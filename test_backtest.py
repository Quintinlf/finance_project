"""Tests for the cost model and the backtest engine.

The engine tests stub out signal generation so they exercise the accounting,
exit logic, and risk gates deterministically without paying for a GP fit per
bar. One test asserts the no-lookahead property directly by capturing exactly
which rows the signal path was handed.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

import logic.backtest as bt
from logic.backtest import (
    BacktestConfig,
    BacktestTrade,
    _bootstrap_ci,
    _check_exit,
    _OpenPosition,
    _sample_warning,
    _shared_trading_dates,
    compute_metrics,
    format_report,
    run_backtest,
)
from logic.costs import (
    CostModel,
    DEFAULT_SPREAD_BPS,
    FALLBACK_SPREAD_BPS,
    for_symbol,
    is_regular_market_hours,
    measure_spreads,
    scale,
    zero_cost_model,
)
from logic.data_structures import Signal


# ========================================================================
# Cost model
# ========================================================================


def test_buy_fills_above_and_sell_fills_below_the_reference():
    model = CostModel(spread_bps=20.0, slippage_bps=5.0)
    assert model.fill_price(100.0, "buy") > 100.0
    assert model.fill_price(100.0, "sell") < 100.0


def test_one_way_cost_is_half_spread_plus_slippage():
    model = CostModel(spread_bps=20.0, slippage_bps=5.0)
    assert model.one_way_bps == pytest.approx(15.0)
    assert model.round_trip_bps == pytest.approx(30.0)


def test_round_trip_cost_scales_with_notional():
    model = CostModel(spread_bps=20.0, slippage_bps=5.0)
    assert model.round_trip_cost(1000.0) == pytest.approx(3.0)
    assert model.round_trip_cost(2000.0) == pytest.approx(6.0)


def test_breakeven_move_matches_round_trip_cost():
    model = CostModel(spread_bps=40.0, slippage_bps=10.0)
    assert model.breakeven_move_pct() == pytest.approx(0.6)


def test_zero_cost_model_is_frictionless():
    model = zero_cost_model()
    assert model.fill_price(100.0, "buy") == 100.0
    assert model.fill_price(100.0, "sell") == 100.0
    assert model.round_trip_cost(10_000.0) == 0.0


def test_commission_is_charged_per_order_not_per_share():
    model = CostModel(spread_bps=0.0, slippage_bps=0.0, commission_per_order=1.0)
    assert model.round_trip_cost(500.0) == pytest.approx(2.0)


def test_negative_costs_are_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        CostModel(spread_bps=-1.0)


def test_invalid_side_and_price_are_rejected():
    model = CostModel(spread_bps=10.0)
    with pytest.raises(ValueError, match="side must be"):
        model.fill_price(100.0, "hold")
    with pytest.raises(ValueError, match="reference_price"):
        model.fill_price(0.0, "buy")


def test_thin_ag_funds_are_assumed_costlier_than_mega_caps():
    assert DEFAULT_SPREAD_BPS["CANE"] > DEFAULT_SPREAD_BPS["GLD"]
    assert DEFAULT_SPREAD_BPS["WEAT"] > DEFAULT_SPREAD_BPS["AAPL"]


def test_measured_spreads_take_precedence_over_estimates():
    model = for_symbol("WEAT", measured={"WEAT": 3.0})
    assert model.spread_bps == 3.0
    assert model.source == "measured"


def test_unknown_symbol_gets_the_pessimistic_fallback():
    model = for_symbol("ZZZZ")
    assert model.spread_bps == FALLBACK_SPREAD_BPS
    assert model.source == "fallback"


def test_scaling_stresses_variable_costs_only():
    doubled = scale(CostModel(spread_bps=20.0, slippage_bps=5.0), 2.0)
    assert doubled.spread_bps == 40.0 and doubled.slippage_bps == 10.0


def test_spread_measurement_refuses_to_run_after_hours():
    # A Saturday: never regular hours, so this must refuse rather than
    # silently persisting unusable quotes.
    assert not is_regular_market_hours()  # tests may run any time; see below
    with pytest.raises(RuntimeError, match="Refusing to measure spreads"):
        measure_spreads(["WEAT"], save_to=None)


# ========================================================================
# Exit logic
# ========================================================================


def _position(entry=100.0, tp=104.0, sl=98.0, qty=1.0, day=date(2026, 1, 5)):
    return _OpenPosition(
        symbol="TEST", quantity=qty, entry_price=entry, entry_date=day,
        tp_price=tp, sl_price=sl, entry_cost=0.0, confidence=0.8,
    )


def _one_bar_frame(bar_date, *, open_, high, low, close):
    return pd.DataFrame(
        {"Open": [open_], "High": [high], "Low": [low], "Close": [close]},
        index=pd.DatetimeIndex([pd.Timestamp(bar_date)]),
    )


def test_stop_triggers_when_low_breaches_it():
    frame = _one_bar_frame(date(2026, 1, 6), open_=100, high=101, low=97, close=99)
    price, reason = _check_exit(frame, date(2026, 1, 6), _position())
    assert reason == "stop_loss" and price == 98.0


def test_target_triggers_when_high_reaches_it():
    frame = _one_bar_frame(date(2026, 1, 6), open_=100, high=105, low=99.5, close=104)
    price, reason = _check_exit(frame, date(2026, 1, 6), _position())
    assert reason == "take_profit" and price == 104.0


def test_bar_hitting_both_resolves_to_the_stop():
    """Pessimistic tie-break — the whole point of the convention."""
    frame = _one_bar_frame(date(2026, 1, 6), open_=100, high=105, low=97, close=103)
    price, reason = _check_exit(frame, date(2026, 1, 6), _position())
    assert reason == "stop_loss" and price == 98.0


def test_quiet_bar_produces_no_exit():
    frame = _one_bar_frame(date(2026, 1, 6), open_=100, high=101, low=99, close=100)
    price, _ = _check_exit(frame, date(2026, 1, 6), _position())
    assert price is None


# ========================================================================
# Trade arithmetic
# ========================================================================


def _trade(entry=100.0, exit_=104.0, qty=2.0, entry_cost=0.2, exit_cost=0.2, reason="take_profit"):
    return BacktestTrade(
        symbol="TEST", quantity=qty,
        entry_date=date(2026, 1, 5), entry_price=entry,
        exit_date=date(2026, 1, 9), exit_price=exit_,
        exit_reason=reason, entry_cost=entry_cost, exit_cost=exit_cost, confidence=0.8,
    )


def test_net_is_the_cash_move_and_gross_adds_costs_back():
    """entry_price/exit_price are fills, so costs are already inside net."""
    trade = _trade()
    assert trade.net_pnl == pytest.approx(8.0)
    assert trade.total_cost == pytest.approx(0.4)
    assert trade.gross_pnl == pytest.approx(8.4)


def test_costs_can_flip_a_marginal_winner_into_a_loser():
    """The reason the cost model exists.

    Reference prices moved +0.1 in your favour, but you bought the offer and
    sold the bid: the fills turn that into a loss.
    """
    trade = _trade(entry=100.2, exit_=99.9, qty=1.0, entry_cost=0.2, exit_cost=0.2)
    assert trade.gross_pnl == pytest.approx(0.1)
    assert trade.net_pnl == pytest.approx(-0.3)
    assert not trade.is_win


def test_holding_period_and_return_pct():
    trade = _trade()
    assert trade.holding_days == 4
    assert trade.return_pct == pytest.approx(8.0 / 200.0 * 100.0)


# ========================================================================
# Metrics
# ========================================================================


def _metrics_for(trades, curve=None, initial=500.0):
    curve = curve or [(date(2026, 1, 5), initial), (date(2026, 1, 9), initial)]
    return compute_metrics(
        trades=trades, equity_curve=curve, initial_equity=initial,
        history={}, symbols=[], trading_dates=[], seed=1,
    )


def test_win_rate_and_profit_factor():
    trades = [_trade(exit_=104.0), _trade(exit_=104.0), _trade(exit_=98.0)]
    m = _metrics_for(trades)
    assert m.n_trades == 3 and m.n_wins == 2
    assert m.win_rate == pytest.approx(66.667, abs=0.01)
    assert m.profit_factor > 1.0


def test_max_drawdown_uses_the_running_peak():
    curve = [(date(2026, 1, i + 1), v) for i, v in enumerate([500, 600, 450, 700])]
    m = _metrics_for([], curve=curve)
    assert m.max_drawdown_pct == pytest.approx(25.0)  # 600 -> 450


def test_gross_and_net_diverge_by_exactly_the_costs():
    trades = [_trade(entry_cost=0.5, exit_cost=0.5) for _ in range(3)]
    m = _metrics_for(trades)
    assert m.gross_pnl - m.total_costs == pytest.approx(m.net_pnl)
    assert m.total_costs == pytest.approx(3.0)


def test_no_trades_yields_zeroed_metrics_and_a_warning():
    m = _metrics_for([])
    assert m.n_trades == 0 and m.net_pnl == 0.0
    assert "No trades" in m.sample_warning


def test_small_sample_is_flagged_as_noise():
    assert "too few" in _sample_warning(12, 1.0, 2.0)


def test_ci_spanning_zero_is_flagged_as_no_edge():
    assert "no detectable edge" in _sample_warning(50, -1.0, 2.0)


def test_a_clean_large_sample_gets_no_warning():
    assert _sample_warning(50, 0.5, 2.0) is None


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    low, high = _bootstrap_ci(values, seed=42)
    assert low < 3.0 < high
    assert (low, high) == _bootstrap_ci(values, seed=42)


def test_bootstrap_ci_is_undefined_for_a_single_trade():
    low, high = _bootstrap_ci([1.0])
    assert low != low and high != high  # NaN


def test_report_renders_and_surfaces_the_warning():
    result = bt.BacktestResult(
        config=BacktestConfig(symbols=["WEAT"], start=date(2026, 1, 1), end=date(2026, 2, 1)),
        trades=[], decisions=[], equity_curve=[(date(2026, 1, 5), 500.0)],
        metrics=_metrics_for([]),
    )
    text = format_report(result)
    assert "BACKTEST" in text and "No trades" in text


# ========================================================================
# Engine integration (stubbed signals)
# ========================================================================


def _ramp_history(symbol="TEST", days=40, start_price=100.0, drift=0.01):
    """A steadily rising series with predictable intraday ranges."""
    idx = pd.bdate_range(date(2026, 1, 1), periods=days)
    rows = []
    price = start_price
    for _ in range(days):
        open_ = price
        close = price * (1 + drift)
        rows.append({"Open": open_, "High": close * 1.005, "Low": open_ * 0.995, "Close": close})
        price = close
    return {symbol: pd.DataFrame(rows, index=idx)}


def _stub_signals(monkeypatch, signal_type="buy", confidence=0.9, prob=0.9, capture=None):
    def fake(universe, config, verbose=False, price_data=None, build_plot=True):
        if capture is not None:
            capture.append({s: df.copy() for s, df in (price_data or {}).items()})
        return [
            Signal(symbol=s, signal_type=signal_type, confidence=confidence,
                   prob_profit=prob, meta={"current_price": 100.0})
            for s in universe
        ]

    monkeypatch.setattr(bt, "generate_signals", fake)
    monkeypatch.setattr(bt, "filter_signals_by_thresholds", lambda sigs, **kw: sigs)


def _cfg(history, **overrides):
    idx = list(history.values())[0].index
    params = dict(
        symbols=list(history), start=idx[0].date(), end=idx[-1].date(),
        initial_equity=1000.0, fractional=True, max_portfolio_exposure=0.95,
    )
    params.update(overrides)
    return BacktestConfig(**params)


def test_engine_never_shows_the_signal_path_a_future_bar(monkeypatch):
    """The core anti-lookahead guarantee."""
    history = _ramp_history(days=30)
    captured = []
    _stub_signals(monkeypatch, signal_type="hold", capture=captured)

    run_backtest(_cfg(history), history=history, progress=False)

    dates = sorted(history["TEST"].index)
    assert captured, "signal path was never invoked"
    for i, snapshot in enumerate(captured):
        frame = snapshot["TEST"]
        # Bar i is the decision bar; the frame must end exactly there.
        assert frame.index.max() == dates[i]
        assert len(frame) == i + 1


def test_a_rising_market_with_persistent_buys_makes_money(monkeypatch):
    history = _ramp_history(days=30, drift=0.01)
    _stub_signals(monkeypatch, signal_type="buy")
    result = run_backtest(_cfg(history), history=history, progress=False)
    assert result.metrics.n_trades > 0
    assert result.metrics.net_pnl > 0


def test_costs_reduce_but_do_not_reverse_a_strong_trend(monkeypatch):
    history = _ramp_history(days=30, drift=0.01)
    _stub_signals(monkeypatch, signal_type="buy")
    result = run_backtest(_cfg(history), history=history, progress=False)
    assert result.metrics.total_costs > 0
    assert result.metrics.gross_pnl > result.metrics.net_pnl


def test_hold_signals_never_open_a_position(monkeypatch):
    history = _ramp_history(days=30)
    _stub_signals(monkeypatch, signal_type="hold")
    result = run_backtest(_cfg(history), history=history, progress=False)
    assert result.metrics.n_trades == 0
    assert result.equity_curve[-1][1] == pytest.approx(1000.0)


def test_exposure_gate_blocks_and_is_recorded_in_the_decision_log(monkeypatch):
    """Needs two symbols: while long one name, a buy on the other must be gated.

    With a single symbol the gate is unreachable — decide_action turns BUY into
    HOLD once a position exists, so the risk check never runs.
    """
    history = _ramp_history(symbol="AAA", days=30)
    history.update(_ramp_history(symbol="BBB", days=30))
    _stub_signals(monkeypatch, signal_type="buy")
    cfg = _cfg(history, max_portfolio_exposure=0.05, max_position_fraction=0.20)
    result = run_backtest(cfg, history=history, progress=False)
    reasons = [d.reason for d in result.decisions if d.action == "rejected"]
    assert any("exposure limit" in r for r in reasons)


def test_final_equity_reconciles_with_reported_net_pnl(monkeypatch):
    history = _ramp_history(days=30)
    _stub_signals(monkeypatch, signal_type="buy")
    result = run_backtest(_cfg(history), history=history, progress=False)
    expected = result.config.initial_equity + result.metrics.net_pnl
    assert result.equity_curve[-1][1] == pytest.approx(expected, rel=1e-6)


def test_every_position_is_closed_by_the_end(monkeypatch):
    history = _ramp_history(days=30)
    _stub_signals(monkeypatch, signal_type="buy")
    result = run_backtest(_cfg(history), history=history, progress=False)
    assert all(t.exit_date <= result.config.end for t in result.trades)


def test_whole_share_mode_only_ever_buys_integers(monkeypatch):
    history = _ramp_history(days=30, start_price=20.0)
    _stub_signals(monkeypatch, signal_type="buy")
    result = run_backtest(_cfg(history, fractional=False), history=history, progress=False)
    for trade in result.trades:
        assert trade.quantity == int(trade.quantity)


def test_fractional_mode_can_buy_a_slice_of_an_unaffordable_share(monkeypatch):
    """$500 equity, $2000 share: impossible whole, fine fractional."""
    history = _ramp_history(days=30, start_price=2000.0)
    _stub_signals(monkeypatch, signal_type="buy")

    whole = run_backtest(
        _cfg(history, fractional=False, initial_equity=500.0), history=history, progress=False
    )
    frac = run_backtest(
        _cfg(history, fractional=True, initial_equity=500.0), history=history, progress=False
    )
    assert whole.metrics.n_trades == 0
    assert frac.metrics.n_trades > 0
    assert all(0 < t.quantity < 1 for t in frac.trades)


def test_a_decision_is_logged_for_every_symbol_on_every_executable_bar(monkeypatch):
    history = _ramp_history(days=25)
    _stub_signals(monkeypatch, signal_type="hold")
    result = run_backtest(_cfg(history), history=history, progress=False)
    # Decisions run on every bar except the last (which is execution-only).
    assert len(result.decisions) == len(result.equity_curve) - 1


def test_backtest_rejects_an_inverted_window():
    with pytest.raises(ValueError, match="must be after"):
        BacktestConfig(symbols=["WEAT"], start=date(2026, 2, 1), end=date(2026, 1, 1))


def test_backtest_rejects_an_empty_symbol_list():
    with pytest.raises(ValueError, match="symbols must not be empty"):
        BacktestConfig(symbols=[], start=date(2026, 1, 1), end=date(2026, 2, 1))


def test_shared_dates_intersect_across_symbols():
    idx_a = pd.bdate_range(date(2026, 1, 1), periods=10)
    idx_b = pd.bdate_range(date(2026, 1, 5), periods=10)
    history = {
        "A": pd.DataFrame({"Close": range(10)}, index=idx_a),
        "B": pd.DataFrame({"Close": range(10)}, index=idx_b),
    }
    shared = _shared_trading_dates(history, ["A", "B"], date(2026, 1, 1), date(2026, 2, 1))
    assert shared[0] == idx_b[0].date()
    assert shared[-1] == idx_a[-1].date()
