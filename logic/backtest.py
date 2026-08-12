"""
Historical replay and backtest engine.

This exists to answer the only question that matters before funding an account:
**does this strategy have an edge?** Nothing else in the repo can answer it. The
live system has produced 45 decisions across five months and never recorded a
single realized P&L, so every parameter in it is currently tuned against
nothing.

Design commitments, because a backtest that flatters you is worse than none:

*No lookahead.* At bar ``t`` the engine hands the signal path only
``history[:t+1]`` and executes at bar ``t+1``'s open. A decision can never see
the bar it trades into.

*The real strategy, not a copy.* Signals come from ``signal_engine`` and the
decision gates are the production ``decide_action`` / ``build_order_plan`` /
``enforce_risk_limits``. Reimplementing them here would measure a lookalike and
tell you nothing about the system that actually trades.

*Costs always on.* Every fill crosses the spread via ``logic.costs``. Results
are reported gross and net so the cost drag is explicit.

*Pessimistic tie-breaks.* When a bar's range spans both the take-profit and the
stop, the stop is assumed to fill first. Intraday order is unknowable from daily
bars, and the optimistic assumption is how backtests manufacture fake edges.

*Honest sample size.* ``BacktestMetrics`` carries a bootstrap confidence
interval on mean trade return and a warning when the sample is too small to
conclude anything. A 60%-win-rate readout over 12 trades is noise.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from logic import eos_bridge
from logic.costs import CostModel, for_symbol, load_measured_spreads
from logic.data_structures import ExecutionConfig, PositionState, Signal
from logic.execution_engine import build_order_plan, decide_action, resolve_exit_pcts
from logic.signal_engine import filter_signals_by_thresholds, generate_signals

ExitReason = Literal["take_profit", "stop_loss", "signal", "end_of_backtest"]

# Bars of history the signal path needs before it can produce anything. The
# forecast wants 200 daily bars and drops rows to build 10 lags.
MIN_WARMUP_BARS = 220

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestConfig:
    """Everything that defines a run. Mirrors ExecutionConfig where they overlap."""

    symbols: Sequence[str]
    start: date
    end: date
    initial_equity: float = 500.0
    min_confidence: float = 0.50
    min_prob_up: float = 0.50
    tp_pct: float = 0.04
    sl_pct: float = 0.02
    base_risk_pct: float = 2.0
    max_position_fraction: float = 0.20
    max_portfolio_exposure: float = 0.30
    max_trades_per_day: int = 3
    fractional: bool = False
    min_notional: float = 1.0
    slippage_bps: float = 5.0
    use_measured_spreads: bool = True
    seed: int = 7

    # EOS finance-domain integration. Default 'off' here rather than the
    # ExecutionConfig default of 'shadow': a full refit runs per symbol per bar,
    # so leaving it on silently would multiply backtest runtime with no effect
    # on results. Turn it on deliberately to measure whether these help.
    eos_mode: str = "off"
    eos_use_garch_exits: bool = False
    eos_use_hurst_confidence: bool = False
    eos_enrichment_stride: int = 5

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be after start ({self.start})")
        if self.initial_equity <= 0:
            raise ValueError(f"initial_equity must be > 0, got {self.initial_equity}")


@dataclass
class BarDecision:
    """One symbol on one bar: what was seen, what was decided, and why.

    This is the record the replay UI will scrub through, so it deliberately
    keeps the reasoning fields rather than just the action.
    """

    bar_date: date
    symbol: str
    close: float
    signal_type: str
    confidence: float
    prob_profit: float
    action: str
    reason: str
    quantity: float = 0.0
    fill_price: Optional[float] = None
    position_qty_before: float = 0.0
    equity: float = 0.0
    rsi: Optional[float] = None
    bb_z_score: Optional[float] = None
    market_regime: Dict[str, float] = field(default_factory=dict)
    belief_entropy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.bar_date.isoformat(),
            "symbol": self.symbol,
            "close": self.close,
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "prob_profit": self.prob_profit,
            "action": self.action,
            "reason": self.reason,
            "quantity": self.quantity,
            "fill_price": self.fill_price,
            "position_qty_before": self.position_qty_before,
            "equity": self.equity,
            "rsi": self.rsi,
            "bb_z_score": self.bb_z_score,
            "market_regime": dict(self.market_regime),
            "belief_entropy": self.belief_entropy,
        }


@dataclass
class BacktestTrade:
    """A completed round trip, with costs separated from price movement."""

    symbol: str
    quantity: float
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    exit_reason: ExitReason
    entry_cost: float
    exit_cost: float
    confidence: float

    @property
    def net_pnl(self) -> float:
        """Actual cash P&L. ``entry_price`` and ``exit_price`` are fills, so the
        spread and slippage are already inside this number — which is why costs
        are added back to get gross rather than subtracted to get net."""
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def total_cost(self) -> float:
        return self.entry_cost + self.exit_cost

    @property
    def gross_pnl(self) -> float:
        """The counterfactual: what this trade would have made frictionlessly."""
        return self.net_pnl + self.total_cost

    @property
    def return_pct(self) -> float:
        notional = self.entry_price * self.quantity
        return (self.net_pnl / notional * 100.0) if notional else 0.0

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_date": self.entry_date.isoformat(),
            "entry_price": self.entry_price,
            "exit_date": self.exit_date.isoformat(),
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "gross_pnl": self.gross_pnl,
            "total_cost": self.total_cost,
            "net_pnl": self.net_pnl,
            "return_pct": self.return_pct,
            "holding_days": self.holding_days,
            "confidence": self.confidence,
        }


@dataclass
class BacktestMetrics:
    """Performance summary. Reports gross and net so cost drag stays visible."""

    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    gross_pnl: float
    total_costs: float
    net_pnl: float
    total_return_pct: float
    gross_return_pct: float
    cost_drag_pct: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    max_drawdown_pct: float
    sharpe: float
    avg_holding_days: float
    buy_hold_return_pct: float
    mean_trade_return_pct: float
    trade_return_ci95: Tuple[float, float]
    sample_warning: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = {k: getattr(self, k) for k in self.__dataclass_fields__}
        payload["trade_return_ci95"] = list(self.trade_return_ci95)
        return payload


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: List[BacktestTrade]
    decisions: List[BarDecision]
    equity_curve: List[Tuple[date, float]]
    metrics: BacktestMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": list(self.config.symbols),
            "start": self.config.start.isoformat(),
            "end": self.config.end.isoformat(),
            "initial_equity": self.config.initial_equity,
            "fractional": self.config.fractional,
            "metrics": self.metrics.to_dict(),
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": [(d.isoformat(), v) for d, v in self.equity_curve],
            "decisions": [d.to_dict() for d in self.decisions],
        }

    def save(self, path: Path) -> Path:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


@dataclass
class _OpenPosition:
    symbol: str
    quantity: float
    entry_price: float
    entry_date: date
    tp_price: float
    sl_price: float
    entry_cost: float
    confidence: float


# ========================================================================
# Data
# ========================================================================


def load_history(
    symbols: Sequence[str],
    start: date,
    end: date,
    *,
    warmup_bars: int = MIN_WARMUP_BARS,
) -> Dict[str, Any]:
    """Download daily OHLCV with enough warmup before ``start`` to seed signals.

    Returns {symbol: DataFrame}. Symbols that fail to download, or that lack
    enough bars to produce a signal, are dropped with a warning rather than
    silently producing empty results.
    """
    import pandas as pd
    import yfinance as yf

    # Pad generously: warmup is in trading days, the fetch window is calendar days.
    padded_start = start - _calendar_days_for_bars(warmup_bars)

    out: Dict[str, Any] = {}
    for symbol in symbols:
        try:
            frame = yf.download(
                symbol,
                start=padded_start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logging.warning("History download failed for %s (%s).", symbol, exc)
            continue

        if frame is None or frame.empty:
            logging.warning("No history returned for %s; excluding it.", symbol)
            continue

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [c[0] if isinstance(c, tuple) else c for c in frame.columns]

        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        if len(frame) < warmup_bars:
            logging.warning(
                "%s has %s bars, need %s for warmup; excluding it.",
                symbol, len(frame), warmup_bars,
            )
            continue

        out[symbol] = frame

    return out


def _calendar_days_for_bars(bars: int):
    from datetime import timedelta

    # ~252 trading days per 365 calendar days, plus slack for holidays.
    return timedelta(days=int(bars * 365 / 252) + 20)


# ========================================================================
# Engine
# ========================================================================


def run_backtest(
    cfg: BacktestConfig,
    *,
    history: Optional[Dict[str, Any]] = None,
    progress: bool = True,
) -> BacktestResult:
    """Replay the strategy bar by bar and return trades, decisions, and metrics."""
    if history is None:
        history = load_history(cfg.symbols, cfg.start, cfg.end)

    symbols = [s for s in cfg.symbols if s in history]
    if not symbols:
        raise ValueError("No usable history for any requested symbol")

    measured = load_measured_spreads() if cfg.use_measured_spreads else {}
    cost_models: Dict[str, CostModel] = {
        s: for_symbol(s, measured=measured, slippage_bps=cfg.slippage_bps) for s in symbols
    }

    trading_dates = _shared_trading_dates(history, symbols, cfg.start, cfg.end)
    if len(trading_dates) < 2:
        raise ValueError(
            f"Only {len(trading_dates)} bars between {cfg.start} and {cfg.end}; "
            "widen the window."
        )

    exec_config = ExecutionConfig(
        execution_mode="simulation",
        allow_short_selling=False,
        dry_run=False,
        base_risk_pct=cfg.base_risk_pct,
        tp_pct=cfg.tp_pct,
        sl_pct=cfg.sl_pct,
        max_position_pct_of_equity=cfg.max_position_fraction * 100.0,
        min_confidence=cfg.min_confidence,
        min_prob_up=cfg.min_prob_up,
        eos_mode=cfg.eos_mode,
        eos_use_garch_exits=cfg.eos_use_garch_exits,
        eos_use_hurst_confidence=cfg.eos_use_hurst_confidence,
        eos_enrichment_stride=cfg.eos_enrichment_stride,
    )

    # Cached enrichments are keyed by symbol and would otherwise leak across
    # runs in the same process, letting one backtest reuse another's fits.
    eos_bridge.reset_enrichment_cache()

    cash = float(cfg.initial_equity)
    positions: Dict[str, _OpenPosition] = {}
    trades: List[BacktestTrade] = []
    decisions: List[BarDecision] = []
    equity_curve: List[Tuple[date, float]] = []

    # Decide on bar i, execute on bar i+1 — so the final bar is decision-only.
    for i, bar_date in enumerate(trading_dates[:-1]):
        next_date = trading_dates[i + 1]

        equity = cash + sum(
            pos.quantity * _bar_value(history[sym], bar_date, "Close")
            for sym, pos in positions.items()
        )
        equity_curve.append((bar_date, equity))

        if progress and i % 20 == 0:
            logging.info(
                "  bar %s/%s  %s  equity=$%.2f  open=%s  trades=%s",
                i + 1, len(trading_dates) - 1, bar_date, equity, len(positions), len(trades),
            )

        # --- Exits first: a position opened earlier may close on this bar ---
        for symbol in list(positions):
            pos = positions[symbol]
            if pos.entry_date >= bar_date:
                continue  # opened today; exits start from the following bar
            exit_price, reason = _check_exit(history[symbol], bar_date, pos)
            if exit_price is None:
                continue
            cash, trade = _close_position(
                pos, exit_price, bar_date, reason, cost_models[symbol], cash
            )
            trades.append(trade)
            del positions[symbol]

        # --- Signals on data through this bar's close ---
        price_data = {s: history[s].loc[:_as_ts(history[s], bar_date)] for s in symbols}
        with _quiet_signal_logging():
            signals = _generate_quiet(symbols, exec_config, price_data)
            signals = filter_signals_by_thresholds(
                signals,
                min_confidence=cfg.min_confidence,
                min_prob_up=cfg.min_prob_up,
            )

        trades_today = 0
        for signal in signals:
            symbol = signal.symbol
            pos = positions.get(symbol)
            close = _bar_value(history[symbol], bar_date, "Close")

            position_state = PositionState(
                symbol=symbol,
                quantity=pos.quantity if pos else 0.0,
                avg_entry_price=pos.entry_price if pos else 0.0,
                side="long" if pos else "flat",
                source="sim",
            )

            action, reason = decide_action(signal, position_state, exec_config)

            record = BarDecision(
                bar_date=bar_date,
                symbol=symbol,
                close=close,
                signal_type=signal.signal_type,
                confidence=float(signal.confidence),
                prob_profit=float(signal.prob_profit),
                action=action,
                reason=reason,
                position_qty_before=position_state.quantity,
                equity=equity,
                rsi=_meta_float(signal, "rsi_value"),
                bb_z_score=_meta_float(signal, "bb_z_score"),
                market_regime=dict(signal.meta.get("market_regime") or {}),
                belief_entropy=_meta_float(signal, "belief_entropy"),
            )

            if action == "sell" and pos is not None:
                next_open = _bar_value(history[symbol], next_date, "Open")
                cash, trade = _close_position(
                    pos, next_open, next_date, "signal", cost_models[symbol], cash
                )
                trades.append(trade)
                del positions[symbol]
                record.quantity = trade.quantity
                record.fill_price = trade.exit_price
                decisions.append(record)
                continue

            if action != "buy":
                decisions.append(record)
                continue

            # --- Buy path: risk gates, then sizing, then fill at next open ---
            invested = sum(
                p.quantity * _bar_value(history[s], bar_date, "Close")
                for s, p in positions.items()
            )
            exposure = invested / equity if equity > 0 else 1.0

            blocked = _pre_trade_block(
                exposure=exposure,
                cfg=cfg,
                trades_today=trades_today,
                equity=equity,
            )
            if blocked:
                record.action, record.reason = "rejected", blocked
                decisions.append(record)
                continue

            next_open = _bar_value(history[symbol], next_date, "Open")
            quantity = _size_position(
                signal=signal,
                position_state=position_state,
                exec_config=exec_config,
                cfg=cfg,
                equity=equity,
                cash=cash,
                price=next_open,
            )
            if quantity <= 0:
                record.action = "skipped"
                record.reason = (
                    f"position_below_minimum: ${equity * cfg.max_position_fraction:,.2f} "
                    f"cap cannot fund a position at ${next_open:,.2f}"
                )
                decisions.append(record)
                continue

            fill = cost_models[symbol].fill_price(next_open, "buy")
            notional = fill * quantity
            if notional > cash:
                record.action, record.reason = "rejected", "insufficient cash"
                decisions.append(record)
                continue

            entry_cost = (fill - next_open) * quantity
            cash -= notional
            # Exit levels come from the same resolver production uses, so a
            # volatility-scaled stop is replayed rather than silently reverted
            # to the flat configured percentage.
            entry_tp_pct, entry_sl_pct = resolve_exit_pcts(signal, exec_config)
            positions[symbol] = _OpenPosition(
                symbol=symbol,
                quantity=quantity,
                entry_price=fill,
                entry_date=next_date,
                tp_price=fill * (1.0 + entry_tp_pct),
                sl_price=fill * (1.0 - entry_sl_pct),
                entry_cost=entry_cost,
                confidence=float(signal.confidence),
            )
            trades_today += 1
            record.quantity = quantity
            record.fill_price = fill
            decisions.append(record)

    # --- Force-close whatever is still open, at the last close ---
    final_date = trading_dates[-1]
    for symbol, pos in list(positions.items()):
        final_close = _bar_value(history[symbol], final_date, "Close")
        cash, trade = _close_position(
            pos, final_close, final_date, "end_of_backtest", cost_models[symbol], cash
        )
        trades.append(trade)
        del positions[symbol]
    equity_curve.append((final_date, cash))

    metrics = compute_metrics(
        trades=trades,
        equity_curve=equity_curve,
        initial_equity=cfg.initial_equity,
        history=history,
        symbols=symbols,
        trading_dates=trading_dates,
        seed=cfg.seed,
    )

    return BacktestResult(
        config=cfg,
        trades=trades,
        decisions=decisions,
        equity_curve=equity_curve,
        metrics=metrics,
    )


@contextlib.contextmanager
def _quiet_signal_logging():
    """Mute the per-symbol signal logs for the duration of one bar.

    ``filter_signals_by_thresholds`` emits a line per symbol per call, which is
    useful once a day and unreadable across thousands of replayed bars.
    """
    loggers = [logging.getLogger("logic.signal_engine"), logging.getLogger()]
    previous = [lg.level for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.WARNING)
        yield
    finally:
        for lg, level in zip(loggers, previous):
            lg.setLevel(level)


def _generate_quiet(symbols, exec_config, price_data) -> List[Signal]:
    """Run the production signal path with its console output suppressed.

    The forecast prints a multi-line report per symbol; across thousands of
    calls that is the single largest source of wall-clock noise and log spam.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return generate_signals(
            list(symbols),
            exec_config,
            verbose=False,
            price_data=price_data,
            build_plot=False,
        )


def _pre_trade_block(*, exposure: float, cfg: BacktestConfig, trades_today: int, equity: float) -> Optional[str]:
    """Risk gates, mirroring production ``enforce_risk_limits`` ordering."""
    if trades_today >= cfg.max_trades_per_day:
        return "Blocked: max trades per day reached"
    if exposure >= cfg.max_portfolio_exposure:
        return (
            f"Blocked: portfolio exposure limit reached "
            f"({exposure:.1%} >= {cfg.max_portfolio_exposure:.0%})"
        )
    if equity <= 0:
        return "Blocked: account wiped out"
    return None


def _size_position(
    *,
    signal: Signal,
    position_state: PositionState,
    exec_config: ExecutionConfig,
    cfg: BacktestConfig,
    equity: float,
    cash: float,
    price: float,
) -> float:
    """Position size, deferring to production sizing for the whole-share case.

    The fractional branch applies the same dollar target but drops the integer
    floor, which is exactly what notional orders change. Production does not
    support fractional yet, so that branch has no counterpart to reuse.
    """
    budget = min(equity * cfg.max_position_fraction, cash)
    if budget <= 0 or price <= 0:
        return 0.0

    if not cfg.fractional:
        plan = build_order_plan(
            signal=signal,
            position_state=position_state,
            config=exec_config,
            account_cash=cash,
            current_price=price,
            account_equity=equity,
            max_position_fraction=cfg.max_position_fraction,
        )
        quantity = float(plan.quantity)
        return quantity if quantity * price <= cash else float(int(cash / price))

    notional = min(budget, equity * cfg.max_position_fraction)
    if notional < cfg.min_notional:
        return 0.0
    return notional / price


def _check_exit(frame, bar_date: date, pos: _OpenPosition) -> Tuple[Optional[float], ExitReason]:
    """Did this bar's range trigger the stop or the target?

    When the bar spans both, the stop wins. Daily bars carry no intrabar
    sequence, and assuming the favourable order is the classic way to
    manufacture an edge that does not exist.
    """
    low = _bar_value(frame, bar_date, "Low")
    high = _bar_value(frame, bar_date, "High")

    hit_stop = low <= pos.sl_price
    hit_target = high >= pos.tp_price

    if hit_stop:
        return pos.sl_price, "stop_loss"
    if hit_target:
        return pos.tp_price, "take_profit"
    return None, "signal"


def _close_position(
    pos: _OpenPosition,
    reference_price: float,
    exit_date: date,
    reason: ExitReason,
    cost_model: CostModel,
    cash: float,
) -> Tuple[float, BacktestTrade]:
    """Realise a position at a reference price and return (new_cash, trade).

    The caller passes the unadjusted reference (stop level, target level, or
    next open); crossing the spread happens here so every exit path is costed
    identically.
    """
    fill = cost_model.fill_price(reference_price, "sell")
    exit_cost = (reference_price - fill) * pos.quantity

    cash += fill * pos.quantity
    trade = BacktestTrade(
        symbol=pos.symbol,
        quantity=pos.quantity,
        entry_date=pos.entry_date,
        entry_price=pos.entry_price,
        exit_date=exit_date,
        exit_price=fill,
        exit_reason=reason,
        entry_cost=pos.entry_cost,
        exit_cost=exit_cost,
        confidence=pos.confidence,
    )
    return cash, trade


# ========================================================================
# Helpers
# ========================================================================


def _as_ts(frame, bar_date: date):
    """Normalise a date to the frame's index type."""
    import pandas as pd

    idx = frame.index
    ts = pd.Timestamp(bar_date)
    if getattr(idx, "tz", None) is not None:
        ts = ts.tz_localize(idx.tz)
    return ts


def _bar_value(frame, bar_date: date, column: str) -> float:
    return float(frame.loc[_as_ts(frame, bar_date), column])


def _meta_float(signal: Signal, key: str) -> Optional[float]:
    value = signal.meta.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _shared_trading_dates(history, symbols, start: date, end: date) -> List[date]:
    """Dates every symbol has a bar for, inside the window.

    Intersecting rather than unioning keeps the portfolio marked consistently:
    a missing bar for one name would otherwise silently freeze its contribution
    to equity.
    """
    common = None
    for symbol in symbols:
        dates = {d.date() if hasattr(d, "date") else d for d in history[symbol].index}
        common = dates if common is None else (common & dates)
    return sorted(d for d in (common or set()) if start <= d <= end)


# ========================================================================
# Metrics
# ========================================================================


def compute_metrics(
    *,
    trades: Sequence[BacktestTrade],
    equity_curve: Sequence[Tuple[date, float]],
    initial_equity: float,
    history: Dict[str, Any],
    symbols: Sequence[str],
    trading_dates: Sequence[date],
    seed: int = 7,
) -> BacktestMetrics:
    """Summarise a run, gross and net, with a sample-size caveat."""
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]

    gross_pnl = sum(t.gross_pnl for t in trades)
    total_costs = sum(t.total_cost for t in trades)
    net_pnl = sum(t.net_pnl for t in trades)

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    values = [v for _, v in equity_curve]
    peak = -float("inf")
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)

    daily_returns = [
        (values[i] / values[i - 1] - 1.0)
        for i in range(1, len(values))
        if values[i - 1] > 0
    ]
    sharpe = _annualised_sharpe(daily_returns)

    trade_returns = [t.return_pct for t in trades]
    mean_trade_return = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
    ci_low, ci_high = _bootstrap_ci(trade_returns, seed=seed)

    return BacktestMetrics(
        n_trades=len(trades),
        n_wins=len(wins),
        n_losses=len(losses),
        win_rate=(len(wins) / len(trades) * 100.0) if trades else 0.0,
        gross_pnl=gross_pnl,
        total_costs=total_costs,
        net_pnl=net_pnl,
        total_return_pct=(net_pnl / initial_equity * 100.0) if initial_equity else 0.0,
        gross_return_pct=(gross_pnl / initial_equity * 100.0) if initial_equity else 0.0,
        cost_drag_pct=(total_costs / initial_equity * 100.0) if initial_equity else 0.0,
        profit_factor=profit_factor,
        avg_win=(sum(t.net_pnl for t in wins) / len(wins)) if wins else 0.0,
        avg_loss=(sum(t.net_pnl for t in losses) / len(losses)) if losses else 0.0,
        expectancy=(net_pnl / len(trades)) if trades else 0.0,
        max_drawdown_pct=max_dd * 100.0,
        sharpe=sharpe,
        avg_holding_days=(
            sum(t.holding_days for t in trades) / len(trades) if trades else 0.0
        ),
        buy_hold_return_pct=_buy_and_hold_return(history, symbols, trading_dates),
        mean_trade_return_pct=mean_trade_return,
        trade_return_ci95=(ci_low, ci_high),
        sample_warning=_sample_warning(len(trades), ci_low, ci_high),
    )


def _annualised_sharpe(daily_returns: Sequence[float]) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mean = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(TRADING_DAYS_PER_YEAR)


def _bootstrap_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 7,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean. Returns (nan, nan) below 2 samples."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1])


def _sample_warning(n_trades: int, ci_low: float, ci_high: float) -> Optional[str]:
    """Say plainly when the result cannot support a conclusion."""
    if n_trades == 0:
        return "No trades were taken — nothing to evaluate."
    if n_trades < 30:
        return (
            f"Only {n_trades} trades. This is far too few to distinguish skill "
            f"from luck; treat every metric here as noise."
        )
    if not math.isnan(ci_low) and ci_low < 0 < ci_high:
        return (
            f"The 95% CI on mean trade return spans zero "
            f"([{ci_low:.2f}%, {ci_high:.2f}%]) — no detectable edge."
        )
    return None


def _buy_and_hold_return(history, symbols: Sequence[str], trading_dates: Sequence[date]) -> float:
    """Equal-weight buy-and-hold over the same window, as a baseline.

    Without this a positive return means nothing: in a rising market, doing
    nothing clever also makes money.
    """
    if not trading_dates or not symbols:
        return 0.0
    first, last = trading_dates[0], trading_dates[-1]
    returns = []
    for symbol in symbols:
        try:
            start_px = _bar_value(history[symbol], first, "Close")
            end_px = _bar_value(history[symbol], last, "Close")
        except (KeyError, ValueError):
            continue
        if start_px > 0:
            returns.append((end_px / start_px - 1.0) * 100.0)
    return sum(returns) / len(returns) if returns else 0.0


def format_report(result: BacktestResult) -> str:
    """Human-readable summary for the console."""
    m = result.metrics
    cfg = result.config
    exits: Dict[str, int] = {}
    for trade in result.trades:
        exits[trade.exit_reason] = exits.get(trade.exit_reason, 0) + 1

    lines = [
        "=" * 66,
        f"BACKTEST  {cfg.start} -> {cfg.end}  ({len(result.equity_curve)} bars)",
        f"Symbols: {', '.join(cfg.symbols)}",
        f"Sizing: {'fractional' if cfg.fractional else 'whole shares'}, "
        f"cap {cfg.max_position_fraction:.0%}/position, "
        f"exposure limit {cfg.max_portfolio_exposure:.0%}",
        "=" * 66,
        f"  Starting equity     ${cfg.initial_equity:,.2f}",
        f"  Ending equity       ${cfg.initial_equity + m.net_pnl:,.2f}",
        f"  Net return          {m.total_return_pct:+.2f}%",
        f"  Gross return        {m.gross_return_pct:+.2f}%   (before costs)",
        f"  Cost drag           {m.cost_drag_pct:.2f}%   (${m.total_costs:,.2f} paid)",
        f"  Buy & hold          {m.buy_hold_return_pct:+.2f}%   (equal weight, same window)",
        "",
        f"  Trades              {m.n_trades}  ({m.n_wins}W / {m.n_losses}L)",
        f"  Win rate            {m.win_rate:.1f}%",
        f"  Profit factor       {m.profit_factor:.2f}",
        f"  Expectancy          ${m.expectancy:+,.2f} per trade",
        f"  Avg win / loss      ${m.avg_win:+,.2f} / ${m.avg_loss:+,.2f}",
        f"  Avg holding         {m.avg_holding_days:.1f} days",
        f"  Exits               {exits or 'none'}",
        "",
        f"  Max drawdown        {m.max_drawdown_pct:.2f}%",
        f"  Sharpe (annualised) {m.sharpe:.2f}",
        f"  Mean trade return   {m.mean_trade_return_pct:+.2f}%  "
        f"95% CI [{m.trade_return_ci95[0]:+.2f}%, {m.trade_return_ci95[1]:+.2f}%]",
        "=" * 66,
    ]
    if m.sample_warning:
        lines += [f"  ⚠  {m.sample_warning}", "=" * 66]
    return "\n".join(lines)
