from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Literal

from zoneinfo import ZoneInfo

from logic.account_equity import resolve_account_equity
from logic.broker_client import create_broker_client
from logic.data_structures import ExecutionConfig
from logic.execution_engine import run_trading_cycle
from logic.fill_reconciler import reconcile_fills
from logic.inverse_routing import format_routing_table, route_signals
from logic.options_engine import (
    filter_expirations_by_dte,
    get_option_chain,
    select_call_contract,
    select_put_contract,
)
from logic.portfolio_state import get_position_states
from logic.position_reconciler import reconcile_position_exits
from logic.risk_config import load_risk_config
from logic.signal_engine import (
    filter_signals_by_thresholds,
    generate_signals,
)
from logic.sqlite_store import init_db, insert_decisions
from logic.thesis import shadow_log_theses
from logic.universe import (
    build_tradeable_universe,
    format_affordability_table,
    lookup as lookup_asset,
)


DEFAULT_DB_PATH = Path("trade_logs") / "trading.db"
LAST_RUN_PATH = Path("trade_logs") / "last_run_date.json"


def _get_eastern_now() -> datetime:
    return datetime.now(tz=ZoneInfo("US/Eastern"))


def _load_last_run_date() -> Optional[str]:
    try:
        if not LAST_RUN_PATH.exists():
            return None
        data = json.loads(LAST_RUN_PATH.read_text())
        return data.get("last_run_date")
    except Exception:
        return None


def _save_last_run_date(run_date: str) -> None:
    LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_run_date": run_date,
        "last_run_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    LAST_RUN_PATH.write_text(json.dumps(payload, indent=2))


def _is_market_day(trading_client) -> bool:
    eastern_now = _get_eastern_now()
    return trading_client.is_market_open(eastern_now.date())


def _summarize_decisions(decision_log) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for entry in decision_log:
        action = entry.action
        summary[action] = summary.get(action, 0) + 1
    return summary


def _held_symbols(broker_client) -> List[str]:
    """Symbols with a non-zero position, regardless of universe membership.

    Passing an empty universe returns only what the broker actually holds, which
    is what we need to make sure a holding stays sellable even after it drops
    out of the affordable set.
    """
    try:
        states = broker_client.get_position_states([])
    except Exception as exc:
        logging.warning("Could not read held positions (%s). Assuming none.", exc)
        return []
    return [sym for sym, state in states.items() if float(getattr(state, "quantity", 0) or 0) != 0]


def _log_options_candidates(symbol: str) -> None:
    chain = get_option_chain(symbol)
    filtered = filter_expirations_by_dte(chain)
    put_contract = select_put_contract(filtered)
    call_contract = select_call_contract(filtered)
    logging.info(
        "OPTIONS CANDIDATES FOR TODAY: %s -> PUT %s | CALL %s",
        symbol,
        put_contract,
        call_contract,
    )


def run_daily_trading_cycle(
    *,
    execution_mode: Literal["simulation", "paper", "live"] = "paper",
    dry_run: bool = False,
    min_confidence: float = 0.50,
    min_prob_up: float = 0.50,
    tp_pct: float = 0.04,
    sl_pct: float = 0.02,
    base_risk_pct: float = 2.0,
    debug_force_strongest_signal: bool = False,
    universe: Optional[List[str]] = None,
    universe_scope: str = "all",
    screen_affordability: bool = True,
    enable_inverse_routing: bool = True,
) -> None:
    logging.info("START DAILY TRADING RUN")

    eastern_now = _get_eastern_now()
    logging.info("DATE/TIME: %s", eastern_now.isoformat())

    broker_client = create_broker_client(execution_mode=execution_mode)

    if not _is_market_day(broker_client):
        logging.info("Not a US market day. Skipping run.")
        return

    last_run_date = _load_last_run_date()
    today_str = eastern_now.date().isoformat()
    if last_run_date == today_str:
        logging.info("Duplicate run detected for %s. Forcing dry_run=True.", today_str)
        dry_run = True

    summary = broker_client.get_account_summary()
    account_cash = float(summary.get("cash", 0.0) or 0.0)
    account_id = str(summary.get("account_id") or "paper-default")

    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        init_db(DEFAULT_DB_PATH)
    except Exception as exc:
        logging.warning("SQLite init failed (%s). Proceeding without DB persistence.", exc)

    equity_ctx = resolve_account_equity(
        broker_client=broker_client,
        db_path=DEFAULT_DB_PATH,
        account_id=account_id,
        fallback_cash=account_cash,
    )
    logging.info(
        "ACCOUNT: equity=$%.2f (source=%s) | cash=$%.2f | exposure=%.1f%%",
        equity_ctx.equity,
        equity_ctx.equity_source,
        equity_ctx.cash,
        equity_ctx.exposure_fraction * 100.0,
    )

    # Record realized P&L from actual broker fills before doing anything else.
    # Exits fire broker-side via bracket orders, so without this step the trades
    # table never learns how any position actually turned out — which is why
    # every historical pnl was zero.
    try:
        reconcile_fills(
            broker_client,
            account_id=account_id,
            db_path=DEFAULT_DB_PATH,
            dry_run=dry_run,
        )
    except Exception as exc:
        logging.warning("Fill reconciliation failed (%s). Continuing with cycle.", exc)

    risk_cfg = load_risk_config()
    # The binding cap is whichever of the two limits is tighter: build_order_plan
    # sizes against config.max_position_pct_of_equity, enforce_risk_limits then
    # re-checks against risk_cfg.max_position_size. Screening on the tighter one
    # keeps the screen honest about what will actually fill.
    position_cap_fraction = min(0.20, float(risk_cfg.max_position_size))

    held = _held_symbols(broker_client)
    if held:
        logging.info("CURRENTLY HELD: %s", ", ".join(sorted(held)))

    if universe is not None:
        logging.info("UNIVERSE: %s symbols from explicit override", len(universe))
    elif not screen_affordability:
        from logic.universe import get_symbols

        universe = get_symbols(universe_scope)
        logging.info("UNIVERSE: scope=%s, %s symbols, affordability screen OFF",
                     universe_scope, len(universe))
    else:
        universe, verdicts = build_tradeable_universe(
            scope=universe_scope,
            account_equity=equity_ctx.equity,
            max_position_fraction=position_cap_fraction,
            always_include=held,
        )
        affordable = sum(1 for v in verdicts if v.affordable)
        logging.info(
            "UNIVERSE SCREEN: scope=%s | %s/%s affordable at $%.2f/position "
            "(%.0f%% of $%.2f equity)\n%s",
            universe_scope,
            affordable,
            len(verdicts),
            equity_ctx.equity * position_cap_fraction,
            position_cap_fraction * 100.0,
            equity_ctx.equity,
            format_affordability_table(verdicts),
        )
        if not universe:
            logging.error(
                "No affordable symbols and no open positions. Nothing to do — "
                "the account needs more capital or a cheaper universe scope."
            )
            return

    logging.info("TRADING UNIVERSE (%s): %s", len(universe), ", ".join(universe))

    config = ExecutionConfig(
        execution_mode=execution_mode,
        allow_short_selling=False,
        dry_run=dry_run,
        base_risk_pct=base_risk_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        max_position_pct_of_equity=20.0,
        min_confidence=min_confidence,
        min_prob_up=min_prob_up,
        debug_force_strongest_signal=debug_force_strongest_signal,
    )

    # Protect existing positions FIRST, every cycle, regardless of signals or
    # universe membership. This is the guardrail for the Mar-24 failure mode:
    # any open long lacking an active exit order gets a fresh GTC OCO exit so
    # positions never sit unmanaged (and never silently pin portfolio exposure).
    try:
        exit_results = reconcile_position_exits(
            broker_client=broker_client,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            dry_run=dry_run,
            verbose=True,
            exit_style="stop",  # downside-only: protect the floor, let winners run
        )
        attached = sum(1 for r in exit_results if r.action == "attached")
        logging.info(
            "EXIT RECONCILE SUMMARY: %s positions checked, %s exits attached",
            len(exit_results),
            attached,
        )
    except Exception as exc:
        logging.warning("Exit reconciliation failed (%s). Continuing with cycle.", exc)

    logging.info("Generating signals...")
    all_signals = generate_signals(universe, config)
    decision_signals = filter_signals_by_thresholds(
        all_signals,
        min_confidence=config.min_confidence,
        min_prob_up=config.min_prob_up,
    )
    directional_candidates = [
        sig for sig in decision_signals if sig.signal_type in {"buy", "sell"}
    ]
    hold_decisions = [sig for sig in decision_signals if sig.signal_type == "hold"]

    if debug_force_strongest_signal:
        strongest_signal = max(
            (sig for sig in all_signals if sig.signal_type != "hold"),
            key=lambda sig: (float(sig.confidence), float(sig.prob_profit)),
            default=None,
        )
        if strongest_signal is not None:
            strongest_signal.meta['threshold_decision'] = 'forced'
            strongest_signal.meta['threshold_reason'] = 'debug override forced strongest non-HOLD signal'
            strongest_signal.meta['forced_debug_signal'] = True
            if strongest_signal not in decision_signals:
                decision_signals.append(strongest_signal)
            if strongest_signal not in directional_candidates:
                directional_candidates.append(strongest_signal)
            logging.info(
                "DEBUG OVERRIDE: forcing strongest non-HOLD signal | symbol=%s | normalized=%s | confidence=%.4f | prob_profit=%.4f",
                strongest_signal.symbol,
                strongest_signal.meta.get('normalized_signal_label', strongest_signal.signal_type).upper(),
                float(strongest_signal.confidence),
                float(strongest_signal.prob_profit),
            )

    logging.info(
        "SIGNALS GENERATED: %s total, %s directional candidates, %s hold decisions",
        len(all_signals),
        len(directional_candidates),
        len(hold_decisions),
    )
    for sig in directional_candidates:
        asset = lookup_asset(sig.symbol)
        logging.info(
            "  %-5s %-4s conf=%.2f prob=%.2f | %s",
            sig.symbol,
            str(sig.signal_type).upper(),
            float(sig.confidence),
            float(sig.prob_profit),
            f"{asset.sector} / {asset.underlying}" if asset else "unclassified",
        )

    # Phase 0 shadow logging: build a TradeThesis per signal and write it to
    # trade_logs/theses/<date>.jsonl. Read-only with respect to trading — nothing
    # downstream consumes these yet. This exists so the thesis shape can be
    # validated against real market days before Phase 1 depends on it.
    theses_logged = shadow_log_theses(all_signals, run_date=today_str)
    logging.info("THESES SHADOW-LOGGED: %s", theses_logged)

    if directional_candidates:
        for sig in directional_candidates:
            symbol = str(getattr(sig, "symbol", ""))
            if symbol:
                _log_options_candidates(symbol)
    else:
        logging.info("OPTIONS CANDIDATES FOR TODAY: none")

    logging.info("Loading portfolio state...")
    position_states = get_position_states(
        universe=universe,
        config=config,
        broker_client=broker_client,
        sim_portfolio={},
    )

    # Long-only book: a SELL on something we do not hold is unexecutable and
    # would be dropped. Route those into long positions in inverse funds, and
    # route a BUY on an underlying whose inverse we hold into closing it.
    if enable_inverse_routing:
        broker_states = broker_client.get_position_states([])
        known_states = {**broker_states, **position_states}
        routing = route_signals(decision_signals, known_states)
        decision_signals = routing.signals
        logging.info("INVERSE ROUTING: %s\n%s",
                     routing.summary(), format_routing_table(routing.outcomes))

        extra = [s for s in routing.extra_symbols if s not in position_states]
        if extra:
            # Proxy tickers need position state too, or decide_action reads them
            # as absent and treats an existing holding as flat.
            logging.info("Loading portfolio state for routed symbols: %s", ", ".join(extra))
            position_states.update(
                get_position_states(
                    universe=extra,
                    config=config,
                    broker_client=broker_client,
                    sim_portfolio={},
                )
            )

    logging.info("Executing trading cycle...")
    decision_log = run_trading_cycle(
        signals=decision_signals,
        position_states=position_states,
        config=config,
        account_cash=account_cash,
        broker_client=broker_client,
        sim_portfolio={},
        db_path=DEFAULT_DB_PATH,
        account_id=account_id,
    )

    summary = _summarize_decisions(decision_log)
    orders_attempted = sum(1 for entry in decision_log if entry.action in {"buy", "sell"})
    orders_submitted = sum(1 for entry in decision_log if getattr(entry, "executed", False))
    logging.info("ORDERS ATTEMPTED: %s", orders_attempted)
    logging.info("ORDERS SUBMITTED: %s", orders_submitted)

    try:
        insert_decisions(account_id=account_id, entries=decision_log, db_path=DEFAULT_DB_PATH)
    except Exception as exc:
        logging.warning("Decision log persistence failed (%s).", exc)

    logging.info("Decision summary: %s", summary)
    logging.info("END DAILY TRADING RUN")

    _save_last_run_date(today_str)
