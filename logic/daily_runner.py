from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Literal

from zoneinfo import ZoneInfo

from logic.broker_client import create_broker_client
from logic.data_structures import ExecutionConfig
from logic.execution_engine import run_trading_cycle
from logic.options_engine import (
    filter_expirations_by_dte,
    get_option_chain,
    select_call_contract,
    select_put_contract,
)
from logic.portfolio_state import get_position_states
from logic.position_reconciler import reconcile_position_exits
from logic.signal_engine import (
    filter_signals_by_thresholds,
    generate_signals,
)
from logic.sqlite_store import init_db, insert_decisions
from logic.thesis import shadow_log_theses


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

    if universe is None:
        universe = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

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

    summary = broker_client.get_account_summary()
    account_cash = float(summary.get("cash", 0.0) or 0.0)
    account_id = str(summary.get("account_id") or "paper-default")

    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        init_db(DEFAULT_DB_PATH)
    except Exception as exc:
        logging.warning("SQLite init failed (%s). Proceeding without DB persistence.", exc)

    logging.info("Loading portfolio state...")
    position_states = get_position_states(
        universe=universe,
        config=config,
        broker_client=broker_client,
        sim_portfolio={},
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
