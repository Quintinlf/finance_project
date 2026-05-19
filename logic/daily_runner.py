from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from zoneinfo import ZoneInfo

from alpaca_exercises import connect_trading_client, get_account_summary, load_alpaca_creds
from data_structures import ExecutionConfig
from execution_engine import run_trading_cycle
from options_engine import (
    filter_expirations_by_dte,
    get_option_chain,
    select_call_contract,
    select_put_contract,
)
from portfolio_state import get_position_states
from signal_engine import filter_signals_by_thresholds, generate_signals
from sqlite_store import init_db, insert_decisions


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
    if eastern_now.weekday() >= 5:
        return False

    if trading_client is None:
        return True

    try:
        from alpaca.trading.requests import GetCalendarRequest  # type: ignore

        req = GetCalendarRequest(
            start=eastern_now.date().isoformat(),
            end=eastern_now.date().isoformat(),
        )
        calendar = trading_client.get_calendar(req)
        return len(calendar) > 0
    except Exception as exc:
        logging.warning("Market calendar check failed (%s). Falling back to weekday check.", exc)
        return True


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
    execution_mode: str = "paper",
    dry_run: bool = False,
    min_confidence: float = 0.50,
    min_prob_up: float = 0.50,
    tp_pct: float = 0.04,
    sl_pct: float = 0.02,
    base_risk_pct: float = 2.0,
    universe: Optional[List[str]] = None,
) -> None:
    logging.info("START DAILY TRADING RUN")

    eastern_now = _get_eastern_now()
    logging.info("DATE/TIME: %s", eastern_now.isoformat())

    trading_client = None
    try:
        creds = load_alpaca_creds()
        trading_client = connect_trading_client(creds, paper=(execution_mode != "live"))
    except Exception as exc:
        logging.warning("Alpaca credentials unavailable (%s). Running without broker client.", exc)

    if not _is_market_day(trading_client):
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
    )

    logging.info("Generating signals...")
    all_signals = generate_signals(universe, config)
    actionable_signals = filter_signals_by_thresholds(
        all_signals,
        min_confidence=config.min_confidence,
        min_prob_up=config.min_prob_up,
    )
    logging.info("SIGNALS GENERATED: %s total, %s pass thresholds", len(all_signals), len(actionable_signals))

    if actionable_signals:
        for sig in actionable_signals:
            symbol = str(getattr(sig, "symbol", ""))
            if symbol:
                _log_options_candidates(symbol)
    else:
        logging.info("OPTIONS CANDIDATES FOR TODAY: none")

    account_cash = 0.0
    account_id = None
    if trading_client is not None:
        summary = get_account_summary(trading_client)
        account_cash = float(summary.get("cash", 0.0) or 0.0)
        try:
            acct = trading_client.get_account()
            account_id = str(getattr(acct, "id", "") or "paper-default")
        except Exception:
            account_id = "paper-default"

    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        init_db(DEFAULT_DB_PATH)
    except Exception as exc:
        logging.warning("SQLite init failed (%s). Proceeding without DB persistence.", exc)

    logging.info("Loading portfolio state...")
    position_states = get_position_states(
        universe=universe,
        config=config,
        alpaca_client=trading_client if config.execution_mode in ["paper", "live"] else None,
        sim_portfolio={},
    )

    logging.info("Executing trading cycle...")
    decision_log = run_trading_cycle(
        signals=actionable_signals,
        position_states=position_states,
        config=config,
        account_cash=account_cash,
        alpaca_client=trading_client if config.execution_mode in ["paper", "live"] else None,
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
