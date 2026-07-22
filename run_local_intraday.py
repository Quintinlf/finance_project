"""
Local laptop-friendly intraday options loop.

While the market is open, polls the signal pipeline every 15 minutes and
routes directional signals through the options contract resolver
(logic.options_engine.get_target_option_contract) and options order router
(logic.execution_engine.route_options_order).

SAFETY: EXECUTION_MODE=live is rejected — this script only ever runs against
simulation or paper. Options order submission itself is opt-in via
ENABLE_OPTIONS_EXECUTION=true; by default every resolved contract is only
logged (dry_run), never submitted.

Meant to be run manually in a terminal on your own machine (not via
GitHub Actions, which already runs the equity daily cycle on a cron).
Press Ctrl+C to stop; the loop exits cleanly on KeyboardInterrupt.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from logic.broker_client import create_broker_client
from logic.data_structures import ExecutionConfig
from logic.execution_engine import route_options_order
from logic.options_engine import get_target_option_contract
from logic.signal_engine import filter_signals_by_thresholds, generate_signals

POLL_INTERVAL_SECONDS = 15 * 60
DEFAULT_UNIVERSE = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _sleep_until_next_check(trading_client) -> None:
    """Sleep until the next market open, or POLL_INTERVAL_SECONDS if that's sooner."""
    try:
        clock = trading_client.get_clock()
        next_open = getattr(clock, "next_open", None)
        if next_open is not None:
            seconds_until_open = (next_open - datetime.now(timezone.utc)).total_seconds()
            if 0 < seconds_until_open < POLL_INTERVAL_SECONDS:
                logging.info(
                    "Market closed. Next open in %.0f minutes; sleeping until then.",
                    seconds_until_open / 60,
                )
                time.sleep(max(1.0, seconds_until_open))
                return
    except Exception:
        pass
    logging.info("Market closed. Checking again in 15 minutes.")
    time.sleep(POLL_INTERVAL_SECONDS)


def run_intraday_loop() -> None:
    configure_logging()

    execution_mode_raw = os.getenv("EXECUTION_MODE", "paper").strip().lower()
    if execution_mode_raw == "live":
        logging.warning("EXECUTION_MODE=live is not permitted for this script; forcing 'paper'.")
    execution_mode = "simulation" if execution_mode_raw == "simulation" else "paper"

    enable_options_execution = os.getenv("ENABLE_OPTIONS_EXECUTION", "false").strip().lower() in {
        "1", "true", "yes", "y",
    }

    universe_env = os.getenv("UNIVERSE", "").strip()
    universe = [s.strip().upper() for s in universe_env.split(",") if s.strip()] or DEFAULT_UNIVERSE

    config = ExecutionConfig(
        execution_mode=execution_mode,
        dry_run=not enable_options_execution,
    )

    broker_client = create_broker_client(execution_mode=execution_mode)
    trading_client = getattr(broker_client, "_trading_client", None)

    logging.info(
        "Starting local intraday loop | mode=%s | options_execution=%s | universe=%s",
        execution_mode,
        enable_options_execution,
        universe,
    )

    try:
        while True:
            is_open = True
            if trading_client is not None:
                try:
                    clock = trading_client.get_clock()
                    is_open = bool(getattr(clock, "is_open", True))
                except Exception as exc:
                    logging.warning("Could not fetch market clock (%s); assuming closed.", exc)
                    is_open = False

            if not is_open:
                _sleep_until_next_check(trading_client)
                continue

            logging.info("Market open. Generating signals...")
            all_signals = generate_signals(universe, config)
            decision_signals = filter_signals_by_thresholds(
                all_signals,
                min_confidence=config.min_confidence,
                min_prob_up=config.min_prob_up,
            )
            directional = [s for s in decision_signals if s.signal_type in {"buy", "sell"}]

            if not directional:
                logging.info("No directional signals this cycle.")

            for sig in directional:
                direction = "LONG" if sig.signal_type == "buy" else "SHORT"
                contract = get_target_option_contract(sig.symbol, direction, expiry_days_out=7)
                if contract is None:
                    logging.info("No option contract resolved for %s (%s)", sig.symbol, direction)
                    continue

                logging.info("Signal %s -> %s contract %s", sig.symbol, direction, contract)
                order_result = route_options_order(
                    contract_symbol=contract,
                    side="buy",
                    qty=1,
                    broker_client=broker_client,
                    dry_run=config.dry_run,
                )
                if order_result.get("error_message"):
                    logging.info("Options order not executed: %s", order_result["error_message"])
                else:
                    logging.info("Options order submitted: %s", order_result.get("broker_order_id"))

            logging.info("Cycle complete. Sleeping %s minutes.", POLL_INTERVAL_SECONDS // 60)
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Shutting down intraday loop cleanly.")
        return


if __name__ == "__main__":
    run_intraday_loop()
