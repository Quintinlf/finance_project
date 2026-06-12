import logging
import os
import sys
from typing import Literal, cast

from logic.daily_runner import run_daily_trading_cycle


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> int:
    configure_logging()

    execution_mode_raw = os.getenv("EXECUTION_MODE", "paper").strip().lower()
    if execution_mode_raw not in {"simulation", "paper", "live"}:
        logging.warning("Invalid EXECUTION_MODE=%s; defaulting to paper.", execution_mode_raw)
        execution_mode_raw = "paper"
    execution_mode = cast(Literal["simulation", "paper", "live"], execution_mode_raw)
    dry_run_env = os.getenv("DRY_RUN", "false").strip().lower()
    dry_run = dry_run_env in {"1", "true", "yes", "y"}

    min_conf = float(os.getenv("MIN_CONF", "0.5"))
    min_prob_up = float(os.getenv("MIN_PROB_UP", "0.5"))
    tp_pct = float(os.getenv("TP_PCT", "0.04"))
    sl_pct = float(os.getenv("SL_PCT", "0.02"))
    base_risk_pct = float(os.getenv("BASE_RISK_PCT", "2.0"))
    debug_force_strongest_signal_env = os.getenv("DEBUG_FORCE_STRONGEST_SIGNAL", "false").strip().lower()
    debug_force_strongest_signal = debug_force_strongest_signal_env in {"1", "true", "yes", "y"}

    universe_env = os.getenv("UNIVERSE", "").strip()
    universe = [s.strip().upper() for s in universe_env.split(",") if s.strip()] or None

    run_daily_trading_cycle(
        execution_mode=execution_mode,
        dry_run=dry_run,
        min_confidence=min_conf,
        min_prob_up=min_prob_up,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        base_risk_pct=base_risk_pct,
        debug_force_strongest_signal=debug_force_strongest_signal,
        universe=universe,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logging.exception("Daily trading run failed")
        sys.exit(1)
