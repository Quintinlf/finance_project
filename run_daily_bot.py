import logging
import os
import sys
import json
import importlib
import uuid
from pathlib import Path
from typing import Literal, cast

DEBUG_LOG_PATH = Path("debug-73ea0c.log")
DEBUG_SESSION_ID = "73ea0c"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "id": f"log_{uuid.uuid4().hex}",
        "timestamp": int(datetime_now_ms()),
        "location": location,
        "message": message,
        "data": data,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # #endregion


def datetime_now_ms() -> int:
    from datetime import datetime, timezone

    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _verify_required_packages() -> tuple[bool, list[str]]:
    required = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "plotly",
        "yfinance",
        "dotenv",
        "tabulate",
        "alpaca",
        "emcee",
    ]
    missing: list[str] = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ModuleNotFoundError:
            missing.append(pkg)
    return (len(missing) == 0, missing)


def main() -> int:
    # FIX: Force stdout/stderr to UTF-8 to prevent UnicodeEncodeError on Windows environments.
    # Without this, console prints containing emojis (e.g., 🔵, 📊) crash silently
    # under standard cp1252 encoding, causing the signal pipeline to fail with 0 signals.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    configure_logging()
    run_id = f"env-check-{uuid.uuid4().hex[:8]}"

    has_deps, missing = _verify_required_packages()
    # #region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H1",
        location="run_daily_bot.py:main",
        message="Dependency preflight result",
        data={"python_executable": sys.executable, "missing_count": len(missing), "missing": missing},
    )
    # #endregion
    if not has_deps:
        # #region agent log
        _debug_log(
            run_id=run_id,
            hypothesis_id="H3",
            location="run_daily_bot.py:main",
            message="Startup halted due to missing dependencies",
            data={"missing": missing},
        )
        # #endregion
        for name in missing:
            print(f"MISSING DEPENDENCY: {name}")
        print("Install required packages with: python -m pip install -r requirements.txt")
        return 1

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

    # UNIVERSE is an explicit ticker override and skips the affordability screen.
    # UNIVERSE_SCOPE picks a named scope from logic/universe.py instead
    # (equities | commodities | agriculture | energy | metals | all).
    universe_env = os.getenv("UNIVERSE", "").strip()
    universe = [s.strip().upper() for s in universe_env.split(",") if s.strip()] or None
    universe_scope = os.getenv("UNIVERSE_SCOPE", "all").strip().lower()
    screen_env = os.getenv("SCREEN_AFFORDABILITY", "true").strip().lower()
    screen_affordability = screen_env in {"1", "true", "yes", "y"}

    # Long-only book, so bearish signals are expressed by buying inverse funds
    # rather than being dropped. Set INVERSE_ROUTING=false to go back to
    # discarding unexecutable SELLs.
    inverse_env = os.getenv("INVERSE_ROUTING", "true").strip().lower()
    enable_inverse_routing = inverse_env in {"1", "true", "yes", "y"}

    from logic.daily_runner import run_daily_trading_cycle

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
        universe_scope=universe_scope,
        screen_affordability=screen_affordability,
        enable_inverse_routing=enable_inverse_routing,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None) or str(exc).replace("No module named ", "").strip("'\"")
        print(f"MISSING DEPENDENCY: {missing_name}")
        print("Install required packages with: python -m pip install -r requirements.txt")
        sys.exit(1)
    except Exception:
        logging.exception("Daily trading run failed")
        sys.exit(1)
