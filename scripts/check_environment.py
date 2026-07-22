"""Environment audit utility for Finance_project.

Checks major runtime dependencies, prints versions, and exits non-zero when
required packages are missing.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

DEBUG_LOG_PATH = Path("debug-73ea0c.log")
DEBUG_SESSION_ID = "73ea0c"

MAJOR_DEPENDENCIES: Dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "yfinance": "yfinance",
    "python-dotenv": "dotenv",
    "tabulate": "tabulate",
    "alpaca-py": "alpaca",
    "emcee": "emcee",
}


def _debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "id": f"log_{uuid.uuid4().hex}",
        "timestamp": int(__import__("time").time() * 1000),
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


def _probe_package(import_name: str) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        return True, str(version)
    except ModuleNotFoundError:
        return False, "missing"
    except Exception:
        return False, "error"


def main() -> int:
    # FIX: Force stdout/stderr to UTF-8 to prevent UnicodeEncodeError on Windows environments.
    # Without this, console prints containing emojis (e.g., 🔵, 📊) crash silently
    # under standard cp1252 encoding, causing the signal pipeline to fail with 0 signals.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    run_id = f"check-env-{uuid.uuid4().hex[:8]}"
    print("Finance_project environment check")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', '<not set>')}")
    print("")

    missing: List[str] = []
    errors: List[str] = []
    versions: Dict[str, str] = {}

    for display_name, import_name in MAJOR_DEPENDENCIES.items():
        ok, status = _probe_package(import_name)
        if ok:
            versions[display_name] = status
            print(f"[OK] {display_name}: {status}")
        elif status == "missing":
            missing.append(display_name)
            print(f"[MISSING] {display_name}")
        else:
            errors.append(display_name)
            print(f"[ERROR] {display_name}: import failed")

    _debug_log(
        run_id=run_id,
        hypothesis_id="H2",
        location="scripts/check_environment.py:main",
        message="Environment dependency probe complete",
        data={
            "python_executable": sys.executable,
            "missing": missing,
            "errors": errors,
            "versions": versions,
        },
    )
    _debug_log(
        run_id=run_id,
        hypothesis_id="H4",
        location="scripts/check_environment.py:main",
        message="Virtual environment state",
        data={"virtual_env": os.environ.get("VIRTUAL_ENV"), "python_executable": sys.executable},
    )

    print("")
    if not missing and not errors:
        print("Summary: all required dependencies are available.")
        return 0

    if missing:
        print("Missing packages:")
        for name in missing:
            print(f"  - {name}")
    if errors:
        print("Packages with import errors:")
        for name in errors:
            print(f"  - {name}")

    print("")
    print("Recommended command:")
    print("  python -m pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
