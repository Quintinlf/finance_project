"""Which feature blocks the live forecast is allowed to use.

The default is the model that was already trading: lagged returns, trailing
volatility, RSI, and Bollinger Bands, with the GP in the ensemble.

After a walk-forward report exists, only a "survives" arm is added, and only
a "fails" arm is removed. "promising" stays as it was: already-live blocks
remain, new blocks are not switched on. A missing or unreadable report
leaves the default in place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

REPORT_PATH = Path("trade_logs") / "research" / "ablation_report.json"

FEATURES_VERSION = "2026-09-29-next-bar"
MODEL_VERSION = "bayesian-gp-holdout-1"

# 2026-09-29 walk-forward, eight symbols, two years, day-block bootstrap.
# TA and volume failed against lagged returns and are not in the live matrix.
# Candle geometry survived that comparison and is included. Regime and lunar
# were positive but their intervals still crossed zero, so they are not added.
# A new report can change this list.
DEFAULT_LIVE_BLOCKS: Tuple[str, ...] = ("lags", "volatility", "candles")

# Arm name -> block that arm is evidence for. Baseline is the reference, not a block.
ARM_TO_BLOCK = {
    "baseline_ta": "ta",
    "baseline_candles": "candles",
    "baseline_volume": "volume",
    "baseline_news": "news",
    "baseline_regime": "regime",
    "baseline_lunar": "lunar",
}


def load_report(path: Path = REPORT_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def live_blocks(report: dict | None = None, *, path: Path = REPORT_PATH) -> List[str]:
    """Blocks for the live Bayesian design matrix, in a stable order."""
    if report is None:
        report = load_report(path)
    arms = (report or {}).get("arms") or {}
    selected = list(DEFAULT_LIVE_BLOCKS)
    for arm_name, block in ARM_TO_BLOCK.items():
        status = str((arms.get(arm_name) or {}).get("status") or "")
        if status == "survives" and block not in selected:
            selected.append(block)
        elif status == "fails" and block in selected:
            selected.remove(block)
    return selected


def use_gp(report: dict | None = None, *, path: Path = REPORT_PATH) -> bool:
    """GP stays off unless a report says it survived against Bayesian.

    The 2026-09-29 comparison was negative, so the fallback is off. A later
    "promising" result does not turn it back on. Only "survives" does.
    """
    if report is None:
        report = load_report(path)
    status = str(((report or {}).get("h6_gp_vs_bayesian") or {}).get("status") or "")
    return status == "survives"


def blocks_label(blocks: Iterable[str]) -> str:
    return "+".join(blocks)
