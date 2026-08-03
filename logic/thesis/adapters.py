"""
Phase 0 compatibility adapters: existing ``Signal`` -> ``TradeThesis``.

Phase 0 changes nothing about how trades are decided. The equity path still runs
on ``Signal``; this module produces a thesis *alongside* it and shadow-logs it to
JSONL so the shape can be validated against real market days before anything
downstream depends on it.

Shadow logs are written to ``trade_logs/theses/<date>.jsonl``, deliberately NOT
to the SQLite decision store. The DecisionTrace schema migration (new tables vs.
widening ``decisions``) is still an open question in the architecture doc, and
guessing at it here would bake in the wrong answer.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from logic.data_structures import Signal
from logic.thesis.context import MarketDirection
from logic.thesis.models import (
    AnalystContribution,
    ConfidenceAttribution,
    TradeThesis,
)

logger = logging.getLogger(__name__)

# The one analyst that exists today: the current ensemble/technical stack.
TECHNICAL_ANALYST_ID = "technical_ensemble"

# Neutral prior. 0.5 means "no view" before any analyst speaks, which makes each
# analyst's delta read as displacement from indifference.
NEUTRAL_BASE_CONFIDENCE = 0.5

DEFAULT_THESIS_LOG_DIR = Path("trade_logs") / "theses"

_DIRECTION_BY_SIGNAL_TYPE = {"buy": "LONG", "sell": "SHORT", "hold": "FLAT"}

# Default horizon: signal_engine builds its expected-return path with horizon=5.
DEFAULT_HOLDING_PERIOD_DAYS = 5


def _classify_time_horizon(holding_period_days: int) -> str:
    if holding_period_days <= 1:
        return "INTRADAY"
    if holding_period_days <= 10:
        return "SWING"
    if holding_period_days <= 60:
        return "POSITION"
    return "LONG_TERM"


def _classify_conviction(confidence: float) -> str:
    if confidence < 0.55:
        return "LOW"
    if confidence < 0.70:
        return "MEDIUM"
    return "HIGH"


def _classify_volatility(meta: Dict[str, Any]) -> str:
    """Read volatility expectation off the MarketState mixture in Signal.meta."""
    market_state = meta.get("market_regime") or {}
    try:
        prob_high_vol = float(market_state.get("prob_high_vol", 0.0))
    except (TypeError, ValueError):
        prob_high_vol = 0.0

    if prob_high_vol >= 0.50:
        return "HIGH"
    if prob_high_vol >= 0.30:
        return "MEDIUM"
    return "LOW"


def _expected_return_from_meta(meta: Dict[str, Any]) -> float:
    """Cumulative expected return of the UNDERLYING over the forecast path.

    Sign is the underlying's, not the position's: a SHORT thesis on a symbol
    forecast to fall carries a negative expected_return. Position-level PnL is
    the Expression layer's concern.
    """
    path = meta.get("expected_return_path")
    if isinstance(path, (list, tuple)) and path:
        try:
            return float(sum(float(step) for step in path))
        except (TypeError, ValueError):
            pass

    try:
        return float(meta.get("ensemble_forecast_return", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_supporting_evidence(signal: Signal) -> List[str]:
    meta = signal.meta or {}
    evidence: List[str] = []

    def _add(label: str, key: str, fmt: str = "{:.4f}") -> None:
        value = meta.get(key)
        if value is None:
            return
        try:
            evidence.append(f"{label}={fmt.format(float(value))}")
        except (TypeError, ValueError):
            evidence.append(f"{label}={value}")

    _add("rsi", "rsi_value", "{:.2f}")
    _add("bb_z_score", "bb_z_score")
    _add("ensemble_forecast_return", "ensemble_forecast_return")
    _add("belief_entropy", "belief_entropy")

    if meta.get("signals_agree") is not None:
        evidence.append(f"signals_agree={bool(meta.get('signals_agree'))}")
    if meta.get("bb_signal"):
        evidence.append(f"bb_signal={meta.get('bb_signal')}")

    return evidence


def signal_to_contribution(signal: Signal) -> AnalystContribution:
    """Wrap the current ensemble output as the single technical analyst's contribution.

    The delta is ``confidence - NEUTRAL_BASE_CONFIDENCE`` so that adding it to the
    neutral base reproduces the signal's own confidence exactly — Phase 0 must not
    move the number, only explain it.
    """
    meta = dict(signal.meta or {})
    # NOTE: "market_regime" is the persisted Signal.meta key written by
    # signal_engine and read back from stored decisions. It holds a MarketState
    # mixture. The key name is intentionally left alone -- renaming it would
    # invalidate existing decision history.
    market_direction = MarketDirection.from_market_state(
        meta.get("market_regime") or {},
        directional_hint=signal.signal_type,
    )

    return AnalystContribution(
        analyst_id=TECHNICAL_ANALYST_ID,
        symbol=signal.symbol,
        confidence_delta=float(signal.confidence) - NEUTRAL_BASE_CONFIDENCE,
        evidence=_build_supporting_evidence(signal),
        tags=["technical", "ensemble"],
        suggested_market_direction=market_direction,
        horizon_hint=_classify_time_horizon(DEFAULT_HOLDING_PERIOD_DAYS),
        meta={
            "prob_profit": float(signal.prob_profit),
            "signal_type": signal.signal_type,
        },
    )


def signal_to_thesis(
    signal: Signal,
    *,
    holding_period_days: int = DEFAULT_HOLDING_PERIOD_DAYS,
) -> TradeThesis:
    """Build a ``TradeThesis`` from an existing ``Signal``.

    Lossless with respect to the trading decision: the resulting thesis carries
    the same confidence the signal did, and nothing consumes it in Phase 0.
    Macro regime, structural themes, and catalysts are left empty because no
    analyst produces them yet — an empty list is honest, a guessed one is not.
    """
    meta = dict(signal.meta or {})
    contribution = signal_to_contribution(signal)
    attribution = ConfidenceAttribution.from_contributions(
        base=NEUTRAL_BASE_CONFIDENCE,
        contributions=[contribution],
    )

    direction = _DIRECTION_BY_SIGNAL_TYPE.get(signal.signal_type, "FLAT")
    reasoning_log = [
        f"signal_type={signal.signal_type} -> direction={direction}",
        (
            f"confidence {attribution.final:.4f} = base {attribution.base:.2f} "
            f"+ {TECHNICAL_ANALYST_ID} {contribution.confidence_delta:+.4f}"
        ),
    ]
    threshold_reason = meta.get("threshold_reason")
    if threshold_reason:
        reasoning_log.append(f"threshold: {threshold_reason}")

    return TradeThesis(
        symbol=signal.symbol,
        direction=direction,
        confidence=attribution.final,
        confidence_attribution=attribution,
        expected_return=_expected_return_from_meta(meta),
        expected_holding_period=holding_period_days,
        time_horizon=_classify_time_horizon(holding_period_days),
        conviction=_classify_conviction(attribution.final),
        volatility_expectation=_classify_volatility(meta),
        market_direction=contribution.suggested_market_direction,
        macro_regime=None,
        structural_themes=[],
        catalysts=[],
        supporting_evidence=list(contribution.evidence),
        reasoning_log=reasoning_log,
        analyst_ids=[TECHNICAL_ANALYST_ID],
        source_signal_id=meta.get("signal_id"),
        meta={
            "prob_profit": float(signal.prob_profit),
            "source_signal_type": signal.signal_type,
            "type_beliefs": dict(signal.type_beliefs or {}),
            "phase": "0-shadow",
        },
    )


def shadow_log_theses(
    signals: Sequence[Signal],
    *,
    log_dir: Path = DEFAULT_THESIS_LOG_DIR,
    run_date: Optional[str] = None,
) -> int:
    """Write one JSON line per thesis. Returns how many were written.

    Never raises. This runs inside the live trading cycle, and a shadow-logging
    bug must not be able to stop a real trading day.
    """
    if not signals:
        return 0

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = run_date or datetime.now(timezone.utc).date().isoformat()
        target = log_dir / f"{stamp}.jsonl"

        lines: List[str] = []
        for signal in signals:
            try:
                lines.append(signal_to_thesis(signal).to_json())
            except Exception as exc:  # one bad signal must not drop the batch
                logger.warning(
                    "Thesis shadow-log skipped %s: %s",
                    getattr(signal, "symbol", "<unknown>"),
                    exc,
                )

        if not lines:
            return 0

        with target.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return len(lines)
    except Exception as exc:
        logger.warning("Thesis shadow logging failed (%s). Trading cycle unaffected.", exc)
        return 0


def load_shadow_theses(path: Path) -> List[TradeThesis]:
    """Read back a shadow-log JSONL file (for inspection and round-trip tests)."""
    theses: List[TradeThesis] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                theses.append(TradeThesis.from_dict(json.loads(line)))
    return theses
