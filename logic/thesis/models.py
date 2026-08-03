"""
Phase 0 thesis-layer domain objects.

    AnalystContribution   what one analyst observed, plus its confidence delta
    ConfidenceAttribution explainable breakdown of how confidence was reached
    TradeThesis           directional view + horizon + context (NO instrument)

A ``TradeThesis`` deliberately carries no instrument, strike, or expiry. It says
what we believe about a symbol and over what horizon; how that view gets
expressed (stock, call, spread) is the Expression layer's job in Phase 1.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

from logic.thesis.context import (
    Catalyst,
    MacroRegime,
    MarketDirection,
    StructuralTheme,
)

SCHEMA_VERSION = "1"

# Floating-point slack when checking that deltas reconcile to raw_final.
_RECONCILE_TOLERANCE = 1e-9

Direction = Literal["LONG", "SHORT", "FLAT"]
TimeHorizon = Literal["INTRADAY", "SWING", "POSITION", "LONG_TERM"]
Conviction = Literal["LOW", "MEDIUM", "HIGH"]
VolatilityExpectation = Literal["LOW", "MEDIUM", "HIGH"]

_VALID_DIRECTIONS = {"LONG", "SHORT", "FLAT"}
_VALID_HORIZONS = {"INTRADAY", "SWING", "POSITION", "LONG_TERM"}
_VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}


@dataclass
class AnalystContribution:
    """One analyst's contribution toward a thesis.

    Analysts never place trades. They emit contributions; the synthesizer turns
    a set of contributions into a ``ConfidenceAttribution`` and a thesis.
    """

    analyst_id: str
    symbol: str
    confidence_delta: float
    evidence: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    suggested_market_direction: Optional[MarketDirection] = None
    suggested_macro_regime: Optional[MacroRegime] = None
    suggested_themes: List[StructuralTheme] = field(default_factory=list)
    suggested_catalysts: List[Catalyst] = field(default_factory=list)
    horizon_hint: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.analyst_id:
            raise ValueError("analyst_id is required")
        delta = float(self.confidence_delta)
        if not math.isfinite(delta):
            raise ValueError(f"confidence_delta must be finite, got {delta}")
        self.confidence_delta = delta
        if self.horizon_hint is not None and self.horizon_hint not in _VALID_HORIZONS:
            raise ValueError(
                f"horizon_hint must be one of {sorted(_VALID_HORIZONS)}, "
                f"got {self.horizon_hint}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyst_id": self.analyst_id,
            "symbol": self.symbol,
            "confidence_delta": self.confidence_delta,
            "evidence": list(self.evidence),
            "tags": list(self.tags),
            "suggested_market_direction": (
                self.suggested_market_direction.to_dict()
                if self.suggested_market_direction
                else None
            ),
            "suggested_macro_regime": (
                self.suggested_macro_regime.to_dict()
                if self.suggested_macro_regime
                else None
            ),
            "suggested_themes": [t.to_dict() for t in self.suggested_themes],
            "suggested_catalysts": [c.to_dict() for c in self.suggested_catalysts],
            "horizon_hint": self.horizon_hint,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalystContribution":
        market_direction = payload.get("suggested_market_direction")
        macro_regime = payload.get("suggested_macro_regime")
        return cls(
            analyst_id=payload["analyst_id"],
            symbol=payload["symbol"],
            confidence_delta=float(payload["confidence_delta"]),
            evidence=list(payload.get("evidence") or []),
            tags=list(payload.get("tags") or []),
            suggested_market_direction=(
                MarketDirection.from_dict(market_direction) if market_direction else None
            ),
            suggested_macro_regime=(
                MacroRegime.from_dict(macro_regime) if macro_regime else None
            ),
            suggested_themes=[
                StructuralTheme.from_dict(t) for t in payload.get("suggested_themes") or []
            ],
            suggested_catalysts=[
                Catalyst.from_dict(c) for c in payload.get("suggested_catalysts") or []
            ],
            horizon_hint=payload.get("horizon_hint"),
            meta=dict(payload.get("meta") or {}),
        )


@dataclass
class ConfidenceAttribution:
    """Explainable confidence: which analyst moved the number, and by how much.

    Two finals are stored on purpose:

        raw_final = base + sum(by_analyst.values())   exact, always reconciles
        final     = clip(raw_final, 0.0, 1.0)         what downstream consumes

    Clipping alone would silently break the audit trail: with base 0.50 and
    deltas summing to +0.65, a lone clipped ``final`` of 1.00 no longer matches
    its own deltas, and the 0.15 of overshoot disappears. Keeping both means the
    invariant always holds and ``overflow`` shows how much conviction piled up
    past the ceiling — which is a signal in itself when calibrating analysts.
    """

    base: float
    by_analyst: Dict[str, float] = field(default_factory=dict)
    raw_final: float = 0.0
    final: float = 0.0

    def __post_init__(self) -> None:
        self.base = float(self.base)
        self.by_analyst = {k: float(v) for k, v in self.by_analyst.items()}

        expected_raw = self.base + sum(self.by_analyst.values())
        if abs(expected_raw - float(self.raw_final)) > _RECONCILE_TOLERANCE:
            raise ValueError(
                "ConfidenceAttribution does not reconcile: "
                f"base ({self.base}) + deltas ({sum(self.by_analyst.values())}) "
                f"= {expected_raw}, but raw_final = {self.raw_final}"
            )
        self.raw_final = float(self.raw_final)

        expected_final = _clip_unit(self.raw_final)
        if abs(expected_final - float(self.final)) > _RECONCILE_TOLERANCE:
            raise ValueError(
                f"final must equal clip(raw_final, 0, 1) = {expected_final}, "
                f"got {self.final}"
            )
        self.final = float(self.final)

    @property
    def overflow(self) -> float:
        """Confidence lost to clipping. Zero when raw_final was already in range."""
        return self.raw_final - self.final

    @property
    def was_clipped(self) -> bool:
        return abs(self.overflow) > _RECONCILE_TOLERANCE

    @classmethod
    def from_contributions(
        cls,
        *,
        base: float,
        contributions: Iterable[AnalystContribution],
    ) -> "ConfidenceAttribution":
        """Build attribution by summing per-analyst deltas onto a base prior.

        Repeated ``analyst_id`` values accumulate rather than overwrite, so an
        analyst emitting several contributions for one symbol lands as a single
        net delta.
        """
        by_analyst: Dict[str, float] = {}
        for contribution in contributions:
            by_analyst[contribution.analyst_id] = (
                by_analyst.get(contribution.analyst_id, 0.0)
                + float(contribution.confidence_delta)
            )

        raw_final = float(base) + sum(by_analyst.values())
        return cls(
            base=float(base),
            by_analyst=by_analyst,
            raw_final=raw_final,
            final=_clip_unit(raw_final),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base,
            "by_analyst": dict(self.by_analyst),
            "raw_final": self.raw_final,
            "final": self.final,
            "overflow": self.overflow,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConfidenceAttribution":
        # 'overflow' is derived on read; it is exported for readability only.
        return cls(
            base=float(payload["base"]),
            by_analyst={k: float(v) for k, v in (payload.get("by_analyst") or {}).items()},
            raw_final=float(payload["raw_final"]),
            final=float(payload["final"]),
        )


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class TradeThesis:
    """A directional view on a symbol over a horizon. Carries no instrument."""

    symbol: str
    direction: Direction
    confidence: float
    confidence_attribution: ConfidenceAttribution
    expected_return: float
    expected_holding_period: int
    time_horizon: TimeHorizon
    conviction: Conviction
    volatility_expectation: VolatilityExpectation
    market_direction: Optional[MarketDirection] = None
    macro_regime: Optional[MacroRegime] = None
    structural_themes: List[StructuralTheme] = field(default_factory=list)
    catalysts: List[Catalyst] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    reasoning_log: List[str] = field(default_factory=list)
    analyst_ids: List[str] = field(default_factory=list)
    source_signal_id: Optional[str] = None
    thesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = SCHEMA_VERSION
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")

        self.direction = str(self.direction).upper()
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(_VALID_DIRECTIONS)}, "
                f"got {self.direction}"
            )

        self.time_horizon = str(self.time_horizon).upper()
        if self.time_horizon not in _VALID_HORIZONS:
            raise ValueError(
                f"time_horizon must be one of {sorted(_VALID_HORIZONS)}, "
                f"got {self.time_horizon}"
            )

        for name in ("conviction", "volatility_expectation"):
            value = str(getattr(self, name)).upper()
            if value not in _VALID_LEVELS:
                raise ValueError(
                    f"{name} must be one of {sorted(_VALID_LEVELS)}, got {value}"
                )
            setattr(self, name, value)

        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

        # The thesis confidence IS the attribution's clipped final. If these ever
        # disagree, the explanation no longer explains the number being used.
        if abs(self.confidence - self.confidence_attribution.final) > _RECONCILE_TOLERANCE:
            raise ValueError(
                f"confidence ({self.confidence}) must equal "
                f"confidence_attribution.final ({self.confidence_attribution.final})"
            )

        self.expected_holding_period = int(self.expected_holding_period)
        if self.expected_holding_period < 0:
            raise ValueError(
                f"expected_holding_period must be >= 0, got {self.expected_holding_period}"
            )

        self.expected_return = float(self.expected_return)
        if not math.isfinite(self.expected_return):
            raise ValueError(f"expected_return must be finite, got {self.expected_return}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "thesis_id": self.thesis_id,
            "created_at": self.created_at.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "confidence_attribution": self.confidence_attribution.to_dict(),
            "expected_return": self.expected_return,
            "expected_holding_period": self.expected_holding_period,
            "time_horizon": self.time_horizon,
            "conviction": self.conviction,
            "volatility_expectation": self.volatility_expectation,
            "market_direction": self.market_direction.to_dict() if self.market_direction else None,
            "macro_regime": self.macro_regime.to_dict() if self.macro_regime else None,
            "structural_themes": [t.to_dict() for t in self.structural_themes],
            "catalysts": [c.to_dict() for c in self.catalysts],
            "supporting_evidence": list(self.supporting_evidence),
            "reasoning_log": list(self.reasoning_log),
            "analyst_ids": list(self.analyst_ids),
            "source_signal_id": self.source_signal_id,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradeThesis":
        market_direction = payload.get("market_direction")
        macro_regime = payload.get("macro_regime")
        return cls(
            symbol=payload["symbol"],
            direction=payload["direction"],
            confidence=float(payload["confidence"]),
            confidence_attribution=ConfidenceAttribution.from_dict(
                payload["confidence_attribution"]
            ),
            expected_return=float(payload["expected_return"]),
            expected_holding_period=int(payload["expected_holding_period"]),
            time_horizon=payload["time_horizon"],
            conviction=payload["conviction"],
            volatility_expectation=payload["volatility_expectation"],
            market_direction=MarketDirection.from_dict(market_direction) if market_direction else None,
            macro_regime=MacroRegime.from_dict(macro_regime) if macro_regime else None,
            structural_themes=[
                StructuralTheme.from_dict(t) for t in payload.get("structural_themes") or []
            ],
            catalysts=[Catalyst.from_dict(c) for c in payload.get("catalysts") or []],
            supporting_evidence=list(payload.get("supporting_evidence") or []),
            reasoning_log=list(payload.get("reasoning_log") or []),
            analyst_ids=list(payload.get("analyst_ids") or []),
            source_signal_id=payload.get("source_signal_id"),
            thesis_id=payload.get("thesis_id") or str(uuid.uuid4()),
            created_at=_parse_timestamp(payload.get("created_at")),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
            meta=dict(payload.get("meta") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "TradeThesis":
        return cls.from_dict(json.loads(raw))


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(str(value))
