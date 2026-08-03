"""
Phase 0 context types for the trading engine architecture.

Four *separate* typed concepts, deliberately not collapsed into one string:

    MarketDirection  directional price view      BULL / BEAR / SIDEWAYS
    MacroRegime      macro policy & cycle        INFLATION_RISING / RATE_CUTS / ...
    StructuralTheme  multi-month narrative       AI_BOOM / NUCLEAR_ENERGY / ...
    Catalyst         discrete near-term trigger  EARNINGS / CPI / FOMC / ...

Each is a small frozen dataclass wrapping an enum ``code`` plus optional
``detail``, so "war" never ends up jammed into a field that also means
"inflation is rising".

Direction vs. state
-------------------
``MarketDirection`` (here) and ``logic.game_utils.MarketState`` answer different
questions and must not be conflated:

    MarketState      is the market trending, ranging, or volatile?  (no direction)
    MarketDirection  which way is it going?                          (BULL / BEAR)

Both classes were briefly named ``MarketDirection``; they were split because a
trending market says nothing about whether it trends up or down.
``MarketDirection.from_market_state()`` bridges one to the other, and refuses to
invent a direction the state mixture does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class MarketDirectionCode(str, Enum):
    """Directional price regime."""

    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class MacroRegimeCode(str, Enum):
    """Macro policy / cycle state."""

    INFLATION_RISING = "INFLATION_RISING"
    INFLATION_FALLING = "INFLATION_FALLING"
    RATE_CUTS = "RATE_CUTS"
    TIGHTENING = "TIGHTENING"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class StructuralThemeCode(str, Enum):
    """Multi-month / multi-year narrative."""

    AI_BOOM = "AI_BOOM"
    NUCLEAR_ENERGY = "NUCLEAR_ENERGY"
    SEMICONDUCTOR_EXPANSION = "SEMICONDUCTOR_EXPANSION"
    DEFENSE_SPENDING = "DEFENSE_SPENDING"
    ENERGY_TRANSITION = "ENERGY_TRANSITION"
    DEGLOBALIZATION = "DEGLOBALIZATION"


class CatalystCode(str, Enum):
    """Discrete near-term trigger."""

    EARNINGS = "EARNINGS"
    CPI = "CPI"
    FOMC = "FOMC"
    JOBS_REPORT = "JOBS_REPORT"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    WAR = "WAR"
    HURRICANE = "HURRICANE"
    REGULATORY = "REGULATORY"


def _coerce_enum(enum_cls, value):
    """Accept an enum member or its string value; raise on anything else."""
    if isinstance(value, enum_cls):
        return value
    return enum_cls(str(value))


@dataclass(frozen=True)
class MarketDirection:
    """Directional price regime with optional supporting confidence."""

    code: MarketDirectionCode
    detail: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _coerce_enum(MarketDirectionCode, self.code))
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @classmethod
    def from_market_state(
        cls,
        state: Mapping[str, float],
        *,
        directional_hint: Optional[str] = None,
    ) -> "MarketDirection":
        """Bridge from ``game_utils.MarketState.as_dict()`` to a directional label.

        The state mixture (prob_trend / prob_range / prob_high_vol) carries no
        direction of its own, so a trending market is only BULL or BEAR once a
        ``directional_hint`` ('buy' / 'sell') says which way. Without a hint a
        trend-dominant state stays UNKNOWN rather than guessing.
        """
        if not state:
            return cls(code=MarketDirectionCode.UNKNOWN)

        prob_trend = float(state.get("prob_trend", 0.0))
        prob_range = float(state.get("prob_range", 0.0))
        dominant = max(state.items(), key=lambda kv: float(kv[1]))[0]

        if dominant == "prob_range":
            return cls(
                code=MarketDirectionCode.SIDEWAYS,
                detail="range-dominant market state",
                confidence=prob_range,
            )

        if dominant == "prob_trend":
            hint = (directional_hint or "").lower()
            if hint == "buy":
                code = MarketDirectionCode.BULL
            elif hint == "sell":
                code = MarketDirectionCode.BEAR
            else:
                code = MarketDirectionCode.UNKNOWN
            return cls(
                code=code,
                detail="trend-dominant market state",
                confidence=prob_trend,
            )

        # High-vol dominant: no directional claim available.
        return cls(
            code=MarketDirectionCode.UNKNOWN,
            detail="high-volatility-dominant market state",
            confidence=float(state.get("prob_high_vol", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "detail": self.detail,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketDirection":
        return cls(
            code=MarketDirectionCode(payload["code"]),
            detail=payload.get("detail"),
            confidence=payload.get("confidence"),
        )


@dataclass(frozen=True)
class MacroRegime:
    """Macro policy / cycle state."""

    code: MacroRegimeCode
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _coerce_enum(MacroRegimeCode, self.code))

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code.value, "detail": self.detail}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MacroRegime":
        return cls(code=MacroRegimeCode(payload["code"]), detail=payload.get("detail"))


@dataclass(frozen=True)
class StructuralTheme:
    """Multi-month / multi-year narrative the symbol participates in."""

    code: StructuralThemeCode
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _coerce_enum(StructuralThemeCode, self.code))

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code.value, "detail": self.detail}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuralTheme":
        return cls(
            code=StructuralThemeCode(payload["code"]), detail=payload.get("detail")
        )


@dataclass(frozen=True)
class Catalyst:
    """Discrete near-term trigger, optionally dated."""

    code: CatalystCode
    event_date: Optional[date] = None
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _coerce_enum(CatalystCode, self.code))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Catalyst":
        raw_date = payload.get("event_date")
        return cls(
            code=CatalystCode(payload["code"]),
            event_date=date.fromisoformat(raw_date) if raw_date else None,
            detail=payload.get("detail"),
        )
