"""
Thesis layer (Phase 0).

Public surface for the intelligence -> expression handoff. Nothing here places
trades or influences the live equity path; Phase 0 only builds and shadow-logs
theses alongside the existing ``Signal`` flow.

``MarketDirection`` (BULL/BEAR/SIDEWAYS) is distinct from
``logic.game_utils.MarketState`` (a probability mixture over trending / ranging /
volatile). Direction says which way; state says what kind. See
``logic/thesis/context.py``.
"""

from logic.thesis.adapters import (
    NEUTRAL_BASE_CONFIDENCE,
    TECHNICAL_ANALYST_ID,
    load_shadow_theses,
    shadow_log_theses,
    signal_to_contribution,
    signal_to_thesis,
)
from logic.thesis.context import (
    Catalyst,
    CatalystCode,
    MacroRegime,
    MacroRegimeCode,
    MarketDirection,
    MarketDirectionCode,
    StructuralTheme,
    StructuralThemeCode,
)
from logic.thesis.models import (
    SCHEMA_VERSION,
    AnalystContribution,
    ConfidenceAttribution,
    TradeThesis,
)

__all__ = [
    "SCHEMA_VERSION",
    "AnalystContribution",
    "ConfidenceAttribution",
    "TradeThesis",
    "Catalyst",
    "CatalystCode",
    "MacroRegime",
    "MacroRegimeCode",
    "MarketDirection",
    "MarketDirectionCode",
    "StructuralTheme",
    "StructuralThemeCode",
    "NEUTRAL_BASE_CONFIDENCE",
    "TECHNICAL_ANALYST_ID",
    "signal_to_thesis",
    "signal_to_contribution",
    "shadow_log_theses",
    "load_shadow_theses",
]
