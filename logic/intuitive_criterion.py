"""
Intuitive Criterion (IC) filter for signal validation.

A signal survives only if at least one rational non-manipulator type can justify
deviating from equilibrium under current payoff assumptions.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from data_structures import Signal
from game_utils import MARKET_TYPES


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        uniform = 1.0 / float(len(clipped))
        return {k: uniform for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def _extract_equilibrium(signal: Signal) -> Dict[str, float]:
    raw = signal.meta.get("equilibrium_payoffs", {})
    if hasattr(raw, "as_dict"):
        raw = raw.as_dict()
    if not isinstance(raw, dict):
        raw = {}

    return {
        "t_trend": float(raw.get("t_trend", 0.0)),
        "t_manipulator": float(raw.get("t_manipulator", 0.0)),
        "t_exhausted": float(raw.get("t_exhausted", 0.0)),
        "t_range": float(raw.get("t_range", 0.0)),
    }


def _extract_deviation(signal: Signal) -> Dict[str, float]:
    by_type = signal.meta.get("deviation_payoff_by_type", {})
    if isinstance(by_type, dict) and all(k in by_type for k in MARKET_TYPES):
        return {k: float(by_type[k]) for k in MARKET_TYPES}

    deviation_raw = signal.meta.get("deviation_payoff", {})
    if hasattr(deviation_raw, "as_dict"):
        deviation_raw = deviation_raw.as_dict()
    if not isinstance(deviation_raw, dict):
        deviation_raw = {}

    net = float(deviation_raw.get("net_payoff_manipulator", 0.0))
    bb_z = float(signal.meta.get("bb_z_score", 0.0))
    trend_hint = float(signal.meta.get("ensemble_forecast_return", 0.0))

    return {
        "t_trend": trend_hint - 0.20 * abs(bb_z),
        "t_manipulator": net,
        "t_exhausted": (-trend_hint * 0.50) + (0.20 * net),
        "t_range": (0.30 * abs(trend_hint)) - (0.15 * abs(bb_z)),
    }


def survives_intuitive_criterion(signal: Signal) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Apply Peters-style elimination logic to a signal.

    Rule: eliminate type t if m(t, A) < x(t).
    """
    prior_beliefs = signal.type_beliefs or {k: 1.0 / len(MARKET_TYPES) for k in MARKET_TYPES}
    prior_beliefs = _normalize({k: float(prior_beliefs.get(k, 0.0)) for k in MARKET_TYPES})

    equilibrium = _extract_equilibrium(signal)
    deviation = _extract_deviation(signal)

    eliminated: Dict[str, Dict[str, float]] = {}
    survivors: Dict[str, float] = {}

    for market_type in MARKET_TYPES:
        m_value = float(deviation.get(market_type, 0.0))
        x_value = float(equilibrium.get(market_type, 0.0))
        if m_value < x_value:
            eliminated[market_type] = {"m": m_value, "x": x_value}
        else:
            survivors[market_type] = float(prior_beliefs.get(market_type, 0.0))

    details: Dict[str, Any] = {
        "prior_beliefs": prior_beliefs,
        "equilibrium_payoffs": equilibrium,
        "deviation_payoffs": deviation,
        "eliminated_types": eliminated,
        "surviving_types": sorted(survivors.keys()),
    }

    if not survivors:
        details["posterior_beliefs"] = {k: 0.0 for k in MARKET_TYPES}
        return False, "All rational market types eliminated by IC", details

    posterior = _normalize(survivors)
    details["posterior_beliefs"] = posterior

    if set(survivors.keys()) == {"t_manipulator"}:
        return False, "Only manipulator type survives IC", details

    return True, "Signal survives intuitive criterion", details
