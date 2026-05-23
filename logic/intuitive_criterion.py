"""
Intuitive Criterion (IC) filter for signal validation.

A signal survives only if at least one rational non-manipulator type can justify
deviating from equilibrium under current payoff assumptions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from logic.data_structures import Signal
from logic.game_utils import MARKET_TYPES


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        uniform = 1.0 / float(len(clipped))
        return {k: uniform for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def _entropy(weights: Dict[str, float]) -> float:
    normalized = _normalize(weights)
    if not normalized:
        return 0.0
    return float(-sum(v * math.log(max(v, 1e-12)) for v in normalized.values()))


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    p_n = _normalize(p)
    q_n = _normalize(q)
    keys = set(p_n.keys()) | set(q_n.keys())
    if not keys:
        return 0.0
    kl = 0.0
    for key in keys:
        p_val = max(float(p_n.get(key, 0.0)), 1e-12)
        q_val = max(float(q_n.get(key, 0.0)), 1e-12)
        kl += p_val * (math.log(p_val) - math.log(q_val))
    return float(max(kl, 0.0))


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    p_n = _normalize(p)
    q_n = _normalize(q)
    keys = set(p_n.keys()) | set(q_n.keys())
    if not keys:
        return 0.0
    midpoint = {k: 0.5 * (p_n.get(k, 0.0) + q_n.get(k, 0.0)) for k in keys}
    return float(0.5 * _kl_divergence(p_n, midpoint) + 0.5 * _kl_divergence(q_n, midpoint))


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
        details["prior_entropy"] = _entropy(prior_beliefs)
        details["posterior_entropy"] = 0.0
        details["kl_divergence"] = None
        details["js_divergence"] = None
        details["eliminated_types_count"] = len(eliminated)
        details["surviving_types_count"] = 0
        details["regime_instability_tag"] = "unstable"
        details["uncertainty_recommendation"] = "no_trade_candidate"

        signal.meta["kl_divergence"] = None
        signal.meta["js_divergence"] = None
        signal.meta["prior_entropy"] = details["prior_entropy"]
        signal.meta["posterior_entropy"] = details["posterior_entropy"]
        signal.meta["regime_instability_tag"] = details["regime_instability_tag"]
        return False, "All rational market types eliminated by IC", details

    posterior = _normalize(survivors)
    details["posterior_beliefs"] = posterior

    prior_entropy = _entropy(prior_beliefs)
    posterior_entropy = _entropy(posterior)
    kl_div = _kl_divergence(posterior, prior_beliefs)
    js_div = _js_divergence(prior_beliefs, posterior)
    eliminated_count = len(eliminated)
    surviving_count = len(survivors)

    details["prior_entropy"] = float(prior_entropy)
    details["posterior_entropy"] = float(posterior_entropy)
    details["kl_divergence"] = float(kl_div)
    details["js_divergence"] = float(js_div)
    details["eliminated_types_count"] = eliminated_count
    details["surviving_types_count"] = surviving_count
    details["regime_instability_tag"] = "unstable" if (posterior_entropy > 1.0 or surviving_count <= 1) else "stable"
    details["uncertainty_recommendation"] = "caution" if details["regime_instability_tag"] == "unstable" else "trade"

    signal.meta["kl_divergence"] = float(kl_div)
    signal.meta["js_divergence"] = float(js_div)
    signal.meta["prior_entropy"] = float(prior_entropy)
    signal.meta["posterior_entropy"] = float(posterior_entropy)
    signal.meta["regime_instability_tag"] = details["regime_instability_tag"]

    if set(survivors.keys()) == {"t_manipulator"}:
        return False, "Only manipulator type survives IC", details

    return True, "Signal survives intuitive criterion", details
