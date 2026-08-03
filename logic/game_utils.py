"""
Game-theoretic utilities for market-type inference and payoff calculations.

This module translates signaling-game concepts into concrete, low-risk helpers
that can be attached to the existing signal generation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import math

import pandas as pd


MARKET_TYPES = ("t_trend", "t_manipulator", "t_exhausted", "t_range")


@dataclass
class MarketState:
    """Probability mixture over market *character*: trending, ranging, or volatile.

    Carries no directional view. A high ``prob_trend`` says the market is
    trending, not which way — see ``logic.thesis.context.MarketDirection`` for
    the BULL/BEAR/SIDEWAYS label, which is a separate concept.
    """

    prob_trend: float
    prob_range: float
    prob_high_vol: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "prob_trend": float(self.prob_trend),
            "prob_range": float(self.prob_range),
            "prob_high_vol": float(self.prob_high_vol),
        }


@dataclass
class TypeEquilibrium:
    """Equilibrium payoff x(t) for each market type."""

    t_trend: float
    t_manipulator: float
    t_exhausted: float
    t_range: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "t_trend": float(self.t_trend),
            "t_manipulator": float(self.t_manipulator),
            "t_exhausted": float(self.t_exhausted),
            "t_range": float(self.t_range),
        }


@dataclass
class SignalDeviation:
    """Deviation payoff m(t, A) components for a single signal event."""

    signal_price_target: float
    hunting_cost: float
    liquidity_profit: float
    net_payoff_manipulator: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "signal_price_target": float(self.signal_price_target),
            "hunting_cost": float(self.hunting_cost),
            "liquidity_profit": float(self.liquidity_profit),
            "net_payoff_manipulator": float(self.net_payoff_manipulator),
        }


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        uniform = 1.0 / float(len(clipped))
        return {k: uniform for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    denom = sum(exps)
    if denom <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [x / denom for x in exps]


def compute_market_state(price_history: pd.DataFrame, period: int = 20) -> MarketState:
    """
    Compute market-character probabilities from recent price action.

    Scores:
        S_trend = trend_strength / sigma
        S_range = 1 / (trend_strength + eps)
        S_vol   = sigma + vol_of_vol
    """
    if price_history is None or price_history.empty or "Close" not in price_history.columns:
        return MarketState(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    close = price_history["Close"].dropna()
    if close.empty:
        return MarketState(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    returns = close.pct_change().dropna()
    window_returns = returns.tail(max(5, period))

    sigma = float(window_returns.std()) if not window_returns.empty else 0.0
    sigma = max(sigma, 1e-6)

    rolling_vol = returns.rolling(5).std().dropna()
    vol_of_vol = float(rolling_vol.tail(max(5, period)).std()) if not rolling_vol.empty else 0.0
    vol_of_vol = max(vol_of_vol, 0.0)

    current_price = float(close.iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema100 = float(close.ewm(span=100, adjust=False).mean().iloc[-1])
    trend_strength = abs(ema20 - ema100) / max(abs(current_price), 1e-8)

    s_trend = trend_strength / sigma
    s_range = 1.0 / (trend_strength + 1e-6)
    s_vol = sigma + vol_of_vol

    p_trend, p_range, p_high_vol = _softmax([s_trend, s_range, s_vol])
    return MarketState(p_trend, p_range, p_high_vol)


def infer_type_beliefs(
    signal_type: str,
    rsi_value: float,
    bb_z_score: float,
    regime: MarketState,
) -> Dict[str, float]:
    """
    Build posterior belief vector beta over market types.

    The priors are regime-conditioned, then adjusted by local signal features.
    """
    beliefs = {
        "t_trend": 0.25 + 0.55 * regime.prob_trend,
        "t_manipulator": 0.15 + 0.55 * regime.prob_high_vol,
        "t_exhausted": 0.15 + 0.25 * regime.prob_high_vol + 0.10 * regime.prob_range,
        "t_range": 0.20 + 0.50 * regime.prob_range,
    }

    if (signal_type == "buy" and rsi_value > 55.0) or (signal_type == "sell" and rsi_value < 45.0):
        beliefs["t_trend"] += 0.20

    if abs(bb_z_score) > 1.6 and 40.0 <= rsi_value <= 60.0:
        beliefs["t_manipulator"] += 0.25

    if rsi_value >= 70.0 or rsi_value <= 30.0:
        beliefs["t_exhausted"] += 0.25

    if abs(bb_z_score) < 0.70:
        beliefs["t_range"] += 0.20

    if signal_type == "hold":
        beliefs["t_range"] += 0.10

    return _normalize(beliefs)


def build_expected_return_path(
    one_step_return: float,
    regime: MarketState,
    horizon: int = 5,
) -> List[float]:
    """Project a small return trajectory from the one-step forecast."""
    horizon = max(1, int(horizon))
    one_step_return = float(one_step_return)

    persistence = 0.45 + 0.40 * regime.prob_trend - 0.20 * regime.prob_high_vol
    persistence = min(0.95, max(0.10, persistence))

    damping = 1.0 - 0.35 * regime.prob_high_vol
    damping = min(1.0, max(0.40, damping))

    path: List[float] = []
    for k in range(horizon):
        projected = one_step_return * (persistence ** k) * damping
        path.append(float(projected))
    return path


def _discounted_sum(values: Sequence[float], gamma: float) -> float:
    total = 0.0
    for idx, val in enumerate(values, start=1):
        total += (gamma ** idx) * float(val)
    return float(total)


def calculate_equilibrium_payoffs(
    expected_returns: Sequence[float],
    regime: MarketState,
) -> TypeEquilibrium:
    """Compute x(t) equilibrium payoffs for each type in return-space units."""
    if not expected_returns:
        expected_returns = [0.0]

    avg_abs = sum(abs(float(x)) for x in expected_returns) / float(len(expected_returns))

    trend_eq = _discounted_sum(expected_returns, gamma=0.95)
    range_eq = (1.0 - min(1.0, regime.prob_trend)) * avg_abs * 0.60
    manip_eq = avg_abs * (0.25 + 0.75 * regime.prob_high_vol)
    exhausted_eq = max(0.0, -trend_eq) * 0.70 + avg_abs * 0.10

    return TypeEquilibrium(
        t_trend=float(trend_eq),
        t_manipulator=float(manip_eq),
        t_exhausted=float(exhausted_eq),
        t_range=float(range_eq),
    )


def calculate_deviation_payoff(
    *,
    current_price: float,
    signal_price_target: float,
    volatility: float,
    volume_spike_ratio: float,
    reversal_potential_pct: float,
) -> SignalDeviation:
    """Compute manipulator deviation payoff m(t, A) = liquidity_profit - hunting_cost."""
    current_price = max(float(current_price), 1e-8)
    signal_price_target = float(signal_price_target)

    distance_pct_to_trigger = abs(signal_price_target - current_price) / current_price
    hunting_cost = max(0.0, float(volatility)) * distance_pct_to_trigger
    liquidity_profit = max(0.0, float(volume_spike_ratio) - 1.0) * max(
        float(reversal_potential_pct),
        distance_pct_to_trigger,
    )

    net = liquidity_profit - hunting_cost
    return SignalDeviation(
        signal_price_target=signal_price_target,
        hunting_cost=float(hunting_cost),
        liquidity_profit=float(liquidity_profit),
        net_payoff_manipulator=float(net),
    )


def build_deviation_from_market_data(
    *,
    signal_type: str,
    current_price: float,
    bb_width: float,
    price_history: pd.DataFrame,
) -> SignalDeviation:
    """Estimate deviation payoff terms directly from local market data."""
    close = price_history["Close"].dropna() if "Close" in price_history.columns else pd.Series(dtype=float)
    returns = close.pct_change().dropna() if not close.empty else pd.Series(dtype=float)
    recent_returns = returns.tail(20)

    volatility = float(recent_returns.std()) if not recent_returns.empty else 0.0
    volatility = max(0.0, volatility)

    if "Volume" in price_history.columns and len(price_history["Volume"].dropna()) >= 5:
        vol_series = price_history["Volume"].dropna().tail(20)
        avg_volume = float(vol_series.mean()) if not vol_series.empty else 0.0
        latest_volume = float(vol_series.iloc[-1]) if not vol_series.empty else 0.0
        volume_spike_ratio = latest_volume / max(avg_volume, 1e-8)
    else:
        volume_spike_ratio = 1.0

    bb_width = float(bb_width) if bb_width is not None else 0.0
    target_move = max(0.003, min(0.03, abs(bb_width) * 0.25))
    if signal_type == "buy":
        signal_price_target = float(current_price) * (1.0 + target_move)
    elif signal_type == "sell":
        signal_price_target = float(current_price) * (1.0 - target_move)
    else:
        signal_price_target = float(current_price)

    short_horizon_drift = abs(float(returns.tail(5).mean())) if not returns.empty else 0.0
    reversal_potential_pct = max(0.002, min(0.05, short_horizon_drift + volatility))

    return calculate_deviation_payoff(
        current_price=float(current_price),
        signal_price_target=float(signal_price_target),
        volatility=float(volatility),
        volume_spike_ratio=float(volume_spike_ratio),
        reversal_potential_pct=float(reversal_potential_pct),
    )