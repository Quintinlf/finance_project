"""
Polar coordinates utilities for finance time series.

Implements the notebook concepts from polar_coordinates_finance.ipynb:
- polar_features: compute (Radius, MomentumAngle) from returns/volatility
- plot_polar_market_cycle: visualize Radius vs Angle in polar coordinates
- bayesian_polar_signal: simple Bayesian-style decision rule from angle

This module is library-only: call functions from your code; nothing runs on import.
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def polar_features(
    df: pd.DataFrame,
    *,
    price_col: str = "Close",
    vol_window: int = 14,
    eps: float = 1e-12,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Convert price data into polar coordinate features for modeling.

    Features:
    - Return: pct_change of price_col
    - Volatility: rolling std of Return over vol_window
    - MomentumAngle (theta): arctan2(Return, Volatility)
    - Radius (r): sqrt(Return^2 + Volatility^2)

    Args:
        df: Input DataFrame with a price column (default 'Close'). Index can be datetime.
        price_col: Column name to use for returns.
        vol_window: Window length for rolling volatility estimate.
        eps: Small number to avoid division-by-zero in angle computation.
        copy: If True, operate on a copy; otherwise mutate df in-place.

    Returns:
        DataFrame with added columns: ['Return', 'Volatility', 'MomentumAngle', 'Radius']
    """
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' not found in DataFrame columns: {list(df.columns)}")

    out = df.copy() if copy else df

    # Compute simple return and volatility proxy
    out["Return"] = out[price_col].pct_change()
    out["Volatility"] = out["Return"].rolling(window=vol_window, min_periods=vol_window).std()

    # Robust angle and radius; avoid zeros in denominator by adding eps
    ret = out["Return"].to_numpy(dtype=float)
    vol = out["Volatility"].to_numpy(dtype=float)
    vol_safe = np.where(np.isfinite(vol), vol, 0.0) + eps

    out["MomentumAngle"] = np.arctan2(ret, vol_safe)
    out["Radius"] = np.sqrt(np.square(ret) + np.square(vol_safe))

    return out


axtype = plt.Axes  # for type hints without importing mpl internals

def plot_polar_market_cycle(
    df: pd.DataFrame,
    ticker: str,
    *,
    price_col: str = "Close",
    vol_window: int = 14,
    show: bool = True,
    ax: Optional[axtype] = None,
):
    """
    Plot Radius vs MomentumAngle in polar coordinates to visualize market cycles.

    Args:
        df: Price DataFrame containing price_col.
        ticker: Ticker symbol for title.
        price_col: Column to compute returns from.
        vol_window: Rolling window for volatility.
        show: If True, call plt.show() at the end.
        ax: Optional existing polar axes to draw onto.

    Returns:
        The matplotlib Axes object used for plotting.
    """
    feats = polar_features(df, price_col=price_col, vol_window=vol_window)

    # Prepare polar plot
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)
        created_fig = True

    theta = feats["MomentumAngle"].to_numpy()
    radius = feats["Radius"].to_numpy()

    # Drop NaNs that arise from initial periods
    mask = np.isfinite(theta) & np.isfinite(radius)
    theta = theta[mask]
    radius = radius[mask]

    ax.plot(theta, radius, color="teal", alpha=0.8, linewidth=1.5)
    ax.set_title(f"Polar Cycle Representation for {ticker}", va="bottom")

    if show and created_fig:
        plt.show()
    return ax


def _wrap_angle(theta: float) -> float:
    """Normalize angle to [0, 2π)."""
    two_pi = 2.0 * np.pi
    t = float(theta) % two_pi
    return t if t >= 0 else t + two_pi


def bayesian_polar_signal(r: float, theta: float) -> Tuple[str, Dict[str, float]]:
    """
    Simple Bayesian-style decision from polar coordinates.

    Heuristic mapping of angle quadrants to market bias:
      - theta near 0 (up/low-vol): favor BUY
      - theta in (π/4, 3π/4): volatile region: favor HOLD
      - theta in (3π/4, 5π/4): downward drift: favor SELL
      - else: uncertain: favor HOLD

    Args:
        r: Radius (magnitude). Included for potential extensions; not used directly here.
        theta: Angle in radians.

    Returns:
        (signal, probs) where signal in {"buy", "hold", "sell"} and probs is a normalized dict.
    """
    t = _wrap_angle(theta)

    probs: Dict[str, float] = {"buy": 1.0 / 3.0, "hold": 1.0 / 3.0, "sell": 1.0 / 3.0}

    if (t < (np.pi / 4.0)) or (t > (7.0 * np.pi / 4.0)):
        # upward / low volatility
        probs["buy"] *= 0.7
    elif (np.pi / 4.0) < t < (3.0 * np.pi / 4.0):
        # high volatility band
        probs["hold"] *= 0.7
    elif (3.0 * np.pi / 4.0) < t < (5.0 * np.pi / 4.0):
        # downward drift
        probs["sell"] *= 0.7
    else:
        # chaotic/uncertain
        probs["hold"] *= 0.6

    total = float(sum(probs.values()))
    if total <= 0 or not np.isfinite(total):
        # Fallback to uniform if something degenerate happens
        probs = {"buy": 1.0 / 3.0, "hold": 1.0 / 3.0, "sell": 1.0 / 3.0}
        total = 1.0

    for k in probs:
        probs[k] = float(probs[k]) / total

    signal = max(probs, key=probs.get)
    return signal, probs


def latest_polar_signal(
    df: pd.DataFrame,
    *,
    price_col: str = "Close",
    vol_window: int = 14,
) -> Tuple[str, Dict[str, float], float, float]:
    """
    Convenience wrapper: compute latest (r, theta) from a price DataFrame and return the signal.

    Returns:
        (signal, probs, r, theta)
    """
    feats = polar_features(df, price_col=price_col, vol_window=vol_window)
    last = feats.dropna(subset=["Radius", "MomentumAngle"]).iloc[-1]
    r = float(last["Radius"]) 
    theta = float(last["MomentumAngle"]) 
    signal, probs = bayesian_polar_signal(r, theta)
    return signal, probs, r, theta
