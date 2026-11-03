"""Causal feature blocks for the walk-forward ablation.

Every column at bar t is a function of prices, volume, or the calendar date
through the close of t. None of them read the next bar's return. The live
forecast and the tournament both call this so a block means the same thing
in both places.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

# Known new moon, used only as a phase origin. The synodic month is the mean
# length; the feature is the angle on that cycle, which is known before the close.
_NEW_MOON = pd.Timestamp("2000-01-06 18:14:00")
_SYNODIC_DAYS = 29.53058867

LAGS = [f"Return_lag{lag}" for lag in range(1, 6)]
VOLATILITY = ["Volatility"]
TA = ["RSI", "BB_Z_Score", "BB_Width"]
CANDLES = ["candle_body", "candle_upper", "candle_lower", "candle_direction"]
VOLUME = ["volume_ratio", "volume_price"]
REGIME = ["variance_ratio"]
LUNAR = ["lunar_sin", "lunar_cos"]
NEWS = ["news_polarity"]

BLOCKS = {
    "lags": LAGS,
    "volatility": VOLATILITY,
    "ta": TA,
    "candles": CANDLES,
    "volume": VOLUME,
    "regime": REGIME,
    "lunar": LUNAR,
    "news": NEWS,
}

# Research arms. "all" is filled after the dict exists so it stays one list.
ARM_BLOCKS = {
    "baseline": ["lags"],
    "baseline_ta": ["lags", "ta"],
    "baseline_candles": ["lags", "candles"],
    "baseline_volume": ["lags", "volume"],
    "baseline_news": ["lags", "news"],
    "baseline_regime": ["lags", "regime"],
    "baseline_lunar": ["lags", "lunar"],
    # News is not in All. A missing headline is not a zero, and requiring it
    # would leave All unscored on every ordinary day. News stays its own arm.
    "all": ["lags", "ta", "candles", "volume", "regime", "lunar"],
}


def columns_for(blocks: Iterable[str]) -> List[str]:
    cols: List[str] = []
    for name in blocks:
        cols.extend(BLOCKS[name])
    return cols


def add_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candle, volume, regime, and lunar columns. Does not drop rows."""
    out = df.copy()
    close = out["Close"].astype(float)
    open_ = out["Open"].astype(float) if "Open" in out.columns else close
    high = out["High"].astype(float) if "High" in out.columns else close
    low = out["Low"].astype(float) if "Low" in out.columns else close
    span = (high - low).astype(float)
    usable = span.replace(0, np.nan)
    body_top = np.maximum(open_, close)
    body_bot = np.minimum(open_, close)
    # A zero-range bar has no geometry. Filling 0 keeps that row in the model
    # instead of dropping it and silently forecasting from the previous close.
    out["candle_body"] = ((close - open_) / usable).fillna(0.0).astype(float)
    out["candle_upper"] = ((high - body_top) / usable).fillna(0.0).astype(float)
    out["candle_lower"] = ((body_bot - low) / usable).fillna(0.0).astype(float)
    out["candle_direction"] = np.sign(close - open_).fillna(0.0).astype(float)

    if "Volume" in out.columns:
        volume = out["Volume"].astype(float)
        trailing = volume.rolling(20, min_periods=20).mean()
        out["volume_ratio"] = (volume / trailing.replace(0, np.nan)).astype(float)
    else:
        out["volume_ratio"] = np.nan
    ret = out["Return"] if "Return" in out.columns else close.pct_change()
    out["volume_price"] = (out["volume_ratio"] * ret).astype(float)

    # Variance ratio: var of 5-day returns over 5 * var of 1-day returns,
    # both measured on a trailing 60-bar window ending at t.
    one = ret.rolling(60, min_periods=60).var()
    five = ret.rolling(5, min_periods=5).sum().rolling(60, min_periods=60).var()
    out["variance_ratio"] = (five / (5.0 * one.replace(0, np.nan))).astype(float)

    lunar_sin, lunar_cos = _lunar_angle(out.index)
    out["lunar_sin"] = lunar_sin
    out["lunar_cos"] = lunar_cos
    if "news_polarity" not in out.columns:
        out["news_polarity"] = np.nan
    return out


def _lunar_angle(index) -> tuple:
    stamps = pd.to_datetime(index)
    if getattr(stamps, "tz", None) is not None:
        stamps = stamps.tz_convert("UTC").tz_localize(None)
    days = (stamps - _NEW_MOON).total_seconds() / 86400.0
    cycle = np.mod(days, _SYNODIC_DAYS) / _SYNODIC_DAYS
    angle = 2.0 * math.pi * cycle
    return np.sin(angle), np.cos(angle)


def snapshot_row(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Last-row feature dicts for a PredictionRecord. Missing values are omitted."""
    row = df.iloc[-1]
    groups = {
        "bb_features": ["BB_Z_Score", "BB_Width", "BB_Upper", "BB_Middle", "BB_Lower"],
        "rsi_features": ["RSI"],
        "candle_features": CANDLES,
        "volume_features": VOLUME,
        "regime_features": REGIME,
        "lunar_features": LUNAR,
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, cols in groups.items():
        block = {}
        for col in cols:
            if col not in df.columns:
                continue
            value = row[col]
            if pd.isna(value):
                continue
            block[col] = float(value)
        out[name] = block
    return out
