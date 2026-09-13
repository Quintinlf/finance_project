"""Size positions by predicted volatility instead of a flat fraction.

Why this and not another direction model: measured on this universe
(`scripts/analyze_volatility.py`, 32,432 overlapping windows), trailing
20-day realized volatility predicts forward 20-day realized volatility with
r = 0.708 (R^2 = 0.50) against an overlap-corrected significance bar of 0.049.
Every symbol in the universe shows it. The best *direction* signal ever
measured here was IC 0.021, which did not survive correction at any horizon.

So: how big the next move will be is roughly half-predictable. Which way it
goes is not predictable at all.

That asymmetry has exactly one honest use, and it is not alpha. A fixed
2%-of-equity position is a completely different bet in UNG (58.5% annualized
vol) than in DBA (13.2%) -- more than four times the risk for the same dollar
amount. Scaling by predicted volatility makes "one position" mean one
consistent quantity of risk rather than one consistent quantity of dollars.

**This cannot create edge.** With no directional signal, expected return stays
where it was. What changes is the distribution: fewer oversized bets on wild
instruments, so drawdowns shrink. That is worth having on its own, and it is
the only thing the measurements actually support.

Deliberately one-directional: the scale factor is clamped at 1.0, so this can
only ever make a position *smaller*. Letting it size up in calm names would
mean levering into low-volatility instruments on the strength of a
relationship that holds on average and breaks precisely during the regime
shifts that matter.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

TRADING_DAYS = 252

# Annualized volatility a "normal" position is sized for. Roughly the middle of
# this universe (equities ~26%, broad commodity baskets ~18%, UNG ~59%).
DEFAULT_TARGET_VOL = 0.25

# Never scale below this: past a point the position rounds to zero shares on a
# small account and the trade simply stops happening, which is a different
# decision than sizing it small.
MIN_SCALE = 0.25

# Bars used to estimate forward volatility. 20 is what the persistence was
# measured at; changing it invalidates that measurement.
DEFAULT_LOOKBACK = 20


@dataclass
class VolatilitySizing:
    symbol: str
    predicted_vol: Optional[float]
    scale: float
    reason: str

    def as_meta(self) -> dict:
        return {
            "vol_predicted": None if self.predicted_vol is None else round(self.predicted_vol, 4),
            "vol_scale": round(self.scale, 4),
            "vol_reason": self.reason,
        }


def realized_vol(returns: Sequence[float]) -> Optional[float]:
    """Annualized standard deviation of a return window."""
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def predict_volatility(
    symbol: str, *, lookback: int = DEFAULT_LOOKBACK, price_history=None
) -> Optional[float]:
    """Forward volatility estimate: trailing realized volatility.

    A deliberately plain estimator. The measured persistence (r = 0.708) is
    for exactly this -- trailing realized vol predicting forward realized vol
    -- so using it keeps the sizing decision backed by the number that was
    actually verified. A GARCH fit might add a little; it would also decouple
    the code from the evidence supporting it.
    """
    try:
        if price_history is None:
            from logic.price_cache import get_history

            price_history = get_history(symbol)
        if price_history is None or len(price_history) < lookback + 1:
            return None

        closes = [float(c) for c in price_history["Close"][-(lookback + 1):]]
        rets = [
            closes[i] / closes[i - 1] - 1.0
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        vol = realized_vol(rets)
        return vol if vol and vol > 0 else None
    except Exception as exc:  # noqa: BLE001
        logging.debug("Volatility estimate failed for %s (%s)", symbol, exc)
        return None


def compute_sizing(
    symbol: str,
    *,
    target_vol: float = DEFAULT_TARGET_VOL,
    lookback: int = DEFAULT_LOOKBACK,
    price_history=None,
) -> VolatilitySizing:
    """Scale factor for one symbol's position size, always <= 1.0."""
    predicted = predict_volatility(symbol, lookback=lookback, price_history=price_history)

    if predicted is None:
        # No estimate is not a reason to take a bigger position.
        return VolatilitySizing(symbol, None, 1.0, "no volatility estimate; unscaled")

    raw_scale = target_vol / predicted
    if raw_scale >= 1.0:
        return VolatilitySizing(
            symbol, predicted, 1.0,
            f"vol {predicted:.1%} at or below {target_vol:.0%} target; unscaled "
            f"(this never sizes UP)",
        )

    scale = max(MIN_SCALE, raw_scale)
    floored = " (floored)" if scale > raw_scale else ""
    return VolatilitySizing(
        symbol, predicted, scale,
        f"vol {predicted:.1%} vs {target_vol:.0%} target -> x{scale:.2f}{floored}",
    )


def apply_volatility_sizing(
    signals: Sequence,
    *,
    target_vol: float = DEFAULT_TARGET_VOL,
    enforce: bool = True,
    verbose: bool = True,
) -> Sequence:
    """Annotate signals with a volatility scale factor.

    With ``enforce=False`` the factor is recorded but ``build_order_plan``
    ignores it, so the effect can be watched before it moves real size.
    """
    for signal in signals:
        if getattr(signal, "signal_type", "") == "hold":
            continue
        if not hasattr(signal, "meta") or signal.meta is None:
            signal.meta = {}

        sizing = compute_sizing(
            str(getattr(signal, "symbol", "")),
            target_vol=target_vol,
            price_history=signal.meta.get("price_history"),
        )
        signal.meta.update(sizing.as_meta())
        signal.meta["vol_sizing_enforced"] = bool(enforce)

        if verbose and sizing.scale < 1.0:
            logging.info(
                "VOL SIZING | symbol=%s | predicted_vol=%.1f%% | scale=x%.2f | %s | %s",
                sizing.symbol, (sizing.predicted_vol or 0) * 100, sizing.scale,
                "ENFORCED" if enforce else "shadow", sizing.reason,
            )

    return signals
