"""Measure whether the forecaster's directional call predicts anything.

The backtester answers "did this strategy make money", which bundles signal
quality together with position sizing, bracket geometry, costs and luck. When
it loses, that number cannot tell you *which* part is broken. This module
isolates the first part: given what the model said on a bar, what did the price
actually do next?

It reads the per-bar decision record a backtest writes (every symbol on every
bar, not just the handful that became trades — roughly 25x the sample) and
joins it to realised forward returns.

Three questions, in order of how much they matter:

1. **Hit rate vs. base rate.** Of the bars where the model said BUY, how often
   did price rise? Compared against the unconditional frequency of an up move
   in the same data, because in a market that drifts upward "60% of BUYs went
   up" is worthless if 60% of *all* bars went up.
2. **Information coefficient.** Rank correlation between the model's
   `prob_profit` and the realised forward return. This is the standard quant
   measure of a predictive signal; an IC of 0 means the ordering carries no
   information. Even a genuinely useful equity signal usually sits around
   0.02-0.05, so the bar is low, but it must be reliably above zero.
3. **Monotonicity in confidence.** If the model is calibrated, high-confidence
   calls should outperform low-confidence ones. If they do not, the confidence
   number is noise and should not be driving position size — which it currently
   does, through `calculate_minimax_multiplier`.

Every statistic is reported with a confidence interval. A point estimate on
this sample size is not evidence of anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_HORIZONS: Tuple[int, ...] = (1, 3, 5, 10)


@dataclass
class Decision:
    """One model opinion on one symbol-bar, plus what happened afterwards."""

    date: str
    symbol: str
    close: float
    signal_type: str
    confidence: float
    prob_profit: float
    forward_returns: Dict[int, float] = field(default_factory=dict)


@dataclass
class HitRate:
    """Directional accuracy against the base rate of an up move."""

    label: str
    n: int
    hits: int
    hit_rate: float
    base_rate: float
    edge: float                 # hit_rate - base_rate, in percentage points
    ci_low: float
    ci_high: float
    significant: bool           # does the edge's CI exclude zero?

    def as_row(self) -> str:
        marker = "  <-- excludes zero" if self.significant else ""
        return (
            f"  {self.label:<18} n={self.n:<6} hit={self.hit_rate*100:5.1f}%  "
            f"base={self.base_rate*100:5.1f}%  edge={self.edge*100:+5.1f}pp  "
            f"95% CI [{self.ci_low*100:+5.1f}, {self.ci_high*100:+5.1f}]{marker}"
        )


@dataclass
class InformationCoefficient:
    horizon: int
    n: int
    ic: float
    ci_low: float
    ci_high: float
    significant: bool

    def as_row(self) -> str:
        marker = "  <-- excludes zero" if self.significant else ""
        return (
            f"  horizon {self.horizon:>2}d   n={self.n:<6} IC={self.ic:+.4f}  "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]{marker}"
        )


@dataclass
class ConfidenceBucket:
    lower: float
    upper: float
    n: int
    mean_forward_return: float
    hit_rate: float

    def as_row(self) -> str:
        return (
            f"  conf [{self.lower:.2f}, {self.upper:.2f})  n={self.n:<6} "
            f"mean fwd={self.mean_forward_return*100:+6.3f}%  hit={self.hit_rate*100:5.1f}%"
        )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _wilson_interval(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because hit rates here sit near the
    tails on small per-bucket samples, where the naive interval misbehaves.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _rank(values: Sequence[float]) -> List[float]:
    """Average ranks, ties shared."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation. 0.0 when undefined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    rx, ry = _rank(xs), _rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def _fisher_ci(r: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Confidence interval for a correlation via Fisher's z transform."""
    if n < 4 or abs(r) >= 1.0:
        return (r, r)
    zr = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = zr - z * se, zr + z * se
    return (math.tanh(lo), math.tanh(hi))


# ---------------------------------------------------------------------------
# Loading and joining
# ---------------------------------------------------------------------------

def load_decisions(backtest_path: Path | str) -> List[Decision]:
    """Read the per-bar decision record out of a backtest result file."""
    payload = json.loads(Path(backtest_path).read_text(encoding="utf-8"))
    decisions: List[Decision] = []
    for row in payload.get("decisions", []):
        try:
            decisions.append(Decision(
                date=str(row["date"]),
                symbol=str(row["symbol"]),
                close=float(row["close"]),
                signal_type=str(row.get("signal_type", "")).lower(),
                confidence=float(row.get("confidence") or 0.0),
                prob_profit=float(row.get("prob_profit") or 0.0),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return decisions


def attach_forward_returns(
    decisions: Sequence[Decision],
    price_history: Dict[str, Any],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> List[Decision]:
    """Join each decision to the realised return over each horizon.

    `price_history` maps symbol to a frame with a 'Close' column indexed by
    date. Decisions whose forward window runs off the end of the data are
    dropped for that horizon rather than padded, which would bias the tail.
    """
    by_symbol: Dict[str, Tuple[List[str], List[float]]] = {}
    for symbol, frame in price_history.items():
        try:
            dates = [str(d)[:10] for d in frame.index]
            closes = [float(c) for c in frame["Close"].values]
            by_symbol[symbol] = (dates, closes)
        except Exception:
            continue

    index_cache: Dict[str, Dict[str, int]] = {
        sym: {d: i for i, d in enumerate(dates)} for sym, (dates, _) in by_symbol.items()
    }

    enriched: List[Decision] = []
    for decision in decisions:
        entry = by_symbol.get(decision.symbol)
        if entry is None:
            continue
        _, closes = entry
        position = index_cache[decision.symbol].get(decision.date[:10])
        if position is None:
            continue
        base = closes[position]
        if base <= 0:
            continue
        for horizon in horizons:
            target = position + horizon
            if target >= len(closes):
                continue
            future = closes[target]
            if future > 0:
                decision.forward_returns[horizon] = (future / base) - 1.0
        if decision.forward_returns:
            enriched.append(decision)
    return enriched


# ---------------------------------------------------------------------------
# The three questions
# ---------------------------------------------------------------------------

def hit_rates(
    decisions: Sequence[Decision], horizon: int = 5
) -> List[HitRate]:
    """Directional accuracy per signal type, against the sample's base rate."""
    usable = [d for d in decisions if horizon in d.forward_returns]
    if not usable:
        return []

    # Base rate: how often price rose at all, across every bar the model saw.
    ups = sum(1 for d in usable if d.forward_returns[horizon] > 0)
    base_rate = ups / len(usable)

    results: List[HitRate] = []
    for label, predicate in (
        ("BUY", lambda d: d.signal_type == "buy"),
        ("SELL", lambda d: d.signal_type == "sell"),
        ("HOLD", lambda d: d.signal_type == "hold"),
    ):
        subset = [d for d in usable if predicate(d)]
        if not subset:
            continue
        # A SELL is correct when price falls.
        if label == "SELL":
            hits = sum(1 for d in subset if d.forward_returns[horizon] < 0)
            reference = 1.0 - base_rate
        else:
            hits = sum(1 for d in subset if d.forward_returns[horizon] > 0)
            reference = base_rate

        n = len(subset)
        rate = hits / n
        lo, hi = _wilson_interval(hits, n)
        results.append(HitRate(
            label=label, n=n, hits=hits, hit_rate=rate, base_rate=reference,
            edge=rate - reference,
            ci_low=lo - reference, ci_high=hi - reference,
            significant=(lo > reference) or (hi < reference),
        ))
    return results


def information_coefficients(
    decisions: Sequence[Decision], horizons: Sequence[int] = DEFAULT_HORIZONS
) -> List[InformationCoefficient]:
    """Rank correlation between predicted probability and realised return."""
    results: List[InformationCoefficient] = []
    for horizon in horizons:
        usable = [d for d in decisions if horizon in d.forward_returns]
        if len(usable) < 10:
            continue
        xs = [d.prob_profit for d in usable]
        ys = [d.forward_returns[horizon] for d in usable]
        ic = spearman(xs, ys)
        lo, hi = _fisher_ci(ic, len(usable))
        results.append(InformationCoefficient(
            horizon=horizon, n=len(usable), ic=ic,
            ci_low=lo, ci_high=hi, significant=(lo > 0) or (hi < 0),
        ))
    return results


def confidence_buckets(
    decisions: Sequence[Decision],
    horizon: int = 5,
    edges: Sequence[float] = (0.0, 0.5, 0.6, 0.7, 0.8, 1.01),
    directional_only: bool = True,
) -> List[ConfidenceBucket]:
    """Forward performance by confidence band, to test calibration."""
    usable = [d for d in decisions if horizon in d.forward_returns]
    if directional_only:
        usable = [d for d in usable if d.signal_type in {"buy", "sell"}]
    buckets: List[ConfidenceBucket] = []
    for lower, upper in zip(edges, edges[1:]):
        subset = [d for d in usable if lower <= d.confidence < upper]
        if not subset:
            continue
        # Sign the return by the direction bet, so BUY and SELL are comparable.
        signed = [
            d.forward_returns[horizon] if d.signal_type == "buy" else -d.forward_returns[horizon]
            for d in subset
        ]
        hits = sum(1 for value in signed if value > 0)
        buckets.append(ConfidenceBucket(
            lower=lower, upper=upper, n=len(subset),
            mean_forward_return=sum(signed) / len(signed),
            hit_rate=hits / len(subset),
        ))
    return buckets


def format_report(
    decisions: Sequence[Decision],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    primary_horizon: int = 5,
) -> str:
    """The whole diagnosis as readable text."""
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append("SIGNAL QUALITY")
    lines.append("=" * 74)
    symbols = {d.symbol for d in decisions}
    lines.append(f"  {len(decisions)} decisions across {len(symbols)} symbols")
    lines.append("")

    lines.append(f"DIRECTIONAL ACCURACY (horizon {primary_horizon}d)")
    lines.append("  Does the call beat simply assuming the base-rate direction?")
    rates = hit_rates(decisions, horizon=primary_horizon)
    if rates:
        lines.extend(rate.as_row() for rate in rates)
    else:
        lines.append("  (insufficient data)")
    lines.append("")

    lines.append("INFORMATION COEFFICIENT")
    lines.append("  Rank correlation of prob_profit with realised forward return.")
    ics = information_coefficients(decisions, horizons)
    if ics:
        lines.extend(ic.as_row() for ic in ics)
    else:
        lines.append("  (insufficient data)")
    lines.append("")

    lines.append(f"CALIBRATION (horizon {primary_horizon}d, directional signals, return signed by the bet)")
    lines.append("  A calibrated model earns more where it claims more confidence.")
    buckets = confidence_buckets(decisions, horizon=primary_horizon)
    if buckets:
        lines.extend(bucket.as_row() for bucket in buckets)
    else:
        lines.append("  (insufficient data)")
    lines.append("")

    any_significant = (
        any(r.significant for r in rates)
        or any(i.significant for i in ics)
    )
    lines.append("=" * 74)
    if any_significant:
        lines.append("  At least one statistic's CI excludes zero. Worth pursuing —")
        lines.append("  confirm it holds out of sample before trading on it.")
    else:
        lines.append("  NO detectable predictive signal: every confidence interval")
        lines.append("  spans zero. Tuning thresholds, exits or position sizing")
        lines.append("  cannot create an edge that is not present here.")
    lines.append("=" * 74)
    return "\n".join(lines)
