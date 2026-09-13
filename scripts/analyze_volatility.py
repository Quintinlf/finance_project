"""Is volatility predictable here, the way direction is not?

The claim worth testing: *direction* is close to unpredictable, but *magnitude*
is not. Volatility clusters -- calm follows calm, turbulence follows turbulence
-- and that persistence is one of the most reproduced findings in empirical
finance (Engle's ARCH, 1982). If it holds on this universe, it is a lever the
direction models never had.

This measures it rather than assuming it, using the same discipline applied to
direction so the two numbers are directly comparable:

* Predict forward realized volatility from trailing realized volatility.
* Report R^2 and rank correlation with confidence intervals.
* Correct the sample size for overlapping windows, exactly as the horizon
  analysis does -- consecutive 20-day vol estimates share 19 of 20 days, so
  naive intervals are far too narrow and would manufacture significance.
* Print the direction IC beside it, because "volatility is predictable" only
  matters if it is predictable by *more* than direction was.

A deliberate caution about what a positive result would and would not mean:
volatility persistence is not by itself a trading edge. It informs position
SIZING (risk less when the tape is wild) and it is the input options are priced
on. It does not tell you which way anything goes. Anyone reading a high R^2
here as "we found alpha" has misread it.

    python scripts/analyze_volatility.py
    python scripts/analyze_volatility.py --lookback 20 --horizon 20
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.price_cache import get_history  # noqa: E402
from logic.universe import get_symbols  # noqa: E402

TRADING_DAYS = 252


def realized_vol(returns: Sequence[float]) -> Optional[float]:
    """Annualized standard deviation of a return window."""
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    try:
        from scipy import stats

        return float(stats.spearmanr(xs, ys).statistic)
    except Exception:  # noqa: BLE001
        return float("nan")


def _effective_n(n: int, horizon: int) -> float:
    """Same overlap correction the horizon analysis uses.

    Consecutive h-day windows share h-1 days. Treating them as independent is
    how a persistence result gets overstated.
    """
    return max(2.0, n / float(max(1, horizon)))


def collect_pairs(
    symbol: str, lookback: int, horizon: int
) -> Tuple[List[float], List[float]]:
    """(trailing vol, forward vol) pairs for one symbol."""
    hist = get_history(symbol)
    if hist is None or hist.empty:
        return [], []

    closes = [float(c) for c in hist["Close"]]
    rets = [
        closes[i] / closes[i - 1] - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]

    past, future = [], []
    for i in range(lookback, len(rets) - horizon):
        p = realized_vol(rets[i - lookback:i])
        f = realized_vol(rets[i:i + horizon])
        if p is None or f is None or p <= 0 or f <= 0:
            continue
        past.append(p)
        future.append(f)
    return past, future


def analyze(
    symbols: Sequence[str], lookback: int, horizon: int, out_path=None
) -> dict:
    rows = []
    all_past: List[float] = []
    all_future: List[float] = []

    for symbol in symbols:
        past, future = collect_pairs(symbol, lookback, horizon)
        if len(past) < 100:
            continue
        all_past.extend(past)
        all_future.extend(future)

        r = _pearson(past, future)
        rows.append({
            "symbol": symbol,
            "n": len(past),
            "effective_n": round(_effective_n(len(past), horizon)),
            "r": None if math.isnan(r) else round(r, 4),
            "r2": None if math.isnan(r) else round(r * r, 4),
            "spearman": round(_spearman(past, future), 4),
            "mean_vol": round(sum(past) / len(past), 4),
        })

    pooled_r = _pearson(all_past, all_future)
    pooled_rho = _spearman(all_past, all_future)
    eff_n = _effective_n(len(all_past), horizon)
    # |r| that would clear significance at the overlap-corrected sample size.
    bar = 1.96 / math.sqrt(eff_n)

    print()
    print("=" * 84)
    print(f"VOLATILITY PERSISTENCE  ({lookback}d trailing -> {horizon}d forward)")
    print("=" * 84)
    print(f"  {'symbol':<8}{'n':>7}{'eff_n':>8}{'r':>9}{'R^2':>9}{'rank':>9}{'mean vol':>11}")
    print("  " + "-" * 80)
    for row in sorted(rows, key=lambda x: -(x["r2"] or 0)):
        print(f"  {row['symbol']:<8}{row['n']:>7}{row['effective_n']:>8}"
              f"{row['r']:>9.4f}{row['r2']:>9.4f}{row['spearman']:>9.4f}"
              f"{row['mean_vol'] * 100:>10.1f}%")

    print()
    print(f"  POOLED  n={len(all_past)}  effective_n={eff_n:.0f}")
    print(f"    Pearson r      : {pooled_r:.4f}   (R^2 = {pooled_r ** 2:.4f})")
    print(f"    Spearman rho   : {pooled_rho:.4f}")
    print(f"    |r| significance bar (overlap-corrected): {bar:.4f}")
    print()

    # The comparison that gives the number meaning.
    direction_ic = 0.021  # best observed in the horizon analysis, GaussianProc @1d
    print(f"  For comparison, the best DIRECTION signal measured was IC {direction_ic:.4f}.")
    if not math.isnan(pooled_rho) and abs(pooled_rho) > abs(direction_ic):
        ratio = abs(pooled_rho) / abs(direction_ic)
        print(f"  Volatility persistence is ~{ratio:.0f}x stronger than anything")
        print(f"  the direction models produced.")
    print()
    print("  NOTE: persistence is not alpha. It says how BIG the move will be,")
    print("  never which way. Its uses are position sizing and options pricing.")
    print()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback": lookback,
        "horizon": horizon,
        "pooled": {
            "n": len(all_past),
            "effective_n": round(eff_n),
            "r": None if math.isnan(pooled_r) else round(pooled_r, 4),
            "r2": None if math.isnan(pooled_r) else round(pooled_r ** 2, 4),
            "spearman": None if math.isnan(pooled_rho) else round(pooled_rho, 4),
            "significance_bar": round(bar, 4),
            "direction_ic_for_comparison": direction_ic,
        },
        "per_symbol": rows,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  -> {out_path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--scope", default="all")
    ap.add_argument("--out", default="trade_logs/volatility_analysis.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    analyze(get_symbols(args.scope), args.lookback, args.horizon, out_path=args.out)


if __name__ == "__main__":
    main()
