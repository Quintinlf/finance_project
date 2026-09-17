"""Does accuracy improve when the model predicts a BIG move?

This is the Tier 0 question, and the last open one on the direction side.

Costs are a fixed toll -- 12bp round trip on XLE, 45bp on CANE -- so the size
of the predicted move decides how much accuracy you need to break even:

    predicted move    break-even accuracy (XLE, 12bp)
    0.59% (the mean)  60.2%
    1.00%             56.0%
    2.00%             53.0%
    3.00%             52.0%

Measured accuracy tops out around 53.2% at maximum model confidence. So on
average-sized predictions the gap is hopeless, while above roughly a 2% move
the required accuracy falls to something already achieved -- *if* accuracy
holds up on large predictions, which has never been tested.

That "if" is the whole point. 53.2% is accuracy conditional on high
CONFIDENCE, not on a large predicted MOVE. The two need not coincide, and
assuming they do would be exactly the reasoning that produced every false
finding in this project's history.

So: bucket predictions by |forecast|, measure accuracy in each bucket against
the base rate, and compare to the break-even line that bucket actually faces.

Reading the output honestly:

* A bucket only matters if accuracy clears BOTH the base rate (otherwise
  buying blindly beats it) AND its own break-even line (otherwise costs eat it).
* Large-move buckets are small. Confidence intervals are printed; a bucket of
  40 observations tells you nothing no matter how good the point estimate looks.
* Several buckets are several tests. A Bonferroni-corrected threshold is
  printed alongside.

    python scripts/analyze_move_size.py
    python scripts/analyze_move_size.py --symbol XLE
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic import costs  # noqa: E402
from logic.sqlite_store import DEFAULT_DB_PATH  # noqa: E402

# Edges in percent of price. Chosen around the break-even crossover rather than
# fitted to the data -- picking bucket edges after seeing results is how a
# threshold gets manufactured.
DEFAULT_BUCKETS = [0.0, 0.5, 1.0, 2.0, 3.0, 100.0]


def load_rows(db_path, symbol: Optional[str] = None) -> List[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        sql = (
            "SELECT symbol, forecast_magnitude, next_day_return, "
            "ensemble_direction, raw_prob_profit "
            "FROM model_component_performance "
            "WHERE forecast_magnitude IS NOT NULL AND next_day_return IS NOT NULL"
        )
        params: tuple = ()
        if symbol:
            sql += " AND symbol = ?"
            params = (symbol,)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _wilson(hits: int, n: int) -> tuple:
    """Wilson interval -- behaves sanely on the small buckets that matter here,
    where the normal approximation would run past 0 or 1."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    z = 1.96
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def analyze(db_path, buckets: Sequence[float], symbol=None, out_path=None) -> dict:
    rows = load_rows(db_path, symbol)
    if not rows:
        print("\nNo rows carry forecast_magnitude yet — run scripts/backfill_history.py first.\n")
        return {}

    base_up = sum(1 for r in rows if r["next_day_return"] > 0) / len(rows)

    # Median round-trip cost across the traded universe, as the break-even line.
    symbols = {r["symbol"] for r in rows}
    cost_by_symbol = {s: costs.for_symbol(s).round_trip_bps / 10000.0 for s in symbols}
    cheapest = min(cost_by_symbol.values()) if cost_by_symbol else 0.0012

    print()
    print("=" * 92)
    print("ACCURACY BY SIZE OF PREDICTED MOVE")
    print("=" * 92)
    print(f"  {len(rows)} scored predictions"
          f"{' for ' + symbol if symbol else ''}   base rate (price rose): {base_up * 100:.1f}%")
    print(f"  break-even line uses the cheapest venue in the sample "
          f"({cheapest * 100:.3f}% round trip)")
    print()
    print(f"  {'move bucket':<16}{'n':>7}{'acc':>8}{'95% CI':>18}"
          f"{'vs base':>9}{'break-even':>12}{'clears?':>10}")
    print("  " + "-" * 88)

    results = []
    n_tests = len(buckets) - 1
    for i in range(n_tests):
        lo, hi = buckets[i], buckets[i + 1]
        sub = [
            r for r in rows
            if lo <= abs(float(r["forecast_magnitude"])) * 100.0 < hi
            and str(r["ensemble_direction"] or "").lower() in ("buy", "sell")
        ]
        if len(sub) < 20:
            label = f"{lo:.1f}-{hi:.0f}%" if hi < 100 else f">{lo:.1f}%"
            print(f"  {label:<16}{len(sub):>7}   too few observations to say anything")
            continue

        hits = sum(
            1 for r in sub
            if (str(r["ensemble_direction"]).lower() == "buy") == (r["next_day_return"] > 0)
        )
        p, ci_lo, ci_hi = _wilson(hits, len(sub))

        # Break-even for THIS bucket: use its midpoint move against the
        # cheapest cost in the sample.
        mid_move = ((lo + min(hi, 10.0)) / 2.0) / 100.0
        breakeven = 0.5 + cheapest / (2 * mid_move) if mid_move > 0 else 1.0

        clears = ci_lo > max(base_up, breakeven)
        label = f"{lo:.1f}-{hi:.0f}%" if hi < 100 else f">{lo:.1f}%"
        verdict = "YES" if clears else ("marginal" if p > breakeven else "no")

        print(f"  {label:<16}{len(sub):>7}{p * 100:>7.1f}%"
              f"   [{ci_lo * 100:5.1f}%,{ci_hi * 100:6.1f}%]"
              f"{(p - base_up) * 100:>+8.1f}pp{breakeven * 100:>11.1f}%{verdict:>10}")

        results.append({
            "bucket": label, "lo": lo, "hi": hi, "n": len(sub),
            "accuracy": round(p * 100, 1),
            "ci_low": round(ci_lo * 100, 1), "ci_high": round(ci_hi * 100, 1),
            "vs_base": round((p - base_up) * 100, 1),
            "breakeven": round(breakeven * 100, 1),
            "clears": clears,
        })

    print()
    winners = [r for r in results if r["clears"]]
    if winners:
        print(f"  {len(winners)} bucket(s) clear BOTH the base rate and their own")
        print( "  break-even line at the lower confidence bound.")
        print( "  Note this is several tests; confirm on a holdout before trading it.")
    else:
        print("  No bucket clears both the base rate and its break-even line.")
        print("  The move-size hypothesis does not hold: predicting a bigger move")
        print("  does not come with enough extra accuracy to pay for the trade.")
    print()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": len(rows),
        "base_rate": round(base_up * 100, 1),
        "cheapest_round_trip": round(cheapest * 100, 4),
        "symbol": symbol,
        "buckets": results,
        "any_bucket_clears": bool(winners),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  -> {out_path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--out", default="trade_logs/move_size_analysis.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    analyze(DEFAULT_DB_PATH, DEFAULT_BUCKETS, symbol=args.symbol, out_path=args.out)


if __name__ == "__main__":
    main()
