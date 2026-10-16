"""Cross affordability against transaction cost, per symbol.

Affordability and cost pull in opposite directions, and nothing in the engine
displayed that tension in one place. Screening on affordability alone actively
selects for the worst-cost names: the instruments cheap enough to trade
profitably (XLE at 12bp, GLD at 12bp) are the expensive ones per share, and the
instruments cheap enough per share (CANE, SOYB at 45bp) are the thin ones.

For a predicted move ``m`` and round-trip cost ``c``, a win returns ``m`` and a
loss costs ``m``, so expected value is ``m(2p - 1) - c`` and break-even sits at

    p* = 0.5 + c / (2m)

That is the accuracy the symbol demands before a trade in it is worth placing.
Printed against the accuracy the model actually delivers, it shows which names
are reachable at all and which are arithmetic dead ends.

Usage:
    python scripts/analyze_tradeability.py [--equity 494] [--cap 0.20]
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import date, timedelta

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from logic.costs import DEFAULT_SLIPPAGE_BPS, DEFAULT_SPREAD_BPS
from logic.price_cache import get_history
from logic.universe import DEFAULT_MIN_SHARES, ALL_ASSETS

# Measured over 24,533 scored predictions; see docs/reports/TECHNICAL_FINDINGS.md.
MEAN_ABS_FORECAST_PCT = 0.586
DELIVERED_ACCURACY = 0.532
BASE_RATE = 0.516


def round_trip_pct(symbol: str) -> float:
    """Full spread crossed once each way, plus slippage charged per fill."""
    spread = DEFAULT_SPREAD_BPS.get(symbol, 30.0)
    return (spread + 2 * DEFAULT_SLIPPAGE_BPS) / 100.0


def break_even_accuracy(cost_pct: float, move_pct: float = MEAN_ABS_FORECAST_PCT) -> float:
    return 0.5 + cost_pct / (2 * move_pct)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, default=494.0)
    ap.add_argument("--cap", type=float, default=0.20)
    ap.add_argument("--min-shares", type=int, default=DEFAULT_MIN_SHARES)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    budget = args.equity * args.cap
    end = date.today()
    start = end - timedelta(days=30)

    rows = []
    for asset in ALL_ASSETS:
        try:
            price = float(get_history(asset.symbol, start=start, end=end)["Close"].iloc[-1])
        except Exception as exc:
            print(f"  ! {asset.symbol}: price unavailable ({type(exc).__name__})")
            continue
        shares = int(budget // price)
        cost = round_trip_pct(asset.symbol)
        rows.append((asset.symbol, price, shares, cost, break_even_accuracy(cost)))

    print(f"\nequity ${args.equity:,.2f} | cap {args.cap:.0%} | ${budget:,.2f} per position")
    print(f"model delivers {DELIVERED_ACCURACY:.1%} against a {BASE_RATE:.1%} base rate\n")
    print(f"{'sym':6}{'price':>10}{'shares':>8}{'cost':>8}{'break-even':>12}   status")
    print("-" * 62)

    reachable = 0
    for symbol, price, shares, cost, be in sorted(rows, key=lambda r: r[3]):
        if shares < 1:
            status = "unreachable"
        elif shares < args.min_shares:
            status = "no sizing resolution"
        else:
            status = "tradeable"
            reachable += 1
        gap = be - DELIVERED_ACCURACY
        flag = "" if gap <= 0 else f"  (short {gap * 100:.1f}pp)"
        print(
            f"{symbol:6}${price:9,.2f}{shares:8d}{cost:7.2f}%{be:11.1%}   "
            f"{status}{flag if status == 'tradeable' else ''}"
        )

    print(f"\n{reachable}/{len(rows)} symbols tradeable at this account size.")
    cheapest = min((r for r in rows if r[2] >= args.min_shares), key=lambda r: r[3], default=None)
    if cheapest:
        print(
            f"Cheapest tradeable: {cheapest[0]} at {cheapest[3]:.2f}% round trip, "
            f"demanding {cheapest[4]:.1%} vs {DELIVERED_ACCURACY:.1%} delivered."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
