"""Gate 1: do option spreads leave room for any volatility edge at all?

This is the cheapest gate in the Tier 2 scope and it runs today, which is the
whole point of it. Gate 0 — whether our forecast beats implied vol — needs
months of accumulated snapshots. Gate 1 needs one measurement during market
hours, and if it fails there is no reason to wait for Gate 0 at all.

The arithmetic. An at-the-money option is approximately

    price ~= 0.4 * S * sigma * sqrt(T)

so its sensitivity to volatility is

    dprice/dsigma ~= 0.4 * S * sqrt(T) ~= price / sigma

A volatility edge of `d_sigma` is therefore worth about `price * d_sigma / sigma`,
while crossing the quoted spread costs `spread_fraction * price` for the round
trip (half the spread entering, half leaving). Setting them equal:

    break-even vol edge = sigma * spread_fraction

That is the result worth internalising: **the volatility edge you need is the
implied vol scaled by the spread you pay.** A 60% vol name quoting a 20% spread
demands a 12-volatility-point forecasting advantage over the options market
before a single trade is worth placing. For reference, the entire measured
volatility persistence in this project (r = 0.708 against *realized* vol, not
against implied) is nowhere near that, and that figure is not even an edge over
the market — it is an edge over nothing.

Quotes outside 09:30-16:00 ET are worthless for this. After-hours option books
routinely quote several times their intraday width, so measuring then would
manufacture a failure as surely as measuring at the wrong time manufactured the
original spread estimates. The script refuses to run outside market hours unless
--force is passed, and labels the output as unusable when it is.

Usage:
    python scripts/measure_option_spreads.py                # all universe symbols
    python scripts/measure_option_spreads.py --symbols UNG,GLD
    python scripts/measure_option_spreads.py --force        # off-hours, diagnostic only
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logic.costs import is_regular_market_hours
from logic.iv_snapshot import _spread_pct_of_mid
from logic.options_engine import _get_spot_price, get_option_chain
from logic.universe import get_symbols

OUT_PATH = Path("trade_logs") / "option_spreads.json"

# Near-the-money and near-dated is where a vol trade would actually live, and
# also where spreads are tightest — so this measurement is deliberately
# generous to the strategy. The wings are worse.
MAX_ABS_MONEYNESS = 0.10
MIN_DTE, MAX_DTE = 7, 60

# The edge the project has actually measured, in vol points, being generous.
# Trailing realized vol predicts forward realized vol at r = 0.708, which given
# typical vol-of-vol on this universe corresponds to single-digit vol points of
# forecast improvement over a naive estimate -- and *zero* measured improvement
# over implied vol, which is the benchmark that matters. Used only to put the
# break-even number in context.
GENEROUS_ASSUMED_EDGE_VOL_POINTS = 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--force", action="store_true", help="measure outside market hours")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    hours_ok = is_regular_market_hours()
    if not hours_ok and not args.force:
        print(
            "REFUSING TO MEASURE: outside 09:30-16:00 ET.\n"
            "After-hours option books quote several times their intraday width, so a\n"
            "measurement taken now would fail Gate 1 for the wrong reason. Re-run during\n"
            "regular market hours, or pass --force for a diagnostic-only reading."
        )
        return 2

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else list(get_symbols("all"))
    )

    print(f"\nGate 1: option spread measurement — {len(symbols)} symbols")
    print(f"market hours: {'YES' if hours_ok else 'NO (diagnostic only, --force)'}")
    print(f"filtered to |moneyness| <= {MAX_ABS_MONEYNESS:.0%}, {MIN_DTE}-{MAX_DTE} DTE\n")
    print(f"{'sym':6}{'n':>5}{'IV':>8}{'spread':>9}{'$ spread':>10}{'break-even edge':>17}")
    print("-" * 56)

    per_symbol = {}
    for symbol in symbols:
        try:
            spot = _get_spot_price(symbol)
            chain = get_option_chain(symbol)
        except Exception as exc:
            print(f"{symbol:6}  ! {type(exc).__name__}")
            continue
        if not spot or not chain:
            print(f"{symbol:6}  ! no chain or spot")
            continue

        today = datetime.now(timezone.utc).date()
        spreads, ivs, dollar_spreads = [], [], []
        for c in chain:
            iv, strike = c.get("implied_volatility"), c.get("strike")
            bid, ask = c.get("bid"), c.get("ask")
            if iv is None or not strike or bid is None or ask is None:
                continue
            if abs((float(strike) - spot) / spot) > MAX_ABS_MONEYNESS:
                continue
            exp = c.get("expiration")
            if exp:
                try:
                    dte = (datetime.fromisoformat(str(exp)).date() - today).days
                except ValueError:
                    continue
                if not (MIN_DTE <= dte <= MAX_DTE):
                    continue
            pct = _spread_pct_of_mid(bid, ask)
            if pct is None:
                continue
            spreads.append(pct)
            ivs.append(float(iv))
            dollar_spreads.append(float(ask) - float(bid))

        if not spreads:
            print(f"{symbol:6}  ! no quotable near-the-money contracts")
            continue

        med_spread = statistics.median(spreads)
        med_iv = statistics.median(ivs)
        med_dollar = statistics.median(dollar_spreads)
        # break-even vol edge = sigma * spread_fraction  (see module docstring)
        be_vol_points = med_iv * (med_spread / 100.0) * 100.0

        per_symbol[symbol] = {
            "contracts": len(spreads),
            "median_iv": med_iv,
            "median_spread_pct_of_mid": med_spread,
            "median_dollar_spread": med_dollar,
            "break_even_vol_points": be_vol_points,
        }
        print(
            f"{symbol:6}{len(spreads):5d}{med_iv:7.1%}{med_spread:8.1f}%"
            f"{med_dollar:10.2f}{be_vol_points:14.1f} pts"
        )

    if not per_symbol:
        print("\nNo measurable contracts. Cannot evaluate Gate 1.")
        return 1

    all_be = [v["break_even_vol_points"] for v in per_symbol.values()]
    all_spread = [v["median_spread_pct_of_mid"] for v in per_symbol.values()]
    med_be = statistics.median(all_be)
    best = min(per_symbol.items(), key=lambda kv: kv[1]["break_even_vol_points"])

    print(f"\nmedian spread across symbols : {statistics.median(all_spread):.1f}% of mid")
    print(f"median break-even vol edge   : {med_be:.1f} vol points")
    print(f"cheapest symbol              : {best[0]} at "
          f"{best[1]['break_even_vol_points']:.1f} vol points")
    print(f"\nFor context, a generous reading of the measured volatility persistence is")
    print(f"~{GENEROUS_ASSUMED_EDGE_VOL_POINTS:.0f} vol points of advantage over a NAIVE forecast — and zero measured")
    print("advantage over implied volatility, which is the benchmark that matters.")

    verdict = (
        "PASS" if best[1]["break_even_vol_points"] <= GENEROUS_ASSUMED_EDGE_VOL_POINTS
        else "FAIL"
    )
    print(f"\nGATE 1 (provisional): {verdict}")
    if verdict == "FAIL":
        print(
            "  Even the cheapest symbol demands more forecasting advantage than the\n"
            "  strategy could plausibly have. Gate 0 would be answering a question that\n"
            "  no longer matters. Stop here rather than accumulating six months of data."
        )
    else:
        print(
            "  Spreads leave room in principle. This does NOT mean there is an edge —\n"
            "  Gate 0 is still the binding question and still needs the accumulated data."
        )
    if not hours_ok:
        print("\n  ** DIAGNOSTIC ONLY: taken outside market hours. Not a verdict. **")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                "regular_market_hours": hours_ok,
                "usable": hours_ok,
                "filters": {
                    "max_abs_moneyness": MAX_ABS_MONEYNESS,
                    "min_dte": MIN_DTE,
                    "max_dte": MAX_DTE,
                },
                "median_break_even_vol_points": med_be,
                "verdict": verdict if hours_ok else "DIAGNOSTIC_ONLY",
                "per_symbol": per_symbol,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
