"""Free-tier coverage probe: can we actually buy our way out of April 2029?

Accumulating IV snapshots reaches a conclusive 30-day Gate 0 read in **2029**.
Buying history collapses that to one month's subscription. Before paying for
anything, three things have to be true, and none of them is documented clearly
enough to take on faith:

1. **ETF options are covered at all.** ThetaData's materials say "US stocks and
   options" and "US Index & Stocks" and never once name an ETF. GLD and SLV
   options are OPRA-listed equity options, so they should be there -- but
   "should be" is how this project previously spent two weeks running stale code
   in CI. ``option/list/symbols`` settles it.

2. **How much history the tier actually grants.** The free tier's own page
   contradicts itself: the prose says one year, the access table says
   2023-06-01. ``option/list/dates`` measures the boundary instead of reading
   about it.

3. **The cheap tier is usable without the IV endpoint.** IV is Standard/Pro
   only. Our plan is to invert Black-Scholes off the EOD quote, which works only
   if the EOD rows carry a two-sided quote that inverts to a sane number. This
   is the step that could actually fail, and it is the reason the probe pulls a
   real day of data rather than stopping at the listings.

Everything here is read-only and runs against the **free** tier at $0. Nothing
is written to the database.

Usage:
    python scripts/probe_thetadata.py
    python scripts/probe_thetadata.py --symbols GLD,SLV --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logic.black_scholes_iv import implied_vol_from_quote
from logic.thetadata_client import (
    PUBLISHED_HISTORY_START,
    ThetaDataClient,
    ThetaDataUnavailable,
    java_version,
    liquid_option_symbols,
)

# Mirrors scripts/analyze_vol_edge.py exactly. Duplicating the constants would
# let the two drift and produce a projection that disagrees with the analysis
# it is projecting.
from scripts.analyze_vol_edge import (  # noqa: E402
    EFFECTIVE_SYMBOL_CLUSTERS,
    MIN_EFFECTIVE_N_CONCLUSIVE,
    MIN_EFFECTIVE_N_PRELIMINARY,
)

HORIZONS = (7, 14, 30)
TRADING_DAYS_PER_YEAR = 252.0

# The probe pulls one day, near the money, short-dated -- the slice the ingest
# will keep. Asking for the whole chain would be slower and prove less.
PROBE_STRIKE_RANGE = 4
PROBE_MAX_DTE = 60


def _verdict(eff_n: float) -> str:
    if eff_n >= MIN_EFFECTIVE_N_CONCLUSIVE:
        return "CONCLUSIVE"
    if eff_n >= MIN_EFFECTIVE_N_PRELIMINARY:
        return "preliminary"
    return "insufficient"


def _project(trading_days: int, n_symbols: int) -> None:
    """What Gate 0 could say if this much history were ingested."""
    print(f"\n  If ingested, {trading_days} trading days x {n_symbols} symbols would support:")
    print(f"    {'horizon':>8}{'raw N':>9}{'effective N':>14}{'verdict':>14}")
    for h in HORIZONS:
        raw = trading_days * n_symbols
        eff = raw / h / EFFECTIVE_SYMBOL_CLUSTERS
        print(f"    {h:>7}d{raw:>9}{eff:>14.0f}{_verdict(eff):>14}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="",
                    help="comma-separated; defaults to the liquid-option set")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    wanted = liquid_option_symbols(args.symbols.split(",") if args.symbols else None)
    client = ThetaDataClient()

    print("\n=== ThetaData coverage probe (free tier, $0) ===")

    # ---------------------------------------------------------------- step 1
    probe = client.probe()
    print(f"\n[1/4] terminal reachable?")
    print(probe.explain())
    if not probe.reachable:
        java = java_version()
        if java is None:
            print("\n  Java is not on PATH, so step 1 is the blocker: install Java 21+ first.")
        else:
            print(f"\n  Java is installed ({java}), so only the terminal itself is missing.")
        print(
            "\nNothing further can be checked until the terminal is running. This is a\n"
            "local-process requirement, not a paid feature -- the free tier is enough\n"
            "for every check in this script."
        )
        return 2

    # ---------------------------------------------------------------- step 2
    print(f"\n[2/4] are ETF options covered?")
    try:
        available = set(client.list_option_symbols())
    except ThetaDataUnavailable as exc:
        print(f"  FAILED: {exc}")
        return 1

    covered = [s for s in wanted if s in available]
    missing = [s for s in wanted if s not in available]
    print(f"  {len(available)} option symbols indexed by the terminal")
    print(f"  covered : {', '.join(covered) if covered else '(none)'}")
    if missing:
        print(f"  MISSING : {', '.join(missing)}")
    if not covered:
        print(
            "\n  None of our symbols have options here. That kills the purchase: the\n"
            "  cost argument rested on these eight names specifically. Do not subscribe."
        )
        return 1

    # ---------------------------------------------------------------- step 3
    print(f"\n[3/4] how much history does this subscription actually grant?")
    print(f"  {'sym':6}{'first date':>13}{'last date':>13}{'days':>8}")
    print("  " + "-" * 38)
    spans = {}
    for symbol in covered:
        try:
            dates = client.list_option_dates(symbol)
        except ThetaDataUnavailable as exc:
            print(f"  {symbol:6}  ! {exc}")
            continue
        if not dates:
            print(f"  {symbol:6}{'(no dates)':>13}")
            continue
        spans[symbol] = dates
        print(f"  {symbol:6}{dates[0]:>13}{dates[-1]:>13}{len(dates):>8}")

    if not spans:
        print("\n  No dated history for any covered symbol. Cannot evaluate.")
        return 1

    earliest = min(d[0] for d in spans.values())
    shortest = min(len(d) for d in spans.values())
    print(f"\n  earliest date across symbols : {earliest}")
    print(f"  shortest per-symbol history  : {shortest} days")
    for tier, start in PUBLISHED_HISTORY_START.items():
        if earliest <= start:
            print(f"  consistent with the published {tier} start ({start}) or better")
            break
    _project(shortest, len(spans))

    # ---------------------------------------------------------------- step 4
    # The step that can actually fail. The cheap tiers have no IV endpoint, so
    # the whole plan depends on EOD quotes inverting to sane implied vols.
    print(f"\n[4/4] do EOD quotes invert to usable implied vol?")
    print("  (the cheap tiers have no IV endpoint; this is the load-bearing check)")

    probe_symbol = covered[0]
    probe_dates = spans.get(probe_symbol, [])
    # Second-to-last available date: the free tier serves only yesterday or
    # older, and the most recent row can be a partially built report.
    probe_date = probe_dates[-2] if len(probe_dates) >= 2 else probe_dates[-1]

    try:
        rows = client.option_history_eod(
            probe_symbol,
            start=probe_date,
            end=probe_date,
            strike_range=PROBE_STRIKE_RANGE,
            max_dte=PROBE_MAX_DTE,
        )
    except ThetaDataUnavailable as exc:
        print(f"  FAILED: {exc}")
        return 1

    print(f"  {probe_symbol} on {probe_date}: {len(rows)} near-the-money rows")
    if not rows:
        print(
            "  Empty. Either strike_range/max_dte filtered everything out, or the\n"
            "  subscription does not serve this date. Re-run with --verbose."
        )
        return 1

    spot = _infer_spot(rows)
    if spot is None:
        print("  Could not infer spot from the returned strikes; cannot invert.")
        return 1
    print(f"  inferred spot (strike midpoint): {spot:.2f}")

    two_sided = inverted = 0
    samples = []
    for r in rows:
        bid, ask = _f(r.get("bid")), _f(r.get("ask"))
        if bid is not None and ask is not None and bid > 0 and ask >= bid:
            two_sided += 1
        iv = implied_vol_from_quote(
            symbol=probe_symbol,
            snapshot_date=probe_date,
            expiration=r.get("expiration"),
            strike=_f(r.get("strike")) or 0.0,
            option_type=str(r.get("right") or "call"),
            spot=spot,
            bid=bid,
            ask=ask,
            close=_f(r.get("close")),
        )
        if iv is not None:
            inverted += 1
            samples.append((r.get("expiration"), _f(r.get("strike")), r.get("right"), iv))

    print(f"  with a two-sided quote : {two_sided}/{len(rows)}")
    print(f"  inverted to an IV      : {inverted}/{len(rows)}")

    if args.verbose and samples:
        print("\n  sample inversions:")
        for exp, strike, right, iv in samples[:8]:
            print(f"    {exp}  {strike:>8.2f} {str(right):<5} IV {iv:6.1%}")

    if inverted == 0:
        print(
            "\n  PROBE FAILED on the load-bearing check. No quote inverted. The cheap\n"
            "  tiers would be unusable and the purchase would have to be Standard or\n"
            "  Pro for their IV endpoint -- re-price the decision before subscribing."
        )
        return 1

    rate = inverted / len(rows)
    ivs = [s[3] for s in samples]
    print(f"  IV range               : {min(ivs):.1%} .. {max(ivs):.1%}")

    print("\n=== VERDICT ===")
    if rate < 0.5:
        print(
            f"PARTIAL: only {rate:.0%} of near-the-money quotes invert. Usable, but the\n"
            "effective sample will be smaller than the day count suggests. Re-run the\n"
            "projection above scaled by this rate before committing."
        )
        return 0

    print(
        f"PASS: {len(covered)}/{len(wanted)} symbols covered, history back to {earliest},\n"
        f"      {rate:.0%} of near-the-money EOD quotes invert to a sane implied vol.\n"
        "\nThe free tier is sufficient to prove the pipeline. For the conclusive 30-day\n"
        "read, one month of a paid tier buys the deeper history -- then cancel.\n"
        "\nNext: python scripts/ingest_option_history.py --dry-run"
    )
    return 0


def _f(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _infer_spot(rows):
    """Approximate spot as the midpoint of the returned strike range.

    ``strike_range=n`` centres the strikes the server returns on spot, so their
    midpoint is close to it. This is only for the probe's inversion sanity
    check; the ingest reads real closes from the price cache instead.
    """
    strikes = sorted({s for s in (_f(r.get("strike")) for r in rows) if s})
    if not strikes:
        return None
    return (strikes[0] + strikes[-1]) / 2.0


if __name__ == "__main__":
    raise SystemExit(main())
