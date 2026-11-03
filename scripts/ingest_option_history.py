"""Backfill purchased option history into the Gate 0 sample.

Run ``scripts/probe_thetadata.py`` first. It costs $0 and answers the three
questions this script assumes: that our ETFs are covered, how far back the
subscription reaches, and that EOD quotes invert to usable implied vol.

Why this exists: accumulating daily snapshots reaches a conclusive 30-day Gate 0
read in **April 2029**. One month of a ThetaData subscription reaches it in an
afternoon, and the subscription can be cancelled immediately after.

Usage:
    python scripts/ingest_option_history.py --dry-run
    python scripts/ingest_option_history.py --start 2016-01-01
    python scripts/ingest_option_history.py --symbols GLD,SLV --start 2020-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logic.sqlite_store import DEFAULT_DB_PATH
from logic.thetadata_client import ThetaDataClient, liquid_option_symbols
from logic.thetadata_ingest import (
    DEFAULT_CHUNK_DAYS,
    DEFAULT_MAX_DTE,
    DEFAULT_STRIKE_RANGE,
    ingest_universe,
)

# Free-tier history start. A sensible default that works without a paid plan;
# --start reaches further once a tier is bought.
DEFAULT_START = "2023-06-01"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--start", type=str, default=DEFAULT_START)
    ap.add_argument("--end", type=str, default="")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    ap.add_argument("--strike-range", type=int, default=DEFAULT_STRIKE_RANGE)
    ap.add_argument("--max-dte", type=int, default=DEFAULT_MAX_DTE)
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and parse but write nothing; ignores saved progress")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s")

    symbols = liquid_option_symbols(args.symbols.split(",") if args.symbols else None)
    start = date.fromisoformat(args.start)
    # Yesterday by default: the free tier serves only previous-day EOD or older,
    # and today's 17:15 report may not exist yet on any tier.
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    if start > end:
        print(f"start {start} is after end {end}; nothing to do.")
        return 2

    client = ThetaDataClient()
    probe = client.probe()
    if not probe.reachable:
        print(probe.explain())
        print("\nRun scripts/probe_thetadata.py for the full pre-flight check.")
        return 2

    print(f"\n=== option history ingest{' (DRY RUN)' if args.dry_run else ''} ===")
    print(f"symbols      : {', '.join(symbols)}")
    print(f"window       : {start} .. {end}  ({(end - start).days} calendar days)")
    print(f"per request  : {args.chunk_days}d chunks, "
          f"{args.strike_range} strikes either side of spot, <= {args.max_dte} DTE")
    print(f"database     : {args.db}")
    if args.dry_run:
        print("\nNothing will be written, and saved progress is ignored so every chunk\n"
              "is re-fetched. Use this to confirm row counts and inversion rate first.")

    stats = ingest_universe(
        symbols,
        start=start,
        end=end,
        client=client,
        db_path=args.db,
        dry_run=args.dry_run,
        chunk_days=args.chunk_days,
        strike_range=args.strike_range,
        max_dte=args.max_dte,
    )

    print(f"\n{stats.summary()}")
    attempted = stats.contracts_written + stats.inversion_failures
    if attempted:
        print(f"inversion rate        : {stats.contracts_written / attempted:.1%} "
              f"({stats.inversion_failures} quotes would not invert)")
    if stats.no_spot:
        print(f"dates without a close : {stats.no_spot} "
              "(no price-cache bar on that date, so spot was unresolvable)")
    if stats.failures:
        print(f"\nfailures ({len(stats.failures)}):")
        for key, msg in list(stats.failures.items())[:10]:
            print(f"  {key}: {msg}")
        print("  Re-running resumes: completed chunks are skipped.")

    if not args.dry_run and stats.contracts_written:
        print("\nNext: python scripts/analyze_vol_edge.py")
        print("It now reports each feed and forecast method separately, so the purchased\n"
              "EOD series and the live INDICATIVE series are never pooled into one number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
