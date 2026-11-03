"""Run the eight-arm walk-forward ablation and write the report.

The report is the input the live feature policy reads. A missing report
leaves the current model in place. This script does not place orders.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from logic.feature_policy import REPORT_PATH
from logic.walk_forward import format_report, run_ablation, write_report

# Liquid enough to have two years of bars, cheap enough to match the paper book.
DEFAULT_SYMBOLS = ["USO", "SLV", "DBA", "UNG", "DBC", "CPER", "WEAT", "CORN"]


def _download(symbols, start, end) -> dict:
    import yfinance as yf

    frames = {}
    for symbol in symbols:
        df = yf.download(symbol, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=True)
        if df is None or df.empty:
            print(f"no bars for {symbol}")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        frames[symbol] = df
        print(f"{symbol}: {len(df)} bars")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Walk-forward feature ablation")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--no-gp", action="store_true", help="Skip the H6 Gaussian-process arm")
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=int(args.years * 365))
    history = _download(args.symbols, start, end)
    if not history:
        raise SystemExit("No history downloaded. Report not written, so nothing is promoted or removed.")

    report = run_ablation(history, draws=args.draws, include_gp=not args.no_gp)
    path = write_report(report, Path(args.out))
    print(format_report(report))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
