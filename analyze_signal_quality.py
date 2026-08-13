"""Does the forecaster's directional call predict anything?

    python analyze_signal_quality.py --backtest trade_logs/backtests/latest.json
    python analyze_signal_quality.py --horizon 3

Reads the per-bar decision record from a backtest run and joins it to realised
forward returns. This separates signal quality from everything the backtest
bundles together with it — sizing, bracket geometry, costs, luck — so a losing
backtest can be attributed rather than just observed.

Uses every decision, not just the ones that became trades, which is roughly 25x
the sample and the difference between "no conclusion" and a real answer.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from logic.signal_quality import (
    DEFAULT_HORIZONS,
    attach_forward_returns,
    format_report,
    load_decisions,
)

BACKTEST_DIR = Path("trade_logs") / "backtests"


def _latest_backtest() -> Path:
    candidates = sorted(BACKTEST_DIR.glob("backtest_*.json"))
    if not candidates:
        raise SystemExit(
            f"No backtest results in {BACKTEST_DIR}. Run run_backtest.py first."
        )
    return candidates[-1]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest", type=Path, default=None,
                        help="backtest result JSON (default: most recent)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="primary forward horizon in trading days (default 5)")
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS),
                        help="comma-separated horizons for the IC table")
    args = parser.parse_args()

    path = args.backtest or _latest_backtest()
    horizons = tuple(int(h) for h in str(args.horizons).split(",") if h.strip())
    if args.horizon not in horizons:
        horizons = tuple(sorted({*horizons, args.horizon}))

    decisions = load_decisions(path)
    if not decisions:
        raise SystemExit(f"No decisions found in {path}.")
    symbols = sorted({d.symbol for d in decisions})
    dates = sorted(d.date for d in decisions)
    print(f"Source: {path}")
    print(f"{len(decisions)} decisions | {len(symbols)} symbols | {dates[0]} -> {dates[-1]}")

    # Pull enough history past the last decision to fill the longest horizon.
    import yfinance as yf

    print(f"Fetching history for {len(symbols)} symbols...")
    price_history = {}
    for symbol in symbols:
        try:
            frame = yf.Ticker(symbol).history(
                start=dates[0], period=None, interval="1d"
            )
            if not frame.empty:
                price_history[symbol] = frame
        except Exception as exc:
            logging.warning("history failed for %s: %s", symbol, exc)

    if not price_history:
        raise SystemExit("Could not fetch any price history.")

    enriched = attach_forward_returns(decisions, price_history, horizons=horizons)
    if not enriched:
        raise SystemExit(
            "No decisions could be joined to forward returns. Check that the "
            "backtest dates overlap the available history."
        )
    dropped = len(decisions) - len(enriched)
    if dropped:
        print(f"({dropped} decisions dropped: no forward data within the window)")
    print()
    print(format_report(enriched, horizons=horizons, primary_horizon=args.horizon))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
