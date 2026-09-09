"""Replay signal generation over past bars to build a scoring sample fast.

The problem this solves: at the live rate of ~14.7 logged predictions per
run-day across 17 symbols, reaching a statistically useful sample takes years:

    17 symbols  ->  ~3,700/year  ->  10,000 in 2.7 years

That is far too slow to answer "does any component actually work?" Replaying
history produces the same measurement in an afternoon, because the models are
deterministic functions of past prices -- a prediction generated today from
the bars up to 2024-06-03 is the same prediction the bot would have made on
2024-06-03.

**Point-in-time correctness is the whole game here.** Every bar is generated
from a price series sliced to that date and nothing after it
(``history.loc[:bar_date]``), mirroring what logic/backtest.py already does.
If that slicing were wrong, every number downstream would be look-ahead
contaminated and worse than useless -- so it is asserted, not assumed.

Rows are written with the historical timestamp and marked
``backfilled``-distinguishable via decision_key, and scored by the same
(split-corrected) return calculation as live predictions.

    python scripts/backfill_history.py --years 2
    python scripts/backfill_history.py --years 1 --scope agriculture --limit-days 60
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.data_structures import ExecutionConfig  # noqa: E402
from logic.model_performance_tracker import (  # noqa: E402
    backfill_next_day_returns_batched,
    init_model_performance_tracker,
    log_model_decision,
    summarize_component_accuracy,
)
from logic.signal_engine import generate_signals  # noqa: E402
from logic.sqlite_store import DEFAULT_DB_PATH, init_db  # noqa: E402
from logic.universe import get_symbols  # noqa: E402


def _load_history(symbols, years: float):
    """Full daily history per symbol, fetched once and sliced per bar."""
    import yfinance as yf

    period = f"{max(1, int(years * 365) + 90)}d"
    history = {}
    for symbol in symbols:
        try:
            df = yf.Ticker(symbol).history(period=period, interval="1d")
            if df is not None and not df.empty:
                history[symbol] = df
                logging.info("history %s: %s bars", symbol, len(df))
        except Exception as exc:  # noqa: BLE001
            logging.warning("history %s failed (%s)", symbol, exc)
    return history


def _trading_days(history, warmup: int):
    """Dates present across the fetched series, past the warmup window.

    Warmup matters: the Bayesian/GP forecasters need enough history to fit, and
    a prediction made from 5 bars is not the prediction the live bot would make
    from 250. Starting after `warmup` bars keeps replayed predictions
    comparable to live ones.
    """
    if not history:
        return []
    longest = max(history.values(), key=len)
    return [d.date() for d in longest.index[warmup:]]


def replay(
    *,
    symbols,
    years: float,
    db_path=DEFAULT_DB_PATH,
    warmup: int = 250,
    limit_days: int = 0,
) -> int:
    init_db(db_path)
    init_model_performance_tracker(db_path)

    history = _load_history(symbols, years)
    if not history:
        logging.error("No history fetched; nothing to replay.")
        return 0

    days = _trading_days(history, warmup)
    if limit_days:
        days = days[-limit_days:]
    logging.info("Replaying %s trading day(s) across %s symbol(s)", len(days), len(history))

    config = ExecutionConfig(execution_mode="simulation", allow_short_selling=False)
    written = 0

    for i, bar_date in enumerate(days, 1):
        price_data = {}
        for symbol, df in history.items():
            sliced = df.loc[: str(bar_date)]
            # A prediction needs enough history to be the same kind of
            # prediction the live bot makes.
            if len(sliced) < warmup:
                continue
            # Point-in-time assertion, not assumption: nothing after bar_date
            # may be visible to the forecaster.
            assert sliced.index[-1].date() <= bar_date, (
                f"look-ahead: {symbol} slice ends {sliced.index[-1].date()} > {bar_date}"
            )
            price_data[symbol] = sliced

        if not price_data:
            continue

        try:
            signals = generate_signals(
                list(price_data), config, verbose=False,
                price_data=price_data, build_plot=False,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("signal generation failed on %s (%s)", bar_date, exc)
            continue

        ts = datetime(bar_date.year, bar_date.month, bar_date.day, 14, 30, tzinfo=timezone.utc)
        for sig in signals or []:
            snapshot = sig.meta.get("component_snapshot")
            if not snapshot:
                continue
            log_model_decision(
                decision_key=f"backfill|{ts.isoformat()}|{sig.symbol}",
                timestamp=ts,
                symbol=sig.symbol,
                action=sig.signal_type,
                price_at_signal=float(sig.meta.get("current_price") or 0.0) or None,
                component_snapshot=snapshot,
                db_path=db_path,
            )
            written += 1

        if i % 10 == 0 or i == len(days):
            logging.info("  %s/%s days replayed, %s predictions logged", i, len(days), written)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=2.0, help="history depth to replay")
    ap.add_argument("--scope", default="all", help="universe scope")
    ap.add_argument("--warmup", type=int, default=250, help="bars required before predicting")
    ap.add_argument("--limit-days", type=int, default=0, help="replay only the last N days")
    ap.add_argument("--no-score", action="store_true", help="skip the scoring pass")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    symbols = get_symbols(args.scope)
    written = replay(
        symbols=symbols, years=args.years, warmup=args.warmup, limit_days=args.limit_days
    )
    logging.info("BACKFILL COMPLETE: %s prediction(s) logged", written)

    if not args.no_score and written:
        total = backfill_next_day_returns_batched(db_path=DEFAULT_DB_PATH)
        logging.info("SCORING COMPLETE: %s newly scored", total)
        report = summarize_component_accuracy(db_path=DEFAULT_DB_PATH)
        if report:
            logging.info("COMPONENT ACCURACY:\n%s", report)


if __name__ == "__main__":
    main()
