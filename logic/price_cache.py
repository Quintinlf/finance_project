"""On-disk daily price cache with rate-limit-aware fetching.

Why this exists: the 3-year backfill fetched full history for 24 symbols, then
the scoring pass tried to fetch the *same* history again to compute returns.
yfinance answered "Too Many Requests" for all 24 symbols, the batched scorer
logged one warning per symbol, returned 0, and moved on. 8,269 predictions
silently went ungraded -- and a daily job would have kept reporting
"MODEL SCORING: 0" indefinitely without anything looking broken.

Two fixes, both needed:

1. **Cache.** Daily bars for a past date do not change from run to run, so
   refetching them is pure waste. Cached CSVs live in ``trade_logs/price_cache``
   and are small enough to commit, which also means CI starts warm instead of
   hammering the API on every run.

2. **Retry with backoff, and a loud failure.** A rate limit is transient;
   giving up on the first one throws away work that would succeed seconds
   later. And when fetching genuinely fails, callers need to tell that apart
   from "this symbol has no data" -- the first is a broken pipeline, the second
   is a normal empty result.

Staleness matters for one specific reason: yfinance back-adjusts history for
splits, so a split retroactively rewrites every prior close. A cache older than
``MAX_CACHE_AGE_DAYS`` is refetched in full rather than extended, which bounds
how long a missed corporate action can sit in the data.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

CACHE_DIR = Path("trade_logs") / "price_cache"

# Refetch rather than reuse beyond this age: a split rewrites historical
# adjusted closes, and a stale cache would silently serve pre-split prices.
MAX_CACHE_AGE_DAYS = 7

_RATE_LIMIT_MARKERS = ("too many requests", "rate limit", "429")


class RateLimited(RuntimeError):
    """Fetching failed because the data provider throttled us."""


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.csv"


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def load_cached(symbol: str, max_age_days: int = MAX_CACHE_AGE_DAYS):
    """Return cached history, or None when missing or too old to trust."""
    path = _cache_path(symbol)
    if not path.exists():
        return None

    age_days = (time.time() - path.stat().st_mtime) / 86400.0
    if age_days > max_age_days:
        logging.debug("price cache for %s is %.1f days old; refetching", symbol, age_days)
        return None

    try:
        import pandas as pd

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not read price cache for %s (%s).", symbol, exc)
        return None


def save_cached(symbol: str, df) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_cache_path(symbol))
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not write price cache for %s (%s).", symbol, exc)


def fetch_with_retry(
    symbol: str,
    *,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    attempts: int = 4,
    base_delay: float = 2.0,
):
    """Fetch daily bars, retrying through transient rate limits.

    Raises ``RateLimited`` when every attempt is throttled, so the caller can
    report a broken pipeline instead of mistaking it for an empty symbol.
    """
    import yfinance as yf

    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = (
                ticker.history(period=period, interval="1d")
                if period
                else ticker.history(start=start, end=end, interval="1d")
            )
            if df is not None and not df.empty:
                return df
            # An empty frame is a real answer (delisted, bad ticker), not a
            # throttle -- do not burn retries on it.
            return df
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_rate_limit(exc):
                raise
            if attempt < attempts:
                delay = base_delay * (2 ** (attempt - 1))
                logging.info(
                    "Rate limited on %s (attempt %s/%s); backing off %.0fs",
                    symbol, attempt, attempts, delay,
                )
                time.sleep(delay)

    raise RateLimited(f"{symbol}: rate limited after {attempts} attempts ({last_exc})")


def get_history(
    symbol: str,
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
    max_age_days: int = MAX_CACHE_AGE_DAYS,
    allow_fetch: bool = True,
):
    """Daily bars for a symbol, preferring cache and falling back to fetching.

    A cached frame is used only when it actually spans the requested window;
    partial coverage triggers a refetch, because silently returning a short
    series would make the caller compute returns from the wrong bars.
    """
    cached = load_cached(symbol, max_age_days=max_age_days)
    if cached is not None and len(cached):
        covered_start = cached.index[0].date()
        covered_end = cached.index[-1].date()
        if (start is None or covered_start <= start) and (end is None or covered_end >= end):
            return cached

    if not allow_fetch:
        return cached

    span_days = 400
    if start is not None:
        span_days = max(span_days, (date.today() - start).days + 60)

    df = fetch_with_retry(symbol, period=f"{span_days}d")
    if df is not None and not df.empty:
        save_cached(symbol, df)
    return df
