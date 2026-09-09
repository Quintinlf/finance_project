"""Tests for the on-disk price cache and rate-limit handling.

Root cause it addresses: the 3-year backfill fetched history for 24 symbols,
then scoring refetched the same history. yfinance answered "Too Many Requests"
for every symbol, the scorer logged warnings, returned 0, and 8,269
predictions went ungraded while the run reported success.
"""

import time
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import logic.price_cache as pc
from logic.price_cache import (
    RateLimited,
    fetch_with_retry,
    get_history,
    load_cached,
    save_cached,
)


def _frame(n=10, start="2026-08-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"Close": [100.0 + i for i in range(n)]}, index=idx)


class _CacheDir:
    """Point the module's CACHE_DIR at a temp directory for the test."""

    def __enter__(self):
        self._tmp = TemporaryDirectory()
        self._orig = pc.CACHE_DIR
        pc.CACHE_DIR = Path(self._tmp.name) / "price_cache"
        return pc.CACHE_DIR

    def __exit__(self, *exc):
        pc.CACHE_DIR = self._orig
        self._tmp.cleanup()


class TestCacheRoundTrip(unittest.TestCase):
    def test_saved_frame_reads_back(self):
        with _CacheDir():
            save_cached("WEAT", _frame())
            back = load_cached("WEAT")
            self.assertIsNotNone(back)
            self.assertEqual(len(back), 10)

    def test_missing_symbol_returns_none(self):
        with _CacheDir():
            self.assertIsNone(load_cached("NOPE"))

    def test_stale_cache_is_rejected(self):
        """Splits retroactively rewrite adjusted closes, so an old cache can
        silently serve pre-split prices."""
        with _CacheDir() as d:
            save_cached("WEAT", _frame())
            old = time.time() - (30 * 86400)
            import os

            os.utime(d / "WEAT.csv", (old, old))
            self.assertIsNone(load_cached("WEAT", max_age_days=7))
            self.assertIsNotNone(load_cached("WEAT", max_age_days=60))


class TestRetry(unittest.TestCase):
    def test_retries_through_a_transient_rate_limit(self):
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Too Many Requests. Rate limited.")
            return _frame()

        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.side_effect = flaky
            with patch("time.sleep"):  # don't actually wait in tests
                out = fetch_with_retry("WEAT", period="400d")
        self.assertEqual(len(out), 10)
        self.assertEqual(calls["n"], 3)

    def test_raises_rate_limited_after_exhausting_attempts(self):
        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.side_effect = RuntimeError("Too Many Requests")
            with patch("time.sleep"):
                with self.assertRaises(RateLimited):
                    fetch_with_retry("WEAT", period="400d", attempts=3)

    def test_non_rate_limit_error_is_not_retried(self):
        """Retrying a malformed request just wastes time."""
        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.side_effect = ValueError("bad ticker")
            with self.assertRaises(ValueError):
                fetch_with_retry("???", period="400d")
            self.assertEqual(tk.return_value.history.call_count, 1)

    def test_empty_result_is_an_answer_not_a_throttle(self):
        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.return_value = pd.DataFrame()
            out = fetch_with_retry("DELISTED", period="400d")
            self.assertTrue(out.empty)
            self.assertEqual(tk.return_value.history.call_count, 1)


class TestGetHistory(unittest.TestCase):
    def test_cache_hit_avoids_the_network_entirely(self):
        with _CacheDir():
            save_cached("WEAT", _frame(n=30, start="2026-08-01"))
            with patch("yfinance.Ticker") as tk:
                out = get_history(
                    "WEAT", start=date(2026, 8, 2), end=date(2026, 8, 20)
                )
                tk.assert_not_called()
            self.assertEqual(len(out), 30)

    def test_partial_coverage_triggers_a_refetch(self):
        """Returning a short series would make the caller compute returns from
        the wrong bars."""
        with _CacheDir():
            save_cached("WEAT", _frame(n=5, start="2026-08-01"))
            with patch("logic.price_cache.fetch_with_retry", return_value=_frame(n=60)) as f:
                get_history("WEAT", start=date(2026, 7, 1), end=date(2026, 9, 1))
                f.assert_called_once()

    def test_allow_fetch_false_never_hits_the_network(self):
        with _CacheDir():
            with patch("yfinance.Ticker") as tk:
                out = get_history("WEAT", start=date(2026, 8, 1), allow_fetch=False)
                tk.assert_not_called()
            self.assertIsNone(out)

    def test_successful_fetch_populates_the_cache(self):
        with _CacheDir():
            with patch("logic.price_cache.fetch_with_retry", return_value=_frame(n=40)):
                get_history("CORN", start=date(2026, 8, 1))
            self.assertIsNotNone(load_cached("CORN"))


if __name__ == "__main__":
    unittest.main()
