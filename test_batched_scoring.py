"""Tests for the batched prediction scorer.

The per-row scorer made one yfinance call per prediction. That is fine for a
handful of live rows and hopeless for a backfilled sample: 11,683 pending rows
meant 11,683 network round trips (1.5-3 hours), and the daily job could only
clear 250 a run -- it would never catch up.

Speed is worthless if the numbers change, so the central test here asserts the
batched path returns exactly what the per-row path returns.
"""

import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from logic.model_performance_tracker import (
    _returns_for_symbol,
    backfill_next_day_returns_batched,
    init_model_performance_tracker,
)
from logic.sqlite_store import connect

import logic.price_cache as pc


def setUpModule():
    """Point the price cache at a temp dir.

    _returns_for_symbol reads the on-disk cache before touching the network,
    so without isolation these tests read real cached CSVs and never exercise
    their yfinance mocks -- passing alone and failing in the full suite.
    """
    global _CACHE_TMP, _CACHE_ORIG
    _CACHE_TMP = TemporaryDirectory()
    _CACHE_ORIG = pc.CACHE_DIR
    pc.CACHE_DIR = Path(_CACHE_TMP.name) / "price_cache"


def tearDownModule():
    pc.CACHE_DIR = _CACHE_ORIG
    _CACHE_TMP.cleanup()


def _hist(dates, closes):
    return pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(dates))


def _seed(db, rows):
    init_model_performance_tracker(db)
    with connect(db) as conn:
        for i, (symbol, ts, direction) in enumerate(rows):
            conn.execute(
                "INSERT INTO model_component_performance (decision_key, timestamp, "
                "symbol, action, price_at_signal, bb_direction, bayesian_direction, "
                "gp_direction, rsi_direction, ensemble_direction) "
                "VALUES (?, ?, ?, 'buy', 100.0, ?, ?, ?, ?, ?)",
                (f"k{i}", ts, symbol, direction, direction, direction, direction, direction),
            )


class TestReturnsForSymbol(unittest.TestCase):
    def test_one_fetch_serves_many_dates(self):
        hist = _hist(
            ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"],
            [100.0, 101.0, 102.0, 103.0],
        )
        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.return_value = hist
            out = _returns_for_symbol("X", [date(2026, 8, 3), date(2026, 8, 4)])
            self.assertEqual(tk.return_value.history.call_count, 1)  # ONE call
        self.assertAlmostEqual(out[date(2026, 8, 3)], 0.01, places=6)
        self.assertAlmostEqual(out[date(2026, 8, 4)], 101.0 / 101.0 * 0 + (102.0 / 101.0 - 1), places=6)

    def test_split_artifact_is_discarded(self):
        """Same >50% guard as the per-row path: a corporate action, not a return."""
        hist = _hist(["2026-08-03", "2026-08-04"], [10.0, 31.0])
        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.return_value = hist
            out = _returns_for_symbol("KOLD", [date(2026, 8, 3)])
        self.assertEqual(out, {})

    def test_date_with_no_following_session_is_skipped(self):
        hist = _hist(["2026-08-03"], [100.0])
        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.return_value = hist
            out = _returns_for_symbol("X", [date(2026, 8, 3)])
        self.assertEqual(out, {})

    def test_ordinary_fetch_failure_is_contained_to_one_symbol(self):
        """A bad symbol must not abort scoring for the other 23."""
        with patch("yfinance.Ticker", side_effect=RuntimeError("network down")):
            self.assertEqual(_returns_for_symbol("X", [date(2026, 8, 3)]), {})

    def test_rate_limit_propagates_so_the_caller_can_report_it(self):
        """The opposite case: a throttle means the pipeline is blocked, and
        reporting 0-scored-and-fine is how 8,269 rows went silently ungraded."""
        from logic.price_cache import RateLimited

        with patch(
            "logic.price_cache.get_history",
            side_effect=RateLimited("X: rate limited after 4 attempts"),
        ):
            with self.assertRaises(RateLimited):
                _returns_for_symbol("X", [date(2026, 8, 3)])


class TestBatchedMatchesPerRow(unittest.TestCase):
    def test_identical_to_the_per_row_calculation(self):
        """Speed must not change a single number."""
        from logic.model_performance_tracker import _compute_realized_return_from_row

        hist = _hist(
            ["2026-08-03", "2026-08-04", "2026-08-05"], [24.19, 24.97, 24.50]
        )
        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.return_value = hist
            batched = _returns_for_symbol("WEAT", [date(2026, 8, 3)])[date(2026, 8, 3)]
            per_row = _compute_realized_return_from_row(
                symbol="WEAT",
                timestamp_iso="2026-08-03T14:30:00+00:00",
                price_at_signal=0.0,
            )
        self.assertAlmostEqual(batched, per_row, places=12)


class TestBackfillBatched(unittest.TestCase):
    def _old_ts(self, days_ago=5):
        return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()

    def test_scores_matured_rows_and_marks_correctness(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            ts = self._old_ts()
            _seed(db, [("WEAT", ts, "buy"), ("WEAT", ts, "sell")])
            day = datetime.fromisoformat(ts).date()
            hist = _hist(
                [day - timedelta(days=1), day, day + timedelta(days=1)],
                [100.0, 100.0, 102.0],
            )
            with patch("yfinance.Ticker") as tk:
                tk.return_value.history.return_value = hist
                n = backfill_next_day_returns_batched(db_path=db)
            self.assertEqual(n, 2)
            with connect(db) as conn:
                rows = conn.execute(
                    "SELECT bayesian_direction, bayesian_correct, next_day_return "
                    "FROM model_component_performance ORDER BY id"
                ).fetchall()
            self.assertGreater(rows[0]["next_day_return"], 0)
            self.assertEqual(rows[0]["bayesian_correct"], 1)  # buy, price rose
            self.assertEqual(rows[1]["bayesian_correct"], 0)  # sell, price rose

    def test_one_fetch_per_symbol_regardless_of_row_count(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            base = datetime.now(timezone.utc) - timedelta(days=30)
            rows = [
                ("WEAT", (base + timedelta(days=i)).isoformat(), "buy")
                for i in range(20)
            ] + [
                ("CORN", (base + timedelta(days=i)).isoformat(), "buy")
                for i in range(20)
            ]
            _seed(db, rows)
            days = [base.date() + timedelta(days=i) for i in range(25)]
            hist = _hist(days, [100.0 + i for i in range(25)])
            with patch("yfinance.Ticker") as tk:
                tk.return_value.history.return_value = hist
                backfill_next_day_returns_batched(db_path=db)
                # 40 rows, 2 symbols -> 2 fetches, not 40.
                self.assertEqual(tk.return_value.history.call_count, 2)

    def test_rows_younger_than_a_day_are_left_alone(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            _seed(db, [("WEAT", datetime.now(timezone.utc).isoformat(), "buy")])
            with patch("yfinance.Ticker") as tk:
                n = backfill_next_day_returns_batched(db_path=db)
                tk.assert_not_called()
            self.assertEqual(n, 0)

    def test_empty_backlog_is_a_noop(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            init_model_performance_tracker(db)
            self.assertEqual(backfill_next_day_returns_batched(db_path=db), 0)


if __name__ == "__main__":
    unittest.main()
