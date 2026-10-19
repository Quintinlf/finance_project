"""Tests for the IV/vol-forecast snapshotter — Phase 0a of the Tier 2 scope.

The point of this module is to accumulate a sample that will be trusted in six
months, by which time nobody will remember how it was collected. So the things
worth testing are the ones that would quietly corrupt that sample rather than
crash: a forecast recorded with the wrong method label, a re-run duplicating a
day, a crossed quote becoming a plausible-looking spread, or a chain failure
silently taking the forecast series down with it.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from logic.iv_snapshot import (
    FORECAST_METHOD,
    MIN_TRADING_DAYS_FOR_GATE0,
    _spread_pct_of_mid,
    coverage,
    init_iv_tables,
    snapshot_universe,
)
from logic.sqlite_store import connect


def _chain(underlying="UNG", *, iv=0.55, bid=0.83, ask=1.03, n=3):
    """A chain shaped like what options_engine.get_option_chain returns."""
    return [
        {
            "symbol": f"{underlying}261218C0001{i}000",
            "underlying": underlying,
            "expiration": "2026-12-18",
            "strike": 10.0 + i,
            "type": "call",
            "delta": 0.5,
            "gamma": 0.15,
            "theta": -0.009,
            "vega": 0.017,
            "implied_volatility": iv,
            "bid": bid,
            "ask": ask,
            "quote_timestamp": "2026-10-16T20:14:59+00:00",
        }
        for i in range(n)
    ]


class TestSpreadArithmetic(unittest.TestCase):
    def test_spread_is_a_percentage_of_mid(self):
        # 0.83/1.03 -> mid 0.93, spread 0.20 -> 21.5%
        self.assertAlmostEqual(_spread_pct_of_mid(0.83, 1.03), 21.505, places=2)

    def test_crossed_quote_returns_none_not_a_negative_spread(self):
        """A crossed book must not become a tidy negative number downstream."""
        self.assertIsNone(_spread_pct_of_mid(1.03, 0.83))

    def test_zero_bid_returns_none(self):
        """A zero bid makes the percentage meaningless, not merely large."""
        self.assertIsNone(_spread_pct_of_mid(0.0, 0.50))

    def test_missing_side_returns_none(self):
        self.assertIsNone(_spread_pct_of_mid(None, 1.03))
        self.assertIsNone(_spread_pct_of_mid(0.83, None))


class TestSnapshotWrites(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "t.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *, chain=None, spot=10.5, forecast=0.48, symbols=("UNG",), date="2026-10-19"):
        with patch("logic.options_engine.get_option_chain",
                   return_value=_chain() if chain is None else chain), \
             patch("logic.options_engine._get_spot_price", return_value=spot), \
             patch("logic.volatility_sizing.predict_volatility", return_value=forecast):
            return snapshot_universe(symbols=list(symbols), db_path=self.db,
                                     snapshot_date=date)

    def test_records_iv_and_the_forecast_together(self):
        res = self._run()
        self.assertTrue(res.ok)
        self.assertEqual(res.forecasts_written, 1)
        self.assertGreater(res.contracts_written, 0)

        with connect(self.db) as conn:
            iv_row = conn.execute("SELECT * FROM iv_snapshots LIMIT 1").fetchone()
            fc_row = conn.execute("SELECT * FROM vol_forecasts LIMIT 1").fetchone()
        self.assertAlmostEqual(iv_row["implied_volatility"], 0.55)
        self.assertAlmostEqual(iv_row["spread_pct_of_mid"], 21.505, places=2)
        self.assertAlmostEqual(fc_row["forecast_vol"], 0.48)

    def test_forecast_carries_its_method_label(self):
        """Changing the estimator later must not silently pool two series.

        Without this label a future analysis would treat trailing-realized-vol
        rows and (say) GARCH rows as one sample and report a number for a model
        that never existed.
        """
        self._run()
        with connect(self.db) as conn:
            row = conn.execute("SELECT forecast_method, lookback_days FROM vol_forecasts").fetchone()
        self.assertEqual(row["forecast_method"], FORECAST_METHOD)
        self.assertEqual(row["lookback_days"], 20)

    def test_rerunning_the_same_day_updates_rather_than_duplicating(self):
        """The daily cycle can run twice; the sample must not double-count."""
        self._run()
        self._run(forecast=0.51)
        with connect(self.db) as conn:
            n_iv = conn.execute("SELECT COUNT(*) AS c FROM iv_snapshots").fetchone()["c"]
            n_fc = conn.execute("SELECT COUNT(*) AS c FROM vol_forecasts").fetchone()["c"]
            fc = conn.execute("SELECT forecast_vol FROM vol_forecasts").fetchone()["forecast_vol"]
        self.assertEqual(n_iv, 3)
        self.assertEqual(n_fc, 1)
        self.assertAlmostEqual(fc, 0.51)  # updated, not appended

    def test_contracts_without_iv_are_skipped_not_stored_as_null(self):
        """Half the INDICATIVE chain has no IV; those rows would only inflate counts."""
        chain = _chain(n=2) + [dict(_chain(n=1)[0], symbol="X", implied_volatility=None)]
        res = self._run(chain=chain)
        self.assertEqual(res.contracts_written, 2)

    def test_far_out_of_the_money_contracts_are_dropped(self):
        """Deep wings say nothing about ATM implied vol and carry the worst quotes."""
        chain = _chain(n=1) + [dict(_chain(n=1)[0], symbol="FAR", strike=100.0)]
        res = self._run(chain=chain, spot=10.5)
        self.assertEqual(res.contracts_written, 1)

    def test_a_chain_failure_still_records_the_forecast(self):
        """The two series are independent; a gap in one must not hole the other.

        If a chain outage also dropped the forecast, the eventual regression
        would be missing exactly the days the options feed was unhealthy, which
        is unlikely to be random with respect to volatility.
        """
        with patch("logic.options_engine.get_option_chain", side_effect=RuntimeError("feed down")), \
             patch("logic.options_engine._get_spot_price", return_value=10.5), \
             patch("logic.volatility_sizing.predict_volatility", return_value=0.48):
            res = snapshot_universe(symbols=["UNG"], db_path=self.db, snapshot_date="2026-10-19")
        self.assertEqual(res.contracts_written, 0)
        self.assertEqual(res.forecasts_written, 1)
        self.assertIn("UNG", res.failures)
        self.assertFalse(res.ok)

    def test_one_symbol_failing_does_not_lose_the_others(self):
        def chain_for(sym):
            if sym == "UNG":
                raise RuntimeError("feed down")
            return _chain(sym)

        with patch("logic.options_engine.get_option_chain", side_effect=chain_for), \
             patch("logic.options_engine._get_spot_price", return_value=10.5), \
             patch("logic.volatility_sizing.predict_volatility", return_value=0.48):
            res = snapshot_universe(symbols=["UNG", "CORN"], db_path=self.db,
                                    snapshot_date="2026-10-19")
        self.assertEqual(res.symbols_with_iv, 1)
        self.assertEqual(res.forecasts_written, 2)


class TestCoverage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "t.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_db_reports_not_ready_without_crashing(self):
        cov = coverage(self.db)
        self.assertEqual(cov["iv_days"], 0)
        self.assertFalse(cov["gate0_ready"])
        self.assertEqual(cov["days_remaining"], MIN_TRADING_DAYS_FOR_GATE0)

    def test_days_remaining_counts_down_by_distinct_date(self):
        """Two runs on one day are one day of history, not two."""
        with patch("logic.options_engine.get_option_chain", return_value=_chain()), \
             patch("logic.options_engine._get_spot_price", return_value=10.5), \
             patch("logic.volatility_sizing.predict_volatility", return_value=0.48):
            snapshot_universe(symbols=["UNG"], db_path=self.db, snapshot_date="2026-10-19")
            snapshot_universe(symbols=["UNG"], db_path=self.db, snapshot_date="2026-10-19")
            snapshot_universe(symbols=["UNG"], db_path=self.db, snapshot_date="2026-10-20")
        cov = coverage(self.db)
        self.assertEqual(cov["iv_days"], 2)
        self.assertEqual(cov["days_remaining"], MIN_TRADING_DAYS_FOR_GATE0 - 2)

    def test_init_is_idempotent(self):
        init_iv_tables(self.db)
        init_iv_tables(self.db)  # must not raise


if __name__ == "__main__":
    unittest.main()
