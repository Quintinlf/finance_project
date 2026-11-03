"""Tests for the purchased-option-history ingest and the IV inversion it needs.

The ingest writes into a sample that Gate 0 will be decided on. So the tests
worth having are the ones that catch a *plausible-looking* corruption rather
than a crash: a spot taken from the wrong day, a forecast that peeked at the
future, a backfilled series pooling with the live one, or a contract id that
fails to collide with itself and so double-counts on a re-run.

The round-trip inversion tests are also load-bearing in a second sense: the
cheap ThetaData tiers have no IV endpoint, so if the inversion is wrong the
whole cost argument for buying history collapses.
"""

import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from logic.black_scholes_iv import (
    bs_price,
    implied_vol_from_quote,
    implied_volatility,
    risk_free_rate,
    year_fraction,
)
from logic.iv_snapshot import FORECAST_METHOD
from logic.sqlite_store import connect
from logic.thetadata_client import _parse_rows, _to_compact_date, _to_iso_date
from logic.thetadata_ingest import (
    FEED,
    FORECAST_METHOD_BACKFILL,
    _chunks,
    _close_on,
    _contract_symbol,
    _history_as_of,
    _rows_from_eod,
    ingest_symbol,
)


def _frame(start="2024-01-01", n=60, price=100.0, step=0.5, wobble=0.0):
    """A minimal daily-close frame shaped like the price cache returns.

    ``wobble`` oscillates the closes around ``price`` instead of trending. A
    flat series is not usable as a fixture: zero variance means zero realized
    vol, and ``predict_volatility`` correctly declines to forecast from it.
    """
    import pandas as pd

    idx = pd.bdate_range(start=start, periods=n)
    if wobble:
        closes = [price + (wobble if i % 2 else -wobble) for i in range(n)]
    else:
        closes = [price + step * i for i in range(n)]
    return pd.DataFrame({"Close": closes}, index=idx)


def _eod_row(**over):
    row = {
        "symbol": "GLD",
        "expiration": "2024-04-19",
        "strike": 100.0,
        "right": "call",
        "created": "2024-02-01T17:15:00.000",
        "date": "20240201",
        "open": 3.0, "high": 3.4, "low": 2.9, "close": 3.2,
        "volume": 120, "count": 11,
        "bid": 3.10, "ask": 3.30,
    }
    row.update(over)
    return row


class TestBlackScholesInversion(unittest.TestCase):
    def test_inverting_a_priced_option_recovers_its_volatility(self):
        """The core guarantee: price(sigma) then invert must return sigma."""
        for vol in (0.10, 0.22, 0.45, 0.90):
            price = bs_price(100.0, 100.0, 0.25, vol, 0.04, option_type="call")
            back = implied_volatility(price, 100.0, 100.0, 0.25,
                                      option_type="call", rate=0.04)
            self.assertIsNotNone(back)
            self.assertAlmostEqual(back, vol, places=4)

    def test_puts_invert_too(self):
        price = bs_price(100.0, 105.0, 0.5, 0.33, 0.04, option_type="put")
        back = implied_volatility(price, 100.0, 105.0, 0.5,
                                  option_type="put", rate=0.04)
        self.assertAlmostEqual(back, 0.33, places=4)

    def test_price_below_intrinsic_returns_none(self):
        """A stale or crossed EOD quote must not invert to a plausible number.

        Deep-in-the-money contracts on thin chains routinely quote below
        intrinsic at the close. Returning a float there would put an
        arbitrage-implied volatility into the Gate 0 sample.
        """
        self.assertIsNone(
            implied_volatility(0.50, 100.0, 50.0, 0.25, option_type="call", rate=0.04)
        )

    def test_absurd_price_returns_none_rather_than_the_bracket_ceiling(self):
        """A price above the 500%-vol ceiling is a broken quote, not a volatile ETF."""
        self.assertIsNone(
            implied_volatility(99.0, 100.0, 100.0, 0.02, option_type="call", rate=0.04)
        )

    def test_expired_and_nonsense_inputs_return_none(self):
        self.assertIsNone(implied_volatility(1.0, 100.0, 100.0, 0.0))
        self.assertIsNone(implied_volatility(0.0, 100.0, 100.0, 0.25))
        self.assertIsNone(implied_volatility(1.0, 0.0, 100.0, 0.25))

    def test_year_fraction_is_calendar_days_not_trading_days(self):
        """Calendar-day convention: the market quotes IV that way.

        Using 252 would make every inverted IV disagree with published figures
        by ~20% in a test whose whole point is comparing our number to theirs.
        """
        self.assertAlmostEqual(year_fraction("2024-01-01", "2024-01-31"), 30 / 365.0)
        self.assertIsNone(year_fraction("2024-01-31", "2024-01-01"))

    def test_rate_varies_by_year(self):
        """A constant rate across 2016-2026 would be a real bias, not rounding."""
        self.assertLess(risk_free_rate("2021-06-01"), 0.01)
        self.assertGreater(risk_free_rate("2023-06-01"), 0.04)

    def test_quote_inversion_prefers_the_mid_over_the_close(self):
        """The last trade can be hours stale; the closing NBBO cannot."""
        mid_iv = implied_vol_from_quote(
            symbol="GLD", snapshot_date="2024-02-01", expiration="2024-05-01",
            strike=100.0, option_type="call", spot=100.0,
            bid=3.0, ask=3.2, close=9.9,
        )
        close_iv = implied_vol_from_quote(
            symbol="GLD", snapshot_date="2024-02-01", expiration="2024-05-01",
            strike=100.0, option_type="call", spot=100.0,
            bid=None, ask=None, close=9.9,
        )
        self.assertIsNotNone(mid_iv)
        self.assertIsNotNone(close_iv)
        self.assertLess(mid_iv, close_iv)

    def test_crossed_quote_falls_back_to_the_close(self):
        iv = implied_vol_from_quote(
            symbol="GLD", snapshot_date="2024-02-01", expiration="2024-05-01",
            strike=100.0, option_type="call", spot=100.0,
            bid=3.3, ask=3.1, close=3.2,
        )
        self.assertIsNotNone(iv)


class TestPointInTimeSlicing(unittest.TestCase):
    """The safeguards that make the backfilled forecast not lookahead-biased."""

    def setUp(self):
        self.hist = _frame(start="2024-01-01", n=60)

    def test_history_as_of_excludes_every_later_bar(self):
        sliced = _history_as_of(self.hist, "2024-02-01")
        self.assertGreater(len(sliced), 0)
        self.assertLessEqual(str(sliced.index[-1])[:10], "2024-02-01")

    def test_a_forecast_cannot_see_a_later_price_spike(self):
        """The decisive test: a future spike must not change a past forecast.

        If it does, the estimator is reading data it would not have had, and
        every Gate 0 number built on the backfill is worthless.
        """
        from logic.volatility_sizing import predict_volatility

        early = predict_volatility("GLD", price_history=_history_as_of(self.hist, "2024-02-01"))

        spiked = self.hist.copy()
        spiked.iloc[-5:, 0] = 10_000.0  # violent move, well after the cutoff
        after = predict_volatility("GLD", price_history=_history_as_of(spiked, "2024-02-01"))

        self.assertIsNotNone(early)
        self.assertAlmostEqual(early, after, places=10)

    def test_close_on_requires_an_exact_date_match(self):
        """Spot must come from the option quote's own day.

        Falling back to the previous close would silently pair a Monday quote
        with a Friday spot, corrupting moneyness and every IV built on it.
        """
        self.assertIsNotNone(_close_on(self.hist, "2024-01-01"))
        # A Sunday: no bar exists, so there is no spot, not a stale one.
        self.assertIsNone(_close_on(self.hist, "2024-01-07"))


class TestRowConstruction(unittest.TestCase):
    def setUp(self):
        self.hist = _frame(start="2024-01-01", n=60, price=100.0, wobble=0.4)

    def _rows(self, raw):
        return _rows_from_eod(raw, symbol="GLD", history=self.hist,
                              captured_at="2024-02-01T00:00:00+00:00",
                              max_moneyness=0.25)

    def test_builds_a_row_with_an_inverted_iv_and_the_right_feed(self):
        rows, failures, missing = self._rows([_eod_row()])
        self.assertEqual(len(rows), 1)
        self.assertEqual(failures, 0)
        self.assertEqual(missing, 0)
        r = rows[0]
        self.assertEqual(r["feed"], FEED)
        self.assertEqual(r["snapshot_date"], "2024-02-01")
        self.assertIsNotNone(r["implied_volatility"])
        # Spot is the close on the snapshot date itself, read from the frame.
        self.assertAlmostEqual(r["spot"], _close_on(self.hist, "2024-02-01"))

    def test_greeks_are_left_null_rather_than_computed(self):
        """The EOD feed carries no greeks.

        Filling them from our own Black-Scholes would let a later analysis treat
        an artefact of our assumptions as an observed quantity.
        """
        rows, _, _ = self._rows([_eod_row()])
        for greek in ("delta", "gamma", "theta", "vega"):
            self.assertIsNone(rows[0][greek])

    def test_far_out_of_the_money_rows_are_dropped(self):
        rows, _, _ = self._rows([_eod_row(strike=400.0)])
        self.assertEqual(rows, [])

    def test_expired_contracts_are_dropped(self):
        rows, _, _ = self._rows([_eod_row(expiration="2024-01-15")])
        self.assertEqual(rows, [])

    def test_uninvertible_quotes_are_counted_not_stored(self):
        """A quote that will not invert is a failure statistic, not a NULL row."""
        rows, failures, _ = self._rows([_eod_row(bid=0.0, ask=0.0, close=0.0)])
        self.assertEqual(rows, [])
        self.assertEqual(failures, 1)

    def test_a_date_with_no_price_bar_is_reported(self):
        rows, _, missing = self._rows([_eod_row(date="20240107")])  # a Sunday
        self.assertEqual(rows, [])
        self.assertEqual(missing, 1)

    def test_puts_and_calls_at_one_strike_get_distinct_contract_ids(self):
        """Otherwise the UNIQUE key collapses a put and a call into one row."""
        rows, _, _ = self._rows([_eod_row(right="call"), _eod_row(right="put")])
        ids = {r["contract_symbol"] for r in rows}
        self.assertEqual(len(ids), len(rows))
        self.assertEqual(len(rows), 2)

    def test_contract_symbol_is_deterministic(self):
        """A re-run must collide and upsert rather than duplicate."""
        a = _contract_symbol("GLD", "2024-04-19", "call", 100.0)
        b = _contract_symbol("GLD", "2024-04-19", "call", 100.0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, _contract_symbol("GLD", "2024-04-19", "put", 100.0))
        self.assertNotEqual(a, _contract_symbol("GLD", "2024-04-19", "call", 100.5))


class TestIngestWrites(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "t.db"
        from logic.iv_snapshot import init_iv_tables
        from logic.thetadata_ingest import init_progress_table

        init_iv_tables(self.db)
        init_progress_table(self.db)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *, raw=None, dry_run=False, start="2024-02-01", end="2024-02-05"):
        class FakeClient:
            def __init__(self, rows):
                self.rows = rows
                self.calls = []

            def option_history_eod(self, symbol, **kw):
                self.calls.append((symbol, kw))
                return self.rows

        client = FakeClient([_eod_row()] if raw is None else raw)
        with patch("logic.thetadata_ingest._load_full_history",
                   return_value=_frame(start="2024-01-01", n=60, wobble=0.4)):
            stats = ingest_symbol(
                client, "GLD",
                start=date.fromisoformat(start), end=date.fromisoformat(end),
                db_path=self.db, dry_run=dry_run,
            )
        return stats, client

    def test_writes_contracts_and_a_backfilled_forecast(self):
        stats, _ = self._run()
        self.assertEqual(stats.contracts_written, 1)
        self.assertEqual(stats.forecasts_written, 1)
        with connect(self.db) as conn:
            iv = conn.execute("SELECT * FROM iv_snapshots").fetchone()
            fc = conn.execute("SELECT * FROM vol_forecasts").fetchone()
        self.assertEqual(iv["feed"], FEED)
        self.assertIsNotNone(fc["forecast_vol"])

    def test_the_backfilled_forecast_is_labelled_distinctly_from_the_live_one(self):
        """The two series must never pool silently.

        They are produced by the same estimator but one is captured live and one
        is reconstructed. If they ever disagree on an overlapping date that is a
        bug, and these labels are what make it findable instead of invisible.
        """
        self._run()
        with connect(self.db) as conn:
            methods = {r["forecast_method"] for r in
                       conn.execute("SELECT forecast_method FROM vol_forecasts")}
        self.assertEqual(methods, {FORECAST_METHOD_BACKFILL})
        self.assertNotEqual(FORECAST_METHOD_BACKFILL, FORECAST_METHOD)

    def test_rerunning_skips_completed_chunks(self):
        _, first = self._run()
        requests_first = len(first.calls)
        stats, second = self._run()
        self.assertGreater(requests_first, 0)
        self.assertEqual(len(second.calls), 0, "a completed chunk was re-fetched")
        self.assertGreater(stats.chunks_skipped, 0)

    def test_rerunning_does_not_duplicate_rows(self):
        self._run()
        # Clear progress so the chunk is genuinely re-pulled, proving the upsert
        # rather than merely the skip.
        with connect(self.db) as conn:
            conn.execute("DELETE FROM ingest_progress")
        self._run()
        with connect(self.db) as conn:
            n = conn.execute("SELECT COUNT(*) AS c FROM iv_snapshots").fetchone()["c"]
        self.assertEqual(n, 1)

    def test_dry_run_writes_nothing_and_ignores_saved_progress(self):
        self._run()
        stats, client = self._run(dry_run=True)
        self.assertGreater(len(client.calls), 0, "dry run should still fetch")
        self.assertEqual(stats.chunks_skipped, 0)
        with connect(self.db) as conn:
            n = conn.execute("SELECT COUNT(*) AS c FROM iv_snapshots").fetchone()["c"]
        self.assertEqual(n, 1)  # unchanged by the dry run

    def test_server_side_filters_are_actually_sent(self):
        """These filters are the reason one month's subscription suffices."""
        _, client = self._run()
        _, kw = client.calls[0]
        self.assertIn("strike_range", kw)
        self.assertIn("max_dte", kw)

    def test_a_symbol_without_price_history_fails_loudly(self):
        class FakeClient:
            def option_history_eod(self, symbol, **kw):
                raise AssertionError("should not be reached without price history")

        with patch("logic.thetadata_ingest._load_full_history", return_value=None):
            stats = ingest_symbol(
                FakeClient(), "GLD",
                start=date(2024, 2, 1), end=date(2024, 2, 5), db_path=self.db,
            )
        self.assertIn("GLD", stats.failures)
        self.assertEqual(stats.contracts_written, 0)


class TestChunking(unittest.TestCase):
    def test_chunks_cover_the_window_without_gaps_or_overlap(self):
        chunks = _chunks(date(2024, 1, 1), date(2024, 3, 31), 31)
        self.assertEqual(chunks[0][0], date(2024, 1, 1))
        self.assertEqual(chunks[-1][1], date(2024, 3, 31))
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
            self.assertEqual((next_start - prev_end).days, 1)

    def test_a_single_day_window_is_one_chunk(self):
        self.assertEqual(_chunks(date(2024, 1, 1), date(2024, 1, 1), 31),
                         [(date(2024, 1, 1), date(2024, 1, 1))])


class TestGateZeroDoesNotPoolTwoSeries(unittest.TestCase):
    """The hazard the ingest introduces into an existing analysis.

    Before the ingest there was exactly one feed and one forecast method in the
    database, so joining without naming them was harmless. The moment a second
    of either exists, the unfiltered query does something subtler than
    duplicating rows -- the ``GROUP BY`` collapses the fan-out, so the row count
    looks untouched, but:

    * ``AVG(s.implied_volatility)`` silently averages a mid-session INDICATIVE
      IV with a 17:15 EOD IV we inverted ourselves, and
    * the bare ``f.forecast_vol`` under that ``GROUP BY`` resolves to whichever
      forecast method SQLite happens to pick from the group.

    An inflated row count would at least be visible. This is a contaminated
    number at an unchanged N, which is precisely the kind of finding this
    project has already had to retract twice.
    """

    IV_LIVE, IV_PURCHASED = 0.25, 0.55

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "t.db"
        from logic.iv_snapshot import init_iv_tables

        init_iv_tables(self.db)
        with connect(self.db) as conn:
            for feed, iv in (("indicative", self.IV_LIVE), (FEED, self.IV_PURCHASED)):
                conn.execute(
                    """
                    INSERT INTO iv_snapshots
                        (snapshot_date, captured_at_utc, underlying, spot,
                         contract_symbol, expiration, dte, strike, option_type,
                         moneyness, implied_volatility, feed)
                    VALUES ('2024-02-01', 'x', 'GLD', 100.0, ?, '2024-03-01',
                            29, 100.0, 'call', 0.0, ?, ?)
                    """,
                    (f"GLD-{feed}", iv, feed),
                )
            for method in (FORECAST_METHOD, FORECAST_METHOD_BACKFILL):
                conn.execute(
                    """
                    INSERT INTO vol_forecasts
                        (snapshot_date, captured_at_utc, symbol, forecast_vol,
                         forecast_method, lookback_days, spot)
                    VALUES ('2024-02-01', 'x', 'GLD', 0.2, ?, 20, 100.0)
                    """,
                    (method,),
                )

    def tearDown(self):
        self._tmp.cleanup()

    def test_both_measurement_processes_are_reported_separately(self):
        from scripts.analyze_vol_edge import available_series

        series = available_series(self.db)
        combos = {(f, m) for f, m, _ in series}
        self.assertIn((FEED, FORECAST_METHOD_BACKFILL), combos)
        self.assertIn(("indicative", FORECAST_METHOD), combos)
        self.assertEqual(len(combos), 4, "every feed x method pair should be visible")

    def _load(self, **kw):
        from scripts.analyze_vol_edge import load_pairs

        with patch("logic.price_cache.get_history",
                   side_effect=lambda s, **k: _frame(start="2024-01-01", n=60, wobble=0.4)):
            return load_pairs(30, db_path=self.db, **kw)

    def test_filtering_by_feed_returns_that_feeds_implied_vol_exactly(self):
        purchased = self._load(feed=FEED, forecast_method=FORECAST_METHOD_BACKFILL)
        live = self._load(feed="indicative", forecast_method=FORECAST_METHOD)

        self.assertEqual(len(purchased), 1)
        self.assertEqual(len(live), 1)
        self.assertAlmostEqual(purchased[0][0], self.IV_PURCHASED)
        self.assertAlmostEqual(live[0][0], self.IV_LIVE)

    def test_the_unfiltered_query_blends_the_two_feeds(self):
        """Documents the defect the filters exist to prevent.

        The unfiltered call returns the same single observation -- the row count
        gives nothing away -- but its implied vol is the mean of two different
        measurement processes rather than either one of them.
        """
        pooled = self._load()
        self.assertEqual(len(pooled), 1, "the row count hides the contamination")
        blended = (self.IV_LIVE + self.IV_PURCHASED) / 2.0
        self.assertAlmostEqual(pooled[0][0], blended)
        self.assertNotAlmostEqual(pooled[0][0], self.IV_LIVE)
        self.assertNotAlmostEqual(pooled[0][0], self.IV_PURCHASED)


class TestClientParsing(unittest.TestCase):
    def test_parses_a_bare_v3_array(self):
        self.assertEqual(_parse_rows('[{"symbol":"GLD"}]'), [{"symbol": "GLD"}])

    def test_parses_the_legacy_header_response_envelope(self):
        """A terminal downgrade must not silently yield zero rows."""
        body = '{"header":{"format":["symbol","close"]},"response":[["GLD",3.2]]}'
        self.assertEqual(_parse_rows(body), [{"symbol": "GLD", "close": 3.2}])

    def test_empty_body_is_empty_not_an_error(self):
        self.assertEqual(_parse_rows("   "), [])

    def test_date_formats_round_trip(self):
        self.assertEqual(_to_iso_date(20240131), "2024-01-31")
        self.assertEqual(_to_iso_date("20240131"), "2024-01-31")
        self.assertEqual(_to_iso_date("2024-01-31"), "2024-01-31")
        self.assertIsNone(_to_iso_date("not a date"))
        self.assertEqual(_to_compact_date("2024-01-31"), "20240131")
        self.assertEqual(_to_compact_date(date(2024, 1, 31)), "20240131")


if __name__ == "__main__":
    unittest.main()
