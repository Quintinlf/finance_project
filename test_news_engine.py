"""Tests for the news diffusion engine.

The concrete case: a headline explicitly about wheat/Ukraine should reach
CORN, SOYB, and DBA too, through the shared "geopolitical_supply_shock" theme
already tagged in logic/universe.py -- even though those symbols are never
named. That propagation, not the keyword matching itself, is the point.
"""

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from logic.news_engine import (
    NewsEvent,
    apply_news_context,
    build_news_events,
    classify_headline,
    init_news_engine,
    persist_news_events,
    score_news_events,
    summarize_news_accuracy,
)
from logic.sqlite_store import connect


def _news_item(headline, symbols, summary="", source="Test Wire", hours_ago=1):
    return SimpleNamespace(
        headline=headline, summary=summary, source=source, symbols=symbols,
        created_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
    )


class TestClassifyHeadline(unittest.TestCase):
    def test_real_headline_matches_the_intended_theme(self):
        """Confirmed live against Alpaca's News API on 2026-08-25."""
        c = classify_headline("Russia Eyes Grain Duty Pause Through Year End on Ukraine Strikes")
        self.assertIn("geopolitical_supply_shock", c["themes"])

    def test_bullish_supply_disruption_language(self):
        c = classify_headline("Wheat shortage as export ban disrupts global grain supply")
        self.assertEqual(c["polarity"], "bullish")
        self.assertGreater(c["polarity_score"], 0)

    def test_bearish_oversupply_language(self):
        c = classify_headline("Bumper corn harvest leads to record glut, prices ease")
        self.assertEqual(c["polarity"], "bearish")
        self.assertLess(c["polarity_score"], 0)

    def test_no_keyword_hits_is_neutral(self):
        c = classify_headline("Quarterly earnings beat analyst expectations")
        self.assertEqual(c["polarity"], "neutral")
        self.assertEqual(c["polarity_score"], 0.0)
        self.assertEqual(c["themes"], [])

    def test_drought_matches_weather_theme(self):
        c = classify_headline("Severe drought threatens Midwest corn and soybean crops")
        self.assertIn("weather_drought", c["themes"])

    def test_opec_production_cut_matches_opec_theme(self):
        c = classify_headline("OPEC+ agrees to another production cut")
        self.assertIn("opec", c["themes"])


class TestDiffusion(unittest.TestCase):
    def test_explicit_tag_produces_an_explicit_event(self):
        item = _news_item(
            "Russia Eyes Grain Duty Pause Through Year End on Ukraine Strikes", ["WEAT"]
        )
        with patch("logic.news_engine._fetch_raw_news", return_value=[item]):
            events = build_news_events(["WEAT", "CORN", "SOYB", "DBA"])
        weat = [e for e in events if e.symbol == "WEAT"]
        self.assertEqual(len(weat), 1)
        self.assertEqual(weat[0].match_type, "explicit")

    def test_theme_diffuses_to_an_untagged_correlated_symbol(self):
        """The actual point of this module: CORN reached without ever being
        named, purely because it shares "geopolitical_supply_shock" with the
        explicitly-tagged WEAT."""
        item = _news_item(
            "Russia Eyes Grain Duty Pause Through Year End on Ukraine Strikes", ["WEAT"]
        )
        with patch("logic.news_engine._fetch_raw_news", return_value=[item]):
            events = build_news_events(["WEAT", "CORN", "SOYB", "DBA", "AAPL"])

        by_symbol = {e.symbol: e for e in events}
        self.assertIn("CORN", by_symbol)
        self.assertEqual(by_symbol["CORN"].match_type, "theme")
        # SOYB/DBA share "food_security" with WEAT, but this headline never
        # matches that theme (no "grain export"/"food security" phrasing) --
        # diffusion follows the theme actually detected in the text, not
        # every theme the tagged symbol happens to carry.
        self.assertNotIn("SOYB", by_symbol)
        self.assertNotIn("DBA", by_symbol)
        # AAPL carries no matching theme and was never explicitly tagged.
        self.assertNotIn("AAPL", by_symbol)

    def test_a_food_security_headline_reaches_soyb_and_dba_too(self):
        """Different headline, different matched theme, wider reach: this one
        actually says "food security" / "grain export", so it also diffuses to
        SOYB and DBA through the shared food_security tag."""
        item = _news_item(
            "War disrupts grain export routes, raising food security fears worldwide", ["WEAT"]
        )
        with patch("logic.news_engine._fetch_raw_news", return_value=[item]):
            events = build_news_events(["WEAT", "CORN", "SOYB", "DBA"])
        symbols = {e.symbol for e in events}
        self.assertEqual(symbols, {"WEAT", "CORN", "SOYB", "DBA"})

    def test_diffusion_only_reaches_symbols_in_the_active_universe(self):
        """A theme match on a symbol outside today's universe must not appear."""
        item = _news_item("Drought threatens wheat and corn harvest", ["WEAT"])
        with patch("logic.news_engine._fetch_raw_news", return_value=[item]):
            events = build_news_events(["WEAT"])  # CORN excluded on purpose
        self.assertNotIn("CORN", {e.symbol for e in events})

    def test_no_theme_match_produces_no_diffusion(self):
        item = _news_item("Quarterly earnings beat expectations", ["AAPL"])
        with patch("logic.news_engine._fetch_raw_news", return_value=[item]):
            events = build_news_events(["AAPL", "MSFT"])
        self.assertEqual([e.symbol for e in events], ["AAPL"])

    def test_fetch_failure_yields_no_events_not_an_exception(self):
        with patch("logic.news_engine._fetch_raw_news", return_value=[]):
            events = build_news_events(["WEAT"])
        self.assertEqual(events, [])


class TestApplyNewsContext(unittest.TestCase):
    def test_matching_signal_gets_shadow_annotated(self):
        sig = SimpleNamespace(symbol="CORN", signal_type="buy", confidence=0.8, prob_profit=0.7, meta={})
        event = NewsEvent(
            headline="Russia Eyes Grain Duty Pause", source="Test",
            published_at=datetime.now(timezone.utc), symbol="CORN",
            match_type="theme", matched_themes=["geopolitical_supply_shock"],
            polarity="bullish", polarity_score=0.5,
        )
        apply_news_context([sig], [event])
        self.assertIn("news_events", sig.meta)
        self.assertEqual(sig.meta["news_themes"], ["geopolitical_supply_shock"])
        self.assertAlmostEqual(sig.meta["news_polarity_score"], 0.5)

    def test_never_changes_the_trade_itself(self):
        """Shadow mode is the whole point: nothing here may alter a live decision."""
        sig = SimpleNamespace(symbol="CORN", signal_type="hold", confidence=0.5, prob_profit=0.5, meta={})
        event = NewsEvent(
            headline="x", source="Test", published_at=datetime.now(timezone.utc),
            symbol="CORN", match_type="explicit", matched_themes=["opec"],
            polarity="bearish", polarity_score=-0.9,
        )
        apply_news_context([sig], [event])
        self.assertEqual(sig.signal_type, "hold")
        self.assertEqual(sig.confidence, 0.5)
        self.assertEqual(sig.prob_profit, 0.5)

    def test_unmatched_signal_is_untouched_but_safe(self):
        sig = SimpleNamespace(symbol="NVDA", signal_type="hold", meta={})
        apply_news_context([sig], [])
        self.assertNotIn("news_events", sig.meta)


class TestPersistenceAndScoring(unittest.TestCase):
    def test_round_trip_and_idempotent_insert(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            event = NewsEvent(
                headline="Test headline", source="Wire",
                published_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
                symbol="WEAT", match_type="explicit",
                matched_themes=["geopolitical_supply_shock"], polarity="bullish",
                polarity_score=0.6,
            )
            written_1 = persist_news_events([event], db_path=db)
            written_2 = persist_news_events([event], db_path=db)  # duplicate
            self.assertEqual(written_1, 1)
            self.assertEqual(written_2, 0)  # INSERT OR IGNORE on event_key

            with connect(db) as conn:
                row = conn.execute("SELECT symbol, polarity FROM news_events").fetchone()
            self.assertEqual(row["symbol"], "WEAT")
            self.assertEqual(row["polarity"], "bullish")

    def test_score_news_events_grades_matured_bullish_call_correctly(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            init_news_engine(db)
            old_ts = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
            with connect(db) as conn:
                conn.execute(
                    "INSERT INTO news_events (event_key, published_at, headline, source, "
                    "symbol, match_type, matched_themes, polarity, polarity_score) "
                    "VALUES ('k1', ?, 'h', 's', 'WEAT', 'explicit', 'grains', 'bullish', 0.5)",
                    (old_ts,),
                )
            import pandas as pd
            hist = pd.DataFrame(
                {"Close": [24.0, 24.0, 24.8]},
                index=pd.DatetimeIndex(
                    [datetime.now(timezone.utc) - timedelta(days=d_) for d_ in (4, 3, 2)]
                ),
            )
            with patch("yfinance.Ticker") as tk:
                tk.return_value.history.return_value = hist
                n = score_news_events(db_path=db)
            self.assertEqual(n, 1)
            with connect(db) as conn:
                row = conn.execute(
                    "SELECT next_day_return, correct FROM news_events WHERE event_key='k1'"
                ).fetchone()
            self.assertGreater(row["next_day_return"], 0)
            self.assertEqual(row["correct"], 1)  # bullish call, price went up

    def test_too_recent_event_is_left_unscored(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            init_news_engine(db)
            with connect(db) as conn:
                conn.execute(
                    "INSERT INTO news_events (event_key, published_at, headline, source, "
                    "symbol, match_type, matched_themes, polarity, polarity_score) "
                    "VALUES ('k2', ?, 'h', 's', 'WEAT', 'explicit', '', 'bullish', 0.5)",
                    (datetime.now(timezone.utc).isoformat(),),
                )
            n = score_news_events(db_path=db)
            self.assertEqual(n, 0)

    def test_summarize_news_accuracy_reports_base_rate_and_splits_by_match_type(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            init_news_engine(db)
            with connect(db) as conn:
                for i in range(12):
                    conn.execute(
                        "INSERT INTO news_events (event_key, published_at, headline, source, "
                        "symbol, match_type, matched_themes, polarity, polarity_score, "
                        "next_day_return, correct) VALUES (?, ?, 'h', 's', 'WEAT', ?, '', "
                        "'bullish', 0.5, 0.01, ?)",
                        (f"k{i}", datetime.now(timezone.utc).isoformat(),
                         "explicit" if i < 8 else "theme", 1 if i % 3 else 0),
                    )
            report = summarize_news_accuracy(db_path=db, min_sample=1)
            self.assertIn("base rate", report)
            self.assertIn("explicit", report)
            self.assertIn("theme", report)

    def test_no_scored_rows_returns_empty_string(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            init_news_engine(db)
            self.assertEqual(summarize_news_accuracy(db_path=db), "")


if __name__ == "__main__":
    unittest.main()
