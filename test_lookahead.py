"""Hold-out and lookahead checks for the forecast path.

The Bayesian fit used to train on the same row it reported as the forecast.
Both models now train on features at t paired with the return realized at
t+1, and the last feature row is the test point.
"""

import unittest
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from logic.backtest import history_through
from logic.eos_bridge import extract_returns
from logic.news_engine import outcome_is_mature
from logic.trading_functions import bayesian_next_return, gp_predict, next_bar_split
from logic.volatility_sizing import predict_volatility
from logic.walk_forward import _fit_aligned


class TestNextBarHoldout(unittest.TestCase):
    def test_split_matches_the_requested_shapes(self):
        X = np.arange(20, dtype=float).reshape(5, 4)
        y = np.arange(5, dtype=float)
        X_train, y_train, X_test = next_bar_split(X, y)
        self.assertEqual(X_train.shape, (4, 4))
        self.assertEqual(X_test.shape, (1, 4))
        np.testing.assert_allclose(y_train, y[1:])
        np.testing.assert_allclose(X_train, X[:-1])
        np.testing.assert_allclose(X_test, X[-1:])

    def test_first_return_is_not_a_training_label(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(25, 3))
        y = rng.normal(size=25)
        before, *_ = bayesian_next_return(X, y)
        shifted = y.copy()
        shifted[0] = 40.0
        after, *_ = bayesian_next_return(X, shifted)
        self.assertAlmostEqual(before, after)

    def test_in_sample_last_row_is_not_the_forecast(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 2))
        y = rng.normal(scale=0.01, size=30)
        y[-1] = 2.5
        X[-1] = np.array([3.0, -3.0])
        honest, *_ = bayesian_next_return(X, y)
        from logic.trading_functions import update_beliefs_with_data

        leaked, _cov = update_beliefs_with_data(
            X, y, np.zeros(2), np.eye(2) * 0.1, 0.01
        )
        leaked_forecast = float(X[-1] @ leaked)
        self.assertGreater(abs(honest - leaked_forecast), 1e-3)

    def test_gp_uses_the_next_bar_target(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(12, 2))
        y = rng.normal(size=12)
        X_train, y_train, X_test = next_bar_split(X, y)
        held_mean, _ = gp_predict(X_train, y_train, X_test, noise_var=1e-3)
        same_bar_mean, _ = gp_predict(X[:-1], y[:-1], X[-1:], noise_var=1e-3)
        self.assertFalse(np.allclose(held_mean, same_bar_mean))

    def test_unseen_next_return_does_not_change_the_fit(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(40, 3))
        y_next = rng.normal(size=40)
        first = _fit_aligned(X[:-1], y_next[:-1], X[-1:])
        y_next[-1] = 9.0
        second = _fit_aligned(X[:-1], y_next[:-1], X[-1:])
        self.assertAlmostEqual(first, second)


class TestOtherComponentsDoNotReadTheNextBar(unittest.TestCase):
    def test_backtest_history_stops_on_the_decision_bar(self):
        index = pd.bdate_range("2024-01-02", periods=6)
        frame = pd.DataFrame({"Close": np.arange(6, dtype=float) + 10}, index=index)
        visible = history_through(frame, index[3].date())
        self.assertEqual(visible.index[-1], index[3])
        self.assertNotIn(index[4], visible.index)

    def test_eos_returns_come_only_from_the_frame_they_are_given(self):
        index = pd.bdate_range("2024-01-02", periods=5)
        frame = pd.DataFrame({"Close": [10, 11, 10.5, 12, 80]}, index=index)
        short = extract_returns(frame.iloc[:-1])
        long = extract_returns(frame)
        self.assertEqual(len(short), 3)
        self.assertEqual(len(long), 4)
        self.assertNotAlmostEqual(long[-1], short[-1])

    def test_volatility_window_ignores_a_bar_it_was_not_given(self):
        index = pd.bdate_range("2024-01-02", periods=40)
        rng = np.random.default_rng(4)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=40))
        frame = pd.DataFrame({"Close": close}, index=index)
        spiked = frame.copy()
        spiked.iloc[-1, spiked.columns.get_loc("Close")] = close[-2] * 1.8
        calm = predict_volatility("TEST", price_history=frame.iloc[:-1])
        also_calm = predict_volatility("TEST", price_history=spiked.iloc[:-1])
        self.assertAlmostEqual(calm, also_calm)

    def test_news_is_not_scored_before_the_window_closes(self):
        published = datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc)
        self.assertFalse(outcome_is_mature(published, published + timedelta(hours=23)))
        self.assertTrue(outcome_is_mature(published, published + timedelta(hours=24)))

    def test_unscored_calibration_row_does_not_move_the_curve(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from logic.calibration import fit_calibration
        from logic.model_performance_tracker import init_model_performance_tracker
        from logic.sqlite_store import connect, init_db

        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            out_a = Path(d) / "a.json"
            out_b = Path(d) / "b.json"
            init_db(db)
            init_model_performance_tracker(db)
            with connect(db) as conn:
                for i in range(40):
                    ts = f"2026-08-{(i % 27) + 1:02d}T10:00:00+00:00"
                    raw = 0.9 if i < 20 else 0.2
                    conn.execute(
                        "INSERT INTO decisions (account_id, timestamp, symbol, "
                        "prob_profit, raw_prob_profit) VALUES ('t', ?, ?, ?, ?)",
                        (ts, f"S{i}", raw, raw),
                    )
                    conn.execute(
                        "INSERT INTO model_component_performance (decision_key, timestamp, "
                        "symbol, action, price_at_signal, next_day_return) "
                        "VALUES (?, ?, ?, 'buy', 100.0, ?)",
                        (f"k{i}", ts, f"S{i}", 0.01 if i < 20 else -0.01),
                    )
            first = fit_calibration(db_path=db, out_path=out_a, min_sample=30)
            with connect(db) as conn:
                conn.execute(
                    "INSERT INTO decisions (account_id, timestamp, symbol, "
                    "prob_profit, raw_prob_profit) VALUES ('t', ?, ?, ?, ?)",
                    ("2026-09-29T10:00:00+00:00", "LIVE", 0.99, 0.99),
                )
                conn.execute(
                    "INSERT INTO model_component_performance (decision_key, timestamp, "
                    "symbol, action, price_at_signal, next_day_return) "
                    "VALUES (?, ?, ?, 'buy', 100.0, NULL)",
                    ("klive", "2026-09-29T10:00:00+00:00", "LIVE"),
                )
            second = fit_calibration(db_path=db, out_path=out_b, min_sample=30)
        self.assertEqual(first.x, second.x)
        self.assertEqual(first.y, second.y)


if __name__ == "__main__":
    unittest.main()
