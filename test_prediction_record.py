"""PredictionRecord is written before the next close exists, then filled later."""

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd

from logic.prediction_log import (
    fill_realized_returns,
    load_records,
    record_from_signal,
    records_from_cycle,
    write_records,
)


class TestPredictionRecord(unittest.TestCase):
    def test_decision_time_record_has_no_outcome(self):
        signal = SimpleNamespace(
            symbol="WEAT",
            signal_type="hold",
            prob_profit=0.52,
            meta={
                "ensemble_forecast_return": 0.004,
                "edge_cost_pct": 0.30,
                "edge_expected_value_pct": -0.1,
                "news_polarity_score": 0.2,
                "news_themes": ["grains"],
                "market_regime": {"trend": 0.4},
                "full_forecast": {
                    "information_cutoff": "2026-09-29T00:00:00",
                    "features_version": "v",
                    "model_version": "m",
                    "bayesian": {"forecast": 0.003},
                    "gp": {"forecast": 0.005},
                    "feature_snapshot": {
                        "bb_features": {"BB_Z_Score": -1.2},
                        "rsi_features": {"RSI": 42.0},
                        "candle_features": {"candle_body": 0.2},
                        "volume_features": {"volume_ratio": 1.1},
                        "lunar_features": {"lunar_sin": 0.3},
                    },
                },
            },
        )
        entry = SimpleNamespace(symbol="WEAT", action="rejected", planned_quantity=0)
        record = records_from_cycle([signal], [entry])[0]
        self.assertEqual(record.action, "rejected")
        self.assertIsNone(record.realized_return)
        self.assertEqual(record.bayesian_prediction, 0.003)
        self.assertEqual(record.news_features["themes"], ["grains"])
        self.assertEqual(record.information_cutoff, "2026-09-29T00:00:00")

    def test_fill_writes_the_outcome_and_leaves_the_forecast(self):
        index = pd.bdate_range("2026-09-28", periods=3)
        frame = pd.DataFrame({"Close": [10.0, 11.0, 12.0]}, index=index)
        record = record_from_signal(
            SimpleNamespace(
                symbol="WEAT",
                signal_type="hold",
                prob_profit=0.5,
                meta={
                    "ensemble_forecast_return": 0.01,
                    "edge_cost_pct": 0.2,
                    "full_forecast": {
                        "information_cutoff": "2026-09-28T00:00:00",
                        "features_version": "v",
                        "model_version": "m",
                        "bayesian": {"forecast": 0.01},
                        "gp": {"forecast": 0.01},
                        "feature_snapshot": {},
                    },
                },
            ),
            action="hold",
            quantity=0,
            timestamp=datetime(2026, 9, 28, tzinfo=timezone.utc).isoformat(),
        )
        with TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            write_records([record.to_dict()], path)
            filled = fill_realized_returns(path, history_lookup=lambda symbol: frame)
            rows = load_records(path)
        self.assertEqual(filled, 1)
        self.assertAlmostEqual(rows[0]["realized_return"], 0.1)
        self.assertAlmostEqual(rows[0]["bayesian_prediction"], 0.01)
        self.assertAlmostEqual(rows[0]["realized_net"], 0.1 - 0.002)


if __name__ == "__main__":
    unittest.main()
