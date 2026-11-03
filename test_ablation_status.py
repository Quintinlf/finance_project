"""The tournament's three statuses, and the three comparisons it reports."""

import unittest

import numpy as np
import pandas as pd

from logic.feature_policy import live_blocks, use_gp
from logic.walk_forward import (
    classify_incremental,
    run_ablation,
    summarize_arm,
)


class TestLivePolicy(unittest.TestCase):
    def test_failed_blocks_leave_and_only_a_survivor_is_added(self):
        report = {
            "arms": {
                "baseline_ta": {"status": "fails"},
                "baseline_candles": {"status": "survives"},
                "baseline_volume": {"status": "fails"},
                "baseline_news": {"status": "not_tested"},
                "baseline_regime": {"status": "promising"},
                "baseline_lunar": {"status": "promising"},
            },
            "h6_gp_vs_bayesian": {"status": "fails"},
        }
        self.assertEqual(live_blocks(report), ["lags", "volatility", "candles"])
        self.assertFalse(use_gp(report))

    def test_promising_does_not_turn_a_removed_model_back_on(self):
        report = {"h6_gp_vs_bayesian": {"status": "promising"}, "arms": {}}
        self.assertFalse(use_gp(report))


class TestStatusRule(unittest.TestCase):
    def test_positive_interval_survives(self):
        self.assertEqual(classify_incremental(0.01, 0.001, 0.02, n_days=80), "survives")

    def test_positive_estimate_with_a_wide_interval_is_promising(self):
        self.assertEqual(classify_incremental(0.01, -0.02, 0.04, n_days=80), "promising")

    def test_negative_estimate_fails(self):
        self.assertEqual(classify_incremental(-0.01, -0.03, 0.01, n_days=80), "fails")

    def test_zero_estimate_fails(self):
        self.assertEqual(classify_incremental(0.0, -0.01, 0.01, n_days=80), "fails")

    def test_short_sample_is_not_tested(self):
        self.assertEqual(classify_incremental(0.02, 0.01, 0.03, n_days=10), "not_tested")


def _rows(name_net, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        day = (pd.Timestamp("2024-01-02") + pd.Timedelta(days=i)).date().isoformat()
        rows.append({"date": day, "symbol": "USO", "net": float(name_net(rng))})
    return rows


class TestReportComparisons(unittest.TestCase):
    def test_summary_keeps_all_three_comparisons(self):
        arm = _rows(lambda rng: 0.01 + rng.normal(0, 0.001), seed=1)
        baseline = _rows(lambda rng: rng.normal(0, 0.001), seed=2)
        hold = _rows(lambda rng: 0.002, seed=3)
        simple = {
            "momentum_5": _rows(lambda rng: 0.0, seed=4),
            "ma_crossover": _rows(lambda rng: 0.0, seed=5),
            "mean_reversion": _rows(lambda rng: -0.001, seed=6),
        }
        result = summarize_arm("baseline_ta", arm, baseline, hold, simple, seed=1)
        self.assertIn(result.status, {"survives", "promising"})
        self.assertIn("mean", result.vs_buy_hold)
        self.assertEqual(set(result.vs_simple), {"momentum_5", "ma_crossover", "mean_reversion"})

    def test_promising_is_not_labeled_fails(self):
        # A small positive edge with noise wide enough to cross zero.
        arm, baseline, hold = [], [], []
        simple = {name: [] for name in ("momentum_5", "ma_crossover", "mean_reversion")}
        for i in range(80):
            day = (pd.Timestamp("2024-01-02") + pd.Timedelta(days=i)).date().isoformat()
            # Positive on average. Half the days are negative, so the
            # interval still crosses zero.
            arm.append({"date": day, "symbol": "USO", "net": 0.05 if i < 40 else -0.04})
            baseline.append({"date": day, "symbol": "USO", "net": 0.0})
            hold.append({"date": day, "symbol": "USO", "net": 0.0005})
            for name in simple:
                simple[name].append({"date": day, "symbol": "USO", "net": 0.0})
        result = summarize_arm("baseline_lunar", arm, baseline, hold, simple, seed=3)
        self.assertEqual(result.status, "promising")
        self.assertGreater(result.incremental_mean, 0)
        self.assertLess(result.incremental_ci[0], 0)
        self.assertGreater(result.incremental_ci[1], 0)


class TestSyntheticWalk(unittest.TestCase):
    def test_one_symbol_emits_a_report(self):
        rng = np.random.default_rng(8)
        index = pd.bdate_range("2022-01-03", periods=160)
        close = 20 * np.cumprod(1 + rng.normal(0.0002, 0.01, size=len(index)))
        frame = pd.DataFrame(
            {
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": rng.integers(1000, 5000, size=len(index)),
            },
            index=index,
        )
        report = run_ablation({"USO": frame}, draws=50, include_gp=False, seed=1)
        self.assertIn("baseline", report["arms"])
        self.assertIn("baseline_lunar", report["arms"])
        self.assertEqual(
            set(report["comparisons"]),
            {"baseline", "buy_and_hold", "momentum_5", "ma_crossover", "mean_reversion"},
        )
        lunar = report["arms"]["baseline_lunar"]
        self.assertIn(lunar["status"], {"survives", "promising", "fails", "not_tested"})
        self.assertIn("vs_buy_hold", lunar)
        self.assertIn("momentum_5", lunar["vs_simple"])


if __name__ == "__main__":
    unittest.main()
