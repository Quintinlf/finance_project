"""Tests for the signal-quality statistics.

These decide whether the strategy gets tuned or rebuilt, so the statistics
themselves need to be right. In particular a hit rate must be judged against
the sample's base rate: in a market that drifts up, "58% of BUYs rose" is worth
nothing if 58% of all bars rose.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from logic.signal_quality import (
    Decision,
    _fisher_ci,
    _wilson_interval,
    attach_forward_returns,
    confidence_buckets,
    format_report,
    hit_rates,
    information_coefficients,
    load_decisions,
    spearman,
)


class FakeFrame:
    """Minimal stand-in for the pandas frame attach_forward_returns expects."""

    def __init__(self, dates, closes):
        self.index = dates
        self._closes = closes

    def __getitem__(self, key):
        assert key == "Close"
        return self

    @property
    def values(self):
        return self._closes


def decision(symbol="X", date="2026-01-01", signal="buy", conf=0.8, prob=0.6, fwd=None):
    d = Decision(
        date=date, symbol=symbol, close=100.0,
        signal_type=signal, confidence=conf, prob_profit=prob,
    )
    if fwd:
        d.forward_returns.update(fwd)
    return d


class TestWilsonInterval(unittest.TestCase):
    def test_brackets_the_point_estimate(self):
        lo, hi = _wilson_interval(30, 100)
        self.assertLess(lo, 0.30)
        self.assertGreater(hi, 0.30)

    def test_interval_narrows_with_sample_size(self):
        narrow = _wilson_interval(300, 1000)
        wide = _wilson_interval(3, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_stays_within_zero_one_at_the_extremes(self):
        lo, hi = _wilson_interval(0, 10)
        self.assertGreaterEqual(lo, 0.0)
        lo, hi = _wilson_interval(10, 10)
        self.assertLessEqual(hi, 1.0)

    def test_empty_sample_is_not_a_crash(self):
        self.assertEqual(_wilson_interval(0, 0), (0.0, 0.0))


class TestSpearman(unittest.TestCase):
    def test_perfect_monotonic_relationships(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_is_rank_based_not_linear(self):
        # Monotonic but wildly non-linear still gives 1.0.
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [1, 10, 1000, 100000]), 1.0)

    def test_constant_input_is_zero_not_nan(self):
        result = spearman([1, 1, 1, 1], [1, 2, 3, 4])
        self.assertEqual(result, 0.0)

    def test_too_few_points_returns_zero(self):
        self.assertEqual(spearman([1, 2], [1, 2]), 0.0)


class TestFisherInterval(unittest.TestCase):
    def test_interval_contains_the_estimate(self):
        lo, hi = _fisher_ci(0.3, 100)
        self.assertLess(lo, 0.3)
        self.assertGreater(hi, 0.3)

    def test_zero_correlation_interval_straddles_zero(self):
        lo, hi = _fisher_ci(0.0, 500)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_large_sample_excludes_zero_for_real_correlation(self):
        lo, hi = _fisher_ci(0.25, 1000)
        self.assertGreater(lo, 0)


class TestForwardReturns(unittest.TestCase):
    def setUp(self):
        self.history = {
            "X": FakeFrame(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                [100.0, 110.0, 121.0, 133.1],
            )
        }

    def test_joins_the_right_future_bar(self):
        d = decision(date="2026-01-01")
        out = attach_forward_returns([d], self.history, horizons=(1, 2))
        self.assertAlmostEqual(out[0].forward_returns[1], 0.10)
        self.assertAlmostEqual(out[0].forward_returns[2], 0.21)

    def test_horizon_past_the_end_is_dropped_not_padded(self):
        d = decision(date="2026-01-03")
        out = attach_forward_returns([d], self.history, horizons=(1, 5))
        self.assertIn(1, out[0].forward_returns)
        self.assertNotIn(5, out[0].forward_returns)

    def test_unknown_symbol_is_skipped(self):
        out = attach_forward_returns([decision(symbol="ZZZ")], self.history)
        self.assertEqual(out, [])

    def test_decision_with_no_usable_horizon_is_excluded(self):
        d = decision(date="2026-01-04")
        out = attach_forward_returns([d], self.history, horizons=(1,))
        self.assertEqual(out, [])


class TestHitRates(unittest.TestCase):
    def test_edge_is_measured_against_the_base_rate(self):
        # 8 of 10 bars rise, so a BUY that is right 8/10 times knows nothing.
        decisions = [decision(signal="buy", fwd={5: 0.01}) for _ in range(8)]
        decisions += [decision(signal="buy", fwd={5: -0.01}) for _ in range(2)]
        rates = hit_rates(decisions, horizon=5)
        buy = next(r for r in rates if r.label == "BUY")
        self.assertAlmostEqual(buy.hit_rate, 0.8)
        self.assertAlmostEqual(buy.base_rate, 0.8)
        self.assertAlmostEqual(buy.edge, 0.0)
        self.assertFalse(buy.significant)

    def test_sell_is_scored_on_downward_moves(self):
        decisions = [decision(signal="sell", fwd={5: -0.02}) for _ in range(9)]
        decisions += [decision(signal="buy", fwd={5: 0.02}) for _ in range(1)]
        rates = hit_rates(decisions, horizon=5)
        sell = next(r for r in rates if r.label == "SELL")
        # All nine SELLs were followed by a fall.
        self.assertAlmostEqual(sell.hit_rate, 1.0)

    def test_genuine_edge_is_flagged_significant(self):
        # Base rate 50%, BUYs right 90% of the time, large sample.
        decisions = [decision(signal="buy", fwd={5: 0.01}) for _ in range(180)]
        decisions += [decision(signal="buy", fwd={5: -0.01}) for _ in range(20)]
        decisions += [decision(signal="hold", fwd={5: -0.01}) for _ in range(160)]
        rates = hit_rates(decisions, horizon=5)
        buy = next(r for r in rates if r.label == "BUY")
        self.assertGreater(buy.edge, 0.0)
        self.assertTrue(buy.significant)

    def test_empty_input_returns_empty(self):
        self.assertEqual(hit_rates([], horizon=5), [])


class TestInformationCoefficient(unittest.TestCase):
    def test_detects_a_real_relationship(self):
        decisions = [
            decision(prob=i / 100.0, fwd={5: (i - 50) / 1000.0}) for i in range(100)
        ]
        ics = information_coefficients(decisions, horizons=(5,))
        self.assertAlmostEqual(ics[0].ic, 1.0, places=6)
        self.assertTrue(ics[0].significant)

    def test_noise_is_not_flagged_significant(self):
        import random
        rng = random.Random(11)
        decisions = [
            decision(prob=rng.random(), fwd={5: rng.gauss(0, 0.01)}) for _ in range(300)
        ]
        ics = information_coefficients(decisions, horizons=(5,))
        self.assertFalse(ics[0].significant)

    def test_horizon_with_too_little_data_is_skipped(self):
        decisions = [decision(fwd={5: 0.01}) for _ in range(3)]
        self.assertEqual(information_coefficients(decisions, horizons=(5,)), [])


class TestConfidenceBuckets(unittest.TestCase):
    def test_sell_returns_are_signed_by_the_bet(self):
        # A SELL followed by a fall is a win, so it must show positive.
        decisions = [decision(signal="sell", conf=0.75, fwd={5: -0.03})]
        buckets = confidence_buckets(decisions, horizon=5)
        self.assertAlmostEqual(buckets[0].mean_forward_return, 0.03)
        self.assertAlmostEqual(buckets[0].hit_rate, 1.0)

    def test_holds_are_excluded_by_default(self):
        decisions = [decision(signal="hold", conf=0.5, fwd={5: 0.01})]
        self.assertEqual(confidence_buckets(decisions, horizon=5), [])

    def test_buckets_split_on_confidence(self):
        decisions = [decision(conf=0.55, fwd={5: 0.01}), decision(conf=0.85, fwd={5: 0.02})]
        buckets = confidence_buckets(decisions, horizon=5)
        self.assertEqual(len(buckets), 2)
        self.assertEqual([b.n for b in buckets], [1, 1])


class TestLoadAndReport(unittest.TestCase):
    def test_load_skips_malformed_rows(self):
        payload = {"decisions": [
            {"date": "2026-01-01", "symbol": "X", "close": 10.0,
             "signal_type": "buy", "confidence": 0.8, "prob_profit": 0.6},
            {"date": "2026-01-02", "symbol": "Y"},           # missing close
            {"date": "2026-01-03", "symbol": "Z", "close": "abc",
             "signal_type": "buy", "confidence": 0.5, "prob_profit": 0.5},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bt.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            decisions = load_decisions(path)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].symbol, "X")

    def test_report_states_no_edge_when_there_is_none(self):
        import random
        rng = random.Random(5)
        decisions = [
            decision(signal=rng.choice(["buy", "sell"]), prob=rng.random(),
                     fwd={5: rng.gauss(0, 0.01)})
            for _ in range(300)
        ]
        report = format_report(decisions, horizons=(5,), primary_horizon=5)
        self.assertIn("NO detectable predictive signal", report)

    def test_report_flags_a_real_edge(self):
        decisions = [
            decision(prob=i / 200.0, fwd={5: (i - 100) / 1000.0}) for i in range(200)
        ]
        report = format_report(decisions, horizons=(5,), primary_horizon=5)
        self.assertIn("excludes zero", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
