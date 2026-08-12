"""Tests for the directional threshold in filter_signals_by_thresholds.

`Signal.prob_profit` is P(next return > 0) — a directional probability, not a
per-trade profit probability. The filter therefore has to judge a SELL against
the down-tail. Before this was fixed, the most confident bearish signals were
the ones most certain to be discarded.
"""

from __future__ import annotations

import unittest

from logic.data_structures import Signal
from logic.signal_engine import filter_signals_by_thresholds


def make_signal(signal_type: str, confidence: float, prob_profit: float) -> Signal:
    return Signal(
        symbol="TEST",
        signal_type=signal_type,
        confidence=confidence,
        prob_profit=prob_profit,
        meta={},
    )


class TestDirectionalThreshold(unittest.TestCase):
    def _filter(self, signals, min_confidence=0.50, min_prob_up=0.50):
        return filter_signals_by_thresholds(
            signals, min_confidence=min_confidence, min_prob_up=min_prob_up
        )

    def test_strong_sell_is_no_longer_discarded(self):
        # The real SLV case: STRONG SELL, confidence 1.0, prob_profit 0.0005.
        # That is a 99.95% down-read and must pass.
        signal = make_signal("sell", 1.0, 0.0005)
        self.assertEqual(len(self._filter([signal])), 1)
        self.assertEqual(signal.meta["threshold_decision"], "pass")
        self.assertAlmostEqual(signal.meta["directional_probability"], 0.9995)

    def test_weak_sell_is_rejected(self):
        # prob_profit 0.60 means P(down) = 0.40, below the 0.50 floor.
        signal = make_signal("sell", 0.9, 0.60)
        self.assertEqual(len(self._filter([signal])), 0)
        self.assertEqual(signal.meta["threshold_decision"], "reject")
        self.assertIn("P(down)", signal.meta["threshold_rejection_reason"])

    def test_buy_threshold_is_unchanged(self):
        strong = make_signal("buy", 0.8, 0.75)
        weak = make_signal("buy", 0.8, 0.25)
        self.assertEqual(len(self._filter([strong])), 1)
        self.assertEqual(len(self._filter([weak])), 0)
        self.assertIn("P(up)", weak.meta["threshold_rejection_reason"])

    def test_confidence_floor_still_applies_to_sells(self):
        # A confident direction does not excuse a low-confidence signal.
        signal = make_signal("sell", 0.10, 0.0005)
        self.assertEqual(len(self._filter([signal])), 0)
        self.assertIn("confidence", signal.meta["threshold_rejection_reason"])

    def test_holds_are_retained_regardless(self):
        signal = make_signal("hold", 0.2, 0.2)
        self.assertEqual(len(self._filter([signal])), 1)
        self.assertEqual(signal.meta["threshold_decision"], "hold")

    def test_directional_probability_recorded_for_every_signal(self):
        buy = make_signal("buy", 0.8, 0.70)
        sell = make_signal("sell", 0.8, 0.30)
        self._filter([buy, sell])
        self.assertAlmostEqual(buy.meta["directional_probability"], 0.70)
        self.assertAlmostEqual(sell.meta["directional_probability"], 0.70)

    def test_symmetric_reads_are_treated_symmetrically(self):
        # A 70% up-read as a BUY and a 70% down-read as a SELL are the same
        # strength of claim and must get the same verdict.
        buy = make_signal("buy", 0.8, 0.70)
        sell = make_signal("sell", 0.8, 0.30)
        self.assertEqual(len(self._filter([buy, sell])), 2)

    def test_boundary_is_inclusive_for_both_directions(self):
        buy = make_signal("buy", 0.5, 0.50)
        sell = make_signal("sell", 0.5, 0.50)
        self.assertEqual(len(self._filter([buy, sell])), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
