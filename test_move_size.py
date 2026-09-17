"""Tests for the move-size analysis (Tier 0).

The question: does accuracy hold up when the model predicts a LARGE move?
Break-even needs 60.2% on a 0.59% forecast and 53.0% on a 2% one, and measured
accuracy tops out near 53.2% -- so the answer decides whether direction
trading is viable here at all.

The trap this guards against is the one that produced every earlier false
finding: large-move buckets are small, so a flattering point estimate on 40
observations means nothing. Hence Wilson intervals and a lower-bound test.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from analyze_move_size import DEFAULT_BUCKETS, _wilson  # noqa: E402


class TestWilsonInterval(unittest.TestCase):
    def test_point_estimate_is_the_hit_rate(self):
        p, _, _ = _wilson(53, 100)
        self.assertAlmostEqual(p, 0.53, places=6)

    def test_small_samples_get_wide_intervals(self):
        """A 60% point estimate on 10 observations must not look convincing."""
        _, lo, hi = _wilson(6, 10)
        self.assertLess(lo, 0.35)
        self.assertGreater(hi, 0.80)

    def test_large_samples_tighten(self):
        _, lo_small, hi_small = _wilson(530, 1000)
        _, lo_big, hi_big = _wilson(5300, 10000)
        self.assertLess(hi_big - lo_big, hi_small - lo_small)

    def test_interval_never_escapes_zero_to_one(self):
        """Where the normal approximation would run past the boundary."""
        _, lo, hi = _wilson(0, 12)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        _, lo, hi = _wilson(12, 12)
        self.assertLessEqual(hi, 1.0)

    def test_empty_bucket_is_not_a_crash(self):
        self.assertEqual(_wilson(0, 0), (0.0, 0.0, 0.0))


class TestBucketEdges(unittest.TestCase):
    def test_buckets_straddle_the_breakeven_crossover(self):
        """Break-even falls below achievable accuracy around a 2% move, so a
        2.0 edge has to exist or the crossover is invisible."""
        self.assertIn(2.0, DEFAULT_BUCKETS)

    def test_buckets_are_ascending_and_cover_everything(self):
        self.assertEqual(DEFAULT_BUCKETS, sorted(DEFAULT_BUCKETS))
        self.assertEqual(DEFAULT_BUCKETS[0], 0.0)
        self.assertGreaterEqual(DEFAULT_BUCKETS[-1], 100.0)


class TestBreakEvenArithmetic(unittest.TestCase):
    """The arithmetic the whole Tier 0 question rests on."""

    @staticmethod
    def _breakeven(cost_frac, move_frac):
        return 0.5 + cost_frac / (2 * move_frac)

    def test_mean_move_needs_unreachable_accuracy(self):
        # XLE 12bp against the 0.586% mean forecast
        self.assertAlmostEqual(self._breakeven(0.0012, 0.00586), 0.602, places=2)

    def test_two_percent_move_is_reachable(self):
        # 53.0% needed vs the 53.2% calibration ceiling
        self.assertLess(self._breakeven(0.0012, 0.02), 0.532)

    def test_expensive_symbol_stays_out_of_reach_even_on_big_moves(self):
        """CANE at 45bp needs more than the ceiling even on a 2% move."""
        self.assertGreater(self._breakeven(0.0045, 0.02), 0.532)

    def test_requirement_falls_as_move_grows(self):
        needs = [self._breakeven(0.0012, m) for m in (0.005, 0.01, 0.02, 0.03)]
        self.assertEqual(needs, sorted(needs, reverse=True))


if __name__ == "__main__":
    unittest.main()
