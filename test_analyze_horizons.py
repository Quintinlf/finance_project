"""Tests for the forward-horizon analysis.

Every accuracy figure before this graded predictions against the next day's
return -- one arbitrary and especially noisy choice. This asks whether signal
exists at 5, 10, or 20 days instead. Two statistical traps make naive versions
of that question misleading, and both are tested here.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from analyze_horizons import _effective_n, _signed  # noqa: E402


class TestSignedSignal(unittest.TestCase):
    def test_buy_is_positive_sell_is_negative(self):
        self.assertEqual(_signed("buy", 0.8), 0.8)
        self.assertEqual(_signed("sell", 0.8), -0.8)

    def test_neutral_is_zero_not_missing(self):
        """A neutral call is a real observation with no directional claim --
        it belongs in the IC sample but not in the accuracy count."""
        self.assertEqual(_signed("hold", 0.5), 0.0)
        self.assertEqual(_signed("neutral", 0.5), 0.0)

    def test_missing_direction_is_none(self):
        self.assertIsNone(_signed(None, 0.5))

    def test_unknown_direction_is_none_rather_than_guessed(self):
        self.assertIsNone(_signed("sideways-ish", 0.5))

    def test_missing_confidence_defaults_to_zero_magnitude(self):
        self.assertEqual(_signed("buy", None), 0.0)


class TestEffectiveN(unittest.TestCase):
    """Overlapping forward windows make naive confidence intervals too narrow.

    Consecutive 20-day forward returns share 19 of their 20 days, so 15,000
    rows carry nowhere near 15,000 independent observations. Without this
    correction almost anything looks significant.
    """

    def test_one_day_horizon_is_unshrunk(self):
        self.assertEqual(_effective_n(1000, 1), 1000)

    def test_longer_horizons_shrink_proportionally(self):
        self.assertEqual(_effective_n(1000, 20), 50)
        self.assertEqual(_effective_n(15000, 20), 750)

    def test_never_drops_below_one(self):
        self.assertGreaterEqual(_effective_n(5, 20), 1.0)

    def test_shrinkage_widens_the_significance_bar(self):
        """The practical consequence: the |IC| needed to claim a finding grows
        with horizon, which is why a 0.0575 IC at 20 days is not a result."""
        import math

        bar_1d = 1.96 / math.sqrt(_effective_n(15000, 1))
        bar_20d = 1.96 / math.sqrt(_effective_n(15000, 20))
        self.assertGreater(bar_20d, bar_1d * 4)


if __name__ == "__main__":
    unittest.main()
