"""Tests for volatility-scaled position sizing.

Grounding: measured on this universe (scripts/analyze_volatility.py, 32,432
overlapping windows), trailing 20-day realized vol predicts forward 20-day vol
with r=0.708 against an overlap-corrected bar of 0.049. The best DIRECTION
signal ever measured was IC 0.021 and survived nothing. So magnitude is ~50%
predictable and direction is not predictable at all.

The sizing rule that follows is deliberately one-directional: it can only
shrink a position, never grow one.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from logic.volatility_sizing import (
    DEFAULT_TARGET_VOL,
    MIN_SCALE,
    apply_volatility_sizing,
    compute_sizing,
    predict_volatility,
    realized_vol,
)


def _history(daily_move: float, n: int = 40):
    """Series alternating +/- daily_move, giving a known realized vol."""
    closes, price = [], 100.0
    for i in range(n):
        price *= (1 + daily_move) if i % 2 == 0 else (1 - daily_move)
        closes.append(price)
    return pd.DataFrame({"Close": closes}, index=pd.date_range("2026-01-01", periods=n))


class TestRealizedVol(unittest.TestCase):
    def test_annualizes_daily_deviation(self):
        # 1% daily moves -> ~1% * sqrt(252) ~= 16% annualized
        vol = realized_vol([0.01, -0.01] * 20)
        self.assertAlmostEqual(vol, 0.01 * (252 ** 0.5), delta=0.02)

    def test_calm_series_has_lower_vol_than_wild(self):
        calm = realized_vol([0.002, -0.002] * 20)
        wild = realized_vol([0.04, -0.04] * 20)
        self.assertLess(calm, wild)

    def test_too_short_returns_none(self):
        self.assertIsNone(realized_vol([0.01]))


class TestComputeSizing(unittest.TestCase):
    def test_high_vol_symbol_is_scaled_down(self):
        """UNG runs ~58% annualized; a flat position there is ~4x the risk of
        the same dollars in DBA."""
        with patch("logic.price_cache.get_history", return_value=_history(0.04)):
            s = compute_sizing("UNG")
        self.assertLess(s.scale, 1.0)
        self.assertGreater(s.predicted_vol, DEFAULT_TARGET_VOL)

    def test_low_vol_symbol_is_never_scaled_up(self):
        """The clamp is the point: levering into calm names relies on a
        relationship that breaks exactly during regime shifts."""
        with patch("logic.price_cache.get_history", return_value=_history(0.001)):
            s = compute_sizing("DBA")
        self.assertEqual(s.scale, 1.0)
        self.assertIn("never sizes UP", s.reason)

    def test_extreme_vol_is_floored_not_zeroed(self):
        """Past a point the position rounds to zero shares, which is a
        different decision than sizing it small."""
        with patch("logic.price_cache.get_history", return_value=_history(0.30)):
            s = compute_sizing("CRAZY")
        self.assertEqual(s.scale, MIN_SCALE)

    def test_missing_history_does_not_grant_a_bigger_position(self):
        with patch("logic.price_cache.get_history", return_value=None):
            s = compute_sizing("NODATA")
        self.assertEqual(s.scale, 1.0)
        self.assertIsNone(s.predicted_vol)
        self.assertIn("no volatility estimate", s.reason)

    def test_scale_is_proportional_to_target_over_predicted(self):
        with patch("logic.price_cache.get_history", return_value=_history(0.04)):
            s = compute_sizing("UNG", target_vol=0.25)
        expected = max(MIN_SCALE, 0.25 / s.predicted_vol)
        self.assertAlmostEqual(s.scale, expected, places=6)

    def test_never_exceeds_one_for_any_target(self):
        with patch("logic.price_cache.get_history", return_value=_history(0.002)):
            s = compute_sizing("CALM", target_vol=0.90)
        self.assertLessEqual(s.scale, 1.0)


class TestApplyVolatilitySizing(unittest.TestCase):
    def _sig(self, symbol="UNG", kind="buy"):
        return SimpleNamespace(symbol=symbol, signal_type=kind, meta={})

    def test_annotates_and_marks_enforcement(self):
        sigs = [self._sig()]
        with patch("logic.price_cache.get_history", return_value=_history(0.04)):
            apply_volatility_sizing(sigs, enforce=True, verbose=False)
        self.assertIn("vol_scale", sigs[0].meta)
        self.assertTrue(sigs[0].meta["vol_sizing_enforced"])
        self.assertLess(sigs[0].meta["vol_scale"], 1.0)

    def test_shadow_mode_records_without_enforcing(self):
        sigs = [self._sig()]
        with patch("logic.price_cache.get_history", return_value=_history(0.04)):
            apply_volatility_sizing(sigs, enforce=False, verbose=False)
        self.assertFalse(sigs[0].meta["vol_sizing_enforced"])
        self.assertLess(sigs[0].meta["vol_scale"], 1.0)  # still measured

    def test_hold_signals_are_skipped(self):
        sigs = [self._sig(kind="hold")]
        apply_volatility_sizing(sigs, verbose=False)
        self.assertNotIn("vol_scale", sigs[0].meta)


class TestSizingIntegration(unittest.TestCase):
    """The scale must actually reach quantity, and only when enforced."""

    def _plan(self, meta):
        from logic.data_structures import ExecutionConfig, PositionState, Signal
        from logic.execution_engine import build_order_plan

        sig = Signal(symbol="UNG", signal_type="buy", confidence=0.8,
                     prob_profit=0.7, meta=dict(meta))
        pos = PositionState(symbol="UNG", quantity=0, avg_entry_price=0.0,
                            side="flat", source="test")
        cfg = ExecutionConfig(execution_mode="simulation", allow_short_selling=False)
        return build_order_plan(sig, pos, cfg, account_cash=10000.0,
                                current_price=10.0, account_equity=10000.0)

    def test_enforced_scale_reduces_quantity(self):
        full = self._plan({"vol_scale": 1.0, "vol_sizing_enforced": True})
        half = self._plan({"vol_scale": 0.5, "vol_sizing_enforced": True})
        self.assertLess(half.quantity, full.quantity)

    def test_unenforced_scale_is_ignored(self):
        full = self._plan({"vol_scale": 1.0, "vol_sizing_enforced": True})
        shadow = self._plan({"vol_scale": 0.5, "vol_sizing_enforced": False})
        self.assertEqual(shadow.quantity, full.quantity)

    def test_absent_metadata_behaves_as_unscaled(self):
        base = self._plan({})
        full = self._plan({"vol_scale": 1.0, "vol_sizing_enforced": True})
        self.assertEqual(base.quantity, full.quantity)


if __name__ == "__main__":
    unittest.main()
