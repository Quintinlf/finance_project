"""Tests for the cost-aware edge gate and the split-adjustment fix.

The gate exists because nothing in the pipeline asked whether a predicted move
was larger than the cost of trading it. On this universe round-trip costs run
0.12% (XLE) to 0.45% (CANE, SOYB) against a mean absolute forecast of 0.586%.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from logic import costs
from logic.edge_gate import DEFAULT_EDGE_MARGIN, apply_edge_gate, evaluate_edge


def _signal(symbol="UNG", side="buy", forecast=0.01, p_dir=0.7):
    return SimpleNamespace(
        symbol=symbol,
        signal_type=side,
        confidence=0.8,
        prob_profit=p_dir,
        meta={"ensemble_forecast_return": forecast, "directional_probability": p_dir},
    )


class TestEvaluateEdge(unittest.TestCase):
    def test_large_move_on_a_cheap_symbol_passes(self):
        v = evaluate_edge(_signal("XLE", forecast=0.02, p_dir=0.75))
        self.assertTrue(v.passes, msg=v.reason)
        self.assertGreater(v.expected_value_pct, 0)

    def test_coin_flip_probability_can_never_pass(self):
        """Below 50% directional probability the EV term is negative outright."""
        v = evaluate_edge(_signal("XLE", forecast=0.05, p_dir=0.50))
        self.assertFalse(v.passes)
        self.assertIn("coin flip", v.reason)

    def test_thin_symbol_costs_swallow_a_typical_move(self):
        """CANE at 45bp round trip against the measured 0.586% mean forecast."""
        v = evaluate_edge(_signal("CANE", forecast=0.00586, p_dir=0.55))
        self.assertFalse(v.passes)
        self.assertAlmostEqual(v.cost_pct, 0.45, places=2)

    def test_same_move_survives_on_a_liquid_symbol(self):
        """Identical forecast, cheaper venue — the cost is what decides it."""
        thin = evaluate_edge(_signal("CANE", forecast=0.01, p_dir=0.70))
        liquid = evaluate_edge(_signal("XLE", forecast=0.01, p_dir=0.70))
        self.assertGreater(liquid.expected_value_pct, thin.expected_value_pct)

    def test_margin_requirement_rejects_a_marginal_trade(self):
        """Barely clearing costs is not worth the risk of being wrong."""
        # 0.57% move at 65% clears XLE's 0.12% cost by only 0.05% — real, but
        # not by the 1.5x demanded.
        v = evaluate_edge(_signal("XLE", forecast=0.0057, p_dir=0.65), margin=DEFAULT_EDGE_MARGIN)
        self.assertFalse(v.passes)
        self.assertGreater(v.expected_value_pct, 0)  # positive, but under the margin
        self.assertIn("margin", v.reason)

    def test_missing_forecast_is_rejected_not_assumed(self):
        sig = _signal()
        sig.meta.pop("ensemble_forecast_return")
        v = evaluate_edge(sig)
        self.assertFalse(v.passes)
        self.assertIn("no forecast magnitude", v.reason)

    def test_sell_side_uses_the_down_tail(self):
        """prob_profit is P(up); a SELL bets on the complement."""
        sig = SimpleNamespace(
            symbol="XLE", signal_type="sell", confidence=0.9, prob_profit=0.10,
            meta={"ensemble_forecast_return": -0.02},
        )
        v = evaluate_edge(sig)
        self.assertAlmostEqual(v.directional_prob, 0.90, places=6)
        self.assertTrue(v.passes, msg=v.reason)


class TestApplyEdgeGate(unittest.TestCase):
    def test_shadow_mode_annotates_but_never_demotes(self):
        sigs = [_signal("CANE", forecast=0.001, p_dir=0.52)]
        apply_edge_gate(sigs, enforce=False, verbose=False)
        self.assertEqual(sigs[0].signal_type, "buy")          # untouched
        self.assertFalse(sigs[0].meta["edge_passes"])          # but recorded
        self.assertIn("edge_expected_value_pct", sigs[0].meta)

    def test_enforce_mode_demotes_a_failing_signal_to_hold(self):
        sigs = [_signal("CANE", forecast=0.001, p_dir=0.52)]
        apply_edge_gate(sigs, enforce=True, verbose=False)
        self.assertEqual(sigs[0].signal_type, "hold")
        self.assertIn("edge gate", sigs[0].meta["threshold_reason"])

    def test_enforce_mode_leaves_a_passing_signal_alone(self):
        sigs = [_signal("XLE", forecast=0.03, p_dir=0.80)]
        apply_edge_gate(sigs, enforce=True, verbose=False)
        self.assertEqual(sigs[0].signal_type, "buy")

    def test_hold_signals_pass_through_untouched(self):
        sigs = [SimpleNamespace(symbol="WEAT", signal_type="hold", meta={})]
        out = apply_edge_gate(sigs, enforce=True, verbose=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].signal_type, "hold")
        self.assertNotIn("edge_passes", out[0].meta)


class TestSplitAdjustmentFix(unittest.TestCase):
    """KOLD reverse-split and the tracker reported a +205% daily return.

    price_at_signal is a raw price recorded at decision time; yfinance
    back-adjusts history. Dividing one by the other fabricates a return across
    any corporate action, and three such rows dominated every average computed
    from the table.
    """

    def _hist(self, dates, closes):
        import pandas as pd
        return pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(dates))

    def test_return_is_computed_within_one_series(self):
        from logic.model_performance_tracker import _compute_realized_return_from_row

        hist = self._hist(["2026-08-12", "2026-08-13", "2026-08-14"], [30.0, 30.0, 30.6])
        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.return_value = hist
            # A stale raw price of 10.00 (pre-reverse-split) must NOT be used.
            ret = _compute_realized_return_from_row(
                symbol="KOLD", timestamp_iso="2026-08-13T03:47:59+00:00", price_at_signal=10.00
            )
        self.assertIsNotNone(ret)
        self.assertAlmostEqual(ret, 0.02, places=4)  # 30.0 -> 30.6, not 30.6/10

    def test_implausible_move_is_discarded(self):
        from logic.model_performance_tracker import _compute_realized_return_from_row

        hist = self._hist(["2026-08-12", "2026-08-13", "2026-08-14"], [10.0, 10.0, 31.0])
        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.return_value = hist
            ret = _compute_realized_return_from_row(
                symbol="KOLD", timestamp_iso="2026-08-13T03:47:59+00:00", price_at_signal=10.00
            )
        self.assertIsNone(ret)

    def test_ordinary_move_survives(self):
        from logic.model_performance_tracker import _compute_realized_return_from_row

        hist = self._hist(["2026-08-12", "2026-08-13", "2026-08-14"], [24.19, 24.19, 24.97])
        with patch("yfinance.Ticker") as tk:
            tk.return_value.history.return_value = hist
            ret = _compute_realized_return_from_row(
                symbol="WEAT", timestamp_iso="2026-08-13T03:47:59+00:00", price_at_signal=24.19
            )
        self.assertAlmostEqual(ret, 0.03224, places=4)


if __name__ == "__main__":
    unittest.main()
