"""Tests for logic.eos_bridge and its wiring into the trading pipeline.

Two things need guarding here:

1. The bridge produces sane numbers when the eos checkout is present.
2. The pipeline behaves *identically to before* when it is absent, or when
   eos_mode is 'off'/'shadow'. A missing sibling repo must never change a
   trade or crash a run.
"""

from __future__ import annotations

import math
import random
import unittest

from logic import eos_bridge
from logic.data_structures import ExecutionConfig, PositionState, Signal
from logic.execution_engine import build_order_plan


def synthetic_prices(n: int = 300, seed: int = 7, vol_break_at: int = 200) -> list:
    """Price path with a deliberate volatility regime break."""
    rng = random.Random(seed)
    prices = [100.0]
    for i in range(n):
        vol = 0.008 if i < vol_break_at else 0.025
        prices.append(prices[-1] * (1 + rng.gauss(0.0004, vol)))
    return prices


class TestReturnExtraction(unittest.TestCase):
    def test_extracts_simple_returns_from_sequence(self):
        returns = eos_bridge.extract_returns([100.0, 110.0, 99.0])
        self.assertEqual(len(returns), 2)
        self.assertAlmostEqual(returns[0], 0.10)
        self.assertAlmostEqual(returns[1], -0.10)

    def test_handles_none_and_empty(self):
        self.assertEqual(eos_bridge.extract_returns(None), [])
        self.assertEqual(eos_bridge.extract_returns([]), [])

    def test_skips_nonpositive_and_nonfinite_prices(self):
        returns = eos_bridge.extract_returns([100.0, 0.0, 50.0, float("nan"), 60.0])
        self.assertTrue(all(math.isfinite(r) for r in returns))


class TestCatalogue(unittest.TestCase):
    def test_catalogue_covers_all_three_domains(self):
        if not eos_bridge.available():
            self.skipTest(f"eos unavailable: {eos_bridge.EOS_IMPORT_ERROR}")
        names = eos_bridge.list_algorithms()
        self.assertEqual(len(names), 27)
        for prefix in ("finance.", "physics_finance.", "quantum_finance."):
            self.assertTrue(
                any(n.startswith(prefix) for n in names), f"no {prefix} algorithms"
            )

    def test_call_returns_plain_dict(self):
        if not eos_bridge.available():
            self.skipTest("eos unavailable")
        result = eos_bridge.call(
            "physics_finance.black_scholes_price",
            spot=100.0, strike=100.0, rate=0.05, volatility=0.2, maturity=1.0,
        )
        self.assertIsInstance(result, dict)

    def test_unknown_algorithm_raises_keyerror(self):
        if not eos_bridge.available():
            self.skipTest("eos unavailable")
        with self.assertRaises(KeyError):
            eos_bridge.call("finance.not_a_real_tool")


class TestTradingHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = synthetic_prices()
        cls.returns = eos_bridge.extract_returns(cls.prices)

    def setUp(self):
        if not eos_bridge.available():
            self.skipTest(f"eos unavailable: {eos_bridge.EOS_IMPORT_ERROR}")

    def test_garch_detects_elevated_volatility(self):
        # The series triples its volatility at bar 200, so the forward
        # forecast must sit above the whole-sample unconditional estimate.
        result = eos_bridge.garch_volatility(returns=self.returns)
        self.assertIsNotNone(result)
        self.assertGreater(result.vol_ratio, 1.0)
        self.assertGreater(result.forecast_next_vol, 0.0)

    def test_helpers_return_none_on_short_series(self):
        short = self.returns[:10]
        self.assertIsNone(eos_bridge.garch_volatility(returns=short))
        self.assertIsNone(eos_bridge.hurst_regime(returns=short))
        self.assertIsNone(eos_bridge.hmm_regime(returns=short))
        self.assertIsNone(eos_bridge.historical_var(returns=short))

    def test_hurst_label_matches_exponent(self):
        result = eos_bridge.hurst_regime(returns=self.returns)
        self.assertIsNotNone(result)
        h = result.hurst_exponent
        expected = (
            "trending" if h > 0.55 else "mean_reverting" if h < 0.45 else "random_walk"
        )
        self.assertEqual(result.label, expected)
        self.assertGreaterEqual(result.strength, 0.0)
        self.assertLessEqual(result.strength, 1.0)

    def test_hmm_label_follows_state_drift_not_index(self):
        result = eos_bridge.hmm_regime(returns=self.returns)
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "risk_on" if result.state_mean >= 0 else "risk_off")
        self.assertGreaterEqual(result.persistence, 0.0)
        self.assertLessEqual(result.persistence, 1.0)

    def test_var_is_positive_and_shortfall_exceeds_it(self):
        result = eos_bridge.historical_var(returns=self.returns, confidence=0.95)
        self.assertIsNotNone(result)
        self.assertGreater(result.value_at_risk, 0.0)
        if result.expected_shortfall is not None:
            self.assertGreaterEqual(result.expected_shortfall, result.value_at_risk)

    def test_quantum_price_levels_bracket_spot(self):
        spot = self.prices[-1]
        result = eos_bridge.quantum_price_levels_for(spot_price=spot, returns=self.returns)
        self.assertIsNotNone(result, "QPL produced no bound states")
        self.assertTrue(result.support, "expected at least one support level")
        self.assertTrue(result.resistance, "expected at least one resistance level")
        for level in result.support:
            self.assertLess(level, spot)
        for level in result.resistance:
            self.assertGreater(level, spot)
        # Nearest-first ordering is what a stop-placement caller relies on.
        self.assertEqual(result.nearest_support, max(result.support))
        self.assertEqual(result.nearest_resistance, min(result.resistance))

    def test_quantum_levels_scale_with_volatility(self):
        # Doubling the return volatility must push the levels further from spot.
        spot = 100.0
        quiet = [r * 0.5 for r in self.returns]
        loud = [r * 2.0 for r in self.returns]
        near = eos_bridge.quantum_price_levels_for(spot_price=spot, returns=quiet)
        far = eos_bridge.quantum_price_levels_for(spot_price=spot, returns=loud)
        self.assertIsNotNone(near)
        self.assertIsNotNone(far)
        self.assertGreater(far.nearest_resistance, near.nearest_resistance)

    def test_build_enrichment_is_json_serializable(self):
        import json
        enrichment = eos_bridge.build_enrichment(
            price_history=self.prices, spot_price=self.prices[-1]
        )
        json.dumps(enrichment)  # must not raise
        self.assertTrue(enrichment["available"])
        self.assertIn("garch", enrichment)

    def test_build_enrichment_omits_disabled_helpers(self):
        enrichment = eos_bridge.build_enrichment(
            price_history=self.prices,
            spot_price=self.prices[-1],
            enable_garch=False,
            enable_qpl=False,
        )
        self.assertNotIn("garch", enrichment)
        self.assertNotIn("quantum_price_levels", enrichment)
        self.assertIn("hurst", enrichment)


class TestEnrichmentStride(unittest.TestCase):
    """Stride caching must cut cost without ever fabricating a fresh fit."""

    def setUp(self):
        if not eos_bridge.available():
            self.skipTest("eos unavailable")
        eos_bridge.reset_enrichment_cache()
        self.prices = synthetic_prices()

    def tearDown(self):
        eos_bridge.reset_enrichment_cache()

    def _enrich(self, key="SYM", stride=1):
        return eos_bridge.build_enrichment(
            price_history=self.prices,
            spot_price=self.prices[-1],
            cache_key=key,
            stride=stride,
        )

    def test_stride_one_never_marks_stale(self):
        for _ in range(3):
            self.assertFalse(self._enrich(stride=1)["stale"])

    def test_stride_reuses_between_refits(self):
        flags = [self._enrich(stride=3)["stale"] for _ in range(7)]
        # Refit on calls 0, 3, 6; reuse in between.
        self.assertEqual(flags, [False, True, True, False, True, True, False])

    def test_reused_result_matches_the_fit_it_came_from(self):
        fresh = self._enrich(stride=4)
        reused = self._enrich(stride=4)
        self.assertTrue(reused["stale"])
        self.assertEqual(fresh["garch"], reused["garch"])
        self.assertEqual(fresh["hurst"], reused["hurst"])

    def test_cache_is_keyed_per_symbol(self):
        a = self._enrich(key="AAA", stride=5)
        b = self._enrich(key="BBB", stride=5)
        # A different symbol must trigger its own fit, not inherit AAA's.
        self.assertFalse(a["stale"])
        self.assertFalse(b["stale"])

    def test_reset_forces_a_fresh_fit(self):
        self._enrich(stride=5)
        self.assertTrue(self._enrich(stride=5)["stale"])
        eos_bridge.reset_enrichment_cache()
        self.assertFalse(self._enrich(stride=5)["stale"])


class TestExitScaling(unittest.TestCase):
    def test_no_enrichment_leaves_percentages_untouched(self):
        for enrichment in (None, {}, {"available": False}, {"garch": "not a dict"}):
            result = eos_bridge.scale_exit_pcts(0.04, 0.02, enrichment)
            self.assertFalse(result["applied"])
            self.assertEqual(result["tp_pct"], 0.04)
            self.assertEqual(result["sl_pct"], 0.02)

    def test_unconverged_garch_is_ignored(self):
        enrichment = {"garch": {"vol_ratio": 1.8, "converged": False}}
        result = eos_bridge.scale_exit_pcts(0.04, 0.02, enrichment)
        self.assertFalse(result["applied"])
        self.assertEqual(result["sl_pct"], 0.02)

    def test_scale_is_clamped_at_both_ends(self):
        high = eos_bridge.scale_exit_pcts(
            0.04, 0.02, {"garch": {"vol_ratio": 99.0, "converged": True}}, max_scale=2.0
        )
        self.assertEqual(high["scale"], 2.0)
        self.assertAlmostEqual(high["sl_pct"], 0.04)

        low = eos_bridge.scale_exit_pcts(
            0.04, 0.02, {"garch": {"vol_ratio": 0.001, "converged": True}}, min_scale=0.5
        )
        self.assertEqual(low["scale"], 0.5)
        self.assertAlmostEqual(low["sl_pct"], 0.01)

    def test_nonfinite_ratio_is_rejected(self):
        for bad in (float("nan"), float("inf"), -1.0, None, "x"):
            result = eos_bridge.scale_exit_pcts(
                0.04, 0.02, {"garch": {"vol_ratio": bad, "converged": True}}
            )
            self.assertFalse(result["applied"], f"accepted bad ratio {bad!r}")


class TestHurstMultiplier(unittest.TestCase):
    def test_no_hurst_result_is_neutral(self):
        result = eos_bridge.hurst_confidence_multiplier("buy", "BUY", {})
        self.assertEqual(result["multiplier"], 1.0)
        self.assertFalse(result["applied"])

    def test_random_walk_is_neutral(self):
        enrichment = {"hurst": {"label": "random_walk", "hurst_exponent": 0.5}}
        result = eos_bridge.hurst_confidence_multiplier("buy", "BUY", enrichment)
        self.assertEqual(result["multiplier"], 1.0)
        self.assertFalse(result["applied"])

    def test_neutral_bollinger_is_not_adjusted(self):
        enrichment = {"hurst": {"label": "mean_reverting", "hurst_exponent": 0.3}}
        result = eos_bridge.hurst_confidence_multiplier("buy", "NEUTRAL", enrichment)
        self.assertFalse(result["applied"])

    def test_mean_reverting_boosts_and_trending_penalizes(self):
        mr = eos_bridge.hurst_confidence_multiplier(
            "buy", "BUY", {"hurst": {"label": "mean_reverting", "hurst_exponent": 0.3}}
        )
        self.assertGreater(mr["multiplier"], 1.0)

        tr = eos_bridge.hurst_confidence_multiplier(
            "buy", "BUY", {"hurst": {"label": "trending", "hurst_exponent": 0.7}}
        )
        self.assertLess(tr["multiplier"], 1.0)


class TestOrderPlanWiring(unittest.TestCase):
    """The wiring must be inert unless eos_mode == 'enforce' with the flag on."""

    def _config(self, **overrides) -> ExecutionConfig:
        params = dict(
            execution_mode="simulation",
            dry_run=True,
            tp_pct=0.04,
            sl_pct=0.02,
            base_risk_pct=2.0,
            max_position_pct_of_equity=20.0,
        )
        params.update(overrides)
        return ExecutionConfig(**params)

    def _signal(self, vol_ratio: float = 2.0) -> Signal:
        return Signal(
            symbol="TEST",
            signal_type="buy",
            confidence=0.7,
            prob_profit=0.6,
            meta={"eos": {"garch": {"vol_ratio": vol_ratio, "converged": True}}},
        )

    def _flat(self) -> PositionState:
        return PositionState(
            symbol="TEST", quantity=0, avg_entry_price=0.0, side="flat", source="sim"
        )

    def test_shadow_mode_does_not_move_exit_prices(self):
        plan = build_order_plan(
            signal=self._signal(),
            position_state=self._flat(),
            config=self._config(eos_mode="shadow", eos_use_garch_exits=True),
            account_cash=10000.0,
            current_price=100.0,
        )
        # Shadow mode must leave the configured 4%/2% exactly in place.
        self.assertAlmostEqual(plan.tp_price, 104.00, places=2)
        self.assertAlmostEqual(plan.sl_price, 98.00, places=2)

    def test_enforce_without_flag_does_not_move_exit_prices(self):
        plan = build_order_plan(
            signal=self._signal(),
            position_state=self._flat(),
            config=self._config(eos_mode="enforce", eos_use_garch_exits=False),
            account_cash=10000.0,
            current_price=100.0,
        )
        self.assertAlmostEqual(plan.tp_price, 104.00, places=2)
        self.assertAlmostEqual(plan.sl_price, 98.00, places=2)

    def test_enforce_with_flag_widens_exits_in_high_vol(self):
        signal = self._signal(vol_ratio=2.0)
        plan = build_order_plan(
            signal=signal,
            position_state=self._flat(),
            config=self._config(eos_mode="enforce", eos_use_garch_exits=True),
            account_cash=10000.0,
            current_price=100.0,
        )
        # vol_ratio 2.0 doubles both legs: 4% -> 8%, 2% -> 4%.
        self.assertAlmostEqual(plan.tp_price, 108.00, places=2)
        self.assertAlmostEqual(plan.sl_price, 96.00, places=2)
        scaling = signal.meta["eos_exit_scaling"]
        self.assertTrue(scaling["enforced"])
        self.assertAlmostEqual(scaling["scale"], 2.0)

    def test_enforce_with_flag_tightens_exits_in_low_vol(self):
        plan = build_order_plan(
            signal=self._signal(vol_ratio=0.5),
            position_state=self._flat(),
            config=self._config(eos_mode="enforce", eos_use_garch_exits=True),
            account_cash=10000.0,
            current_price=100.0,
        )
        self.assertAlmostEqual(plan.tp_price, 102.00, places=2)
        self.assertAlmostEqual(plan.sl_price, 99.00, places=2)

    def test_signal_without_eos_meta_still_plans(self):
        signal = Signal(
            symbol="TEST", signal_type="buy", confidence=0.7, prob_profit=0.6, meta={}
        )
        plan = build_order_plan(
            signal=signal,
            position_state=self._flat(),
            config=self._config(eos_mode="enforce", eos_use_garch_exits=True),
            account_cash=10000.0,
            current_price=100.0,
        )
        self.assertAlmostEqual(plan.tp_price, 104.00, places=2)
        self.assertAlmostEqual(plan.sl_price, 98.00, places=2)
        self.assertFalse(signal.meta["eos_exit_scaling"]["enforced"])


class TestGracefulDegradation(unittest.TestCase):
    def test_helpers_return_none_when_unavailable(self):
        original = eos_bridge.EOS_AVAILABLE
        try:
            eos_bridge.EOS_AVAILABLE = False
            self.assertIsNone(eos_bridge.garch_volatility(returns=[0.01] * 200))
            self.assertIsNone(eos_bridge.hurst_regime(returns=[0.01] * 200))
            self.assertIsNone(eos_bridge.hmm_regime(returns=[0.01] * 200))
            self.assertIsNone(eos_bridge.historical_var(returns=[0.01] * 200))
            self.assertIsNone(
                eos_bridge.quantum_price_levels_for(spot_price=100.0, returns=[0.01] * 200)
            )
            enrichment = eos_bridge.build_enrichment([100.0, 101.0], spot_price=101.0)
            self.assertFalse(enrichment["available"])
            with self.assertRaises(eos_bridge.EosUnavailable):
                eos_bridge.call("finance.ar1_garch11_fit", returns=[0.01] * 100)
        finally:
            eos_bridge.EOS_AVAILABLE = original

    def test_status_reports_root_and_count(self):
        snapshot = eos_bridge.status()
        self.assertIn("available", snapshot)
        self.assertIn("eos_root", snapshot)
        self.assertIn("algorithm_count", snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
