"""Tests for the Phase 0 thesis layer (types, attribution math, adapters)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from logic.data_structures import Signal
from logic.thesis import (
    AnalystContribution,
    Catalyst,
    CatalystCode,
    ConfidenceAttribution,
    MacroRegime,
    MacroRegimeCode,
    MarketDirection,
    MarketDirectionCode,
    StructuralTheme,
    StructuralThemeCode,
    TradeThesis,
    load_shadow_theses,
    shadow_log_theses,
    signal_to_thesis,
)
from logic.thesis.adapters import NEUTRAL_BASE_CONFIDENCE, TECHNICAL_ANALYST_ID


def _contribution(analyst_id: str, delta: float) -> AnalystContribution:
    return AnalystContribution(
        analyst_id=analyst_id, symbol="AAPL", confidence_delta=delta
    )


def _signal(
    *,
    signal_type: str = "buy",
    confidence: float = 0.72,
    prob_profit: float = 0.61,
    meta: dict | None = None,
) -> Signal:
    base_meta = {
        "rsi_value": 41.2,
        "bb_z_score": -1.35,
        "ensemble_forecast_return": 0.004,
        "expected_return_path": [0.004, 0.003, 0.002, 0.001, 0.001],
        "belief_entropy": 0.82,
        "signals_agree": True,
        "bb_signal": "oversold",
        # Persisted Signal.meta key holding a MarketState mixture. Deliberately
        # still named "market_regime" -- renaming it would break decision history.
        "market_regime": {
            "prob_trend": 0.61,
            "prob_range": 0.24,
            "prob_high_vol": 0.15,
        },
    }
    base_meta.update(meta or {})
    return Signal(
        symbol="AAPL",
        signal_type=signal_type,
        confidence=confidence,
        prob_profit=prob_profit,
        meta=base_meta,
    )


class TestConfidenceAttribution(unittest.TestCase):
    def test_deltas_reconcile_to_raw_final(self):
        attribution = ConfidenceAttribution.from_contributions(
            base=0.5,
            contributions=[_contribution("technical", 0.2), _contribution("macro", 0.05)],
        )
        self.assertAlmostEqual(attribution.raw_final, 0.75)
        self.assertAlmostEqual(attribution.final, 0.75)
        self.assertFalse(attribution.was_clipped)
        self.assertAlmostEqual(attribution.overflow, 0.0)
        self.assertAlmostEqual(
            attribution.base + sum(attribution.by_analyst.values()),
            attribution.raw_final,
        )

    def test_overflow_is_preserved_not_silently_dropped(self):
        """The bug this design exists to prevent: clipped confidence losing its deltas."""
        attribution = ConfidenceAttribution.from_contributions(
            base=0.5,
            contributions=[
                _contribution("technical", 0.30),
                _contribution("macro", 0.25),
                _contribution("news", 0.10),
            ],
        )
        self.assertAlmostEqual(attribution.raw_final, 1.15)
        self.assertAlmostEqual(attribution.final, 1.0)
        self.assertAlmostEqual(attribution.overflow, 0.15)
        self.assertTrue(attribution.was_clipped)
        # Invariant still holds against raw_final even though final was clipped.
        self.assertAlmostEqual(
            attribution.base + sum(attribution.by_analyst.values()),
            attribution.raw_final,
        )

    def test_negative_deltas_clip_at_zero(self):
        attribution = ConfidenceAttribution.from_contributions(
            base=0.5, contributions=[_contribution("news", -0.9)]
        )
        self.assertAlmostEqual(attribution.raw_final, -0.4)
        self.assertAlmostEqual(attribution.final, 0.0)
        self.assertAlmostEqual(attribution.overflow, -0.4)

    def test_repeated_analyst_ids_accumulate(self):
        attribution = ConfidenceAttribution.from_contributions(
            base=0.5,
            contributions=[_contribution("news", 0.1), _contribution("news", 0.05)],
        )
        self.assertAlmostEqual(attribution.by_analyst["news"], 0.15)
        self.assertAlmostEqual(attribution.raw_final, 0.65)

    def test_rejects_non_reconciling_construction(self):
        with self.assertRaises(ValueError):
            ConfidenceAttribution(
                base=0.5, by_analyst={"technical": 0.2}, raw_final=0.9, final=0.9
            )

    def test_rejects_final_that_is_not_clipped_raw(self):
        with self.assertRaises(ValueError):
            ConfidenceAttribution(
                base=0.5, by_analyst={"technical": 0.6}, raw_final=1.1, final=1.1
            )

    def test_round_trip(self):
        original = ConfidenceAttribution.from_contributions(
            base=0.5, contributions=[_contribution("technical", 0.7)]
        )
        restored = ConfidenceAttribution.from_dict(original.to_dict())
        self.assertEqual(restored.to_dict(), original.to_dict())
        self.assertAlmostEqual(restored.overflow, 0.2)


class TestContextTypes(unittest.TestCase):
    def test_regime_types_stay_distinct(self):
        """Four separate concepts must not collapse into one another."""
        market = MarketDirection(code=MarketDirectionCode.BULL)
        macro = MacroRegime(code=MacroRegimeCode.RATE_CUTS)
        theme = StructuralTheme(code=StructuralThemeCode.AI_BOOM)
        catalyst = Catalyst(code=CatalystCode.CPI, event_date=date(2026, 8, 12))

        self.assertNotEqual(type(market), type(macro))
        self.assertEqual(catalyst.to_dict()["event_date"], "2026-08-12")
        self.assertEqual(theme.to_dict()["code"], "AI_BOOM")

    def test_accepts_string_codes(self):
        self.assertEqual(MarketDirection(code="BEAR").code, MarketDirectionCode.BEAR)

    def test_rejects_unknown_code(self):
        with self.assertRaises(ValueError):
            MacroRegime(code="NOT_A_REAL_REGIME")

    def test_context_round_trips(self):
        catalyst = Catalyst(code=CatalystCode.EARNINGS, event_date=date(2026, 9, 1))
        self.assertEqual(Catalyst.from_dict(catalyst.to_dict()), catalyst)

    def test_market_state_bridge_needs_direction_for_trend(self):
        mixture = {"prob_trend": 0.7, "prob_range": 0.2, "prob_high_vol": 0.1}
        self.assertEqual(
            MarketDirection.from_market_state(mixture, directional_hint="buy").code,
            MarketDirectionCode.BULL,
        )
        self.assertEqual(
            MarketDirection.from_market_state(mixture, directional_hint="sell").code,
            MarketDirectionCode.BEAR,
        )
        # No hint: the mixture alone carries no direction, so do not invent one.
        self.assertEqual(
            MarketDirection.from_market_state(mixture).code, MarketDirectionCode.UNKNOWN
        )

    def test_market_state_bridge_maps_range_to_sideways(self):
        mixture = {"prob_trend": 0.2, "prob_range": 0.7, "prob_high_vol": 0.1}
        self.assertEqual(
            MarketDirection.from_market_state(mixture, directional_hint="buy").code,
            MarketDirectionCode.SIDEWAYS,
        )


class TestTradeThesis(unittest.TestCase):
    def _thesis(self, **overrides) -> TradeThesis:
        attribution = ConfidenceAttribution.from_contributions(
            base=0.5, contributions=[_contribution("technical", 0.22)]
        )
        params = dict(
            symbol="AAPL",
            direction="LONG",
            confidence=attribution.final,
            confidence_attribution=attribution,
            expected_return=0.011,
            expected_holding_period=5,
            time_horizon="SWING",
            conviction="MEDIUM",
            volatility_expectation="LOW",
        )
        params.update(overrides)
        return TradeThesis(**params)

    def test_confidence_must_match_attribution(self):
        attribution = ConfidenceAttribution.from_contributions(
            base=0.5, contributions=[_contribution("technical", 0.22)]
        )
        with self.assertRaises(ValueError):
            self._thesis(confidence=0.99, confidence_attribution=attribution)

    def test_rejects_invalid_direction_and_horizon(self):
        with self.assertRaises(ValueError):
            self._thesis(direction="SIDEWAYS")
        with self.assertRaises(ValueError):
            self._thesis(time_horizon="FOREVER")

    def test_normalizes_case(self):
        thesis = self._thesis(direction="long", conviction="medium")
        self.assertEqual(thesis.direction, "LONG")
        self.assertEqual(thesis.conviction, "MEDIUM")

    def test_json_round_trip_preserves_everything(self):
        original = self._thesis(
            market_direction=MarketDirection(code=MarketDirectionCode.BULL, confidence=0.61),
            macro_regime=MacroRegime(code=MacroRegimeCode.RATE_CUTS),
            structural_themes=[StructuralTheme(code=StructuralThemeCode.AI_BOOM)],
            catalysts=[Catalyst(code=CatalystCode.EARNINGS, event_date=date(2026, 8, 1))],
            supporting_evidence=["rsi=41.20"],
            analyst_ids=["technical"],
        )
        restored = TradeThesis.from_json(original.to_json())
        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_thesis_carries_no_instrument_fields(self):
        """Design commitment #2: instrument choice belongs to the Expression layer."""
        payload = self._thesis().to_dict()
        for forbidden in ("strike", "expiry", "instrument", "contract", "quantity", "side"):
            self.assertNotIn(forbidden, payload)


class TestSignalAdapter(unittest.TestCase):
    def test_confidence_is_preserved_exactly(self):
        """Phase 0 must explain the number, never move it."""
        for confidence in (0.0, 0.33, 0.5, 0.72, 1.0):
            signal = _signal(confidence=confidence)
            thesis = signal_to_thesis(signal)
            self.assertAlmostEqual(thesis.confidence, confidence)
            self.assertAlmostEqual(thesis.confidence_attribution.final, confidence)
            self.assertFalse(thesis.confidence_attribution.was_clipped)

    def test_attribution_reconciles_from_signal(self):
        thesis = signal_to_thesis(_signal(confidence=0.72))
        attribution = thesis.confidence_attribution
        self.assertAlmostEqual(attribution.base, NEUTRAL_BASE_CONFIDENCE)
        self.assertAlmostEqual(attribution.by_analyst[TECHNICAL_ANALYST_ID], 0.22)
        self.assertAlmostEqual(
            attribution.base + sum(attribution.by_analyst.values()),
            attribution.raw_final,
        )

    def test_direction_mapping(self):
        self.assertEqual(signal_to_thesis(_signal(signal_type="buy")).direction, "LONG")
        self.assertEqual(signal_to_thesis(_signal(signal_type="sell")).direction, "SHORT")
        self.assertEqual(signal_to_thesis(_signal(signal_type="hold")).direction, "FLAT")

    def test_expected_return_sums_forecast_path(self):
        thesis = signal_to_thesis(_signal())
        self.assertAlmostEqual(thesis.expected_return, 0.011)

    def test_falls_back_to_one_step_return_without_path(self):
        signal = _signal(meta={"expected_return_path": None})
        self.assertAlmostEqual(signal_to_thesis(signal).expected_return, 0.004)

    def test_unpopulated_context_is_empty_not_guessed(self):
        thesis = signal_to_thesis(_signal())
        self.assertIsNone(thesis.macro_regime)
        self.assertEqual(thesis.structural_themes, [])
        self.assertEqual(thesis.catalysts, [])

    def test_volatility_expectation_tracks_regime(self):
        calm = signal_to_thesis(_signal())
        self.assertEqual(calm.volatility_expectation, "LOW")

        stormy = signal_to_thesis(
            _signal(
                meta={
                    "market_regime": {
                        "prob_trend": 0.2,
                        "prob_range": 0.2,
                        "prob_high_vol": 0.6,
                    }
                }
            )
        )
        self.assertEqual(stormy.volatility_expectation, "HIGH")

    def test_survives_empty_meta(self):
        signal = Signal(symbol="MSFT", signal_type="buy", confidence=0.6, prob_profit=0.55)
        thesis = signal_to_thesis(signal)
        self.assertEqual(thesis.symbol, "MSFT")
        self.assertAlmostEqual(thesis.confidence, 0.6)
        self.assertAlmostEqual(thesis.expected_return, 0.0)


class TestShadowLogging(unittest.TestCase):
    def test_write_then_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "theses"
            written = shadow_log_theses(
                [_signal(), _signal(signal_type="sell", confidence=0.66)],
                log_dir=log_dir,
                run_date="2026-07-24",
            )
            self.assertEqual(written, 2)

            restored = load_shadow_theses(log_dir / "2026-07-24.jsonl")
            self.assertEqual(len(restored), 2)
            self.assertEqual(restored[0].direction, "LONG")
            self.assertEqual(restored[1].direction, "SHORT")

    def test_appends_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "theses"
            shadow_log_theses([_signal()], log_dir=log_dir, run_date="2026-07-24")
            shadow_log_theses([_signal()], log_dir=log_dir, run_date="2026-07-24")
            self.assertEqual(len(load_shadow_theses(log_dir / "2026-07-24.jsonl")), 2)

    def test_never_raises_on_bad_input(self):
        """A shadow-logging bug must not be able to stop a live trading day."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "theses"
            broken = object()  # not a Signal at all
            self.assertEqual(
                shadow_log_theses([broken], log_dir=log_dir, run_date="2026-07-24"), 0
            )

    def test_empty_input_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "theses"
            self.assertEqual(shadow_log_theses([], log_dir=log_dir), 0)
            self.assertFalse(log_dir.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
