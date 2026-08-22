"""Tests for probability calibration.

Measured 2026-08-21 from this account's own history: when the ensemble
claimed >=90% confidence the market would rise, it actually rose 68% of the
time -- below the 75.6% base rate for the same period. Confident calls were
the worst calls. Nothing upstream ever checked, so `prob_profit` fed the
threshold filter, the edge gate, and position sizing at face value.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from logic.calibration import (
    MIN_SAMPLE,
    CalibrationCurve,
    apply_probability_calibration,
    fit_calibration,
    load_calibration,
)
from logic.sqlite_store import init_db, connect


def _seed_db(db_path: Path, pairs):
    """Write (claimed_prob, realized_up) pairs into decisions + component tables."""
    init_db(db_path)
    from logic.model_performance_tracker import init_model_performance_tracker

    init_model_performance_tracker(db_path)
    with connect(db_path) as conn:
        for i, (p, up) in enumerate(pairs):
            ts = f"2026-08-{(i % 27) + 1:02d}T10:00:00+00:00"
            conn.execute(
                "INSERT INTO decisions (account_id, timestamp, symbol, prob_profit) "
                "VALUES (?, ?, ?, ?)",
                ("test", ts, f"SYM{i}", p),
            )
            ret = 0.01 if up else -0.01
            conn.execute(
                "INSERT INTO model_component_performance "
                "(decision_key, timestamp, symbol, action, price_at_signal, next_day_return) "
                "VALUES (?, ?, ?, 'buy', 100.0, ?)",
                (f"key{i}", ts, f"SYM{i}", ret),
            )


def _signal(prob_profit=0.7, side="buy"):
    return SimpleNamespace(symbol="UNG", signal_type=side, prob_profit=prob_profit, meta={})


class TestFitCalibration(unittest.TestCase):
    def test_below_min_sample_returns_none_and_writes_nothing(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            out = Path(d) / "calibration.json"
            _seed_db(db, [(0.9, True)] * 5)
            curve = fit_calibration(db_path=db, out_path=out, min_sample=MIN_SAMPLE)
            self.assertIsNone(curve)
            self.assertFalse(out.exists())

    def test_perfectly_calibrated_data_recovers_identity_ish_curve(self):
        """Claims that exactly match outcome frequency should map close to themselves."""
        pairs = [(0.9, True)] * 18 + [(0.9, False)] * 2 + [(0.1, True)] * 2 + [(0.1, False)] * 18
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            out = Path(d) / "calibration.json"
            _seed_db(db, pairs)
            curve = fit_calibration(db_path=db, out_path=out, min_sample=MIN_SAMPLE)
            self.assertIsNotNone(curve)
            self.assertEqual(curve.n, 40)
            self.assertGreater(curve.apply(0.9), 0.7)
            self.assertLess(curve.apply(0.1), 0.3)
            self.assertTrue(out.exists())

    def test_overconfident_claims_are_pulled_toward_the_true_frequency(self):
        """The real finding: claims of 90%+ that only came true ~68% of the time."""
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            out = Path(d) / "calibration.json"
            pairs = [(0.95, True)] * 34 + [(0.95, False)] * 16  # 68% true at claim=0.95
            _seed_db(db, pairs)
            curve = fit_calibration(db_path=db, out_path=out, min_sample=MIN_SAMPLE)
            self.assertIsNotNone(curve)
            calibrated = curve.apply(0.95)
            self.assertLess(calibrated, 0.95)
            self.assertAlmostEqual(calibrated, 0.68, delta=0.05)

    def test_monotonic_output_despite_noisy_input(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            out = Path(d) / "calibration.json"
            import random

            random.seed(0)
            pairs = [(p / 100.0, random.random() < (p / 100.0)) for p in range(1, 100)]
            _seed_db(db, pairs)
            curve = fit_calibration(db_path=db, out_path=out, min_sample=MIN_SAMPLE)
            self.assertIsNotNone(curve)
            outs = [curve.apply(x / 100.0) for x in range(1, 100)]
            self.assertTrue(all(a <= b + 1e-9 for a, b in zip(outs, outs[1:])))

    def test_persisted_curve_round_trips(self):
        with TemporaryDirectory() as d:
            db = Path(d) / "t.db"
            out = Path(d) / "calibration.json"
            pairs = [(0.9, True)] * 30 + [(0.1, False)] * 30
            _seed_db(db, pairs)
            fit_calibration(db_path=db, out_path=out, min_sample=MIN_SAMPLE)
            reloaded = load_calibration(out)
            self.assertIsNotNone(reloaded)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["n"], 60)


class TestApplyProbabilityCalibration(unittest.TestCase):
    def test_no_curve_leaves_prob_profit_unchanged_but_annotates(self):
        sigs = [_signal(0.85)]
        apply_probability_calibration(sigs, curve=None, path=Path("/nonexistent/x.json"), verbose=False)
        self.assertEqual(sigs[0].prob_profit, 0.85)
        self.assertEqual(sigs[0].meta["raw_prob_profit"], 0.85)
        self.assertFalse(sigs[0].meta["calibration_applied"])

    def test_curve_overrides_prob_profit_and_keeps_raw_for_audit(self):
        curve = CalibrationCurve(fitted_at="2026-08-21T00:00:00+00:00", n=78, x=[0.0, 1.0], y=[0.6, 0.6])
        sigs = [_signal(0.98)]
        apply_probability_calibration(sigs, curve=curve, verbose=False)
        self.assertAlmostEqual(sigs[0].prob_profit, 0.6, places=4)
        self.assertEqual(sigs[0].meta["raw_prob_profit"], 0.98)
        self.assertTrue(sigs[0].meta["calibration_applied"])

    def test_a_confident_sell_gets_correctly_downweighted(self):
        """The finding in practice: a claimed near-certain DOWN call, once the
        curve reflects a base-rate-dominated market, should stop looking like
        a near-certain trade."""
        # Curve says: regardless of the raw claim, empirical P(up) is ~0.75.
        curve = CalibrationCurve(fitted_at="x", n=78, x=[0.0, 1.0], y=[0.75, 0.75])
        sig = _signal(prob_profit=0.05, side="sell")  # claimed 95% confident DOWN
        apply_probability_calibration([sig], curve=curve, verbose=False)
        # prob_profit (P(up)) is now 0.75; a SELL's directional confidence is
        # 1 - prob_profit = 0.25, correctly humbled from the raw 0.95 claim.
        self.assertAlmostEqual(sig.prob_profit, 0.75, places=4)


if __name__ == "__main__":
    unittest.main()
