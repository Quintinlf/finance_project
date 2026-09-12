"""Probability calibration: map a claimed P(up) to its empirical frequency.

Measured 2026-08-21 from this account's own scored history: when the ensemble
claimed >=90% confidence the market would rise, it actually rose 68% of the
time -- BELOW the 75.6% base rate for the same period. The model's most
confident calls were its worst ones. Nothing upstream of this module ever
checked that; `prob_profit` was consumed at face value by the signal filter,
the edge gate, and position sizing alike.

This fits an isotonic (monotonic, non-parametric) regression from claimed
`prob_profit` to realized up/down outcome, using every decision matched to a
scored next-day return. Isotonic rather than a parametric curve (Platt/sigmoid)
because the miscalibration here is not a simple shift -- it reverses direction
at the top end -- and isotonic regression is guaranteed monotonic but otherwise
free to bend around exactly that non-monotonic bias.

Refuses to calibrate below MIN_SAMPLE (30). A curve fit on a couple dozen points
is noise dressed up as correction, and passing raw probabilities through
unchanged is the honest default until there is enough history to do better.
The isotonic fit is stored as a small list of (x, y) breakpoints in JSON rather
than a pickled sklearn object, so it survives a scikit-learn version bump and
is human-readable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from logic.sqlite_store import DEFAULT_DB_PATH, connect

CALIBRATION_PATH = Path("trade_logs") / "calibration.json"

# Below this many matched (claim, outcome) pairs, isotonic regression is
# curve-fitting noise. Raw probabilities pass through unchanged instead.
#
# 30 was indefensibly low. Isotonic regression is non-parametric and will
# happily fit a step function through a few dozen noisy points: the curve fit
# on 249 live rows collapsed to a single step, mapping essentially every input
# to a constant 0.649. Applied to live signals that silently meant "approve
# every BUY, reject every SELL" -- a trading policy nobody chose, emerging from
# an artifact. Several hundred observations is the minimum for this to mean
# anything.
MIN_SAMPLE = 500

# A fitted curve whose output varies by less than this across its whole input
# range is constant in every way that matters: the model's confidence no longer
# influences the decision at all.
DEGENERATE_SPREAD = 0.02


@dataclass
class CalibrationCurve:
    fitted_at: str
    n: int
    x: List[float]  # claimed prob_profit breakpoints, ascending
    y: List[float]  # empirical P(up) at each breakpoint

    def apply(self, p: float) -> float:
        # Piecewise-linear interpolation through the isotonic breakpoints;
        # clipped to the observed range rather than extrapolated beyond it.
        if not self.x:
            return float(p)
        import numpy as np

        return float(np.clip(np.interp(p, self.x, self.y), 0.0, 1.0))

    def is_degenerate(self, tolerance: float = DEGENERATE_SPREAD) -> bool:
        """True when the curve maps effectively everything to one value.

        Measured over the *plausible* input range rather than the raw
        breakpoints: a lone breakpoint near 0 can make the nominal spread look
        large (0.0 to 0.649) while every realistic input still lands on the
        same constant.
        """
        probe = [i / 20.0 for i in range(1, 20)]
        outputs = [self.apply(p) for p in probe]
        return (max(outputs) - min(outputs)) < tolerance

    def to_dict(self) -> dict:
        return {"fitted_at": self.fitted_at, "n": self.n, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationCurve":
        return cls(fitted_at=d["fitted_at"], n=int(d["n"]), x=list(d["x"]), y=list(d["y"]))


def _matched_history(db_path: Union[str, Path]) -> List[tuple]:
    """(RAW claimed prob_profit, realized outcome) pairs from scored history.

    Trains on ``raw_prob_profit``, never ``prob_profit``. Once calibration is
    live, ``prob_profit`` holds the *calibrated* value, so fitting on it would
    feed the curve its own output: the fit flattens, that flatter value gets
    persisted tomorrow, and within days the curve collapses to a constant
    regardless of what the models actually said. COALESCE covers rows written
    before the raw column existed.

    Two sources, unioned:

    1. ``model_component_performance.raw_prob_profit`` -- the direct path, and
       the only one that covers backfilled history. Replayed predictions have
       no ``decisions`` row (nothing was traded), so a decisions-only query saw
       247 rows while ~12,000 scored predictions sat unused.
    2. The decisions join, for live rows written before that column existed.

    Joined on symbol + minute-truncated timestamp: decisions.timestamp and
    model_component_performance.timestamp are written from the same generation
    pass, so truncating to the minute lines them up without a shared surrogate
    key that does not exist yet.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT raw_prob_profit, next_day_return
            FROM model_component_performance
            WHERE next_day_return IS NOT NULL
              AND raw_prob_profit IS NOT NULL

            UNION ALL

            SELECT COALESCE(d.raw_prob_profit, d.prob_profit), m.next_day_return
            FROM decisions d
            JOIN model_component_performance m
              ON d.symbol = m.symbol
             AND substr(d.timestamp, 1, 16) = substr(m.timestamp, 1, 16)
            WHERE m.next_day_return IS NOT NULL
              AND d.prob_profit IS NOT NULL
              AND m.raw_prob_profit IS NULL
            """.strip()
        ).fetchall()
    return [(float(p), float(r)) for p, r in rows]


def fit_calibration(
    *,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    out_path: Path = CALIBRATION_PATH,
    min_sample: int = MIN_SAMPLE,
) -> Optional[CalibrationCurve]:
    """Refit the calibration curve from scored history and persist it.

    Returns None (and leaves any existing file untouched) when there are fewer
    than ``min_sample`` matched pairs -- there is nothing trustworthy to fit
    yet, and today's curve should not be silently deleted for a run that
    happens to have less data joined than yesterday's.
    """
    pairs = _matched_history(db_path)
    if len(pairs) < min_sample:
        logging.info(
            "CALIBRATION: %s matched prediction(s), need %s — passing raw probabilities through",
            len(pairs), min_sample,
        )
        return None

    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    p = np.array([x[0] for x in pairs])
    y = np.array([1.0 if x[1] > 0 else 0.0 for x in pairs])

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)

    curve = CalibrationCurve(
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n=len(pairs),
        x=[float(v) for v in iso.X_thresholds_],
        y=[float(v) for v in iso.y_thresholds_],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(curve.to_dict(), indent=2), encoding="utf-8")
    logging.info("CALIBRATION: refit from %s matched prediction(s), saved to %s", curve.n, out_path)

    if curve.is_degenerate():
        constant = curve.apply(0.5)
        logging.warning(
            "CALIBRATION COLLAPSED TO A CONSTANT (%.3f): the models' confidence no "
            "longer affects any decision. Downstream this means every BUY clears the "
            "directional threshold and every SELL fails it -- a trading policy that "
            "fell out of the fit rather than being chosen. Either the models carry no "
            "signal (consistent with the horizon analysis) or the sample is still too "
            "thin to fit.",
            constant,
        )
    return curve


def load_calibration(path: Path = CALIBRATION_PATH) -> Optional[CalibrationCurve]:
    if not path.exists():
        return None
    try:
        return CalibrationCurve.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not load calibration curve at %s (%s).", path, exc)
        return None


def apply_probability_calibration(
    signals: Sequence[Any],
    *,
    curve: Optional[CalibrationCurve] = None,
    path: Path = CALIBRATION_PATH,
    verbose: bool = True,
) -> Sequence[Any]:
    """Replace each signal's claimed prob_profit with its calibrated value.

    The raw claim is kept at ``meta['raw_prob_profit']`` for audit; every
    downstream consumer (the threshold filter, the edge gate, position sizing)
    reads ``signal.prob_profit`` and gets the corrected number without needing
    to know calibration exists. With no curve fitted yet, this is a no-op --
    signals pass through with raw_prob_profit == prob_profit.
    """
    if curve is None:
        curve = load_calibration(path)

    if curve is None:
        for signal in signals:
            if not hasattr(signal, "meta") or signal.meta is None:
                signal.meta = {}
            signal.meta["raw_prob_profit"] = float(getattr(signal, "prob_profit", 0.0) or 0.0)
            signal.meta["calibration_applied"] = False
        return signals

    if curve.is_degenerate():
        # NOT applied. A constant curve gives every symbol the same
        # prob_profit, which downstream means "every BUY clears the threshold,
        # every SELL fails it" -- a trading policy imposed by a fitting
        # artifact rather than chosen. Raw claims are also imperfect
        # (overconfident, per the measurements), but at least the logged reason
        # then reflects what the model actually said.
        #
        # This also matters because fit_calibration leaves a previously-fitted
        # file in place when the sample later falls below min_sample, so
        # without this check a bad curve would keep applying indefinitely.
        logging.warning(
            "CALIBRATION DEGENERATE (constant %.3f, n=%s) — NOT applied. It would "
            "make prob_profit identical for every symbol and silently reduce the "
            "threshold filter to 'approve every buy, reject every sell'. Passing "
            "raw probabilities through instead.",
            curve.apply(0.5), curve.n,
        )
        for signal in signals:
            if not hasattr(signal, "meta") or signal.meta is None:
                signal.meta = {}
            signal.meta["raw_prob_profit"] = float(getattr(signal, "prob_profit", 0.0) or 0.0)
            signal.meta["calibration_applied"] = False
            signal.meta["calibration_skipped_reason"] = "degenerate curve"
        return signals

    moved = 0
    for signal in signals:
        raw = float(getattr(signal, "prob_profit", 0.0) or 0.0)
        calibrated = curve.apply(raw)
        if not hasattr(signal, "meta") or signal.meta is None:
            signal.meta = {}
        signal.meta["raw_prob_profit"] = raw
        signal.meta["calibrated_prob_profit"] = calibrated
        signal.meta["calibration_applied"] = True
        signal.meta["calibration_n"] = curve.n
        signal.prob_profit = calibrated
        if abs(calibrated - raw) > 0.05:
            moved += 1

    if verbose and signals:
        logging.info(
            "CALIBRATION APPLIED: %s signal(s), fit on n=%s (fitted %s) — %s moved by >5pp",
            len(signals), curve.n, curve.fitted_at, moved,
        )

    return signals
