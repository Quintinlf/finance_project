"""Cost-aware edge gate.

The signal filter checks confidence and directional probability, but nothing
ever asked the question that decides whether a trade can make money: **is the
predicted move bigger than what it costs to trade it?**

The measured answer, from this account's own logs, is usually no. Mean absolute
ensemble forecast is 0.586% per day. Round-trip cost is 0.12% on XLE, 0.25% on
BNO, 0.45% on CANE and SOYB. So on the thin agricultural names the cost eats
three quarters of the entire predicted move before direction is even considered
-- and directional accuracy measured over 78 scored predictions is 46-47%,
below a coin flip.

Expected value per trade is therefore:

    EV = (2p - 1) * |forecast| - round_trip_cost

where p is the probability the move goes the signal's way. With p below 0.5 the
first term is *negative* and no cost level rescues it. This gate makes that
arithmetic explicit and refuses trades that cannot clear their own costs,
instead of discovering it later in the P&L.

This is a necessary condition, not a sufficient one. Passing the gate does not
mean a trade is good; failing it means the trade cannot be good.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from logic import costs


# Require the edge to clear costs by this multiple before trading. 1.0 would
# mean "break even in expectation", which is not worth the risk; demanding the
# move be worth meaningfully more than the friction is the whole point.
DEFAULT_EDGE_MARGIN = 1.5


@dataclass
class EdgeVerdict:
    symbol: str
    passes: bool
    expected_move_pct: float      # |forecast|, in percent
    directional_prob: float       # P(move goes the signal's way)
    cost_pct: float               # round-trip cost, in percent
    expected_value_pct: float     # EV after costs, in percent
    reason: str

    def as_meta(self) -> Dict[str, Any]:
        return {
            "edge_passes": self.passes,
            "edge_expected_move_pct": round(self.expected_move_pct, 4),
            "edge_directional_prob": round(self.directional_prob, 4),
            "edge_cost_pct": round(self.cost_pct, 4),
            "edge_expected_value_pct": round(self.expected_value_pct, 4),
            "edge_reason": self.reason,
        }


def evaluate_edge(
    signal: Any,
    *,
    margin: float = DEFAULT_EDGE_MARGIN,
    cost_model: Optional[costs.CostModel] = None,
) -> EdgeVerdict:
    """Score one signal's expected value net of the cost of trading it."""
    symbol = str(getattr(signal, "symbol", "") or "")
    meta = getattr(signal, "meta", None) or {}

    model = cost_model or costs.for_symbol(symbol)
    cost_pct = float(model.round_trip_bps) / 100.0

    # The forecaster's own point estimate of tomorrow's return.
    try:
        forecast = abs(float(meta.get("ensemble_forecast_return", 0.0) or 0.0))
    except (TypeError, ValueError):
        forecast = 0.0
    expected_move_pct = forecast * 100.0

    # Probability the move goes the way this signal is betting. The filter
    # stores this; fall back to prob_profit oriented by side.
    directional_prob = meta.get("directional_probability")
    if directional_prob is None:
        prob = float(getattr(signal, "prob_profit", 0.5) or 0.5)
        directional_prob = (1.0 - prob) if getattr(signal, "signal_type", "") == "sell" else prob
    directional_prob = float(directional_prob)

    ev_pct = (2.0 * directional_prob - 1.0) * expected_move_pct - cost_pct
    required = cost_pct * float(margin)

    if expected_move_pct <= 0:
        reason = "no forecast magnitude to trade on"
        passes = False
    elif directional_prob <= 0.5:
        reason = (
            f"directional probability {directional_prob:.1%} is a coin flip or worse — "
            f"no cost level makes this positive"
        )
        passes = False
    elif ev_pct <= 0:
        reason = (
            f"predicted move {expected_move_pct:.3f}% at {directional_prob:.1%} "
            f"does not cover {cost_pct:.3f}% round-trip cost (EV {ev_pct:+.3f}%)"
        )
        passes = False
    elif ev_pct < required:
        reason = (
            f"EV {ev_pct:+.3f}% clears costs but not the {margin:.1f}x margin "
            f"({required:.3f}% required)"
        )
        passes = False
    else:
        reason = (
            f"EV {ev_pct:+.3f}% on a {expected_move_pct:.3f}% move at "
            f"{directional_prob:.1%} vs {cost_pct:.3f}% cost"
        )
        passes = True

    return EdgeVerdict(
        symbol=symbol,
        passes=passes,
        expected_move_pct=expected_move_pct,
        directional_prob=directional_prob,
        cost_pct=cost_pct,
        expected_value_pct=ev_pct,
        reason=reason,
    )


def apply_edge_gate(
    signals: list,
    *,
    margin: float = DEFAULT_EDGE_MARGIN,
    enforce: bool = False,
    verbose: bool = True,
) -> list:
    """Annotate every signal with its cost-adjusted edge.

    With ``enforce=False`` (the default) this only records and logs the verdict,
    so the gate can be observed against live signals before it is allowed to
    change any behaviour. With ``enforce=True`` failing directional signals are
    demoted to HOLD.

    Shadow-first is deliberate: this gate would currently reject nearly every
    trade the strategy generates, and that conclusion deserves to be watched
    before it is wired to the order path.
    """
    kept = []
    rejected = 0

    for signal in signals:
        if getattr(signal, "signal_type", "") == "hold":
            kept.append(signal)
            continue

        verdict = evaluate_edge(signal, margin=margin)
        if not hasattr(signal, "meta") or signal.meta is None:
            signal.meta = {}
        signal.meta.update(verdict.as_meta())

        if verbose:
            logging.info(
                "EDGE GATE | symbol=%s | move=%.3f%% | p_dir=%.3f | cost=%.3f%% | "
                "EV=%+.3f%% | decision=%s | reason=%s",
                verdict.symbol,
                verdict.expected_move_pct,
                verdict.directional_prob,
                verdict.cost_pct,
                verdict.expected_value_pct,
                "PASS" if verdict.passes else ("REJECT" if enforce else "PASS (shadow)"),
                verdict.reason,
            )

        if enforce and not verdict.passes:
            signal.signal_type = "hold"
            signal.meta["threshold_decision"] = "hold"
            signal.meta["threshold_reason"] = f"edge gate: {verdict.reason}"
            rejected += 1

        kept.append(signal)

    if verbose and signals:
        logging.info(
            "EDGE GATE SUMMARY: %s directional signal(s) evaluated, %s demoted to HOLD (%s)",
            sum(1 for s in signals if getattr(s, "signal_type", "") != "hold") + rejected,
            rejected,
            "enforcing" if enforce else "shadow mode",
        )

    return kept
