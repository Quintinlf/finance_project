"""Route bearish signals into long positions in inverse funds.

The book is long-only: shorting is disabled in `ExecutionConfig` and the Alpaca
account itself reports `shorting_enabled=False`. So `decide_action` drops every
SELL on a symbol we do not already hold, which on a bearish day means the bot
does nothing at all. On 2026-08-11 that was three of three directional signals.

This module translates those signals instead of discarding them:

    SELL <underlying>, flat      ->  BUY  <inverse proxy>
    BUY  <underlying>, hold proxy ->  SELL <inverse proxy>   (close the bearish bet)
    SELL <underlying>, hold it    ->  unchanged (a normal long exit)

The second rule matters as much as the first. Without it, a proxy bought on a
bearish signal could only ever be exited by its stop, because a later bullish
signal on the underlying names a different ticker than the position we hold.

What this does NOT do is pretend an inverse fund is a mirror image. Almost every
listed inverse commodity product is -2x and resets daily, so:

  * exit percentages are divided by the leverage factor (see
    `universe.scale_exits_for_leverage`), otherwise a 2% stop on a -2x fund
    fires on a 1% underlying move — half the move the signal asked for;
  * `prob_profit` is inverted, because the routed signal is a bet on the proxy
    going up, which is the underlying going down;
  * symbols with no listed inverse (all of agriculture, platinum, palladium,
    industrial and broad-basket metals) are reported as unroutable rather than
    silently dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from logic.data_structures import Signal
from logic.universe import InverseProxy, resolve_inverse, inverse_for

logger = logging.getLogger(__name__)


@dataclass
class RoutingOutcome:
    """What happened to one signal, for the run log and the audit trail."""

    original_symbol: str
    original_signal: str
    action: str               # 'routed_entry' | 'routed_exit' | 'unchanged' | 'unroutable'
    routed_symbol: Optional[str] = None
    leverage: Optional[float] = None
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_symbol": self.original_symbol,
            "original_signal": self.original_signal,
            "action": self.action,
            "routed_symbol": self.routed_symbol,
            "leverage": self.leverage,
            "reason": self.reason,
        }


@dataclass
class RoutingResult:
    signals: List[Signal] = field(default_factory=list)
    outcomes: List[RoutingOutcome] = field(default_factory=list)

    @property
    def extra_symbols(self) -> List[str]:
        """Proxy tickers introduced by routing, for position-state lookup."""
        seen: List[str] = []
        for outcome in self.outcomes:
            if outcome.routed_symbol and outcome.routed_symbol not in seen:
                seen.append(outcome.routed_symbol)
        return seen

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.action] = counts.get(outcome.action, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing to route"


def _held_quantity(position_states: Mapping[str, Any], symbol: str) -> float:
    state = position_states.get(symbol)
    if state is None:
        return 0.0
    try:
        return float(getattr(state, "quantity", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_routed_signal(
    original: Signal,
    proxy: InverseProxy,
    side: str,
    reason: str,
    proxy_price: float,
) -> Signal:
    """Clone a signal onto the proxy ticker, inverting the directional read."""
    # The routed signal bets on the proxy rising, which is the underlying
    # falling — so P(proxy up) is P(underlying down) = 1 - prob_profit.
    inverted_prob = max(0.0, min(1.0, 1.0 - float(original.prob_profit)))
    prob_profit = inverted_prob if side == "buy" else float(original.prob_profit)

    meta = dict(original.meta)
    # Re-price onto the proxy. `build_order_plan` derives both position size
    # and the bracket levels from meta['current_price'], so carrying the
    # underlying's price here produces exit prices for the wrong instrument.
    # UNG at ~$10 routed to KOLD at ~$29.81 yielded a $10.20 take-profit and
    # a broker rejection: "take_profit.limit_price must be >= base_price".
    meta["underlying_price"] = original.meta.get("current_price")
    meta["current_price"] = float(proxy_price)
    meta["inverse_routing"] = {
        "routed_from": original.symbol,
        "original_signal_type": original.signal_type,
        "proxy_symbol": proxy.symbol,
        "proxy_name": proxy.name,
        "leverage": proxy.leverage,
        "instrument": proxy.instrument,
        "note": proxy.note,
        "reason": reason,
    }
    # Consumed by execution_engine.resolve_exit_pcts: a -2x fund travels twice
    # as far, so the exit bands must be divided by the leverage factor to keep
    # the underlying move that triggers them equal to what the signal intended.
    meta["exit_leverage_divisor"] = proxy.leverage
    # These describe the underlying's market state, not the proxy's. Kept for
    # continuity of the audit trail, flagged so nothing reads them as native.
    meta["market_state_refers_to"] = original.symbol

    return Signal(
        symbol=proxy.symbol,
        signal_type=side,  # type: ignore[arg-type]
        confidence=float(original.confidence),
        prob_profit=prob_profit,
        type_beliefs=dict(original.type_beliefs),
        meta=meta,
    )


def route_signals(
    signals: Sequence[Signal],
    position_states: Mapping[str, Any],
    *,
    price_lookup: Optional[Callable[[Sequence[str]], Dict[str, Optional[float]]]] = None,
    verbose: bool = True,
) -> RoutingResult:
    """Translate unexecutable bearish signals into tradable long proxy positions.

    `position_states` should cover both the universe and any proxy tickers
    already held, so the exit rule can see them. HOLD signals pass through
    untouched.
    """
    result = RoutingResult()

    for signal in signals:
        symbol = signal.symbol
        signal_type = str(signal.signal_type).lower()

        if signal_type == "hold":
            result.signals.append(signal)
            continue

        held_underlying = _held_quantity(position_states, symbol)

        # --- SELL on something we own: an ordinary long exit, leave it alone.
        if signal_type == "sell" and held_underlying > 0:
            result.signals.append(signal)
            result.outcomes.append(RoutingOutcome(
                symbol, signal_type, "unchanged",
                reason=f"holding {held_underlying:g} shares; normal long exit",
            ))
            continue

        proxy_hint = inverse_for(symbol)
        held_proxy = (
            _held_quantity(position_states, proxy_hint.symbol) if proxy_hint else 0.0
        )

        # --- BUY on the underlying while holding its inverse: close the bet.
        if signal_type == "buy" and proxy_hint is not None and held_proxy > 0:
            proxy, detail, proxy_price = resolve_inverse(symbol, price_lookup=price_lookup)
            if proxy is None or proxy_price is None:
                # Cannot price the proxy, so cannot build a valid exit order.
                # Say so rather than emitting one at the underlying's price.
                result.outcomes.append(RoutingOutcome(
                    symbol, signal_type, "unroutable",
                    reason=f"holding {proxy_hint.symbol} but cannot price it: {detail}",
                ))
                if verbose:
                    logger.warning(
                        "INVERSE ROUTE (exit blocked) | holding %s but cannot price it: %s",
                        proxy_hint.symbol, detail,
                    )
                continue
            routed = _build_routed_signal(
                signal, proxy, "sell",
                reason=f"bullish on {symbol} while holding {proxy.symbol}; closing the bearish position",
                proxy_price=proxy_price,
            )
            result.signals.append(routed)
            result.outcomes.append(RoutingOutcome(
                symbol, signal_type, "routed_exit",
                routed_symbol=proxy.symbol, leverage=proxy.leverage,
                reason=f"exiting {held_proxy:g} shares of {proxy.symbol}",
            ))
            if verbose:
                logger.info(
                    "INVERSE ROUTE (exit) | BUY %s -> SELL %s @ $%.2f (closing bearish position)",
                    symbol, proxy.symbol, proxy_price,
                )
            continue

        # --- BUY on the underlying, flat: nothing to translate.
        if signal_type == "buy":
            result.signals.append(signal)
            continue

        # --- SELL while flat: the case that used to be dropped entirely.
        proxy, detail, proxy_price = resolve_inverse(symbol, price_lookup=price_lookup)
        if proxy is None or proxy_price is None:
            result.outcomes.append(RoutingOutcome(
                symbol, signal_type, "unroutable", reason=detail,
            ))
            if verbose:
                logger.info("INVERSE ROUTE (skipped) | SELL %s not actionable: %s", symbol, detail)
            continue

        routed = _build_routed_signal(
            signal, proxy, "buy",
            reason=f"bearish on {symbol} with no short capability; expressing via {proxy.symbol}",
            proxy_price=proxy_price,
        )
        result.signals.append(routed)
        result.outcomes.append(RoutingOutcome(
            symbol, signal_type, "routed_entry",
            routed_symbol=proxy.symbol, leverage=proxy.leverage, reason=detail,
        ))
        if verbose:
            logger.info(
                "INVERSE ROUTE (entry) | SELL %s -> BUY %s | -%gx | %s",
                symbol, proxy.symbol, proxy.leverage, detail,
            )

    return result


def format_routing_table(outcomes: Sequence[RoutingOutcome]) -> str:
    """Readable summary of routing decisions for the daily run log."""
    if not outcomes:
        return "  (no directional signals to route)"
    lines = [f"  {'FROM':<7} {'SIGNAL':<6} {'ACTION':<14} {'TO':<7} REASON"]
    for outcome in outcomes:
        lines.append(
            f"  {outcome.original_symbol:<7} {outcome.original_signal.upper():<6} "
            f"{outcome.action:<14} {(outcome.routed_symbol or '-'):<7} {outcome.reason}"
        )
    return "\n".join(lines)
