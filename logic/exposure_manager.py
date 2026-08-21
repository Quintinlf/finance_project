"""
Portfolio exposure trimming.

Root-cause context: the exposure guardrail in ``enforce_risk_limits`` is a
one-way valve — it blocks new BUYs while exposure sits at or above
``max_portfolio_exposure``, but nothing in the cycle ever brings exposure back
*down*. Combined with the default stop-only exit policy ("let winners run", see
logic/position_reconciler.py), a position that appreciates has no take-profit,
never exits, and silently consumes the entire exposure budget forever.

That is exactly what happened to this account: BAC opened 2026-03-24 at $47.38
and ran to ~$64. With BNO alongside it, exposure pinned at 36.6% against a 30%
cap, so every single BUY — including a UNG signal at 0.90 confidence and 0.98
P(up) — was rejected with "portfolio exposure limit reached" on every run from
August 14 onward. The account was structurally unable to trade.

This module closes the loop: when the book is over its exposure cap, trim the
largest holdings back under it so the strategy can act again. Shares reserved by
a resting protective SELL are not sellable, so any protective order on a trimmed
symbol is cancelled first; ``reconcile_position_exits`` re-attaches a fresh GTC
exit to the remaining shares later in the same cycle (hence trim must run
BEFORE reconciliation).

Trimming is deliberately conservative:
  * It only ever runs when exposure is already **over** the cap.
  * It sells the minimum whole share count that gets back under target.
  * It never opens a position, never shorts, and never touches cash.
  * In dry-run mode nothing is submitted; intended actions are returned.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional


# Trim to this many percentage points *below* the cap rather than exactly to it.
# Landing precisely on the cap leaves zero headroom, so the very next BUY is
# rejected again and the trim bought nothing. A small buffer means one trim
# actually restores the ability to enter a position.
DEFAULT_RELEASE_BUFFER = 0.05


@dataclass
class TrimResult:
    symbol: str
    action: str  # 'trimmed' | 'dry_run' | 'skipped' | 'error'
    qty: int
    detail: str
    order_id: Optional[str] = None


def _enum_str(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _cancel_resting_sells(trading_client: Any, symbol: str) -> int:
    """Cancel open SELL orders on ``symbol`` so its shares become sellable.

    Alpaca reserves shares backing a resting protective sell; without this a
    trim fails with "insufficient qty available for order".
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    cancelled = 0
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        orders = trading_client.get_orders(filter=req)
    except Exception as exc:
        logging.warning("Trim: could not list open orders for %s (%s).", symbol, exc)
        return 0

    for order in orders or []:
        if _enum_str(getattr(order, "side", None)) != "sell":
            continue
        try:
            trading_client.cancel_order_by_id(str(getattr(order, "id", "") or ""))
            cancelled += 1
        except Exception as exc:
            logging.warning("Trim: could not cancel order on %s (%s).", symbol, exc)
    return cancelled


def _positions_by_value(trading_client: Any) -> List[Any]:
    """Long positions, richest first — the cheapest way back under the cap."""
    try:
        positions = trading_client.get_all_positions() or []
    except Exception as exc:
        logging.warning("Trim: could not fetch positions (%s).", exc)
        return []

    longs = [p for p in positions if float(getattr(p, "qty", 0) or 0) > 0]
    return sorted(
        longs,
        key=lambda p: abs(float(getattr(p, "market_value", 0) or 0)),
        reverse=True,
    )


def trim_to_exposure_cap(
    *,
    broker_client: Any,
    equity: float,
    exposure: float,
    max_exposure: float,
    release_buffer: float = DEFAULT_RELEASE_BUFFER,
    dry_run: bool = True,
    verbose: bool = True,
) -> List[TrimResult]:
    """Sell down over-cap holdings until portfolio exposure is back under target.

    Returns one TrimResult per position acted on (empty when already compliant).
    Never raises: a broker failure is reported as an 'error' result so the daily
    cycle continues.
    """
    results: List[TrimResult] = []

    if equity <= 0 or exposure <= max_exposure:
        return results

    trading_client = getattr(broker_client, "_trading_client", None)
    if trading_client is None:
        # In-memory simulation broker has no resting orders or live book to trim.
        return results

    target_exposure = max(0.0, max_exposure - release_buffer)
    excess_dollars = (exposure - target_exposure) * float(equity)

    logging.info(
        "EXPOSURE TRIM: %.1f%% exposed against a %.0f%% cap — freeing $%.2f to reach %.0f%%",
        exposure * 100.0,
        max_exposure * 100.0,
        excess_dollars,
        target_exposure * 100.0,
    )

    for pos in _positions_by_value(trading_client):
        if excess_dollars <= 0:
            break

        symbol = str(getattr(pos, "symbol", "") or "")
        held_qty = int(float(getattr(pos, "qty", 0) or 0))
        price = float(getattr(pos, "current_price", 0) or 0)

        if held_qty < 1 or price <= 0:
            results.append(
                TrimResult(symbol, "skipped", 0, f"no usable qty/price (qty={held_qty}, price={price})")
            )
            continue

        # Round *up*: selling one share short of the target leaves the book over
        # the cap and the whole trim accomplishes nothing.
        want_qty = min(held_qty, math.ceil(excess_dollars / price))
        if want_qty < 1:
            continue

        freed = want_qty * price
        detail = (
            f"sell {want_qty}/{held_qty} sh @ ${price:.2f} frees ${freed:.2f}"
        )

        if dry_run:
            results.append(TrimResult(symbol, "dry_run", want_qty, f"would {detail}"))
            excess_dollars -= freed
            continue

        cancelled = _cancel_resting_sells(trading_client, symbol)
        order = broker_client.place_market_order(
            symbol=symbol, qty=want_qty, side="sell", time_in_force="day"
        )

        if order is None:
            err = getattr(broker_client, "last_order_error", None) or "unknown rejection"
            results.append(TrimResult(symbol, "error", 0, f"trim sell rejected: {err}"))
            continue

        note = detail
        if cancelled:
            note += f" (cancelled {cancelled} resting sell order(s) to free shares)"
        results.append(TrimResult(symbol, "trimmed", want_qty, note, str(getattr(order, "id", "") or "")))
        excess_dollars -= freed

    if excess_dollars > 0 and results:
        logging.warning(
            "EXPOSURE TRIM: still $%.2f over target after trimming every eligible position.",
            excess_dollars,
        )

    if verbose:
        for r in results:
            logging.info("EXPOSURE TRIM | %s | %s | %s", r.symbol, r.action.upper(), r.detail)

    return results
