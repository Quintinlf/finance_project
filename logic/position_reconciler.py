"""
Position exit reconciliation.

Root-cause context: on 2026-03-24 the bot opened BAC/TSLA with *day-TIF*
bracket exit legs. Those legs expired unfilled the same afternoon and were
never resubmitted, leaving the positions with no stop-loss / take-profit for
months. Because those two positions consumed ~95% of a tiny (~$527) account,
the portfolio-exposure guardrail then rejected every new buy across the whole
universe — 4 months of silent no-trades.

This module closes that gap: every cycle, any open long position that lacks an
active protective SELL order gets a fresh **GTC** exit re-attached. GTC (not
day) is the fix — the exit persists across sessions instead of expiring
overnight. Levels are anchored to the position's *current* price, not its
(possibly stale) average entry, so a position that has already run up is
protected at today's value rather than being dumped at a long-ago entry level.

Default policy is a **GTC trailing stop** (`exit_style="trailing"`). The
earlier fixed-stop default protected the downside but never harvested a winner,
so an appreciating position held its market value — and therefore its share of
the portfolio exposure budget — indefinitely. Pass `exit_style="stop"` for a
fixed stop, or `exit_style="oco"` for a full take-profit + stop-loss OCO.

Only long positions are protected for now (the system does not short). The
submission path is wrapped so a rejected exit is reported, never raised, and
never takes down the daily cycle. In dry-run mode nothing is submitted — the
intended actions are returned for logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExitReconciliationResult:
    symbol: str
    action: str  # 'attached' | 'already_protected' | 'skipped' | 'dry_run' | 'error'
    detail: str
    order_id: Optional[str] = None


def _enum_str(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _open_orders_by_symbol(trading_client: Any) -> Dict[str, List[Any]]:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    orders = trading_client.get_orders(filter=req)
    by_symbol: Dict[str, List[Any]] = {}
    for order in orders or []:
        symbol = str(getattr(order, "symbol", "") or "")
        by_symbol.setdefault(symbol, []).append(order)
    return by_symbol


def _has_protective_sell(orders: List[Any]) -> bool:
    """True if an open SELL order already protects the position (limit/stop/OCO/bracket)."""
    for order in orders:
        if _enum_str(getattr(order, "side", None)) != "sell":
            continue
        order_type = _enum_str(getattr(order, "type", None))
        order_class = _enum_str(getattr(order, "order_class", None))
        # "trailing_stop" must be in this set: without it a trailing exit is not
        # recognised as protection and a fresh one is stacked on every cycle,
        # eventually over-committing the position's shares.
        if order_type in {"limit", "stop", "stop_limit", "trailing_stop"} or order_class in {"oco", "bracket", "oto"}:
            return True
    return False


def _submit_protective_stop(
    trading_client: Any,
    *,
    symbol: str,
    qty: int,
    sl_price: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Attach a GTC stop-loss SELL. Downside protection without capping upside."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import StopOrderRequest

    try:
        stop_req = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=sl_price,
        )
        order = trading_client.submit_order(stop_req)
        return str(getattr(order, "id", "") or ""), None
    except Exception as exc:
        return None, f"GTC stop submission failed ({exc})"


def _submit_trailing_stop(
    trading_client: Any,
    *,
    symbol: str,
    qty: int,
    trail_pct: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Attach a GTC trailing stop SELL.

    A fixed stop and a "let winners run" policy combine badly: the stop sits at
    its original level while the position appreciates, so the winner is never
    harvested and its (growing) market value permanently occupies the portfolio
    exposure budget. A trailing stop keeps the downside protection, follows the
    position up, and does eventually exit — which is what frees the budget back
    up for the next signal.
    """
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import TrailingStopOrderRequest

    try:
        req = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            trail_percent=round(float(trail_pct) * 100.0, 2),
        )
        order = trading_client.submit_order(req)
        return str(getattr(order, "id", "") or ""), None
    except Exception as exc:
        return None, f"GTC trailing stop submission failed ({exc})"


def _submit_protective_oco(
    trading_client: Any,
    *,
    symbol: str,
    qty: int,
    tp_price: float,
    sl_price: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Attach a GTC OCO sell (TP + SL). Fall back to a GTC stop if OCO is rejected.

    Downside protection is the priority: if the broker rejects the OCO for any
    reason, a lone GTC stop-loss still caps the loss, which is strictly better
    than the current unprotected state.
    """
    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
    from alpaca.trading.requests import (
        LimitOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
    )

    try:
        oco_req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.OCO,
            limit_price=tp_price,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=sl_price),
        )
        order = trading_client.submit_order(oco_req)
        return str(getattr(order, "id", "") or ""), None
    except Exception as exc_oco:
        order_id, err = _submit_protective_stop(
            trading_client, symbol=symbol, qty=qty, sl_price=sl_price
        )
        if order_id is not None:
            return order_id, f"OCO rejected ({exc_oco}); attached GTC stop-only fallback"
        return None, f"OCO failed ({exc_oco}); stop fallback failed ({err})"


def reconcile_position_exits(
    *,
    broker_client: Any,
    tp_pct: float,
    sl_pct: float,
    dry_run: bool = True,
    verbose: bool = True,
    exit_style: str = "trailing",
) -> List[ExitReconciliationResult]:
    """Ensure every open long position has an active protective GTC exit order.

    exit_style:
        "trailing" (default) — attach a GTC trailing stop at ``sl_pct``. Protects
            the downside, follows a winner up, and still eventually exits so the
            position stops occupying the portfolio exposure budget forever.
        "stop" — attach a fixed GTC stop-loss only. Never harvests a winner; see
            logic/exposure_manager.py for why that deadlocked this account.
        "oco" — attach a GTC OCO (take-profit + stop-loss) trade plan.

    Operates directly on the underlying Alpaca trading client when available.
    For the in-memory PaperBrokerClient (simulation mode) there are no live
    resting orders to reconcile, so this is a no-op.
    """
    style = str(exit_style).lower()
    results: List[ExitReconciliationResult] = []

    trading_client = getattr(broker_client, "_trading_client", None)
    if trading_client is None:
        return results

    try:
        positions = trading_client.get_all_positions()
    except Exception as exc:
        logging.warning("Exit reconcile: could not fetch positions (%s).", exc)
        return results

    try:
        orders_by_symbol = _open_orders_by_symbol(trading_client)
    except Exception as exc:
        logging.warning("Exit reconcile: could not fetch open orders (%s).", exc)
        orders_by_symbol = {}

    for pos in positions or []:
        symbol = str(getattr(pos, "symbol", "") or "")
        qty = float(getattr(pos, "qty", 0) or 0)

        if qty <= 0:
            # Short positions are not managed here (system does not short).
            results.append(ExitReconciliationResult(symbol, "skipped", f"non-long position qty={qty}"))
            continue

        existing = orders_by_symbol.get(symbol, [])
        if _has_protective_sell(existing):
            results.append(
                ExitReconciliationResult(symbol, "already_protected", "open protective sell order exists")
            )
            continue

        # Anchor exit levels to CURRENT price, not stale average entry.
        anchor = float(getattr(pos, "current_price", 0) or 0) or float(getattr(pos, "avg_entry_price", 0) or 0)
        if anchor <= 0:
            results.append(ExitReconciliationResult(symbol, "skipped", "no usable price to anchor exits"))
            continue

        tp_price = round(anchor * (1 + tp_pct), 2)
        sl_price = round(anchor * (1 - sl_pct), 2)
        int_qty = int(qty)
        if style == "trailing":
            levels = f"trail {sl_pct:.1%}"
        elif style == "stop":
            levels = f"SL ${sl_price}"
        else:
            levels = f"TP ${tp_price} / SL ${sl_price}"

        if dry_run:
            results.append(
                ExitReconciliationResult(
                    symbol,
                    "dry_run",
                    f"would attach GTC {style} sell qty={int_qty} @ anchor ${anchor:.2f} ({levels})",
                )
            )
            continue

        if style == "oco":
            order_id, err = _submit_protective_oco(
                trading_client, symbol=symbol, qty=int_qty, tp_price=tp_price, sl_price=sl_price
            )
        elif style == "trailing":
            order_id, err = _submit_trailing_stop(
                trading_client, symbol=symbol, qty=int_qty, trail_pct=sl_pct
            )
            if order_id is None:
                # Downside protection beats no protection: fall back to a fixed stop.
                order_id, fallback_err = _submit_protective_stop(
                    trading_client, symbol=symbol, qty=int_qty, sl_price=sl_price
                )
                err = (
                    f"{err}; attached fixed GTC stop fallback"
                    if order_id is not None
                    else f"{err}; stop fallback failed ({fallback_err})"
                )
        else:
            order_id, err = _submit_protective_stop(
                trading_client, symbol=symbol, qty=int_qty, sl_price=sl_price
            )

        if order_id is None:
            results.append(ExitReconciliationResult(symbol, "error", err or "unknown exit submission error"))
        else:
            note = f"GTC {style} exit attached qty={int_qty} ({levels})"
            if err:
                note += f" [{err}]"
            results.append(ExitReconciliationResult(symbol, "attached", note, order_id))

    if verbose:
        for r in results:
            logging.info("EXIT RECONCILE | %s | %s | %s", r.symbol, r.action.upper(), r.detail)

    return results
