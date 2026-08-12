"""
Realized P&L reconciliation from broker fills.

The trades table has never held a real P&L. ``mark_trade_closed`` exists but is
called from nowhere, and ``upsert_broker_closed_trade`` hardcodes ``pnl = 0.0``
while setting ``entry_price = exit_price``, so anything it imports is zero by
construction. Exits happen broker-side via bracket orders and were never
reconciled back, which is why five months of trading shows no performance
record at all.

This module closes that loop: pull actual fills, FIFO-match them into round
trips, compute realized P&L, and persist it.

FIFO is the convention here because it matches how the broker reports cost
basis and how US tax lots default. It matters whenever a position was built in
more than one fill at different prices — averaging instead would quietly
misstate the P&L of a partial exit.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from logic.sqlite_store import DEFAULT_DB_PATH, connect, utc_now_iso


@dataclass(frozen=True)
class Fill:
    """One executed order. The atom of truth — everything else is derived."""

    order_id: str
    symbol: str
    side: str  # 'buy' | 'sell'
    quantity: float
    price: float
    filled_at: datetime

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"fill quantity must be > 0, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"fill price must be > 0, got {self.price}")
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"side must be buy/sell, got {self.side!r}")


@dataclass(frozen=True)
class RoundTrip:
    """A matched buy->sell pair (or the matched portion of one)."""

    symbol: str
    quantity: float
    entry_price: float
    entry_at: datetime
    entry_order_id: str
    exit_price: float
    exit_at: datetime
    exit_order_id: str

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.quantity

    @property
    def return_pct(self) -> float:
        return (self.pnl / self.cost_basis * 100.0) if self.cost_basis else 0.0

    @property
    def holding_days(self) -> int:
        return (self.exit_at - self.entry_at).days

    @property
    def trade_id(self) -> str:
        """Stable id so re-running reconciliation updates instead of duplicating."""
        return f"rt:{self.entry_order_id}:{self.exit_order_id}"


@dataclass(frozen=True)
class OpenLot:
    """A buy that has not been matched by a sell yet."""

    symbol: str
    quantity: float
    price: float
    opened_at: datetime
    order_id: str


@dataclass
class ReconcileResult:
    round_trips: List[RoundTrip] = field(default_factory=list)
    open_lots: List[OpenLot] = field(default_factory=list)
    unmatched_sells: List[Fill] = field(default_factory=list)
    persisted: int = 0

    @property
    def realized_pnl(self) -> float:
        return sum(rt.pnl for rt in self.round_trips)

    @property
    def wins(self) -> int:
        return sum(1 for rt in self.round_trips if rt.pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for rt in self.round_trips if rt.pnl <= 0)

    @property
    def win_rate(self) -> float:
        total = len(self.round_trips)
        return (self.wins / total * 100.0) if total else 0.0


def fetch_fills(broker_client, *, limit: int = 500) -> List[Fill]:
    """Pull filled orders from the broker, oldest first.

    Uses closed orders rather than the account-activities endpoint, which this
    SDK version (alpaca-py 0.43.x) does not expose on TradingClient.
    """
    trading_client = getattr(broker_client, "_trading_client", None)
    if trading_client is None:
        logging.warning("Broker client exposes no trading client; no fills to reconcile.")
        return []

    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=limit, nested=True)
        )
    except Exception as exc:
        logging.warning("Could not fetch orders for reconciliation (%s).", exc)
        return []

    fills: List[Fill] = []
    for order in _flatten_orders(orders):
        filled_at = getattr(order, "filled_at", None)
        filled_qty = _to_float(getattr(order, "filled_qty", None))
        fill_price = _to_float(getattr(order, "filled_avg_price", None))
        if filled_at is None or not filled_qty or not fill_price:
            continue

        side = getattr(order, "side", None)
        side_value = str(getattr(side, "value", side) or "").lower()
        if side_value not in {"buy", "sell"}:
            continue

        fills.append(
            Fill(
                order_id=str(getattr(order, "id", "")),
                symbol=str(getattr(order, "symbol", "")),
                side=side_value,
                quantity=filled_qty,
                price=fill_price,
                filled_at=filled_at,
            )
        )

    return sorted(fills, key=lambda f: f.filled_at)


def _flatten_orders(orders: Iterable) -> List:
    """Bracket orders nest their exit legs; those legs are real fills too."""
    out: List = []
    for order in orders or []:
        out.append(order)
        for leg in getattr(order, "legs", None) or []:
            out.append(leg)
    return out


def _to_float(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def match_fifo(fills: Sequence[Fill]) -> Tuple[List[RoundTrip], List[OpenLot], List[Fill]]:
    """Pair buys with sells per symbol, oldest lot first.

    Partial fills are handled by splitting lots, so a 3-share buy closed by a
    1-share and a 2-share sell produces two round trips at the same entry price.

    Sells with no matching lot are returned separately rather than dropped —
    they mean either a short (which the engine disallows) or history older than
    the fetch window, and silently discarding them would understate activity.
    """
    lots: Dict[str, Deque[List]] = defaultdict(deque)
    round_trips: List[RoundTrip] = []
    unmatched: List[Fill] = []

    for fill in sorted(fills, key=lambda f: f.filled_at):
        if fill.side == "buy":
            lots[fill.symbol].append([fill.quantity, fill])
            continue

        remaining = fill.quantity
        queue = lots[fill.symbol]
        while remaining > 1e-12 and queue:
            lot_qty, lot_fill = queue[0]
            matched = min(lot_qty, remaining)

            round_trips.append(
                RoundTrip(
                    symbol=fill.symbol,
                    quantity=matched,
                    entry_price=lot_fill.price,
                    entry_at=lot_fill.filled_at,
                    entry_order_id=lot_fill.order_id,
                    exit_price=fill.price,
                    exit_at=fill.filled_at,
                    exit_order_id=fill.order_id,
                )
            )

            remaining -= matched
            if lot_qty - matched <= 1e-12:
                queue.popleft()
            else:
                queue[0][0] = lot_qty - matched

        if remaining > 1e-12:
            unmatched.append(
                Fill(
                    order_id=fill.order_id,
                    symbol=fill.symbol,
                    side="sell",
                    quantity=remaining,
                    price=fill.price,
                    filled_at=fill.filled_at,
                )
            )

    open_lots = [
        OpenLot(
            symbol=symbol,
            quantity=qty,
            price=lot_fill.price,
            opened_at=lot_fill.filled_at,
            order_id=lot_fill.order_id,
        )
        for symbol, queue in lots.items()
        for qty, lot_fill in queue
    ]

    return round_trips, open_lots, unmatched


def persist_round_trips(
    round_trips: Sequence[RoundTrip],
    *,
    account_id: str,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> int:
    """Write round trips to the trades table with their real P&L.

    Keyed on the deterministic ``trade_id``, so reconciling repeatedly is safe
    and idempotent rather than piling up duplicates.
    """
    written = 0
    with connect(db_path) as conn:
        for rt in round_trips:
            conn.execute(
                """
                INSERT INTO trades(
                  trade_id, account_id, symbol, side, qty,
                  entry_price, exit_price, tp_price, sl_price,
                  status, confidence, pnl,
                  attempted_at, opened_at, closed_at,
                  broker_order_id, error
                ) VALUES (?, ?, ?, 'buy', ?, ?, ?, NULL, NULL, 'CLOSED', NULL, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(trade_id) DO UPDATE SET
                  exit_price = excluded.exit_price,
                  closed_at  = excluded.closed_at,
                  pnl        = excluded.pnl,
                  status     = 'CLOSED'
                """.strip(),
                (
                    rt.trade_id,
                    account_id,
                    rt.symbol,
                    rt.quantity,
                    rt.entry_price,
                    rt.exit_price,
                    rt.pnl,
                    _iso(rt.entry_at),
                    _iso(rt.entry_at),
                    _iso(rt.exit_at),
                    rt.exit_order_id,
                ),
            )
            written += 1
    return written


def _iso(moment: datetime) -> str:
    try:
        return moment.isoformat()
    except AttributeError:
        return utc_now_iso()


def reconcile_fills(
    broker_client,
    *,
    account_id: str,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    dry_run: bool = False,
    verbose: bool = True,
) -> ReconcileResult:
    """Fetch fills, match them, and record realized P&L. Read-only if dry_run."""
    fills = fetch_fills(broker_client)
    if not fills:
        if verbose:
            logging.info("FILL RECONCILE: no filled orders found.")
        return ReconcileResult()

    round_trips, open_lots, unmatched = match_fifo(fills)
    result = ReconcileResult(
        round_trips=round_trips, open_lots=open_lots, unmatched_sells=unmatched
    )

    if not dry_run and round_trips:
        result.persisted = persist_round_trips(
            round_trips, account_id=account_id, db_path=db_path
        )

    if verbose:
        logging.info(
            "FILL RECONCILE: %s fills -> %s round trips, realized P&L $%.2f "
            "(%sW/%sL, %.1f%% win rate), %s lots still open%s",
            len(fills), len(round_trips), result.realized_pnl,
            result.wins, result.losses, result.win_rate, len(open_lots),
            " [dry run, nothing written]" if dry_run else f", {result.persisted} rows written",
        )
        for rt in round_trips:
            logging.info(
                "  %-6s %6.2f sh  %s -> %s  $%.2f -> $%.2f  P&L $%+.2f (%+.2f%%, %sd)",
                rt.symbol, rt.quantity,
                rt.entry_at.date(), rt.exit_at.date(),
                rt.entry_price, rt.exit_price, rt.pnl, rt.return_pct, rt.holding_days,
            )
        for lot in open_lots:
            logging.info(
                "  OPEN %-6s %6.2f sh @ $%.2f since %s",
                lot.symbol, lot.quantity, lot.price, lot.opened_at.date(),
            )
        for sell in unmatched:
            logging.warning(
                "  UNMATCHED SELL %-6s %.2f sh @ $%.2f on %s "
                "(short, or entry predates the fetch window)",
                sell.symbol, sell.quantity, sell.price, sell.filled_at.date(),
            )

    return result
