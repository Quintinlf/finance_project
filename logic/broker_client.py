"""Broker client abstraction for execution, portfolio, and daily orchestration.

The trading engine uses this module to avoid depending on raw Alpaca clients.
AlpacaBrokerClient wraps the Alpaca SDK, while PaperBrokerClient provides a
safe fallback when Alpaca credentials are unavailable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional

from logic.data_structures import PositionState


@dataclass
class BrokerOrder:
    id: str
    symbol: str
    qty: int
    side: str
    order_type: str
    status: str = "accepted"
    submitted_at: datetime = field(default_factory=datetime.utcnow)


class BrokerClient(ABC):
    @abstractmethod
    def get_account_summary(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_position_states(self, universe: Iterable[str]) -> Dict[str, PositionState]:
        raise NotImplementedError

    @abstractmethod
    def place_market_order(self, *, symbol: str, qty: int, side: str, time_in_force: str = "day") -> Any:
        raise NotImplementedError

    @abstractmethod
    def place_bracket_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        time_in_force: str = "day",
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def is_market_open(self, trading_date: date) -> bool:
        raise NotImplementedError


class PaperBrokerClient(BrokerClient):
    def __init__(self, *, initial_cash: float = 100000.0, account_id: str = "paper-default") -> None:
        self._cash = float(initial_cash)
        self._account_id = account_id
        self._positions: Dict[str, PositionState] = {}
        self._last_prices: Dict[str, float] = {}

    def get_account_summary(self) -> Dict[str, Any]:
        portfolio_value = self._cash
        for symbol, state in self._positions.items():
            price = self._last_prices.get(symbol, float(state.avg_entry_price or 0.0))
            portfolio_value += float(state.quantity) * float(price)

        return {
            "cash": self._cash,
            "buying_power": max(self._cash * 2.0, 0.0),
            "portfolio_value": portfolio_value,
            "account_id": self._account_id,
            "status": "PAPER",
        }

    def get_position_states(self, universe: Iterable[str]) -> Dict[str, PositionState]:
        position_states: Dict[str, PositionState] = {}
        for symbol in universe:
            if symbol in self._positions:
                position_states[symbol] = self._positions[symbol]
            else:
                position_states[symbol] = PositionState(
                    symbol=symbol,
                    quantity=0,
                    avg_entry_price=0.0,
                    side="flat",
                    source="paper",
                )
        return position_states

    def place_market_order(self, *, symbol: str, qty: int, side: str, time_in_force: str = "day") -> BrokerOrder:
        price = self._last_prices.get(symbol, 100.0)
        self._apply_fill(symbol=symbol, qty=qty, side=side, price=price)
        return BrokerOrder(
            id=f"PAPER-{symbol}-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="market",
            status="filled",
        )

    def place_bracket_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        time_in_force: str = "day",
    ) -> BrokerOrder:
        order = self.place_market_order(symbol=symbol, qty=qty, side=side, time_in_force=time_in_force)
        order.order_type = "bracket"
        return order

    def is_market_open(self, trading_date: date) -> bool:
        return trading_date.weekday() < 5

    def _apply_fill(self, *, symbol: str, qty: int, side: str, price: float) -> None:
        self._last_prices[symbol] = float(price)
        signed_qty = int(qty) if side.lower() == "buy" else -int(qty)
        current = self._positions.get(
            symbol,
            PositionState(symbol=symbol, quantity=0, avg_entry_price=0.0, side="flat", source="paper"),
        )
        next_qty = float(current.quantity) + signed_qty

        if next_qty == 0:
            self._positions[symbol] = PositionState(
                symbol=symbol,
                quantity=0,
                avg_entry_price=0.0,
                side="flat",
                source="paper",
                unrealized_pl=0.0,
            )
            self._cash -= signed_qty * float(price)
            return

        same_direction = (current.quantity >= 0 and next_qty > 0) or (current.quantity <= 0 and next_qty < 0)
        if same_direction and current.quantity != 0:
            total_qty = abs(float(current.quantity)) + abs(float(signed_qty))
            weighted_cost = (abs(float(current.quantity)) * float(current.avg_entry_price)) + (
                abs(float(signed_qty)) * float(price)
            )
            avg_price = weighted_cost / total_qty if total_qty else float(price)
        else:
            avg_price = float(price)

        side_value = "long" if next_qty > 0 else "short"
        self._positions[symbol] = PositionState(
            symbol=symbol,
            quantity=next_qty,
            avg_entry_price=avg_price,
            side=side_value,
            source="paper",
        )
        self._cash -= signed_qty * float(price)


class AlpacaBrokerClient(BrokerClient):
    def __init__(self, trading_client: Any, *, paper: bool = True) -> None:
        self._trading_client = trading_client
        self._paper = paper

    def get_account_summary(self) -> Dict[str, Any]:
        from logic.alpaca_exercises import get_account_summary

        summary = dict(get_account_summary(self._trading_client))
        try:
            account = self._trading_client.get_account()
            summary.setdefault("account_id", str(getattr(account, "id", "")))
            summary.setdefault("status", str(getattr(account, "status", "")))
        except Exception:
            summary.setdefault("account_id", "paper-default" if self._paper else "alpaca-live")
        return summary

    def get_position_states(self, universe: Iterable[str]) -> Dict[str, PositionState]:
        from logic.alpaca_exercises import get_positions

        position_states: Dict[str, PositionState] = {}
        for pos in get_positions(self._trading_client):
            qty = float(pos.qty)
            position_states[pos.symbol] = PositionState(
                symbol=pos.symbol,
                quantity=qty,
                avg_entry_price=float(pos.avg_entry_price),
                side="long" if qty > 0 else "short" if qty < 0 else "flat",
                source="alpaca" if not self._paper else "paper",
                unrealized_pl=float(getattr(pos, "unrealized_pl", 0.0) or 0.0),
            )

        for symbol in universe:
            if symbol not in position_states:
                position_states[symbol] = PositionState(
                    symbol=symbol,
                    quantity=0,
                    avg_entry_price=0.0,
                    side="flat",
                    source="alpaca" if not self._paper else "paper",
                )
        return position_states

    def place_market_order(self, *, symbol: str, qty: int, side: str, time_in_force: str = "day") -> Any:
        from alpaca.trading.enums import TimeInForce
        from logic.alpaca_exercises import place_market_order

        tif = TimeInForce.DAY if str(time_in_force).lower() == "day" else TimeInForce.GTC
        return place_market_order(self._trading_client, symbol=symbol, qty=qty, side=side, tif=tif)

    def place_bracket_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        time_in_force: str = "day",
    ) -> Any:
        from alpaca.trading.enums import TimeInForce
        from logic.alpaca_exercises import place_bracket_order

        tif = TimeInForce.DAY if str(time_in_force).lower() == "day" else TimeInForce.GTC
        return place_bracket_order(
            self._trading_client,
            symbol=symbol,
            qty=qty,
            side=side,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            tif=tif,
        )

    def is_market_open(self, trading_date: date) -> bool:
        if trading_date.weekday() >= 5:
            return False
        try:
            from alpaca.trading.requests import GetCalendarRequest

            req = GetCalendarRequest(start=trading_date.isoformat(), end=trading_date.isoformat())
            calendar = self._trading_client.get_calendar(req)
            return len(calendar) > 0
        except Exception:
            return True


def create_broker_client(
    *,
    execution_mode: str,
    creds: Optional[Any] = None,
    paper: Optional[bool] = None,
    initial_cash: float = 100000.0,
) -> BrokerClient:
    """Return a valid broker client for the requested execution mode."""
    mode = str(execution_mode).lower()
    if mode == "simulation":
        return PaperBrokerClient(initial_cash=initial_cash)

    try:
        if creds is None:
            from logic.alpaca_exercises import load_alpaca_creds

            creds = load_alpaca_creds()
        from logic.alpaca_exercises import connect_trading_client

        paper_mode = paper if paper is not None else mode != "live"
        trading_client = connect_trading_client(creds, paper=paper_mode)
        return AlpacaBrokerClient(trading_client, paper=paper_mode)
    except Exception:
        return PaperBrokerClient(initial_cash=initial_cash)