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
import logging
import os

from logic.data_structures import PositionState


@dataclass
class BrokerOrder:
    id: str
    symbol: str
    qty: float
    side: str
    order_type: str
    status: str = "accepted"
    submitted_at: datetime = field(default_factory=datetime.utcnow)


def _order_qty(alpaca_order: Any, fallback: float) -> float:
    """Read an order quantity off an Alpaca order response.

    Alpaca documents Order.qty as a *string* that "can take up to 9 decimal
    points" (GET/POST /v2/orders), so fractional-share orders come back as e.g.
    "0.5". Parsing that with int() raises ValueError, which would lose track of
    an order that the broker has already accepted.
    """
    raw = getattr(alpaca_order, "qty", None)
    if raw is None:
        return float(fallback)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(fallback)


class BrokerClient(ABC):
    @abstractmethod
    def get_account_summary(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_position_states(self, universe: Iterable[str]) -> Dict[str, PositionState]:
        raise NotImplementedError

    @abstractmethod
    def place_market_order(self, *, symbol: str, qty: int, side: str, time_in_force: str = "day") -> Optional[BrokerOrder]:
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
    ) -> Optional[BrokerOrder]:
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
            # Mirrors the GET /v2/account blocking flags so callers can gate on
            # the same keys regardless of which broker client is in use.
            "trading_blocked": False,
            "account_blocked": False,
            "trade_suspended_by_user": False,
            "can_trade": True,
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
        # Reason the most recent submission was rejected, so the caller can put
        # the broker's own words in the decision log instead of a bare None.
        self.last_order_error: Optional[str] = None

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

    def place_market_order(
        self, *, symbol: str, qty: int, side: str, time_in_force: str = "day"
    ) -> Optional[BrokerOrder]:
        from alpaca.trading.enums import TimeInForce
        from logic.alpaca_exercises import OrderSubmissionError, place_market_order

        tif = TimeInForce.DAY if str(time_in_force).lower() == "day" else TimeInForce.GTC
        self.last_order_error = None
        try:
            alpaca_order = place_market_order(
                self._trading_client, symbol=symbol, qty=qty, side=side, tif=tif
            )
        except OrderSubmissionError as exc:
            self.last_order_error = exc.message
            return None

        # Wrap Alpaca order response in BrokerOrder dataclass
        if alpaca_order is None:
            self.last_order_error = "broker returned no order object"
            return None

        status = getattr(alpaca_order, "status", "accepted")
        if hasattr(status, "value"):
            status = status.value
        
        return BrokerOrder(
            id=str(getattr(alpaca_order, "id", "")),
            symbol=str(getattr(alpaca_order, "symbol", symbol)),
            qty=_order_qty(alpaca_order, qty),
            side=side,
            order_type="market",
            status=str(status),
            submitted_at=datetime.utcnow(),
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
    ) -> Optional[BrokerOrder]:
        from alpaca.trading.enums import TimeInForce
        from logic.alpaca_exercises import OrderSubmissionError, place_bracket_order

        tif = TimeInForce.DAY if str(time_in_force).lower() == "day" else TimeInForce.GTC
        self.last_order_error = None
        try:
            result = place_bracket_order(
                self._trading_client,
                symbol=symbol,
                qty=qty,
                side=side,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                tif=tif,
            )
        except OrderSubmissionError as exc:
            self.last_order_error = exc.message
            return None

        # Wrap Alpaca bracket order response in BrokerOrder dataclass
        if result is None:
            self.last_order_error = "broker returned no bracket result"
            return None

        main_order = result.get("main_order") if isinstance(result, dict) else result
        if main_order is None:
            self.last_order_error = "bracket result contained no main order"
            return None

        status = getattr(main_order, "status", "accepted")
        if hasattr(status, "value"):
            status = status.value
        
        return BrokerOrder(
            id=str(getattr(main_order, "id", "")),
            symbol=str(getattr(main_order, "symbol", symbol)),
            qty=_order_qty(main_order, qty),
            side=side,
            order_type="bracket",
            status=str(status),
            submitted_at=datetime.utcnow(),
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


class BrokerConnectionError(RuntimeError):
    """Could not reach the real broker when the caller required one."""


def create_broker_client(
    *,
    execution_mode: str,
    creds: Optional[Any] = None,
    paper: Optional[bool] = None,
    initial_cash: float = 100000.0,
    strict: Optional[bool] = None,
) -> BrokerClient:
    """Return a broker client for the requested execution mode.

    When `execution_mode` is 'paper' or 'live' and the Alpaca connection cannot
    be established, this used to fall back to an in-memory `PaperBrokerClient`
    seeded with $100,000 — silently, via a bare `except Exception`. The run then
    "traded" a fake book that vanished when the process exited, and reported
    success. That is exactly how a scheduled GitHub Actions job ran green for
    months without a single order ever reaching Alpaca.

    Now the failure is logged loudly with its cause, and in `strict` mode it
    raises instead. Strict defaults to True in CI (where nobody is watching a
    terminal and a fake fill is worthless) and False locally, so ad-hoc
    experimentation without credentials still works. Override explicitly, or
    set STRICT_BROKER=true/false.
    """
    mode = str(execution_mode).lower()
    if mode == "simulation":
        return PaperBrokerClient(initial_cash=initial_cash)

    if strict is None:
        strict_env = os.getenv("STRICT_BROKER", "").strip().lower()
        if strict_env in {"1", "true", "yes", "y"}:
            strict = True
        elif strict_env in {"0", "false", "no", "n"}:
            strict = False
        else:
            # No terminal to watch and no human to notice a fake fill.
            strict = os.getenv("CI", "").strip().lower() in {"1", "true"}

    try:
        if creds is None:
            from logic.alpaca_exercises import load_alpaca_creds

            creds = load_alpaca_creds()
        from logic.alpaca_exercises import connect_trading_client

        paper_mode = paper if paper is not None else mode != "live"
        trading_client = connect_trading_client(creds, paper=paper_mode)
        client = AlpacaBrokerClient(trading_client, paper=paper_mode)
        # Force one real API call now. Constructing a TradingClient does no I/O,
        # so bad credentials would otherwise stay hidden until the first order.
        client.get_account_summary()
        return client
    except Exception as exc:
        message = (
            f"Could not connect to Alpaca in '{mode}' mode: {type(exc).__name__}: {exc}. "
            f"Check that APCA_API_KEY_ID/APCA_API_SECRET_KEY (or ALPACA_API_KEY/"
            f"ALPACA_SECRET_KEY, or API_KEY/SECRET_KEY) are set in the environment. "
            f"In GitHub Actions these come from repository secrets."
        )
        if strict:
            logging.error("BROKER CONNECTION FAILED (strict mode): %s", message)
            raise BrokerConnectionError(message) from exc

        logging.error(
            "BROKER CONNECTION FAILED: %s\n"
            "  Falling back to an IN-MEMORY simulated broker with $%.2f. "
            "NOTHING FROM THIS RUN WILL REACH ALPACA and all state is discarded "
            "when the process exits.",
            message, initial_cash,
        )
        return PaperBrokerClient(initial_cash=initial_cash)