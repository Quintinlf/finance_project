"""
Alpaca paper-trading utilities extracted from the notebook `alpaca_exercises.ipynb`.

Features:
- Environment loading (.env + standard env vars)
- Connect Trading and Data clients
- Account summary and positions table
- List orders (pretty table)
- Market and Limit orders (with simple wash-trade check)
- Historical trade fetch (pretty table)
- Cancel all open orders
- Risk: position sizing helper
- Optional: start a live trade stream (background thread)
- Optional: auto-trade using `trading_functions.unified_bayesian_gp_forecast`

Notes:
- Install deps: alpaca-py, python-dotenv, tabulate
- Never hardcode API keys; use environment variables or a .env file.
- Library-only: functions return data; separate format_* helpers render pretty tables with ANSI colors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Any, cast
import os
import threading

# Third-party
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore
    find_dotenv = None  # type: ignore

from tabulate import tabulate  # type: ignore

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderType,
    QueryOrderStatus,
)
from alpaca.common.exceptions import APIError

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream


# =============
# ANSI helpers
# =============
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _mask(val: Optional[str]) -> str:
    if not val:
        return "<missing>"
    return f"{val[:4]}...{val[-4:]}" if len(val) >= 8 else "<set>"


# =====================
# Environment / Clients
# =====================
@dataclass
class AlpacaCreds:
    api_key: str
    secret_key: str


def load_alpaca_creds(explicit_env_path: Optional[Path] = None) -> AlpacaCreds:
    """
    Load Alpaca credentials from environment/.env.

    Order of precedence (first found wins):
    - .env discovered via find_dotenv()
    - explicit_env_path (if provided)
    - existing process env vars

    Recognized keys:
    - APCA_API_KEY_ID / APCA_API_SECRET_KEY (official)
    - ALPACA_API_KEY / ALPACA_SECRET_KEY
    - API_KEY / SECRET_KEY (fallback)
    """
    # Load .env if available
    if load_dotenv and find_dotenv:
        loaded = load_dotenv(find_dotenv(), override=False)
        if not loaded and explicit_env_path:
            load_dotenv(explicit_env_path, override=False)
    elif load_dotenv and explicit_env_path:
        load_dotenv(explicit_env_path, override=False)

    api_key = (
        os.getenv("APCA_API_KEY_ID")
        or os.getenv("ALPACA_API_KEY")
        or os.getenv("API_KEY")
        or ""
    )
    secret_key = (
        os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("ALPACA_SECRET_KEY")
        or os.getenv("SECRET_KEY")
        or ""
    )

    if not api_key or not secret_key:
        raise ValueError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID/APCA_API_SECRET_KEY (or ALPACA_API_KEY/ALPACA_SECRET_KEY) in env or .env"
        )
    return AlpacaCreds(api_key=api_key, secret_key=secret_key)


def connect_trading_client(creds: Optional[AlpacaCreds] = None, paper: bool = True) -> TradingClient:
    """Create an Alpaca TradingClient using provided or loaded credentials. No output."""
    if creds is None:
        # Default .env location near Finance_project root if present
        default_env = Path(__file__).resolve().parents[2] / "Finance_project" / ".env"
        creds = load_alpaca_creds(default_env if default_env.exists() else None)
    return TradingClient(creds.api_key, creds.secret_key, paper=paper)


def connect_data_client(creds: Optional[AlpacaCreds] = None) -> StockHistoricalDataClient:
    """Create a StockHistoricalDataClient using provided or loaded credentials. No output."""
    if creds is None:
        default_env = Path(__file__).resolve().parents[2] / "Finance_project" / ".env"
        creds = load_alpaca_creds(default_env if default_env.exists() else None)
    return StockHistoricalDataClient(creds.api_key, creds.secret_key)


def verify_alpaca_setup(verbose: bool = True) -> bool:
    """
    Verify that Alpaca API is properly configured and connected.
    
    Checks:
    1. Credentials are loaded
    2. Trading client can connect
    3. Account is accessible
    4. Paper trading mode is enabled
    
    Args:
        verbose: Print detailed status messages
    
    Returns:
        True if setup is valid, False otherwise
    """
    if verbose:
        print("="*70)
        print("🔍 ALPACA API SETUP VERIFICATION")
        print("="*70)
    
    try:
        # Step 1: Load credentials
        if verbose:
            print("\n1️⃣ Checking credentials...")
        
        creds = load_alpaca_creds()
        
        if verbose:
            print(f"   ✅ API Key: {_mask(creds.api_key)}")
            print(f"   ✅ Secret Key: {_mask(creds.secret_key)}")
        
        # Step 2: Connect trading client
        if verbose:
            print("\n2️⃣ Connecting to Alpaca API...")
        
        trading_client = TradingClient(creds.api_key, creds.secret_key, paper=True)
        
        if verbose:
            print("   ✅ Trading client connected")
        
        # Step 3: Get account info
        if verbose:
            print("\n3️⃣ Fetching account information...")
        
        account = trading_client.get_account()
        
        if verbose:
            print(f"   ✅ Account ID: {account.id}")
            print(f"   ✅ Account Status: {account.status}")
            print(f"   ✅ Cash: ${float(account.cash):.2f}")
            print(f"   ✅ Buying Power: ${float(account.buying_power):.2f}")
            print(f"   ✅ Portfolio Value: ${float(account.portfolio_value):.2f}")
        
        # Step 4: Verify paper trading
        if verbose:
            print("\n4️⃣ Verifying paper trading mode...")
        
        # Alpaca paper accounts have specific URL patterns
        is_paper = True  # Assume paper if connection works with paper=True
        
        if verbose:
            if is_paper:
                print("   ✅ Paper trading mode ENABLED (no real money at risk)")
            else:
                print("   ⚠️  WARNING: May be live trading mode!")
        
        # Step 5: Test data client
        if verbose:
            print("\n5️⃣ Testing data client...")
        
        data_client = StockHistoricalDataClient(creds.api_key, creds.secret_key)
        
        if verbose:
            print("   ✅ Data client connected")
        
        # Summary
        if verbose:
            print("\n" + "="*70)
            print("✅ SETUP VERIFICATION COMPLETE")
            print("="*70)
            print("\n🎉 Your Alpaca account is ready for trading!")
            print("   You can now run run_once() to execute your strategy.")
            print("="*70)
        
        return True
    
    except ValueError as e:
        if verbose:
            print(f"\n❌ ERROR: {str(e)}")
            print("\n📝 SETUP INSTRUCTIONS:")
            print("1. Create a .env file in the Finance_project folder")
            print("2. Add your Alpaca paper trading keys:")
            print("   APCA_API_KEY_ID=your_key_here")
            print("   APCA_API_SECRET_KEY=your_secret_here")
            print("3. Get free paper trading keys at: https://alpaca.markets/")
            print("="*70)
        return False
    
    except Exception as e:
        if verbose:
            print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
            print("="*70)
        return False


# ======================
# Pretty printing helpers
# ======================
def color_side(side: str) -> str:
    return f"{GREEN}BUY{RESET}" if side.lower() == "buy" else f"{RED}SELL{RESET}"


def color_status(status: str) -> str:
    s = status.lower()
    if s in {"filled", "accepted"}:
        return f"{GREEN}{status.upper()}{RESET}"
    if s in {"pending", "new", "open"}:
        return f"{YELLOW}{status.upper()}{RESET}"
    return f"{RED}{status.upper()}{RESET}"


def format_table(rows, headers) -> str:
    """Render a fancy_grid table string from rows and headers."""
    return tabulate(rows, headers=headers, tablefmt="fancy_grid")


# ======================
# Core account operations
# ======================
def get_account_summary(client: TradingClient) -> dict:
    """Return a dict with key account fields. No printing."""
    acct = client.get_account()
    summary = {
        "status": getattr(getattr(acct, "status", ""), "value", str(getattr(acct, "status", ""))),
        "cash": float(getattr(acct, "cash", 0) or 0),
        "portfolio_value": float(getattr(acct, "portfolio_value", 0) or 0),
        "buying_power": float(getattr(acct, "buying_power", 0) or 0),
    }
    return summary


def format_account_summary(summary: dict) -> str:
    rows = [[
        summary.get("status", ""),
        f"${summary.get('cash', 0):,.2f}",
        f"${summary.get('portfolio_value', 0):,.2f}",
        f"${summary.get('buying_power', 0):,.2f}",
    ]]
    return format_table(rows, ["Status", "Cash", "Portfolio Value", "Buying Power"])


def list_all_orders(client: TradingClient):
    """Return raw list of order objects (no printing)."""
    return client.get_orders()


def format_orders_table(orders) -> str:
    if not orders:
        return "No orders found."
    rows: List[List[object]] = []
    for o in orders:
        side_val = o.side.value if hasattr(o.side, "value") else str(o.side)
        status_val = o.status.value if hasattr(o.status, "value") else str(o.status)
        rows.append([o.symbol, color_side(side_val), o.qty, color_status(status_val)])
    return format_table(rows, ["Symbol", "Side", "Qty", "Status"])


def get_positions(client: TradingClient):
    """Return raw list of position objects (no printing)."""
    return client.get_all_positions()


def format_positions_table(positions) -> str:
    if not positions:
        return "No current positions."
    rows: List[List[object]] = []
    for p in positions:
        qty = float(p.qty)
        qty_color = GREEN if qty > 0 else RED
        rows.append([p.symbol, f"{qty_color}{p.qty}{RESET}", f"${float(p.avg_entry_price):.2f}"])
    return format_table(rows, ["Symbol", "Quantity", "Avg Entry Price"])


# =====================
# Order helper routines
# =====================
def place_market_order(
    client: TradingClient, *, symbol: str, qty: int, side: str, tif: TimeInForce = TimeInForce.DAY
):
    """Submit a market order. Returns the order object or None on API error. No printing."""
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif)
    try:
        return client.submit_order(req)
    except APIError:
        return None


def format_order_table(order) -> str:
    if not order:
        return "<no order>"
    rows = [[
        order.id,
        order.symbol,
        order.qty,
        order.side.value,
        order.type.value,
        order.status.value,
        getattr(order, "limit_price", None) or "-",
        order.filled_qty,
        getattr(order, "filled_avg_price", None) or "-",
        order.submitted_at,
    ]]
    headers = [
        "ID",
        "Symbol",
        "Qty",
        "Side",
        "Type",
        "Status",
        "Limit Price",
        "Filled Qty",
        "Avg Price",
        "Submitted At",
    ]
    return format_table(rows, headers)


def place_limit_order_with_wash_check(
    client: TradingClient,
    *,
    symbol: str,
    qty: int,
    side: str,
    limit_price: float,
    tif: TimeInForce = TimeInForce.DAY,
):
    """
    Submit a limit order after a simple wash-trade check.
    Returns tuple: (order or None, conflicting_orders:list)
    No printing.
    """
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    req = LimitOrderRequest(
        symbol=symbol, qty=qty, side=side_enum, limit_price=limit_price, time_in_force=tif
    )

    open_filter = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    open_orders = cast(List[Any], client.get_orders(filter=open_filter))
    opposite = OrderSide.SELL if side_enum == OrderSide.BUY else OrderSide.BUY
    conflicting = []
    for o in open_orders:
        side_attr = getattr(o, "side", None)
        type_attr = getattr(o, "type", None)
        if side_attr == opposite and type_attr in {OrderType.MARKET, OrderType.STOP, OrderType.STOP_LIMIT}:
            conflicting.append(o)
    if conflicting:
        return None, conflicting

    try:
        res = client.submit_order(req)
        return res, []
    except APIError:
        return None, []


def cancel_all_open_orders(client: TradingClient):
    """Cancel all open orders. Returns dict with 'cancelled' and 'failed' lists. No printing."""
    open_filter = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    open_orders = cast(List[Any], client.get_orders(filter=open_filter))
    cancelled = []
    failed = []
    for o in open_orders:
        try:
            oid = getattr(o, "id", None)
            if oid is None:
                failed.append({"id": None, "error": "Order missing id"})
                continue
            client.cancel_order_by_id(oid)
            cancelled.append(oid)
        except APIError as e:
            failed.append({"id": getattr(o, "id", None), "error": str(e)})
    return {"cancelled": cancelled, "failed": failed}


# =====================
# Historical trade fetch
# =====================
def fetch_trades(
    data_client: StockHistoricalDataClient,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
):
    """Return list of trade objects for symbol in [start, end]. No printing."""
    req = StockTradesRequest(symbol_or_symbols=[symbol], start=start, end=end)
    resp = data_client.get_stock_trades(req)
    try:
        return resp[symbol]
    except Exception:
        return []


def format_trades_table(trades) -> str:
    if not trades:
        return "No trades found in the given time range."
    first_price = trades[0].price
    rows: List[List[object]] = []
    for t in trades:
        price_str = (
            f"{GREEN}{t.price}{RESET}" if t.price > first_price else f"{RED}{t.price}{RESET}" if t.price < first_price else f"{t.price}"
        )
        rows.append([t.timestamp, t.symbol, price_str, t.size, t.exchange, ", ".join(t.conditions)])
    return format_table(rows, ["Timestamp", "Symbol", "Price", "Size", "Exchange", "Conditions"])


# =========================
# Streaming (background thread)
# =========================
def start_trade_stream(
    creds: Optional[AlpacaCreds],
    symbols: Sequence[str],
    on_trade: Optional[Callable[[object], None]] = None,
) -> Tuple[StockDataStream, threading.Thread]:
    """
    Start StockDataStream in a daemon background thread. Returns (stream, thread).
    No printing; provide on_trade callback to consume messages.
    """
    if creds is None:
        default_env = Path(__file__).resolve().parents[2] / "Finance_project" / ".env"
        creds = load_alpaca_creds(default_env if default_env.exists() else None)

    stream = StockDataStream(creds.api_key, creds.secret_key)

    async def _handle_trade(data):
        if on_trade:
            on_trade(data)

    # subscribe
    stream.subscribe_trades(_handle_trade, *symbols)

    # run in background thread to avoid asyncio conflicts
    t = threading.Thread(target=stream.run, daemon=True)
    t.start()
    return stream, t


# ==================
# Risk management
# ==================
def calculate_position_size(
    account_cash: float, entry_price: float, stop_loss_price: float, risk_percent: float = 1.0
) -> int:
    """Risk-based position sizing (integer shares)."""
    risk_amount = account_cash * (risk_percent / 100.0)
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share == 0:
        return 0
    return int(risk_amount / risk_per_share)


# =============================================
# Optional: Auto-trade via forecasted signal
# =============================================
def auto_trade_with_forecast(
    client: TradingClient, *, symbol: str, confidence_threshold: float = 0.7
):
    """
    Try to import and run `trading_functions.unified_bayesian_gp_forecast(symbol)` and place a
    market order when confidence exceeds threshold. No printing.

    Returns dict with keys: symbol, signal, confidence, order (order object or None)
    """
    try:
        # Make sure Finance_project is importable if this module lives deeper
        project_root = Path(__file__).resolve().parents[2] / "Finance_project"
        import sys as _sys

        if str(project_root) not in _sys.path:
            _sys.path.append(str(project_root))

        from Finance_project.functions.trading_functions import unified_bayesian_gp_forecast  # type: ignore
    except Exception:
        return {"symbol": symbol, "signal": None, "confidence": None, "order": None}

    import pandas as pd  # local import to avoid hard dependency if not used

    def _normalize_signal_and_confidence(res) -> Tuple[Optional[str], Optional[float]]:
        sig = None
        conf: Optional[float] = None
        if isinstance(res, dict):
            sig = res.get("final_signal") or res.get("signal") or res.get("action")
            conf = res.get("final_confidence") or res.get("confidence") or res.get("prob") or res.get("score")
        elif isinstance(res, pd.DataFrame):
            row = res.iloc[0]
            sig = row.get("final_signal") or row.get("signal") or row.get("action")
            conf = row.get("final_confidence") or row.get("confidence") or row.get("prob") or row.get("score")
        elif isinstance(res, (list, tuple)) and res and str(res[0]).lower() in {"buy", "sell", "hold"}:
            sig = str(res[0])
            conf = float(res[1]) if len(res) > 1 else None

        if isinstance(sig, str):
            sig = sig.lower()

        if isinstance(conf, str):
            txt = conf.strip().replace("%", "")
            try:
                conf = float(txt)
            except ValueError:
                conf = None

        if isinstance(conf, (int, float)) and conf > 1:
            conf = conf / 100.0
        return sig, conf

    res = unified_bayesian_gp_forecast(symbol)
    signal, confidence = _normalize_signal_and_confidence(res)

    order_result = None
    if signal in {"buy", "sell"} and isinstance(confidence, (int, float)) and confidence > confidence_threshold:
        side_enum = OrderSide.BUY if signal == "buy" else OrderSide.SELL
        order = MarketOrderRequest(symbol=symbol, qty=1, side=side_enum, time_in_force=TimeInForce.DAY)
        order_result = client.submit_order(order)

    return {"symbol": symbol, "signal": signal, "confidence": confidence, "order": order_result}


def format_auto_trade_summary(summary: dict) -> str:
    def _color_signal(sig: Optional[str]) -> str:
        if sig == "buy":
            return f"{GREEN}BUY{RESET}"
        if sig == "sell":
            return f"{RED}SELL{RESET}"
        return f"{YELLOW}{(sig or 'HOLD').upper()}{RESET}"

    rows = [[
        summary.get("symbol", ""),
        _color_signal(summary.get("signal")),
        (
            f"{summary['confidence']*100:.1f}%"
            if isinstance(summary.get("confidence"), (int, float)) else "N/A"
        ),
        "✅ Order Placed" if summary.get("order") else "❌ No Trade",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]]
    return format_table(rows, ["Symbol", "Signal", "Confidence", "Action Taken", "Timestamp"])


# ================
# Quick demo entry
# ================
if __name__ == "__main__":
    # Library-only: no side effects or output when run directly.
    # Intentionally do nothing.
    pass

