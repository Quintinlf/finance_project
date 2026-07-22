"""
Canonical account equity resolution for execution and risk sizing.

Single source of truth priority:
  1. Live broker account (equity / portfolio_value)
  2. Latest DB snapshot portfolio_value
  3. Latest DB snapshot equity
  4. Fallback cash balance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from pathlib import Path

from logic.broker_client import BrokerClient
from logic.sqlite_store import get_latest_account_snapshot


@dataclass
class AccountEquityContext:
    equity: float
    cash: float
    portfolio_value: float
    equity_source: str
    daily_return: Optional[float]
    exposure_fraction: float


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
        if parsed <= 0:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _compute_exposure_fraction(*, portfolio_value: float, cash: float) -> float:
    """Invested fraction of portfolio (0–1), comparable to max_portfolio_exposure."""
    if portfolio_value <= 0:
        return 0.0
    invested = max(0.0, portfolio_value - cash)
    return max(0.0, min(1.0, invested / portfolio_value))


def _broker_raw_account_fields(broker_client: BrokerClient) -> Dict[str, Any]:
    """Best-effort pull of extended Alpaca account fields when available."""
    trading_client = getattr(broker_client, "_trading_client", None)
    if trading_client is None:
        return {}
    try:
        acct = trading_client.get_account()
    except Exception:
        return {}
    return {
        "equity": getattr(acct, "equity", None),
        "last_equity": getattr(acct, "last_equity", None),
        "portfolio_value": getattr(acct, "portfolio_value", None),
        "cash": getattr(acct, "cash", None),
    }


def resolve_account_equity(
    *,
    broker_client: BrokerClient,
    db_path: Optional[Union[str, Path]] = None,
    account_id: Optional[str] = None,
    fallback_cash: float = 0.0,
) -> AccountEquityContext:
    """Resolve equity, daily return, and exposure from one consistent hierarchy."""
    summary: Dict[str, Any] = {}
    try:
        summary = dict(broker_client.get_account_summary() or {})
    except Exception:
        summary = {}

    broker_cash = _safe_float(summary.get("cash"))
    raw = _broker_raw_account_fields(broker_client)
    raw_equity = _safe_float(raw.get("equity"))
    broker_portfolio = _safe_float(summary.get("portfolio_value")) or raw_equity

    snap = None
    if db_path is not None:
        try:
            snap = get_latest_account_snapshot(db_path=db_path, account_id=account_id)
        except Exception:
            snap = None

    snap_portfolio = _safe_float(snap.get("portfolio_value")) if snap else None
    snap_equity = _safe_float(snap.get("equity")) if snap else None
    snap_cash = _safe_float(snap.get("cash")) if snap else None

    equity_source = "fallback_cash"
    equity = _safe_float(fallback_cash) or 0.0
    cash = broker_cash or snap_cash or _safe_float(fallback_cash) or 0.0
    portfolio_value = broker_portfolio or snap_portfolio or equity

    if raw_equity is not None:
        equity = raw_equity
        equity_source = "broker_live"
        portfolio_value = broker_portfolio or raw_equity
    elif broker_portfolio is not None:
        equity = broker_portfolio
        equity_source = "broker_portfolio_value"
        portfolio_value = broker_portfolio
    elif snap_portfolio is not None:
        equity = snap_portfolio
        equity_source = "snapshot_portfolio_value"
        portfolio_value = snap_portfolio
        if snap_cash is not None:
            cash = snap_cash
    elif snap_equity is not None:
        equity = snap_equity
        equity_source = "snapshot_equity"
        portfolio_value = snap_equity
        if snap_cash is not None:
            cash = snap_cash
    elif broker_cash is not None:
        equity = broker_cash
        equity_source = "broker_cash"
        portfolio_value = broker_cash
        cash = broker_cash
    elif _safe_float(fallback_cash) is not None:
        equity = float(fallback_cash)
        equity_source = "fallback_cash"
        portfolio_value = equity
        cash = equity

    daily_return: Optional[float] = None
    last_equity = _safe_float(raw.get("last_equity"))
    if raw_equity is not None and last_equity is not None:
        try:
            daily_return = raw_equity / last_equity - 1.0
        except Exception:
            daily_return = None
    elif snap and snap.get("last_equity") is not None and snap_equity is not None:
        try:
            daily_return = float(snap_equity) / float(snap["last_equity"]) - 1.0
        except Exception:
            daily_return = None

    exposure_fraction = _compute_exposure_fraction(
        portfolio_value=portfolio_value,
        cash=cash,
    )

    return AccountEquityContext(
        equity=float(equity),
        cash=float(cash),
        portfolio_value=float(portfolio_value),
        equity_source=equity_source,
        daily_return=daily_return,
        exposure_fraction=exposure_fraction,
    )
