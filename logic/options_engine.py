from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


def _log_stub(symbol: str, reason: str) -> None:
    logging.warning("Options data unavailable for %s (%s). Using placeholder candidates.", symbol, reason)


def get_option_chain(symbol: str) -> List[Dict[str, Any]]:
    """
    Fetch option chain data for a symbol using Alpaca options data client.

    Returns a list of dicts with keys:
    - symbol
    - expiration
    - strike
    - type ("call" or "put")
    - delta (optional)
    """
    try:
        from alpaca.data.historical import OptionHistoricalDataClient  # type: ignore
        from alpaca.data.requests import OptionChainRequest  # type: ignore
        from alpaca.data.timeframe import TimeFrame  # type: ignore
        from alpaca.data.enums import OptionsFeed  # type: ignore

        try:
            from alpaca_exercises import load_alpaca_creds

            creds = load_alpaca_creds()
            client = OptionHistoricalDataClient(creds.api_key, creds.secret_key)
        except Exception:
            client = OptionHistoricalDataClient()
        req = OptionChainRequest(
            symbol_or_symbols=[symbol],
            start=datetime.utcnow() - timedelta(days=3),
            end=datetime.utcnow(),
            timeframe=TimeFrame.Day,
            feed=OptionsFeed.OPRA,
        )
        chain = client.get_option_chain(req)
        rows: List[Dict[str, Any]] = []
        for opt in chain:
            rows.append(
                {
                    "symbol": symbol,
                    "expiration": str(getattr(opt, "expiration_date", "")),
                    "strike": float(getattr(opt, "strike_price", 0.0) or 0.0),
                    "type": str(getattr(opt, "type", "")).lower(),
                    "delta": getattr(opt, "delta", None),
                }
            )
        return rows
    except Exception as exc:
        _log_stub(symbol, str(exc))

    # Placeholder if Alpaca options data client is unavailable
    today = datetime.utcnow().date()
    placeholder_exp = today + timedelta(days=30)
    return [
        {
            "symbol": symbol,
            "expiration": placeholder_exp.isoformat(),
            "strike": None,
            "type": "call",
            "delta": None,
        },
        {
            "symbol": symbol,
            "expiration": placeholder_exp.isoformat(),
            "strike": None,
            "type": "put",
            "delta": None,
        },
    ]


def filter_expirations_by_dte(
    chain: List[Dict[str, Any]],
    min_dte: int = 21,
    max_dte: int = 45,
) -> List[Dict[str, Any]]:
    """Filter chain rows by days-to-expiration range."""
    today = datetime.utcnow().date()
    filtered: List[Dict[str, Any]] = []
    for row in chain:
        exp_str = row.get("expiration") or ""
        try:
            exp_date = datetime.fromisoformat(exp_str).date()
        except Exception:
            continue
        dte = (exp_date - today).days
        if min_dte <= dte <= max_dte:
            filtered.append(row)
    return filtered


def _select_by_delta(
    chain: List[Dict[str, Any]],
    target_delta: float,
    option_type: str,
) -> Optional[Dict[str, Any]]:
    best = None
    best_distance = None
    for row in chain:
        if row.get("type") != option_type:
            continue
        delta = row.get("delta")
        if delta is None:
            continue
        try:
            distance = abs(float(delta) - float(target_delta))
        except Exception:
            continue
        if best_distance is None or distance < best_distance:
            best = row
            best_distance = distance
    return best


def select_put_contract(chain: List[Dict[str, Any]], target_delta: float = 0.30) -> Dict[str, Any]:
    """Select a put contract closest to target delta (placeholder if needed)."""
    selected = _select_by_delta(chain, target_delta, "put")
    if selected is None:
        return {
            "symbol": chain[0].get("symbol") if chain else "",
            "expiration": chain[0].get("expiration") if chain else None,
            "strike": chain[0].get("strike") if chain else None,
            "type": "put",
            "delta": None,
        }
    return selected


def select_call_contract(chain: List[Dict[str, Any]], target_delta: float = 0.30) -> Dict[str, Any]:
    """Select a call contract closest to target delta (placeholder if needed)."""
    selected = _select_by_delta(chain, target_delta, "call")
    if selected is None:
        return {
            "symbol": chain[0].get("symbol") if chain else "",
            "expiration": chain[0].get("expiration") if chain else None,
            "strike": chain[0].get("strike") if chain else None,
            "type": "call",
            "delta": None,
        }
    return selected
