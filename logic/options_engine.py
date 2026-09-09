from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


def _log_stub(symbol: str, reason: str) -> None:
    logging.warning("Options data unavailable for %s (%s). Using placeholder candidates.", symbol, reason)


def _parse_occ_symbol(contract_symbol: str) -> Optional[Dict[str, Any]]:
    """Decode strike, expiration, and type from an OCC option symbol.

    Alpaca keys the chain by OCC symbol, e.g. ``UNG260916P00017000``:

        UNG      underlying, variable length
        260916   expiration, YYMMDD
        P        C(all) or P(ut)
        00017000 strike x 1000, zero-padded to 8 digits -> $17.00

    Everything the selectors need is in the key; the snapshot object itself
    carries only quotes and (on OPRA) greeks.
    """
    import re

    match = re.fullmatch(r"([A-Z]+)(\d{6})([CP])(\d{8})", contract_symbol.strip().upper())
    if not match:
        return None
    _underlying, yymmdd, kind, strike_raw = match.groups()
    try:
        expiration = datetime.strptime(yymmdd, "%y%m%d").date()
    except ValueError:
        return None
    return {
        "expiration": expiration.isoformat(),
        "strike": int(strike_raw) / 1000.0,
        "type": "call" if kind == "C" else "put",
    }


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
        from alpaca.data.enums import OptionsFeed  # type: ignore

        try:
            from logic.alpaca_exercises import load_alpaca_creds

            creds = load_alpaca_creds()
            client = OptionHistoricalDataClient(creds.api_key, creds.secret_key)
        except Exception:
            client = OptionHistoricalDataClient()
        # OPRA is the real-time feed and requires a signed OPRA agreement on
        # the Alpaca account. This account does not have one, so every chain
        # request failed with "OPRA agreement is not signed" and the engine
        # silently fell back to placeholder contracts with strike=None and
        # delta=None -- decorative data that could never price a trade.
        #
        # INDICATIVE is the free, delayed feed and returns real strikes and
        # greeks. Delayed data is unsuitable for intraday options execution,
        # but this bot decides once a day off daily bars, so a delayed chain is
        # consistent with everything else it does. Try OPRA first so the code
        # upgrades itself the day an agreement is signed.
        chain = None
        last_error = None
        for feed in (OptionsFeed.OPRA, OptionsFeed.INDICATIVE):
            try:
                chain = client.get_option_chain(
                    OptionChainRequest(underlying_symbol=symbol, feed=feed)
                )
                if feed is OptionsFeed.INDICATIVE:
                    logging.info(
                        "Options chain for %s served from the INDICATIVE (delayed) "
                        "feed; OPRA unavailable (%s).",
                        symbol, last_error,
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = " ".join(str(exc).split())
                continue
        if chain is None:
            raise RuntimeError(f"no options feed available ({last_error})")
        # get_option_chain returns a dict keyed by OCC contract symbol, so
        # iterating it yields *strings*, not objects. The previous code called
        # getattr(str, "strike_price") on each key, which silently produced
        # strike=None and expiration="" for every contract -- a chain that
        # looked populated but could not price anything.
        rows: List[Dict[str, Any]] = []
        for contract_symbol, snapshot in chain.items():
            parsed = _parse_occ_symbol(str(contract_symbol))
            if parsed is None:
                continue
            greeks = getattr(snapshot, "greeks", None)
            rows.append(
                {
                    "symbol": str(contract_symbol),
                    "underlying": symbol,
                    "expiration": parsed["expiration"],
                    "strike": parsed["strike"],
                    "type": parsed["type"],
                    # Greeks are None on the INDICATIVE feed; selection falls
                    # back to moneyness when that happens.
                    "delta": getattr(greeks, "delta", None) if greeks else None,
                    "implied_volatility": getattr(snapshot, "implied_volatility", None),
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
            # Put deltas are NEGATIVE. Comparing them to a positive target
            # directly made "closest to 0.30" resolve to the delta nearest
            # +0.30 on the number line -- which for puts is the one closest to
            # zero, i.e. the furthest out-of-the-money, nearly worthless
            # contract. A real 0.30-delta put (-0.30) scored as the *worst*
            # match. Compare magnitudes so both sides mean the same thing.
            distance = abs(abs(float(delta)) - abs(float(target_delta)))
        except Exception:
            continue
        if best_distance is None or distance < best_distance:
            best = row
            best_distance = distance
    return best


# Roughly where a 0.30-delta contract sits for typical equity/ETF vol at ~30
# DTE. A crude stand-in for the real thing, and labeled as such wherever the
# selection is reported.
_DELTA_TO_MONEYNESS = {0.30: 0.06, 0.25: 0.08, 0.40: 0.03, 0.50: 0.0}


def _select_by_moneyness(
    chain: List[Dict[str, Any]],
    target_delta: float,
    option_type: str,
    spot: float,
) -> Optional[Dict[str, Any]]:
    """Pick a contract by distance out-of-the-money when greeks are missing.

    The INDICATIVE (free) feed returns no greeks, so delta-based selection
    silently fell through to ``chain[0]`` -- an arbitrary strike. Approximating
    the target delta by moneyness is far from exact, but it selects a contract
    in the right region of the chain instead of an arbitrary one.
    """
    if spot <= 0:
        return None
    offset = _DELTA_TO_MONEYNESS.get(round(float(target_delta), 2), 0.06)
    # OTM is above spot for calls, below for puts.
    target_strike = spot * (1.0 + offset) if option_type == "call" else spot * (1.0 - offset)

    best = None
    best_distance = None
    for row in chain:
        if row.get("type") != option_type:
            continue
        strike = row.get("strike")
        if not strike:
            continue
        distance = abs(float(strike) - target_strike)
        if best_distance is None or distance < best_distance:
            best = dict(row)
            best["selection_basis"] = (
                f"moneyness proxy for {target_delta:.2f} delta "
                f"(no greeks on the delayed feed); spot ${spot:.2f}, "
                f"target strike ${target_strike:.2f}"
            )
            best_distance = distance
    return best


def select_put_contract(chain: List[Dict[str, Any]], target_delta: float = 0.30) -> Dict[str, Any]:
    """Select a put contract closest to target delta (placeholder if needed)."""
    selected = _select_by_delta(chain, target_delta, "put")
    if selected is None and chain:
        # No greeks (delayed feed): approximate the target delta by moneyness
        # rather than falling through to an arbitrary chain[0].
        underlying = chain[0].get("underlying") or chain[0].get("symbol") or ""
        spot = _get_spot_price(underlying) or 0.0
        selected = _select_by_moneyness(chain, target_delta, "put", spot)
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
    if selected is None and chain:
        # No greeks (delayed feed): approximate the target delta by moneyness
        # rather than falling through to an arbitrary chain[0].
        underlying = chain[0].get("underlying") or chain[0].get("symbol") or ""
        spot = _get_spot_price(underlying) or 0.0
        selected = _select_by_moneyness(chain, target_delta, "call", spot)
    if selected is None:
        return {
            "symbol": chain[0].get("symbol") if chain else "",
            "expiration": chain[0].get("expiration") if chain else None,
            "strike": chain[0].get("strike") if chain else None,
            "type": "call",
            "delta": None,
        }
    return selected


def _get_spot_price(symbol: str) -> Optional[float]:
    """Best-effort last daily close, used to find the strike closest to ATM."""
    try:
        import yfinance as yf

        data = yf.download(symbol, period="5d", interval="1d", progress=False)
        if data.empty:
            return None
        close = data["Close"].iloc[-1]
        return float(close.item() if hasattr(close, "item") else close)
    except Exception:
        return None


def _build_osi_symbol(symbol: str, expiration: str, option_type: str, strike: float) -> str:
    """Build a standard OSI option symbol, e.g. AAPL260717C00325000."""
    exp_date = datetime.fromisoformat(expiration).date()
    exp_part = exp_date.strftime("%y%m%d")
    type_char = "C" if option_type == "call" else "P"
    strike_part = f"{int(round(strike * 1000)):08d}"
    return f"{symbol.upper()}{exp_part}{type_char}{strike_part}"


def get_target_option_contract(
    symbol: str,
    signal_direction: str,
    expiry_days_out: int = 7,
) -> Optional[str]:
    """
    Resolve the OSI symbol for the strike closest to at-the-money, at the
    expiration closest to expiry_days_out, for a directional signal.

    signal_direction: 'LONG' -> call, 'SHORT' -> put.
    Returns None if no chain, no matching expiration, or no spot price is
    available. Note: the chain returned by get_option_chain doesn't carry
    volume/open-interest, so this picks nearest-to-spot strike only — it
    does not filter by liquidity.
    """
    direction = str(signal_direction).upper()
    if direction not in {"LONG", "SHORT"}:
        raise ValueError(f"signal_direction must be 'LONG' or 'SHORT', got {signal_direction!r}")
    option_type = "call" if direction == "LONG" else "put"

    chain = get_option_chain(symbol)
    if not chain:
        return None

    today = datetime.utcnow().date()

    def _dte(exp_str: str) -> Optional[int]:
        try:
            return (datetime.fromisoformat(exp_str).date() - today).days
        except Exception:
            return None

    expirations = {row.get("expiration") for row in chain if row.get("expiration")}
    valid_expirations = [exp for exp in expirations if _dte(exp) is not None]
    if not valid_expirations:
        return None

    target_expiry = min(valid_expirations, key=lambda exp: abs(_dte(exp) - expiry_days_out))

    candidates = [
        row for row in chain
        if row.get("expiration") == target_expiry
        and row.get("type") == option_type
        and row.get("strike") is not None
    ]
    if not candidates:
        return None

    spot_price = _get_spot_price(symbol)
    if spot_price is None:
        return None

    best = min(candidates, key=lambda row: abs(float(row["strike"]) - spot_price))
    return _build_osi_symbol(symbol, target_expiry, option_type, float(best["strike"]))
