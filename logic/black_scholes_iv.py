"""Implied volatility by inverting Black-Scholes on an EOD option quote.

ThetaData's EOD endpoint carries prices and the closing NBBO, but implied
volatility is a Standard/Pro-only endpoint. That is not a reason to pay the
higher tier: IV is not data, it is a function of data. Given spot, strike, time,
rate and the option's price, there is exactly one volatility that reproduces
that price, and finding it is a one-dimensional root find.

### Why bisection rather than Newton

Newton on vega converges in three or four steps and fails badly on precisely the
contracts this ingest will be full of: deep wings and near-expiry contracts where
vega approaches zero, so the Newton step explodes. Bisection on a bracket is
slower and completely reliable, and we are inverting a few hundred thousand
quotes once, offline. Reliability is free here and speed is worth nothing.

### The assumptions, stated rather than buried

Three inputs are not observable in the data and have to be assumed:

1. **Risk-free rate.** Rates went from ~0.3% to over 5% across the window this
   ingest covers, so a single constant would be a real bias, not a rounding
   error. ``ANNUAL_RISK_FREE`` holds annual averages of the 3-month bill. That
   is coarse -- it is not a term structure and not a daily curve -- but it is
   the difference between a few basis points of error and several hundred.

2. **Dividend yield.** These are commodity ETFs. GLD, SLV and IAU pay nothing;
   XLE pays a real yield. ``ANNUAL_DIVIDEND_YIELD`` carries a per-symbol figure
   and defaults to zero, which is right for the metals and wrong-but-small for
   the rest.

3. **Early exercise.** US ETF options are American. We price them as European.
   For calls on a non-dividend-paying underlying this is exact; for puts and for
   dividend payers it understates value, so the inverted IV is slightly low. The
   error is largest deep in the money, which is one more reason the ingest keeps
   only near-the-money strikes.

None of these break Gate 0, because Gate 0 is a question about *incremental
information* in a nested regression, and a smooth bias in the level of implied
vol is absorbed by the regression's slope on implied vol. What would break it is
an inconsistent bias -- which is why every number here is a fixed function of
(symbol, date) and never of anything fitted.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional, Union

# Annual averages of the 3-month US Treasury bill, used as the risk-free rate
# for contracts in that year. Coarse on purpose: see module docstring.
ANNUAL_RISK_FREE = {
    2012: 0.0009, 2013: 0.0006, 2014: 0.0003, 2015: 0.0005, 2016: 0.0032,
    2017: 0.0093, 2018: 0.0194, 2019: 0.0206, 2020: 0.0037, 2021: 0.0004,
    2022: 0.0202, 2023: 0.0507, 2024: 0.0515, 2025: 0.0430, 2026: 0.0380,
}
FALLBACK_RISK_FREE = 0.0400

# Commodity ETF distribution yields. Zero is correct for the metals trusts,
# which hold bullion and distribute nothing.
ANNUAL_DIVIDEND_YIELD = {
    "GLD": 0.0, "IAU": 0.0, "SLV": 0.0, "PPLT": 0.0, "PALL": 0.0,
    "USO": 0.0, "UNG": 0.0, "BNO": 0.0, "UGA": 0.0,
    "CORN": 0.0, "WEAT": 0.0, "SOYB": 0.0, "CANE": 0.0, "DBA": 0.0,
    "DBB": 0.0, "CPER": 0.0,
    "DBC": 0.0280, "GSG": 0.0250, "XLE": 0.0320,
}

TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0

# Bracket for the root find. An option implying more than 500% vol is a broken
# quote, not a volatile underlying, and returning None is the honest answer.
MIN_VOL, MAX_VOL = 1e-4, 5.0
TOLERANCE = 1e-6
MAX_ITERATIONS = 100


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def risk_free_rate(as_of: Union[str, date]) -> float:
    """The assumed annualized risk-free rate for a given date."""
    if isinstance(as_of, date):
        year = as_of.year
    else:
        try:
            year = int(str(as_of)[:4])
        except (TypeError, ValueError):
            return FALLBACK_RISK_FREE
    return ANNUAL_RISK_FREE.get(year, FALLBACK_RISK_FREE)


def dividend_yield(symbol: str) -> float:
    return ANNUAL_DIVIDEND_YIELD.get(str(symbol).upper(), 0.0)


def bs_price(
    spot: float,
    strike: float,
    years: float,
    vol: float,
    rate: float,
    *,
    option_type: str = "call",
    div_yield: float = 0.0,
) -> Optional[float]:
    """Black-Scholes-Merton price with continuous dividend yield."""
    if spot <= 0 or strike <= 0 or years <= 0 or vol <= 0:
        return None

    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol * vol) * years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc_r = math.exp(-rate * years)
    disc_q = math.exp(-div_yield * years)

    if str(option_type).lower().startswith("c"):
        return spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
    return strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1)


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    years: float,
    *,
    option_type: str = "call",
    rate: float = FALLBACK_RISK_FREE,
    div_yield: float = 0.0,
) -> Optional[float]:
    """The volatility that reproduces ``price``, or None if none can.

    Returns None rather than a number whenever the quote cannot be inverted:
    non-positive price, expired contract, a price below intrinsic value (an
    arbitrage that in practice means a stale or crossed EOD quote), or a price
    above what even 500% vol would justify. Each of those is a data-quality
    filter, and letting any of them through as a plausible-looking float is how
    a sample gets quietly poisoned.
    """
    if price is None or price <= 0 or spot <= 0 or strike <= 0 or years <= 0:
        return None

    lo_price = bs_price(spot, strike, years, MIN_VOL, rate,
                        option_type=option_type, div_yield=div_yield)
    hi_price = bs_price(spot, strike, years, MAX_VOL, rate,
                        option_type=option_type, div_yield=div_yield)
    if lo_price is None or hi_price is None:
        return None

    # Below the zero-vol floor the quote implies negative time value; above the
    # ceiling it implies more than MAX_VOL. Neither is invertible.
    if price <= lo_price + TOLERANCE or price >= hi_price:
        return None

    lo, hi = MIN_VOL, MAX_VOL
    for _ in range(MAX_ITERATIONS):
        mid = 0.5 * (lo + hi)
        guess = bs_price(spot, strike, years, mid, rate,
                         option_type=option_type, div_yield=div_yield)
        if guess is None:
            return None
        if abs(guess - price) < TOLERANCE or (hi - lo) < TOLERANCE:
            return mid
        # Price is monotonically increasing in vol, so the sign of the residual
        # tells us which half of the bracket holds the root.
        if guess < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def implied_vol_from_quote(
    *,
    symbol: str,
    snapshot_date: Union[str, date],
    expiration: Union[str, date],
    strike: float,
    option_type: str,
    spot: float,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    close: Optional[float] = None,
) -> Optional[float]:
    """Invert the mid quote for one contract, falling back to the close.

    The mid of the closing NBBO is preferred over the last trade price: the last
    trade on a thin commodity-ETF option may be hours stale and on the wrong
    side of a book that has since moved, whereas the NBBO is the price at which
    the contract was actually biddable at the close. ``close`` is used only when
    no two-sided quote exists.
    """
    years = year_fraction(snapshot_date, expiration)
    if years is None or years <= 0:
        return None

    price: Optional[float] = None
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        price = (float(bid) + float(ask)) / 2.0
    elif close is not None and float(close) > 0:
        price = float(close)
    if price is None:
        return None

    return implied_volatility(
        price,
        float(spot),
        float(strike),
        years,
        option_type=option_type,
        rate=risk_free_rate(snapshot_date),
        div_yield=dividend_yield(symbol),
    )


def year_fraction(
    as_of: Union[str, date], expiration: Union[str, date]
) -> Optional[float]:
    """Time to expiry in years, on a calendar-day basis.

    Calendar days rather than trading days, because that is the convention the
    options market quotes IV on. Using 252 here would make our inverted IV
    systematically disagree with every published figure by about 20%, which
    would be a self-inflicted wound in a test whose entire purpose is comparing
    our number to the market's.
    """
    d0 = _as_date(as_of)
    d1 = _as_date(expiration)
    if d0 is None or d1 is None:
        return None
    days = (d1 - d0).days
    if days <= 0:
        return None
    return days / CALENDAR_DAYS_PER_YEAR


def _as_date(value: Union[str, date, None]) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if len(s) == 8 and s.isdigit():
        s = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None
