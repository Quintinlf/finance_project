"""
Tradeable universe definition and affordability screening.

Two jobs, deliberately together:

1. **What we are allowed to trade.** Assets carry metadata (asset class, sector,
   underlying exposure, theme tags) rather than being bare ticker strings. The
   theme tags exist so a future news/macro analyst can answer "a conflict is
   disrupting grain exports — which of my tradeable assets does that touch?"
   without hardcoding a symbol list at the news layer. ``THEMES`` is the shared
   vocabulary for that join.

2. **What we can actually afford.** The engine sizes positions as
   ``max_position_pct_of_equity`` of account equity, so a symbol whose share
   price exceeds that cap can never fill — ``build_order_plan`` computes
   ``max_shares = 0`` and ``enforce_risk_limits`` rejects with
   ``position_below_minimum``. Screening up front turns that silent per-symbol
   rejection into one explicit, readable verdict table.

Commodities here are **ETFs, not futures**: the only broker adapter is Alpaca
(``logic/broker_client.py``), which trades equities, options, and crypto — no
futures. WEAT/CORN/USO/GLD-style funds give commodity exposure through the
existing plumbing. Real contracts (ZW, ZC, CL) would need a second broker
adapter plus contract/margin/roll handling in the risk layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

AssetClass = Literal["equity", "commodity_etf", "sector_etf", "broad_etf"]
Sector = Literal[
    "technology",
    "agriculture",
    "energy",
    "precious_metals",
    "industrial_metals",
    "broad_commodity",
]

# Shared theme vocabulary. The news/macro analyst emits themes from this set and
# joins them to assets; keeping it closed means a typo fails loudly instead of
# silently matching nothing.
THEMES: Tuple[str, ...] = (
    "grains",
    "softs",
    "food_security",
    "crude",
    "natgas",
    "refined_products",
    "opec",
    "geopolitical_supply_shock",
    "safe_haven",
    "inflation_hedge",
    "dollar_weakness",
    "industrial_demand",
    "china_demand",
    "energy_transition",
    "weather_drought",
    "mega_cap_tech",
    "ai_capex",
    "consumer_demand",
)


@dataclass(frozen=True)
class Asset:
    """One tradeable instrument plus the metadata the analysts key off."""

    symbol: str
    name: str
    asset_class: AssetClass
    sector: Sector
    underlying: str
    themes: Tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = [t for t in self.themes if t not in THEMES]
        if unknown:
            raise ValueError(
                f"{self.symbol}: unknown theme(s) {unknown}; add them to universe.THEMES first"
            )


# --- Agriculture -----------------------------------------------------------
# The "war disrupts grain exports" case lives here. Wheat and corn are the most
# conflict-sensitive listed ag exposures available as US ETFs.
_AGRICULTURE: Tuple[Asset, ...] = (
    Asset("WEAT", "Teucrium Wheat Fund", "commodity_etf", "agriculture", "wheat futures",
          ("grains", "food_security", "geopolitical_supply_shock", "weather_drought")),
    Asset("CORN", "Teucrium Corn Fund", "commodity_etf", "agriculture", "corn futures",
          ("grains", "food_security", "geopolitical_supply_shock", "weather_drought")),
    Asset("SOYB", "Teucrium Soybean Fund", "commodity_etf", "agriculture", "soybean futures",
          ("grains", "food_security", "china_demand", "weather_drought")),
    Asset("CANE", "Teucrium Sugar Fund", "commodity_etf", "agriculture", "sugar futures",
          ("softs", "weather_drought")),
    Asset("DBA", "Invesco DB Agriculture Fund", "commodity_etf", "agriculture", "diversified ag basket",
          ("grains", "softs", "food_security", "inflation_hedge")),
)

# --- Energy ----------------------------------------------------------------
_ENERGY: Tuple[Asset, ...] = (
    Asset("UNG", "United States Natural Gas Fund", "commodity_etf", "energy", "natural gas futures",
          ("natgas", "geopolitical_supply_shock", "weather_drought")),
    Asset("BNO", "United States Brent Oil Fund", "commodity_etf", "energy", "Brent crude futures",
          ("crude", "opec", "geopolitical_supply_shock")),
    Asset("USO", "United States Oil Fund", "commodity_etf", "energy", "WTI crude futures",
          ("crude", "opec", "geopolitical_supply_shock")),
    Asset("UGA", "United States Gasoline Fund", "commodity_etf", "energy", "RBOB gasoline futures",
          ("refined_products", "crude", "consumer_demand")),
    Asset("XLE", "Energy Select Sector SPDR", "sector_etf", "energy", "US energy equities",
          ("crude", "opec", "energy_transition")),
)

# --- Metals ----------------------------------------------------------------
_METALS: Tuple[Asset, ...] = (
    Asset("SLV", "iShares Silver Trust", "commodity_etf", "precious_metals", "physical silver",
          ("safe_haven", "inflation_hedge", "industrial_demand", "energy_transition")),
    Asset("IAU", "iShares Gold Trust", "commodity_etf", "precious_metals", "physical gold",
          ("safe_haven", "inflation_hedge", "dollar_weakness", "geopolitical_supply_shock")),
    Asset("GLD", "SPDR Gold Shares", "commodity_etf", "precious_metals", "physical gold",
          ("safe_haven", "inflation_hedge", "dollar_weakness", "geopolitical_supply_shock")),
    Asset("PPLT", "abrdn Platinum Shares", "commodity_etf", "precious_metals", "physical platinum",
          ("industrial_demand", "inflation_hedge")),
    Asset("PALL", "abrdn Palladium Shares", "commodity_etf", "precious_metals", "physical palladium",
          ("industrial_demand",)),
    Asset("CPER", "United States Copper Index Fund", "commodity_etf", "industrial_metals", "copper futures",
          ("industrial_demand", "china_demand", "energy_transition")),
    Asset("DBB", "Invesco DB Base Metals Fund", "commodity_etf", "industrial_metals", "base metals basket",
          ("industrial_demand", "china_demand")),
)

# --- Broad commodity -------------------------------------------------------
_BROAD_COMMODITY: Tuple[Asset, ...] = (
    Asset("DBC", "Invesco DB Commodity Index Fund", "broad_etf", "broad_commodity", "diversified commodity basket",
          ("inflation_hedge", "crude", "grains", "industrial_demand")),
    Asset("GSG", "iShares S&P GSCI Commodity-Indexed Trust", "broad_etf", "broad_commodity", "GSCI commodity index",
          ("inflation_hedge", "crude", "grains")),
)

# --- Equities (the pre-existing default universe) --------------------------
_EQUITIES: Tuple[Asset, ...] = (
    Asset("AAPL", "Apple Inc.", "equity", "technology", "consumer hardware",
          ("mega_cap_tech", "consumer_demand", "china_demand")),
    Asset("MSFT", "Microsoft Corp.", "equity", "technology", "software and cloud",
          ("mega_cap_tech", "ai_capex")),
    Asset("AMZN", "Amazon.com Inc.", "equity", "technology", "e-commerce and cloud",
          ("mega_cap_tech", "ai_capex", "consumer_demand")),
    Asset("NVDA", "NVIDIA Corp.", "equity", "technology", "AI accelerators",
          ("mega_cap_tech", "ai_capex", "china_demand")),
    Asset("GOOGL", "Alphabet Inc.", "equity", "technology", "search and cloud",
          ("mega_cap_tech", "ai_capex")),
)

COMMODITIES: Tuple[Asset, ...] = _AGRICULTURE + _ENERGY + _METALS + _BROAD_COMMODITY
EQUITIES: Tuple[Asset, ...] = _EQUITIES
ALL_ASSETS: Tuple[Asset, ...] = EQUITIES + COMMODITIES

UniverseScope = Literal["equities", "commodities", "agriculture", "energy", "metals", "all"]

_SCOPES: Dict[str, Tuple[Asset, ...]] = {
    "equities": EQUITIES,
    "commodities": COMMODITIES,
    "agriculture": _AGRICULTURE,
    "energy": _ENERGY,
    "metals": _METALS,
    "all": ALL_ASSETS,
}

_BY_SYMBOL: Dict[str, Asset] = {a.symbol: a for a in ALL_ASSETS}


def get_assets(scope: str = "all") -> Tuple[Asset, ...]:
    """Return the assets for a named scope."""
    key = str(scope).strip().lower()
    if key not in _SCOPES:
        raise ValueError(f"Unknown universe scope {scope!r}; expected one of {sorted(_SCOPES)}")
    return _SCOPES[key]


def get_symbols(scope: str = "all") -> List[str]:
    """Return just the tickers for a named scope."""
    return [a.symbol for a in get_assets(scope)]


def lookup(symbol: str) -> Optional[Asset]:
    """Return asset metadata for a ticker, or None if it isn't in the universe."""
    return _BY_SYMBOL.get(str(symbol).strip().upper())


def assets_for_theme(theme: str) -> List[Asset]:
    """Every asset tagged with a theme. The join the news analyst will use."""
    if theme not in THEMES:
        raise ValueError(f"Unknown theme {theme!r}; expected one of {list(THEMES)}")
    return [a for a in ALL_ASSETS if theme in a.themes]


# ========================================================================
# Inverse proxies — making bearish signals actionable without shorting
# ========================================================================
#
# The account is long-only (shorting is disabled in ExecutionConfig and the
# Alpaca account reports shorting_enabled=False), so a SELL on a symbol we do
# not hold is unexecutable and gets dropped. Buying an inverse fund expresses
# the same bearish view with a long position.
#
# Three things make these NOT drop-in mirrors, and the trading layer has to
# respect all three:
#
# 1. **Leverage.** Almost every listed inverse commodity product is -2x. A 2%
#    stop on a -2x fund is triggered by a 1% move in the underlying, so exit
#    percentages must be divided by `leverage` to preserve the intent of the
#    signal that generated them. See `scale_exits_for_leverage`.
# 2. **Daily reset.** Leveraged funds rebalance daily and track the *daily*
#    return, not the holding-period return. Over a choppy multi-day hold they
#    decay relative to the naive inverse. Average holding here is ~9 days, so
#    this is a real cost, not a footnote.
# 3. **Coverage is partial.** No liquid inverse product exists for agriculture,
#    platinum, palladium, industrial metals, or broad commodity baskets. Those
#    SELL signals stay unactionable, and `coverage_report` says so explicitly
#    rather than letting it look like everything is handled.
#
# Listings change — ETNs get called, funds close. Nothing here is trusted
# blindly: `resolve_inverse` price-checks the proxy before it is routed to.


@dataclass(frozen=True)
class InverseProxy:
    """A long instrument that expresses a bearish view on `underlying_symbol`."""

    symbol: str
    name: str
    underlying_symbol: str
    leverage: float          # magnitude of inverse exposure: 1.0, 2.0, 3.0
    instrument: Literal["etf", "etn"]
    note: str = ""

    def __post_init__(self) -> None:
        if self.leverage <= 0:
            raise ValueError(f"{self.symbol}: leverage must be positive, got {self.leverage}")


# Keyed by the symbol the SELL signal fired on.
INVERSE_PROXIES: Dict[str, InverseProxy] = {
    # --- Energy ---
    "UNG": InverseProxy("KOLD", "ProShares UltraShort Bloomberg Natural Gas", "UNG", 2.0, "etf",
                        "natural gas is the most volatile underlying here; -2x compounds that"),
    "USO": InverseProxy("SCO", "ProShares UltraShort Bloomberg Crude Oil", "USO", 2.0, "etf"),
    "BNO": InverseProxy("SCO", "ProShares UltraShort Bloomberg Crude Oil", "BNO", 2.0, "etf",
                        "SCO tracks WTI; BNO is Brent, so the hedge is imperfect"),
    "UGA": InverseProxy("SCO", "ProShares UltraShort Bloomberg Crude Oil", "UGA", 2.0, "etf",
                        "gasoline proxied by crude; crack spread moves are unhedged"),
    "XLE": InverseProxy("DUG", "ProShares UltraShort Oil & Gas", "XLE", 2.0, "etf"),
    # --- Precious metals ---
    "SLV": InverseProxy("ZSL", "ProShares UltraShort Silver", "SLV", 2.0, "etf"),
    "GLD": InverseProxy("GLL", "ProShares UltraShort Gold", "GLD", 2.0, "etf"),
    "IAU": InverseProxy("GLL", "ProShares UltraShort Gold", "IAU", 2.0, "etf"),
    # --- Equities ---
    # Single-stock inverse funds exist for some mega caps but are thin; the
    # Nasdaq-100 short is the liquid way to be bearish on this basket, at the
    # cost of trading the index rather than the name.
    "AAPL": InverseProxy("PSQ", "ProShares Short QQQ", "AAPL", 1.0, "etf", "index proxy, not single-name"),
    "MSFT": InverseProxy("PSQ", "ProShares Short QQQ", "MSFT", 1.0, "etf", "index proxy, not single-name"),
    "AMZN": InverseProxy("PSQ", "ProShares Short QQQ", "AMZN", 1.0, "etf", "index proxy, not single-name"),
    "NVDA": InverseProxy("PSQ", "ProShares Short QQQ", "NVDA", 1.0, "etf", "index proxy, not single-name"),
    "GOOGL": InverseProxy("PSQ", "ProShares Short QQQ", "GOOGL", 1.0, "etf", "index proxy, not single-name"),
}

# Symbols with no inverse product worth trading. Listed explicitly so the gap is
# visible in the coverage report instead of being inferred from a missing key.
NO_INVERSE_COVERAGE: Dict[str, str] = {
    "WEAT": "no liquid inverse wheat fund listed in the US",
    "CORN": "no liquid inverse corn fund listed in the US",
    "SOYB": "no liquid inverse soybean fund listed in the US",
    "CANE": "no liquid inverse sugar fund listed in the US",
    "DBA": "no liquid inverse agriculture basket listed in the US",
    "PPLT": "no liquid inverse platinum fund listed in the US",
    "PALL": "no liquid inverse palladium fund listed in the US",
    "CPER": "no liquid inverse copper fund listed in the US",
    "DBB": "no liquid inverse base-metals fund listed in the US",
    "DBC": "no liquid inverse broad-commodity fund listed in the US",
    "GSG": "no liquid inverse broad-commodity fund listed in the US",
}


def inverse_for(symbol: str) -> Optional[InverseProxy]:
    """The inverse proxy for a symbol, or None when none is mapped."""
    return INVERSE_PROXIES.get(str(symbol).strip().upper())


def inverse_symbols(scope: str = "all") -> List[str]:
    """Distinct inverse tickers needed to cover a scope, in a stable order."""
    seen: List[str] = []
    for symbol in get_symbols(scope):
        proxy = inverse_for(symbol)
        if proxy is not None and proxy.symbol not in seen:
            seen.append(proxy.symbol)
    return seen


def scale_exits_for_leverage(
    tp_pct: float, sl_pct: float, leverage: float
) -> Tuple[float, float]:
    """Convert exit percentages expressed on the underlying to the proxy's scale.

    A signal saying "take profit after a 4% fall" must become a 2% target on a
    -2x fund, because the fund moves twice as far. Without this the trade exits
    on half the underlying move the signal actually asked for.
    """
    factor = max(float(leverage), 1e-9)
    return float(tp_pct) / factor, float(sl_pct) / factor


def coverage_report(scope: str = "all") -> str:
    """Human-readable table of which symbols can express a bearish view."""
    lines = [f"{'SYMBOL':<8} {'INVERSE':<9} {'LEV':<5} STATUS"]
    for symbol in get_symbols(scope):
        proxy = inverse_for(symbol)
        if proxy is None:
            reason = NO_INVERSE_COVERAGE.get(symbol, "unmapped")
            lines.append(f"{symbol:<8} {'-':<9} {'-':<5} NO COVERAGE ({reason})")
        else:
            lines.append(
                f"{symbol:<8} {proxy.symbol:<9} {proxy.leverage:<5.1f} "
                f"ok{(' - ' + proxy.note) if proxy.note else ''}"
            )
    return "\n".join(lines)


def resolve_inverse(
    symbol: str,
    price_lookup: Optional[Callable[[Sequence[str]], Dict[str, Optional[float]]]] = None,
) -> Tuple[Optional[InverseProxy], str]:
    """Return the tradable inverse proxy for `symbol`, plus why if there isn't one.

    Price-checks the proxy before returning it. Inverse products are unusually
    prone to closure and reverse splits — a stale ticker in the table would
    otherwise turn into a rejected order at the broker with a confusing message.
    """
    proxy = inverse_for(symbol)
    if proxy is None:
        return None, NO_INVERSE_COVERAGE.get(
            str(symbol).strip().upper(), f"no inverse proxy mapped for {symbol}"
        )

    lookup_fn = price_lookup or fetch_last_prices
    try:
        prices = lookup_fn([proxy.symbol])
    except Exception as exc:
        return None, f"{proxy.symbol}: price lookup failed ({exc})"

    price = prices.get(proxy.symbol)
    if price is None or price <= 0:
        return None, f"{proxy.symbol}: no current price (delisted or untradable?)"
    return proxy, f"{proxy.symbol} @ ${price:,.2f} (-{proxy.leverage:g}x {symbol})"


# ========================================================================
# Affordability screening
# ========================================================================


@dataclass(frozen=True)
class AffordabilityVerdict:
    """Why a symbol can or cannot be bought at the current account size."""

    symbol: str
    price: Optional[float]
    max_position_dollars: float
    max_shares: int
    affordable: bool
    reason: str


def fetch_last_prices(symbols: Sequence[str]) -> Dict[str, Optional[float]]:
    """Best-effort last daily close per symbol, batched through yfinance.

    Returns None for any symbol that fails to resolve (delisted, bad ticker,
    network error) rather than raising — an unpriceable symbol is simply
    unaffordable, which is the same outcome the caller needs.
    """
    wanted = [str(s).strip().upper() for s in symbols if str(s).strip()]
    prices: Dict[str, Optional[float]] = {s: None for s in wanted}
    if not wanted:
        return prices

    try:
        import yfinance as yf

        data = yf.download(
            wanted,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="column",
        )
    except Exception as exc:
        logging.warning("Price fetch failed for affordability screen (%s).", exc)
        return prices

    if data is None or len(data) == 0:
        return prices

    try:
        closes = data["Close"]
    except Exception:
        return prices

    # Single-symbol downloads collapse to a Series; multi-symbol give a frame.
    if hasattr(closes, "columns"):
        for symbol in wanted:
            if symbol in closes.columns:
                series = closes[symbol].dropna()
                if len(series):
                    prices[symbol] = float(series.iloc[-1])
    else:
        series = closes.dropna()
        if len(series) and len(wanted) == 1:
            prices[wanted[0]] = float(series.iloc[-1])

    return prices


def screen_affordability(
    symbols: Sequence[str],
    *,
    account_equity: float,
    max_position_fraction: float,
    price_lookup: Optional[Callable[[Sequence[str]], Mapping[str, Optional[float]]]] = None,
) -> List[AffordabilityVerdict]:
    """Decide, per symbol, whether one share fits inside the position cap.

    This mirrors the arithmetic in ``build_order_plan`` and
    ``enforce_risk_limits`` (``max_dollar_position = equity * max_position_size``)
    so the screen and the executor agree. Verdicts are returned for every
    symbol, affordable or not, so the rejected ones stay visible in the run log
    instead of vanishing.
    """
    lookup_fn = price_lookup or fetch_last_prices
    wanted = [str(s).strip().upper() for s in symbols if str(s).strip()]
    prices = dict(lookup_fn(wanted))

    equity = float(account_equity)
    fraction = float(max_position_fraction)
    budget = equity * fraction

    verdicts: List[AffordabilityVerdict] = []
    for symbol in wanted:
        price = prices.get(symbol)

        if price is None or price <= 0:
            verdicts.append(
                AffordabilityVerdict(
                    symbol=symbol,
                    price=None,
                    max_position_dollars=budget,
                    max_shares=0,
                    affordable=False,
                    reason="no price available",
                )
            )
            continue

        max_shares = int(budget / price)
        if max_shares < 1:
            verdicts.append(
                AffordabilityVerdict(
                    symbol=symbol,
                    price=price,
                    max_position_dollars=budget,
                    max_shares=0,
                    affordable=False,
                    reason=(
                        f"1 share costs ${price:,.2f} but the cap is "
                        f"${budget:,.2f} ({fraction:.0%} of ${equity:,.2f})"
                    ),
                )
            )
            continue

        verdicts.append(
            AffordabilityVerdict(
                symbol=symbol,
                price=price,
                max_position_dollars=budget,
                max_shares=max_shares,
                affordable=True,
                reason=f"up to {max_shares} share(s) at ${price:,.2f}",
            )
        )

    return verdicts


def build_tradeable_universe(
    *,
    scope: str = "all",
    account_equity: float,
    max_position_fraction: float,
    always_include: Iterable[str] = (),
    price_lookup: Optional[Callable[[Sequence[str]], Mapping[str, Optional[float]]]] = None,
) -> Tuple[List[str], List[AffordabilityVerdict]]:
    """Resolve a scope to the symbols worth running signals on.

    ``always_include`` is for symbols we already hold. An unaffordable holding
    still needs signals generated so it can be *sold* — the position cap gates
    opening a position, not closing one — so held symbols bypass the screen.

    Returns ``(symbols, verdicts)``. Verdicts cover the scope only, so the
    caller can log exactly what was screened out and why.
    """
    scope_symbols = get_symbols(scope)
    held = [str(s).strip().upper() for s in always_include if str(s).strip()]

    verdicts = screen_affordability(
        scope_symbols,
        account_equity=account_equity,
        max_position_fraction=max_position_fraction,
        price_lookup=price_lookup,
    )

    tradeable = [v.symbol for v in verdicts if v.affordable]
    for symbol in held:
        if symbol not in tradeable:
            tradeable.append(symbol)

    return tradeable, verdicts


def format_affordability_table(verdicts: Sequence[AffordabilityVerdict]) -> str:
    """Render verdicts as aligned text for the run log."""
    if not verdicts:
        return "(no symbols screened)"

    ordered = sorted(verdicts, key=lambda v: (not v.affordable, v.price or float("inf")))
    width = max(len(v.symbol) for v in ordered)
    lines = []
    for v in ordered:
        mark = "OK " if v.affordable else "-- "
        price = f"${v.price:,.2f}" if v.price is not None else "n/a"
        lines.append(f"  {mark}{v.symbol:<{width}}  {price:>10}  {v.reason}")
    return "\n".join(lines)
