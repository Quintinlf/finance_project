"""
Transaction cost model.

Until now nothing in the engine charged anything for trading: no spread, no
slippage, no fees. On a $500 account targeting 4% moves that omission is not a
rounding error — crossing a 60bp spread on a thin commodity ETF costs 120bp per
round trip, which is 30% of the target. A backtest without costs will report an
edge that does not survive contact with a real order book.

Two numbers, kept separate on purpose:

    spread_bps    half the quoted bid-ask, paid on entry AND exit. This is the
                  cost of crossing to the other side of the book.
    slippage_bps  everything else — the market moving between the decision and
                  the fill, and market impact. Charged per fill, like spread.

Commissions are $0 at Alpaca for US equities and ETFs, but the field exists so
the model stays correct if the broker ever changes.

The default spread estimates below are ESTIMATES, not measurements, and are
deliberately conservative. Replace them with real numbers by running
``measure_spreads()`` during regular market hours — it writes a JSON snapshot
that ``load_measured_spreads()`` picks up automatically. Quotes taken outside
09:30-16:00 ET are worthless for this (after-hours books show 6-10% spreads on
names that trade at 1bp intraday), so the measurement refuses to run then.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, time
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Optional

from zoneinfo import ZoneInfo

Side = Literal["buy", "sell"]

MEASURED_SPREADS_PATH = Path("trade_logs") / "measured_spreads.json"

_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)
_EASTERN = ZoneInfo("US/Eastern")

# Conservative per-symbol estimates in basis points of the FULL quoted spread.
# Tiering rationale: mega-cap equities and the largest metal trusts quote inside
# a couple of bp; sector and broad-commodity funds a few bp; single-commodity
# Teucrium-style funds are small and thin and routinely quote 20-60bp.
# These are placeholders for measurement — see the module docstring.
DEFAULT_SPREAD_BPS: Dict[str, float] = {
    # Mega-cap equities
    "AAPL": 1.0, "MSFT": 1.0, "AMZN": 1.5, "NVDA": 1.0, "GOOGL": 1.5,
    # Large, liquid commodity trusts
    "GLD": 1.5, "IAU": 3.0, "SLV": 3.0, "XLE": 2.0,
    # Broad commodity baskets
    "DBC": 6.0, "GSG": 8.0, "DBA": 8.0,
    # Energy
    "USO": 5.0, "BNO": 15.0, "UNG": 8.0, "UGA": 30.0,
    # Industrial and secondary precious metals
    "CPER": 15.0, "DBB": 30.0, "PPLT": 20.0, "PALL": 30.0,
    # Single-commodity agriculture — the thinnest names in the universe
    "WEAT": 20.0, "CORN": 20.0, "SOYB": 35.0, "CANE": 35.0,
}

# Used when a symbol has neither a measurement nor a default. Deliberately
# pessimistic: an unknown instrument is more likely thin than liquid.
FALLBACK_SPREAD_BPS = 40.0

# Slippage beyond the quoted spread. Market orders on a small account are not
# moving these books, so this covers decision-to-fill drift more than impact.
DEFAULT_SLIPPAGE_BPS = 5.0


@dataclass(frozen=True)
class CostModel:
    """Per-symbol cost assumptions and the fill arithmetic that uses them."""

    spread_bps: float
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    commission_per_order: float = 0.0
    commission_bps: float = 0.0
    source: str = "default_estimate"

    def __post_init__(self) -> None:
        for name in ("spread_bps", "slippage_bps", "commission_bps", "commission_per_order"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
            object.__setattr__(self, name, value)

    @property
    def one_way_bps(self) -> float:
        """Cost of a single fill: half the spread, plus slippage and fees."""
        return self.spread_bps / 2.0 + self.slippage_bps + self.commission_bps

    @property
    def round_trip_bps(self) -> float:
        """Cost of a complete entry-and-exit cycle."""
        return 2.0 * self.one_way_bps

    def fill_price(self, reference_price: float, side: Side) -> float:
        """Where a market order actually fills, relative to the mid/reference.

        Buys fill above the reference, sells below — you always cross the spread
        in the direction that costs you.
        """
        price = float(reference_price)
        if price <= 0:
            raise ValueError(f"reference_price must be > 0, got {price}")
        adjustment = 1.0 + self.one_way_bps / 10_000.0
        if str(side).lower() == "buy":
            return price * adjustment
        if str(side).lower() == "sell":
            return price / adjustment
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    def fee_for_order(self, notional: float) -> float:
        """Flat per-order commission. Separate from the bps costs baked into fills."""
        return self.commission_per_order if abs(float(notional)) > 0 else 0.0

    def round_trip_cost(self, notional: float) -> float:
        """Total expected cost, in dollars, of entering and exiting at ``notional``."""
        value = abs(float(notional))
        return value * self.round_trip_bps / 10_000.0 + 2.0 * self.fee_for_order(value)

    def breakeven_move_pct(self) -> float:
        """Percent price move required just to cover a round trip."""
        return self.round_trip_bps / 100.0


def for_symbol(
    symbol: str,
    *,
    measured: Optional[Mapping[str, float]] = None,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> CostModel:
    """Build the cost model for one symbol, preferring measured spreads."""
    key = str(symbol).strip().upper()

    if measured and key in measured:
        return CostModel(
            spread_bps=float(measured[key]),
            slippage_bps=slippage_bps,
            source="measured",
        )
    if key in DEFAULT_SPREAD_BPS:
        return CostModel(
            spread_bps=DEFAULT_SPREAD_BPS[key],
            slippage_bps=slippage_bps,
            source="default_estimate",
        )
    return CostModel(
        spread_bps=FALLBACK_SPREAD_BPS,
        slippage_bps=slippage_bps,
        source="fallback",
    )


def zero_cost_model() -> CostModel:
    """A frictionless model. Only for isolating cost impact in comparisons."""
    return CostModel(spread_bps=0.0, slippage_bps=0.0, source="zero")


def scale(model: CostModel, factor: float) -> CostModel:
    """Scale a model's variable costs — useful for stress-testing assumptions."""
    return replace(
        model,
        spread_bps=model.spread_bps * float(factor),
        slippage_bps=model.slippage_bps * float(factor),
        source=f"{model.source}_x{factor:g}",
    )


# ========================================================================
# Spread measurement
# ========================================================================


def is_regular_market_hours(now: Optional[datetime] = None) -> bool:
    """True only during 09:30-16:00 ET on a weekday.

    Quotes outside this window are unusable for spread estimation — the book is
    thin or stale and shows spreads orders of magnitude wider than the real
    intraday cost.
    """
    moment = (now or datetime.now(tz=_EASTERN)).astimezone(_EASTERN)
    if moment.weekday() >= 5:
        return False
    return _MARKET_OPEN <= moment.time() <= _MARKET_CLOSE


def measure_spreads(
    symbols: Iterable[str],
    *,
    force: bool = False,
    save_to: Optional[Path] = MEASURED_SPREADS_PATH,
) -> Dict[str, float]:
    """Sample live quoted spreads in bps and persist them.

    Refuses to run outside regular hours unless ``force=True``, because the
    resulting numbers would be garbage that then silently poisons every
    backtest that loads them.
    """
    wanted = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not wanted:
        return {}

    if not is_regular_market_hours() and not force:
        raise RuntimeError(
            "Refusing to measure spreads outside 09:30-16:00 ET — after-hours "
            "quotes are not representative. Pass force=True to override."
        )

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        from logic.alpaca_exercises import load_alpaca_creds

        creds = load_alpaca_creds()
        client = StockHistoricalDataClient(creds.api_key, creds.secret_key)
        quotes = client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=wanted)
        )
    except Exception as exc:
        logging.warning("Spread measurement failed (%s).", exc)
        return {}

    measured: Dict[str, float] = {}
    for symbol in wanted:
        quote = quotes.get(symbol)
        if quote is None:
            continue
        try:
            bid = float(quote.bid_price or 0.0)
            ask = float(quote.ask_price or 0.0)
        except (TypeError, ValueError):
            continue
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (ask + bid) / 2.0
        measured[symbol] = (ask - bid) / mid * 10_000.0

    if save_to is not None and measured:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "measured_at": datetime.now(tz=_EASTERN).isoformat(),
            "spreads_bps": measured,
        }
        existing = _read_snapshot(save_to)
        if existing:
            merged = dict(existing.get("spreads_bps") or {})
            merged.update(measured)
            payload["spreads_bps"] = merged
        save_to.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return measured


def _read_snapshot(path: Path) -> Optional[Dict]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception:
        return None


def load_measured_spreads(path: Path = MEASURED_SPREADS_PATH) -> Dict[str, float]:
    """Load persisted spread measurements, or an empty map if none exist."""
    snapshot = _read_snapshot(path)
    if not snapshot:
        return {}
    raw = snapshot.get("spreads_bps") or {}
    out: Dict[str, float] = {}
    for symbol, value in raw.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            out[str(symbol).upper()] = parsed
    return out
