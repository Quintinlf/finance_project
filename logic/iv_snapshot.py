"""Daily implied-volatility and vol-forecast snapshots. Phase 0a of the Tier 2 scope.

Alpaca serves option chains as *current snapshots only* — there is no historical
IV endpoint, and yfinance has none either. So the data needed to answer Gate 0
("does our volatility forecast beat implied volatility?") does not exist yet and
cannot be bought back cheaply. It has to be accumulated going forward, which
means the clock starts the day this module first runs and not a day earlier.

Two things are recorded, and the second one is the one that is easy to get wrong:

1. **Implied volatility**, per contract, with greeks and the bid/ask quote. The
   quote matters as much as the IV: an option that prices attractively and
   quotes 0.83/1.03 is a 21% round trip, and Gate 1 kills the strategy on cost
   long before Gate 0 gets a say.

2. **Our own forecast, as it stood on the day.** This is not redundant with
   being able to recompute it later from price history. Recomputing it in six
   months would fit the estimator with knowledge of the period it is being
   tested on, and the resulting number would be lookahead-biased garbage that
   looks like a result. Point-in-time capture is the only version of this test
   that means anything. ``forecast_method`` records *which* estimator produced
   the number, so changing the estimator later does not silently pool two
   different series into one sample.

Nothing here trades, sizes, or influences a decision. It only writes rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from logic.sqlite_store import DEFAULT_DB_PATH, connect, utc_now_iso

# The estimator whose persistence was actually measured (r = 0.708): trailing
# 20-day realized vol predicting forward realized vol. Recorded alongside every
# forecast so a future change of estimator is visible in the data rather than
# silently mixed into it.
FORECAST_METHOD = "trailing_realized_vol_20d"

# Contracts this far from spot tell us nothing about at-the-money implied vol
# and triple the row count. Deep wings also carry the widest spreads and the
# least reliable IV.
MAX_MONEYNESS = 0.25

# Below this there is no point running Gate 0. Overlapping 20-day windows mean
# the effective sample is far smaller than the row count, and this project has
# already learned twice what small-n findings are worth: every dramatic early
# accuracy claim evaporated at n=22k, and the base rate itself moved 24 points.
# ~126 trading days is six months of daily snapshots, which given ~20-day
# overlap is on the order of only 6 independent observations per symbol — thin
# even then, which is exactly why the honest answer may be to buy history.
MIN_TRADING_DAYS_FOR_GATE0 = 126


@dataclass(frozen=True)
class SnapshotResult:
    """What one run of the snapshotter managed to record."""

    snapshot_date: str
    symbols_attempted: int
    symbols_with_iv: int
    contracts_written: int
    forecasts_written: int
    failures: Dict[str, str]

    @property
    def ok(self) -> bool:
        return self.contracts_written > 0 and self.forecasts_written > 0

    def summary(self) -> str:
        return (
            f"IV SNAPSHOT {self.snapshot_date}: {self.contracts_written} contracts "
            f"across {self.symbols_with_iv}/{self.symbols_attempted} symbols, "
            f"{self.forecasts_written} forecasts"
            + (f", {len(self.failures)} failed" if self.failures else "")
        )


def init_iv_tables(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
    """Create the snapshot tables. Safe to call repeatedly."""
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS iv_snapshots (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date      TEXT NOT NULL,
                captured_at_utc    TEXT NOT NULL,
                underlying         TEXT NOT NULL,
                spot               REAL,
                contract_symbol    TEXT NOT NULL,
                expiration         TEXT,
                dte                INTEGER,
                strike             REAL,
                option_type        TEXT,
                moneyness          REAL,
                implied_volatility REAL,
                delta              REAL,
                gamma              REAL,
                theta              REAL,
                vega               REAL,
                bid                REAL,
                ask                REAL,
                mid                REAL,
                spread_pct_of_mid  REAL,
                feed               TEXT,
                quote_timestamp    TEXT,
                UNIQUE(snapshot_date, contract_symbol)
            );

            CREATE INDEX IF NOT EXISTS idx_iv_snap_date
                ON iv_snapshots(snapshot_date);
            CREATE INDEX IF NOT EXISTS idx_iv_snap_underlying
                ON iv_snapshots(underlying, snapshot_date);

            -- Our forecast, as it stood on the snapshot date. See module
            -- docstring: recomputing this later is lookahead bias, not a test.
            CREATE TABLE IF NOT EXISTS vol_forecasts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date   TEXT NOT NULL,
                captured_at_utc TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                forecast_vol    REAL,
                forecast_method TEXT NOT NULL,
                lookback_days   INTEGER,
                spot            REAL,
                UNIQUE(snapshot_date, symbol, forecast_method)
            );

            CREATE INDEX IF NOT EXISTS idx_vol_fc_symbol
                ON vol_forecasts(symbol, snapshot_date);
            """
        )


def _spread_pct_of_mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    """Full quoted spread as a percentage of mid.

    This is the number Gate 1 lives or dies on, so it is computed once here
    rather than in each caller. A crossed or zero-bid quote returns None instead
    of a huge or negative number that would look like data.
    """
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid * 100.0


def _rows_for_symbol(
    symbol: str,
    *,
    snapshot_date: str,
    captured_at: str,
    max_moneyness: float,
) -> List[Dict[str, Any]]:
    """Build snapshot rows for one underlying, IV-bearing contracts only."""
    from logic.options_engine import _get_spot_price, get_option_chain

    spot = _get_spot_price(symbol)
    chain = get_option_chain(symbol)
    if not chain:
        return []

    today = date.fromisoformat(snapshot_date)
    rows: List[Dict[str, Any]] = []
    for c in chain:
        iv = c.get("implied_volatility")
        if iv is None:
            # Roughly half the INDICATIVE chain comes back without IV. Those
            # rows carry no information for Gate 0 and would only inflate counts.
            continue

        strike = c.get("strike")
        moneyness = None
        if strike and spot and spot > 0:
            moneyness = (float(strike) - float(spot)) / float(spot)
            if abs(moneyness) > max_moneyness:
                continue

        dte = None
        exp = c.get("expiration")
        if exp:
            try:
                dte = (date.fromisoformat(str(exp)) - today).days
            except ValueError:
                dte = None

        bid, ask = c.get("bid"), c.get("ask")
        mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None

        rows.append(
            {
                "snapshot_date": snapshot_date,
                "captured_at_utc": captured_at,
                "underlying": symbol,
                "spot": spot,
                "contract_symbol": c.get("symbol"),
                "expiration": exp,
                "dte": dte,
                "strike": strike,
                "option_type": c.get("type"),
                "moneyness": moneyness,
                "implied_volatility": float(iv),
                "delta": c.get("delta"),
                "gamma": c.get("gamma"),
                "theta": c.get("theta"),
                "vega": c.get("vega"),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct_of_mid": _spread_pct_of_mid(bid, ask),
                "feed": "indicative",
                "quote_timestamp": c.get("quote_timestamp"),
            }
        )
    return rows


def snapshot_universe(
    symbols: Optional[Sequence[str]] = None,
    *,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    snapshot_date: Optional[str] = None,
    max_moneyness: float = MAX_MONEYNESS,
) -> SnapshotResult:
    """Record today's IV chain and today's vol forecast for each symbol.

    Failures are collected per symbol rather than raised: a chain that does not
    resolve should cost that one symbol's row, not the whole day's snapshot.
    Re-running on the same date is idempotent — the UNIQUE constraints mean a
    second run updates rather than duplicating.
    """
    from logic.volatility_sizing import DEFAULT_LOOKBACK, predict_volatility

    if symbols is None:
        from logic.universe import get_symbols

        symbols = get_symbols("all")

    snap_date = snapshot_date or datetime.now(timezone.utc).date().isoformat()
    captured_at = utc_now_iso()

    init_iv_tables(db_path)

    failures: Dict[str, str] = {}
    all_rows: List[Dict[str, Any]] = []
    forecasts: List[Dict[str, Any]] = []
    with_iv = 0

    for symbol in symbols:
        # Bound before the try: on a chain failure the forecast block below
        # still runs, and an unbound name there would turn a recoverable chain
        # outage into a lost forecast row -- silently biasing the eventual
        # sample against exactly the days the options feed was unhealthy.
        rows: List[Dict[str, Any]] = []
        try:
            rows = _rows_for_symbol(
                symbol,
                snapshot_date=snap_date,
                captured_at=captured_at,
                max_moneyness=max_moneyness,
            )
            if rows:
                with_iv += 1
                all_rows.extend(rows)
            else:
                logging.info("IV snapshot: no IV-bearing contracts for %s", symbol)
        except Exception as exc:  # noqa: BLE001
            failures[symbol] = f"{type(exc).__name__}: {exc}"
            logging.warning("IV snapshot failed for %s (%s)", symbol, exc)

        # The forecast is recorded even when the chain fails. The two series are
        # independent, and a gap in one should not punch a hole in the other.
        try:
            fc = predict_volatility(symbol)
            spot = rows[0]["spot"] if rows else None
            forecasts.append(
                {
                    "snapshot_date": snap_date,
                    "captured_at_utc": captured_at,
                    "symbol": symbol,
                    "forecast_vol": fc,
                    "forecast_method": FORECAST_METHOD,
                    "lookback_days": DEFAULT_LOOKBACK,
                    "spot": spot,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures[f"{symbol}:forecast"] = f"{type(exc).__name__}: {exc}"

    written = _write_rows(all_rows, forecasts, db_path=db_path)
    result = SnapshotResult(
        snapshot_date=snap_date,
        symbols_attempted=len(symbols),
        symbols_with_iv=with_iv,
        contracts_written=written[0],
        forecasts_written=written[1],
        failures=failures,
    )
    logging.info(result.summary())
    return result


def _write_rows(
    iv_rows: Iterable[Dict[str, Any]],
    forecast_rows: Iterable[Dict[str, Any]],
    *,
    db_path: Union[str, Path],
) -> tuple[int, int]:
    iv_rows = list(iv_rows)
    forecast_rows = list(forecast_rows)
    with connect(db_path) as conn:
        if iv_rows:
            conn.executemany(
                """
                INSERT INTO iv_snapshots (
                    snapshot_date, captured_at_utc, underlying, spot,
                    contract_symbol, expiration, dte, strike, option_type,
                    moneyness, implied_volatility, delta, gamma, theta, vega,
                    bid, ask, mid, spread_pct_of_mid, feed, quote_timestamp
                ) VALUES (
                    :snapshot_date, :captured_at_utc, :underlying, :spot,
                    :contract_symbol, :expiration, :dte, :strike, :option_type,
                    :moneyness, :implied_volatility, :delta, :gamma, :theta, :vega,
                    :bid, :ask, :mid, :spread_pct_of_mid, :feed, :quote_timestamp
                )
                ON CONFLICT(snapshot_date, contract_symbol) DO UPDATE SET
                    implied_volatility = excluded.implied_volatility,
                    delta = excluded.delta,
                    bid = excluded.bid,
                    ask = excluded.ask,
                    mid = excluded.mid,
                    spread_pct_of_mid = excluded.spread_pct_of_mid,
                    captured_at_utc = excluded.captured_at_utc
                """,
                iv_rows,
            )
        if forecast_rows:
            conn.executemany(
                """
                INSERT INTO vol_forecasts (
                    snapshot_date, captured_at_utc, symbol,
                    forecast_vol, forecast_method, lookback_days, spot
                ) VALUES (
                    :snapshot_date, :captured_at_utc, :symbol,
                    :forecast_vol, :forecast_method, :lookback_days, :spot
                )
                ON CONFLICT(snapshot_date, symbol, forecast_method) DO UPDATE SET
                    forecast_vol = excluded.forecast_vol,
                    captured_at_utc = excluded.captured_at_utc
                """,
                forecast_rows,
            )
    return len(iv_rows), len(forecast_rows)


def atm_iv_by_symbol(
    snapshot_date: str,
    *,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    target_dte: int = 30,
) -> Dict[str, float]:
    """At-the-money IV per symbol for one date, nearest the target expiry.

    "At the money" is the smallest |moneyness| available rather than an exact
    strike match, because these chains are sparse and an exact match usually
    does not exist.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT underlying, implied_volatility
            FROM iv_snapshots
            WHERE snapshot_date = ?
              AND implied_volatility IS NOT NULL
              AND moneyness IS NOT NULL
              AND dte IS NOT NULL
              AND dte > 0
            GROUP BY underlying
            HAVING MIN(ABS(dte - ?) * 1000 + ABS(moneyness) * 100)
            """,
            (snapshot_date, target_dte),
        ).fetchall()
    return {r["underlying"]: float(r["implied_volatility"]) for r in rows}


def coverage(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """How much history has accumulated, and whether Gate 0 can run yet."""
    init_iv_tables(db_path)
    with connect(db_path) as conn:
        iv = conn.execute(
            """
            SELECT COUNT(DISTINCT snapshot_date) AS days,
                   COUNT(*) AS rows,
                   MIN(snapshot_date) AS first_date,
                   MAX(snapshot_date) AS last_date,
                   COUNT(DISTINCT underlying) AS symbols
            FROM iv_snapshots
            """
        ).fetchone()
        fc = conn.execute(
            "SELECT COUNT(DISTINCT snapshot_date) AS days, COUNT(*) AS rows FROM vol_forecasts"
        ).fetchone()

    days = int(iv["days"] or 0)
    return {
        "iv_days": days,
        "iv_rows": int(iv["rows"] or 0),
        "iv_symbols": int(iv["symbols"] or 0),
        "first_date": iv["first_date"],
        "last_date": iv["last_date"],
        "forecast_days": int(fc["days"] or 0),
        "forecast_rows": int(fc["rows"] or 0),
        "days_required": MIN_TRADING_DAYS_FOR_GATE0,
        "days_remaining": max(0, MIN_TRADING_DAYS_FOR_GATE0 - days),
        "gate0_ready": days >= MIN_TRADING_DAYS_FOR_GATE0,
    }
