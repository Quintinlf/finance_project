"""Batched, resumable backfill of historical option EOD into ``iv_snapshots``.

This exists to collapse a timeline. Accumulating daily snapshots reaches a
conclusive 30-day Gate 0 verdict in **April 2029**; ingesting purchased history
reaches it in an afternoon. The destination is the schema Phase 0a already
writes, so ``scripts/analyze_vol_edge.py`` runs against the result unchanged.

### The honest part: is a backfilled forecast lookahead-biased?

``logic/iv_snapshot`` warns, correctly, that recomputing our forecast later
would be lookahead bias rather than a test. That warning is about **fitted**
estimators: anything whose parameters are chosen with knowledge of the period it
is scored on is cheating, and no amount of care fixes it after the fact.

Our estimator is not fitted. It is trailing 20-day realized volatility -- a
fixed, parameter-free function of the 21 closes before the snapshot date. There
is nothing to fit, so recomputing it from a strictly truncated history gives the
number we *would* have produced on that day, not a number informed by the
future. Two safeguards keep that claim true rather than merely intended:

* ``_history_as_of`` slices to bars dated **on or before** the snapshot date and
  is the only way this module obtains prices. ``predict_volatility`` is then
  called with that frame, so the backfilled series goes through the identical
  code path as the live one.
* The rows are labelled ``FORECAST_METHOD_BACKFILL``, distinct from the live
  ``FORECAST_METHOD``. The two never pool silently. If the backfilled and
  live-captured series ever disagree on an overlapping date, that is a bug worth
  finding, and the labels are what make it findable.

What is genuinely *not* recoverable this way is implied vol, which is why it has
to be bought: no amount of cleverness reconstructs what the options market
quoted on a Tuesday in 2019.

### Resumability

A multi-year pull across eight symbols is thousands of requests at 2-8
concurrent, so it will be interrupted. ``ingest_progress`` records each
completed (symbol, chunk) so a re-run skips finished work instead of re-pulling
it. Chunks are recorded only after their rows commit, so an interrupted chunk is
retried rather than silently skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from logic.black_scholes_iv import implied_vol_from_quote
from logic.iv_snapshot import MAX_MONEYNESS, init_iv_tables
from logic.sqlite_store import DEFAULT_DB_PATH, connect, utc_now_iso
from logic.thetadata_client import ThetaDataClient, ThetaDataUnavailable

# Distinguishes purchased EOD history from the live INDICATIVE snapshots. The
# two are different measurement processes -- different time of day, different
# IV derivation -- and mixing them in one regression would pool two samples.
FEED = "thetadata_eod"

# Deliberately NOT equal to iv_snapshot.FORECAST_METHOD. See module docstring.
FORECAST_METHOD_BACKFILL = "trailing_realized_vol_20d_backfill"

# Server-side filters. These are the whole reason a month's subscription is
# enough: without them, expiration=* strike=* returns the full chain for every
# listed expiry on every date, overwhelmingly deep wings we would discard.
DEFAULT_STRIKE_RANGE = 6
DEFAULT_MAX_DTE = 60

# One request per symbol-month. Small enough that an interruption loses little,
# large enough that per-request overhead does not dominate.
DEFAULT_CHUNK_DAYS = 31

__all__ = [
    "FEED",
    "FORECAST_METHOD_BACKFILL",
    "IngestStats",
    "ingest_symbol",
    "ingest_universe",
    "init_progress_table",
]


@dataclass
class IngestStats:
    symbols: int = 0
    chunks_done: int = 0
    chunks_skipped: int = 0
    rows_fetched: int = 0
    contracts_written: int = 0
    forecasts_written: int = 0
    inversion_failures: int = 0
    no_spot: int = 0
    failures: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        inverted = self.contracts_written
        attempted = inverted + self.inversion_failures
        rate = f"{inverted / attempted:.0%}" if attempted else "n/a"
        return (
            f"INGEST: {self.contracts_written} contracts and "
            f"{self.forecasts_written} forecasts across {self.symbols} symbols; "
            f"{self.chunks_done} chunks pulled, {self.chunks_skipped} already done; "
            f"inversion rate {rate}"
            + (f"; {len(self.failures)} failures" if self.failures else "")
        )


def init_progress_table(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ingest_progress (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source       TEXT NOT NULL,
                symbol       TEXT NOT NULL,
                chunk_start  TEXT NOT NULL,
                chunk_end    TEXT NOT NULL,
                rows_written INTEGER,
                completed_at TEXT NOT NULL,
                UNIQUE(source, symbol, chunk_start, chunk_end)
            );
            """
        )


def _chunk_done(
    symbol: str, start: date, end: date, *, db_path: Union[str, Path]
) -> bool:
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM ingest_progress
            WHERE source = ? AND symbol = ? AND chunk_start = ? AND chunk_end = ?
            """,
            (FEED, symbol, start.isoformat(), end.isoformat()),
        ).fetchone()
    return row is not None


def _mark_chunk_done(
    symbol: str, start: date, end: date, rows: int, *, db_path: Union[str, Path]
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ingest_progress
                (source, symbol, chunk_start, chunk_end, rows_written, completed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, symbol, chunk_start, chunk_end) DO UPDATE SET
                rows_written = excluded.rows_written,
                completed_at = excluded.completed_at
            """,
            (FEED, symbol, start.isoformat(), end.isoformat(), rows, utc_now_iso()),
        )


def _chunks(start: date, end: date, chunk_days: int) -> List[Tuple[date, date]]:
    out = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=chunk_days - 1), end)
        out.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return out


# ------------------------------------------------------------------- prices


def _load_full_history(symbol: str, start: date, end: date):
    """Daily bars spanning the ingest window, fetched once per symbol."""
    from logic.price_cache import get_history

    return get_history(symbol, start=start - timedelta(days=90), end=end)


def _history_as_of(history, as_of: str):
    """Bars dated on or before ``as_of``. The only price access in this module.

    This slice is what makes the backfilled forecast point-in-time rather than
    lookahead-biased, so it is a single chokepoint by design: nothing else here
    touches a price frame.
    """
    if history is None or len(history) == 0:
        return None
    try:
        return history.loc[:as_of]
    except Exception:  # noqa: BLE001
        return None


def _close_on(history, as_of: str) -> Optional[float]:
    """The close on ``as_of`` exactly, or None.

    Deliberately not the most recent close at or before the date: pairing an
    option quote from one day with a spot from another would corrupt moneyness
    and every inverted IV built on it.
    """
    sliced = _history_as_of(history, as_of)
    if sliced is None or len(sliced) == 0:
        return None
    try:
        label = str(sliced.index[-1])[:10]
    except Exception:  # noqa: BLE001
        return None
    if label != as_of:
        return None
    try:
        return float(sliced["Close"].iloc[-1])
    except Exception:  # noqa: BLE001
        return None


# -------------------------------------------------------------------- rows


def _rows_from_eod(
    raw: Iterable[Dict[str, Any]],
    *,
    symbol: str,
    history,
    captured_at: str,
    max_moneyness: float,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Turn EOD bars into ``iv_snapshots`` rows, inverting IV per contract.

    Returns (rows, inversion_failures, dates_without_spot).
    """
    rows: List[Dict[str, Any]] = []
    failures = 0
    missing_spot = 0
    spot_cache: Dict[str, Optional[float]] = {}

    for r in raw:
        snap = _iso(r.get("date")) or _iso(r.get("created"))
        exp = _iso(r.get("expiration"))
        strike = _f(r.get("strike"))
        if not snap or not exp or not strike:
            continue

        if snap not in spot_cache:
            spot_cache[snap] = _close_on(history, snap)
            if spot_cache[snap] is None:
                missing_spot += 1
        spot = spot_cache[snap]
        if spot is None or spot <= 0:
            continue

        moneyness = (strike - spot) / spot
        if abs(moneyness) > max_moneyness:
            continue

        try:
            dte = (date.fromisoformat(exp) - date.fromisoformat(snap)).days
        except ValueError:
            continue
        if dte <= 0:
            continue

        right = _normalize_right(r.get("right"))
        bid, ask = _f(r.get("bid")), _f(r.get("ask"))
        close = _f(r.get("close"))
        iv = implied_vol_from_quote(
            symbol=symbol,
            snapshot_date=snap,
            expiration=exp,
            strike=strike,
            option_type=right,
            spot=spot,
            bid=bid,
            ask=ask,
            close=close,
        )
        if iv is None:
            # A quote that will not invert carries no information for Gate 0.
            # Storing it as a NULL IV would only inflate row counts, and the
            # project has already learned what inflated counts are worth.
            failures += 1
            continue

        mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
        rows.append(
            {
                "snapshot_date": snap,
                "captured_at_utc": captured_at,
                "underlying": symbol,
                "spot": spot,
                "contract_symbol": _contract_symbol(symbol, exp, right, strike),
                "expiration": exp,
                "dte": dte,
                "strike": strike,
                "option_type": right,
                "moneyness": moneyness,
                "implied_volatility": iv,
                # The EOD endpoint carries no greeks. Left NULL rather than
                # computed, so a later analysis cannot mistake a Black-Scholes
                # artefact of our own making for an observed quantity.
                "delta": None,
                "gamma": None,
                "theta": None,
                "vega": None,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct_of_mid": _spread_pct(bid, ask),
                "feed": FEED,
                "quote_timestamp": r.get("created"),
            }
        )

    return rows, failures, missing_spot


def _backfill_forecast_rows(
    symbol: str,
    snapshot_dates: Sequence[str],
    *,
    history,
    captured_at: str,
) -> List[Dict[str, Any]]:
    """Point-in-time trailing-vol forecasts for each date, from truncated history."""
    from logic.volatility_sizing import DEFAULT_LOOKBACK, predict_volatility

    out = []
    for snap in sorted(set(snapshot_dates)):
        sliced = _history_as_of(history, snap)
        if sliced is None or len(sliced) == 0:
            continue
        # No minimum-bar check here on purpose. predict_volatility already
        # requires lookback + 1 bars and returns None otherwise, so a second
        # threshold would be a competing source of truth about how much history
        # the estimator needs -- and a stricter one would silently discard the
        # earliest dates of every ingest window for no gain.
        fc = predict_volatility(symbol, price_history=sliced)
        if fc is None:
            continue
        try:
            spot = float(sliced["Close"].iloc[-1])
        except Exception:  # noqa: BLE001
            spot = None
        out.append(
            {
                "snapshot_date": snap,
                "captured_at_utc": captured_at,
                "symbol": symbol,
                "forecast_vol": fc,
                "forecast_method": FORECAST_METHOD_BACKFILL,
                "lookback_days": DEFAULT_LOOKBACK,
                "spot": spot,
            }
        )
    return out


# ------------------------------------------------------------------ writing


def _write(
    iv_rows: Sequence[Dict[str, Any]],
    forecast_rows: Sequence[Dict[str, Any]],
    *,
    db_path: Union[str, Path],
) -> Tuple[int, int]:
    """Reuses the Phase 0a upserts so both sources write identically."""
    from logic.iv_snapshot import _write_rows

    return _write_rows(iv_rows, forecast_rows, db_path=db_path)


# ------------------------------------------------------------------- driver


def ingest_symbol(
    client: ThetaDataClient,
    symbol: str,
    *,
    start: date,
    end: date,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    strike_range: int = DEFAULT_STRIKE_RANGE,
    max_dte: int = DEFAULT_MAX_DTE,
    max_moneyness: float = MAX_MONEYNESS,
    stats: Optional[IngestStats] = None,
    dry_run: bool = False,
) -> IngestStats:
    """Pull one symbol's option history, chunk by chunk, resuming where possible."""
    stats = stats or IngestStats()
    stats.symbols += 1
    captured_at = utc_now_iso()
    history = _load_full_history(symbol, start, end)
    if history is None or len(history) == 0:
        stats.failures[symbol] = "no price history; cannot resolve spot or forecast"
        return stats

    for chunk_start, chunk_end in _chunks(start, end, chunk_days):
        if not dry_run and _chunk_done(symbol, chunk_start, chunk_end, db_path=db_path):
            stats.chunks_skipped += 1
            continue

        try:
            raw = client.option_history_eod(
                symbol,
                start=chunk_start,
                end=chunk_end,
                strike_range=strike_range,
                max_dte=max_dte,
            )
        except ThetaDataUnavailable as exc:
            stats.failures[f"{symbol}:{chunk_start}"] = str(exc)
            logging.warning("ingest %s %s..%s failed: %s", symbol, chunk_start, chunk_end, exc)
            continue

        stats.rows_fetched += len(raw)
        rows, inv_failures, missing_spot = _rows_from_eod(
            raw,
            symbol=symbol,
            history=history,
            captured_at=captured_at,
            max_moneyness=max_moneyness,
        )
        stats.inversion_failures += inv_failures
        stats.no_spot += missing_spot

        forecasts = _backfill_forecast_rows(
            symbol,
            [r["snapshot_date"] for r in rows],
            history=history,
            captured_at=captured_at,
        )

        if dry_run:
            stats.chunks_done += 1
            stats.contracts_written += len(rows)
            stats.forecasts_written += len(forecasts)
            continue

        n_iv, n_fc = _write(rows, forecasts, db_path=db_path)
        stats.contracts_written += n_iv
        stats.forecasts_written += n_fc
        # Marked only after the rows commit, so an interrupted chunk is retried
        # rather than skipped as if it had succeeded.
        _mark_chunk_done(symbol, chunk_start, chunk_end, n_iv, db_path=db_path)
        stats.chunks_done += 1
        logging.info(
            "ingest %s %s..%s: %d rows, %d forecasts", symbol, chunk_start, chunk_end, n_iv, n_fc
        )

    return stats


def ingest_universe(
    symbols: Sequence[str],
    *,
    start: date,
    end: date,
    client: Optional[ThetaDataClient] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    dry_run: bool = False,
    **kwargs: Any,
) -> IngestStats:
    client = client or ThetaDataClient()
    init_iv_tables(db_path)
    init_progress_table(db_path)

    stats = IngestStats()
    for symbol in symbols:
        ingest_symbol(
            client, symbol, start=start, end=end, db_path=db_path,
            stats=stats, dry_run=dry_run, **kwargs,
        )
    logging.info(stats.summary())
    return stats


# ------------------------------------------------------------------ helpers


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> Optional[str]:
    from logic.thetadata_client import _to_iso_date

    return _to_iso_date(value)


def _normalize_right(value: Any) -> str:
    s = str(value or "call").strip().lower()
    return "put" if s.startswith("p") else "call"


def _spread_pct(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    from logic.iv_snapshot import _spread_pct_of_mid

    return _spread_pct_of_mid(bid, ask)


def _contract_symbol(symbol: str, expiration: str, right: str, strike: float) -> str:
    """OCC-style identifier, built because the EOD rows carry no contract id.

    ``iv_snapshots`` is keyed UNIQUE(snapshot_date, contract_symbol), so this
    must be deterministic: the same contract pulled twice has to collide and
    upsert rather than duplicate.
    """
    yy_mm_dd = expiration.replace("-", "")[2:]
    letter = "P" if right == "put" else "C"
    strike_thousandths = int(round(float(strike) * 1000))
    return f"{symbol}{yy_mm_dd}{letter}{strike_thousandths:08d}"
