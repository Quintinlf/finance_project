"""Tests for FIFO fill matching and realized-P&L persistence."""

import sqlite3
from datetime import datetime, timedelta

import pytest

from logic.fill_reconciler import (
    Fill,
    ReconcileResult,
    RoundTrip,
    match_fifo,
    persist_round_trips,
)
from logic.sqlite_store import init_db

BASE = datetime(2026, 3, 24, 13, 30)


def _fill(symbol, side, qty, price, day_offset=0, order_id=None):
    return Fill(
        order_id=order_id or f"{symbol}-{side}-{day_offset}-{qty}",
        symbol=symbol,
        side=side,
        quantity=qty,
        price=price,
        filled_at=BASE + timedelta(days=day_offset),
    )


# --- Fill validation -------------------------------------------------------


def test_nonsense_fills_are_rejected():
    with pytest.raises(ValueError, match="quantity must be"):
        _fill("WEAT", "buy", 0, 20.0)
    with pytest.raises(ValueError, match="price must be"):
        _fill("WEAT", "buy", 1, 0.0)
    with pytest.raises(ValueError, match="side must be"):
        _fill("WEAT", "hold", 1, 20.0)


# --- FIFO matching ---------------------------------------------------------


def test_simple_round_trip_computes_real_pnl():
    """The TSLA trade the live system never recorded."""
    fills = [
        _fill("TSLA", "buy", 1, 379.74, 0),
        _fill("TSLA", "sell", 1, 337.00, 121),
    ]
    trips, open_lots, unmatched = match_fifo(fills)

    assert len(trips) == 1 and not open_lots and not unmatched
    assert trips[0].pnl == pytest.approx(-42.74)
    assert trips[0].return_pct == pytest.approx(-11.255, abs=0.01)
    assert trips[0].holding_days == 121


def test_an_unsold_buy_stays_an_open_lot():
    trips, open_lots, _ = match_fifo([_fill("BAC", "buy", 2, 47.38)])
    assert not trips
    assert len(open_lots) == 1
    assert open_lots[0].quantity == 2 and open_lots[0].price == 47.38


def test_fifo_consumes_the_oldest_lot_first():
    """Averaging instead of FIFO would report the wrong P&L here."""
    fills = [
        _fill("WEAT", "buy", 1, 20.0, 0),
        _fill("WEAT", "buy", 1, 30.0, 1),
        _fill("WEAT", "sell", 1, 25.0, 2),
    ]
    trips, open_lots, _ = match_fifo(fills)

    assert len(trips) == 1
    assert trips[0].entry_price == 20.0   # oldest lot, not the $25 average
    assert trips[0].pnl == pytest.approx(5.0)
    assert open_lots[0].price == 30.0


def test_a_partial_sell_splits_a_lot():
    fills = [
        _fill("CORN", "buy", 3, 18.0, 0),
        _fill("CORN", "sell", 1, 20.0, 1),
    ]
    trips, open_lots, _ = match_fifo(fills)

    assert len(trips) == 1 and trips[0].quantity == 1
    assert len(open_lots) == 1 and open_lots[0].quantity == pytest.approx(2.0)


def test_one_sell_spanning_two_lots_produces_two_round_trips():
    fills = [
        _fill("CANE", "buy", 1, 9.0, 0),
        _fill("CANE", "buy", 2, 10.0, 1),
        _fill("CANE", "sell", 3, 11.0, 2),
    ]
    trips, open_lots, _ = match_fifo(fills)

    assert len(trips) == 2 and not open_lots
    assert sum(t.pnl for t in trips) == pytest.approx(2.0 + 2.0)


def test_symbols_are_matched_independently():
    fills = [
        _fill("WEAT", "buy", 1, 20.0, 0),
        _fill("CORN", "buy", 1, 18.0, 0),
        _fill("CORN", "sell", 1, 19.0, 1),
    ]
    trips, open_lots, _ = match_fifo(fills)

    assert len(trips) == 1 and trips[0].symbol == "CORN"
    assert len(open_lots) == 1 and open_lots[0].symbol == "WEAT"


def test_a_sell_without_a_lot_is_surfaced_not_swallowed():
    trips, _, unmatched = match_fifo([_fill("GLD", "sell", 1, 370.0)])
    assert not trips
    assert len(unmatched) == 1 and unmatched[0].symbol == "GLD"


def test_oversized_sell_reports_only_the_excess_as_unmatched():
    fills = [
        _fill("SLV", "buy", 1, 50.0, 0),
        _fill("SLV", "sell", 3, 52.0, 1),
    ]
    trips, _, unmatched = match_fifo(fills)

    assert len(trips) == 1 and trips[0].quantity == 1
    assert unmatched[0].quantity == pytest.approx(2.0)


def test_fractional_quantities_match_cleanly():
    fills = [
        _fill("MSFT", "buy", 0.25, 400.0, 0),
        _fill("MSFT", "sell", 0.25, 420.0, 1),
    ]
    trips, open_lots, unmatched = match_fifo(fills)

    assert len(trips) == 1 and not open_lots and not unmatched
    assert trips[0].pnl == pytest.approx(5.0)


def test_matching_is_ordered_by_fill_time_not_input_order():
    later = _fill("WEAT", "buy", 1, 30.0, 5)
    earlier = _fill("WEAT", "buy", 1, 20.0, 0)
    sell = _fill("WEAT", "sell", 1, 25.0, 9)
    trips, _, _ = match_fifo([later, sell, earlier])
    assert trips[0].entry_price == 20.0


def test_no_fills_yields_nothing():
    trips, open_lots, unmatched = match_fifo([])
    assert not trips and not open_lots and not unmatched


# --- Aggregate result ------------------------------------------------------


def test_result_aggregates_pnl_and_win_rate():
    trips, _, _ = match_fifo([
        _fill("A", "buy", 1, 10.0, 0), _fill("A", "sell", 1, 12.0, 1),
        _fill("B", "buy", 1, 10.0, 0), _fill("B", "sell", 1, 9.0, 1),
        _fill("C", "buy", 1, 10.0, 0), _fill("C", "sell", 1, 11.0, 1),
    ])
    result = ReconcileResult(round_trips=trips)

    assert result.realized_pnl == pytest.approx(2.0)
    assert result.wins == 2 and result.losses == 1
    assert result.win_rate == pytest.approx(66.667, abs=0.01)


def test_a_flat_trade_counts_as_a_loss_not_a_win():
    """Breakeven before costs is not a win — costs make it negative."""
    trips, _, _ = match_fifo([
        _fill("A", "buy", 1, 10.0, 0), _fill("A", "sell", 1, 10.0, 1),
    ])
    assert ReconcileResult(round_trips=trips).wins == 0


# --- Persistence -----------------------------------------------------------


def _trip(symbol="TSLA", entry=379.74, exit_=337.0, qty=1.0, eid="e1", xid="x1"):
    return RoundTrip(
        symbol=symbol, quantity=qty,
        entry_price=entry, entry_at=BASE, entry_order_id=eid,
        exit_price=exit_, exit_at=BASE + timedelta(days=120), exit_order_id=xid,
    )


def test_persisted_trade_carries_the_real_pnl(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    persist_round_trips([_trip()], account_id="acct", db_path=db)

    rows = sqlite3.connect(db).execute(
        "SELECT symbol, qty, entry_price, exit_price, pnl, status FROM trades"
    ).fetchall()
    assert len(rows) == 1
    symbol, qty, entry, exit_, pnl, status = rows[0]
    assert (symbol, qty, status) == ("TSLA", 1.0, "CLOSED")
    assert pnl == pytest.approx(-42.74)
    assert entry != exit_, "entry and exit must differ or P&L is zero by construction"


def test_reconciling_twice_updates_rather_than_duplicates(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    persist_round_trips([_trip()], account_id="acct", db_path=db)
    persist_round_trips([_trip()], account_id="acct", db_path=db)

    count = sqlite3.connect(db).execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert count == 1


def test_a_corrected_exit_price_overwrites_the_old_row(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    persist_round_trips([_trip(exit_=337.0)], account_id="acct", db_path=db)
    persist_round_trips([_trip(exit_=340.0)], account_id="acct", db_path=db)

    pnl = sqlite3.connect(db).execute("SELECT pnl FROM trades").fetchone()[0]
    assert pnl == pytest.approx(-39.74)


def test_distinct_round_trips_get_distinct_rows(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    persist_round_trips(
        [_trip(eid="e1", xid="x1"), _trip(symbol="BAC", eid="e2", xid="x2")],
        account_id="acct", db_path=db,
    )
    count = sqlite3.connect(db).execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert count == 2
