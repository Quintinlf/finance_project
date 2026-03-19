"""SQLite persistence layer (single source of truth).

This module owns:
- the locked schema (tables + indexes)
- all SQL statements used by the trading engine

DB file location (by default): trade_logs/trading.db
Timestamp convention: store UTC ISO-8601 strings.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union


DEFAULT_DB_PATH = Path("trade_logs") / "trading.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fingerprint_api_key(api_key: str, length: int = 12) -> str:
    """Return a stable, non-reversible fingerprint for an API key id (never store secrets)."""
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest[:length]


@contextmanager
def connect(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
    """Create/open the DB and ensure all locked tables + indexes exist (idempotent)."""
    with connect(db_path) as conn:
        # Tables (locked schema)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              account_id TEXT,
              api_fingerprint TEXT,
                            status TEXT,
              timestamp DATETIME,
              cash REAL,
              buying_power REAL,
              portfolio_value REAL,
              equity REAL,
              last_equity REAL,
              margin_used REAL
            )
            """.strip()
        )

        # Backward-compatible migration for DBs created before status column existed.
        account_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(accounts)").fetchall()
        }
        if "status" not in account_cols:
            conn.execute("ALTER TABLE accounts ADD COLUMN status TEXT")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
              trade_id TEXT PRIMARY KEY,
              account_id TEXT,
              symbol TEXT,
              side TEXT,
              qty REAL,
              entry_price REAL,
              exit_price REAL,
              tp_price REAL,
              sl_price REAL,
              status TEXT,
              confidence REAL,
              pnl REAL,
              attempted_at DATETIME,
              opened_at DATETIME,
              closed_at DATETIME,
              broker_order_id TEXT,
              error TEXT
            )
            """.strip()
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
              account_id TEXT,
              symbol TEXT,
              qty REAL,
              avg_price REAL,
              unrealized_pnl REAL,
              updated_at DATETIME,
              PRIMARY KEY(account_id, symbol)
            )
            """.strip()
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_performance (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              symbol TEXT,
              prediction REAL,
              confidence REAL,
              actual_return REAL,
              error REAL,
              timestamp DATETIME
            )
            """.strip()
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_trends (
              timestamp DATETIME,
              spy_trend REAL,
              volatility REAL,
              market_regime TEXT
            )
            """.strip()
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT,
                timestamp DATETIME,
                symbol TEXT,
                signal_type TEXT,
                confidence REAL,
                prob_profit REAL,
                position_quantity_before REAL,
                position_side_before TEXT,
                execution_mode TEXT,
                action TEXT,
                reason TEXT,
                planned_quantity REAL,
                planned_entry_price REAL,
                planned_tp_price REAL,
                planned_sl_price REAL,
                executed INTEGER,
                broker_order_id TEXT,
                execution_timestamp DATETIME,
                error_message TEXT
            )
            """.strip()
        )

        # Indexes (non-breaking additions)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_account_status ON trades(account_id, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_opened ON trades(symbol, opened_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(status, symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_account ON positions(account_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_decisions_account_time ON decisions(account_id, timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_decisions_action_time ON decisions(action, timestamp)"
        )


def insert_account_snapshot(
    *,
    account_id: Optional[str],
    api_fingerprint: Optional[str],
    status: Optional[str] = None,
    cash: Optional[float],
    buying_power: Optional[float] = None,
    portfolio_value: Optional[float] = None,
    equity: Optional[float] = None,
    last_equity: Optional[float] = None,
    margin_used: Optional[float] = None,
    timestamp: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> int:
    ts = timestamp or utc_now_iso()
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO accounts(
              account_id, api_fingerprint, status, timestamp,
              cash, buying_power, portfolio_value,
              equity, last_equity, margin_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.strip(),
            (
                account_id,
                api_fingerprint,
                status,
                ts,
                cash,
                buying_power,
                portfolio_value,
                equity,
                last_equity,
                margin_used,
            ),
        )
        return int(cur.lastrowid)


def upsert_positions(
    *,
    account_id: str,
    positions: Iterable[Mapping[str, Any]],
    updated_at: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    ts = updated_at or utc_now_iso()
    rows: List[tuple] = []
    for pos in positions:
        rows.append(
            (
                account_id,
                str(pos.get("symbol")),
                pos.get("qty"),
                pos.get("avg_price"),
                pos.get("unrealized_pnl"),
                ts,
            )
        )

    if not rows:
        return

    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO positions(account_id, symbol, qty, avg_price, unrealized_pnl, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id, symbol) DO UPDATE SET
              qty=excluded.qty,
              avg_price=excluded.avg_price,
              unrealized_pnl=excluded.unrealized_pnl,
              updated_at=excluded.updated_at
            """.strip(),
            rows,
        )


def create_trade_attempt(
    *,
    trade_id: str,
    account_id: Optional[str],
    symbol: str,
    side: str,
    qty: float,
    entry_price: Optional[float] = None,
    exit_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    status: Optional[str],
    confidence: Optional[float] = None,
    pnl: Optional[float] = None,
    attempted_at: Optional[str] = None,
    opened_at: Optional[str] = None,
    closed_at: Optional[str] = None,
    broker_order_id: Optional[str] = None,
    error: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    # Hard rule: attempted_at must be populated in the initial INSERT.
    attempted_ts = attempted_at or utc_now_iso()

    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO trades(
              trade_id, account_id, symbol, side, qty,
              entry_price, exit_price, tp_price, sl_price,
              status, confidence, pnl,
              attempted_at, opened_at, closed_at,
              broker_order_id, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.strip(),
            (
                trade_id,
                account_id,
                symbol,
                side,
                qty,
                entry_price,
                exit_price,
                tp_price,
                sl_price,
                status,
                confidence,
                pnl,
                attempted_ts,
                opened_at,
                closed_at,
                broker_order_id,
                error,
            ),
        )


def set_trade_broker_order_id(
    *,
    trade_id: str,
    broker_order_id: str,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE trades SET broker_order_id=? WHERE trade_id=?",
            (broker_order_id, trade_id),
        )


def mark_trade_open(
    *,
    trade_id: str,
    entry_price: Optional[float],
    opened_at: Optional[str] = None,
    status: str = "OPEN",
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    opened_ts = opened_at or utc_now_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE trades
            SET status=?, entry_price=?, opened_at=?
            WHERE trade_id=?
            """.strip(),
            (status, entry_price, opened_ts, trade_id),
        )


def mark_trade_closed(
    *,
    trade_id: str,
    exit_price: Optional[float],
    closed_at: Optional[str] = None,
    status: str = "CLOSED",
    pnl: Optional[float] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    closed_ts = closed_at or utc_now_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE trades
            SET status=?, exit_price=?, closed_at=?, pnl=?
            WHERE trade_id=?
            """.strip(),
            (status, exit_price, closed_ts, pnl, trade_id),
        )


def mark_trade_failed(
    *,
    trade_id: str,
    error: str,
    status: str = "FAILED",
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE trades SET status=?, error=? WHERE trade_id=?",
            (status, error, trade_id),
        )


def record_model_performance(
    *,
    symbol: str,
    prediction: Optional[float],
    confidence: Optional[float],
    actual_return: Optional[float] = None,
    error: Optional[float] = None,
    timestamp: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> int:
    ts = timestamp or utc_now_iso()
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO model_performance(symbol, prediction, confidence, actual_return, error, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """.strip(),
            (symbol, prediction, confidence, actual_return, error, ts),
        )
        return int(cur.lastrowid)


def insert_market_trend_snapshot(
    *,
    spy_trend: Optional[float],
    volatility: Optional[float],
    market_regime: Optional[str],
    timestamp: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    ts = timestamp or utc_now_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO market_trends(timestamp, spy_trend, volatility, market_regime)
            VALUES (?, ?, ?, ?)
            """.strip(),
            (ts, spy_trend, volatility, market_regime),
        )


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def get_latest_account_snapshot(
    *,
    account_id: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    with connect(db_path) as conn:
        if account_id:
            row = conn.execute(
                """
                SELECT account_id, api_fingerprint, timestamp,
                      status,
                       cash, buying_power, portfolio_value,
                       equity, last_equity, margin_used
                FROM accounts
                WHERE account_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """.strip(),
                (account_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT account_id, api_fingerprint, timestamp,
                      status,
                       cash, buying_power, portfolio_value,
                       equity, last_equity, margin_used
                FROM accounts
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """.strip(),
            ).fetchone()

    return _row_to_dict(row) if row else None


def get_positions_snapshot(
    *,
    account_id: str,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT account_id, symbol, qty, avg_price, unrealized_pnl, updated_at
            FROM positions
            WHERE account_id = ?
            ORDER BY symbol ASC
            """.strip(),
            (account_id,),
        ).fetchall()

    return [_row_to_dict(r) for r in rows]


def insert_decisions(
    *,
    account_id: str,
    entries: Iterable[Any],
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    rows: List[tuple] = []

    for entry in entries:
        if hasattr(entry, "to_dict"):
            data = entry.to_dict()
        elif isinstance(entry, Mapping):
            data = dict(entry)
        else:
            continue

        executed_value = data.get("executed")
        executed = 1 if executed_value in (True, "True", 1, "1") else 0

        rows.append(
            (
                account_id,
                data.get("timestamp") or utc_now_iso(),
                data.get("symbol"),
                data.get("signal_type"),
                data.get("confidence"),
                data.get("prob_profit"),
                data.get("position_quantity_before"),
                data.get("position_side_before"),
                data.get("execution_mode"),
                data.get("action"),
                data.get("reason"),
                data.get("planned_quantity"),
                data.get("planned_entry_price"),
                data.get("planned_tp_price"),
                data.get("planned_sl_price"),
                executed,
                data.get("broker_order_id"),
                data.get("execution_timestamp"),
                data.get("error_message"),
            )
        )

    if not rows:
        return

    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO decisions(
              account_id, timestamp, symbol, signal_type,
              confidence, prob_profit, position_quantity_before,
              position_side_before, execution_mode, action, reason,
              planned_quantity, planned_entry_price, planned_tp_price,
              planned_sl_price, executed, broker_order_id,
              execution_timestamp, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.strip(),
            rows,
        )


def get_execution_summary_sqlite(
    *,
    account_id: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    where_clause = ""
    params: Sequence[Any] = ()
    if account_id:
        where_clause = "WHERE account_id = ?"
        params = (account_id,)

    with connect(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT
              COUNT(*) AS total_decisions,
              SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) AS total_executed,
              SUM(CASE WHEN action = 'rejected' THEN 1 ELSE 0 END) AS total_rejected,
              SUM(CASE WHEN action = 'hold' THEN 1 ELSE 0 END) AS total_holds
            FROM decisions
            {where_clause}
            """.strip(),
            params,
        ).fetchone()

    total_decisions = int(row["total_decisions"] or 0)
    total_executed = int(row["total_executed"] or 0)
    total_rejected = int(row["total_rejected"] or 0)
    total_holds = int(row["total_holds"] or 0)
    execution_rate = (total_executed / total_decisions * 100.0) if total_decisions else 0.0

    return {
        "total_decisions": total_decisions,
        "total_executed": total_executed,
        "total_rejected": total_rejected,
        "total_holds": total_holds,
        "execution_rate": execution_rate,
    }


def get_recent_decisions_sqlite(
    *,
    account_id: Optional[str] = None,
    limit: int = 10,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    where_clause = ""
    params: List[Any] = []
    if account_id:
        where_clause = "WHERE account_id = ?"
        params.append(account_id)

    params.append(limit)

    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT account_id, timestamp, symbol, signal_type, confidence,
                   prob_profit, action, reason, executed,
                   broker_order_id, execution_timestamp, error_message
            FROM decisions
            {where_clause}
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """.strip(),
            tuple(params),
        ).fetchall()

    return [_row_to_dict(r) for r in rows]


def get_recent_closed_trades_sqlite(
    *,
    account_id: Optional[str] = None,
    limit: int = 10,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    where_parts = ["status = 'CLOSED'"]
    params: List[Any] = []

    if account_id:
        where_parts.append("account_id = ?")
        params.append(account_id)

    params.append(limit)
    where_clause = " AND ".join(where_parts)

    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT trade_id, account_id, symbol, side, qty,
                   entry_price, exit_price, pnl, status,
                   opened_at, closed_at, broker_order_id
            FROM trades
            WHERE {where_clause}
            ORDER BY COALESCE(closed_at, opened_at, attempted_at) DESC
            LIMIT ?
            """.strip(),
            tuple(params),
        ).fetchall()

    return [_row_to_dict(r) for r in rows]


def get_realized_pnl_sqlite(
    *,
    account_id: Optional[str] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> float:
    where_clause = "WHERE status = 'CLOSED'"
    params: Sequence[Any] = ()

    if account_id:
        where_clause += " AND account_id = ?"
        params = (account_id,)

    with connect(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT SUM(COALESCE(pnl, 0.0)) AS total_realized_pnl
            FROM trades
            {where_clause}
            """.strip(),
            params,
        ).fetchone()

    return float(row["total_realized_pnl"] or 0.0)


def upsert_broker_closed_trade(
    *,
    account_id: str,
    broker_order_id: str,
    symbol: str,
    side: str,
    qty: float,
    fill_price: Optional[float],
    filled_at: Optional[str],
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> None:
    """Best-effort snapshot of Alpaca closed fills into trades table."""
    ts = filled_at or utc_now_iso()

    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT trade_id, status, entry_price FROM trades WHERE broker_order_id = ? LIMIT 1",
            (broker_order_id,),
        ).fetchone()

        if existing:
            trade_id = existing["trade_id"]
            status = str(existing["status"] or "")
            if status != "CLOSED":
                conn.execute(
                    """
                    UPDATE trades
                    SET status = 'CLOSED',
                        exit_price = ?,
                        closed_at = ?
                    WHERE trade_id = ?
                    """.strip(),
                    (fill_price, ts, trade_id),
                )
            return

        synthetic_trade_id = f"broker:{broker_order_id}"
        conn.execute(
            """
            INSERT INTO trades(
              trade_id, account_id, symbol, side, qty,
              entry_price, exit_price, tp_price, sl_price,
              status, confidence, pnl,
              attempted_at, opened_at, closed_at,
              broker_order_id, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.strip(),
            (
                synthetic_trade_id,
                account_id,
                symbol,
                side,
                qty,
                fill_price,
                fill_price,
                None,
                None,
                "CLOSED",
                None,
                0.0,
                ts,
                ts,
                ts,
                broker_order_id,
                "Imported from broker closed order snapshot",
            ),
        )


def get_recent_model_performance(
    *,
    limit: int = 20,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT symbol, prediction, confidence, actual_return, error, timestamp
            FROM model_performance
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """.strip(),
            (limit,),
        ).fetchall()

    return [_row_to_dict(r) for r in rows]
