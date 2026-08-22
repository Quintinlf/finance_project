"""Export the bot's state to a single JSON file for the static dashboard.

The daily workflow already commits everything the dashboard needs -- trading.db,
run_snapshots.csv, run-*.log, and the backtest JSONs. This script flattens those
into ``docs/data.json`` so the dashboard is a plain static page with no server,
no API keys in the browser, and nothing to keep running between market days.

Run it after the trading cycle:

    python scripts/build_dashboard.py

Everything is best-effort: a missing or malformed input produces an empty
section and a note, never a failed build. A dashboard that renders with one
panel missing is far more useful than a workflow step that goes red.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
TRADE_LOGS = ROOT / "trade_logs"
DB_PATH = TRADE_LOGS / "trading.db"
SNAPSHOTS = TRADE_LOGS / "run_snapshots.csv"
BACKTESTS = TRADE_LOGS / "backtests"
OUT_PATH = ROOT / "docs" / "data.json"

# Signal lines the runner writes, e.g.
# "SIGNAL FILTER | symbol=UNG | original=STRONG BUY | ... | decision=PASS | reason=..."
_SIGNAL_RE = re.compile(r"SIGNAL FILTER \| (.+)$")
_ACCOUNT_RE = re.compile(
    r"ACCOUNT: equity=\$([\d.]+) \(source=(\w+)\) \| cash=\$([\d.]+) \| exposure=([\d.]+)%"
)


def _safe(fn, default):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a broken panel must not fail the build
        print(f"  ! {fn.__name__ if hasattr(fn, '__name__') else fn}: {exc}")
        return default


def _latest_run_log() -> Optional[Path]:
    logs = sorted(TRADE_LOGS.glob("run-*.log"))
    return logs[-1] if logs else None


def load_equity_curve() -> List[Dict[str, Any]]:
    """Portfolio value over time, from the per-run snapshot CSV."""
    if not SNAPSHOTS.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with SNAPSHOTS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append(
                    {
                        "time": row["utc_timestamp"][:10],
                        "value": float(row["portfolio_value"]),
                        "exposure": float(row["exposure"]),
                        "daily_return": float(row.get("daily_return") or 0.0),
                    }
                )
            except (KeyError, ValueError):
                continue
    # Lightweight Charts requires ascending, unique timestamps; keep the last
    # snapshot of each day.
    by_day: Dict[str, Dict[str, Any]] = {r["time"]: r for r in rows}
    return [by_day[k] for k in sorted(by_day)]


def load_trades() -> List[Dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, side, qty, entry_price, exit_price, status, pnl,
                   confidence, opened_at, closed_at, attempted_at, error
            FROM trades
            ORDER BY COALESCE(opened_at, attempted_at) DESC
            """
        ).fetchall()
    finally:
        conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "symbol": r["symbol"],
                "side": r["side"],
                "qty": float(r["qty"] or 0),
                "entry": float(r["entry_price"]) if r["entry_price"] else None,
                "exit": float(r["exit_price"]) if r["exit_price"] else None,
                "status": r["status"],
                "pnl": float(r["pnl"]) if r["pnl"] is not None else None,
                "opened_at": r["opened_at"],
                "closed_at": r["closed_at"],
                "attempted_at": r["attempted_at"],
                "error": r["error"],
            }
        )
    return out


def load_decisions(limit: int = 200) -> List[Dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT timestamp, symbol, signal_type, confidence, prob_profit,
                   action, reason, executed, planned_quantity,
                   planned_entry_price, error_message
            FROM decisions ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def parse_latest_run() -> Dict[str, Any]:
    """Signals, account state, and blockers from the most recent run log."""
    log = _latest_run_log()
    if log is None:
        return {"date": None, "signals": [], "blockers": [], "note": "no run log found"}

    text = log.read_text(encoding="utf-8", errors="replace")
    signals: List[Dict[str, Any]] = []
    for line in text.splitlines():
        m = _SIGNAL_RE.search(line)
        if not m:
            continue
        fields: Dict[str, str] = {}
        for part in m.group(1).split(" | "):
            if "=" in part:
                k, _, v = part.partition("=")
                fields[k.strip()] = v.strip()
        if not fields.get("symbol"):
            continue
        signals.append(
            {
                "symbol": fields.get("symbol"),
                "signal": fields.get("normalized", fields.get("original", "")),
                "original": fields.get("original", ""),
                "confidence": float(fields.get("confidence") or 0),
                "prob_profit": float(fields.get("prob_profit") or 0),
                "decision": fields.get("decision", ""),
                "reason": fields.get("reason", ""),
            }
        )

    account: Dict[str, Any] = {}
    for m in _ACCOUNT_RE.finditer(text):
        account = {
            "equity": float(m.group(1)),
            "equity_source": m.group(2),
            "cash": float(m.group(3)),
            "exposure": float(m.group(4)) / 100.0,
        }

    # The panel that matters most on this bot: why nothing traded.
    blockers = [
        line.split(" | ", 2)[-1].strip()
        for line in text.splitlines()
        if "rejection_reason:" in line or "Blocked:" in line
    ]

    positions = []
    for m in re.finditer(
        r"OPEN (\w+)\s+([\d.]+) sh @ \$([\d.]+) since ([\d-]+)", text
    ):
        positions.append(
            {
                "symbol": m.group(1),
                "qty": float(m.group(2)),
                "entry": float(m.group(3)),
                "since": m.group(4),
            }
        )

    return {
        "date": log.stem.replace("run-", ""),
        "account": account,
        "signals": signals,
        "positions": positions,
        "blockers": sorted(set(blockers))[:12],
        "eos_available": "EOS FINANCE DOMAINS AVAILABLE" in text,
        "orders_submitted": _int_after(text, "ORDERS SUBMITTED:"),
        "orders_attempted": _int_after(text, "ORDERS ATTEMPTED:"),
    }


def _int_after(text: str, marker: str) -> Optional[int]:
    m = re.search(re.escape(marker) + r"\s*(\d+)", text)
    return int(m.group(1)) if m else None


def load_backtests() -> List[Dict[str, Any]]:
    """Every backtest's headline metrics, so the dashboard can show the spread.

    Showing one window in isolation is how a strategy looks profitable; the
    honest view is all of them side by side with their confidence intervals.
    """
    if not BACKTESTS.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(BACKTESTS.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = data.get("metrics") or {}
        if not metrics:
            continue
        out.append(
            {
                "name": path.stem,
                "start": data.get("start"),
                "end": data.get("end"),
                "symbols": len(data.get("symbols") or []),
                "n_trades": metrics.get("n_trades"),
                "win_rate": metrics.get("win_rate"),
                "net_pnl": metrics.get("net_pnl"),
                "total_return_pct": metrics.get("total_return_pct"),
                "buy_hold_return_pct": metrics.get("buy_hold_return_pct"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "profit_factor": metrics.get("profit_factor"),
                "cost_drag_pct": metrics.get("cost_drag_pct"),
                "ci95": metrics.get("trade_return_ci95"),
                "warning": metrics.get("sample_warning"),
            }
        )
    return out


def load_component_accuracy() -> Dict[str, Any]:
    """Measured directional accuracy per model component, with the base rate.

    An accuracy figure alone is misleading: 58% correct looks like skill until
    you notice the market rose on 76% of the sampled days, at which point it is
    worse than always guessing up. The base rate travels with the numbers.
    """
    if not DB_PATH.exists():
        return {}
    import math

    conn = sqlite3.connect(DB_PATH)
    try:
        base_n, base_up = conn.execute(
            """
            SELECT COUNT(*), SUM(CASE WHEN next_day_return > 0 THEN 1 ELSE 0 END)
            FROM model_component_performance WHERE next_day_return IS NOT NULL
            """
        ).fetchone()
        base_n = int(base_n or 0)
        if not base_n:
            return {}

        components = []
        for label, prefix in [
            ("Bollinger", "bb"), ("Bayesian", "bayesian"), ("Gaussian Process", "gp"),
            ("RSI", "rsi"), ("Ensemble", "ensemble"),
        ]:
            n, k = conn.execute(
                f"SELECT COUNT({prefix}_correct), SUM({prefix}_correct) "
                f"FROM model_component_performance WHERE {prefix}_correct IS NOT NULL"
            ).fetchone()
            n = int(n or 0)
            if not n:
                continue
            p_hat = int(k or 0) / n
            se = math.sqrt(max(p_hat * (1 - p_hat), 0.0) / n)
            components.append({
                "name": label, "n": n, "accuracy": round(p_hat * 100, 1),
                "ci_low": round(max(0.0, p_hat - 1.96 * se) * 100, 1),
                "ci_high": round(min(1.0, p_hat + 1.96 * se) * 100, 1),
            })
    finally:
        conn.close()

    return {
        "base_rate": round(int(base_up or 0) / base_n * 100, 1),
        "base_n": base_n,
        "components": components,
    }


def summarise(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    closed = [t for t in trades if t["pnl"] is not None]
    wins = [t for t in closed if t["pnl"] > 0]
    failed = [t for t in trades if t["status"] == "FAILED"]
    return {
        "filled": len([t for t in trades if t["status"] in {"OPEN", "CLOSED"}]),
        "failed": len(failed),
        "closed": len(closed),
        "realized_pnl": round(sum(t["pnl"] for t in closed), 2),
        "win_rate": round(100.0 * len(wins) / len(closed), 1) if closed else None,
    }


def main() -> None:
    print("Building dashboard data...")
    trades = _safe(load_trades, [])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "equity_curve": _safe(load_equity_curve, []),
        "trades": trades,
        "summary": summarise(trades),
        "decisions": _safe(load_decisions, []),
        "latest_run": _safe(parse_latest_run, {}),
        "backtests": _safe(load_backtests, []),
        "component_accuracy": _safe(load_component_accuracy, {}),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  equity points : {len(payload['equity_curve'])}")
    print(f"  trades        : {len(payload['trades'])}")
    print(f"  decisions     : {len(payload['decisions'])}")
    print(f"  signals today : {len(payload['latest_run'].get('signals', []))}")
    print(f"  backtests     : {len(payload['backtests'])}")
    print(f"  scored preds  : {payload['component_accuracy'].get('base_n', 0)}")
    print(f"  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
