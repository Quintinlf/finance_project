"""Model performance instrumentation for component-level signal tracking.

This module is evaluation-only and does not alter trading logic or weights.
It logs per-decision component predictions, computes rolling metrics, and
generates summary reports from SQLite.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from logic.sqlite_store import DEFAULT_DB_PATH, connect

ComponentSnapshot = Dict[str, Any]

ACTIONABLE_DIRECTIONS = {"buy", "sell"}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _norm_cdf(x: float) -> float:
    # Standard normal CDF via erf; avoids dependency overhead.
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _normalize_direction(label: Any) -> str:
    txt = str(label or "").strip().lower()
    if txt in {"buy", "strong buy", "bull", "bullish", "long"}:
        return "buy"
    if txt in {"sell", "strong sell", "bear", "bearish", "short"}:
        return "sell"
    return "neutral"


def _direction_confidence_from_forecast(forecast: Optional[float], std: Optional[float]) -> Tuple[str, float]:
    if forecast is None:
        return "neutral", 0.0
    mean = float(forecast)
    if abs(mean) <= 1e-12:
        return "neutral", 0.0

    direction = "buy" if mean > 0 else "sell"
    sigma = float(std) if std is not None else 0.0
    if sigma <= 1e-10:
        # No uncertainty estimate: treat directional confidence as moderate.
        return direction, 0.5

    z = abs(mean / sigma)
    return direction, _clip01(_norm_cdf(z))


def _bb_confidence_from_zscore(bb_z_score: Optional[float]) -> float:
    if bb_z_score is None:
        return 0.0
    # 1.5 threshold is trigger point; confidence saturates around 3.0.
    z = abs(float(bb_z_score))
    return _clip01((z - 1.5) / 1.5)


def _rsi_direction_confidence(rsi_value: Optional[float]) -> Tuple[str, float]:
    if rsi_value is None:
        return "neutral", 0.0
    rsi = float(rsi_value)
    if rsi < 30.0:
        # deeper oversold => higher buy confidence
        return "buy", _clip01((30.0 - rsi) / 30.0)
    if rsi > 70.0:
        # deeper overbought => higher sell confidence
        return "sell", _clip01((rsi - 70.0) / 30.0)
    return "neutral", _clip01(abs(rsi - 50.0) / 20.0 * 0.25)


def build_component_snapshot(
    *,
    forecast_result: Dict[str, Any],
    bb_signal: str,
    bb_z_score: Optional[float],
    ensemble_signal: str,
    ensemble_confidence: float,
    rsi_value: Optional[float],
) -> ComponentSnapshot:
    """Build normalized component prediction payload for one decision."""
    bayesian = forecast_result.get("bayesian", {}) if isinstance(forecast_result, dict) else {}
    gp = forecast_result.get("gp", {}) if isinstance(forecast_result, dict) else {}

    bayesian_direction, bayesian_conf = _direction_confidence_from_forecast(
        bayesian.get("forecast"),
        bayesian.get("std"),
    )
    gp_direction, gp_conf = _direction_confidence_from_forecast(
        gp.get("forecast"),
        gp.get("std"),
    )
    rsi_direction, rsi_conf = _rsi_direction_confidence(rsi_value)
    bb_direction = _normalize_direction(bb_signal)
    bb_conf = _bb_confidence_from_zscore(bb_z_score)
    ensemble_direction = _normalize_direction(ensemble_signal)

    directions = {
        "bollinger_bands": bb_direction,
        "bayesian_model": bayesian_direction,
        "gaussian_process": gp_direction,
        "rsi": rsi_direction,
        "ensemble": ensemble_direction,
    }
    confidences = {
        "bollinger_bands": bb_conf,
        "bayesian_model": bayesian_conf,
        "gaussian_process": gp_conf,
        "rsi": rsi_conf,
        "ensemble": _clip01(float(ensemble_confidence)),
    }

    return {
        "directions": directions,
        "confidences": confidences,
        "raw_rsi_value": float(rsi_value) if rsi_value is not None else None,
        "agreement_score_raw": compute_agreement_score(directions),
        "agreement_score_weighted": compute_weighted_agreement_score(directions, confidences),
    }


def compute_agreement_score(signals: Dict[str, Any]) -> float:
    """Compute unweighted alignment score between actionable model directions.

    Returns in [0, 1], where 1 means all actionable directions are identical.
    """
    normalized = [_normalize_direction(v) for v in signals.values()]
    actionable = [v for v in normalized if v in ACTIONABLE_DIRECTIONS]
    n = len(actionable)
    if n <= 1:
        return 1.0

    same_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if actionable[i] == actionable[j]:
                same_pairs += 1
    if total_pairs == 0:
        return 1.0
    return _clip01(same_pairs / float(total_pairs))


def compute_weighted_agreement_score(signals: Dict[str, Any], confidences: Dict[str, Any]) -> float:
    """Compute confidence-weighted agreement score in [0, 1]."""
    pairs: List[Tuple[str, float]] = []
    for name, direction in signals.items():
        d = _normalize_direction(direction)
        if d in ACTIONABLE_DIRECTIONS:
            pairs.append((d, _clip01(float(confidences.get(name, 0.0) or 0.0))))

    n = len(pairs)
    if n <= 1:
        return 1.0

    same_weight = 0.0
    total_weight = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = pairs[i][1] * pairs[j][1]
            total_weight += w
            if pairs[i][0] == pairs[j][0]:
                same_weight += w
    if total_weight <= 1e-12:
        return 1.0
    return _clip01(same_weight / total_weight)


def _component_correct(direction: str, next_day_return: Optional[float]) -> Optional[int]:
    if next_day_return is None:
        return None
    d = _normalize_direction(direction)
    r = float(next_day_return)
    if d == "buy":
        return 1 if r > 0.0 else 0
    if d == "sell":
        return 1 if r < 0.0 else 0
    return None


def init_model_performance_tracker(db_path: Union[str, "PathLike[str]"] = DEFAULT_DB_PATH) -> None:
    """Ensure instrumentation table and indexes exist (idempotent)."""
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_component_performance (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              decision_key TEXT UNIQUE,
              timestamp DATETIME NOT NULL,
              symbol TEXT NOT NULL,
              action TEXT,
              price_at_signal REAL,
              next_day_return REAL,

              bb_direction TEXT,
              bb_confidence REAL,
              bb_correct INTEGER,

              bayesian_direction TEXT,
              bayesian_confidence REAL,
              bayesian_correct INTEGER,

              gp_direction TEXT,
              gp_confidence REAL,
              gp_correct INTEGER,

              rsi_direction TEXT,
              rsi_confidence REAL,
              rsi_value REAL,
              rsi_correct INTEGER,

              ensemble_direction TEXT,
              ensemble_confidence REAL,
              ensemble_correct INTEGER,

              agreement_score_raw REAL,
              agreement_score_weighted REAL,
              bb_disagrees_with_ensemble INTEGER DEFAULT 0
            )
            """.strip()
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_component_perf_symbol_time "
            "ON model_component_performance(symbol, timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_component_perf_time "
            "ON model_component_performance(timestamp)"
        )


def log_model_decision(
    *,
    decision_key: str,
    timestamp: datetime,
    symbol: str,
    action: str,
    price_at_signal: Optional[float],
    component_snapshot: Optional[ComponentSnapshot],
    next_day_return: Optional[float] = None,
    db_path: Union[str, "PathLike[str]"] = DEFAULT_DB_PATH,
) -> None:
    """Insert one component-level decision row.

    This is best-effort and idempotent (decision_key UNIQUE + INSERT OR IGNORE).
    """
    if not component_snapshot:
        return

    dirs = component_snapshot.get("directions", {})
    confs = component_snapshot.get("confidences", {})

    bb_direction = _normalize_direction(dirs.get("bollinger_bands"))
    bay_direction = _normalize_direction(dirs.get("bayesian_model"))
    gp_direction = _normalize_direction(dirs.get("gaussian_process"))
    rsi_direction = _normalize_direction(dirs.get("rsi"))
    ens_direction = _normalize_direction(dirs.get("ensemble"))

    payload = (
        decision_key,
        timestamp.astimezone(timezone.utc).isoformat(),
        symbol,
        action,
        float(price_at_signal) if price_at_signal is not None else None,
        float(next_day_return) if next_day_return is not None else None,
        bb_direction,
        float(confs.get("bollinger_bands", 0.0) or 0.0),
        _component_correct(bb_direction, next_day_return),
        bay_direction,
        float(confs.get("bayesian_model", 0.0) or 0.0),
        _component_correct(bay_direction, next_day_return),
        gp_direction,
        float(confs.get("gaussian_process", 0.0) or 0.0),
        _component_correct(gp_direction, next_day_return),
        rsi_direction,
        float(confs.get("rsi", 0.0) or 0.0),
        component_snapshot.get("raw_rsi_value"),
        _component_correct(rsi_direction, next_day_return),
        ens_direction,
        float(confs.get("ensemble", 0.0) or 0.0),
        _component_correct(ens_direction, next_day_return),
        float(component_snapshot.get("agreement_score_raw", 1.0) or 1.0),
        float(component_snapshot.get("agreement_score_weighted", 1.0) or 1.0),
        1 if (bb_direction in ACTIONABLE_DIRECTIONS and ens_direction in ACTIONABLE_DIRECTIONS and bb_direction != ens_direction) else 0,
    )

    try:
        init_model_performance_tracker(db_path)
        with connect(db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO model_component_performance (
                  decision_key, timestamp, symbol, action, price_at_signal, next_day_return,
                  bb_direction, bb_confidence, bb_correct,
                  bayesian_direction, bayesian_confidence, bayesian_correct,
                  gp_direction, gp_confidence, gp_correct,
                  rsi_direction, rsi_confidence, rsi_value, rsi_correct,
                  ensemble_direction, ensemble_confidence, ensemble_correct,
                  agreement_score_raw, agreement_score_weighted, bb_disagrees_with_ensemble
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """.strip(),
                payload,
            )
    except Exception:
        # Must never block trading flow.
        return


def _safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    xs = [float(v) for v in x[:n]]
    ys = [float(v) for v in y[:n]]
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((v - mx) ** 2 for v in xs)
    vy = sum((v - my) ** 2 for v in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / math.sqrt(vx * vy)


def _component_strategy_return(direction: str, next_day_return: float) -> Optional[float]:
    d = _normalize_direction(direction)
    if d == "buy":
        return float(next_day_return)
    if d == "sell":
        return -float(next_day_return)
    return None


def _latest_rolling_rate(values: Iterable[int], window: int) -> Optional[float]:
    dq: Deque[int] = deque(maxlen=window)
    for v in values:
        dq.append(int(v))
    if not dq:
        return None
    return sum(dq) / float(len(dq))


def _latest_rolling_sharpe(returns: Iterable[float], window: int) -> Optional[float]:
    dq: Deque[float] = deque(maxlen=window)
    for v in returns:
        dq.append(float(v))
    if len(dq) < 2:
        return None
    vals = list(dq)
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(1, (len(vals) - 1))
    std = math.sqrt(var)
    if std <= 1e-12:
        return None
    return (mean / std) * math.sqrt(252.0)


def _collect_component_rows(db_path: Union[str, "PathLike[str]"], component_prefix: str) -> List[Dict[str, Any]]:
    direction_col = f"{component_prefix}_direction"
    correct_col = f"{component_prefix}_correct"

    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT timestamp, next_day_return, {direction_col} AS direction, {correct_col} AS is_correct
            FROM model_component_performance
            WHERE {correct_col} IS NOT NULL
            ORDER BY timestamp ASC, id ASC
            """.strip()
        ).fetchall()
    return [dict(r) for r in rows]


def _component_metrics(db_path: Union[str, "PathLike[str]"], component_prefix: str) -> Dict[str, Any]:
    rows = _collect_component_rows(db_path, component_prefix)
    if not rows:
        return {
            "sample_size": 0,
            "rolling_win_rate_10": None,
            "rolling_win_rate_20": None,
            "cumulative_win_rate": None,
            "rolling_sharpe_20": None,
        }

    correct_series = [int(r["is_correct"]) for r in rows]
    strat_returns: List[float] = []
    for r in rows:
        ret = r.get("next_day_return")
        direction = r.get("direction")
        if ret is None:
            continue
        sret = _component_strategy_return(str(direction or ""), float(ret))
        if sret is not None:
            strat_returns.append(float(sret))

    return {
        "sample_size": len(correct_series),
        "rolling_win_rate_10": _latest_rolling_rate(correct_series, 10),
        "rolling_win_rate_20": _latest_rolling_rate(correct_series, 20),
        "cumulative_win_rate": sum(correct_series) / float(len(correct_series)),
        "rolling_sharpe_20": _latest_rolling_sharpe(strat_returns, 20),
    }


def _compute_realized_return_from_row(symbol: str, timestamp_iso: str, price_at_signal: float) -> Optional[float]:
    # Lazy import keeps normal decision path fast.
    import yfinance as yf

    try:
        dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        signal_date = dt.date()
        start = (signal_date - timedelta(days=2)).isoformat()
        end = (signal_date + timedelta(days=10)).isoformat()
        hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
        closes = list(hist.get("Close", []))
        index_dates = [idx.date() for idx in hist.index]
        if not closes:
            return None

        next_close: Optional[float] = None
        seen_signal_or_after = False
        for i, d in enumerate(index_dates):
            if d >= signal_date:
                if not seen_signal_or_after:
                    seen_signal_or_after = True
                    continue
                next_close = float(closes[i])
                break
        if next_close is None or price_at_signal <= 0.0:
            return None
        return (next_close / float(price_at_signal)) - 1.0
    except Exception:
        return None


def backfill_next_day_returns(
    *,
    db_path: Union[str, "PathLike[str]"] = DEFAULT_DB_PATH,
    max_rows: int = 250,
) -> int:
    """Populate next_day_return + correctness for matured rows.

    Returns number of rows updated.
    """
    init_model_performance_tracker(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, symbol, timestamp, price_at_signal,
                   bb_direction, bayesian_direction, gp_direction, rsi_direction, ensemble_direction
            FROM model_component_performance
            WHERE next_day_return IS NULL
              AND price_at_signal IS NOT NULL
            ORDER BY timestamp ASC, id ASC
            LIMIT ?
            """.strip(),
            (int(max_rows),),
        ).fetchall()

    updates: List[Tuple[Any, ...]] = []
    now_utc = datetime.now(timezone.utc)
    for row in rows:
        ts = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # Wait until we likely have next trading day close.
        if now_utc - ts < timedelta(hours=24):
            continue

        ret = _compute_realized_return_from_row(
            symbol=str(row["symbol"]),
            timestamp_iso=str(row["timestamp"]),
            price_at_signal=float(row["price_at_signal"]),
        )
        if ret is None:
            continue
        updates.append(
            (
                float(ret),
                _component_correct(str(row["bb_direction"]), ret),
                _component_correct(str(row["bayesian_direction"]), ret),
                _component_correct(str(row["gp_direction"]), ret),
                _component_correct(str(row["rsi_direction"]), ret),
                _component_correct(str(row["ensemble_direction"]), ret),
                int(row["id"]),
            )
        )

    if not updates:
        return 0

    with connect(db_path) as conn:
        conn.executemany(
            """
            UPDATE model_component_performance
            SET next_day_return = ?,
                bb_correct = ?,
                bayesian_correct = ?,
                gp_correct = ?,
                rsi_correct = ?,
                ensemble_correct = ?
            WHERE id = ?
            """.strip(),
            updates,
        )
    return len(updates)


def generate_performance_report(
    *,
    db_path: Union[str, "PathLike[str]"] = DEFAULT_DB_PATH,
    backfill_returns: bool = True,
) -> Dict[str, Any]:
    """Generate report with per-model stats and agreement analyses."""
    init_model_performance_tracker(db_path)
    if backfill_returns:
        backfill_next_day_returns(db_path=db_path)

    components = {
        "bollinger_bands": "bb",
        "bayesian_model": "bayesian",
        "gaussian_process": "gp",
        "rsi": "rsi",
        "ensemble": "ensemble",
    }

    per_model: Dict[str, Any] = {}
    for display_name, prefix in components.items():
        per_model[display_name] = _component_metrics(db_path, prefix)

    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT agreement_score_raw, agreement_score_weighted, ensemble_correct
            FROM model_component_performance
            WHERE ensemble_correct IS NOT NULL
            ORDER BY timestamp ASC, id ASC
            """.strip()
        ).fetchall()

        disagreements = conn.execute(
            """
            SELECT next_day_return, bb_direction, ensemble_direction, bb_correct, ensemble_correct
            FROM model_component_performance
            WHERE bb_disagrees_with_ensemble = 1
              AND next_day_return IS NOT NULL
              AND bb_correct IS NOT NULL
              AND ensemble_correct IS NOT NULL
            ORDER BY timestamp ASC, id ASC
            """.strip()
        ).fetchall()

    raw_agreement = [float(r["agreement_score_raw"]) for r in rows]
    weighted_agreement = [float(r["agreement_score_weighted"]) for r in rows]
    ensemble_acc = [float(r["ensemble_correct"]) for r in rows]

    bb_rows = [dict(r) for r in disagreements]
    bb_total = len(bb_rows)
    bb_wins = sum(int(r["bb_correct"]) for r in bb_rows) if bb_rows else 0
    ens_wins = sum(int(r["ensemble_correct"]) for r in bb_rows) if bb_rows else 0

    bb_strategy_returns: List[float] = []
    ens_strategy_returns: List[float] = []
    for r in bb_rows:
        ret = float(r["next_day_return"])
        bb_sr = _component_strategy_return(str(r["bb_direction"]), ret)
        ens_sr = _component_strategy_return(str(r["ensemble_direction"]), ret)
        if bb_sr is not None:
            bb_strategy_returns.append(bb_sr)
        if ens_sr is not None:
            ens_strategy_returns.append(ens_sr)

    return {
        "per_model_stats": per_model,
        "agreement_accuracy_correlation": {
            "raw_agreement_vs_ensemble_accuracy": _safe_pearson(raw_agreement, ensemble_acc),
            "weighted_agreement_vs_ensemble_accuracy": _safe_pearson(weighted_agreement, ensemble_acc),
            "sample_size": len(ensemble_acc),
        },
        "bb_disagrees_with_ensemble": {
            "sample_size": bb_total,
            "bb_win_rate": (bb_wins / bb_total) if bb_total else None,
            "ensemble_win_rate": (ens_wins / bb_total) if bb_total else None,
            "bb_avg_strategy_return": (sum(bb_strategy_returns) / len(bb_strategy_returns)) if bb_strategy_returns else None,
            "ensemble_avg_strategy_return": (sum(ens_strategy_returns) / len(ens_strategy_returns)) if ens_strategy_returns else None,
        },
    }
