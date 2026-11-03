"""PredictionRecord: one forecast, written before its outcome exists.

The paper bot appends a row at decision time. A later run fills
``realized_return`` from the next close. Nothing in that fill is allowed to
rewrite the forecast fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PREDICTIONS_PATH = Path("trade_logs") / "predictions" / "predictions.jsonl"

FORECAST_FIELDS = (
    "timestamp",
    "symbol",
    "information_cutoff",
    "features_version",
    "model_version",
    "bayesian_prediction",
    "gp_prediction",
    "bb_features",
    "rsi_features",
    "candle_features",
    "volume_features",
    "news_features",
    "regime_features",
    "lunar_features",
    "ensemble_probability",
    "expected_return",
    "estimated_cost",
    "net_ev",
    "action",
    "quantity",
)


@dataclass
class PredictionRecord:
    timestamp: str
    symbol: str
    information_cutoff: str
    features_version: str
    model_version: str
    bayesian_prediction: Optional[float] = None
    gp_prediction: Optional[float] = None
    bb_features: Dict[str, Any] = field(default_factory=dict)
    rsi_features: Dict[str, Any] = field(default_factory=dict)
    candle_features: Dict[str, Any] = field(default_factory=dict)
    volume_features: Dict[str, Any] = field(default_factory=dict)
    news_features: Dict[str, Any] = field(default_factory=dict)
    regime_features: Dict[str, Any] = field(default_factory=dict)
    lunar_features: Dict[str, Any] = field(default_factory=dict)
    ensemble_probability: Optional[float] = None
    expected_return: Optional[float] = None
    estimated_cost: Optional[float] = None
    net_ev: Optional[float] = None
    action: str = "hold"
    quantity: float = 0.0
    realized_return: Optional[float] = None
    realized_net: Optional[float] = None
    arm: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _cutoff_day(value: str) -> Optional[date]:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(text[:10]), datetime.min.time())
        except ValueError:
            return None
    return parsed.date()


def record_from_signal(signal: Any, *, action: str, quantity: float, timestamp: Optional[str] = None) -> PredictionRecord:
    """Build a record from a live Signal. Outcome fields stay empty."""
    meta = getattr(signal, "meta", None) or {}
    forecast = meta.get("full_forecast") or {}
    snap = forecast.get("feature_snapshot") or {}
    news = {}
    if meta.get("news_polarity_score") is not None:
        news["news_polarity"] = meta.get("news_polarity_score")
    if meta.get("news_themes"):
        news["themes"] = list(meta.get("news_themes") or [])
    regime = meta.get("market_regime") or snap.get("regime_features") or {}
    cost = meta.get("edge_cost_pct")
    ev = meta.get("edge_expected_value_pct")
    when = timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")
    cutoff = str(forecast.get("information_cutoff") or when)
    return PredictionRecord(
        timestamp=when,
        symbol=str(getattr(signal, "symbol", "")),
        information_cutoff=cutoff,
        features_version=str(forecast.get("features_version") or ""),
        model_version=str(forecast.get("model_version") or ""),
        bayesian_prediction=_float_or_none((forecast.get("bayesian") or {}).get("forecast")),
        gp_prediction=_float_or_none((forecast.get("gp") or {}).get("forecast")),
        bb_features=dict(snap.get("bb_features") or {}),
        rsi_features=dict(snap.get("rsi_features") or {}),
        candle_features=dict(snap.get("candle_features") or {}),
        volume_features=dict(snap.get("volume_features") or {}),
        news_features=news,
        regime_features=dict(regime) if isinstance(regime, dict) else {},
        lunar_features=dict(snap.get("lunar_features") or {}),
        ensemble_probability=_float_or_none(getattr(signal, "prob_profit", None)),
        expected_return=_float_or_none(meta.get("ensemble_forecast_return")),
        estimated_cost=_float_or_none(cost),
        net_ev=_float_or_none(ev),
        action=str(action or "hold"),
        quantity=float(quantity or 0.0),
    )


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def append_records(records: Iterable[PredictionRecord], path: Path = PREDICTIONS_PATH) -> int:
    rows = list(records)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), default=str) + "\n")
    return len(rows)


def load_records(path: Path = PREDICTIONS_PATH) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_records(records: Iterable[dict], path: Path = PREDICTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, default=str) + "\n")


def records_from_cycle(signals: Iterable[Any], decision_log: Iterable[Any]) -> List[PredictionRecord]:
    """One record per scored symbol, including holds and rejections.

    Action and quantity come from the decision log when that symbol was
    planned. A symbol the log never saw keeps its signal type and quantity 0.
    """
    by_symbol: Dict[str, Any] = {}
    for entry in decision_log or []:
        symbol = str(getattr(entry, "symbol", "") or "")
        if symbol:
            by_symbol[symbol] = entry
    stamped = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    seen = set()
    for signal in signals or []:
        symbol = str(getattr(signal, "symbol", "") or "")
        seen.add(symbol)
        entry = by_symbol.get(symbol)
        if entry is None:
            meta = getattr(signal, "meta", None) or {}
            if meta.get("threshold_decision") == "reject" or meta.get("edge_passes") is False:
                action = "rejected"
            else:
                action = str(getattr(signal, "signal_type", "hold"))
            quantity = 0.0
        else:
            action = str(getattr(entry, "action", "hold"))
            quantity = float(getattr(entry, "planned_quantity", 0) or 0)
        records.append(record_from_signal(signal, action=action, quantity=quantity, timestamp=stamped))
    return records


def fill_realized_returns(path: Path = PREDICTIONS_PATH, *, history_lookup=None) -> int:
    """Fill outcomes for rows whose next close is now known.

    ``history_lookup(symbol)`` returns a DataFrame with a Close column. The
    forecast fields on the row are left as they were written.
    """
    rows = load_records(path)
    if not rows:
        return 0
    if history_lookup is None:
        from logic.price_cache import get_history

        history_lookup = lambda symbol: get_history(symbol, allow_fetch=True)  # noqa: E731

    updated = 0
    cache: Dict[str, Any] = {}
    for row in rows:
        if row.get("realized_return") is not None:
            continue
        symbol = str(row.get("symbol") or "")
        cutoff = _cutoff_day(str(row.get("information_cutoff") or ""))
        if not symbol or cutoff is None:
            continue
        if symbol not in cache:
            try:
                cache[symbol] = history_lookup(symbol)
            except Exception:
                cache[symbol] = None
        frame = cache[symbol]
        realized = _next_close_return(frame, cutoff)
        if realized is None:
            continue
        row["realized_return"] = realized
        cost = row.get("estimated_cost")
        # estimated_cost is stored in percent, matching the edge gate.
        if cost is None:
            row["realized_net"] = realized
        else:
            row["realized_net"] = realized - float(cost) / 100.0
        updated += 1
    if updated:
        write_records(rows, path)
    return updated


def _next_close_return(frame, cutoff: date) -> Optional[float]:
    if frame is None or len(getattr(frame, "index", [])) < 2 or "Close" not in getattr(frame, "columns", []):
        return None
    days = []
    for label in frame.index:
        try:
            days.append(pd_timestamp_date(label))
        except Exception:
            days.append(None)
    closes = [float(v) for v in frame["Close"].tolist()]
    for i, day in enumerate(days):
        if day is None or day < cutoff:
            continue
        if day > cutoff:
            return None
        if i + 1 >= len(closes):
            return None
        if closes[i] == 0:
            return None
        return closes[i + 1] / closes[i] - 1.0
    return None


def pd_timestamp_date(label) -> date:
    import pandas as pd

    if isinstance(label, date) and not isinstance(label, datetime):
        return label
    stamp = pd.Timestamp(label)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.date()
