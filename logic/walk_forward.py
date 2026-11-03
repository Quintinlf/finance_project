"""Purged walk-forward ablation.

At bar t the model is fit on features from bars strictly before t paired with
the return those bars were followed by. The score is the return from t to
t+1, minus a round trip when the forecast is large enough to trade.

Statuses:

- survives: incremental net return vs baseline is positive and the day-block
  bootstrap interval lies above zero. Only this is promoted.
- promising: positive point estimate, interval still crosses zero. Keep, and
  collect more data. Do not delete.
- fails: incremental net return is not positive.
- not_tested: too few days to score.

The report also compares each arm with buy-and-hold and with three simple
rules (5-day momentum, 10/20 moving-average crossover, one-day mean
reversion). Those comparisons are evidence. They do not change the status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from logic.costs import for_symbol
from logic.feature_blocks import ARM_BLOCKS, add_research_features, columns_for
from logic.feature_policy import REPORT_PATH
from logic.trading_functions import gp_predict, update_beliefs_with_data

MIN_TRAIN = 80
MIN_DAYS = 40
BOOTSTRAP_DRAWS = 1000

SIMPLE_RULES = ("momentum_5", "ma_crossover", "mean_reversion")


@dataclass
class ArmResult:
    name: str
    status: str
    n_observations: int
    n_days: int
    incremental_mean: Optional[float] = None
    incremental_ci: Optional[List[float]] = None
    own_mean_net: Optional[float] = None
    own_ci: Optional[List[float]] = None
    vs_buy_hold: Dict[str, Optional[float]] = field(default_factory=dict)
    vs_simple: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "n_observations": self.n_observations,
            "n_days": self.n_days,
            "incremental_mean": self.incremental_mean,
            "incremental_ci": self.incremental_ci,
            "own_mean_net": self.own_mean_net,
            "own_ci": self.own_ci,
            "vs_buy_hold": self.vs_buy_hold,
            "vs_simple": self.vs_simple,
            "note": self.note,
        }


def classify_incremental(mean: Optional[float], lo: Optional[float], hi: Optional[float], *, n_days: int) -> str:
    """Status from the baseline comparison only."""
    if n_days < MIN_DAYS or mean is None or lo is None or hi is None:
        return "not_tested"
    if mean > 0 and lo > 0:
        return "survives"
    if mean > 0:
        return "promising"
    return "fails"


def _standardize(train: np.ndarray, test: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def _fit_aligned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    min_rows: int = MIN_TRAIN,
) -> Optional[float]:
    """Posterior mean at X_test. X_train is already paired with the next return."""
    if len(X_train) < min_rows or not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
        return None
    X_train, X_test = _standardize(X_train, X_test)
    prior = np.zeros(X_train.shape[1])
    prior_cov = np.eye(X_train.shape[1]) * 0.1
    try:
        posterior, _cov = update_beliefs_with_data(X_train, y_train, prior, prior_cov, 0.01)
    except np.linalg.LinAlgError:
        return None
    return float(np.asarray(X_test @ posterior).reshape(-1)[0])


def _fit_gp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Optional[float]:
    if len(X_train) < 30:
        return None
    # Trailing window keeps the kernel inversion small. The holdout is the
    # rows passed in; the caller already excluded the test target.
    window = 60
    X_train = X_train[-window:]
    y_train = y_train[-window:]
    X_train, X_test = _standardize(X_train, X_test)
    try:
        mean, _var = gp_predict(X_train, y_train, X_test, noise_var=1e-3)
    except np.linalg.LinAlgError:
        return None
    value = float(np.asarray(mean).reshape(-1)[0])
    if not np.isfinite(value):
        return None
    return value


def prepare_frame(df: pd.DataFrame, news_polarity: Optional[pd.Series] = None) -> pd.DataFrame:
    """Causal features plus the next-bar return that will be the label."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [col[0] if isinstance(col, tuple) else col for col in out.columns]
    out["Return"] = out["Close"].astype(float).pct_change()
    for lag in range(1, 11):
        out[f"Return_lag{lag}"] = out["Return"].shift(lag)
    delta = out["Close"].astype(float).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    mid = out["Close"].astype(float).rolling(20).mean()
    std = out["Close"].astype(float).rolling(20).std()
    out["BB_Middle"] = mid
    out["BB_Upper"] = mid + 2 * std
    out["BB_Lower"] = mid - 2 * std
    out["BB_Z_Score"] = (out["Close"].astype(float) - mid) / (std + 1e-8)
    out["BB_Width"] = (out["BB_Upper"] - out["BB_Lower"]) / (mid + 1e-8)
    out["Volatility"] = out["Return"].rolling(5).std()
    out = add_research_features(out)
    if news_polarity is not None:
        aligned = news_polarity.reindex(out.index)
        out["news_polarity"] = aligned.astype(float)
    out["next_return"] = out["Return"].shift(-1)
    close = out["Close"].astype(float)
    out["sma10"] = close.rolling(10).mean()
    out["sma20"] = close.rolling(20).mean()
    return out


def _cost_fraction(symbol: str) -> float:
    return float(for_symbol(symbol).round_trip_bps) / 10_000.0


def _strategy_net(prediction: float, realized: float, cost: float) -> float:
    """One-day position. Flat when the forecast does not clear the round trip."""
    if not np.isfinite(prediction) or not np.isfinite(realized):
        return float("nan")
    if prediction > cost:
        position = 1.0
    elif prediction < -cost:
        position = -1.0
    else:
        return 0.0
    return position * realized - cost


def _rule_position(row: pd.Series, rule: str) -> float:
    if rule == "momentum_5":
        lags = [row.get(f"Return_lag{i}") for i in range(1, 6)]
        if any(pd.isna(v) for v in lags):
            return 0.0
        return float(np.sign(sum(lags)))
    if rule == "ma_crossover":
        fast, slow = row.get("sma10"), row.get("sma20")
        if pd.isna(fast) or pd.isna(slow) or fast == slow:
            return 0.0
        return 1.0 if fast > slow else -1.0
    if rule == "mean_reversion":
        last = row.get("Return_lag1")
        if pd.isna(last) or last == 0:
            return 0.0
        return float(-np.sign(last))
    return 0.0


def _rule_net(position: float, realized: float, cost: float) -> float:
    if position == 0 or not np.isfinite(realized):
        return 0.0
    return float(position) * realized - cost


def score_symbol(
    symbol: str,
    frame: pd.DataFrame,
    *,
    arms: Sequence[str] = tuple(ARM_BLOCKS),
    include_gp: bool = True,
) -> Dict[str, List[dict]]:
    """Walk one symbol. Each item is one decision day for one arm or rule."""
    prepared = frame if "next_return" in frame.columns else prepare_frame(frame)
    cost = _cost_fraction(symbol)
    columns = {arm: columns_for(ARM_BLOCKS[arm]) for arm in arms}
    out: Dict[str, List[dict]] = {arm: [] for arm in arms}
    out["buy_hold"] = []
    for rule in SIMPLE_RULES:
        out[rule] = []
    if include_gp:
        out["gp_baseline"] = []

    index = prepared.index
    for i in range(MIN_TRAIN, len(prepared) - 1):
        row = prepared.iloc[i]
        realized = row["next_return"]
        if not np.isfinite(realized):
            continue
        day = pd.Timestamp(index[i]).date().isoformat()
        history = prepared.iloc[: i + 1]
        # Training rows are every bar before i whose next return is known,
        # which is rows 0..i-1. The label is next_return, already the t+1 return.
        train = history.iloc[:-1]
        y_train = train["next_return"].to_numpy(dtype=float)

        def predict(cols: List[str], model: str) -> Optional[float]:
            use = [c for c in cols if c in history.columns]
            if len(use) != len(cols):
                return None
            X_all = history[use].to_numpy(dtype=float)
            if not np.isfinite(X_all[-1]).all():
                return None
            train_X = X_all[:-1]
            mask = np.isfinite(train_X).all(axis=1) & np.isfinite(y_train)
            # News days are sparse. Dropping empty rows is correct; requiring
            # the test row itself to be finite is what skips days with no headline.
            if mask.sum() < MIN_TRAIN and "news_polarity" not in use:
                return None
            if "news_polarity" in use:
                # Score only where a headline exists. Do not fill missing news with 0.
                if not np.isfinite(X_all[-1, use.index("news_polarity")]):
                    return None
                if mask.sum() < 20:
                    return None
            train_X = train_X[mask]
            y = y_train[mask]
            X_test = X_all[-1:].copy()
            if model == "gp":
                return _fit_gp(train_X, y, X_test)
            min_rows = 20 if "news_polarity" in use else MIN_TRAIN
            return _fit_aligned(train_X, y, X_test, min_rows=min_rows)

        realized_f = float(realized)
        out["buy_hold"].append({"date": day, "symbol": symbol, "net": realized_f, "cost": cost})
        for rule in SIMPLE_RULES:
            position = _rule_position(row, rule)
            out[rule].append({
                "date": day,
                "symbol": symbol,
                "net": _rule_net(position, realized_f, cost),
            })
        for arm in arms:
            pred = predict(columns[arm], "bayesian")
            if pred is None:
                continue
            out[arm].append({
                "date": day,
                "symbol": symbol,
                "prediction": pred,
                "realized": realized_f,
                "net": _strategy_net(pred, realized_f, cost),
            })
        if include_gp:
            pred = predict(columns_for(ARM_BLOCKS["baseline"]), "gp")
            if pred is not None:
                out["gp_baseline"].append({
                    "date": day,
                    "symbol": symbol,
                    "prediction": pred,
                    "realized": realized_f,
                    "net": _strategy_net(pred, realized_f, cost),
                })
    return out


def _by_date(rows: Sequence[dict], key: str = "net") -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        grouped.setdefault(row["date"], []).append(float(row[key]))
    return grouped


def _point_mean(grouped: Dict[str, List[float]]) -> Optional[float]:
    values = [v for bucket in grouped.values() for v in bucket]
    if not values:
        return None
    return float(np.mean(values))


def bootstrap_mean_ci(
    grouped: Dict[str, List[float]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 7,
) -> tuple:
    """Day-block bootstrap. A day is drawn with every symbol that traded that day."""
    point = _point_mean(grouped)
    dates = list(grouped)
    if point is None or len(dates) < 2:
        return point, None, None
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    n = len(dates)
    for draw in range(draws):
        chosen = rng.integers(0, n, size=n)
        sample = [value for j in chosen for value in grouped[dates[j]]]
        means[draw] = float(np.mean(sample)) if sample else np.nan
    lo, hi = np.nanquantile(means, [0.025, 0.975])
    return point, float(lo), float(hi)


def _difference_by_date(left: Dict[str, List[float]], right: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Per-date mean(left) - mean(right), so unequal symbol counts still compare."""
    out: Dict[str, List[float]] = {}
    for day, values in left.items():
        other = right.get(day)
        if not other or not values:
            continue
        out[day] = [float(np.mean(values) - np.mean(other))]
    return out


def _comparison(left_rows: Sequence[dict], right_rows: Sequence[dict], *, seed: int, draws: int = BOOTSTRAP_DRAWS) -> dict:
    diff = _difference_by_date(_by_date(left_rows), _by_date(right_rows))
    mean, lo, hi = bootstrap_mean_ci(diff, draws=draws, seed=seed)
    return {
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
    }


def summarize_arm(
    name: str,
    rows: Sequence[dict],
    baseline_rows: Sequence[dict],
    buy_hold_rows: Sequence[dict],
    simple_rows: Dict[str, Sequence[dict]],
    *,
    seed: int = 7,
    draws: int = BOOTSTRAP_DRAWS,
) -> ArmResult:
    if name == "baseline":
        own_grouped = _by_date(rows)
        own_mean, own_lo, own_hi = bootstrap_mean_ci(own_grouped, draws=draws, seed=seed)
        n_days = len(own_grouped)
        return ArmResult(
            name=name,
            status="reference",
            n_observations=len(rows),
            n_days=n_days,
            incremental_mean=0.0,
            incremental_ci=[0.0, 0.0],
            own_mean_net=own_mean,
            own_ci=[own_lo, own_hi] if own_lo is not None else None,
            vs_buy_hold=_comparison(rows, buy_hold_rows, seed=seed, draws=draws),
            vs_simple={rule: _comparison(rows, simple_rows[rule], seed=seed, draws=draws) for rule in SIMPLE_RULES},
            note="Reference arm. Incremental return versus itself is zero.",
        )

    # News (and any sparse arm) is compared to baseline on the same dates only.
    arm_dates = {row["date"] for row in rows}
    baseline_matched = [row for row in baseline_rows if row["date"] in arm_dates]
    diff = _difference_by_date(_by_date(rows), _by_date(baseline_matched))
    inc_mean, inc_lo, inc_hi = bootstrap_mean_ci(diff, draws=draws, seed=seed)
    own_mean, own_lo, own_hi = bootstrap_mean_ci(_by_date(rows), draws=draws, seed=seed + 1)
    n_days = len({row["date"] for row in rows})
    status = classify_incremental(inc_mean, inc_lo, inc_hi, n_days=n_days)
    note = ""
    if name == "baseline_news" and status == "not_tested":
        note = "News is scored only on days with a stored headline. This window does not have enough of them."
    elif status == "promising":
        note = "Positive estimate, uncertainty interval still crosses zero. Keep and collect more data."
    elif status == "survives":
        note = "Promoted. Incremental net return versus baseline cleared zero."
    elif status == "fails":
        note = "Incremental net return versus baseline is not positive."
    if name == "all":
        note = (note + " " if note else "") + (
            "All is every price and calendar block. News stays a separate arm "
            "because a day with no headline is not a zero."
        )
    return ArmResult(
        name=name,
        status=status,
        n_observations=len(rows),
        n_days=n_days,
        incremental_mean=inc_mean,
        incremental_ci=[inc_lo, inc_hi] if inc_lo is not None else None,
        own_mean_net=own_mean,
        own_ci=[own_lo, own_hi] if own_lo is not None else None,
        vs_buy_hold=_comparison(rows, [r for r in buy_hold_rows if r["date"] in arm_dates], seed=seed, draws=draws),
        vs_simple={
            rule: _comparison(rows, [r for r in simple_rows[rule] if r["date"] in arm_dates], seed=seed, draws=draws)
            for rule in SIMPLE_RULES
        },
        note=note,
    )


def run_ablation(
    history: Dict[str, pd.DataFrame],
    *,
    news: Optional[Dict[str, pd.Series]] = None,
    draws: int = BOOTSTRAP_DRAWS,
    include_gp: bool = True,
    seed: int = 7,
) -> dict:
    """Run every arm on the supplied history and return the report dict."""
    pooled: Dict[str, List[dict]] = {}
    for symbol, frame in history.items():
        series = None if not news else news.get(symbol)
        prepared = prepare_frame(frame, series)
        scored = score_symbol(symbol, prepared, include_gp=include_gp)
        for key, rows in scored.items():
            pooled.setdefault(key, []).extend(rows)

    baseline = pooled.get("baseline", [])
    buy_hold = pooled.get("buy_hold", [])
    simple = {rule: pooled.get(rule, []) for rule in SIMPLE_RULES}
    arms = {}
    for name in ARM_BLOCKS:
        arms[name] = summarize_arm(
            name,
            pooled.get(name, []),
            baseline,
            buy_hold,
            simple,
            seed=seed,
            draws=draws,
        ).as_dict()

    h6_rows = pooled.get("gp_baseline", [])
    h6 = summarize_arm("gp_baseline", h6_rows, baseline, buy_hold, simple, seed=seed, draws=draws)
    h6_dict = h6.as_dict()
    h6_dict["note"] = (
        "GP versus Bayesian on lagged returns only. "
        + (h6.note or "")
    )
    report = {
        "comparisons": [
            "baseline",
            "buy_and_hold",
            "momentum_5",
            "ma_crossover",
            "mean_reversion",
        ],
        "status_rule": {
            "survives": "incremental mean > 0 and bootstrap interval entirely above 0; promote",
            "promising": "incremental mean > 0 but interval crosses 0; keep, do not delete",
            "fails": "incremental mean <= 0; remove from the live path if it is already there",
            "not_tested": "fewer than the minimum number of days",
        },
        "bootstrap": "day-block, so symbols on the same day are resampled together",
        "bootstrap_draws": draws,
        "arms": arms,
        "h6_gp_vs_bayesian": h6_dict,
    }
    return report


def write_report(report: dict, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def format_report(report: dict) -> str:
    lines = ["Ablation tournament", ""]
    header = f"{'arm':<20} {'status':<12} {'incr':>10} {'ci':>24} {'vs hold':>10} {'vs mom':>10}"
    lines.append(header)
    for name, arm in (report.get("arms") or {}).items():
        lines.append(_format_arm_line(name, arm))
    h6 = report.get("h6_gp_vs_bayesian") or {}
    lines.append(_format_arm_line("gp_vs_bayesian", h6))
    lines.append("")
    lines.append("Status uses the baseline comparison only. vs hold and vs mom are reported, not gates.")
    return "\n".join(lines)


def _format_arm_line(name: str, arm: dict) -> str:
    inc = arm.get("incremental_mean")
    ci = arm.get("incremental_ci") or [None, None]
    hold = (arm.get("vs_buy_hold") or {}).get("mean")
    mom = ((arm.get("vs_simple") or {}).get("momentum_5") or {}).get("mean")
    return (
        f"{name:<20} {str(arm.get('status')):<12} "
        f"{_fmt(inc):>10} {(_fmt(ci[0]) + ' .. ' + _fmt(ci[1])):>24} "
        f"{_fmt(hold):>10} {_fmt(mom):>10}"
    )


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.5f}"
