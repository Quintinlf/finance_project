"""Personal portfolio withdrawal planning utilities (local/private workflow).

This module is intentionally separate from the algo trading engine. It helps load
Fidelity export CSV files and generate transparent, rule-based withdrawal plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


CurrencyLike = Union[str, float, int, None]


@dataclass
class PlannerConfig:
    """Configuration for rule-based withdrawal planning."""

    ladder_percentages: Tuple[float, ...] = (30.0, 20.0, 15.0, 5.0)
    include_technical_overlay: bool = True
    technical_lookback_days: int = 45
    conservative_mode: bool = True


def get_common_target_presets() -> List[float]:
    """Return quick target presets used in the private notebook."""
    return [5.0, 10.0, 15.0, 20.0, 30.0]


def _parse_currency(value: CurrencyLike) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text == "--":
        return np.nan

    sign = -1.0 if text.startswith("-") else 1.0
    text = text.replace("$", "").replace(",", "").replace("+", "").replace("-", "")
    try:
        return sign * float(text)
    except ValueError:
        return np.nan


def _parse_percent(value: CurrencyLike) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text == "--":
        return np.nan

    sign = -1.0 if text.startswith("-") else 1.0
    text = text.replace("%", "").replace("+", "").replace("-", "")
    try:
        return sign * float(text)
    except ValueError:
        return np.nan


def _clean_symbol(symbol: CurrencyLike) -> str:
    if symbol is None:
        return ""
    return str(symbol).strip().replace("*", "")


def load_fidelity_positions(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load and normalize Fidelity portfolio export CSV.

    The Fidelity export appends disclaimer lines at the bottom; those are skipped.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Portfolio file not found: {path}")

    df = pd.read_csv(path, engine="python", skipfooter=3, index_col=False, encoding="utf-8-sig")

    # Remove unnamed trailing columns from export format.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]

    # Drop rows that do not contain a symbol entry.
    if "Symbol" in df.columns:
        df = df[df["Symbol"].astype(str).str.strip() != ""]

    rename_map = {
        "Account Number": "account_number",
        "Account Name": "account_name",
        "Symbol": "symbol_raw",
        "Description": "description",
        "Quantity": "quantity",
        "Last Price": "last_price",
        "Last Price Change": "last_price_change",
        "Current Value": "current_value",
        "Today's Gain/Loss Dollar": "today_gain_loss_dollar",
        "Today's Gain/Loss Percent": "today_gain_loss_percent",
        "Total Gain/Loss Dollar": "total_gain_loss_dollar",
        "Total Gain/Loss Percent": "total_gain_loss_percent",
        "Percent Of Account": "percent_of_account",
        "Cost Basis Total": "cost_basis_total",
        "Average Cost Basis": "average_cost_basis",
        "Type": "position_type",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Parse numeric fields.
    for col in [
        "last_price",
        "last_price_change",
        "current_value",
        "today_gain_loss_dollar",
        "total_gain_loss_dollar",
        "cost_basis_total",
        "average_cost_basis",
    ]:
        if col in df.columns:
            df[col] = df[col].map(_parse_currency)

    for col in ["today_gain_loss_percent", "total_gain_loss_percent", "percent_of_account"]:
        if col in df.columns:
            df[col] = df[col].map(_parse_percent)

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    if "symbol_raw" in df.columns:
        symbol_source = df["symbol_raw"]
    elif "symbol" in df.columns:
        symbol_source = df["symbol"]
    else:
        symbol_source = pd.Series([""] * len(df), index=df.index)
    df["symbol"] = symbol_source.map(_clean_symbol)

    # Keep rows with a positive current value for planning.
    if "current_value" in df.columns:
        df = df[df["current_value"].fillna(0.0) > 0.0].copy()

    df["is_cash_like"] = df["symbol"].str.upper().eq("SPAXX")
    return df.reset_index(drop=True)


def combine_portfolio_exports(paths: Iterable[Union[str, Path]]) -> pd.DataFrame:
    """Combine multiple fidelity export files into one normalized frame."""
    frames = [load_fidelity_positions(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    # Keep the latest row for each account/symbol pair by current value priority.
    combined = combined.sort_values(by=["account_number", "symbol", "current_value"], ascending=[True, True, False])
    combined = combined.drop_duplicates(subset=["account_number", "symbol"], keep="first")
    return combined.reset_index(drop=True)


def _fetch_technical_scores(symbols: Sequence[str], lookback_days: int = 45) -> Dict[str, float]:
    """Return sell-priority multipliers from simple candlestick/trend context.

    Higher score means stronger preference to sell from that position.
    """
    scores = {s: 1.0 for s in symbols}
    if yf is None:
        return scores

    period = f"{max(lookback_days, 10)}d"
    for symbol in symbols:
        if not symbol:
            continue
        try:
            history = yf.Ticker(symbol).history(period=period, interval="1d")
            if history.empty or len(history) < 12:
                continue

            close = history["Close"].astype(float)
            open_ = history["Open"].astype(float)

            ma10 = float(close.tail(10).mean())
            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-6]) if len(close) >= 6 else float(close.iloc[0])
            trend = (last_close - prev_close) / max(abs(prev_close), 1e-8)

            # Weak trend and red candle increase sell preference.
            score = 1.0
            if last_close < ma10:
                score += 0.20
            if trend < 0:
                score += min(0.20, abs(trend) * 2.0)
            if float(close.iloc[-1]) < float(open_.iloc[-1]):
                score += 0.10

            scores[symbol] = float(np.clip(score, 0.75, 1.75))
        except Exception:
            # Keep default when data fetch fails.
            continue

    return scores


def _base_sell_scores(df: pd.DataFrame, conservative_mode: bool = True) -> pd.Series:
    """Compute transparent base sell-priority score per position."""
    score = pd.Series(1.0, index=df.index, dtype=float)

    # Prefer trimming winners and high concentration first.
    if "total_gain_loss_percent" in df.columns:
        gain_component = df["total_gain_loss_percent"].fillna(0.0) / 100.0
        score = score + gain_component.clip(lower=-0.20, upper=0.40)

    if "percent_of_account" in df.columns:
        concentration_component = df["percent_of_account"].fillna(0.0) / 100.0
        score = score + concentration_component.clip(lower=0.0, upper=0.30)

    if conservative_mode:
        # In conservative mode, avoid liquidating deep losers aggressively.
        if "total_gain_loss_percent" in df.columns:
            deep_loser_penalty = (df["total_gain_loss_percent"].fillna(0.0) < -25.0).astype(float) * 0.20
            score = score - deep_loser_penalty

    return score.clip(lower=0.25, upper=2.50)


def plan_withdrawal(
    positions: pd.DataFrame,
    target_amount: float,
    config: Optional[PlannerConfig] = None,
) -> pd.DataFrame:
    """Build a rule-based withdrawal plan for a requested cash amount.

    The planner allocates sell dollars across positions by priority and ladder caps
    (e.g. 30%, then 20%, then 15%, then 5% of each position value per pass).
    """
    if config is None:
        config = PlannerConfig()

    if target_amount <= 0:
        raise ValueError("target_amount must be positive")

    df = positions.copy()
    required_cols = {"symbol", "current_value", "quantity", "last_price", "account_name"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"positions missing required columns: {missing}")

    # Exclude cash-like rows from sell recommendations.
    if "is_cash_like" in df.columns:
        df = df[~df["is_cash_like"].fillna(False)].copy()

    df = df[df["current_value"].fillna(0.0) > 0.0].copy()
    if df.empty:
        raise ValueError("No eligible positions with positive current value")

    gain_pct = (
        df["total_gain_loss_percent"].fillna(0.0)
        if "total_gain_loss_percent" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    concentration_pct = (
        df["percent_of_account"].fillna(0.0)
        if "percent_of_account" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    base_scores = _base_sell_scores(df, conservative_mode=config.conservative_mode)

    if config.include_technical_overlay:
        tech_scores = _fetch_technical_scores(df["symbol"].tolist(), lookback_days=config.technical_lookback_days)
        tech_mult = df["symbol"].map(lambda s: tech_scores.get(str(s), 1.0))
    else:
        tech_mult = pd.Series(1.0, index=df.index)

    df["sell_priority_score"] = (base_scores * tech_mult).clip(lower=0.20, upper=3.00)
    df["_gain_pct"] = gain_pct
    df["_concentration_pct"] = concentration_pct
    df["_tech_multiplier"] = tech_mult

    def _build_rationale(row: pd.Series) -> str:
        reasons: List[str] = []

        gain = float(row.get("_gain_pct", 0.0))
        concentration = float(row.get("_concentration_pct", 0.0))
        tech = float(row.get("_tech_multiplier", 1.0))

        if gain > 0:
            reasons.append("trim winner")
        if concentration >= 8.0:
            reasons.append("high concentration")
        if config.conservative_mode and gain < -25.0:
            reasons.append("conservative loser protection")
        if config.include_technical_overlay and tech > 1.1:
            reasons.append("weak short-term trend")

        if not reasons:
            reasons.append("balanced diversification trim")
        return ", ".join(reasons)

    df["sell_rationale"] = df.apply(_build_rationale, axis=1)
    df = df.sort_values(by=["sell_priority_score", "current_value"], ascending=[False, False]).reset_index(drop=True)

    remaining = float(target_amount)
    allocation = pd.Series(0.0, index=df.index)

    # Ladder passes across ranked positions.
    for rung_pct in config.ladder_percentages:
        if remaining <= 1e-9:
            break

        rung_ratio = max(float(rung_pct), 0.0) / 100.0
        if rung_ratio <= 0.0:
            continue

        for idx in df.index:
            if remaining <= 1e-9:
                break

            position_value_raw = pd.to_numeric(df.at[idx, "current_value"], errors="coerce")
            already_raw = pd.to_numeric(allocation.at[idx], errors="coerce")
            position_value = float(position_value_raw) if pd.notna(position_value_raw) else 0.0
            already = float(already_raw) if pd.notna(already_raw) else 0.0
            step_cap = position_value * rung_ratio
            max_remaining_for_position = max(position_value - already, 0.0)
            step_available = max(min(step_cap, max_remaining_for_position), 0.0)
            if step_available <= 0.0:
                continue

            take = min(step_available, remaining)
            allocation.at[idx] = already + take
            remaining -= take

    # If the target is not reached, use weighted top-off from remaining capacity.
    if remaining > 1e-9:
        capacities = (df["current_value"] - allocation).clip(lower=0.0)
        weighted_capacity = capacities * df["sell_priority_score"]
        total_weighted_capacity = float(weighted_capacity.sum())

        if total_weighted_capacity > 0.0:
            proportional_take = (weighted_capacity / total_weighted_capacity) * remaining
            proportional_take = np.minimum(proportional_take, capacities)
            allocation += proportional_take
            remaining -= float(proportional_take.sum())

    df["suggested_sell_value"] = allocation.round(2)
    df = df[df["suggested_sell_value"] > 0.0].copy()

    price_ref = df["last_price"].copy()
    missing_price = price_ref.isna() | (price_ref <= 0)
    price_ref.loc[missing_price] = (
        df.loc[missing_price, "current_value"] / df.loc[missing_price, "quantity"].replace(0, np.nan)
    )
    df["price_reference"] = price_ref

    df["estimated_shares_to_sell"] = (
        df["suggested_sell_value"] / df["price_reference"].replace(0, np.nan)
    ).fillna(0.0)

    df["suggested_sell_pct_of_position"] = (
        (df["suggested_sell_value"] / df["current_value"].replace(0, np.nan)) * 100.0
    ).fillna(0.0)

    df["remaining_position_value"] = (df["current_value"] - df["suggested_sell_value"]).clip(lower=0.0).round(2)

    selected_cols = [
        "account_name",
        "symbol",
        "description",
        "quantity",
        "current_value",
        "sell_priority_score",
        "sell_rationale",
        "suggested_sell_value",
        "suggested_sell_pct_of_position",
        "estimated_shares_to_sell",
        "price_reference",
        "remaining_position_value",
    ]
    selected_cols = [c for c in selected_cols if c in df.columns]
    plan = df[selected_cols].sort_values(by=["suggested_sell_value", "sell_priority_score"], ascending=[False, False])

    plan.attrs["requested_target_amount"] = float(target_amount)
    plan.attrs["planned_total"] = float(plan["suggested_sell_value"].sum()) if "suggested_sell_value" in plan.columns else 0.0
    plan.attrs["unmet_amount"] = float(max(remaining, 0.0))
    return plan.reset_index(drop=True)


def summarize_portfolio(positions: pd.DataFrame) -> Dict[str, float]:
    """Return high-level portfolio totals for notebook display."""
    total_value = float(positions.get("current_value", pd.Series(dtype=float)).fillna(0.0).sum())
    total_gain = float(positions.get("total_gain_loss_dollar", pd.Series(dtype=float)).fillna(0.0).sum())
    positions_count = int((positions.get("current_value", pd.Series(dtype=float)).fillna(0.0) > 0).sum())

    return {
        "total_value": round(total_value, 2),
        "total_gain_loss_dollar": round(total_gain, 2),
        "positions_count": float(positions_count),
    }


def summarize_portfolio_by_account(positions: pd.DataFrame) -> pd.DataFrame:
    """Return per-account totals for quick Roth vs brokerage visibility."""
    if positions.empty or "account_name" not in positions.columns:
        return pd.DataFrame(columns=["account_name", "total_value", "total_gain_loss_dollar", "positions_count"])

    grouped = (
        positions.groupby("account_name", dropna=False)
        .agg(
            total_value=("current_value", "sum"),
            total_gain_loss_dollar=("total_gain_loss_dollar", "sum"),
            positions_count=("symbol", "count"),
        )
        .reset_index()
    )
    grouped["portfolio_weight_pct"] = (
        grouped["total_value"] / max(float(grouped["total_value"].sum()), 1e-8)
    ) * 100.0
    return grouped.sort_values("total_value", ascending=False).reset_index(drop=True)


def build_sell_checklist(plan: pd.DataFrame) -> pd.DataFrame:
    """Create a concise execution checklist from the raw withdrawal plan."""
    if plan.empty:
        return plan.copy()

    checklist_cols = [
        "account_name",
        "symbol",
        "suggested_sell_value",
        "estimated_shares_to_sell",
        "suggested_sell_pct_of_position",
        "sell_rationale",
    ]
    checklist_cols = [c for c in checklist_cols if c in plan.columns]
    checklist = plan[checklist_cols].copy()

    if "suggested_sell_value" in checklist.columns:
        checklist = checklist.sort_values("suggested_sell_value", ascending=False)

    if "estimated_shares_to_sell" in checklist.columns:
        checklist["estimated_shares_to_sell"] = checklist["estimated_shares_to_sell"].round(4)
    if "suggested_sell_pct_of_position" in checklist.columns:
        checklist["suggested_sell_pct_of_position"] = checklist["suggested_sell_pct_of_position"].round(2)

    return checklist.reset_index(drop=True)
