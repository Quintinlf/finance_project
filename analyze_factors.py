"""Which inputs, if any, predict forward returns?

    python analyze_factors.py
    python analyze_factors.py --horizon 3

Yesterday's lesson was that a statistic from one window means nothing: the
signal's information coefficient read -0.098 in one period and +0.043 in the
next, and both looked significant. So every number here is computed on two
independent windows and printed side by side. A feature is interesting only if
it keeps its sign and rough magnitude across both.

Three analyses, all via the eos finance domains (imported through
`logic.eos_bridge`, since MCP cannot reach a headless process):

1. **PCA on the returns matrix** (`finance.pca_factor_model`). How many
   independent bets are actually in the universe? 24 tickers that move as 4
   factors give an effective sample far below the raw row count, which is
   precisely why the naive confidence intervals lied.

2. **Feature importance** (`finance.ridge_regression` +
   `finance.permutation_feature_importance`). Tests the *inputs* rather than
   the assembled model, so a weak-but-real feature is not masked by the
   ensemble that consumes it.

3. **Hypothesis tests** (`physics_finance.regression_hypothesis_test`).
   Per-coefficient t-statistics and an overall F-test.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from logic import eos_bridge
from logic.signal_quality import attach_forward_returns, load_decisions

BACKTEST_DIR = Path("trade_logs") / "backtests"

# Features carried on every decision record. Deliberately the raw inputs, not
# the model's own output, apart from prob_profit/confidence which are included
# to see whether the ensemble adds anything over its parts.
FEATURE_KEYS: Tuple[str, ...] = (
    "rsi",
    "bb_z_score",
    "confidence",
    "prob_profit",
    "belief_entropy",
)


def _standardize(columns: List[List[float]]) -> List[List[float]]:
    """Z-score each column so ridge's penalty falls evenly across features."""
    out: List[List[float]] = []
    for column in columns:
        n = len(column)
        mean = sum(column) / n
        var = sum((v - mean) ** 2 for v in column) / max(1, n - 1)
        sd = math.sqrt(var) if var > 0 else 1.0
        out.append([(v - mean) / sd for v in column])
    return out


def build_feature_matrix(
    backtest_path: Path, horizon: int, price_history: Dict[str, Any]
) -> Tuple[List[List[float]], List[float], List[str], int]:
    """Assemble (X, y, feature_names, n_rows) from a backtest's decision log."""
    import json

    payload = json.loads(Path(backtest_path).read_text(encoding="utf-8"))
    raw_rows = payload.get("decisions", [])

    decisions = load_decisions(backtest_path)
    enriched = attach_forward_returns(decisions, price_history, horizons=(horizon,))
    # Key the raw rows so the extra feature columns can be recovered.
    by_key = {(str(r.get("date")), str(r.get("symbol"))): r for r in raw_rows}

    # Discover regime sub-keys from the first row that carries them.
    regime_keys: List[str] = []
    for row in raw_rows:
        regime = row.get("market_regime")
        if isinstance(regime, dict) and regime:
            regime_keys = sorted(regime)
            break

    feature_names = list(FEATURE_KEYS) + [f"regime.{k}" for k in regime_keys]
    rows: List[List[float]] = []
    targets: List[float] = []

    for decision in enriched:
        raw = by_key.get((decision.date, decision.symbol))
        if raw is None:
            continue
        values: List[float] = []
        ok = True
        for key in FEATURE_KEYS:
            value = raw.get(key)
            if value is None:
                ok = False
                break
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                ok = False
                break
            if not math.isfinite(fvalue):
                ok = False
                break
            values.append(fvalue)
        if not ok:
            continue
        regime = raw.get("market_regime") or {}
        if regime_keys:
            if not isinstance(regime, dict):
                continue
            for key in regime_keys:
                try:
                    values.append(float(regime.get(key, 0.0)))
                except (TypeError, ValueError):
                    values.append(0.0)
        rows.append(values)
        targets.append(decision.forward_returns[horizon])

    if not rows:
        return [], [], feature_names, 0

    # Standardize column-wise, then transpose back to row-major.
    columns = [[row[i] for row in rows] for i in range(len(feature_names))]
    columns = _standardize(columns)
    X = [[columns[i][r] for i in range(len(feature_names))] for r in range(len(rows))]
    return X, targets, feature_names, len(rows)


def factor_structure(price_history: Dict[str, Any], n_components: int = 6) -> Optional[Dict[str, Any]]:
    """PCA on aligned daily returns: how many independent bets are there?"""
    symbols = sorted(price_history)
    series: Dict[str, List[float]] = {}
    for symbol in symbols:
        returns = eos_bridge.extract_returns(price_history[symbol])
        if returns:
            series[symbol] = returns
    if len(series) < 3:
        return None

    length = min(len(v) for v in series.values())
    if length < 30:
        return None
    names = sorted(series)
    # pca_factor_model wants observations as rows, assets as columns.
    matrix = [[series[name][-length:][t] for name in names] for t in range(length)]
    try:
        return eos_bridge.call(
            "finance.pca_factor_model",
            returns=matrix,
            n_components=min(n_components, len(names)),
            asset_names=names,
        )
    except Exception as exc:
        logging.warning("PCA failed: %s", exc)
        return None


def effective_sample_shrink(horizon: int, n_symbols: int, n_factors: int) -> float:
    """How much to divide raw t-statistics by, given non-independent rows.

    Two violations of the independence the OLS t-statistic assumes, and both
    are large here:

    * **Overlapping targets.** An h-day forward return computed on consecutive
      bars shares h-1 days with its neighbour, deflating the variance of the
      estimate by roughly a factor of h.
    * **Cross-sectional correlation.** The rows are not `n_symbols` independent
      bets. PCA says the universe collapses to `n_factors` — SOYB/CORN/WEAT/DBA
      are essentially one grain factor — so the symbol dimension contributes
      only `n_factors` worth of independent information.

    Effective n is therefore about raw_n / (horizon * n_symbols / n_factors),
    and a t-statistic scales with sqrt(n). Ignoring this is exactly how the
    signal's information coefficient looked significant in two windows with
    opposite signs.
    """
    ratio = max(1.0, float(n_symbols) / max(1, n_factors))
    return math.sqrt(max(1.0, float(horizon)) * ratio)


def analyse_window(
    X: List[List[float]], y: List[float], feature_names: List[str]
) -> Dict[str, Any]:
    """Ridge fit, permutation importance, and per-coefficient t-tests."""
    result: Dict[str, Any] = {}
    if not X or len(X) < 30:
        return result
    try:
        result["ridge"] = eos_bridge.call(
            "finance.ridge_regression", X=X, y=y, alpha=1.0, feature_names=feature_names
        )
    except Exception as exc:
        logging.warning("ridge failed: %s", exc)
    try:
        result["importance"] = eos_bridge.call(
            "finance.permutation_feature_importance",
            X=X, y=y, alpha=1.0, feature_names=feature_names, n_repeats=10, seed=7,
        )
    except Exception as exc:
        logging.warning("permutation importance failed: %s", exc)
    try:
        result["hypothesis"] = eos_bridge.call(
            "physics_finance.regression_hypothesis_test",
            X=X, y=y, feature_names=feature_names,
        )
    except Exception as exc:
        logging.warning("hypothesis test failed: %s", exc)
    return result


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-a", type=Path,
                        default=BACKTEST_DIR / "backtest_20260811_220137.json")
    parser.add_argument("--window-b", type=Path,
                        default=BACKTEST_DIR / "oos_window.json")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    if not eos_bridge.available():
        raise SystemExit(f"eos domains unavailable: {eos_bridge.EOS_IMPORT_ERROR}")

    import yfinance as yf

    windows: List[Tuple[str, Path]] = [("A", args.window_a), ("B", args.window_b)]
    loaded: Dict[str, Dict[str, Any]] = {}

    for label, path in windows:
        if not path.exists():
            raise SystemExit(f"Missing backtest: {path}")
        decisions = load_decisions(path)
        symbols = sorted({d.symbol for d in decisions})
        dates = sorted(d.date for d in decisions)
        print(f"Window {label}: {path.name} | {len(decisions)} decisions | "
              f"{len(symbols)} symbols | {dates[0]} -> {dates[-1]}")

        history: Dict[str, Any] = {}
        for symbol in symbols:
            try:
                frame = yf.Ticker(symbol).history(start=dates[0], interval="1d")
                if not frame.empty:
                    history[symbol] = frame
            except Exception as exc:
                logging.warning("history failed for %s: %s", symbol, exc)

        X, y, names, n = build_feature_matrix(path, args.horizon, history)
        loaded[label] = {
            "path": path, "history": history, "X": X, "y": y,
            "names": names, "n": n, "dates": (dates[0], dates[-1]),
        }
        print(f"  -> {n} usable rows, {len(names)} features")
    print()

    # --- 1. Factor structure -------------------------------------------------
    print("=" * 78)
    print("1. FACTOR STRUCTURE  (how many independent bets are in the universe?)")
    print("=" * 78)
    shrink: Dict[str, float] = {}
    for label, _ in windows:
        pca = factor_structure(loaded[label]["history"])
        n_symbols = len(loaded[label]["history"]) or 1
        if not pca:
            print(f"  Window {label}: PCA unavailable; assuming no compression")
            shrink[label] = effective_sample_shrink(args.horizon, n_symbols, n_symbols)
            continue
        ratios = pca["explained_variance_ratio"]
        cumulative = 0.0
        n_for_80 = len(ratios)
        for i, ratio in enumerate(ratios, 1):
            cumulative += ratio
            if cumulative >= 0.80:
                n_for_80 = i
                break
        shown = ", ".join(f"{r*100:.1f}%" for r in ratios[:6])
        print(f"  Window {label}: components explain [{shown}]")
        print(f"             {n_for_80} component(s) cover 80% of variance "
              f"across {len(pca['asset_names'])} symbols")
        shrink[label] = effective_sample_shrink(args.horizon, n_symbols, n_for_80)
        n_eff = int(loaded[label]["n"] / (shrink[label] ** 2))
        print(f"             -> effective n = {n_eff} (not {loaded[label]['n']}); "
              f"t-statistics divided by {shrink[label]:.2f}")
    print()

    # --- 2 & 3. Features -----------------------------------------------------
    print("=" * 78)
    print(f"2. FEATURE -> {args.horizon}d FORWARD RETURN  (both windows, side by side)")
    print("=" * 78)
    results = {label: analyse_window(loaded[label]["X"], loaded[label]["y"],
                                     loaded[label]["names"])
               for label, _ in windows}

    names = loaded["A"]["names"]
    ra, rb = results.get("A", {}), results.get("B", {})
    if not ra or not rb:
        print("  Insufficient data for regression on one or both windows.")
        return 1

    print(f"  Ridge R^2:  window A = {ra['ridge']['r_squared']:+.5f}   "
          f"window B = {rb['ridge']['r_squared']:+.5f}")
    print("  (R^2 near zero means the features explain essentially none of the")
    print("   variation in forward returns.)")
    print()

    ha, hb = ra.get("hypothesis", {}), rb.get("hypothesis", {})
    ia, ib = ra.get("importance", {}), rb.get("importance", {})

    print("  t-adj is the raw t-statistic divided by the shrink factor above.")
    print("  |t-adj| > 1.96 in BOTH windows, with the same sign, is the bar.")
    print()
    print(f"  {'FEATURE':<20} {'COEF A':>10} {'t A':>7} {'t-adj A':>8} "
          f"{'COEF B':>10} {'t B':>7} {'t-adj B':>8}  STABLE?")
    print(f"  {'-'*20} {'-'*10} {'-'*7} {'-'*8} {'-'*10} {'-'*7} {'-'*8}  {'-'*7}")
    stable_hits: List[str] = []
    for i, name in enumerate(names):
        try:
            ca, ta = ha["coefficients"][i], ha["t_statistics"][i]
            cb, tb = hb["coefficients"][i], hb["t_statistics"][i]
        except (KeyError, IndexError):
            continue
        if ta is None or tb is None or math.isnan(ta) or math.isnan(tb):
            continue
        adj_a, adj_b = ta / shrink["A"], tb / shrink["B"]
        same_sign = (ca > 0) == (cb > 0)
        both_sig = abs(adj_a) > 1.96 and abs(adj_b) > 1.96
        verdict = "YES" if (same_sign and both_sig) else ("sign" if same_sign else "no")
        if same_sign and both_sig:
            stable_hits.append(name)
        print(f"  {name:<20} {ca:>+10.5f} {ta:>7.2f} {adj_a:>8.2f} "
              f"{cb:>+10.5f} {tb:>7.2f} {adj_b:>8.2f}  {verdict}")
    print()

    if ia and ib:
        print("  Permutation importance (drop in R^2 when the column is shuffled):")
        for i, name in enumerate(names):
            try:
                va = ia["importances_mean"][i]
                vb = ib["importances_mean"][i]
            except (KeyError, IndexError):
                continue
            print(f"    {name:<20} A={va:+.6f}   B={vb:+.6f}")
        print()

    print("  Overall F-test:")
    for label, h in (("A", ha), ("B", hb)):
        if h:
            print(f"    window {label}: F={h['f_statistic']:.3f}  p={h['f_p_value']:.4f}  "
                  f"adj R^2={h['adjusted_r_squared']:+.5f}")
    print()

    print("=" * 78)
    if stable_hits:
        print(f"  STABLE ACROSS BOTH WINDOWS: {', '.join(stable_hits)}")
        print("  Same sign and p<0.05 in two independent periods. Worth a")
        print("  dedicated strategy test — but confirm on a third window first.")
    else:
        print("  NOTHING STABLE. No feature clears |t-adj| > 1.96 with a consistent")
        print("  sign across both windows. On this evidence the inputs carry no")
        print("  usable directional information, and reweighting them cannot")
        print("  create any. Note how far the raw t-statistics overstate this:")
        print("  that gap is the whole reason single-window results looked good.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
