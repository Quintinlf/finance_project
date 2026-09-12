"""Does the signal work at any horizon, or only the one we happened to measure?

Every accuracy number so far graded predictions against the NEXT DAY's return.
That is one arbitrary choice out of many, and it is the noisiest one available:
daily returns are dominated by microstructure and news, while whatever a
Bayesian/GP fit on 250 daily bars is picking up -- if anything -- is more
plausibly a slower drift.

So this asks the obvious question the pipeline never asked: holding the
predictions fixed, does accuracy or information coefficient improve at 5, 10,
or 20 days?

Two honest caveats, stated up front because they decide how to read the output:

1. **Multiple comparisons.** Testing 4 horizons x 5 components is 20 tests. At
   a 5% threshold roughly one will look "significant" by chance alone. A
   Bonferroni-corrected threshold is printed alongside the raw one, and a
   result that only clears the raw threshold should be treated as noise.

2. **Overlapping windows.** A 20-day forward return computed on consecutive
   days shares 19 of its 20 days with the previous observation. The effective
   sample is therefore far smaller than the row count, so the naive confidence
   interval is too narrow. An overlap-adjusted effective N is reported and
   used for the intervals.

    python scripts/analyze_horizons.py
    python scripts/analyze_horizons.py --horizons 1,5,20 --scope agriculture
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.price_cache import get_history  # noqa: E402
from logic.sqlite_store import DEFAULT_DB_PATH  # noqa: E402

COMPONENTS = [
    ("Bollinger", "bb"),
    ("Bayesian", "bayesian"),
    ("GaussianProc", "gp"),
    ("RSI", "rsi"),
    ("ENSEMBLE", "ensemble"),
]


def load_predictions(db_path, scope_symbols: Optional[Sequence[str]] = None) -> List[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cols = ", ".join(
            f"{p}_direction, {p}_confidence" for _, p in COMPONENTS
        )
        rows = conn.execute(
            f"SELECT symbol, timestamp, {cols} FROM model_component_performance"
        ).fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        symbol = str(r["symbol"])
        if scope_symbols and symbol not in scope_symbols:
            continue
        out.append(dict(r))
    return out


def forward_returns(symbol: str, horizons: Sequence[int]) -> Dict[date, Dict[int, float]]:
    """Forward return at each horizon for every bar, from the cached series."""
    hist = get_history(symbol)
    if hist is None or hist.empty:
        return {}

    closes = [float(c) for c in hist["Close"]]
    dates = [idx.date() for idx in hist.index]
    by_date: Dict[date, Dict[int, float]] = {}

    for i, d in enumerate(dates):
        entry: Dict[int, float] = {}
        for h in horizons:
            j = i + h
            if j >= len(closes) or closes[i] <= 0:
                continue
            ret = closes[j] / closes[i] - 1.0
            # Same guard as the scorer: a move this large across a few sessions
            # is a corporate action the adjustment missed, not a return.
            if abs(ret) > 0.5 * max(1, h / 5):
                continue
            entry[h] = ret
        if entry:
            by_date[d] = entry
    return by_date


def _signed(direction: Optional[str], confidence: Optional[float]) -> Optional[float]:
    """Direction and confidence as one signed number, for correlation."""
    if direction is None:
        return None
    d = str(direction).lower()
    conf = float(confidence or 0.0)
    if d in ("buy", "long", "up"):
        return conf
    if d in ("sell", "short", "down"):
        return -conf
    if d in ("neutral", "hold", "flat"):
        return 0.0
    return None


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    try:
        from scipy import stats

        return float(stats.spearmanr(xs, ys).statistic)
    except Exception:  # noqa: BLE001
        return float("nan")


def _effective_n(n: int, horizon: int) -> float:
    """Shrink N for overlapping forward windows.

    Consecutive h-day forward returns share h-1 days, so treating them as
    independent makes every confidence interval too narrow. Dividing by the
    horizon is the standard rough correction.
    """
    return max(1.0, n / float(max(1, horizon)))


def analyze(db_path, horizons: Sequence[int], scope_symbols=None, out_path=None) -> dict:
    preds = load_predictions(db_path, scope_symbols)
    logging.info("Loaded %s predictions", len(preds))

    symbols = sorted({p["symbol"] for p in preds})
    fwd: Dict[str, Dict[date, Dict[int, float]]] = {}
    for symbol in symbols:
        fwd[symbol] = forward_returns(symbol, horizons)
    logging.info("Forward returns built for %s symbol(s)", len(fwd))

    # base rate per horizon: how often did price simply rise?
    base: Dict[int, List[float]] = defaultdict(list)
    # (component, horizon) -> (hits, total, signed values, returns)
    acc: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0])
    ic_data: Dict[Tuple[str, int], Tuple[List[float], List[float]]] = defaultdict(
        lambda: ([], [])
    )

    for p in preds:
        symbol = p["symbol"]
        try:
            d = date.fromisoformat(str(p["timestamp"])[:10])
        except ValueError:
            continue
        rets = fwd.get(symbol, {}).get(d)
        if not rets:
            continue

        for h, ret in rets.items():
            base[h].append(1.0 if ret > 0 else 0.0)

        for label, prefix in COMPONENTS:
            signal = _signed(p.get(f"{prefix}_direction"), p.get(f"{prefix}_confidence"))
            if signal is None:
                continue
            for h, ret in rets.items():
                xs, ys = ic_data[(label, h)]
                xs.append(signal)
                ys.append(ret)
                if signal == 0.0:
                    continue  # a neutral call makes no directional claim
                hit = 1 if ((signal > 0) == (ret > 0)) else 0
                acc[(label, h)][0] += hit
                acc[(label, h)][1] += 1

    print()
    print("=" * 88)
    print("FORWARD-HORIZON ANALYSIS")
    print("=" * 88)
    for h in horizons:
        b = base[h]
        print(f"  {h:>3}-day base rate (price simply rose): "
              f"{100.0 * sum(b) / len(b):.1f}%  (n={len(b)})" if b else f"  {h}-day: no data")

    # 20 tests -> one will look significant by chance at p<0.05.
    n_tests = len(COMPONENTS) * len(horizons)
    print()
    print(f"  {n_tests} tests run. Raw |IC| threshold shown alongside a "
          f"Bonferroni-corrected one;")
    print( "  a result clearing only the raw bar is noise, not a finding.")
    print()
    results: List[dict] = []
    print(f"  {'component':<14}{'horizon':>8}{'n':>8}{'eff_n':>8}"
          f"{'acc':>8}{'vs base':>9}{'IC':>9}{'|IC| bar':>10}  verdict")
    print("  " + "-" * 86)

    for label, _prefix in COMPONENTS:
        for h in horizons:
            hits, total = acc[(label, h)]
            xs, ys = ic_data[(label, h)]
            if total < 30 or len(xs) < 30:
                continue

            p_hat = hits / total
            eff_n = _effective_n(total, h)
            se = math.sqrt(max(p_hat * (1 - p_hat), 0.0) / eff_n)
            base_rate = sum(base[h]) / len(base[h]) if base[h] else 0.5

            ic = _spearman(xs, ys)
            # |IC| that would be significant at this effective sample size.
            ic_raw_bar = 1.96 / math.sqrt(max(2.0, _effective_n(len(xs), h)))
            ic_bonf_bar = (1.96 + math.log(n_tests)) / math.sqrt(
                max(2.0, _effective_n(len(xs), h))
            )

            if not math.isnan(ic) and abs(ic) > ic_bonf_bar:
                verdict = "SURVIVES correction" + (" (wrong sign)" if ic < 0 else "")
            elif not math.isnan(ic) and abs(ic) > ic_raw_bar:
                verdict = "raw-significant only -> noise"
            elif p_hat - 1.96 * se > base_rate:
                verdict = "beats base rate"
            else:
                verdict = "nothing"

            print(f"  {label:<14}{h:>8}{total:>8}{eff_n:>8.0f}"
                  f"{p_hat * 100:>7.1f}%{(p_hat - base_rate) * 100:>+8.1f}pp"
                  f"{ic:>9.4f}{ic_bonf_bar:>10.4f}  {verdict}")

            results.append({
                "component": label, "horizon": h, "n": total,
                "effective_n": round(eff_n), "accuracy": round(p_hat * 100, 1),
                "base_rate": round(base_rate * 100, 1),
                "vs_base": round((p_hat - base_rate) * 100, 1),
                "ic": None if math.isnan(ic) else round(ic, 4),
                "ic_threshold": round(ic_bonf_bar, 4),
                "verdict": verdict,
            })

    print()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tests": n_tests,
        "base_rates": {
            str(h): round(100.0 * sum(base[h]) / len(base[h]), 1)
            for h in horizons if base[h]
        },
        "results": results,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  -> {out_path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", default="1,5,10,20")
    ap.add_argument("--scope", default=None, help="limit to a universe scope")
    ap.add_argument("--out", default="trade_logs/horizon_analysis.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    scope_symbols = None
    if args.scope:
        from logic.universe import get_symbols

        scope_symbols = set(get_symbols(args.scope))

    analyze(DEFAULT_DB_PATH, horizons, scope_symbols, out_path=args.out)


if __name__ == "__main__":
    main()
