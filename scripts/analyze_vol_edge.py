"""Gate 0: does our volatility forecast carry information implied vol does not?

This is the question the whole Tier 2 scope rests on, and it is not "can we
predict volatility." We can — trailing realized vol predicts forward realized
vol at r = 0.708. That number is an edge over *nothing*. We would be trading
against implied volatility, which is the options market's own forecast of the
identical quantity, and it is already very good.

So the test is a nested regression:

    realized_vol ~ a + b * implied_vol                      (restricted)
    realized_vol ~ a + b * implied_vol + c * our_forecast   (full)

Gate 0 passes only if `c` is significantly non-zero. A high R-squared on our
forecast alone proves nothing whatsoever — implied vol would score similarly,
and the two are heavily collinear. What matters is the *incremental* term.

### Why this reports INSUFFICIENT for months

Each observation needs `horizon` days of forward data, and consecutive daily
observations share almost all of their window. Overlapping windows inflate the
row count without adding information, so the script reports an effective sample
size, not a raw one:

    effective_N ~= raw_N / horizon_days

and then discounts further for cross-sectional correlation, because 19 commodity
ETFs on the same day are nothing like 19 independent draws (GLD and IAU are the
same asset; DBC and GSG are near-duplicates).

This project has twice been burned by exactly this. Component accuracy that
looked like "21.4% worse than chance" at n=14 became 49.9% at n=6,814, and the
base rate itself moved from 75.6% to 51.6%. A number produced here before the
sample supports it would not be a preliminary answer — it would be noise with a
p-value attached, and acting on it would be worse than waiting.

### Horizons, and why the short one matters

Shorter horizons resolve faster, because the windows overlap less: a 7-day test
reaches a usable effective sample in roughly a quarter the calendar time of a
30-day test. The short horizon is therefore the one to watch first. Its
weakness is that 7-day realized vol is a noisy estimate of anything, so it will
show a weaker relationship even if the effect is real — treat it as an early
warning rather than the verdict.

Usage:
    python scripts/analyze_vol_edge.py
    python scripts/analyze_vol_edge.py --horizon 7
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logic.iv_snapshot import coverage, init_iv_tables
from logic.sqlite_store import DEFAULT_DB_PATH, connect

HORIZONS = (7, 14, 30)

# Effective (not raw) observations required. Below the first, any number printed
# is noise. Between the two, the direction is worth looking at but not acting on.
MIN_EFFECTIVE_N_PRELIMINARY = 30
MIN_EFFECTIVE_N_CONCLUSIVE = 100

# 19 commodity ETFs on one day are not 19 independent observations. GLD/IAU are
# the same metal, DBC/GSG are near-duplicate baskets, and the energy and grain
# complexes each move together. A factor of ~4 effective independent clusters is
# a generous read of this universe; the discount is applied to effective_N.
EFFECTIVE_SYMBOL_CLUSTERS = 4.0


def _ols_with_r2(y, xs):
    """Least squares via normal equations. Returns (coeffs, r2, resid_var, n).

    Written out rather than pulled from statsmodels because the only consumer is
    this script and the matrix is at most 3x3.
    """
    n = len(y)
    k = len(xs) + 1
    if n <= k:
        return None, None, None, n

    cols = [[1.0] * n] + [list(x) for x in xs]
    xtx = [[sum(cols[i][r] * cols[j][r] for r in range(n)) for j in range(k)] for i in range(k)]
    xty = [sum(cols[i][r] * y[r] for r in range(n)) for i in range(k)]

    # Gaussian elimination with partial pivoting.
    aug = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(aug[r][c]))
        if abs(aug[piv][c]) < 1e-12:
            return None, None, None, n
        aug[c], aug[piv] = aug[piv], aug[c]
        for r in range(k):
            if r == c:
                continue
            f = aug[r][c] / aug[c][c]
            for cc in range(c, k + 1):
                aug[r][cc] -= f * aug[c][cc]
    beta = [aug[i][k] / aug[i][i] for i in range(k)]

    fitted = [sum(beta[i] * cols[i][r] for i in range(k)) for r in range(n)]
    ybar = sum(y) / n
    sst = sum((v - ybar) ** 2 for v in y)
    sse = sum((y[r] - fitted[r]) ** 2 for r in range(n))
    r2 = 1.0 - sse / sst if sst > 0 else None
    resid_var = sse / (n - k)
    return beta, r2, resid_var, n


def available_series(db_path=DEFAULT_DB_PATH):
    """The (feed, forecast_method) pairs actually present in the database.

    Each combination is a distinct measurement process. Live INDICATIVE
    snapshots are taken mid-session with vendor IV; purchased EOD history is
    taken at the 17:15 report with IV we inverted ourselves. Regressing them
    together would pool two samples and call the result one number, so the
    caller analyses each combination separately rather than guessing.
    """
    init_iv_tables(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT s.feed AS feed, f.forecast_method AS method,
                   COUNT(*) AS n
            FROM iv_snapshots s
            JOIN vol_forecasts f
              ON f.snapshot_date = s.snapshot_date AND f.symbol = s.underlying
            GROUP BY s.feed, f.forecast_method
            ORDER BY n DESC
            """
        ).fetchall()
    return [(r["feed"], r["method"], int(r["n"] or 0)) for r in rows]


def load_pairs(horizon: int, db_path=DEFAULT_DB_PATH, *, feed=None, forecast_method=None):
    """(implied, forecast, realized) triples, one per symbol-date with forward data.

    ``feed`` and ``forecast_method`` are not optional refinements. Without them
    the join fans out across every forecast method stored for a date, and the
    GROUP BY then collapses that fan-out by picking an arbitrary row -- so once
    a backfilled series exists alongside the live one, an unfiltered call
    silently mixes them and the mixture is not reproducible between runs.
    """
    init_iv_tables(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.snapshot_date   AS d,
                   s.underlying      AS sym,
                   AVG(s.implied_volatility) AS iv,
                   f.forecast_vol    AS fc
            FROM iv_snapshots s
            JOIN vol_forecasts f
              ON f.snapshot_date = s.snapshot_date AND f.symbol = s.underlying
            WHERE s.implied_volatility IS NOT NULL
              AND f.forecast_vol IS NOT NULL
              AND s.dte BETWEEN ? AND ?
              AND ABS(s.moneyness) <= 0.05
              AND (? IS NULL OR s.feed = ?)
              AND (? IS NULL OR f.forecast_method = ?)
            GROUP BY s.snapshot_date, s.underlying
            ORDER BY s.snapshot_date
            """,
            (max(1, horizon - 7), horizon + 14, feed, feed,
             forecast_method, forecast_method),
        ).fetchall()

    if not rows:
        return []

    # Realized vol over the forward window, computed from the price cache rather
    # than stored, because it is a function of data we already keep and storing
    # it would create a second source of truth to keep in sync.
    from logic.price_cache import get_history
    from logic.volatility_sizing import realized_vol
    from datetime import date, timedelta

    out = []
    cache = {}
    for r in rows:
        sym, d = r["sym"], r["d"]
        try:
            start = date.fromisoformat(d)
        except ValueError:
            continue
        if sym not in cache:
            try:
                cache[sym] = get_history(sym, start=start - timedelta(days=5),
                                         end=date.today())
            except Exception:
                cache[sym] = None
        hist = cache[sym]
        if hist is None or len(hist) == 0:
            continue
        try:
            fwd = hist.loc[str(start):str(start + timedelta(days=int(horizon * 1.6)))]
        except Exception:
            continue
        closes = [float(c) for c in fwd["Close"]]
        if len(closes) < max(5, horizon // 2):
            continue  # forward window has not elapsed yet
        rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))
                if closes[i - 1] > 0]
        rv = realized_vol(rets)
        if rv is None or rv <= 0:
            continue
        out.append((float(r["iv"]), float(r["fc"]), float(rv)))
    return out


def report_horizon(horizon: int, db_path=DEFAULT_DB_PATH, *,
                   feed=None, forecast_method=None) -> None:
    triples = load_pairs(horizon, db_path=db_path,
                         feed=feed, forecast_method=forecast_method)
    raw_n = len(triples)
    eff_n = raw_n / horizon / EFFECTIVE_SYMBOL_CLUSTERS if raw_n else 0.0

    print(f"\n--- horizon {horizon}d ---")
    print(f"raw observations      : {raw_n}")
    print(f"effective observations: {eff_n:.1f}  "
          f"(raw / {horizon}d overlap / {EFFECTIVE_SYMBOL_CLUSTERS:.0f} symbol clusters)")

    if eff_n < MIN_EFFECTIVE_N_PRELIMINARY:
        need = MIN_EFFECTIVE_N_PRELIMINARY * horizon * EFFECTIVE_SYMBOL_CLUSTERS
        print(f"VERDICT: INSUFFICIENT SAMPLE — need ~{need:.0f} raw rows for even a "
              f"preliminary read.")
        print("         No number is reported, deliberately. See module docstring.")
        return

    iv = [t[0] for t in triples]
    fc = [t[1] for t in triples]
    rv = [t[2] for t in triples]

    _, r2_restricted, var_r, _ = _ols_with_r2(rv, [iv])
    beta, r2_full, var_f, n = _ols_with_r2(rv, [iv, fc])
    if beta is None:
        print("VERDICT: regression failed (collinear or degenerate).")
        return

    print(f"R^2  implied vol only        : {r2_restricted:.4f}")
    print(f"R^2  implied vol + forecast  : {r2_full:.4f}")
    print(f"incremental R^2              : {r2_full - r2_restricted:+.4f}")
    print(f"forecast coefficient         : {beta[2]:+.4f}")

    # F test on the single added regressor.
    if var_f and var_f > 0 and var_r is not None:
        k_full = 3
        f_stat = (var_r * (n - k_full + 1) - var_f * (n - k_full)) / var_f
        print(f"F statistic (1, {n - k_full})        : {f_stat:.3f}")
        passes = f_stat > 4.0  # ~5% critical value for large n, 1 numerator df
    else:
        passes = False

    status = "PRELIMINARY" if eff_n < MIN_EFFECTIVE_N_CONCLUSIVE else "CONCLUSIVE"
    print(f"VERDICT ({status}): {'PASS' if passes else 'FAIL'} — the forecast "
          f"{'adds' if passes else 'does not add'} information over implied vol.")
    if status == "PRELIMINARY":
        print(f"         Direction only. Do not act on this until effective N "
              f">= {MIN_EFFECTIVE_N_CONCLUSIVE}.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    cov = coverage(args.db)
    print("\n=== Gate 0: does our vol forecast beat implied vol? ===")
    print(f"snapshot days accumulated : {cov['iv_days']}")
    print(f"date range                : {cov['first_date']} .. {cov['last_date']}")
    print(f"IV rows / forecast rows   : {cov['iv_rows']} / {cov['forecast_rows']}")
    print(f"symbols covered           : {cov['iv_symbols']}")

    if cov["iv_days"] == 0:
        print("\nNo snapshots recorded yet. Run logic/iv_snapshot.snapshot_universe "
              "(wired into the daily cycle) and come back.")
        return 1

    series = available_series(args.db)
    if not series:
        print("\nNo joinable snapshot/forecast pairs yet.")
        return 1

    # One report per measurement process. Pooling an INDICATIVE mid-session
    # series with a purchased 17:15 EOD series would answer a question about
    # neither of them.
    print(f"\nmeasurement processes present: {len(series)}")
    for feed, method, n in series:
        print(f"  feed={feed or '?'} method={method or '?'}  ({n} joined rows)")

    bar = "=" * 62
    for feed, method, _n in series:
        print(f"\n{bar}\nfeed={feed}  forecast_method={method}\n{bar}")
        for h in ([args.horizon] if args.horizon else HORIZONS):
            report_horizon(h, db_path=args.db, feed=feed, forecast_method=method)

    print("\nNote: the 7d horizon resolves soonest but measures the noisiest "
          "realized-vol\nestimate. Treat it as an early warning, not the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
