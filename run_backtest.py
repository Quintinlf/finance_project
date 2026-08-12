"""
Backtest entrypoint.

    python run_backtest.py --scope agriculture --start 2025-01-01 --end 2026-08-01
    python run_backtest.py --symbols WEAT,CORN --fractional --equity 500

Results are written to trade_logs/backtests/<timestamp>.json, which carries the
full per-bar decision record — that file is what the replay UI will read.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from logic.backtest import BacktestConfig, format_report, run_backtest
from logic.universe import get_symbols

OUTPUT_DIR = Path("trade_logs") / "backtests"


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Replay the strategy over history.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scope", default="agriculture",
                        help="universe scope: equities|commodities|agriculture|energy|metals|all")
    source.add_argument("--symbols", help="explicit comma-separated tickers")

    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--equity", type=float, default=500.0)
    parser.add_argument("--fractional", action="store_true",
                        help="allow notional/fractional sizing instead of whole shares")
    parser.add_argument("--tp", type=float, default=0.04, help="take profit fraction")
    parser.add_argument("--sl", type=float, default=0.02, help="stop loss fraction")
    parser.add_argument("--min-conf", type=float, default=0.50)
    parser.add_argument("--min-prob", type=float, default=0.50)
    parser.add_argument("--max-exposure", type=float, default=0.30)
    parser.add_argument("--position-cap", type=float, default=0.20)
    parser.add_argument("--no-costs", action="store_true",
                        help="run frictionless, to isolate how much cost is eating")
    parser.add_argument("--out", type=Path, default=None)

    eos = parser.add_argument_group(
        "eos finance domains",
        "physics_finance / quantum_finance / finance_ml algorithms via logic.eos_bridge",
    )
    eos.add_argument("--eos-mode", choices=["off", "shadow", "enforce"], default="off",
                     help="off: skip entirely. shadow: compute and record, change nothing. "
                          "enforce: let the --eos-* switches below affect trades.")
    eos.add_argument("--eos-garch-exits", action="store_true",
                     help="scale tp/sl by the GARCH volatility ratio (needs --eos-mode enforce)")
    eos.add_argument("--eos-hurst-confidence", action="store_true",
                     help="adjust confidence by Hurst regime (needs --eos-mode enforce)")
    eos.add_argument("--eos-stride", type=int, default=5,
                     help="refit the eos estimators every Nth bar per symbol (default 5)")

    args = parser.parse_args()

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else get_symbols(args.scope)
    )

    cfg = BacktestConfig(
        symbols=symbols,
        start=args.start,
        end=args.end,
        initial_equity=args.equity,
        min_confidence=args.min_conf,
        min_prob_up=args.min_prob,
        tp_pct=args.tp,
        sl_pct=args.sl,
        max_position_fraction=args.position_cap,
        max_portfolio_exposure=args.max_exposure,
        fractional=args.fractional,
        slippage_bps=0.0 if args.no_costs else 5.0,
        use_measured_spreads=not args.no_costs,
        eos_mode=args.eos_mode,
        eos_use_garch_exits=args.eos_garch_exits,
        eos_use_hurst_confidence=args.eos_hurst_confidence,
        eos_enrichment_stride=max(1, args.eos_stride),
    )

    if args.eos_mode != "off":
        from logic import eos_bridge

        snapshot = eos_bridge.status()
        if not snapshot["available"]:
            logging.error(
                "--eos-mode %s requested but the eos domains are not importable: %s",
                args.eos_mode, snapshot["import_error"],
            )
            return 1
        logging.info(
            "EOS: %s algorithms from %s | mode=%s stride=%s garch_exits=%s hurst_conf=%s",
            snapshot["algorithm_count"], snapshot["eos_root"], args.eos_mode,
            args.eos_stride, args.eos_garch_exits, args.eos_hurst_confidence,
        )
        if args.eos_mode == "shadow" and (args.eos_garch_exits or args.eos_hurst_confidence):
            logging.warning(
                "--eos-garch-exits/--eos-hurst-confidence have no effect in shadow mode; "
                "pass --eos-mode enforce to activate them."
            )

    if args.no_costs:
        import logic.backtest as bt
        from logic.costs import zero_cost_model

        bt.for_symbol = lambda *a, **k: zero_cost_model()
        logging.warning("Running WITHOUT costs — results are not achievable in practice.")

    logging.info("Loading history for %s symbols...", len(symbols))
    result = run_backtest(cfg)

    print()
    print(format_report(result))

    out = args.out or OUTPUT_DIR / f"backtest_{datetime.now():%Y%m%d_%H%M%S}.json"
    result.save(out)
    print(f"\nFull per-bar record: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception:
        logging.exception("Backtest failed")
        sys.exit(1)
