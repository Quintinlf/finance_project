"""Step through one daily candle and the PredictionRecord written at that close.

The realized return is shown only once the file has been filled from the next
close. Stepping does not place orders and does not refit anything.

Examples:
    python scripts/replay_predictions.py --symbol WEAT
    python scripts/replay_predictions.py --symbol WEAT --index 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from logic.prediction_log import PREDICTIONS_PATH, load_records


def _matching(records, symbol):
    rows = records
    if symbol:
        rows = [row for row in records if str(row.get("symbol")) == symbol.upper()]
    return rows


def _print_step(row: dict, candle: dict) -> None:
    print(f"{row.get('symbol')}  cutoff {row.get('information_cutoff')}")
    if candle:
        print(
            "  candle  O {open}  H {high}  L {low}  C {close}".format(
                open=_num(candle.get("Open")),
                high=_num(candle.get("High")),
                low=_num(candle.get("Low")),
                close=_num(candle.get("Close")),
            )
        )
    else:
        print("  candle  (no bar stored for this cutoff)")
    print(f"  bayesian { _num(row.get('bayesian_prediction')) }   gp { _num(row.get('gp_prediction')) }")
    print(f"  expected { _num(row.get('expected_return')) }   cost { _num(row.get('estimated_cost')) }%   net_ev { _num(row.get('net_ev')) }")
    print(f"  action {row.get('action')}   quantity {row.get('quantity')}")
    for name in (
        "bb_features",
        "rsi_features",
        "candle_features",
        "volume_features",
        "news_features",
        "regime_features",
        "lunar_features",
    ):
        block = row.get(name) or {}
        if block:
            print(f"  {name}: {block}")
    realized = row.get("realized_return")
    if realized is None:
        print("  realized return: not yet known")
    else:
        print(f"  realized return: {_num(realized)}   net {_num(row.get('realized_net'))}")


def _num(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _candle_for(row: dict) -> dict:
    symbol = str(row.get("symbol") or "")
    cutoff = str(row.get("information_cutoff") or "")[:10]
    if not symbol or not cutoff:
        return {}
    try:
        from logic.price_cache import get_history

        frame = get_history(symbol, allow_fetch=False)
    except Exception:
        return {}
    if frame is None or frame.empty:
        return {}
    stamps = pd.to_datetime(frame.index)
    if getattr(stamps, "tz", None) is not None:
        stamps = stamps.tz_convert("UTC").tz_localize(None)
    wanted = pd.Timestamp(cutoff)
    matches = frame.loc[stamps.normalize() == wanted.normalize()]
    if matches.empty:
        return {}
    bar = matches.iloc[-1]
    return {col: bar[col] for col in ("Open", "High", "Low", "Close") if col in matches.columns}


def main():
    parser = argparse.ArgumentParser(description="Replay one prediction candle at a time")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--index", type=int, default=0, help="Which matching record to show, from the start")
    parser.add_argument("--path", default=str(PREDICTIONS_PATH))
    args = parser.parse_args()

    records = _matching(load_records(Path(args.path)), args.symbol)
    if not records:
        raise SystemExit("No prediction records to replay yet.")
    index = args.index
    if index < 0 or index >= len(records):
        raise SystemExit(f"index {index} is outside 0..{len(records) - 1}")
    row = records[index]
    print(f"step {index + 1} of {len(records)}")
    _print_step(row, _candle_for(row))
    if index + 1 < len(records):
        print(f"\nnext: python scripts/replay_predictions.py --symbol {row.get('symbol')} --index {index + 1}")


if __name__ == "__main__":
    main()
