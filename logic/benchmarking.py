"""Benchmark bundle utilities for reproducible A/B experiments.

This module freezes symbols, historical windows, seeds, and config snapshots so
simulation changes can be compared on a stable dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast
import json


@dataclass
class BenchmarkWindow:
    """Named historical slice for benchmark replay."""

    label: str
    start_date: str
    end_date: str


@dataclass
class BenchmarkBundle:
    """Frozen bundle used for milestone comparisons."""

    name: str
    symbols: List[str]
    windows: List[BenchmarkWindow]
    seeds: List[int] = field(default_factory=lambda: [42])
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["windows"] = [asdict(w) for w in self.windows]
        return data


def default_windows() -> List[BenchmarkWindow]:
    """Return broad bull/bear/sideways windows for baseline experiments."""
    return [
        BenchmarkWindow(label="bull_2020_2021", start_date="2020-04-01", end_date="2021-12-31"),
        BenchmarkWindow(label="bear_2022", start_date="2022-01-01", end_date="2022-12-31"),
        BenchmarkWindow(label="sideways_2023", start_date="2023-01-01", end_date="2023-12-31"),
    ]


def build_default_bundle(
    *,
    name: str = "physics_upgrade_baseline",
    symbols: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    config_snapshot: Optional[Mapping[str, Any]] = None,
) -> BenchmarkBundle:
    """Create the default frozen benchmark bundle for milestone regression."""
    bundle_symbols = list(symbols) if symbols else ["SPY", "QQQ", "AAPL", "BTC-USD"]
    bundle_seeds = [int(x) for x in seeds] if seeds else [42, 1337, 2026]
    return BenchmarkBundle(
        name=name,
        symbols=bundle_symbols,
        windows=default_windows(),
        seeds=bundle_seeds,
        config_snapshot=dict(config_snapshot or {}),
    )


def freeze_benchmark_bundle(
    *,
    output_path: Union[str, Path],
    name: str = "physics_upgrade_baseline",
    symbols: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    config_snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist the benchmark bundle to disk as immutable experiment input."""
    bundle = build_default_bundle(
        name=name,
        symbols=symbols,
        seeds=seeds,
        config_snapshot=config_snapshot,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = bundle.to_dict()
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def freeze_bundle_from_execution_config(
    *,
    output_path: Union[str, Path],
    config: Any,
    name: str = "physics_upgrade_baseline",
    symbols: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Freeze benchmark using an ExecutionConfig-like object.

    The config object may expose a `to_dict()` method (preferred) or `__dict__`.
    """
    snapshot: Dict[str, Any] = {}
    if hasattr(config, "to_dict") and callable(config.to_dict):
        raw = cast(Any, config).to_dict()
        if isinstance(raw, Mapping):
            snapshot = {str(k): v for k, v in raw.items()}
    elif hasattr(config, "__dict__"):
        raw = cast(Any, config).__dict__
        if isinstance(raw, Mapping):
            snapshot = {str(k): v for k, v in raw.items()}

    return freeze_benchmark_bundle(
        output_path=output_path,
        name=name,
        symbols=symbols,
        seeds=seeds,
        config_snapshot=snapshot,
    )


def load_benchmark_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a previously frozen benchmark bundle JSON file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
