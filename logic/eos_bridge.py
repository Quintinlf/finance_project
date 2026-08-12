"""Bridge to the EOS server's finance domains.

The `eos` repo (sibling checkout, package name ``eos``) exposes 27 finance
algorithms across three domains: ``physics_finance``, ``quantum_finance``, and
``finance_ml``. Those are published to agents over MCP, but MCP is an
agent-to-tool protocol — ``run_daily_bot.py`` is a headless process with no MCP
client, so the MCP surface is unreachable at runtime. This module imports the
same algorithm functions directly as Python instead, which is the only way the
trading pipeline can actually use them.

Two layers live here:

1. `ALGORITHMS` / `call` — the raw catalogue. Every algorithm the eos finance
   domains expose, keyed by its MCP tool name (e.g. ``finance.ar1_garch11_fit``)
   so a name works identically here and in an MCP session. Results come back as
   plain dicts, not pydantic models from a foreign repo.

2. The trading-facing helpers (`garch_volatility`, `hurst_regime`, `hmm_regime`,
   `historical_var`, `quantum_price_levels_for`). These take an OHLCV frame,
   handle the calibration the raw algorithms leave to the caller, and return
   local dataclasses. This is what `signal_engine` and `execution_engine` call.

Everything degrades quietly. If the eos checkout is missing, or an algorithm
fails to converge on a short series, helpers return ``None`` and the pipeline
runs exactly as it did before. A missing sibling repo must never take down a
trading run.

Point `EOS_ROOT` at the checkout to override the default sibling-directory
lookup.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Locating and importing the eos checkout
# ---------------------------------------------------------------------------

_DEFAULT_EOS_ROOT = Path(__file__).resolve().parent.parent.parent / "mcp_server"


def _resolve_eos_root() -> Path:
    override = os.environ.get("EOS_ROOT", "").strip()
    return Path(override).expanduser().resolve() if override else _DEFAULT_EOS_ROOT


EOS_ROOT = _resolve_eos_root()

EOS_AVAILABLE = False
EOS_IMPORT_ERROR: Optional[str] = None

_finance_ml: Any = None
_physics_finance: Any = None
_quantum_finance: Any = None

try:
    if not EOS_ROOT.exists():
        raise ImportError(
            f"eos checkout not found at {EOS_ROOT}. Set EOS_ROOT to the "
            f"mcp_server directory to enable the finance domains."
        )
    if str(EOS_ROOT) not in sys.path:
        sys.path.insert(0, str(EOS_ROOT))

    from domains.finance_ml import algorithms as _finance_ml  # type: ignore[no-redef]
    from domains.physics_finance import algorithms as _physics_finance  # type: ignore[no-redef]
    from domains.quantum_finance import algorithms as _quantum_finance  # type: ignore[no-redef]

    EOS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on local checkout
    EOS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    logger.info(
        "eos finance domains unavailable (%s). Trading pipeline continues with "
        "built-in logic only.",
        EOS_IMPORT_ERROR,
    )


class EosUnavailable(RuntimeError):
    """Raised when an eos algorithm is called but the checkout is missing."""


def available() -> bool:
    """True when the eos finance algorithms can be called."""
    return EOS_AVAILABLE


def status() -> Dict[str, Any]:
    """Diagnostic snapshot — safe to log at startup."""
    return {
        "available": EOS_AVAILABLE,
        "eos_root": str(EOS_ROOT),
        "import_error": EOS_IMPORT_ERROR,
        "algorithm_count": len(ALGORITHMS) if EOS_AVAILABLE else 0,
    }


# ---------------------------------------------------------------------------
# Layer 1: the raw algorithm catalogue
# ---------------------------------------------------------------------------

def _build_catalogue() -> Dict[str, Callable[..., Any]]:
    """Map MCP tool names to the underlying callables.

    Names match the eos MCP tool names exactly, so a call written against an
    MCP session translates to `call(...)` here without renaming anything.
    """
    if not EOS_AVAILABLE:
        return {}
    return {
        # finance_ml — statistical / ML baselines
        "finance.ridge_regression": _finance_ml.ridge_regression,
        "finance.permutation_feature_importance": _finance_ml.permutation_feature_importance,
        "finance.bayesian_linear_regression": _finance_ml.bayesian_linear_regression,
        "finance.pca_factor_model": _finance_ml.pca_factor_model,
        "finance.ar1_garch11_fit": _finance_ml.ar1_garch11_fit,
        "finance.gaussian_hmm_regime": _finance_ml.gaussian_hmm_regime,
        "finance.q_learning_trading": _finance_ml.q_learning_trading,
        # physics_finance — portfolio theory, derivatives, risk, control
        "physics_finance.markowitz_portfolio": _physics_finance.markowitz_portfolio,
        "physics_finance.capm_expected_return": _physics_finance.capm_expected_return,
        "physics_finance.simulate_gbm_paths": _physics_finance.simulate_gbm_paths,
        "physics_finance.black_scholes_price": _physics_finance.black_scholes_price,
        "physics_finance.black_scholes_greeks": _physics_finance.black_scholes_greeks,
        "physics_finance.value_at_risk": _physics_finance.value_at_risk,
        "physics_finance.regression_hypothesis_test": _physics_finance.regression_hypothesis_test,
        "physics_finance.arma_forecast": _physics_finance.arma_forecast,
        "physics_finance.hill_tail_estimator": _physics_finance.hill_tail_estimator,
        "physics_finance.path_integral_option_price": _physics_finance.path_integral_option_price,
        "physics_finance.lqr_control": _physics_finance.lqr_control,
        "physics_finance.shannon_entropy": _physics_finance.shannon_entropy,
        "physics_finance.channel_capacity": _physics_finance.channel_capacity,
        # quantum_finance — QFSE, chaos, fractals, fuzzy logic
        "quantum_finance.qfse_energy_levels": _quantum_finance.qfse_energy_levels,
        "quantum_finance.quantum_price_levels": _quantum_finance.quantum_price_levels,
        "quantum_finance.lyapunov_exponent": _quantum_finance.lyapunov_exponent,
        "quantum_finance.hurst_exponent": _quantum_finance.hurst_exponent,
        "quantum_finance.fuzzy_trading_signal": _quantum_finance.fuzzy_trading_signal,
        "quantum_finance.genetic_algorithm_ma_crossover": _quantum_finance.genetic_algorithm_ma_crossover,
        "quantum_finance.lee_oscillator_simulate": _quantum_finance.lee_oscillator_simulate,
    }


ALGORITHMS: Dict[str, Callable[..., Any]] = _build_catalogue()


def list_algorithms() -> List[str]:
    """Names of every eos finance algorithm reachable from this project."""
    return sorted(ALGORITHMS)


def call(name: str, **kwargs: Any) -> Dict[str, Any]:
    """Invoke an eos algorithm by MCP tool name and return a plain dict.

    Raises EosUnavailable if the checkout is missing and KeyError for an
    unknown name. Algorithm-level errors (bad input, non-convergence) surface
    as whatever the algorithm raises — callers that need silence should use the
    trading-facing helpers below, which swallow those.
    """
    if not EOS_AVAILABLE:
        raise EosUnavailable(
            f"Cannot call '{name}': eos finance domains are not importable "
            f"({EOS_IMPORT_ERROR})."
        )
    try:
        handler = ALGORITHMS[name]
    except KeyError:
        raise KeyError(
            f"Unknown eos algorithm '{name}'. Available: {', '.join(list_algorithms())}"
        ) from None
    result = handler(**kwargs)
    # Every eos algorithm returns a pydantic model; unwrap so nothing
    # downstream depends on a foreign repo's model classes.
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return {"result": result}


# ---------------------------------------------------------------------------
# Layer 2: trading-facing helpers
# ---------------------------------------------------------------------------

@dataclass
class GarchVolatility:
    """One-step-ahead conditional volatility from an AR(1)-GARCH(1,1) fit."""
    forecast_next_vol: float
    unconditional_vol: float
    vol_ratio: float          # forecast / unconditional; >1 means vol is elevated
    alpha: float
    beta: float
    persistence: float        # alpha + beta; ->1 means shocks decay slowly
    converged: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HurstRegime:
    """Rescaled-range Hurst exponent with a trading-relevant label."""
    hurst_exponent: float
    label: str                # 'trending' | 'mean_reverting' | 'random_walk'
    strength: float           # |H - 0.5| * 2, clipped to [0, 1]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HmmRegime:
    """Current state of a 2-state Gaussian HMM fitted on returns."""
    current_state: int
    n_states: int
    state_mean: float
    state_std: float
    label: str                # 'risk_on' | 'risk_off'
    persistence: float        # P(stay in current state) from the transition matrix
    converged: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TailRisk:
    """Historical VaR and expected shortfall on the return distribution."""
    confidence: float
    value_at_risk: float          # positive number = loss fraction
    expected_shortfall: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PriceLevels:
    """Quantum price levels (QPLs) bracketing spot, nearest level first."""
    spot_price: float
    support: List[float]
    resistance: List[float]
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_returns(price_history: Any, column: str = "Close") -> List[float]:
    """Simple returns from an OHLCV frame, NaN/inf dropped.

    Accepts a pandas DataFrame (the shape signal_engine already has), a Series,
    or a bare sequence of prices.
    """
    if price_history is None:
        return []
    try:
        if hasattr(price_history, "columns"):
            series = price_history[column]
        elif hasattr(price_history, "values") and not isinstance(price_history, (list, tuple)):
            series = price_history
        else:
            series = None

        if series is not None:
            prices = [float(v) for v in series.values]
        else:
            prices = [float(v) for v in price_history]
    except Exception as exc:
        logger.debug("Could not extract prices for eos helpers: %s", exc)
        return []

    returns: List[float] = []
    for previous, current in zip(prices, prices[1:]):
        if previous <= 0 or not math.isfinite(previous) or not math.isfinite(current):
            continue
        value = (current / previous) - 1.0
        if math.isfinite(value):
            returns.append(value)
    return returns


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(variance, 0.0))


def garch_volatility(
    price_history: Any = None,
    returns: Optional[Sequence[float]] = None,
    min_observations: int = 60,
) -> Optional[GarchVolatility]:
    """Fit AR(1)-GARCH(1,1) and report the next-bar volatility forecast.

    This is the piece worth having: a fixed 2% stop is wrong in both directions
    — too tight in a high-vol tape, needlessly wide in a quiet one. `vol_ratio`
    is the scaling factor for turning a static stop into a volatility-aware one.

    Returns None when the eos checkout is missing, the series is too short for
    a stable MLE fit, or the optimizer does not converge.
    """
    if not EOS_AVAILABLE:
        return None
    series = list(returns) if returns is not None else extract_returns(price_history)
    if len(series) < min_observations:
        logger.debug("GARCH skipped: %s returns < %s required", len(series), min_observations)
        return None
    try:
        fit = _finance_ml.ar1_garch11_fit(series)
    except Exception as exc:
        logger.debug("GARCH fit failed: %s", exc)
        return None

    unconditional = _std(series)
    forecast = float(fit.forecast_next_vol)
    if not math.isfinite(forecast) or forecast <= 0 or unconditional <= 0:
        return None

    return GarchVolatility(
        forecast_next_vol=forecast,
        unconditional_vol=unconditional,
        vol_ratio=forecast / unconditional,
        alpha=float(fit.alpha),
        beta=float(fit.beta),
        persistence=float(fit.alpha) + float(fit.beta),
        converged=bool(fit.converged),
    )


def hurst_regime(
    price_history: Any = None,
    returns: Optional[Sequence[float]] = None,
    min_observations: int = 64,
    trending_threshold: float = 0.55,
    mean_reverting_threshold: float = 0.45,
) -> Optional[HurstRegime]:
    """Classify the series as trending, mean-reverting, or a random walk.

    Fed the *returns* series, not cumulated prices — the eos tool documents
    that requirement and an already-cumulated series biases H upward toward 1.

    H > 0.55 is persistent (trend-following logic is appropriate); H < 0.45 is
    anti-persistent (mean-reversion logic, i.e. the Bollinger leg, is
    appropriate); in between there is no exploitable memory.
    """
    if not EOS_AVAILABLE:
        return None
    series = list(returns) if returns is not None else extract_returns(price_history)
    if len(series) < min_observations:
        logger.debug("Hurst skipped: %s returns < %s required", len(series), min_observations)
        return None
    try:
        result = _quantum_finance.hurst_exponent(series)
    except Exception as exc:
        logger.debug("Hurst estimation failed: %s", exc)
        return None

    h = float(result.hurst_exponent)
    if not math.isfinite(h):
        return None

    if h > trending_threshold:
        label = "trending"
    elif h < mean_reverting_threshold:
        label = "mean_reverting"
    else:
        label = "random_walk"

    return HurstRegime(
        hurst_exponent=h,
        label=label,
        strength=min(1.0, abs(h - 0.5) * 2.0),
    )


def hmm_regime(
    price_history: Any = None,
    returns: Optional[Sequence[float]] = None,
    n_states: int = 2,
    min_observations: int = 60,
    seed: int = 0,
    n_iter: int = 30,
) -> Optional[HmmRegime]:
    """Fit a Gaussian HMM on returns and decode the regime of the latest bar.

    Complements the project's own `compute_market_state`, which assigns regime
    probabilities from hand-tuned rules. This one estimates the states from the
    data via Baum-Welch and decodes with Viterbi, so the two can be compared
    against each other in the backtest rather than assumed equivalent.

    `n_iter` defaults well below the algorithm's own 100 because this is the
    single most expensive helper. On a converging series Baum-Welch hits the
    tolerance around iteration 20; on a degenerate one (no real two-regime
    structure) it never converges and the extra iterations move the
    log-likelihood in the fourth decimal while tripling the cost. 30 buys
    convergence where it exists and stops wasting time where it does not.
    """
    if not EOS_AVAILABLE:
        return None
    series = list(returns) if returns is not None else extract_returns(price_history)
    if len(series) < min_observations:
        logger.debug("HMM skipped: %s returns < %s required", len(series), min_observations)
        return None
    try:
        result = _finance_ml.gaussian_hmm_regime(
            series, n_states=n_states, n_iter=n_iter, seed=seed
        )
    except Exception as exc:
        logger.debug("HMM fit failed: %s", exc)
        return None

    if not result.state_path:
        return None
    current = int(result.state_path[-1])
    try:
        mean = float(result.means[current])
        std = float(result.stds[current])
        persistence = float(result.transition_matrix[current][current])
    except (IndexError, TypeError):
        return None

    # Label by the state's own drift rather than its index — Baum-Welch state
    # ordering is arbitrary and flips between fits.
    label = "risk_on" if mean >= 0 else "risk_off"

    return HmmRegime(
        current_state=current,
        n_states=int(result.n_states),
        state_mean=mean,
        state_std=std,
        label=label,
        persistence=persistence,
        converged=bool(result.converged),
    )


def historical_var(
    price_history: Any = None,
    returns: Optional[Sequence[float]] = None,
    confidence: float = 0.95,
    min_observations: int = 30,
) -> Optional[TailRisk]:
    """Historical VaR and expected shortfall for a single symbol's returns."""
    if not EOS_AVAILABLE:
        return None
    series = list(returns) if returns is not None else extract_returns(price_history)
    if len(series) < min_observations:
        return None
    try:
        result = _physics_finance.value_at_risk(
            series, confidence=confidence, method="historical", portfolio_value=1.0
        )
    except Exception as exc:
        logger.debug("VaR computation failed: %s", exc)
        return None

    var = float(result.value_at_risk)
    if not math.isfinite(var):
        return None
    shortfall = result.expected_shortfall
    return TailRisk(
        confidence=float(result.confidence),
        value_at_risk=var,
        expected_shortfall=(
            float(shortfall) if shortfall is not None and math.isfinite(float(shortfall)) else None
        ),
    )


def quantum_price_levels_for(
    spot_price: float,
    price_history: Any = None,
    returns: Optional[Sequence[float]] = None,
    n_levels: int = 3,
    barrier_sigmas: float = 3.0,
    min_observations: int = 30,
    n_grid: int = 150,
) -> Optional[PriceLevels]:
    """Quantum price levels (support/resistance) around spot.

    The raw QFSE tools take the potential coefficients (gamma, eta, delta,
    upsilon) as free parameters and leave calibration to the caller. Two
    choices matter here, and getting either wrong yields no bound states at all:

    1. *Units.* The QFSE is solved on a finite-difference grid with hbar = m = 1,
       so the kinetic coefficient scales as 1/dr^2. Feeding it raw returns
       (dr ~ 1e-4) makes kinetic energy swamp the potential and every eigenstate
       lands above the barrier. So the equation is solved in standardized units
       (r in sigmas, spot = 1.0) and the resulting turning radii are rescaled by
       sigma afterwards.

    2. *Domain.* V(r) = a*r^2 - b*r^4 turns negative beyond r = sqrt(a/b), and
       those negative-energy edge states sort below the real bound states and
       crowd them out of the lowest `n_levels`. The grid is therefore truncated
       at the barrier peak, where V is still non-negative.

    In standardized units, a = 1/2 (curvature matched to unit variance) and
    placing the barrier top at `barrier_sigmas` = k fixes b = 1/(4k^2), i.e.
    delta = 1 and upsilon = 1/k^2. The barrier peak then sits at exactly r = k.

    Treat the output as a structural level estimate, not a forecast: it is a
    restatement of the return distribution's width in price terms.

    `n_grid` drives an O(n^3) dense eigendecomposition. 150 is used rather than
    the algorithm's 400 default because the turning radii agree to within
    0.0003 sigma between the two while costing a tenth as much.
    """
    if not EOS_AVAILABLE or spot_price <= 0:
        return None
    series = list(returns) if returns is not None else extract_returns(price_history)
    if len(series) < min_observations:
        return None

    sigma = _std(series)
    if sigma <= 0 or not math.isfinite(sigma):
        return None

    k = max(float(barrier_sigmas), 1e-6)
    delta = 1.0             # a = delta/2 = 1/2 in standardized units
    upsilon = 1.0 / (k ** 2)  # b = upsilon/4 = 1/(4k^2) puts the barrier at r = k

    try:
        levels_input = _quantum_finance.qfse_energy_levels(
            gamma=1.0,
            eta=1.0,
            delta=delta,
            upsilon=upsilon,
            r_max=k,  # truncate at the barrier peak; V >= 0 across the domain
            n_grid=max(50, int(n_grid)),
            n_levels=max(1, n_levels),
        )
        # spot_price=1.0 keeps this in standardized space, so upper_price - 1
        # is the turning radius in sigmas. Rescaling below converts to price.
        result = _quantum_finance.quantum_price_levels(
            spot_price=1.0,
            energies=list(levels_input.energies),
            gamma=1.0,
            eta=1.0,
            delta=delta,
            upsilon=upsilon,
        )
    except Exception as exc:
        logger.debug("Quantum price levels failed: %s", exc)
        return None

    support: List[float] = []
    resistance: List[float] = []
    for level in result.levels:
        if not level.bound_state:
            continue
        turning_radius = float(level.upper_price) - 1.0  # in sigmas
        if not math.isfinite(turning_radius) or turning_radius <= 0:
            continue
        offset = turning_radius * sigma  # back to return units
        lower = spot_price * (1.0 - offset)
        upper = spot_price * (1.0 + offset)
        if math.isfinite(lower) and 0 < lower < spot_price:
            support.append(lower)
        if math.isfinite(upper) and upper > spot_price:
            resistance.append(upper)

    support.sort(reverse=True)   # nearest support first
    resistance.sort()            # nearest resistance first

    if not support and not resistance:
        return None

    return PriceLevels(
        spot_price=float(spot_price),
        support=support,
        resistance=resistance,
        nearest_support=support[0] if support else None,
        nearest_resistance=resistance[0] if resistance else None,
    )


# ---------------------------------------------------------------------------
# Enrichment used by signal_engine
# ---------------------------------------------------------------------------

_ENRICHMENT_CACHE: Dict[str, Dict[str, Any]] = {}
_ENRICHMENT_COUNTER: Dict[str, int] = {}


def reset_enrichment_cache() -> None:
    """Drop all cached enrichments. Call between backtest runs and in tests."""
    _ENRICHMENT_CACHE.clear()
    _ENRICHMENT_COUNTER.clear()


def build_enrichment(
    price_history: Any,
    spot_price: float,
    enable_garch: bool = True,
    enable_hurst: bool = True,
    enable_hmm: bool = True,
    enable_var: bool = True,
    enable_qpl: bool = True,
    cache_key: Optional[str] = None,
    stride: int = 1,
) -> Dict[str, Any]:
    """Run the enabled eos helpers once and package the results for Signal.meta.

    Returns a JSON-serializable dict. Keys are absent (not null) when a helper
    is disabled or could not produce a result, so a consumer checking `in` gets
    an honest answer about what was actually computed.

    Returns are extracted once and shared across all five helpers — each of
    these otherwise re-walks the same frame.

    A full refit costs roughly a quarter-second per symbol, which is nothing
    for one live run a day but hours across a multi-year backtest. Passing a
    `cache_key` (the symbol) with `stride` > 1 refits only every Nth call and
    reuses the previous result in between. That is safe with respect to
    look-ahead — a reused result is fitted on strictly *older* data — but the
    reused copy is tagged `stale: True` and carries `stale_age` so nothing
    downstream mistakes it for a fresh fit. Estimates over a 200+ observation
    window barely move bar to bar, so a stride of 5 costs very little accuracy.
    """
    if cache_key is not None and stride > 1:
        count = _ENRICHMENT_COUNTER.get(cache_key, 0)
        _ENRICHMENT_COUNTER[cache_key] = count + 1
        cached = _ENRICHMENT_CACHE.get(cache_key)
        if cached is not None and count % stride != 0:
            reused = dict(cached)
            reused["stale"] = True
            reused["stale_age"] = count % stride
            return reused

    enrichment: Dict[str, Any] = {"available": EOS_AVAILABLE}
    if not EOS_AVAILABLE:
        enrichment["import_error"] = EOS_IMPORT_ERROR
        return enrichment

    returns = extract_returns(price_history)
    enrichment["n_returns"] = len(returns)
    if not returns:
        return enrichment

    if enable_garch:
        garch = garch_volatility(returns=returns)
        if garch is not None:
            enrichment["garch"] = garch.as_dict()

    if enable_hurst:
        hurst = hurst_regime(returns=returns)
        if hurst is not None:
            enrichment["hurst"] = hurst.as_dict()

    if enable_hmm:
        hmm = hmm_regime(returns=returns)
        if hmm is not None:
            enrichment["hmm"] = hmm.as_dict()

    if enable_var:
        tail = historical_var(returns=returns)
        if tail is not None:
            enrichment["tail_risk"] = tail.as_dict()

    if enable_qpl and spot_price > 0:
        levels = quantum_price_levels_for(spot_price=spot_price, returns=returns)
        if levels is not None:
            enrichment["quantum_price_levels"] = levels.as_dict()

    enrichment["stale"] = False
    if cache_key is not None and stride > 1:
        _ENRICHMENT_CACHE[cache_key] = enrichment

    return enrichment


# ---------------------------------------------------------------------------
# Risk sizing derived from the enrichment
# ---------------------------------------------------------------------------

def scale_exit_pcts(
    tp_pct: float,
    sl_pct: float,
    enrichment: Optional[Dict[str, Any]],
    max_scale: float = 2.0,
    min_scale: float = 0.5,
) -> Dict[str, Any]:
    """Scale take-profit and stop-loss by the GARCH volatility ratio.

    A stop set at a fixed 2% is a different bet in a 1%-vol tape than in a
    4%-vol one. Scaling both legs by `vol_ratio` keeps the stop at a roughly
    constant number of standard deviations, which is what the risk model
    actually intends.

    The scale is clamped to [min_scale, max_scale] so one bad GARCH fit cannot
    produce an absurd stop. Returns the original percentages unchanged whenever
    no usable GARCH result is present.
    """
    result: Dict[str, Any] = {
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "scale": 1.0,
        "applied": False,
        "reason": "no garch result",
    }
    if not enrichment:
        return result
    garch = enrichment.get("garch")
    if not isinstance(garch, dict):
        return result
    if not garch.get("converged", False):
        result["reason"] = "garch did not converge"
        return result

    ratio = garch.get("vol_ratio")
    try:
        scale = float(ratio)
    except (TypeError, ValueError):
        return result
    if not math.isfinite(scale) or scale <= 0:
        result["reason"] = "non-finite vol_ratio"
        return result

    scale = max(min_scale, min(max_scale, scale))
    result.update(
        tp_pct=float(tp_pct) * scale,
        sl_pct=float(sl_pct) * scale,
        scale=scale,
        applied=True,
        reason=f"garch vol_ratio={ratio:.3f} clamped to {scale:.3f}",
    )
    return result


def hurst_confidence_multiplier(
    signal_type: str,
    bb_signal: str,
    enrichment: Optional[Dict[str, Any]],
    boost: float = 1.10,
    penalty: float = 0.90,
) -> Dict[str, Any]:
    """Reward or penalize a signal based on whether its logic matches the tape.

    The Bollinger leg is a mean-reversion bet: it says BUY when price is
    stretched *below* the band. That bet is well-founded in an anti-persistent
    series (H < 0.45) and works against the tape in a persistent one (H > 0.55).
    `signal_engine` currently applies a fixed +15%/-15% adjustment on
    forecast/Bollinger agreement without asking which of the two regimes holds.

    This returns a multiplier only — the caller decides whether to apply it.
    """
    result: Dict[str, Any] = {"multiplier": 1.0, "applied": False, "reason": "no hurst result"}
    if not enrichment:
        return result
    hurst = enrichment.get("hurst")
    if not isinstance(hurst, dict):
        return result

    label = str(hurst.get("label", ""))
    if label == "random_walk":
        result["reason"] = "hurst indicates random walk; no adjustment"
        return result

    normalized_bb = str(bb_signal).upper()
    normalized_signal = str(signal_type).upper()
    if normalized_bb == "NEUTRAL" or normalized_bb != normalized_signal:
        result["reason"] = "bollinger leg not driving this signal"
        return result

    # Bollinger agreed with the direction, so a mean-reversion bet is in play.
    if label == "mean_reverting":
        multiplier, why = boost, "mean-reverting tape supports the Bollinger leg"
    else:
        multiplier, why = penalty, "trending tape works against the Bollinger leg"

    result.update(
        multiplier=multiplier,
        applied=True,
        reason=f"H={hurst.get('hurst_exponent'):.3f} ({label}); {why}",
    )
    return result
