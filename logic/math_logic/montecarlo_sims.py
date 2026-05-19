# montecarlo_simulations.py

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

try:
    from scipy.stats import levy_stable
except Exception:  # pragma: no cover - fallback guarded in runtime checks
    levy_stable = None


# ===========================================================
# Monte Carlo + MCMC utilities (library-only, no printing)
# ===========================================================

def estimate_alpha_stable_params(
    observed_returns: np.ndarray,
    *,
    min_samples: int = 80,
) -> Dict[str, Any]:
    """Estimate alpha-stable parameters from observed returns.

    Returns a dict with keys: alpha, beta, loc, scale, success, fallback_mode,
    sample_size, and error (if any).
    """
    data = np.asarray(observed_returns, dtype=float)
    data = data[np.isfinite(data)]

    fallback = {
        "alpha": 2.0,
        "beta": 0.0,
        "loc": float(np.mean(data)) if data.size else 0.0,
        # alpha=2 corresponds to Normal(mu, variance=2*scale^2)
        "scale": float(max(np.std(data) / np.sqrt(2.0), 1e-8)) if data.size else 1e-4,
        "success": False,
        "fallback_mode": "gaussian",
        "sample_size": int(data.size),
        "error": None,
    }

    if data.size < min_samples:
        fallback["error"] = f"insufficient samples ({data.size} < {min_samples})"
        return fallback

    if levy_stable is None:
        fallback["error"] = "scipy.stats.levy_stable unavailable"
        return fallback

    try:
        alpha, beta, loc, scale = levy_stable.fit(data)
        alpha = float(np.clip(alpha, 1.10, 2.0))
        beta = float(np.clip(beta, -1.0, 1.0))
        loc = float(loc)
        scale = float(max(scale, 1e-8))
        return {
            "alpha": alpha,
            "beta": beta,
            "loc": loc,
            "scale": scale,
            "success": True,
            "fallback_mode": None,
            "sample_size": int(data.size),
            "error": None,
        }
    except Exception as exc:
        fallback["error"] = str(exc)
        return fallback


def estimate_rolling_alpha_stable_params(
    observed_returns: np.ndarray,
    *,
    window: int = 252,
    step: int = 21,
    min_samples: int = 80,
) -> List[Dict[str, Any]]:
    """Estimate alpha-stable parameters over rolling windows."""
    data = np.asarray(observed_returns, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < min_samples:
        return []

    window = max(int(window), min_samples)
    step = max(1, int(step))
    results: List[Dict[str, Any]] = []
    for end_idx in range(window, data.size + 1, step):
        start_idx = end_idx - window
        segment = data[start_idx:end_idx]
        params = estimate_alpha_stable_params(segment, min_samples=min_samples)
        params["start_idx"] = int(start_idx)
        params["end_idx"] = int(end_idx)
        results.append(params)
    return results


def _sample_shocks(
    *,
    rng: np.random.Generator,
    distribution: str,
    shape: Tuple[int, int],
    stable_params: Optional[Dict[str, float]] = None,
    max_abs_shock: float = 20.0,
) -> np.ndarray:
    """Sample normalized shocks for path simulation."""
    dist = str(distribution).lower()
    if dist == "normal":
        return rng.standard_normal(shape)

    if dist != "alpha_stable":
        raise ValueError("distribution must be 'normal' or 'alpha_stable'")

    if levy_stable is None:
        return rng.standard_normal(shape)

    params = stable_params or {}
    alpha = float(np.clip(params.get("alpha", 1.7), 1.10, 2.0))
    beta = float(np.clip(params.get("beta", 0.0), -1.0, 1.0))
    loc = float(params.get("loc", 0.0))
    scale = float(max(params.get("scale", 1.0), 1e-8))

    raw = levy_stable.rvs(alpha, beta, loc=loc, scale=scale, size=shape, random_state=rng)
    # Keep simulation numerically stable for exponentiation while preserving heavy tails.
    raw = np.clip(raw, -max_abs_shock, max_abs_shock)
    return raw


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    N_sim: int,
    seed: Optional[int] = None,
    *,
    distribution: str = "normal",
    stable_params: Optional[Dict[str, float]] = None,
    max_abs_shock: float = 20.0,
) -> np.ndarray:
    """
    Simulate GBM-like price paths using either normal or alpha-stable shocks.
    Returns array of shape (N_sim, N_steps).
    """
    rng = np.random.default_rng(seed)
    N_steps = int(T / dt)

    shocks = _sample_shocks(
        rng=rng,
        distribution=distribution,
        shape=(N_sim, N_steps),
        stable_params=stable_params,
        max_abs_shock=max_abs_shock,
    )

    if str(distribution).lower() == "alpha_stable":
        alpha = float(np.clip((stable_params or {}).get("alpha", 1.7), 1.10, 2.0))
        scale_term = sigma * (dt ** (1.0 / alpha))
    else:
        scale_term = sigma * np.sqrt(dt)

    daily_returns = (mu * dt) + (scale_term * shocks)
    price_paths = S0 * np.exp(np.cumsum(daily_returns, axis=1))
    return price_paths


def tail_event_frequency(
    standardized_returns: np.ndarray,
    *,
    sigma_levels: Tuple[int, int, int] = (3, 5, 10),
) -> Dict[str, float]:
    """Count and rate of standardized return exceedances for each sigma level."""
    data = np.asarray(standardized_returns, dtype=float)
    data = np.abs(data[np.isfinite(data)])
    total = int(data.size)
    metrics: Dict[str, float] = {"n_total": float(total)}
    if total == 0:
        for lvl in sigma_levels:
            metrics[f"count_{lvl}sigma"] = 0.0
            metrics[f"rate_{lvl}sigma"] = 0.0
        return metrics

    for lvl in sigma_levels:
        count = int(np.sum(data >= float(lvl)))
        metrics[f"count_{lvl}sigma"] = float(count)
        metrics[f"rate_{lvl}sigma"] = float(count / total)
    return metrics


def tail_event_frequency_from_paths(
    price_paths: np.ndarray,
    *,
    sigma_levels: Tuple[int, int, int] = (3, 5, 10),
) -> Dict[str, float]:
    """Compute tail exceedance diagnostics from simulated price paths."""
    paths = np.asarray(price_paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] < 2:
        return tail_event_frequency(np.array([]), sigma_levels=sigma_levels)

    log_returns = np.diff(np.log(np.maximum(paths, 1e-12)), axis=1).reshape(-1)
    sigma = float(np.std(log_returns))
    if sigma <= 0.0:
        return tail_event_frequency(np.array([]), sigma_levels=sigma_levels)

    standardized = log_returns / sigma
    metrics = tail_event_frequency(standardized, sigma_levels=sigma_levels)
    metrics["sigma_reference"] = sigma
    return metrics


def risk_metrics(final_prices: np.ndarray, alpha: float = 0.95) -> Dict[str, float]:
    """
    Compute expected final price, VaR, and CVaR at level alpha (e.g., 0.95).
    """
    if final_prices.size == 0:
        return {"expected_final": float("nan"), "VaR": float("nan"), "CVaR": float("nan")}
    expected_final = float(np.mean(final_prices))
    var_cut = float(np.percentile(final_prices, (1 - alpha) * 100))
    cvar = float(np.mean(final_prices[final_prices <= var_cut]))
    return {"expected_final": expected_final, "VaR": var_cut, "CVaR": cvar}


def mcmc_posterior_mu_sigma(
    observed_returns: np.ndarray,
    dt: float,
    *,
    nwalkers: int = 40,
    steps: int = 2000,
    burn: int = 500,
    thin: int = 10,
    init_center: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Fit posterior for (mu, sigma) using emcee. Returns dict with samples and chains.
    No plotting or printing; consumers can summarize as needed.
    """
    try:
        import emcee  # type: ignore
    except Exception as e:
        raise ImportError("emcee is required for MCMC. Install with `pip install emcee`.\n" + str(e))

    rng = np.random.default_rng(seed)
    data = np.asarray(observed_returns, dtype=float)

    def _log_like(theta, data, dt):
        mu_, sigma_ = theta
        if sigma_ <= 0:
            return -np.inf
        sd = sigma_ * np.sqrt(dt)
        z = (data - mu_ * dt) / sd
        return -0.5 * np.sum(z ** 2 + np.log(2 * np.pi * sd ** 2))

    def _log_prior(theta):
        mu_, sigma_ = theta
        if -1.0 < mu_ < 1.0 and 0.0 < sigma_ < 2.0:
            return 0.0
        return -np.inf

    def _log_post(theta, data, dt):
        lp = _log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + _log_like(theta, data, dt)

    ndim = 2
    if init_center is None:
        # simple moment estimates as a starting point
        m = float(np.mean(data) / dt)
        s = float(np.std(data) / np.sqrt(dt))
        init_center = (np.clip(m, -0.5, 0.5), np.clip(s, 1e-3, 1.0))

    initial_pos = np.array(init_center) + 0.05 * rng.standard_normal(size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_post, args=(data, dt))
    sampler.run_mcmc(initial_pos, steps, progress=False)

    samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    return {"samples": samples, "chain": sampler.get_chain()}


def posterior_predictive_final_prices(
    S0: float,
    T: float,
    dt: float,
    samples: np.ndarray,
    *,
    per_param_sims: int = 600,
    n_param_samples: int = 400,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Draw a subset of (mu, sigma) from posterior samples and simulate final prices.
    Returns 1D array of final prices across all simulations.
    """
    rng = np.random.default_rng(seed)
    n_param_samples = min(n_param_samples, samples.shape[0])
    sel_idx = rng.choice(samples.shape[0], size=n_param_samples, replace=False)
    sel_params = samples[sel_idx]

    finals = []
    for mu_post, sigma_post in sel_params:
        paths = simulate_gbm_paths(S0, float(mu_post), float(sigma_post), T, dt, per_param_sims,
                                   seed=int(rng.integers(0, 1_000_000)))
        finals.append(paths[:, -1])
    return np.concatenate(finals, axis=0) if finals else np.array([])


def compare_fixed_vs_posterior(
    baseline_final_prices: np.ndarray,
    predictive_final_prices: np.ndarray,
    *,
    alpha: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Compute risk metrics for fixed-parameter MC vs posterior-predictive.
    Returns dict {fixed: {...}, predictive: {...}}.
    """
    return {
        "fixed": risk_metrics(baseline_final_prices, alpha=alpha),
        "predictive": risk_metrics(predictive_final_prices, alpha=alpha),
    }


def compare_normal_vs_alpha_stable(
    *,
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    N_sim: int,
    stable_params: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    alpha: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Run paired simulations and return risk + tail diagnostics by distribution."""
    normal_paths = simulate_gbm_paths(
        S0,
        mu,
        sigma,
        T,
        dt,
        N_sim,
        seed=seed,
        distribution="normal",
    )
    stable_paths = simulate_gbm_paths(
        S0,
        mu,
        sigma,
        T,
        dt,
        N_sim,
        seed=seed,
        distribution="alpha_stable",
        stable_params=stable_params,
    )

    normal_finals = normal_paths[:, -1]
    stable_finals = stable_paths[:, -1]
    return {
        "normal": {
            **risk_metrics(normal_finals, alpha=alpha),
            **tail_event_frequency_from_paths(normal_paths),
        },
        "alpha_stable": {
            **risk_metrics(stable_finals, alpha=alpha),
            **tail_event_frequency_from_paths(stable_paths),
        },
    }


# ===============================
# RISK ANALYSIS MODULE
# ===============================

class RiskModel:
    def __init__(
        self,
        mu: float = 0.07,
        sigma: float = 0.2,
        T: float = 1,
        dt: float = 1 / 252,
        distribution: str = "normal",
        stable_params: Optional[Dict[str, float]] = None,
    ):
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.distribution = distribution
        self.stable_params = stable_params or None
        self.rng = np.random.default_rng(42)

    def simulate_gbm_paths(self, S0: float, N_sim: int = 10000):
        return simulate_gbm_paths(
            S0,
            self.mu,
            self.sigma,
            self.T,
            self.dt,
            N_sim,
            seed=int(self.rng.integers(0, 1_000_000)),
            distribution=self.distribution,
            stable_params=self.stable_params,
        )

    def risk_metrics(self, final_prices: np.ndarray, alpha: float = 0.95) -> Dict[str, float]:
        expected_final = float(np.mean(final_prices))
        var_cut = float(np.percentile(final_prices, (1 - alpha) * 100))
        cvar = float(np.mean(final_prices[final_prices <= var_cut]))
        return {"expected_final": expected_final, "VaR": var_cut, "CVaR": cvar}

    def estimate_parameters_mcmc(self, observed_returns):
        import emcee
        def log_likelihood(theta, data, dt):
            mu_, sigma_ = theta
            if sigma_ <= 0: return -np.inf
            sd = sigma_ * np.sqrt(dt)
            z = (data - mu_ * dt) / sd
            return -0.5 * np.sum(z**2 + np.log(2 * np.pi * sd**2))

        def log_prior(theta):
            mu_, sigma_ = theta
            return 0.0 if (-1.0 < mu_ < 1.0 and 0.0 < sigma_ < 2.0) else -np.inf

        def log_posterior(theta, data, dt):
            lp = log_prior(theta)
            return lp + log_likelihood(theta, data, dt) if np.isfinite(lp) else -np.inf

        ndim, nwalkers = 2, 40
        initial_center = np.array([self.mu, self.sigma])
        initial_pos = initial_center + 0.05 * self.rng.standard_normal(size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(observed_returns, self.dt))
        sampler.run_mcmc(initial_pos, 2000, progress=False)
        samples = sampler.get_chain(discard=500, thin=10, flat=True)
        self.mu, self.sigma = np.mean(samples[:, 0]), np.mean(samples[:, 1])
        return samples

    def combined_mc_mcmc(self, S0, observed_returns):
        mcmc_samples = self.estimate_parameters_mcmc(observed_returns)
        sel_idx = self.rng.choice(mcmc_samples.shape[0], size=300, replace=False)
        sel_params = mcmc_samples[sel_idx]
        predictive_finals = []
        for mu_post, sigma_post in sel_params:
            paths = simulate_gbm_paths(
                S0,
                float(mu_post),
                float(sigma_post),
                self.T,
                self.dt,
                10000,
                seed=int(self.rng.integers(0, 1_000_000)),
                distribution=self.distribution,
                stable_params=self.stable_params,
            )
            predictive_finals.append(paths[:, -1])
        predictive_finals = np.concatenate(predictive_finals)
        return self.risk_metrics(predictive_finals)


# ===========================================================
# TRADING STRATEGY MONTE CARLO SIMULATION
# ===========================================================

def monte_carlo_strategy_simulation(
    initial_capital=100,
    avg_trades_per_day=2,
    win_rate=0.55,
    avg_win_pct=4.0,
    avg_loss_pct=2.0,
    days=30,
    num_simulations=1000,
    seed=None,
    distribution: str = "normal",
    stable_params: Optional[Dict[str, float]] = None,
):
    """
    Simulate trading strategy P&L over multiple paths.
    
    Args:
        initial_capital: Starting capital ($)
        avg_trades_per_day: Average number of trades per day
        win_rate: Probability of winning trade (0.0 to 1.0)
        avg_win_pct: Average win size as percentage
        avg_loss_pct: Average loss size as percentage
        days: Number of trading days to simulate
        num_simulations: Number of Monte Carlo paths
        seed: Random seed for reproducibility
    
    Returns:
        Dict with simulation results and statistics
    """
    rng = np.random.default_rng(seed)
    
    # Storage for results
    final_capitals = np.zeros(num_simulations)
    all_paths = []
    max_drawdowns = []
    realized_trade_pct_changes: List[float] = []
    
    for sim in range(num_simulations):
        capital = initial_capital
        capital_path = [capital]
        peak_capital = capital
        max_dd = 0
        
        for day in range(days):
            # Random number of trades per day (Poisson distribution)
            num_trades = rng.poisson(avg_trades_per_day)
            
            for _ in range(num_trades):
                # Determine win or loss
                is_win = rng.random() < win_rate
                
                if is_win:
                    # Win: sample from normal distribution around avg_win_pct
                    pct_change = rng.normal(avg_win_pct, avg_win_pct * 0.3)
                    pct_change = max(0, pct_change)  # Can't be negative
                else:
                    # Loss: sample from normal distribution around avg_loss_pct
                    pct_change = -rng.normal(avg_loss_pct, avg_loss_pct * 0.3)
                    pct_change = min(0, pct_change)  # Can't be positive

                if str(distribution).lower() == "alpha_stable":
                    shock = float(
                        _sample_shocks(
                            rng=rng,
                            distribution="alpha_stable",
                            shape=(1, 1),
                            stable_params=stable_params,
                            max_abs_shock=12.0,
                        )[0, 0]
                    )
                    # Inflate magnitude under heavy-tail shocks while preserving trade sign.
                    pct_change *= (1.0 + (0.20 * abs(shock)))

                realized_trade_pct_changes.append(float(pct_change))
                
                # Apply to capital
                capital *= (1 + pct_change / 100)
                capital = max(0, capital)  # Can't go below 0
                
                # Track peak and drawdown
                if capital > peak_capital:
                    peak_capital = capital
                drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
                max_dd = max(max_dd, drawdown)
            
            capital_path.append(capital)
        
        final_capitals[sim] = capital
        all_paths.append(capital_path)
        max_drawdowns.append(max_dd)
    
    # Calculate statistics
    returns = (final_capitals - initial_capital) / initial_capital * 100
    
    results = {
        'final_capitals': final_capitals,
        'all_paths': np.array(all_paths),
        'max_drawdowns': np.array(max_drawdowns),
        'mean_final': np.mean(final_capitals),
        'median_final': np.median(final_capitals),
        'std_final': np.std(final_capitals),
        'mean_return_pct': np.mean(returns),
        'median_return_pct': np.median(returns),
        'percentile_5': np.percentile(final_capitals, 5),
        'percentile_25': np.percentile(final_capitals, 25),
        'percentile_75': np.percentile(final_capitals, 75),
        'percentile_95': np.percentile(final_capitals, 95),
        'prob_profit': np.mean(final_capitals > initial_capital),
        'prob_loss_50pct': np.mean(final_capitals < initial_capital * 0.5),
        'mean_max_drawdown': np.mean(max_drawdowns),
        'worst_drawdown': np.max(max_drawdowns),
        'distribution': distribution,
    }

    # Tail-event diagnostics: frequency of 3/5/10 sigma-equivalent trade shocks.
    trade_changes = np.asarray(realized_trade_pct_changes, dtype=float)
    trade_sigma = float(np.std(trade_changes)) if trade_changes.size else 0.0
    if trade_sigma > 0.0:
        standardized = trade_changes / trade_sigma
        tail_metrics = tail_event_frequency(standardized, sigma_levels=(3, 5, 10))
        results['simulated_crash_freq_3sigma'] = tail_metrics.get('rate_3sigma', 0.0)
        results['simulated_crash_freq_5sigma'] = tail_metrics.get('rate_5sigma', 0.0)
        results['simulated_crash_freq_10sigma'] = tail_metrics.get('rate_10sigma', 0.0)
        results['simulated_crash_count_3sigma'] = tail_metrics.get('count_3sigma', 0.0)
        results['simulated_crash_count_5sigma'] = tail_metrics.get('count_5sigma', 0.0)
        results['simulated_crash_count_10sigma'] = tail_metrics.get('count_10sigma', 0.0)
    else:
        results['simulated_crash_freq_3sigma'] = 0.0
        results['simulated_crash_freq_5sigma'] = 0.0
        results['simulated_crash_freq_10sigma'] = 0.0
        results['simulated_crash_count_3sigma'] = 0.0
        results['simulated_crash_count_5sigma'] = 0.0
        results['simulated_crash_count_10sigma'] = 0.0
    
    return results


def plot_monte_carlo_results(results, initial_capital=100, show=True):
    """
    Visualize Monte Carlo simulation results with 4-panel plot.
    
    Args:
        results: Dict returned from monte_carlo_strategy_simulation()
        initial_capital: Starting capital for reference
        show: Whether to call plt.show()
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Final Capital Distribution (Histogram)
    ax1 = axes[0, 0]
    ax1.hist(results['final_capitals'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(initial_capital, color='red', linestyle='--', linewidth=2, label=f'Initial: ${initial_capital}')
    ax1.axvline(results['mean_final'], color='green', linestyle='--', linewidth=2, label=f'Mean: ${results["mean_final"]:.2f}')
    ax1.axvline(results['median_final'], color='orange', linestyle='--', linewidth=2, label=f'Median: ${results["median_final"]:.2f}')
    ax1.set_xlabel('Final Capital ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Final Capital Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Return Distribution
    ax2 = axes[0, 1]
    returns = (results['final_capitals'] - initial_capital) / initial_capital * 100
    ax2.hist(returns, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax2.axvline(results['mean_return_pct'], color='green', linestyle='--', linewidth=2, label=f'Mean: {results["mean_return_pct"]:.1f}%')
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Return Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Sample Paths (20 random simulations)
    ax3 = axes[1, 0]
    sample_indices = np.random.choice(len(results['all_paths']), size=min(20, len(results['all_paths'])), replace=False)
    for idx in sample_indices:
        ax3.plot(results['all_paths'][idx], alpha=0.3, color='gray')
    
    # Overlay mean path
    mean_path = np.mean(results['all_paths'], axis=0)
    ax3.plot(mean_path, color='blue', linewidth=3, label='Mean Path')
    ax3.axhline(initial_capital, color='red', linestyle='--', linewidth=2, label='Initial Capital')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Capital ($)')
    ax3.set_title('Sample Capital Paths (20 simulations)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Maximum Drawdown Distribution
    ax4 = axes[1, 1]
    ax4.hist(results['max_drawdowns'] * 100, bins=50, color='darkred', alpha=0.7, edgecolor='black')
    ax4.axvline(results['mean_max_drawdown'] * 100, color='orange', linestyle='--', linewidth=2, 
                label=f'Mean: {results["mean_max_drawdown"]*100:.1f}%')
    ax4.set_xlabel('Max Drawdown (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Maximum Drawdown Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if show:
        plt.show()
    
    return fig


def print_monte_carlo_summary(results, initial_capital=100):
    """Print formatted summary of Monte Carlo simulation results."""
    print("=" * 70)
    print("📊 MONTE CARLO SIMULATION RESULTS")
    print("=" * 70)
    print(f"\n💰 Capital Statistics:")
    print(f"   Initial Capital:    ${initial_capital:.2f}")
    print(f"   Mean Final:         ${results['mean_final']:.2f}")
    print(f"   Median Final:       ${results['median_final']:.2f}")
    print(f"   Std Deviation:      ${results['std_final']:.2f}")
    
    print(f"\n📈 Return Statistics:")
    print(f"   Mean Return:        {results['mean_return_pct']:.2f}%")
    print(f"   Median Return:      {results['median_return_pct']:.2f}%")
    
    print(f"\n📊 Percentiles:")
    print(f"   5th Percentile:     ${results['percentile_5']:.2f}")
    print(f"   25th Percentile:    ${results['percentile_25']:.2f}")
    print(f"   75th Percentile:    ${results['percentile_75']:.2f}")
    print(f"   95th Percentile:    ${results['percentile_95']:.2f}")
    
    print(f"\n🎲 Probabilities:")
    print(f"   Prob of Profit:     {results['prob_profit']:.1%}")
    print(f"   Prob of 50%+ Loss:  {results['prob_loss_50pct']:.1%}")
    
    print(f"\n⚠️  Risk Metrics:")
    print(f"   Mean Max Drawdown:  {results['mean_max_drawdown']*100:.2f}%")
    print(f"   Worst Drawdown:     {results['worst_drawdown']*100:.2f}%")
    print(f"   Crash Freq 3σ:      {results.get('simulated_crash_freq_3sigma', 0.0):.3%}")
    print(f"   Crash Freq 5σ:      {results.get('simulated_crash_freq_5sigma', 0.0):.3%}")
    print(f"   Crash Freq 10σ:     {results.get('simulated_crash_freq_10sigma', 0.0):.3%}")
    print("=" * 70)
