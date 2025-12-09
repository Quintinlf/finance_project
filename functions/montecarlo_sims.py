# montecarlo_simulations.py

import numpy as np
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt


# ===========================================================
# Monte Carlo + MCMC utilities (library-only, no printing)
# ===========================================================

def simulate_gbm_paths(S0: float, mu: float, sigma: float, T: float, dt: float, N_sim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Simulate GBM-like price paths using log-returns ~ Normal(mu*dt, sigma*sqrt(dt)).
    Returns array of shape (N_sim, N_steps).
    """
    rng = np.random.default_rng(seed)
    N_steps = int(T / dt)
    daily_returns = rng.normal(loc=(mu * dt), scale=(sigma * np.sqrt(dt)), size=(N_sim, N_steps))
    price_paths = S0 * np.exp(np.cumsum(daily_returns, axis=1))
    return price_paths


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


# ===============================
# RISK ANALYSIS MODULE
# ===============================

class RiskModel:
    def __init__(self, mu: float = 0.07, sigma: float = 0.2, T: float = 1, dt: float = 1/252):
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.rng = np.random.default_rng(42)

    def simulate_gbm_paths(self, S0: float, N_sim: int = 10000):
        N_steps = int(self.T / self.dt)
        daily_returns = self.rng.normal(
            loc=(self.mu * self.dt),
            scale=(self.sigma * np.sqrt(self.dt)),
            size=(N_sim, N_steps)
        )
        return S0 * np.exp(np.cumsum(daily_returns, axis=1))

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
            paths = self.simulate_gbm_paths(S0)
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
    seed=None
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
        'worst_drawdown': np.max(max_drawdowns)
    }
    
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
    print("=" * 70)
