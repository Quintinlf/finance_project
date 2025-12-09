"""
Risk Management Module

Features:
- Adaptive threshold calculation using Bayesian updating
- MCMC-based parameter optimization for trading thresholds
- Position sizing and risk metrics
- Portfolio risk analysis

This module helps optimize trading parameters dynamically based on performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ===========================================================
# ADAPTIVE THRESHOLDS (Bayesian Parameter Tuning)
# ===========================================================

def adaptive_threshold_calculator(
    trade_history: List[Dict],
    initial_min_conf: float = 0.65,
    alpha: float = 0.1,
    target_win_rate: float = 0.55
) -> float:
    """
    Calculate adaptive MIN_CONF threshold using Bayesian updating.
    
    The algorithm adjusts the confidence threshold based on:
    - Win rate vs target win rate
    - Profit factor (total wins / total losses)
    - Recent performance trends
    
    Args:
        trade_history: List of dicts with 'confidence' and 'profit' keys
        initial_min_conf: Starting threshold (default 0.65)
        alpha: Learning rate (0.0 to 1.0, default 0.1)
        target_win_rate: Desired win rate (default 0.55)
    
    Returns:
        New recommended MIN_CONF threshold (bounded between 0.55 and 0.80)
    """
    if not trade_history or len(trade_history) < 10:
        print(f"⚠️ Not enough trades ({len(trade_history)}/10 min) for adaptation. Using initial threshold.")
        return initial_min_conf
    
    df = pd.DataFrame(trade_history)
    
    # Calculate performance metrics
    total_trades = len(df)
    wins = (df['profit'] > 0).sum()
    losses = (df['profit'] < 0).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    total_profit = df[df['profit'] > 0]['profit'].sum()
    total_loss = abs(df[df['profit'] < 0]['profit'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    # Bayesian adjustment logic
    current_threshold = initial_min_conf
    
    # Rule 1: Win rate adjustment
    win_rate_error = win_rate - target_win_rate
    if win_rate < target_win_rate - 0.05:
        # Win rate too low -> increase threshold (be more selective)
        adjustment = alpha * abs(win_rate_error)
        current_threshold += adjustment
    elif win_rate > target_win_rate + 0.10:
        # Win rate very high -> decrease threshold (capture more opportunities)
        adjustment = alpha * 0.5 * win_rate_error
        current_threshold -= adjustment
    
    # Rule 2: Profit factor adjustment
    if profit_factor < 1.2:
        # Poor profit factor -> increase threshold
        current_threshold += alpha * 0.05
    elif profit_factor > 2.0:
        # Excellent profit factor -> can afford to lower threshold slightly
        current_threshold -= alpha * 0.02
    
    # Rule 3: Recent performance (last 20% of trades)
    recent_cutoff = int(total_trades * 0.8)
    recent_trades = df.iloc[recent_cutoff:]
    recent_win_rate = (recent_trades['profit'] > 0).sum() / len(recent_trades) if len(recent_trades) > 0 else 0
    
    if recent_win_rate < 0.45:
        # Recent performance declining -> increase threshold
        current_threshold += alpha * 0.03
    
    # Bound the threshold
    new_threshold = np.clip(current_threshold, 0.55, 0.80)
    
    print(f"🎯 Adaptive Threshold Calculation:")
    print(f"   Current Win Rate: {win_rate:.1%} (Target: {target_win_rate:.1%})")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   Recent Win Rate: {recent_win_rate:.1%}")
    print(f"   Old Threshold: {initial_min_conf:.2f}")
    print(f"   New Threshold: {new_threshold:.2f}")
    
    return float(new_threshold)


# ===========================================================
# MCMC PARAMETER OPTIMIZATION
# ===========================================================

def mcmc_optimize_thresholds(
    trade_history: List[Dict],
    n_samples: int = 5000,
    burn_in: int = 1000,
    thin: int = 10,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Use MCMC (Markov Chain Monte Carlo) to find optimal trading thresholds.
    
    Optimizes:
    - MIN_CONF: Minimum confidence threshold
    - TP_PCT: Take profit percentage
    - SL_PCT: Stop loss percentage
    
    Uses Metropolis-Hastings sampling to explore the parameter space.
    
    Args:
        trade_history: List of dicts with 'confidence', 'profit', 'entry_price', etc.
        n_samples: Total MCMC samples
        burn_in: Number of initial samples to discard
        thin: Keep every Nth sample (reduces autocorrelation)
        seed: Random seed
    
    Returns:
        Dict with optimal parameters: {'MIN_CONF', 'TP_PCT', 'SL_PCT', 'expected_pnl'}
    """
    try:
        import emcee
    except ImportError:
        raise ImportError("emcee required for MCMC. Install with: pip install emcee")
    
    if not trade_history or len(trade_history) < 30:
        raise ValueError(f"Need at least 30 trades for MCMC optimization (got {len(trade_history)})")
    
    df = pd.DataFrame(trade_history)
    rng = np.random.default_rng(seed)
    
    def simulate_strategy_pnl(min_conf, tp_pct, sl_pct):
        """Simulate P&L given thresholds by replaying historical trades."""
        total_pnl = 0
        trade_count = 0
        
        for _, trade in df.iterrows():
            # Skip trades below confidence threshold
            if trade.get('confidence', 0) < min_conf:
                continue
            
            trade_count += 1
            
            # Simulate outcome based on actual market movement
            actual_pnl = trade.get('profit', 0)
            entry = trade.get('entry_price', 100)
            
            # Apply TP/SL limits
            max_profit = entry * (tp_pct / 100)
            max_loss = -entry * (sl_pct / 100)
            
            capped_pnl = np.clip(actual_pnl, max_loss, max_profit)
            total_pnl += capped_pnl
        
        # Penalize if too few trades
        if trade_count < 10:
            return -1000
        
        # Expected PnL per trade
        return total_pnl / trade_count if trade_count > 0 else -1000
    
    def log_likelihood(theta):
        """Log likelihood for MCMC: higher = better parameters."""
        min_conf, tp_pct, sl_pct = theta
        
        # Prior constraints
        if not (0.5 <= min_conf <= 0.85):
            return -np.inf
        if not (1.0 <= tp_pct <= 10.0):
            return -np.inf
        if not (0.5 <= sl_pct <= 5.0):
            return -np.inf
        if tp_pct <= sl_pct:  # TP should be > SL
            return -np.inf
        
        # Simulate strategy with these parameters
        expected_pnl = simulate_strategy_pnl(min_conf, tp_pct, sl_pct)
        
        # Return log-likelihood (proportional to PnL)
        return expected_pnl
    
    # Initialize walkers
    n_dim = 3
    n_walkers = 20
    
    # Starting point: current "good guess" parameters
    initial = np.array([0.65, 4.0, 2.0])  # [MIN_CONF, TP_PCT, SL_PCT]
    pos = initial + 0.05 * rng.standard_normal((n_walkers, n_dim))
    
    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_likelihood)
    print(f"🔬 Running MCMC optimization ({n_samples} samples)...")
    sampler.run_mcmc(pos, n_samples, progress=False)
    
    # Extract samples
    samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    
    # Find best parameters (highest likelihood)
    best_idx = np.argmax(sampler.get_log_prob(discard=burn_in, thin=thin, flat=True))
    best_params = samples[best_idx]
    
    # Also get posterior means
    mean_params = np.mean(samples, axis=0)
    
    optimal_min_conf = float(best_params[0])
    optimal_tp_pct = float(best_params[1])
    optimal_sl_pct = float(best_params[2])
    
    # Calculate expected PnL with optimal parameters
    expected_pnl = simulate_strategy_pnl(optimal_min_conf, optimal_tp_pct, optimal_sl_pct)
    
    print(f"\n✅ MCMC Optimization Complete!")
    print(f"   Optimal MIN_CONF: {optimal_min_conf:.3f} (mean: {mean_params[0]:.3f})")
    print(f"   Optimal TP_PCT:   {optimal_tp_pct:.2f}% (mean: {mean_params[1]:.2f}%)")
    print(f"   Optimal SL_PCT:   {optimal_sl_pct:.2f}% (mean: {mean_params[2]:.2f}%)")
    print(f"   Expected PnL/Trade: ${expected_pnl:.2f}")
    
    return {
        'MIN_CONF': optimal_min_conf,
        'TP_PCT': optimal_tp_pct,
        'SL_PCT': optimal_sl_pct,
        'expected_pnl': expected_pnl,
        'posterior_mean_MIN_CONF': float(mean_params[0]),
        'posterior_mean_TP_PCT': float(mean_params[1]),
        'posterior_mean_SL_PCT': float(mean_params[2]),
        'samples': samples
    }


# ===========================================================
# POSITION SIZING & RISK METRICS
# ===========================================================

def calculate_position_size(
    account_balance: float,
    risk_per_trade_pct: float,
    stop_loss_pct: float,
    price: float
) -> int:
    """
    Calculate optimal position size using fixed-risk position sizing.
    
    Args:
        account_balance: Total account value ($)
        risk_per_trade_pct: Percentage of account to risk per trade (e.g., 1.0 = 1%)
        stop_loss_pct: Stop loss percentage (e.g., 2.0 = 2%)
        price: Entry price per share ($)
    
    Returns:
        Number of shares to buy (integer)
    """
    risk_amount = account_balance * (risk_per_trade_pct / 100)
    loss_per_share = price * (stop_loss_pct / 100)
    
    if loss_per_share <= 0:
        return 0
    
    shares = int(risk_amount / loss_per_share)
    return max(0, shares)


def portfolio_risk_metrics(positions: List[Dict], account_value: float) -> Dict:
    """
    Calculate portfolio-level risk metrics.
    
    Args:
        positions: List of dicts with 'symbol', 'qty', 'entry_price', 'current_price'
        account_value: Total account value
    
    Returns:
        Dict with risk metrics
    """
    if not positions:
        return {
            'total_exposure': 0,
            'concentration_ratio': 0,
            'largest_position_pct': 0,
            'num_positions': 0,
            'avg_position_size': 0
        }
    
    position_values = []
    for pos in positions:
        qty = pos.get('qty', 0)
        price = pos.get('current_price', pos.get('entry_price', 0))
        value = qty * price
        position_values.append(value)
    
    total_exposure = sum(position_values)
    concentration_ratio = total_exposure / account_value if account_value > 0 else 0
    largest_position = max(position_values) if position_values else 0
    largest_position_pct = largest_position / account_value if account_value > 0 else 0
    
    return {
        'total_exposure': total_exposure,
        'concentration_ratio': concentration_ratio,
        'largest_position_pct': largest_position_pct,
        'num_positions': len(positions),
        'avg_position_size': total_exposure / len(positions) if positions else 0
    }


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio (risk-adjusted return).
    
    Args:
        returns: Array of returns (as decimals, e.g., 0.05 = 5%)
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


def max_drawdown(capital_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from capital curve.
    
    Args:
        capital_curve: Array of capital values over time
    
    Returns:
        Tuple of (max_drawdown_pct, peak_idx, trough_idx)
    """
    running_max = np.maximum.accumulate(capital_curve)
    drawdowns = (capital_curve - running_max) / running_max
    
    max_dd = np.min(drawdowns)
    trough_idx = np.argmin(drawdowns)
    peak_idx = np.argmax(capital_curve[:trough_idx+1]) if trough_idx > 0 else 0
    
    return float(max_dd), int(peak_idx), int(trough_idx)
