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
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import warnings
import math
warnings.filterwarnings('ignore')

if TYPE_CHECKING:
    from logic.data_structures import Signal


# ===========================================================
# ADAPTIVE THRESHOLDS (Bayesian Parameter Tuning)
# ===========================================================

def adaptive_threshold_calculator(
    trade_history: List[Dict],
    initial_min_conf: float = 0.65,
    alpha: float = 0.1,
    target_win_rate: float = 0.55,
    include_open_positions: bool = True
) -> float:
    """
    Calculate adaptive MIN_CONF threshold using Bayesian updating.
    
    The algorithm adjusts the confidence threshold based on:
    - Win rate vs target win rate
    - Profit factor (total wins / total losses)
    - Recent performance trends
    
    NEW: Can now learn from OPEN positions (unrealized P&L) in addition to closed trades!
    
    Args:
        trade_history: List of dicts with 'confidence' and 'profit' keys
                      Can include both closed trades and open positions (status='open')
        initial_min_conf: Starting threshold (default 0.65)
        alpha: Learning rate (0.0 to 1.0, default 0.1)
        target_win_rate: Desired win rate (default 0.55)
        include_open_positions: If True, uses unrealized P&L from open positions
    
    Returns:
        New recommended MIN_CONF threshold (bounded between 0.55 and 0.80)
    """
    if not trade_history:
        print(f"⚠️ No trade data available. Using initial threshold.")
        return initial_min_conf
    
    df = pd.DataFrame(trade_history)
    
    # Separate closed and open positions
    if 'status' in df.columns:
        closed_trades = df[df['status'] != 'open']
        open_positions = df[df['status'] == 'open']
    else:
        closed_trades = df
        open_positions = pd.DataFrame()
    
    # Need at least some data points
    total_data_points = len(closed_trades) + (len(open_positions) if include_open_positions else 0)
    
    if total_data_points < 5:
        print(f"⚠️ Not enough data ({total_data_points}/5 min) for adaptation. Using initial threshold.")
        return initial_min_conf
    
    # Use appropriate dataset
    if include_open_positions and len(open_positions) > 0:
        analysis_df = df  # Use both closed and open
        print(f"📊 Analyzing {len(closed_trades)} closed trades + {len(open_positions)} open positions")
    else:
        analysis_df = closed_trades
        print(f"📊 Analyzing {len(closed_trades)} closed trades only")
    
    # Calculate performance metrics
    total_trades = len(analysis_df)
    wins = (analysis_df['profit'] > 0).sum()
    losses = (analysis_df['profit'] < 0).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    total_profit = analysis_df[analysis_df['profit'] > 0]['profit'].sum()
    total_loss = abs(analysis_df[analysis_df['profit'] < 0]['profit'].sum())
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
    recent_trades = analysis_df.iloc[recent_cutoff:]
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


def calculate_minimax_multiplier(signal: 'Signal', verbose: bool = False) -> float:
    """
    Calculate minimax confidence multiplier for hybrid position sizing.
    
    Implements the minimax principle: size the trade as if the worst plausible market
    regime arrives next. This is NOT a risk calculator—it is a confidence modulator.
    
    Formula:
        1. Extract equilibrium payoff (eq_i) for each market type i
        2. Extract deviation payoff (dev_i) for each market type i  
        3. Compute worst-case payoff: P_i = eq_i + dev_i
        4. Minimax payoff: W = min(P_i) across all types
        5. Volatility proxy: σ = |ensemble_forecast_std| + 1e-6
        6. Raw Kelly fraction: f_raw = W / σ
        7. Bounded multiplier: f_mm = clip(f_raw, 0, 1)
    
    Args:
        signal: Signal object with meta dict containing equilibrium/deviation payoffs
        verbose: If True, print diagnostic info (for debugging)
    
    Returns:
        Minimax multiplier (float, 0.0 to 1.0)
        - 0.0: Adversarial setup (no edge, don't trade)
        - 0.3-0.8: Typical confidence levels
        - 1.0: Strong alignment across regimes  
    
    Note:
        All payoff and volatility values are assumed to be already computed and stored
        in signal.meta during signal generation (Phase 2).
    """
    # Try to extract equilibrium payoffs
    eq_payoffs = signal.meta.get('equilibrium_payoffs')
    if eq_payoffs is None:
        if verbose:
            print("  [minimax] No equilibrium payoffs in signal.meta; returning 0.5 (neutral)")
        return 0.5
    
    # Convert TypeEquilibrium dataclass to dict if needed
    if hasattr(eq_payoffs, 'as_dict'):
        eq_dict = eq_payoffs.as_dict()
    elif isinstance(eq_payoffs, dict):
        eq_dict = eq_payoffs
    else:
        if verbose:
            print(f"  [minimax] Unexpected equilibrium_payoffs type: {type(eq_payoffs)}")
        return 0.5
    
    # Extract deviation payoffs for each market type
    dev_dict = signal.meta.get('deviation_payoff_by_type', {})
    
    # Define market types (from game_utils.MARKET_TYPES)
    market_types = ['t_trend', 't_manipulator', 't_exhausted', 't_range']
    dev_keys = ['trend', 'manipulator', 'exhausted', 'range']  # Shortened keys in dict
    
    # Build payoff vector: P_i = eq_i + dev_i for each type
    payoff_vector = []
    for mt, dk in zip(market_types, dev_keys):
        eq_i = eq_dict.get(mt, 0.0)
        dev_i = dev_dict.get(dk, 0.0)
        p_i = eq_i + dev_i  # Corrected formula (no regime probability weighting)
        payoff_vector.append(p_i)
    
    # Compute minimax payoff: worst-case across all market types
    w_worst_case = min(payoff_vector)
    
    # Extract volatility proxy from signal
    ensemble_std = signal.meta.get('ensemble_forecast_std', 0.0)
    if ensemble_std is None:
        ensemble_std = 0.0
    
    sigma_trade = abs(float(ensemble_std)) + 1e-6  # Add epsilon to avoid divide-by-zero
    
    # Compute raw Kelly fraction
    f_raw = w_worst_case / sigma_trade
    
    # Clip to [0, 1] for hybrid multiplier mode
    f_mm = float(np.clip(f_raw, 0, 1))
    
    # Store diagnostics in signal.meta
    signal.meta['minimax_payoff_vector'] = payoff_vector
    signal.meta['minimax_worst_case_payoff'] = float(w_worst_case)
    signal.meta['minimax_multiplier'] = f_mm
    
    if verbose:
        print(f"  [minimax] Payoff vector: {[round(p, 4) for p in payoff_vector]}")
        print(f"  [minimax] Worst-case W: {round(w_worst_case, 4)}")
        print(f"  [minimax] Trade vol σ: {round(sigma_trade, 4)}")
        print(f"  [minimax] Raw Kelly f: {round(f_raw, 4)}")
        print(f"  [minimax] Clipped multiplier: {round(f_mm, 4)}")
    
    return f_mm


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


# ===========================================================
# MAE/MFE TRACKING (Maximum Adverse/Favorable Excursion)
# ===========================================================

def calculate_maximum_adverse_excursion(
    entry_price: float,
    intraday_prices: List[float],
    side: str = 'long'
) -> Tuple[float, float]:
    """
    Calculate Maximum Adverse Excursion (MAE) for a trade.
    
    MAE measures the worst price movement against a position during its lifetime.
    This is crucial for stop-loss optimization and risk management.
    
    Args:
        entry_price: Entry price of the trade
        intraday_prices: List of prices during the trade's lifetime
        side: 'long' or 'short' position
        
    Returns:
        Tuple of (mae_dollars, mae_percent)
        
    Example:
        >>> # Long position: entered at $100, worst price was $97
        >>> mae_dollars, mae_pct = calculate_maximum_adverse_excursion(100, [99, 97, 101], 'long')
        >>> print(f"MAE: ${mae_dollars:.2f} ({mae_pct:.1f}%)")
        MAE: $3.00 (-3.0%)
    """
    if not intraday_prices or len(intraday_prices) == 0:
        return 0.0, 0.0
    
    prices = np.array(intraday_prices)
    
    if side.lower() == 'long':
        # For long: MAE is the maximum loss (entry - lowest price)
        worst_price = np.min(prices)
        mae_dollars = entry_price - worst_price
        mae_percent = ((entry_price - worst_price) / entry_price) * 100
    else:  # short
        # For short: MAE is the maximum loss (highest price - entry)
        worst_price = np.max(prices)
        mae_dollars = worst_price - entry_price
        mae_percent = ((worst_price - entry_price) / entry_price) * 100
    
    return float(mae_dollars), float(mae_percent)


def calculate_maximum_favorable_excursion(
    entry_price: float,
    intraday_prices: List[float],
    side: str = 'long'
) -> Tuple[float, float]:
    """
    Calculate Maximum Favorable Excursion (MFE) for a trade.
    
    MFE measures the best price movement in favor of a position during its lifetime.
    Useful for take-profit optimization and understanding profit potential.
    
    Args:
        entry_price: Entry price of the trade
        intraday_prices: List of prices during the trade's lifetime
        side: 'long' or 'short' position
        
    Returns:
        Tuple of (mfe_dollars, mfe_percent)
        
    Example:
        >>> # Long position: entered at $100, best price was $105
        >>> mfe_dollars, mfe_pct = calculate_maximum_favorable_excursion(100, [101, 105, 103], 'long')
        >>> print(f"MFE: ${mfe_dollars:.2f} ({mfe_pct:.1f}%)")
        MFE: $5.00 (5.0%)
    """
    if not intraday_prices or len(intraday_prices) == 0:
        return 0.0, 0.0
    
    prices = np.array(intraday_prices)
    
    if side.lower() == 'long':
        # For long: MFE is the maximum profit (highest price - entry)
        best_price = np.max(prices)
        mfe_dollars = best_price - entry_price
        mfe_percent = ((best_price - entry_price) / entry_price) * 100
    else:  # short
        # For short: MFE is the maximum profit (entry - lowest price)
        best_price = np.min(prices)
        mfe_dollars = entry_price - best_price
        mfe_percent = ((entry_price - best_price) / entry_price) * 100
    
    return float(mfe_dollars), float(mfe_percent)


def analyze_mae_mfe_distribution(trade_history: List[Dict]) -> Dict[str, float]:
    """
    Analyze MAE/MFE distribution across historical trades.
    
    This analysis helps optimize stop-loss and take-profit levels by showing:
    - Average MAE for winning vs losing trades
    - Average MFE for winning vs losing trades
    - Optimal stop-loss distance (average MAE for winners + buffer)
    - Optimal take-profit distance (average MFE for winners - buffer)
    
    Args:
        trade_history: List of dicts with keys:
            - 'entry_price': float
            - 'intraday_prices': List[float]
            - 'exit_price': float
            - 'side': 'long' or 'short'
            - 'profit': float (optional, calculated from prices if missing)
            
    Returns:
        dict: Analysis results with recommended stop-loss and take-profit levels
        
    Example:
        >>> analysis = analyze_mae_mfe_distribution(trade_history)
        >>> print(f"Recommended SL: {analysis['recommended_sl_pct']:.2f}%")
        >>> print(f"Recommended TP: {analysis['recommended_tp_pct']:.2f}%")
    """
    if not trade_history or len(trade_history) < 10:
        return {
            'error': 'Need at least 10 trades for MAE/MFE analysis',
            'n_trades': len(trade_history) if trade_history else 0
        }
    
    winners = []
    losers = []
    
    for trade in trade_history:
        entry = trade.get('entry_price')
        prices = trade.get('intraday_prices', [])
        exit_price = trade.get('exit_price')
        side = trade.get('side', 'long')
        
        if not entry or not prices:
            continue
        
        # Calculate profit if not provided
        if 'profit' in trade:
            profit = trade['profit']
        elif exit_price:
            if side.lower() == 'long':
                profit = exit_price - entry
            else:
                profit = entry - exit_price
        else:
            continue
        
        # Calculate MAE and MFE
        mae_dollars, mae_pct = calculate_maximum_adverse_excursion(entry, prices, side)
        mfe_dollars, mfe_pct = calculate_maximum_favorable_excursion(entry, prices, side)
        
        trade_data = {
            'mae_pct': mae_pct,
            'mfe_pct': mfe_pct,
            'profit': profit
        }
        
        if profit > 0:
            winners.append(trade_data)
        else:
            losers.append(trade_data)
    
    if not winners and not losers:
        return {'error': 'No valid trade data with MAE/MFE'}
    
    # Calculate statistics
    winner_mae_avg = np.mean([t['mae_pct'] for t in winners]) if winners else 0
    winner_mfe_avg = np.mean([t['mfe_pct'] for t in winners]) if winners else 0
    loser_mae_avg = np.mean([t['mae_pct'] for t in losers]) if losers else 0
    loser_mfe_avg = np.mean([t['mfe_pct'] for t in losers]) if losers else 0
    
    all_mae = [t['mae_pct'] for t in winners + losers]
    all_mfe = [t['mfe_pct'] for t in winners + losers]
    
    # Recommendations
    # Stop-loss: Set slightly beyond average MAE for winners (to avoid premature stops)
    recommended_sl_pct = winner_mae_avg * 1.2 if winner_mae_avg > 0 else 2.0
    
    # Take-profit: Set at average MFE for winners (capture most of the move)
    recommended_tp_pct = winner_mfe_avg * 0.8 if winner_mfe_avg > 0 else 4.0
    
    return {
        'n_trades': len(trade_history),
        'n_winners': len(winners),
        'n_losers': len(losers),
        'win_rate': len(winners) / (len(winners) + len(losers)) if (winners or losers) else 0,
        'winner_mae_avg': winner_mae_avg,
        'winner_mfe_avg': winner_mfe_avg,
        'loser_mae_avg': loser_mae_avg,
        'loser_mfe_avg': loser_mfe_avg,
        'overall_mae_avg': np.mean(all_mae),
        'overall_mfe_avg': np.mean(all_mfe),
        'mae_std': np.std(all_mae),
        'mfe_std': np.std(all_mfe),
        'recommended_sl_pct': recommended_sl_pct,
        'recommended_tp_pct': recommended_tp_pct
    }


def madl_loss(actual_returns: np.ndarray, 
              predicted_returns: np.ndarray,
              penalty_weight: float = 2.0) -> float:
    """
    Calculate Mean Absolute Directional Loss (MADL).
    
    MADL is a specialized loss function for trading that penalizes:
    1. Incorrect direction predictions (heavily)
    2. Magnitude errors (moderately)
    
    This aligns model optimization with actual trading goals better than MAE/MSE.
    
    Formula:
        MADL = mean(|actual - predicted| * (1 + penalty * wrong_direction))
    
    Args:
        actual_returns: Actual return values
        predicted_returns: Predicted return values
        penalty_weight: Multiplier for wrong direction (default 2.0)
        
    Returns:
        float: MADL loss value
        
    Reference:
        "Mean Absolute Directional Loss as a New Loss Function for 
         Optimization of Machine Learning Models" (arXiv, 2024)
         
    Example:
        >>> actual = np.array([0.02, -0.01, 0.03])
        >>> predicted = np.array([0.025, 0.005, 0.025])  # Wrong direction on 2nd
        >>> loss = madl_loss(actual, predicted)
        >>> print(f"MADL: {loss:.4f}")
    """
    actual = np.array(actual_returns)
    predicted = np.array(predicted_returns)
    
    # Base error (magnitude)
    magnitude_error = np.abs(actual - predicted)
    
    # Direction penalty (1.0 if correct, 1.0 + penalty_weight if wrong)
    actual_sign = np.sign(actual)
    predicted_sign = np.sign(predicted)
    direction_penalty = np.where(
        actual_sign == predicted_sign,
        1.0,
        1.0 + penalty_weight
    )
    
    # Combined loss
    madl = magnitude_error * direction_penalty
    
    return float(np.mean(madl))


def optimize_stop_loss_from_mae(
    symbol: str,
    trade_history: List[Dict],
    min_sl_pct: float = 1.0,
    max_sl_pct: float = 5.0
) -> float:
    """
    Optimize stop-loss percentage based on historical MAE analysis.
    
    Uses MAE distribution to find optimal stop-loss that:
    - Avoids stopping out winning trades prematurely
    - Limits losses on losing trades
    - Adapts to symbol-specific volatility
    
    Args:
        symbol: Stock ticker symbol
        trade_history: Historical trades for this symbol
        min_sl_pct: Minimum stop-loss % (default 1.0%)
        max_sl_pct: Maximum stop-loss % (default 5.0%)
        
    Returns:
        float: Optimized stop-loss percentage
        
    Example:
        >>> optimal_sl = optimize_stop_loss_from_mae('AAPL', trade_history)
        >>> print(f"Optimal SL for AAPL: {optimal_sl:.2f}%")
    """
    # Filter trades for this symbol
    symbol_trades = [t for t in trade_history if t.get('symbol') == symbol]
    
    if len(symbol_trades) < 5:
        # Not enough data, use moderate default
        return 2.0
    
    # Get MAE/MFE analysis
    analysis = analyze_mae_mfe_distribution(symbol_trades)
    
    if 'error' in analysis:
        return 2.0
    
    # Use recommended SL from analysis, clamped to min/max
    recommended_sl = analysis['recommended_sl_pct']
    optimal_sl = max(min_sl_pct, min(max_sl_pct, recommended_sl))
    
    return float(optimal_sl)


def print_mae_mfe_report(analysis: Dict):
    """
    Print formatted MAE/MFE analysis report.
    
    Args:
        analysis: Output from analyze_mae_mfe_distribution()
    """
    if 'error' in analysis:
        print(f"⚠️  {analysis['error']}")
        return
    
    print("=" * 80)
    print("  📊 MAE/MFE ANALYSIS REPORT")
    print("=" * 80)
    print(f"  Total Trades:           {analysis['n_trades']}")
    print(f"  Winners:                {analysis['n_winners']}")
    print(f"  Losers:                 {analysis['n_losers']}")
    print(f"  Win Rate:               {analysis['win_rate']:.1%}")
    print()
    print("  WINNERS:")
    print(f"    Avg MAE (drawdown):   {analysis['winner_mae_avg']:.2f}%")
    print(f"    Avg MFE (peak profit): {analysis['winner_mfe_avg']:.2f}%")
    print()
    print("  LOSERS:")
    print(f"    Avg MAE (drawdown):   {analysis['loser_mae_avg']:.2f}%")
    print(f"    Avg MFE (peak profit): {analysis['loser_mfe_avg']:.2f}%")
    print()
    print("  OVERALL:")
    print(f"    Avg MAE:              {analysis['overall_mae_avg']:.2f}% ± {analysis['mae_std']:.2f}%")
    print(f"    Avg MFE:              {analysis['overall_mfe_avg']:.2f}% ± {analysis['mfe_std']:.2f}%")
    print()
    print("  📋 RECOMMENDATIONS:")
    print(f"    Optimal Stop-Loss:    {analysis['recommended_sl_pct']:.2f}%")
    print(f"    Optimal Take-Profit:  {analysis['recommended_tp_pct']:.2f}%")
    print()
    print("  💡 INSIGHTS:")
    if analysis['winner_mae_avg'] < analysis['loser_mae_avg']:
        print(f"    ✅ Winners have lower drawdown than losers (good signal quality)")
    else:
        print(f"    ⚠️  Winners experience similar drawdown to losers (signal needs work)")
    
    if analysis['winner_mfe_avg'] > analysis['recommended_tp_pct'] * 1.5:
        print(f"    💰 Consider wider take-profit to capture more upside")
    
    print("=" * 80)
