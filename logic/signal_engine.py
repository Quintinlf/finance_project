"""
Signal Generation Engine

Wraps the existing forecasting logic and provides a clean interface
for generating trading signals across a universe of stocks.

Key function: generate_signals(universe, config) -> List[Signal]
"""

import yfinance as yf
from typing import List, Dict, Any
from datetime import datetime

from data_structures import Signal, ExecutionConfig
from trading_functions import unified_bayesian_gp_forecast, calculate_bollinger_bands


def generate_signals(
    universe: List[str],
    config: ExecutionConfig,
    verbose: bool = False
) -> List[Signal]:
    """
    Generate trading signals for all symbols in the universe.
    
    Args:
        universe: list of ticker symbols to analyze
        config: execution configuration (contains thresholds)
        verbose: whether to print progress
    
    Returns:
        List of Signal objects, one per symbol analyzed
    
    Process:
        1. Fetch recent price data for each symbol
        2. Calculate Bollinger Bands
        3. Run Bayesian + GP forecast
        4. Combine signals and adjust confidence based on alignment
        5. Package as Signal object
    """
    signals = []
    
    for i, symbol in enumerate(universe, 1):
        if verbose:
            print(f"[{i}/{len(universe)}] Analyzing {symbol}...")
        
        try:
            signal = _generate_single_signal(symbol, config)
            if signal:
                signals.append(signal)
        except Exception as e:
            if verbose:
                print(f"  ❌ Error on {symbol}: {str(e)[:80]}")
            continue
    
    return signals


def _generate_single_signal(symbol: str, config: ExecutionConfig) -> Signal:
    """
    Generate signal for a single symbol.
    
    Returns:
        Signal object or None if insufficient data
    """
    # Fetch price data
    ticker_data = yf.Ticker(symbol)
    price_history = ticker_data.history(period="3mo", interval="1d")
    
    if price_history.empty:
        return None
    
    # Calculate Bollinger Bands
    df_with_bb = calculate_bollinger_bands(price_history, window=20, num_std=2)
    bb_z_score = df_with_bb['BB_Z_Score'].iloc[-1]
    current_price = df_with_bb['Close'].iloc[-1]
    bb_width = df_with_bb['BB_Width'].iloc[-1]
    
    # Determine Bollinger Band signal
    if bb_z_score < -1.5:
        bb_signal = "BUY"
    elif bb_z_score > 1.5:
        bb_signal = "SELL"
    else:
        bb_signal = "NEUTRAL"
    
    # Run forecast model
    forecast_result = unified_bayesian_gp_forecast(symbol)
    
    if not forecast_result:
        return None
    
    # Extract forecast components
    base_signal = forecast_result['final_signal']  # BUY, SELL, or HOLD
    base_confidence = forecast_result['final_confidence']
    # Use corrected probability of profit if available, otherwise fall back to formula
    if 'prob_profit' in forecast_result['ensemble']:
        ensemble_forecast = forecast_result['ensemble']['prob_profit']
    else:
        # Fallback: calculate from return distribution using normal CDF
        from scipy import stats
        raw_forecast = forecast_result['ensemble']['forecast']
        ensemble_std = forecast_result['ensemble'].get('std', 0.01)
        ensemble_forecast = float(stats.norm.cdf(raw_forecast / (ensemble_std + 1e-8)))
    ensemble_z_score = forecast_result['ensemble'].get('z_score', 0.0)
    rsi_value = forecast_result['rsi']['value']
    
    # Check signal alignment
    signals_agree = (
        (base_signal == 'BUY' and bb_signal == 'BUY') or
        (base_signal == 'SELL' and bb_signal == 'SELL') or
        bb_signal == 'NEUTRAL'
    )
    
    # Adjust confidence based on technical alignment
    combined_confidence = base_confidence
    if signals_agree and bb_signal != 'NEUTRAL':
        # Boost confidence when forecast and BB agree
        combined_confidence = min(1.0, base_confidence * 1.15)
    elif not signals_agree and bb_signal != 'NEUTRAL':
        # Reduce confidence when they conflict
        combined_confidence = base_confidence * 0.85
    
    # Normalize signal type to lowercase for consistency
    signal_type = base_signal.lower()
    
    # Build Signal object
    signal = Signal(
        symbol=symbol,
        signal_type=signal_type,
        confidence=combined_confidence,
        prob_profit=ensemble_forecast,
        meta={
            'base_confidence': base_confidence,
            'bb_signal': bb_signal,
            'bb_z_score': bb_z_score,
            'bb_width': bb_width,
            'ensemble_z_score': ensemble_z_score,
            'rsi_value': rsi_value,
            'signals_agree': signals_agree,
            'current_price': current_price,
            'full_forecast': forecast_result  # Keep for advanced use
        }
    )
    
    return signal


def filter_signals_by_thresholds(
    signals: List[Signal],
    min_confidence: float,
    min_prob_up: float
) -> List[Signal]:
    """
    Filter signals that meet minimum thresholds.
    
    Args:
        signals: list of Signal objects
        min_confidence: minimum confidence threshold
        min_prob_up: minimum probability of profit
    
    Returns:
        Filtered list containing only actionable signals
    """
    actionable = []
    
    for signal in signals:
        # Only consider buy/sell signals (skip hold)
        if signal.signal_type == 'hold':
            continue
        
        # Check thresholds
        meets_confidence = signal.confidence >= min_confidence
        meets_prob = signal.prob_profit >= min_prob_up
        
        if meets_confidence and meets_prob:
            actionable.append(signal)
    
    return actionable
