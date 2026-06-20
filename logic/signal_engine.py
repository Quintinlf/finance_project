"""
Signal Generation Engine

Wraps the existing forecasting logic and provides a clean interface
for generating trading signals across a universe of stocks.

Key function: generate_signals(universe, config) -> List[Signal]
"""

import logging
import math
from typing import List, Dict, Any, Optional, Literal, cast

import yfinance as yf

from logic.data_structures import Signal, ExecutionConfig
from logic.trading_functions import unified_bayesian_gp_forecast, calculate_bollinger_bands
from logic.game_utils import (
    compute_market_regime,
    infer_type_beliefs,
    build_expected_return_path,
    calculate_equilibrium_payoffs,
    build_deviation_from_market_data,
)
from logic.model_performance_tracker import build_component_snapshot


def _normalize_prob_map(values: Dict[str, float]) -> Dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in values.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        if not clipped:
            return {}
        uniform = 1.0 / float(len(clipped))
        return {k: uniform for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def _shannon_entropy(values: Dict[str, float]) -> float:
    normalized = _normalize_prob_map(values)
    if not normalized:
        return 0.0
    return float(-sum(p * math.log(max(p, 1e-12)) for p in normalized.values()))


def _normalize_forecast_label(label: Any) -> Literal['buy', 'sell', 'hold']:
    normalized = str(label).strip().lower().replace("_", " ")
    if normalized in {"strong buy", "buy"}:
        return cast(Literal['buy', 'sell', 'hold'], 'buy')
    if normalized in {"strong sell", "sell"}:
        return cast(Literal['buy', 'sell', 'hold'], 'sell')
    if normalized in {"hold", "neutral"}:
        return cast(Literal['buy', 'sell', 'hold'], 'hold')
    return cast(Literal['buy', 'sell', 'hold'], 'hold')


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


def _generate_single_signal(symbol: str, config: ExecutionConfig) -> Optional[Signal]:
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
    
    original_signal_label = str(base_signal)
    normalized_signal_type = _normalize_forecast_label(original_signal_label)
    signal_strength = float(base_confidence)

    # Check signal alignment using the normalized directional label.
    normalized_signal_upper = normalized_signal_type.upper()
    signals_agree = (
        (normalized_signal_upper == 'BUY' and bb_signal == 'BUY') or
        (normalized_signal_upper == 'SELL' and bb_signal == 'SELL') or
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
    
    if original_signal_label != normalized_signal_upper:
        logging.info(
            "SIGNAL NORMALIZED | symbol=%s | original=%s | normalized=%s | confidence=%.4f | prob_profit=%.4f",
            symbol,
            original_signal_label,
            normalized_signal_upper,
            float(combined_confidence),
            float(ensemble_forecast),
        )

    signal_type = normalized_signal_type

    # Game-theory context: regime -> beliefs -> payoffs.
    regime = compute_market_regime(price_history)
    one_step_return = float(forecast_result['ensemble'].get('forecast', 0.0))
    expected_return_path = build_expected_return_path(one_step_return, regime, horizon=5)
    equilibrium_payoffs = calculate_equilibrium_payoffs(expected_return_path, regime)
    deviation_payoff = build_deviation_from_market_data(
        signal_type=signal_type,
        current_price=float(current_price),
        bb_width=float(bb_width),
        price_history=price_history,
    )
    type_beliefs = infer_type_beliefs(
        signal_type=signal_type,
        rsi_value=float(rsi_value),
        bb_z_score=float(bb_z_score),
        regime=regime,
    )

    deviation_payoff_by_type = {
        't_trend': float(one_step_return - 0.25 * deviation_payoff.hunting_cost),
        't_manipulator': float(deviation_payoff.net_payoff_manipulator),
        't_exhausted': float((-0.50 * one_step_return) + (0.20 * deviation_payoff.net_payoff_manipulator)),
        't_range': float((0.35 * abs(one_step_return)) - (0.15 * deviation_payoff.hunting_cost)),
    }

    regime_probs = regime.as_dict()
    belief_entropy = _shannon_entropy(type_beliefs)
    regime_entropy = _shannon_entropy(regime_probs)
    max_belief_prob = max(type_beliefs.values()) if type_beliefs else 0.0
    instability_tag = 'unstable' if (belief_entropy > 1.0 or max_belief_prob < 0.45) else 'stable'
    component_snapshot = build_component_snapshot(
        forecast_result=forecast_result,
        bb_signal=bb_signal,
        bb_z_score=float(bb_z_score),
        ensemble_signal=normalized_signal_upper,
        ensemble_confidence=float(combined_confidence),
        rsi_value=float(rsi_value),
    )
    
    # Build Signal object
    signal = Signal(
        symbol=symbol,
        signal_type=signal_type,
        confidence=combined_confidence,
        prob_profit=ensemble_forecast,
        type_beliefs=type_beliefs,
        meta={
            'base_confidence': base_confidence,
            'signal_strength': signal_strength,
            'original_signal_label': original_signal_label,
            'normalized_signal_label': normalized_signal_upper,
            'confidence_adjustment_factor': float(combined_confidence / signal_strength) if signal_strength else 0.0,
            'bb_signal': bb_signal,
            'bb_z_score': bb_z_score,
            'bb_width': bb_width,
            'ensemble_z_score': ensemble_z_score,
            'ensemble_forecast_std': float(forecast_result['ensemble'].get('std', 0.0) or 0.0),
            'ensemble_forecast_return': one_step_return,
            'rsi_value': rsi_value,
            'signals_agree': signals_agree,
            'current_price': current_price,
            'market_regime': regime_probs,
            'expected_return_path': expected_return_path,
            'equilibrium_payoffs': equilibrium_payoffs.as_dict(),
            'deviation_payoff': deviation_payoff.as_dict(),
            'deviation_payoff_by_type': deviation_payoff_by_type,
            'belief_entropy': belief_entropy,
            'uncertainty_entropy': belief_entropy,
            'regime_entropy': regime_entropy,
            'max_belief_probability': float(max_belief_prob),
            'uncertainty_regime_tag': instability_tag,
            'uncertainty_source': 'signal_engine',
            'uncertainty_gate_mode_snapshot': config.uncertainty_gate_mode,
            'kl_divergence': None,
            'js_divergence': None,
            'component_snapshot': component_snapshot,
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
    evaluated: List[Signal] = []
    
    for signal in signals:
        original_label = str(signal.meta.get('original_signal_label', signal.signal_type.upper()))
        normalized_label = str(signal.meta.get('normalized_signal_label', signal.signal_type)).upper()
        confidence = float(signal.confidence)
        prob_profit = float(signal.prob_profit)

        if signal.signal_type == 'hold':
            reason = 'explicit HOLD decision retained for logging'
            signal.meta['threshold_decision'] = 'hold'
            signal.meta['threshold_reason'] = reason
            logging.info(
                "SIGNAL FILTER | symbol=%s | original=%s | normalized=%s | confidence=%.4f | prob_profit=%.4f | decision=HOLD | reason=%s",
                signal.symbol,
                original_label,
                normalized_label,
                confidence,
                prob_profit,
                reason,
            )
            evaluated.append(signal)
            continue
        
        meets_confidence = confidence >= min_confidence
        meets_prob = prob_profit >= min_prob_up

        if meets_confidence and meets_prob:
            signal.meta['threshold_decision'] = 'pass'
            signal.meta['threshold_reason'] = 'meets confidence and probability thresholds'
            logging.info(
                "SIGNAL FILTER | symbol=%s | original=%s | normalized=%s | confidence=%.4f | prob_profit=%.4f | decision=PASS | reason=meets confidence and probability thresholds",
                signal.symbol,
                original_label,
                normalized_label,
                confidence,
                prob_profit,
            )
            evaluated.append(signal)
            continue

        rejection_reasons = []
        if not meets_confidence:
            rejection_reasons.append(f"confidence {confidence:.4f} < {min_confidence:.4f}")
        if not meets_prob:
            rejection_reasons.append(f"prob_profit {prob_profit:.4f} < {min_prob_up:.4f}")
        reason = '; '.join(rejection_reasons) if rejection_reasons else 'unknown threshold failure'
        signal.meta['threshold_decision'] = 'reject'
        signal.meta['threshold_rejection_reason'] = reason
        signal.meta['threshold_reason'] = reason
        logging.info(
            "SIGNAL FILTER | symbol=%s | original=%s | normalized=%s | confidence=%.4f | prob_profit=%.4f | decision=REJECT | reason=%s",
            signal.symbol,
            original_label,
            normalized_label,
            confidence,
            prob_profit,
            reason,
        )
    
    return evaluated
