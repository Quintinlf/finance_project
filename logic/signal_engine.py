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
from logic import eos_bridge
from logic.trading_functions import unified_bayesian_gp_forecast, calculate_bollinger_bands
from logic.game_utils import (
    compute_market_state,
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
    verbose: bool = False,
    price_data: Optional[Dict[str, Any]] = None,
    build_plot: bool = True,
) -> List[Signal]:
    """
    Generate trading signals for all symbols in the universe.
    
    Args:
        universe: list of ticker symbols to analyze
        config: execution configuration (contains thresholds)
        verbose: whether to print progress
    
    price_data: optional {symbol: OHLCV frame} of pre-fetched history. When
        given, no download happens for that symbol and the signal is computed
        only from the rows supplied. The backtester passes history truncated at
        the bar under evaluation; that truncation is what keeps the replay
        honest about what was knowable at decision time.
    build_plot: pass False to skip figure construction inside the forecast.

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
            signal = _generate_single_signal(
                symbol,
                config,
                price_history=(price_data or {}).get(symbol),
                build_plot=build_plot,
            )
            if signal:
                signals.append(signal)
        except Exception as e:
            if verbose:
                print(f"  ❌ Error on {symbol}: {str(e)[:80]}")
            continue
    
    return signals


def _generate_single_signal(
    symbol: str,
    config: ExecutionConfig,
    price_history: Optional[Any] = None,
    build_plot: bool = True,
) -> Optional[Signal]:
    """
    Generate signal for a single symbol.

    price_history: everything knowable as of the decision bar, or None to fetch
        live. When supplied it is sliced here into the same two windows
        production uses — ~3 months for Bollinger/regime, 200 days for the
        forecast — so a replayed signal matches a live one bar for bar.

    Returns:
        Signal object or None if insufficient data
    """
    forecast_history = None
    if price_history is None:
        # Fetch a year rather than 3 months so the eos enrichment has enough
        # observations for a stable GARCH/HMM fit (both need >= 60 returns; a
        # 3-month window yields ~62 and fits badly). The Bollinger/regime slice
        # below is still the same trailing ~63 bars, so signal behaviour is
        # unchanged — only the history retained alongside it is longer.
        ticker_data = yf.Ticker(symbol)
        eos_history = ticker_data.history(period="1y", interval="1d")
        price_history = eos_history.tail(63)
    else:
        eos_history = price_history
        forecast_history = price_history.tail(200)
        price_history = price_history.tail(63)  # ~3 months of trading days

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
    forecast_result = unified_bayesian_gp_forecast(
        symbol,
        price_history=forecast_history,
        build_plot=build_plot,
    )

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
    regime = compute_market_state(price_history)
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

    # EOS enrichment: GARCH volatility, Hurst regime, HMM regime, tail risk,
    # and quantum price levels from the physics/quantum/finance-ML domains.
    # In 'shadow' mode this only populates meta; in 'enforce' mode the Hurst
    # multiplier below is allowed to move confidence.
    eos_enrichment: Dict[str, Any] = {}
    eos_mode = getattr(config, 'eos_mode', 'off')
    if eos_mode in {'shadow', 'enforce'}:
        try:
            eos_enrichment = eos_bridge.build_enrichment(
                price_history=eos_history,
                spot_price=float(current_price),
                cache_key=symbol,
                stride=int(getattr(config, 'eos_enrichment_stride', 1) or 1),
            )
        except Exception as exc:
            # Never let an enrichment failure cost a trading decision.
            logging.warning("EOS enrichment failed for %s (%s). Continuing.", symbol, exc)
            eos_enrichment = {'available': False, 'error': str(exc)}

    hurst_adjustment = eos_bridge.hurst_confidence_multiplier(
        signal_type=signal_type,
        bb_signal=bb_signal,
        enrichment=eos_enrichment,
    )
    if eos_mode == 'enforce' and getattr(config, 'eos_use_hurst_confidence', False):
        if hurst_adjustment.get('applied'):
            pre_hurst_confidence = combined_confidence
            combined_confidence = max(
                0.0, min(1.0, combined_confidence * float(hurst_adjustment['multiplier']))
            )
            logging.info(
                "EOS HURST ADJUST | symbol=%s | %.4f -> %.4f | %s",
                symbol,
                pre_hurst_confidence,
                combined_confidence,
                hurst_adjustment.get('reason'),
            )
            hurst_adjustment['enforced'] = True
    if eos_enrichment:
        eos_enrichment['hurst_confidence_adjustment'] = hurst_adjustment

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
            'eos': eos_enrichment,
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
        min_prob_up: minimum probability the move goes the signal's way

    Returns:
        Filtered list containing only actionable signals

    Note on direction: `Signal.prob_profit` carries what the forecaster calls
    `prob_profit_up` — P(next return > 0) from the ensemble's normal CDF. It is
    a *directional* probability, not "probability this trade makes money". So a
    SELL must be judged against the down-tail, 1 - prob_profit. Comparing a SELL
    to `prob_profit >= min_prob_up` rejects exactly the most confident bearish
    calls: a STRONG SELL at prob_profit=0.0005 is a 99.95% down-read, and the
    naive test throws it out for being "too low".
    """
    evaluated: List[Signal] = []

    for signal in signals:
        original_label = str(signal.meta.get('original_signal_label', signal.signal_type.upper()))
        normalized_label = str(signal.meta.get('normalized_signal_label', signal.signal_type)).upper()
        confidence = float(signal.confidence)
        prob_profit = float(signal.prob_profit)
        # Probability the market moves the way this signal is betting.
        directional_prob = (1.0 - prob_profit) if signal.signal_type == 'sell' else prob_profit
        signal.meta['directional_probability'] = directional_prob

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
        meets_prob = directional_prob >= min_prob_up

        if meets_confidence and meets_prob:
            signal.meta['threshold_decision'] = 'pass'
            signal.meta['threshold_reason'] = 'meets confidence and probability thresholds'
            logging.info(
                "SIGNAL FILTER | symbol=%s | original=%s | normalized=%s | confidence=%.4f | prob_profit=%.4f | p_dir=%.4f | decision=PASS | reason=meets confidence and probability thresholds",
                signal.symbol,
                original_label,
                normalized_label,
                confidence,
                prob_profit,
                directional_prob,
            )
            evaluated.append(signal)
            continue

        rejection_reasons = []
        if not meets_confidence:
            rejection_reasons.append(f"confidence {confidence:.4f} < {min_confidence:.4f}")
        if not meets_prob:
            direction_word = 'down' if signal.signal_type == 'sell' else 'up'
            rejection_reasons.append(
                f"P({direction_word}) {directional_prob:.4f} < {min_prob_up:.4f}"
            )
        reason = '; '.join(rejection_reasons) if rejection_reasons else 'unknown threshold failure'
        signal.meta['threshold_decision'] = 'reject'
        signal.meta['threshold_rejection_reason'] = reason
        signal.meta['threshold_reason'] = reason
        logging.info(
            "SIGNAL FILTER | symbol=%s | original=%s | normalized=%s | confidence=%.4f | prob_profit=%.4f | p_dir=%.4f | decision=REJECT | reason=%s",
            signal.symbol,
            original_label,
            normalized_label,
            confidence,
            prob_profit,
            directional_prob,
            reason,
        )

    return evaluated
