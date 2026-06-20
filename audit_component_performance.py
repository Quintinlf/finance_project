"""
Model Component Performance Audit
==================================

Analyzes the predictive performance of each ensemble component:
- Bayesian forecast
- GP forecast
- RSI signal
- Bollinger Bands signal

Computes win rates, errors, and correlation with realized returns.
"""

import sqlite3
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

DB_PATH = Path("trade_logs/trading.db")

def get_decision_records():
    """Fetch all decision records with timestamps."""
    if not DB_PATH.exists():
        print("ERROR: Database not found at " + str(DB_PATH))
        return []
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    try:
        rows = conn.execute("""
            SELECT 
                id,
                symbol, 
                signal_type,
                confidence,
                prob_profit,
                timestamp,
                action,
                executed,
                planned_entry_price
            FROM decisions
            WHERE signal_type IN ('buy', 'sell')
            ORDER BY timestamp ASC
        """).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_realized_return(symbol, entry_date_str, days_ahead=1):
    """
    Fetch realized return from entry date to days_ahead days later.
    
    Returns: (realized_pct_return, success)
    """
    try:
        # Parse entry date
        entry_date = datetime.fromisoformat(entry_date_str).date()
        end_date = entry_date + timedelta(days=days_ahead+5)  # Fetch extra days
        
        # Fetch price data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=entry_date, end=end_date)
        
        if len(hist) < 2:
            return None, False
        
        entry_price = hist['Close'].iloc[0]
        exit_price = hist['Close'].iloc[-1]
        
        pct_return = (exit_price - entry_price) / entry_price
        return pct_return, True
    except Exception as e:
        return None, False


def estimate_component_signals(symbol):
    """
    Re-run models to estimate component signals for analysis.
    
    This is a simplified version - in production, you'd store these values.
    """
    try:
        # Fetch price history
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo", interval="1d")
        
        if len(hist) < 30:
            return None
        
        # Calculate returns
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        
        # Simple Bayesian signal: if recent return is positive
        bayesian_signal = 1.0 if returns.iloc[-1] > 0 else -1.0
        bayesian_forecast = returns.iloc[-5:].mean() * 10  # Scaled
        
        # Simple GP signal: trend following
        gp_signal = 1.0 if returns.iloc[-5:].mean() > 0 else -1.0
        gp_forecast = returns.iloc[-5:].mean() * 8
        
        # RSI signal
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        rsi_signal = 1.0 if current_rsi < 30 else (-1.0 if current_rsi > 70 else 0.0)
        
        # Bollinger Bands signal
        sma = hist['Close'].rolling(window=20).mean()
        std = hist['Close'].rolling(window=20).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        bb_z_score = (hist['Close'].iloc[-1] - sma.iloc[-1]) / (std.iloc[-1] + 1e-8)
        bb_signal = 1.0 if bb_z_score < -1.5 else (-1.0 if bb_z_score > 1.5 else 0.0)
        
        return {
            'bayesian_forecast': bayesian_forecast,
            'bayesian_signal': bayesian_signal,
            'gp_forecast': gp_forecast,
            'gp_signal': gp_signal,
            'rsi_signal': rsi_signal,
            'rsi_value': current_rsi,
            'bb_signal': bb_signal,
            'bb_z_score': bb_z_score,
        }
    except Exception as e:
        return None


def analyze_component_performance():
    """Main audit function."""
    decisions = get_decision_records()
    
    if not decisions:
        print("ERROR: No decision records found")
        return
    
    print("\n" + "="*80)
    print("MODEL COMPONENT PERFORMANCE AUDIT")
    print("="*80)
    print("\nAnalyzing " + str(len(decisions)) + " trading signals\n")
    
    # Component results storage
    component_performance = {
        'bayesian': {'hits': 0, 'total': 0, 'errors': [], 'returns': []},
        'gp': {'hits': 0, 'total': 0, 'errors': [], 'returns': []},
        'rsi': {'hits': 0, 'total': 0, 'errors': [], 'returns': []},
        'bb': {'hits': 0, 'total': 0, 'errors': [], 'returns': []},
        'ensemble': {'hits': 0, 'total': 0, 'errors': [], 'returns': []},
    }
    
    analyzed_count = 0
    
    for decision in decisions:
        symbol = decision['symbol']
        signal_type = decision['signal_type']  # 'buy' or 'sell'
        entry_timestamp = decision['timestamp']
        confidence = decision['confidence']
        
        # Get realized return
        realized_return, success = get_realized_return(symbol, entry_timestamp, days_ahead=1)
        
        if not success or realized_return is None:
            continue
        
        # Estimate component signals
        components = estimate_component_signals(symbol)
        if not components:
            continue
        
        analyzed_count += 1
        
        # Expected direction for correctness check
        expected_direction = 1.0 if signal_type == 'buy' else -1.0
        realized_direction = 1.0 if realized_return > 0 else (-1.0 if realized_return < 0 else 0.0)
        
        # Bayesian component
        bay_forecast = components['bayesian_forecast']
        bay_predicted = 1.0 if bay_forecast > 0 else (-1.0 if bay_forecast < 0 else 0.0)
        if bay_predicted != 0:
            component_performance['bayesian']['total'] += 1
            if bay_predicted == realized_direction:
                component_performance['bayesian']['hits'] += 1
            component_performance['bayesian']['errors'].append(abs(bay_forecast - realized_return))
            component_performance['bayesian']['returns'].append(realized_return)
        
        # GP component
        gp_forecast = components['gp_forecast']
        gp_predicted = 1.0 if gp_forecast > 0 else (-1.0 if gp_forecast < 0 else 0.0)
        if gp_predicted != 0:
            component_performance['gp']['total'] += 1
            if gp_predicted == realized_direction:
                component_performance['gp']['hits'] += 1
            component_performance['gp']['errors'].append(abs(gp_forecast - realized_return))
            component_performance['gp']['returns'].append(realized_return)
        
        # RSI component
        rsi_signal = components['rsi_signal']
        if rsi_signal != 0:
            component_performance['rsi']['total'] += 1
            if rsi_signal == realized_direction:
                component_performance['rsi']['hits'] += 1
            component_performance['rsi']['returns'].append(realized_return)
        
        # BB component
        bb_signal = components['bb_signal']
        if bb_signal != 0:
            component_performance['bb']['total'] += 1
            if bb_signal == realized_direction:
                component_performance['bb']['hits'] += 1
            component_performance['bb']['returns'].append(realized_return)
        
        # Ensemble (confidence > 0.5 = predicted positive)
        ensemble_predicted = 1.0 if confidence > 0.5 else -1.0
        component_performance['ensemble']['total'] += 1
        if ensemble_predicted == realized_direction:
            component_performance['ensemble']['hits'] += 1
        component_performance['ensemble']['errors'].append(abs(confidence - (realized_return + 1.0)))
        component_performance['ensemble']['returns'].append(realized_return)
    
    print("\nAnalyzed " + str(analyzed_count) + " signals with available price data\n")
    
    if analyzed_count == 0:
        print("ERROR: Could not retrieve price data for analysis")
        return
    
    # Print results
    print("="*80)
    print("COMPONENT PERFORMANCE METRICS")
    print("="*80)
    
    results_summary = {}
    
    for component, data in component_performance.items():
        if data['total'] == 0:
            continue
        
        win_rate = 100.0 * data['hits'] / data['total'] if data['total'] > 0 else 0
        
        mae = np.mean(data['errors']) if data['errors'] else 0
        rmse = np.sqrt(np.mean([e**2 for e in data['errors']])) if data['errors'] else 0
        
        returns = np.array(data['returns'])
        avg_return = np.mean(returns) if len(returns) > 0 else 0
        std_return = np.std(returns) if len(returns) > 0 else 1e-8
        sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        
        results_summary[component] = {
            'win_rate': win_rate,
            'mae': mae,
            'rmse': rmse,
            'avg_return': avg_return,
            'sharpe': sharpe,
            'sample_size': data['total'],
            'returns': returns
        }
        
        print("\n[" + component.upper() + "]")
        print("  Win Rate:          {:.1f}% ({}/{})".format(win_rate, data['hits'], data['total']))
        print("  Mean Abs Error:    {:.4f}".format(mae))
        print("  RMSE:              {:.4f}".format(rmse))
        print("  Avg Realized Ret:  {:.4f} ({:.2f}%)".format(avg_return, avg_return*100))
        print("  Sharpe Ratio:      {:.2f}".format(sharpe))
    
    # Ranking
    print("\n" + "="*80)
    print("COMPONENT RANKING")
    print("="*80)
    
    print("\nBy Win Rate:")
    ranked = sorted(results_summary.items(), key=lambda x: x[1]['win_rate'], reverse=True)
    for i, (component, metrics) in enumerate(ranked, 1):
        print("  {}. {} - {:.1f}%".format(i, component.upper(), metrics['win_rate']))
    
    print("\nBy Sharpe Ratio:")
    ranked = sorted(results_summary.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    for i, (component, metrics) in enumerate(ranked, 1):
        print("  {}. {} - {:.2f}".format(i, component.upper(), metrics['sharpe']))
    
    # Estimate optimal weights
    print("\n" + "="*80)
    print("OPTIMAL ENSEMBLE WEIGHTS (Estimated)")
    print("="*80)
    
    if len(results_summary) >= 2:
        # Weight by Sharpe ratio
        sharpe_scores = {k: max(0, v['sharpe']) for k, v in results_summary.items()}
        total_sharpe = sum(sharpe_scores.values())
        
        if total_sharpe > 0:
            print("\nBased on Sharpe Ratio weighting:")
            for component, score in sorted(sharpe_scores.items(), key=lambda x: x[1], reverse=True):
                weight = score / total_sharpe
                print("  {}: {:.2f}%".format(component.upper(), weight * 100))
    
    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    analyze_component_performance()
