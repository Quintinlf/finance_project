"""
Trading Assistant Module

Provides automated performance analysis and actionable trading recommendations.
The assistant analyzes your trading history and tells you exactly what to do next.

Features:
- Performance metric calculation (win rate, profit factor, Sharpe ratio, etc.)
- Automated recommendation generation with priority levels
- Explicit action items based on strategy performance
- Portfolio health checks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def trading_assistant(
    trade_history: List[Dict],
    current_positions: List[Dict],
    account_summary: Dict
) -> List[Dict]:
    """
    Analyzes trading performance and provides prioritized recommendations.
    
    Args:
        trade_history: List of dicts with 'confidence' and 'profit' keys
        current_positions: List of dicts with 'symbol', 'qty', 'entry' keys
        account_summary: Dict with 'cash', 'portfolio_value' keys
    
    Returns:
        List of recommendation dicts with 'priority', 'action', 'reason', 'details'
    """
    recommendations = []
    
    if not trade_history or len(trade_history) < 10:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'RUN_ONCE',
            'reason': 'Not enough trading history. Execute more trades to build data.',
            'details': f'Only {len(trade_history)} trades recorded. Need 20+ for reliable analysis.'
        })
        return recommendations
    
    # Convert to DataFrame
    df = pd.DataFrame(trade_history)
    
    # Calculate metrics
    total_trades = len(df)
    wins = (df['profit'] > 0).sum()
    losses = (df['profit'] < 0).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    total_profit = df[df['profit'] > 0]['profit'].sum()
    total_loss = abs(df[df['profit'] < 0]['profit'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    avg_win = df[df['profit'] > 0]['profit'].mean() if wins > 0 else 0
    avg_loss = abs(df[df['profit'] < 0]['profit'].mean()) if losses > 0 else 0
    
    # Calculate drawdown
    cumulative = (df['profit'].cumsum() + account_summary.get('cash', 100))
    running_max = cumulative.expanding().max()
    drawdown = ((cumulative - running_max) / running_max * 100).min()
    
    # Expected value per trade
    ev_per_trade = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # Sharpe ratio (simplified)
    returns = df['profit'] / account_summary.get('portfolio_value', 100)
    sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
    
    print("="*70)
    print("🤖 TRADING ASSISTANT ANALYSIS")
    print("="*70)
    print(f"\n📊 Performance Metrics:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {win_rate:.1%} ({wins}W / {losses}L)")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   Avg Win: ${avg_win:.2f}")
    print(f"   Avg Loss: ${avg_loss:.2f}")
    print(f"   Expected Value/Trade: ${ev_per_trade:.2f}")
    print(f"   Max Drawdown: {drawdown:.1f}%")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    
    # DECISION LOGIC
    
    # 1. Check if strategy is losing money
    if ev_per_trade < 0 or profit_factor < 1.0:
        recommendations.append({
            'priority': 'CRITICAL',
            'action': 'TUNE_THRESHOLDS',
            'reason': 'Strategy is losing money on average.',
            'details': f'EV/trade = ${ev_per_trade:.2f}, Profit Factor = {profit_factor:.2f}. ' +
                       'Increase MIN_CONF to 0.75+ or widen TP/SL ratio.'
        })
    
    # 2. Check win rate
    if win_rate < 0.45:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'INCREASE_MIN_CONF',
            'reason': f'Win rate too low ({win_rate:.1%}).',
            'details': f'Current win rate {win_rate:.1%} < 45%. Raise MIN_CONF by 0.10 to filter weaker signals.'
        })
    elif win_rate > 0.70:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'DECREASE_MIN_CONF',
            'reason': f'Win rate very high ({win_rate:.1%}) - may be too conservative.',
            'details': f'You can afford to lower MIN_CONF by 0.05 to capture more opportunities.'
        })
    
    # 3. Check drawdown
    if drawdown < -15:
        recommendations.append({
            'priority': 'CRITICAL',
            'action': 'REDUCE_POSITION_SIZE',
            'reason': f'Excessive drawdown ({drawdown:.1f}%).',
            'details': 'Cut position sizes in half or tighten stop losses to preserve capital.'
        })
    
    # 4. Check if not trading enough
    if total_trades < 20:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'EXPAND_UNIVERSE',
            'reason': 'Low trade frequency.',
            'details': f'Only {total_trades} trades. Add more stocks to UNIVERSE or lower MIN_CONF slightly.'
        })
    
    # 5. Check concentration risk
    if current_positions:
        num_positions = len(current_positions)
        if num_positions > 10:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'REDUCE_MAX_ORDERS',
                'reason': f'Too many positions ({num_positions}).',
                'details': 'Lower MAX_ORDERS_PER_RUN to focus on best opportunities.'
            })
        elif num_positions < 3 and total_trades > 20:
            recommendations.append({
                'priority': 'LOW',
                'action': 'INCREASE_MAX_ORDERS',
                'reason': 'Under-diversified portfolio.',
                'details': f'Only {num_positions} positions. Consider MAX_ORDERS_PER_RUN = 5-8.'
            })
    
    # 6. Check if profitable but can optimize
    if ev_per_trade > 0 and 0.50 <= win_rate <= 0.65 and profit_factor > 1.3:
        recommendations.append({
            'priority': 'LOW',
            'action': 'RUN_MONTE_CARLO',
            'reason': 'Strategy looks profitable! Quantify expected returns.',
            'details': 'Run Monte Carlo simulation to estimate monthly P&L distribution.'
        })
    
    # 7. Suggest backtesting if never done
    if total_trades < 30:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'BACKTEST_STRATEGY',
            'reason': 'Need historical validation.',
            'details': 'Test strategy on 6-12 months of historical data before committing more capital.'
        })
    
    # 8. Good performance - stay the course
    if ev_per_trade > 1.0 and win_rate > 0.52 and profit_factor > 1.5 and drawdown > -10:
        recommendations.append({
            'priority': 'LOW',
            'action': 'CONTINUE_CURRENT_STRATEGY',
            'reason': '✅ Strategy performing well!',
            'details': f'Win rate {win_rate:.1%}, PF {profit_factor:.2f}, EV ${ev_per_trade:.2f}/trade. Keep executing.'
        })
    
    # Sort by priority
    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 99))
    
    # Print recommendations
    print(f"\n🎯 RECOMMENDED ACTIONS (in priority order):")
    print("="*70)
    
    for i, rec in enumerate(recommendations, 1):
        priority_color = {
            'CRITICAL': '🔴',
            'HIGH': '🟠',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }.get(rec['priority'], '⚪')
        
        print(f"\n{i}. {priority_color} [{rec['priority']}] {rec['action']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Details: {rec['details']}")
    
    print("\n" + "="*70)
    
    return recommendations


def format_assistant_summary(recommendations: List[Dict]) -> str:
    """
    Format recommendations as a concise summary string.
    
    Args:
        recommendations: List of recommendation dicts from trading_assistant()
    
    Returns:
        Formatted string with top 3 recommendations
    """
    if not recommendations:
        return "No recommendations available."
    
    summary_lines = ["📋 Top Recommendations:"]
    
    for i, rec in enumerate(recommendations[:3], 1):
        emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(rec['priority'], '⚪')
        summary_lines.append(f"{i}. {emoji} {rec['action']}: {rec['reason']}")
    
    return "\n".join(summary_lines)


def get_quick_health_check(
    trade_history: List[Dict],
    min_trades: int = 10
) -> Dict[str, any]:
    """
    Quick health check of trading strategy (no detailed recommendations).
    
    Args:
        trade_history: List of trades
        min_trades: Minimum trades needed for analysis
    
    Returns:
        Dict with health status: 'status', 'win_rate', 'profit_factor', 'message'
    """
    if not trade_history or len(trade_history) < min_trades:
        return {
            'status': 'INSUFFICIENT_DATA',
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'message': f'Need {min_trades - len(trade_history)} more trades for analysis.'
        }
    
    df = pd.DataFrame(trade_history)
    
    wins = (df['profit'] > 0).sum()
    total = len(df)
    win_rate = wins / total
    
    total_profit = df[df['profit'] > 0]['profit'].sum()
    total_loss = abs(df[df['profit'] < 0]['profit'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    # Determine status
    if profit_factor > 1.5 and win_rate > 0.52:
        status = 'HEALTHY'
        message = '✅ Strategy performing well!'
    elif profit_factor > 1.0 and win_rate > 0.45:
        status = 'ACCEPTABLE'
        message = '⚠️ Strategy marginally profitable. Monitor closely.'
    else:
        status = 'UNHEALTHY'
        message = '🔴 Strategy losing money. Tune parameters immediately.'
    
    return {
        'status': status,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': total,
        'message': message
    }
