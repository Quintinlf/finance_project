"""
Open Position Evaluator

Evaluates current OPEN positions against their original forecasts/expectations.
This allows threshold tuning even BEFORE trades are closed.

Key insight: We don't need to wait for trades to close to learn!
We can evaluate how well our predictions matched reality for open positions.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from alpaca.trading.client import TradingClient


def evaluate_open_positions(
    trading_client: TradingClient,
    decision_log_path: str = "trade_logs/decisions.csv"
) -> List[Dict[str, Any]]:
    """
    Evaluate all open positions vs their original entry predictions.
    
    This compares:
    - Entry price vs current price (unrealized P&L)
    - Original confidence vs actual performance
    - Original prob_profit vs reality (is it moving in predicted direction?)
    - Time in trade vs expected holding period
    
    Args:
        trading_client: Alpaca trading client
        decision_log_path: Path to decision log CSV
    
    Returns:
        List of evaluation dicts, one per open position
    """
    from alpaca_exercises import get_positions
    from trade_log import load_decision_log
    
    # Get current open positions from Alpaca
    positions = get_positions(trading_client)
    
    if not positions:
        print("No open positions to evaluate")
        return []
    
    # Load decision history to find original entry signals
    decision_log = load_decision_log(decision_log_path)
    if not decision_log:
        print("Warning: No decision log found. Cannot compare to original predictions.")
        return []
    
    decision_df = pd.DataFrame(decision_log)
    
    evaluations = []
    
    for pos in positions:
        symbol = pos.symbol
        qty = float(pos.qty)
        entry_price = float(pos.avg_entry_price)
        current_price = float(pos.current_price)
        unrealized_pl = float(pos.unrealized_pl)
        unrealized_plpc = float(pos.unrealized_plpc)
        
        # Find original entry decision
        entry_decisions = decision_df[
            (decision_df['symbol'] == symbol) & 
            (decision_df['action'] == 'buy') & 
            (decision_df['executed'] == 'True')
        ]
        
        if entry_decisions.empty:
            # No logged entry found (might be from before logging system)
            original_confidence = None
            original_prob_profit = None
            entry_timestamp = None
        else:
            # Use most recent entry (in case of multiple buys)
            last_entry = entry_decisions.iloc[-1]
            original_confidence = float(last_entry.get('confidence', 0))
            original_prob_profit = float(last_entry.get('prob_profit', 0))
            entry_timestamp = pd.to_datetime(last_entry.get('timestamp'))
        
        # Calculate days in trade
        if entry_timestamp:
            days_in_trade = (datetime.now() - entry_timestamp).days
        else:
            days_in_trade = None
        
        # Direction correctness: did it move in predicted direction?
        price_change_pct = (current_price - entry_price) / entry_price
        direction_correct = price_change_pct > 0  # We predicted "buy" (up)
        
        # Performance score: how well did prediction match reality?
        # Higher confidence should correlate with better performance
        if original_confidence is not None:
            # Expected return based on confidence (rough heuristic)
            expected_return_pct = original_prob_profit * 0.05 if original_prob_profit else 0.02
            performance_score = price_change_pct / expected_return_pct if expected_return_pct > 0 else 0
        else:
            performance_score = None
        
        evaluation = {
            'symbol': symbol,
            'quantity': qty,
            'entry_price': entry_price,
            'current_price': current_price,
            'unrealized_pl': unrealized_pl,
            'unrealized_plpc': unrealized_plpc,
            'price_change_pct': price_change_pct,
            'direction_correct': direction_correct,
            'days_in_trade': days_in_trade,
            'original_confidence': original_confidence,
            'original_prob_profit': original_prob_profit,
            'performance_score': performance_score,
            'entry_timestamp': entry_timestamp.isoformat() if entry_timestamp else None
        }
        
        evaluations.append(evaluation)
    
    return evaluations


def print_position_evaluation_summary(evaluations: List[Dict[str, Any]]):
    """
    Print a formatted summary of position evaluations.
    """
    if not evaluations:
        print("No evaluations to display")
        return
    
    print("="*80)
    print("📊 OPEN POSITION EVALUATION")
    print("="*80)
    print()
    
    # Overall stats
    total_positions = len(evaluations)
    positions_with_data = [e for e in evaluations if e['original_confidence'] is not None]
    
    if positions_with_data:
        avg_unrealized_plpc = sum(e['unrealized_plpc'] for e in evaluations) / len(evaluations)
        correct_direction = sum(1 for e in positions_with_data if e['direction_correct'])
        direction_accuracy = correct_direction / len(positions_with_data) * 100
        avg_confidence = sum(e['original_confidence'] for e in positions_with_data) / len(positions_with_data)
        
        print(f"Total Open Positions: {total_positions}")
        print(f"Average Unrealized P&L: {avg_unrealized_plpc:.2%}")
        print(f"Direction Accuracy: {direction_accuracy:.1f}% ({correct_direction}/{len(positions_with_data)})")
        print(f"Average Entry Confidence: {avg_confidence:.2%}")
        print()
    else:
        print(f"Total Open Positions: {total_positions}")
        print("⚠️ No historical data found for these positions")
        print("   (Positions may have been opened before logging system)")
        print()
    
    # Individual positions
    print("Individual Position Details:")
    print("-"*80)
    
    for eval in evaluations:
        direction_emoji = "✅" if eval['direction_correct'] else "❌"
        pl_emoji = "🟢" if eval['unrealized_pl'] > 0 else "🔴"
        
        print(f"\n{eval['symbol']}: {direction_emoji} {pl_emoji}")
        print(f"  Entry: ${eval['entry_price']:.2f} → Current: ${eval['current_price']:.2f}")
        print(f"  Unrealized P&L: ${eval['unrealized_pl']:.2f} ({eval['unrealized_plpc']:.2%})")
        
        if eval['original_confidence']:
            print(f"  Entry Confidence: {eval['original_confidence']:.2%}")
            print(f"  Entry Prob Profit: {eval['original_prob_profit']:.2%}")
            if eval['performance_score']:
                print(f"  Performance Score: {eval['performance_score']:.2f}x expected")
        
        if eval['days_in_trade']:
            print(f"  Days in Trade: {eval['days_in_trade']}")
    
    print()
    print("="*80)


def convert_open_positions_to_trade_history_format(
    evaluations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert open position evaluations to trade_history format for threshold calculator.
    
    This allows us to use UNREALIZED performance to tune thresholds,
    rather than waiting for trades to close.
    
    Args:
        evaluations: Output from evaluate_open_positions()
    
    Returns:
        List in trade_history format (confidence, profit, etc.)
    """
    trade_history = []
    
    for eval in evaluations:
        # Skip positions without original prediction data
        if eval['original_confidence'] is None:
            continue
        
        # Use unrealized P&L as "profit" (even though position is open)
        # This is our best estimate of how the prediction is performing
        trade_record = {
            'symbol': eval['symbol'],
            'confidence': eval['original_confidence'],
            'prob_profit': eval['original_prob_profit'],
            'profit': eval['unrealized_pl'],  # Unrealized - not locked in yet
            'return_pct': eval['unrealized_plpc'],
            'entry_price': eval['entry_price'],
            'current_price': eval['current_price'],
            'status': 'open',  # Mark as open position
            'days_in_trade': eval['days_in_trade']
        }
        
        trade_history.append(trade_record)
    
    return trade_history


# ========================================================================
# ACCOUNT BALANCE EXPLAINER
# ========================================================================

def explain_account_balance(trading_client: TradingClient):
    """
    Explain Alpaca account balance, especially negative cash situations.
    
    Negative cash is NORMAL when you have open positions because:
    - Cash = Starting cash - All purchases
    - Portfolio Value = Cash + Position market values
    - Buying Power considers margin/equity
    """
    from alpaca_exercises import get_account_summary, get_positions
    
    account = get_account_summary(trading_client)
    positions = get_positions(trading_client)
    
    print("="*80)
    print("💰 ACCOUNT BALANCE EXPLANATION")
    print("="*80)
    print()
    
    cash = account['cash']
    portfolio_value = account['portfolio_value']
    buying_power = account['buying_power']
    
    # Calculate position values
    if positions:
        total_position_cost = sum(float(p.qty) * float(p.avg_entry_price) for p in positions)
        total_position_market_value = sum(float(p.qty) * float(p.current_price) for p in positions)
        total_unrealized_pl = sum(float(p.unrealized_pl) for p in positions)
    else:
        total_position_cost = 0
        total_position_market_value = 0
        total_unrealized_pl = 0
    
    print("📊 Account Breakdown:")
    print(f"   Cash: ${cash:,.2f}")
    print(f"   Position Market Value: ${total_position_market_value:,.2f}")
    print(f"   Portfolio Value (Total Equity): ${portfolio_value:,.2f}")
    print(f"   Buying Power: ${buying_power:,.2f}")
    print()
    
    if cash < 0:
        print("⚠️  NEGATIVE CASH - Why is this happening?")
        print()
        print("This is NORMAL with open positions! Here's why:")
        print()
        print("1. You started with some initial cash (e.g., $100,000)")
        print(f"2. You bought ${total_position_cost:,.2f} worth of stocks")
        print(f"3. Cash = Initial - Purchases = ${cash:,.2f} (negative!)")
        print()
        print("But your PORTFOLIO VALUE is what matters:")
        print(f"   Portfolio = Cash + Position Values")
        print(f"   Portfolio = ${cash:,.2f} + ${total_position_market_value:,.2f}")
        print(f"   Portfolio = ${portfolio_value:,.2f}")
        print()
        print("You have:")
        print(f"   • {len(positions)} open positions")
        print(f"   • ${total_position_market_value:,.2f} in stocks")
        print(f"   • ${total_unrealized_pl:,.2f} unrealized P&L")
        print()
        
        if total_unrealized_pl > 0:
            print(f"🟢 Your positions are UP ${total_unrealized_pl:,.2f}")
            print(f"   When you sell, your cash will increase by the gain!")
        elif total_unrealized_pl < 0:
            print(f"🔴 Your positions are DOWN ${total_unrealized_pl:,.2f}")
            print(f"   You haven't lost until you sell (paper loss)")
        
        print()
        print("💡 To increase cash:")
        print("   1. Sell some positions (close winning trades)")
        print("   2. Wait for bracket orders to hit TP (take profit)")
        print("   3. Close losing positions to free up capital")
        
    else:
        print("✅ Positive cash balance")
        print(f"   You have ${cash:,.2f} available to trade")
        print(f"   You can buy ~${buying_power:,.2f} worth of stocks (with margin)")
    
    print()
    print("="*80)
