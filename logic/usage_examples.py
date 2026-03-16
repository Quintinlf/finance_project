"""
USAGE EXAMPLES: Production Trading Engine

This file shows how to use the trading engine modules programmatically.
These examples go beyond the notebook integration to show customization options.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Example 1: Basic Trading Cycle (Simplest Usage)
# ═══════════════════════════════════════════════════════════════════════════

def example_1_basic_cycle():
    """Run a complete trading cycle with default settings."""
    from signal_engine import generate_signals
    from portfolio_state import get_position_states
    from execution_engine import run_trading_cycle
    from trade_log import log_decisions_batch
    from data_structures import ExecutionConfig
    from alpaca_exercises import connect_trading_client, get_account_summary
    
    # Configuration
    config = ExecutionConfig(
        execution_mode='paper',
        allow_short_selling=False,
        dry_run=True,  # ALWAYS start with dry_run!
        risk_per_trade_pct=0.02,
        take_profit_pct=0.04,
        stop_loss_pct=0.02,
        max_positions=5,
        min_confidence=0.65,
        min_prob_up=0.50
    )
    
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    # Connect to Alpaca
    trading_client = connect_trading_client(paper=True)
    account = get_account_summary(trading_client)
    
    # 1. Generate signals
    print("Generating signals...")
    signals = generate_signals(universe, config)
    
    # 2. Get current positions
    print("Loading portfolio state...")
    position_states = get_position_states(universe, config, trading_client)
    
    # 3. Run trading cycle
    print("Executing trading cycle...")
    decision_log = run_trading_cycle(
        signals=signals,
        position_states=position_states,
        config=config,
        account_cash=account['cash'],
        alpaca_client=trading_client
    )
    
    # 4. Log decisions
    print("Logging decisions...")
    log_decisions_batch(decision_log)
    
    print(f"✅ Complete! {len(decision_log)} decisions logged.")


# ═══════════════════════════════════════════════════════════════════════════
# Example 2: Simulation Mode with Portfolio Tracking
# ═══════════════════════════════════════════════════════════════════════════

def example_2_simulation_mode():
    """Run multiple days of simulated trading with portfolio tracking."""
    from signal_engine import generate_signals
    from portfolio_state import get_position_states
    from execution_engine import run_trading_cycle
    from data_structures import ExecutionConfig
    
    config = ExecutionConfig(
        execution_mode='simulation',  # Simulation mode
        allow_short_selling=False,
        dry_run=False,  # Execute in simulation
        risk_per_trade_pct=0.02,
        take_profit_pct=0.04,
        stop_loss_pct=0.02,
        max_positions=3,
        min_confidence=0.60,
        min_prob_up=0.50
    )
    
    universe = ["AAPL", "MSFT", "GOOGL"]
    
    # Initialize simulation portfolio
    sim_portfolio = {}
    initial_cash = 100000.0
    current_cash = initial_cash
    
    # Simulate 5 trading days
    for day in range(1, 6):
        print(f"\n{'='*60}")
        print(f"DAY {day}")
        print(f"{'='*60}")
        
        # Generate fresh signals each day
        signals = generate_signals(universe, config)
        
        # Get position states (from sim_portfolio)
        position_states = get_position_states(
            universe=universe,
            config=config,
            alpaca_client=None,  # No Alpaca in simulation
            sim_portfolio=sim_portfolio
        )
        
        # Run trading cycle
        decision_log = run_trading_cycle(
            signals=signals,
            position_states=position_states,
            config=config,
            account_cash=current_cash,
            alpaca_client=None,
            sim_portfolio=sim_portfolio  # Pass for updates
        )
        
        # Extract updated portfolio and cash
        for entry in decision_log:
            if entry.executed and 'updated_sim_portfolio' in entry.execution_result:
                sim_portfolio = entry.execution_result['updated_sim_portfolio']
                # Update cash (simplified - would need to track from execution)
        
        # Print day summary
        executed_count = sum(1 for e in decision_log if e.executed)
        print(f"Executed: {executed_count} trades")
        print(f"Open positions: {len([k for k,v in sim_portfolio.items() if v['quantity'] > 0])}")
        print(f"Cash: ${current_cash:,.2f}")
    
    # Calculate final portfolio value
    total_position_value = sum(
        pos['quantity'] * pos['avg_entry_price'] 
        for pos in sim_portfolio.values()
    )
    final_value = current_cash + total_position_value
    
    print(f"\n{'='*60}")
    print(f"SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Initial capital: ${initial_cash:,.2f}")
    print(f"Final value: ${final_value:,.2f}")
    print(f"Profit/Loss: ${final_value - initial_cash:,.2f}")
    print(f"Return: {(final_value / initial_cash - 1) * 100:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════
# Example 3: Custom Signal Processing with Filtering
# ═══════════════════════════════════════════════════════════════════════════

def example_3_custom_filtering():
    """Generate signals and apply custom filtering logic."""
    from signal_engine import generate_signals, filter_signals_by_thresholds
    from data_structures import ExecutionConfig
    
    config = ExecutionConfig(
        execution_mode='paper',
        allow_short_selling=False,
        dry_run=True,
        risk_per_trade_pct=0.02,
        take_profit_pct=0.04,
        stop_loss_pct=0.02,
        max_positions=5,
        min_confidence=0.50,  # Lower threshold initially
        min_prob_up=0.45
    )
    
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
    
    # Generate all signals (before filtering)
    print("Generating signals for all stocks...")
    all_signals = generate_signals(universe, config)
    
    print(f"Total signals: {len(all_signals)}")
    for sig in all_signals:
        print(f"  {sig.symbol}: {sig.signal_type} (conf={sig.confidence:.2f}, prob={sig.prob_profit:.2f})")
    
    # Apply basic filtering
    print("\nFiltering by min thresholds...")
    filtered = filter_signals_by_thresholds(
        all_signals,
        min_confidence=0.65,  # Higher than config
        min_prob_up=0.55
    )
    
    print(f"After filtering: {len(filtered)} signals")
    for sig in filtered:
        print(f"  {sig.symbol}: {sig.signal_type} (conf={sig.confidence:.2f}, prob={sig.prob_profit:.2f})")
    
    # Custom filtering: only BUY signals
    buy_signals = [s for s in filtered if s.signal_type == 'buy']
    print(f"\nBUY signals only: {len(buy_signals)}")
    
    # Custom filtering: sort by confidence
    top_3 = sorted(buy_signals, key=lambda s: s.confidence, reverse=True)[:3]
    print(f"\nTop 3 by confidence:")
    for sig in top_3:
        print(f"  {sig.symbol}: conf={sig.confidence:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# Example 4: Manual Decision Logic (Step-by-Step)
# ═══════════════════════════════════════════════════════════════════════════

def example_4_manual_decisions():
    """
    Manually control each step of the trading cycle.
    Useful for debugging or custom workflows.
    """
    from signal_engine import generate_signals
    from portfolio_state import get_position_states
    from execution_engine import decide_action, build_order_plan, execute_order_plan
    from data_structures import ExecutionConfig
    from alpaca_exercises import connect_trading_client, get_account_summary
    
    config = ExecutionConfig(
        execution_mode='paper',
        allow_short_selling=False,
        dry_run=True,
        risk_per_trade_pct=0.02,
        take_profit_pct=0.04,
        stop_loss_pct=0.02,
        max_positions=5,
        min_confidence=0.65,
        min_prob_up=0.50
    )
    
    universe = ["AAPL"]
    
    trading_client = connect_trading_client(paper=True)
    account = get_account_summary(trading_client)
    
    # Step 1: Generate signal for one stock
    signals = generate_signals(universe, config)
    signal = signals[0]
    
    print(f"Signal for {signal.symbol}:")
    print(f"  Type: {signal.signal_type}")
    print(f"  Confidence: {signal.confidence:.2f}")
    print(f"  Prob Profit: {signal.prob_profit:.2f}")
    
    # Step 2: Get position state
    position_states = get_position_states(universe, config, trading_client)
    position_state = position_states[signal.symbol]
    
    print(f"\nPosition State:")
    print(f"  Side: {position_state.side}")
    print(f"  Quantity: {position_state.quantity}")
    
    # Step 3: Decide action
    action, reason = decide_action(signal, position_state, config)
    
    print(f"\nDecision:")
    print(f"  Action: {action}")
    print(f"  Reason: {reason}")
    
    # Step 4: Build order plan (if actionable)
    if action in ['buy', 'sell']:
        import yfinance as yf
        ticker_data = yf.Ticker(signal.symbol)
        current_price = ticker_data.history(period='1d')['Close'].iloc[-1]
        
        order_plan = build_order_plan(
            signal=signal,
            position_state=position_state,
            config=config,
            account_cash=account['cash'],
            current_price=current_price
        )
        
        print(f"\nOrder Plan:")
        print(f"  Symbol: {order_plan.symbol}")
        print(f"  Side: {order_plan.side}")
        print(f"  Quantity: {order_plan.quantity}")
        print(f"  Entry Type: {order_plan.entry_type}")
        print(f"  TP Price: ${order_plan.tp_price:.2f}" if order_plan.tp_price else "  TP Price: None")
        print(f"  SL Price: ${order_plan.sl_price:.2f}" if order_plan.sl_price else "  SL Price: None")
        
        # Step 5: Execute order plan
        result = execute_order_plan(
            order_plan=order_plan,
            config=config,
            alpaca_client=trading_client,
            sim_portfolio=None
        )
        
        print(f"\nExecution Result:")
        print(f"  Executed: {result.get('executed', False)}")
        print(f"  Reason: {result.get('reason', 'N/A')}")
        if 'broker_order_id' in result:
            print(f"  Order ID: {result['broker_order_id']}")
    else:
        print(f"\nNo order plan needed (action={action})")


# ═══════════════════════════════════════════════════════════════════════════
# Example 5: Analyzing Decision Logs
# ═══════════════════════════════════════════════════════════════════════════

def example_5_analyze_logs():
    """Analyze historical decision logs for insights."""
    from trade_log import load_decision_log, get_execution_summary
    import pandas as pd
    
    # Load all decisions
    decisions = load_decision_log()
    
    if not decisions:
        print("No decisions logged yet. Run some trading cycles first!")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(decisions)
    
    print("="*60)
    print("DECISION LOG ANALYSIS")
    print("="*60)
    
    # Summary statistics
    summary = get_execution_summary()
    print(f"\nTotal Decisions: {summary['total_decisions']}")
    print(f"Executed: {summary['total_executed']}")
    print(f"Rejected: {summary['total_rejected']}")
    print(f"Holds: {summary['total_holds']}")
    print(f"Execution Rate: {summary['execution_rate']:.1f}%")
    
    # Group by action
    print("\nDecisions by Action:")
    action_counts = df['action'].value_counts()
    for action, count in action_counts.items():
        print(f"  {action}: {count}")
    
    # Group by symbol
    print("\nMost Traded Symbols:")
    symbol_counts = df[df['executed'] == 'True']['symbol'].value_counts().head(5)
    for symbol, count in symbol_counts.items():
        print(f"  {symbol}: {count} trades")
    
    # Average confidence by action
    print("\nAverage Confidence by Action:")
    avg_conf = df.groupby('action')['confidence'].mean()
    for action, conf in avg_conf.items():
        print(f"  {action}: {conf:.2f}")
    
    # Rejection reasons
    print("\nRejection Reasons:")
    rejected = df[df['action'] == 'rejected']
    if len(rejected) > 0:
        reasons = rejected['reason'].value_counts()
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")
    else:
        print("  No rejections")


# ═══════════════════════════════════════════════════════════════════════════
# Example 6: Short Selling Enabled
# ═══════════════════════════════════════════════════════════════════════════

def example_6_short_selling():
    """
    Enable short selling to open short positions on SELL signals.
    USE WITH CAUTION - only for advanced strategies!
    """
    from signal_engine import generate_signals
    from portfolio_state import get_position_states
    from execution_engine import run_trading_cycle
    from data_structures import ExecutionConfig
    from alpaca_exercises import connect_trading_client, get_account_summary
    
    config = ExecutionConfig(
        execution_mode='paper',
        allow_short_selling=True,  # ⚠️ ENABLE SHORT SELLING ⚠️
        dry_run=True,  # Still use dry_run for testing!
        risk_per_trade_pct=0.02,
        take_profit_pct=0.04,
        stop_loss_pct=0.02,
        max_positions=5,
        min_confidence=0.65,
        min_prob_up=0.50
    )
    
    universe = ["AAPL", "MSFT", "GOOGL"]
    
    trading_client = connect_trading_client(paper=True)
    account = get_account_summary(trading_client)
    
    print("⚠️  SHORT SELLING ENABLED ⚠️")
    print("SELL signals on flat positions will open SHORT positions\n")
    
    # Generate signals
    signals = generate_signals(universe, config)
    
    # Get positions
    position_states = get_position_states(universe, config, trading_client)
    
    # Run cycle
    decision_log = run_trading_cycle(
        signals=signals,
        position_states=position_states,
        config=config,
        account_cash=account['cash'],
        alpaca_client=trading_client
    )
    
    # Show decisions
    print("Decisions:")
    for entry in decision_log:
        print(f"  {entry.symbol}: {entry.action} ({entry.reason})")
        
    # Count shorts
    short_positions = [
        e for e in decision_log 
        if e.action == 'sell' and e.position_state_before['side'] == 'flat'
    ]
    
    print(f"\nNew SHORT positions opened: {len(short_positions)}")


# ═══════════════════════════════════════════════════════════════════════════
# Example 7: Integration with Existing analyze_symbol_core (Hybrid Approach)
# ═══════════════════════════════════════════════════════════════════════════

def example_7_hybrid_with_legacy():
    """
    Use new signal engine but integrate with existing analyze_symbol_core
    for backward compatibility during transition.
    """
    from signal_engine import generate_signals
    from data_structures import ExecutionConfig
    # from trading_functions import analyze_symbol_core  # Your existing function
    
    config = ExecutionConfig(
        execution_mode='paper',
        allow_short_selling=False,
        dry_run=True,
        risk_per_trade_pct=0.02,
        take_profit_pct=0.04,
        stop_loss_pct=0.02,
        max_positions=5,
        min_confidence=0.65,
        min_prob_up=0.50
    )
    
    symbol = "AAPL"
    
    # New way: use signal engine
    signals = generate_signals([symbol], config)
    new_signal = signals[0]
    
    print("New Engine Signal:")
    print(f"  Type: {new_signal.signal_type}")
    print(f"  Confidence: {new_signal.confidence:.2f}")
    
    # Old way: use analyze_symbol_core
    # old_signal = analyze_symbol_core(symbol, paper=True)
    # 
    # print("\nLegacy Engine Signal:")
    # print(f"  Type: {old_signal['signal']}")
    # print(f"  Confidence: {old_signal['combined_conf']:.2f}")
    
    # You can compare or choose which to use
    # During transition, you might run both and compare results


# ═══════════════════════════════════════════════════════════════════════════
# Main: Run Examples
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("PRODUCTION TRADING ENGINE - USAGE EXAMPLES")
    print("="*60)
    print("\nAvailable examples:")
    print("1. Basic trading cycle")
    print("2. Simulation mode with portfolio tracking")
    print("3. Custom signal filtering")
    print("4. Manual step-by-step decisions")
    print("5. Analyze decision logs")
    print("6. Short selling enabled")
    print("7. Hybrid with legacy analyze_symbol_core")
    print()
    
    choice = input("Enter example number to run (1-7): ").strip()
    
    if choice == "1":
        example_1_basic_cycle()
    elif choice == "2":
        example_2_simulation_mode()
    elif choice == "3":
        example_3_custom_filtering()
    elif choice == "4":
        example_4_manual_decisions()
    elif choice == "5":
        example_5_analyze_logs()
    elif choice == "6":
        example_6_short_selling()
    elif choice == "7":
        example_7_hybrid_with_legacy()
    else:
        print("Invalid choice. Run this script again and enter 1-7.")
