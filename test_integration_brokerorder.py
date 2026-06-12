#!/usr/bin/env python
"""
Integration test: Simulate complete trading flow from signal to execution
This verifies the entire BrokerOrder fix chain works end-to-end
"""

from datetime import datetime
from logic.broker_client import create_broker_client
from logic.data_structures import (
    Signal, PositionState, ExecutionConfig, OrderPlan
)
from logic.execution_engine import execute_order_plan

def integration_test_paper_trading():
    """Full integration: signal → execution with BrokerOrder"""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Paper Trading Signal -> Execution (BrokerOrder Fix)")
    print("="*80)
    
    # 1. Create broker client
    print("\n[1] Creating paper broker client...")
    broker = create_broker_client(
        execution_mode='paper',
        initial_cash=100000.0
    )
    print(f"    Broker type: {type(broker).__name__}")
    
    # 2. Get account summary
    print("\n[2] Getting account summary...")
    account = broker.get_account_summary()
    print(f"    Cash: ${account.get('cash', 0):.2f}")
    print(f"    Portfolio Value: ${account.get('portfolio_value', 0):.2f}")
    
    # 3. Check positions
    print("\n[3] Checking positions...")
    positions = broker.get_position_states(['AAPL', 'MSFT', 'NVDA'])
    for symbol, pos in positions.items():
        print(f"    {symbol}: qty={pos.quantity}, side={pos.side}")
    
    # 4. Create simulated signals (as if from model)
    print("\n[4] Creating BUY signals...")
    signals = [
        Signal(
            symbol='AAPL',
            signal_type='buy',
            confidence=0.75,
            prob_profit=0.65,
            meta={'current_price': 195.50}
        ),
        Signal(
            symbol='MSFT',
            signal_type='buy',
            confidence=0.82,
            prob_profit=0.70,
            meta={'current_price': 420.25}
        ),
        Signal(
            symbol='NVDA',
            signal_type='buy',
            confidence=0.88,
            prob_profit=0.75,
            meta={'current_price': 145.75}
        ),
    ]
    print(f"    Generated {len(signals)} buy signals")
    
    # 5. Create execution config
    print("\n[5] Creating execution config...")
    config = ExecutionConfig(
        execution_mode='paper',
        dry_run=False,  # Actually execute
        tp_pct=0.04,    # 4% take profit
        sl_pct=0.02,    # 2% stop loss
        base_risk_pct=2.0  # 2% risk per trade
    )
    print(f"    Mode: {config.execution_mode}, TP: {config.tp_pct*100}%, SL: {config.sl_pct*100}%")
    
    # 6. Build order plans
    print("\n[6] Building order plans...")
    order_plans = []
    for signal in signals:
        pos = positions.get(signal.symbol, PositionState(
            symbol=signal.symbol, quantity=0, avg_entry_price=0.0,
            side='flat', source='paper'
        ))
        
        # Determine order quantity
        current_price = signal.meta.get('current_price', 100.0)
        account_equity = account.get('cash', 100000.0)
        qty = max(1, int((account_equity * config.base_risk_pct / 100) / (current_price * config.sl_pct)))
        
        plan = OrderPlan(
            symbol=signal.symbol,
            side='buy',
            quantity=qty,
            entry_type='market',
            tp_price=round(current_price * (1 + config.tp_pct), 2),
            sl_price=round(current_price * (1 - config.sl_pct), 2),
            time_in_force='day',
            reason=f"Signal: BUY, confidence={signal.confidence:.2f}"
        )
        order_plans.append(plan)
        print(f"    {signal.symbol}: BUY {qty} shares @ ${current_price:.2f} (TP: ${plan.tp_price}, SL: ${plan.sl_price})")
    
    # 7. Execute orders
    print("\n[7] Executing orders...")
    executed_count = 0
    failed_count = 0
    
    for i, plan in enumerate(order_plans, 1):
        print(f"\n    [{i}/3] Executing {plan.symbol} order...")
        
        try:
            result = execute_order_plan(
                order_plan=plan,
                config=config,
                broker_client=broker,
                verbose=False
            )
            
            if result['executed']:
                print(f"        [OK] EXECUTED - Order ID: {result['broker_order_id']}")
                executed_count += 1
            else:
                print(f"        [FAILED] {result['error_message']}")
                failed_count += 1
                
        except Exception as e:
            print(f"        [ERROR] {e}")
            failed_count += 1
    
    # 8. Summary
    print("\n" + "="*80)
    print("INTEGRATION TEST RESULTS")
    print("="*80)
    print(f"Orders Attempted:  {len(order_plans)}")
    print(f"Orders Executed:   {executed_count}")
    print(f"Orders Failed:     {failed_count}")
    
    if executed_count == len(order_plans):
        print("\n[SUCCESS] All orders executed without BrokerOrder subscripting errors!")
        print("          The trading flow is now operational.")
        return True
    else:
        print(f"\n[WARNING] Only {executed_count}/{len(order_plans)} orders executed")
        return False

if __name__ == "__main__":
    import sys
    try:
        success = integration_test_paper_trading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
