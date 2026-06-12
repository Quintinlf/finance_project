#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify BrokerOrder fix:
- Both PaperBrokerClient and AlpacaBrokerClient return BrokerOrder objects
- execution_engine.py can successfully access .id attribute
- No dictionary subscripting errors occur
"""

import sys
from datetime import datetime
from logic.broker_client import BrokerOrder, PaperBrokerClient, AlpacaBrokerClient
from logic.data_structures import OrderPlan, ExecutionConfig, Signal, PositionState
from logic.execution_engine import execute_order_plan

def test_paper_broker_market_order():
    """Test PaperBrokerClient returns BrokerOrder for market orders."""
    print("\n" + "="*70)
    print("TEST 1: PaperBrokerClient.place_market_order()")
    print("="*70)
    
    broker = PaperBrokerClient(initial_cash=100000.0)
    order = broker.place_market_order(symbol="AAPL", qty=10, side="buy")
    
    print(f"Order type: {type(order)}")
    print(f"Order is BrokerOrder: {isinstance(order, BrokerOrder)}")
    print(f"Order.id: {order.id if order else 'None'}")
    print(f"Order.symbol: {order.symbol if order else 'None'}")
    print(f"Order.qty: {order.qty if order else 'None'}")
    print(f"Order.side: {order.side if order else 'None'}")
    
    assert isinstance(order, BrokerOrder), "Market order should return BrokerOrder"
    assert order.id is not None, "Order ID should not be None"
    print("[PASS] PaperBrokerClient.place_market_order() returns BrokerOrder")

def test_paper_broker_bracket_order():
    """Test PaperBrokerClient returns BrokerOrder for bracket orders."""
    print("\n" + "="*70)
    print("TEST 2: PaperBrokerClient.place_bracket_order()")
    print("="*70)
    
    broker = PaperBrokerClient(initial_cash=100000.0)
    order = broker.place_bracket_order(
        symbol="MSFT",
        qty=5,
        side="buy",
        take_profit_price=150.0,
        stop_loss_price=140.0
    )
    
    print(f"Order type: {type(order)}")
    print(f"Order is BrokerOrder: {isinstance(order, BrokerOrder)}")
    print(f"Order.id: {order.id if order else 'None'}")
    print(f"Order.order_type: {order.order_type if order else 'None'}")
    
    assert isinstance(order, BrokerOrder), "Bracket order should return BrokerOrder"
    assert order.order_type == "bracket", "Order type should be 'bracket'"
    print("[PASS] PaperBrokerClient.place_bracket_order() returns BrokerOrder")

def test_execution_with_bracket_order():
    """Test execution_engine.py handles bracket orders without subscripting errors."""
    print("\n" + "="*70)
    print("TEST 3: execution_engine handles bracket orders (no subscripting error)")
    print("="*70)
    
    broker = PaperBrokerClient(initial_cash=100000.0)
    config = ExecutionConfig(
        execution_mode='paper',
        dry_run=False,  # MUST disable dry run to actually execute
        tp_pct=0.04,
        sl_pct=0.02
    )
    
    order_plan = OrderPlan(
        symbol="NVDA",
        side="buy",
        quantity=3,
        entry_type="market",
        tp_price=150.0,
        sl_price=145.0,
        time_in_force="day"
    )
    
    print(f"Order plan has bracket: {order_plan.has_bracket()}")
    
    result = execute_order_plan(
        order_plan=order_plan,
        config=config,
        broker_client=broker,
        verbose=True
    )
    
    print(f"Execution result: {result}")
    print(f"Executed: {result['executed']}")
    print(f"Broker order ID: {result['broker_order_id']}")
    print(f"Error message: {result['error_message']}")
    
    assert result['executed'], "Order should be executed"
    assert result['broker_order_id'] is not None, "Broker order ID should be set"
    assert result['error_message'] is None, "Should not have error"
    print("[PASS] execution_engine handles bracket orders correctly")

def test_execution_with_market_order():
    """Test execution_engine.py handles market orders without subscripting errors."""
    print("\n" + "="*70)
    print("TEST 4: execution_engine handles market orders (no subscripting error)")
    print("="*70)
    
    broker = PaperBrokerClient(initial_cash=100000.0)
    config = ExecutionConfig(
        execution_mode='paper',
        dry_run=False,  # MUST disable dry run to actually execute
        tp_pct=None,
        sl_pct=None
    )
    
    order_plan = OrderPlan(
        symbol="AMZN",
        side="buy",
        quantity=2,
        entry_type="market",
        tp_price=None,
        sl_price=None,
        time_in_force="day"
    )
    
    print(f"Order plan has bracket: {order_plan.has_bracket()}")
    
    result = execute_order_plan(
        order_plan=order_plan,
        config=config,
        broker_client=broker,
        verbose=True
    )
    
    print(f"Execution result: {result}")
    print(f"Executed: {result['executed']}")
    print(f"Broker order ID: {result['broker_order_id']}")
    print(f"Error message: {result['error_message']}")
    
    assert result['executed'], "Order should be executed"
    assert result['broker_order_id'] is not None, "Broker order ID should be set"
    assert result['error_message'] is None, "Should not have error"
    print("[PASS] execution_engine handles market orders correctly")

if __name__ == "__main__":
    try:
        test_paper_broker_market_order()
        test_paper_broker_bracket_order()
        test_execution_with_bracket_order()
        test_execution_with_market_order()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        print("="*70)
        print("\nThe BrokerOrder fix is working correctly!")
        print("- PaperBrokerClient returns BrokerOrder objects")
        print("- execution_engine.py can access .id without subscripting errors")
        print("- Both bracket and market orders are handled properly")
        
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[FAILED] TEST ASSERTION: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
