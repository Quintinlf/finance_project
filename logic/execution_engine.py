"""
Execution Engine

Core decision logic and trade execution routing.
This is the heart of the trading system - where signals meet reality.

Key components:
1. decide_action: explicit logic gates (BUY/SELL/HOLD vs owns/doesn't_own)
2. build_order_plan: converts decision to concrete order with sizing/TP/SL  
3. execute_order_plan: routes to simulation, paper, or live execution
4. run_trading_cycle: orchestrates the full decision → execution flow
"""

from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import uuid

from data_structures import (
    Signal, PositionState, ExecutionConfig, OrderPlan, DecisionLogEntry
)
from portfolio_state import update_sim_portfolio_after_trade
from risk_management import calculate_position_size
from alpaca.trading.enums import TimeInForce
from alpaca.trading.client import TradingClient
from alpaca_exercises import place_market_order, place_bracket_order

from sqlite_store import (
    DEFAULT_DB_PATH,
    init_db,
    create_trade_attempt,
    set_trade_broker_order_id,
    mark_trade_open,
    mark_trade_failed,
)


# ========================================================================
# STEP 1: DECISION LOGIC (Logic Gates)
# ========================================================================

def decide_action(
    signal: Signal,
    position_state: PositionState,
    config: ExecutionConfig
) -> Tuple[str, str]:
    """
    Core decision logic: determine what action to take.
    
    Args:
        signal: trading signal from model
        position_state: current position in this asset
        config: execution configuration
    
    Returns:
        (action, reason) tuple where:
            action: 'buy', 'sell', 'hold', or 'rejected'
            reason: human-readable explanation
    
    Logic Gates (EXPLICIT):
        SIGNAL = BUY:
            - If we don't own → action=BUY (open long)
            - If we own (long) → action=HOLD (already in position)
            - If we're short and shorting disabled → action=REJECTED
        
        SIGNAL = SELL:
            - If we own (long) → action=SELL (close position)
            - If we don't own and shorting disabled → action=REJECTED (can't sell what we don't own)
            - If we don't own and shorting enabled → action=SELL (open short)
        
        SIGNAL = HOLD:
            - action=HOLD (do nothing)
    """
    signal_type = signal.signal_type
    current_side = position_state.side
    current_qty = position_state.quantity
    
    # ---- CASE: BUY SIGNAL ----
    if signal_type == 'buy':
        if current_side == 'flat':
            # We don't own shares → allowed to buy
            action = 'buy'
            reason = f"Signal=BUY, position=flat → opening long position"
            
        elif current_side == 'long':
            # We already own shares → hold (optional: could add to position later)
            action = 'hold'
            reason = f"Signal=BUY but already long {current_qty} shares → hold"
            
        elif current_side == 'short':
            # We're short → complicated (would need to close short first, then buy)
            # For now, reject unless you explicitly handle this
            action = 'rejected'
            reason = f"Signal=BUY but currently short {current_qty} shares → rejected (short covering not implemented)"
        
        else:
            action = 'rejected'
            reason = f"Signal=BUY with unknown position side '{current_side}'"
    
    # ---- CASE: SELL SIGNAL ----
    elif signal_type == 'sell':
        if current_side == 'long' and current_qty > 0:
            # We own shares → allowed to sell
            action = 'sell'
            reason = f"Signal=SELL, position=long ({current_qty} shares) → closing position"
            
        elif current_side == 'flat' and not config.allow_short_selling:
            # We don't own and shorting disabled → cannot sell
            action = 'rejected'
            reason = f"Signal=SELL but position=flat and short selling disabled"
            
        elif current_side == 'flat' and config.allow_short_selling:
            # We don't own but shorting enabled → open short
            action = 'sell'
            reason = f"Signal=SELL, position=flat, shorting enabled → opening short position"
            
        elif current_side == 'short':
            # Already short → hold (optional: could add to short later)
            action = 'hold'
            reason = f"Signal=SELL but already short {current_qty} shares → hold"
        
        else:
            action = 'rejected'
            reason = f"Signal=SELL with unknown position side '{current_side}'"
    
    # ---- CASE: HOLD SIGNAL ----
    elif signal_type == 'hold':
        action = 'hold'
        reason = f"Signal=HOLD → no action"
    
    else:
        action = 'rejected'
        reason = f"Unknown signal type '{signal_type}'"
    
    return action, reason


# ========================================================================
# STEP 2: ORDER PLANNING (Sizing + TP/SL)
# ========================================================================

def build_order_plan(
    signal: Signal,
    position_state: PositionState,
    config: ExecutionConfig,
    account_cash: float,
    current_price: float
) -> OrderPlan:
    """
    Build a concrete order plan with sizing and TP/SL levels.
    
    Args:
        signal: trading signal
        position_state: current position
        config: execution config (TP/SL percentages, risk settings)
        account_cash: available cash
        current_price: current market price
    
    Returns:
        OrderPlan with all order details
    
    Process:
        1. Calculate position size using risk management
        2. For SELL: cap at owned quantity
        3. Calculate TP/SL prices based on config
        4. Determine if bracket order (TP + SL) or market only
    """
    symbol = signal.symbol
    
    # Determine side (buy or sell)
    if signal.signal_type == 'buy':
        side = 'buy'
    elif signal.signal_type == 'sell':
        side = 'sell'
    else:
        raise ValueError(f"Cannot build order plan for signal type '{signal.signal_type}'")
    
    # Calculate position size
    if side == 'buy':
        # Calculate shares to buy based on risk
        quantity = calculate_position_size(
            account_balance=account_cash,
            risk_per_trade_pct=config.base_risk_pct,
            stop_loss_pct=config.sl_pct * 100,  # Function expects percentage as 0-100
            price=current_price
        )
        
        # Cap at max position size
        max_shares = int((account_cash * config.max_position_pct_of_equity / 100) / current_price)
        quantity = min(quantity, max_shares)
        quantity = max(1, quantity)  # At least 1 share
        
        # Calculate TP/SL for long position
        tp_price = round(current_price * (1 + config.tp_pct), 2)
        sl_price = round(current_price * (1 - config.sl_pct), 2)
        
    else:  # sell
        if position_state.side == 'long':
            # Closing long position: sell what we own
            quantity = int(abs(position_state.quantity))
            
            # TP/SL for closing don't apply in same way
            # (we're exiting, not entering a new position)
            tp_price = None
            sl_price = None
            
        else:
            # Opening short position (if shorting enabled)
            quantity = calculate_position_size(
                account_balance=account_cash,
                risk_per_trade_pct=config.base_risk_pct,
                stop_loss_pct=config.sl_pct * 100,
                price=current_price
            )
            max_shares = int((account_cash * config.max_position_pct_of_equity / 100) / current_price)
            quantity = min(quantity, max_shares)
            quantity = max(1, quantity)
            
            # For shorts, TP is below entry, SL is above
            tp_price = round(current_price * (1 - config.tp_pct), 2)
            sl_price = round(current_price * (1 + config.sl_pct), 2)
    
    # Build OrderPlan
    plan = OrderPlan(
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_type='market',
        limit_price=None,
        tp_price=tp_price,
        sl_price=sl_price,
        time_in_force='day',
        reason=f"Signal: {signal.signal_type.upper()}, confidence={signal.confidence:.2f}"
    )
    
    return plan


# ========================================================================
# STEP 3: EXECUTION ROUTING (Simulation vs Alpaca)
# ========================================================================

def execute_order_plan(
    order_plan: OrderPlan,
    config: ExecutionConfig,
    alpaca_client=None,
    sim_portfolio: Optional[Dict[str, PositionState]] = None,
    verbose: bool = True
) -> Dict:
    """
    Execute an order plan by routing to the appropriate executor.
    
    Args:
        order_plan: the order to execute
        config: execution configuration
        alpaca_client: Alpaca trading client (required for paper/live)
        sim_portfolio: simulation portfolio (required for simulation)
        verbose: whether to print execution details
    
    Returns:
        Dictionary with:
            - executed: bool (True if order was submitted/simulated)
            - broker_order_id: str or None
            - error_message: str or None
            - updated_sim_portfolio: dict or None (for simulation mode)
    
    Logic:
        - If dry_run=True: log only, don't execute
        - If simulation: update sim_portfolio
        - If paper/live: submit to Alpaca
    """
    result = {
        'executed': False,
        'broker_order_id': None,
        'error_message': None,
        'updated_sim_portfolio': None
    }
    
    # DRY RUN: log but don't execute
    if config.dry_run:
        if verbose:
            print(f"🟡 DRY RUN: {order_plan.side.upper()} {order_plan.quantity} {order_plan.symbol} @ market")
            if order_plan.has_bracket():
                print(f"   TP: ${order_plan.tp_price:.2f}, SL: ${order_plan.sl_price:.2f}")
        result['error_message'] = 'Dry run mode - order not executed'
        return result
    
    # SIMULATION MODE
    if config.execution_mode == 'simulation':
        if sim_portfolio is None:
            result['error_message'] = 'Simulation mode requires sim_portfolio'
            return result
        
        # Simulate execution at market price (use planned price)
        # In real simulation, you'd use actual historical price at that timestamp
        execution_price = order_plan.limit_price if order_plan.limit_price else 100.0  # Placeholder
        
        try:
            updated_portfolio = update_sim_portfolio_after_trade(
                sim_portfolio=sim_portfolio,
                symbol=order_plan.symbol,
                side=order_plan.side,
                quantity=order_plan.quantity,
                price=execution_price
            )
            
            result['executed'] = True
            result['broker_order_id'] = f"SIM-{order_plan.symbol}-{datetime.now().strftime('%H%M%S')}"
            result['updated_sim_portfolio'] = updated_portfolio
            
            if verbose:
                print(f"✅ SIMULATION: {order_plan.side.upper()} {order_plan.quantity} {order_plan.symbol}")
        
        except Exception as e:
            result['error_message'] = str(e)
            if verbose:
                print(f"❌ Simulation error: {e}")
        
        return result
    
    # PAPER / LIVE MODE (Alpaca)
    if config.execution_mode in ['paper', 'live']:
        if alpaca_client is None:
            result['error_message'] = 'Paper/live mode requires alpaca_client'
            return result
        
        try:
            # Choose between bracket and market order
            if order_plan.has_bracket():
                # Submit bracket order (market entry + TP/SL)
                # place_bracket_order returns a dict: {'main_order': <Order>, ...}
                order_result = place_bracket_order(
                    client=alpaca_client,
                    symbol=order_plan.symbol,
                    qty=order_plan.quantity,
                    side=order_plan.side,
                    take_profit_price=order_plan.tp_price,
                    stop_loss_price=order_plan.sl_price,
                    tif=TimeInForce.DAY
                )
                order = order_result['main_order'] if order_result else None
            else:
                # Submit simple market order
                # place_market_order returns an Order object directly
                order = place_market_order(
                    client=alpaca_client,
                    symbol=order_plan.symbol,
                    qty=order_plan.quantity,
                    side=order_plan.side,
                    tif=TimeInForce.DAY
                )
            
            if order:
                result['executed'] = True
                result['broker_order_id'] = order.id
                
                if verbose:
                    order_type = "BRACKET" if order_plan.has_bracket() else "MARKET"
                    print(f"✅ {config.execution_mode.upper()}: {order_type} {order_plan.side.upper()} " +
                          f"{order_plan.quantity} {order_plan.symbol}")
                    if order_plan.has_bracket():
                        print(f"   TP: ${order_plan.tp_price:.2f} | SL: ${order_plan.sl_price:.2f}")
                    print(f"   Order ID: {order.id}")
            else:
                result['error_message'] = 'Order submission returned None'
        
        except Exception as e:
            result['error_message'] = str(e)
            if verbose:
                print(f"❌ Alpaca execution error: {e}")
        
        return result
    
    # Unknown mode
    result['error_message'] = f"Unknown execution mode: {config.execution_mode}"
    return result


# ========================================================================
# STEP 4: ORCHESTRATION (End-to-End Trading Cycle)
# ========================================================================

def run_trading_cycle(
    signals: List[Signal],
    position_states: Dict[str, PositionState],
    config: ExecutionConfig,
    account_cash: float,
    alpaca_client: Optional[TradingClient] = None,
    sim_portfolio: Optional[Dict[str, PositionState]] = None,
    verbose: bool = True,
    db_path: Optional[Union[str, Path]] = None,
    account_id: Optional[str] = None
) -> List[DecisionLogEntry]:
    """
    Run a complete trading cycle: signals → decisions → execution.
    
    Args:
        signals: list of trading signals
        position_states: dict of current positions
        config: execution configuration
        account_cash: available cash for position sizing
        alpaca_client: Alpaca client (for paper/live)
        sim_portfolio: simulation portfolio (for simulation)
        verbose: print progress
    
    Returns:
        List of DecisionLogEntry records (one per signal)
    
    Process for EACH signal:
        1. Look up current position state
        2. Call decide_action (logic gates)
        3. If action is buy/sell:
            - Build order plan (sizing + TP/SL)
            - Execute order plan (route to sim/paper/live)
        4. Build DecisionLogEntry
        5. Return all entries for logging
    """
    log_entries = []

    # Optional: SQLite persistence (trade attempts).
    resolved_db_path: Optional[Union[str, Path]] = None
    if db_path is not None:
        resolved_db_path = db_path
        init_db(resolved_db_path)
    elif config.execution_mode in ['paper', 'live', 'simulation']:
        # Keep opt-in by default; notebook can explicitly pass db_path.
        resolved_db_path = None
    
    for signal in signals:
        symbol = signal.symbol
        
        # Get current position
        position_state = position_states.get(symbol)
        if position_state is None:
            # Should not happen if position_states was built correctly
            position_state = PositionState(
                symbol=symbol,
                quantity=0,
                avg_entry_price=0.0,
                side='flat',
                source=config.execution_mode
            )
        
        # STEP 1: Decide action
        action, reason = decide_action(signal, position_state, config)
        
        # Initialize log entry
        log_entry = DecisionLogEntry(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal.signal_type,
            confidence=signal.confidence,
            prob_profit=signal.prob_profit,
            position_quantity_before=position_state.quantity,
            position_side_before=position_state.side,
            execution_mode=config.execution_mode,
            action=action,
            reason=reason
        )
        
        # STEP 2 & 3: If actionable, plan and execute
        if action in ['buy', 'sell']:
            # Get current price from signal meta
            current_price = signal.meta.get('current_price', 0.0)
            
            if current_price == 0.0:
                log_entry.error_message = 'No current price available in signal'
                # Persist as a FAILED attempt (engine-side skip).
                if resolved_db_path is not None:
                    trade_id = str(uuid.uuid4())
                    create_trade_attempt(
                        trade_id=trade_id,
                        account_id=account_id,
                        symbol=symbol,
                        side=action,
                        qty=0.0,
                        entry_price=None,
                        tp_price=None,
                        sl_price=None,
                        status='FAILED',
                        confidence=float(signal.confidence),
                        error=log_entry.error_message,
                        db_path=resolved_db_path,
                    )
                log_entries.append(log_entry)
                continue
            
            try:
                # Build order plan
                order_plan = build_order_plan(
                    signal=signal,
                    position_state=position_state,
                    config=config,
                    account_cash=account_cash,
                    current_price=current_price
                )
                
                # Store plan details in log
                log_entry.planned_quantity = order_plan.quantity
                log_entry.planned_entry_price = current_price
                log_entry.planned_tp_price = order_plan.tp_price
                log_entry.planned_sl_price = order_plan.sl_price
                
                if verbose:
                    print(f"\n🎯 {symbol}: {action.upper()}")
                    print(f"   Quantity: {order_plan.quantity} shares")
                    print(f"   Entry: ${current_price:.2f}")
                    if order_plan.tp_price:
                        print(f"   TP: ${order_plan.tp_price:.2f}")
                    if order_plan.sl_price:
                        print(f"   SL: ${order_plan.sl_price:.2f}")
                
                # Persist trade attempt before execution (hard rule: attempted_at set at INSERT time).
                trade_id: Optional[str] = None
                if resolved_db_path is not None:
                    trade_id = str(uuid.uuid4())
                    create_trade_attempt(
                        trade_id=trade_id,
                        account_id=account_id,
                        symbol=symbol,
                        side=order_plan.side,
                        qty=float(order_plan.quantity),
                        entry_price=None,
                        exit_price=None,
                        tp_price=order_plan.tp_price,
                        sl_price=order_plan.sl_price,
                        status='OPEN' if not config.dry_run else 'FAILED',
                        confidence=float(signal.confidence),
                        error='DRY_RUN' if config.dry_run else None,
                        db_path=resolved_db_path,
                    )

                # Execute order plan
                execution_result = execute_order_plan(
                    order_plan=order_plan,
                    config=config,
                    alpaca_client=alpaca_client,
                    sim_portfolio=sim_portfolio,
                    verbose=verbose
                )
                
                # Record execution result
                log_entry.executed = execution_result['executed']
                log_entry.broker_order_id = execution_result['broker_order_id']
                log_entry.execution_timestamp = datetime.now() if execution_result['executed'] else None
                log_entry.error_message = execution_result['error_message']

                # Persist outcome.
                if resolved_db_path is not None and trade_id is not None:
                    if execution_result.get('broker_order_id'):
                        set_trade_broker_order_id(
                            trade_id=trade_id,
                            broker_order_id=str(execution_result['broker_order_id']),
                            db_path=resolved_db_path,
                        )

                    if execution_result.get('executed'):
                        # We don't currently reconcile fill prices here; record the observed price and timestamp.
                        mark_trade_open(
                            trade_id=trade_id,
                            entry_price=float(current_price),
                            db_path=resolved_db_path,
                        )
                    else:
                        mark_trade_failed(
                            trade_id=trade_id,
                            error=str(execution_result.get('error_message') or 'Order not executed'),
                            db_path=resolved_db_path,
                        )
                
                # Update sim portfolio if simulation mode
                if execution_result.get('updated_sim_portfolio'):
                    sim_portfolio.update(execution_result['updated_sim_portfolio'])
            
            except Exception as e:
                log_entry.error_message = f"Planning/execution error: {str(e)}"
                if verbose:
                    print(f"❌ Error on {symbol}: {e}")

                if resolved_db_path is not None:
                    trade_id = str(uuid.uuid4())
                    create_trade_attempt(
                        trade_id=trade_id,
                        account_id=account_id,
                        symbol=symbol,
                        side=action,
                        qty=0.0,
                        entry_price=None,
                        tp_price=None,
                        sl_price=None,
                        status='FAILED',
                        confidence=float(signal.confidence),
                        error=log_entry.error_message,
                        db_path=resolved_db_path,
                    )
        
        else:
            # action was 'hold' or 'rejected' - no execution
            if verbose and action == 'rejected':
                print(f"⛔ {symbol}: {reason}")
        
        log_entries.append(log_entry)
    
    return log_entries
