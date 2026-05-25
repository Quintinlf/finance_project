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
from zoneinfo import ZoneInfo
from datetime import timezone
import csv

from logic.risk_config import load_risk_config, PortfolioRiskConfig
from logic.sqlite_store import DEFAULT_DB_PATH, get_latest_account_snapshot, connect

from logic.broker_client import BrokerClient
from logic.data_structures import (
    Signal, PositionState, ExecutionConfig, OrderPlan, DecisionLogEntry
)
from logic.portfolio_state import update_sim_portfolio_after_trade
from logic.risk_management import calculate_position_size, calculate_minimax_multiplier
from logic.intuitive_criterion import survives_intuitive_criterion

from logic.sqlite_store import (
    DEFAULT_DB_PATH,
    init_db,
    create_trade_attempt,
    set_trade_broker_order_id,
    mark_trade_open,
    mark_trade_failed,
    insert_uncertainty_metric,
    insert_tail_risk_metric,
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

    # Intuitive Criterion (IC): reject strategically irrational deviations.
    if action in {'buy', 'sell'}:
        ic_pass, ic_reason, ic_details = survives_intuitive_criterion(signal)
        signal.meta['ic_filter_result'] = ic_details
        if isinstance(ic_details, dict) and 'posterior_beliefs' in ic_details:
            signal.type_beliefs = ic_details['posterior_beliefs']
        if not ic_pass:
            action = 'rejected'
            reason = f"IC veto: {ic_reason}"
    
    return action, reason


def apply_safety_veto(
    *,
    action: str,
    signal: Signal,
    position_state: PositionState,
    config: ExecutionConfig,
) -> Tuple[str, Optional[str]]:
    """Hard veto layer applied after action selection and before order planning."""
    if not config.enforce_safety_veto:
        return action, None

    current_drawdown_pct = signal.meta.get('current_drawdown_pct')
    if (
        action == 'buy'
        and config.max_allowed_drawdown_pct is not None
        and current_drawdown_pct is not None
        and float(current_drawdown_pct) > float(config.max_allowed_drawdown_pct)
    ):
        return (
            'hold',
            (
                f"Safety veto: BUY blocked (drawdown {float(current_drawdown_pct):.2f}% "
                f"> max {float(config.max_allowed_drawdown_pct):.2f}%)"
            ),
        )

    # Prevent invalid negative/zero price actions from reaching execution.
    current_price = signal.meta.get('current_price')
    if action in {'buy', 'sell'} and (current_price is None or float(current_price) <= 0.0):
        return 'hold', 'Safety veto: actionable signal blocked due to invalid current_price'

    return action, None


def apply_uncertainty_gate(
    *,
    action: str,
    signal: Signal,
    config: ExecutionConfig,
) -> Tuple[str, Optional[str], bool, bool, Optional[str]]:
    """Apply uncertainty gate in off/shadow/enforce modes.

    Returns:
        (next_action, next_reason, would_veto, enforced_veto, veto_reason)
    """
    if action not in {'buy', 'sell'}:
        return action, None, False, False, None

    mode = str(config.uncertainty_gate_mode or 'off').lower()
    if mode == 'off':
        return action, None, False, False, None

    entropy_val = signal.meta.get('belief_entropy')
    kl_val = signal.meta.get('kl_divergence')

    reasons: List[str] = []
    if (
        config.max_belief_entropy is not None
        and entropy_val is not None
        and float(entropy_val) > float(config.max_belief_entropy)
    ):
        reasons.append(
            f"entropy {float(entropy_val):.4f} > {float(config.max_belief_entropy):.4f}"
        )

    if (
        config.max_kl_divergence is not None
        and kl_val is not None
        and float(kl_val) > float(config.max_kl_divergence)
    ):
        reasons.append(
            f"kl {float(kl_val):.4f} > {float(config.max_kl_divergence):.4f}"
        )

    if not reasons:
        return action, None, False, False, None

    veto_reason = "Uncertainty gate: " + "; ".join(reasons)
    if mode == 'shadow':
        return action, None, True, False, veto_reason

    return 'rejected', veto_reason, True, True, veto_reason


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
    
    # Calculate position size with minimax confidence scaling
    if side == 'buy':
        # Calculate base shares to buy based on risk
        base_qty = calculate_position_size(
            account_balance=account_cash,
            risk_per_trade_pct=config.base_risk_pct,
            stop_loss_pct=config.sl_pct * 100,  # Function expects percentage as 0-100
            price=current_price
        )
        
        # Cap at max position size
        max_shares = int((account_cash * config.max_position_pct_of_equity / 100) / current_price)
        base_qty = min(base_qty, max_shares)
        
        # Apply minimax confidence multiplier (hybrid mode)
        mm = calculate_minimax_multiplier(signal)
        quantity = int(base_qty * mm)
        quantity = max(1, quantity)  # At least 1 share
        
        # Store minimax multiplier in signal metadata for audit trail
        signal.meta['minimax_applied_multiplier'] = mm
        
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
            # Calculate base shares to sell based on risk
            base_qty = calculate_position_size(
                account_balance=account_cash,
                risk_per_trade_pct=config.base_risk_pct,
                stop_loss_pct=config.sl_pct * 100,
                price=current_price
            )
            max_shares = int((account_cash * config.max_position_pct_of_equity / 100) / current_price)
            base_qty = min(base_qty, max_shares)
            
            # Apply minimax confidence multiplier (hybrid mode)
            mm = calculate_minimax_multiplier(signal)
            quantity = int(base_qty * mm)
            quantity = max(1, quantity)
            
            # Store minimax multiplier in signal metadata for audit trail
            signal.meta['minimax_applied_multiplier'] = mm
            
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
    broker_client: BrokerClient,
    sim_portfolio: Optional[Dict[str, PositionState]] = None,
    verbose: bool = True
) -> Dict:
    """
    Execute an order plan by routing to the appropriate executor.
    
    Args:
        order_plan: the order to execute
        config: execution configuration
        broker_client: broker client for paper/live execution
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
    
    # PAPER / LIVE MODE (broker client)
    if config.execution_mode in ['paper', 'live']:
        try:
            # Choose between bracket and market order
            if order_plan.has_bracket():
                # Submit bracket order (market entry + TP/SL)
                order_result = broker_client.place_bracket_order(
                    symbol=order_plan.symbol,
                    qty=order_plan.quantity,
                    side=order_plan.side,
                    take_profit_price=order_plan.tp_price,
                    stop_loss_price=order_plan.sl_price,
                    time_in_force=order_plan.time_in_force,
                )
                order = order_result['main_order'] if order_result else None
            else:
                # Submit simple market order
                order = broker_client.place_market_order(
                    symbol=order_plan.symbol,
                    qty=order_plan.quantity,
                    side=order_plan.side,
                    time_in_force=order_plan.time_in_force,
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
                print(f"❌ Broker execution error: {e}")
        
        return result
    
    # Unknown mode
    result['error_message'] = f"Unknown execution mode: {config.execution_mode}"
    return result


# ========================================================================
# STEP 4: ORCHESTRATION (End-to-End Trading Cycle)
# ========================================================================


# ========================================================================
# STEP 2.5: RISK LAYER (HARD GUARDRAILS)
# ========================================================================

def enforce_risk_limits(
    order_plan: OrderPlan,
    action: str,
    position_state: PositionState,
    risk_cfg: PortfolioRiskConfig,
    account_cash: float,
    trades_today: int,
    daily_return: float,
    exposure: float,
    db_path: Union[str, Path],
    current_price: float,
) -> Tuple[bool, Optional[OrderPlan], Optional[str]]:
    """Return (allowed, possibly_modified_plan, reason_if_blocked).

    This enforces hard guardrails like daily loss limits, max trades per day,
    portfolio exposure caps, and max position size. It may return a modified
    OrderPlan with a reduced quantity when capping is applied.
    """
    # Duplicate prevention: if already long and attempting to open long
    if action == 'buy' and position_state.side == 'long' and getattr(position_state, 'quantity', 0):
        return False, None, 'Blocked: duplicate order - existing open position'

    # Daily loss kill switch
    try:
        if daily_return <= -float(risk_cfg.daily_loss_limit):
            return False, None, 'Daily loss limit reached — trading halted.'
    except Exception:
        pass

    # Max trades per day
    try:
        if trades_today >= int(risk_cfg.max_trades_per_day) and action in {'buy', 'sell'}:
            return False, None, 'Blocked: max trades per day reached'
    except Exception:
        pass

    # Exposure cap for new BUYs
    if action == 'buy':
        try:
            if exposure >= float(risk_cfg.max_portfolio_exposure):
                return False, None, 'Blocked: portfolio exposure limit reached'
        except Exception:
            pass

        # Max position size (notional)
        try:
            latest = get_latest_account_snapshot(db_path=db_path)
            account_equity = float(latest.get('equity') or latest.get('portfolio_value') or account_cash)
        except Exception:
            account_equity = float(account_cash)

        try:
            max_dollar_position = account_equity * float(risk_cfg.max_position_size)
            proposed_notional = float(order_plan.quantity) * float(current_price)
            if proposed_notional > max_dollar_position:
                # Cap quantity
                capped_qty = int(max_dollar_position / float(current_price))
                if capped_qty < 1:
                    return False, None, 'Blocked: max position size caps order below minimum notional ($1)'
                # create a shallow copy of order_plan with adjusted quantity
                new_plan = OrderPlan(
                    symbol=order_plan.symbol,
                    side=order_plan.side,
                    quantity=capped_qty,
                    entry_type=order_plan.entry_type,
                    limit_price=order_plan.limit_price,
                    tp_price=order_plan.tp_price,
                    sl_price=order_plan.sl_price,
                    time_in_force=order_plan.time_in_force,
                    reason=(order_plan.reason + ' | Capped by max_position_size')
                )
                return True, new_plan, None
        except Exception:
            pass

    return True, order_plan, None


def run_trading_cycle(
    signals: List[Signal],
    position_states: Dict[str, PositionState],
    config: ExecutionConfig,
    account_cash: float,
    broker_client: BrokerClient,
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
        broker_client: broker client (for paper/live)
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
    
    # Load risk configuration (single source for enforcement)
    risk_cfg: PortfolioRiskConfig = load_risk_config()

    resolved_db_path: Optional[Union[str, Path]] = None

    # Compute pre-trade snapshot values (trades today, exposure, daily return)
    # Prefer the passed db_path, else use default DB location.
    snapshot_db_path = resolved_db_path if resolved_db_path is not None else DEFAULT_DB_PATH

    def _get_trades_executed_today(db_path: Union[str, Path]) -> int:
        # Count decisions.executed=1 where execution_timestamp in US/Eastern is today
        count = 0
        try:
            with connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT execution_timestamp FROM decisions WHERE executed = 1"
                ).fetchall()

            eastern = ZoneInfo("US/Eastern")
            now_eastern = datetime.now(tz=eastern).date()

            for r in rows:
                ts = r[0]
                if not ts:
                    continue
                try:
                    dt = datetime.fromisoformat(ts)
                    # assume stored as UTC if no tzinfo
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    dt_eastern = dt.astimezone(eastern).date()
                    if dt_eastern == now_eastern:
                        count += 1
                except Exception:
                    continue
        except Exception:
            # If DB not available, return 0 to be conservative
            return 0
        return count

    def _compute_daily_return(db_path: Union[str, Path]) -> float:
        # Use latest account snapshot: compare equity to last_equity
        try:
            snap = get_latest_account_snapshot(db_path=db_path)
            if not snap:
                # Missing snapshot — conservative: return a sentinel to trigger block
                return -1.0
            equity = snap.get('equity') or snap.get('portfolio_value') or None
            last_equity = snap.get('last_equity')
            if equity is None or last_equity is None:
                return -1.0
            try:
                return float(equity) / float(last_equity) - 1.0
            except Exception:
                return -1.0
        except Exception:
            return -1.0

    def _compute_current_exposure(db_path: Union[str, Path], account_cash: float) -> float:
        # Prefer account snapshot values
        try:
            snap = get_latest_account_snapshot(db_path=db_path)
            if snap:
                portfolio_value = snap.get('portfolio_value')
                cash = snap.get('cash')
                if portfolio_value is None or cash is None:
                    return 1.0
                try:
                    cash = float(cash)
                    pv = float(portfolio_value)
                    if cash <= 0:
                        return 1.0
                    return pv / cash
                except Exception:
                    return 1.0
        except Exception:
            return 1.0
        # Fallback: use provided account_cash and assume portfolio_value = account_cash
        return 0.0

    trades_today = _get_trades_executed_today(snapshot_db_path)
    daily_return = _compute_daily_return(snapshot_db_path)
    exposure = _compute_current_exposure(snapshot_db_path, account_cash)

    # Pre-trade snapshot logging
    snapshot_path = Path("trade_logs") / "run_snapshots.csv"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_header = not snapshot_path.exists()
        with snapshot_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow([
                    "utc_timestamp",
                    "portfolio_value",
                    "exposure",
                    "trades_today",
                    "daily_return",
                    "max_portfolio_exposure",
                    "max_position_size",
                    "max_trades_per_day",
                    "daily_loss_limit",
                    "paper_trading_mode",
                ])
            latest_portfolio_value = None
            try:
                latest = get_latest_account_snapshot(db_path=snapshot_db_path)
                latest_portfolio_value = latest.get('portfolio_value') if latest else None
            except Exception:
                latest_portfolio_value = None
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                latest_portfolio_value,
                f"{exposure:.4f}",
                trades_today,
                f"{daily_return:.6f}",
                risk_cfg.max_portfolio_exposure,
                risk_cfg.max_position_size,
                risk_cfg.max_trades_per_day,
                risk_cfg.daily_loss_limit,
                risk_cfg.paper_trading_mode,
            ])
    except Exception:
        # best-effort logging
        pass

    # Print a concise pre-trade snapshot
    if verbose:
        print("\n📋 Pre-trade snapshot:")
        print(f"   Exposure: {exposure:.4f}")
        print(f"   Trades today (executed): {trades_today}")
        print(f"   Daily return: {daily_return:.4f}")
        print(f"   Risk config: exposure={risk_cfg.max_portfolio_exposure}, max_pos={risk_cfg.max_position_size}, max_trades/day={risk_cfg.max_trades_per_day}, daily_loss_limit={risk_cfg.daily_loss_limit}, paper_mode={risk_cfg.paper_trading_mode}")

    # If paper trading mode is enabled, print a clear warning
    if risk_cfg.paper_trading_mode and verbose:
        print("🟡 PAPER TRADING MODE: True — executing in paper mode with guardrails enabled.")
    # Optional: SQLite persistence (trade attempts).
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

        # UNCERTAINTY GATE: optional off/shadow/enforce veto layer.
        gate_action, gate_reason, would_veto, enforced_veto, veto_reason = apply_uncertainty_gate(
            action=action,
            signal=signal,
            config=config,
        )
        if gate_reason is not None:
            action = gate_action
            reason = gate_reason

        signal.meta['uncertainty_would_veto'] = bool(would_veto)
        signal.meta['uncertainty_enforced_veto'] = bool(enforced_veto)
        signal.meta['uncertainty_veto_reason'] = veto_reason

        # SAFETY VETO: final guardrail before planning/execution.
        vetoed_action, veto_reason = apply_safety_veto(
            action=action,
            signal=signal,
            position_state=position_state,
            config=config,
        )
        if veto_reason:
            action = vetoed_action
            reason = veto_reason
        
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

        # Attach uncertainty/tail diagnostics to decision log entry.
        log_entry.belief_entropy = (
            float(signal.meta.get('belief_entropy')) if signal.meta.get('belief_entropy') is not None else None
        )
        log_entry.kl_divergence = (
            float(signal.meta.get('kl_divergence')) if signal.meta.get('kl_divergence') is not None else None
        )
        log_entry.uncertainty_veto_flag = bool(would_veto)
        log_entry.uncertainty_veto_reason = veto_reason
        log_entry.tail_cvar_estimate = (
            float(signal.meta.get('tail_cvar_estimate')) if signal.meta.get('tail_cvar_estimate') is not None else None
        )
        log_entry.simulated_crash_freq_3sigma = (
            float(signal.meta.get('simulated_crash_freq_3sigma'))
            if signal.meta.get('simulated_crash_freq_3sigma') is not None
            else None
        )
        log_entry.simulated_crash_freq_5sigma = (
            float(signal.meta.get('simulated_crash_freq_5sigma'))
            if signal.meta.get('simulated_crash_freq_5sigma') is not None
            else None
        )
        log_entry.simulated_crash_freq_10sigma = (
            float(signal.meta.get('simulated_crash_freq_10sigma'))
            if signal.meta.get('simulated_crash_freq_10sigma') is not None
            else None
        )

        if resolved_db_path is not None:
            insert_uncertainty_metric(
                account_id=account_id,
                symbol=symbol,
                belief_entropy=log_entry.belief_entropy,
                kl_divergence=log_entry.kl_divergence,
                veto_flag=bool(would_veto),
                veto_reason=veto_reason,
                regime_probs_snapshot=signal.meta.get('market_regime'),
                db_path=resolved_db_path,
            )

            insert_tail_risk_metric(
                account_id=account_id,
                symbol=symbol,
                alpha=(float(signal.meta.get('stable_alpha')) if signal.meta.get('stable_alpha') is not None else None),
                beta=(float(signal.meta.get('stable_beta')) if signal.meta.get('stable_beta') is not None else None),
                stable_scale=(
                    float(signal.meta.get('stable_scale')) if signal.meta.get('stable_scale') is not None else None
                ),
                stable_loc=(float(signal.meta.get('stable_loc')) if signal.meta.get('stable_loc') is not None else None),
                cvar_estimate=log_entry.tail_cvar_estimate,
                tail_percentiles=signal.meta.get('tail_percentiles'),
                simulated_crash_freq={
                    'rate_3sigma': log_entry.simulated_crash_freq_3sigma,
                    'rate_5sigma': log_entry.simulated_crash_freq_5sigma,
                    'rate_10sigma': log_entry.simulated_crash_freq_10sigma,
                },
                db_path=resolved_db_path,
            )
        
        # STEP 2 & 3: If actionable, plan and execute
        if action in ['buy', 'sell']:
            # Get current price from signal meta
            current_price = signal.meta.get('current_price', 0.0)

            # Early-exit when price is missing or invalid. Persist a FAILED attempt if DB enabled.
            if current_price is None or float(current_price) <= 0.0:
                log_entry.error_message = 'No current price available in signal'
                log_entry.action = 'rejected'
                log_entry.executed = False

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

                # RISK CHECK: enforce hard guardrails before attempting to persist/execute
                try:
                    trades_today = 0
                    daily_return = 0.0
                    exposure = 0.0
                    try:
                        # best-effort helpers if available
                        if resolved_db_path is not None:
                            trades_today = int(count_trades_today(db_path=resolved_db_path))
                    except Exception:
                        trades_today = 0
                    try:
                        latest_snapshot = get_latest_account_snapshot(db_path=resolved_db_path) if resolved_db_path is not None else {}
                        daily_return = float(latest_snapshot.get('daily_return', 0.0) or 0.0)
                        exposure = float(latest_snapshot.get('exposure', 0.0) or 0.0)
                    except Exception:
                        daily_return = 0.0
                        exposure = 0.0

                    allowed, maybe_plan, block_reason = enforce_risk_limits(
                        order_plan=order_plan,
                        action=action,
                        position_state=position_state,
                        risk_cfg=risk_cfg,
                        account_cash=account_cash,
                        trades_today=trades_today,
                        daily_return=daily_return,
                        exposure=exposure,
                        db_path=resolved_db_path,
                        current_price=current_price,
                    )
                    if not allowed:
                        log_entry.action = 'rejected'
                        log_entry.reason = block_reason
                        log_entry.error_message = block_reason
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
                                error=block_reason,
                                db_path=resolved_db_path,
                            )
                        log_entries.append(log_entry)
                        continue

                    # If risk layer returned a modified plan, adopt it
                    if maybe_plan is not None and maybe_plan is not order_plan:
                        order_plan = maybe_plan
                except Exception:
                    # On any unexpected error in risk checks, proceed conservatively and block
                    log_entry.action = 'rejected'
                    log_entry.reason = 'Risk check subsystem error'
                    log_entry.error_message = 'Risk check subsystem error'
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
                    broker_client=broker_client,
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
