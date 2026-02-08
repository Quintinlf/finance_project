"""
Portfolio State Management

Provides a consistent view of current holdings across simulation,
paper trading, and live trading modes.

Key function: get_position_states(execution_mode, ...) -> Dict[str, PositionState]
"""

from typing import Dict, Optional
from data_structures import PositionState, ExecutionConfig
from alpaca_exercises import get_positions


def get_position_states(
    universe: list,
    config: ExecutionConfig,
    alpaca_client=None,
    sim_portfolio: Optional[Dict[str, PositionState]] = None
) -> Dict[str, PositionState]:
    """
    Get current position state for all symbols in universe.
    
    Args:
        universe: list of symbols to check
        config: execution configuration
        alpaca_client: Alpaca trading client (required for paper/live)
        sim_portfolio: simulation portfolio dict (required for simulation mode)
    
    Returns:
        Dictionary mapping symbol -> PositionState
        Symbols with no position will have quantity=0, side='flat'
    
    Process:
        - If simulation mode: read from sim_portfolio
        - If paper/live: query Alpaca positions
        - Ensure every symbol in universe has an entry
    """
    position_states = {}
    
    if config.execution_mode == 'simulation':
        # Use simulation portfolio
        if sim_portfolio is None:
            sim_portfolio = {}
        
        for symbol in universe:
            if symbol in sim_portfolio:
                position_states[symbol] = sim_portfolio[symbol]
            else:
                # No position in simulation
                position_states[symbol] = PositionState(
                    symbol=symbol,
                    quantity=0,
                    avg_entry_price=0.0,
                    side='flat',
                    source='sim'
                )
    
    else:
        # Query Alpaca for paper/live positions
        if alpaca_client is None:
            raise ValueError("Alpaca client required for paper/live trading")
        
        alpaca_positions = get_positions(alpaca_client)
        
        # Build position map from Alpaca data
        alpaca_position_map = {}
        for pos in alpaca_positions:
            qty = float(pos.qty)
            alpaca_position_map[pos.symbol] = PositionState(
                symbol=pos.symbol,
                quantity=qty,
                avg_entry_price=float(pos.avg_entry_price),
                side='long' if qty > 0 else 'short' if qty < 0 else 'flat',
                source='alpaca',
                unrealized_pl=float(pos.unrealized_pl)
            )
        
        # Ensure all universe symbols have entries
        for symbol in universe:
            if symbol in alpaca_position_map:
                position_states[symbol] = alpaca_position_map[symbol]
            else:
                # No position at broker
                position_states[symbol] = PositionState(
                    symbol=symbol,
                    quantity=0,
                    avg_entry_price=0.0,
                    side='flat',
                    source='alpaca'
                )
    
    return position_states


def update_sim_portfolio_after_trade(
    sim_portfolio: Dict[str, PositionState],
    symbol: str,
    side: str,
    quantity: int,
    price: float
) -> Dict[str, PositionState]:
    """
    Update simulation portfolio after a trade.
    
    Args:
        sim_portfolio: current simulation portfolio
        symbol: ticker traded
        side: 'buy' or 'sell'
        quantity: shares traded
        price: execution price
    
    Returns:
        Updated portfolio dict
    
    Logic:
        - BUY: increase quantity, update avg entry price
        - SELL: reduce quantity, compute realized P&L
    """
    if symbol not in sim_portfolio:
        sim_portfolio[symbol] = PositionState(
            symbol=symbol,
            quantity=0,
            avg_entry_price=0.0,
            side='flat',
            source='sim'
        )
    
    current_state = sim_portfolio[symbol]
    
    if side == 'buy':
        # Buying shares: increase position
        new_quantity = current_state.quantity + quantity
        
        if current_state.quantity == 0:
            # Opening new position
            new_avg_price = price
        else:
            # Adding to existing position
            total_cost = (current_state.quantity * current_state.avg_entry_price) + (quantity * price)
            new_avg_price = total_cost / new_quantity
        
        sim_portfolio[symbol] = PositionState(
            symbol=symbol,
            quantity=new_quantity,
            avg_entry_price=new_avg_price,
            side='long' if new_quantity > 0 else 'flat',
            source='sim'
        )
    
    elif side == 'sell':
        # Selling shares: reduce position
        new_quantity = current_state.quantity - quantity
        
        # Compute realized P&L
        realized_pl = quantity * (price - current_state.avg_entry_price)
        
        if new_quantity <= 0:
            # Position closed or flipped
            sim_portfolio[symbol] = PositionState(
                symbol=symbol,
                quantity=0,
                avg_entry_price=0.0,
                side='flat',
                source='sim',
                unrealized_pl=0.0
            )
        else:
            # Partial close
            sim_portfolio[symbol] = PositionState(
                symbol=symbol,
                quantity=new_quantity,
                avg_entry_price=current_state.avg_entry_price,  # Entry price unchanged
                side='long',
                source='sim'
            )
    
    return sim_portfolio
