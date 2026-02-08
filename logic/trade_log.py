"""
Trade and Decision Logging

Persistent logging of all trading decisions and executions.
Logs EVERY decision including holds and rejections for full traceability.

Key functions:
- log_decision: append a single decision to the log file
- load_decision_log: read all logged decisions
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from data_structures import DecisionLogEntry


# Default log file location
DEFAULT_LOG_DIR = Path("trade_logs")
DEFAULT_DECISION_LOG = DEFAULT_LOG_DIR / "decisions.csv"


def ensure_log_directory():
    """Create log directory if it doesn't exist."""
    DEFAULT_LOG_DIR.mkdir(exist_ok=True)


def log_decision(
    entry: DecisionLogEntry,
    log_file: Path = DEFAULT_DECISION_LOG
):
    """
    Append a decision log entry to the CSV file.
    
    Args:
        entry: DecisionLogEntry to log
        log_file: path to CSV log file
    
    Creates file with headers if it doesn't exist.
    Appends entry as new row if file exists.
    """
    ensure_log_directory()
    
    # Convert entry to dict
    entry_dict = entry.to_dict()
    
    # Check if file exists
    file_exists = log_file.exists()
    
    # Open in append mode
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=entry_dict.keys())
        
        # Write header if new file
        if not file_exists:
            writer.writeheader()
        
        # Write entry
        writer.writerow(entry_dict)


def log_decisions_batch(
    entries: List[DecisionLogEntry],
    log_file: Path = DEFAULT_DECISION_LOG
):
    """
    Append multiple decision entries at once.
    
    More efficient than calling log_decision repeatedly.
    """
    if not entries:
        return
    
    ensure_log_directory()
    
    # Convert all entries
    entry_dicts = [entry.to_dict() for entry in entries]
    
    # Check if file exists
    file_exists = log_file.exists()
    
    # Open in append mode
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=entry_dicts[0].keys())
        
        # Write header if new file
        if not file_exists:
            writer.writeheader()
        
        # Write all entries
        writer.writerows(entry_dicts)


def load_decision_log(
    log_file: Path = DEFAULT_DECISION_LOG
) -> List[Dict[str, Any]]:
    """
    Load all decision log entries from CSV.
    
    Returns:
        List of dictionaries, one per logged decision
    """
    if not log_file.exists():
        return []
    
    with open(log_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_recent_decisions(
    n: int = 10,
    log_file: Path = DEFAULT_DECISION_LOG
) -> List[Dict[str, Any]]:
    """
    Get the N most recent decisions from the log.
    
    Args:
        n: number of recent decisions to return
        log_file: path to log file
    
    Returns:
        List of decision dicts (most recent first)
    """
    all_decisions = load_decision_log(log_file)
    
    # Return last N entries, reversed (most recent first)
    return list(reversed(all_decisions[-n:]))


def count_executions_today(
    log_file: Path = DEFAULT_DECISION_LOG
) -> int:
    """
    Count how many trades were executed today.
    
    Useful for daily trade limits or reporting.
    """
    decisions = load_decision_log(log_file)
    today_str = datetime.now().date().isoformat()
    
    executed_today = 0
    for entry in decisions:
        timestamp_str = entry.get('timestamp', '')
        if timestamp_str.startswith(today_str) and entry.get('executed') == 'True':
            executed_today += 1
    
    return executed_today


def get_execution_summary(
    log_file: Path = DEFAULT_DECISION_LOG
) -> Dict[str, Any]:
    """
    Get summary statistics from the decision log.
    
    Returns:
        Dictionary with:
            - total_decisions: total number of decisions logged
            - total_executed: number of orders executed
            - total_rejected: number of signals rejected
            - total_holds: number of hold decisions
            - execution_rate: % of decisions that resulted in execution
    """
    decisions = load_decision_log(log_file)
    
    if not decisions:
        return {
            'total_decisions': 0,
            'total_executed': 0,
            'total_rejected': 0,
            'total_holds': 0,
            'execution_rate': 0.0
        }
    
    total_decisions = len(decisions)
    total_executed = sum(1 for d in decisions if d.get('executed') == 'True')
    total_rejected = sum(1 for d in decisions if d.get('action') == 'rejected')
    total_holds = sum(1 for d in decisions if d.get('action') == 'hold')
    
    execution_rate = (total_executed / total_decisions * 100) if total_decisions > 0 else 0.0
    
    return {
        'total_decisions': total_decisions,
        'total_executed': total_executed,
        'total_rejected': total_rejected,
        'total_holds': total_holds,
        'execution_rate': execution_rate
    }


# ========================================================================
# TRADE HISTORY (Separate from Decisions)
# ========================================================================

DEFAULT_TRADE_LOG = DEFAULT_LOG_DIR / "trades.csv"


def log_closed_trade(
    symbol: str,
    side: str,
    quantity: int,
    entry_price: float,
    exit_price: float,
    entry_time: datetime,
    exit_time: datetime,
    profit: float,
    return_pct: float,
    signal_confidence: float = None,
    tp_price: float = None,
    sl_price: float = None,
    log_file: Path = DEFAULT_TRADE_LOG
):
    """
    Log a completed trade (entry and exit).
    
    This is different from decision log - only records closed positions.
    Used for performance analysis and strategy tuning.
    """
    ensure_log_directory()
    
    trade_record = {
        'symbol': symbol,
        'side': side,
        'quantity': quantity,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'entry_time': entry_time.isoformat(),
        'exit_time': exit_time.isoformat(),
        'profit': profit,
        'return_pct': return_pct,
        'signal_confidence': signal_confidence,
        'tp_price': tp_price,
        'sl_price': sl_price
    }
    
    file_exists = log_file.exists()
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trade_record.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(trade_record)


def load_trade_history(
    log_file: Path = DEFAULT_TRADE_LOG
) -> List[Dict[str, Any]]:
    """
    Load all closed trades from the trade log.
    
    Returns:
        List of trade dictionaries
    """
    if not log_file.exists():
        return []
    
    with open(log_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)
