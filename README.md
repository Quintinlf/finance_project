# Finance Project: Algorithmic Trading Bot

This repository contains an end-to-end algorithmic trading system built for research, simulation, paper trading, and live execution through Alpaca.

The bot combines probabilistic forecasting and technical indicators, then applies explicit position-aware decision logic before executing or rejecting trades.

## What This Project Is

This project is split into two major parts:

1. A production-style trading engine in the `logic/` folder.
2. A large learning and experimentation area in `Exercises/` for ML-in-finance notebooks.

If you only want to understand the trading bot, start with the files in `logic/` and `trading.ipynb`.

## What The Trading Bot Does

At a high level, each cycle does this:

1. Generate signals for each symbol in your universe.
2. Read current portfolio state (flat/long/short).
3. Decide action with clear logic gates.
4. Build order plan (size + optional TP/SL).
5. Execute in simulation, paper, or live mode.
6. Persist decisions/trade attempts for auditing and analysis.

## How It Works

### 1) Signal Generation

The signal engine (`logic/signal_engine.py`) wraps forecast functions from `logic/trading_functions.py`:

- `unified_bayesian_gp_forecast(...)`
- `calculate_bollinger_bands(...)`
- `bayesian_rsi_signal(...)`

It outputs normalized `Signal` objects (`buy`, `sell`, `hold`) with:

- confidence score
- probability of profit
- metadata (Bollinger z-score, RSI, current price, etc.)

### 2) Position State

`logic/portfolio_state.py` creates a unified position view for every symbol:

- `flat` (no position)
- `long`
- `short`

It supports both broker-backed states (paper/live) and in-memory simulation state.

### 3) Decision Engine

`logic/execution_engine.py` is the core of the bot. It enforces explicit rules in `decide_action(...)`:

- BUY + flat -> buy
- BUY + long -> hold
- SELL + long -> sell
- SELL + flat + shorting disabled -> rejected
- HOLD -> hold

This is where behavior is intentionally deterministic and auditable.

### 4) Order Planning And Execution

If action is actionable (`buy` or `sell`):

- `build_order_plan(...)` computes size with risk rules and TP/SL levels.
- `execute_order_plan(...)` routes by mode:
  - `simulation`: updates in-memory portfolio
  - `paper` / `live`: submits orders to Alpaca (`market` or `bracket`)

### 5) Logging And Persistence

The system supports two logging paths:

- CSV decision logs (`logic/trade_log.py`, default: `trade_logs/decisions.csv`)
- SQLite persistence (`logic/sqlite_store.py`, default: `trade_logs/trading.db`)

`run_trading_cycle(...)` can write broker trade attempts and outcomes to SQLite when a `db_path` is provided.

## Core Architecture Files

- `logic/data_structures.py`: typed data classes (`Signal`, `PositionState`, `ExecutionConfig`, `OrderPlan`, `DecisionLogEntry`)
- `logic/signal_engine.py`: signal generation and filtering
- `logic/portfolio_state.py`: position retrieval and simulation updates
- `logic/execution_engine.py`: decision gates, order plans, routing, orchestration
- `logic/risk_management.py`: position sizing and adaptive threshold utilities
- `logic/position_evaluator.py`: evaluates open positions against original signal quality
- `logic/sqlite_store.py`: durable SQLite schema and trade/decision storage
- `logic/trade_log.py`: CSV logging utilities

## Execution Modes

Configure mode via `ExecutionConfig`:

- `simulation`: no broker calls, fully local behavior
- `paper`: Alpaca paper account (safe real integration)
- `live`: real market execution

Important safety switches:

- `dry_run=True`: never submit real orders
- `allow_short_selling=False`: blocks opening shorts by default

## Typical Trading Cycle

The orchestration function is `run_trading_cycle(...)` in `logic/execution_engine.py`.

Minimal flow:

1. `signals = generate_signals(universe, config)`
2. `position_states = get_position_states(universe, config, alpaca_client=trading_client, sim_portfolio=sim_portfolio)`
3. `decision_log = run_trading_cycle(...)`
4. Persist decisions with CSV and/or SQLite

## Quick Start

### Prerequisites

- Python 3.10+
- Alpaca API credentials (for paper/live)

### Install

```bash
pip install -r requirements.txt
```

### Run

The primary workflow currently lives in `trading.ipynb`:

1. Configure parameters and mode (`dry_run` recommended first).
2. Verify Alpaca connection (paper account first).
3. Run forecasting and trading cycle cells.
4. Review logs in `trade_logs/`.

For implementation details, see `TRADING_ENGINE_README.md` and `COMPLETE_WORKFLOW.md`.

## Repository Layout

- `logic/`: production trading modules
- `trade_logs/`: runtime logs and database
- `trading.ipynb`: notebook-driven execution workflow
- `Exercises/`: educational notebooks and ML finance study material

## Notes And Disclaimer

- This project is educational/research software and not financial advice.
- Always validate in `dry_run` and `paper` mode before any live execution.
- Live trading carries real financial risk.
