# Finance Project: Algorithmic Trading Bot

This repository contains an end-to-end algorithmic trading system built for research, simulation, paper trading, and live execution through Alpaca.

The bot combines:
- **Probabilistic forecasting** (Bayesian regression, Gaussian Processes)
- **Technical indicators** (Bollinger Bands, RSI, momentum)
- **Game-theoretic payoff analysis** (market type inference, Nash equilibrium, Minimax strategies)
- **Explicit position-aware decision logic** before executing or rejecting trades

## ✨ Key Innovations

### Game Theory Integration
This project uniquely integrates **signaling game theory** into live trading:
- **Market Type Inference**: Real-time classification into Trend, Range, Manipulator, or Exhausted states
- **Equilibrium Payoff Calculation**: Nash equilibrium analysis for different trader types
- **Intuitive Criterion Filtering**: Peters-style elimination of dominated strategies
- **Minimax Decision-Making**: Risk-minimizing position sizing based on game-theoretic payoff matrices
- **Rational Deviation Analysis**: Detects when informed traders should deviate from equilibrium

### Trading Engine Features
- Real-time signal generation with confidence scoring
- Adaptive threshold tuning based on trade history
- Multi-mode execution (simulation, paper, live)
- Position-aware decision gates
- Risk management with stop-loss and take-profit
- SQLite persistence for full audit trail
- Monte Carlo strategy simulation
- Performance analytics with win/loss tracking

## What This Project Is

This project is split into two major parts:

1. **Production Trading Engine** (`logic/` folder): Full-featured algorithmic trading system with game theory
2. **Learning & Research** (`Exercises/` folder): Educational notebooks and ML finance study material

If you only want to understand the trading bot, start with the files in `logic/` and `trading.ipynb`.

## What The Trading Bot Does

At a high level, each cycle does this:

1. Generate signals for each symbol in your universe.
2. Read current portfolio state (flat/long/short).
3. Decide action with clear logic gates.
4. Build order plan (size + optional TP/SL).
5. Execute in simulation, paper, or live mode.
6. Persist decisions/trade attempts for auditing and analysis.

## 🎮 Game Theory Implementation

This system implements **information-asymmetric signaling game theory** from financial markets literature:

### Market Type Inference (`logic/game_utils.py`)

The bot classifies market regimes into **four types** based on recent price action and volatility:

1. **t_trend** - Directionally persistent moves; traders follow fundamentals
2. **t_manipulator** - High-volatility regime with possible informed manipulation
3. **t_exhausted** - Overbought/oversold extremes; mean reversion likely
4. **t_range** - Low-trending, bounded movements; mean-reverting behavior

**Inference Method**: Uses EMA-based trend strength, rolling volatility of volatility, and Bollinger Band width to compute Bayesian posterior probabilities over market types.

### Equilibrium Payoff Calculation (`logic/game_utils.py`)

For each market type, computes **Nash equilibrium payoffs** x(t):

$$x_t = \gamma^{-1} \sum_{k=1}^H \gamma^k \mathbb{E}[r_{t+k} \mid \text{regime}]$$

- Expected returns are discounted from one-step forecasts
- Persistence and damping are regime-dependent
- Reflects what equilibrium traders earn in each regime

### Type Beliefs and Rationality (`logic/game_utils.py`)

Constructs posterior belief distribution β over market types:

- **Prior**: Regime probabilities (trend, range, volatility)
- **Likelihood**: Signal type (buy/sell/hold), RSI levels, Bollinger z-scores
- **Posterior**: Normalized beliefs βi = p(type | observables)

Used to weight payoffs when trader types have heterogeneous beliefs.

### Deviation Payoff Analysis (`logic/game_utils.py`)

Computes **off-equilibrium deviation payoffs** m(t, A) for each market type:

- **Signal price target**: Expected move if signal is correct
- **Hunting cost**: Adverse selection from informed traders
- **Liquidity profit**: Execution benefits in responsive regimes
- **Net manipulator payoff**: Asymmetric payoff for informed traders

Deviation payoffs reflect: *What do informed traders earn by deviating from equilibrium?*

### Intuitive Criterion Filter (`logic/intuitive_criterion.py`)

Implements **Peters-style elimination** of irrational beliefs:

**Rule**: Eliminate market type t if m(t, A) < x(t)
- If deviation payoff is worse than equilibrium, traders won't deviate
- Remaining types are **survivor types** under IC

Applied to every signal:
1. Extract prior beliefs over types
2. Compute equilibrium and deviation payoffs
3. Eliminate dominated types
4. Update posterior belief distribution
5. Keep signal only if **at least one rational type survives**

### Minimax Position Sizing (`logic/risk_management.py`)

Uses game-theoretic minimax criterion for position sizing:

```
Position Size = Base × minimax_multiplier(payoff_matrix)
```

Where `minimax_multiplier` reflects:
- Alignment of equilibrium and deviation payoffs across market types
- Robustness to regime uncertainty  
- Confidence that signal is rational across all possible types

**Range**: 0.0 (no edge) to 1.0 (strong alignment across types)

---

## How It Works

### 1) Signal Generation

The signal engine (`logic/signal_engine.py`) wraps forecast functions from `logic/trading_functions.py`:

- `unified_bayesian_gp_forecast(...)` - Ensemble of Bayesian & Gaussian Process models
- `calculate_bollinger_bands(...)` - Price extremes and mean reversion signals
- `bayesian_rsi_signal(...)` - Momentum with probabilistic interpretation

**Game Theory Integration**:
1. Compute market regime (trend strength, volatility, range-bound behavior)
2. Build expected return path over 5-step horizon
3. Calculate equilibrium payoffs for each market type
4. Infer trader type beliefs from RSI, Bollinger Bands, regime
5. Compute deviation payoffs (what informed traders earn by trading)
6. Apply Intuitive Criterion filter (eliminate irrational types)
7. Compute minimax multiplier for position sizing

**Signal Output**: Normalized `Signal` objects with:
- Type: `buy`, `sell`, or `hold`
- Confidence score (0.0-1.0)
- Probability of profit
- Game-theoretic metadata:
  - Market regime probabilities
  - Equilibrium payoffs by type
  - Deviation payoffs by type
  - Type beliefs (posterior over market types)
  - Minimax alignment score

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

### Core Engine Modules
- `logic/data_structures.py`: Type-safe dataclasses
  - `Signal`: buy/sell/hold with confidence, payoff metadata
  - `PositionState`: flat/long/short tracking
  - `ExecutionConfig`: configuration parameters
  - `OrderPlan`: size, TP, SL, order type
  - `DecisionLogEntry`: audit trail entry
  - `RLObservation`: state features for learning

- `logic/signal_engine.py`: Signal generation pipeline
  - Forecast ensemble (Bayesian + Gaussian Process)
  - Bollinger Bands integration
  - RSI momentum signals
  - **Game theory integration** (regime → equilibrium → minimax)

- `logic/portfolio_state.py`: Position state management
  - Unified interface for simulation/paper/live
  - Handles long, short, and flat positions
  - Integrates with Alpaca broker API

- `logic/execution_engine.py`: Core trading orchestration
  - Decision logic gates (BUY+flat→buy, SELL+long→sell)
  - Order plan building with risk limits
  - Execution routing (simulation/paper/live)
  - Full trading cycle runner: `run_trading_cycle(...)`

- `logic/risk_management.py`: Position sizing & risk
  - `calculate_position_size(...)`: kelly-fraction based sizing
  - `calculate_minimax_multiplier(...)`: **game-theoretic position adjustment**
  - `mcmc_optimize_thresholds(...)`: Bayesian parameter tuning
  - Adaptive confidence threshold updates

- `logic/position_evaluator.py`: Live position analytics
  - Compares current prices to original signal quality
  - Tracks performance vs forecast accuracy
  - Used for threshold auto-tuning

### Game-Theoretic Analysis
- `logic/game_utils.py`: **Signaling game theory core**
  - Market regime inference: compute_market_regime(...)
  - Equilibrium payoff calculation: calculate_equilibrium_payoffs(...)
  - Type belief inference: infer_type_beliefs(...)
  - Expected return path projection: build_expected_return_path(...)
  - Risk-weighted deviation payoff: build_deviation_from_market_data(...)

- `logic/intuitive_criterion.py`: **Peters-style IC filtering**
  - `survives_intuitive_criterion(...)`: Eliminate irrational types
  - Prior belief normalization
  - Posterior belief updating
  - Survivor type identification

### Forecasting & Analysis
- `logic/trading_functions.py`: Forecast ensemble
  - `unified_bayesian_gp_forecast(...)`: Combined forecasts
  - `bayesian_linear_regression(...)`: Bayesian point forecasts
  - `gaussian_process_forecast(...)`: GP with uncertainty
  - `gbm(...)`: Geometric Brownian Motion simulator

- `logic/signal_engine.py`: Signal generation & filtering
  - `generate_signals(...)`: Full pipeline
  - `filter_signals_by_thresholds(...)`: Confidence filtering
  - Game theory context integration

### Persistence & Logging
- `logic/sqlite_store.py`: SQLite database layer
  - Schema: decisions, accounts, positions, trades
  - Durable audit trail
  - Query interface for analytics

- `logic/trade_log.py`: CSV logging
  - `log_decision(...)`: Append to CSV decision log
  - Default: `trade_logs/decisions.csv`

- `logic/alpaca_exercises.py`: Alpaca broker integration
  - Account management: `get_account_summary(...)`
  - Order execution: `place_bracket_order(...)`
  - Position retrieval: `get_positions(...)`

### Utilities
- `logic/trading_assistant.py`: Recommendation engine
  - `trading_assistant(...)`: Generate actionable recommendations
  - Evaluates strategy performance
  - Suggests next actions (backtest, tune, simulate, etc.)

- `logic/math_logic/montecarlo_sims.py`: Monte Carlo simulation
  - `monte_carlo_strategy_simulation(...)`: Strategy P&L projection
  - Path simulation with draw-down tracking
  - Returns distribution analysis

- `logic/math_logic/series_sequences.py`: Mathematical sequences
  - Arithmetic and geometric progressions
  - Series summation utilities

- `logic/math_logic/polar_coordinates_finance.py`: Polar space analysis
  - `bayesian_polar_signal(...)`: Bayesian belief in polar coordinates
  - Visual representation of market state

## Execution Modes

Configure mode via `ExecutionConfig`:

- `simulation`: no broker calls, fully local behavior
- `paper`: Alpaca paper account (safe real integration)
- `live`: real market execution

Important safety switches:

- `dry_run=True`: never submit real orders
- `allow_short_selling=False`: blocks opening shorts by default

## Complete Feature List

### ✅ Core Trading Engine
- [x] Real-time signal generation (buy/sell/hold)
- [x] Multi-mode execution (simulation, paper, live)
- [x] Position state tracking (flat, long, short)
- [x] Deterministic decision gates
- [x] Position-aware order planning
- [x] Risk management with TP/SL
- [x] Short selling support (optional)
- [x] Bracket order execution
- [x] Market and limit order types

### ✅ Signal Generation & Forecasting
- [x] Bayesian linear regression
- [x] Gaussian Process forecasting
- [x] Ensemble forecast aggregation
- [x] Bollinger Bands (price extremes)
- [x] RSI momentum signals
- [x] Bayesian RSI interpretation
- [x] Confidence scoring
- [x] Probability of profit calculation
- [x] Multi-symbol batch processing

### ✅ Game Theory Integration
- [x] Market regime inference (4 types)
- [x] Equilibrium payoff calculation
- [x] Type belief inference (Bayesian)
- [x] Intuitive Criterion filtering
- [x] Deviation payoff analysis
- [x] Minimax position sizing
- [x] Rational strategy elimination

### ✅ Risk Management
- [x] Kelly Criterion position sizing
- [x] Minimax-adjusted sizing
- [x] Stop-loss and take-profit levels
- [x] Max positions limit
- [x] Risk-per-trade percentage
- [x] Adaptive threshold tuning
- [x] Position evaluation analytics
- [x] Win/loss tracking
- [x] Profit factor analysis

### ✅ Persistence & Auditing
- [x] SQLite trade log with full schema
- [x] CSV decision log
- [x] Account state snapshots
- [x] Position history
- [x] Trade outcomes
- [x] Decision audit trail
- [x] Query interface for analytics

### ✅ Analytics & Optimization
- [x] Monte Carlo strategy simulation
- [x] P&L distribution analysis
- [x] Draw-down tracking
- [x] Win rate calculation
- [x] MCMC threshold optimization
- [x] Performance statistics
- [x] Trading assistant recommendations
- [x] Position performance evaluation

### ✅ Broker Integration
- [x] Alpaca API connection
- [x] Paper trading account
- [x] Live trading account
- [x] Account balance retrieval
- [x] Position listing
- [x] Order submission (market & bracket)
- [x] Order tracking
- [x] Credential validation

### ✅ Learning & Research
- [x] Reinforcement learning exercises
- [x] ML-in-finance notebooks
- [x] Bollinger Bands analysis
- [x] RSI strategy notebooks
- [x] Option pricing (QLBS model)
- [x] Q-Learning wealth management
- [x] Inverse RL models
- [x] Chapter-based curriculum

## Workflow Integration

### Stage 1: Setup & Configuration
- Define stock universe
- Configure execution mode (simulation/paper/live)
- Set risk parameters (TP%, SL%, risk per trade)
- Set threshold values (MIN_CONF, MIN_PROB_UP)

### Stage 2: Signal Generation
- Compute market regime for each stock
- Generate buy/sell/hold signals
- Calculate equilibrium & deviation payoffs
- Apply Intuitive Criterion filtering
- Compute minimax multipliers

### Stage 3: Position Analysis (if positions exist)
- Evaluate open positions
- Compare to original predictions
- Calculate performance scores
- Recommend threshold adjustments

### Stage 4: Trading Logic
- Retrieve current positions
- Apply decision gates (BUY+flat, SELL+long, etc.)
- Build order plans with TP/SL
- Route to execution (simulation/paper/live)

### Stage 5: Persistence
- Log decisions to CSV
- Store trade outcomes to SQLite
- Update account snapshots
- Track performance metrics

### Stage 6: Analysis (optional)
- Run Monte Carlo simulation
- Generate strategy recommendations
- Tune thresholds via MCMC
- Export analytics

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
