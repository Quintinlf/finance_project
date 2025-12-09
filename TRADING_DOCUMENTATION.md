# Trading System Documentation

## Complete Documentation for Your Automated Paper Trading System

This document contains all the comprehensive documentation for your trading system.
The actual trading notebook (`trading.ipynb`) is kept minimal with just executable code.

---

[Previous documentation content - the 5 W's + H, core functions deep dive, real-world workflows, etc. are all here in this markdown file for reference, keeping the notebook clean]

## System Overview

**WHO**: Retail traders using algorithmic strategies  
**WHAT**: Automated paper trading with Bayesian ML forecasting  
**WHEN**: During market hours, daily/weekly scans  
**WHERE**: Yahoo Finance (data) + Alpaca API (execution)  
**WHY**: Probabilistic, multi-model, automated, risk-managed approach  
**HOW**: Data → Feature Engineering → Bayesian + GP + RSI → Ensemble → Execution

## Core Modules

### 1. `trading_functions.py`
- `unified_bayesian_gp_forecast(ticker)` - Main forecasting brain
- `calculate_bollinger_bands(df)` - Technical analysis
- `bayesian_rsi_signal(rsi)` - RSI interpretation
- **Now includes**: Bollinger Bands z-scores as features!

### 2. `risk_management.py`
- `adaptive_threshold_calculator(trade_history)` - Bayesian parameter tuning
- `mcmc_optimize_thresholds(trade_history)` - MCMC optimization
- `calculate_position_size(...)` - Risk-based sizing
- `portfolio_risk_metrics(positions)` - Portfolio analysis

### 3. `trading_assistant.py`
- `trading_assistant(trades, positions, account)` - Automated recommendations
- `get_quick_health_check(trades)` - Fast health check
- **Tells you exactly what to do based on performance**

### 4. `sector_analysis.py`
- `analyze_all_sectors()` - Rank all S&P 500 sectors
- `get_top_sectors(n=3)` - Get best sectors to trade
- `auto_trade_sectors()` - Automated sector rotation
- `compare_sector_momentum()` - Momentum rankings

### 5. `deeplearning_nmr.py`
- `NMRSignalProcessor` - Deep learning model framework
- `integrate_nmr_with_ensemble()` - Combine with Bayesian/GP
- **Template for your custom deep learning model**

### 6. `montecarlo_sims.py`
- `monte_carlo_strategy_simulation()` - Strategy P&L simulation
- `plot_monte_carlo_results()` - 4-panel visualization
- `print_monte_carlo_summary()` - Statistics summary
- **Answers: "How much money will I make?"**

### 7. `alpaca_exercises.py`
- `verify_alpaca_setup()` - Check API connectivity
- `connect_trading_client()` - Connect to Alpaca
- `market_order()`, `limit_order()` - Order execution
- Account management utilities

## Trading Workflow

```
1. Verify Setup
   └─> verify_alpaca_setup()

2. Analyze Market
   ├─> unified_bayesian_gp_forecast(ticker)  # Individual stocks
   └─> analyze_all_sectors()                  # Sector analysis

3. Check Risk
   ├─> monte_carlo_strategy_simulation()  # Estimate returns
   └─> adaptive_threshold_calculator()     # Optimize parameters

4. Get Guidance
   └─> trading_assistant()  # What should I do?

5. Execute
   └─> run_once()  # Place trades

6. Monitor & Optimize
   ├─> mcmc_optimize_thresholds()  # MCMC parameter tuning
   └─> trading_assistant()         # Continuous feedback
```

## Key Configuration Parameters

- `MIN_CONF`: Minimum confidence threshold (default 0.65)
- `MIN_PROB_UP`: Ensemble fallback (default 0.60)
- `MIN_Z`: Statistical edge requirement (default 0.20)
- `TP_PCT`: Take profit percentage (default 0.04 = 4%)
- `SL_PCT`: Stop loss percentage (default 0.02 = 2%)
- `MAX_ORDERS_PER_RUN`: Position limit (default 20)
- `DRY_RUN`: Test mode flag (default True)

## Best Practices

1. **Always start with DRY_RUN=True** to test without real trades
2. **Run verify_alpaca_setup()** before first execution
3. **Check trading_assistant()** daily for recommendations
4. **Use Monte Carlo** to quantify expected returns
5. **Optimize thresholds** with MCMC after 30+ trades
6. **Monitor sectors** with sector_analysis for diversification
7. **Backtest** with yfinance data before live trading

## Troubleshooting

**Q: "No data available for ticker"**  
A: Check ticker symbol is valid, try different period (e.g., "100d" instead of "200d")

**Q: "Insufficient buying power"**  
A: Reduce position sizes or enable USE_NOTIONAL for testing

**Q: "Signal confidence below threshold"**  
A: Lower MIN_CONF or check that forecast models are working

**Q: "MCMC optimization failing"**  
A: Need at least 30 trades in history, install emcee: `pip install emcee`

---

## References

- **Alpaca API Docs**: https://alpaca.markets/docs/
- **Yahoo Finance**: https://pypi.org/project/yfinance/
- **Bayesian Inference**: Bishop, Pattern Recognition and Machine Learning
- **Gaussian Processes**: Rasmussen & Williams, GPs for ML

---

*Last Updated: 2025-12-08*
