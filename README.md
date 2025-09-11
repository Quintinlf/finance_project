# 💹 Finance Project

Lightweight Python tools for portfolio inspection, return forecasting, technical indicators, and basic backtesting. Demonstrates skills in quantitative finance, data analysis, machine learning, and automation. Uses Yahoo Finance for data access.

---

## 📈 Project Overview

- Bayesian ML models and statistical methods for forecasting and portfolio analysis.  
- End-to-end pipeline: data ingestion → feature processing → predictions → simulations.  
- Interactive charts, technical indicators, and backtesting scaffolding for notebook-friendly workflow.  
- Designed for **real-world financial analysis and automation**, portfolio-ready code.

---

## 🔑 Features

- **Portfolio inspection:** Tabular + summary views  
  - `display_portfolio_ohlcv_heads(tickers, period, interval)` – fetch and display recent OHLCV per ticker  
- **Interactive charts:**  
  - `create_individual_portfolio_charts()` – Plotly-based per-ticker closing price charts  
- **Technical indicators:**  
  - `rsi_calculator(df)` – add RSI to a DataFrame  
- **Forecasting:**  
  - `forecast_next_day_return(ticker)` – next-day return model with lagged returns, volatility, RSI  
- **Backtesting scaffolding:**  
  - `Position` and `Strategy` classes for trade simulation  
- **Quant utilities:**  
  - `gbm()` – geometric Brownian motion paths  
  - `bsformula()` – Black–Scholes pricing  
  - `update_beliefs_with_data()`, `g1()`, `g0()` – simple Bayesian helpers  

---

## 📂 Repository Structure

- `algo_bot/` — main module code  
  - `trading_functions.py` – core toolkit  
  - `api_functions.py` – Alpaca API placeholders  
- `exercises/` — math and financial practice pre-implementation 
- `fintech.md` — personal notes  

---
# 🧑‍💼 Author & Notes

## About the Author
I’m **Quintin**, a quantitative finance and Python enthusiast focused on building real-world trading automation tools. This repository demonstrates my skills in:

- Quantitative finance & portfolio analysis  
- Machine learning for financial forecasting  
- Python programming & automation  
- Interactive visualization & data pipelines  

**GitHub:** [Quintinlf](https://github.com/Quintinlf)  
**License:** MIT License — free to use with attribution

---

## Notes
- **Data Source:** Yahoo Finance via `yfinance`  
- **Recommended Environment:** Jupyter Notebook or VS Code for best visualization experience  
- **Purpose:** Educational and portfolio demonstration; not investment advice  
- **Dependencies:** Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, yfinance, ipython (optional), alpaca-trade-api (optional)  



## 🚀 Setup & About Me

Python 3.9+ recommended.

```bash
git clone https://github.com/Quintinlf/finance_project.git
pip install yfinance pandas numpy matplotlib scipy plotly
# Optional for notebooks:
pip install ipython
# Optional for Alpaca API:
pip install alpaca-trade-api python-dotenv
