# Quant Mini Lab — Backtesting, Options, and Portfolio Optimization (Python)

A small, test-driven quantitative finance lab with three components:
1) **Backtesting** of simple multi-asset strategies with transaction costs and robustness checks  
2) **Options pricing** (Black–Scholes, Greeks, implied volatility) + Monte Carlo with variance reduction  
3) **Portfolio optimization** (constrained mean–variance in CVXPY) with rolling out-of-sample evaluation

The goal is correctness, reproducibility, and clear diagnostics (benchmarks, sanity checks, sensitivity).

---

## Project structure

- `src/qmlab/`  
  - `data.py` — Yahoo Finance data loader + returns + walk-forward splits  
  - `metrics.py` — performance + drawdowns + turnover  
  - `backtest.py` — signals, weights, transaction costs, backtest engine  
  - `bs.py` — Black–Scholes prices/Greeks, implied vol, Monte Carlo + variance reduction  
  - `optimizer.py` — constrained mean–variance + rolling OOS backtest  
- `tests/` — `pytest` unit tests for key components  
- `notebooks/`  
  - `01_backtest.ipynb`  
  - `02_options_pricing.ipynb`  
  - `03_portfolio_optimization.ipynb`

---

## Key findings (from notebooks)

### 1) Backtesting (SPY, QQQ, IWM, TLT, GLD — 2015–2024)
- **TSMOM (sign-based), L=60, costs=5 bps**: small positive drift but weak Sharpe; moderate drawdowns; turnover ~0.10.  
- **Mean Reversion (z-score trigger), L=20, z=1.0**: structurally loss-making, large drawdowns; high turnover (~0.42) and strong sensitivity to costs.  
- **Benchmarks dominate** on this sample (SPY and equal-weight are hard to beat in a long-equity-dominated decade).  
- Sensitivity analysis shows **longer lookbacks reduce turnover** and improve cost-robustness for TSMOM.

### 2) Options pricing
- Black–Scholes call/put prices pass **put–call parity** to numerical precision (~1e−14).  
- **Implied vol inversion** recovers the original σ to numerical precision (~1e−14).  
- Monte Carlo estimators converge toward BS; variance reduction (antithetic + control variate) often reduces error for a fixed path budget.

### 3) Portfolio optimization (constrained mean–variance, rolling OOS)
- Constrained mean–variance optimizer (long-only, cap, turnover constraint) delivers a reasonable risk-adjusted profile with **very low turnover** (~0.01).  
- Constraints are verified empirically (fully invested, long-only, cap respected, turnover bounded).  
- Equal-weight remains a strong baseline on this small universe; SPY has higher return with higher volatility/drawdown.

---

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt