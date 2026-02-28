# src/qmlab/optimizer.py
from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp


def estimate_mean_cov(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    r = returns.copy().astype(float).dropna(how="any")
    if r.empty:
        raise ValueError("Not enough data to estimate mean/cov.")
    mu = r.mean().values  # daily mean
    Sigma = np.cov(r.values, rowvar=False, ddof=1)
    return mu, Sigma


def _make_psd(Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Small ridge to avoid numerical issues
    n = Sigma.shape[0]
    return Sigma + eps * np.eye(n)


def solve_mean_variance(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_aversion: float = 10.0,
    long_only: bool = True,
    w_max: float | None = 0.35,
    w_prev: np.ndarray | None = None,
    turnover_limit: float | None = None,
) -> np.ndarray:
    """
    Solve:
      max  mu^T w - (risk_aversion/2) w^T Sigma w
      s.t. sum(w)=1
           w>=0 (optional)
           w<=w_max (optional)
           ||w - w_prev||_1 <= turnover_limit (optional, needs w_prev)
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)

    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError("Sigma must be (n,n) matching mu length.")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")
    if w_max is not None and (w_max <= 0 or w_max > 1.0):
        raise ValueError("w_max must be in (0,1].")

    Sigma_psd = _make_psd(Sigma)

    w = cp.Variable(n)

    quad = cp.quad_form(w, Sigma_psd)
    obj = cp.Maximize(mu @ w - 0.5 * risk_aversion * quad)

    constraints = [cp.sum(w) == 1]

    if long_only:
        constraints.append(w >= 0)

    if w_max is not None:
        constraints.append(w <= w_max)

    if turnover_limit is not None:
        if w_prev is None:
            raise ValueError("turnover_limit set but w_prev is None.")
        w_prev = np.asarray(w_prev, dtype=float).reshape(-1)
        if w_prev.shape[0] != n:
            raise ValueError("w_prev length must match mu.")
        if turnover_limit <= 0:
            raise ValueError("turnover_limit must be positive.")
        constraints.append(cp.norm1(w - w_prev) <= turnover_limit)

    prob = cp.Problem(obj, constraints)

    # Try a couple solvers for robustness
    try:
        prob.solve(solver=cp.ECOS, warm_start=True)
    except Exception:
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
        except Exception:
            prob.solve(solver=cp.CLARABEL, warm_start=True)

    if w.value is None:
        raise RuntimeError("Optimization failed to produce a solution.")

    sol = np.asarray(w.value, dtype=float).reshape(-1)

    # Clean numerical noise
    sol[np.abs(sol) < 1e-12] = 0.0
    # Re-normalize (sometimes tiny drift)
    s = sol.sum()
    if s != 0:
        sol = sol / s

    return sol


def rolling_backtest_optimizer(
    returns: pd.DataFrame,
    train_days: int = 252,
    rebalance_every: int = 21,
    cost_bps: float = 0.0,
    drop_warmup: bool = True,
    **solver_kwargs,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Rolling OOS backtest:
      - At i = train_days, train on [i-train_days : i)
      - Compute weights w_i
      - Hold for rebalance_every days (weights forward-filled)
      - Portfolio return uses weights_{t-1} * returns_t (no lookahead)

    If drop_warmup=True, returns only the period where weights are available
    (i.e., after the first optimization date).
    """
    r = returns.copy().astype(float).dropna(how="any")
    if len(r) <= train_days + 1:
        raise ValueError("Not enough data for rolling backtest.")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive.")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative.")

    idx = r.index
    cols = r.columns
    n = len(cols)

    # Use NaN until the first allocation, then forward-fill weights between rebalances
    weights = pd.DataFrame(np.nan, index=idx, columns=cols)
    w_prev = np.ones(n) / n

    for i in range(train_days, len(r), rebalance_every):
        train = r.iloc[i - train_days : i]
        mu, Sigma = estimate_mean_cov(train)

        w = solve_mean_variance(mu, Sigma, w_prev=w_prev, **solver_kwargs)

        end = min(i + rebalance_every, len(r))
        weights.iloc[i:end] = w
        w_prev = w

    # Carry last weights forward until next rebalance
    weights = weights.ffill()

    if drop_warmup:
        first_active = weights.dropna(how="all").index[0]
        weights = weights.loc[first_active:]
        r = r.loc[first_active:]

    # Portfolio return with 1-day shift to avoid lookahead
    w_used = weights.shift(1).fillna(0.0)
    r, w_used = r.align(w_used, join="inner", axis=0)

    gross = (w_used * r).sum(axis=1)

    # Transaction costs from turnover (computed on target weights)
    if cost_bps > 0:
        tw = weights.diff().abs().sum(axis=1)
        if len(tw) > 0:
            tw.iloc[0] = 0.0
        costs = (cost_bps / 1e4) * tw
        gross = gross - costs.shift(1).fillna(0.0)

    gross.name = "portfolio_return"
    return gross, weights