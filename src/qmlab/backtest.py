# src/qmlab/backtest.py
from __future__ import annotations

import numpy as np
import pandas as pd


def signal_tsmom(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Time-series momentum signal:
      signal_t = sign( sum_{i=1..lookback} log(1 + r_{t-i}) )

    Output in {-1, 0, +1}.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    r = returns.copy().astype(float)

    if (r <= -1.0).any().any():
        raise ValueError("Returns contain values <= -100%, log(1+r) undefined.")

    log_r = np.log1p(r)
    score = log_r.rolling(lookback).sum()
    sig = np.sign(score)
    sig = sig.replace(0.0, 0.0)
    return sig.fillna(0.0)


def signal_meanrev(returns: pd.DataFrame, lookback: int, z_entry: float = 1.0) -> pd.DataFrame:
    """
    Simple mean-reversion signal on cumulative return:
      z_t = roll_sum(r) / (roll_std(r) * sqrt(lookback))
      signal = -sign(z) if |z| > z_entry else 0
    """
    if lookback <= 1:
        raise ValueError("lookback must be >= 2.")
    if z_entry <= 0:
        raise ValueError("z_entry must be positive.")

    r = returns.copy().astype(float)
    roll_sum = r.rolling(lookback).sum()
    roll_std = r.rolling(lookback).std(ddof=1)

    z = roll_sum / (roll_std * np.sqrt(lookback))
    sig = -np.sign(z)
    sig = sig.where(z.abs() > z_entry, 0.0)
    return sig.fillna(0.0)


def signal_to_weights(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Convert signals {-1,0,1} to weights with unit gross exposure:

      w_{t,i} = signal_{t,i} / N_active_t

    where N_active_t = number of non-zero signals that day.
    """
    s = signal.copy().astype(float).fillna(0.0)
    active = (s != 0.0).sum(axis=1).replace(0, np.nan)
    w = s.div(active, axis=0).fillna(0.0)
    return w


def apply_transaction_costs(weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    """
    Transaction cost series based on turnover:
      turnover_t = sum_i |w_t - w_{t-1}|
      cost_rate = cost_bps / 1e4
      costs applied to next day's return in `backtest` via shift(1).
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative.")

    w = weights.copy().astype(float).fillna(0.0)
    tw = (w.diff().abs()).sum(axis=1)
    if len(tw) > 0:
        tw.iloc[0] = 0.0
    cost_rate = cost_bps / 1e4
    costs = cost_rate * tw
    costs.name = "costs"
    return costs


def backtest(returns: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 0.0) -> pd.Series:
    """
    Portfolio return:
      r_p,t = sum_i w_{t-1,i} * r_{t,i} - costs_{t-1}

    Notes
    -----
    - weights are shifted by 1 to avoid lookahead
    - costs are computed from weight changes and applied with a shift(1)
    """
    r = returns.copy().astype(float)
    w = weights.copy().astype(float).fillna(0.0)

    # Align
    r, w = r.align(w, join="inner", axis=0)

    w_used = w.shift(1).fillna(0.0)
    gross = (w_used * r).sum(axis=1)

    costs = apply_transaction_costs(w, cost_bps=cost_bps).shift(1).fillna(0.0)
    net = gross - costs
    net.name = "portfolio_return"
    return net