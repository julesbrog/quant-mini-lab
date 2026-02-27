from __future__ import annotations

import numpy as np
import pandas as pd


def _as_clean_series(x: pd.Series | np.ndarray | list[float]) -> pd.Series:
    """Convert input to a numeric pd.Series and drop NaNs."""
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)

    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def equity_curve(r: pd.Series, start: float = 1.0) -> pd.Series:
    """
    Build an equity curve from simple returns.

    equity_t = start * Π_{i<=t} (1 + r_i)

    Notes
    -----
    - Expects simple returns (not log returns).
    - Drops NaNs.
    """
    s = _as_clean_series(r)
    if s.empty:
        return pd.Series(dtype=float)

    if (s <= -1.0).any():
        raise ValueError("Found returns <= -100%. Equity curve is undefined.")

    eq = start * (1.0 + s).cumprod()
    eq.name = "equity"
    return eq


def drawdown_series(r: pd.Series, start: float = 1.0) -> pd.Series:
    """
    Drawdown series from returns.

    dd_t = equity_t / max_{u<=t}(equity_u) - 1
    """
    eq = equity_curve(r, start=start)
    if eq.empty:
        return pd.Series(dtype=float)

    peak = eq.cummax()
    dd = eq / peak - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(r: pd.Series, start: float = 1.0) -> float:
    """Maximum drawdown (most negative drawdown)."""
    dd = drawdown_series(r, start=start)
    if dd.empty:
        return float("nan")
    return float(dd.min())


def annualized_return(r: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized return using geometric compounding (CAGR-like).

    ann_ret = (Π (1+r))^(periods_per_year / n) - 1
    """
    s = _as_clean_series(r)
    if s.empty:
        return float("nan")
    if (s <= -1.0).any():
        raise ValueError("Found returns <= -100%. Annualized return is undefined.")

    growth = (1.0 + s).prod()
    n = len(s)
    return float(growth ** (periods_per_year / n) - 1.0)


def annualized_vol(r: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized volatility: std(r) * sqrt(periods_per_year)."""
    s = _as_clean_series(r)
    if len(s) < 2:
        return float("nan")
    return float(s.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(
    r: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe ratio (annualized), assuming rf_annual is an annual simple rate.

    excess_daily ≈ r - rf_annual / periods_per_year
    sharpe = mean(excess_daily) / std(excess_daily) * sqrt(periods_per_year)
    """
    s = _as_clean_series(r)
    if len(s) < 2:
        return float("nan")

    rf_daily = rf_annual / periods_per_year
    ex = s - rf_daily

    vol = ex.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")

    return float(ex.mean() / vol * np.sqrt(periods_per_year))


def turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Daily turnover from a weights DataFrame.

    turnover_t = Σ_i |w_{t,i} - w_{t-1,i}|

    Notes
    -----
    - Treats NaNs as 0 weights.
    - First date turnover is 0.
    """
    if weights is None or weights.empty:
        return pd.Series(dtype=float)

    w = weights.copy()
    w = w.apply(pd.to_numeric, errors="coerce").astype(float)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tw = w.diff().abs().sum(axis=1)
    if len(tw) > 0:
        tw.iloc[0] = 0.0
    tw.name = "turnover"
    return tw


def summarize_performance(
    r: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Convenience helper: return key metrics in a dict."""
    s = _as_clean_series(r)
    if s.empty:
        return {
            "annualized_return": float("nan"),
            "annualized_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    return {
        "annualized_return": annualized_return(s, periods_per_year=periods_per_year),
        "annualized_vol": annualized_vol(s, periods_per_year=periods_per_year),
        "sharpe": sharpe(s, rf_annual=rf_annual, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(s),
    }