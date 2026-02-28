import numpy as np
import pandas as pd

from qmlab.metrics import (
    equity_curve,
    drawdown_series,
    max_drawdown,
    annualized_return,
    annualized_vol,
    sharpe,
    turnover,
)


def test_equity_curve_constant_returns():
    r = pd.Series([0.01, 0.01, 0.01], index=pd.date_range("2020-01-01", periods=3))
    eq = equity_curve(r, start=1.0)
    assert np.isclose(eq.iloc[-1], (1.01) ** 3)


def test_drawdown_and_max_drawdown_simple():
    # Equity: 1 -> 1.1 -> 1.0 -> 1.2
    r = pd.Series([0.10, -0.0909090909, 0.20], index=pd.date_range("2020-01-01", periods=3))
    dd = drawdown_series(r)
    mdd = max_drawdown(r)
    assert dd.min() <= 0
    assert np.isclose(mdd, dd.min())
    # peak after first return is 1.1, then equity goes to 1.0 => dd = 1.0/1.1 - 1 = -0.090909...
    assert np.isclose(mdd, -0.0909090909, atol=1e-6)


def test_annualized_vol_zero_for_constant_returns():
    r = pd.Series([0.01] * 50)
    assert np.isclose(annualized_vol(r), 0.0)


def test_sharpe_nan_when_zero_vol():
    r = pd.Series([0.0] * 100)
    assert np.isnan(sharpe(r))


def test_turnover_basic():
    idx = pd.date_range("2020-01-01", periods=3)
    w = pd.DataFrame(
        {"A": [0.5, 0.5, 0.0], "B": [0.5, 0.5, 1.0]},
        index=idx,
    )
    tw = turnover(w)
    assert tw.iloc[0] == 0.0
    # turnover on last date: |0-0.5| + |1-0.5| = 1.0
    assert np.isclose(tw.iloc[-1], 1.0)