import numpy as np
import pandas as pd

from qmlab.backtest import signal_to_weights, apply_transaction_costs, backtest


def test_signal_to_weights_gross_exposure_1_when_active():
    idx = pd.date_range("2020-01-01", periods=2)
    sig = pd.DataFrame({"A": [1, -1], "B": [1, 0]}, index=idx)
    w = signal_to_weights(sig)

    # day1: two active => weights 0.5,0.5 => gross 1.0
    assert np.isclose(w.iloc[0].abs().sum(), 1.0)
    # day2: one active => gross 1.0
    assert np.isclose(w.iloc[1].abs().sum(), 1.0)


def test_apply_transaction_costs_nonnegative():
    idx = pd.date_range("2020-01-01", periods=3)
    w = pd.DataFrame({"A": [1.0, 0.0, 1.0]}, index=idx)
    costs = apply_transaction_costs(w, cost_bps=10.0)
    assert (costs >= 0).all()


def test_backtest_uses_shifted_weights():
    idx = pd.date_range("2020-01-01", periods=3)
    r = pd.DataFrame({"A": [0.10, 0.00, 0.00]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=idx)

    # If it used same-day weights, first return would be 0.10.
    # With shift, first day uses 0 weight => 0 return.
    rp = backtest(r, w, cost_bps=0.0)
    assert np.isclose(rp.iloc[0], 0.0)