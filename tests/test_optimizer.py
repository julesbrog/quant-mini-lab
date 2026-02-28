# tests/test_optimizer.py
import numpy as np
import pandas as pd

from qmlab.optimizer import estimate_mean_cov, solve_mean_variance


def test_solve_mean_variance_constraints():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, size=(200, 4)), columns=list("ABCD"))

    mu, Sigma = estimate_mean_cov(r)
    w = solve_mean_variance(mu, Sigma, risk_aversion=10.0, long_only=True, w_max=0.6)

    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-8).all()
    assert (w <= 0.6 + 1e-6).all()


def test_turnover_constraint_respected():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, size=(250, 3)), columns=list("ABC"))

    mu, Sigma = estimate_mean_cov(r)
    w_prev = np.array([0.7, 0.2, 0.1])
    limit = 0.05

    w = solve_mean_variance(
        mu, Sigma,
        risk_aversion=15.0,
        long_only=True,
        w_max=1.0,
        w_prev=w_prev,
        turnover_limit=limit,
    )

    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-8).all()
    assert np.sum(np.abs(w - w_prev)) <= limit + 1e-5