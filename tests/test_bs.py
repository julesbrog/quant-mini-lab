# tests/test_bs.py
import numpy as np

from qmlab.bs import bs_price, implied_vol


def test_put_call_parity():
    S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.02, 0.01, 0.25
    C = bs_price(S, K, T, r, sigma, option="call", q=q)
    P = bs_price(S, K, T, r, sigma, option="put", q=q)

    lhs = C - P
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert np.isclose(lhs, rhs, atol=1e-8)


def test_implied_vol_inverts_bs():
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.01, 0.0, 0.30
    price = bs_price(S, K, T, r, sigma, option="call", q=q)
    iv = implied_vol(price, S, K, T, r, option="call", q=q)
    assert np.isclose(iv, sigma, atol=1e-6)


def test_call_price_increases_with_S():
    K, T, r, q, sigma = 100.0, 1.0, 0.01, 0.0, 0.2
    c1 = bs_price(90.0, K, T, r, sigma, option="call", q=q)
    c2 = bs_price(110.0, K, T, r, sigma, option="call", q=q)
    assert c2 > c1