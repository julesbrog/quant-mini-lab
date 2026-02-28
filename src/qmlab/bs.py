# src/qmlab/bs.py
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _validate_inputs(S: float, K: float, T: float, sigma: float):
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    if T == 0 or sigma == 0:
        # Convention: d1/d2 not used in that case; caller should handle T==0 or sigma==0
        return float("nan"), float("nan")
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option: str = "call",
    q: float = 0.0,
) -> float:
    """
    Black–Scholes price with continuous dividend yield q.

    option in {"call","put"}
    """
    _validate_inputs(S, K, T, sigma)
    option = option.lower()
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'.")

    if T == 0:
        payoff = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
        return float(payoff)

    if sigma == 0:
        # Deterministic forward
        fwd = S * math.exp((r - q) * T)
        disc = math.exp(-r * T)
        payoff = max(fwd - K, 0.0) if option == "call" else max(K - fwd, 0.0)
        return float(disc * payoff)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option == "call":
        return float(disc_q * S * norm.cdf(d1) - disc_r * K * norm.cdf(d2))
    else:
        return float(disc_r * K * norm.cdf(-d2) - disc_q * S * norm.cdf(-d1))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option: str = "call",
    q: float = 0.0,
) -> dict:
    """
    Returns Greeks: delta, gamma, vega, theta, rho.

    Notes
    -----
    - theta returned per YEAR (consistent with r and T in years).
    - vega is per 1.0 change in vol (not per 1%); divide by 100 if you want per 1%.
    """
    _validate_inputs(S, K, T, sigma)
    option = option.lower()
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'.")

    if T == 0 or sigma == 0:
        # Greeks are not well-behaved at expiry; return NaNs except delta in some cases
        return {"delta": float("nan"), "gamma": float("nan"), "vega": float("nan"),
                "theta": float("nan"), "rho": float("nan")}

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    sqrtT = math.sqrt(T)

    if option == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -disc_q * (S * pdf_d1 * sigma) / (2 * sqrtT)
            - r * disc_r * K * norm.cdf(d2)
            + q * disc_q * S * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (
            -disc_q * (S * pdf_d1 * sigma) / (2 * sqrtT)
            + r * disc_r * K * norm.cdf(-d2)
            - q * disc_q * S * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = disc_q * S * pdf_d1 * sqrtT

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option: str = "call",
    q: float = 0.0,
    sigma_low: float = 1e-8,
    sigma_high: float = 5.0,
) -> float:
    """Solve sigma such that bs_price(..., sigma) == price using Brent."""
    if price <= 0:
        raise ValueError("Option price must be positive.")
    _validate_inputs(S, K, T, sigma=1e-8)

    if T == 0:
        raise ValueError("Implied vol undefined at expiry (T=0).")

    option = option.lower()
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'.")

    def f(sig: float) -> float:
        return bs_price(S, K, T, r, sig, option=option, q=q) - price

    # Expand bracket if needed (rare but can happen)
    a, b = sigma_low, sigma_high
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        b = 10.0
        fb = f(b)
        if fa * fb > 0:
            raise ValueError("Could not bracket implied volatility. Check input price.")

    return float(brentq(f, a, b, maxiter=200, xtol=1e-10))


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option: str = "call",
    q: float = 0.0,
    n_paths: int = 200_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = 0,
) -> float:
    """
    Monte Carlo pricer under risk-neutral measure:
      S_T = S * exp((r-q-0.5*sigma^2)T + sigma*sqrt(T)*Z)

    Variance reduction:
      - antithetic: uses Z and -Z
      - control variate: uses S_T with known E[S_T] = S*exp((r-q)T)
    """
    _validate_inputs(S, K, T, sigma)
    option = option.lower()
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'.")
    if T == 0:
        payoff = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
        return float(payoff)
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    rng = np.random.default_rng(seed)

    m = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal(m)
    if antithetic:
        Z = np.concatenate([Z, -Z])
    if len(Z) < n_paths:
        # if n_paths odd, pad one sample
        Z = np.concatenate([Z, rng.standard_normal(1)])

    drift = (r - q - 0.5 * sigma * sigma) * T
    diff = sigma * np.sqrt(T) * Z
    ST = S * np.exp(drift + diff)

    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc = np.exp(-r * T)
    Y = disc * payoff

    if control_variate:
        # Control variate with ST (undiscounted), or use discounted ST*exp(-rT)
        # We'll use discounted ST (so expectation is known under Q):
        X = disc * ST
        EX = disc * (S * np.exp((r - q) * T))  # E[ST] discounted by exp(-rT)
        Xc = X - EX

        var_X = np.var(Xc, ddof=1)
        if var_X > 0:
            cov_YX = np.cov(Y, Xc, ddof=1)[0, 1]
            b = cov_YX / var_X
            Y = Y - b * Xc

    return float(np.mean(Y))