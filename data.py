from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_prices_yf(
    tickers: List[str],
    start: str,
    end: str,
    price_field: str = "Adj Close",
) -> pd.DataFrame:
    """
    Download daily prices from Yahoo Finance using yfinance.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols (e.g., ["SPY", "QQQ"]).
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    price_field : str
        Which field to use among: "Adj Close", "Close", "Open", "High", "Low".

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date (DatetimeIndex), columns = tickers, float prices.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")
    if price_field not in {"Adj Close", "Close", "Open", "High", "Low"}:
        raise ValueError(f"Unsupported price_field: {price_field}")

    try:
        import yfinance as yf  # local import to keep module lightweight
    except ImportError as e:
        raise ImportError(
            "yfinance is required for load_prices_yf. Install it with: pip install yfinance"
        ) from e

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if raw is None or raw.empty:
        raise ValueError("No data returned by yfinance. Check tickers and date range.")

    # yfinance returns:
    # - MultiIndex columns when multiple tickers: (Field, Ticker)
    # - Single-level columns when one ticker: Field columns
    if isinstance(raw.columns, pd.MultiIndex):
        if price_field not in raw.columns.get_level_values(0):
            raise ValueError(f"Field '{price_field}' not found in downloaded data.")
        prices = raw[price_field].copy()
    else:
        if price_field not in raw.columns:
            raise ValueError(f"Field '{price_field}' not found in downloaded data.")
        prices = raw[[price_field]].copy()
        prices.columns = tickers[:1]  # name the single column with the ticker

    # Clean index and values
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.apply(pd.to_numeric, errors="coerce").astype(float)

    # Drop rows where all tickers are missing
    prices = prices.dropna(how="all")

    # Light fill for occasional missing values (holidays are fine; we keep the index as-is)
    prices = prices.ffill(limit=3)

    # Keep only requested tickers and consistent column order
    # (yfinance sometimes returns extra columns ordering)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Missing tickers in downloaded data: {missing}")

    prices = prices[tickers]

    return prices


def compute_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """
    Compute returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Price levels indexed by date.
    kind : str
        "simple" for simple returns, "log" for log-returns.

    Returns
    -------
    pd.DataFrame
        Returns aligned to the same columns; first row is dropped due to differencing.
    """
    if prices is None or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame.")
    if kind not in {"simple", "log"}:
        raise ValueError("kind must be either 'simple' or 'log'.")

    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce").astype(float)

    if kind == "simple":
        rets = prices.pct_change()
    else:
        # log returns require strictly positive prices
        if (prices <= 0).any().any():
            raise ValueError("Log returns require strictly positive prices.")
        rets = np.log(prices).diff()

    # Drop rows that are all NaN (first row typically)
    rets = rets.dropna(how="all")

    return rets


def train_test_splits(
    df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Rolling walk-forward splits using index positions (not calendar days).

    Example:
      - train_days=252, test_days=63, step_days=63
      - Split 1: [0:252] train, [252:315] test
      - Split 2: [63:315] train, [315:378] test
      - etc.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed data (prices or returns).
    train_days : int
        Number of rows in training window.
    test_days : int
        Number of rows in test window.
    step_days : int
        Step size to move the window forward each split.

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        List of (train_df, test_df) splits.
    """
    if df is None or df.empty:
        raise ValueError("df must be a non-empty DataFrame.")
    for name, v in [("train_days", train_days), ("test_days", test_days), ("step_days", step_days)]:
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    df = df.sort_index()
    n = len(df)

    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    start = 0

    while True:
        train_start = start
        train_end = train_start + train_days
        test_end = train_end + test_days

        if test_end > n:
            break

        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        splits.append((train_df, test_df))
        start += step_days

    return splits


