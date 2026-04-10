"""Shared test fixtures."""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def trending_prices():
    """Price series with a clear uptrend — strategies should perform well."""
    rng = np.random.default_rng(42)
    n = 500
    drift = 0.0003
    noise = rng.normal(0, 0.01, n)
    log_returns = drift + noise
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    return prices


@pytest.fixture
def random_prices():
    """Random walk prices — no edge to exploit."""
    rng = np.random.default_rng(99)
    n = 500
    log_returns = rng.normal(0, 0.01, n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    return prices


@pytest.fixture
def simple_signals(trending_prices):
    """Momentum signals on trending data."""
    n = len(trending_prices)
    signals = np.zeros(n)
    for i in range(50, n):
        fast = np.mean(trending_prices[i-10:i])
        slow = np.mean(trending_prices[i-50:i])
        signals[i] = 1.0 if fast > slow else -1.0
    return signals


@pytest.fixture
def ohlcv_arrays(trending_prices):
    """Synthetic OHLCV from close prices."""
    rng = np.random.default_rng(42)
    close = trending_prices
    high = close * (1 + rng.uniform(0, 0.02, len(close)))
    low = close * (1 - rng.uniform(0, 0.02, len(close)))
    open_ = close * (1 + rng.normal(0, 0.005, len(close)))
    volume = rng.uniform(1e6, 5e6, len(close))
    return {"open": open_, "high": high, "low": low,
            "close": close, "volume": volume}


@pytest.fixture
def spy_data_path():
    p = Path("datasets/spy_daily.parquet")
    if p.exists():
        return str(p)
    pytest.skip("spy_daily.parquet not available")


@pytest.fixture
def momentum_strategy_path():
    p = Path("strategies/momentum.py")
    if p.exists():
        return str(p)
    pytest.skip("momentum.py not available")


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path
