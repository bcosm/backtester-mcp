import numpy as np
import pytest
from backtester_mcp.orderbook import (
    estimate_spread_roll,
    estimate_spread_corwin_schultz,
    FillModel,
)


def test_roll_spread_positive_on_bouncy_data():
    """Data with bid-ask bounce should produce a positive spread estimate."""
    rng = np.random.default_rng(42)
    n = 5000
    # flat mid so autocovariance is dominated by bounce, not drift
    mid = np.full(n, 100.0) + rng.normal(0, 0.01, n)
    spread = 1.0
    bounce = rng.choice([-1, 1], n) * spread / 2
    prices = mid + bounce

    estimated = estimate_spread_roll(prices)
    assert estimated > 0


def test_corwin_schultz_positive():
    rng = np.random.default_rng(42)
    n = 500
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.abs(close) + 50
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))

    spread = estimate_spread_corwin_schultz(high, low)
    assert spread >= 0


def test_fill_price_worse_than_mid():
    """Buy fills should be above mid, sell fills below."""
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0, 0.5, 500))
    prices = np.abs(prices) + 50
    volume = rng.integers(100_000, 1_000_000, 500).astype(float)

    model = FillModel(prices, volume=volume)

    mid = 100.0
    buy_fills = [model.fill_price(mid, 1000, is_buy=True) for _ in range(100)]
    sell_fills = [model.fill_price(mid, 1000, is_buy=False) for _ in range(100)]

    # on average, buys should be above mid and sells below
    assert np.mean(buy_fills) > mid
    assert np.mean(sell_fills) < mid


def test_larger_orders_have_more_impact():
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0, 0.5, 500))
    prices = np.abs(prices) + 50
    volume = np.full(500, 1_000_000.0)

    model = FillModel(prices, volume=volume)

    mid = 100.0
    small_fills = [model.fill_price(mid, 100, is_buy=True) for _ in range(200)]
    large_fills = [model.fill_price(mid, 500_000, is_buy=True) for _ in range(200)]

    # larger orders should have worse (higher) average fill
    assert np.mean(large_fills) > np.mean(small_fills)
