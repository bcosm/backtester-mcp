import numpy as np
import pytest
from backtester_mcp.robustness import pbo, bootstrap_sharpe


def _make_trending_data(n=2000, n_strats=10, seed=42):
    """Strategies with real edge on trending data — should have low PBO."""
    rng = np.random.default_rng(seed)

    # trending prices
    trend = np.cumsum(rng.normal(0.001, 0.02, n)) + 100
    prices = np.abs(trend)

    returns_list = []
    for i in range(n_strats):
        # momentum strategies with similar lookbacks — all should work on a trend
        lookback = 10 + i * 5
        fast = np.convolve(prices, np.ones(lookback) / lookback, mode="full")[:n]
        slow = np.convolve(prices, np.ones(lookback * 3) / (lookback * 3), mode="full")[:n]
        signals = np.where(fast > slow, 1.0, -1.0)
        signals[:lookback * 3] = 0

        from backtester_mcp.engine import backtest
        result = backtest(prices, signals, fees=0.0, slippage=0.0)
        returns_list.append(result.returns)

    min_len = min(len(r) for r in returns_list)
    return np.column_stack([r[:min_len] for r in returns_list])


def _make_noise_data(n=2000, n_strats=20, seed=99):
    """Random strategies on random walks — should have high PBO."""
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    prices = np.abs(prices) + 50

    returns_list = []
    for _ in range(n_strats):
        signals = rng.choice([-1.0, 0.0, 1.0], size=n)
        from backtester_mcp.engine import backtest
        result = backtest(prices, signals, fees=0.0, slippage=0.0)
        returns_list.append(result.returns)

    min_len = min(len(r) for r in returns_list)
    return np.column_stack([r[:min_len] for r in returns_list])


def test_pbo_flags_overfit_strategies():
    matrix = _make_noise_data(n_strats=40)
    result = pbo(matrix, n_splits=10)
    # random strategies should not consistently rank well OOS
    assert result["pbo"] > 0.0


def test_pbo_low_for_genuine_edge():
    matrix = _make_trending_data()
    result = pbo(matrix, n_splits=10)
    # trending data with momentum strategies should have lower PBO
    assert result["pbo"] < 0.7


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.01, 1000)  # positive drift

    result = bootstrap_sharpe(returns, n_samples=5000)
    assert result["ci_lower"] <= result["sharpe"] <= result["ci_upper"]


def test_bootstrap_zero_returns():
    returns = np.zeros(500)
    result = bootstrap_sharpe(returns, n_samples=1000)
    assert result["sharpe"] == 0.0
    assert result["ci_includes_zero"]


def test_pbo_needs_at_least_two_strategies():
    with pytest.raises(ValueError):
        pbo(np.random.default_rng(0).normal(0, 1, (100, 1)))
