"""Tests for portfolio backtesting."""

import numpy as np
import pytest

from backtester_mcp.portfolio import backtest_portfolio


def test_single_component(trending_prices, simple_signals):
    result = backtest_portfolio([{
        "prices": trending_prices,
        "signals": simple_signals,
        "weight": 1.0,
        "name": "momentum",
    }])
    assert "portfolio_metrics" in result
    assert "component_metrics" in result
    assert len(result["component_metrics"]) == 1
    assert result["diversification_ratio"] == pytest.approx(1.0, abs=0.01)


def test_two_components(trending_prices, simple_signals, random_prices):
    rng = np.random.default_rng(77)
    random_signals = rng.choice([-1.0, 0.0, 1.0], size=len(random_prices))

    result = backtest_portfolio([
        {"prices": trending_prices, "signals": simple_signals,
         "weight": 0.6, "name": "momentum"},
        {"prices": random_prices, "signals": random_signals,
         "weight": 0.4, "name": "random"},
    ])
    assert len(result["component_metrics"]) == 2
    assert len(result["correlation_matrix"]) == 2
    assert result["diversification_ratio"] > 0


def test_weights_normalize(trending_prices, simple_signals):
    result = backtest_portfolio([
        {"prices": trending_prices, "signals": simple_signals,
         "weight": 3.0, "name": "a"},
        {"prices": trending_prices, "signals": simple_signals,
         "weight": 7.0, "name": "b"},
    ])
    # weights should sum to 1 after normalization
    total = sum(c["weight"] for c in result["component_metrics"])
    assert pytest.approx(total) == 1.0


def test_empty_components():
    with pytest.raises(ValueError, match="at least one"):
        backtest_portfolio([])
