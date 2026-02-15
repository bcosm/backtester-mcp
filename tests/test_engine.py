import numpy as np
import pytest
from backtester_mcp.engine import backtest


def test_buy_and_hold_matches_simple_return():
    """Buy-and-hold should roughly match the raw price return minus costs."""
    prices = np.linspace(100, 150, 252)  # steady uptrend
    signals = np.ones(252)

    result = backtest(prices, signals, fees=0.0, slippage=0.0)

    # with zero costs, final equity / initial should match price return
    expected_return = prices[-1] / prices[0] - 1
    actual_return = result.metrics["total_return"]
    assert abs(actual_return - expected_return) < 0.01


def test_fees_reduce_returns():
    prices = np.linspace(100, 120, 252)
    signals = np.ones(252)

    no_fees = backtest(prices, signals, fees=0.0, slippage=0.0)
    with_fees = backtest(prices, signals, fees=0.01, slippage=0.005)

    assert with_fees.metrics["total_return"] < no_fees.metrics["total_return"]


def test_all_cash_returns_zero():
    prices = np.linspace(100, 200, 252)
    signals = np.zeros(252)

    result = backtest(prices, signals)
    assert result.metrics["total_return"] == pytest.approx(0.0, abs=0.001)
    assert result.metrics["num_trades"] == 0


def test_short_signal_profits_on_decline():
    prices = np.linspace(100, 60, 252)  # steady downtrend
    signals = -np.ones(252)

    result = backtest(prices, signals, fees=0.0, slippage=0.0)
    assert result.metrics["total_return"] > 0.2  # shorts profit on declines


def test_equity_curve_length_matches_prices():
    prices = np.random.default_rng(0).normal(100, 1, 500).cumsum()
    prices = np.abs(prices) + 50
    signals = np.ones(500)

    result = backtest(prices, signals)
    assert len(result.equity_curve) == 500
    assert len(result.returns) == 500


def test_backtest_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        backtest(np.ones(100), np.ones(50))
