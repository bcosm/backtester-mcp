"""Tests for metrics module."""

import numpy as np
import pytest

from backtester_mcp.metrics import (
    sharpe, sortino, max_drawdown, max_drawdown_duration,
    calmar, win_rate, profit_factor, total_return, cagr, volatility,
    compute_all, TRADING_DAYS,
)


def test_sharpe_zero_returns():
    returns = np.zeros(100)
    assert sharpe(returns) == 0.0


def test_sharpe_positive_consistent():
    returns = np.full(252, 0.001)
    s = sharpe(returns)
    assert s > 10  # very consistent positive returns -> high Sharpe


def test_sortino_no_downside():
    returns = np.full(100, 0.01)
    assert sortino(returns) == float("inf")


def test_sortino_all_negative():
    returns = np.full(100, -0.01)
    s = sortino(returns)
    assert s < 0


def test_max_drawdown_no_drawdown():
    equity = np.linspace(100, 200, 100)
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_known():
    equity = np.array([100, 110, 90, 95, 80, 120])
    dd = max_drawdown(equity)
    # peak at 110, trough at 80 -> -27.27%
    assert pytest.approx(dd, abs=0.01) == -0.2727


def test_max_drawdown_duration_known():
    equity = np.array([100, 110, 105, 95, 100, 115, 120])
    d = max_drawdown_duration(equity)
    assert d == 3  # 110 -> 105 -> 95 -> 100 (3 bars below peak)


def test_win_rate_known():
    returns = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.005])
    # non-zero: 0.01, -0.02, 0.03, -0.01, 0.005 = 3 wins / 5 trades
    assert pytest.approx(win_rate(returns)) == 0.6


def test_profit_factor_known():
    returns = np.array([0.10, -0.05, 0.08, -0.03])
    pf = profit_factor(returns)
    assert pytest.approx(pf) == (0.10 + 0.08) / (0.05 + 0.03)


def test_profit_factor_no_losses():
    returns = np.array([0.01, 0.02, 0.03])
    assert profit_factor(returns) == float("inf")


def test_total_return_known():
    equity = np.array([100, 150])
    assert pytest.approx(total_return(equity)) == 0.5


def test_cagr_one_year():
    equity = np.ones(252) * 100
    equity[-1] = 110
    c = cagr(equity)
    assert pytest.approx(c, abs=0.01) == 0.10


def test_volatility_constant():
    returns = np.full(100, 0.01)
    assert volatility(returns) == pytest.approx(0.0, abs=1e-10)


def test_compute_all_keys():
    returns = np.random.default_rng(42).normal(0.001, 0.01, 252)
    equity = 100 * np.cumprod(1 + returns)
    result = compute_all(returns, equity, 10)
    expected_keys = {"sharpe", "sortino", "max_drawdown", "max_drawdown_duration",
                     "calmar", "win_rate", "profit_factor", "total_return",
                     "cagr", "volatility", "num_trades"}
    assert set(result.keys()) == expected_keys
    assert result["num_trades"] == 10
