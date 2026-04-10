"""Tests for sensitivity / execution scenarios module."""

import numpy as np
import pytest

from backtester_mcp.sensitivity import run_execution_scenarios, stress_test


def test_execution_scenarios_three_modes(trending_prices, simple_signals):
    result = run_execution_scenarios(trending_prices, simple_signals)
    assert "optimistic" in result
    assert "base" in result
    assert "conservative" in result
    assert "fill_summary" in result


def test_optimistic_better_than_conservative(trending_prices, simple_signals):
    result = run_execution_scenarios(trending_prices, simple_signals)
    opt = result["optimistic"]["total_return"]
    con = result["conservative"]["total_return"]
    assert opt >= con


def test_execution_scenarios_with_ohlcv(trending_prices, simple_signals, ohlcv_arrays):
    result = run_execution_scenarios(
        trending_prices, simple_signals,
        high=ohlcv_arrays["high"], low=ohlcv_arrays["low"],
        volume=ohlcv_arrays["volume"],
    )
    assert result["fill_summary"]["method_used"] in (
        "corwin_schultz", "roll", "default_fallback"
    )


def test_stress_test_scenarios(trending_prices, simple_signals):
    result = stress_test(trending_prices, simple_signals)
    assert "high_fees" in result
    assert "wide_spread" in result
    assert "first_half" in result
    assert "second_half" in result
    for name, metrics in result.items():
        if "error" not in metrics:
            assert "sharpe" in metrics


def test_stress_high_fees_worse(trending_prices, simple_signals):
    from backtester_mcp.engine import backtest
    base = backtest(trending_prices, simple_signals)
    result = stress_test(trending_prices, simple_signals)
    assert result["high_fees"]["total_return"] <= base.metrics["total_return"]
