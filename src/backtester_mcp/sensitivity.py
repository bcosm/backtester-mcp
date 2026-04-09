"""Execution scenarios and stress testing."""
from __future__ import annotations

import numpy as np

from backtester_mcp.engine import backtest
from backtester_mcp.orderbook import FillModel


def run_execution_scenarios(
    prices: np.ndarray,
    signals: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
) -> dict:
    """Run optimistic / base / conservative backtests side by side."""
    fm = FillModel(prices, high=high, low=low, volume=volume)
    half_spread = fm.spread / 2.0

    optimistic = backtest(prices, signals, fees=0.0, slippage=0.0)
    base = backtest(prices, signals, fees=0.001, slippage=half_spread)
    conservative = backtest(
        prices, signals, fees=0.002, slippage=fm.spread
    )

    return {
        "optimistic": optimistic.metrics,
        "base": base.metrics,
        "conservative": conservative.metrics,
        "fill_summary": fm.summary(),
    }


def stress_test(
    prices: np.ndarray,
    signals: np.ndarray,
) -> dict:
    """Run backtest under varied cost and regime assumptions."""
    prices = np.asarray(prices, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.float64)
    n = len(prices)
    mid = n // 2

    results: dict[str, dict] = {}
    results["high_fees"] = backtest(
        prices, signals, fees=0.003, slippage=0.001
    ).metrics
    results["wide_spread"] = backtest(
        prices, signals, fees=0.001, slippage=0.002
    ).metrics
    results["first_half"] = backtest(
        prices[:mid], signals[:mid]
    ).metrics
    results["second_half"] = backtest(
        prices[mid:], signals[mid:]
    ).metrics

    # high-volatility regime filter
    rets = np.diff(prices) / prices[:-1]
    if len(rets) > 20:
        kernel = np.ones(20) / 20
        rolling_var = np.convolve(rets**2, kernel, mode="valid")
        rolling_vol = np.sqrt(rolling_var)
        med_vol = float(np.median(rolling_vol))
        offset = len(prices) - len(rolling_vol)
        mask = np.zeros(n, dtype=bool)
        mask[offset:] = rolling_vol > med_vol
        if np.sum(mask) >= 50:
            results["high_volatility"] = backtest(
                prices[mask], signals[mask]
            ).metrics

    return results
