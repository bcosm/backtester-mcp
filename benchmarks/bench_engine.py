"""Backtest engine speed at various data scales."""

import time
import numpy as np
from backtester_mcp.engine import backtest


def bench(n_bars, warmup=True):
    rng = np.random.default_rng(42)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bars)))
    signals = rng.choice([-1.0, 0.0, 1.0], size=n_bars)

    if warmup:
        backtest(prices[:100], signals[:100])

    t0 = time.perf_counter()
    result = backtest(prices, signals)
    dt = time.perf_counter() - t0
    return dt, result.metrics["sharpe"]


if __name__ == "__main__":
    print("backtester-mcp engine benchmark")
    print(f"{'bars':>10}  {'time (ms)':>12}  {'sharpe':>8}")
    print("-" * 34)

    for n in [1_000, 10_000, 100_000, 500_000]:
        dt, sharpe = bench(n)
        print(f"{n:>10,}  {dt*1000:>11.1f}ms  {sharpe:>8.4f}")

    # JIT warmup comparison
    prices = 100.0 * np.exp(np.cumsum(
        np.random.default_rng(42).normal(0, 0.01, 10000)))
    signals = np.random.default_rng(42).choice([-1.0, 0.0, 1.0], size=10000)

    t0 = time.perf_counter()
    backtest(prices, signals)
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    backtest(prices, signals)
    warm = time.perf_counter() - t0

    print(f"\nJIT warmup: {cold*1000:.1f}ms cold -> {warm*1000:.1f}ms warm")
