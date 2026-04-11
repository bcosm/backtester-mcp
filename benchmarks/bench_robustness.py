"""PBO, bootstrap, and walk-forward timing."""

import time
import numpy as np
from backtester_mcp.robustness import pbo, bootstrap_sharpe, perturbation_pbo
from backtester_mcp.engine import backtest


def momentum_signals(prices, fast_period=10, slow_period=50):
    n = len(prices)
    signals = np.zeros(n)
    for i in range(slow_period, n):
        if np.mean(prices[i-fast_period:i]) > np.mean(prices[i-slow_period:i]):
            signals[i] = 1.0
        else:
            signals[i] = -1.0
    return signals


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 2000
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))

    # PBO on 20 random strategies
    returns_list = []
    for _ in range(20):
        sigs = rng.choice([-1.0, 0.0, 1.0], size=n)
        r = backtest(prices, sigs, fees=0.0, slippage=0.0)
        returns_list.append(r.returns)
    min_len = min(len(r) for r in returns_list)
    matrix = np.column_stack([r[:min_len] for r in returns_list])

    t0 = time.perf_counter()
    result = pbo(matrix, n_splits=10)
    dt_pbo = time.perf_counter() - t0
    print(f"PBO (20 strategies, {n} bars): {dt_pbo*1000:.0f}ms  score={result['pbo']:.4f}")

    # Bootstrap Sharpe
    signals = momentum_signals(prices)
    r = backtest(prices, signals)
    t0 = time.perf_counter()
    bs = bootstrap_sharpe(r.returns)
    dt_bs = time.perf_counter() - t0
    print(f"Bootstrap Sharpe (10k samples):  {dt_bs*1000:.0f}ms  sharpe={bs['sharpe']:.4f}")

    # Perturbation PBO
    t0 = time.perf_counter()
    ppbo = perturbation_pbo(
        momentum_signals, prices,
        {"fast_period": 10, "slow_period": 50}, n_variants=20,
    )
    dt_ppbo = time.perf_counter() - t0
    print(f"Perturbation PBO (20 variants):  {dt_ppbo*1000:.0f}ms  pbo={ppbo['pbo']:.4f}")
