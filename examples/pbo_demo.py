"""Demonstrate PBO catching overfitting vs real signal."""

import numpy as np
from backtester_mcp.robustness import pbo, bootstrap_sharpe, perturbation_pbo
from backtester_mcp.engine import backtest


def make_random_strategies(prices, n_strats=20, seed=0):
    """Random strategies on real data — should have high PBO."""
    rng = np.random.default_rng(seed)
    n = len(prices)
    returns_list = []
    for _ in range(n_strats):
        signals = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.3, 0.4, 0.3])
        result = backtest(prices, signals, fees=0.0, slippage=0.0)
        returns_list.append(result.returns)
    min_len = min(len(r) for r in returns_list)
    return np.column_stack([r[:min_len] for r in returns_list])


def make_momentum_signals(prices, fast=10, slow=50):
    n = len(prices)
    signals = np.zeros(n)
    for i in range(slow, n):
        if np.mean(prices[i-fast:i]) > np.mean(prices[i-slow:i]):
            signals[i] = 1.0
        else:
            signals[i] = -1.0
    return signals


if __name__ == "__main__":
    from backtester_mcp import load

    df = load("datasets/spy_daily.parquet")
    prices = df["close"].to_numpy()

    # random strategies: should have high PBO
    print("Random strategies (should be overfit):")
    matrix = make_random_strategies(prices)
    result = pbo(matrix, n_splits=10)
    print(f"  PBO score: {result['pbo']:.4f}")
    print(f"  Combinations tested: {result['n_combinations']}")
    if result["pbo"] > 0.5:
        print("  -> high PBO: these strategies are likely overfit\n")

    # momentum with perturbation PBO: should be lower
    print("Momentum strategy (perturbation PBO):")
    ppbo = perturbation_pbo(
        make_momentum_signals, prices,
        {"fast": 10, "slow": 50}, n_variants=20,
    )
    print(f"  PBO score: {ppbo['pbo']:.4f}")
    print(f"  Param ranges: {ppbo['param_ranges']}\n")

    # bootstrap check
    print("Bootstrap Sharpe CI for momentum:")
    signals = make_momentum_signals(prices)
    r = backtest(prices, signals)
    bs = bootstrap_sharpe(r.returns)
    print(f"  Sharpe: {bs['sharpe']:.4f}")
    print(f"  95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")
    print(f"  CI includes zero: {bs['ci_includes_zero']}")
