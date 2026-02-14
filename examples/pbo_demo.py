"""Demonstrate PBO detecting overfitting vs. a real signal."""

import numpy as np
from backtester_mcp.robustness import pbo, bootstrap_sharpe
from backtester_mcp.engine import backtest


def make_overfit_strategies(prices, n_strats=20, seed=0):
    """Generate strategies optimized on random noise — should have high PBO."""
    rng = np.random.default_rng(seed)
    n = len(prices)
    returns_list = []

    for _ in range(n_strats):
        # random entry/exit — no real edge
        signals = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.3, 0.4, 0.3])
        result = backtest(prices, signals, fees=0.0, slippage=0.0)
        returns_list.append(result.returns)

    min_len = min(len(r) for r in returns_list)
    return np.column_stack([r[:min_len] for r in returns_list])


if __name__ == "__main__":
    from backtester-mcp import load
    df = load("datasets/spy_daily.parquet")
    prices = df["close"].to_numpy()

    print("generating random strategies (should be overfit)...")
    matrix = make_overfit_strategies(prices)
    result = pbo(matrix, n_splits=10)
    print(f"  PBO score: {result['pbo']:.4f}")
    print(f"  combinations tested: {result['n_combinations']}")
    if result["pbo"] > 0.5:
        print("  -> high PBO: these strategies are likely overfit\n")

    # bootstrap check on a single random strategy
    print("bootstrap Sharpe CI for a random strategy:")
    single_returns = matrix[:, 0]
    bs = bootstrap_sharpe(single_returns)
    print(f"  Sharpe: {bs['sharpe']:.4f}")
    print(f"  95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")
    print(f"  CI includes zero: {bs['ci_includes_zero']}")
