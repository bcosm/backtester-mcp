"""20-line quickstart: load data, run backtest, print metrics."""

import numpy as np
from backtester-mcp import load, backtest

# load price data
df = load("datasets/spy_daily.parquet")
prices = df["close"].to_numpy()

# simple moving average crossover signal
fast = np.convolve(prices, np.ones(10)/10, mode="full")[:len(prices)]
slow = np.convolve(prices, np.ones(50)/50, mode="full")[:len(prices)]
signals = np.where(fast > slow, 1.0, -1.0)
signals[:50] = 0  # no signal until MAs warm up

# run backtest
result = backtest(prices, signals)

# print results
for k, v in result.metrics.items():
    print(f"  {k}: {v}")
