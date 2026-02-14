"""Dual moving average crossover."""

import numpy as np

DEFAULT_PARAMS = {"fast_period": 10, "slow_period": 50}


def generate_signals(prices: np.ndarray, fast_period: int = 10,
                     slow_period: int = 50) -> np.ndarray:
    signals = np.zeros(len(prices))

    fast_ma = np.convolve(prices, np.ones(fast_period) / fast_period, mode="full")[:len(prices)]
    slow_ma = np.convolve(prices, np.ones(slow_period) / slow_period, mode="full")[:len(prices)]

    # no signal until both MAs are warm
    for i in range(slow_period, len(prices)):
        if fast_ma[i] > slow_ma[i]:
            signals[i] = 1.0
        elif fast_ma[i] < slow_ma[i]:
            signals[i] = -1.0

    return signals
