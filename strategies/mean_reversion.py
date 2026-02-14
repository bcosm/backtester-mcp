"""Bollinger band mean reversion."""

import numpy as np

DEFAULT_PARAMS = {"lookback": 20, "num_std": 2.0}


def generate_signals(prices: np.ndarray, lookback: int = 20,
                     num_std: float = 2.0) -> np.ndarray:
    signals = np.zeros(len(prices))

    for i in range(lookback, len(prices)):
        window = prices[i - lookback:i]
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            continue

        upper = mean + num_std * std
        lower = mean - num_std * std

        if prices[i] < lower:
            signals[i] = 1.0   # buy the dip
        elif prices[i] > upper:
            signals[i] = -1.0  # sell the rip
        else:
            signals[i] = signals[i - 1]

    return signals
