"""Simple prediction market strategy for binary contracts (price in 0-1).

Goes long when price is below fair value estimate, short when above.
Uses a smoothed midpoint as the "fair value" anchor.
"""

import numpy as np

DEFAULT_PARAMS = {"lookback": 30, "threshold": 0.05}


def generate_signals(prices: np.ndarray, lookback: int = 30,
                     threshold: float = 0.05) -> np.ndarray:
    signals = np.zeros(len(prices))

    for i in range(lookback, len(prices)):
        fair = np.mean(prices[i - lookback:i])
        deviation = prices[i] - fair

        if deviation < -threshold:
            signals[i] = 1.0   # price below fair — buy
        elif deviation > threshold:
            signals[i] = -1.0  # price above fair — sell
        # otherwise hold previous
        else:
            signals[i] = signals[i - 1]

    return signals
