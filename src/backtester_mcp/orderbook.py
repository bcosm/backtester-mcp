"""Stochastic order book estimation and fill simulation."""

import numpy as np
from numba import njit


def estimate_spread_roll(prices: np.ndarray) -> float:
    """Roll (1984) spread estimator from close prices.

    Spread = 2 * sqrt(-cov(delta_p_t, delta_p_{t-1}))
    When covariance is positive (no bounce), returns 0.
    """
    dp = np.diff(prices)
    if len(dp) < 2:
        return 0.0
    cov = np.mean(dp[1:] * dp[:-1]) - np.mean(dp[1:]) * np.mean(dp[:-1])
    if cov >= 0:
        return 0.0
    return float(2.0 * np.sqrt(-cov) / np.mean(prices))


def estimate_spread_corwin_schultz(high: np.ndarray, low: np.ndarray) -> float:
    """Corwin & Schultz (2012) spread estimator from OHLC data.

    Uses the ratio of high-low ranges over 1 and 2 day windows.
    """
    beta = np.log(high[:-1] / low[:-1])**2 + np.log(high[1:] / low[1:])**2
    gamma = np.log(
        np.maximum(high[:-1], high[1:]) / np.minimum(low[:-1], low[1:])
    )**2

    beta_mean = np.mean(beta)
    gamma_mean = np.mean(gamma)

    k = 2 * np.sqrt(2) - 1
    alpha = (np.sqrt(2 * beta_mean) - np.sqrt(beta_mean)) / (3 - 2 * np.sqrt(2))
    alpha -= np.sqrt(gamma_mean / (3 - 2 * np.sqrt(2)))

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(max(spread, 0.0))


def estimate_volume_profile(volume: np.ndarray, n_levels: int = 10) -> np.ndarray:
    """Estimate an exponential decay order book shape from volume data.

    Returns relative liquidity at each price level away from mid.
    """
    avg_vol = np.mean(volume)
    decay = 0.5
    levels = np.array([avg_vol * decay**i for i in range(n_levels)])
    return levels / np.sum(levels)


@njit(cache=True)
def _simulate_fill(mid_price, spread, order_size, daily_volume,
                   volatility, is_buy, rng_seed):
    """Core fill simulation with market impact."""
    np.random.seed(rng_seed)

    half_spread = spread * mid_price / 2.0

    # square-root market impact: impact ~ sigma * sqrt(Q / V)
    if daily_volume > 0:
        participation = abs(order_size) / daily_volume
        impact = volatility * mid_price * np.sqrt(min(participation, 1.0))
    else:
        impact = 0.0

    # random component calibrated to the spread
    noise = np.random.normal(0, half_spread * 0.3)

    if is_buy:
        fill_price = mid_price + half_spread + impact + abs(noise)
    else:
        fill_price = mid_price - half_spread - impact - abs(noise)

    return fill_price


class FillModel:
    """Estimates fills from price data alone — no per-asset configuration needed."""

    def __init__(self, prices: np.ndarray, high: np.ndarray = None,
                 low: np.ndarray = None, volume: np.ndarray = None):
        if high is not None and low is not None:
            self.spread = estimate_spread_corwin_schultz(high, low)
        else:
            self.spread = estimate_spread_roll(prices)

        # fallback: if estimator gives 0, use a conservative default
        if self.spread <= 0:
            self.spread = 0.001

        returns = np.diff(prices) / prices[:-1]
        self.volatility = float(np.std(returns))
        self.avg_volume = float(np.mean(volume)) if volume is not None else 0.0
        self._seed_counter = 0

    def fill_price(self, mid: float, order_size: float, is_buy: bool) -> float:
        self._seed_counter += 1
        return float(_simulate_fill(
            mid, self.spread, order_size, self.avg_volume,
            self.volatility, is_buy, self._seed_counter
        ))

    def summary(self) -> dict:
        return {
            "estimated_spread": round(self.spread, 6),
            "daily_volatility": round(self.volatility, 6),
            "avg_volume": round(self.avg_volume, 2),
        }
