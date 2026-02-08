from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    metrics: dict


@njit(cache=True)
def _core_backtest(prices, signals, fees, slippage, initial_capital):
    n = len(prices)
    equity = np.empty(n)
    returns = np.empty(n)
    positions = np.empty(n)

    cash = initial_capital
    pos = 0.0
    equity[0] = initial_capital
    returns[0] = 0.0
    positions[0] = signals[0]
    num_trades = 0

    # enter initial position
    if signals[0] != 0.0:
        cost_per_unit = prices[0] * (1.0 + signals[0] * (fees + slippage))
        pos = signals[0] * (cash / abs(cost_per_unit))
        cash -= pos * cost_per_unit
        num_trades += 1

    for i in range(1, n):
        # mark to market
        equity[i] = cash + pos * prices[i]
        returns[i] = equity[i] / equity[i - 1] - 1.0

        target = signals[i]
        positions[i] = target

        if target != signals[i - 1]:
            # close current position
            if pos != 0.0:
                fill = prices[i] * (1.0 - np.sign(pos) * (fees + slippage))
                cash += pos * fill
                pos = 0.0
                num_trades += 1

            # open new position
            if target != 0.0:
                portfolio_val = cash + pos * prices[i]
                cost_per_unit = prices[i] * (1.0 + target * (fees + slippage))
                pos = target * (portfolio_val / abs(cost_per_unit))
                cash -= pos * cost_per_unit
                num_trades += 1

        equity[i] = cash + pos * prices[i]

    return equity, returns, positions, num_trades


def backtest(
    prices: np.ndarray,
    signals: np.ndarray,
    fees: float = 0.001,
    slippage: float = 0.0005,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Run a vectorized backtest on a price series with a signal array."""
    prices = np.asarray(prices, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.float64)

    if len(prices) != len(signals):
        raise ValueError("prices and signals must have the same length")

    equity, returns, positions, num_trades = _core_backtest(
        prices, signals, fees, slippage, initial_capital
    )

    from backtester_mcp.metrics import compute_all
    metrics = compute_all(returns, equity, num_trades)

    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        positions=positions,
        metrics=metrics,
    )
