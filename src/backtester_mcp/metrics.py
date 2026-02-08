import numpy as np

TRADING_DAYS = 252


def sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS))


def sortino(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    down_std = np.sqrt(np.mean(downside**2))
    if down_std == 0:
        return 0.0
    return float(np.mean(excess) / down_std * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def max_drawdown_duration(equity: np.ndarray) -> int:
    """Number of periods in the longest drawdown."""
    peak = np.maximum.accumulate(equity)
    in_dd = equity < peak
    if not np.any(in_dd):
        return 0
    longest = 0
    current = 0
    for v in in_dd:
        if v:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def calmar(returns: np.ndarray, equity: np.ndarray) -> float:
    dd = max_drawdown(equity)
    if dd == 0:
        return float("inf")
    cagr_val = cagr(equity)
    return float(cagr_val / abs(dd))


def win_rate(returns: np.ndarray) -> float:
    trades = returns[returns != 0]
    if len(trades) == 0:
        return 0.0
    return float(np.sum(trades > 0) / len(trades))


def profit_factor(returns: np.ndarray) -> float:
    gains = np.sum(returns[returns > 0])
    losses = abs(np.sum(returns[returns < 0]))
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def total_return(equity: np.ndarray) -> float:
    return float(equity[-1] / equity[0] - 1.0)


def cagr(equity: np.ndarray) -> float:
    n_years = len(equity) / TRADING_DAYS
    if n_years == 0:
        return 0.0
    return float((equity[-1] / equity[0]) ** (1.0 / n_years) - 1.0)


def volatility(returns: np.ndarray) -> float:
    return float(np.std(returns) * np.sqrt(TRADING_DAYS))


def compute_all(returns: np.ndarray, equity: np.ndarray, num_trades: int) -> dict:
    return {
        "sharpe": round(sharpe(returns), 4),
        "sortino": round(sortino(returns), 4),
        "max_drawdown": round(max_drawdown(equity), 4),
        "max_drawdown_duration": max_drawdown_duration(equity),
        "calmar": round(calmar(returns, equity), 4),
        "win_rate": round(win_rate(returns), 4),
        "profit_factor": round(profit_factor(returns), 4),
        "total_return": round(total_return(equity), 4),
        "cagr": round(cagr(equity), 4),
        "volatility": round(volatility(returns), 4),
        "num_trades": num_trades,
    }
