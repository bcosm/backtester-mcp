"""Multi-strategy / multi-asset portfolio backtesting."""
from __future__ import annotations

import numpy as np

from backtester_mcp.engine import backtest
from backtester_mcp.metrics import compute_all, sharpe, volatility


def backtest_portfolio(
    components: list[dict],
    initial_capital: float = 100_000.0,
) -> dict:
    """Run backtests for each component and combine into a portfolio.

    Each component: {"prices": ndarray, "signals": ndarray,
                     "weight": float, "name": str}
    """
    if not components:
        raise ValueError("need at least one component")

    # normalize weights
    total_w = sum(c["weight"] for c in components)
    weights = [c["weight"] / total_w for c in components]

    results = []
    for comp in components:
        r = backtest(comp["prices"], comp["signals"])
        results.append(r)

    # find common length (all components must cover same period)
    min_len = min(len(r.equity_curve) for r in results)

    # combine equity curves (weighted, rebased)
    combined_equity = np.zeros(min_len)
    for w, r in zip(weights, results):
        eq = r.equity_curve[:min_len]
        combined_equity += w * (eq / eq[0]) * initial_capital

    combined_returns = np.diff(combined_equity) / combined_equity[:-1]
    combined_returns = np.concatenate([[0.0], combined_returns])

    portfolio_metrics = compute_all(combined_returns, combined_equity, 0)

    # per-component metrics
    component_metrics = []
    for comp, r in zip(components, results):
        component_metrics.append({
            "name": comp["name"],
            "weight": comp["weight"] / total_w,
            "metrics": r.metrics,
        })

    # correlation matrix of component returns
    returns_matrix = np.column_stack([
        r.returns[:min_len] for r in results
    ])
    corr = np.corrcoef(returns_matrix.T).tolist()

    # diversification ratio: sum(w_i * vol_i) / portfolio_vol
    component_vols = [volatility(r.returns[:min_len]) for r in results]
    weighted_vol_sum = sum(w * v for w, v in zip(weights, component_vols))
    port_vol = volatility(combined_returns)
    div_ratio = weighted_vol_sum / port_vol if port_vol > 0 else 1.0

    return {
        "portfolio_metrics": portfolio_metrics,
        "component_metrics": component_metrics,
        "correlation_matrix": corr,
        "diversification_ratio": round(div_ratio, 4),
    }
