# backtester-mcp

Local-first backtesting engine with built-in overfitting detection. Asset-class agnostic. MCP-native.

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)

## The Problem

QuantConnect requires Docker + C#, supports 9 hardcoded asset classes, and ships zero statistical robustness tools. Solo quants and AI agents need something that's `pip install`, works on any price series (equities, crypto, prediction markets) and tells you if your strategy is overfit before you risk real money.

backtester-mcp is a backtesting engine built for this. Vectorized execution on NumPy + Numba, automatic overfitting detection via Probability of Backtest Overfitting (PBO), and a native MCP server so AI agents can run backtests directly.

## Quick Start

```bash
pip install backtester-mcp
```

```python
import numpy as np
from backtester_mcp import load, backtest

df = load("datasets/spy_daily.parquet")
prices = df["close"].to_numpy()

# moving average crossover signal
fast = np.convolve(prices, np.ones(10)/10, mode="full")[:len(prices)]
slow = np.convolve(prices, np.ones(50)/50, mode="full")[:len(prices)]
signals = np.where(fast > slow, 1.0, -1.0)
signals[:50] = 0

result = backtest(prices, signals)
print(result.metrics)
```

```
{'sharpe': 0.2801, 'sortino': 0.2723, 'max_drawdown': -0.2889,
 'calmar': 0.0477, 'win_rate': 0.5233, 'profit_factor': 1.0541,
 'total_return': 0.162, 'cagr': 0.0138, 'volatility': 0.1649,
 'num_trades': 137}
```

## Key Features

| Feature | QuantConnect | backtester-mcp |
|---|---|---|
| Setup | Docker + .NET | `pip install` |
| Engine | C# (Python wrapper) | Pure Python + NumPy + Numba |
| Asset classes | 9 hardcoded | Any price series |
| Overfitting detection | None | PBO + Bootstrap Sharpe + DSR |
| Fill simulation | Hand-tuned per asset | Auto-estimated from data |
| AI agent interface | Cloud API wrapper | Native MCP server |
| Cost | $60+/month | Free & open source |

## CLI Usage

```bash
# Run a backtest
backtester-mcp backtest --strategy strategies/momentum.py --data datasets/spy_daily.parquet

# Run with bootstrap Sharpe analysis
backtester-mcp backtest --strategy strategies/momentum.py --data datasets/spy_daily.parquet --pbo

# Optimize parameters (Bayesian search + automatic PBO check)
backtester-mcp optimize --strategy strategies/momentum.py --data datasets/spy_daily.parquet \
  --param fast_period:5:50 --param slow_period:20:200

# Generate HTML report
backtester-mcp report --strategy strategies/momentum.py --data datasets/spy_daily.parquet -o report.html
```

Output from `backtester-mcp backtest`:
```
========================================
  Strategy: momentum
  Data: datasets/spy_daily.parquet (2765 bars)
========================================
  sharpe................... 0.2801
  sortino.................. 0.2723
  max_drawdown............. -28.89%
  max_drawdown_duration.... 849
  calmar................... 0.0477
  win_rate................. 52.33%
  profit_factor............ 1.0541
  total_return............. 16.20%
  cagr..................... 1.38%
  volatility............... 16.49%
  num_trades............... 137
```

## MCP Server

backtester-mcp exposes backtesting tools via the Model Context Protocol, so AI agents (Claude, etc.) can run backtests, validate robustness, and optimize parameters directly.

```json
{
  "mcpServers": {
    "backtester-mcp": {
      "command": "backtester-mcp",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

Available tools: `backtest_strategy`, `validate_robustness`, `optimize_parameters`, `compare_strategies`.

## How PBO Works

Probability of Backtest Overfitting (PBO) answers: "if I pick the best strategy from an in-sample optimization, what's the probability it underperforms out-of-sample?"

It works by splitting your backtest data into S sub-periods, then forming all combinations of S/2 sub-periods as in-sample and the rest as out-of-sample (combinatorially symmetric cross-validation). For each combination, it checks whether the strategy that ranked best in-sample still performs above median out-of-sample. The PBO score is the fraction of combinations where it doesn't. A score above 0.5 means your strategy is more likely overfit than not.

This is from Lopez de Prado (2018), "The Probability of Backtest Overfitting," *Journal of Computational Finance*.

## Architecture

```
Data (CSV/Parquet/DuckDB)
  → Engine (vectorized backtest on NumPy arrays, Numba-accelerated)
    → Metrics (Sharpe, Sortino, drawdown, etc.)
    → Robustness (PBO, bootstrap Sharpe, deflated Sharpe ratio)
    → Report (self-contained HTML)
    → Manifest (reproducible JSON audit trail)
```

Strategies are plain functions: `f(prices, **params) → signals`. No class hierarchies, no inheritance, no plugin system.

## Stochastic Order Book

Most backtesting engines either ignore fill simulation or require hand-tuned models per asset class. backtester-mcp automatically estimates fill characteristics from your price data:

- **Spread estimation**: Roll (1984) estimator from close prices, or Corwin-Schultz (2012) from OHLC data
- **Market impact**: square-root model calibrated to observed volatility
- **Stochastic fills**: randomized around the estimated spread, so your backtest doesn't assume perfect execution

This works on any price series (equities, crypto, prediction markets) without configuration.

## License

Apache 2.0
