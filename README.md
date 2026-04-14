# backtester-mcp

Local-first backtesting engine with built-in overfitting detection. Asset-class agnostic. MCP-native.

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![Tests](https://github.com/bcosm/backtester-mcp/actions/workflows/test.yml/badge.svg)

## The Problem

QuantConnect requires Docker + C#, supports 9 hardcoded asset classes, and ships zero statistical robustness tools. Solo quants and AI agents need something that's `pip install`, works on any price series (equities, crypto, prediction markets) and tells you if your strategy is overfit before you risk real money.

backtester-mcp is a validation layer for AI-generated trading strategies. Vectorized execution on NumPy + Numba, automatic overfitting detection via PBO, walk-forward validation, execution scenario analysis, and a native MCP server so AI agents can validate strategies directly.

## Quick Start

```bash
pip install backtester-mcp
```

Runs on synthetic data out of the box, no datasets to download:

```python
import numpy as np
from backtester_mcp import backtest

# synthetic price series, reproducible
rng = np.random.default_rng(0)
prices = np.cumprod(1 + rng.normal(0, 0.01, 2000))

# moving average crossover signal
fast = np.convolve(prices, np.ones(10) / 10, mode="full")[:len(prices)]
slow = np.convolve(prices, np.ones(50) / 50, mode="full")[:len(prices)]
signals = np.where(fast > slow, 1.0, -1.0)
signals[:50] = 0

result = backtest(prices, signals)
print(result.metrics)
```

To run against a real dataset, clone the repo (`git clone https://github.com/bcosm/backtester-mcp`) and point `load()` at anything in `datasets/` or your own CSV/Parquet file.

## Key Features

| Feature | QuantConnect | backtester-mcp |
|---|---|---|
| Setup | Docker + .NET | `pip install` |
| Engine | C# (Python wrapper) | Pure Python + NumPy + Numba |
| Asset classes | 9 hardcoded | Any price series |
| Overfitting detection | None | PBO + Bootstrap Sharpe + DSR + Walk-Forward |
| Execution realism | Hand-tuned per asset | Auto-estimated from data, 3 scenario modes |
| AI agent interface | Cloud API wrapper | Native MCP server (13 tools) |
| Run persistence | Cloud-dependent | Local DuckDB registry |
| Cost | Free tier; $60+/mo | Free & open source |

## Validation Pipeline

backtester-mcp runs a full validation pipeline that tells you whether to trust a strategy:

1. **Backtest** with realistic estimated fills
2. **Bootstrap Sharpe CI**: is the Sharpe distinguishable from zero?
3. **Deflated Sharpe**: does it survive correction for multiple testing?
4. **PBO (perturbation)**: are the exact parameters robust, or did you get lucky?
5. **Walk-forward validation**: does the strategy hold up out-of-sample?
6. **Execution scenarios**: optimistic, base, and conservative cost assumptions

```bash
backtester-mcp backtest -s strategies/momentum.py -d datasets/spy_daily.parquet \
  --robustness --execution-scenarios --walk-forward
```

## CLI Usage

```bash
# Basic backtest
backtester-mcp backtest -s strategies/momentum.py -d datasets/spy_daily.parquet

# Full validation: robustness + execution scenarios
backtester-mcp backtest -s strategies/momentum.py -d datasets/spy_daily.parquet \
  --robustness --execution-scenarios

# Override strategy parameters
backtester-mcp backtest -s strategies/momentum.py -d datasets/spy_daily.parquet \
  --set fast_period=20 --set slow_period=100

# Estimated fills from market data
backtester-mcp backtest -s strategies/momentum.py -d datasets/spy_daily.parquet \
  --realistic-fills

# Optimize with Bayesian search + PBO check
backtester-mcp optimize -s strategies/momentum.py -d datasets/spy_daily.parquet \
  -p fast_period:5:50 -p slow_period:20:200

# Generate HTML validation report
backtester-mcp report -s strategies/momentum.py -d datasets/spy_daily.parquet \
  --robustness --execution-scenarios -o report.html

# Persist and compare runs
backtester-mcp backtest -s strategies/momentum.py -d datasets/spy_daily.parquet --save-run
backtester-mcp list-runs
backtester-mcp show-run <run-id>
backtester-mcp compare-runs <run-id-1> <run-id-2>
```

## MCP Server

backtester-mcp exposes 13 tools via the Model Context Protocol for AI agents:

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

If your MCP client (e.g. Claude Desktop) can't find the `backtester-mcp` binary because the venv isn't on its PATH, use the Python-module form instead:

```json
{
  "mcpServers": {
    "backtester-mcp": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "backtester_mcp", "serve", "--transport", "stdio"]
    }
  }
}
```

**Available tools:**

| Tool | Purpose |
|---|---|
| `backtest_strategy` | Run a backtest (flat, estimated, or conservative fills) |
| `validate_strategy` | Full validation pipeline with pass/caution verdict |
| `validate_robustness` | Bootstrap Sharpe CI + DSR + PBO |
| `optimize_parameters` | Bayesian parameter search with PBO check |
| `compare_strategies` | Compare multiple strategies side by side |
| `register_dataset` | Register data from file path, CSV, or base64 |
| `profile_dataset` | Detailed dataset statistics and quality check |
| `save_run` | Persist results to local DuckDB store |
| `list_runs` | List recent runs, filterable by dataset/strategy |
| `load_run` | Retrieve full results for a run |
| `compare_runs` | Compare metrics across saved runs |
| `generate_report` | Create HTML validation report |
| `strategy_template` | Get a parameterized strategy code template |

The `validate_strategy` tool runs the full pipeline in one call and returns a structured verdict:

```json
{
  "verdict": "caution",
  "reasons": ["bootstrap CI includes zero", "conservative scenario negative return"],
  "metrics": {"sharpe": 0.28, ...},
  "pbo": {"pbo": 0.36, ...},
  "scenarios": {"optimistic": {...}, "base": {...}, "conservative": {...}}
}
```

## Performance

Benchmarks on a 2024 thin-and-light laptop CPU (Intel Core Ultra 7 256V, Lunar Lake generation; performance is in the same ballpark as an Apple M3 MacBook Air or Ryzen 7 7840U for single-threaded NumPy/Numba workloads). Python 3.13, numba 0.65, min of 5 runs after JIT warmup:

| Bars | Time |
|---|---|
| 1,000 | ~0.1 ms |
| 10,000 | ~0.6 ms |
| 100,000 | ~8.5 ms |
| 500,000 | ~56 ms |

The first call in a process pays for numba JIT compilation (roughly 1 to 3 seconds depending on the function). Subsequent calls hit numba's on-disk cache and run at the numbers above.

## How PBO Works

Probability of Backtest Overfitting (PBO) answers: "if I pick the best strategy from an in-sample optimization, what's the probability it underperforms out-of-sample?"

It works by splitting your backtest data into S sub-periods, then forming all combinations of S/2 sub-periods as in-sample and the rest as out-of-sample (combinatorially symmetric cross-validation). For each combination, it checks whether the strategy that ranked best in-sample still performs above median out-of-sample. The PBO score is the fraction of combinations where it doesn't. A score above 0.5 means your strategy is more likely overfit than not.

For single strategies, backtester-mcp uses **perturbation PBO**: it generates variants by jittering parameters within +/-20%, runs each variant, then computes PBO over the resulting returns matrix. This answers "would nearby parameters work just as well, or did you get lucky with these exact numbers?"

From Lopez de Prado (2018), "The Probability of Backtest Overfitting," *Journal of Computational Finance*.

## Architecture

```
Data (CSV/Parquet/DuckDB) -> Engine (NumPy + Numba) -> Metrics
                                                    -> Robustness (PBO, Bootstrap, DSR, Walk-Forward)
                                                    -> Sensitivity (Execution Scenarios, Stress Test)
                                                    -> Report (HTML)
                                                    -> Manifest (JSON audit trail)
                                                    -> Store (DuckDB persistence)
```

Strategies are plain functions: `f(prices, **params) -> signals`. No class hierarchies, no inheritance, no plugin system.

## Stochastic Order Book

Most backtesting engines either ignore fill simulation or require hand-tuned models per asset class. backtester-mcp automatically estimates fill characteristics from your price data:

- **Spread estimation**: Corwin-Schultz (2012) from OHLC data, Roll (1984) from close prices
- **Market impact**: square-root model calibrated to observed volatility
- **Three execution modes**: optimistic (zero cost), base (estimated), conservative (2x spread)

This works on any price series without configuration.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Open an issue before submitting a PR.

## License

Apache 2.0
