"""MCP server exposing backtesting tools to AI agents."""

import json
import tempfile
import os
from pathlib import Path

import numpy as np


def _strategy_from_code(code: str):
    """Compile strategy code string into a module with generate_signals."""
    import importlib.util
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(code)
    tmp.close()
    try:
        spec = importlib.util.spec_from_file_location("strategy", tmp.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.unlink(tmp.name)


def run_server(transport: str = "stdio"):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError("install mcp extra: pip install backtester-mcp[mcp]")

    mcp = FastMCP("backtester-mcp")

    @mcp.tool()
    def backtest_strategy(strategy_code: str, data_path: str,
                          params: dict = None) -> str:
        """Run a backtest and return metrics."""
        from backtester_mcp.data import load, to_arrays
        from backtester_mcp.engine import backtest

        df = load(data_path)
        prices = to_arrays(df)["close"]
        mod = _strategy_from_code(strategy_code)
        signals = mod.generate_signals(prices, **(params or {}))
        result = backtest(prices, signals)
        return json.dumps(result.metrics, indent=2)

    @mcp.tool()
    def validate_robustness(strategy_code: str, data_path: str,
                            params: dict = None) -> str:
        """Run bootstrap Sharpe CI + deflated Sharpe ratio."""
        from backtester_mcp.data import load, to_arrays
        from backtester_mcp.engine import backtest
        from backtester_mcp.robustness import bootstrap_sharpe, deflated_sharpe

        df = load(data_path)
        prices = to_arrays(df)["close"]
        mod = _strategy_from_code(strategy_code)
        signals = mod.generate_signals(prices, **(params or {}))
        result = backtest(prices, signals)
        bs = bootstrap_sharpe(result.returns)
        dsr = deflated_sharpe(
            observed_sharpe=bs["sharpe"],
            n_returns=len(result.returns),
            n_strategies=1,
        )
        return json.dumps({
            "bootstrap_sharpe": {
                "sharpe": bs["sharpe"],
                "ci_lower": bs["ci_lower"],
                "ci_upper": bs["ci_upper"],
                "ci_includes_zero": bs["ci_includes_zero"],
            },
            "deflated_sharpe": {
                "dsr": dsr["dsr"],
                "p_value": dsr["p_value"],
                "expected_max_sharpe": dsr["expected_max_sharpe"],
            },
            "metrics": result.metrics,
        }, indent=2)

    @mcp.tool()
    def optimize_parameters(strategy_code: str, data_path: str,
                            param_space: dict) -> str:
        """Optimize strategy parameters and return best params + PBO score."""
        from backtester_mcp.data import load, to_arrays
        from backtester_mcp.optimize import optimize

        df = load(data_path)
        prices = to_arrays(df)["close"]
        mod = _strategy_from_code(strategy_code)

        # convert param_space values to tuples
        space = {k: tuple(v) for k, v in param_space.items()}
        result = optimize(mod.generate_signals, prices, space)

        out = {k: v for k, v in result.items() if k != "logits"}
        return json.dumps(out, indent=2, default=str)

    @mcp.tool()
    def compare_strategies(strategies: list[dict]) -> str:
        """Compare multiple strategies. Each dict: {code, data_path, params}."""
        from backtester_mcp.data import load, to_arrays
        from backtester_mcp.engine import backtest

        results = []
        for i, s in enumerate(strategies):
            df = load(s["data_path"])
            prices = to_arrays(df)["close"]
            mod = _strategy_from_code(s["code"])
            signals = mod.generate_signals(prices, **(s.get("params") or {}))
            r = backtest(prices, signals)
            results.append({"strategy_index": i, "metrics": r.metrics})

        results.sort(key=lambda x: x["metrics"]["sharpe"], reverse=True)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def upload_dataset(file_path: str, name: str = None) -> str:
        """Register a local dataset file for use in backtests."""
        from backtester_mcp.data import load

        path = Path(file_path).resolve()
        if not path.exists():
            return json.dumps({"error": f"file not found: {file_path}"})

        df = load(str(path))
        rows = len(df)
        cols = list(df.columns)
        label = name or path.stem

        return json.dumps({
            "name": label,
            "path": str(path),
            "rows": rows,
            "columns": cols,
        }, indent=2)

    mcp.run(transport=transport)
