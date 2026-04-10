"""MCP server exposing backtesting tools to AI agents."""

import json
import tempfile
import os
import base64
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


def _make_response(data: dict, warnings: list = None) -> str:
    out = dict(data)
    if warnings:
        out["warnings"] = warnings
    out["reproducibility"] = {"engine_version": "0.1.0"}
    return json.dumps(out, indent=2, default=str)


def run_server(transport: str = "stdio"):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError("install mcp extra: pip install backtester-mcp[mcp]")

    mcp = FastMCP("backtester-mcp")

    @mcp.tool()
    def backtest_strategy(strategy_code: str, data_path: str,
                          params: dict = None,
                          fill_mode: str = "flat") -> str:
        """Run a backtest and return metrics.

        fill_mode: "flat" (default fees/slippage), "estimated" (data-driven),
                   "conservative" (2x spread).
        """
        from backtester_mcp.data import load, to_arrays
        from backtester_mcp.engine import backtest

        df = load(data_path)
        arrays = to_arrays(df)
        prices = arrays["close"]
        mod = _strategy_from_code(strategy_code)
        signals = mod.generate_signals(prices, **(params or {}))

        warnings = []
        if fill_mode == "estimated":
            from backtester_mcp.orderbook import FillModel
            fm = FillModel(prices, high=arrays.get("high"),
                           low=arrays.get("low"), volume=arrays.get("volume"))
            result = backtest(prices, signals, fees=0.001,
                              slippage=fm.spread / 2)
        elif fill_mode == "conservative":
            from backtester_mcp.orderbook import FillModel
            fm = FillModel(prices, high=arrays.get("high"),
                           low=arrays.get("low"), volume=arrays.get("volume"))
            result = backtest(prices, signals, fees=0.002,
                              slippage=fm.spread)
        else:
            result = backtest(prices, signals)

        return _make_response({"metrics": result.metrics}, warnings)

    @mcp.tool()
    def validate_robustness(strategy_code: str, data_path: str,
                            params: dict = None) -> str:
        """Run bootstrap Sharpe CI + deflated Sharpe + PBO."""
        from backtester_mcp.data import load, to_arrays
        from backtester_mcp.engine import backtest
        from backtester_mcp.robustness import (
            bootstrap_sharpe, deflated_sharpe, perturbation_pbo,
        )

        df = load(data_path)
        prices = to_arrays(df)["close"]
        mod = _strategy_from_code(strategy_code)
        p = params or getattr(mod, "DEFAULT_PARAMS", {})
        signals = mod.generate_signals(prices, **p)
        result = backtest(prices, signals)
        bs = bootstrap_sharpe(result.returns)
        dsr = deflated_sharpe(
            observed_sharpe=bs["sharpe"],
            n_returns=len(result.returns),
            n_strategies=1,
        )

        out = {
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
        }

        if p:
            ppbo = perturbation_pbo(mod.generate_signals, prices, p)
            out["pbo"] = ppbo

        return _make_response(out)

    @mcp.tool()
    def validate_strategy(strategy_code: str, data_path: str,
                          params: dict = None,
                          param_space: dict = None) -> str:
        """Full validation pipeline: backtest + robustness + PBO + scenarios."""
        from backtester_mcp.data import load, to_arrays
        from backtester_mcp.engine import backtest
        from backtester_mcp.robustness import (
            bootstrap_sharpe, deflated_sharpe, perturbation_pbo,
        )
        from backtester_mcp.sensitivity import run_execution_scenarios

        df = load(data_path)
        arrays = to_arrays(df)
        prices = arrays["close"]
        mod = _strategy_from_code(strategy_code)
        p = params or getattr(mod, "DEFAULT_PARAMS", {})
        signals = mod.generate_signals(prices, **p)
        result = backtest(prices, signals)

        bs = bootstrap_sharpe(result.returns)
        dsr = deflated_sharpe(
            observed_sharpe=bs["sharpe"],
            n_returns=len(result.returns),
            n_strategies=1,
        )
        scenarios = run_execution_scenarios(
            prices, signals, high=arrays.get("high"),
            low=arrays.get("low"), volume=arrays.get("volume"),
        )

        reasons = []
        if bs["ci_includes_zero"]:
            reasons.append("bootstrap CI includes zero")
        if dsr["p_value"] > 0.05:
            reasons.append(f"DSR p-value={dsr['p_value']:.4f}")
        if scenarios["conservative"]["total_return"] < 0:
            reasons.append("conservative scenario negative return")

        pbo_out = None
        if p:
            pbo_out = perturbation_pbo(mod.generate_signals, prices, p)
            if pbo_out.get("pbo") is not None and pbo_out["pbo"] > 0.5:
                reasons.append(f"PBO={pbo_out['pbo']:.2f}")

        wf_out = None
        if param_space:
            from backtester_mcp.robustness import walk_forward
            space = {k: tuple(v) for k, v in param_space.items()}
            wf_out = walk_forward(mod.generate_signals, prices, space,
                                  n_windows=3, n_trials=20)
            if wf_out["stability_score"] < 0.5:
                reasons.append("walk-forward stability < 50%")

        verdict = "caution" if reasons else "pass"

        return _make_response({
            "verdict": verdict,
            "reasons": reasons,
            "metrics": result.metrics,
            "bootstrap_sharpe": {
                "sharpe": bs["sharpe"],
                "ci_lower": bs["ci_lower"],
                "ci_upper": bs["ci_upper"],
                "ci_includes_zero": bs["ci_includes_zero"],
            },
            "deflated_sharpe": {
                "dsr": dsr["dsr"],
                "p_value": dsr["p_value"],
            },
            "pbo": pbo_out,
            "walk_forward": wf_out,
            "scenarios": {k: v for k, v in scenarios.items()
                          if k != "fill_summary"},
        })

    @mcp.tool()
    def optimize_parameters(strategy_code: str, data_path: str,
                            param_space: dict) -> str:
        """Optimize strategy parameters and return best params + PBO score."""
        from backtester_mcp.data import load, to_arrays
        from backtester_mcp.optimize import optimize

        df = load(data_path)
        prices = to_arrays(df)["close"]
        mod = _strategy_from_code(strategy_code)

        space = {k: tuple(v) for k, v in param_space.items()}
        result = optimize(mod.generate_signals, prices, space)

        out = {k: v for k, v in result.items() if k != "logits"}
        return _make_response(out)

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
        return _make_response({"strategies": results})

    @mcp.tool()
    def register_dataset(file_path: str = None, name: str = None,
                         csv_text: str = None, base64_data: str = None) -> str:
        """Register a dataset from file path, inline CSV, or base64."""
        from backtester_mcp.data import load
        from backtester_mcp.store import register_dataset as _register

        if csv_text:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False)
            tmp.write(csv_text)
            tmp.close()
            path = tmp.name
        elif base64_data:
            raw = base64.b64decode(base64_data)
            tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
            tmp.write(raw)
            tmp.close()
            path = tmp.name
        elif file_path:
            path = str(Path(file_path).resolve())
            if not Path(path).exists():
                return json.dumps({"error": f"file not found: {file_path}"})
        else:
            return json.dumps({"error": "provide file_path, csv_text, or base64_data"})

        df = load(path)
        date_range = None
        if hasattr(df.index, "min"):
            try:
                date_range = (str(df.index.min()), str(df.index.max()))
            except Exception:
                pass

        label = name or Path(path).stem
        dataset_id = _register(
            name=label, path=path, row_count=len(df),
            columns=list(df.columns), date_range=date_range,
        )
        return _make_response({
            "dataset_id": dataset_id,
            "name": label,
            "rows": len(df),
            "columns": list(df.columns),
        })

    @mcp.tool()
    def profile_dataset(dataset_id: str) -> str:
        """Return detailed statistics for a registered dataset."""
        from backtester_mcp.store import profile_dataset as _profile
        return _make_response(_profile(dataset_id))

    @mcp.tool()
    def save_run(strategy_name: str, metrics: dict,
                 strategy_code: str = "", params: dict = None,
                 dataset_id: str = "", validation: dict = None) -> str:
        """Persist backtest results and return run ID."""
        from backtester_mcp.store import save_run as _save
        run_id = _save(
            strategy_name=strategy_name,
            strategy_code=strategy_code,
            params=params,
            dataset_id=dataset_id,
            metrics=metrics,
            validation=validation,
        )
        return _make_response({"run_id": run_id})

    @mcp.tool()
    def list_runs(dataset_id: str = None, strategy_name: str = None,
                  limit: int = 50) -> str:
        """List recent runs, filterable by dataset or strategy."""
        from backtester_mcp.store import list_runs as _list
        runs = _list(dataset_id=dataset_id, strategy_name=strategy_name,
                     limit=limit)
        return _make_response({"runs": runs})

    @mcp.tool()
    def load_run(run_id: str) -> str:
        """Retrieve full results for a run ID."""
        from backtester_mcp.store import get_run
        run = get_run(run_id)
        if not run:
            return json.dumps({"error": f"run not found: {run_id}"})
        return _make_response(run)

    @mcp.tool()
    def compare_runs(run_ids: list[str]) -> str:
        """Compare metrics across multiple run IDs."""
        from backtester_mcp.store import compare_runs as _compare
        runs = _compare(run_ids)
        return _make_response({"runs": runs})

    @mcp.tool()
    def generate_report(run_id: str = None, metrics: dict = None,
                        equity_curve: list = None, returns: list = None,
                        output_path: str = "report.html") -> str:
        """Generate HTML report from run data or inline results."""
        from backtester_mcp.report import generate_report as _gen
        from backtester_mcp.report import save_report

        if run_id:
            from backtester_mcp.store import get_run
            run = get_run(run_id)
            if not run:
                return json.dumps({"error": f"run not found: {run_id}"})
            m = run["metrics"]
            # generate a basic report without equity data
            html = _gen(
                equity=np.ones(10),
                returns=np.zeros(10),
                metrics=m,
                manifest=run.get("manifest"),
            )
        else:
            eq = np.array(equity_curve or [100000])
            ret = np.array(returns or [0.0])
            html = _gen(equity=eq, returns=ret, metrics=metrics or {})

        save_report(html, output_path)
        return _make_response({"report_path": output_path})

    @mcp.tool()
    def strategy_template(strategy_type: str = "momentum") -> str:
        """Return a parameterized strategy code template."""
        templates = {
            "momentum": '''import numpy as np

DEFAULT_PARAMS = {"fast_period": 10, "slow_period": 50}

def generate_signals(prices, fast_period=10, slow_period=50):
    n = len(prices)
    signals = np.zeros(n)
    for i in range(slow_period, n):
        fast_ma = np.mean(prices[i - fast_period:i])
        slow_ma = np.mean(prices[i - slow_period:i])
        signals[i] = 1.0 if fast_ma > slow_ma else -1.0
    return signals
''',
            "mean_reversion": '''import numpy as np

DEFAULT_PARAMS = {"lookback": 20, "num_std": 2.0}

def generate_signals(prices, lookback=20, num_std=2.0):
    n = len(prices)
    signals = np.zeros(n)
    for i in range(lookback, n):
        window = prices[i - lookback:i]
        mean = np.mean(window)
        std = np.std(window)
        if prices[i] < mean - num_std * std:
            signals[i] = 1.0
        elif prices[i] > mean + num_std * std:
            signals[i] = -1.0
    return signals
''',
            "prediction_market": '''import numpy as np

DEFAULT_PARAMS = {"lookback": 30, "threshold": 0.05}

def generate_signals(prices, lookback=30, threshold=0.05):
    n = len(prices)
    signals = np.zeros(n)
    for i in range(lookback, n):
        fair = np.mean(prices[i - lookback:i])
        if prices[i] < fair - threshold:
            signals[i] = 1.0
        elif prices[i] > fair + threshold:
            signals[i] = -1.0
    return signals
''',
            "custom": '''import numpy as np

DEFAULT_PARAMS = {}

def generate_signals(prices, **params):
    """Custom strategy template. Replace this logic with your own."""
    n = len(prices)
    signals = np.zeros(n)
    # your signal logic here
    return signals
''',
        }
        code = templates.get(strategy_type, templates["custom"])
        return _make_response({"strategy_type": strategy_type, "code": code})

    mcp.run(transport=transport)
