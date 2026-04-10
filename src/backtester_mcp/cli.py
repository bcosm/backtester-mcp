"""CLI entry point."""

import argparse
import importlib.util
import sys
import json
from pathlib import Path

import numpy as np

from backtester_mcp.data import load, to_arrays
from backtester_mcp.engine import backtest
from backtester_mcp.manifest import create_manifest, save_manifest


def _load_strategy(path: str):
    """Dynamically load a strategy module from a .py file."""
    spec = importlib.util.spec_from_file_location("strategy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_param_range(s: str) -> tuple[str, int | float, int | float]:
    """Parse 'name:low:high' into (name, low, high)."""
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"param format must be name:low:high, got '{s}'")
    name = parts[0]
    lo, hi = parts[1], parts[2]
    if "." in lo or "." in hi:
        return name, float(lo), float(hi)
    return name, int(lo), int(hi)


def _parse_set_param(s: str) -> tuple[str, int | float | str]:
    """Parse 'key=value' into (key, typed_value)."""
    if "=" not in s:
        raise ValueError(f"--set format must be key=value, got '{s}'")
    key, val = s.split("=", 1)
    try:
        return key, int(val)
    except ValueError:
        pass
    try:
        return key, float(val)
    except ValueError:
        return key, val


def _print_metrics(metrics: dict):
    for k, v in metrics.items():
        if k == "max_drawdown":
            print(f"  {k:.<25} {v:.2%}")
        elif k in ("total_return", "cagr", "volatility", "win_rate"):
            print(f"  {k:.<25} {v:.2%}")
        elif k == "max_drawdown_duration":
            print(f"  {k:.<25} {v} bars")
        elif isinstance(v, float):
            print(f"  {k:.<25} {v:.4f}")
        else:
            print(f"  {k:.<25} {v}")


def _print_scenario_table(scenarios: dict):
    """Print a comparison table of execution scenario metrics."""
    keys = ["sharpe", "total_return", "max_drawdown", "cagr"]
    header = f"  {'metric':.<20} {'optimistic':>12} {'base':>12} {'conservative':>12}"
    print(header)
    print(f"  {'-'*56}")
    for k in keys:
        row = f"  {k:.<20}"
        for mode in ("optimistic", "base", "conservative"):
            v = scenarios[mode].get(k, 0)
            if k in ("total_return", "max_drawdown", "cagr"):
                row += f" {v:>11.2%}"
            else:
                row += f" {v:>11.4f}"
        print(row)


def cmd_backtest(args):
    df = load(args.data)
    arrays = to_arrays(df)
    prices = arrays.get("close")
    if prices is None:
        print("error: no close/price column found in data", file=sys.stderr)
        sys.exit(1)

    mod = _load_strategy(args.strategy)
    params = dict(getattr(mod, "DEFAULT_PARAMS", {}))

    # apply --set overrides
    for s in (args.set or []):
        key, val = _parse_set_param(s)
        params[key] = val

    # apply --params-json overrides
    if args.params_json:
        params.update(json.loads(args.params_json))

    signals = mod.generate_signals(prices, **params)
    result = backtest(prices, signals)

    print(f"\n{'='*40}")
    print(f"  Strategy: {Path(args.strategy).stem}")
    print(f"  Data: {args.data} ({len(prices)} bars)")
    print(f"{'='*40}")
    _print_metrics(result.metrics)

    if args.realistic_fills:
        from backtester_mcp.orderbook import FillModel
        fm = FillModel(prices, high=arrays.get("high"),
                       low=arrays.get("low"), volume=arrays.get("volume"))
        print(f"\n  Fill model: {fm.method_used}")
        print(f"  Estimated spread: {fm.spread:.6f}")
        fill_result = backtest(prices, signals, fees=0.001,
                                slippage=fm.spread / 2)
        print(f"\n  With estimated fills:")
        _print_metrics(fill_result.metrics)

    if args.execution_scenarios:
        from backtester_mcp.sensitivity import run_execution_scenarios
        scenarios = run_execution_scenarios(
            prices, signals, high=arrays.get("high"),
            low=arrays.get("low"), volume=arrays.get("volume"),
        )
        print(f"\n  Execution Scenarios:")
        _print_scenario_table(scenarios)
        print(f"  Spread method: {scenarios['fill_summary']['method_used']}")

    if args.robustness:
        from backtester_mcp.robustness import (
            bootstrap_sharpe, deflated_sharpe, perturbation_pbo,
        )
        bs = bootstrap_sharpe(result.returns)
        print(f"\n  Bootstrap Sharpe 95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")
        if bs["ci_includes_zero"]:
            print("  ** CI includes zero: edge may not be statistically significant")

        dsr = deflated_sharpe(
            observed_sharpe=bs["sharpe"],
            n_returns=len(result.returns),
            n_strategies=1,
        )
        print(f"  Deflated Sharpe p-value: {dsr['p_value']:.4f}")

        if params:
            ppbo = perturbation_pbo(mod.generate_signals, prices, params)
            if ppbo.get("pbo") is not None:
                print(f"  PBO (param perturbation): {ppbo['pbo']:.4f}")
                if ppbo["pbo"] > 0.5:
                    print("  ** High PBO — parameterization may be overfit")
            elif "warning" in ppbo:
                print(f"  PBO: skipped ({ppbo['warning']})")

    if args.walk_forward:
        from backtester_mcp.robustness import walk_forward
        # need param_space for walk-forward — derive from DEFAULT_PARAMS
        default_params = getattr(mod, "DEFAULT_PARAMS", {})
        if default_params:
            param_space = {}
            for k, v in default_params.items():
                if isinstance(v, int):
                    param_space[k] = (max(1, v // 2), v * 2)
                elif isinstance(v, float):
                    param_space[k] = (v * 0.5, v * 2.0)
            wf = walk_forward(mod.generate_signals, prices, param_space,
                              n_windows=3, n_trials=20)
            print(f"\n  Walk-Forward ({len(wf['windows'])} windows):")
            print(f"  Aggregate OOS Sharpe: {wf['aggregate_oos_sharpe']:.4f}")
            print(f"  Stability: {wf['stability_score']:.0%}")
            for i, w in enumerate(wf["windows"]):
                oos = w["oos_metrics"]
                print(f"    Window {i+1}: Sharpe={oos['sharpe']:.4f}  "
                      f"Return={oos['total_return']:.2%}")
        else:
            print("\n  Walk-forward: skipped (no DEFAULT_PARAMS to optimize)")

    # generate manifest
    date_range = None
    if hasattr(df.index, 'min'):
        try:
            date_range = (str(df.index.min().date()), str(df.index.max().date()))
        except Exception:
            pass

    manifest = create_manifest(
        strategy_name=Path(args.strategy).stem,
        params=params,
        data_path=args.data,
        data_rows=len(prices),
        date_range=date_range,
        metrics=result.metrics,
    )

    if args.manifest:
        save_manifest(manifest, args.manifest)
        print(f"\n  Manifest saved to {args.manifest}")

    if args.save_run:
        from backtester_mcp.store import save_run
        run_id = save_run(
            strategy_name=Path(args.strategy).stem,
            params=params,
            metrics=result.metrics,
            manifest=manifest,
        )
        print(f"\n  Run saved: {run_id}")

    return result, manifest


def cmd_optimize(args):
    df = load(args.data)
    arrays = to_arrays(df)
    prices = arrays.get("close")
    if prices is None:
        print("error: no close/price column found", file=sys.stderr)
        sys.exit(1)

    mod = _load_strategy(args.strategy)
    param_space = {}
    for p in args.param:
        name, lo, hi = _parse_param_range(p)
        param_space[name] = (lo, hi)

    from backtester_mcp.optimize import optimize
    result = optimize(
        strategy_fn=mod.generate_signals,
        prices=prices,
        param_space=param_space,
        n_trials=args.trials,
    )

    print(f"\n  Best params: {result['best_params']}")
    print(f"  Best {result['metric_name']}: {result['best_metric']:.4f}")
    if "pbo" in result and result["pbo"] is not None:
        print(f"  PBO score: {result['pbo']:.4f}")
        if result["pbo"] > 0.5:
            print("  ** High PBO — optimized parameters likely overfit")

    if args.save_run:
        from backtester_mcp.store import save_run
        run_id = save_run(
            strategy_name=Path(args.strategy).stem,
            params=result["best_params"],
            metrics={"best_" + result["metric_name"]: result["best_metric"]},
        )
        print(f"  Run saved: {run_id}")


def cmd_report(args):
    result, manifest = cmd_backtest(args)

    from backtester_mcp.report import generate_report, save_report
    from backtester_mcp.robustness import bootstrap_sharpe

    pbo_result = None
    bootstrap_result = None
    wf_result = None
    scenarios = None

    if args.robustness:
        from backtester_mcp.robustness import perturbation_pbo
        bootstrap_result = bootstrap_sharpe(result.returns)
        mod = _load_strategy(args.strategy)
        params = getattr(mod, "DEFAULT_PARAMS", {})
        if params:
            pbo_result = perturbation_pbo(
                mod.generate_signals,
                to_arrays(load(args.data))["close"],
                params,
            )

    if args.walk_forward:
        from backtester_mcp.robustness import walk_forward
        mod = _load_strategy(args.strategy)
        default_params = getattr(mod, "DEFAULT_PARAMS", {})
        if default_params:
            prices = to_arrays(load(args.data))["close"]
            param_space = {}
            for k, v in default_params.items():
                if isinstance(v, int):
                    param_space[k] = (max(1, v // 2), v * 2)
                elif isinstance(v, float):
                    param_space[k] = (v * 0.5, v * 2.0)
            wf_result = walk_forward(
                mod.generate_signals, prices, param_space,
                n_windows=3, n_trials=20,
            )

    if args.execution_scenarios:
        from backtester_mcp.sensitivity import run_execution_scenarios
        df = load(args.data)
        arrays = to_arrays(df)
        prices = arrays["close"]
        mod = _load_strategy(args.strategy)
        params = getattr(mod, "DEFAULT_PARAMS", {})
        signals = mod.generate_signals(prices, **params)
        scenarios = run_execution_scenarios(
            prices, signals, high=arrays.get("high"),
            low=arrays.get("low"), volume=arrays.get("volume"),
        )

    html = generate_report(
        equity=result.equity_curve,
        returns=result.returns,
        metrics=result.metrics,
        manifest=manifest,
        pbo_result=pbo_result,
        bootstrap_result=bootstrap_result,
        walk_forward_result=wf_result,
        scenarios=scenarios,
    )

    out = args.output or "report.html"
    save_report(html, out)
    print(f"\n  Report saved to {out}")


def cmd_serve(args):
    from backtester_mcp.mcp_server import run_server
    run_server(transport=args.transport)


def cmd_list_runs(args):
    from backtester_mcp.store import list_runs
    runs = list_runs(limit=args.limit if hasattr(args, "limit") else 20)
    if not runs:
        print("  No saved runs.")
        return
    print(f"\n  {'ID':36}  {'Strategy':15}  {'Date':25}  {'Sharpe':>8}")
    print(f"  {'-'*86}")
    for r in runs:
        s = r["metrics"].get("sharpe", "—")
        sharpe_str = f"{s:.4f}" if isinstance(s, float) else str(s)
        print(f"  {r['id']:36}  {r['strategy_name']:15}  "
              f"{r['created_at'][:19]:25}  {sharpe_str:>8}")


def cmd_show_run(args):
    from backtester_mcp.store import get_run
    run = get_run(args.run_id)
    if not run:
        print(f"  Run not found: {args.run_id}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(run, indent=2, default=str))


def cmd_compare_runs(args):
    from backtester_mcp.store import compare_runs
    runs = compare_runs(args.run_ids)
    if not runs:
        print("  No matching runs.")
        return

    keys = ["sharpe", "total_return", "max_drawdown", "cagr"]
    header = f"  {'metric':.<20}"
    for r in runs:
        label = r["strategy_name"][:12]
        header += f" {label:>12}"
    print(header)
    print(f"  {'-' * (20 + 13 * len(runs))}")
    for k in keys:
        row = f"  {k:.<20}"
        for r in runs:
            v = r["metrics"].get(k, 0)
            if k in ("total_return", "max_drawdown", "cagr"):
                row += f" {v:>11.2%}"
            else:
                row += f" {v:>11.4f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        prog="backtester-mcp",
        description="Local-first backtesting engine with overfitting detection",
    )
    sub = parser.add_subparsers(dest="command")

    # backtest
    bt = sub.add_parser("backtest", help="run a backtest")
    bt.add_argument("--strategy", "-s", required=True)
    bt.add_argument("--data", "-d", required=True)
    bt.add_argument("--robustness", action="store_true",
                     help="bootstrap Sharpe CI + DSR + PBO")
    bt.add_argument("--realistic-fills", action="store_true",
                     help="use estimated fills from data")
    bt.add_argument("--execution-scenarios", action="store_true",
                     help="run optimistic/base/conservative comparison")
    bt.add_argument("--walk-forward", action="store_true",
                     help="run walk-forward validation")
    bt.add_argument("--set", action="append",
                     help="override param: key=value (repeatable)")
    bt.add_argument("--params-json",
                     help="JSON blob of param overrides")
    bt.add_argument("--save-run", action="store_true",
                     help="persist results to local store")
    bt.add_argument("--manifest", "-m", help="save manifest to this path")

    # optimize
    opt = sub.add_parser("optimize", help="optimize strategy parameters")
    opt.add_argument("--strategy", "-s", required=True)
    opt.add_argument("--data", "-d", required=True)
    opt.add_argument("--param", "-p", action="append", required=True,
                     help="param:low:high (e.g. fast_period:5:50)")
    opt.add_argument("--trials", "-n", type=int, default=100)
    opt.add_argument("--save-run", action="store_true")

    # report
    rpt = sub.add_parser("report", help="generate HTML report")
    rpt.add_argument("--strategy", "-s", required=True)
    rpt.add_argument("--data", "-d", required=True)
    rpt.add_argument("--robustness", action="store_true",
                      help="include robustness analysis in report")
    rpt.add_argument("--walk-forward", action="store_true",
                      help="include walk-forward results")
    rpt.add_argument("--execution-scenarios", action="store_true",
                      help="include execution scenarios")
    rpt.add_argument("--realistic-fills", action="store_true")
    rpt.add_argument("--output", "-o", default="report.html")
    rpt.add_argument("--manifest", "-m")
    rpt.add_argument("--set", action="append")
    rpt.add_argument("--params-json", default=None)
    rpt.add_argument("--save-run", action="store_true")

    # serve
    srv = sub.add_parser("serve", help="start MCP server")
    srv.add_argument("--transport", default="stdio", choices=["stdio"])

    # list-runs
    lr = sub.add_parser("list-runs", help="list saved runs")
    lr.add_argument("--limit", type=int, default=20)

    # show-run
    sr = sub.add_parser("show-run", help="show details of a saved run")
    sr.add_argument("run_id")

    # compare-runs
    cr = sub.add_parser("compare-runs", help="compare metrics across runs")
    cr.add_argument("run_ids", nargs="+")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "backtest": cmd_backtest,
        "optimize": cmd_optimize,
        "report": cmd_report,
        "serve": cmd_serve,
        "list-runs": cmd_list_runs,
        "show-run": cmd_show_run,
        "compare-runs": cmd_compare_runs,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
