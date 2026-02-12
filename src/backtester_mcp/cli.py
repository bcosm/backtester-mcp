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


def _parse_param(s: str) -> tuple[str, int, int]:
    """Parse 'name:low:high' into (name, low, high)."""
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"param format must be name:low:high, got '{s}'")
    return parts[0], int(parts[1]), int(parts[2])


def cmd_backtest(args):
    df = load(args.data)
    arrays = to_arrays(df)
    prices = arrays.get("close")
    if prices is None:
        print("error: no close/price column found in data", file=sys.stderr)
        sys.exit(1)

    mod = _load_strategy(args.strategy)
    signals = mod.generate_signals(prices, **getattr(mod, "DEFAULT_PARAMS", {}))
    result = backtest(prices, signals)

    print(f"\n{'='*40}")
    print(f"  Strategy: {Path(args.strategy).stem}")
    print(f"  Data: {args.data} ({len(prices)} bars)")
    print(f"{'='*40}")
    for k, v in result.metrics.items():
        if k == "max_drawdown":
            print(f"  {k:.<25} {v:.2%}")
        elif k == "total_return":
            print(f"  {k:.<25} {v:.2%}")
        elif k == "cagr":
            print(f"  {k:.<25} {v:.2%}")
        elif k == "volatility":
            print(f"  {k:.<25} {v:.2%}")
        elif k == "win_rate":
            print(f"  {k:.<25} {v:.2%}")
        elif isinstance(v, float):
            print(f"  {k:.<25} {v:.4f}")
        else:
            print(f"  {k:.<25} {v}")

    # PBO analysis
    if args.pbo:
        from backtester_mcp.robustness import bootstrap_sharpe
        bs = bootstrap_sharpe(result.returns)
        print(f"\n  Bootstrap Sharpe 95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")
        if bs["ci_includes_zero"]:
            print("  ** CI includes zero — edge may not be statistically significant")

    # generate manifest
    date_range = None
    if hasattr(df.index, 'min'):
        try:
            date_range = (str(df.index.min().date()), str(df.index.max().date()))
        except Exception:
            pass

    manifest = create_manifest(
        strategy_name=Path(args.strategy).stem,
        params=getattr(mod, "DEFAULT_PARAMS", {}),
        data_path=args.data,
        data_rows=len(prices),
        date_range=date_range,
        metrics=result.metrics,
    )

    if args.manifest:
        save_manifest(manifest, args.manifest)
        print(f"\n  Manifest saved to {args.manifest}")

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
        name, lo, hi = _parse_param(p)
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
    if "pbo" in result:
        print(f"  PBO score: {result['pbo']:.4f}")
        if result["pbo"] > 0.5:
            print("  ** High PBO — optimized parameters likely overfit")


def cmd_report(args):
    result, manifest = cmd_backtest(args)

    from backtester_mcp.report import generate_report, save_report

    pbo_result = None
    bootstrap_result = None
    if args.pbo:
        from backtester_mcp.robustness import bootstrap_sharpe
        bootstrap_result = bootstrap_sharpe(result.returns)

    html = generate_report(
        equity=result.equity_curve,
        returns=result.returns,
        metrics=result.metrics,
        manifest=manifest,
        pbo_result=pbo_result,
        bootstrap_result=bootstrap_result,
    )

    out = args.output or "report.html"
    save_report(html, out)
    print(f"\n  Report saved to {out}")


def cmd_serve(args):
    from backtester_mcp.mcp_server import run_server
    run_server(transport=args.transport)


def main():
    parser = argparse.ArgumentParser(prog="backtester-mcp", description="AI-first backtesting engine")
    sub = parser.add_subparsers(dest="command")

    # backtest
    bt = sub.add_parser("backtest", help="run a backtest")
    bt.add_argument("--strategy", "-s", required=True)
    bt.add_argument("--data", "-d", required=True)
    bt.add_argument("--pbo", action="store_true", help="run bootstrap Sharpe analysis")
    bt.add_argument("--manifest", "-m", help="save manifest to this path")

    # optimize
    opt = sub.add_parser("optimize", help="optimize strategy parameters")
    opt.add_argument("--strategy", "-s", required=True)
    opt.add_argument("--data", "-d", required=True)
    opt.add_argument("--param", "-p", action="append", required=True,
                     help="param:low:high (e.g. fast_period:5:50)")
    opt.add_argument("--trials", "-n", type=int, default=100)

    # report
    rpt = sub.add_parser("report", help="generate HTML report")
    rpt.add_argument("--strategy", "-s", required=True)
    rpt.add_argument("--data", "-d", required=True)
    rpt.add_argument("--pbo", action="store_true")
    rpt.add_argument("--output", "-o", default="report.html")
    rpt.add_argument("--manifest", "-m")

    # serve
    srv = sub.add_parser("serve", help="start MCP server")
    srv.add_argument("--transport", default="stdio", choices=["stdio"])

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "backtest": cmd_backtest,
        "optimize": cmd_optimize,
        "report": cmd_report,
        "serve": cmd_serve,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
