"""Self-contained HTML report generation."""

import json
import numpy as np
from backtester_mcp.metrics import TRADING_DAYS


def _equity_svg(equity: np.ndarray, width: int = 800, height: int = 250) -> str:
    n = len(equity)
    if n < 2:
        return ""
    lo, hi = float(np.min(equity)), float(np.max(equity))
    pad = (hi - lo) * 0.05 or 1.0
    lo -= pad
    hi += pad

    points = []
    for i in range(n):
        x = i / (n - 1) * width
        y = height - (equity[i] - lo) / (hi - lo) * height
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{polyline}" fill="none" stroke="#4fc3f7" stroke-width="1.5"/>'
        f'</svg>'
    )


def _drawdown_svg(equity: np.ndarray, width: int = 800, height: int = 120) -> str:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    n = len(dd)
    if n < 2:
        return ""
    lo = float(np.min(dd))
    if lo == 0:
        lo = -0.01

    points = [f"0,0"]
    for i in range(n):
        x = i / (n - 1) * width
        y = -(dd[i] / lo) * height
        points.append(f"{x:.1f},{y:.1f}")
    points.append(f"{width},0")

    polygon = " ".join(points)
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polygon points="{polygon}" fill="rgba(244,67,54,0.4)" stroke="#f44336" stroke-width="1"/>'
        f'</svg>'
    )


def _monthly_heatmap(returns: np.ndarray, dates=None) -> str:
    n = len(returns)
    block = 21
    months = n // block
    if months < 1:
        return "<p>Not enough data for monthly heatmap.</p>"

    rows_html = []
    for m in range(months):
        start = m * block
        end = min(start + block, n)
        monthly_ret = float(np.prod(1 + returns[start:end]) - 1)
        pct = monthly_ret * 100

        if monthly_ret >= 0:
            bg = f"rgba(76,175,80,{min(abs(monthly_ret)*10, 0.8):.2f})"
        else:
            bg = f"rgba(244,67,54,{min(abs(monthly_ret)*10, 0.8):.2f})"

        rows_html.append(
            f'<td style="background:{bg};padding:4px 8px;text-align:center">'
            f'{pct:+.1f}%</td>'
        )

    table = '<table style="border-collapse:collapse;margin:16px 0">'
    for row_start in range(0, len(rows_html), 12):
        table += "<tr>" + "".join(rows_html[row_start:row_start+12]) + "</tr>"
    table += "</table>"
    return table


def _metrics_table(metrics: dict) -> str:
    rows = ""
    fmt = {
        "sharpe": "{:.3f}", "sortino": "{:.3f}", "calmar": "{:.3f}",
        "max_drawdown": "{:.2%}", "total_return": "{:.2%}", "cagr": "{:.2%}",
        "volatility": "{:.2%}", "win_rate": "{:.2%}", "profit_factor": "{:.3f}",
    }
    for k, v in metrics.items():
        if isinstance(v, (np.ndarray, list)):
            continue
        f = fmt.get(k, "{}")
        display = f.format(v) if isinstance(v, float) else str(v)
        label = k.replace("_", " ").title()
        rows += f"<tr><td>{label}</td><td>{display}</td></tr>"
    return (
        '<table style="border-collapse:collapse;width:100%;max-width:400px">'
        f"<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _scenario_table(scenarios: dict) -> str:
    keys = ["sharpe", "total_return", "max_drawdown", "cagr"]
    fmt_map = {"sharpe": "{:.3f}", "total_return": "{:.2%}",
               "max_drawdown": "{:.2%}", "cagr": "{:.2%}"}
    header = "<tr><th>Metric</th>"
    for mode in ("optimistic", "base", "conservative"):
        header += f"<th>{mode.title()}</th>"
    header += "</tr>"

    rows = ""
    for k in keys:
        rows += f"<tr><td>{k.replace('_',' ').title()}</td>"
        for mode in ("optimistic", "base", "conservative"):
            v = scenarios[mode].get(k, 0)
            f = fmt_map.get(k, "{:.4f}")
            rows += f"<td>{f.format(v)}</td>"
        rows += "</tr>"

    return (
        '<table style="border-collapse:collapse;width:100%;max-width:600px">'
        f"<thead>{header}</thead><tbody>{rows}</tbody></table>"
    )


def _walk_forward_section(wf: dict) -> str:
    html = f"""
    <h2>Walk-Forward Validation</h2>
    <p>Aggregate OOS Sharpe: <strong>{wf['aggregate_oos_sharpe']:.4f}</strong></p>
    <p>Stability: <strong>{wf['stability_score']:.0%}</strong>
    ({sum(1 for w in wf['windows'] if w['oos_metrics']['sharpe'] > 0)}/{len(wf['windows'])} windows positive)</p>
    <table style="border-collapse:collapse;width:100%;max-width:600px">
    <thead><tr><th>Window</th><th>Train bars</th><th>Test bars</th>
    <th>OOS Sharpe</th><th>OOS Return</th></tr></thead><tbody>"""

    for i, w in enumerate(wf["windows"]):
        oos = w["oos_metrics"]
        train_bars = w["train_end"] - w["train_start"]
        test_bars = w["test_end"] - w["test_start"]
        html += (f"<tr><td>{i+1}</td><td>{train_bars}</td><td>{test_bars}</td>"
                 f"<td>{oos['sharpe']:.4f}</td><td>{oos['total_return']:.2%}</td></tr>")

    html += "</tbody></table>"
    return html


def generate_report(
    equity: np.ndarray,
    returns: np.ndarray,
    metrics: dict,
    manifest: dict = None,
    pbo_result: dict = None,
    bootstrap_result: dict = None,
    walk_forward_result: dict = None,
    scenarios: dict = None,
) -> str:
    """Generate a self-contained HTML report string."""

    equity_chart = _equity_svg(equity)
    dd_chart = _drawdown_svg(equity)
    heatmap = _monthly_heatmap(returns)
    metrics_html = _metrics_table(metrics)

    pbo_section = ""
    if pbo_result and pbo_result.get("pbo") is not None:
        pbo_section = f"""
        <h2>Overfitting Analysis (PBO)</h2>
        <p>PBO Score: <strong>{pbo_result['pbo']:.4f}</strong>
        ({pbo_result.get('n_combinations', '?')} combinations tested)</p>
        <p>{'&#9888; High probability of overfitting' if pbo_result['pbo'] > 0.5
           else '&#10004; Low overfitting risk'}</p>
        """

    bootstrap_section = ""
    if bootstrap_result:
        bootstrap_section = f"""
        <h2>Bootstrap Sharpe</h2>
        <p>Point estimate: {bootstrap_result['sharpe']:.4f}</p>
        <p>95% CI: [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]</p>
        <p>{'&#9888; CI includes zero' if bootstrap_result['ci_includes_zero']
           else '&#10004; Statistically significant'}</p>
        """

    wf_section = ""
    if walk_forward_result:
        wf_section = _walk_forward_section(walk_forward_result)

    scenario_section = ""
    if scenarios:
        fill = scenarios.get("fill_summary", {})
        scenario_section = f"""
        <h2>Execution Scenarios</h2>
        <p>Fill method: {fill.get('method_used', 'unknown')}
        | Spread: {fill.get('estimated_spread', 0):.6f}</p>
        {_scenario_table(scenarios)}
        """

    manifest_section = ""
    if manifest:
        manifest_section = (
            "<h2>Run Manifest</h2>"
            f'<pre style="background:#1e1e1e;padding:12px;border-radius:4px;overflow-x:auto">'
            f'{json.dumps(manifest, indent=2, default=str)}</pre>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>backtester-mcp report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #121212; color: #e0e0e0; margin: 0; padding: 24px; }}
  h1 {{ color: #4fc3f7; margin-bottom: 4px; }}
  h2 {{ color: #81c784; margin-top: 32px; }}
  table {{ font-size: 14px; }}
  th, td {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid #333; }}
  th {{ color: #aaa; }}
  pre {{ font-size: 12px; color: #aaa; }}
  svg {{ display: block; margin: 8px 0; }}
</style>
</head>
<body>
<h1>backtester-mcp backtest report</h1>
<h2>Equity Curve</h2>
{equity_chart}
<h2>Drawdown</h2>
{dd_chart}
<h2>Metrics</h2>
{metrics_html}
<h2>Monthly Returns</h2>
{heatmap}
{pbo_section}
{bootstrap_section}
{wf_section}
{scenario_section}
{manifest_section}
</body>
</html>"""


def save_report(html: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(html)
