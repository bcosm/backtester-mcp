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
    """Simple monthly returns table. If no dates, approximate from trading days."""
    n = len(returns)
    # approximate monthly blocks (~21 trading days)
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

    # arrange in rows of 12
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


def generate_report(
    equity: np.ndarray,
    returns: np.ndarray,
    metrics: dict,
    manifest: dict = None,
    pbo_result: dict = None,
    bootstrap_result: dict = None,
) -> str:
    """Generate a self-contained HTML report string."""

    equity_chart = _equity_svg(equity)
    dd_chart = _drawdown_svg(equity)
    heatmap = _monthly_heatmap(returns)
    metrics_html = _metrics_table(metrics)

    pbo_section = ""
    if pbo_result:
        pbo_section = f"""
        <h2>Overfitting Analysis</h2>
        <p>PBO Score: <strong>{pbo_result['pbo']:.4f}</strong>
        ({pbo_result['n_combinations']} combinations tested)</p>
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
{manifest_section}
</body>
</html>"""


def save_report(html: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(html)
