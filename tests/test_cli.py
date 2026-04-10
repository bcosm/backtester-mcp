"""CLI integration tests."""

import subprocess
import sys
import pytest
from pathlib import Path


PYTHON = sys.executable
CLI = [PYTHON, "-m", "backtester_mcp.cli"]

SPY = "datasets/spy_daily.parquet"
MOMENTUM = "strategies/momentum.py"


def _run(*args, timeout=120):
    result = subprocess.run(
        [*CLI, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return result


@pytest.fixture(autouse=True)
def check_data():
    if not Path(SPY).exists() or not Path(MOMENTUM).exists():
        pytest.skip("test datasets/strategies not available")


def test_backtest_basic():
    r = _run("backtest", "-s", MOMENTUM, "-d", SPY)
    assert r.returncode == 0
    assert "sharpe" in r.stdout


def test_backtest_robustness():
    r = _run("backtest", "-s", MOMENTUM, "-d", SPY, "--robustness")
    assert r.returncode == 0
    assert "Bootstrap Sharpe" in r.stdout
    assert "Deflated Sharpe" in r.stdout
    assert "PBO" in r.stdout


def test_backtest_execution_scenarios():
    r = _run("backtest", "-s", MOMENTUM, "-d", SPY, "--execution-scenarios")
    assert r.returncode == 0
    assert "optimistic" in r.stdout
    assert "conservative" in r.stdout


def test_backtest_realistic_fills():
    r = _run("backtest", "-s", MOMENTUM, "-d", SPY, "--realistic-fills")
    assert r.returncode == 0
    assert "Fill model" in r.stdout


def test_backtest_set_params():
    r = _run("backtest", "-s", MOMENTUM, "-d", SPY,
             "--set", "fast_period=20", "--set", "slow_period=100")
    assert r.returncode == 0
    assert "sharpe" in r.stdout


def test_backtest_save_run():
    r = _run("backtest", "-s", MOMENTUM, "-d", SPY, "--save-run")
    assert r.returncode == 0
    assert "Run saved:" in r.stdout


def test_optimize():
    r = _run("optimize", "-s", MOMENTUM, "-d", SPY,
             "-p", "fast_period:5:30", "-p", "slow_period:20:100",
             "-n", "10")
    assert r.returncode == 0
    assert "Best params" in r.stdout


def test_report(tmp_path):
    out = str(tmp_path / "test_report.html")
    r = _run("report", "-s", MOMENTUM, "-d", SPY, "-o", out)
    assert r.returncode == 0
    assert Path(out).exists()
    content = Path(out).read_text()
    assert "backtester-mcp" in content


def test_list_runs():
    r = _run("list-runs")
    assert r.returncode == 0


def test_help():
    r = _run("--help")
    assert r.returncode == 0
    assert "backtest" in r.stdout


def test_backtest_missing_strategy():
    r = _run("backtest", "-s", "nonexistent.py", "-d", SPY)
    assert r.returncode != 0
