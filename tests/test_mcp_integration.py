"""MCP server integration tests over stdio.

Tests the real server subprocess with JSON-RPC protocol, verifying
initialization handshake, tool listing, and tool call responses.
"""

import json
import os
import subprocess
import sys
import threading
import time

import pytest


PYTHON = sys.executable
DATA_PATH = os.path.abspath("datasets/spy_daily.parquet")


MOMENTUM_CODE = """import numpy as np
DEFAULT_PARAMS = {"fast_period": 10, "slow_period": 50}
def generate_signals(prices, fast_period=10, slow_period=50, **params):
    n = len(prices)
    signals = np.zeros(n)
    for i in range(slow_period, n):
        fast = float(np.mean(prices[i-fast_period:i]))
        slow = float(np.mean(prices[i-slow_period:i]))
        signals[i] = 1.0 if fast > slow else -1.0
    return signals
"""

# Pre-warm JIT and import scipy before starting MCP server to avoid
# Numba hanging on Windows when stdout is captured by subprocess, and
# scipy writing to stdout on first import.
_SERVER_SCRIPT = """
import sys, os
from backtester_mcp.engine import backtest
import numpy as np
prices = np.array([100.0, 101.0, 99.0, 102.0, 98.0, 97.0, 103.0])
signals = np.array([0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0])
backtest(prices, signals)
import scipy.stats
from backtester_mcp.mcp_server import run_server
run_server(transport="stdio")
"""


class MCPClient:
    """Minimal MCP client for testing over stdio."""

    def __init__(self):
        self._id = 0
        self.proc = subprocess.Popen(
            [PYTHON, "-c", _SERVER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # wait for server to start
        time.sleep(1.5)
        self._initialize()

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def send(self, method: str, params: dict = None, is_notification: bool = False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not is_notification:
            msg["id"] = self._next_id()
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        if is_notification:
            return None
        return self._recv()

    def call_tool(self, name: str, arguments: dict = None, timeout: int = 30):
        """Call an MCP tool and return the parsed JSON from the text content."""
        resp = self.send("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        assert "result" in resp, f"Expected result, got: {resp}"
        content = resp["result"].get("content", [])
        assert len(content) > 0, "Empty content in tool response"
        text = content[0].get("text", "")
        is_error = resp["result"].get("isError", False)
        return json.loads(text), is_error

    def _recv(self, timeout: int = 120):
        result = [None]

        def _read():
            try:
                result[0] = self.proc.stdout.readline()
            except Exception:
                pass

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=timeout)
        assert result[0] is not None, "No response from MCP server (timeout)"
        return json.loads(result[0])

    def _initialize(self):
        resp = self.send("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "pytest-mcp", "version": "1.0"},
        })
        assert "result" in resp
        assert "serverInfo" in resp["result"]
        self.server_info = resp["result"]
        self.send("notifications/initialized", params={}, is_notification=True)

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture(scope="module")
def mcp():
    """Start MCP server once for all tests in this module."""
    if not os.path.exists(DATA_PATH):
        pytest.skip("spy_daily.parquet not available")
    client = MCPClient()
    yield client
    client.close()


def test_initialization_handshake(mcp):
    assert mcp.server_info["serverInfo"]["name"] == "backtester-mcp"
    assert "capabilities" in mcp.server_info
    assert "tools" in mcp.server_info["capabilities"]


EXPECTED_TOOLS = [
    "backtest_strategy",
    "validate_robustness",
    "validate_strategy",
    "optimize_parameters",
    "compare_strategies",
    "register_dataset",
    "profile_dataset",
    "save_run",
    "list_runs",
    "load_run",
    "compare_runs",
    "generate_report",
    "strategy_template",
]


def test_tools_list_returns_all_13(mcp):
    resp = mcp.send("tools/list", {})
    tools = resp["result"]["tools"]
    names = [t["name"] for t in tools]
    assert len(names) == 13
    for expected in EXPECTED_TOOLS:
        assert expected in names, f"Missing tool: {expected}"
    for tool in tools:
        assert "description" in tool, f"{tool['name']} missing description"
        assert "inputSchema" in tool, f"{tool['name']} missing inputSchema"


def test_backtest_strategy_returns_metrics(mcp):
    data, is_error = mcp.call_tool("backtest_strategy", {
        "strategy_code": MOMENTUM_CODE,
        "data_path": DATA_PATH,
    })
    assert not is_error
    assert "metrics" in data
    metrics = data["metrics"]
    for key in ("sharpe", "sortino", "max_drawdown", "total_return",
                "cagr", "volatility", "win_rate", "profit_factor"):
        assert key in metrics, f"Missing metric: {key}"
    assert isinstance(metrics["sharpe"], float)
    assert "reproducibility" in data


def test_validate_robustness_returns_bootstrap_dsr_pbo(mcp):
    data, is_error = mcp.call_tool("validate_robustness", {
        "strategy_code": MOMENTUM_CODE,
        "data_path": DATA_PATH,
        "params": {"fast_period": 10, "slow_period": 50},
    })
    assert not is_error
    assert "bootstrap_sharpe" in data
    bs = data["bootstrap_sharpe"]
    assert "sharpe" in bs
    assert "ci_lower" in bs
    assert "ci_upper" in bs
    assert "ci_includes_zero" in bs

    assert "deflated_sharpe" in data
    dsr = data["deflated_sharpe"]
    assert "dsr" in dsr
    assert "p_value" in dsr

    assert "pbo" in data
    pbo = data["pbo"]
    assert "pbo" in pbo
    assert "n_combinations" in pbo


def test_validate_strategy_returns_verdict(mcp):
    data, is_error = mcp.call_tool("validate_strategy", {
        "strategy_code": MOMENTUM_CODE,
        "data_path": DATA_PATH,
        "params": {"fast_period": 10, "slow_period": 50},
    })
    assert not is_error
    assert "verdict" in data
    assert data["verdict"] in ("pass", "caution")
    assert "reasons" in data
    assert isinstance(data["reasons"], list)
    assert "metrics" in data
    assert "bootstrap_sharpe" in data
    assert "scenarios" in data
    scenarios = data["scenarios"]
    for mode in ("optimistic", "base", "conservative"):
        assert mode in scenarios, f"Missing scenario: {mode}"


def test_register_dataset_with_inline_csv(mcp):
    csv_text = "date,close\n"
    for i in range(100):
        csv_text += f"2024-01-{(i % 28) + 1:02d},{100 + i * 0.5}\n"

    data, is_error = mcp.call_tool("register_dataset", {
        "csv_text": csv_text,
        "name": "test_inline",
    })
    assert not is_error
    assert "dataset_id" in data
    assert data["rows"] == 100


def test_save_list_load_run_cycle(mcp):
    # Save a run
    data, is_error = mcp.call_tool("save_run", {
        "strategy_name": "mcp_test_strat",
        "metrics": {"sharpe": 1.23, "total_return": 0.15},
    })
    assert not is_error
    run_id = data["run_id"]
    assert len(run_id) == 36  # UUID

    # List runs
    data, is_error = mcp.call_tool("list_runs", {})
    assert not is_error
    runs = data["runs"]
    run_ids = [r["id"] for r in runs]
    assert run_id in run_ids

    # Load run
    data, is_error = mcp.call_tool("load_run", {"run_id": run_id})
    assert not is_error
    assert data["strategy_name"] == "mcp_test_strat"
    assert data["metrics"]["sharpe"] == 1.23


def test_strategy_template_returns_valid_python(mcp):
    data, is_error = mcp.call_tool("strategy_template", {
        "strategy_type": "momentum",
    })
    assert not is_error
    assert "code" in data
    code = data["code"]
    assert "generate_signals" in code
    assert "DEFAULT_PARAMS" in code
    # verify it's valid Python
    compile(code, "<template>", "exec")


def test_error_bad_strategy_code(mcp):
    """Invalid strategy code should return an error, not crash the server."""
    resp = mcp.send("tools/call", {
        "name": "backtest_strategy",
        "arguments": {
            "strategy_code": "this is not valid python!!!",
            "data_path": DATA_PATH,
        },
    })
    # Server should respond (not crash/hang)
    assert resp is not None
    # Either isError in result or error in response
    if "result" in resp:
        is_error = resp["result"].get("isError", False)
        assert is_error, "Expected isError for invalid strategy code"
    elif "error" in resp:
        pass  # JSON-RPC error is also acceptable
    else:
        pytest.fail("Expected error response for bad strategy code")


def test_error_missing_data_file(mcp):
    """Nonexistent data path should return error."""
    resp = mcp.send("tools/call", {
        "name": "backtest_strategy",
        "arguments": {
            "strategy_code": MOMENTUM_CODE,
            "data_path": "/nonexistent/path/data.parquet",
        },
    })
    assert resp is not None
    if "result" in resp:
        is_error = resp["result"].get("isError", False)
        assert is_error, "Expected isError for missing data"
    elif "error" in resp:
        pass
    else:
        pytest.fail("Expected error response for missing file")


def test_server_still_alive_after_errors(mcp):
    """Verify the server didn't crash from the error tests above."""
    data, is_error = mcp.call_tool("strategy_template", {
        "strategy_type": "custom",
    })
    assert not is_error
    assert "code" in data
