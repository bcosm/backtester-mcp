"""Tests for examples running without errors."""

import subprocess
import sys
import pytest
from pathlib import Path

PYTHON = sys.executable


def _run_example(name, timeout=120):
    path = Path("examples") / name
    if not path.exists():
        pytest.skip(f"{name} not found")
    return subprocess.run(
        [PYTHON, str(path)],
        capture_output=True, text=True, timeout=timeout,
    )


def test_quickstart():
    r = _run_example("quickstart.py")
    assert r.returncode == 0
    assert "sharpe" in r.stdout.lower() or "total_return" in r.stdout.lower()


def test_pbo_demo():
    r = _run_example("pbo_demo.py")
    assert r.returncode == 0
    assert "PBO" in r.stdout or "pbo" in r.stdout.lower()


def test_mcp_example():
    r = _run_example("mcp_example.py")
    assert r.returncode == 0
