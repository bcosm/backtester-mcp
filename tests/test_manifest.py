"""Tests for manifest module."""

import json
import pytest
from pathlib import Path

from backtester_mcp.manifest import create_manifest, save_manifest, load_manifest


def test_create_manifest(spy_data_path):
    m = create_manifest(
        strategy_name="test",
        params={"fast": 10},
        data_path=spy_data_path,
        data_rows=100,
        date_range=("2020-01-01", "2024-01-01"),
        metrics={"sharpe": 1.5},
    )
    assert m["strategy"] == "test"
    assert m["parameters"] == {"fast": 10}
    assert m["data_rows"] == 100
    assert "timestamp" in m
    assert "data_hash" in m


def test_manifest_roundtrip(tmp_dir, spy_data_path):
    m = create_manifest(
        strategy_name="roundtrip",
        params={},
        data_path=spy_data_path,
        data_rows=50,
        date_range=None,
        metrics={"sharpe": 0.5},
    )
    path = str(tmp_dir / "manifest.json")
    save_manifest(m, path)

    loaded = load_manifest(path)
    assert loaded["strategy"] == "roundtrip"
    assert loaded["metrics"]["sharpe"] == 0.5


def test_create_manifest_missing_file():
    m = create_manifest(
        strategy_name="test",
        params={},
        data_path="nonexistent.parquet",
        data_rows=0,
        date_range=None,
        metrics={},
    )
    assert m["data_hash"] == "unknown"
