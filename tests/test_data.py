"""Tests for data loading module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from backtester_mcp.data import load, to_arrays


def test_load_parquet(spy_data_path):
    df = load(spy_data_path)
    assert len(df) > 100
    assert "close" in df.columns


def test_load_csv(tmp_dir):
    csv_path = tmp_dir / "test.csv"
    csv_path.write_text(
        "date,close,volume\n"
        "2024-01-01,100.0,1000000\n"
        "2024-01-02,101.0,1100000\n"
        "2024-01-03,99.5,900000\n"
    )
    df = load(str(csv_path))
    assert len(df) == 3
    assert "close" in df.columns


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load("nonexistent.parquet")


def test_load_unsupported_format(tmp_dir):
    p = tmp_dir / "test.xyz"
    p.write_text("data")
    with pytest.raises(ValueError, match="unsupported"):
        load(str(p))


def test_column_normalization(tmp_dir):
    csv_path = tmp_dir / "test.csv"
    csv_path.write_text(
        "Date,Close,Volume,Adj Close\n"
        "2024-01-01,100.0,1000,99.5\n"
        "2024-01-02,101.0,1100,100.5\n"
    )
    df = load(str(csv_path))
    assert "close" in df.columns


def test_to_arrays(spy_data_path):
    df = load(spy_data_path)
    arrays = to_arrays(df)
    assert "close" in arrays
    assert isinstance(arrays["close"], np.ndarray)
    assert arrays["close"].dtype == np.float64


def test_to_arrays_ohlcv(tmp_dir):
    csv_path = tmp_dir / "ohlcv.csv"
    csv_path.write_text(
        "date,open,high,low,close,volume\n"
        "2024-01-01,99,101,98,100,1000000\n"
        "2024-01-02,100,102,99,101,1100000\n"
    )
    df = load(str(csv_path))
    arrays = to_arrays(df)
    assert set(arrays.keys()) == {"open", "high", "low", "close", "volume"}


def test_duckdb_query():
    df = load("SELECT 1 as close, 100 as volume")
    assert len(df) == 1
    assert "close" in df.columns
