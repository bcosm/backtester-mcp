"""Tests for DuckDB store module."""

import pytest
from backtester_mcp.store import (
    register_dataset, get_dataset, list_datasets, profile_dataset,
    save_run, get_run, list_runs, compare_runs, _get_db,
)


@pytest.fixture(autouse=True)
def clean_db():
    """Ensure clean state for each test."""
    con = _get_db()
    con.execute("DELETE FROM datasets")
    con.execute("DELETE FROM runs")
    con.close()
    yield


def test_register_dataset(spy_data_path):
    ds_id = register_dataset(
        name="spy_test", path=spy_data_path,
        row_count=2765, columns=["open", "high", "low", "close", "volume"],
    )
    assert len(ds_id) == 36  # UUID format


def test_register_dataset_dedup(spy_data_path):
    id1 = register_dataset(
        name="spy_1", path=spy_data_path,
        row_count=2765, columns=["close"],
    )
    id2 = register_dataset(
        name="spy_2", path=spy_data_path,
        row_count=2765, columns=["close"],
    )
    assert id1 == id2  # same content hash


def test_get_dataset(spy_data_path):
    ds_id = register_dataset(
        name="spy_get", path=spy_data_path,
        row_count=100, columns=["close"],
    )
    ds = get_dataset(ds_id)
    assert ds is not None
    assert ds["name"] == "spy_get"


def test_get_dataset_not_found():
    assert get_dataset("nonexistent-id") is None


def test_list_datasets(spy_data_path):
    register_dataset(
        name="ds1", path=spy_data_path,
        row_count=100, columns=["close"],
    )
    datasets = list_datasets()
    assert len(datasets) >= 1


def test_profile_dataset(spy_data_path):
    ds_id = register_dataset(
        name="spy_profile", path=spy_data_path,
        row_count=2765, columns=["close"],
    )
    profile = profile_dataset(ds_id)
    assert profile["row_count"] > 0
    assert "close" in profile["columns"]
    assert "ohlcv_available" in profile


def test_save_and_get_run():
    run_id = save_run(
        strategy_name="test_strat",
        params={"fast": 10},
        metrics={"sharpe": 1.5, "total_return": 0.2},
    )
    run = get_run(run_id)
    assert run is not None
    assert run["strategy_name"] == "test_strat"
    assert run["metrics"]["sharpe"] == 1.5


def test_list_runs_filter():
    save_run(strategy_name="strat_a", metrics={"sharpe": 1.0})
    save_run(strategy_name="strat_b", metrics={"sharpe": 2.0})

    all_runs = list_runs()
    assert len(all_runs) >= 2

    filtered = list_runs(strategy_name="strat_a")
    assert all(r["strategy_name"] == "strat_a" for r in filtered)


def test_compare_runs():
    id1 = save_run(strategy_name="a", metrics={"sharpe": 1.0})
    id2 = save_run(strategy_name="b", metrics={"sharpe": 2.0})
    result = compare_runs([id1, id2])
    assert len(result) == 2


def test_get_run_not_found():
    assert get_run("nonexistent-run-id") is None
