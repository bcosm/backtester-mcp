"""DuckDB-backed dataset registry and run persistence."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np


_DB_DIR = Path.home() / ".backtester-mcp"
_DB_PATH = _DB_DIR / "registry.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT,
    path TEXT,
    hash TEXT,
    schema TEXT,
    row_count INTEGER,
    date_range TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    strategy_name TEXT,
    strategy_code TEXT,
    params TEXT,
    dataset_id TEXT,
    metrics TEXT,
    validation TEXT,
    report_path TEXT,
    manifest TEXT,
    git_sha TEXT,
    engine_version TEXT,
    created_at TEXT
);
"""


def _get_db() -> duckdb.DuckDBPyConnection:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(_DB_PATH))
    for stmt in _SCHEMA.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)
    return con


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def register_dataset(
    name: str,
    path: str,
    row_count: int,
    columns: list[str],
    date_range: tuple[str, str] | None = None,
) -> str:
    """Register a dataset file. Deduplicates by content hash."""
    resolved = str(Path(path).resolve())
    file_hash = _hash_file(resolved)

    con = _get_db()
    existing = con.execute(
        "SELECT id FROM datasets WHERE hash = ?", [file_hash]
    ).fetchone()
    if existing:
        con.close()
        return existing[0]

    dataset_id = str(uuid.uuid4())
    con.execute(
        "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            dataset_id,
            name,
            resolved,
            file_hash,
            json.dumps(columns),
            row_count,
            json.dumps(list(date_range)) if date_range else None,
            datetime.now(timezone.utc).isoformat(),
        ],
    )
    con.close()
    return dataset_id


def get_dataset(dataset_id: str) -> dict | None:
    con = _get_db()
    row = con.execute(
        "SELECT * FROM datasets WHERE id = ?", [dataset_id]
    ).fetchone()
    con.close()
    if not row:
        return None
    return _dataset_row_to_dict(row)


def list_datasets() -> list[dict]:
    con = _get_db()
    rows = con.execute(
        "SELECT * FROM datasets ORDER BY created_at DESC"
    ).fetchall()
    con.close()
    return [_dataset_row_to_dict(r) for r in rows]


def _dataset_row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "name": row[1],
        "path": row[2],
        "hash": row[3],
        "columns": json.loads(row[4]) if row[4] else [],
        "row_count": row[5],
        "date_range": json.loads(row[6]) if row[6] else None,
        "created_at": row[7],
    }


def profile_dataset(dataset_id: str) -> dict:
    """Load a registered dataset and compute detailed statistics."""
    ds = get_dataset(dataset_id)
    if not ds:
        raise ValueError(f"dataset not found: {dataset_id}")

    from backtester_mcp.data import load, to_arrays

    df = load(ds["path"])
    arrays = to_arrays(df)
    profile = {
        "id": dataset_id,
        "name": ds["name"],
        "row_count": len(df),
        "columns": list(df.columns),
        "ohlcv_available": {
            col: col in arrays for col in ["open", "high", "low", "close", "volume"]
        },
    }

    if hasattr(df.index, "min") and hasattr(df.index, "max"):
        try:
            profile["date_range"] = (
                str(df.index.min()), str(df.index.max())
            )
        except Exception:
            pass

    if "close" in arrays:
        close = arrays["close"]
        rets = np.diff(close) / close[:-1]
        profile["price_range"] = (float(np.min(close)), float(np.max(close)))
        profile["returns_stats"] = {
            "mean": round(float(np.mean(rets)), 6),
            "std": round(float(np.std(rets)), 6),
            "min": round(float(np.min(rets)), 6),
            "max": round(float(np.max(rets)), 6),
        }
        # detect bounded price (prediction market: 0-1 range)
        if float(np.min(close)) >= 0 and float(np.max(close)) <= 1:
            profile["price_type"] = "bounded_0_1"
        else:
            profile["price_type"] = "continuous"

    # detect frequency from index spacing
    if hasattr(df.index, "to_series"):
        try:
            diffs = df.index.to_series().diff().dropna()
            median_gap = diffs.median()
            hours = median_gap.total_seconds() / 3600
            if hours < 0.1:
                profile["frequency"] = "tick"
            elif hours < 1.5:
                profile["frequency"] = "hourly"
            elif hours < 8:
                profile["frequency"] = "4h"
            elif hours < 30:
                profile["frequency"] = "daily"
            else:
                profile["frequency"] = "weekly_or_longer"
        except Exception:
            pass

    # missing data
    total_cells = len(df) * len(df.columns)
    missing = int(df.isna().sum().sum()) if total_cells > 0 else 0
    profile["missing_pct"] = round(missing / total_cells * 100, 2) if total_cells else 0.0

    return profile


def save_run(
    strategy_name: str,
    strategy_code: str = "",
    params: dict | None = None,
    dataset_id: str = "",
    metrics: dict | None = None,
    validation: dict | None = None,
    report_path: str = "",
    manifest: dict | None = None,
    git_sha: str = "",
    engine_version: str = "0.1.0",
) -> str:
    """Persist a run and return its UUID."""
    run_id = str(uuid.uuid4())
    con = _get_db()
    con.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run_id,
            strategy_name,
            strategy_code,
            json.dumps(params or {}),
            dataset_id,
            json.dumps(metrics or {}, default=str),
            json.dumps(validation or {}, default=str),
            report_path,
            json.dumps(manifest or {}, default=str),
            git_sha,
            engine_version,
            datetime.now(timezone.utc).isoformat(),
        ],
    )
    con.close()
    return run_id


def get_run(run_id: str) -> dict | None:
    con = _get_db()
    row = con.execute(
        "SELECT * FROM runs WHERE id = ?", [run_id]
    ).fetchone()
    con.close()
    if not row:
        return None
    return _run_row_to_dict(row)


def list_runs(
    dataset_id: str | None = None,
    strategy_name: str | None = None,
    limit: int = 50,
) -> list[dict]:
    con = _get_db()
    query = "SELECT * FROM runs WHERE 1=1"
    args = []
    if dataset_id:
        query += " AND dataset_id = ?"
        args.append(dataset_id)
    if strategy_name:
        query += " AND strategy_name = ?"
        args.append(strategy_name)
    query += f" ORDER BY created_at DESC LIMIT {limit}"
    rows = con.execute(query, args).fetchall()
    con.close()
    return [_run_row_to_dict(r) for r in rows]


def compare_runs(run_ids: list[str]) -> list[dict]:
    con = _get_db()
    placeholders = ", ".join(["?"] * len(run_ids))
    rows = con.execute(
        f"SELECT * FROM runs WHERE id IN ({placeholders})", run_ids
    ).fetchall()
    con.close()
    return [_run_row_to_dict(r) for r in rows]


def _run_row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "strategy_name": row[1],
        "strategy_code": row[2],
        "params": json.loads(row[3]) if row[3] else {},
        "dataset_id": row[4],
        "metrics": json.loads(row[5]) if row[5] else {},
        "validation": json.loads(row[6]) if row[6] else {},
        "report_path": row[7],
        "manifest": json.loads(row[8]) if row[8] else {},
        "git_sha": row[9],
        "engine_version": row[10],
        "created_at": row[11],
    }
