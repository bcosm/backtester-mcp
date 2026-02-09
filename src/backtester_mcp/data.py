from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb

_OHLCV_ALIASES = {
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "price", "adj close", "adj_close", "adjclose"],
    "volume": ["volume", "vol", "v"],
}

_DATE_ALIASES = ["date", "timestamp", "time", "datetime", "dt", "ts"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_lower = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_lower)

    # find and set date index
    for alias in _DATE_ALIASES:
        if alias in df.columns:
            df[alias] = pd.to_datetime(df[alias])
            df = df.set_index(alias).sort_index()
            break

    # normalize OHLCV names
    rename = {}
    for canon, aliases in _OHLCV_ALIASES.items():
        if canon in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = canon
                break
    if rename:
        df = df.rename(columns=rename)

    return df


def load(source: str) -> pd.DataFrame:
    """Load price data from CSV, Parquet, or DuckDB query."""
    source = source.strip()

    if source.upper().startswith("SELECT ") or source.upper().startswith("FROM "):
        return _load_duckdb(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {source}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in (".csv", ".tsv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported file format: {suffix}")

    return _normalize_columns(df)


def _load_duckdb(query: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(query).fetchdf()
    con.close()
    return _normalize_columns(df)


def to_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract available price/volume arrays from a normalized DataFrame."""
    out = {}
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            out[col] = df[col].to_numpy(dtype=np.float64)
    return out
