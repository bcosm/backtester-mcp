"""Reproducible run manifests."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"


def create_manifest(
    strategy_name: str,
    params: dict,
    data_path: str,
    data_rows: int,
    date_range: tuple[str, str] | None,
    metrics: dict,
) -> dict:
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine_version": "0.1.0",
        "strategy": strategy_name,
        "parameters": params,
        "data_hash": _hash_file(data_path) if Path(data_path).exists() else "unknown",
        "data_rows": data_rows,
        "date_range": list(date_range) if date_range else None,
        "metrics": {k: v for k, v in metrics.items()
                    if not isinstance(v, np.ndarray)},
    }
    return manifest


def save_manifest(manifest: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
