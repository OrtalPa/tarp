"""Tidy long-format results store (one parquet).

One row per observation. Scalar (non-layer) metrics use ``layer = -1``. Experiments append
rows; analysis/plots read them back. Rows are de-duplicated on the identity key.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from tarp.pipeline import cache

COLUMNS = [
    "experiment", "model", "dataset", "seed", "pooling", "ft_regime",
    "condition", "layer", "metric", "value", "run_hash", "timestamp",
]
_KEY = ["experiment", "model", "dataset", "seed", "pooling", "ft_regime", "condition", "layer", "metric"]

NO_LAYER = -1  # sentinel for scalar metrics


def load() -> pd.DataFrame:
    p = cache.results_path()
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=COLUMNS)


def append_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return load()
    new = pd.DataFrame(rows)
    ts = datetime.now().isoformat(timespec="seconds")
    if "timestamp" not in new:
        new["timestamp"] = ts
    for c in COLUMNS:
        if c not in new:
            new[c] = None
    new = new[COLUMNS]
    new["layer"] = new["layer"].fillna(NO_LAYER).astype(int)

    df = pd.concat([load(), new], ignore_index=True)
    df = df.drop_duplicates(subset=_KEY, keep="last").reset_index(drop=True)

    p = cache.results_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df
