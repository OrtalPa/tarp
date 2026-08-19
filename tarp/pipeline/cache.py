"""On-disk cache layout under ``outputs/`` (gitignored).

```
outputs/
  reps/<extraction_key>.npz            # per-layer pooled representations
  runs/ft/<finetune_key>/adapter/      # saved LoRA adapter
  runs/ft/<finetune_key>/meta.json     # {test_acc, val_acc}
  runs/probe/<model>__<ds>__<pool>__s<seed>.json
  results/results.parquet
  figures/
```
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs")


def reps_path(key: str) -> Path:
    return ROOT / "reps" / f"{key}.npz"


def ft_dir(finetune_key: str) -> Path:
    return ROOT / "runs" / "ft" / finetune_key


def probe_path(model: str, dataset: str, pooling: str, seed: int) -> Path:
    return ROOT / "runs" / "probe" / f"{model}__{dataset}__{pooling}__s{seed}.json"


def results_path() -> Path:
    return ROOT / "results" / "results.parquet"


def figures_dir() -> Path:
    d = ROOT / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- representations ---------------------------------------------------------
def has_reps(key: str) -> bool:
    return reps_path(key).exists()


def load_reps(key: str) -> np.ndarray | None:
    p = reps_path(key)
    return np.load(p)["reps"] if p.exists() else None


def save_reps(key: str, arr: np.ndarray) -> None:
    # float16 halves disk/IO; pooled reps tolerate it (metrics upcast to float64).
    p = reps_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, reps=arr.astype(np.float16))


# --- json --------------------------------------------------------------------
def load_json(path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def save_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
