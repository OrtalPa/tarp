"""Linear CKA (Centered Kernel Alignment), Kornblith et al. 2019.

Measures similarity between two representation *spaces* over the same N inputs. Invariant
to orthogonal transforms and isotropic scaling; 1.0 for identical (up to those) spaces.
Primary metric for representation shift (frozen layer L vs fine-tuned layer L).
"""

from __future__ import annotations

import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """X (N, d1), Y (N, d2) over the same N inputs -> CKA in [0, 1]."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    xx = np.linalg.norm(X.T @ X, ord="fro")
    yy = np.linalg.norm(Y.T @ Y, ord="fro")
    denom = xx * yy
    if denom == 0.0:
        return float("nan")
    return float(xy / denom)


def cka_per_layer(reps_a: np.ndarray, reps_b: np.ndarray,
                  max_n: int = 8000, seed: int = 0) -> list[float]:
    """reps_* shape (L, N, H) aligned on the same inputs -> CKA at each layer.

    For very large N (e.g. 60k-example test splits) the float64 upcast inside ``linear_cka``
    is memory-heavy, especially for deep models (ModernBERT, 23 layers). CKA is a stable
    statistic, so we estimate it on a fixed seeded subsample of ``max_n`` inputs — applied to
    the SAME indices of both stacks (they must stay aligned). No-op when N <= max_n."""
    if reps_a.shape[0] != reps_b.shape[0]:
        raise ValueError(f"layer mismatch: {reps_a.shape[0]} vs {reps_b.shape[0]}")
    n = reps_a.shape[1]
    if max_n is not None and n > max_n:
        idx = np.random.default_rng(seed).choice(n, size=max_n, replace=False)
        reps_a = reps_a[:, idx, :]
        reps_b = reps_b[:, idx, :]
    return [linear_cka(reps_a[i], reps_b[i]) for i in range(reps_a.shape[0])]
