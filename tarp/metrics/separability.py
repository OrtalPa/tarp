"""Frozen-only separability metrics (RQ3 candidate predictors).

Computed on frozen representations + labels — no fine-tuning. Hypothesis (H3): datasets
whose frozen classes are poorly separated benefit more from task-aware adaptation. Used by
Exp5; not needed by the Exp1 slice.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def silhouette(X: np.ndarray, y: np.ndarray, max_n: int = 4000, seed: int = 0) -> float:
    X, y = _subsample(_f32(X), y, max_n, seed)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(silhouette_score(X, y))


def intra_inter_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Mean within-class spread / mean between-centroid distance. Lower = better separated."""
    X = _f32(X)
    classes = np.unique(y)
    centroids = np.stack([X[y == c].mean(axis=0) for c in classes])
    intra = np.mean([np.linalg.norm(X[y == c] - centroids[i], axis=1).mean()
                     for i, c in enumerate(classes)])
    n = len(classes)
    if n < 2:
        return float("nan")
    dists = [np.linalg.norm(centroids[i] - centroids[j])
             for i in range(n) for j in range(i + 1, n)]
    inter = float(np.mean(dists))
    return float(intra / inter) if inter > 0 else float("nan")


def knn_accuracy(X: np.ndarray, y: np.ndarray, k: int = 5, cv: int = 3,
                 max_n: int = 4000, seed: int = 0) -> float:
    """Cross-validated k-NN accuracy on frozen features (a linear-free separability read)."""
    X, y = _subsample(_f32(X), y, max_n, seed)
    clf = KNeighborsClassifier(n_neighbors=k)
    return float(cross_val_score(clf, X, y, cv=cv).mean())


def anisotropy(X: np.ndarray, max_pairs: int = 5000, seed: int = 0) -> float:
    """Mean absolute cosine between random pairs of vectors (high = anisotropic space)."""
    X = _f32(X)
    rng = np.random.default_rng(seed)
    n = len(X)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = (Xn[i] * Xn[j]).sum(axis=1)
    return float(np.abs(cos).mean())


def effective_dimension(X: np.ndarray) -> float:
    """Participation ratio of PCA spectrum: (Σλ)^2 / Σλ^2. Higher = info spread over more dims."""
    X = _f32(X)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12))


def _f32(X: np.ndarray) -> np.ndarray:
    """Upcast cached reps (stored float16) to float32 — np.linalg / sklearn need >=float32."""
    return np.asarray(X, dtype=np.float32)


def _subsample(X, y, max_n, seed):
    if len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]
