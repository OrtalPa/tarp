"""Cosine-based representation shift (simple baseline to CKA).

Per-example cosine between the *same* input's frozen vs fine-tuned vector, averaged.
Low mean cosine = the vectors moved a lot. Unlike CKA this is alignment-sensitive, which
is exactly why we keep CKA as primary and this as a contrast.
"""

from __future__ import annotations

import numpy as np


def paired_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """a, b shape (N, H), row i of a paired with row i of b -> mean cosine."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float((an * bn).sum(axis=1).mean())


def cosine_per_layer(reps_a: np.ndarray, reps_b: np.ndarray) -> list[float]:
    """reps_* shape (L, N, H) aligned -> mean paired cosine at each layer."""
    return [paired_cosine(reps_a[i], reps_b[i]) for i in range(reps_a.shape[0])]
