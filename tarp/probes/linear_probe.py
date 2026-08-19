"""Frozen linear probe.

Trains a logistic-regression classifier on frozen representations (per layer) and reads
accuracy on the eval split. This is the "Joint Frozen" analog: the encoder is untouched,
only a linear head is fit. The headline ``frozen_acc`` is the **final-layer** probe
(the encoder-output analog, matching MulTaBench); per-layer accuracies feed RQ3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    per_layer: list[float]   # accuracy at each layer (index 0 = embeddings)
    final: float             # accuracy at the last layer (headline frozen_acc)
    best_layer: int
    best: float


def _probe_layer(x_tr, y_tr, x_te, y_te, standardize: bool, seed: int, max_iter: int) -> float:
    if standardize:
        scaler = StandardScaler().fit(x_tr)
        x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)
    clf = LogisticRegression(max_iter=max_iter, C=1.0, random_state=seed)
    clf.fit(x_tr, y_tr)
    return float((clf.predict(x_te) == y_te).mean())


def probe_all_layers(
    train_reps: np.ndarray,   # (L+1, Ntr, H)
    train_y: np.ndarray,
    test_reps: np.ndarray,    # (L+1, Nte, H)
    test_y: np.ndarray,
    standardize: bool = True,
    seed: int = 0,
    max_iter: int = 1000,
    max_train: int | None = 6000,   # cap probe train size for speed (seeded, consistent per layer)
) -> ProbeResult:
    n_layers = train_reps.shape[0]
    idx = np.arange(train_reps.shape[1])
    if max_train is not None and len(idx) > max_train:
        idx = np.random.default_rng(seed).choice(len(idx), size=max_train, replace=False)
    tr_y = train_y[idx]
    accs = [
        _probe_layer(train_reps[i][idx], tr_y, test_reps[i], test_y, standardize, seed, max_iter)
        for i in range(n_layers)
    ]
    best_layer = int(np.argmax(accs))
    return ProbeResult(
        per_layer=accs, final=accs[-1], best_layer=best_layer, best=accs[best_layer]
    )
