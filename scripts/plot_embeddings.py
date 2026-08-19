"""Qualitative embedding visualization: 2D PCA (via SVD) of the final-layer test
representations, frozen vs. fine-tuned, colored by class.

Reuses the cached reps that Exp1 already extracted (``run_conditions`` — no accelerator, no
re-fine-tuning). Illustrates the Exp4 finding directly: for a high-gap task (emotion) the
frozen classes are an overlapping blob that fine-tuning pulls apart; for a low-gap task
(banking77/agnews) the frozen classes are already reasonably arranged.

    uv run python scripts/plot_embeddings.py                 # roberta on emotion/trec/agnews
    uv run python scripts/plot_embeddings.py --model bert --datasets emotion banking77
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.decomposition import PCA

from tarp import plots
from tarp.config import RunConfig
from tarp.pipeline import run_conditions
from tarp.registry import resolve_dataset, resolve_model


def _project(X: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, float]:
    """PCA (SVD) to 2D on float32 reps; return coords + total explained-variance ratio."""
    X = np.asarray(X, dtype=np.float32)
    if len(X) > n:  # thin out for a legible scatter
        X = X[np.random.default_rng(seed).choice(len(X), n, replace=False)]
    p = PCA(n_components=2, random_state=seed).fit(X)
    return p.transform(X), float(p.explained_variance_ratio_.sum())


def _subsample_idx(n_total: int, n: int, seed: int) -> np.ndarray:
    if n_total <= n:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta")
    ap.add_argument("--datasets", nargs="+", default=["emotion", "trec", "agnews"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-points", type=int, default=800)
    args = ap.parse_args()

    panels = []
    for ds in args.datasets:
        cfg = RunConfig(model=resolve_model(args.model), dataset=resolve_dataset(ds), seed=args.seed)
        rr = run_conditions(cfg)
        # one shared subsample so frozen/ft panels show the *same* points, colored identically
        idx = _subsample_idx(len(rr.labels), args.max_points, args.seed)
        y = np.asarray(rr.labels)[idx]
        fxy, fev = _project(rr.frozen_reps[-1][idx], args.max_points, args.seed)
        txy, tev = _project(rr.ft_reps[-1][idx], args.max_points, args.seed)
        panels.append(dict(dataset=ds, label_names=rr.label_names, labels=y,
                           frozen_xy=fxy, ft_xy=txy, frozen_ev=fev, ft_ev=tev))
        print(f"[emb] {ds:9s} {args.model}: classes={len(np.unique(y))} "
              f"pts={len(y)} PCA-var frozen={fev:.0%} ft={tev:.0%}")

    out = plots.plot_embedding_projection(panels, args.model, f"emb_{args.model}_pca.png")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
