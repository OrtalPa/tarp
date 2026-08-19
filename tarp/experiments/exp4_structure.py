"""Exp4 — Layer-wise structural probing (structure, not accuracy).

Where in the network does *class structure* live, and how does fine-tuning reshape it?
Exp2 localizes *change* (CKA drops in the top layers) but is accuracy/label-agnostic; Exp3
shows frozen separability predicts the gap but only at the **final** layer. Exp4 fills the
middle: it evaluates the Exp3 separability descriptors (intra/inter ratio, silhouette, kNN
acc, anisotropy, effective dimension) at **every** layer, for both the **frozen** and the
**fine-tuned** encoder, and emits the per-layer curves.

Pure cache reuse: ``run_conditions`` returns the cached per-layer frozen & FT test reps that
Exp1 already extracted — no accelerator, no re-fine-tuning. All compute here is CPU numpy/sklearn.
The per-layer curves + frozen→FT structural delta live in ``tarp.analysis.structure`` and
render under ``report --experiment exp4``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tarp.config import RunConfig
from tarp.experiments.base import Experiment
from tarp.metrics import separability as sep
from tarp.pipeline import run_conditions
from tarp.registry import register_experiment
from tarp.results import append_rows

# Structural descriptors evaluated per layer. Same functions as Exp3 (final layer only);
# here we sweep them across depth. Direction reminders for the reader / report:
#   intra_inter_ratio  lower = better separated
#   silhouette, knn_acc higher = better separated
#   anisotropy         higher = more degenerate (vectors share a direction)
#   eff_dim            higher = variance spread over more dimensions
_METRICS = ["intra_inter_ratio", "silhouette", "knn_acc", "anisotropy", "eff_dim"]


def _layer_features(X, y, seed: int) -> dict[str, float]:
    """All structural descriptors for one layer's (N, H) representation."""
    return {
        "intra_inter_ratio": sep.intra_inter_ratio(X, y),
        "silhouette": sep.silhouette(X, y, seed=seed),
        "knn_acc": sep.knn_accuracy(X, y, seed=seed),
        "anisotropy": sep.anisotropy(X, seed=seed),
        "eff_dim": sep.effective_dimension(X),
    }


@register_experiment("exp4")
class LayerStructure(Experiment):
    def run(self, configs: list[RunConfig]) -> pd.DataFrame:
        rows: list[dict] = []
        n = len(configs)
        print(f"\n[exp4] layer-wise structure over {n} pair(s)", flush=True)

        for i, cfg in enumerate(configs, 1):
            print(f"[exp4] ({i}/{n}) start {cfg.dataset.key}/{cfg.model.key}", flush=True)
            rr = run_conditions(cfg)
            labels = rr.labels
            base = dict(
                experiment="exp4", model=cfg.model.key, dataset=cfg.dataset.key,
                seed=cfg.seed, pooling=cfg.pooling, ft_regime=cfg.finetune.regime,
                run_hash=cfg.run_id(),
            )
            # keep the accuracy gap alongside so the report can relate structure to the gap
            rows.append({**base, "condition": "delta", "layer": -1, "metric": "gap", "value": rr.gap})

            for state, reps in (("frozen", rr.frozen_reps), ("ft", rr.ft_reps)):
                n_layers = reps.shape[0]
                for L in range(n_layers):
                    feats = _layer_features(reps[L], labels, cfg.seed)
                    for name, val in feats.items():
                        rows.append({**base, "condition": state, "layer": L,
                                     "metric": name, "value": float(val)})
            # readable progress: frozen vs ft intra_inter at the final layer
            fro = sep.intra_inter_ratio(rr.frozen_reps[-1], labels)
            fin = sep.intra_inter_ratio(rr.ft_reps[-1], labels)
            print(f"[exp4] ({i}/{n}) {cfg.dataset.key:9s} {cfg.model.key:11s} "
                  f"intra_inter(top): frozen={fro:.2f} -> ft={fin:.2f}  gap={rr.gap:+.3f}",
                  flush=True)

        df = append_rows(rows)
        print(f"[exp4] wrote {len(rows)} rows. Run `report --experiment exp4` for the curves.")
        return df
