"""Exp3 — Predicting the gap from frozen features alone (RQ3, H3). Compute + emit only.

For each (model, dataset) we compute cheap **frozen-only** descriptors of the final-layer
representation space — the frozen linear-probe accuracy plus label-aware separability
(silhouette, intra/inter ratio, kNN acc) and label-free geometry (anisotropy, effective
dimension) — and emit them alongside the (fine-tuning-derived) target ``gap``. No fine-tuning
happens here beyond what Exp1 already cached: ``run_conditions`` returns the cached frozen
reps + gap. The regression that predicts gap from these features (leave-one-dataset-out CV,
univariate Spearman, and the binary needs-TAR threshold) lives in
``tarp.analysis.frozen_predictor`` and runs under ``report --experiment exp3``.
"""

from __future__ import annotations

import pandas as pd

from tarp.config import RunConfig
from tarp.experiments.base import Experiment
from tarp.metrics import separability as sep
from tarp.pipeline import run_conditions
from tarp.registry import register_experiment
from tarp.results import append_rows


def _frozen_features(reps_final, labels, seed: int) -> dict[str, float]:
    """Separability / geometry descriptors of one frozen representation space."""
    return {
        "silhouette": sep.silhouette(reps_final, labels, seed=seed),
        "intra_inter_ratio": sep.intra_inter_ratio(reps_final, labels),
        "knn_acc": sep.knn_accuracy(reps_final, labels, seed=seed),
        "anisotropy": sep.anisotropy(reps_final, seed=seed),
        "eff_dim": sep.effective_dimension(reps_final),
    }


@register_experiment("exp3")
class FrozenPredictor(Experiment):
    def run(self, configs: list[RunConfig]) -> pd.DataFrame:
        rows: list[dict] = []
        n = len(configs)
        print(f"\n[exp3] frozen predictors over {n} pair(s)", flush=True)

        for i, cfg in enumerate(configs, 1):
            print(f"[exp3] ({i}/{n}) start {cfg.dataset.key}/{cfg.model.key}", flush=True)
            rr = run_conditions(cfg)
            feats = _frozen_features(rr.frozen_reps[-1], rr.labels, cfg.seed)

            base = dict(
                experiment="exp3", model=cfg.model.key, dataset=cfg.dataset.key,
                seed=cfg.seed, pooling=cfg.pooling, ft_regime=cfg.finetune.regime,
                run_hash=cfg.run_id(),
            )
            # target + the frozen-accuracy baseline predictor
            rows += [
                {**base, "condition": "delta", "layer": -1, "metric": "gap", "value": rr.gap},
                {**base, "condition": "frozen", "layer": -1, "metric": "frozen_acc", "value": rr.frozen_acc},
            ]
            # frozen-only candidate predictors (final layer)
            for name, val in feats.items():
                rows.append({**base, "condition": "frozen", "layer": -1, "metric": name, "value": float(val)})

            feat_str = "  ".join(f"{k}={v:.3f}" for k, v in feats.items())
            print(f"[exp3] ({i}/{n}) {cfg.dataset.key:9s} {cfg.model.key:11s} "
                  f"gap={rr.gap:+.3f} frozen={rr.frozen_acc:.3f}  {feat_str}", flush=True)

        df = append_rows(rows)
        print(f"[exp3] wrote {len(rows)} rows. Run `report --experiment exp3` for the prediction.")
        return df
