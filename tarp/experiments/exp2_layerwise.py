"""Exp2 — Where adaptation happens; full-FT vs LoRA (RQ2, H2). Compute + emit only.

For each (model, dataset) we run **both** fine-tuning regimes — LoRA-last-3 (reusing the
adapters Exp1 already cached) and full fine-tuning — and emit, per regime, the per-layer
frozen→FT CKA/cosine, the per-layer frozen probe accuracy, and the scalar gap. The
layer-wise aggregation (mean ± CI across datasets, per model/regime) and the figures live
in ``tarp.analysis.layerwise`` and are produced by ``report --experiment exp2``.

Nothing here re-fine-tunes the LoRA arm: its ``finetune_key`` matches Exp1, so the cached
adapter + reps are reused. Only the full-FT arm trains (once per model/dataset/seed).
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from tarp.config import FinetuneSpec, RunConfig
from tarp.experiments.base import Experiment
from tarp.metrics import cka_per_layer, cosine_per_layer
from tarp.pipeline import run_conditions
from tarp.registry import register_experiment
from tarp.results import append_rows

# Full fine-tuning updates every weight, so it uses the standard small transformer LR
# (LoRA's 2e-4 would be far too hot for a full update).
FULL_FT = FinetuneSpec(regime="full", epochs=3, lr=2e-5, batch_size=32)


@register_experiment("exp2")
class LayerWise(Experiment):
    def run(self, configs: list[RunConfig]) -> pd.DataFrame:
        rows: list[dict] = []
        n = len(configs)
        print(f"\n[exp2] layer-wise, full-FT vs LoRA over {n} pair(s)", flush=True)

        for i, cfg in enumerate(configs, 1):
            lora_cfg = cfg  # default finetune spec = lora_last3
            full_cfg = replace(cfg, finetune=FULL_FT)
            for regime_cfg in (lora_cfg, full_cfg):
                regime = regime_cfg.finetune.regime
                print(f"[exp2] ({i}/{n}) {cfg.dataset.key}/{cfg.model.key} [{regime}] ...", flush=True)
                rr = run_conditions(regime_cfg)
                cka = cka_per_layer(rr.frozen_reps, rr.ft_reps)
                cos = cosine_per_layer(rr.frozen_reps, rr.ft_reps)
                base = dict(
                    experiment="exp2", model=cfg.model.key, dataset=cfg.dataset.key,
                    seed=cfg.seed, pooling=cfg.pooling, ft_regime=regime,
                    run_hash=regime_cfg.run_id(),
                )
                rows += [
                    {**base, "condition": "frozen", "layer": -1, "metric": "frozen_acc", "value": rr.frozen_acc},
                    {**base, "condition": "ft", "layer": -1, "metric": "ft_acc", "value": rr.ft_acc},
                    {**base, "condition": "delta", "layer": -1, "metric": "gap", "value": rr.gap},
                ]
                for L in range(len(cka)):
                    rows += [
                        {**base, "condition": "shift", "layer": L, "metric": "cka", "value": cka[L]},
                        {**base, "condition": "shift", "layer": L, "metric": "cosine", "value": cos[L]},
                        {**base, "condition": "frozen", "layer": L, "metric": "probe_acc", "value": rr.per_layer_probe_acc[L]},
                    ]
                print(
                    f"[exp2] ({i}/{n}) {cfg.dataset.key:9s} {cfg.model.key:11s} [{regime:10s}] "
                    f"frozen={rr.frozen_acc:.3f} ft={rr.ft_acc:.3f} gap={rr.gap:+.3f}",
                    flush=True,
                )

        df = append_rows(rows)
        print(f"[exp2] wrote {len(rows)} rows. Run `report --experiment exp2` for layer curves.")
        return df
