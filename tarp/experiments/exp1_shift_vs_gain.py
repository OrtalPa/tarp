"""Exp1 — Representation shift vs. adaptation gain (RQ1, H1). Compute + emit only.

Per (model, dataset): compute the gap and per-layer frozen→FT CKA/cosine, write tidy rows,
and save the per-run CKA curve. The cross-config correlation + scatter live in
``tarp.analysis.shift_gain`` and are produced by the ``report`` command (compute vs analysis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tarp import plots
from tarp.config import RunConfig
from tarp.experiments.base import Experiment
from tarp.metrics import cka_per_layer, cosine_per_layer
from tarp.pipeline import run_conditions
from tarp.registry import register_experiment
from tarp.results import append_rows


@register_experiment("exp1")
class ShiftVsGain(Experiment):
    def run(self, configs: list[RunConfig]) -> pd.DataFrame:
        rows: list[dict] = []
        n = len(configs)
        print(f"\n[exp1] shift-vs-gain over {n} run(s)", flush=True)

        for i, cfg in enumerate(configs, 1):
            print(f"[exp1] ({i}/{n}) start {cfg.dataset.key}/{cfg.model.key} seed={cfg.seed}", flush=True)
            rr = run_conditions(cfg)
            cka = cka_per_layer(rr.frozen_reps, rr.ft_reps)
            cos = cosine_per_layer(rr.frozen_reps, rr.ft_reps)
            base = dict(
                experiment="exp1", model=cfg.model.key, dataset=cfg.dataset.key,
                seed=cfg.seed, pooling=cfg.pooling, ft_regime=cfg.finetune.regime,
                run_hash=cfg.run_id(),
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

            shift = float(1.0 - np.mean(cka[1:]))  # mean over transformer layers
            fig = plots.plot_cka_layers(cfg, cka)
            print(
                f"  {cfg.dataset.key:9s} {cfg.model.key:11s} "
                f"frozen={rr.frozen_acc:.3f} ft={rr.ft_acc:.3f} gap={rr.gap:+.3f} "
                f"shift={shift:.3f}  fig={fig.name}",
                flush=True,
            )

        df = append_rows(rows)
        print(f"[exp1] wrote {len(rows)} rows. Run `report --experiment exp1` for the correlation + scatter.")
        return df
