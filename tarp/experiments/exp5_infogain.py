"""Exp5 — Information gain vs. loss during adaptation (what's dropped vs amplified).

(a) Target-vs-generic trade-off (quantitative). For each (model, dataset A):
    - ``target_gain`` = the adaptation gap on A (probe(FT) − probe(frozen) = the gap).
    - ``generic_loss`` = mean over the *other* selected datasets B of
      ``frozen_probe(B) − probe(B decoded through the A-fine-tuned encoder)`` — the
      transferable information A-adaptation dropped (see ``pipeline.cross_task``).
    We emit both per A; the report scatters them (gain up, generic decodability down).

(b) Token-saliency shift (qualitative). For a few examples, the change in last-layer
    [CLS] attention frozen → fine-tuned (``pipeline.saliency``) — saved as a JSON artifact.

Cross-task is scoped to the datasets the caller selected, per model/seed/pooling. The LoRA
adapters are reused from Exp1 (no re-fine-tuning); only cross-task extraction runs.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from tarp.config import RunConfig
from tarp.data import load_task
from tarp.experiments.base import Experiment
from tarp.pipeline import cache, run_conditions
from tarp.pipeline.cross_task import ft_source_probe_acc
from tarp.registry import register_experiment
from tarp.results import append_rows

_SALIENCY_EXAMPLES = 5


@register_experiment("exp5")
class InfoGainLoss(Experiment):
    def run(self, configs: list[RunConfig]) -> pd.DataFrame:
        # cross-task comparisons are only meaningful within one (model, seed, pooling) group
        groups: dict[tuple, list[RunConfig]] = defaultdict(list)
        for cfg in configs:
            groups[(cfg.model.key, cfg.seed, cfg.pooling)].append(cfg)

        rows: list[dict] = []
        print(f"\n[exp5] info gain vs loss over {len(configs)} pair(s) "
              f"in {len(groups)} group(s)", flush=True)

        for (model, seed, pooling), group in groups.items():
            if len(group) < 2:
                print(f"[exp5] {model}: need >=2 datasets for cross-task loss — skipping", flush=True)
                continue

            # per-dataset frozen probe acc + gap (cached from Exp1 → cheap)
            frozen_acc, gap = {}, {}
            for cfg in group:
                rr = run_conditions(cfg)
                frozen_acc[cfg.dataset.key], gap[cfg.dataset.key] = rr.frozen_acc, rr.gap

            cross = {}  # (A, B) -> probe acc of B through A-adapted encoder
            for a in group:
                A = a.dataset.key
                for b in group:
                    if b.dataset.key == A:
                        continue
                    cross[(A, b.dataset.key)] = ft_source_probe_acc(
                        a, b.dataset, seed=seed, pooling=pooling, max_length=a.max_length)

            base_common = dict(experiment="exp5", model=model, seed=seed,
                               pooling=pooling, ft_regime=group[0].finetune.regime)
            for a in group:
                A = a.dataset.key
                losses = [frozen_acc[b.dataset.key] - cross[(A, b.dataset.key)]
                          for b in group if b.dataset.key != A]
                generic_loss = float(np.mean(losses))
                base = {**base_common, "dataset": A, "run_hash": a.run_id()}
                rows += [
                    {**base, "condition": "delta", "layer": -1, "metric": "target_gain", "value": gap[A]},
                    {**base, "condition": "delta", "layer": -1, "metric": "generic_loss", "value": generic_loss},
                    {**base, "condition": "frozen", "layer": -1, "metric": "frozen_acc", "value": frozen_acc[A]},
                ]
                print(f"[exp5] {A:9s} {model:11s} target_gain={gap[A]:+.3f} "
                      f"generic_loss={generic_loss:+.3f}", flush=True)

            # full pairwise matrix kept as a side artifact (too fine-grained for the tidy table)
            cache.save_json(
                cache.figures_dir().parent / "artifacts" / f"exp5_cross_{model}_s{seed}_{pooling}.json",
                {f"{a}->{b}": v for (a, b), v in cross.items()},
            )

            self._saliency(group[0])

        df = append_rows(rows)
        print(f"[exp5] wrote {len(rows)} rows. Run `report --experiment exp5` for the trade-off.")
        return df

    def _saliency(self, cfg: RunConfig) -> None:
        """Best-effort Exp5(b): CLS-attention shift on a few examples of ``cfg``'s dataset."""
        try:
            from tarp.pipeline.saliency import saliency_shift

            texts = load_task(cfg.dataset, cfg.seed).test[0][:_SALIENCY_EXAMPLES]
            records = saliency_shift(cfg, texts)
            out = cache.figures_dir().parent / "artifacts" / f"exp5_saliency_{cfg.model.key}_{cfg.dataset.key}.json"
            cache.save_json(out, records)
            print(f"[exp5] saliency ({cfg.dataset.key}/{cfg.model.key}) -> {out}", flush=True)
        except Exception as e:  # attention is a side diagnostic — never fail the experiment on it
            print(f"[exp5] saliency skipped ({cfg.dataset.key}/{cfg.model.key}): {e}", flush=True)
