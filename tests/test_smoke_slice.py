"""Fast end-to-end smoke: a tiny (200-example, 1-epoch) run through the real pipeline.

Uses a distinct dataset key ("trecsmoke") so its cache never collides with real runs.
"""

import numpy as np

from tarp.config import DatasetSpec, FinetuneSpec, RunConfig
from tarp.pipeline import run_conditions
from tarp.registry import resolve_model


def test_smoke_slice_end_to_end():
    ds = DatasetSpec(
        "trecsmoke", "SetFit/TREC-QC", "text", "label_coarse", 6, subsample_train=200
    )
    cfg = RunConfig(
        model=resolve_model("distilbert"),
        dataset=ds,
        finetune=FinetuneSpec(epochs=1),
        max_length=64,
    )
    rr = run_conditions(cfg, force=True)

    assert np.isfinite(rr.frozen_acc)
    assert np.isfinite(rr.ft_acc)
    assert np.isfinite(rr.gap)
    # frozen and fine-tuned reps have the same shape and align with the labels
    assert rr.frozen_reps.shape == rr.ft_reps.shape
    assert rr.frozen_reps.shape[1] == len(rr.labels)
    # per-layer probe accuracy has one entry per layer (incl. embeddings)
    assert len(rr.per_layer_probe_acc) == rr.frozen_reps.shape[0]
