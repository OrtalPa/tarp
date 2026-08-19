"""Cross-task representation probing (Exp5a).

Push dataset **B**'s inputs through an encoder that was fine-tuned on dataset **A**, then
probe B with its *own* labels. Comparing this to B's frozen probe accuracy quantifies the
generic / transferable information that A-adaptation preserved vs. dropped (feature
distortion, Kumar et al. 2022). Reps are cached per (source adapter, data, split) so a
given A→B extraction happens once.
"""

from __future__ import annotations

import hashlib

import numpy as np

from tarp.config import DatasetSpec, RunConfig
from tarp.data import load_task
from tarp.pipeline.extract import get_or_extract
from tarp.pipeline.run import load_finetuned_encoder
from tarp.probes import probe_all_layers

# The probe caps its own train at 6000; extracting a dataset's full 10-15k train through the
# source encoder would just be thrown away. Cap cross-task train extraction to match (seeded).
_CROSS_MAX_TRAIN = 6000


def _cross_key(ft_key: str, data_key: str, pooling: str, split: str,
               seed: int, max_length: int, max_train: int | None) -> str:
    raw = f"{ft_key}|{data_key}|{pooling}|{split}|s{seed}|len{max_length}|cap{max_train}"
    return f"cross__{data_key}__{split}__{pooling}__s{seed}__{hashlib.sha1(raw.encode()).hexdigest()[:10]}"


def _cap(texts: list[str], labels, max_train: int | None, seed: int):
    """Seeded subsample of a train split (no-op for test / already-small splits)."""
    if max_train is None or len(texts) <= max_train:
        return texts, labels
    idx = np.random.default_rng(seed).choice(len(texts), size=max_train, replace=False)
    return [texts[i] for i in idx], labels[idx]


def ft_source_probe_acc(src_cfg: RunConfig, data_spec: DatasetSpec,
                        seed: int = 0, pooling: str = "mean", max_length: int = 128,
                        max_train: int | None = _CROSS_MAX_TRAIN) -> float:
    """Final-layer linear-probe accuracy for ``data_spec`` decoded through the encoder
    fine-tuned per ``src_cfg`` (source task A). The source classification head is irrelevant
    here — we only read the (A-adapted) hidden states."""
    task = load_task(data_spec, seed)
    ft_key = src_cfg.finetune_key()
    tr_texts, tr_labels = _cap(task.train[0], task.train[1], max_train, seed)

    _enc: dict = {}

    def provider():
        if "e" not in _enc:  # build the A-adapted encoder at most once for both splits
            _enc["e"] = load_finetuned_encoder(src_cfg, src_cfg.dataset.num_labels)
        return _enc["e"]

    tr = get_or_extract(_cross_key(ft_key, data_spec.key, pooling, "train", seed, max_length, max_train),
                        tr_texts, pooling, provider)
    te = get_or_extract(_cross_key(ft_key, data_spec.key, pooling, "test", seed, max_length, max_train),
                        task.test[0], pooling, provider)
    return probe_all_layers(tr, tr_labels, te, task.test[1], seed=seed).final
