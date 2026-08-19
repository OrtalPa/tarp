"""Standardized dataset loading.

``load_task(spec, seed)`` returns train/val/test as ``(texts, int_labels)`` with a
consistent ``0..K-1`` label encoding, handling each dataset's quirks in one place:

- parquet-native mirrors (``datasets>=3`` dropped script loaders),
- TREC coarse (6) vs fine (50) labels — the spec picks the field,
- CLINC ``plus`` → drop the out-of-scope class → exactly 150 intents,
- SST-2 hidden test labels → evaluate on ``validation`` (via ``spec.eval_split``),
- optional seeded train subsampling,
- a ``val`` split for fine-tuning early-stopping (carved from train if absent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from datasets import ClassLabel, load_dataset

from tarp.config import DatasetSpec

Split = tuple[list[str], np.ndarray]


@dataclass
class TaskData:
    train: Split
    val: Split
    test: Split
    label_names: list[str]
    num_labels: int


def _numeric_aware_sorted(values) -> list:
    vals = list(values)
    try:
        return sorted(vals, key=lambda v: int(v))
    except (ValueError, TypeError):
        return sorted(vals, key=str)


def _label_tools(train_feat, raw_value_iters) -> tuple[list[str], Callable]:
    """Return (ordered label names, raw_value -> name) function.

    ClassLabel features carry authoritative names; otherwise we build the name set
    from the union of raw values across all splits (so eval labels are never missing).
    """
    if isinstance(train_feat, ClassLabel):
        names = list(train_feat.names)
        return names, (lambda v: names[int(v)])
    seen = set()
    for it in raw_value_iters:
        seen.update(it)
    names = [str(v) for v in _numeric_aware_sorted(seen)]
    return names, (lambda v: str(v))


def _encode_split(dset, text_field, label_field, raw_to_name, name_to_new, drop_set):
    texts, labels = [], []
    for txt, raw in zip(dset[text_field], dset[label_field]):
        name = raw_to_name(raw)
        if name in drop_set:
            continue
        texts.append(str(txt))
        labels.append(name_to_new[name])
    return texts, np.asarray(labels, dtype=np.int64)


def _subsample(split: Split, n: int | None, seed: int) -> Split:
    texts, labels = split
    if n is None or len(texts) <= n:
        return split
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=n, replace=False)
    return [texts[i] for i in idx], labels[idx]


def _stratified_carve(split: Split, frac: float, seed: int) -> tuple[Split, Split]:
    """Split into (larger, smaller) with a seeded, roughly class-stratified holdout."""
    texts, labels = split
    rng = np.random.default_rng(seed)
    keep, hold = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * frac)))
        hold.extend(idx[:k].tolist())
        keep.extend(idx[k:].tolist())
    rng.shuffle(keep)
    rng.shuffle(hold)
    keep_arr = np.asarray(keep, dtype=int)
    hold_arr = np.asarray(hold, dtype=int)
    big = ([texts[i] for i in keep_arr], labels[keep_arr])
    small = ([texts[i] for i in hold_arr], labels[hold_arr])
    return big, small


def load_task(spec: DatasetSpec, seed: int = 0) -> TaskData:
    dd = load_dataset(spec.hf_id, spec.hf_config)

    eval_name = spec.eval_split
    if eval_name not in dd:
        raise KeyError(f"{spec.key}: eval split '{eval_name}' not in {list(dd)}")
    if "train" not in dd:
        raise KeyError(f"{spec.key}: no 'train' split in {list(dd)}")

    train_raw, eval_raw = dd["train"], dd[eval_name]
    feat = train_raw.features[spec.label_field]
    raw_iters = [train_raw[spec.label_field], eval_raw[spec.label_field]]
    if "validation" in dd and "validation" != eval_name:
        raw_iters.append(dd["validation"][spec.label_field])

    names, raw_to_name = _label_tools(feat, raw_iters)
    drop_set = set(spec.drop_labels)
    kept_names = [n for n in names if n not in drop_set]
    name_to_new = {n: i for i, n in enumerate(kept_names)}
    if len(kept_names) != spec.num_labels:
        raise ValueError(
            f"{spec.key}: expected {spec.num_labels} labels, got {len(kept_names)} "
            f"(after dropping {sorted(drop_set)})"
        )

    def enc(dset):
        return _encode_split(
            dset, spec.text_field, spec.label_field, raw_to_name, name_to_new, drop_set
        )

    train = _subsample(enc(train_raw), spec.subsample_train, seed)
    test = enc(eval_raw)

    if "validation" in dd and "validation" != eval_name:
        val = enc(dd["validation"])
    else:
        train, val = _stratified_carve(train, frac=0.1, seed=seed)

    return TaskData(train=train, val=val, test=test,
                    label_names=kept_names, num_labels=spec.num_labels)
