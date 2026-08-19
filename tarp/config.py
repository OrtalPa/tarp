"""Typed, hashable run configuration.

Plain frozen dataclasses (no YAML/Hydra, per the approved plan). Frozen so they are
hashable and usable as cache keys. The cache keys are deliberately *layered*:

- ``finetune_key`` excludes pooling and the experiment, so a
  (model, dataset, ft-config, seed) pair is fine-tuned **exactly once** and reused.
- ``extraction_key`` identifies a cached per-layer representation tensor for a given
  model state (frozen or a specific fine-tuned adapter), pooling, and split.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


def _short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


@dataclass(frozen=True)
class ModelSpec:
    key: str      # short handle, e.g. "distilbert"
    hf_id: str    # HuggingFace id, e.g. "distilbert-base-uncased"
    family: str   # "bert" | "roberta" | "distilbert"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_id: str
    text_field: str
    label_field: str
    num_labels: int
    hf_config: Optional[str] = None
    eval_split: str = "test"              # which split has usable labels for evaluation
    drop_labels: tuple[str, ...] = ()     # label *names* to drop, e.g. ("oos",)
    subsample_train: Optional[int] = None  # cap train size for speed (seeded)
    domain: str = ""                      # task-family tag for Exp1 grouping


@dataclass(frozen=True)
class FinetuneSpec:
    regime: str = "lora_last3"   # "lora_last3" | "full"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    epochs: int = 3
    lr: float = 2e-4
    batch_size: int = 32
    n_last_layers: int = 3       # #top transformer layers LoRA adapts (lora_last3)

    def tag(self) -> str:
        if self.regime == "full":
            return f"full-e{self.epochs}-lr{self.lr}-b{self.batch_size}"
        return (
            f"lora{self.n_last_layers}-r{self.r}a{self.alpha}d{self.dropout}"
            f"-e{self.epochs}-lr{self.lr}-b{self.batch_size}"
        )


@dataclass(frozen=True)
class RunConfig:
    model: ModelSpec
    dataset: DatasetSpec
    finetune: FinetuneSpec = field(default_factory=FinetuneSpec)
    pooling: str = "mean"        # "mean" | "cls"
    seed: int = 0
    max_length: int = 128

    # --- cache keys -------------------------------------------------------
    def finetune_key(self) -> str:
        """Adapter cache key. Independent of pooling / experiment: a given
        (model, dataset, ft-config, seed, max_length) is fine-tuned once."""
        raw = "|".join([
            self.model.key, self.dataset.key, self.finetune.tag(),
            f"seed{self.seed}", f"len{self.max_length}",
        ])
        return (
            f"{self.model.key}__{self.dataset.key}__{self.finetune.tag()}"
            f"__s{self.seed}__{_short_hash(raw)}"
        )

    def extraction_key(self, state: str, pooling: str, split: str) -> str:
        """Per-layer representation cache key.

        state: "frozen" or "ft:<finetune_key>".
        """
        state_tag = "frozen" if state == "frozen" else "ft"
        # seed matters: the train subsample (and val carve) are seeded, so reps differ.
        raw = "|".join([
            self.model.key, self.dataset.key, split, state, pooling,
            f"seed{self.seed}", f"len{self.max_length}",
        ])
        return (
            f"{self.model.key}__{self.dataset.key}__{split}__{state_tag}"
            f"__{pooling}__s{self.seed}__{_short_hash(raw)}"
        )

    def run_id(self) -> str:
        """Human-ish id for a full run record (model, dataset, pooling, seed)."""
        raw = "|".join([self.finetune_key(), self.pooling])
        return (
            f"{self.model.key}__{self.dataset.key}__{self.pooling}"
            f"__s{self.seed}__{_short_hash(raw)}"
        )
