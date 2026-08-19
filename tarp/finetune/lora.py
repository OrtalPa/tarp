"""LoRA configuration targeting the **last N transformer layers** (mirrors MulTaBench).

We resolve explicit `nn.Linear` module names in the top-N layers so LoRA adapts only
those; the rest of the encoder stays frozen. The classification head is trained via
`modules_to_save`. Works across BERT/RoBERTa (`encoder.layer.<i>`) and DistilBERT
(`transformer.layer.<i>`) since both expose `.layer.<i>.`.
"""

from __future__ import annotations

import re

import torch.nn as nn
from peft import LoraConfig, TaskType

from tarp.config import FinetuneSpec

# Matches transformer blocks across families: BERT/RoBERTa `encoder.layer.<i>`,
# DistilBERT `transformer.layer.<i>`, ModernBERT `model.layers.<i>` (plural).
_LAYER_RE = re.compile(r"\.layers?\.(\d+)\.")
# Task-head modules to train alongside LoRA: `classifier` (all), `pre_classifier`
# (DistilBERT), `head` (ModernBERT's 768×768 prediction head). BERT's `pooler` is added
# dynamically below (suffix match). None of these names collide across families.
_HEAD_MODULES = ("classifier", "pre_classifier", "head")


def last_n_linear_module_names(model, n_last: int) -> list[str]:
    linears = [
        name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)
    ]
    layer_idxs = {int(m.group(1)) for n in linears if (m := _LAYER_RE.search(n))}
    if not layer_idxs:
        raise RuntimeError("could not locate transformer layers via '.layer(s).<i>.'")
    max_layer = max(layer_idxs)
    keep = set(range(max_layer - n_last + 1, max_layer + 1))
    return [
        n for n in linears
        if (m := _LAYER_RE.search(n)) and int(m.group(1)) in keep
    ]


def present_head_modules(model) -> list[str]:
    """Classification-head modules to train alongside the LoRA layers.

    Includes the ``pooler`` when the family has one (BERT): it sits in the [CLS]→classifier
    path, and leaving it frozen would give BERT a strictly weaker (half-size) trainable head
    than RoBERTa/DistilBERT — an unfair cross-model comparison and a cause of underfitting on
    many-class tasks. ``modules_to_save`` matches by name suffix, so "pooler" catches
    ``<base>.pooler``."""
    names = [name for name, _ in model.named_modules()]
    have = set(names)
    heads = [h for h in _HEAD_MODULES if h in have]
    if any(n.endswith("pooler") for n in names):
        heads.append("pooler")
    return heads


def build_lora_config(model, spec: FinetuneSpec) -> LoraConfig:
    targets = last_n_linear_module_names(model, spec.n_last_layers)
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        bias="none",
        target_modules=targets,
        modules_to_save=present_head_modules(model),
    )
