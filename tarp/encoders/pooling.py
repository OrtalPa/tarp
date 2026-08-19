"""Pooling: collapse a layer's token matrix (B, T, H) into one vector per input (B, H).

``mean`` is the pinned primary (masked average over real tokens); ``cls`` is the
secondary robustness view (first token: ``[CLS]`` for BERT/DistilBERT, ``<s>`` for
RoBERTa).
"""

from __future__ import annotations

import torch

POOLERS = ("mean", "cls")


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over the token dimension. hidden (B,T,H), mask (B,T)."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """First-token vector. hidden (B,T,H) -> (B,H)."""
    return hidden[:, 0]


def get_pooler(name: str):
    if name == "mean":
        return mean_pool
    if name == "cls":
        return cls_pool
    raise ValueError(f"unknown pooling '{name}'. options: {POOLERS}")
