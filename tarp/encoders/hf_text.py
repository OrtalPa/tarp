"""HuggingFace text encoder (BERT / RoBERTa / DistilBERT).

Wraps any model that returns ``hidden_states`` under ``output_hidden_states=True`` — so
the *same* class handles the frozen base model (``AutoModel``) and a fine-tuned
sequence-classification model (used to extract post-FT representations).
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from tarp.config import ModelSpec
from tarp.encoders.base import Encoder
from tarp.encoders.pooling import get_pooler


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _config_dims(config) -> tuple[int, int]:
    hidden = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
    n_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layers", None)
    return int(n_layers), int(hidden)


def _load_kwargs(spec: ModelSpec) -> dict:
    """Family-scoped from_pretrained kwargs. ModernBERT defaults to FlashAttention-2, which is
    not available in every environment → force sdpa, which is always supported. Other families
    keep their default (unchanged, so existing caches stay valid)."""
    return {"attn_implementation": "sdpa"} if spec.family == "modernbert" else {}


class HFTextEncoder(Encoder):
    def __init__(self, model, tokenizer, spec: ModelSpec, max_length: int = 128, device: str | None = None):
        self.spec = spec
        self.max_length = max_length
        self.device = device or default_device()
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()
        self.num_layers, self.hidden_size = _config_dims(self.model.config)

    # --- constructors ----------------------------------------------------
    @classmethod
    def frozen(cls, spec: ModelSpec, max_length: int = 128, device: str | None = None) -> "HFTextEncoder":
        tok = AutoTokenizer.from_pretrained(spec.hf_id)
        model = AutoModel.from_pretrained(spec.hf_id, **_load_kwargs(spec))
        return cls(model, tok, spec, max_length, device)

    @classmethod
    def wrap(cls, model, tokenizer, spec: ModelSpec, max_length: int = 128, device: str | None = None) -> "HFTextEncoder":
        """Wrap an already-instantiated model (e.g. a fine-tuned seq-cls model)."""
        return cls(model, tokenizer, spec, max_length, device)

    # --- Encoder API -----------------------------------------------------
    @torch.no_grad()
    def encode(self, texts: list[str], pooling: str = "mean", batch_size: int = 64) -> np.ndarray:
        pool_fn = get_pooler(pooling)
        per_layer: list[list[np.ndarray]] | None = None
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            out = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
            hs = out.hidden_states  # tuple len (num_layers + 1), each (B, T, H)
            if per_layer is None:
                per_layer = [[] for _ in hs]
            mask = enc["attention_mask"]
            for i, h in enumerate(hs):
                per_layer[i].append(pool_fn(h, mask).float().cpu().numpy())
        if per_layer is None:
            return np.empty((self.num_layers + 1, 0, self.hidden_size), dtype=np.float32)
        return np.stack([np.concatenate(ch, axis=0) for ch in per_layer], axis=0).astype(np.float32)

    def classification_model(self, num_labels: int):
        return AutoModelForSequenceClassification.from_pretrained(
            self.spec.hf_id, num_labels=num_labels, **_load_kwargs(self.spec)
        )
