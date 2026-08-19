"""Encoder interface.

An ``Encoder`` turns a list of inputs into per-layer pooled representations — a single
array of shape ``(num_layers + 1, N, hidden_size)`` (index 0 = embedding layer). This
decouples the pipeline/metrics from any specific model. Text encoders implement it now;
the same contract would admit a vision encoder later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    num_layers: int
    hidden_size: int

    @abstractmethod
    def encode(self, texts: list[str], pooling: str = "mean", batch_size: int = 64) -> np.ndarray:
        """Return per-layer pooled reps, shape (num_layers + 1, N, hidden_size)."""

    @abstractmethod
    def classification_model(self, num_labels: int):
        """Return a fresh HF sequence-classification model for fine-tuning."""
