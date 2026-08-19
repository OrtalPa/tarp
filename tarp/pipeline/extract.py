"""Cached per-layer representation extraction.

``get_or_extract`` returns cached reps if present, else builds the encoder lazily (via
``encoder_provider``) and extracts. The lazy provider means a fully-cached run never
loads a model onto the compute device.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from tarp.pipeline import cache


def get_or_extract(
    key: str, texts: list[str], pooling: str, encoder_provider: Callable
) -> np.ndarray:
    cached = cache.load_reps(key)
    if cached is not None:
        return cached
    reps = encoder_provider().encode(texts, pooling=pooling)
    cache.save_reps(key, reps)
    return reps
