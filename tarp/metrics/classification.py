"""Classification metrics."""

from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())
