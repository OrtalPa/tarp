"""Importing this package registers all experiments in the registry."""

from tarp.experiments import (  # noqa: F401
    exp1_shift_vs_gain,
    exp2_layerwise,
    exp3_frozen_predictor,
    exp4_structure,
    exp5_infogain,
)
from tarp.experiments.base import Experiment

__all__ = ["Experiment"]
