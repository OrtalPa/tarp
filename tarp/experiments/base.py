"""Experiment interface: take a list of RunConfigs, emit tidy result rows."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from tarp.config import RunConfig


class Experiment(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, configs: list[RunConfig]) -> pd.DataFrame:
        """Execute over the given configs and return the appended results table."""
