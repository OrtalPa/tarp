"""Vertical slice: DistilBERT + TREC via Exp1.

Equivalent to:
    uv run python -m tarp.cli run --experiment exp1 --models distilbert --datasets trec --seeds 0
"""

import tarp.experiments  # noqa: F401  (registers experiments)
from tarp.config import RunConfig
from tarp.registry import get_experiment, resolve_dataset, resolve_model

if __name__ == "__main__":
    cfg = RunConfig(model=resolve_model("distilbert"), dataset=resolve_dataset("trec"))
    get_experiment("exp1")().run([cfg])
