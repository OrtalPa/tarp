"""Registries for models, datasets, and experiments (plain Python).

Adding a model/dataset = add a spec below. Adding an experiment = decorate its class
with ``@register_experiment("name")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tarp.config import DatasetSpec, ModelSpec

if TYPE_CHECKING:
    from tarp.experiments.base import Experiment

# --- models -----------------------------------------------------------------
MODELS: dict[str, ModelSpec] = {
    "distilbert": ModelSpec("distilbert", "distilbert-base-uncased", "distilbert"),
    "bert": ModelSpec("bert", "bert-base-uncased", "bert"),
    "roberta": ModelSpec("roberta", "roberta-base", "roberta"),
    # Modern encoder (2024) — 4th family, tests whether frozen-predictability generalizes
    # beyond 2018-era models. Names blocks `model.layers.<i>`; needs sdpa attention (see
    # encoders/hf_text.py).
    "modernbert": ModelSpec("modernbert", "answerdotai/ModernBERT-base", "modernbert"),
}

# --- datasets (see PROJECT.md §6; quirks encoded here) ----------------------
DATASETS: dict[str, DatasetSpec] = {
    # SST-2 test labels are hidden (-1) → evaluate on the validation split.
    "sst2": DatasetSpec(
        "sst2", "stanfordnlp/sst2", "sentence", "label", 2,
        eval_split="validation", subsample_train=15000, domain="sentiment",
    ),
    "agnews": DatasetSpec(
        "agnews", "fancyzhx/ag_news", "text", "label", 4,
        subsample_train=15000, domain="topic",
    ),
    # TREC: use the coarse 6-class label, not the 50-way fine label.
    "trec": DatasetSpec("trec", "SetFit/TREC-QC", "text", "label_coarse", 6, domain="question"),
    "emotion": DatasetSpec(
        "emotion", "dair-ai/emotion", "text", "label", 6,
        subsample_train=15000, domain="emotion",
    ),
    "banking77": DatasetSpec("banking77", "mteb/banking77", "text", "label", 77, domain="intent"),
    # CLINC "plus" has 150 intents + an out-of-scope bucket → drop oos → 150.
    "clinc150": DatasetSpec(
        "clinc150", "clinc/clinc_oos", "text", "intent", 150,
        hf_config="plus", drop_labels=("oos",), subsample_train=15000, domain="intent",
    ),
    # --- Exp3 expansion (short-text, same domains; span the need spectrum) -------
    # sentiment: short movie snippets (train/validation/test).
    "rotten_tomatoes": DatasetSpec(
        "rotten_tomatoes", "cornell-movie-review-data/rotten_tomatoes", "text", "label", 2,
        domain="sentiment",
    ),
    # topic: clean Wikipedia abstracts (train/test only → val carved from train).
    "dbpedia14": DatasetSpec(
        "dbpedia14", "fancyzhx/dbpedia_14", "content", "label", 14,
        subsample_train=15000, domain="topic",
    ),
    # topic: noisy Q&A — use the short title only to stay single-sentence/comparable.
    "yahoo": DatasetSpec(
        "yahoo", "community-datasets/yahoo_answers_topics", "question_title", "topic", 10,
        subsample_train=15000, domain="topic",
    ),
    # emotion: tweets (train/validation/test) — 2nd emotion set to add high-gap points.
    "tweet_emotion": DatasetSpec(
        "tweet_emotion", "cardiffnlp/tweet_eval", "text", "label", 4,
        hf_config="emotion", domain="emotion",
    ),
    # intent: virtual-assistant utterances, 60 intents (train/validation/test).
    "massive": DatasetSpec(
        "massive", "mteb/amazon_massive_intent", "text", "label", 60,
        hf_config="en", domain="intent",
    ),
}


def dataset_domain(key: str) -> str:
    return DATASETS[key].domain


def resolve_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model '{key}'. known: {sorted(MODELS)}")
    return MODELS[key]


def resolve_dataset(key: str) -> DatasetSpec:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset '{key}'. known: {sorted(DATASETS)}")
    return DATASETS[key]


# --- experiments ------------------------------------------------------------
_EXPERIMENTS: dict[str, type["Experiment"]] = {}


def register_experiment(name: str):
    def deco(cls: type["Experiment"]) -> type["Experiment"]:
        cls.name = name
        _EXPERIMENTS[name] = cls
        return cls

    return deco


def get_experiment(name: str) -> type["Experiment"]:
    if name not in _EXPERIMENTS:
        raise KeyError(
            f"unknown experiment '{name}'. known: {sorted(_EXPERIMENTS)}"
        )
    return _EXPERIMENTS[name]


def experiment_names() -> list[str]:
    return sorted(_EXPERIMENTS)
