"""Shared run primitive: execute all conditions for one RunConfig, with layered caching.

``run_conditions(cfg)`` returns a ``RunResult`` holding everything the experiments need:
frozen & fine-tuned per-layer test reps, the labels, frozen-probe & fine-tuned accuracy,
per-layer probe accuracies, and frozen train reps (for RQ3 separability).

Caching guarantees fine-tuning happens **once** per ``finetune_key`` (independent of
pooling/experiment); a fully-cached repeat run runs no model forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tarp.config import RunConfig
from tarp.data import load_task
from tarp.encoders import HFTextEncoder
from tarp.finetune import finetune
from tarp.pipeline import cache
from tarp.pipeline.extract import get_or_extract
from tarp.probes import probe_all_layers
from tarp.seed import set_seed


@dataclass
class RunResult:
    frozen_reps: np.ndarray        # (L+1, Nte, H) frozen test reps
    ft_reps: np.ndarray            # (L+1, Nte, H) fine-tuned test reps
    labels: np.ndarray             # test labels
    frozen_acc: float
    ft_acc: float
    per_layer_probe_acc: list[float]
    frozen_train_reps: np.ndarray  # (L+1, Ntr, H) frozen train reps (for separability)
    train_labels: np.ndarray
    num_labels: int
    label_names: list[str]

    @property
    def gap(self) -> float:
        return self.ft_acc - self.frozen_acc


def load_finetuned_encoder(cfg: RunConfig, num_labels: int,
                           attn_implementation: str | None = None) -> HFTextEncoder:
    """Reload a cached fine-tuned model as an encoder. LoRA runs save an adapter over the
    base model; full fine-tuning saves the whole model — both live under ``ft_dir/adapter``.
    Public so Exp5 can push *other* datasets' inputs through a source-task-adapted encoder.
    ``attn_implementation="eager"`` is needed when the caller wants attention weights back
    (SDPA/flash return none) — e.g. the Exp5b saliency diagnostic."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    saved = str(cache.ft_dir(cfg.finetune_key()) / "adapter")
    tok = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    # ModernBERT falls back to sdpa where FlashAttention-2 is unavailable; default it when the
    # caller didn't ask for a specific backend.
    if attn_implementation is None and cfg.model.family == "modernbert":
        attn_implementation = "sdpa"
    kw = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    if cfg.finetune.regime == "full":
        model = AutoModelForSequenceClassification.from_pretrained(saved, num_labels=num_labels, **kw)
    else:
        from peft import PeftModel

        base = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.hf_id, num_labels=num_labels, **kw
        )
        model = PeftModel.from_pretrained(base, saved)
    return HFTextEncoder.wrap(model, tok, cfg.model, cfg.max_length)


def _save_finetuned(cfg: RunConfig, res) -> None:
    d = cache.ft_dir(cfg.finetune_key())
    res.model.save_pretrained(str(d / "adapter"))
    cache.save_json(d / "meta.json", {"test_acc": res.test_acc, "val_acc": res.val_acc})
    np.save(d / "test_logits.npy", res.test_logits)


def run_conditions(cfg: RunConfig, force: bool = False) -> RunResult:
    set_seed(cfg.seed)
    task = load_task(cfg.dataset, cfg.seed)

    # Lazily build the frozen encoder only if some extraction is uncached.
    _cache: dict[str, HFTextEncoder] = {}

    def frozen_encoder() -> HFTextEncoder:
        if "enc" not in _cache:
            _cache["enc"] = HFTextEncoder.frozen(cfg.model, cfg.max_length)
        return _cache["enc"]

    # --- frozen representations (train + test) ---------------------------
    k_tr = cfg.extraction_key("frozen", cfg.pooling, "train")
    k_te = cfg.extraction_key("frozen", cfg.pooling, "test")
    frozen_train = get_or_extract(k_tr, task.train[0], cfg.pooling, frozen_encoder)
    frozen_test = get_or_extract(k_te, task.test[0], cfg.pooling, frozen_encoder)

    # --- frozen linear probe (cached) ------------------------------------
    ppath = cache.probe_path(cfg.model.key, cfg.dataset.key, cfg.pooling, cfg.seed)
    pj = None if force else cache.load_json(ppath)
    if pj is None:
        pr = probe_all_layers(frozen_train, task.train[1], frozen_test, task.test[1], seed=cfg.seed)
        pj = {"per_layer": pr.per_layer, "final": pr.final,
              "best_layer": pr.best_layer, "best": pr.best}
        cache.save_json(ppath, pj)
    frozen_acc = float(pj["final"])
    per_layer_probe = list(pj["per_layer"])

    # --- fine-tune (cached adapter + metrics) ----------------------------
    fkey = cfg.finetune_key()
    meta = None if force else cache.load_json(cache.ft_dir(fkey) / "meta.json")
    k_ft = cfg.extraction_key(f"ft:{fkey}", cfg.pooling, "test")

    if meta is not None and cache.has_reps(k_ft):
        ft_acc = float(meta["test_acc"])
        ft_test = cache.load_reps(k_ft)
    else:
        if meta is not None and (cache.ft_dir(fkey) / "adapter").exists():
            ft_enc = load_finetuned_encoder(cfg, task.num_labels)
            ft_acc = float(meta["test_acc"])
        else:
            res = finetune(frozen_encoder(), task, cfg, cache.ft_dir(fkey) / "trainer")
            _save_finetuned(cfg, res)
            ft_enc = HFTextEncoder.wrap(res.model, res.tokenizer, cfg.model, cfg.max_length)
            ft_acc = float(res.test_acc)
        ft_test = get_or_extract(k_ft, task.test[0], cfg.pooling, lambda: ft_enc)

    return RunResult(
        frozen_reps=frozen_test,
        ft_reps=ft_test,
        labels=np.asarray(task.test[1]),
        frozen_acc=frozen_acc,
        ft_acc=ft_acc,
        per_layer_probe_acc=per_layer_probe,
        frozen_train_reps=frozen_train,
        train_labels=np.asarray(task.train[1]),
        num_labels=task.num_labels,
        label_names=task.label_names,
    )
