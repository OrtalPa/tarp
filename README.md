# Predicting the Need for Task-Aware Adaptation from Frozen Representations

NLP final project (RUNI 2026), based on **MulTaBench**.

Fine-tuning an encoder on a target task ("task-aware adaptation") helps a lot on some datasets
and barely at all on others, but you normally only find out by paying for the fine-tune. This
project asks whether that outcome is **predictable in advance**, from the frozen encoder's
representations alone.

We define the *adaptation gain* as `gap = acc(fine-tuned) − acc(frozen linear probe)` and
measure it over a grid of **4 encoder families × 11 text-classification datasets × 3 seeds**
(BERT, RoBERTa, DistilBERT, ModernBERT; sentiment, topic, question, emotion and intent tasks).
Five experiments then relate that gain to properties of the frozen representations — how much
fine-tuning shifts them (CKA), where along depth it acts, which frozen descriptors predict the
gain, how class structure is reshaped layer by layer, and what generic information adaptation
trades away.

The headline result: frozen-probe accuracy alone orders datasets by adaptation need almost
monotonically (Spearman ≈ −0.80), so the need for adaptation can be estimated before any
fine-tuning takes place.

## Layout

```
tarp/                  the harness (installable-free package, run via `python -m tarp.cli`)
  cli.py               entry point:  `run` (compute)  |  `report` (analyze + plot from cache)
  config.py            frozen dataclasses for runs + the layered cache keys
  registry.py          model / dataset / experiment registries
  data/                dataset loading and split handling
  encoders/            HuggingFace encoder wrapper + pooling (mean, [CLS])
  finetune/            LoRA and full fine-tuning
  probes/              linear probes over frozen representations
  pipeline/            caching and all model-side effects (extraction, cross-task, saliency)
  experiments/         exp1–exp5 orchestration → tidy rows in results.parquet
  metrics/             pure numeric functions (CKA, separability, similarity)
  analysis/            pure statistics over the results table
  plots.py             figure generation

scripts/               run drivers and one-off helpers (full grid, figure regeneration, dataset peeks)
tests/                 unit tests for the pure layers + one end-to-end smoke test
outputs/               results.parquet, figures, text reports and per-run metadata
  figures/regenerated/ the final figure set
```

## Running it

```bash
uv sync                                     # install dependencies
uv run pytest                               # test suite (CPU-only, ~1 min)

# one experiment over a model x dataset x seed grid
uv run python -m tarp.cli run --experiment exp1 --models distilbert --datasets trec --seeds 0

# statistics + figures from the cached results, no recompute
uv run python -m tarp.cli report --experiment exp1
```

`run` is the expensive half and caches aggressively: a given (model, dataset, config, seed) is
fine-tuned exactly once and its representations are reused across experiments. `report` only
reads `outputs/results/results.parquet`. `scripts/run_all.sh` drives the full grid.
