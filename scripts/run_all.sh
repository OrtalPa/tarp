#!/usr/bin/env bash
# Full robustness + ModernBERT pass.
#
# Seeds {0, 42, 11} × 11 datasets, for Exp1/Exp3/Exp4:
#   - seed 0  : ONLY ModernBERT is new (distilbert/bert/roberta already ran at seed 0).
#   - seed 42 : all 4 models (distilbert, bert, roberta, modernbert).
#   - seed 11 : all 4 models.
# Then a combined across-seeds report (mean ± std error bars).
#
# ADDITIVE / NON-DESTRUCTIVE: `run` only APPENDS to results.parquet (dedup key includes seed, so
# seed-0 rows are untouched); per-pair CKA figures are seed-tagged (`_s<seed>_cka.png`); the
# combined report writes only NEW files (report_allseeds.txt, *_allseeds.png). No canonical plot
# or existing experiment is overwritten. NOT using `set -e` so one failed pair can't abort the
# whole overnight run — failures are logged and the script keeps going.
#
# Launch so it survives closing the terminal (reparented to init):
#   setsid nohup bash scripts/run_all.sh > outputs/run_all.log 2>&1 &
# Watch:  tail -f outputs/run_all.log
set -uo pipefail
cd "$(dirname "$0")/.."

DATASETS="sst2 agnews trec emotion banking77 clinc150 rotten_tomatoes dbpedia14 yahoo tweet_emotion massive"
ALL4="distilbert bert roberta modernbert"

run_seed () {  # $1 = seed ; $2 = space-separated models
  local seed="$1"; local models="$2"
  for exp in exp1 exp3 exp4; do
    echo "########## seed=$seed  $exp  models=[$models] ##########  $(date)"
    uv run python -m tarp.cli run --experiment "$exp" \
      --models $models --datasets $DATASETS --seeds "$seed" \
      || echo "WARN: seed=$seed $exp FAILED (continuing)  $(date)"
  done
}

run_seed 0  "modernbert"
run_seed 42 "$ALL4"
run_seed 11 "$ALL4"

echo "########## combined across-seeds report ##########  $(date)"
PYTHONPATH=. uv run python scripts/report_combined.py \
  || echo "WARN: combined report FAILED  $(date)"

echo "ALL DONE  $(date)"
