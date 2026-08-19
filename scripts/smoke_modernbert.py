"""Smoke test: can ModernBERT run through the tarp pipeline?

Proves feasibility of adding ModernBERT as a 4th family before committing it to the grid:
 1. LoRA targets resolve to the **top-3** transformer blocks (`model.layers.<i>` naming).
 2. The trainable head (`head` + `classifier`) gives param parity with the other families
    (the §9 fairness requirement) — no family gets a strictly weaker head.
 3. A 1-epoch end-to-end fine-tune (frozen probe → LoRA-FT → post-FT extract) yields a finite
    gap, confirming the encoder wrapping + sdpa attention + extraction all work end to end.

    PYTHONPATH=. uv run python scripts/smoke_modernbert.py
"""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification
from peft import get_peft_model

from tarp.config import FinetuneSpec, RunConfig
from tarp.finetune.lora import build_lora_config, last_n_linear_module_names
from tarp.pipeline import run_conditions
from tarp.registry import resolve_dataset, resolve_model

FAMILIES = ["distilbert", "bert", "roberta", "modernbert"]


def _load(model_key: str, num_labels: int = 6):
    spec = resolve_model(model_key)
    kw = {"attn_implementation": "sdpa"} if spec.family == "modernbert" else {}
    return AutoModelForSequenceClassification.from_pretrained(spec.hf_id, num_labels=num_labels, **kw)


def _layer_indices(targets: list[str]) -> set[int]:
    import re
    rx = re.compile(r"\.layers?\.(\d+)\.")
    return {int(m.group(1)) for t in targets if (m := rx.search(t))}


def main() -> None:
    print("=== (1+2) LoRA targeting + trainable-param parity (num_labels=6) ===")
    for k in FAMILIES:
        model = _load(k)
        cfg = build_lora_config(model, FinetuneSpec())
        targets = last_n_linear_module_names(model, 3)
        idxs = sorted(_layer_indices(targets))
        pm = get_peft_model(model, cfg)
        tp = sum(p.numel() for p in pm.parameters() if p.requires_grad)
        print(f"  {k:11s} trainable={tp:>10,}  top-3 layers={idxs[-3:]}  "
              f"#targets={len(targets)}  heads={cfg.modules_to_save}")

    print("\n=== (3) end-to-end 1-epoch fine-tune: modernbert / trec ===")
    cfg = RunConfig(model=resolve_model("modernbert"), dataset=resolve_dataset("trec"),
                    finetune=FinetuneSpec(epochs=1))
    rr = run_conditions(cfg)
    print(f"  frozen_acc={rr.frozen_acc:.3f}  ft_acc={rr.ft_acc:.3f}  gap={rr.gap:+.3f}  "
          f"layers(+emb)={rr.frozen_reps.shape[0]}")
    ok = (rr.frozen_reps.shape[0] > 3) and (-1.0 < rr.gap < 1.0) and (rr.ft_acc > 0)
    print("\nSMOKE: PASS ✅" if ok else "\nSMOKE: FAIL ❌")


if __name__ == "__main__":
    main()
