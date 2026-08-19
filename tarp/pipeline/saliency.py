"""Token saliency via last-layer [CLS] attention, before vs after fine-tuning (Exp5b).

Qualitative diagnostic: for a handful of examples, read how much the final-layer [CLS]
query attends to each token, for the frozen base encoder vs. the fine-tuned one. The text
analog of MulTaBench's attention maps — the fine-tuned model should concentrate on
class-discriminative words. Best-effort: attention is a side diagnostic (our primary
pooling is mean), so callers guard failures rather than aborting the experiment.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from tarp.config import ModelSpec, RunConfig
from tarp.pipeline.run import load_finetuned_encoder


def _cls_attention(model, tokenizer, text: str, device: str, max_length: int = 128):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_attentions=True)
    att = out.attentions[-1][0]          # (heads, T, T) last layer
    cls_to_tokens = att[:, 0, :].mean(0)  # average heads; query = position 0 ([CLS]/<s>)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return tokens, cls_to_tokens.detach().cpu().numpy()


def saliency_shift(src_cfg: RunConfig, texts: list[str], top_k: int = 6,
                   max_length: int = 128) -> list[dict]:
    """Per text: tokens ranked by increase in [CLS] attention after fine-tuning.

    ``src_cfg`` names the fine-tuned encoder (its own dataset/adapter). Returns a list of
    ``{text, top_gained}`` records, where ``top_gained`` are the tokens whose CLS-attention
    grew most from frozen → fine-tuned.
    """
    spec: ModelSpec = src_cfg.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    # eager attention is required to read attention weights back (SDPA/flash return none)
    frozen = AutoModel.from_pretrained(spec.hf_id, attn_implementation="eager").to(device).eval()
    ft = load_finetuned_encoder(src_cfg, src_cfg.dataset.num_labels,
                                attn_implementation="eager").model.to(device).eval()

    records = []
    for text in texts:
        toks_f, att_f = _cls_attention(frozen, tok, text, device, max_length)
        toks_t, att_t = _cls_attention(ft, tok, text, device, max_length)
        m = min(len(att_f), len(att_t))
        delta = att_t[:m] - att_f[:m]
        order = np.argsort(-delta)
        top = [(toks_t[i], float(delta[i])) for i in order[:top_k]
               if toks_t[i] not in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]")]
        records.append({"text": text, "top_gained": top})
    return records
