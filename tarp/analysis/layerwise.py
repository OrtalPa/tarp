"""Exp2 analysis — where adaptation happens, LoRA vs full-FT.

Pure statistics over the results table. Produces (1) per-layer CKA curves aggregated across
datasets — mean ± 95% CI, per (model, regime) — to localize adaptation, and (2) a per-pair
gap comparison between the two fine-tuning regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_KEYS = ["model", "dataset", "seed"]


def _exp2(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["experiment"] == "exp2"]


def layer_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, ft_regime, layer): mean CKA and 95% CI half-width across datasets/seeds."""
    d = _exp2(df)
    cka = d[(d["metric"] == "cka") & (d["layer"] >= 0)]
    out = (cka.groupby(["model", "ft_regime", "layer"])["value"]
           .agg(cka_mean="mean", std="std", n="count").reset_index())
    sem = out["std"].fillna(0.0) / np.sqrt(out["n"].clip(lower=1))
    out["ci"] = (1.96 * sem).where(out["n"] > 1, 0.0)
    return out.drop(columns="std").sort_values(["model", "ft_regime", "layer"]).reset_index(drop=True)


def gap_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, dataset, seed): gap under each regime, side by side."""
    d = _exp2(df)
    gap = d[d["metric"] == "gap"]
    wide = (gap.pivot_table(index=_KEYS, columns="ft_regime", values="value")
            .reset_index())
    wide.columns.name = None
    return wide


def format_report(curves: pd.DataFrame, gaps: pd.DataFrame) -> str:
    lines = ["Exp2 — layer-wise adaptation (CKA mean across datasets, per model/regime):", ""]
    for (model, regime), g in curves.groupby(["model", "ft_regime"]):
        vals = "  ".join(f"L{int(r.layer)}={r.cka_mean:.2f}" for r in g.itertuples())
        lines.append(f"  {model:11s} [{regime:10s}] {vals}")
    lines += ["", "Per-pair gap by regime:"]
    if {"lora_last3", "full"} <= set(gaps.columns):
        g = gaps.dropna(subset=["lora_last3", "full"])
        lines.append(g.round(4).to_string(index=False))
        if len(g) >= 2:
            r = stats.pearsonr(g["lora_last3"], g["full"])[0]
            lines.append(f"\ngap correlation LoRA vs full-FT: pearson={r:+.3f}  "
                         f"mean(full−lora)={(g['full'] - g['lora_last3']).mean():+.4f}")
    else:
        lines.append("  (need both regimes present — run exp2 over the grid first)")
    return "\n".join(lines)
