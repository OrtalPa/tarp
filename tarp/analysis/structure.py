"""Exp4 analysis — layer-wise structure, frozen vs. fine-tuned.

Pure statistics over the results table. Produces (1) per-layer structural curves
(mean ± 95% CI across datasets) for each metric, per (model, condition); and (2) a
frozen→FT *structural delta* at the final layer, related to the accuracy gap — i.e. does
the amount of structure fine-tuning injects at the top track how much accuracy it buys?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_KEYS = ["model", "dataset", "seed"]
METRICS = ["intra_inter_ratio", "silhouette", "knn_acc", "anisotropy", "eff_dim"]
# lower = better separated for intra_inter_ratio; higher = better for silhouette/knn.
_LOWER_IS_BETTER = {"intra_inter_ratio"}


def _exp4(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["experiment"] == "exp4"]


def structure_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, metric, condition, layer): mean value + 95% CI half-width across datasets."""
    d = _exp4(df)
    d = d[(d["layer"] >= 0) & (d["metric"].isin(METRICS))]
    out = (d.groupby(["model", "metric", "condition", "layer"])["value"]
           .agg(mean="mean", std="std", n="count").reset_index())
    sem = out["std"].fillna(0.0) / np.sqrt(out["n"].clip(lower=1))
    out["ci"] = (1.96 * sem).where(out["n"] > 1, 0.0)
    return (out.drop(columns="std")
            .sort_values(["metric", "model", "condition", "layer"]).reset_index(drop=True))


def _final_layer_values(df: pd.DataFrame, metric: str, condition: str) -> pd.DataFrame:
    """The metric at each pair's top (max-index) layer, for one condition."""
    d = _exp4(df)
    d = d[(d["metric"] == metric) & (d["condition"] == condition) & (d["layer"] >= 0)]
    top = d.loc[d.groupby(_KEYS)["layer"].idxmax()]
    return top[_KEYS + ["value"]].rename(columns={"value": condition})


def top_delta(df: pd.DataFrame, metric: str = "intra_inter_ratio") -> pd.DataFrame:
    """Per (model, dataset, seed): frozen vs. FT ``metric`` at the final layer, the frozen→FT
    delta, and the accuracy gap. Delta is signed so that *positive = fine-tuning improved
    separability* regardless of the metric's native direction."""
    d = _exp4(df)
    fro = _final_layer_values(df, metric, "frozen")
    fin = _final_layer_values(df, metric, "ft").rename(columns={"ft": "ft_val"})
    gap = (d[(d["metric"] == "gap")][_KEYS + ["value"]].rename(columns={"value": "gap"}))
    t = fro.rename(columns={"frozen": "frozen_val"}).merge(fin, on=_KEYS).merge(gap, on=_KEYS)
    raw = t["frozen_val"] - t["ft_val"]  # drop in the metric under FT
    t["improve"] = raw if metric in _LOWER_IS_BETTER else -raw
    return t.sort_values("gap").reset_index(drop=True)


def format_report(curves: pd.DataFrame, deltas: dict[str, pd.DataFrame]) -> str:
    lines = ["Exp4 — layer-wise structure (mean across datasets), frozen vs fine-tuned:", ""]
    for metric in METRICS:
        cm = curves[curves["metric"] == metric]
        if cm.empty:
            continue
        lines.append(f"[{metric}]")
        for (model, cond), g in cm.groupby(["model", "condition"]):
            g = g.sort_values("layer")
            vals = " ".join(f"L{int(r.layer)}={r.mean:.2f}" for r in g.itertuples())
            lines.append(f"  {model:11s} {cond:6s} {vals}")
        lines.append("")

    lines.append("Final-layer frozen→FT structural improvement vs. accuracy gap:")
    for metric, t in deltas.items():
        ok = t.dropna(subset=["improve", "gap"])
        if len(ok) >= 3:
            r = stats.pearsonr(ok["improve"], ok["gap"])[0]
            rho = stats.spearmanr(ok["improve"], ok["gap"])[0]
            lines.append(f"  {metric:18s} corr(improvement, gap): "
                         f"pearson={r:+.3f} spearman={rho:+.3f} (n={len(ok)})")
    return "\n".join(lines)
