"""Exp1 analysis — representation shift vs. adaptation gain.

Pure statistics over the results table (no model forward passes, no I/O beyond reading the parquet). Builds
one row per (model, dataset, seed) and relates the shift summary to the gap, pooled and
grouped by model / domain, with a partial correlation controlling for frozen accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tarp.registry import dataset_domain
from tarp.results import load

_KEYS = ["model", "dataset", "seed"]


def build_table(df: pd.DataFrame | None = None, experiment: str = "exp1",
                ft_regime: str = "lora_last3") -> pd.DataFrame:
    """One row per (model, dataset, seed): shift, gap, frozen_acc, ft_acc, domain."""
    df = load() if df is None else df
    d = df[(df["experiment"] == experiment) & (df["ft_regime"] == ft_regime)]

    def scalar(metric, name):
        return d[d["metric"] == metric][_KEYS + ["value"]].rename(columns={"value": name})

    gap = scalar("gap", "gap")
    frozen = scalar("frozen_acc", "frozen_acc")
    ft = scalar("ft_acc", "ft_acc")
    # shift = 1 - mean CKA over transformer layers (exclude embeddings layer 0)
    cka = d[(d["metric"] == "cka") & (d["layer"] >= 1)]
    shift = (cka.groupby(_KEYS)["value"].mean().reset_index()
             .assign(shift=lambda x: 1.0 - x["value"])[_KEYS + ["shift"]])

    t = gap.merge(shift, on=_KEYS).merge(frozen, on=_KEYS).merge(ft, on=_KEYS)
    t["domain"] = t["dataset"].map(dataset_domain)
    return t.sort_values("gap").reset_index(drop=True)


def _corr(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(len(x))}
    return {"pearson": float(stats.pearsonr(x, y)[0]),
            "spearman": float(stats.spearmanr(x, y)[0]), "n": int(len(x))}


def _partial_corr(x, y, z) -> float:
    """corr(x, y | z): Pearson of residuals after linearly regressing out z."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    if len(x) < 4:
        return float("nan")
    Z = np.c_[np.ones_like(z), z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(stats.pearsonr(rx, ry)[0])


def correlations(t: pd.DataFrame) -> dict:
    pooled = _corr(t["shift"], t["gap"])
    pooled["partial_given_frozen"] = _partial_corr(t["shift"], t["gap"], t["frozen_acc"])
    return {
        "pooled": pooled,
        "per_model": {m: _corr(g["shift"], g["gap"]) for m, g in t.groupby("model")},
        "per_domain": {d: _corr(g["shift"], g["gap"]) for d, g in t.groupby("domain")},
    }


def format_report(t: pd.DataFrame, corr: dict) -> str:
    lines = ["shift-vs-gain table:", t.round(4).to_string(index=False), ""]
    p = corr["pooled"]
    lines.append(
        f"pooled (n={p['n']}): pearson={p['pearson']:+.3f}  spearman={p['spearman']:+.3f}  "
        f"partial(shift,gap | frozen_acc)={p['partial_given_frozen']:+.3f}"
    )
    lines.append("per-model:  " + "  ".join(
        f"{m}: r={v['pearson']:+.2f}(n={v['n']})" for m, v in corr["per_model"].items()))
    lines.append("per-domain: " + "  ".join(
        f"{d}: r={v['pearson']:+.2f}(n={v['n']})" for d, v in corr["per_domain"].items()))
    return "\n".join(lines)
