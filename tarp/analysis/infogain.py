"""Exp5 analysis — target gain vs. generic loss during adaptation.

Pure statistics over the results table. One row per (model, dataset, seed) with the target
gain (the gap) and the generic-information loss; we relate them across pairs (does adapting
harder for the target cost more generic decodability?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_KEYS = ["model", "dataset", "seed"]


def build_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (model, dataset, seed): target_gain, generic_loss, frozen_acc, domain."""
    from tarp.registry import dataset_domain
    from tarp.results import load

    df = load() if df is None else df
    d = df[df["experiment"] == "exp5"]

    def scalar(metric: str) -> pd.DataFrame:
        return d[d["metric"] == metric][_KEYS + ["value"]].rename(columns={"value": metric})

    t = scalar("target_gain")
    for m in ("generic_loss", "frozen_acc"):
        t = t.merge(scalar(m), on=_KEYS, how="left")
    t["domain"] = t["dataset"].map(dataset_domain)
    return t.sort_values("target_gain").reset_index(drop=True)


def correlations(t: pd.DataFrame) -> dict:
    x, y = t["target_gain"].to_numpy(float), t["generic_loss"].to_numpy(float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "pearson": np.nan, "spearman": np.nan}
    return {"n": int(ok.sum()),
            "pearson": float(stats.pearsonr(x[ok], y[ok])[0]),
            "spearman": float(stats.spearmanr(x[ok], y[ok])[0])}


def format_report(t: pd.DataFrame, corr: dict) -> str:
    lines = ["Exp5 — target gain vs. generic loss:", "",
             t.round(4).to_string(index=False), "",
             f"correlation(target_gain, generic_loss): n={corr['n']}  "
             f"pearson={corr['pearson']:+.3f}  spearman={corr['spearman']:+.3f}"]
    return "\n".join(lines)
