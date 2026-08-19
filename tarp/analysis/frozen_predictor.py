"""Exp3 analysis — predict the adaptation gap from frozen-only features.

Pure statistics over the results table. Each data point is a (model, dataset, seed) pair.
We report (1) univariate Spearman(feature, gap); (2) a leave-one-dataset-out multivariate
linear regression, predicted-vs-actual rank correlation + R²; (3) whether the separability
features add signal *beyond* frozen accuracy; (4) a binary needs-TAR threshold read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

_KEYS = ["model", "dataset", "seed"]
# frozen-only candidate predictors emitted by Exp3 (frozen_acc handled separately as baseline)
FEATURES = ["frozen_acc", "silhouette", "intra_inter_ratio", "knn_acc", "anisotropy", "eff_dim"]
# The model we actually headline and recommend: frozen accuracy alone. Everything reported as
# a single number (the binary needs-TAR read, the predicted-vs-actual figure) is scored with
# this, not with all of FEATURES — the full set overfits leave-one-dataset-out (R² 0.155 vs
# 0.356), so judging the recommendation by it would understate what we recommend.
HEADLINE_FEATURES = ["frozen_acc"]


def build_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (model, dataset, seed): gap (+ headroom-normalized) and every frozen feature."""
    from tarp.results import load

    df = load() if df is None else df
    d = df[df["experiment"] == "exp3"]

    def scalar(metric: str) -> pd.DataFrame:
        return d[d["metric"] == metric][_KEYS + ["value"]].rename(columns={"value": metric})

    t = scalar("gap")
    for m in FEATURES:
        t = t.merge(scalar(m), on=_KEYS, how="left")
    t["gap_headroom"] = t["gap"] / (1.0 - t["frozen_acc"]).clip(lower=1e-6)
    return t.sort_values("gap").reset_index(drop=True)


def univariate(t: pd.DataFrame, target: str = "gap") -> pd.DataFrame:
    """Spearman(feature, target) for each frozen feature."""
    out = []
    for f in FEATURES:
        x, y = t[f].to_numpy(float), t[target].to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(y))
        rho, p = (stats.spearmanr(x[ok], y[ok]) if ok.sum() >= 3 else (np.nan, np.nan))
        out.append({"feature": f, "spearman": float(rho), "p": float(p), "n": int(ok.sum())})
    return pd.DataFrame(out).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def _loo_predict(t: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    """Leave-one-dataset-out predictions aligned to ``t``'s row order."""
    X = t[features].to_numpy(float)
    y = t[target].to_numpy(float)
    datasets = t["dataset"].to_numpy()
    pred = np.full(len(t), np.nan)
    for ds in np.unique(datasets):
        te = datasets == ds
        tr = ~te
        if tr.sum() < 2:
            continue
        model = LinearRegression().fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def loo_regression(t: pd.DataFrame, features: list[str], target: str = "gap") -> dict:
    """Leave-one-dataset-out CV: predicted-vs-actual Spearman/Pearson and out-of-fold R²."""
    pred = _loo_predict(t, features, target)
    y = t[target].to_numpy(float)
    ok = ~(np.isnan(pred) | np.isnan(y))
    if ok.sum() < 3:
        return {"features": features, "n": int(ok.sum()), "spearman": np.nan,
                "pearson": np.nan, "r2": np.nan}
    yp, yt = pred[ok], y[ok]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "features": features,
        "n": int(ok.sum()),
        "spearman": float(stats.spearmanr(yp, yt)[0]),
        "pearson": float(stats.pearsonr(yp, yt)[0]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def needs_tar(t: pd.DataFrame, features: list[str], delta: float = 0.05) -> dict:
    """Binary read: threshold gap at delta, predict via LOO regression, report accuracy."""
    pred = _loo_predict(t, features, "gap")
    y = t["gap"].to_numpy(float)
    ok = ~(np.isnan(pred) | np.isnan(y))
    if ok.sum() < 3:
        return {"delta": delta, "n": int(ok.sum()), "accuracy": np.nan, "positives": int((y[ok] >= delta).sum())}
    yhat = (pred[ok] >= delta).astype(int)
    ytrue = (y[ok] >= delta).astype(int)
    # majority-class accuracy, the baseline any binary read has to beat
    base = max(ytrue.mean(), 1.0 - ytrue.mean())
    return {
        "delta": delta,
        "features": features,
        "n": int(ok.sum()),
        "accuracy": float((yhat == ytrue).mean()),
        "majority": float(base),
        "positives": int(ytrue.sum()),
    }


def analyze(t: pd.DataFrame) -> dict:
    sep_only = [f for f in FEATURES if f != "frozen_acc"]
    return {
        "univariate": univariate(t),
        "loo_all": loo_regression(t, FEATURES),
        "loo_sep_only": loo_regression(t, sep_only),
        "loo_frozen_only": loo_regression(t, ["frozen_acc"]),
        "needs_tar": needs_tar(t, HEADLINE_FEATURES),
    }


def format_report(t: pd.DataFrame, res: dict) -> str:
    lines = ["Exp3 — predicting the gap from frozen features:", "",
             t.round(4).to_string(index=False), "",
             "Univariate Spearman(feature, gap):", res["univariate"].round(3).to_string(index=False), ""]

    def line(tag: str, r: dict) -> str:
        return (f"  {tag:16s} n={r['n']}  spearman={r['spearman']:+.3f}  "
                f"pearson={r['pearson']:+.3f}  R2={r['r2']:+.3f}")

    lines += ["Leave-one-dataset-out regression (predicted vs actual gap):",
              line("all features", res["loo_all"]),
              line("separability", res["loo_sep_only"]),
              line("frozen_acc only", res["loo_frozen_only"]), ""]
    nt = res["needs_tar"]
    lines.append(f"Binary needs-TAR (gap>={nt['delta']}, features={'+'.join(nt['features'])}): "
                 f"accuracy={nt['accuracy']:.3f} vs majority={nt['majority']:.3f} "
                 f"(n={nt['n']}, positives={nt['positives']})")
    return "\n".join(lines)
