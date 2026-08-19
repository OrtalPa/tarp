"""Combined across-seeds report — run after scripts/run_all.sh.

Computes each headline statistic **per seed** and reports **mean ± std across seeds** (the
robustness / error-bar view the pooled report can't give, since pooling treats the same dataset
under different seeds as independent). Reuses the tested `analysis/` functions per-seed.

Writes ONLY new files — never overwrites canonical plots/reports:
  outputs/report_allseeds.txt
  outputs/figures/exp3_feature_spearman_allseeds.png

    PYTHONPATH=. uv run python scripts/report_combined.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from tarp.analysis import frozen_predictor as FP  # noqa: E402
from tarp.analysis import structure as ST  # noqa: E402
from tarp.results import load  # noqa: E402


def _partial(x, y, z) -> float:
    """Partial Pearson corr(x, y | z): residualize x and y on [1, z], correlate residuals."""
    X = np.c_[np.ones_like(z), z]
    bx = np.linalg.lstsq(X, x, rcond=None)[0]
    by = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(stats.pearsonr(x - X @ bx, y - X @ by)[0])


def exp1_partial_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["experiment"] == "exp1"]
    out = []
    for s, g in d.groupby("seed"):
        cka = g[(g["metric"] == "cka") & (g["layer"] > 0)].groupby(["model", "dataset"])["value"].mean()
        gap = g[g["metric"] == "gap"].set_index(["model", "dataset"])["value"]
        fa = g[g["metric"] == "frozen_acc"].set_index(["model", "dataset"])["value"]
        t = pd.concat([cka.rename("mcka"), gap.rename("gap"), fa.rename("fa")], axis=1).dropna()
        if len(t) > 3:
            shift = 1.0 - t["mcka"].to_numpy()
            out.append({"seed": int(s), "n": len(t),
                        "partial(shift,gap|frozen_acc)": _partial(shift, t["gap"].to_numpy(), t["fa"].to_numpy())})
    return pd.DataFrame(out)


def exp3_univariate_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s in sorted(df[df["experiment"] == "exp3"]["seed"].unique()):
        t = FP.build_table(df[df["seed"] == s])
        u = FP.univariate(t).set_index("feature")["spearman"]
        out.append({"seed": int(s), **u.to_dict()})
    return pd.DataFrame(out)


def exp3_lodo_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s in sorted(df[df["experiment"] == "exp3"]["seed"].unique()):
        t = FP.build_table(df[df["seed"] == s])
        fa = FP.loo_regression(t, ["frozen_acc"])
        allf = FP.loo_regression(t, FP.FEATURES)
        out.append({"seed": int(s), "frozen_acc_R2": fa["r2"], "frozen_acc_spear": fa["spearman"],
                    "all_R2": allf["r2"], "all_spear": allf["spearman"]})
    return pd.DataFrame(out)


def exp4_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s in sorted(df[df["experiment"] == "exp4"]["seed"].unique()):
        td = ST.top_delta(df[df["seed"] == s], "intra_inter_ratio").dropna(subset=["improve", "gap"])
        if len(td) > 3:
            out.append({"seed": int(s), "n": len(td),
                        "improve_vs_gap_spearman": float(stats.spearmanr(td["improve"], td["gap"])[0])})
    return pd.DataFrame(out)


def _summ(df: pd.DataFrame, drop=("seed", "n")) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in drop]
    return df[cols].agg(["mean", "std"]).T.round(3)


def _bar_with_err(u: pd.DataFrame, out_path: str) -> None:
    feats = [c for c in u.columns if c != "seed"]
    mean = u[feats].mean()
    std = u[feats].std().fillna(0.0)
    order = mean.abs().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(order)), mean[order], yerr=std[order], capsize=4, color="tab:blue")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Spearman(feature, gap)")
    ax.set_title(f"Exp3: frozen-feature → gap, mean ± std across seeds {sorted(u['seed'])}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    df = load()
    seeds = sorted(int(s) for s in df["seed"].unique())
    L = [f"Combined across-seeds report — seeds present: {seeds}", "=" * 60, ""]

    u = exp3_univariate_per_seed(df)
    L += ["### Exp3 — univariate Spearman(feature, gap) per seed", u.round(3).to_string(index=False), "",
          "mean ± std across seeds:", _summ(u).to_string(), ""]

    lodo = exp3_lodo_per_seed(df)
    L += ["### Exp3 — leave-one-dataset-out per seed", lodo.round(3).to_string(index=False), "",
          "mean ± std:", _summ(lodo).to_string(), ""]

    p1 = exp1_partial_per_seed(df)
    L += ["### Exp1 — partial(shift, gap | frozen_acc) per seed", p1.round(3).to_string(index=False), "",
          "mean ± std:", _summ(p1).to_string(), ""]

    e5 = exp4_per_seed(df)
    L += ["### Exp4 — top-layer intra_inter improvement ↔ gap per seed", e5.round(3).to_string(index=False), "",
          "mean ± std:", _summ(e5).to_string(), ""]

    txt = "\n".join(L)
    with open("outputs/report_allseeds.txt", "w") as f:
        f.write(txt + "\n")
    fig_path = "outputs/figures/exp3_feature_spearman_allseeds.png"
    _bar_with_err(u, fig_path)

    print(txt)
    print(f"\nwrote -> outputs/report_allseeds.txt\nfigure -> {fig_path}")


if __name__ == "__main__":
    main()
