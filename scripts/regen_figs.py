"""Regenerate the 7 report.md figures into outputs/figures/regenerated/.

Rebuilds every figure embedded in report.md with the post-renumber experiment-number
titles (Exp1/Exp3/Exp4/Exp5). On every scatter, domain is encoded as color and model as
marker shape (so each point shows both without per-point text labels, which previously
overlapped into unreadable clumps). The exp3 feature-vs-gap figure is reconstructed here
because no plotting function for it existed in the package. Reads the cached full grid
(seeds 0/11/42 x 4 models) from results.parquet — no recompute, no model forward passes.

    PYTHONPATH=. python scripts/regen_figs.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tarp.analysis import frozen_predictor as FP  # noqa: E402
from tarp.analysis import infogain, layerwise, shift_gain, structure  # noqa: E402
from tarp.results import load  # noqa: E402

OUT = Path("outputs/figures/regenerated")
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("outputs/figures")

# stable, high-contrast marker per model so shape encodes model (color still encodes domain)
MODEL_MARKERS = {"bert": "o", "distilbert": "s", "roberta": "^", "modernbert": "D"}


def _scatter_domain_model(ax, t, xcol, ycol):
    """Scatter with color=domain, marker=model; returns the two legends to place."""
    from matplotlib.lines import Line2D

    domains = sorted(t["domain"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    dcolor = {d: cmap(i % 10) for i, d in enumerate(domains)}
    models = [m for m in MODEL_MARKERS if m in set(t["model"])]
    for d in domains:
        for m in models:
            g = t[(t["domain"] == d) & (t["model"] == m)]
            if len(g):
                ax.scatter(g[xcol], g[ycol], s=48, color=dcolor[d],
                           marker=MODEL_MARKERS[m], edgecolors="white", linewidths=0.4)
    dom_handles = [Line2D([], [], marker="o", ls="", color=dcolor[d], label=d) for d in domains]
    mdl_handles = [Line2D([], [], marker=MODEL_MARKERS[m], ls="", color="0.35", label=m)
                   for m in models]
    return dom_handles, mdl_handles


def _add_two_legends(ax, dom_handles, mdl_handles, loc1="upper left", loc2="lower right"):
    leg1 = ax.legend(handles=dom_handles, title="domain", fontsize=8, loc=loc1)
    ax.add_artist(leg1)
    ax.legend(handles=mdl_handles, title="model", fontsize=8, loc=loc2)


def save(fig, name):
    out = OUT / name
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {out}")


def fig_exp1(df):
    t = shift_gain.build_table(df)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    dom, mdl = _scatter_domain_model(ax, t, "shift", "gap")
    ax.set_xlabel("representation shift  (1 − mean CKA over transformer layers)")
    ax.set_ylabel("adaptation gap  (ft_acc − frozen_acc)")
    ax.set_title("Exp1: representation shift vs adaptation gain")
    _add_two_legends(ax, dom, mdl, loc1="center right", loc2="lower right")
    ax.grid(alpha=0.3)
    save(fig, "exp1_shift_vs_gain_final3seed.png")


def fig_exp2(df):
    curves = layerwise.layer_curves(df)
    models = sorted(curves["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        gm = curves[curves["model"] == model]
        for regime, g in gm.groupby("ft_regime"):
            g = g.sort_values("layer")
            ax.plot(g["layer"], g["cka_mean"], marker="o", label=regime)
            ax.fill_between(g["layer"], g["cka_mean"] - g["ci"], g["cka_mean"] + g["ci"], alpha=0.2)
        ax.set_title(model)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_ylabel("CKA(frozen, FT)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(title="regime", fontsize=8)
    save(fig, "exp2_layer_curves.png")


def fig_exp3_feature(df):
    """Reconstructed: frozen intra/inter-class ratio vs adaptation gap, per (model, dataset, seed)."""
    from tarp.registry import dataset_domain

    t = FP.build_table(df).dropna(subset=["intra_inter_ratio", "gap"])
    t = t.assign(domain=t["dataset"].map(dataset_domain))
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    dom, mdl = _scatter_domain_model(ax, t, "intra_inter_ratio", "gap")
    ax.set_xlabel("frozen intra/inter-class distance ratio")
    ax.set_ylabel("adaptation gap  (ft_acc − frozen_acc)")
    ax.set_title("Exp3: frozen intra/inter-class ratio vs adaptation gap")
    _add_two_legends(ax, dom, mdl, loc1="upper right", loc2="lower right")
    ax.grid(alpha=0.3)
    save(fig, "exp3_feature_vs_gap_final3seed.png")


def fig_exp3_pred(df):
    from matplotlib.lines import Line2D
    from tarp.registry import dataset_domain

    t = FP.build_table(df)
    # the model the report headlines (frozen accuracy alone), not the full feature set —
    # plotting FEATURES here would show a model the text argues against.
    pred = FP._loo_predict(t, FP.HEADLINE_FEATURES, "gap")
    y = t["gap"].to_numpy(float)
    ok = ~(np.isnan(pred) | np.isnan(y))
    t = t[ok].reset_index(drop=True).assign(_pred=pred[ok], domain=lambda d: d["dataset"].map(dataset_domain))
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    dom, mdl = _scatter_domain_model(ax, t, "gap", "_pred")
    lo = float(min(y[ok].min(), pred[ok].min()))
    hi = float(max(y[ok].max(), pred[ok].max()))
    id_line = Line2D([lo, hi], [lo, hi], ls="--", color="grey", lw=0.8, label="perfect prediction")
    ax.add_line(id_line)
    ax.set_xlabel("actual gap")
    ax.set_ylabel("predicted gap (LOO, frozen-probe accuracy)")
    ax.set_title("Exp3: predicting the gap from frozen-probe accuracy")
    _add_two_legends(ax, dom + [id_line], mdl, loc1="upper left", loc2="lower right")
    ax.grid(alpha=0.3)
    save(fig, "exp3_predicted_vs_actual_final3seed.png")


def fig_exp4(df):
    curves = structure.structure_curves(df)
    metrics = list(dict.fromkeys(curves["metric"]))
    models = sorted(curves["model"].unique())
    colors = {"frozen": "tab:blue", "ft": "tab:orange"}
    fig, axes = plt.subplots(len(metrics), len(models),
                             figsize=(4.2 * len(models), 2.6 * len(metrics)), squeeze=False)
    for r, metric in enumerate(metrics):
        for c, model in enumerate(models):
            ax = axes[r][c]
            g = curves[(curves["metric"] == metric) & (curves["model"] == model)]
            for cond, gc in g.groupby("condition"):
                gc = gc.sort_values("layer")
                ax.plot(gc["layer"], gc["mean"], marker="o", ms=3, color=colors.get(cond), label=cond)
                ax.fill_between(gc["layer"], gc["mean"] - gc["ci"], gc["mean"] + gc["ci"],
                                color=colors.get(cond), alpha=0.15)
            if r == 0:
                ax.set_title(model)
            if c == 0:
                ax.set_ylabel(metric, fontsize=8)
            if r == len(metrics) - 1:
                ax.set_xlabel("layer (0 = embeddings)", fontsize=8)
            ax.grid(alpha=0.3)
            if r == 0 and c == len(models) - 1:
                ax.legend(fontsize=7)
    fig.suptitle("Exp4: layer-wise representation structure (frozen vs fine-tuned)", y=1.002)
    save(fig, "exp4_structure_curves_final3seed.png")


def fig_exp5(df):
    t = infogain.build_table(df)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    dom, mdl = _scatter_domain_model(ax, t, "target_gain", "generic_loss")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("target gain (adaptation gap)")
    ax.set_ylabel("generic loss (mean decodability drop on other tasks)")
    ax.set_title("Exp5: information gained on target vs. lost generically")
    _add_two_legends(ax, dom, mdl, loc1="upper left", loc2="lower right")
    ax.grid(alpha=0.3)
    save(fig, "exp5_gain_vs_loss.png")


def fig_emb():
    """No experiment number in title and no overlap issue; needs cached encoder reps to
    recompute, so copy the existing figure through unchanged."""
    src = SRC / "emb_roberta_pca.png"
    shutil.copy2(src, OUT / src.name)
    print(f"copied -> {OUT / src.name} (unchanged; no exp-number/overlap issue)")


def main():
    df = load()
    fig_exp1(df)
    fig_exp2(df)
    fig_exp3_feature(df)
    fig_exp3_pred(df)
    fig_exp4(df)
    fig_exp5(df)
    fig_emb()


if __name__ == "__main__":
    main()
