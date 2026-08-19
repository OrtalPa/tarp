"""Matplotlib figures -> outputs/figures/ (headless Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tarp.config import RunConfig  # noqa: E402
from tarp.pipeline import cache  # noqa: E402


def plot_cka_layers(cfg: RunConfig, cka: list[float]):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(cka)), cka, marker="o")
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("CKA(frozen, fine-tuned)")
    ax.set_title(f"{cfg.dataset.key} / {cfg.model.key} ({cfg.pooling}) — per-layer shift")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    out = cache.figures_dir() / f"exp1_{cfg.dataset.key}_{cfg.model.key}_{cfg.pooling}_s{cfg.seed}_cka.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_layer_curves(curves: pd.DataFrame):
    """Exp2: per-layer CKA (mean ± CI over datasets), one line per (model, regime)."""
    fig, axes = plt.subplots(1, curves["model"].nunique(), figsize=(5.2 * curves["model"].nunique(), 4),
                             squeeze=False)
    for ax, (model, gm) in zip(axes[0], curves.groupby("model")):
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
    out = cache.figures_dir() / "exp2_layer_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_infogain(t: pd.DataFrame):
    """Exp5: target gain vs. generic loss, colored by domain."""
    domains = sorted(t["domain"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    colors = {d: cmap(i % 10) for i, d in enumerate(domains)}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for d in domains:
        g = t[t["domain"] == d]
        ax.scatter(g["target_gain"], g["generic_loss"], s=45, color=colors[d], label=d)
    for _, r in t.iterrows():
        ax.annotate(f"{r['dataset']}/{r['model']}", (r["target_gain"], r["generic_loss"]),
                    fontsize=6, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("target gain (adaptation gap)")
    ax.set_ylabel("generic loss (mean decodability drop on other tasks)")
    ax.set_title("Exp5: information gained on target vs. lost generically")
    ax.legend(title="domain", fontsize=8)
    ax.grid(alpha=0.3)
    out = cache.figures_dir() / "exp5_gain_vs_loss.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_frozen_predictor(t: pd.DataFrame, pred, features: list[str]):
    """Exp3: leave-one-dataset-out predicted vs. actual gap."""
    y = t["gap"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y, pred, s=45)
    for actual, p, name in zip(y, pred, t["dataset"] + "/" + t["model"]):
        ax.annotate(name, (actual, p), fontsize=6, alpha=0.6)
    lo = float(min(np.nanmin(y), np.nanmin(pred)))
    hi = float(max(np.nanmax(y), np.nanmax(pred)))
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8)
    ax.set_xlabel("actual gap")
    ax.set_ylabel("predicted gap (LOO, frozen features)")
    ax.set_title("Exp3: predicting the gap from frozen features")
    ax.grid(alpha=0.3)
    out = cache.figures_dir() / "exp3_predicted_vs_actual.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_embedding_projection(panels: list[dict], model: str, out_name: str):
    """Qualitative: 2D PCA of final-layer test reps, frozen (left) vs fine-tuned (right),
    one row per dataset, points colored by class. ``panels`` items carry precomputed 2D
    coords so this stays a pure figure function.

    Each panel: {dataset, label_names, labels, frozen_xy, ft_xy, frozen_ev, ft_ev}.
    """
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(panels), 2, figsize=(9, 4.2 * len(panels)), squeeze=False)
    for r, p in enumerate(panels):
        y = np.asarray(p["labels"])
        names = p["label_names"]
        for c, (xy, ev, state) in enumerate([
            (p["frozen_xy"], p["frozen_ev"], "frozen"),
            (p["ft_xy"], p["ft_ev"], "fine-tuned"),
        ]):
            ax = axes[r][c]
            for k in np.unique(y):
                m = y == k
                lbl = names[k] if names and k < len(names) else str(k)
                ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6,
                           color=cmap(int(k) % 10), label=lbl)
            ax.set_title(f"{p['dataset']} · {state}  (PCA var={ev:.0%})", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 1 and len(np.unique(y)) <= 10:
                ax.legend(fontsize=6, markerscale=1.5, loc="best")
    fig.suptitle(f"Final-layer representations ({model}): frozen vs fine-tuned (2D PCA)", y=1.001)
    out = cache.figures_dir() / out_name
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_structure_curves(curves: pd.DataFrame):
    """Exp4: per-layer structural metrics, frozen vs FT overlaid. Rows = metric, cols = model."""
    metrics = list(dict.fromkeys(curves["metric"]))  # preserve report order
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
                ax.plot(gc["layer"], gc["mean"], marker="o", ms=3,
                        color=colors.get(cond, None), label=cond)
                ax.fill_between(gc["layer"], gc["mean"] - gc["ci"], gc["mean"] + gc["ci"],
                                color=colors.get(cond, None), alpha=0.15)
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
    out = cache.figures_dir() / "exp4_structure_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_shift_vs_gain(t: pd.DataFrame):
    """Scatter of representation shift vs adaptation gap, colored by domain."""
    domains = sorted(t["domain"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {d: cmap(i % 10) for i, d in enumerate(domains)}

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for d in domains:
        g = t[t["domain"] == d]
        ax.scatter(g["shift"], g["gap"], s=45, color=colors[d], label=d)
    for _, r in t.iterrows():
        ax.annotate(f"{r['dataset']}/{r['model']}", (r["shift"], r["gap"]),
                    fontsize=6, alpha=0.6)
    ax.set_xlabel("representation shift  (1 − mean CKA over transformer layers)")
    ax.set_ylabel("adaptation gap  (ft_acc − frozen_acc)")
    ax.set_title("Exp1: representation shift vs adaptation gain")
    ax.legend(title="domain", fontsize=8)
    ax.grid(alpha=0.3)
    out = cache.figures_dir() / "exp1_shift_vs_gain.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
