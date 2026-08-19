"""Command-line entry point.

    uv run python -m tarp.cli run --experiment exp1 --models distilbert --datasets trec --seeds 0

Grids are the cartesian product of --models x --datasets x --seeds.
"""

from __future__ import annotations

import argparse

import tarp.experiments  # noqa: F401  (import triggers experiment registration)
from tarp.config import RunConfig
from tarp.registry import (
    DATASETS,
    MODELS,
    experiment_names,
    get_experiment,
    resolve_dataset,
    resolve_model,
)


def build_configs(models, datasets, seeds, pooling) -> list[RunConfig]:
    return [
        RunConfig(model=resolve_model(m), dataset=resolve_dataset(d), pooling=pooling, seed=s)
        for m in models
        for d in datasets
        for s in seeds
    ]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="tarp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment over a model x dataset x seed grid")
    r.add_argument("--experiment", required=True, help=f"one of {experiment_names()}")
    r.add_argument("--models", nargs="+", default=["distilbert"], help=f"any of {list(MODELS)}")
    r.add_argument("--datasets", nargs="+", default=["trec"], help=f"any of {list(DATASETS)}")
    r.add_argument("--seeds", nargs="+", type=int, default=[0])
    r.add_argument("--pooling", default="mean", choices=["mean", "cls"])

    rep = sub.add_parser("report", help="analyze cached results + regenerate figures (no recompute)")
    rep.add_argument("--experiment", required=True)
    rep.add_argument("--ft-regime", default="lora_last3")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        exp = get_experiment(args.experiment)()
        configs = build_configs(args.models, args.datasets, args.seeds, args.pooling)
        exp.run(configs)
    elif args.cmd == "report":
        _report(args.experiment, args.ft_regime)


def _report(experiment: str, ft_regime: str) -> None:
    from tarp import plots
    from tarp.analysis import frozen_predictor, infogain, layerwise, shift_gain, structure
    from tarp.results import load

    if experiment == "exp1":
        t = shift_gain.build_table(ft_regime=ft_regime)
        corr = shift_gain.correlations(t)
        print(shift_gain.format_report(t, corr))
        print(f"figure -> {plots.plot_shift_vs_gain(t)}")

    elif experiment == "exp2":
        df = load()
        curves = layerwise.layer_curves(df)
        gaps = layerwise.gap_by_regime(df)
        print(layerwise.format_report(curves, gaps))
        print(f"figure -> {plots.plot_layer_curves(curves)}")

    elif experiment == "exp3":
        t = frozen_predictor.build_table()
        res = frozen_predictor.analyze(t)
        print(frozen_predictor.format_report(t, res))
        feats = frozen_predictor.HEADLINE_FEATURES  # plot the model we recommend, not all of FEATURES
        pred = frozen_predictor._loo_predict(t, feats, "gap")
        print(f"figure -> {plots.plot_frozen_predictor(t, pred, feats)}")

    elif experiment == "exp4":
        df = load()
        curves = structure.structure_curves(df)
        deltas = {m: structure.top_delta(df, m) for m in ("intra_inter_ratio", "knn_acc", "silhouette")}
        print(structure.format_report(curves, deltas))
        print(f"figure -> {plots.plot_structure_curves(curves)}")

    elif experiment == "exp5":
        t = infogain.build_table()
        corr = infogain.correlations(t)
        print(infogain.format_report(t, corr))
        print(f"figure -> {plots.plot_infogain(t)}")

    else:
        raise SystemExit(f"no report implemented for '{experiment}' yet")


if __name__ == "__main__":
    main()
