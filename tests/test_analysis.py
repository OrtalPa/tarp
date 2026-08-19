"""Pure-stats tests for the Exp2/Exp3/Exp4/Exp5 analysis layers (no accelerator, no I/O)."""

import numpy as np
import pandas as pd

from tarp.analysis import frozen_predictor, infogain, layerwise, structure
from tarp.results.store import COLUMNS


def _frame(rows):
    df = pd.DataFrame([{**{c: None for c in COLUMNS}, "seed": 0, "pooling": "mean", **r} for r in rows])
    df = df[COLUMNS]
    df["layer"] = df["layer"].fillna(-1).astype(int)
    return df


def test_exp2_layer_curves_and_gap_pivot():
    rows = []
    for regime in ("lora_last3", "full"):
        for d in ("a", "b"):
            for L in range(3):
                rows.append(dict(experiment="exp2", model="m", dataset=d, ft_regime=regime,
                                 condition="shift", layer=L, metric="cka", value=1.0 - 0.1 * L))
            rows.append(dict(experiment="exp2", model="m", dataset=d, ft_regime=regime,
                             condition="delta", layer=-1, metric="gap", value=0.2))
    df = _frame(rows)
    curves = layerwise.layer_curves(df)
    # 2 regimes x 3 layers = 6 rows, mean CKA decreasing with layer
    assert len(curves) == 6
    lora_l0 = curves[(curves.ft_regime == "lora_last3") & (curves.layer == 0)]["cka_mean"].iloc[0]
    lora_l2 = curves[(curves.ft_regime == "lora_last3") & (curves.layer == 2)]["cka_mean"].iloc[0]
    assert lora_l0 > lora_l2
    gaps = layerwise.gap_by_regime(df)
    assert {"lora_last3", "full"} <= set(gaps.columns)


def test_exp5_correlation_sign():
    rows = []
    for i, d in enumerate(["sst2", "agnews", "trec", "emotion", "banking77", "clinc150"]):
        tg = 0.05 * (i + 1)
        rows += [dict(experiment="exp5", model="m", dataset=d, ft_regime="lora_last3",
                      condition="delta", layer=-1, metric="target_gain", value=tg),
                 dict(experiment="exp5", model="m", dataset=d, ft_regime="lora_last3",
                      condition="delta", layer=-1, metric="generic_loss", value=0.5 * tg)]
    t = infogain.build_table(_frame(rows))
    corr = infogain.correlations(t)
    assert corr["n"] == 6 and corr["pearson"] > 0.9  # perfectly co-linear by construction


def test_exp3_frozen_only_beats_noise():
    rng = np.random.default_rng(0)
    rows = []
    for d in list("abcdef"):
        fa = float(rng.uniform(0.5, 0.9))
        gap = (1 - fa) * 0.6  # gap is a clean function of frozen_acc
        rows += [dict(experiment="exp3", model="m", dataset=d, ft_regime="lora_last3",
                      condition="delta", layer=-1, metric="gap", value=gap),
                 dict(experiment="exp3", model="m", dataset=d, ft_regime="lora_last3",
                      condition="frozen", layer=-1, metric="frozen_acc", value=fa)]
        for feat in ("silhouette", "intra_inter_ratio", "knn_acc", "anisotropy", "eff_dim"):
            rows.append(dict(experiment="exp3", model="m", dataset=d, ft_regime="lora_last3",
                             condition="frozen", layer=-1, metric=feat, value=float(rng.normal())))
    t = frozen_predictor.build_table(_frame(rows))
    assert "gap_headroom" in t.columns
    res = frozen_predictor.loo_regression(t, ["frozen_acc"])
    assert res["n"] == 6 and res["pearson"] > 0.9


def test_exp4_structure_curves_and_delta():
    # Plant: frozen intra_inter high & flat; FT lowers it at the top layer, more so when gap is big.
    rows = []
    for d, gap in [("emotion", 0.30), ("banking77", 0.00)]:
        rows.append(dict(experiment="exp4", model="m", dataset=d, ft_regime="lora_last3",
                         condition="delta", layer=-1, metric="gap", value=gap))
        for L in range(4):
            rows.append(dict(experiment="exp4", model="m", dataset=d, ft_regime="lora_last3",
                             condition="frozen", layer=L, metric="intra_inter_ratio", value=3.0))
            # FT improves (lowers) intra_inter only at the final layer, proportional to the gap
            ft_val = 3.0 - (gap * 5 if L == 3 else 0.0)
            rows.append(dict(experiment="exp4", model="m", dataset=d, ft_regime="lora_last3",
                             condition="ft", layer=L, metric="intra_inter_ratio", value=ft_val))
    df = _frame(rows)
    curves = structure.structure_curves(df)
    # frozen curve is flat at 3.0; ft final layer < ft early layer
    ft = curves[(curves.condition == "ft")].sort_values("layer")
    assert ft["mean"].iloc[-1] < ft["mean"].iloc[0]
    # improvement (frozen→ft, lower-is-better) must be positive and rank with the gap
    td = structure.top_delta(df, "intra_inter_ratio")
    assert td.loc[td.dataset == "emotion", "improve"].iloc[0] > \
           td.loc[td.dataset == "banking77", "improve"].iloc[0]
