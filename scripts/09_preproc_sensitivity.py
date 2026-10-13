#!/usr/bin/env python3
"""
09_preproc_sensitivity.py -- how much of the linear / neural failure in the cold protocols
is a preprocessing artifact rather than a property of the split?

This script exists because `04_uncertainty.py`'s consistency assert caught `03_models.py`
disagreeing with itself (davis/scaffold/ridge: 0.8409 stored vs 1.4973 recomputed), and the
cause turned out to be two successive versions of the training-only z-scoring guard in
`03_models.standardize()`:

  floor_1e-6   the original bug: divide by max(sd, 1e-6). A fingerprint bit that is absent
               from every training ligand but present in a test ligand becomes z ~ 1e6.
  keep_clip10  the first fix: leave train-constant columns at scale 1 and clip the rest to
               +-10. It bounds the extrapolation but does not remove it -- 59 clipped
               columns still add up to a ~590-unit mean shift on a test row.
  drop_clip10  the shipped guard: DROP columns that are constant (sd < MIN_TRAIN_SD) on the
               training rows, then z-score and clip what survives.

All three are computed from training rows only, so none of them leaks test information; they
differ only in what they do with a feature the training set never gave a scale to. A column
with no training variance carries no training information for a linear or neural model, so
dropping it is the ordinary choice -- and it is also the choice that decides whether Ridge
looks like a model that loses to the null on `scaffold` (0.8522) or one that beats it.

Usage
  python scripts/09_preproc_sensitivity.py                    # ridge + xgboost, both datasets,
                                                              # + MLP on DAVIS only
  python scripts/09_preproc_sensitivity.py --mlp-datasets all --datasets davis

The committed results/preproc_sensitivity.csv is this call (60 fits = DAVIS x 5 protocols x
4 models x 3 guards, seed 0 -- seed 0 because 03 stores per-seed rows and the shipped guard
has to reproduce them exactly):

  python scripts/09_preproc_sensitivity.py --datasets davis \
      --models ridge,xgboost,randomforest,mlp
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dti_core import Featuriser, evaluate                    # noqa: E402

_spec = importlib.util.spec_from_file_location("m09_03", SCRIPTS / "03_models.py")
m03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m03)

PROC = ROOT / "data" / "processed"
SPL = PROC / "splits"
RESULTS = ROOT / "results"
GUARDS = ["floor_1e-6", "keep_clip10", "drop_clip10"]


def standardize_guard(Xtr, Xv, Xte, guard):
    """The three historical guards, side by side. Train statistics only, always."""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    if guard == "floor_1e-6":                                  # the bug
        scale = np.maximum(sd, 1e-6)
        keep = np.ones(len(sd), bool)
        z = lambda X: (X[:, keep] - mu[keep]) / scale[keep]    # noqa: E731
    elif guard == "keep_clip10":                               # the incomplete fix
        scale = np.where(sd < m03.MIN_TRAIN_SD, 1.0, sd)
        keep = np.ones(len(sd), bool)
        z = lambda X: np.clip((X - mu) / scale, -m03.Z_CLIP, m03.Z_CLIP)  # noqa: E731
    elif guard == "drop_clip10":                               # shipped
        keep = sd >= m03.MIN_TRAIN_SD
        z = lambda X: np.clip((X[:, keep] - mu[keep]) / sd[keep],  # noqa: E731
                              -m03.Z_CLIP, m03.Z_CLIP)
    else:
        raise ValueError(guard)
    return z(Xtr), z(Xv), z(Xte), int((~keep).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="davis,kiba")
    ap.add_argument("--protocols", default="random,scaffold,cold_drug,cold_target,temporal")
    ap.add_argument("--models", default="ridge,xgboost,mlp",
                    help="trees are the invariance control: they should barely move")
    ap.add_argument("--mlp-datasets", default="davis",
                    help="fit the MLP only for these datasets (a KIBA MLP fit x 3 guards is "
                         "~15 min and DAVIS is where the guard changed the conclusion)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = [x for x in args.models.split(",") if x != "mlp"]
    if "mlp" in args.models.split(","):
        models += (["mlp"] if args.mlp_datasets == "all" else
                   [f"mlp@{d.strip()}" for d in args.mlp_datasets.split(",")])
    f = Featuriser()
    # The shipped grid, so the ablation can prove that `drop_clip10` is the guard that
    # produced it: recomputing that arm must land on 03's stored RMSE, to 1e-5.
    ref = (pd.read_csv(RESULTS / "metrics.csv")
           if (RESULTS / "metrics.csv").exists() else pd.DataFrame())
    rows = []
    for dataset in args.datasets.split(","):
        ycol = "pkd" if dataset == "davis" else "kiba_score"
        base = pd.read_csv(PROC / f"{dataset}.csv")[["pair_id", "ligand_id", "target_id", ycol]]
        base = base.set_index("pair_id")
        for protocol in args.protocols.split(","):
            sp_df = pd.read_csv(SPL / f"{dataset}_{protocol}.csv")
            y = base.loc[sp_df.pair_id, ycol].to_numpy(float)
            lig, tgt = sp_df.ligand_id.to_numpy(), sp_df.target_id.to_numpy()
            m = (sp_df.split == "train").to_numpy()
            v = (sp_df.split == "valid").to_numpy()
            t = (sp_df.split == "test").to_numpy()
            null = float(np.sqrt(np.mean((y[t] - y[m].mean()) ** 2)))
            for model in models:
                if "@" in model:                       # `mlp@davis` = fit the MLP on DAVIS only
                    name, only = model.split("@", 1)
                    if only != dataset:
                        continue
                else:
                    name = model
                view = m03.MODELS_VIEWS[name]
                if view is None:
                    continue
                Xall = f.blocks(lig, tgt, view)
                for guard in GUARDS:
                    t0 = time.time()
                    Xtr, Xv, Xte, n_drop = standardize_guard(Xall[m], Xall[v], Xall[t], guard)
                    pred, info = m03.fit_predict(view, name, Xtr, y[m], Xte, args.seed,
                                                 valid=(Xv, y[v]))
                    ev = evaluate(y[t], pred)
                    # The shipped guard must reproduce the shipped grid, or this whole
                    # ablation is describing a model nobody ran.
                    r03 = ok03 = None
                    if guard == "drop_clip10" and len(ref):
                        sel = ref[(ref.dataset == dataset) & (ref.protocol == protocol)
                                  & (ref.model == name) & (ref.seed == args.seed)]
                        if len(sel):
                            r03 = float(sel.iloc[0].rmse)
                            ok03 = bool(abs(r03 - ev["rmse"]) < 1e-5)
                            if not ok03:
                                print(f"  !! shipped guard disagrees with 03 on "
                                      f"{dataset}/{protocol}/{name}: {ev['rmse']:.6f} vs "
                                      f"{r03:.6f} stored")
                    rows.append({"dataset": dataset, "protocol": protocol, "model": name,
                                 "guard": guard, "seed": args.seed, "view": view,
                                 "n_cols_built": int(Xall.shape[1]), "n_cols_dropped": n_drop,
                                 "n_cols_used": int(Xtr.shape[1]),
                                 "absz_max_test": round(float(np.abs(Xte).max()), 4),
                                 "null_rmse": round(null, 4),
                                 "rmse": round(ev["rmse"], 4),
                                 "rmse_03_stored": None if r03 is None else round(r03, 4),
                                 "shipped_guard_matches_03": ok03,
                                 "ci_harrell": round(ev["ci_harrell"], 4),
                                 "pearson_r": round(ev["pearson_r"], 4),
                                 "beats_null": bool(ev["rmse"] < null),
                                 "alpha": info.get("alpha"),
                                 "seconds": round(time.time() - t0, 1)})
                    print(f"{dataset:<6}{protocol:<12}{name:<11}{guard:<13}"
                          f"RMSE {ev['rmse']:.4f} null {null:.4f} "
                          f"{'BEATS' if ev['rmse'] < null else 'loses'} "
                          f"|z|max {np.abs(Xte).max():.4g}"
                          + ("" if ok03 is None else
                             ("  matches_03" if ok03 else "  DIFFERS_FROM_03")), flush=True)

    if not rows:
        sys.exit("nothing computed")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "preproc_sensitivity.csv", index=False)

    # headline: the guard decides whether a model looks better or worse than the null
    flips = []
    for (ds, pr, md), sub in df.groupby(["dataset", "protocol", "model"]):
        p = sub.set_index("guard").rmse
        nl = float(sub.null_rmse.iloc[0])
        if len(p) == 3 and (p["drop_clip10"] < nl) != (p["keep_clip10"] < nl):
            flips.append({"dataset": ds, "protocol": pr, "model": md,
                          "keep_clip10": round(float(p["keep_clip10"]), 4),
                          "drop_clip10": round(float(p["drop_clip10"]), 4),
                          "null": round(nl, 4)})
    tree = df[df.model.isin(["xgboost", "randomforest"])]
    r = df[df.model == "ridge"]
    tmove = (tree.groupby(["dataset", "protocol", "model"]).rmse.agg(lambda s: float(np.nanmax(s) - np.nanmin(s))))
    rng = (df.groupby(["dataset", "protocol", "model"]).rmse
             .agg(lambda s: float(np.nanmax(s) - np.nanmin(s)))).reset_index(name="range")
    worst = {m: rng[rng.model == m].sort_values("range", ascending=False).head(1)
                 .to_dict("records")[0]
             for m in ["ridge", "xgboost", "randomforest", "mlp"]
             if (df.model == m).any()}
    summary = {
        "script": "scripts/09_preproc_sensitivity.py",
        "guards": GUARDS,
        "n_rows": int(len(df)),
        "shipped_guard_reproduces_stored_grid": {
            "rows_checked": int(df["shipped_guard_matches_03"].notna().sum()),
            "rows_mismatched": int((df["shipped_guard_matches_03"] == False).sum())},  # noqa: E712
        "null_flips_caught_by_the_guard": flips,
        "worst_cell_per_model": {m: {"dataset": w["dataset"], "protocol": w["protocol"],
                                     "rmse_range": round(float(w["range"]), 4)}
                                 for m, w in worst.items()},
        "max_rmse_range_across_guards": {
            "ridge": round(float(r.groupby(["dataset", "protocol", "model"]).rmse
                                 .agg(lambda s: float(np.nanmax(s) - np.nanmin(s))).max()), 4),
            "trees": round(float(tmove.max()), 4) if len(tmove) else None,
            "mlp": (round(float(df[df.model == "mlp"].groupby(["dataset", "protocol"])
                               .rmse.agg(lambda s: float(np.nanmax(s) - np.nanmin(s))).max()), 4)
                    if (df.model == "mlp").any() else None)},
        "reading": ("trees barely move (a split threshold does not care whether an unseen "
                    "column is at 1e6, 10 or absent), while the linear and neural arms change "
                    "by up to the range above -- so their cold-protocol numbers are a "
                    "preprocessing decision as much as a property of the split"),
    }
    (RESULTS / "preproc_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + df.pivot_table(index=["dataset", "protocol", "model"], columns="guard",
                                values="rmse").round(4).to_string())
    print("\nwrote", RESULTS / "preproc_sensitivity.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
