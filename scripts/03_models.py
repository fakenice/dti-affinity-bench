#!/usr/bin/env python3
"""
03_models.py -- baselines under every protocol, with null-model floors and seed error bars.

Models (all classical / shallow on purpose: the question is which protocol measures what,
not who has the biggest network)

  mean        predict the training mean                                (null floor)
  ridge       closed-form L2 on the compact dense block                (693 dims)
  xgboost     gradient boosting on the compact dense block             (693 dims)
  randomforest  bagged trees on the compact dense block                (693 dims)
  mlp         DeepDTA-style 3-layer MLP on the dense block             (949 dims)

Dense, not sparse, is the default view for Ridge (`MODELS_VIEWS`), and the dims are those
of the *built* blocks -- `standardize()` additionally drops the columns that are constant
on the training rows, so the width a model actually sees is a few dozen dims smaller and
protocol-dependent.

Features are the concatenation of a ligand block and a protein block, exactly as in the
sequence-based DTI literature (ECFP4 for the compound, composition vectors for the
protein). Two matrix views are built from the same per-entity tables:
  sparse_full = log1p(ECFP4-2048 counts) ++ [13 descriptors, 20 AAC, 400 DPC] (z-scored
                on training rows only)
  dense_compact = log1p(the 256 ECFP4 bits with the highest training-set frequency) ++
                the same 433 columns; dense_mlp adds the 512-bit binary ECFP4 instead.
  Bit selection is unsupervised (frequency of the bit among ligands), never label-based.

Metrics, reported for every (dataset, protocol, model, seed) row
  RMSE, MAE, Pearson r, Spearman rho,
  CI  = concordance index counting tied labels as 0.5  (the field's convention),
  CI_strict = concordance index over pairs with *unequal* labels only.
Both are printed because DAVIS puts 69.6% of its labels on the censoring ceiling, so the
tied-label fraction is large and the two calibers disagree -- a real reporting hazard.

Usage
  python scripts/03_models.py --quick     # DAVIS, ridge+xgboost, 1 seed (smoke test)
  python scripts/03_models.py             # full grid
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dti_core import Featuriser, evaluate, selftest   # shared with 04 and the notebook

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SPL = PROC / "splits"
RESULTS = ROOT / "results"
PRED = RESULTS / "predictions"

N_TOP_BITS = 256
SEEDS_FULL = [0, 1, 2]
SEEDS_ONE = [0]
MIN_TRAIN_SD = 1e-3   # below this a training column has no usable scale (see standardize)
Z_CLIP = 10.0         # bound on |z| after standardising


# --------------------------------------------------------------------------- models

def standardize(Xtr, Xv, Xte):
    """z-score with TRAINING rows only -- valid and test never contribute statistics.

    Two guards, both needed because the feature blocks are sparse count/binary columns:

    * a column that is **constant on the training rows carries no training information**
      for a linear or neural model, and its test value is pure out-of-support
      extrapolation. Such columns are therefore *dropped* (`keep` mask from train only).
      Dividing by sd instead (or by a 1e-6 floor) turns "this bit is absent from every
      training ligand but present in a test ligand" into |z| ~ 1e6 -- measured 2e6 on the
      DAVIS scaffold / cold-drug / temporal test blocks.
    * the z values of the surviving columns are clipped to +-Z_CLIP, so a rare-but-not
      constant bit cannot raise a feature to an arbitrary number either.

    Dropping is the part that actually matters, and it is measured, not assumed: for
    Ridge on DAVIS `scaffold`, leaving the 59 train-constant columns in and clipping them
    to +-10 still gives test RMSE **1.4973** -- statistically indistinguishable from the
    broken 1e-6-floor run (1.4968) -- whereas dropping them gives **0.8786** against a
    0.8522 global-mean null. 59 columns at +10 each is a 590-unit mean shift; a clip bounds
    an extrapolation, it does not remove it. Tree models are invariant to this guard
    (monotone transforms of a column do not change split thresholds), only linear and
    neural ones are exposed, which is why the effect is checked on Ridge and the MLP.

    Both are computed from the training rows only, so neither can leak test information.
    The sparse_full override is passed through unstandardised (see MODELS_VIEWS note).
    """
    if sp.issparse(Xtr):
        return Xtr, Xv, Xte
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    keep = sd >= MIN_TRAIN_SD
    z = lambda X: np.clip((X[:, keep] - mu[keep]) / sd[keep], -Z_CLIP, Z_CLIP)
    return z(Xtr), z(Xv), z(Xte)


def fit_predict(view, model, Xtr, ytr, Xte, seed, valid=None):
    t0 = time.time()
    if model == "mean":
        return np.full(len(Xte), float(np.mean(ytr))), {"null_model": "global train mean"}
    if model == "ridge":
        best, info = None, {}
        for alpha in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
            m = Ridge(alpha=alpha, fit_intercept=True)
            m.fit(Xtr, ytr)
            rmse_v = float(np.sqrt(np.mean((valid[1] - m.predict(valid[0])) ** 2))) if valid else float("nan")
            if best is None or rmse_v < best[0]:
                best, info = (rmse_v, alpha, m), {"alpha": alpha, "cv_rmse_on_valid": rmse_v}
        return best[2].predict(Xte), info
    if model == "xgboost":
        m = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                         random_state=seed, n_jobs=-1, tree_method="hist",
                         early_stopping_rounds=40 if valid is not None else None)
        if valid is not None:
            m.fit(Xtr, ytr, eval_set=[(valid[0], valid[1])], verbose=False)
        else:
            m.fit(Xtr, ytr)
        return m.predict(Xte), {"best_iteration": int(getattr(m, "best_iteration", -1))}
    if model == "randomforest":
        m = RandomForestRegressor(n_estimators=150, max_depth=None, min_samples_leaf=20,
                                  max_features=0.3, n_jobs=-1, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict(Xte), {"oob_free": 1}
    if model == "mlp":
        return _mlp(view, Xtr, ytr, Xte, seed, valid)
    raise ValueError(model)


def _mlp(view, Xtr, ytr, Xte, seed, valid):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    # Xtr / Xte / valid arrive already standardised by standardize(). Rescaling them a
    # second time here would undo that function's guards, so the tensors are built direct.
    Xt = torch.tensor(np.asarray(Xtr, np.float32))
    yt = torch.tensor(np.asarray(ytr, np.float32)).view(-1, 1)
    Xv = torch.tensor(np.asarray(Xte, np.float32))
    Xval = torch.tensor(np.asarray(valid[0], np.float32)) if valid is not None else None
    yval = np.asarray(valid[1], np.float32) if valid is not None else None
    n = Xt.shape[0]
    d = Xt.shape[1]
    net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.25),
                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.25),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    bs = 512
    best_val, best_state, epochs_no_gain = float("inf"), None, 0
    for ep in range(25):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        if Xval is not None:
            with torch.no_grad():
                pv = net(Xval).numpy().ravel()
            rv = float(np.sqrt(np.mean((yval - pv) ** 2)))
            if rv < best_val - 1e-4:
                best_val, epochs_no_gain = rv, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                epochs_no_gain += 1
                if epochs_no_gain >= 5:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        out = net(Xv).numpy().ravel()
    return out, {"fit_seconds": round(time.time() - t0, 1),
                 "valid_rmse": None if Xval is None else round(best_val, 4)}


# --------------------------------------------------------------------------- main

# One matrix view per model. `dense_compact` (693 dims: 256 most-frequent ECFP4 count
# bits + 13 ligand descriptors + 424 protein composition columns) is used for the linear
# and tree models because z-scoring a 2,485-column sparse block would materialise ~1.2 GB
# of float32 for KIBA; `sparse_full` stays available through --views for anyone who wants
# the full-fingerprint variant (it is then left unstandardised, which the alpha search
# partly absorbs -- stated, not hidden).
MODELS_VIEWS = {
    "mean": None,
    "ridge": "dense_compact",
    "xgboost": "dense_compact",
    "randomforest": "dense_compact",
    "mlp": "dense_mlp",
}
VIEWS_OVERRIDE = {"sparse_full": "sparse_full"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--protocols", default=None,
                    help="comma-separated subset of random,scaffold,cold_drug,cold_target,temporal")
    ap.add_argument("--ridge-view", default="dense_compact",
                    choices=["dense_compact", "sparse_full"],
                    help="feature block for the linear model (see MODELS_VIEWS)")
    ap.add_argument("--one-seed", action="store_true",
                    help="single seed -- whole-grid smoke test without the 3x cost")
    ap.add_argument("--resume", action="store_true",
                    help="keep the existing metrics.csv and skip (dataset, protocol, model, seed) "
                         "rows that are already present -- the grid is ~70 min and a crash "
                         "half-way must not throw it away")
    args = ap.parse_args()

    PRED.mkdir(parents=True, exist_ok=True)
    st = selftest()
    print("concordance selftest, fast vs brute force (abs diff):",
          json.dumps({k: (v.get("abs_diff", v) if isinstance(v, dict) else v)
                         for k, v in st.items()}, default=float))
    f = Featuriser()
    print("feature block dims:", f.dims)
    datasets = (args.datasets.split(",") if args.datasets
                else (["davis"] if args.quick else ["davis", "kiba"]))
    protocols = (args.protocols.split(",") if args.protocols
                 else ["random", "scaffold", "cold_drug", "cold_target", "temporal"])
    models = (args.models.split(",") if args.models
              else (["mean", "ridge", "xgboost"] if args.quick
                    else ["mean", "ridge", "xgboost", "randomforest", "mlp"]))
    seeds = SEEDS_ONE if args.quick else SEEDS_FULL
    if args.one_seed:
        seeds = SEEDS_ONE
    MODELS_VIEWS["ridge"] = args.ridge_view

    OUTCSV = RESULTS / ("metrics_quick.csv" if args.quick else "metrics.csv")
    rows = []
    done = set()
    if args.resume and OUTCSV.exists():
        prev = pd.read_csv(OUTCSV)
        rows = prev.to_dict("records")
        done = {(r.dataset, r.protocol, r.model, int(r.seed)) for r in prev.itertuples()}
        print(f"resume: {len(done)} rows kept from {OUTCSV.name}")

    def dump():
        """Rewrite the metrics table after every row so a crash cannot lose the grid."""
        df = pd.DataFrame(rows)
        cols = ["dataset", "protocol", "model", "seed", "n_test", "rmse", "mae", "pearson_r",
                "spearman_rho", "ci_harrell", "ci_sklearn_like", "label_tie_fraction", "seconds"] + \
               [c for c in df.columns if c.startswith("info_") or c in
                ("rmse_over_train_sd", "error")]
        df = df[[c for c in dict.fromkeys(cols) if c in df.columns]]
        df.to_csv(OUTCSV, index=False)
        return df
    for dataset in datasets:
        ycol = "pkd" if dataset == "davis" else "kiba_score"
        base = pd.read_csv(PROC / f"{dataset}.csv")[["pair_id", "ligand_id", "target_id", ycol]]
        base = base.set_index("pair_id")
        for protocol in protocols:
            sp_df = pd.read_csv(SPL / f"{dataset}_{protocol}.csv")
            y = base.loc[sp_df.pair_id, ycol].to_numpy(float)
            lig_ids = sp_df.ligand_id.to_numpy()
            tgt_ids = sp_df.target_id.to_numpy()
            m = (sp_df.split == "train").to_numpy()
            v = (sp_df.split == "valid").to_numpy()
            t = (sp_df.split == "test").to_numpy()
            if not t.any():
                print(f"skip {dataset}/{protocol}: empty test")
                continue
            ytr = y[m]
            # ---- entity-level null baselines (train means, no learning) ----
            def entity_mean(ids_tr, y_tr, ids_te):
                s = pd.Series(y_tr, index=ids_tr)
                g = s.groupby(level=0).mean()
                grand = float(s.mean())
                return np.array([g[i] if i in g.index else grand for i in ids_te], dtype=float)

            for name, pred in (
                    ("mean_ligand_train", entity_mean(lig_ids[m], ytr, lig_ids[t])),
                    ("mean_target_train", entity_mean(tgt_ids[m], ytr, tgt_ids[t])),
            ):
                if (dataset, protocol, name, 0) in done:
                    continue
                row = {"dataset": dataset, "protocol": protocol, "model": name, "seed": 0}
                row.update(evaluate(y[t], pred))
                row["seconds"] = 0.0
                rows.append(row)
                dump()
                np.savez_compressed(PRED / f"{dataset}_{protocol}_{name}_seed0.npz",
                                    pair_id=sp_df.pair_id.to_numpy()[t],
                                    y=y[t].astype(np.float32), yhat=pred.astype(np.float32))
                print(f"{dataset:<6}{protocol:<12}{name:<13}         "
                      f"RMSE {row['rmse']:.4f} CI {row['ci_harrell']:.4f}", flush=True)

            for model in models:
                view = MODELS_VIEWS[model]
                if view is None:
                    Xtr = np.zeros((int(m.sum()), 1))
                    Xv = np.zeros((int(v.sum()), 1))
                    Xte = np.zeros((int(t.sum()), 1))
                else:
                    Xall = f.blocks(lig_ids, tgt_ids, view)
                    Xtr, Xv, Xte = standardize(Xall[m], Xall[v], Xall[t])
                    del Xall
                for seed in (seeds if model != "mean" else [0]):
                    if (dataset, protocol, model, seed) in done:
                        continue
                    t0 = time.time()
                    srow = {"dataset": dataset, "protocol": protocol, "model": model, "seed": seed}
                    if sp.issparse(Xtr) and model in ("xgboost", "randomforest", "mlp"):
                        print("  skip", model, "under a sparse view")
                        continue
                    valid = (Xv, y[v]) if model in ("ridge", "xgboost", "mlp") else None
                    try:
                        pred, info = fit_predict(view, model, Xtr, ytr, Xte, seed, valid)
                        srow.update(evaluate(y[t], pred))
                        srow["seconds"] = round(time.time() - t0, 1)
                        srow.update({f"info_{k}": vv for k, vv in info.items()})
                        if model == "mean":
                            srow["rmse_over_train_sd"] = round(srow["rmse"] / float(np.std(ytr)), 4)
                    except Exception as e:                                  # noqa: BLE001
                        srow["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                        srow["seconds"] = round(time.time() - t0, 1)
                        srow.update({k: float("nan") for k in
                                     ("n_test", "rmse", "mae", "pearson_r", "spearman_rho",
                                      "ci_harrell", "ci_sklearn_like", "label_tie_fraction")})
                        print(f"  !! {dataset}/{protocol}/{model}/seed{seed} failed: {srow['error']}",
                              flush=True)
                    tag = f"{dataset}_{protocol}_{model}_seed{seed}"
                    if "error" not in srow:
                        np.savez_compressed(PRED / f"{tag}.npz",
                                            pair_id=sp_df.pair_id.to_numpy()[t],
                                            y=y[t].astype(np.float32),
                                            yhat=pred.astype(np.float32))
                    rows.append(srow)
                    dump()
                    if "error" not in srow:
                        print(f"{dataset:<6}{protocol:<12}{model:<13}seed{seed} "
                              f"RMSE {srow['rmse']:.4f} CI {srow['ci_harrell']:.4f} "
                              f"CI_sk {srow['ci_sklearn_like']:.4f} r {srow['pearson_r']:.4f} "
                              f"({srow['seconds']}s)", flush=True)

    df = dump()      # same column set as every intermediate write, incl. error/rmse_over_train_sd

    if not args.quick:
        agg = (df.groupby(["dataset", "protocol", "model"])
                 .agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                      ci=("ci_harrell", "mean"), ci_sklearn_like=("ci_sklearn_like", "mean"),
                      pearson_r=("pearson_r", "mean"),
                      spearman_rho=("spearman_rho", "mean"), n_seeds=("seed", "nunique"))
                 .round(4).reset_index())
        agg.to_csv(RESULTS / "metrics_by_protocol.csv", index=False)
        print("\n" + agg.to_string(index=False))
    print("\nwrote", RESULTS / ("metrics_quick.csv" if args.quick else "metrics.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
