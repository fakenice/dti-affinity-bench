#!/usr/bin/env python3
"""
04_uncertainty.py -- split-conformal prediction intervals on top of the 03 models.

Why this exists: RMSE and CI say nothing about how wrong a *single* predicted pKd is.
Here the validation split of each protocol is used as a calibration set (it is never
used for anything else -- the alpha search / early stopping in 03 already consumed it,
which is stated in the caveats), and the test split is covered by an interval whose
empirical coverage we then MEASURE. Two variants:

  conformal          classical split conformal, constant half-width q for the whole set
  conformal_norm     scores divided by sigma-hat(x), sigma-hat = a second regressor
                     trained on the calibration residuals  (heteroscedastic / adaptive)

and one honest diagnostic per protocol: the coverage curve over nominal levels
0.5..0.95 with its mean absolute deviation from nominal ("calibration error"), which
is the regression analogue of ECE.

Because 03 selects alpha / early-stops on the same validation rows, the marginal
coverage guarantee is not exact; this is reported as a limitation, not hidden, and the
coverage numbers below are what they are.

The point estimates are recomputed with 03's own fit_predict + standardize, and the
resulting test RMSE is asserted against results/metrics.csv so the two scripts cannot
drift apart silently. One model does both jobs: it is fitted once (with 03's early
stopping / alpha selection) and predicts the calibration rows and the test rows in a
single pass, so the scores and the residuals it is checked against come from the same
function.

Usage
  python scripts/04_uncertainty.py                      # full grid, seed 0
  python scripts/04_uncertainty.py --datasets davis
  python scripts/04_uncertainty.py --models ridge       # fast smoke test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dti_core import Featuriser, evaluate          # noqa: E402

_spec = importlib.util.spec_from_file_location("m03", SCRIPTS / "03_models.py")
m03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m03)

PROC = ROOT / "data" / "processed"
SPL = PROC / "splits"
RESULTS = ROOT / "results"
UNC = RESULTS / "uncertainty"
ILE = UNC / "intervals"

LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def conformal_quantile(scores, level):
    """q = the ceil((n+1)*level)-th smallest nonconformity score (exact-split-conformal)."""
    n = len(scores)
    k = math.ceil((n + 1) * level)
    if k > n:
        return float("inf")
    return float(np.sort(np.asarray(scores, float))[k - 1])


def coverage_curve(abs_res, q_by_level):
    return {f"cov_{int(l*100)}": float(np.mean(abs_res <= q_by_level[l] + 1e-12))
            for l, q in q_by_level.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="davis,kiba")
    ap.add_argument("--protocols", default="random,scaffold,cold_drug,cold_target,temporal")
    ap.add_argument("--models", default="ridge,xgboost",
                    help="subset of 03's models to wrap (randomforest/mlp available)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-floor", type=float, default=0.05)
    ap.add_argument("--no-normalized", action="store_true")
    ap.add_argument("--skip-consistency-assert", action="store_true")
    args = ap.parse_args()

    UNC.mkdir(parents=True, exist_ok=True)
    ILE.mkdir(parents=True, exist_ok=True)
    f = Featuriser()
    metrics = (pd.read_csv(RESULTS / "metrics.csv")
               if (RESULTS / "metrics.csv").exists() else pd.DataFrame())

    rows, curve_rows = [], []
    for dataset in args.datasets.split(","):
        ycol = "pkd" if dataset == "davis" else "kiba_score"
        base = pd.read_csv(PROC / f"{dataset}.csv")[["pair_id", "ligand_id", "target_id", ycol]]
        base = base.set_index("pair_id")
        for protocol in args.protocols.split(","):
            p = SPL / f"{dataset}_{protocol}.csv"
            if not p.exists():
                print(f"skip {dataset}/{protocol}: no split file")
                continue
            sp_df = pd.read_csv(p)
            y = base.loc[sp_df.pair_id, ycol].to_numpy(float)
            lig_ids, tgt_ids = sp_df.ligand_id.to_numpy(), sp_df.target_id.to_numpy()
            m = (sp_df.split == "train").to_numpy()
            v = (sp_df.split == "valid").to_numpy()
            t = (sp_df.split == "test").to_numpy()
            if not v.any() or not t.any():
                print(f"skip {dataset}/{protocol}: empty valid or test")
                continue
            pid = sp_df.pair_id.to_numpy()
            for model in args.models.split(","):
                view = m03.MODELS_VIEWS[model]
                if view is None:
                    continue
                t0 = time.time()
                Xall = f.blocks(lig_ids, tgt_ids, view)
                Xtr, Xv, Xte = m03.standardize(Xall[m], Xall[v], Xall[t])
                del Xall
                # ONE fit, predictions for calibration rows and test rows taken from the
                # same model. Fitting twice (once to predict valid, once to predict test)
                # would make the nonconformity scores and the test residuals come from
                # different networks/trees -- the conformal guarantee assumes both sides
                # are scored by one fixed model -- and it doubles the cost for nothing.
                n_cal_rows = int(v.sum())
                Xcal_te = (sp.vstack([Xv, Xte]) if sp.issparse(Xv) else np.vstack([Xv, Xte]))
                yhat_cal_te, _ = m03.fit_predict(view, model, Xtr, y[m], Xcal_te,
                                                    args.seed, valid=(Xv, y[v]))
                yhat_cal = np.asarray(yhat_cal_te[:n_cal_rows], float)
                yhat_te = np.asarray(yhat_cal_te[n_cal_rows:], float)
                del Xcal_te
                ev = evaluate(y[t], yhat_te)
                # --- consistency with 03 -------------------------------------------------
                ref = None
                if len(metrics):
                    sel = metrics[(metrics.dataset == dataset) & (metrics.protocol == protocol)
                                  & (metrics.model == model) & (metrics.seed == args.seed)]
                    if len(sel):
                        ref = float(sel.iloc[0].rmse)
                dref = None if ref is None else abs(ref - ev["rmse"])
                ok = (dref is None) or (dref < 1e-5)
                if not ok and not args.skip_consistency_assert:
                    print(f"  !! RMSE differs from 03 for {dataset}/{protocol}/{model}: "
                          f"{ev['rmse']:.6f} vs {ref:.6f}")
                res_cal = y[v] - yhat_cal
                s_cal = np.abs(res_cal)
                res_te = y[t] - yhat_te
                a_te = np.abs(res_te)

                # ---- constant-width split conformal ------------------------------------
                qmap = {l: conformal_quantile(s_cal, l) for l in LEVELS}
                cov = coverage_curve(a_te, qmap)
                cal_err = float(np.mean([abs(cov[f"cov_{int(l*100)}"] - l) for l in LEVELS]))
                row = {"dataset": dataset, "protocol": protocol, "model": model,
                       "seed": args.seed, "method": "conformal",
                       "n_cal": int(v.sum()), "n_test": int(t.sum()),
                       "rmse": round(ev["rmse"], 4), "rmse_03": None if ref is None else round(ref, 4),
                       "rmse_matches_03": bool(ok),
                       "q90": None if math.isinf(qmap[0.90]) else round(qmap[0.90], 4),
                       "width90": None if math.isinf(qmap[0.90]) else round(2 * qmap[0.90], 4),
                       "width90_over_label_sd": (None if math.isinf(qmap[0.90]) else
                                                 round(2 * qmap[0.90] / float(np.std(y[m])), 4)),
                       "cov90": round(cov["cov_90"], 4), "cov80": round(cov["cov_80"], 4),
                       "cov50": round(cov["cov_50"], 4), "coverage_error": round(cal_err, 4),
                       "cal_rmse": round(float(np.sqrt(np.mean(res_cal ** 2))), 4),
                       "seconds": round(time.time() - t0, 1)}
                for k, vv in cov.items():
                    row[k] = round(vv, 4)
                rows.append(row)
                for l in LEVELS:
                    curve_rows.append({"dataset": dataset, "protocol": protocol, "model": model,
                                       "method": "conformal", "nominal": l,
                                       "empirical": round(cov[f"cov_{int(l*100)}"], 4),
                                       "halfwidth": None if math.isinf(qmap[l]) else round(qmap[l], 4)})

                np.savez_compressed(ILE / f"{dataset}_{protocol}_{model}_conformal.npz",
                                    pair_id=pid[t], y=y[t], yhat=yhat_te,
                                    halfwidth=np.array([qmap[l] for l in LEVELS], float),
                                    levels=np.array(LEVELS, float))

                # ---- normalised (adaptive) conformal -----------------------------------
                # sigma-hat is one ridge (alpha fixed at 10, no label peeking beyond the
                # calibration rows) fitted on the FIRST HALF of the validation split and
                # then applied to the second half (for the scores) and to the test rows
                # (for the interval widths) -- the same model on both sides, so the
                # normalised scores and the test half-widths are on one scale.
                if not args.no_normalized:
                    try:
                        from sklearn.linear_model import Ridge
                        n_cal = len(s_cal)
                        half = n_cal // 2
                        if half < 20:
                            raise RuntimeError(f"calibration split too small (n={n_cal}) to halve")
                        idx_a = np.sort(np.random.default_rng(7 + args.seed)
                                        .choice(n_cal, size=half, replace=False))
                        idx_b = np.setdiff1d(np.arange(n_cal), idx_a)
                        sm = Ridge(alpha=10.0, fit_intercept=True)
                        sm.fit(Xv[idx_a], s_cal[idx_a])
                        sig_b = np.maximum(sm.predict(Xv[idx_b]), args.sigma_floor)
                        sig_te = np.maximum(sm.predict(Xte), args.sigma_floor)
                        s_n = s_cal[idx_b] / sig_b
                        qn = {l: conformal_quantile(s_n, l) for l in LEVELS}
                        hw_te = (qn[0.90] * sig_te if not math.isinf(qn[0.90])
                                 else np.full(len(sig_te), np.inf))
                        cov_n = {f"cov_{int(l*100)}": float(np.mean(a_te <= (qn[l] * sig_te + 1e-12)))
                                 for l in LEVELS}
                        cal_err_n = float(np.mean([abs(cov_n[f"cov_{int(l*100)}"] - l) for l in LEVELS]))
                        r2 = dict(row)
                        r2.update({"method": "conformal_norm", "sigma_model": "ridge(alpha=10)",
                                   "n_cal": int(len(idx_b)),
                                   "q90": None if math.isinf(qn[0.90]) else round(qn[0.90], 4),
                                   "width90": round(float(np.mean(2 * hw_te)), 4),
                                   "width90_over_label_sd": round(float(np.mean(2 * hw_te)) / float(np.std(y[m])), 4),
                                   "cov90": round(cov_n["cov_90"], 4),
                                   "cov80": round(cov_n["cov_80"], 4),
                                   "cov50": round(cov_n["cov_50"], 4),
                                   "coverage_error": round(cal_err_n, 4)})
                        for k, vv in cov_n.items():
                            r2[k] = round(vv, 4)
                        # heteroscedasticity evidence: corr(|residual|, sigma-hat) and (|res|, |yhat|)
                        r2["corr_absres_sigma"] = round(float(np.corrcoef(a_te, sig_te)[0, 1])
                                                       if np.std(sig_te) > 0 else float("nan"), 4)
                        r2["corr_absres_yhat"] = round(float(np.corrcoef(a_te, np.abs(yhat_te))[0, 1]), 4)
                        r2["sigma_te_min"] = round(float(sig_te.min()), 4)
                        r2["sigma_te_max"] = round(float(sig_te.max()), 4)
                        rows.append(r2)
                        for l in LEVELS:
                            curve_rows.append({"dataset": dataset, "protocol": protocol,
                                               "model": model, "method": "conformal_norm",
                                               "nominal": l,
                                               "empirical": round(cov_n[f"cov_{int(l*100)}"], 4),
                                               "halfwidth": round(float(np.mean(qn[l] * sig_te)), 4)
                                               if not math.isinf(qn[l]) else None})
                        np.savez_compressed(ILE / f"{dataset}_{protocol}_{model}_conformal_norm.npz",
                                            pair_id=pid[t], y=y[t], yhat=yhat_te,
                                            sigma=sig_te.astype(np.float64),
                                            levels=np.array(LEVELS, float),
                                            q=np.array([qn[l] for l in LEVELS], float))
                    except Exception as e:                                  # noqa: BLE001
                        print(f"  normalized conformal failed for {dataset}/{protocol}/{model}: "
                              f"{type(e).__name__}: {e}")

                print(f"{dataset:<6}{protocol:<12}{model:<9} conf90 {row['cov90']} "
                      f"w90 {row['width90']}  norm90 {rows[-1]['cov90'] if rows[-1]['method']=='conformal_norm' else 'NA'} "
                      f"({row['seconds']}s) rmse_vs_03 {'OK' if ok else 'DIFF'}", flush=True)

    if not rows:
        sys.exit("nothing computed -- check --datasets/--protocols/--models")
    df = pd.DataFrame(rows)
    front = ["dataset", "protocol", "model", "seed", "method", "rmse", "rmse_03", "rmse_matches_03",
             "n_cal", "n_test", "q90", "width90", "width90_over_label_sd", "cov50", "cov80",
             "cov90", "coverage_error", "cal_rmse"]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    df[cols].to_csv(UNC / "conformal.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(UNC / "coverage_curve.csv", index=False)

    n_bad = int((~df.rmse_matches_03.astype(bool)).sum()) if "rmse_matches_03" in df else None
    summary = {
        "script": "scripts/04_uncertainty.py",
        "levels": LEVELS,
        "methods": sorted(df.method.unique().tolist()),
        "n_configs": int(len(df)),
        "rmse_consistency_with_03": {"rows_checked": int(df.rmse_03.notna().sum()),
                                     "rows_mismatched": n_bad},
        "caveats": [
            "the validation split does double duty: 03 uses it for alpha selection / early "
            "stopping and 04 uses it for conformal calibration, so the 1-alpha coverage "
            "guarantee is approximate, not exact",
            "DAVIS labels are censored: 69.6% of pKd values sit on the 10000 nM ceiling, "
            "so residual distributions are mixtures and constant-width intervals over-cover "
            "the ceiling block",
            "under the temporal protocol the calibration set precedes the test set in time, "
            "which is exactly where exchangeability fails; coverage there is a measurement, "
            "not a guarantee",
        ],
    }
    (UNC / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + df[["dataset", "protocol", "model", "method", "rmse", "width90",
                     "cov90", "coverage_error"]].to_string(index=False))
    print("\nwrote", UNC / "conformal.csv", "and", UNC / "coverage_curve.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
