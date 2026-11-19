#!/usr/bin/env python3
"""
17_tobit_baseline.py -- a censored-likelihood (Tobit) baseline, as promised by README S2 note 4.

The benchmark's own data summary records the problem: **69.64 % of DAVIS rows sit exactly at
the assay floor pKd = 5.0** (a Kd of ">= 10 uM" recorded as a number), while KIBA has
**1 row at its floor** and is essentially uncensored. Every committed baseline regresses the
observed value as if it were exact, so on DAVIS roughly two-thirds of the training signal is
a fabricated point mass at 5.0. The README flags the correct treatment (Tobit likelihood or
a two-stage classifier-then-regressor); this script lands the Tobit arm and uses KIBA as the
negative control the design needs: with ~0 % censoring, Tobit must collapse onto ordinary
penalised least squares, and if it does not, the likelihood implementation is wrong.

Model (one-sided left Tobit; DAVIS floor L = 5.0; KIBA floor = 0.0, one row):
    latent y* = x'b + eps,  eps ~ N(0, sigma^2)
    observed y = L if y* <= L, else y*
per-row log-likelihood:
    y > L : -log sigma - 0.5 ((y - mu)/sigma)^2
    y = L : log Phi((L - mu)/sigma)
prediction is the conditional mean of the OBSERVED variable,
    E[y|x] = L*Phi(a) + mu*(1-Phi(a)) + sigma*phi(a),  a = (L - mu)/sigma,
which is the RMSE-optimal point forecast for censored labels; ranking is evaluated from the
same vector, and the ceiling-tie caveat of the tables applies unchanged.

Protocol, identical to 03_models so rows are directly comparable:
  * same dense_compact block, same training-only standardize() guard (imported from 03);
  * ridge penalty lambda on b (intercept and log-sigma exempt) chosen on the VALID block's
    censored negative log-likelihood over (1, 3, 10, 30, 100, 300) -- the same search grid
    03 uses for ridge alphas, so "which lambda" is not a free parameter of this script;
  * metrics from dti_core.evaluate on the TEST block (RMSE, MAE, r, rho, both CI calibers);
  * all five protocols; MLE is deterministic (no data sampling), so one run, seed 0.

Sanity controls written into results/tobit_baseline.json:
  1. KIBA vs ridge: |RMSE(tobit) - RMSE(ridge)| should be small on every protocol
     (near-zero censoring => the two models coincide);
  2. DAVIS floor-recovery: among test rows observed exactly at 5.0, the share with
     E[y|x] within 0.1 of the floor. Note this is expected to be well under 1: the
     censored conditional mean carries a + sigma*phi(a) term, so the RMSE-optimal
     forecast deliberately sits above the floor even where 70 % of labels land on it;
  3. lambda sensitivity: every lambda in the grid is also scored on the TEST block, and
     the JSON reports the spread of test RMSE across the grid plus the regret of the
     NLL-selected lambda vs the test-optimal one (lambda_sensitivity). The choice is
     measured, not assumed -- if the spread is tiny, lambda is not what separates this
     row from ridge; if it is large, the grid is a live hyper-parameter here.

Outputs results/metrics_tobit.csv, results/tobit_baseline.json, per-block predictions under
results/predictions (same npz schema as 03).

Usage
  python scripts/17_tobit_baseline.py                 # both datasets, all protocols
  python scripts/17_tobit_baseline --datasets davis   # one dataset
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
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dti_core import Featuriser, evaluate                      # noqa: E402

_spec = importlib.util.spec_from_file_location("m17_03", SCRIPTS / "03_models.py")
m03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m03)

PROC = ROOT / "data" / "processed"
SPL = PROC / "splits"
RESULTS = ROOT / "results"
PRED = RESULTS / "predictions"

LAMBDAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
FLOORS = {"davis": 5.0, "kiba": 0.0}
PROTOCOLS = ["random", "scaffold", "cold_drug", "cold_target", "temporal"]


# --------------------------------------------------------------------------- Tobit MLE

def _nll_grad(theta, X, y, L, lam):
    """Censored Gaussian NLL (negative log-likelihood + ridge on slopes) and gradient.

    theta = [b (p), sigma_raw (1)]; sigma = exp(sigma_raw) keeps the scale positive.
    Column 0 of X is the constant, so b[0] is the intercept and is penalty-exempt,
    together with sigma. Derivatives, per observation, with mu = x'b, r = (y-mu)/s,
    z = (L-mu)/s, ratio = phi(z)/Phi(z):
      uncensored : dll/dmu = r/s        dll/ds = (r^2 - 1)/s
      censored   : dll/dmu = -ratio/s   dll/ds = -ratio*z/s
    Convention: the 0.5*log(2*pi) of the Gaussian density is dropped. It is constant in
    theta, so the MLE, the lambda selection and every metric in this script are unchanged;
    only the absolute NLL numbers in results/*.json sit 0.9189 nats per uncensored row
    below a textbook log-likelihood. tests/test_tobit_baseline.py pins this convention.
    """
    b, s = theta[:-1], float(np.exp(theta[-1]))
    mu = X @ b
    n = len(y)
    cens = y <= L + 1e-12
    unc = ~cens
    ll = np.zeros(n)
    g_mu = np.zeros(n)
    g_s = 0.0
    if unc.any():
        r = (y[unc] - mu[unc]) / s
        ll[unc] = -np.log(s) - 0.5 * r * r
        g_mu[unc] = r / s
        g_s += float(((r * r - 1.0) / s).sum())
    if cens.any():
        z = (L - mu[cens]) / s
        ratio = norm.pdf(z) / np.maximum(ndtr(z), 1e-300)   # inverse Mills, stable
        ll[cens] = log_ndtr(z)
        g_mu[cens] = -ratio / s
        g_s += float((-ratio * z / s).sum())
    nll = -float(ll.sum()) + lam * float(b[1:] @ b[1:])
    g_b = -(X.T @ g_mu) + np.concatenate([[0.0], 2.0 * lam * b[1:]])
    g_theta = np.concatenate([g_b, [-g_s * s]])             # chain ds/dsigma_raw = s
    return nll, g_theta


def fit_tobit(Xtr, ytr, L, lam, init=None):
    p = Xtr.shape[1]
    theta0 = np.zeros(p + 1)
    if init is None:
        # OLS warm start on the uncensored rows only (censoring-aware enough, cheap)
        keep = ytr > L + 1e-12
        if keep.sum() > p + 1:
            beta, *_ = np.linalg.lstsq(Xtr[keep], ytr[keep], rcond=None)
            theta0[:-1] = beta
            theta0[-1] = np.log(max(float(np.std(ytr[keep] - Xtr[keep] @ beta)), 0.1))
        else:
            theta0[-1] = np.log(max(float(np.std(ytr)), 0.1))
    else:
        theta0 = init
    res = minimize(_nll_grad, theta0, args=(Xtr, ytr, L, lam), jac=True,
                   method="L-BFGS-B", options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-9})
    return res


def predict_tobit(theta, X, L):
    b, s = theta[:-1], float(np.exp(theta[-1]))
    mu = X @ b
    a = (L - mu) / s
    Phi_a, phi_a = ndtr(a), norm.pdf(a)
    mean_obs = L * Phi_a + mu * (1.0 - Phi_a) + s * phi_a
    return mean_obs, mu, s


def _nll_only(theta, X, y, L):
    return _nll_grad(theta, X, y, L, 0.0)[0]


# --------------------------------------------------------------------------- selftest

def gradient_selftest() -> dict:
    """Analytic vs central-difference gradient of the penalised censored NLL."""
    rng = np.random.default_rng(7)
    n, p = 80, 4
    L = 0.6
    X = np.hstack([np.ones((n, 1)), rng.normal(size=(n, p))])
    b_true = np.r_[1.0, rng.normal(size=p)]
    y = np.clip(X @ b_true + 0.7 * rng.normal(size=n), L, None)
    lam = 3.0
    theta = np.r_[rng.normal(size=p + 1), 0.2]
    g0, ga = _nll_grad(theta, X, y, L, lam)
    gn = np.zeros_like(theta)
    h = 1e-6
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        gn[i] = (_nll_grad(tp, X, y, L, lam)[0] - _nll_grad(tm, X, y, L, lam)[0]) / (2 * h)
    denom = np.maximum(np.abs(gn), 1e-8)
    return {"analytic_vs_numeric_max_rel_dev": float(((np.abs(ga - gn) / denom)).max())}


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="davis,kiba")
    args = ap.parse_args()

    st = gradient_selftest()
    print("Tobit gradient selftest:", json.dumps(st), flush=True)
    assert st["analytic_vs_numeric_max_rel_dev"] < 1e-4, "gradient selftest failed"

    f = Featuriser()
    rows, per_cell = [], {}
    for dataset in args.datasets.split(","):
        ycol = "pkd" if dataset == "davis" else "kiba_score"
        L = FLOORS[dataset]
        base = pd.read_csv(PROC / f"{dataset}.csv")[
            ["pair_id", "ligand_id", "target_id", ycol]].set_index("pair_id")
        for protocol in PROTOCOLS:
            t0 = time.time()
            sp_df = pd.read_csv(SPL / f"{dataset}_{protocol}.csv")
            y = base.loc[sp_df.pair_id, ycol].to_numpy(float)
            m = (sp_df.split == "train").to_numpy()
            v = (sp_df.split == "valid").to_numpy()
            te = (sp_df.split == "test").to_numpy()
            Xall = f.blocks(sp_df.ligand_id.to_numpy(), sp_df.target_id.to_numpy(),
                            "dense_compact")
            Xtr, Xv, Xte = m03.standardize(Xall[m], Xall[v], Xall[te])
            del Xall
            Xtr = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
            Xv = np.hstack([np.ones((Xv.shape[0], 1)), Xv])
            Xte = np.hstack([np.ones((Xte.shape[0], 1)), Xte])

            # lambda by censored NLL on the valid block; warm-start the next lambda.
            # Every lambda also gets its TEST metrics recorded, so "did the choice of
            # lambda move the headline number?" is measured, not asserted.
            best = None
            init = None
            lambda_path = []
            for lam in LAMBDAS:
                res = fit_tobit(Xtr, y[m], L, lam, init=init)
                init = res.x
                nll_v = _nll_only(res.x, Xv, y[v], L) / Xv.shape[0]
                yhat_l, _, _ = predict_tobit(res.x, Xte, L)
                eval_l = evaluate(y[te], yhat_l)
                lambda_path.append({"lambda": lam,
                                    "valid_nll_per_row": round(float(nll_v), 6),
                                    "test_rmse": round(float(eval_l["rmse"]), 6),
                                    "test_spearman": round(float(eval_l["spearman_rho"]), 6),
                                    "test_ci_harrell": round(float(eval_l["ci_harrell"]), 6)})
                if best is None or nll_v < best[0]:
                    best = (nll_v, lam, res)
            nll_v, lam, res = best
            yhat, mu_te, sigma = predict_tobit(res.x, Xte, L)

            srow = {"dataset": dataset, "protocol": protocol, "model": "tobit", "seed": 0,
                    "lambda": lam, "sigma": round(float(sigma), 4),
                    "valid_nll_per_row": round(float(nll_v), 6),
                    "cens_frac_train": round(float((y[m] <= L + 1e-12).mean()), 4),
                    "cens_frac_test": round(float((y[te] <= L + 1e-12).mean()), 4)}
            srow.update(evaluate(y[te], yhat))
            srow["seconds"] = round(time.time() - t0, 1)
            rows.append(srow)

            # floor-recovery control: predicted mass at the observed-floor rows
            at_floor = y[te] <= L + 1e-12
            floor_hit = float(np.mean(np.abs(yhat[at_floor] - L) < 0.1)) if at_floor.any() else None
            per_cell[f"{dataset}/{protocol}"] = {
                "lambda": lam, "rmse": round(float(srow["rmse"]), 4),
                "ci_harrell": round(float(srow["ci_harrell"]), 4),
                "ci_sklearn_like": round(float(srow["ci_sklearn_like"]), 4),
                "sigma": round(float(sigma), 4),
                "floor_recovery_share": None if floor_hit is None else round(floor_hit, 4),
                "lambda_path": lambda_path}
            np.savez_compressed(PRED / f"{dataset}_{protocol}_tobit_seed0.npz",
                                pair_id=sp_df.pair_id.to_numpy()[te],
                                y=y[te].astype(np.float32),
                                yhat=yhat.astype(np.float32))
            print(f"{dataset:<6}{protocol:<12}tobit        lambda {lam:>6} "
                  f"RMSE {srow['rmse']:.4f} CI {srow['ci_harrell']:.4f} "
                  f"CI_sk {srow['ci_sklearn_like']:.4f} ({srow['seconds']}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "metrics_tobit.csv", index=False)

    # ---- controls against the committed table --------------------------------
    # Ridge is compared on results/metrics.csv (per-seed rows, averaged over the
    # same seeds 03 ran) rather than the aggregated metrics_by_protocol.csv, so
    # the column names below are the ones evaluate() actually emits.
    mp = pd.read_csv(RESULTS / "metrics.csv")
    ridge = (mp[mp.model.eq("ridge")]
             .groupby(["dataset", "protocol"])[["rmse", "ci_harrell"]]
             .mean())
    controls = {}
    for dataset, kind in (("kiba", "near_zero_censoring_expect_close_to_ridge"),
                          ("davis", "censored_expect_differs")):
        mine = df[df.dataset.eq(dataset)]
        cmp = {r.protocol: {
            "tobit_rmse": round(float(r.rmse), 4),
            "ridge_rmse": round(float(ridge.loc[(dataset, r.protocol), "rmse"]), 4),
            "tobit_ci": round(float(r.ci_harrell), 4),
            "ridge_ci": round(float(ridge.loc[(dataset, r.protocol), "ci_harrell"]), 4),
        } for r in mine.itertuples()}
        # kiba control: the Tobit-vs-ridge RMSE gap must be small everywhere
        if dataset == "kiba":
            gaps = [abs(v["tobit_rmse"] - v["ridge_rmse"]) for v in cmp.values()]
            controls[dataset] = {"expectation": kind,
                                 "max_abs_rmse_gap_vs_ridge": round(max(gaps), 4),
                                 "per_protocol": cmp}
        else:
            controls[dataset] = {"expectation": kind, "per_protocol": cmp}
    # lambda sensitivity, measured rather than asserted. The honest question is not
    # "does the valid-block NLL move" (a ratio of that is unstable when the best NLL
    # sits near zero, which is exactly what kiba's cold_drug cell does), it is
    # "does the HEADLINE test RMSE move across the grid, and how much does choosing
    # lambda by NLL cost vs the test-optimal lambda". Both are reported per cell.
    sensitivity = {}
    for dataset in args.datasets.split(","):
        rmse_spread_pct, regret_pct, nll_spread = [], [], []
        rows_per_dataset = []
        for key, cell in per_cell.items():
            if not key.startswith(dataset + "/"):
                continue
            path = cell["lambda_path"]
            rm = [e["test_rmse"] for e in path]
            nl = [e["valid_nll_per_row"] for e in path]
            oracle = min(rm)
            rmse_spread_pct.append((max(rm) - oracle) / oracle * 100.0)
            regret_pct.append((cell["rmse"] - oracle) / oracle * 100.0)
            nll_spread.append(max(nl) - min(nl))
            rows_per_dataset.append({
                "protocol": key.split("/")[1], "selected_lambda": cell["lambda"],
                "test_rmse_at_selected": cell["rmse"],
                "test_rmse_best_lambda": round(oracle, 4),
                "best_lambda_on_test": path[int(np.argmin(rm))]["lambda"],
                "valid_nll_spread_nats_per_row": round(max(nl) - min(nl), 6)})
        med = float(np.median(rmse_spread_pct))
        sensitivity[dataset] = {
            "grid": list(LAMBDAS),
            "median_test_rmse_spread_across_grid_pct": round(med, 3),
            "max_test_rmse_spread_across_grid_pct": round(float(np.max(rmse_spread_pct)), 3),
            "median_selection_regret_vs_test_optimal_lambda_pct": round(
                float(np.median(regret_pct)), 4),
            "max_selection_regret_vs_test_optimal_lambda_pct": round(
                float(np.max(regret_pct)), 4),
            "median_valid_nll_spread_nats_per_row": round(float(np.median(nll_spread)), 6),
            "per_protocol": rows_per_dataset,
            "reading": ("test RMSE barely moves across the lambda grid, so the selected "
                        "lambda is not what separates this row from ridge"
                        if med < 2.0 else
                        "test RMSE is lambda-sensitive; the selected value moves the "
                        "headline number, so the grid -- not the model -- is a live "
                        "hyper-parameter here")}
    summary = {
        "generated_by": "scripts/17_tobit_baseline.py",
        "gradient_selftest": st,
        "model": ("one-sided left Tobit on dense_compact features; lambda on the valid "
                  "block's censored NLL over the same grid 03 uses for ridge alphas; "
                  "point forecast is E[observed y | x], so RMSE/MAE and both CI calibers "
                  "are computed exactly as every other row of metrics.csv."),
        "floors": FLOORS,
        "controls": controls,
        "lambda_sensitivity": sensitivity,
        "per_cell": per_cell,
    }
    (RESULTS / "tobit_baseline.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
