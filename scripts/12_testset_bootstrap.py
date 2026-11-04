#!/usr/bin/env python3
"""12_testset_bootstrap.py -- put intervals on the whole 150-fit grid, not on one cell.

03_models.py stores a point estimate per (dataset, protocol, model, seed). 07 turns those
points into tables, and the README turns the tables into sentences like "within a protocol
the three models that do not collapse differ by at most 0.1864" and "switching split costs
2.6-7.9x what switching model costs". Neither sentence has ever carried an uncertainty
statement: a 0.0005 gap between XGBoost and a random forest on DAVIS `cold_drug` is printed
in the same type as a 0.48 gap between protocols.

This script resamples the *stored* test predictions -- no model is refitted -- and reports,
for every cell:

  marginal intervals   RMSE / MAE / Harrell CI per arm under three resampling schemes
  model contrasts      the 10 model-level differences per cell (6 learned-vs-learned +
                       4 learned-vs-global-mean), Holm-adjusted *within that family of 10*
                       -- this is the family the README's sentences actually claim
  paired intervals     all 105 within-cell arm-vs-arm differences, kept as an artifact
  exact sign test      paired |error| and squared-error sign tests (no bootstrap involved),
                       Holm-adjusted within the 105-pair family
  protocol contrasts   per model, the widest protocol-minus-protocol RMSE difference

Why two families. A percentile-bootstrap two-sided p-value cannot go below 2/(B+1): with
B=2000 the floor is 1.0e-3, so Holm over the 105 arm pairs (threshold 0.05/105 = 4.8e-4)
rejects *nothing, ever, whatever the effect size*. The first version of this script reported
"0/660 model-vs-model RMSE gaps significant" and that number was an arithmetic property of
B and m, not a property of the data. `holm_resolvable()` below computes the check and the
arm-level table now carries `holm_underpowered` instead of a verdict. The model-level family
of 10 needs 2/(B+1) < 0.005, i.e. B > 398, which rmse/mae satisfy at B=2000 and the CI
bootstrap satisfies at the new default B=600 (floor 3.3e-3 < 5e-3). Multiplicity *across*
the 10 cells is not controlled; a whole-grid family of 100 would need B > 3998.

Three resampling schemes, because the rows are not exchangeable in the same way everywhere:

  row      resample test rows; assumes rows are independent
  target   resample kinase targets as clusters; rows of one target are correlated
           (same protein, many ligands) so this is the honest scheme for a per-target claim
  ligand   resample ligands as clusters; the honest scheme for a per-compound claim

The CI (ranking) bootstrap is the expensive one -- `dti_core.ci_harrell` is a per-row Python
loop -- so it is run with fewer reps than RMSE/MAE and only under `row` and `target`.

Caveat on the protocol contrasts: the five protocols of one dataset draw their test rows
from the same 30 056 / 118 254 pair table and *overlap* (e.g. `random` and `scaffold` share
most test rows). Combining two cells' bootstrap distributions as if they were independent
ignores that overlap, which makes the resulting interval wider than a joint resampling would
give. It is a conservative interval, not a tight one.

Caveat on the model contrasts: a learned model's statistic is the mean over its three fitted
seeds, and those three seeds are *the same three fits in every bootstrap draw*. The interval
therefore carries test-set sampling variability only, not refit randomness; seed-to-seed
spread is visible in the arm-level table (seed_i vs seed_j rows of the same model) and in
`results/variance_decomposition.json`, not here. Averaging the seeds is the right estimand
for "which model class is better" and the wrong one for "how much would my RMSE move if I
refit".

Outputs results/testset_bootstrap_ci.csv, results/testset_bootstrap_models.csv,
results/testset_bootstrap_pairs.csv, results/testset_bootstrap.json.

Usage
  python scripts/12_testset_bootstrap.py [--reps 2000] [--ci-reps 600] [--jobs 6]
  python scripts/12_testset_bootstrap.py --quick      # davis/random, 200 + 40 reps
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dti_core import ci_harrell  # noqa: E402

PROC = ROOT / "data" / "processed"
PRED = ROOT / "results" / "predictions"
RES = ROOT / "results"

DATASETS = ["davis", "kiba"]
PROTOCOLS = ["random", "scaffold", "cold_drug", "cold_target", "temporal"]
LEARNED = ["ridge", "xgboost", "randomforest", "mlp"]
NULLS = ["mean", "mean_ligand_train", "mean_target_train"]
SEEDS_LEARNED = [0, 1, 2]
# arms are ordered deterministically: nulls first (1 seed), then learned x 3 seeds
ARMS = [(m, 0) for m in NULLS] + [(m, s) for m in LEARNED for s in SEEDS_LEARNED]
BASE_SEED = 20261103
LABEL_COL = {"davis": "pkd", "kiba": "kiba_score"}

# the model-level contrast family: 6 learned-vs-learned + 4 learned-vs-global-mean = 10.
# m=10 is what makes Holm resolvable at B=2000 (rmse/mae) and B=600 (Harrell CI).
MODEL_SET = LEARNED + ["mean"]
MODEL_CONTRASTS = list(itertools.combinations(LEARNED, 2)) + [(m, "mean") for m in LEARNED]
# the family the README's headline sentence is read off: RMSE, structure-aware resample
PRIMARY = ("target", "rmse")


# --------------------------------------------------------------------------- loading

def load_cell(ds: str, proto: str) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Return (arm_names, y, P[arms x n], clusters) for one dataset x protocol cell."""
    y_ref = None
    pid_ref = None
    rows = []
    for model, seed in ARMS:
        f = PRED / f"{ds}_{proto}_{model}_seed{seed}.npz"
        if not f.exists():
            raise SystemExit(f"missing prediction file {f}")
        z = np.load(f)
        pid, y, yh = z["pair_id"], z["y"].astype(np.float64), z["yhat"].astype(np.float64)
        if pid_ref is None:
            pid_ref, y_ref = pid, y
        else:
            # every arm of a cell must be scored on exactly the same rows in the same order,
            # otherwise a "paired" difference is comparing two different test sets
            assert np.array_equal(pid, pid_ref), f"{f.name}: pair_id order differs"
            assert np.allclose(y, y_ref, atol=0, rtol=0), f"{f.name}: labels differ"
        rows.append(yh)
    P = np.vstack(rows)

    lab = pd.read_csv(PROC / f"{ds}.csv")[["pair_id", "ligand_id", "target_id"]]
    lab = lab.set_index("pair_id").loc[pid_ref]
    clusters = {
        "target": lab["target_id"].to_numpy(),
        "ligand": lab["ligand_id"].to_numpy(),
    }
    names = [f"{m}_seed{s}" for m, s in ARMS]
    return names, y_ref, P, clusters


def cluster_index_groups(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.argsort(groups, kind="stable")
    parts = np.split(order, np.cumsum(counts)[:-1])
    return uniq, parts


def draw_indices(kind: str, n: int, groups, parts, rng) -> np.ndarray:
    if kind == "row":
        return rng.integers(0, n, size=n)
    take = rng.integers(0, len(parts), size=len(parts))
    return np.concatenate([parts[t] for t in take])


# --------------------------------------------------------------------------- statistics

def point_stats(y: np.ndarray, P: np.ndarray) -> pd.DataFrame:
    err = P - y
    return pd.DataFrame({
        "rmse": np.sqrt((err ** 2).mean(axis=1)),
        "mae": np.abs(err).mean(axis=1),
        "ci_harrell": [ci_harrell(y, P[a]) for a in range(P.shape[0])],
    })


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, monotone, capped at 1."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def sign_test(va: np.ndarray, vb: np.ndarray) -> float:
    """Exact two-sided sign test on paired per-row losses (|error| or squared error)."""
    win = int((va < vb).sum())
    lose = int((va > vb).sum())
    if win + lose == 0:
        return 1.0
    return float(binomtest(win, win + lose, 0.5).pvalue)


def boot_p(delta: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p-value for H0: delta = 0."""
    B = len(delta)
    le = float((delta <= 0).sum())
    ge = float((delta >= 0).sum())
    return float(min(1.0, 2.0 * min((le + 1.0) / (B + 1.0), (ge + 1.0) / (B + 1.0))))


def holm_resolvable(B: int, m: int, alpha: float = 0.05) -> tuple[float, bool]:
    """(p-value floor, can Holm over m tests ever reject at alpha?).

    A percentile bootstrap cannot produce a two-sided p below 2/(B+1) -- that is the
    all-draws-on-one-side case -- and Holm's smallest threshold is alpha/m. If the floor is
    not below that threshold, "no significant difference" is a property of B and m, not of
    the data, and must not be reported as a finding.
    """
    floor = 2.0 / (B + 1.0)
    return floor, bool(floor < alpha / m)


def model_mean_matrix(R: np.ndarray, names: list[str], models: list[str]) -> np.ndarray:
    """(B x len(models)) bootstrap statistic averaged over each model's fitted seeds.

    The same fits are reused in every draw, so this propagates test-set sampling only --
    see the module docstring's caveat.
    """
    cols = []
    for m in models:
        idx = [a for a, nm in enumerate(names) if nm.rsplit("_seed", 1)[0] == m]
        assert idx, f"{m}: no arm in {names}"
        cols.append(R[:, idx].mean(axis=1))
    return np.column_stack(cols)


# --------------------------------------------------------------------------- one cell

def stable_id(*parts) -> int:
    """Deterministic 31-bit id (Python's hash() is salted per process, so it would make
    the bootstrap seeds irreproducible across runs)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def run_cell(job: dict) -> dict:
    ds, proto = job["dataset"], job["protocol"]
    reps, ci_reps = job["reps"], job["ci_reps"]
    t0 = time.time()
    names, y, P, clusters = load_cell(ds, proto)
    n = len(y)
    n_arms = P.shape[0]
    E2 = (P - y) ** 2
    AE = np.abs(P - y)

    marg = {m: {} for m in ["rmse", "mae", "ci_harrell"]}   # metric -> resample -> (B x arms)
    pairs = list(itertools.combinations(range(n_arms), 2))

    for resample in ["row", "target", "ligand"]:
        if resample == "row":
            groups = parts = None
        else:
            groups, parts = cluster_index_groups(clusters[resample])
        want_ci = resample in ("row", "target")
        B_ci = ci_reps if want_ci else 0
        R_rmse = np.empty((reps, n_arms))
        R_mae = np.empty((reps, n_arms))
        R_ci = np.empty((B_ci, n_arms))
        rng = np.random.default_rng(np.random.SeedSequence(
            [BASE_SEED, stable_id(ds, proto, resample, "rmse_mae")]))
        for b in range(reps):
            idx = draw_indices(resample, n, groups, parts, rng)
            R_rmse[b] = np.sqrt(E2[:, idx].mean(axis=1))
            R_mae[b] = AE[:, idx].mean(axis=1)
        if B_ci:
            rng_ci = np.random.default_rng(np.random.SeedSequence(
                [BASE_SEED, stable_id(ds, proto, resample, "ci")]))
            for b in range(B_ci):
                idx = draw_indices(resample, n, groups, parts, rng_ci)
                yy = y[idx]
                for a in range(n_arms):
                    R_ci[b, a] = ci_harrell(yy, P[a][idx])
        marg["rmse"][resample] = R_rmse
        marg["mae"][resample] = R_mae
        if B_ci:
            marg["ci_harrell"][resample] = R_ci

    pt = point_stats(y, P)

    # exact sign tests are computed on the observed rows, so they are resample-independent:
    # `mae` compares |error|, `sq` compares squared error (the loss behind RMSE)
    sign = {}
    for i, j in pairs:
        sign[(i, j)] = {"mae": sign_test(AE[i], AE[j]), "sq": sign_test(E2[i], E2[j])}

    return {"dataset": ds, "protocol": proto, "n_test": n, "names": names,
            "clusters": {k: int(len(np.unique(v))) for k, v in clusters.items()},
            "point": pt.to_dict(orient="list"), "marg": marg, "pairs": pairs,
            "sign": {f"{k[0]}|{k[1]}": v for k, v in sign.items()},
            "seconds": round(time.time() - t0, 1),
            "reps": reps, "ci_reps": ci_reps}


# --------------------------------------------------------------------------- assembly

def percentile(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0)


def assemble(results: list[dict], metrics: pd.DataFrame
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    ci_rows, pair_rows, model_rows = [], [], []
    summary: dict = {"generated_by": "scripts/12_testset_bootstrap.py",
                     "base_seed": BASE_SEED, "cells": [],
                     "model_contrast_family": [list(c) for c in MODEL_CONTRASTS],
                     "primary_family": {"resample": PRIMARY[0], "metric": PRIMARY[1]}}

    # cross-check the recomputed point estimates against the committed grid
    max_pt_dev = 0.0
    for r in sorted(results, key=lambda d: (d["dataset"], d["protocol"])):
        ds, proto = r["dataset"], r["protocol"]
        names = r["names"]
        for a, name in enumerate(names):
            model, seed = name.rsplit("_seed", 1)
            row = metrics[(metrics.dataset == ds) & (metrics.protocol == proto)
                          & (metrics.model == model) & (metrics.seed == int(seed))]
            assert len(row) == 1, f"{ds}/{proto}/{name}: {len(row)} rows in metrics.csv"
            for metric in ("rmse", "mae", "ci_harrell"):
                stored = float(row.iloc[0][metric])
                recomputed = float(r["point"][metric][a])
                max_pt_dev = max(max_pt_dev, abs(stored - recomputed))
                assert abs(stored - recomputed) < 2e-4, (
                    f"{ds}/{proto}/{name}/{metric}: stored {stored} vs recomputed {recomputed}")

        cell_sum = {"dataset": ds, "protocol": proto, "n_test": r["n_test"],
                    "n_target_clusters": r["clusters"]["target"],
                    "n_ligand_clusters": r["clusters"]["ligand"],
                    "reps_rmse_mae": r["reps"], "reps_ci": r["ci_reps"],
                    "seconds": r["seconds"]}

        # ---- marginal intervals
        for metric, byres in r["marg"].items():
            for resample, R in byres.items():
                lo, hi = percentile(R)
                for a, name in enumerate(names):
                    ci_rows.append({
                        "dataset": ds, "protocol": proto, "model": name.rsplit("_seed", 1)[0],
                        "seed": int(name.rsplit("_seed", 1)[1]), "metric": metric,
                        "resample": resample, "n_test": r["n_test"],
                        "point": r["point"][metric][a],
                        "lo95": lo[a], "hi95": hi[a], "width95": hi[a] - lo[a],
                        "valid_reps": int(np.isfinite(R[:, a]).sum()), "reps": R.shape[0],
                    })

        # ---- paired intervals + Holm within (cell, metric, resample)
        cell_sig = {"model_vs_model_rmse_holm": 0, "model_vs_model_pairs": 0,
                    "largest_significant_model_vs_model_rmse": None,
                    "learned_beat_mean_null_rmse": {}, "ci_row_significant": {},
                    "arm_pair_holm_resolvable": {}, "arm_pair_ci_excludes_zero": {},
                    "arm_pair_sign_holm_significant": {}}

        # exact sign tests are resample-independent (they use the observed rows), so Holm
        # over the 105-pair family is applied once per metric. Unlike the bootstrap p, the
        # exact binomial p has no resolution floor, so this family *is* testable -- at the
        # cost of treating rows as exchangeable, i.e. ignoring within-target correlation.
        sign_holm = {}
        for metric, key in (("rmse", "sq"), ("mae", "mae")):
            sp = np.array([r["sign"][f"{i}|{j}"][key] for i, j in r["pairs"]])
            adj_sp = holm(sp)
            sign_holm[metric] = {pair: float(adj_sp[k]) for k, pair in enumerate(r["pairs"])}

        for (resample, metric), R in sorted(r_pair_container(r).items()):
            # D is (B x n_pairs): the paired difference is a statistic in its own right, so
            # its interval comes from resampling, not from the two marginal intervals
            D = np.column_stack([R[:, i] - R[:, j] for i, j in r["pairs"]])
            lo, hi = percentile(D)
            raw_p = [boot_p(D[:, k]) for k in range(D.shape[1])]
            adj = holm(np.array(raw_p))
            floor, resolvable = holm_resolvable(R.shape[0], len(r["pairs"]))
            n_excl = 0
            n_sign_sig = 0
            for k, (i, j) in enumerate(r["pairs"]):
                dpoint = float(r["point"][metric][i] - r["point"][metric][j])
                excl = bool(np.isfinite(lo[k]) and (lo[k] > 0 or hi[k] < 0))
                n_excl += int(excl)
                rec = {"dataset": ds, "protocol": proto, "metric": metric,
                       "resample": resample,
                       "model_a": names[i], "model_b": names[j],
                       "delta_point": dpoint, "delta_lo95": lo[k], "delta_hi95": hi[k],
                       "boot_p": raw_p[k], "holm_p": adj[k], "p_floor": floor,
                       # a verdict from this Holm would be meaningless whenever
                       # `holm_underpowered` is True -- read `ci_excludes_zero` instead
                       "holm_underpowered": not resolvable,
                       "ci_excludes_zero": excl}
                if metric in sign_holm:
                    rec["sign_test_p"] = r["sign"][f"{i}|{j}"][
                        "sq" if metric == "rmse" else "mae"]
                    rec["sign_test_holm_p"] = sign_holm[metric][(i, j)]
                    rec["significant_sign_holm05"] = bool(
                        rec["sign_test_holm_p"] < 0.05 and excl)
                    n_sign_sig += int(rec["significant_sign_holm05"])
                pair_rows.append(rec)
            cell_sig["arm_pair_holm_resolvable"][f"{metric}/{resample}"] = bool(resolvable)
            cell_sig["arm_pair_ci_excludes_zero"][f"{metric}/{resample}"] = (
                f"{n_excl}/{len(r['pairs'])}")
            if metric in sign_holm:
                cell_sig["arm_pair_sign_holm_significant"][f"{metric}/{resample}"] = (
                    f"{n_sign_sig}/{len(r['pairs'])}")

            # ---- model-level family of 10: 6 learned-vs-learned + 4 learned-vs-global-mean.
            # This is the family the README's sentences claim, and at m=10 Holm is resolvable.
            M = model_mean_matrix(R, names, MODEL_SET)
            mpoint = {m: float(np.mean([r["point"][metric][a] for a, nm in enumerate(names)
                                        if nm.rsplit("_seed", 1)[0] == m]))
                      for m in MODEL_SET}
            contrasts = MODEL_CONTRASTS
            Dm = np.column_stack([M[:, MODEL_SET.index(a)] - M[:, MODEL_SET.index(b)]
                                  for a, b in contrasts])
            lom, him = percentile(Dm)
            raw_m = [boot_p(Dm[:, k]) for k in range(Dm.shape[1])]
            adj_m = holm(np.array(raw_m))
            mfloor, mresolvable = holm_resolvable(R.shape[0], len(contrasts))
            sgn = -1.0 if metric in ("rmse", "mae") else 1.0   # lower error / higher C-index
            for k, (a, b) in enumerate(contrasts):
                dp = mpoint[a] - mpoint[b]
                sig = bool(adj_m[k] < 0.05 and mresolvable
                           and np.isfinite(lom[k]) and (lom[k] > 0 or him[k] < 0))
                model_rows.append({
                    "dataset": ds, "protocol": proto, "metric": metric, "resample": resample,
                    "model_a": a, "model_b": b, "contrast": "learned_vs_learned"
                    if b != "mean" else "learned_vs_global_mean_null",
                    "delta_point": dp, "delta_lo95": lom[k], "delta_hi95": him[k],
                    "delta_favours": (a if sgn * dp < 0 else b) if dp != 0 else "tie",
                    "boot_p": raw_m[k], "holm_p": adj_m[k], "p_floor": mfloor,
                    "holm_resolvable": mresolvable,
                    "ci_excludes_zero": bool(np.isfinite(lom[k]) and (lom[k] > 0 or him[k] < 0)),
                    "significant_holm05": sig,
                    "family_size": len(contrasts), "reps": int(R.shape[0])})
                if metric == "rmse" and resample == "target":
                    if b != "mean":
                        cell_sig["model_vs_model_pairs"] += 1
                        if sig:
                            cell_sig["model_vs_model_rmse_holm"] += 1
                            cur = cell_sig["largest_significant_model_vs_model_rmse"]
                            if cur is None or abs(dp) > abs(cur["delta_point"]):
                                cell_sig["largest_significant_model_vs_model_rmse"] = {
                                    "a": a, "b": b, "delta_point": round(dp, 4),
                                    "lo95": round(float(lom[k]), 4),
                                    "hi95": round(float(him[k]), 4),
                                    "holm_p": float(adj_m[k])}
                    else:
                        # orient: positive = the learned model has the SMALLER rmse
                        cell_sig["learned_beat_mean_null_rmse"][a] = {
                            "delta_rmse_mean_minus_model": round(-dp, 4),
                            "holm_p": float(adj_m[k]), "significant": sig,
                            "holm_resolvable": bool(mresolvable),
                            "point_estimate_better": bool(-dp > 0)}
                if metric == "ci_harrell" and resample == "row":
                    if b != "mean" and sig:
                        cell_sig["ci_row_significant"][f"{a}-vs-{b}"] = round(dp, 4)
        cell_sum.update(cell_sig)
        summary["cells"].append(cell_sum)

    # ---- protocol contrasts (per model x seed, widest protocol difference in RMSE)
    proto_contrasts = []
    by_cell = {(r["dataset"], r["protocol"]): r for r in results}
    for ds in [d for d in DATASETS if any(k[0] == d for k in by_cell)]:
        cells = [p for p in PROTOCOLS if (ds, p) in by_cell]
        if len(cells) < 2:
            continue   # a protocol contrast needs at least two protocols in the same dataset
        reps = min(by_cell[(ds, p)]["marg"]["rmse"]["row"].shape[0] for p in cells)
        for a, name in enumerate(by_cell[(ds, cells[0])]["names"]):
            draws = {p: by_cell[(ds, p)]["marg"]["rmse"]["row"][:reps, a] for p in cells}
            pts = {p: float(by_cell[(ds, p)]["point"]["rmse"][a]) for p in cells}
            lo_p = min(pts, key=pts.get)
            hi_p = max(pts, key=pts.get)
            # conservative: the two cells' bootstrap draws are combined as if independent,
            # but the protocols share test rows (see module docstring)
            d = draws[hi_p] - draws[lo_p]
            dl, dh = np.nanpercentile(d, [2.5, 97.5])
            proto_contrasts.append({
                "dataset": ds, "model": name, "metric": "rmse",
                "easiest_protocol": lo_p, "easiest_rmse": round(pts[lo_p], 4),
                "hardest_protocol": hi_p, "hardest_rmse": round(pts[hi_p], 4),
                "spread_point": round(pts[hi_p] - pts[lo_p], 4),
                "spread_lo95": round(float(dl), 4), "spread_hi95": round(float(dh), 4),
                "excludes_zero": bool(dl > 0), "reps_used": int(reps)})
    summary["protocol_contrasts_rmse"] = proto_contrasts

    spread = [c["spread_point"] for c in proto_contrasts]
    if spread:
        summary["protocol_spread_range_rmse"] = [round(min(spread), 4), round(max(spread), 4)]
        summary["protocol_spread_excludes_zero_frac"] = round(
            float(np.mean([c["excludes_zero"] for c in proto_contrasts])), 4)
    else:
        summary["protocol_spread_range_rmse"] = None
        summary["protocol_spread_excludes_zero_frac"] = None
    summary["point_estimate_max_abs_deviation_vs_metrics_csv"] = float(max_pt_dev)

    # headline numbers for the README, read off the *model-level* family (m=10), because the
    # arm-level Holm (m=105) has a p-floor above its own rejection threshold and can never
    # reject -- reporting its zero as a finding would be an arithmetic artefact.
    mm = [c for c in summary["cells"]]
    mdf = pd.DataFrame(model_rows)
    # bracket indexing on purpose: `mdf.resample` is the DataFrame resample *method*, and
    # attribute access silently returns it instead of the column
    prim = mdf[(mdf["metric"] == PRIMARY[1]) & (mdf["resample"] == PRIMARY[0])] if len(mdf) \
        else mdf
    summary["cells_with_significant_model_vs_model_rmse_target_bootstrap"] = [
        f"{c['dataset']}/{c['protocol']}" for c in mm
        if c["model_vs_model_rmse_holm"] > 0]
    summary["model_vs_model_rmse_pairs_tested_per_cell"] = int(
        mm[0]["model_vs_model_pairs"]) if mm else 0
    summary["total_model_vs_model_significant"] = int(
        sum(c["model_vs_model_rmse_holm"] for c in mm))
    summary["total_model_vs_model_tested"] = int(
        sum(c["model_vs_model_pairs"] for c in mm))
    beat = {f"{c['dataset']}/{c['protocol']}": c["learned_beat_mean_null_rmse"] for c in mm}
    flat = [(cell, m, v) for cell, d in beat.items() for m, v in d.items()]
    # two separate facts, because conflating them is how "False" ended up looking like a
    # finding: the point estimate can favour the model while Holm is unresolvable at this B
    summary["learned_beat_mean_null_point_estimate_all_cells"] = bool(
        all(v["point_estimate_better"] for _, _, v in flat)) if flat else None
    summary["learned_beat_mean_null_significant_all_cells"] = bool(
        all(v["significant"] for _, _, v in flat)) if flat else None
    summary["learned_beat_mean_null_counts"] = {
        "tested": len(flat),
        "point_estimate_better": int(sum(v["point_estimate_better"] for _, _, v in flat)),
        "significant_after_holm": int(sum(v["significant"] for _, _, v in flat)),
        # "significant" is two-sided, so it does not say whose side. Splitting it is the
        # difference between "33/40 contrasts are significant" (true, and readable as a win for
        # the models) and "24 favour the model, 9 favour the global-mean null" (also true).
        # Provenance: these two keys and the list below were added AFTER the committed
        # results/testset_bootstrap.json was written, so that file does not contain them;
        # 07_report.py derives the same values from each cell's deltas and holm_p and stamps
        # the derivation in summary.json["testset_bootstrap"]["source"].
        "significant_in_models_favour": int(sum(
            v["significant"] and v["delta_rmse_mean_minus_model"] > 0 for _, _, v in flat)),
        "significant_in_nulls_favour": int(sum(
            v["significant"] and v["delta_rmse_mean_minus_model"] < 0 for _, _, v in flat)),
        "holm_unresolvable_blocks": int(sum(not v["holm_resolvable"] for _, _, v in flat))}
    summary["learned_beat_mean_null_significant_in_nulls_favour"] = [
        {"cell": cell, "model": m,
         "delta_rmse_mean_minus_model": v["delta_rmse_mean_minus_model"],
         "holm_p": v["holm_p"]}
        for cell, m, v in flat
        if v["significant"] and v["delta_rmse_mean_minus_model"] < 0]
    # keep the old key name so 07_report.py and the README's merge stay wired up
    summary["learned_beat_mean_null_all_cells"] = summary[
        "learned_beat_mean_null_significant_all_cells"]
    # name the exceptions rather than printing a bare False: which cell, which model, and
    # whether it lost the point estimate or just could not clear an underpowered Holm
    summary["learned_beat_mean_null_exceptions"] = [
        {"cell": cell, "model": m,
         "delta_rmse_mean_minus_model": v["delta_rmse_mean_minus_model"],
         "holm_p": v["holm_p"], "holm_resolvable": v["holm_resolvable"],
         "point_estimate_worse_than_null": not v["point_estimate_better"]}
        for cell, m, v in flat if not v["significant"]]
    # keyed per cell: the same metric/resample label recurs in all 10 cells, so a flat
    # dict would silently keep only the last one
    summary["arm_level_holm_resolvable"] = {
        f"{c['dataset']}/{c['protocol']} {k}": bool(v)
        for c in mm for k, v in c["arm_pair_holm_resolvable"].items()}
    summary["arm_level_ci_excludes_zero"] = {
        f"{c['dataset']}/{c['protocol']} {k}": v
        for c in mm for k, v in c["arm_pair_ci_excludes_zero"].items()}
    summary["arm_level_sign_holm_significant"] = {
        f"{c['dataset']}/{c['protocol']} {k}": v
        for c in mm for k, v in c["arm_pair_sign_holm_significant"].items()}
    summary["sign_test_caveat"] = ("exact sign tests treat test rows as exchangeable and "
                                   "therefore ignore within-target / within-ligand "
                                   "correlation; they are anti-conservative relative to the "
                                   "cluster bootstraps and are reported alongside, not "
                                   "instead of, them")
    if len(prim):
        # per cell, the *closest* learned-vs-learned RMSE pair: this is the number the
        # README's "the models that do not collapse differ by at most X" sentence rests on,
        # and it is the one contrast most likely to fail Holm.
        rows = []
        lv = prim[prim["contrast"] == "learned_vs_learned"].copy()
        lv["abs_delta"] = lv.delta_point.abs()
        for (dsx, px), g in lv.groupby(["dataset", "protocol"]):
            r0 = g.sort_values("abs_delta", kind="stable").iloc[0]
            rows.append({"cell": f"{dsx}/{px}", "closest_pair": f"{r0.model_a}-{r0.model_b}",
                         "delta_rmse": round(float(r0.delta_point), 4),
                         "ci_lo95": round(float(r0.delta_lo95), 4),
                         "ci_hi95": round(float(r0.delta_hi95), 4),
                         "holm_p": round(float(r0.holm_p), 5),
                         "significant": bool(r0.significant_holm05)})
        summary["closest_learned_pair_per_cell_primary_family"] = rows
        summary["primary_family_learned_vs_learned_significant"] = (
            f"{int(lv.significant_holm05.sum())}/{len(lv)}")
        summary["primary_family_holm_resolvable"] = bool(lv.holm_resolvable.all())
        summary["primary_family_reps"] = sorted(set(int(x) for x in prim.reps))

    return pd.DataFrame(ci_rows), mdf, pd.DataFrame(pair_rows), summary


def r_pair_container(r: dict) -> dict:
    """Flatten {(resample, metric): R} into the iteration order used by assemble()."""
    out = {}
    for metric, byres in r["marg"].items():
        for resample, R in byres.items():
            out[(resample, metric)] = R
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000, help="RMSE/MAE bootstrap reps")
    ap.add_argument("--ci-reps", type=int, default=600,
                    help="Harrell-CI bootstrap reps (must exceed 398 for Holm over the "
                         "10-contrast model family to be resolvable at alpha=0.05)")
    ap.add_argument("--jobs", type=int, default=6, help="parallel cells (one process each)")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--protocols", nargs="*", default=PROTOCOLS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    reps, ci_reps = (200, 40) if args.quick else (args.reps, args.ci_reps)
    datasets = ["davis"] if args.quick else args.datasets
    protocols = ["random"] if args.quick else args.protocols

    metrics = pd.read_csv(RES / "metrics.csv")
    jobs = [{"dataset": d, "protocol": p, "reps": reps, "ci_reps": ci_reps}
            for d in datasets for p in protocols]
    print(f"{len(jobs)} cells, reps={reps} (rmse/mae) / {ci_reps} (harrell ci), "
          f"jobs={args.jobs}", flush=True)

    t0 = time.time()
    workers = max(1, min(args.jobs, len(jobs)))
    if workers == 1:
        results = [run_cell(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run_cell, jobs))
    for r in sorted(results, key=lambda d: (d["dataset"], d["protocol"])):
        print(f"  {r['dataset']}/{r['protocol']}: n={r['n_test']} "
              f"targets={r['clusters']['target']} ligands={r['clusters']['ligand']} "
              f"{r['seconds']} s", flush=True)

    ci_df, model_df, pair_df, summary = assemble(results, metrics)
    ci_df.to_csv(RES / "testset_bootstrap_ci.csv", index=False)
    model_df.to_csv(RES / "testset_bootstrap_models.csv", index=False)
    pair_df.to_csv(RES / "testset_bootstrap_pairs.csv", index=False)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["n_ci_rows"] = int(len(ci_df))
    summary["n_model_rows"] = int(len(model_df))
    summary["n_pair_rows"] = int(len(pair_df))
    (RES / "testset_bootstrap.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {len(ci_df)} marginal rows, {len(model_df)} model-contrast rows, "
          f"{len(pair_df)} paired rows, {summary['wall_seconds']} s wall")
    print(f"point estimates agree with results/metrics.csv to "
          f"{summary['point_estimate_max_abs_deviation_vs_metrics_csv']:.2e}")
    if summary["protocol_spread_range_rmse"] is not None:
        print(f"protocol RMSE spread across cells: "
              f"{summary['protocol_spread_range_rmse']} "
              f"({summary['protocol_spread_excludes_zero_frac']:.0%} of model x dataset "
              f"contrasts exclude zero)")
    print(f"primary family = {PRIMARY[1]} under the {PRIMARY[0]} resample, "
          f"m={len(MODEL_CONTRASTS)} per cell, reps={summary.get('primary_family_reps')}, "
          f"Holm resolvable={summary.get('primary_family_holm_resolvable')}")
    print(f"model-vs-model RMSE gaps significant after Holm (primary family): "
          f"{summary['total_model_vs_model_significant']}/"
          f"{summary['total_model_vs_model_tested']}")
    nb = summary["learned_beat_mean_null_counts"]
    print(f"learned model vs global-mean null on RMSE (primary family): "
          f"{nb['point_estimate_better']}/{nb['tested']} point estimates favour the model, "
          f"{nb['significant_after_holm']}/{nb['tested']} significant after Holm -- "
          f"{nb['significant_in_models_favour']} in the model's favour and "
          f"{nb['significant_in_nulls_favour']} in the NULL's favour "
          f"({nb['holm_unresolvable_blocks']} of those contrasts sat in a block where Holm "
          f"is unresolvable at this B)")
    for e in summary["learned_beat_mean_null_significant_in_nulls_favour"]:
        print(f"  null significantly better: {e['cell']} {e['model']} "
              f"delta(mean - model)={e['delta_rmse_mean_minus_model']:+.4f} "
              f"holm_p={e['holm_p']:.4g}")
    for e in summary["learned_beat_mean_null_exceptions"]:
        why = ("point estimate is WORSE than the null" if e["point_estimate_worse_than_null"]
               else "Holm unresolvable at this B -- not a null result"
               if not e["holm_resolvable"] else "interval spans zero")
        print(f"  exception: {e['cell']} {e['model']} "
              f"delta(mean - model)={e['delta_rmse_mean_minus_model']:+.4f} "
              f"holm_p={e['holm_p']:.4g}  [{why}]")
    under = [k for k, v in summary["arm_level_holm_resolvable"].items() if not v]
    print(f"arm-level Holm (m=105) unresolvable in {len(under)}/"
          f"{len(summary['arm_level_holm_resolvable'])} cell x metric x resample blocks "
          f"-- those blocks report ci_excludes_zero, not a verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
