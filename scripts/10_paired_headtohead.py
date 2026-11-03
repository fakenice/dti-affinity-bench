#!/usr/bin/env python3
"""10_paired_headtohead.py -- is a Vina-vs-model "win" distinguishable from noise?

06_docking_analysis.py declares a winner from two marginal point estimates (CI_vina >
CI_03). Those CIs are computed on the same rows but never compared *as a difference*, so a
0.001 gap is reported the same way as a 0.20 gap. With 21-28 docked test pairs per protocol
that is the difference between a result and a coin flip.

This script keeps the identical pairing and asks the paired question three ways:

  delta_rmse      RMSE_vina - RMSE_model, positive = the model is closer in pKd
  delta_ci        CI_model - CI_vina (Harrell), positive = the model ranks better
  row sign test   how many pairs the model gets closer in |error|; exact binomial

Each delta gets a 95 % interval from two bootstraps: rows resampled with replacement, and
targets resampled as clusters (docked pairs of the same kinase are not independent).

Outputs results/docking/paired_headtohead.csv and .json. No model is refitted; predictions
are the seed-0 rows 06 already used.

Usage
  python scripts/10_paired_headtohead.py [--reps 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dti_core import ci_denominators, ci_harrell   # noqa: E402

PROC = ROOT / "data" / "processed"
SPL = PROC / "splits"
DOCK = ROOT / "results" / "docking"
PRED = ROOT / "results" / "predictions"
OUT = DOCK / "paired_headtohead"

PROTOCOLS = ["random", "scaffold", "cold_drug", "cold_target", "temporal"]
LEARNED = ["ridge", "xgboost", "randomforest", "mlp"]
NULLS = ["mean", "mean_ligand_train", "mean_target_train"]
SEED = 20261103


def load_pairs() -> pd.DataFrame:
    p = pd.read_csv(DOCK / "pairs.csv")
    p = p[p.vina_best.notna()].copy()
    lab = pd.read_csv(PROC / "davis.csv")[["pair_id", "pkd"]].rename(
        columns={"pkd": "pkd_label"})
    p = p.merge(lab, on="pair_id", how="left", validate="one_to_one")
    p["neg_vina"] = -p.vina_best
    return p


def splits() -> dict[str, pd.Series]:
    out = {}
    for proto in PROTOCOLS:
        f = SPL / f"davis_{proto}.csv"
        if f.exists():
            out[proto] = pd.read_csv(f)[["pair_id", "split"]].set_index("pair_id").split
    return out


def ci_fast(y: np.ndarray, p: np.ndarray) -> float:
    """Harrell CI, prediction ties 0.5, denominator = unequal-label pairs only."""
    iu = np.triu_indices(len(y), 1)
    if len(iu[0]) == 0:
        return float("nan")
    ya, yb = y[iu[0]], y[iu[1]]
    pa, pb = p[iu[0]], p[iu[1]]
    keep = ya != yb
    if not keep.any():
        return float("nan")
    hi = np.where(ya[keep] > yb[keep], pa[keep], pb[keep])
    lo = np.where(ya[keep] > yb[keep], pb[keep], pa[keep])
    return float(np.mean((hi > lo) + 0.5 * (hi == lo)))


def delta_stats(y: np.ndarray, p_model: np.ndarray, p_vina: np.ndarray) -> dict:
    e_m = p_model - y
    e_v = p_vina - y
    # Both deltas are oriented so positive = the model is better:
    #   delta_rmse = RMSE_vina - RMSE_model (positive = model closer)
    #   delta_ci   = CI_model - CI_vina     (positive = model ranks better)
    return {
        "rmse_model": float(np.sqrt(np.mean(e_m ** 2))),
        "rmse_vina": float(np.sqrt(np.mean(e_v ** 2))),
        "delta_rmse": float(np.sqrt(np.mean(e_v ** 2)) - np.sqrt(np.mean(e_m ** 2))),
        "ci_model": ci_fast(y, p_model),
        "ci_vina": ci_fast(y, p_vina),
        "delta_ci": ci_fast(y, p_model) - ci_fast(y, p_vina),
    }


def bootstrap(y, p_model, p_vina, groups, reps, rng, cluster):
    """Return 95 % intervals for delta_rmse and delta_ci.

    cluster=False: resample rows with replacement.
    cluster=True : resample target groups with replacement, concatenate their rows.
    """
    n = len(y)
    d_rmse = np.empty(reps)
    d_ci = np.empty(reps)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    for b in range(reps):
        if cluster:
            take = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_group[g] for g in take])
        else:
            idx = rng.integers(0, n, size=n)
        yy, pm, pv = y[idx], p_model[idx], p_vina[idx]
        d_rmse[b] = np.sqrt(np.mean((pv - yy) ** 2)) - np.sqrt(np.mean((pm - yy) ** 2))
        d_ci[b] = ci_fast(yy, pm) - ci_fast(yy, pv)
    # A resample can put every row on the DAVIS censoring ceiling, leaving zero comparable
    # label pairs; nanpercentile keeps that replicate out of the interval instead of
    # poisoning it. valid_ci_reps reports how many replicates actually carried a CI.
    q = np.nanpercentile
    return {
        "delta_rmse_ci95_row": [float(q(d_rmse, 2.5)), float(q(d_rmse, 97.5))],
        "delta_ci_ci95_row": [float(q(d_ci, 2.5)), float(q(d_ci, 97.5))],
        "valid_ci_reps": int(np.isfinite(d_ci).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()

    p = load_pairs()
    sp = splits()
    rng = np.random.default_rng(SEED)
    rows = []

    for proto, s in sp.items():
        te = p[p.pair_id.isin(s[s == "test"].index)].copy()
        if len(te) < 8:
            continue
        tr = p[p.pair_id.isin(s[s == "train"].index)]
        if len(tr) < 8:
            continue
        b, c = np.polyfit(tr.neg_vina.to_numpy(float), tr.pkd_label.to_numpy(float), 1)
        te["pred_vina"] = c + b * te.neg_vina.to_numpy(float)

        for f in sorted(PRED.glob(f"davis_{proto}_*_seed0.npz")):
            model = f.stem[len(f"davis_{proto}_"):].rsplit("_seed", 1)[0]
            if model not in LEARNED + NULLS:
                continue
            z = np.load(f)
            pr = pd.DataFrame({"pair_id": z["pair_id"], "yhat": z["yhat"]})
            j = te.merge(pr, on="pair_id", how="inner", validate="one_to_one")
            if len(j) < 8:
                continue
            y = j.pkd_label.to_numpy(float)
            pm = j.yhat.to_numpy(float)
            pv = j.pred_vina.to_numpy(float)
            base = delta_stats(y, pm, pv)
            groups = j.target_id.to_numpy()
            boots = bootstrap(y, pm, pv, groups, a.reps, rng, cluster=False)
            bootc = bootstrap(y, pm, pv, groups, a.reps, rng, cluster=True)
            closer = int(np.sum(np.abs(pm - y) < np.abs(pv - y)))
            equidistant = int(np.sum(np.abs(pm - y) == np.abs(pv - y)))
            sign_p = (binomtest(closer, closer + (len(y) - closer - equidistant), 0.5,
                                alternative="two-sided").pvalue
                      if (len(y) - equidistant) > 0 else float("nan"))
            def verdict(d, lo, hi):
                if lo > 0:
                    return "model"
                if hi < 0:
                    return "vina"
                return "undecided"
            rows.append({
                "protocol": proto, "model": model, "learned": model in LEARNED,
                "n_rows": int(len(j)), "n_targets": int(j.target_id.nunique()),
                "n_comparable_pairs": int(ci_denominators(y)["n_comparable_pairs"]),
                **{k: round(v, 4) for k, v in base.items()},
                "delta_rmse_lo_row": round(boots["delta_rmse_ci95_row"][0], 4),
                "delta_rmse_hi_row": round(boots["delta_rmse_ci95_row"][1], 4),
                "delta_ci_lo_row": round(boots["delta_ci_ci95_row"][0], 4),
                "delta_ci_hi_row": round(boots["delta_ci_ci95_row"][1], 4),
                "delta_rmse_lo_cluster": round(bootc["delta_rmse_ci95_row"][0], 4),
                "delta_rmse_hi_cluster": round(bootc["delta_rmse_ci95_row"][1], 4),
                "delta_ci_lo_cluster": round(bootc["delta_ci_ci95_row"][0], 4),
                "delta_ci_hi_cluster": round(bootc["delta_ci_ci95_row"][1], 4),
                "valid_ci_reps_row": boots["valid_ci_reps"],
                "valid_ci_reps_cluster": bootc["valid_ci_reps"],
                "point_win": ("tie" if abs(base["delta_ci"]) < 1e-12 else
                              ("model" if base["delta_ci"] > 0 else "vina")),
                "rmse_verdict_row": verdict(base["delta_rmse"],
                                            *boots["delta_rmse_ci95_row"]),
                "ci_verdict_row": verdict(base["delta_ci"], *boots["delta_ci_ci95_row"]),
                "ci_verdict_cluster": verdict(base["delta_ci"],
                                              *bootc["delta_ci_ci95_row"]),
                "rows_model_closer": closer,
                "rows_vina_closer": int(len(y) - closer - equidistant),
                "rows_equidistant": equidistant,
                "sign_test_p": float(sign_p),
            })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # invariant checks: the table is the same 16 learned cells 06 reports, and the two
    # stored deltas are exactly the two metric differences (a sign slip here is silent).
    assert len(df[df.learned]) == 16, f"expected 16 learned cells, got {len(df[df.learned])}"
    assert set(df.point_win) <= {"model", "vina", "tie"}
    assert np.allclose(df.delta_ci, df.ci_model - df.ci_vina, atol=2e-4)
    assert np.allclose(df.delta_rmse, df.rmse_vina - df.rmse_model, atol=2e-4)
    df.to_csv(OUT.with_suffix(".csv"), index=False)

    learned = df[df.learned]
    summary = {
        "generated_by": "scripts/10_paired_headtohead.py",
        "reps": a.reps, "seed": SEED,
        "note": ("Pairing is identical to 06: same test pair_ids, same Vina regression fit on "
                 "the train block, same seed-0 model predictions. Both deltas are oriented so "
                 "positive favours the model: delta_rmse = RMSE_vina - RMSE_model and "
                 "delta_ci = CI_model - CI_vina. Row bootstrap resamples pairs; cluster "
                 "bootstrap resamples kinase targets because pairs inside a target are "
                 "dependent."),
        "learned_cells": int(len(learned)),
        "point_wins": learned.point_win.value_counts().to_dict(),
        "ci_wins_point_estimate": {
            "vina": sorted(set(learned[learned.point_win == "vina"].protocol)),
            "model": sorted(set(learned[learned.point_win == "model"].protocol)),
        },
        "significant_ci_rowbootstrap": {
            "vina": int((learned.ci_verdict_row == "vina").sum()),
            "model": int((learned.ci_verdict_row == "model").sum()),
            "undecided": int((learned.ci_verdict_row == "undecided").sum()),
            "rows": learned[learned.ci_verdict_row != "undecided"][
                ["protocol", "model", "delta_ci", "delta_ci_lo_row", "delta_ci_hi_row"]
            ].round(4).to_dict("records"),
        },
        "significant_ci_clusterbootstrap": {
            "vina": int((learned.ci_verdict_cluster == "vina").sum()),
            "model": int((learned.ci_verdict_cluster == "model").sum()),
            "undecided": int((learned.ci_verdict_cluster == "undecided").sum()),
            "rows": learned[learned.ci_verdict_cluster != "undecided"][
                ["protocol", "model", "delta_ci", "delta_ci_lo_cluster", "delta_ci_hi_cluster"]
            ].round(4).to_dict("records"),
        },
        "rmse_significant_cells": int((learned.rmse_verdict_row != "undecided").sum()),
        "sign_test_below_0.05": int((learned.sign_test_p < 0.05).sum()),
        "min_sign_test_p": float(learned.sign_test_p.min()),
    }
    OUT.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
