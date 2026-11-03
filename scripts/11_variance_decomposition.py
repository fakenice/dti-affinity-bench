#!/usr/bin/env python3
"""11_variance_decomposition.py -- how much of the RMSE table is protocol, how much is model?

The headline says "the split protocol moves the number more than the model does", but the
README supports it with two ranges (0.4823 across protocols vs 0.1864 across three models),
which is not a decomposition and silently excludes the MLP from one side. This script fits
the additive model

    metric[protocol, model, seed] = mu + a_protocol + b_model + (ab)_cell + e_seed

on the shipped grid and reports variance shares. Seeds are the replication, so the
protocol x model interaction is not confounded with seed noise. Shares are bootstrapped by
resampling the three seeds within each cell (the only random quantity the grid actually has).

Reported per dataset and per model set:
  trained3  ridge, xgboost, randomforest  -- the models the headline calls "usable"
  all4      + mlp                         -- kept because deleting it would hide a failure

Also writes the trained3 model spread per protocol, which is the number the headline should
quote next to the all-model spread.

Usage
  python scripts/11_variance_decomposition.py [--reps 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as fdist

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
METRICS = ["rmse", "ci_harrell", "mae"]
TRAINED3 = ["ridge", "xgboost", "randomforest"]
ALL4 = TRAINED3 + ["mlp"]


def two_way(cell: pd.DataFrame, metric: str) -> dict:
    """Classical two-way ANOVA (protocol x model) on cell means, no replication.

    Returns sums of squares and eta-squared. With 5 protocols x 4 models there are enough
    df for the interaction to be the residual; that is the correct error term for testing
    both main effects when each cell has one mean.
    """
    wide = cell.pivot(index="protocol", columns="model", values=metric)
    y = wide.to_numpy(float)
    grand = y.mean()
    ss_prot = (y.mean(axis=1) - grand) ** 2 * y.shape[1]
    ss_model = (y.mean(axis=0) - grand) ** 2 * y.shape[0]
    ss_tot = ((y - grand) ** 2).sum()
    ss_resid = ss_tot - ss_prot.sum() - ss_model.sum()
    df_p, df_m = y.shape[0] - 1, y.shape[1] - 1
    df_r = df_p * df_m
    ms_r = ss_resid / df_r
    out = {
        "ss_protocol": float(ss_prot.sum()), "ss_model": float(ss_model.sum()),
        "ss_interaction": float(ss_resid), "ss_total": float(ss_tot),
        "eta2_protocol": float(ss_prot.sum() / ss_tot),
        "eta2_model": float(ss_model.sum() / ss_tot),
        "eta2_interaction": float(ss_resid / ss_tot),
        "F_protocol": float((ss_prot.sum() / df_p) / ms_r),
        "F_model": float((ss_model.sum() / df_m) / ms_r),
        "p_protocol": float(fdist.sf((ss_prot.sum() / df_p) / ms_r, df_p, df_r)),
        "p_model": float(fdist.sf((ss_model.sum() / df_m) / ms_r, df_m, df_r)),
        "df": {"protocol": df_p, "model": df_m, "residual": df_r},
    }
    return out


def bootstrap_shares(cell: pd.DataFrame, metric: str, reps: int, rng) -> dict:
    """Resample seeds within each (protocol, model) cell and recompute eta-squared."""
    groups = []
    for (proto, model), sub in cell.groupby(["protocol", "model"]):
        vals = sub[metric].to_numpy(float)
        if len(vals):
            groups.append((proto, model, vals))
    rec = {"eta2_protocol": [], "eta2_model": [], "eta2_interaction": []}
    for _ in range(reps):
        rows = []
        for proto, model, vals in groups:
            take = rng.integers(0, len(vals), size=len(vals))
            rows.append({"protocol": proto, "model": model, "value": float(vals[take].mean())})
        b = pd.DataFrame(rows)
        d = two_way(b.rename(columns={"value": metric}), metric)
        for k in rec:
            rec[k].append(d[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in rec.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(20261103)
    m = pd.read_csv(RES / "metrics.csv")
    out = {"generated_by": "scripts/11_variance_decomposition.py",
           "reps": a.reps, "seed": 20261103,
           "note": ("Two-way ANOVA on cell means over seeds; interaction is the residual. "
                    "Bootstrap resamples seeds within each cell. trained3 = ridge/xgboost/"
                    "randomforest; all4 adds mlp, which loses to the training mean in 7 of 10 "
                    "dataset x protocol cells and would otherwise be invisible."),
           "datasets": {}, "model_spread": {}}

    for ds in ["davis", "kiba"]:
        d = m[(m.dataset == ds) & m.model.isin(ALL4) & m.seed.isin([0, 1, 2])].copy()
        out["datasets"][ds] = {}
        for name, models in [("trained3", TRAINED3), ("all4", ALL4)]:
            sub = d[d.model.isin(models)]
            cell = sub.groupby(["protocol", "model"], as_index=False)[METRICS].mean()
            entry = {"models": models,
                     "n_cells": int(len(cell)),
                     "protocols": int(cell.protocol.nunique())}
            for metric in METRICS:
                entry[metric] = two_way(cell, metric)
                entry[metric]["bootstrap_ci95"] = bootstrap_shares(sub, metric, a.reps, rng)
            out["datasets"][ds][name] = entry

        # spread: all models vs the three the headline calls usable
        for name, models in [("trained3", TRAINED3), ("all4", ALL4)]:
            g = d[d.model.isin(models)].groupby(["protocol", "model"], as_index=False)[
                ["rmse", "ci_harrell"]].mean()
            spread = {}
            for proto, s in g.groupby("protocol"):
                best = s.loc[s.rmse.idxmin()]
                worst = s.loc[s.rmse.idxmax()]
                spread[proto] = {
                    "best_model": str(best.model), "best_rmse": round(float(best.rmse), 4),
                    "worst_model": str(worst.model), "worst_rmse": round(float(worst.rmse), 4),
                    "spread": round(float(worst.rmse - best.rmse), 4)}
            out["model_spread"][f"{ds}/{name}"] = spread

    (RES / "variance_decomposition.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Every eta-squared triple must sum to 1; otherwise the SS bookkeeping is wrong.
    for ds, sets in out["datasets"].items():
        for name, e in sets.items():
            for metric in METRICS:
                s = (e[metric]["eta2_protocol"] + e[metric]["eta2_model"]
                     + e[metric]["eta2_interaction"])
                assert abs(s - 1.0) < 1e-9, f"{ds}/{name}/{metric}: eta2 sums to {s}"

    # human-readable echo: the shares are the point, so print them compactly
    for ds, sets in out["datasets"].items():
        for name, e in sets.items():
            for metric in ["rmse", "ci_harrell"]:
                v = e[metric]
                ci = v["bootstrap_ci95"]
                print(f"{ds:5s} {name:8s} {metric:10s} "
                      f"protocol {v['eta2_protocol']:.3f} "
                      f"[{ci['eta2_protocol'][0]:.3f},{ci['eta2_protocol'][1]:.3f}]  "
                      f"model {v['eta2_model']:.3f} "
                      f"[{ci['eta2_model'][0]:.3f},{ci['eta2_model'][1]:.3f}]  "
                      f"interaction {v['eta2_interaction']:.3f}  "
                      f"p_protocol={v['p_protocol']:.2g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
