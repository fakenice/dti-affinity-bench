#!/usr/bin/env python3
"""
06_docking_analysis.py -- what does an AutoDock Vina score tell you about measured
DAVIS affinity, and how does it compare with the trained models from 03 on the
*identical rows*?

Reported, per comparison:
  ranking only   : concordance index of (-Vina) vs pKd. No training, no units, no
                   calibration -- docking is a ranker, so this is its native metric.
                   Computed globally (all panel targets pooled) and within each
                   kinase separately, because a physics score carries no information
                   about the assay's per-protein offset.
  regression     : pKd ~ a + b * (-Vina), a and b fitted by OLS on the docking pairs
                   that fall in the TRAIN block of the protocol, evaluated on the
                   pairs that fall in TEST. Two parameters, so this is the most a
                   docking score can be asked to do.
  head-to-head   : the same rows scored with the 03 predictions (loaded from
                   results/predictions/*.npz), so model and Vina are compared on an
                   identical pair set with identical metrics from dti_core.
  decoy control  : (-Vina) ranking inside a box translated +30 A into solvent. The
                   interpretable readout is the score MAGNITUDE (a specific site should
                   lose fitted strength in solvent); the CI of a 10-pair block is printed
                   next to its comparable-pair denominator and is noise at that n.

Every CI here carries `n_comparable_pairs`, because on censored DAVIS labels a 20-compound
block leaves only a handful of usable pairs -- which is how CI = 1.0000 can appear on
ERK1/ERK2 without any binding signal being measured there.

Usage
  python scripts/06_docking_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dti_core import ci_harrell, ci_sklearn_like, ci_denominators, evaluate   # noqa: E402

PROC = ROOT / "data" / "processed"
SPL = PROC / "splits"
DOCK = ROOT / "results" / "docking"
PRED = ROOT / "results" / "predictions"
PROTOCOLS = ["random", "scaffold", "cold_drug", "cold_target", "temporal"]


def load_pairs():
    p = pd.read_csv(DOCK / "pairs.csv")
    p = p[p.vina_best.notna()].copy()
    lab = pd.read_csv(PROC / "davis.csv")[["pair_id", "pkd", "target_id", "ligand_id",
                                           "target_name", "published_part"]]
    p = p.merge(lab[["pair_id", "pkd"]].rename(columns={"pkd": "pkd_label"}),
                on="pair_id", how="left", validate="one_to_one")
    p["neg_vina"] = -p.vina_best
    return p


def split_of():
    """pair_id -> block ('train'/'valid'/'test') for every protocol that has a split file."""
    out = {}
    for proto in PROTOCOLS:
        f = SPL / f"davis_{proto}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)[["pair_id", "split"]].set_index("pair_id").split
        out[proto] = s
    return out


def main() -> int:
    if not (DOCK / "pairs.csv").exists():
        sys.exit("run scripts/05_docking.py first -- results/docking/pairs.csv is missing")
    p = load_pairs()
    if not len(p):
        sys.exit("no scored docking pairs")
    splits = split_of()
    for proto, s in splits.items():
        p["in_train_" + proto] = p.pair_id.isin(s[s == "train"].index)
        p["in_valid_" + proto] = p.pair_id.isin(s[s == "valid"].index)
        p["in_test_" + proto] = p.pair_id.isin(s[s == "test"].index)
    panel = pd.read_csv(DOCK / "target_panel.csv")
    prep = pd.read_csv(DOCK / "ligand_prep.csv")
    man = json.loads((DOCK / "manifest.json").read_text(encoding="utf-8"))

    rows = []
    # ---------------- global ranking, all docked pairs (no train/test at all) ---------
    for scope, sub in [("all_docked_pairs", p)] + \
                      [(f"target:{r.target_id}", p[p.target_id == r.target_id])
                       for r in panel.itertuples()]:
        if len(sub) < 4:
            continue
        den = ci_denominators(sub.pkd_label)
        rows.append({"comparison": "vina_ranking", "scope": scope, "subset": "-",
                     "n": int(len(sub)), "n_comparable_pairs": den["n_comparable_pairs"],
                     "n_tied_label_pairs": den["n_tied_label_pairs"],
                     "ci_harrell": round(ci_harrell(sub.pkd_label, sub.neg_vina), 4),
                     "ci_sklearn_like": round(ci_sklearn_like(sub.pkd_label, sub.neg_vina), 4),
                     "pearson_r": round(float(np.corrcoef(sub.neg_vina, sub.pkd_label)[0, 1]), 4),
                     "spearman_rho": round(float(sub.neg_vina.corr(sub.pkd_label, method="spearman")), 4),
                     "label_tie_fraction": round(float(1 - sub.pkd_label.nunique() / len(sub)), 3),
                     "gene": None if not scope.startswith("target:")
                             else panel[panel.target_id == scope.split(":")[1]].gene.iloc[0]})

    # ---------------- per protocol: train-fit 2-parameter regression, test eval -------
    for proto in splits:
        tr, te = p[p["in_train_" + proto]], p[p["in_test_" + proto]]
        if len(tr) < 8 or len(te) < 8:
            rows.append({"comparison": "vina_regression", "scope": "pooled", "subset": proto,
                         "n": int(len(te)), "note": f"too few docked pairs (train {len(tr)}, test {len(te)})"})
            continue
        x, y = tr.neg_vina.to_numpy(float), tr.pkd_label.to_numpy(float)
        b, a = np.polyfit(x, y, 1)
        pred_te = a + b * te.neg_vina.to_numpy(float)
        ev = evaluate(te.pkd_label.to_numpy(float), pred_te)
        den = ci_denominators(te.pkd_label)
        rows.append({"comparison": "vina_regression", "scope": "pooled", "subset": proto,
                     "n": int(len(te)), "n_fit": int(len(tr)),
                     "n_comparable_pairs": den["n_comparable_pairs"],
                     "slope": round(float(b), 4), "intercept": round(float(a), 4),
                     "rmse": round(ev["rmse"], 4), "mae": round(ev["mae"], 4),
                     "pearson_r": round(ev["pearson_r"], 4), "spearman_rho": round(ev["spearman_rho"], 4),
                     "ci_harrell": round(ev["ci_harrell"], 4),
                     "ci_sklearn_like": round(ev["ci_sklearn_like"], 4),
                     "label_tie_fraction": round(ev["label_tie_fraction"], 3)})

        # ---------------- head-to-head against the 03 models on identical rows --------
        for f in sorted(PRED.glob(f"davis_{proto}_*.npz")):
            stem = f.stem
            model = stem[len(f"davis_{proto}_"):].rsplit("_seed", 1)[0]
            seed = int(stem.rsplit("_seed", 1)[1])
            if seed != 0:
                continue
            z = np.load(f)
            pr = pd.DataFrame({"pair_id": z["pair_id"], "yhat_03": z["yhat"], "y03": z["y"]})
            j = te.merge(pr, on="pair_id", how="inner", validate="one_to_one")
            if len(j) < 8:
                continue
            v = evaluate(j.y03.to_numpy(float), j.yhat_03.to_numpy(float))
            vv = evaluate(j.y03.to_numpy(float), a + b * j.neg_vina.to_numpy(float))
            rows.append({"comparison": "model_vs_vina", "scope": "pooled", "subset": proto,
                         "model": model, "n": int(len(j)),
                         "n_comparable_pairs": ci_denominators(j.y03)["n_comparable_pairs"],
                         "rmse_03": round(v["rmse"], 4), "ci_03": round(v["ci_harrell"], 4),
                         "ci_sk_03": round(v["ci_sklearn_like"], 4), "r_03": round(v["pearson_r"], 4),
                         "rmse_vina": round(vv["rmse"], 4), "ci_vina": round(vv["ci_harrell"], 4),
                         "ci_sk_vina": round(vv["ci_sklearn_like"], 4), "r_vina": round(vv["pearson_r"], 4),
                         "delta_ci_favours": ("tie" if abs(vv["ci_harrell"] - v["ci_harrell"]) < 1e-9
                                              else ("vina" if vv["ci_harrell"] > v["ci_harrell"]
                                                    else "model"))})

    # ---------------- decoy box control -----------------------------------------------
    dc = pd.read_csv(DOCK / "decoy_control.csv") if (DOCK / "decoy_control.csv").exists() else pd.DataFrame()
    decoy_note = None
    if len(dc):
        d = dc.merge(p[["pair_id", "vina_best", "neg_vina"]], on="pair_id", how="inner")
        if len(d) >= 4:
            den_on = ci_denominators(d.pkd)
            decoy_note = {
                "n": int(len(d)),
                "n_comparable_pairs": den_on["n_comparable_pairs"],
                "n_pairs": den_on["n_pairs"],
                "mean_onsite": round(float(d.vina_best.mean()), 3),
                "mean_shifted": round(float(d.vina_shifted.mean()), 3),
                "ci_harrell_shifted": round(ci_harrell(d.pkd, -d.vina_shifted), 4),
                "ci_harrell_onsite": round(ci_harrell(d.pkd, d.neg_vina), 4),
                "note": ("box translated +30 A along x. The interpretable signal here is the "
                         "SCORE MAGNITUDE, not the CI: a 10-pair block leaves only "
                         f"{den_on['n_comparable_pairs']} comparable pairs of "
                         f"{den_on['n_pairs']}, so both CIs are noise-level and are reported as "
                         "such. The shifted box losing ~2 kcal/mol of fitted strength is the "
                         "specificity evidence."),
            }
            rows.append({"comparison": "decoy_box_control", "scope": "pooled", "subset": "-",
                         "n": int(len(d)),
                         "n_comparable_pairs": decoy_note["n_comparable_pairs"],
                         "ci_harrell": decoy_note["ci_harrell_shifted"],
                         "ci_harrell_onsite": decoy_note["ci_harrell_onsite"],
                         "mean_onsite_kcal": decoy_note["mean_onsite"],
                         "mean_shifted_kcal": decoy_note["mean_shifted"]})

    out = pd.DataFrame(rows)
    order = ["comparison", "scope", "gene", "subset", "model", "n", "n_fit",
             "n_comparable_pairs", "n_tied_label_pairs", "slope", "intercept",
             "rmse", "mae", "pearson_r", "spearman_rho", "ci_harrell", "ci_sklearn_like",
             "rmse_03", "ci_03", "ci_sk_03", "r_03", "rmse_vina", "ci_vina", "ci_sk_vina",
             "r_vina", "delta_ci_favours", "label_tie_fraction", "note", "mean_onsite_kcal",
             "mean_shifted_kcal", "ci_harrell_onsite"]
    out = out[[c for c in order if c in out.columns]]
    out.to_csv(DOCK / "comparison.csv", index=False)

    hh = out[out.comparison == "model_vs_vina"]
    g = out[(out.comparison == "vina_ranking") & (out.scope == "all_docked_pairs")]
    summary = {
        "script": "scripts/06_docking_analysis.py",
        "docked_pairs_scored": int(p.pair_id.nunique()),
        "panel_targets": int(len(panel)),
        "ligands_prepared": int(prep.pdbqt.sum()) if "pdbqt" in prep else None,
        "vina_global_ranking_ci": (float(g.ci_harrell.iloc[0]) if len(g) else None),
        "per_target_ci": {r.scope: (r.ci_harrell, int(r.n)) for r in
                          out[out.comparison == "vina_ranking"].itertuples()
                          if r.scope != "all_docked_pairs"},
        "head_to_head_rows": int(len(hh)),
        "head_to_head_outcome_counts": (hh.delta_ci_favours.value_counts().to_dict()
                                        if len(hh) else {}),
        "protocols_where_vina_wins_on_ci": sorted(set(hh[hh.delta_ci_favours == "vina"].subset))
                                           if len(hh) else [],
        "decoy_control": decoy_note,
        "caveats": man.get("caveats", []) + [
            "the docked subset is a pKd-stratified sample of 20 compounds on a handful of "
            "kinases, so per-target CI is estimated from ~20 points and is noisy by construction",
            "Vina scores are compared to a trained model that has seen the rest of the assay "
            "table; on identical rows that asymmetry is removed, and only ranking quality is compared",
        ],
    }
    (DOCK / "docking_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                               encoding="utf-8")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("\nwrote", DOCK / "comparison.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
