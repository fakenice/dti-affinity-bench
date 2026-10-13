#!/usr/bin/env python3
"""
07_report.py -- turn everything under results/ into figures, one markdown report and a
machine-readable summary.json.

Read-only with respect to every other artefact: it never refits, it only reports what
00-06 wrote, and it degrades gracefully (a missing docking run or a missing uncertainty
run drops that section with a printed reason instead of failing the whole report).

Usage
    python scripts/07_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
PROC = ROOT / "data" / "processed"
FIG.mkdir(parents=True, exist_ok=True)

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from dti_core import ci_denominators                   # noqa: E402  (pair counts, fig4)

PROTO_ORDER = ["random", "scaffold", "cold_drug", "cold_target", "temporal"]
MODEL_ORDER = ["mean", "mean_ligand_train", "mean_target_train", "ridge", "mlp",
               "randomforest", "xgboost"]
COL = {"random": "#4477AA", "scaffold": "#228833", "cold_drug": "#CCBB44",
       "cold_target": "#AA3377", "temporal": "#66CCEE"}


def exists(p):
    ok = Path(p).exists()
    if not ok:
        print(f"  [skip figure] missing {Path(p).relative_to(ROOT)}")
    return ok


def md_table(df, floatfmt="{:.4f}"):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in d.itertuples(index=False))
    return "\n".join([head, sep, body])


def fig1_leakage(au):
    piv = au.pivot_table(index=["dataset", "protocol"],
                         values=["test_pairs_with_near_duplicate_ligand_ge0.7",
                                 "test_scaffold_present_in_train",
                                 "test_targets_absent_from_train"], aggfunc="mean")
    piv = piv.reindex(pd.MultiIndex.from_product([["davis", "kiba"], PROTO_ORDER],
                                                 names=["dataset", "protocol"])).dropna(how="all")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(piv))
    w = 0.26
    for k, (col, lab) in enumerate([
            ("test_pairs_with_near_duplicate_ligand_ge0.7", "test ligand has a >=0.7 Tanimoto twin in train"),
            ("test_scaffold_present_in_train", "test scaffold already seen in train"),
            ("test_targets_absent_from_train", "test targets unseen in train")]):
        ax.barh(y + (k - 1) * w, (piv[col] * 100).to_numpy(float), height=w, label=lab,
                color=["#CC3311", "#CCBB44", "#228833"][k])
    ax.set_yticks(y, [f"{d}/{p}" for d, p in piv.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of test rows")
    ax.set_title("What each split protocol actually holds out (leakage audit)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_leakage_by_protocol.png", dpi=150)
    plt.close(fig)


def fig2_heatmap(m, dataset, value, fname, title, vmin=None, vmax=None, better_low=True):
    piv = (m[m.dataset == dataset].groupby(["model", "protocol"])[value].mean()
             .unstack("protocol").reindex(index=[x for x in MODEL_ORDER if x in m.model.unique()],
                                          columns=[c for c in PROTO_ORDER if c in m.protocol.unique()]))
    arr = piv.to_numpy(float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        # the grid is still filling in, or this dataset/protocol combo has no rows yet:
        # draw nothing rather than crash the whole report over one empty panel
        print(f"  [fig2] skip {fname}: no {value} rows for {dataset}")
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    # 'rocket'/'mako' are seaborn names and do not exist in matplotlib's own registry --
    # cividis is (it is also colour-vision-deficiency safe). Reversed when low values are
    # the good ones, so light = good in either direction.
    im = ax.imshow(arr, cmap="cividis_r" if better_low else "cividis",
                   vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=8)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                # pick the label colour from the actual cell luminance, not from an assumed
                # ramp direction -- white on cividis' pale-yellow end is unreadable
                rgb = im.cmap((arr[i, j] - lo) / (hi - lo + 1e-9))[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if lum < 0.55 else "black")
    ax.set_title(f"{title} -- {dataset}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)
    return fname


def fig3_ci_calibers(m):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, ds in zip(axes, ["davis", "kiba"]):
        s = m[(m.dataset == ds) & m.model.isin(MODEL_ORDER)]
        if not len(s):
            ax.set_title(f"{ds} -- no grid rows yet")
            continue
        for p in PROTO_ORDER:
            ss = s[s.protocol == p]
            ax.scatter(ss.ci_harrell, ss.rmse, c=COL[p], s=34, label=p,
                       edgecolor="k", linewidth=.3)
        ax.set_xlabel("concordance index, Harrell calibre\n(label ties dropped, pred ties = 0.5)")
        ax.axhline(np.nanmin(s.rmse), color="grey", lw=.6, ls=":")
        ax.set_title(ds)
        ax.invert_xaxis()
    axes[0].set_ylabel("test RMSE (label units)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Ranking quality and error magnitude disagree -- that is the point of the audit",
                 y=1.05, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig3_rmse_vs_ci.png", dpi=150)
    plt.close(fig)


def fig4_ties(m):
    """Three different things that all get called "ties", measured on every test block.

    Conflating them is the cheapest way to mis-read a concordance index, so the figure keeps
    them apart and writes the numbers to results/label_tie_measures.csv:

      bars   -- % of label PAIRS that are exact ties. These are exactly the pairs
                `ci_harrell` removes from its denominator, so this is the number that says how
                much of the ranking question is answerable at all.
      dots   -- % of ROWS sitting on the modal label value. For DAVIS that value is the assay
                ceiling pKd 5.0, so this is the 69.64 % censoring share quoted in the text.
      crosses-- 1 - distinct labels / rows, i.e. the `label_tie_fraction` column stored in
                results/metrics.csv. It is a REDUNDANCY measure, not a pair measure: it
                saturates near 1 on any large block (KIBA included), which is why plotting it
                alone was misleading.
    """
    rows = []
    for ds in ["davis", "kiba"]:
        ycol = "pkd" if ds == "davis" else "kiba_score"
        base = pd.read_csv(PROC / f"{ds}.csv")[["pair_id", ycol]].set_index("pair_id")
        for p in PROTO_ORDER:
            f = PROC / "splits" / f"{ds}_{p}.csv"
            if not f.exists():
                continue
            sp_df = pd.read_csv(f)
            y = base.loc[sp_df.loc[sp_df.split == "test", "pair_id"], ycol].to_numpy(float)
            if not len(y):
                continue
            den = ci_denominators(y)
            counts = np.unique(y, return_counts=True)[1]
            sel = m[(m.dataset == ds) & (m.protocol == p)]
            rows.append({"dataset": ds, "protocol": p, "n_test": int(len(y)),
                         "n_distinct_labels": int(len(counts)),
                         "pct_label_pairs_tied": round(100 * den["n_tied_label_pairs"]
                                                       / max(den["n_pairs"], 1), 2),
                         "pct_rows_on_modal_label": round(100 * counts.max() / len(y), 2),
                         "pct_redundancy_metrics_csv": round(
                             100 * float(sel.label_tie_fraction.mean()), 2) if len(sel) else None})
    t = pd.DataFrame(rows)
    t.to_csv(RES / "label_tie_measures.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    idx = np.arange(len(t))
    ax.barh(idx - 0.22, t.pct_label_pairs_tied, height=0.42, color="#BB5566",
            label="% of label PAIRS that are exact ties (what ci_harrell drops)")
    ax.scatter(t.pct_rows_on_modal_label, idx + 0.22, marker="o", s=42, color="#004488",
               zorder=3, label="% of ROWS on the modal label (DAVIS: the pKd 5.0 ceiling)")
    ax.scatter(t.pct_redundancy_metrics_csv, idx, marker="x", s=46, color="#555555",
               zorder=3, label="1 - distinct/rows  (= metrics.csv label_tie_fraction)")
    ax.set_yticks(idx, [f"{r.dataset}/{r.protocol}" for r in t.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("percent")
    ax.set_title("How 'tied' are the labels? Three answers, and only the first one is the "
                 "concordance denominator")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", lw=.4, alpha=.4)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_label_ties.png", dpi=150)
    plt.close(fig)
    return t


def fig5_conformal(cf):
    """cf = results/uncertainty/coverage_curve.csv (long form: nominal / empirical / method).

    Solid lines are the constant-width split-conformal intervals, dashed the normalised
    variant, both for the XGBoost wrapper at seed 0; the dotted diagonal is what a valid
    1-alpha guarantee would trace.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, ds in zip(axes, ["davis", "kiba"]):
        for p in PROTO_ORDER:
            for method, ls, lw, msz, lab in [("conformal", "-", 1.7, 4, p),
                                             ("conformal_norm", "--", 1.2, 3, None)]:
                ss = cf[(cf.dataset == ds) & (cf.protocol == p) & (cf.model == "xgboost")
                        & (cf.method == method)].sort_values("nominal")
                if not len(ss) or ss.empirical.isna().all():
                    continue
                ax.plot(ss.nominal.to_numpy(float), ss.empirical.to_numpy(float), ls,
                        color=COL[p], lw=lw, marker="o", ms=msz, label=lab)
        ax.plot([0.5, 0.95], [0.5, 0.95], ls=":", c="k", lw=.9, label="ideal 1-alpha")
        ax.set_xlabel("nominal coverage")
        ax.set_title(ds)
    axes[0].set_ylabel("measured coverage of the test block")
    axes[0].legend(fontsize=7)
    fig.suptitle("Measured split-conformal coverage (XGBoost wrapper, seed 0) -- "
                 "solid = constant width, dashed = normalised", y=1.05, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig5_conformal_coverage.png", dpi=150)
    plt.close(fig)


def fig6_docking(dp, panel):
    d = dp[dp.vina_best.notna()].copy()
    if not len(d):
        print("  [skip figure] no scored docking pairs")
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    d["neg_vina"] = -d.vina_best
    for r in panel.itertuples():
        s = d[d.target_id == r.target_id]
        if len(s):
            ax.scatter(s.neg_vina, s.pkd, s=30, alpha=.85, edgecolor="k", lw=.3,
                       label=f"{r.gene} {r.pdb_id}_{r.chain}")
    b, a = np.polyfit(d.neg_vina, d.pkd, 1)
    xs = np.linspace(float(d.neg_vina.min()), float(d.neg_vina.max()), 20)
    r = float(np.corrcoef(d.neg_vina, d.pkd)[0, 1])
    rho = float(d.neg_vina.corr(d.pkd, method="spearman"))
    ax.plot(xs, a + b * xs, c="k", lw=1.2, label=f"OLS r={r:.3f}, rho={rho:.3f}")
    ax.set_xlabel("-Vina score (kcal/mol)")
    ax.set_ylabel("measured pKd")
    ax.set_title("Every docked pair: -Vina score vs measured pKd (flat cloud = the assay ceiling)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_docking_vs_pkd.png", dpi=150)
    plt.close(fig)


def readme_numbers(m, cfwide, comp):
    """Every figure the README quotes, computed here so the prose cannot drift from the
    tables it came from. Written to results/readme_numbers.json."""
    out = {"generated_by": "scripts/07_report.py"}
    learned = ["ridge", "xgboost", "randomforest", "mlp"]
    if m is not None:
        g = (m[m.model.isin(learned)]
              .groupby(["dataset", "protocol", "model"])[
                  ["rmse", "ci_harrell", "ci_sklearn_like", "pearson_r"]]
              .mean().reset_index())
        floors = m[m.model == "mean"].groupby(["dataset", "protocol"]).rmse.mean().to_dict()
        if len(g):
            best = g.loc[g.groupby(["dataset", "protocol"]).rmse.idxmin()]
            out["best_learned_per_protocol"] = {
                f"{r.dataset}/{r.protocol}": {
                    "model": r.model, "rmse": round(float(r.rmse), 4),
                    "ci_harrell": round(float(r.ci_harrell), 4),
                    "ci_sklearn_like": round(float(r.ci_sklearn_like), 4),
                    "null_rmse": (None if (r.dataset, r.protocol) not in floors else
                                  round(float(floors[(r.dataset, r.protocol)]), 4)),
                    "beats_null": (None if (r.dataset, r.protocol) not in floors else
                                   bool(r.rmse < floors[(r.dataset, r.protocol)]))}
                for r in best.itertuples()}
            spread = {}
            for ds in g.dataset.unique():
                s = best[best.dataset == ds].set_index("protocol").rmse
                spread[ds] = {"best_of": round(float(s.min()), 4),
                              "worst_of": round(float(s.max()), 4),
                              "range": round(float(s.max() - s.min()), 4),
                              "easiest": str(s.idxmin()), "hardest": str(s.idxmax())}
            out["protocol_spread_of_the_best_model"] = spread
            out["model_spread_within_protocol"] = {
                f"{ds}/{pr}": round(float(sub.rmse.max() - sub.rmse.min()), 4)
                for (ds, pr), sub in g.groupby(["dataset", "protocol"])}
        d = (m.ci_harrell - m.ci_sklearn_like).abs()
        if len(d.dropna()):
            i = int(d.idxmax())
            out["ci_caliber_gap_max"] = {
                "abs_diff": round(float(d.loc[i]), 4),
                "row": f"{m.loc[i].dataset}/{m.loc[i].protocol}/{m.loc[i].model}/seed{int(m.loc[i].seed)}",
                "ci_harrell": round(float(m.loc[i].ci_harrell), 4),
                "ci_sklearn_like": round(float(m.loc[i].ci_sklearn_like), 4)}
        losers = []
        for (ds, pr), sub in g.groupby(["dataset", "protocol"]):
            fl = floors.get((ds, pr))
            if fl is None:
                continue
            losers += [{"dataset": ds, "protocol": pr, "model": r.model,
                        "rmse": round(float(r.rmse), 4), "null": round(float(fl), 4)}
                       for r in sub.itertuples() if r.rmse > fl]
        out["learned_models_worse_than_the_global_mean"] = losers
    if cfwide is not None and len(cfwide):
        c = cfwide[cfwide.method == "conformal"]
        n = cfwide[cfwide.method == "conformal_norm"]
        w = (None if not len(c) else
             {"rows": int(len(c)),
              "rmse_all_match_03": bool(cfwide.rmse_matches_03.astype(bool).all()),
              "cov90_min_mean_max": [round(float(c.cov90.min()), 4),
                                     round(float(c.cov90.mean()), 4),
                                     round(float(c.cov90.max()), 4)],
              "coverage_error_mean": round(float(c.coverage_error.mean()), 4),
              "width90_over_label_sd_mean": round(float(c.width90_over_label_sd.mean()), 4),
              "worst_row": None if not len(c) else
              f"{c.loc[c.coverage_error.abs().idxmax()].dataset}/"
              f"{c.loc[c.coverage_error.abs().idxmax()].protocol}/"
              f"{c.loc[c.coverage_error.abs().idxmax()].model}"})
        if len(n) and c is not None and len(c):
            w["normalised_vs_constant_mean_width90"] = [round(float(n.width90.mean()), 4),
                                                        round(float(c.width90.mean()), 4)]
            w["normalised_vs_constant_mean_cov90"] = [round(float(n.cov90.mean()), 4),
                                                      round(float(c.cov90.mean()), 4)]
        out["conformal"] = w
    if comp is not None and len(comp):
        rk = comp[comp.comparison == "vina_ranking"]
        rr = comp[comp.comparison == "vina_regression"]
        hh = comp[comp.comparison == "model_vs_vina"]
        hl = hh[hh.model.isin(learned)] if len(hh) else hh
        dc = comp[comp.comparison == "decoy_box_control"]
        enough = rk[rk.n_comparable_pairs >= 50]
        out["docking"] = {
            "global_ranking": (None if not len(rk) else {
                "ci_harrell": float(rk[rk.scope == "all_docked_pairs"].ci_harrell.iloc[0])
                if (rk.scope == "all_docked_pairs").any() else None,
                "n_comparable_pairs": int(rk[rk.scope == "all_docked_pairs"]
                                          .n_comparable_pairs.iloc[0])
                if (rk.scope == "all_docked_pairs").any() else None}),
            "per_target": [{"target": r.scope, "n": int(r.n),
                            "n_comparable_pairs": None if pd.isna(r.n_comparable_pairs)
                            else int(r.n_comparable_pairs),
                            "ci_harrell": None if pd.isna(r.ci_harrell) else float(r.ci_harrell)}
                           for r in rk.itertuples() if r.scope != "all_docked_pairs"],
            "per_target_with_ge50_comparable_pairs": [
                {"target": r.scope, "n_comparable_pairs": int(r.n_comparable_pairs),
                 "ci_harrell": float(r.ci_harrell)} for r in enough.itertuples()],
            "regression": [{"protocol": r.subset, "n_test": int(r.n),
                            "n_fit": None if pd.isna(r.n_fit) else int(r.n_fit),
                            "rmse": None if pd.isna(r.rmse) else float(r.rmse),
                            "ci_harrell": None if pd.isna(r.ci_harrell) else float(r.ci_harrell),
                            "n_comparable_pairs": None if pd.isna(r.n_comparable_pairs)
                            else int(r.n_comparable_pairs)} for r in rr.itertuples()],
            "head_to_head": {"rows": int(len(hh)),
                             "protocols_where_vina_wins_ci": sorted(set(
                                 hh[hh.delta_ci_favours == "vina"].subset)) if len(hh) else [],
                             "min_n_comparable_pairs": (None if hh.empty or
                                                        hh.n_comparable_pairs.isna().all()
                                                        else int(hh.n_comparable_pairs.min()))},
            # the deterministic nulls (mean / mean_ligand_train / mean_target_train) sit in the
            # same table; the README quotes trained-model-vs-Vina, so count learned rows only.
            "head_to_head_learned_only": {
                "rows": int(len(hl)),
                "outcome_counts": (hl.delta_ci_favours.value_counts().to_dict()
                                   if len(hl) else {}),
                "vina_wins_in": sorted(set(hl[hl.delta_ci_favours == "vina"].subset))
                               if len(hl) else []},
            "decoy": (None if not len(dc) else {
                "mean_onsite_kcal": float(dc.iloc[0].mean_onsite_kcal),
                "mean_shifted_kcal": float(dc.iloc[0].mean_shifted_kcal),
                "n_comparable_pairs": None if pd.isna(dc.iloc[0].n_comparable_pairs)
                else int(dc.iloc[0].n_comparable_pairs)})}
    return out


def main() -> int:
    built, missing = [], []
    m = pd.read_csv(RES / "metrics.csv") if exists(RES / "metrics.csv") else None
    au = pd.read_csv(RES / "leakage_audit.csv") if exists(RES / "leakage_audit.csv") else None
    cf = pd.read_csv(RES / "uncertainty" / "coverage_curve.csv") \
        if exists(RES / "uncertainty" / "coverage_curve.csv") else None
    cfwide = pd.read_csv(RES / "uncertainty" / "conformal.csv") \
        if (RES / "uncertainty" / "conformal.csv").exists() else None
    dp = pd.read_csv(RES / "docking" / "pairs.csv") \
        if exists(RES / "docking" / "pairs.csv") else None
    comp = pd.read_csv(RES / "docking" / "comparison.csv") \
        if exists(RES / "docking" / "comparison.csv") else None
    panel = pd.read_csv(RES / "docking" / "target_panel.csv") \
        if (RES / "docking" / "target_panel.csv").exists() else None
    dsum = json.loads((PROC / "data_summary.json").read_text(encoding="utf-8")) \
        if (PROC / "data_summary.json").exists() else {}

    if au is not None:
        fig1_leakage(au); built.append("fig1_leakage_by_protocol.png")
    if m is not None:
        built += [f for f in (
            fig2_heatmap(m, "davis", "rmse", "fig2a_davis_rmse.png", "test RMSE (pKd units)"),
            fig2_heatmap(m, "kiba", "rmse", "fig2b_kiba_rmse.png", "test RMSE (KIBA units)"),
            fig2_heatmap(m, "davis", "ci_harrell", "fig2c_davis_ci.png",
                         "concordance index (Harrell: comparable pairs only)", better_low=False),
            fig2_heatmap(m, "kiba", "ci_harrell", "fig2d_kiba_ci.png",
                         "concordance index (Harrell: comparable pairs only)", better_low=False),
        ) if f]
        fig3_ci_calibers(m); built.append("fig3_rmse_vs_ci.png")
        fig4_ties(m); built.append("fig4_label_ties.png")
    if cf is not None:
        fig5_conformal(cf); built.append("fig5_conformal_coverage.png")
    if dp is not None and panel is not None:
        fig6_docking(dp, panel); built.append("fig6_docking_vs_pkd.png")

    # ------------------------------------------------------------------ report.md
    L = ["# DTI affinity benchmark -- results report", "",
         "Generated by `scripts/07_report.py`. Every number below is read from a file "
         "written by the numbered scripts; nothing here is re-fitted.", ""]
    if dsum:
        dv, kv = dsum.get("davis", {}), dsum.get("kiba", {})
        L += ["## 1. Data", "", md_table(pd.DataFrame([
            {"dataset": "DAVIS", "rows": dv.get("rows"),
             "ligands": dv.get("distinct_ligand_smiles"),
             "targets_by_sequence": dv.get("distinct_target_sequences"),
             "targets_by_name": dv.get("distinct_target_names"),
             "label": "pKd (Kd nM, clipped 0.016-10000)",
             "on_censor_ceiling": dv.get("fraction_at_censor_ceiling_pKd_5")},
            {"dataset": "KIBA", "rows": kv.get("rows"),
             "ligands": kv.get("distinct_ligand_smiles"),
             "targets_by_sequence": kv.get("distinct_target_sequences"),
             "targets_by_name": kv.get("distinct_target_keys"),
             "label": "KIBA score (0-17.2)",
             "on_censor_ceiling": None if not kv.get("rows") else
                                   round(kv["rows_at_floor"] / kv["rows"], 6)},
        ])), ""]
    if au is not None:
        L += ["## 2. Split protocols and leakage", "", md_table(au[[
            "dataset", "protocol", "n_train", "n_valid", "n_test", "test_ligands",
            "test_ligands_absent_from_train", "test_targets_absent_from_train",
            "test_scaffold_present_in_train",
            "test_pairs_with_near_duplicate_ligand_ge0.7",
            "target_max_dpc_cosine_to_train_target_mean"]]), ""]
    if m is not None:
        agg = (m.groupby(["dataset", "protocol", "model"])[["rmse", "ci_harrell", "ci_sklearn_like",
                                                            "pearson_r", "spearman_rho"]]
                 .mean().round(4).reset_index())
        # seed-averaged table WITH the spread, written next to metrics.csv because the
        # report text promises it and a promise in a report must be a file
        mbp = (m.groupby(["dataset", "protocol", "model"])
                .agg(n_seeds=("seed", "nunique"), n_test=("n_test", "max"),
                     rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                     ci_harrell_mean=("ci_harrell", "mean"), ci_harrell_sd=("ci_harrell", "std"),
                     ci_sklearn_like_mean=("ci_sklearn_like", "mean"),
                     pearson_r_mean=("pearson_r", "mean"))
                .round(4).reset_index())
        mbp["rmse_sd"] = mbp["rmse_sd"].fillna(0.0)
        mbp.to_csv(RES / "metrics_by_protocol.csv", index=False)
        imgs = [x for f in built if f.startswith("fig2")
                for x in (f"![{f.removesuffix('.png').replace('_', ' ')}](figures/{f})", "")]
        L += ["## 3. Models x protocols (mean over seeds)", "",
              "Per-seed spread is in `results/metrics.csv`; the seed-averaged table with its "
              "standard deviation is `results/metrics_by_protocol.csv`. CI is the Harrell "
              "calibre (comparable pairs only, tied labels dropped) — `ci_sklearn_like` is the "
              "all-pairs calibre and is the one DAVIS's censoring ceiling drags toward 0.5.",
              "", *imgs, md_table(agg), ""]
    if cfwide is not None:
        L += ["## 4. Uncertainty (split conformal, seed 0)", "", md_table(cfwide[[
            c for c in ["dataset", "protocol", "model", "method", "n_cal", "rmse", "q90",
                        "width90", "cov50", "cov80", "cov90", "coverage_error",
                        "rmse_matches_03", "corr_absres_sigma"]
            if c in cfwide.columns]]), ""]
    if dp is not None and panel is not None:
        L += ["## 5. AutoDock Vina arm", "",
              "Docking panel:", "", md_table(panel[[
                  c for c in ["target_id", "gene", "uniprot_acc", "pdb_id", "chain", "resolution",
                              "pdbe_coverage", "identity_vs_davis", "identity_vs_uniprot",
                              "cocrystal_ligand", "cocrystal_heavy", "meeko_bad_residues_dropped"]
                  if c in panel.columns]]), "",
              f"Scored pairs: {int(dp.vina_best.notna().sum())} / {len(dp)}", ""]
        cs = RES / "docking" / "comparison.csv"
        if cs.exists():
            L += [md_table(pd.read_csv(cs)), ""]
        dc = RES / "docking" / "decoy_control.csv"
        if dc.exists():
            L += ["Box-shifted (non-specific) control:", "", md_table(pd.read_csv(dc)), ""]
    (RES / "report.md").write_text("\n".join(L), encoding="utf-8")

    # ------------------------------------------------------------------ summary.json
    s = {"generated_by": "scripts/07_report.py", "figures_built": built}
    if m is not None:
        trained = m[m.model.isin(["ridge", "xgboost", "randomforest", "mlp"])]
        best = trained.groupby(["dataset", "protocol", "model"])[["rmse", "ci_harrell"]].mean()
        # reset_index first: on a MultiIndex, groupby(level=...).idxmin() returns the
        # *innermost* labels only, so best.loc[that] silently looks up the wrong level
        # (this is what raised "too many values to unpack" here).
        br = best.reset_index()
        pick = br.loc[br.groupby(["dataset", "protocol"]).rmse.idxmin()]
        s["best_model_per_protocol"] = {}
        for r in pick.itertuples():
            s["best_model_per_protocol"][f"{r.dataset}/{r.protocol}"] = {
                "model": str(r.model),
                "rmse": round(float(r.rmse), 4),
                "ci_harrell": round(float(r.ci_harrell), 4)}
        s["null_floor"] = {f"{r.dataset}/{r.protocol}": round(float(r.rmse), 4)
                           for r in m[m.model == "mean"].itertuples()}
        if au is not None:
            def aud(dataset, protocol, column):
                sel = au[(au.dataset == dataset) & (au.protocol == protocol)]
                return float(sel.iloc[0][column]) if len(sel) else None
            s["headline_leakage"] = {
                "kiba_random_near_duplicate_ge0.7":
                    aud("kiba", "random", "test_pairs_with_near_duplicate_ligand_ge0.7"),
                "kiba_scaffold_near_duplicate_ge0.7":
                    aud("kiba", "scaffold", "test_pairs_with_near_duplicate_ligand_ge0.7"),
                "kiba_cold_drug_scaffold_present_in_train":
                    aud("kiba", "cold_drug", "test_scaffold_present_in_train"),
                "davis_cold_target_targets_absent_from_train":
                    aud("davis", "cold_target", "test_targets_absent_from_train"),
                "davis_label_tie_fraction_mean":
                    float(m[m.dataset == "davis"].label_tie_fraction.mean()),
            }
    (RES / "summary.json").write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")
    rn = readme_numbers(m, cfwide, comp)
    (RES / "readme_numbers.json").write_text(json.dumps(rn, indent=2, default=float),
                                             encoding="utf-8")
    print("\nwrote results/report.md, results/summary.json, results/readme_numbers.json; "
          "figures:", built)
    return 0


if __name__ == "__main__":
    sys.exit(main())
