#!/usr/bin/env python3
"""
02_splits.py -- four evaluation protocols plus an explicit leakage audit.

The scientific claim this repository wants to make is about *protocols*, not about a new
network. So the same pairs are cut four different ways and every cut is measured for how
much similarity it lets leak from train into test:

  random      80/20 at the pair level, seeded. The optimistic ceiling: a test ligand's
              near-duplicates are almost always in the training set.
  scaffold    generic Murcko scaffold-grouped (Bock & Kelly 2016): scaffolds sorted by
              size, greedily assigned to test then valid until budget, remainder train.
              No scaffold is split across sides.
  cold_drug   test ligands never appear in training (target set may repeat).
  cold_target test targets never appear in training (ligand set may repeat).
  temporal    train on pairs deposited before a cutoff, test on the rest. Only possible
              on the dated subset (DAVIS 83.8%, KIBA 37.8% of rows), so its test set is
              *smaller and distribution-shifted* -- both facts are reported.

For DAVIS the published cold-drug / cold-target parts shipped inside the source CSVs
(`published_part` in {cold_drug, cold_target}) are used to cross-check our own cold
definitions; agreement is reported, not assumed.

Leakage audit per protocol and side:
  * fraction of test pairs whose ligand / target id is absent from train (structural, 0 or 1)
  * max ECFP4 Tanimoto from each test ligand to any training ligand: mean, median,
    share >= 0.7 ("near-duplicate") and >= 0.95
  * share of test pairs whose Murcko scaffold occurs in train
  * max cosine similarity of dipeptide composition between a test target and any
    training target (sequence-level family leakage), share >= 0.95
Outputs: data/processed/splits/*.csv  +  results/leakage_audit.csv/json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SPL = PROC / "splits"
RESULTS = ROOT / "results"

SEED = 2026
FRAC_TEST = 0.20
FRAC_VALID = 0.10


# --------------------------------------------------------------------------- helpers

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_split(idx, n, rng):
    perm = rng.permutation(n)
    n_test = int(round(FRAC_TEST * n))
    n_valid = int(round(FRAC_VALID * n))
    out = np.array(["train"] * n, dtype=object)
    out[perm[:n_test]] = "test"
    out[perm[n_test:n_test + n_valid]] = "valid"
    return out


def scaffold_split(lig_of_row, scaff_of_lig, rng, frac_test=FRAC_TEST, frac_valid=FRAC_VALID):
    """Grouped scaffold split: whole scaffolds go to one side, biggest first."""
    rows_by_scaffold = {}
    for r, l in enumerate(lig_of_row):
        rows_by_scaffold.setdefault(scaff_of_lig[l], []).append(r)
    order = sorted(rows_by_scaffold, key=lambda s: (-len(rows_by_scaffold[s]), s))
    n = len(lig_of_row)
    out = np.array(["train"] * n, dtype=object)
    n_test, n_valid = int(round(frac_test * n)), int(round(frac_valid * n))
    got_test = got_valid = 0
    for sc in order:
        rows = rows_by_scaffold[sc]
        if got_test + len(rows) <= n_test and got_test < n_test:
            for r in rows:
                out[r] = "test"
            got_test += len(rows)
        elif got_valid + len(rows) <= n_valid and got_valid < n_valid:
            for r in rows:
                out[r] = "valid"
            got_valid += len(rows)
    return out


def cold_split(keys, rng, frac=FRAC_TEST, frac_valid=FRAC_VALID):
    """Hold out whole entities (ligands or targets). `secondary` is used only to report."""
    uniq = sorted(set(keys))
    counts = pd.Series(keys).value_counts()
    sizes = counts.reindex(uniq).to_numpy()
    # biggest groups first, ties broken by id so the cut is reproducible
    order = sorted(range(len(uniq)), key=lambda i: (-int(sizes[i]), str(uniq[i])))
    n = len(keys)
    out = np.array(["train"] * n, dtype=object)
    budget, budget_v = int(round(frac * n)), int(round(frac_valid * n))
    got = got_v = 0
    for i in order:
        ent = uniq[i]
        rows = np.where(keys == ent)[0]
        if got + len(rows) <= budget:
            out[rows] = "test"
            got += len(rows)
        elif got_v + len(rows) <= budget_v:
            out[rows] = "valid"
            got_v += len(rows)
    return out


def temporal_split(dates, rng, frac=FRAC_TEST):
    """Forward-looking temporal split: train on pairs deposited before the cutoff, test
    on those deposited at or after it. Undated rows are excluded from this protocol and
    the exclusion count is reported. The source table carries year-resolution only, so
    the cutoff is snapped to a date boundary -- no single date is split across sides.
    (The upstream audit repo does the opposite, testing on the *earliest* rows; that is a
    contamination probe, not a deployment protocol, so we do not copy it.)"""
    n = len(dates)
    ints = np.asarray(pd.to_datetime(pd.Series(list(dates))).to_numpy(),
                      dtype="datetime64[ns]").astype(np.int64)
    known = ints > 0
    out = np.array(["train"] * n, dtype=object)
    out[~known] = "excluded"
    idx = np.where(known)[0]
    if len(idx) == 0:
        return out, None
    v = ints[idx]
    q = np.quantile(v, 1 - frac)
    uniq = np.unique(v)
    after = uniq[uniq >= q]
    if len(after) == 0:
        return out, None
    cut = int(after[0])
    out[idx[v >= cut]] = "test"
    pre = idx[v < cut]
    if len(pre):
        vp = ints[pre]
        qp = np.quantile(vp, 0.9)
        up = np.unique(vp)
        cp = up[up >= qp]
        if len(cp):
            out[pre[vp >= int(cp[0])]] = "valid"
    return out, str(pd.Timestamp(cut, unit="ns").date())


def n_of(x):
    return len(x)


# --------------------------------------------------------------------------- leakage audit

def audit(pairs, split, lig_ids, target_ids, tanimoto, lig_index, scaff, prot_sim, prot_index):
    """Quantify train->test similarity under one cut."""
    tr = np.where(split == "train")[0]
    te = np.where(split == "test")[0]
    va = np.where(split == "valid")[0]
    if len(te) == 0 or len(tr) == 0:
        return {}
    tr_lig = set(lig_ids[tr])
    te_lig = lig_ids[te]
    unseen_lig = float(np.mean([l not in tr_lig for l in te_lig]))
    # max Tanimoto from each test ligand to any training ligand
    tr_li = np.array(sorted({lig_index[l] for l in tr_lig}))
    best = {}
    if len(tr_li):
        sub = tanimoto[np.array([lig_index[l] for l in te_lig])[:, None], tr_li[None, :]]
        mx = sub.max(axis=1)
        for l, m in zip(te_lig, mx):
            best[l] = float(m)
    test_max_sim = np.array([best.get(l, np.nan) for l in te_lig])
    # scaffold overlap
    tr_sc = {scaff[l] for l in tr_lig if l in scaff}
    sc_overlap = float(np.mean([scaff[l] in tr_sc for l in te_lig]))
    # target family similarity
    tr_tgt = set(target_ids[tr])
    unseen_tgt = float(np.mean([t not in tr_tgt for t in target_ids[te]]))
    tr_ti = np.array(sorted({prot_index[t] for t in tr_tgt}))
    if len(tr_ti) and prot_sim is not None:
        ps = prot_sim[np.array([prot_index[t] for t in target_ids[te]])[:, None],
                      tr_ti[None, :]].max(axis=1)
    else:
        ps = np.full(len(te), np.nan)
    out = {
        "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
        "test_fraction": round(len(te) / (len(tr) + len(va) + len(te)), 4),
        "test_ligands": int(len(set(te_lig))),
        "test_ligands_absent_from_train": round(unseen_lig, 4),
        "test_targets_absent_from_train": round(unseen_tgt, 4),
        "test_scaffold_present_in_train": round(sc_overlap, 4),
        "ecfp4_max_similarity_to_train_ligand_mean": None if np.isnan(test_max_sim).all()
            else round(float(np.nanmean(test_max_sim)), 4),
        "ecfp4_max_similarity_median": None if np.isnan(test_max_sim).all()
            else round(float(np.nanmedian(test_max_sim)), 4),
        "test_pairs_with_near_duplicate_ligand_ge0.7": round(float(np.nanmean(test_max_sim >= 0.7)), 4)
            if not np.isnan(test_max_sim).all() else None,
        "test_pairs_with_near_duplicate_ligand_ge0.95": round(float(np.nanmean(test_max_sim >= 0.95)), 4)
            if not np.isnan(test_max_sim).all() else None,
        "target_max_dpc_cosine_to_train_target_mean": None if np.isnan(ps).all()
            else round(float(np.nanmean(ps)), 4),
        "test_pairs_with_target_family_ge0.95": None if np.isnan(ps).all()
            else round(float(np.nanmean(ps >= 0.95)), 4),
    }
    if len(va):
        va_lig = lig_ids[va]
        out["valid_pairs_with_near_duplicate_ligand_ge0.7"] = round(float(np.nanmean(
            [best.get(l, np.nan) >= 0.7 for l in va_lig])), 4)
    return {k: v for k, v in out.items()}


# --------------------------------------------------------------------------- main

def main() -> int:
    import scipy.sparse as sp
    SPL.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = _rng(SEED)

    lig = pd.read_csv(PROC / "ligands.csv")
    tg = pd.read_csv(PROC / "targets.csv")
    scaf = pd.read_csv(PROC / "features" / "ligand_scaffolds.csv")
    tanimoto = np.load(PROC / "features" / "ligand_tanimoto.npy")
    pf = pd.read_csv(PROC / "features" / "protein_features.csv")
    scaff = dict(zip(scaf.ligand_id, scaf.scaffold))
    lig_index = dict(zip(lig.ligand_id, range(len(lig))))
    prot_index = dict(zip(tg.target_id, range(len(tg))))

    # protein similarity: cosine over the 400-dim dipeptide composition
    D = pf[[c for c in pf.columns if c.startswith("dpc_")]].to_numpy(dtype=np.float64)
    order = {t: i for i, t in enumerate(pf.target_id)}
    D = D[[order[t] for t in tg.target_id]]
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    Dn = D / np.where(norms == 0, 1, norms)
    prot_sim = (Dn @ Dn.T).astype(np.float32)
    off = prot_sim[~np.eye(len(prot_sim), dtype=bool)]
    print(f"protein dpc-cosine: max off-diagonal {off.max():.4f}, "
          f"pairs >= 0.95: {(off >= 0.95).sum() // 2}")

    audits, split_meta = [], {}
    for dataset, ycol in (("davis", "pkd"), ("kiba", "kiba_score")):
        p = pd.read_csv(PROC / f"{dataset}.csv")
        lig_ids = p.ligand_id.to_numpy()
        tgt_ids = p.target_id.to_numpy()
        dates = pd.to_datetime(p.get("deposit_date"), utc=True, errors="coerce").to_numpy(
            dtype="datetime64[ns]") if "deposit_date" in p else np.full(len(p), np.datetime64("NaT"))
        protocols = {
            "random": lambda: random_split(None, len(p), rng),
            "scaffold": lambda: scaffold_split(lig_ids, scaff, rng),
            "cold_drug": lambda: cold_split(lig_ids, rng),
            "cold_target": lambda: cold_split(tgt_ids, rng),
            "temporal": lambda: temporal_split(dates, rng),
        }
        for name, fn in protocols.items():
            res = fn()
            split = res[0] if isinstance(res, tuple) else res
            cutoff = res[1] if isinstance(res, tuple) else None
            out = p[["pair_id", "ligand_id", "target_id"]].copy()
            out["split"] = split
            if name == "temporal":
                out = out[out.split != "excluded"]
            out.to_csv(SPL / f"{dataset}_{name}.csv", index=False)
            a = audit(p, split, lig_ids, tgt_ids, tanimoto, lig_index, scaff, prot_sim, prot_index)
            a.update({"dataset": dataset, "protocol": name,
                      "rows_in_protocol": int(len(out)),
                      "temporal_cutoff": cutoff,
                      "excluded_rows": int((split == "excluded").sum())})
            audits.append(a)
            split_meta[f"{dataset}_{name}"] = {
                "counts": {k: int(v) for k, v in pd.Series(split).value_counts().items()},
                "n_rows": int(len(out))}
            print(f"{dataset:<6} {name:<12} train {list(split).count('train'):>7} "
                  f"valid {list(split).count('valid'):>6} test {list(split).count('test'):>6}"
                  f"  excluded {list(split).count('excluded'):>6}")

    # What do the *published* DAVIS parts actually hold out? (checked, not assumed)
    pv = pd.read_csv(PROC / "davis.csv")
    check = {}
    train_l = set(pv.loc[pv.published_part == "train", "ligand_id"])
    train_t = set(pv.loc[pv.published_part == "train", "target_id"])
    train_lv = set(pv.loc[pv.published_part.isin(["train", "val"]), "ligand_id"])
    train_tv = set(pv.loc[pv.published_part.isin(["train", "val"]), "target_id"])
    for part in ("train", "val", "test", "cold_drug", "cold_target"):
        sub = pv[pv.published_part == part]
        if part == "train" or not len(sub):
            continue
        L, T = set(sub.ligand_id), set(sub.target_id)
        check[part] = {
            "rows": int(len(sub)), "n_ligands": len(L), "n_targets": len(T),
            "frac_ligand_rows_seen_in_train": round(float(np.isin(sub.ligand_id, list(train_l)).mean()), 4),
            "frac_target_rows_seen_in_train": round(float(np.isin(sub.target_id, list(train_t)).mean()), 4),
            "frac_ligand_rows_seen_in_train_or_val": round(float(np.isin(sub.ligand_id, list(train_lv)).mean()), 4),
            "frac_target_rows_seen_in_train_or_val": round(float(np.isin(sub.target_id, list(train_tv)).mean()), 4),
        }
    # our cold protocols vs the published subsets (row overlap; budgets differ by design)
    for ours, theirs in (("cold_drug", "cold_drug"), ("cold_target", "cold_target")):
        sp_df = pd.read_csv(SPL / f"davis_{ours}.csv")
        their_ids = set(pv.loc[pv.published_part == theirs, "pair_id"])
        our_test = set(sp_df.loc[sp_df.split == "test", "pair_id"])
        inter = len(our_test & their_ids)
        check[f"ours_{ours}_vs_published"] = {
            "our_test_pairs": len(our_test), "published_pairs": len(their_ids),
            "intersection": inter,
            "jaccard": round(inter / max(len(our_test | their_ids), 1), 4),
            "note": "the published subsets are 4-5x smaller, so the cuts differ; what is "
                    "verified is the *property*, reported per part above",
        }

    ad = pd.DataFrame(audits)
    cols = ["dataset", "protocol", "rows_in_protocol", "n_train", "n_valid", "n_test",
            "test_fraction", "test_ligands", "test_ligands_absent_from_train",
            "test_targets_absent_from_train", "test_scaffold_present_in_train",
            "ecfp4_max_similarity_to_train_ligand_mean", "ecfp4_max_similarity_median",
            "test_pairs_with_near_duplicate_ligand_ge0.7",
            "test_pairs_with_near_duplicate_ligand_ge0.95",
            "target_max_dpc_cosine_to_train_target_mean",
            "test_pairs_with_target_family_ge0.95", "excluded_rows", "temporal_cutoff"]
    ad = ad[[c for c in cols if c in ad.columns]]
    ad.to_csv(RESULTS / "leakage_audit.csv", index=False)
    (RESULTS / "leakage_audit.json").write_text(json.dumps(
        {"audit": ad.to_dict(orient="records"), "published_subset_cross_check": check,
         "published_part_finding": (
             "In the delivered DAVIS partition, `cold_drug` really is ligand-disjoint from "
             "`train` (0 of its ligand rows appear in train) but its targets are all warm, "
             "while `cold_target` is target-disjoint (88% of its target rows unseen in "
             "train) with all ligands warm. `val` is not entity-disjoint at all: 100% of its "
             "ligands and 100% of its targets also appear in train. So evaluating on these "
             "files tests different things, and a number quoted as 'DAVIS cold-drug' depends "
             "on which of the shipped blocks was used."),
         "target_similarity_caveat": (
             "target_max_dpc_cosine_to_train_target_mean is 1.0 for every protocol except "
             "cold_target, because the identical target sequence is then present in the "
             "training set by construction; the metric only carries information about "
             "protein-family leakage under cold_target."),
         "protein_similarity": {"metric": "cosine over 400-dim dipeptide composition",
                                "max_off_diagonal": round(float(off.max()), 4),
                                "n_target_pairs_ge_0.95": int((off >= 0.95).sum() // 2)}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n", ad.to_string(index=False))
    print("\ncross-check vs published DAVIS cold-start parts:")
    print(json.dumps(check, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
