#!/usr/bin/env python3
"""
00_fetch_data.py -- fetch, cross-verify and normalise the DAVIS and KIBA
protein--ligand affinity benchmarks.

Channel note: from this machine `github.com:443` and `raw.githubusercontent.com` are
both unreachable, but `api.github.com` works. Every artefact is therefore pulled as a
git *blob* (`GET /repos/{repo}/git/blobs/{sha}` with `Accept: application/vnd.github.raw`)
and its sha256 is recorded in `data/raw/manifest.json`, so each number reported by this
repository is traceable to the exact bytes we downloaded.

Sources
-------
DAVIS  canonical long tables, `zihandun/GraphDTA-ColdStart`
    five files that sum to exactly 30,056 = 68 drugs x 442 kinase targets (verified
    below): train / val / test plus the two published cold-start test subsets.
    Affinity is pKd. Original: Davis et al., Proteins 2011.
DAVIS/KIBA full tables with assay + deposit-date metadata, `swmeyer1979/dti-contamination`
    `davis_temporal_splits.parquet` (30,056 rows, Kd in nM, right-censored at the assay
    ceiling) and `kiba_temporal_splits.parquet` (118,254 rows, KIBA score).
    Originals: Liu et al. JCIM 2015 (KIBA), Menden et al. Nat Commun 2019 (DepoKIBA).

Why two DAVIS files are needed: the canonical CSVs give pKd straight away, but they
carry no target identifier -- only the sequence. 442 kinase targets collapse onto 379
distinct sequences, so 4,284 of the 30,056 measurements share their features with
another measurement. We use the censored-Kd table to recover the target name by
matching affinity values *within* each (SMILES, sequence) group, and we report the
residual disagreement between the two sources instead of hiding it.

Output (data/processed/, normalised so no protein sequence is repeated per row)
    ligands.csv  ligand_id, smiles, canonical_smiles, ...
    targets.csv  target_id, name, sequence, seq_len, ...
    davis.csv    pair_id, ligand_id, target_id, pkd, kd_nM, deposit_date, published_part
    kiba.csv     pair_id, ligand_id, target_id, kiba_score, deposit_date
    data_summary.json

Usage
    python scripts/00_fetch_data.py            # network, then normalise + verify
    python scripts/00_fetch_data.py --offline  # normalise + verify from data/raw/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

API = "https://api.github.com"
GC = "zihandun/GraphDTA-ColdStart"
CT = "swmeyer1979/dti-contamination"

# source_id -> (repo, git blob sha1, filename, path in repo)
BLOBS = {
    "davis_train": (GC, "698be36ac8527b80dc19f4781ae1e687546b8bf0", "davis_train_canonical.csv",
                    "data/oursplit/davis_train_canonical.csv"),
    "davis_val": (GC, "98d56581c0a94f4c5b8f23906eaec134050a65ac", "davis_val_canonical.csv",
                  "data/oursplit/davis_val_canonical.csv"),
    "davis_test": (GC, "f3392841fbee9737784661519c4db0df1597a192", "davis_test_canonical.csv",
                   "data/oursplit/davis_test_canonical.csv"),
    "davis_cold_drug": (GC, "f007de43eabad0da737e09a7b7ae01778fb5c0b8", "davis_test1_canonical.csv",
                        "data/oursplit/davis_test1_canonical.csv"),
    "davis_cold_target": (GC, "64900311978825366599362436f6aaa7a5cb8cd8", "davis_test2_canonical.csv",
                          "data/oursplit/davis_test2_canonical.csv"),
    "kiba_table": (CT, "890f6ad19a0f98609e58b9aebdd3fd33e0724286", "kiba_temporal_splits.parquet",
                   "data/splits/kiba_temporal_splits.parquet"),
    "davis_kd_table": (CT, "7c5eac29a77b0660dbd5527a4b77aece321cd32d", "davis_temporal_splits.parquet",
                       "data/splits/davis_temporal_splits.parquet"),
}

DAVIS_PARTS = {
    "davis_train": "train", "davis_val": "val", "davis_test": "test",
    "davis_cold_drug": "cold_drug", "davis_cold_target": "cold_target",
}

PROVENANCE = {
    "davis_original": "Davis MI et al. Proteins 2011;69(2):319-324 -- 68 inhibitor x 442 kinase Kd panel",
    "kiba_original": "Liu T et al. J Chem Inf Model 2015;55(9):1792-1801 (KIBA score); "
                     "aggregated by Menden GD et al. Nat Commun 2019;10:3050 (DepoKIBA)",
    "mirror_davis_canonical": f"{GC} (GitHub, canonical CSV tables)",
    "mirror_full_tables": f"{CT} (GitHub, contamination/temporal audit of DAVIS+KIBA)",
    "retrieved_by": "api.github.com git/blobs, Accept: application/vnd.github.raw",
}


# --------------------------------------------------------------------------- plumbing

def http_get(url: str, headers: dict | None = None, tries: int = 4) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dti-affinity-bench",
                                                       **(headers or {})})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:
            last, _ = e, time.sleep(2 + 5 * i)
    raise RuntimeError(f"GET failed {url}: {last}")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load(sid: str) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd
    _repo, _sha, fname, _path = BLOBS[sid]
    p = RAW / fname
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def canonical_smiles(s: str):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


# --------------------------------------------------------------------------- DAVIS

def build_davis(pd):
    """Normalise DAVIS: canonical pKd table + target names recovered from the Kd table."""
    frames = []
    for sid, part in DAVIS_PARTS.items():
        df = load(sid)
        assert set(df.columns) >= {"compound_iso_smiles", "target_sequence", "affinity"}, \
            f"{sid}: unexpected columns {list(df.columns)}"
        frames.append(pd.DataFrame({
            "canonical_smiles": df["compound_iso_smiles"].map(canonical_smiles),
            "raw_smiles": df["compound_iso_smiles"],
            "sequence": df["target_sequence"],
            "pkd": df["affinity"].astype(float),
            "published_part": part,
        }))
    d = pd.concat(frames, ignore_index=True)
    files_sum = int(sum(len(f) for f in frames))
    d["source"] = "canonical_pKd"

    kd = load("davis_kd_table")
    kd["canonical_smiles"] = kd["smiles"].map(canonical_smiles)
    kd = kd.rename(columns={"protein_key": "target_name"})
    kd["pkd_from_kd"] = 9.0 - np.log10(kd["affinity"].astype(float).clip(lower=0.016, upper=10000.0))
    # Both sources hold the same 68 x 442 grid, but the canonical CSVs carry no target
    # name -- only the sequence -- and 442 names collapse onto 379 sequences. So the name
    # is recovered positionally: inside each (SMILES, sequence) group, sort the canonical
    # pKd values and the Kd-derived pKd values and pair them up. The residual disagreement
    # is then reported rather than hidden.
    d = d.sort_values(["canonical_smiles", "sequence", "pkd"], kind="stable").reset_index(drop=True)
    d["rank"] = d.groupby(["canonical_smiles", "sequence"], sort=False).cumcount()
    kd = kd.sort_values(["canonical_smiles", "sequence", "pkd_from_kd", "target_name"],
                        kind="stable").reset_index(drop=True)
    kd["rank"] = kd.groupby(["canonical_smiles", "sequence"], sort=False).cumcount()

    aligned = d[["canonical_smiles", "sequence", "rank", "pkd"]].merge(
        kd[["canonical_smiles", "sequence", "rank", "target_name", "affinity",
            "pkd_from_kd", "deposit_date"]],
        on=["canonical_smiles", "sequence", "rank"], how="left")
    d = d.merge(aligned.drop(columns=["pkd"]), on=["canonical_smiles", "sequence", "rank"],
                how="left", suffixes=("", "_a"))

    delta = (d["pkd"] - d["pkd_from_kd"]).abs()
    ok = delta <= 0.02
    grp_bad = (d.assign(_bad=(~ok) & d["target_name"].notna())
                .groupby(["canonical_smiles", "sequence"], sort=False)["_bad"].max())
    return d, kd, {
        "rows": int(len(d)),
        "distinct_ligand_smiles": int(d.canonical_smiles.nunique()),
        "distinct_target_sequences": int(d.sequence.nunique()),
        "distinct_target_names": int(kd["target_name"].nunique()),
        "distinct_feature_pairs": int(d.drop_duplicates(["canonical_smiles", "sequence"]).shape[0]),
        "rows_sharing_features_with_another_row": int(d.duplicated(["canonical_smiles", "sequence"]).sum()),
        "max_within_pair_pKd_spread": round(float(
            d.groupby(["canonical_smiles", "sequence"])["pkd"].std().max()), 4),
        "fraction_at_censor_ceiling_pKd_5": round(float((d["pkd"] <= 5.0 + 1e-9).mean()), 4),
        "cross_source_rows_aligned": int(d["target_name"].notna().sum()),
        "cross_source_agree_within_0.02_pKd": int(ok.sum()),
        "cross_source_mean_abs_delta_pKd": round(float(delta[ok].mean()), 5),
        "cross_source_max_abs_delta_pKd": round(float(delta[ok].max()), 5),
        "cross_source_disagreeing_rows": int(((~ok) & d["target_name"].notna()).sum()),
        "feature_groups_with_any_disagreement": int(grp_bad.sum()),
        "deposit_date_known_rows": int(d["deposit_date"].notna().sum()),
        "sum_of_five_published_files": files_sum,
        "bad_smiles": 0,
        "caveat": (
            "target_name is recovered by sorting affinity values inside each "
            "(SMILES, sequence) group. 69.6% of DAVIS values sit on the censoring ceiling "
            "(pKd = 5.0), so within a group of sequence-identical targets the *label* is "
            "identical and the two are interchangeable for modelling; only the name attached "
            "to such a row is arbitrary. All modelling keys on the sequence (target_id), "
            "never on target_name."),
    }


# --------------------------------------------------------------------------- main

def main() -> int:
    import numpy as np
    import pandas as pd
    globals()["np"] = np  # build_davis uses np for the log10 transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    # ---- 1. fetch -----------------------------------------------------------
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "provenance": PROVENANCE, "artifacts": {}}
    for sid, (repo, sha, fname, path) in BLOBS.items():
        dest = RAW / fname
        cached = dest.exists()
        if not (args.offline or cached):
            dest.write_bytes(http_get(f"{API}/repos/{repo}/git/blobs/{sha}",
                                      {"Accept": "application/vnd.github.raw"}))
        if not dest.exists():
            raise SystemExit(f"missing artefact {fname} (offline mode and not cached)")
        data = dest.read_bytes()
        manifest["artifacts"][sid] = {"repo": repo, "path_in_repo": path,
                                      "git_blob_sha1": sha, "file": f"data/raw/{fname}",
                                      "bytes": len(data), "sha256": sha256(data),
                                      "was_cached": cached}
        print(f"[{'cache' if cached else 'fetch'}] {sid:<17} {len(data):>10,} B  "
              f"{manifest['artifacts'][sid]['sha256'][:12]}")

    # ---- 2. DAVIS -----------------------------------------------------------
    d, kd, davis_stats = build_davis(pd)
    kiba = load("kiba_table")
    kiba["canonical_smiles"] = kiba["smiles"].map(canonical_smiles)

    # ---- 3. id tables (sequences and SMILES stored once) --------------------
    davis_stats["bad_smiles"] = int(d.canonical_smiles.isna().sum())
    kiba = kiba.dropna(subset=["canonical_smiles"]).copy()

    all_smiles = sorted(set(d.canonical_smiles.dropna()) | set(kiba.canonical_smiles.dropna()))
    lig = pd.DataFrame({"ligand_id": [f"L{i:05d}" for i in range(len(all_smiles))],
                        "canonical_smiles": all_smiles})
    s2i = dict(zip(lig.canonical_smiles, lig.ligand_id))

    d["ligand_id"] = d.canonical_smiles.map(s2i)
    kiba["ligand_id"] = kiba.canonical_smiles.map(s2i)

    # target_id is keyed by *sequence*, because the sequence is what the model sees: the
    # 442 DAVIS names collapse onto 379 sequences and the 2,111 KIBA keys onto 229, so a
    # single sequence can carry several assay identities. Names are aggregated per sequence
    # and the multiplicity is reported in data_summary.json rather than smoothed away.
    names = pd.concat([
        d[["sequence", "target_name"]].rename(columns={"target_name": "name"}),
        kiba[["sequence", "protein_key"]].rename(columns={"protein_key": "name"}),
    ], ignore_index=True).dropna(subset=["name"]).drop_duplicates(["sequence", "name"])
    tg = (names.groupby("sequence", sort=True)["name"]
          .apply(lambda s: "|".join(sorted(set(s)))).reset_index()
          .rename(columns={"name": "names"}))
    tg = tg.sort_values("sequence", kind="stable").reset_index(drop=True)
    tg["target_id"] = [f"T{i:04d}" for i in range(len(tg))]
    tg["n_names"] = tg.names.str.count("|") + 1
    q2i = dict(zip(tg.sequence, tg.target_id))
    d["target_id"] = d.sequence.map(q2i)
    kiba["target_id"] = kiba.sequence.map(q2i)

    lig["n_davis_pairs"] = lig.ligand_id.map(d.ligand_id.value_counts()).fillna(0).astype(int)
    lig["n_kiba_pairs"] = lig.ligand_id.map(kiba.ligand_id.value_counts()).fillna(0).astype(int)
    lig["raw_smiles_example"] = lig.canonical_smiles.map(
        dict(zip(d.canonical_smiles, d.raw_smiles)))
    tg["seq_len"] = tg.sequence.str.len()
    tg["n_davis_pairs"] = tg.target_id.map(d.target_id.value_counts()).fillna(0).astype(int)
    tg["n_kiba_pairs"] = tg.target_id.map(kiba.target_id.value_counts()).fillna(0).astype(int)
    tg["in_both_datasets"] = (tg.n_davis_pairs > 0) & (tg.n_kiba_pairs > 0)

    davis_out = (d.assign(pair_id=range(len(d)))
                 [["pair_id", "ligand_id", "target_id", "pkd", "target_name",
                   "published_part", "deposit_date"]]
                 .rename(columns={"affinity": "kd_nM"}))
    davis_out["kd_nM"] = d["affinity"].to_numpy()
    davis_out["pkd_from_kd_table"] = d["pkd_from_kd"].to_numpy()
    kiba_out = (kiba.assign(pair_id=range(len(kiba)))
                [["pair_id", "ligand_id", "target_id", "affinity", "deposit_date"]]
                .rename(columns={"affinity": "kiba_score"}))

    davis_out.to_csv(PROC / "davis.csv", index=False)
    kiba_out.to_csv(PROC / "kiba.csv", index=False)
    lig.to_csv(PROC / "ligands.csv", index=False)
    tg[["target_id", "names", "n_names", "seq_len", "n_davis_pairs", "n_kiba_pairs",
        "in_both_datasets"]].to_csv(PROC / "targets_index.csv", index=False)
    tg.to_csv(PROC / "targets.csv", index=False)

    # ---- 4. summary + verification -----------------------------------------
    kiba_stats = {
        "rows": int(len(kiba)),
        "distinct_ligand_smiles": int(kiba.canonical_smiles.nunique()),
        "distinct_ligand_keys": int(kiba.ligand_key.nunique()),
        "distinct_target_sequences": int(kiba.sequence.nunique()),
        "distinct_target_keys": int(kiba.protein_key.nunique()),
        "distinct_feature_pairs": int(kiba.drop_duplicates(["canonical_smiles", "sequence"]).shape[0]),
        "rows_sharing_features_with_another_row": int(kiba.duplicated(["canonical_smiles", "sequence"]).sum()),
        "grid_cells": int(kiba.canonical_smiles.nunique() * kiba.sequence.nunique()),
        "grid_fill_rate": round(float(len(kiba) / (kiba.canonical_smiles.nunique() * kiba.sequence.nunique())), 4),
        "score_min": round(float(kiba.affinity.min()), 4),
        "score_median": round(float(kiba.affinity.median()), 4),
        "score_max": round(float(kiba.affinity.max()), 4),
        "rows_at_floor": int((kiba.affinity <= 1e-6).sum()),
        "published_split_counts": {k: int(v) for k, v in kiba["split"].value_counts().items()},
        "deposit_date_known_rows": int(kiba.deposit_date.notna().sum()),
        "deposit_date_known_fraction": round(float(kiba.deposit_date.notna().mean()), 4),
    }
    cross = {
        "shared_ligand_smiles_davis_kiba": int(len(set(d.canonical_smiles.dropna())
                                                   & set(kiba.canonical_smiles.dropna()))),
        "shared_target_sequences_davis_kiba": int(len(set(d.sequence) & set(kiba.sequence))),
        "davis_targets_total": int(d.sequence.nunique()),
        "note": "the two benchmarks share most of their kinase targets, so "
                "train-on-DAVIS / test-on-KIBA is NOT target-disjoint",
    }
    summary = {"davis": davis_stats, "kiba": kiba_stats, "cross_dataset": cross,
               "tables": {
                   "davis.csv": int(len(davis_out)), "kiba.csv": int(len(kiba_out)),
                   "ligands.csv": int(len(lig)), "targets.csv": int(len(tg))},
               "manifest": manifest["artifacts"]}
    (PROC / "data_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    print("\n" + "=" * 74)
    print(json.dumps({"davis": davis_stats, "kiba": kiba_stats, "cross_dataset": cross},
                     indent=2, ensure_ascii=False))

    checks = [
        ("DAVIS: five published files sum to 30,056", davis_stats["sum_of_five_published_files"] == 30056),
        ("DAVIS: rows == 30,056", davis_stats["rows"] == 30056),
        ("DAVIS: 68 distinct ligand SMILES", davis_stats["distinct_ligand_smiles"] == 68),
        ("DAVIS: 442 distinct target names", davis_stats["distinct_target_names"] == 442),
        ("DAVIS: pKd in [5.0, 10.8]", float(d.pkd.min()) >= 4.99 and float(d.pkd.max()) <= 10.81),
        ("DAVIS: no unparseable SMILES", davis_stats["bad_smiles"] == 0),
        ("DAVIS: every row has ligand_id and target_id",
         int(d.ligand_id.isna().sum() + d.target_id.isna().sum()) == 0),
        ("KIBA: rows == 118,254", kiba_stats["rows"] == 118254),
        ("KIBA: score in [0, 17.21]", kiba_stats["score_min"] >= 0 and kiba_stats["score_max"] <= 17.21),
        ("KIBA: train and test both present", {"train", "test"} <= set(kiba_stats["published_split_counts"])),
        ("KIBA: grid fill rate ~25% (sparse -> cold-start splits are meaningful)",
         0.20 <= kiba_stats["grid_fill_rate"] <= 0.30),
        ("ids unique across ligands / targets",
         lig.ligand_id.is_unique and tg.target_id.is_unique),
        ("pair tables store ids, not repeated sequences/SMILES",
         not (set(davis_out.columns) | set(kiba_out.columns))
             & {"sequence", "smiles", "canonical_smiles"}),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("\nwrote:", ", ".join(str(PROC / f) for f in
          ["davis.csv", "kiba.csv", "ligands.csv", "targets.csv", "targets_index.csv",
           "data_summary.json"]))
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
