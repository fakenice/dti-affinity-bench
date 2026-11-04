#!/usr/bin/env python3
"""
14_external_chembl.py -- the headline claim, re-measured on a third, independently curated table.

The repository's headline is a claim about *evaluation protocols*, not about a network: the way
a DTI table is cut up moves RMSE more than the choice of model does. Until this script, that
claim was supported only by DAVIS and KIBA -- two tables that are both small, both kinase-centred
and both widely re-used, which is exactly the configuration in which a protocol artefact can look
like a property of the field. Here the same five protocols, the same four learned models, the same
three nulls, the same two concordance calibres and the same paired-bootstrap families are run on
ChEMBL kinase Ki data pulled by 13_fetch_chembl.py: 19,088 pairs, 14,713 ligands, 16 human kinases,
curated by a different source, on a different label scale (pKi from median Ki, not pKd from a
dose-response experiment), with a different censoring pattern and a different publication history.

Nothing in the shipped numbers is recomputed or rewritten. The external table is *self-contained
by construction*:

  * features live in data/processed/external/features/, splits in data/processed/external/splits/,
    results in results/external_chembl/;
  * dti_core.Featuriser's PROC/FEAT module globals are redirected inside this process only, so the
    256 "most frequent ECFP4 bits" are selected over the external ligand set. Appending these 14,713
    ligands to the shipped data/processed/ligands.csv instead would have changed Featuriser.top_bits
    for DAVIS and KIBA too, and with them every committed number in results/metrics.csv.

Reuse, not re-derivation: the feature builders, the split functions, the leakage audit, the models,
the nulls and the bootstrap statistics are imported from 01_features.py, 02_splits.py, 03_models.py,
11_variance_decomposition.py and 12_testset_bootstrap.py via importlib (their digit-prefixed filenames
are not importable as module names). Two files were refactored to make that possible, and the
refactor was *measured* rather than asserted: 01_features.py's extraction of `bit_vectors` /
`scaffold_table` and 03_models.py's hoisting of `entity_mean` were followed by a re-run of 01 whose
seven output files came back byte-identical to the pre-refactor sha256s recorded in
data/processed/features/features_manifest.json, and by a 03 smoke run reproducing the shipped entity
nulls (davis/random: mean_ligand_train RMSE 0.810575, mean_target_train 0.875423, mean 0.922429).

One number is deliberately NOT shared: the dense pairwise Tanimoto table. 01_features.py materialises
2,127 x 2,127 float32 (18 MB) for the shipped ligands; the external set would need 14,713 x 14,713 =
867 MB. `BlockTanimoto` below answers exactly the block-index expression 02_splits.audit() performs,
computing those entries from the same RDKit bit vectors with BulkTanimotoSimilarity, so the leakage
metric is the same quantity without the quadratic memory. The substitution is checked, not assumed:
`--selftest` compares BlockTanimoto against a dense matrix on a random sub-block and asserts equality.

Stages (idempotent, resumable; `--stage all` runs them in order):

  features  ECFP4 counts/binary + descriptors + protein composition + Murcko scaffolds
  splits    random / scaffold / cold_drug / cold_target / temporal, written as chembl_*.csv
  audit     02_splits.audit() per protocol, incl. max Tanimoto to train and scaffold overlap
  grid      3 nulls + 4 models x 5 protocols x 3 seeds, metrics.csv written after every row
  analyse   variance shares (11), paired bootstrap + Holm families (12), headline cross-table

Label note that bounds the comparison: pKi and DAVIS pKd share the 9 - log10(nM) definition but not
the measurement process, and KIBA's 0-17.2 consensus score shares neither. RMSE magnitudes across the
three tables are therefore *not* interchangeable; what is compared is the protocol-vs-model structure,
and the README says so.

Usage
  python scripts/14_external_chembl.py --stage all --quick      # 1 seed, mean+ridge, smoke
  python scripts/14_external_chembl.py --stage all               # the full external grid (background it)
  python scripts/14_external_chembl.py --stage grid --resume     # continue an interrupted grid
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PROC = ROOT / "data" / "processed"
EXT = PROC / "external"
EXTFEAT = EXT / "features"
EXTSPL = EXT / "splits"
RESULTS = ROOT / "results"
OUT = RESULTS / "external_chembl"
PRED = OUT / "predictions"

sys.path.insert(0, str(SCRIPTS))

DATASET = "chembl"
YCOL = "pki"
PROTOCOLS = ["random", "scaffold", "cold_drug", "cold_target", "temporal"]
NULLS = ["mean", "mean_ligand_train", "mean_target_train"]
LEARNED = ["ridge", "xgboost", "randomforest", "mlp"]
MODEL_SET = LEARNED + ["mean"]
MODEL_CONTRASTS = list(itertools.combinations(LEARNED, 2)) + [(m, "mean") for m in LEARNED]
SEEDS = [0, 1, 2]
BASE_SEED = 20261104          # deterministic 31-bit per-cell seeds, same rule as 12
STAGES = ["features", "splits", "audit", "grid", "analyse"]


# --------------------------------------------------------------------------- reuse

def load_sibling(name: str):
    """Import a digit-prefixed sibling script (not importable by module name) by file path."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class BlockTanimoto:
    """Answers `t[R, C]` (broadcast integer index arrays) with the Tanimoto similarities of those
    index pairs, computed from RDKit bit vectors instead of a materialised n x n matrix.

    02_splits.audit() asks for exactly one thing of the similarity table -- the block
    `t[test_ligand_rows[:, None], train_ligand_cols[None, :]]` -- so a lazy block provider is a
    drop-in for the dense ndarray at that call site. Two paths:

      * the outer-product shape audit() uses: one BulkTanimotoSimilarity call per requested row,
        against the requested columns in order. No copy, no mask, and the peak allocation is the
        returned block itself (the largest external case is ~5.7k x 14k float32 = 320 MB, against
        867 MB for the full dense matrix -- and *not* 1.3 GB, which a naive broadcast-then-mask
        implementation would allocate in index arrays alone).
      * any other broadcast shape: the answer is still drawn from unique(R) x unique(C), so that
        sub-block is computed and gathered. Used by --selftest.

    Both paths are checked against a dense matrix in --selftest.
    """

    def __init__(self, fps):
        self.fps = fps
        self.shape = (len(fps), len(fps))
        self.queries = 0
        self.entries = 0

    def _bulk(self, rows, cols):
        from rdkit.DataStructs import cDataStructs
        cols_fps = [self.fps[int(j)] for j in cols]
        out = np.empty((len(rows), len(cols)), dtype=np.float32)
        for a, ri in enumerate(rows):
            out[a] = cDataStructs.BulkTanimotoSimilarity(self.fps[int(ri)], cols_fps)
        self.queries += len(rows)
        self.entries += out.size
        return out

    def __getitem__(self, key):
        r, c = key
        ra, ca = np.asarray(r), np.asarray(c)
        if ra.ndim == 2 and ca.ndim == 2 and ra.shape[1] == 1 and ca.shape[0] == 1:
            return self._bulk(ra.ravel(), ca.ravel())          # the audit() call pattern
        rb, cb = np.broadcast_arrays(ra, ca)
        rows, cols = np.unique(rb), np.unique(cb)
        block = self._bulk(rows, cols)
        pr = {int(v): a for a, v in enumerate(rows)}
        pc = {int(v): b for b, v in enumerate(cols)}
        gi = np.fromiter((pr[int(v)] for v in rb.ravel()), np.int64, rb.size)
        gj = np.fromiter((pc[int(v)] for v in cb.ravel()), np.int64, cb.size)
        return block[gi, gj].reshape(rb.shape)


def selftest_block_tanimoto(n: int = 120) -> dict:
    """BlockTanimoto must equal the dense matrix on both call shapes, or the leakage audit of the
    external table would be measuring a different quantity from the shipped one."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    smis = ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)ccc1ccccc12", "CCN(CC)CC",
            "O=C(N)c1ccc2[nH]ccc2c1", "C1CCCCC1", "c1ccnc(c1)C(=O)N", "CC(C)C(N)C(=O)O",
            "c1ccc(cc1)Oc1ccccc1", "CC(=O)NCc1ccccc1O"]
    rng = np.random.default_rng(0)
    pool = []
    for i in range(n):
        m = Chem.MolFromSmiles(smis[i % len(smis)])
        side = "".join("cC1"[(j + i) % 3] for j in range(3 + (i % 5)))
        mm = Chem.MolFromSmiles(Chem.MolToSmiles(m) + side)
        pool.append(mm if mm is not None else m)
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in pool]
    from rdkit.DataStructs import cDataStructs
    dense = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dense[i] = cDataStructs.BulkTanimotoSimilarity(fps[i], fps)
    bt = BlockTanimoto(fps)
    checks = {}

    # 1. the outer-product shape 02_splits.audit() uses, with repeated rows and repeated columns
    R = rng.integers(0, n, size=(37, 1))
    C = rng.integers(0, n, size=(1, 41))
    got = bt[R, C]
    checks["audit_call_pattern"] = {
        "block_shape": list(got.shape), "max_abs_diff_vs_dense": float(np.abs(got - dense[R, C]).max()),
        "has_repeated_rows": bool(len(np.unique(R)) < R.shape[0]),
        "has_repeated_cols": bool(len(np.unique(C)) < C.shape[1])}

    # 2. the general path: an arbitrary broadcast shape whose entries are not a rectangle
    a = rng.integers(0, n, size=(23, 7))
    b = rng.integers(0, n, size=(23, 7))
    got2 = bt[a, b]
    checks["elementwise_pairs"] = {
        "block_shape": list(got2.shape),
        "max_abs_diff_vs_dense": float(np.abs(got2 - dense[a, b]).max())}

    # 3. paired indices (t[i, i] style) -- the case a row-wise shortcut is most likely to get wrong
    d = np.arange(n, dtype=np.int64)
    checks["paired_diagonal"] = {
        "block_shape": [n],
        "max_abs_diff_vs_dense": float(np.abs(bt[d, d] - dense[d, d]).max())}

    worst = max(v["max_abs_diff_vs_dense"] for v in checks.values())
    assert worst < 1e-6, f"BlockTanimoto disagrees with the dense matrix by {worst}"
    checks["worst_abs_diff"] = worst
    checks["pass"] = True
    return checks


# --------------------------------------------------------------------------- stage: features

def stage_features(args) -> dict:
    f01 = load_sibling("01_features")
    EXTFEAT.mkdir(parents=True, exist_ok=True)
    f01.PROC = EXT           # the builders read only the tables passed to them; this keeps
    f01.FEAT = EXTFEAT       # any module-level path reference pointed at the external tree
    lig = pd.read_csv(EXT / "ligands.csv")
    tg = pd.read_csv(EXT / "targets.csv")
    t0 = time.time()
    print(f"external ligands {len(lig)}  targets {len(tg)}  rdkit {f01.RDKIT_VERSION}")
    lig_meta = f01.build_ligand_features(lig)
    prot_meta = f01.build_protein_features(tg)
    sdf = f01.scaffold_table(lig)
    sdf.to_csv(EXTFEAT / "ligand_scaffolds.csv", index=False)

    # 01 hard-codes its output paths in the metadata it returns; re-derive them from the files
    # actually written here and re-hash those files, so nothing in this manifest is borrowed.
    for meta in (lig_meta, prot_meta):
        for key, rec in list(meta.items()):
            if isinstance(rec, dict) and "path" in rec:
                name = Path(rec["path"]).name
                p = EXTFEAT / name
                rec["path"] = p.relative_to(ROOT).as_posix()
                rec["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()

    fetch = json.loads((EXT / "external_manifest.json").read_text(encoding="utf-8"))
    man = {
        "generated_by": "scripts/14_external_chembl.py --stage features",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_table": {"pairs": fetch["counts"]["pairs"], "ligands": len(lig),
                         "targets": len(tg), "chembl_release": fetch["chembl_release"],
                         "label": fetch["label_definition"]},
        "rdkit_version": f01.RDKIT_VERSION, "numpy_version": np.__version__,
        "builders": "01_features.build_ligand_features / build_protein_features / "
                    "scaffold_table / bit_vectors (imported, not copied)",
        "fingerprint": "Morgan radius 2 (ECFP4), folded; counts for 2048, binary for 1024, 512",
        "scaffold": "RDKit MurckoScaffoldSmiles, generic (no chirality), '_no_ring_' if acyclic",
        "ligand": lig_meta, "protein_features": prot_meta,
        "scaffolds": {"path": EXTFEAT.relative_to(ROOT).joinpath("ligand_scaffolds.csv").as_posix(),
                      "n_scaffolds": int(sdf.scaffold.nunique()),
                      "largest_scaffold": {str(k): int(v) for k, v in
                                           sdf.scaffold.value_counts().head(3).items()},
                      "sha256": hashlib.sha256((EXTFEAT / "ligand_scaffolds.csv").read_bytes()).hexdigest()},
        "dense_tanimoto_omitted": {
            "why": f"a dense {len(lig)} x {len(lig)} float32 similarity matrix would be "
                   f"{len(lig) ** 2 * 4 / 1e6:.0f} MB (the shipped 2,127-ligand one is 18 MB); "
                   "14 computes only the blocks the leakage audit asks for, from bit_vectors()",
            "equivalence": "BlockTanimoto vs dense checked in --selftest"},
        "unparseable_smiles": {"n": len(lig_meta.get("_bad_smiles", [])),
                               "note": "RDKit substitutes methane for these, exactly as 01 does "
                                       "for the shipped table; a count is reported rather than "
                                       "the structures silently replaced"},
        "wall_seconds": round(time.time() - t0, 1),
    }
    (EXTFEAT / "features_manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    print(f"scaffolds {man['scaffolds']['n_scaffolds']} of {len(sdf)} ligands; "
          f"unparseable SMILES {man['unparseable_smiles']['n']}; "
          f"bad ligand ids {lig_meta['_bad_smiles'][:3]}")
    print("wrote", EXTFEAT, f"({man['wall_seconds']}s)")
    return man


# --------------------------------------------------------------------------- stage: splits

def external_pairs() -> pd.DataFrame:
    return pd.read_csv(EXT / "chembl_ki.csv")


def temporal_dates(pairs: pd.DataFrame) -> list:
    """first_year -> 'YYYY-01-01', missing year -> None (02.temporal_split excludes those rows).

    The external table carries year resolution only, and the year is the *publication* year of the
    source document, not a deposition date -- a documented substitution recorded in the fetch
    manifest. 02's split snaps its cut to a date boundary, so with Jan-1 dates every boundary is a
    whole year and no year can straddle the cut.
    """
    y = pairs["first_year"].to_numpy()
    return [f"{int(v)}-01-01" if pd.notna(v) else None for v in y]


def stage_splits(args) -> dict:
    f02 = load_sibling("02_splits")
    EXTSPL.mkdir(parents=True, exist_ok=True)
    pairs = external_pairs()
    lig_ids = pairs.ligand_id.to_numpy()
    tgt_ids = pairs.target_id.to_numpy()
    scaff = dict(zip(pd.read_csv(EXTFEAT / "ligand_scaffolds.csv").ligand_id,
                     pd.read_csv(EXTFEAT / "ligand_scaffolds.csv").scaffold))
    dates = temporal_dates(pairs)
    rng = f02._rng(f02.SEED)
    protocols = {
        "random": lambda: f02.random_split(None, len(pairs), rng),
        "scaffold": lambda: f02.scaffold_split(lig_ids, scaff, rng),
        "cold_drug": lambda: f02.cold_split(lig_ids, rng),
        "cold_target": lambda: f02.cold_split(tgt_ids, rng),
        "temporal": lambda: f02.temporal_split(dates, rng),
    }
    meta = {}
    for name, fn in protocols.items():
        res = fn()
        split = res[0] if isinstance(res, tuple) else res
        cutoff = res[1] if isinstance(res, tuple) else None
        out = pairs[["pair_id", "ligand_id", "target_id"]].copy()
        out["split"] = split
        kept = out[out.split != "excluded"] if name == "temporal" else out
        kept.to_csv(EXTSPL / f"{DATASET}_{name}.csv", index=False)
        counts = {k: int(v) for k, v in pd.Series(split).value_counts().items()}
        meta[name] = {"counts": counts, "rows_in_protocol": int(len(kept)),
                      "temporal_cutoff": cutoff,
                      "excluded_rows": int((split == "excluded").sum()),
                      "test_fraction_of_kept": round(counts.get("test", 0) / max(len(kept), 1), 4)}
        print(f"  {name:<12} train {counts.get('train', 0):>6}  valid {counts.get('valid', 0):>5}"
              f"  test {counts.get('test', 0):>6}  excluded {counts.get('excluded', 0):>5}"
              f"   cutoff {cutoff}")
    (EXTSPL / "splits_manifest.json").write_text(json.dumps({
        "generated_by": "scripts/14_external_chembl.py --stage splits",
        "functions": "02_splits.random_split / scaffold_split / cold_split / temporal_split "
                     "(imported, not copied)",
        "seed": f02.SEED, "frac_test": f02.FRAC_TEST, "frac_valid": f02.FRAC_VALID,
        "n_pairs_total": int(len(pairs)),
        "temporal_note": "date resolution is one year and the date is the source publication year, "
                         "not a deposition date; undated pairs are excluded and counted",
        "protocols": meta}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", EXTSPL)
    return meta


# --------------------------------------------------------------------------- stage: audit

def protein_similarity(f01) -> tuple[np.ndarray, dict]:
    """Cosine over the 400-dim dipeptide composition, in the same way 02_splits.main builds it."""
    pf = pd.read_csv(EXTFEAT / "protein_features.csv").set_index("target_id")
    tg = pd.read_csv(EXT / "targets.csv")
    D = pf[[c for c in pf.columns if c.startswith("dpc_")]].loc[tg.target_id].to_numpy(float)
    n = np.linalg.norm(D, axis=1, keepdims=True)
    Dn = D / np.where(n == 0, 1, n)
    sim = (Dn @ Dn.T).astype(np.float32)
    return sim, {t: i for i, t in enumerate(tg.target_id)}


def stage_audit(args) -> list:
    f01 = load_sibling("01_features")
    f02 = load_sibling("02_splits")
    pairs = external_pairs()
    lig = pd.read_csv(EXT / "ligands.csv")
    f01.FEAT = EXTFEAT
    t0 = time.time()
    fps = f01.bit_vectors(lig)
    tanimoto = BlockTanimoto(fps)
    lig_index = {l: i for i, l in enumerate(lig.ligand_id)}
    scdf = pd.read_csv(EXTFEAT / "ligand_scaffolds.csv")
    scaff = dict(zip(scdf.ligand_id, scdf.scaffold))
    prot_sim, prot_index = protein_similarity(f01)
    off = prot_sim[~np.eye(len(prot_sim), dtype=bool)]
    print(f"protein dpc-cosine: max off-diagonal {off.max():.4f}, "
          f"pairs >= 0.95: {(off >= 0.95).sum() // 2}")

    lig_ids = pairs.ligand_id.to_numpy()
    tgt_ids = pairs.target_id.to_numpy()
    rows = []
    for name in PROTOCOLS:
        sp_df = pd.read_csv(EXTSPL / f"{DATASET}_{name}.csv")
        # audit() indexes by row position of `pairs`, so the split column must be re-attached to
        # every pair. A protocol table that dropped rows (temporal excludes undated pairs) means
        # "excluded from this protocol" for the missing ones -- a dict lookup, not a per-pair scan
        # of 19,088 rows (which would be 3.6e8 comparisons per protocol).
        split_of_pair = dict(zip(sp_df.pair_id, sp_df.split))
        full = pairs[["pair_id", "ligand_id", "target_id"]].copy()
        full["split"] = [split_of_pair.get(pid, "excluded") for pid in full.pair_id]
        a = f02.audit(full, np.asarray(full["split"]), lig_ids, tgt_ids, tanimoto,
                      lig_index, scaff, prot_sim, prot_index)
        a.update({"dataset": DATASET, "protocol": name,
                  "rows_in_protocol": int(len(sp_df)),
                  "excluded_rows": int(len(pairs) - len(sp_df))})
        rows.append(a)
        print(f"  {name:<12} test {a.get('n_test', 0):>6}  "
              f"max-Tanimoto mean {a.get('ecfp4_max_similarity_to_train_ligand_mean')}  "
              f">=0.7 {a.get('test_pairs_with_near_duplicate_ligand_ge0.7')}  "
              f"scaffold-in-train {a.get('test_scaffold_present_in_train')}")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "leakage_audit.csv", index=False)
    (OUT / "leakage_audit.json").write_text(json.dumps({
        "generated_by": "scripts/14_external_chembl.py --stage audit",
        "audit_function": "02_splits.audit (imported, not copied)",
        "similarity_provider": "BlockTanimoto over 01_features.bit_vectors (same fingerprints, "
                               "no dense n x n matrix)",
        "block_queries": tanimoto.queries, "similarity_entries_computed": tanimoto.entries,
        "dense_matrix_would_be_mb": round(len(lig) ** 2 * 4 / 1e6),
        "protein_dpc_cosine_max_offdiagonal": round(float(off.max()), 4),
        "rows": rows, "wall_seconds": round(time.time() - t0, 1)}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"wrote {OUT / 'leakage_audit.csv'} ({tanimoto.entries:,} similarity entries, "
          f"{round(time.time() - t0, 1)}s)")
    return rows


# --------------------------------------------------------------------------- stage: grid

def _dump_rows(rows: list, path: Path) -> pd.DataFrame:
    """Same column set 03_models.py writes, so the two tables can be read by the same code."""
    df = pd.DataFrame(rows)
    cols = ["dataset", "protocol", "model", "seed", "n_test", "rmse", "mae", "pearson_r",
            "spearman_rho", "ci_harrell", "ci_sklearn_like", "label_tie_fraction", "seconds"] + \
           [c for c in df.columns if c.startswith("info_") or c in
            ("rmse_over_train_sd", "train_sd", "error")]
    df = df[[c for c in dict.fromkeys(cols) if c in df.columns]]
    df.to_csv(path, index=False)
    return df


def stage_grid(args) -> pd.DataFrame:
    import dti_core as core
    core.PROC = EXT
    core.FEAT = EXTFEAT
    f03 = load_sibling("03_models")
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)

    st = core.selftest()
    print("concordance selftest, fast vs brute force (abs diff):",
          json.dumps({k: (v.get("abs_diff", v) if isinstance(v, dict) else v)
                      for k, v in st.items()}, default=float))
    f = core.Featuriser()
    print("external feature block dims:", f.dims)

    protocols = args.protocols.split(",") if args.protocols else PROTOCOLS
    models = args.models.split(",") if args.models else (
        ["mean", "ridge"] if args.quick else ["mean", "ridge", "xgboost", "randomforest", "mlp"])
    seeds = [0] if args.quick else (args.seeds or SEEDS)

    METRICS = OUT / "metrics.csv"
    rows, done = [], set()
    if args.resume and METRICS.exists():
        prev = pd.read_csv(METRICS)
        rows = prev.to_dict("records")
        done = {(r.dataset, r.protocol, r.model, int(r.seed)) for r in prev.itertuples()}
        print(f"resume: {len(done)} rows kept from {METRICS.name}")

    base = pd.read_csv(EXT / "chembl_ki.csv")[["pair_id", "ligand_id", "target_id", YCOL]].set_index("pair_id")
    for protocol in protocols:
        spl = pd.read_csv(EXTSPL / f"{DATASET}_{protocol}.csv")
        y = base.loc[spl.pair_id, YCOL].to_numpy(float)
        lig_ids, tgt_ids = spl.ligand_id.to_numpy(), spl.target_id.to_numpy()
        m = (spl.split == "train").to_numpy()
        v = (spl.split == "valid").to_numpy()
        t = (spl.split == "test").to_numpy()
        if not t.any():
            print(f"skip {protocol}: empty test")
            continue
        ytr = y[m]
        train_sd = float(np.std(ytr))
        for name, pred in (("mean_ligand_train", f03.entity_mean(lig_ids[m], ytr, lig_ids[t])),
                           ("mean_target_train", f03.entity_mean(tgt_ids[m], ytr, tgt_ids[t]))):
            if (DATASET, protocol, name, 0) in done:
                continue
            row = {"dataset": DATASET, "protocol": protocol, "model": name, "seed": 0}
            row.update(core.evaluate(y[t], pred))
            row["seconds"] = 0.0
            row["train_sd"] = round(train_sd, 4)
            row["rmse_over_train_sd"] = round(row["rmse"] / train_sd, 4)
            rows.append(row)
            _dump_rows(rows, METRICS)
            np.savez_compressed(PRED / f"{DATASET}_{protocol}_{name}_seed0.npz",
                                pair_id=spl.pair_id.to_numpy()[t], y=y[t].astype(np.float32),
                                yhat=pred.astype(np.float32))
            print(f"{protocol:<12}{name:<18} RMSE {row['rmse']:.4f} "
                  f"CI {row['ci_harrell']:.4f}  (null)", flush=True)

        for model in models:
            view = f03.MODELS_VIEWS[model]
            if view is None:
                Xtr = np.zeros((int(m.sum()), 1)); Xv = np.zeros((int(v.sum()), 1))
                Xte = np.zeros((int(t.sum()), 1))
            else:
                Xall = f.blocks(lig_ids, tgt_ids, view)
                Xtr, Xv, Xte = f03.standardize(Xall[m], Xall[v], Xall[t])
                del Xall
            for seed in ([0] if model == "mean" else seeds):
                if (DATASET, protocol, model, seed) in done:
                    continue
                if sp.issparse(Xtr) and model in ("xgboost", "randomforest", "mlp"):
                    print("  skip", model, "under a sparse view")
                    continue
                t0 = time.time()
                srow = {"dataset": DATASET, "protocol": protocol, "model": model, "seed": seed}
                valid = (Xv, y[v]) if model in ("ridge", "xgboost", "mlp") else None
                try:
                    pred, info = f03.fit_predict(view, model, Xtr, ytr, Xte, seed, valid)
                    srow.update(core.evaluate(y[t], pred))
                    srow["seconds"] = round(time.time() - t0, 1)
                    srow.update({f"info_{k}": vv for k, vv in info.items()})
                    srow["train_sd"] = round(train_sd, 4)
                    srow["rmse_over_train_sd"] = round(srow["rmse"] / train_sd, 4)
                except Exception as e:                                    # noqa: BLE001
                    srow["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    srow["seconds"] = round(time.time() - t0, 1)
                    srow.update({k: float("nan") for k in
                                 ("n_test", "rmse", "mae", "pearson_r", "spearman_rho",
                                  "ci_harrell", "ci_sklearn_like", "label_tie_fraction")})
                    print(f"  !! {protocol}/{model}/seed{seed} failed: {srow['error']}", flush=True)
                if "error" not in srow:
                    np.savez_compressed(PRED / f"{DATASET}_{protocol}_{model}_seed{seed}.npz",
                                        pair_id=spl.pair_id.to_numpy()[t],
                                        y=y[t].astype(np.float32),
                                        yhat=pred.astype(np.float32))
                rows.append(srow)
                _dump_rows(rows, METRICS)
                if "error" not in srow:
                    print(f"{protocol:<12}{model:<18}seed{seed} RMSE {srow['rmse']:.4f} "
                          f"CI {srow['ci_harrell']:.4f} CI_sk {srow['ci_sklearn_like']:.4f} "
                          f"r {srow['pearson_r']:.4f} ({srow['seconds']}s)", flush=True)
    df = _dump_rows(rows, METRICS)
    agg = (df.groupby(["dataset", "protocol", "model"])
             .agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                  ci=("ci_harrell", "mean"), ci_sklearn_like=("ci_sklearn_like", "mean"),
                  pearson_r=("pearson_r", "mean"), spearman_rho=("spearman_rho", "mean"),
                  n_seeds=("seed", "nunique")).round(4).reset_index())
    agg.to_csv(OUT / "metrics_by_protocol.csv", index=False)
    print("\n" + agg.to_string(index=False))
    print("\nwrote", METRICS)
    return df


# --------------------------------------------------------------------------- stage: analyse

def load_cell(proto: str, arms: list) -> tuple[list, np.ndarray, np.ndarray, dict]:
    """(arm_names, y, P[arms x n], clusters) for one protocol, with the same pairing contract as
    12_testset_bootstrap.load_cell: every arm must be scored on exactly the same rows in the same
    order, or a 'paired' difference is comparing two different test sets. 12's own loader is
    hard-wired to the shipped layout (data/processed/{ds}.csv, results/predictions), so the
    external tree needs a loader that points at its own."""
    pid_ref = y_ref = None
    blocks = []
    for model, seed in arms:
        fp = PRED / f"{DATASET}_{proto}_{model}_seed{seed}.npz"
        if not fp.exists():
            raise SystemExit(f"missing prediction file {fp} (run --stage grid first)")
        z = np.load(fp)
        pid, y, yh = z["pair_id"], z["y"].astype(np.float64), z["yhat"].astype(np.float64)
        if pid_ref is None:
            pid_ref, y_ref = pid, y
        else:
            assert np.array_equal(pid, pid_ref), f"{fp.name}: pair_id order differs"
            assert np.allclose(y, y_ref, atol=0, rtol=0), f"{fp.name}: labels differ"
        blocks.append(yh)
    lab = pd.read_csv(EXT / "chembl_ki.csv")[["pair_id", "ligand_id", "target_id"]] \
        .set_index("pair_id").loc[pid_ref]
    return ([f"{m}_seed{s}" for m, s in arms], y_ref, np.vstack(blocks),
            {"target": lab["target_id"].to_numpy(), "ligand": lab["ligand_id"].to_numpy()})


# --------------------------------------------------------------------------- one bootstrap cell

RESAMPLES = ("row", "target", "ligand")
# the same three resampling units 12_testset_bootstrap uses. `ligand` is the interesting one on
# this table: 14,713 ligands against 19,088 pairs means most ligands appear in one pair only, so
# ligand-cluster resampling is close to row resampling and the two intervals should agree.


def run_cell_ext(job: dict) -> dict:
    """Paired bootstrap for one protocol, in the process 12-style job pools give it.

    Kept at module level, with every path derived from module constants, so it is picklable and
    re-runnable in a spawned child. Returns the (B x arms) draw matrices; the intervals, Holm
    families and protocol contrasts are assembled by the parent, as in 12's assemble()."""
    proto, reps, ci_reps = job["protocol"], job["reps"], job["ci_reps"]
    f12 = load_sibling("12_testset_bootstrap")
    import dti_core as core
    t0 = time.time()
    learned = [m for m in LEARNED if (PRED / f"{DATASET}_{proto}_{m}_seed0.npz").exists()]
    arms = ([(m, 0) for m in NULLS]
            + [(m, s) for m in learned for s in SEEDS
               if (PRED / f"{DATASET}_{proto}_{m}_seed{s}.npz").exists()])
    names, y, P, clusters = load_cell(proto, arms)
    n, n_arms = len(y), P.shape[0]
    E2 = (P - y) ** 2
    AE = np.abs(P - y)
    marg = {m: {} for m in ("rmse", "mae", "ci_harrell")}
    for kind in RESAMPLES:
        groups, parts = (None, None) if kind == "row" else f12.cluster_index_groups(clusters[kind])
        rng = np.random.default_rng(np.random.SeedSequence(
            [BASE_SEED, f12.stable_id(DATASET, proto, kind, "rmse_mae")]))
        R_rmse = np.empty((reps, n_arms))
        R_mae = np.empty((reps, n_arms))
        for b in range(reps):
            idx = f12.draw_indices(kind, n, groups, parts, rng)
            R_rmse[b] = np.sqrt(E2[:, idx].mean(axis=1))
            R_mae[b] = AE[:, idx].mean(axis=1)
        marg["rmse"][kind] = R_rmse
        marg["mae"][kind] = R_mae
        if kind in ("row", "target"):
            # CI is the expensive statistic (38 ms per arm-draw at this n), so it runs at ci_reps,
            # not reps -- the same split 12 uses (2,000 for rmse/mae, 600 for the CI).
            rng_ci = np.random.default_rng(np.random.SeedSequence(
                [BASE_SEED, f12.stable_id(DATASET, proto, kind, "ci")]))
            R_ci = np.empty((ci_reps, n_arms))
            for b in range(ci_reps):
                idx = f12.draw_indices(kind, n, groups, parts, rng_ci)
                yy = y[idx]
                for a in range(n_arms):
                    R_ci[b, a] = core.ci_harrell(yy, P[a][idx])
            marg["ci_harrell"][kind] = R_ci
    return {"protocol": proto, "names": names, "n_test": n,
            "n_clusters": {k: int(len(np.unique(clusters[k]))) for k in ("target", "ligand")},
            "point_rmse": list(np.sqrt(E2.mean(axis=1))), "marg": marg,
            "resamples": list(RESAMPLES), "reps": reps, "ci_reps": ci_reps,
            "seconds": round(time.time() - t0, 1)}


def stage_analyse(args) -> dict:
    f11 = load_sibling("11_variance_decomposition")
    f12 = load_sibling("12_testset_bootstrap")
    metrics = pd.read_csv(OUT / "metrics.csv")
    reps, ci_reps = (200, 40) if args.quick else (args.reps, args.ci_reps)
    if ci_reps < 400:
        print(f"note: ci_reps={ci_reps} is below the 400 needed for a percentile-bootstrap p to "
              f"ever clear Holm's alpha/m at m={len(MODEL_CONTRASTS)}; the CI contrasts will be "
              f"reported with resolvable=False rather than as 'no difference'.")
    metrics["dataset"] = DATASET
    summary: dict = {"generated_by": "scripts/14_external_chembl.py --stage analyse",
                     "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "table": DATASET, "label": YCOL,
                     "boot_reps": reps, "ci_reps": ci_reps,
                     "holm_note": f12.__doc__.strip().split("\n")[0] if f12.__doc__ else None}

    # ---- 1. variance shares: protocol vs model, the headline statistic -------------------
    vd = {}
    rng = np.random.default_rng(BASE_SEED)
    for label, models in (("trained3", f11.TRAINED3), ("all4", f11.ALL4)):
        cell = metrics[metrics.model.isin(models)]
        if cell.empty:
            continue
        means = (cell.groupby(["protocol", "model"])[f11.METRICS].mean().reset_index())
        entry = {"models": models, "n_cells": int(len(means)),
                 "protocols": int(means.protocol.nunique()),
                 "note": "two-way ANOVA on the (protocol x model) cell means, seeds averaged "
                         "first -- the same construction 11_variance_decomposition uses, so "
                         "eta2 here is comparable with the shipped davis/kiba eta2"}
        for metric in f11.METRICS:
            if metric not in means.columns:
                continue
            d = f11.two_way(means, metric)
            d["bootstrap_ci95"] = f11.bootstrap_shares(cell, metric, reps, rng)
            entry[metric] = d
        vd[label] = entry
    summary["variance_decomposition"] = vd

    # ---- 2. spreads the headline sentences quote ------------------------------------------
    learned = [m for m in LEARNED if m in set(metrics.model)]
    # per (protocol, model) mean over seeds is the atom every headline number is built from;
    # groupby("protocol").rmse.mean() alone would average *across* models and answer a different
    # question from "the best model under each protocol".
    per_cell_all = metrics[metrics.model.isin(learned)].groupby(["protocol", "model"]).rmse.mean()
    best = (per_cell_all.groupby(level=0).min().sort_values().reset_index()
            .rename(columns={"rmse": "best_rmse"}))
    summary["protocol_spread_of_the_best_model"] = {
        DATASET: {"per_protocol_best_rmse": {r.protocol: round(r.best_rmse, 4)
                                             for r in best.itertuples()},
                  "easiest": best.iloc[0].protocol, "best_of": round(float(best.iloc[0].best_rmse), 4),
                  "hardest": best.iloc[-1].protocol,
                  "worst_of": round(float(best.iloc[-1].best_rmse), 4),
                  "range": round(float(best.iloc[-1].best_rmse - best.iloc[0].best_rmse), 4)}}
    t3 = metrics[metrics.model.isin(f11.TRAINED3)]
    # per (protocol, model) mean over seeds, then the widest gap between models inside one
    # protocol -- the same construction 07_report.py reads for the shipped tables, so the two
    # numbers mean the same thing. Keyed "dataset/protocol" like results/readme_numbers.json.
    per_cell = t3.groupby(["protocol", "model"]).rmse.mean()
    summary["model_spread_within_protocol_trained3"] = {
        f"{DATASET}/{p}": round(float(g.max() - g.min()), 4)
        for p, g in per_cell.groupby(level=0)}
    summary["model_spread_within_protocol_all4"] = {
        f"{DATASET}/{p}": round(float(g.max() - g.min()), 4)
        for p, g in metrics[metrics.model.isin(f11.ALL4)]
        .groupby(["protocol", "model"]).rmse.mean().groupby(level=0)}
    null = metrics[metrics.model == "mean"].groupby("protocol").rmse.mean()
    summary["best_learned_minus_global_mean_null"] = {
        p: round(float(b - null.get(p, np.nan)), 4)
        for p, b in zip(best.protocol, best.best_rmse)}
    # same row shape as readme_numbers.json["learned_models_worse_than_the_global_mean"], so the
    # two tables can be read by one loop: every (protocol, model) cell whose seed-mean RMSE is
    # worse than the global-train-mean null on that protocol.
    summary["learned_models_worse_than_the_global_mean"] = [
        {"dataset": DATASET, "protocol": p, "model": mo, "rmse": round(float(v), 4),
         "null": round(float(null[p]), 4)}
        for (p, mo), v in per_cell_all.items() if p in null.index and v >= null[p]]
    summary["protocols_where_no_learned_model_beats_the_global_mean"] = {
        "count": int(sum(1 for p, g in per_cell_all.groupby(level=0)
                         if p in null.index and g.min() >= null[p])),
        "of_protocols": int(len(null)),
        "detail": {p: {"global_mean_null": round(float(null[p]), 4),
                       "best_learned": round(float(g.min()), 4),
                       "beaten_by_null": bool(g.min() >= null[p])}
                   for p, g in per_cell_all.groupby(level=0) if p in null.index}}
    gap = (metrics.ci_harrell - metrics.ci_sklearn_like).abs()
    a_gap = gap.idxmax()
    summary["ci_caliber_gap_max"] = {
        "abs_diff": round(float(gap.iloc[a_gap]), 4),
        "row": f"{DATASET}/{metrics.iloc[a_gap].protocol}/{metrics.iloc[a_gap].model}/"
               f"seed{int(metrics.iloc[a_gap].seed)}",
        "ci_harrell": round(float(metrics.iloc[a_gap].ci_harrell), 4),
        "ci_sklearn_like": round(float(metrics.iloc[a_gap].ci_sklearn_like), 4)}
    summary["label_tie_fraction_range"] = [round(float(metrics.label_tie_fraction.min()), 4),
                                           round(float(metrics.label_tie_fraction.max()), 4)]

    # ---- 3. paired bootstrap: marginals, the family of 10 model contrasts, protocol spread
    # The per-protocol bootstrap runs in its own process (12 does the same): a Harrell CI on
    # ~3,800 test rows costs 38 ms here, and 2 x ci_reps x arms x protocols is ~90,000 calls,
    # i.e. an hour serially and ~10 minutes across the pool.
    rmse_draws = {}
    cells_out = []
    protos = [p for p in PROTOCOLS if (PRED / f"{DATASET}_{p}_mean_seed0.npz").exists()]
    jobs = [{"protocol": p, "reps": reps, "ci_reps": ci_reps, "arms_by_proto": {}} for p in protos]
    workers = max(1, min(args.jobs, len(jobs)))
    print(f"{len(jobs)} protocol cells, reps={reps} (rmse/mae) / {ci_reps} (harrell ci), "
          f"jobs={workers}", flush=True)
    t_pool = time.time()
    if workers == 1:
        results = [run_cell_ext(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run_cell_ext, jobs))
    print(f"bootstrap pool: {round(time.time() - t_pool, 1)}s", flush=True)
    for r in results:
        proto, names, marg = r["protocol"], r["names"], r["marg"]
        cell = {"protocol": proto, "n_test": r["n_test"], "arms": names,
                "n_clusters": r["n_clusters"], "seconds": r["seconds"],
                "reps_rmse_mae": r["reps"], "reps_ci": r["ci_reps"]}
        # cross-check the recomputed point RMSE against metrics.csv, the way 12 does: the
        # bootstrap must be scoring the same predictions the grid wrote.
        worst_dev = 0.0
        for nm, pv in zip(names, r["point_rmse"]):
            model, seed = nm.rsplit("_seed", 1)
            row = metrics[(metrics.protocol == proto) & (metrics.model == model)
                          & (metrics.seed == int(seed))]
            assert len(row) == 1, f"{proto}/{nm}: {len(row)} rows in metrics.csv"
            worst_dev = max(worst_dev, abs(float(row.iloc[0]["rmse"]) - float(pv)))
        cell["point_rmse_max_abs_dev_vs_metrics_csv"] = round(worst_dev, 6)
        cell["point_rmse"] = {nm: round(float(p), 4) for nm, p in zip(names, r["point_rmse"])}
        cell["marginal_ci95"] = {
            kind: {met: {nm: [round(float(np.nanpercentile(marg[met][kind][:, a], 2.5)), 4),
                              round(float(np.nanpercentile(marg[met][kind][:, a], 97.5)), 4)]
                         for a, nm in enumerate(names)}
                     for met in ("rmse", "mae", "ci_harrell") if kind in marg[met]}
            for kind in r["resamples"]}
        rmse_draws[proto] = (names, marg["rmse"]["row"], dict(zip(names, r["point_rmse"])))
        # model-level family of 10 (6 learned-vs-learned + 4 learned-vs-global-mean), seed-averaged,
        # Holm within the family -- the same family 12 reports for the shipped tables.
        have_models = {n.rsplit("_seed", 1)[0] for n in names}
        for met in ("rmse", "mae", "ci_harrell"):
            if met not in marg or not all(m in have_models for m in MODEL_SET):
                continue
            mat = f12.model_mean_matrix(marg[met]["row"], names, MODEL_SET)
            pvals, deltas, cis = [], [], []
            for a, b in MODEL_CONTRASTS:
                ia, ib = MODEL_SET.index(a), MODEL_SET.index(b)
                d = mat[:, ia] - mat[:, ib]
                deltas.append(float(d.mean()))
                cis.append([round(float(np.nanpercentile(d, 2.5)), 4),
                            round(float(np.nanpercentile(d, 97.5)), 4)])
                pvals.append(f12.boot_p(d))
            adj = f12.holm(np.array(pvals))
            floor, resolvable = f12.holm_resolvable(int(mat.shape[0]), len(MODEL_CONTRASTS))
            cell[f"model_contrasts_{met}"] = [
                {"pair": f"{a}-vs-{b}", "delta_mean": round(dl, 4), "ci95": ci,
                 "p_boot": round(p, 5), "p_holm": round(q, 5),
                 "significant_at_0.05": bool(q < 0.05)}
                for (a, b), dl, ci, p, q in zip(MODEL_CONTRASTS, deltas, cis, pvals, adj)]
            cell[f"holm_family_{met}"] = {"m": len(MODEL_CONTRASTS), "reps": int(mat.shape[0]),
                                         "p_floor": round(floor, 6), "resolvable": resolvable,
                                         "n_significant": int(sum(q < 0.05 for q in adj))}
        cells_out.append(cell)

    # protocol contrasts: for each arm, the widest protocol-minus-protocol RMSE gap, reusing the
    # row-resample draws already computed above (12's assemble() does the same with
    # marg["rmse"]["row"]). Arms are matched by name, not by position, because a protocol whose
    # predictions are missing contributes fewer arms.
    proto_contrasts = []
    have = [p for p in PROTOCOLS if p in rmse_draws]
    if len(have) >= 2:
        reps_min = min(rmse_draws[p][1].shape[0] for p in have)
        all_arms = [a for a in rmse_draws[have[0]][0]
                    if all(a in rmse_draws[p][0] for p in have)]
        for arm in all_arms:
            draws, pts = {}, {}
            for p in have:
                names, D, pt = rmse_draws[p]
                draws[p] = D[:reps_min, names.index(arm)]
                pts[p] = float(pt[arm])
            lo_p, hi_p = min(pts, key=pts.get), max(pts, key=pts.get)
            if lo_p == hi_p:
                continue
            d = draws[hi_p] - draws[lo_p]
            dl, dh = np.nanpercentile(d, [2.5, 97.5])
            proto_contrasts.append({
                "dataset": DATASET, "model": arm, "metric": "rmse",
                "easiest_protocol": lo_p, "easiest_rmse": round(pts[lo_p], 4),
                "hardest_protocol": hi_p, "hardest_rmse": round(pts[hi_p], 4),
                "spread_point": round(pts[hi_p] - pts[lo_p], 4),
                "spread_lo95": round(float(dl), 4), "spread_hi95": round(float(dh), 4),
                "excludes_zero": bool(dl > 0), "reps_used": int(reps_min),
                "caveat": "the protocols of one table draw their test rows from the same pairs, so "
                          "the two bootstrap draws are combined as if independent and the interval "
                          "is conservative (same treatment as 12_testset_bootstrap)"})
    spread = [c["spread_point"] for c in proto_contrasts]
    summary["protocol_contrasts_rmse"] = proto_contrasts
    summary["protocol_spread_range_rmse"] = ([round(min(spread), 4), round(max(spread), 4)]
                                             if spread else None)
    summary["protocol_spread_excludes_zero_frac"] = (
        round(float(np.mean([c["excludes_zero"] for c in proto_contrasts])), 4)
        if spread else None)
    summary["cells"] = cells_out

    # ---- 4. side by side with the shipped claim -------------------------------------------
    summary["cross_table"] = cross_table()

    (OUT / "paired_bootstrap.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    flat = []
    for c in cells_out:
        for key in ("model_contrasts_rmse",):
            for row in c.get(key, []):
                flat.append({"protocol": c["protocol"], **row})
    if flat:
        pd.DataFrame(flat).to_csv(OUT / "model_contrasts_holm.csv", index=False)
    rows = []
    for c in cells_out:
        for kind, mm in c["marginal_ci95"].items():
            for met, per in mm.items():
                for arm, (lo, hi) in per.items():
                    rows.append({"protocol": c["protocol"], "resample": kind, "metric": met,
                                 "arm": arm, "point": c["point_rmse"][arm],
                                 "lo95": lo, "hi95": hi})
    pd.DataFrame(rows).to_csv(OUT / "arm_marginal_intervals.csv", index=False)
    write_manifest(summary)
    print_headline(summary)
    return summary


def cross_table() -> dict:
    """The three headline sentences, read off the shipped artifacts (read-only) and placed next
    to the external table's own. Nothing here re-runs a shipped model."""
    out = {}
    try:
        rn = json.loads((RESULTS / "readme_numbers.json").read_text(encoding="utf-8"))
        vd = json.loads((RESULTS / "variance_decomposition.json").read_text(encoding="utf-8"))
        out["shipped_protocol_spread_of_the_best_model"] = rn.get("protocol_spread_of_the_best_model")
        out["shipped_model_spread_within_protocol_trained3"] = rn.get(
            "model_spread_within_protocol_trained3")
        out["shipped_learned_models_worse_than_global_mean"] = rn.get(
            "learned_models_worse_than_the_global_mean")
        out["shipped_eta2_protocol_rmse"] = {
            ds: {k: round(vv["rmse"]["eta2_protocol"], 4) for k, vv in blk.items()
                 if isinstance(vv, dict) and "rmse" in vv}
            for ds, blk in vd.get("datasets", {}).items()}
        out["shipped_eta2_model_rmse"] = {
            ds: {k: round(vv["rmse"]["eta2_model"], 4) for k, vv in blk.items()
                 if isinstance(vv, dict) and "rmse" in vv}
            for ds, blk in vd.get("datasets", {}).items()}
        out["note"] = ("read-only from results/readme_numbers.json and "
                       "results/variance_decomposition.json; RMSE across tables is not comparable "
                       "(pKi / pKd / KIBA consensus score are different scales), the protocol-vs-"
                       "model structure is")
    except FileNotFoundError as e:
        out["error"] = f"shipped artifacts unreadable: {e}"
    return out


def write_manifest(summary: dict) -> None:
    files = {}
    for p in sorted(OUT.glob("*")):
        if p.is_file() and p.name != "external_run_manifest.json":   # this file cannot hash itself
            files[p.name] = {"path": p.relative_to(ROOT).as_posix(),
                             "bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    for p in sorted(PRED.glob("*.npz")) if PRED.exists() else []:
        files[f"predictions/{p.name}"] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    fetch = json.loads((EXT / "external_manifest.json").read_text(encoding="utf-8"))
    feat = json.loads((EXTFEAT / "features_manifest.json").read_text(encoding="utf-8")) \
        if (EXTFEAT / "features_manifest.json").exists() else {}
    spl = json.loads((EXTSPL / "splits_manifest.json").read_text(encoding="utf-8")) \
        if (EXTSPL / "splits_manifest.json").exists() else {}
    man = {
        "generated_by": "scripts/14_external_chembl.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "table": {"source": "ChEMBL via scripts/13_fetch_chembl.py",
                  "chembl_release": fetch.get("chembl_release"),
                  "pairs": fetch.get("counts", {}).get("pairs"),
                  "ligands": fetch.get("counts", {}).get("ligands"),
                  "targets": fetch.get("counts", {}).get("targets"),
                  "label": fetch.get("label_definition"),
                  "row_sampling": fetch.get("row_sampling", {}).get("coverage_per_target"),
                  # the final fetcher run was served from the disk page cache (0 bytes on the
                  # wire), so the number that bounds what was actually downloaded is the cache
                  # size; the three-run network total is recorded in the fetch manifest's own
                  # transfer block and in results/run_13_fetch_chembl*.log.
                  "fetch_run_network_bytes": fetch.get("transfer", {}).get("bytes_downloaded"),
                  "fetch_run_network_bytes_note": fetch.get("transfer", {}).get(
                      "bytes_this_process_only"),
                  "raw_cache_bytes_on_disk": fetch.get("transfer", {}).get("raw_cache_bytes_on_disk"),
                  "download_budget_mb": fetch.get("transfer", {}).get("budget_mb"),
                  "download_budget_exceeded": fetch.get("transfer", {}).get("budget_exceeded")},
        "features": {"builders": feat.get("builders"), "rdkit": feat.get("rdkit_version"),
                     "unparseable_smiles": feat.get("unparseable_smiles")},
        "splits": {"seed": spl.get("seed"), "frac_test": spl.get("frac_test"),
                   "protocols": {k: {"test": v["counts"].get("test", 0),
                                      "excluded": v["excluded_rows"],
                                      "cutoff": v["temporal_cutoff"]}
                                 for k, v in spl.get("protocols", {}).items()},
                   "temporal_note": spl.get("temporal_note")},
        "metrics_rows": int(len(pd.read_csv(OUT / "metrics.csv"))) if (OUT / "metrics.csv").exists() else 0,
        "headline": {
            "protocol_spread_range_rmse": summary.get("protocol_spread_range_rmse"),
            "protocol_spread_excludes_zero_frac": summary.get("protocol_spread_excludes_zero_frac"),
            "eta2_protocol_rmse": {k: v.get("rmse", {}).get("eta2_protocol")
                                   for k, v in summary.get("variance_decomposition", {}).items()},
            "eta2_model_rmse": {k: v.get("rmse", {}).get("eta2_model")
                                for k, v in summary.get("variance_decomposition", {}).items()},
            "protocols_where_null_wins": summary.get("protocols_where_no_learned_model_beats_the_global_mean"),
            "ci_caliber_gap_max": summary.get("ci_caliber_gap_max")},
        "files": files,
    }
    (OUT / "external_run_manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")


def print_headline(s: dict) -> None:
    vd = s.get("variance_decomposition", {}).get("trained3", {})
    print("\n=== external table: the headline, re-measured ===")
    if vd:
        r = vd.get("rmse", {})
        print(f"  eta2 protocol {r.get('eta2_protocol', float('nan')):.4f}   "
              f"eta2 model {r.get('eta2_model', float('nan')):.4f}   "
              f"interaction {r.get('eta2_interaction', float('nan')):.4f}")
        ci = r.get("bootstrap_ci95", {})
        print(f"  protocol CI95 {ci.get('eta2_protocol')}  model CI95 {ci.get('eta2_model')}")
    ps = s.get("protocol_spread_of_the_best_model", {}).get(DATASET, {})
    print(f"  best-model RMSE by protocol: {ps.get('per_protocol_best_rmse')}")
    print(f"  spread of the best model: {ps.get('easiest')} {ps.get('best_of')} -> "
          f"{ps.get('hardest')} {ps.get('worst_of')}  (range {ps.get('range')})")
    print(f"  protocol spread (hardest-easiest) RMSE: {s.get('protocol_spread_range_rmse')}"
          f"  frac excluding zero: {s.get('protocol_spread_excludes_zero_frac')}")
    w = s.get("protocols_where_no_learned_model_beats_the_global_mean", {})
    print(f"  null beats every learned model on {w.get('count')}/{w.get('of_protocols')} protocols")
    print(f"  CI caliber gap (Harrell vs sklearn-like), max: {s.get('ci_caliber_gap_max')}")
    print(f"  label tie fraction range: {s.get('label_tie_fraction_range')}")


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", help="all | " + "|".join(STAGES))
    ap.add_argument("--quick", action="store_true", help="1 seed, mean+ridge only (smoke)")
    ap.add_argument("--resume", action="store_true", help="keep results/external_chembl/metrics.csv")
    ap.add_argument("--protocols", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated; default 0,1,2")
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap reps for rmse/mae")
    ap.add_argument("--ci-reps", type=int, default=600,
                    help="Harrell-CI bootstrap reps (>=400 needed for Holm over m=10)")
    ap.add_argument("--jobs", type=int, default=6, help="parallel protocol cells (one process each)")
    ap.add_argument("--selftest", action="store_true", help="check BlockTanimoto and exit")
    args = ap.parse_args()
    if args.seeds:
        args.seeds = [int(s) for s in args.seeds.split(",")]
    if args.selftest:
        print(json.dumps(selftest_block_tanimoto(), indent=1))
        return 0

    t0 = time.time()
    stages = STAGES if args.stage == "all" else args.stage.split(",")
    summary = None
    for st in stages:
        print(f"\n### stage {st} " + "-" * 40)
        if st == "features":
            stage_features(args)
        elif st == "splits":
            stage_splits(args)
        elif st == "audit":
            stage_audit(args)
        elif st == "grid":
            stage_grid(args)
        elif st == "analyse":
            summary = stage_analyse(args)
    print(f"\n[14] stages {stages} done in {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
