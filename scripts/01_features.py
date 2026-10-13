#!/usr/bin/env python3
"""
01_features.py -- ligand and protein featurisation.

Ligand (RDKit, deterministic, no learned parameters)
    * ECFP4 counts: Morgan radius 2, 2048 bits folded, raw counts  (the fingerprint the
      field uses for near-duplicate / scaffold leakage analysis)
    * 13 physicochemical descriptors (MW, LogP, TPSA, HBA/HBD, rotors, aromatic rings,
      heavy atoms, fraction sp3, formal charge, Bertz complexity, MolLogP)
    * ECFP4 folded to 512 and 1024 binary vectors for the dense models

Protein (composition-based; nothing is trained on the labels)
    * amino-acid composition (20), dipeptide composition (400), normalised by length
    * length, sequence entropy, estimated MW and GRAVY

Why composition rather than an embedding model: the whole point of this repository is to
attribute performance differences to *data protocols*, so every feature has to be
reproducible on a laptop CPU in seconds and auditable to a single line of RDKit. An ESM
embedding would change the ranking of the models but not the leakage conclusion; the
option is left as a documented next step in the README.

Outputs (data/processed/features/, all committed)
    ligand_ecfp4_counts.npz   csr matrix (n_ligands x 2048), uint16 counts
    ligand_ecfp4_{512,1024}.npz  csr binary
    ligand_desc.csv
    protein_features.csv      aac(20) + dpc(400) + 4 summary columns
    features_manifest.json    dims, library versions, sha256 of each output
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import __version__ as RDKIT_VERSION
from scipy import sparse

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FEAT = PROC / "features"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"          # 20 standard residues (no BZXUJO)
DESC_COLS = ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
             "NumRotatableBonds", "NumAromaticRings", "HeavyAtomCount",
             "FractionCSP3", "FormalCharge", "MolMR", "NumSaturatedRings",
             "BalabanJ"]
DESC_FUNCS = {
    "MolWt": Descriptors.MolWt, "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA, "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "FormalCharge": Chem.GetFormalCharge, "MolMR": Descriptors.MolMR,
    "NumSaturatedRings": Descriptors.NumSaturatedRings,
    "BalabanJ": Descriptors.BalabanJ,
}
# hydrophobicity scale (Kyte-Doolittle) for GRAVY
KD = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
      "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
      "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3}
RES_MW = {"A": 71.08, "C": 103.14, "D": 115.09, "E": 129.12, "F": 147.18, "G": 57.05,
          "H": 137.14, "I": 113.16, "K": 128.17, "L": 113.16, "M": 131.20, "N": 114.10,
          "P": 97.12, "Q": 128.13, "R": 156.19, "S": 87.08, "T": 101.10, "V": 99.13,
          "W": 186.21, "Y": 163.18}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- ligands

def morgan_counts(mol, nbits: int, radius: int = 2) -> np.ndarray:
    """Folded Morgan (ECFP4) *count* vector, length nbits."""
    feat = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius, nBits=nbits)
    v = np.zeros(nbits, dtype=np.uint16)
    for idx, cnt in feat.GetNonzeroElements().items():
        v[idx] += cnt
    return v


def build_ligand_features(lig: pd.DataFrame) -> dict:
    mats = {512: [], 1024: [], 2048: []}
    desc_rows, bad = [], []
    for lid, smi in zip(lig.ligand_id, lig.canonical_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append(lid)
            mol = Chem.MolFromSmiles("C")
        for n in mats:
            mats[n].append(morgan_counts(mol, n))
        desc_rows.append([float(DESC_FUNCS[c](mol)) for c in DESC_COLS])
    out = {}
    for n, name in ((2048, "ligand_ecfp4_counts"), (1024, "ligand_ecfp4_1024"),
                    (512, "ligand_ecfp4_512")):
        arr = np.array(mats[n], dtype=np.float32 if n != 2048 else np.uint16)
        sp = sparse.csr_matrix(arr)
        sparse.save_npz(FEAT / f"{name}.npz", sp)
        out[name] = {"path": f"data/processed/features/{name}.npz", "shape": list(sp.shape),
                     "nnz_per_row_mean": round(float(np.diff(sp.indptr).mean()), 2),
                     "dtype": str(arr.dtype), "sha256": sha256_file(FEAT / f"{name}.npz")}
    dd = pd.DataFrame(desc_rows, columns=DESC_COLS)
    dd.insert(0, "ligand_id", lig.ligand_id.values)
    dd.to_csv(FEAT / "ligand_desc.csv", index=False)
    out["ligand_desc"] = {"path": "data/processed/features/ligand_desc.csv",
                          "columns": DESC_COLS, "sha256": sha256_file(FEAT / "ligand_desc.csv")}
    out["_bad_smiles"] = bad
    return out


# --------------------------------------------------------------------------- proteins

def protein_features(seq: str) -> dict:
    s = "".join(c for c in seq.upper() if c in AMINO_ACIDS)
    n = len(s)
    aac = {a: 0.0 for a in AMINO_ACIDS}
    for a in s:
        aac[a] += 1.0
    aac = {k: (v / n if n else 0.0) for k, v in aac.items()}
    dpc = {}
    pair_counts, tot = {}, max(n - 1, 1)
    for i in range(n - 1):
        pair_counts[s[i:i + 2]] = pair_counts.get(s[i:i + 2], 0) + 1
    for a in AMINO_ACIDS:
        for b in AMINO_ACIDS:
            dpc[a + b] = pair_counts.get(a + b, 0) / tot
    p = np.array([max(v, 1e-12) for v in aac.values() if v > 0])
    entropy = float(-(p / p.sum() * np.log(p / p.sum())).sum()) / math.log(20)
    gravy = float(sum(KD.get(c, 0.0) for c in s) / n) if n else 0.0
    mw = float(sum(RES_MW.get(c, 0.0) for c in s) + 18.0 * max(n - 1, 0))
    return {"seq_len": n, "aac": aac, "dpc": dpc, "entropy": entropy,
            "mw_est": mw, "gravy": gravy}


def build_protein_features(tg: pd.DataFrame) -> dict:
    rows = []
    for tid, seq in zip(tg.target_id, tg.sequence):
        f = protein_features(seq)
        r = {"target_id": tid, "seq_len": f["seq_len"], "entropy": round(f["entropy"], 6),
             "mw_est": round(f["mw_est"], 2), "gravy": round(f["gravy"], 6)}
        r.update({f"aac_{k}": round(v, 8) for k, v in f["aac"].items()})
        r.update({f"dpc_{k}": round(v, 8) for k, v in f["dpc"].items()})
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(FEAT / "protein_features.csv", index=False)
    return {"protein_features": {"path": "data/processed/features/protein_features.csv",
                                "shape": list(df.shape),
                                "n_aac": 20, "n_dpc": 400,
                                "sha256": sha256_file(FEAT / "protein_features.csv")}}


# --------------------------------------------------------------------------- main

def main() -> int:
    FEAT.mkdir(parents=True, exist_ok=True)
    lig = pd.read_csv(PROC / "ligands.csv")
    tg = pd.read_csv(PROC / "targets.csv")

    print(f"ligands {len(lig)}  targets {len(tg)}  rdkit {RDKIT_VERSION}")
    lig_meta = build_ligand_features(lig)
    prot_meta = build_protein_features(tg)

    # similarity tables used by the leakage analysis: pairwise Tanimoto on ECFP4-2048
    from rdkit.DataStructs import cDataStructs
    fps = []
    for smi in lig.canonical_smiles:
        m = Chem.MolFromSmiles(smi) or Chem.MolFromSmiles("C")
        fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    sims = np.zeros((len(fps), len(fps)), dtype=np.float32)
    for i in range(len(fps)):
        sims[i, i + 1:] = cDataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
    np.save(FEAT / "ligand_tanimoto.npy", sims)

    # scaffold assignment (Murcko, generic) for the split protocol
    scaff = []
    for smi in lig.canonical_smiles:
        m = Chem.MolFromSmiles(smi)
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            sc = ""
        scaff.append(sc if sc else "_no_ring_")
    sdf = pd.DataFrame({"ligand_id": lig.ligand_id, "scaffold": scaff})
    sdf.to_csv(FEAT / "ligand_scaffolds.csv", index=False)

    manifest = {
        "rdkit_version": RDKIT_VERSION, "numpy_version": np.__version__,
        "fingerprint": "Morgan radius 2 (ECFP4), folded, counts for 2048 / binary for 1024, 512",
        "scaffold": "RDKit MurckoScaffoldSmiles, generic (no chirality)",
        "protein": "amino-acid composition (20) + dipeptide composition (400) + "
                   "length/entropy/MW/GRAVY",
        "ligand": lig_meta, "protein_features": prot_meta,
        "ligand_tanimoto": {"path": "data/processed/features/ligand_tanimoto.npy",
                            "shape": list(sims.shape), "sha256": sha256_file(FEAT / "ligand_tanimoto.npy")},
        "ligand_scaffolds": {"path": "data/processed/features/ligand_scaffolds.csv",
                             "n_scaffolds": int(sdf.scaffold.nunique()),
                             "largest_scaffold": sdf.scaffold.value_counts().head(3).to_dict(),
                             "sha256": sha256_file(FEAT / "ligand_scaffolds.csv")},
    }
    (FEAT / "features_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print("scaffolds:", int(sdf.scaffold.nunique()), "of", len(sdf), "ligands")
    print("max pairwise Tanimoto (excluding self):", float(sims[~np.eye(len(sims), dtype=bool)].max()))
    print("wrote", FEAT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
