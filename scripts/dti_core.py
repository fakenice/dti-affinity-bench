#!/usr/bin/env python3
"""
dti_core.py -- shared feature assembly and metrics for the affinity benchmark.

Why a separate module: scripts 03, 04 and the notebook all need *identical* features and
identical metric definitions, and neither should redefine them (a notebook that quietly
computes concordance differently than the table in the README is exactly the failure mode
this repository is about).

Concordance index, two calibers, both reported:

  ci_harrell      denominator = comparable pairs only (labels unequal). Tied predictions
                  get 0.5 credit. This is Harrell's C as used in the survival and lifelines
                  literature, and as used by the DTI field when it says "CI".
  ci_sklearn_like denominator = every pair; a pair with equal labels contributes 0.5.
                  This is what scikit-learn's (now removed) `concordance_index` returned.
                  On DAVIS it is dragged toward 0.5 by the assay ceiling: 69.6% of rows sit
                  at pKd = 5.0, so most pairs are label-tied and carry no signal.

Both are computed in O(n log n) with a Fenwick tree over prediction ranks; a brute-force
O(n^2) cross-check runs on small synthetic data in `selftest()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FEAT = PROC / "features"

N_TOP_BITS = 256


# --------------------------------------------------------------------------- concordance

class _Fenwick:
    __slots__ = ("tree", "n")

    def __init__(self, n: int):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def add(self, i: int, v: int = 1) -> None:
        i += 1
        while i <= self.n:
            self.tree[i] += v
            i += i & (-i)

    def prefix(self, i: int) -> int:
        """sum over indices 0..i-1 (i elements strictly below index i)"""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s


def ci_denominators(y) -> dict:
    """Pair counts behind a concordance index, so a CI can be read as evidence or noise.

    `ci_harrell` divides by the number of *comparable* pairs, i.e. pairs whose two labels
    differ. On censored data (DAVIS: 69.6% of labels sit on the assay ceiling) that
    denominator is a tiny slice of all pairs, and a 20-point block can therefore report
    CI = 1.000 on a handful of pairs. This function makes the denominator explicit so the
    per-target numbers in 06 are interpretable instead of flattering.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    total = n * (n - 1) // 2
    counts = np.unique(y, return_counts=True)[1]
    tied = int(sum(c * (c - 1) // 2 for c in counts))
    return {"n_pairs": int(total), "n_tied_label_pairs": tied,
            "n_comparable_pairs": int(total - tied)}


def ci_harrell(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n < 2:
        return float("nan")
    ur, counts = np.unique(p, return_counts=True)
    pr = np.searchsorted(ur, p).astype(np.int64)      # p values are all in ur
    ft = _Fenwick(len(ur))
    total_counts = np.zeros(len(ur), dtype=np.int64)
    order = np.argsort(y, kind="stable")
    ys = y[order]
    conc = tie = disc = 0
    seen = 0
    start = 0
    while start < n:
        end = start
        while end < n and ys[end] == ys[start]:
            end += 1
        grp = order[start:end]
        gp = pr[grp]
        for k in range(len(grp)):
            r = gp[k]
            lt = ft.prefix(r)                      # strictly smaller prediction
            eq = total_counts[r]                   # equal prediction
            le = ft.prefix(r + 1)                  # <= r
            gt = seen - le
            conc += lt
            tie += eq
            disc += gt
        for r in gp:
            ft.add(int(r), 1)
            total_counts[r] += 1
        seen += len(grp)
        start = end
    comparable = conc + tie + disc
    if comparable == 0:
        return float("nan")
    return float((conc + 0.5 * tie) / comparable)


def ci_sklearn_like(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n < 2:
        return float("nan")
    ur, counts = np.unique(p, return_counts=True)
    pr = np.searchsorted(ur, p).astype(np.int64)
    ft = _Fenwick(len(ur))
    total_counts = np.zeros(len(ur), dtype=np.int64)
    order = np.argsort(y, kind="stable")
    ys = y[order]
    conc = tie_pred = 0
    tied_y_pairs = 0
    seen = 0
    start = 0
    while start < n:
        end = start
        while end < n and ys[end] == ys[start]:
            end += 1
        grp = order[start:end]
        gp = pr[grp]
        for r in gp:
            conc += ft.prefix(int(r))
            tie_pred += total_counts[int(r)]
        for r in gp:
            ft.add(int(r), 1)
            total_counts[int(r)] += 1
        m = len(grp)
        tied_y_pairs += m * (m - 1) // 2
        seen += m
        start = end
    total = n * (n - 1) // 2
    # every prediction tie counted here is between rows with unequal labels (groups are
    # processed in ascending label order and only earlier groups are in the tree)
    numerator = conc + 0.5 * tie_pred + 0.5 * tied_y_pairs
    return float(numerator / total)


def ci_bruteforce(y, p, mode="harrell"):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    n = len(y)
    i, j = np.triu_indices(n, 1)
    a, b = y[i], y[j]
    pa, pb = p[i], p[j]
    same_y = a == b
    conc = (a != b) & (((a < b) & (pa < pb)) | ((a > b) & (pa > pb)))
    tie_p = (a != b) & (pa == pb)
    if mode == "harrell":
        comp = (~same_y).sum()
        return float((conc.sum() + 0.5 * tie_p.sum()) / comp) if comp else float("nan")
    total = n * (n - 1) // 2
    return float((conc.sum() + 0.5 * tie_p.sum() + 0.5 * same_y.sum()) / total)


# --------------------------------------------------------------------------- metrics

def evaluate(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ny = len(np.unique(y))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(pearsonr(y, p)[0]) if np.std(p) > 0 and np.std(y) > 0 else float("nan")
        rho = float(spearmanr(y, p)[0]) if np.std(p) > 0 and np.std(y) > 0 else float("nan")
    return {
        "n_test": int(len(y)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
        "pearson_r": r,
        "spearman_rho": rho,
        "ci_harrell": ci_harrell(y, p),
        "ci_sklearn_like": ci_sklearn_like(y, p),
        "label_tie_fraction": float(1 - ny / len(y)),
    }


def selftest(rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    out = {}
    for n in (12, 60, 220):
        y = rng.integers(0, 5, n).astype(float)
        p = rng.random(n).round(3)
        for mode, fast, slow in (("harrell", ci_harrell(y, p), ci_bruteforce(y, p, "harrell")),
                                 ("sklearn_like", ci_sklearn_like(y, p),
                                  ci_bruteforce(y, p, "sklearn_like"))):
            out[f"n{n}_{mode}"] = {"fast": round(fast, 10), "brute": round(slow, 10),
                                   "abs_diff": abs(fast - slow)}
            assert abs(fast - slow) < 1e-9, (n, mode, fast, slow)
        # all-tied labels and perfect predictions
    out["perfect"] = ci_harrell(np.arange(10.0), np.arange(10.0))
    out["reversed"] = ci_harrell(np.arange(10.0), -np.arange(10.0))
    assert out["perfect"] == 1.0 and out["reversed"] == 0.0
    return out


# --------------------------------------------------------------------------- features

class Featuriser:
    """Concatenation of a ligand block and a protein block; three matrix views."""

    def __init__(self):
        self.lig = pd.read_csv(PROC / "ligands.csv")
        self.lig_index = {l: i for i, l in enumerate(self.lig.ligand_id)}
        self.fp = sp.load_npz(FEAT / "ligand_ecfp4_counts.npz").astype(np.float32)
        self.fp512 = sp.load_npz(FEAT / "ligand_ecfp4_512.npz").astype(np.float32)
        d = pd.read_csv(FEAT / "ligand_desc.csv").set_index("ligand_id")
        self.desc = d.loc[self.lig.ligand_id].to_numpy(np.float32)
        pf = pd.read_csv(FEAT / "protein_features.csv").set_index("target_id")
        self.prot_ids = list(pf.index)
        self.prot_index = {t: i for i, t in enumerate(self.prot_ids)}
        dense = pf[[c for c in pf.columns if c.startswith(("aac_", "dpc_"))]].to_numpy(np.float32)
        extras = pf[["seq_len", "entropy", "mw_est", "gravy"]].to_numpy(np.float32)
        self.prot_dense = np.hstack([dense, extras])
        freq = np.asarray((self.fp > 0).sum(axis=0)).ravel()
        self.top_bits = np.argsort(-freq, kind="stable")[:N_TOP_BITS]
        self.dims = {"sparse_full": self.fp.shape[1] + self.desc.shape[1] + self.prot_dense.shape[1],
                     "dense_compact": N_TOP_BITS + self.desc.shape[1] + self.prot_dense.shape[1],
                     "dense_mlp": self.fp512.shape[1] + self.desc.shape[1] + self.prot_dense.shape[1]}

    def blocks(self, lig_ids, target_ids, view):
        li = np.fromiter((self.lig_index[l] for l in lig_ids), dtype=np.int64, count=len(lig_ids))
        ti = np.fromiter((self.prot_index[t] for t in target_ids), dtype=np.int64,
                         count=len(target_ids))
        if view == "sparse_full":
            f = self.fp[li].tocsr(copy=True)
            f.data = np.log1p(f.data)
            dense = np.hstack([self.desc[li], self.prot_dense[ti]])
            return sp.hstack([f, sp.csr_matrix(dense)], format="csr")
        if view == "dense_compact":
            fp = np.log1p(np.asarray(self.fp[li][:, self.top_bits].todense(), dtype=np.float32))
            return np.hstack([fp, self.desc[li], self.prot_dense[ti]])
        if view == "dense_mlp":
            fp = np.asarray(self.fp512[li].todense(), dtype=np.float32)
            return np.hstack([fp, self.desc[li], self.prot_dense[ti]])
        raise ValueError(view)
