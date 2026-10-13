#!/usr/bin/env python3
"""
05_docking.py -- AutoDock Vina arm: a training-free, physics-based predictor,
scored on the SAME (kinase, compound) rows and later with the SAME metrics as
03_models.py.

Pipeline, per DAVIS kinase target
  gene symbol / UniProt accession   (data/processed/targets.csv `names`)
    -> reviewed human UniProt sequence          (rest.uniprot.org)
    -> PDBe SIFTS best_structures               (www.ebi.ac.uk/pdbe)
       filter: X-ray, resolution <= --max-res, SIFTS coverage >= --min-cov,
               observed unp span > 100 residues
    -> coordinates                              (files.rcsb.org/download/<pdb>.pdb)
    -> pick the co-crystal drug-like ligand (heavy atoms >= --min-heavy, buffer /
       ion / cryoprotectant blacklist) and use its centroid + bounding box
       + padding as the Vina search box
    -> keep the selected chain's amino acids only (gemmi), parameterise with
       meeko (mk_prepare_receptor, gasteiger charges, -x drops residues that
       fail template matching), dock every DAVIS ligand with tools/vina.exe

Compounds are built from the DAVIS canonical SMILES with RDKit ETKDG + MMFF,
i.e. the docked pose is NOT the crystallographic pose and no protonation/tautomer
micro-state is modelled. That is part of the point: this arm measures what a
rigid-receptor docking score can tell you about measured affinity without ever
being trained on it.

Caching: all downloads and intermediate PDBQT live in data/raw/docking/ (git
ignored); every table the analysis reads is written to results/docking/.

Usage
    python scripts/05_docking.py --max-targets 6 --ligands 24 --jobs 6
    python scripts/05_docking.py --offline              # cached files only
    python scripts/05_docking.py --only-ids T0006       # single target, debug
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Lipinski

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "docking"
OUT = ROOT / "results" / "docking"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

VINA = ROOT / "tools" / "vina.exe"
MKREC = shutil.which("mk_prepare_receptor.exe") or "mk_prepare_receptor"
MKLIG = shutil.which("mk_prepare_ligand.exe") or "mk_prepare_ligand"

ACC_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{2}[0-9][A-Z0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$")

# ligand_monomers entries that are never the drug of interest
BLACKLIST = {
    "HOH", "SO4", "PO4", "GOL", "EDO", "PEG", "P6G", "1PE", "MPD", "DMS", "ACT",
    "TRS", "TAR", "CIT", "EPE", "MES", "IMD", "BME", "FMT", "NO3", "IPA", "ACY",
    "THS", "BEN", "CL", "NA", "K", "MG", "CA", "MN", "ZN", "FE", "FE2", "CO", "NI",
    "CU", "CU1", "SR", "BA", "CS", "W", "BR", "I", "IOD", "FL", "CN", "AZI", "UNX",
    "SCN", "MLI", "OXG", "NAW", "PPI", "PG4", "DXE", "2PE", "7PE", "ATE", "CDS",
    "TLA", "LAC", "GLY", "SER", "CYS", "ALA", "PAS", "SIN", "ACE", "FTT", "BTB",
}


# --------------------------------------------------------------------------- http
def http_get(url, dest: Path, accept=None, offline=False, timeout=120, tries=3):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline:
        raise RuntimeError(f"--offline and cache miss for {dest.name} ({url})")
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dti-affinity-bench/0.1",
                                                       "Accept": accept or "*/*"})
            body = urllib.request.urlopen(req, timeout=timeout).read()
            if not body:
                raise RuntimeError("empty body")
            dest.write_bytes(body)
            return dest
        except Exception as e:                                   # noqa: BLE001
            last, _ = e, time.sleep(1.0 * (k + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def get_json(url, name, offline=False):
    p = RAW / name
    http_get(url, p, accept="application/json", offline=offline)
    return json.loads(p.read_text(encoding="utf-8"))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ local align
def sw_align(q, s, match=2, mism=-1, gap=-2):
    """Smith-Waterman; `gap` is a NEGATIVE penalty. Returns (identity, aln_len, q_cov, s_cov)."""
    q, s = str(q).upper(), str(s).upper()
    if not q or not s:
        return (float("nan"), 0, 0.0, 0.0)
    assert gap <= 0 and mism <= 0
    m, n = len(q), len(s)
    F = np.zeros((m + 1, n + 1), dtype=np.int32)
    best, bi, bj = 0, 0, 0
    for i in range(1, m + 1):
        qi = q[i - 1]
        row, prev = F[i], F[i - 1]
        for j in range(1, n + 1):
            d = prev[j - 1] + (match if qi == s[j - 1] else mism)
            row[j] = max(0, d, prev[j] + gap, row[j - 1] + gap)
            if row[j] > best:
                best, bi, bj = int(row[j]), i, j
    if best == 0:
        return (0.0, 0, 0.0, 0.0)
    i, j, matches, aln = bi, bj, 0, 0
    while i > 0 and j > 0 and F[i, j] > 0:
        cur = int(F[i, j])
        if cur == int(F[i - 1, j - 1]) + (match if q[i - 1] == s[j - 1] else mism):
            matches += int(q[i - 1] == s[j - 1])
            aln += 1
            i, j = i - 1, j - 1
        elif cur == int(F[i - 1, j]) + gap:
            aln += 1
            i -= 1
        else:
            aln += 1
            j -= 1
    return (matches / aln if aln else 0.0, aln, aln / m, aln / n)


def chain_polymer_seq(struct, chain_name):
    out = []
    for res in struct[0][chain_name]:
        info = gemmi.find_tabulated_residue(res.name)
        if info and info.is_amino_acid():
            out.append(info.one_letter_code.upper())
    return "".join(out)


# --------------------------------------------------------------------- uniprot
def resolve_uniprot(names_field, gene_symbol, offline=False):
    toks = [t.strip() for t in str(names_field).split("|") if t.strip()]
    for t in toks:
        if ACC_RE.match(t):
            j = get_json(f"https://rest.uniprot.org/uniprotkb/{t}.json", f"up_{t}.json", offline)
            return t, j["sequence"]["value"], "accession_in_davis_names"
    sym = gene_symbol or (re.split(r"[\s|]", toks[0])[0] if toks else None)
    if not sym:
        return None, None, "no_symbol"
    url = ("https://rest.uniprot.org/uniprotkb/search?query="
           f"gene%3A{sym}+AND+organism_id%3A9606+AND+reviewed%3Atrue"
           "&format=json&size=1&fields=accession,sequence")
    hits = get_json(url, f"up_gene_{sym}.json", offline).get("results", [])
    if not hits:
        return None, None, "uniprot_no_hit"
    h = hits[0]
    return h["primaryAccession"], h["sequence"]["value"], "uniprot_gene_search"


def pick_structure(acc, max_res, min_cov, offline=False):
    j = get_json(f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{acc}",
                 f"pdbe_{acc}.json", offline)
    out = []
    for r in j.get(acc, []):
        if r.get("experimental_method") != "X-ray diffraction":
            continue
        res = r.get("resolution")
        if res is None or res > max_res:
            continue
        span = (r.get("unp_end", 0) or 0) - (r.get("unp_start", 0) or 0) + 1
        cov = r.get("coverage")
        if cov is None:
            L = r.get("unp_length") or span
            cov = span / L if L else 0.0
        if cov < min_cov or (r.get("end", 0) - r.get("start", 0) + 1) < 100:
            continue
        r["cov"] = float(cov)
        out.append(r)
    out.sort(key=lambda r: (round(float(r["resolution"]), 3), -r["cov"], r["pdb_id"], r["chain_id"]))
    return out


# ----------------------------------------------------------------- co-crystal lig
def cocrystal_ligand(pdb_id, chain_name, min_heavy, offline=False):
    j = get_json(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/ligand_monomers/{pdb_id}",
                 f"ligmono_{pdb_id}.json", offline)
    entries = j.get(pdb_id.lower()) or j.get(pdb_id.upper()) or []
    ids = []
    for m in entries:
        cid = m.get("chem_comp_id")
        if m.get("chain_id") == chain_name and cid and cid not in BLACKLIST and cid not in ids:
            ids.append(cid)
    best = None
    for cid in ids:
        cj = get_json(f"https://data.rcsb.org/rest/v1/core/chemcomp/{cid}", f"cc_{cid}.json", offline)
        fw = (cj.get("chem_comp") or {}).get("formula_weight") or 0.0
        smiles = None
        descs = cj.get("pdbx_chem_comp_descriptor") or []
        for d in descs:
            if d.get("type") == "SMILES_CANONICAL" and str(d.get("program", "")).startswith("OpenEye"):
                smiles = d.get("descriptor")
        if smiles is None:
            for d in descs:
                if str(d.get("type", "")).startswith("SMILES"):
                    smiles = d.get("descriptor")
                    break
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() < min_heavy:
            continue
        cand = (cid, smiles, int(mol.GetNumHeavyAtoms()), float(fw))
        if best is None or cand[2] > best[2]:
            best = cand
    return best


def comp_atoms(struct, chain_name, comp_id):
    pts = []
    for res in struct[0][chain_name]:
        if res.name == comp_id:
            pts += [[a.pos.x, a.pos.y, a.pos.z] for a in res]
    return np.array(pts) if pts else None


# ------------------------------------------------------------------------ meeko
def prepare_receptor(struct, chain_name, box, out_base):
    rec = gemmi.Structure()
    rec.add_model(gemmi.Model("1"))
    nc = gemmi.Chain(chain_name)
    for res in struct[0][chain_name]:
        info = gemmi.find_tabulated_residue(res.name)
        if info and info.is_amino_acid():
            nc.add_residue(res)
    if len(nc) == 0:
        return False, "empty_chain", 0
    rec[0].add_chain(nc)
    rec.setup_entities()
    p = Path(str(out_base) + ".trimmed.pdb")
    rec.write_pdb(str(p))
    cx, cy, cz = box["center"]
    sx, sy, sz = box["size"]
    cmd = [MKREC, "--read_pdb", str(p),
           "--box_center", f"{cx:.3f}", f"{cy:.3f}", f"{cz:.3f}",
           "--box_size", f"{sx:.3f}", f"{sy:.3f}", f"{sz:.3f}",
           "--charge_model", "gasteiger", "-x", "-o", str(out_base), "-p", "-j"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(RAW))
    ok = Path(str(out_base) + ".pdbqt").exists() and r.returncode == 0
    blob = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"Template matching failed for: \[(.*?)\]", blob, re.S)
    n_bad = len(m.group(1).split(",")) if m else 0
    return ok, blob[-300:], n_bad


def prepare_ligand(mol, base):
    sdf = Path(str(base) + ".sdf")
    Chem.MolToMolFile(mol, str(sdf), confId=0)
    q = Path(str(base) + ".pdbqt")
    r = subprocess.run([MKLIG, "-i", str(sdf), "-o", str(q)],
                       capture_output=True, text=True, cwd=str(RAW))
    return q.exists(), ((r.stdout or "") + (r.stderr or ""))[-300:]


# -------------------------------------------------------------------------- vina
def run_vina(rec_pdbqt, lig_pdbqt, box, out_pdbqt, exhaustiveness, num_modes, seed,
             cpu="1", verbosity=0):
    cx, cy, cz = box["center"]
    sx, sy, sz = box["size"]
    cmd = [str(VINA), "--receptor", str(rec_pdbqt), "--ligand", str(lig_pdbqt),
           "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}", "--center_z", f"{cz:.3f}",
           "--size_x", f"{sx:.1f}", "--size_y", f"{sy:.1f}", "--size_z", f"{sz:.1f}",
           "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes),
           "--min_rmsd", "1.0", "--seed", str(seed), "--cpu", cpu,
           "--verbosity", str(verbosity), "--out", str(out_pdbqt)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    modes, rmsd = [], []
    if out_pdbqt.exists():
        for line in out_pdbqt.read_text(errors="ignore").splitlines():
            m = re.match(r"REMARK VINA RESULT:\s+([-0-9.]+)\s+([0-9.]+)", line.strip())
            if m:
                modes.append(float(m.group(1)))
                rmsd.append(float(m.group(2)))
    return {"vina_rc": r.returncode, "vina_seconds": round(dt, 1),
            "vina_best": modes[0] if modes else None,
            "vina_n_modes": len(modes),
            "vina_rmsd_max_from_best": (max(rmsd[1:]) if len(rmsd) > 1 else 0.0),
            "vina_err": ((r.stderr or "") + (r.stdout or ""))[-200:] if r.returncode else ""}


# --------------------------------------------------------------------------- main
def selftest():
    """sw_align regression checks -- the panel gate depends on these numbers."""
    a = "MAAAAAAGAGPEMVRGQVFDVGPRYTNLSYIGEGAYGMVCSAYDNVNKVRVAIKKISPFEHQTYCQRTLREIK"
    checks = []
    iden, aln, _, _ = sw_align(a, a)
    checks.append(("identical", abs(iden - 1.0) < 1e-9 and aln == len(a), (iden, aln)))
    sub = a[10:60]
    iden, aln, _, _ = sw_align(sub, a)
    checks.append(("substring", abs(iden - 1.0) < 1e-9 and aln == len(sub), (iden, aln)))
    mut = list(a)
    mut[5] = "W" if mut[5] != "W" else "K"
    iden, aln, _, _ = sw_align("".join(mut), a)
    checks.append(("one_mismatch", abs(iden - (len(a) - 1) / len(a)) < 1e-6, (iden, aln)))
    iden, aln, _, _ = sw_align("FFFFFFFF", "LLLLLLLL")
    checks.append(("disjoint_low", iden < 0.3, (iden, aln)))
    rows = [{"check": n, "ok": bool(ok), "value": [round(float(x), 4) for x in v]}
            for n, ok, v in checks]
    print(json.dumps(rows, indent=1), flush=True)
    if not all(c[1] for c in checks):
        sys.exit("sw_align selftest FAILED -- do not trust identity_vs_davis")
    print("[selftest] sw_align PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-targets", type=int, default=6)
    ap.add_argument("--try-limit", type=int, default=60, help="candidate DAVIS targets examined")
    ap.add_argument("--ligands", type=int, default=24, help="ligands (quantile-spaced on pKd); 0 = all")
    ap.add_argument("--max-res", type=float, default=2.6)
    ap.add_argument("--min-cov", type=float, default=0.35)
    ap.add_argument("--min-heavy", type=int, default=15)
    ap.add_argument("--min-identity", type=float, default=0.90)
    ap.add_argument("--padding", type=float, default=5.0)
    ap.add_argument("--max-box-dim", type=float, default=30.0)
    ap.add_argument("--exhaustiveness", type=int, default=4)
    ap.add_argument("--num-modes", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--decoy", type=int, default=12, help="pairs re-docked with a box shifted into solvent")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--only-ids", default=None, help="comma-separated target_ids (debug)")
    ap.add_argument("--resume", action="store_true", help="skip pairs whose vina output already exists")
    args = ap.parse_args()
    selftest()

    if not VINA.exists():
        sys.exit(f"missing {VINA} (AutoDock Vina 1.2.7 win exe) -- see docs/data_sources.md")
    (OUT / "versions.json").write_text(json.dumps({
        "rdkit": Chem.rdBase.rdkitVersion, "gemmi": gemmi.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0],
        "vina_sha256": sha256(VINA), "vina_bytes": VINA.stat().st_size,
        "mk_prepare_receptor": MKREC, "mk_prepare_ligand": MKLIG}, indent=2), encoding="utf-8")

    d = pd.read_csv(ROOT / "data" / "processed" / "davis.csv")
    lig = pd.read_csv(ROOT / "data" / "processed" / "ligands.csv")
    tgt = pd.read_csv(ROOT / "data" / "processed" / "targets.csv")
    lmap = dict(zip(lig.ligand_id, lig.canonical_smiles))
    tmap = dict(zip(tgt.target_id, tgt.names))
    seqmap = dict(zip(tgt.target_id, tgt.sequence))
    name_by_tid = d.groupby("target_id").target_name.first().to_dict()

    # ---------------------------------------------------------------- target panel
    panel, rejects = [], []
    cand_ids = (args.only_ids.split(",") if args.only_ids else sorted(d.target_id.unique()))
    for tid in cand_ids[:args.try_limit]:
        if len(panel) >= args.max_targets:
            break
        gene, names = name_by_tid.get(tid), tmap.get(tid)
        note = {"target_id": tid, "gene": gene, "davis_names": names}
        try:
            acc, useq, how = resolve_uniprot(names, gene, args.offline)
            note["uniprot_acc"], note["uniprot_source"] = acc, how
            if not acc:
                panel_note(rejects, note, "no_uniprot")
                continue
            cands = pick_structure(acc, args.max_res, args.min_cov, args.offline)
            note["n_structures_pass_filters"] = len(cands)
            got = None
            for c in cands[:4]:
                pdb_id, ch = str(c["pdb_id"]).lower(), c["chain_id"]
                p = RAW / f"{pdb_id}.pdb"
                http_get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", p, offline=args.offline)
                st = gemmi.read_structure(str(p))
                st.setup_entities()
                try:
                    st.remove_alternative_conformations()
                except Exception:                                  # noqa: BLE001
                    pass
                if ch not in [x.name for x in st[0]]:
                    continue
                pseq = chain_polymer_seq(st, ch)
                if len(pseq) < 80:
                    continue
                id_d, aln, qc, _ = sw_align(pseq, seqmap[tid])
                id_u, _, _, _ = sw_align(pseq, useq)
                lc = cocrystal_ligand(pdb_id, ch, args.min_heavy, args.offline)
                if lc is None:
                    continue
                pts = comp_atoms(st, ch, lc[0])
                if pts is None or len(pts) < 5:
                    continue
                ctr, ext = pts.mean(0), pts.max(0) - pts.min(0) + 2 * args.padding
                if np.any(np.array(ext) > args.max_box_dim) or np.any(np.array(ext) < 10):
                    continue
                box = {"center": [round(float(v), 3) for v in ctr],
                       "size": [round(float(v), 3) for v in ext]}
                base = RAW / f"rec_{tid}_{pdb_id}_{ch}"
                ok, msg, nbad = prepare_receptor(st, ch, box, base)
                if not ok:
                    continue
                got = {"pdb_id": pdb_id, "chain": ch, "resolution": float(c["resolution"]),
                       "pdbe_coverage": round(float(c["cov"]), 3), "pdb_seq_len": len(pseq),
                       "identity_vs_davis": round(float(id_d), 4), "aln_len_vs_davis": int(aln),
                       "davis_seq_len": int(len(str(seqmap[tid]))),
                       "identity_vs_uniprot": round(float(id_u), 4),
                       "cocrystal_ligand": lc[0], "cocrystal_smiles": lc[1],
                       "cocrystal_heavy": lc[2], "cocrystal_weight": lc[3],
                       "receptor_pdbqt": str(Path(str(base) + ".pdbqt").relative_to(ROOT)), "box": box,
                       "meeko_bad_residues_dropped": int(nbad), "meeko_tail": msg}
                break
            if got is None:
                panel_note(rejects, note, "no_usable_structure")
                continue
            note.update(got)
            if note["identity_vs_davis"] < args.min_identity:
                panel_note(rejects, note, f"identity_vs_davis<{args.min_identity}")
                continue
            note["reject"] = ""
            panel.append(note)
            print(f"[panel] {tid} {gene} -> {got['pdb_id']}_{got['chain']} {got['resolution']}A "
                  f"cov={got['pdbe_coverage']} id_davis={got['identity_vs_davis']} "
                  f"id_uniprot={got['identity_vs_uniprot']} lig={got['cocrystal_ligand']} "
                  f"box={got['box']['size']} ({len(panel)}/{args.max_targets})", flush=True)
        except Exception as e:                                     # noqa: BLE001
            panel_note(rejects, note, f"error:{type(e).__name__}:{str(e)[:140]}")
    if not panel:
        sys.exit("no DAVIS target passed the structural filters -- loosen --max-res/--min-cov/--min-identity")

    panel_df = pd.DataFrame(panel)
    panel_df["box_json"] = panel_df.box.map(json.dumps)
    panel_df.drop(columns=["box"]).to_csv(OUT / "target_panel.csv", index=False)

    # ---------------------------------------------------------------- ligand panel
    sub = d[d.target_id.isin([p["target_id"] for p in panel])]
    per_lig = sub.groupby("ligand_id").pkd.agg(["mean", "std", "count"])
    order = list(per_lig.sort_values("mean").index)
    if args.ligands and args.ligands < len(order):
        pick = sorted({order[i] for i in np.linspace(0, len(order) - 1, args.ligands).round().astype(int)})
    else:
        pick = order
    print(f"[ligands] preparing {len(pick)} of {len(order)} DAVIS compounds", flush=True)

    prep = []
    for lid in pick:
        smi = lmap[lid]
        row = {"ligand_id": lid, "smiles": smi,
               "mean_pkd_panel": round(float(per_lig.loc[lid, "mean"]), 4)}
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                raise RuntimeError("MolFromSmiles returned None")
            frags = Chem.GetMolFrags(m, asMols=True)
            if len(frags) > 1:
                row["n_fragments_input"] = len(frags)
                m = max(frags, key=lambda f: f.GetNumHeavyAtoms())
                row["largest_fragment_only"] = True
            Chem.SanitizeMol(m)
            row["heavy"] = int(m.GetNumHeavyAtoms())
            row["formula"] = rdMolDescriptors.CalcMolFormula(m)
            row["rotatable"] = int(Lipinski.NumRotatableBonds(m))
            h = Chem.AddHs(m)
            ps = AllChem.ETKDGv3()
            ps.randomSeed = 42
            row["embed"] = int(AllChem.EmbedMolecule(h, ps))
            if row["embed"] != 0:
                row["embed"] = int(AllChem.EmbedMolecule(h, useRandomCoords=True, randomSeed=42))
            row["mmff"] = int(AllChem.MMFFOptimizeMolecule(h, maxIters=2000)) if h.GetNumConformers() else -1
            row["pdbqt"] = False
            if h.GetNumConformers():
                okk, msg = prepare_ligand(h, RAW / lid)
                row["pdbqt"] = bool(okk)
                if not okk:
                    row["prep_err"] = msg
        except Exception as e:                                     # noqa: BLE001
            row["pdbqt"] = False
            row["error"] = f"{type(e).__name__}:{str(e)[:140]}"
        prep.append(row)
        if not row["pdbqt"]:
            print(f"  [lig] {lid} FAILED {row.get('error', row.get('prep_err', ''))[:100]}", flush=True)
    PREP_COLS = ["ligand_id", "smiles", "mean_pkd_panel", "heavy", "formula", "rotatable",
                "n_fragments_input", "largest_fragment_only", "embed", "mmff", "pdbqt",
                "prep_err", "error"]
    pd.DataFrame(prep).reindex(columns=PREP_COLS).to_csv(OUT / "ligand_prep.csv", index=False)
    good = [r["ligand_id"] for r in prep if r.get("pdbqt")]
    print(f"[ligands] {len(good)}/{len(pick)} pdbqt ready", flush=True)

    # ---------------------------------------------------------------------- docking
    pairs = sub[sub.ligand_id.isin(good)].sort_values("pair_id")
    if len(pairs) == 0 or not good:
        sys.exit(f"no dockable ligand survived preparation ({len(good)}/{len(pick)}); "
                 "see results/docking/ligand_prep.csv")
    boxmap = {p["target_id"]: (ROOT / p["receptor_pdbqt"], p["box"]) for p in panel}
    jobs = []
    for r in pairs.itertuples():
        recq, box = boxmap[r.target_id]
        outq = RAW / f"out_{r.target_id}_{r.ligand_id}.pdbqt"
        jobs.append({"pair_id": int(r.pair_id), "target_id": r.target_id, "ligand_id": r.ligand_id,
                     "pkd": float(r.pkd), "rec": recq, "box": box, "out": outq,
                     "cached": outq.exists() and outq.stat().st_size > 200})
    n_cached = sum(1 for j in jobs if j["cached"])
    print(f"[dock] {len(jobs)} pairs ({n_cached} already cached), exhaustiveness="
          f"{args.exhaustiveness}, num_modes={args.num_modes}, jobs={args.jobs}", flush=True)

    def work(j):
        if j["cached"]:
            txt = j["out"].read_text(errors="ignore")
            sc = re.findall(r"REMARK VINA RESULT:\s+([-0-9.]+)\s+([0-9.]+)", txt)
            return j["pair_id"], {"vina_best": float(sc[0][0]) if sc else None,
                                  "vina_n_modes": len(sc),
                                  "vina_rmsd_max_from_best": (max(float(x[1]) for x in sc[1:])
                                                              if len(sc) > 1 else 0.0),
                                  "vina_seconds": 0.0, "vina_rc": 0, "cached": True}
        res = run_vina(j["rec"], RAW / f"{j['ligand_id']}.pdbqt", j["box"], j["out"],
                       args.exhaustiveness, args.num_modes, seed=42 + j["pair_id"])
        res["cached"] = False
        return j["pair_id"], res

    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            pid, res = f.result()
            results[pid] = res
            if n % 20 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  [dock] {n}/{len(jobs)}  {el/60:.1f} min  {el/max(n,1):.1f}s/pair", flush=True)

    rows = []
    for j in jobs:
        res = results.get(j["pair_id"], {})
        rows.append({"pair_id": j["pair_id"], "target_id": j["target_id"], "ligand_id": j["ligand_id"],
                    "pkd": j["pkd"], "vina_best": res.get("vina_best"),
                    "vina_n_modes": res.get("vina_n_modes"),
                    "vina_rmsd_max_from_best": res.get("vina_rmsd_max_from_best"),
                    "vina_rc": res.get("vina_rc"), "vina_seconds": res.get("vina_seconds"),
                    "from_cache": bool(res.get("cached"))})
    out = pd.DataFrame(rows).sort_values("pair_id")
    meta = pairs[["pair_id", "target_name", "kd_nM", "published_part"]].copy()
    meta["pair_id"] = meta.pair_id.astype(int)
    out = out.merge(meta, on="pair_id", how="left")
    out.to_csv(OUT / "pairs.csv", index=False)
    print(f"[dock] {OUT/'pairs.csv'} rows={len(out)} scored={int(out.vina_best.notna().sum())} "
          f"failures={int((out.vina_rc != 0).sum())}", flush=True)

    # ----------------------------------------------------- decoy-box specificity control
    decoy = []
    if args.decoy and len(jobs):
        sel = jobs[::max(1, len(jobs) // args.decoy)][:args.decoy]
        print(f"[decoy] {len(sel)} pairs re-docked with the box shifted +30 A in x", flush=True)
        for j in sel:
            b2 = {"center": [j["box"]["center"][0] + 30.0] + j["box"]["center"][1:],
                  "size": j["box"]["size"]}
            o2 = RAW / f"decoy_{j['pair_id']}.pdbqt"
            res = run_vina(j["rec"], RAW / f"{j['ligand_id']}.pdbqt", b2, o2,
                           args.exhaustiveness, args.num_modes, seed=42 + j["pair_id"])
            decoy.append({"pair_id": j["pair_id"], "target_id": j["target_id"],
                          "ligand_id": j["ligand_id"], "pkd": j["pkd"],
                          "vina_shifted": res.get("vina_best"), "vina_rc": res.get("vina_rc")})
        pd.DataFrame(decoy).to_csv(OUT / "decoy_control.csv", index=False)
        if decoy:
            print(f"[decoy] mean shifted score {np.nanmean([d['vina_shifted'] for d in decoy if d['vina_shifted'] is not None]):.2f} "
                  f"vs on-site mean {out.vina_best.mean():.2f}", flush=True)

    # --------------------------------------------------------------------- manifest
    man = {
        "script": "scripts/05_docking.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vina_binary": {"path": "tools/vina.exe", "sha256": sha256(VINA),
                        "bytes": VINA.stat().st_size,
                        "provenance": "AutoDock Vina 1.2.7 Windows exe, GitHub release asset"},
        "structural_filters": {"xray_only": True, "max_resolution_A": args.max_res,
                               "min_sifts_coverage": args.min_cov,
                               "min_cocrystal_heavy_atoms": args.min_heavy,
                               "min_local_identity_vs_davis": args.min_identity,
                               "box_padding_A": args.padding, "max_box_dim_A": args.max_box_dim},
        "docking_settings": {"exhaustiveness": args.exhaustiveness, "num_modes": args.num_modes,
                             "min_rmsd": 1.0, "seed_base": 42, "cpu_per_job": 1,
                             "parallel_jobs": args.jobs, "grid_spacing_A": 0.375},
        "targets_examined": min(len(cand_ids), args.try_limit),
        "panel": [{"target_id": p["target_id"], "gene": p["gene"], "uniprot": p["uniprot_acc"],
                   "pdb": p["pdb_id"], "chain": p["chain"], "resolution_A": p["resolution"],
                   "sifts_coverage": p["pdbe_coverage"],
                   "identity_vs_davis": p["identity_vs_davis"],
                   "identity_vs_uniprot": p["identity_vs_uniprot"],
                   "cocrystal_ligand": p["cocrystal_ligand"],
                   "cocrystal_heavy_atoms": p["cocrystal_heavy"],
                   "meeko_bad_residues_dropped": p["meeko_bad_residues_dropped"],
                   "box_center": p["box"]["center"], "box_size": p["box"]["size"]} for p in panel],
        "rejected": rejects,
        "counts": {"panel_targets": len(panel), "ligands_requested": len(pick),
                   "ligands_prepared": len(good), "pairs": len(out),
                   "pairs_scored": int(out.vina_best.notna().sum()),
                   "vina_failures": int((out.vina_rc != 0).sum()),
                   "decoy_pairs": len(decoy)},
        "wallclock_dock_seconds": round(time.time() - t0, 1),
        "offline": args.offline,
        "caveats": [
            "docked poses come from RDKit ETKDG+MMFF of the DAVIS canonical SMILES, not from any crystal structure; single tautomer/protonation state, no pH model",
            "receptor is rigid (side chains fixed at the co-crystal conformation) and one apo/cryo structure stands in for a whole kinase",
            "the search box is centred on a DIFFERENT compound (the co-crystal ligand), so a DAVIS ligand that binds elsewhere is scored in the wrong site",
            "69.6% of DAVIS pKd values sit at the assay ceiling (Kd clipped at 10000 nM), which caps any correlation with a continuous score",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    print("[done] results/docking/{target_panel,ligand_prep,pairs,decoy_control}.csv + manifest.json", flush=True)


def panel_note(rejects, note, why):
    if any(r.get("target_id") == note.get("target_id") for r in rejects):
        return
    n = dict(note)
    n["reject"] = why
    rejects.append(n)
    print(f"[skip] {note.get('target_id')} {note.get('gene')}: {why}", flush=True)


if __name__ == "__main__":
    main()
