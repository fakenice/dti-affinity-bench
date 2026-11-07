#!/usr/bin/env python3
"""
15_external_bindingdb.py -- run the headline claim on the BindingDB table (the fourth external
table), reusing 14_external_chembl.py's evaluation machinery *unchanged*.

14 is the reference implementation of the five-protocol / four-model / three-null / paired-
bootstrap grid. It is byte-identical to the version that produced the committed ChEMBL numbers;
this adapter does not edit it. Instead it redirects the same module globals 14 already uses to
isolate an external run (configure_paths), points DATASET at "bindingdb", and swaps the pair
table read from chembl_ki.csv to bindingdb_ki.csv. Two reads in 14 hardcode the filename
(stage_grid line ~476, stage_analyse line ~583); a narrowly-scoped pd.read_csv wrapper redirects
only `external*/chembl_ki.csv` to the BindingDB table while those two stages run, then restores
pd.read_csv exactly. Every other read (ligands.csv, targets.csv, manifests, split files) is
unchanged.

Provenance: 14's stage manifests carry "generated_by: scripts/14_external_chembl.py" (the engine
that actually ran). After each stage this adapter appends a "bindingdb_adapter" block recording
that the run was driven by 15_external_bindingdb.py on the BindingDB table, so the output is not
mistaken for a ChEMBL run. The committed ChEMBL tree (data/processed/external/,
results/external_chembl/) is never written: configure_bindingdb() moves the whole tree to
external_bindingdb/ and results/external_bindingdb/ (same out-tag semantics as the fetch).

Temporal caveat carried forward from the fetch: most BindingDB rows are doi-only with no pmid,
so most pairs are undated and the temporal arm runs on the pmid-dated subset (the fetch manifest
records the coverage). This is a documented limitation, not a silent one.

Usage (mirror 14):
    python scripts/15_external_bindingdb.py --stage all --quick --out-tag _plumbing   # smoke
    python scripts/15_external_bindingdb.py --stage all                               # full grid
    python scripts/15_external_bindingdb.py --stage grid --resume                     # continue
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
sys.path.insert(0, str(SCRIPTS))


def _load14():
    spec = importlib.util.spec_from_file_location("external_chembl_engine",
                                                  SCRIPTS / "14_external_chembl.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["external_chembl_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


ext14 = _load14()

# committed ChEMBL globals, captured before any redirection so restore_chembl() is exact
_CHEMBL_GLOBALS = {k: getattr(ext14, k)
                   for k in ("EXT", "EXTFEAT", "EXTSPL", "OUT", "PRED", "DATASET")}
_ORIGINAL_EXTERNAL_PAIRS = ext14.external_pairs
_ORIGINAL_READ_CSV = pd.read_csv
_ORIGINAL_CONFIGURE_PATHS = ext14.configure_paths


def configure_bindingdb(out_tag: str = "") -> None:
    """Redirect the whole external tree to the BindingDB directories and namespace outputs.

    Uses the SAME out-tag semantics as 15_fetch_bindingdb.py (a suffix on "external_bindingdb"),
    so the adapter reads exactly the directory the fetch wrote: --out-tag _plumbing on both
    scripts lines up. 14's own configure_paths names outputs "external_chembl"+tag, which would
    split the input/output naming; here both input and output are "external_bindingdb"+tag."""
    ext14.EXT = PROC / ("external_bindingdb" + out_tag)
    ext14.EXTFEAT = ext14.EXT / "features"
    ext14.EXTSPL = ext14.EXT / "splits"
    ext14.OUT = RESULTS / ("external_bindingdb" + out_tag)
    ext14.PRED = ext14.OUT / "predictions"
    ext14.DATASET = "bindingdb"
    ext14.external_pairs = lambda: pd.read_csv(ext14.EXT / "bindingdb_ki.csv")
    # 14's bootstrap worker run_cell_ext re-applies the tree tag by calling configure_paths(tag).
    # Redirect that call here too, so the worker lands on the BindingDB tree with DATASET=bindingdb
    # instead of 14's ChEMBL defaults. (The bootstrap is forced in-process below; a Windows 'spawn'
    # child re-imports 14 fresh and would not inherit this redirection.)
    ext14.configure_paths = configure_bindingdb


def restore_chembl() -> None:
    """Undo configure_bindingdb(): committed ChEMBL paths, dataset name, pair reader and the
    configure_paths redirect (restored first, so the call below is the real ChEMBL one)."""
    ext14.configure_paths = _ORIGINAL_CONFIGURE_PATHS
    _ORIGINAL_CONFIGURE_PATHS("")
    for k, v in _CHEMBL_GLOBALS.items():
        setattr(ext14, k, v)
    ext14.external_pairs = _ORIGINAL_EXTERNAL_PAIRS
    pd.read_csv = _ORIGINAL_READ_CSV


def _redirected_read_csv(original):
    """A pd.read_csv wrapper that maps only the external pair-table filename to BindingDB.
    `external*/chembl_ki.csv` (the two hardcoded reads in 14) becomes `bindingdb_ki.csv` in the
    same directory; every other path is passed through untouched."""
    def wrapper(filepath_or_buffer, *a, **kw):
        if isinstance(filepath_or_buffer, (str, Path)):
            p = Path(filepath_or_buffer)
            if p.name == "chembl_ki.csv" and p.parent.name.startswith("external"):
                filepath_or_buffer = p.parent / "bindingdb_ki.csv"
        return original(filepath_or_buffer, *a, **kw)
    return wrapper


def _annotate(manifest_path: Path, stage: str) -> None:
    """Append adapter provenance to a stage manifest 14 wrote (whose generated_by names 14)."""
    if not manifest_path.exists():
        return
    try:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    man["bindingdb_adapter"] = {
        "driven_by": "scripts/15_external_bindingdb.py",
        "engine": "scripts/14_external_chembl.py (imported, unchanged)",
        "table": "BindingDB Ki (data/processed/external_bindingdb/bindingdb_ki.csv)",
        "stage": stage,
        "note": "generated_by names the reused engine; this block records the BindingDB driver",
    }
    manifest_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")


def stage_features(args):
    res = ext14.stage_features(args)
    _annotate(ext14.EXTFEAT / "features_manifest.json", "features")
    return res


def stage_splits(args):
    res = ext14.stage_splits(args)
    _annotate(ext14.EXTSPL / "splits_manifest.json", "splits")
    return res


def stage_audit(args):
    res = ext14.stage_audit(args)
    _annotate(ext14.OUT / "leakage_audit.json", "audit")
    return res


def _correct_table_source(manifest_path: Path) -> None:
    """14's run manifest hardcodes table.source = 'ChEMBL via 13_fetch_chembl.py'. The fetch
    manifest it embeds IS the BindingDB one (source field says BindingDB), so correct the outer
    label to match rather than leave a contradictory provenance string."""
    if not manifest_path.exists():
        return
    try:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    tbl = man.get("table")
    if isinstance(tbl, dict):
        tbl["source"] = "BindingDB via scripts/15_fetch_bindingdb.py"
        tbl["source_note"] = ("the embedded chembl_release block is the BindingDB fetch stamp "
                              "(BindingDB REST carries no version); 14's hardcoded 'ChEMBL' "
                              "label was corrected by the adapter")
    man["bindingdb_adapter"] = {
        "driven_by": "scripts/15_external_bindingdb.py",
        "engine": "scripts/14_external_chembl.py (imported, unchanged)",
        "table": "BindingDB Ki (data/processed/external_bindingdb/bindingdb_ki.csv)",
        "stage": "analyse",
        "note": "generated_by names the reused engine; this block records the BindingDB driver",
    }
    manifest_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")


def stage_grid(args):
    pd.read_csv = _redirected_read_csv(_ORIGINAL_READ_CSV)
    try:
        return ext14.stage_grid(args)
    finally:
        pd.read_csv = _ORIGINAL_READ_CSV


def stage_analyse(args):
    # The paired bootstrap normally runs in a ProcessPoolExecutor. On Windows the pool uses
    # 'spawn': each child re-imports 14 fresh and cannot inherit this adapter's in-process
    # redirection (DATASET, paths, pair reader, configure_paths). So the bootstrap is forced
    # in-process (jobs=1), where run_cell_ext's configure_paths call hits the redirected
    # configure_bindingdb and the pd.read_csv redirect below covers load_cell's chembl_ki read.
    # Cost: the bootstrap is serial (~1 h for the full grid) rather than ~10 min across a pool.
    args.jobs = 1
    pd.read_csv = _redirected_read_csv(_ORIGINAL_READ_CSV)
    try:
        res = ext14.stage_analyse(args)
    finally:
        pd.read_csv = _ORIGINAL_READ_CSV
    _correct_table_source(ext14.OUT / "external_run_manifest.json")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", help="all | " + "|".join(ext14.STAGES))
    ap.add_argument("--quick", action="store_true", help="1 seed, mean+ridge only (smoke)")
    ap.add_argument("--resume", action="store_true", help="keep results/external_bindingdb/metrics.csv")
    ap.add_argument("--protocols", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated; default 0,1,2")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--ci-reps", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out-tag", default="",
                    help="suffix on 'external_bindingdb', matching 15_fetch_bindingdb.py's "
                         "--out-tag; default '' reads data/processed/external_bindingdb/ and "
                         "writes results/external_bindingdb/. A smoke run should pass a distinct "
                         "tag (e.g. _plumbing) so it cannot overwrite the real BindingDB grid.")
    args = ap.parse_args()
    if args.quick and not args.out_tag:
        args.out_tag = "_quick"
    configure_bindingdb(args.out_tag)
    if args.seeds:
        args.seeds = [int(s) for s in args.seeds.split(",")]

    if not (ext14.EXT / "bindingdb_ki.csv").exists():
        print(f"missing {ext14.EXT / 'bindingdb_ki.csv'} -- run 15_fetch_bindingdb.py first",
              file=sys.stderr)
        return 2

    t0 = time.time()
    stages = ext14.STAGES if args.stage == "all" else args.stage.split(",")
    dispatch = {"features": stage_features, "splits": stage_splits, "audit": stage_audit,
                "grid": stage_grid, "analyse": stage_analyse}
    for st in stages:
        print(f"\n### [bindingdb] stage {st} " + "-" * 40)
        dispatch[st](args)
    print(f"\n[15_bindingdb] stages {stages} done in {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())