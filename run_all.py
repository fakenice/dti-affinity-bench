#!/usr/bin/env python3
"""One-command reproduction of every number in this repository.

    python run_all.py                      # features -> splits -> models -> intervals -> report
    python run_all.py --fetch              # also re-download DAVIS/KIBA (needs api.github.com)
    python run_all.py --with-docking       # adds the AutoDock Vina arm (needs tools/vina.exe)
    python run_all.py --offline            # no network at all; docking reuses data/raw/docking
    python run_all.py --quick              # smoke test: davis/random, 2 models, 1 seed
    python run_all.py --stage report       # re-render tables + figures from results/*.csv
    python run_all.py --stage notebook     # regenerate and EXECUTE notebook/*.ipynb
    python run_all.py --stage bootstrap    # intervals for all 150 fits (no refitting, CPU only)
    python run_all.py --stage external     # external validation on a ChEMBL kinase Ki table
                                           # (network fetch + its own features/splits/grid)

Stages run in dependency order and each one is a separate script, so a failure tells you
exactly which step broke. Wall time on an 8-core laptop: ~4 h for the full grid is an
estimate, not a measurement -- the measured stages are 03 (69.8 min summed fit time, from
the `seconds` column of results/metrics.csv) and the stage-12 test-set bootstrap (82 min,
4,936 s at 6 cells in parallel, 2000 RMSE/MAE reps + 600 Harrell-CI reps; ~7 min with
--quick). The 600 ci-reps is a floor, not a preference -- Holm over the 10-contrast model
family cannot reject below B=398, and the 105-pair arm-level family is not resolvable at
all within this budget (the script stamps those blocks `holm_underpowered` instead of
printing a verdict). Use --bootstrap-ci-reps 0 to skip the CI part entirely. Stage 12 reads
results/predictions/*.npz, which is gitignored, so a fresh clone needs stage `models` first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("fetch",     "00_fetch_data.py",     "download + hash-pin the raw activity tables"),
    ("features",  "01_features.py",       "ECFP4 fingerprints, protein composition, scaffolds"),
    ("splits",    "02_splits.py",         "5 protocols + leakage audit"),
    ("models",    "03_models.py",         "null baselines + 4 learned models x 3 seeds"),
    ("intervals", "04_uncertainty.py",    "split-conformal intervals, measured coverage"),
    ("docking",   "05_docking.py",        "AutoDock Vina arm (panel + decoy control)"),
    ("dockstats", "06_docking_analysis.py", "Vina vs labels vs the 03 models on identical rows"),
    ("variance",  "11_variance_decomposition.py", "protocol vs model variance shares (ANOVA)"),
    ("bootstrap", "12_testset_bootstrap.py", "test-set bootstrap intervals over all 150 fits"),
    ("paired",    "10_paired_headtohead.py", "paired bootstraps for the Vina head-to-head"),
    ("report",    "07_report.py",         "figures + results/report.md + results/summary.json"),
    ("notebook",  "08_notebook.py",       "writes and executes notebook/dti_affinity_bench.ipynb"),
    ("sensitivity", "09_preproc_sensitivity.py", "three standardisation guards x models x protocols"),
    ("extfetch",   "13_fetch_chembl.py",   "download + hash-pin the ChEMBL kinase Ki panel (network)"),
    ("external",   "14_external_chembl.py", "external validation: features/splits/audit/grid/analyse"),
]
OPTIONAL_STAGE = "docking"
OPTIONAL_STAGES = (OPTIONAL_STAGE, "dockstats", "sensitivity")
# the external-validation pair is reachable as one stage name: `--stage external` runs the
# fetcher and then the pipeline, because the pipeline needs the fetched tree.
EXTERNAL_PAIR = ("extfetch", "external")


def run(script: str, extra: list[str]) -> None:
    print(f"\n### python scripts/{script} {' '.join(extra)}".rstrip())
    r = subprocess.run([sys.executable, "-u", f"scripts/{script}", *extra], cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"{script} failed with exit {r.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[s[0] for s in STAGES] + ["all"], default="all")
    ap.add_argument("--fetch", action="store_true", help="include the fetch stage")
    ap.add_argument("--with-docking", action="store_true",
                    help=f"include the {OPTIONAL_STAGE} stage (needs tools/vina.exe, see docs/data_sources.md)")
    ap.add_argument("--with-sensitivity", action="store_true",
                    help="include 09_preproc_sensitivity.py (re-fits under three standardisation guards)")
    ap.add_argument("--offline", action="store_true", help="pass --offline to the network scripts")
    ap.add_argument("--quick", action="store_true", help="small grid, for a smoke test")
    ap.add_argument("--resume-grid", action="store_true",
                    help="let 03 keep existing results/metrics.csv rows instead of recomputing "
                         "the grid (fast, but only correct if the code has not changed)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel vina jobs")
    ap.add_argument("--bootstrap-jobs", type=int, default=6,
                    help="parallel cells for 12_testset_bootstrap.py (one process each)")
    ap.add_argument("--bootstrap-reps", type=int, default=2000,
                    help="RMSE/MAE bootstrap reps in stage `bootstrap`")
    ap.add_argument("--bootstrap-ci-reps", type=int, default=600,
                    help="Harrell-CI bootstrap reps in stage `bootstrap` (the expensive part; "
                         "must exceed 398 for Holm over the 10-contrast model family to be "
                         "resolvable, see scripts/12_testset_bootstrap.py)")
    ap.add_argument("--external-budget-mb", type=float, default=175.0,
                    help="download budget for the ChEMBL panel in stage `extfetch` "
                         "(the run behind this repo's external validation transferred 77.5 MB)")
    ap.add_argument("--external-reps", type=int, default=2000,
                    help="bootstrap reps for the external table's analyse stage")
    ap.add_argument("--external-ci-reps", type=int, default=600,
                    help="Harrell-CI bootstrap reps for the external table's analyse stage")
    args = ap.parse_args()

    want = args.stage
    if want == "external":
        order = [s for s in STAGES if s[0] in EXTERNAL_PAIR]
    else:
        order = list(STAGES) if want == "all" else [s for s in STAGES if s[0] == want]
    if want == "all" and not args.fetch:
        order = [s for s in order if s[0] != "fetch"]
    # the external table needs the network and its own ~15 min of compute, so it is never
    # part of the default run: ask for it by name (`--stage external`).
    if want == "all":
        order = [s for s in order if s[0] not in EXTERNAL_PAIR]
    if want == "all" and not args.with_docking:
        order = [s for s in order if s[0] not in (OPTIONAL_STAGE, "dockstats", "paired")]
    if want == "all" and not args.with_sensitivity:
        order = [s for s in order if s[0] != "sensitivity"]
    # the sensitivity ablation is an audit of the preprocessing decision, not part of the
    # numbers the README tables are built from -- it re-fits models under three guards.
    if want == "all" and not args.with_sensitivity:
        order = [s for s in order if s[0] != "sensitivity"]

    for name, script, _why in order:
        extra: list[str] = []
        if name == "fetch":
            if args.offline:
                extra.append("--offline")
        elif name == "models":
            # `--resume` keeps rows already in results/metrics.csv. That is a real hazard: the
            # grid once ended up holding cells from two different code versions, which 04's
            # consistency assert then reported as DIFF. A full run therefore recomputes
            # everything unless the caller asks to resume.
            extra += ["--resume"] * args.resume_grid + (["--quick"] if args.quick else [])
        elif name == "intervals":
            if args.quick:
                extra += ["--datasets", "davis", "--protocols", "random"]
        elif name == "docking":
            if not (ROOT / "tools" / "vina.exe").exists():
                raise SystemExit("tools/vina.exe missing -- see docs/data_sources.md "
                                 "for the version + sha256 to install, or drop --with-docking")
            extra += (["--offline"] if args.offline else [])
            extra += (["--jobs", str(args.jobs)] if not args.quick else
                      ["--max-targets", "1", "--try-limit", "8", "--ligands", "6",
                       "--decoy", "2", "--jobs", "2", "--resume"])
        elif name == "notebook" and args.quick:
            extra.append("--timeout")
            extra.append("300")
        elif name == "bootstrap":
            # the Harrell-CI reps are the expensive part (ci_harrell is a per-row Python loop),
            # so they are a separate knob from the cheap RMSE/MAE reps
            extra += (["--quick"] if args.quick else
                      ["--reps", str(args.bootstrap_reps),
                       "--ci-reps", str(args.bootstrap_ci_reps),
                       "--jobs", str(args.bootstrap_jobs)])
        elif name == "extfetch":
            extra += ["--budget-mb", str(args.external_budget_mb)]
            if args.quick:
                # the smoke fetch and the smoke grid both run in a tagged tree, so a quick run
                # can never overwrite the committed table in data/processed/external/
                extra += ["--max-targets", "4", "--rows-per-target", "400",
                          "--out-tag", "_quick"]
        elif name == "external":
            extra += ["--quick"] if args.quick else [
                "--reps", str(args.external_reps),
                "--ci-reps", str(args.external_ci_reps),
                "--jobs", "5"]
        run(script, extra)
    print("\nall requested stages finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
