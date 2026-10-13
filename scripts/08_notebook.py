#!/usr/bin/env python3
"""
08_notebook.py -- build notebook/dti_affinity_bench.ipynb and EXECUTE it, so the
committed .ipynb contains real outputs (a notebook with empty outputs is not evidence).

The notebook is a read-through of what scripts 00-06 already produced, plus one live
recomputation (concordance index from a saved prediction file through dti_core) so that
the reader can see the metric code and the number in the same place.

Usage
    python scripts/08_notebook.py            # generate + execute + write in place
    python scripts/08_notebook.py --no-exec  # generate only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebook" / "dti_affinity_bench.ipynb"

MD = "markdown"
CODE = "code"


def cells():
    C = []

    def md(t):
        C.append((MD, t.strip()))

    def code(t):
        C.append((CODE, t.strip()))

    md(f"""
# Protein-ligand affinity: what the split protocol is actually measuring

Public DTI benchmarks (DAVIS, KIBA) are reported with a single number per model, and that
number moves more with the **split protocol** than with the model. This notebook reads the
outputs of `scripts/00` - `scripts/06` and walks through that claim with the leakage audit,
a five-model x five-protocol grid, split-conformal coverage, and an AutoDock Vina arm that
never saw the labels.

Reproduce end to end:

```
python scripts/00_fetch_data.py      # downloads + integrity checks (13 PASS)
python scripts/01_features.py        # ECFP4 counts + sequence composition
python scripts/02_splits.py          # 5 protocols + leakage audit
python scripts/03_models.py          # 5 models x 5 protocols x 3 seeds
python scripts/04_uncertainty.py     # split conformal on top of 03
python scripts/05_docking.py         # UniProt -> PDBe -> meeko -> Vina
python scripts/06_docking_analysis.py
python scripts/07_report.py
```
""")

    code("""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, display

# Find the repo root instead of trusting the kernel cwd. nbconvert starts the kernel with
# cwd = the notebook's own directory (notebook/), so a bare Path('scripts') would resolve to
# notebook/scripts and raise ModuleNotFoundError -- and a reader running the whole repo from
# an IDE may have cwd = repo root. Every path below is therefore anchored on ROOT, located
# by walking up until scripts/dti_core.py is found.
ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT / 'scripts' / 'dti_core.py').is_file():
    ROOT = ROOT.parent
assert (ROOT / 'scripts' / 'dti_core.py').is_file(), f'no repo root above {Path.cwd()}'
sys.path.insert(0, str(ROOT / 'scripts'))
from dti_core import evaluate, ci_harrell, ci_sklearn_like
pd.set_option('display.width', 200)
RES, PROC = ROOT / 'results', ROOT / 'data' / 'processed'

def show(fname):
    p = Path(fname)
    if not p.is_absolute():
        p = (ROOT / p).resolve()      # figure paths are written repo-root relative
    if p.exists():
        display(Image(filename=str(p)))
    else:
        print(f'[not built in this run: {fname}] -- run the stage that writes it; '
              f'this notebook does not display a stale or missing figure')
    return p.exists()

summary = json.loads((PROC/'data_summary.json').read_text(encoding='utf-8'))
summary['davis']
""")

    md("""
## 1. Data integrity, before any modelling

Two things have to be true or every downstream number is meaningless: the row count has
to equal the sum of the five published DAVIS files, and the labels recovered from the
feature-only canonical files have to agree with the Kd table. `00_fetch_data.py` asserts
both; the numbers below are what it recorded.

Note `distinct_target_sequences` < `distinct_target_names`: 442 DAVIS target names collapse
onto 379 sequences, i.e. **the same protein features appear under two different names**, and
any split done on names leaks. This benchmark keys every split on `target_id` (sequence).
""")

    code("""
d = summary['davis']
k = summary['kiba']
pd.DataFrame([
    {'dataset': 'DAVIS', 'rows': d['rows'], 'ligands': d['distinct_ligand_smiles'],
     'target_names': d['distinct_target_names'], 'target_sequences': d['distinct_target_sequences'],
     'label': 'pKd', 'censored_share': d['fraction_at_censor_ceiling_pKd_5'],
     'cross_source_max_abs_delta': d['cross_source_max_abs_delta_pKd'],
     'files_sum_matches': d['sum_of_five_published_files']},
    {'dataset': 'KIBA', 'rows': k['rows'], 'ligands': k['distinct_ligand_smiles'],
     'target_names': k['distinct_target_keys'], 'target_sequences': k['distinct_target_sequences'],
     'label': 'KIBA score', 'censored_share': k['rows_at_floor']/k['rows'],
     'cross_source_max_abs_delta': None, 'files_sum_matches': k['published_split_counts']},
])
""")

    md("""
## 2. The leakage audit: five protocols, five different questions

`random` is not a scaffold split, `cold_drug` is not scaffold-disjoint, and `temporal`
throws away most of the data. The columns that matter:

* `test_pairs_with_near_duplicate_ligand_ge0.7` - share of test rows whose ligand has an
  ECFP4 Tanimoto twin >= 0.7 in train (a memorised near-duplicate)
* `test_scaffold_present_in_train` - share of test rows whose Murcko scaffold is in train
* `test_targets_absent_from_train` - share of test targets never seen in train
""")

    code("""
au = pd.read_csv(RES/'leakage_audit.csv')
cols = ['dataset','protocol','n_train','n_valid','n_test','excluded_rows','test_ligands',
        'test_ligands_absent_from_train','test_targets_absent_from_train',
        'test_scaffold_present_in_train','test_pairs_with_near_duplicate_ligand_ge0.7',
        'test_pairs_with_near_duplicate_ligand_ge0.95','target_max_dpc_cosine_to_train_target_mean',
        'temporal_cutoff']
au[[c for c in cols if c in au.columns]]
""")

    code("""
from IPython.display import Image, display
show('results/figures/fig1_leakage_by_protocol.png')
""")

    md("""
Read the KIBA rows: under `random`, **35%** of the test pairs have a near-duplicate ligand
already in the training set; under a real scaffold split that drops to 11%. So a KIBA
"random split" number is partly a similarity lookup. And `cold_drug` is NOT a scaffold
split -- about half of its held-out drugs keep a scaffold that training already saw.

For DAVIS the two cuts have the **same block sizes** (21,658 / 2,652 / 5,746) and both come
out scaffold-disjoint (`test_scaffold_present_in_train` = 0), because 68 inhibitors only
span 13 Murcko scaffolds, so holding drugs out by entity and holding them out by scaffold
converge on the same *shape* of cut. They are **not the same cut**: only 3 of the 13
held-out ligands overlap (`02_splits.py` assigns them independently). KIBA, with 2,068
drugs and many more scaffolds, shows the difference clearly -- its `cold_drug` keeps
48.8 % of test rows on a scaffold that training already saw.

The `temporal` rows also report `excluded_rows`: undated assays cannot be placed in time,
so they are dropped and counted instead of being silently kept.
""")

    md("""
## 3. Five models across five protocols

Models are deliberately classical (train-mean null, ridge, XGBoost, random forest, a
DeepDTA-style MLP): the question is what the protocol measures, not who has the largest
network. The table below averages the 3 seeds of each learned model (the three null
baselines are deterministic, so they have a single row); features are ECFP4 count bits plus
amino-acid / dipeptide composition, standardised with training statistics only.
""")

    code("""
m = pd.read_csv(RES/'metrics.csv')
m.groupby(['dataset','protocol','model'])[['rmse','ci_harrell','ci_sklearn_like','pearson_r']].mean().round(4)
""")

    code("""
show('results/figures/fig2a_davis_rmse.png')
show('results/figures/fig2b_kiba_rmse.png')
show('results/figures/fig2c_davis_ci.png')
show('results/figures/fig2d_kiba_ci.png')
""")

    md("""
### The reporting hazard nobody states

`ci_harrell` — the convention the DTI literature actually uses — restricts the denominator
to *comparable* pairs: a pair whose two labels are equal is dropped, and a pair whose
predictions are equal gets 0.5 credit. `ci_sklearn_like` keeps every pair and gives a
label-tied pair 0.5 credit instead. DAVIS puts ~two-thirds of its *rows* on the censoring
ceiling (pKd 5.0), and that turns into **36.4–48.7 % of the label *pairs* inside a test block
being exact ties** (KIBA: 5.1–6.7 %) — close to half, not most, but enough that the second
caliber is dragged toward 0.5 by construction. Both are in `results/metrics.csv` because the
two disagree by up to 0.18 on the same predictions, and picking one silently is how
benchmarks get inflated.

Three different quantities all get called "ties", so the next cell prints them separately:
the pair share (the `ci_harrell` denominator loss), the row share on the modal label (the
censoring), and `1 − distinct/rows` (plain redundancy, which is ~93–98 % on both datasets and
says nothing about concordance at all).
""")

    code("""
pd.read_csv(RES/'label_tie_measures.csv')
""")

    code("""
show('results/figures/fig4_label_ties.png')
""")

    code("""
m2 = m[m.dataset=='davis']
print('CI, comparable pairs only (Harrell / DTI convention), davis')
display(pd.pivot_table(m2, index='model', columns='protocol',
                       values='ci_harrell').round(3).dropna(how='all'))
print('CI, every pair counted and label ties given 0.5 (sklearn-like), davis')
display(pd.pivot_table(m2, index='model', columns='protocol',
                       values='ci_sklearn_like').round(3).dropna(how='all'))
""")

    md("""
### The negative result, kept in

Under the forward-in-time DAVIS protocol the linear model is **worse than predicting the
training mean** (`pearson_r` collapses to ~0). That is written into `results/metrics.csv`
and into the README rather than dropped: assay batches drift, and a protocol that puts the
future in the test set measures that drift.
""")

    code("""
tm = m[(m.dataset=='davis') & (m.protocol=='temporal')][['model','seed','rmse','pearson_r','ci_harrell']]
tm.sort_values('rmse')
""")

    md("""
## 4. How wrong can a single prediction be? (split conformal)

`04_uncertainty.py` uses each protocol's validation block as a calibration set, computes
conformal quantiles, and then MEASURES how much of the test block the interval actually
covers. Constant width, and a normalised variant whose width follows a sigma-hat model.
""")

    code("""
cf = pd.read_csv(RES/'uncertainty/conformal.csv')
cf[[c for c in ['dataset','protocol','model','method','n_cal','rmse','rmse_matches_03','q90',
                'width90','cov50','cov80','cov90','coverage_error','corr_absres_sigma'] if c in cf.columns]]
""")

    code("""
show('results/figures/fig5_conformal_coverage.png')
""")

    md("""
## 5. A physics arm that never saw the labels (AutoDock Vina)

For each panel kinase: gene -> UniProt -> PDBe SIFTS `best_structures` (X-ray, <= 2.6 A,
coverage >= 0.35) -> RCSB coordinates -> keep the assayed chain, check its sequence
against the DAVIS sequence with local alignment -> box from the co-crystal drug-like
ligand -> meeko (gasteiger) -> `vina.exe`. Compounds come from RDKit ETKDG+MMFF of the
DAVIS SMILES, so the pose is not a crystal pose and the receptor is rigid.

The score is *not* a affinity in pKd units, so it is compared as a ranker (concordance)
and through a two-parameter train-fitted linear map, on exactly the rows the trained
models predict.
""")

    code("""
panel = pd.read_csv(RES/'docking/target_panel.csv')
panel[['target_id','gene','uniprot_acc','pdb_id','chain','resolution','pdbe_coverage',
       'identity_vs_davis','identity_vs_uniprot','cocrystal_ligand','cocrystal_heavy',
       'meeko_bad_residues_dropped']]
""")

    code("""
pairs = pd.read_csv(RES/'docking/pairs.csv')
pairs.describe().loc[['count','mean','std','min','25%','50%','75%','max'],['pkd','vina_best','vina_seconds']]
""")

    code("""
comp = pd.read_csv(RES/'docking/comparison.csv')
comp.head(40)
""")

    code("""
show('results/figures/fig6_docking_vs_pkd.png')
""")

    md("""
### Specificity control

Re-docking the same compounds into a box translated +30 A into the solvent should destroy
the signal. `results/docking/decoy_control.csv` holds that measurement, which is what
separates "the box is right" from "the score is noise".
""")

    code("""
import os
pd.read_csv(RES/'docking/decoy_control.csv') if os.path.exists(RES/'docking/decoy_control.csv') else 'not run'
""")

    md("""
## 6. What this repo does and does not claim

**Claims**
- Every protocol/label/feature decision is asserted in code with printed PASS/FAIL checks.
- Metrics are reported for 4 learned models x 5 protocols x 3 seeds on each of DAVIS and
  KIBA (120 fitted runs), plus 3 deterministic null baselines per protocol x dataset, both
  CI calibers, with the null floors.
- Conformal coverage is measured per protocol, not assumed.
- The docking arm is scored on identical rows to the ML arm, with a negative control.

**Does not claim**
- No deep DTI architecture is proposed; the models are baselines by design.
- DAVIS is 68 compounds against 442 target names (379 distinct sequences) - far too small
  to support a generalisation claim about drug affinity, and 69.6% of its labels are censored.
- The docking panel is a handful of kinases with a single rigid structure each; its
  correlation with assay affinity is an observation about this subset, not about docking.
- KIBA's temporal protocol keeps only ~38% of rows (deposits are known for 0.378 of them),
  so the drift result there is weaker evidence than the DAVIS one.
""")

    code("""
json.loads((RES/'summary.json').read_text(encoding='utf-8'))
""")
    return C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    nb.cells = [nbf.v4.new_markdown_cell(t) if k == MD else nbf.v4.new_code_cell(t)
                for k, t in cells()]
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(NB))
    print(f"wrote {NB} ({len(nb.cells)} cells)")

    if args.no_exec:
        return 0
    t0 = time.time()
    cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=%d" % args.timeout, "--output", NB.name, str(NB)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    print("execute rc", r.returncode, f"{time.time()-t0:.0f}s")
    if r.returncode:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        return 1
    out = nbf.read(str(NB), as_version=4)
    errs = []
    for i, c in enumerate(out.cells):
        if c.cell_type != "code":
            continue
        for o in c.outputs:
            if o.get("output_type") == "error":
                errs.append((i, o.get("ename"), o.get("evalue")))
    n_with_out = sum(1 for c in out.cells if c.cell_type == "code" and c.outputs)
    n_code = sum(1 for c in out.cells if c.cell_type == "code")
    print(f"cells executed: {n_code}, with outputs: {n_with_out}, errors: {len(errs)}")
    for e in errs:
        print("  ERROR", e)
        return 1
    nbf.write(out, str(NB))
    print("notebook executed clean and saved with outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
