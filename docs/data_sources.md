# Data sources, versions and how to re-fetch everything

Everything below was retrieved on **2026-10-13 (UTC)** from this machine. `data/raw/` is
git-ignored (92 MB of third-party downloads) but every file in it is hash-pinned here and
in `data/raw/manifest.json`, so a fresh clone can verify it is looking at the same bytes.

## 1. Bioactivity tables (the labels)

| Dataset | Rows used here | Source of record | What we actually downloaded |
|---|---|---|---|
| DAVIS | 30,056 | Davis MI *et al.* **Proteins** 2011;69(2):319-324 — 68 inhibitors × 442 kinases, $K_d$ | 5 canonical CSVs from the `zihandun/GraphDTA-ColdStart` mirror |
| KIBA | 118,254 | Liu T *et al.* **J Chem Inf Model** 2015;55(9):1792-1801 (KIBA score), aggregated by Menden GD *et al.* **Nat Commun** 2019;10:3050 (DepoKIBA) | `kiba_temporal_splits.parquet` from the `swmeyer1979/dti-contamination` mirror |

Why mirrors rather than the original web pages: the DAVIS host and the original KIBA
table are not reliably fetchable, and both mirrors above are *canonicalised* (SMILES
already normalised, splits already published), which removes a whole class of silent
parsing differences. The cost is that we inherit their curation — hence the cross-checks
in `scripts/00_fetch_data.py`: the five published DAVIS files sum to exactly 30,056 rows
(`sum_of_five_published_files`), and the independently-distributed DAVIS tables agree with
each other to <0.02 pKd on **all 30,056 rows** (`cross_source_disagreeing_rows: 0`).

Retrieval channel: `https://api.github.com/repos/{repo}/git/blobs/…` with
`Accept: application/vnd.github.raw`. (`github.com:443` and `raw.githubusercontent.com`
were unreachable from this network — only the API host resolved.)

### Hash-pinned raw artifacts

| file | repo path | bytes | sha256 |
|---|---|---|---|
| `davis_train_canonical.csv` | `zihandun/GraphDTA-ColdStart:data/oursplit/` | 17,009,061 | `bf376e50d68568b8379a943b4b0b2b326d3fad56803f38ab00e8105624c37cdb` |
| `davis_val_canonical.csv` | same | 4,417,222 | `1afb863dde386690e1c9eadd2a86417f7b6e42f561f75ab76f24981c308aa5cf` |
| `davis_test_canonical.csv` | same | 478,891 | `9bcfcce2937e899618edbc7f46ef58f3de1011daeb193d7281832eb6f0e6041c` |
| `davis_test1_canonical.csv` (cold-drug) | same | 2,078,897 | `4ed0b509a5f8cfcb93cc00e6a8d99fc4a30dc426fae86ed56e867064972f80a8` |
| `davis_test2_canonical.csv` (cold-target) | same | 1,952,425 | `b520db3da356230d86b438858426d2bf6a1096169edf73d944d296d67d2dc87b` |
| `kiba_temporal_splits.parquet` | `swmeyer1979/dti-contamination:data/splits/` | 1,157,228 | `0f4004e9e84efb9762186ddd698d9c3efb00cb132ee4572413ee4d6c349effed` |
| `davis_temporal_splits.parquet` | `swmeyer1979/dti-contamination:data/splits/` | 477,417 | `c5e705cf78b2907f63dd35d38037912c360a3746f761a7ad5ad4ec6257c6b34b` |

The last two parquet tables supply the **deposit dates** used by the forward-looking
`temporal` split (DAVIS 25,194/30,056 rows dated; KIBA 44,702/118,254 = 37.8 %).

## 1b. External validation table (ChEMBL kinase Ki)

Retrieved on **2026-11-04 (UTC)** by `scripts/13_fetch_chembl.py`, from the ChEMBL REST API
(`https://www.ebi.ac.uk/chembl/api/data`) plus the UniProt REST service for the protein
sequences. ChEMBL_37, release date 2026-05-01, with the version read live at run time.

| item | value |
|---|---|
| panel | 16 human SINGLE PROTEIN kinase targets, ≥ 300 exact-standard Ki rows each |
| pairs / ligands / targets / documents | 19,088 / 14,713 / 16 / 398 |
| label | pKi = 9 − log₁₀(median Ki / nM), median over a pair's curated rows |
| pKi range | 4.0 – 11.0 (mean 7.786, sd 1.476) |
| transferred | 59.2 MB + 18.3 MB + 0.0 MB over three runs = **77.5 MB** (cap: 200 MB) |
| raw page cache | `data/raw/chembl/` (104,814,223 bytes, git-ignored), 1,123 pages, each sha256 listed in `data/processed/external/external_manifest.json → raw_cache_sha256` |

The output tables and their sha256 are in
`data/processed/external/external_manifest.json → files` (`chembl_ki.csv`, `ligands.csv`,
`targets.csv`, `documents.csv`); the derived features and the five split files carry their own
manifests in `data/processed/external/features/` and `data/processed/external/splits/`.

Two limits the manifest records instead of hiding: the download cap forced the fetcher to
sample activity pages, so three of the sixteen targets are `complete: false`; and the public
API exposes no per-activity deposition date, so this table's `temporal` protocol uses the
**publication year** of the source document instead of a deposition date (README §9).

## 2. Structural arm (AutoDock Vina)

Four REST endpoints, all cached under `data/raw/docking/` (66 MB, git-ignored) and reusable
with `--offline`:

| Purpose | Endpoint |
|---|---|
| DAVIS target name → reviewed human UniProt entry | `https://rest.uniprot.org/uniprotkb/search`, `…/uniprotkb/{acc}.json` |
| UniProt → deposited structures (PDBe SIFTS) | `https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{acc}` |
| coordinates | `https://files.rcsb.org/download/{PDB_ID}.pdb` |
| co-crystal ligand SDF / chemcomp details | `https://www.ebi.ac.uk/pdbe/api/pdb/entry/ligand_monomers/{pdb_id}`, `https://data.rcsb.org/rest/v1/core/chemcomp/{cid}` |

Note: `data.rcsb.org/rest/v1/core/chemcomp/{cid}` is used because both
`www.ebi.ac.uk/pdbe/api/pdb/compound/detail/{cid}` and `files.rcsb.org/ligands/{cid}/…`
return 404 from here. A 404 from `best_structures` is PDBe's way of reporting "no PDB entry
is mapped to this accession" — `scripts/05_docking.py` records it as a rejected candidate
rather than swallowing it.

### Vina binary — not vendored

`tools/vina.exe` is git-ignored; you must place it there yourself.

| | |
|---|---|
| Program | AutoDock Vina **1.2.7** (Windows exe), Apache-2.0 |
| Obtained from | the official GitHub release asset for 1.2.7 |
| Size | 1,233,920 bytes |
| sha256 | `e0c4b2715e0c1a74f6e92d0f3be0328ac97542eafbc111e6b1efad897a73cce5` |
| Verified by | `scripts/05_docking.py` writes this hash into `results/docking/manifest.json` on every run |

Prepare scripts come from `meeko` 0.8.0 (`mk_prepare_receptor`, `mk_prepare_ligand`),
installed by `requirements.txt` into the Python `Scripts` directory and located with
`shutil.which`.

## 3. Software versions

`scripts/05_docking.py` dumps the exact runtime into `results/docking/versions.json`;
the run that produced the committed results used:

```
python 3.13.7   rdkit 2026.03.3   numpy 2.3.4   pandas 3.0.3   scikit-learn 1.9.1
xgboost 3.4.1   torch 2.6.0+cu124 gemmi 0.7.5   meeko 0.8.0    matplotlib 3.11.0
```

Feature arrays carry their own provenance (`data/processed/features/features_manifest.json`:
RDKit version, fingerprint definition, shape, non-zero density and sha256 per array).

## 4. Re-fetching from a clean clone

```bash
python scripts/00_fetch_data.py          # needs api.github.com; writes data/raw/ + manifest
python scripts/01_features.py            # deterministic, offline
python scripts/02_splits.py              # deterministic, offline
```

`data/processed/` (the 8 MB of tidy tables and the 16.7 MB of split assignments) is
**committed**, so you can skip straight to `03_models.py` and reproduce the headline
numbers without any network access. The Vina arm additionally needs `tools/vina.exe`.

## 5. Licence notes

DAVIS data are CC-BY (Davis *et al.* 2011); KIBA derives from ChEMBL/BindingDB/Predicting
IC50 aggregated in Menden *et al.* 2019 — check those terms before redistribution. PDB
entries are accessed under the RCSB/PDBe open data terms; each deposited structure used in
the panel is recorded with its PDB ID in `results/docking/target_panel.csv` so the specific
experimental entries can be cited. AutoDock Vina is Apache-2.0, Meeko LGPL-3.0, RDKit
BSD-3-Clause.
