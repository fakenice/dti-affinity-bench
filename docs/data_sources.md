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

## 1c. Second external validation table (BindingDB kinase Ki)

Retrieved on **2026-11-07 (UTC)** by `scripts/15_fetch_bindingdb.py`, from the BindingDB REST
endpoint `https://www.bindingdb.org/rest/getLigandsByUniprots` — which is **open and needs no API
key** (an earlier note in `13_fetch_chembl.py` claimed BindingDB required an emailed key; that was
checked against the live service on 2026-11-07 and corrected) — plus the PubMed E-utilities
`esummary` service (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi`) for publication
years. Raw probe evidence (homepage, API docs, a single-target JSON and its response headers) is
committed under `results/bindingdb_probe/`.

| item | value |
|---|---|
| panel | 16 human kinase targets, requested by UniProt accession, cutoff 100,000 nM, ≥ 50 usable Ki rows each |
| pairs / ligands / targets / documents | 18,478 / 15,012 / 16 / 298 |
| rows | 183,390 fetched → 24,110 kept (`Ki` only, censored bounds `</>/~` dropped, non-numeric excluded) |
| row arithmetic | `183,390 − Σrow_filters_applied (159,280) = 24,110` exactly; the per-target `exact_ki_kept` sums to 24,117 because those 7 rows were still counted before the SMILES-parse step (`unparseable_smiles: 7`), i.e. the two totals differ by one documented filter, not by lost data |
| label | pKi = 9 − log₁₀(median Ki / nM), median over a pair's exact Ki rows (n_rows / ki_iqr_nM carried for audit) |
| ligand identity | standard InChIKey (RDKit) — two ChEMBL/monomer ids sharing a structure are one ligand, so a duplicate cannot straddle train/test |
| target attribution | rows kept only when the response's `query` protein name equals the queried target (a single-UniProt response also carries off-target selectivity rows) |
| pKi range | 4.00 – 12.24 (mean 8.173, sd 1.386) |
| transferred | 41,948,907 bytes over 16 HTTP requests, 0 retries (budget 175 MB, authorised cap 200 MB recorded in `scripts/15_fetch_bindingdb.py`, `budget_exceeded: false`). The 16 requests are **14 target pages + 2 PubMed pages**, not "16 requests because 16 targets" — the other two target pages were served from cache (see "wire vs disk" below) |
| raw page cache | `data/raw/bindingdb/` (git-ignored); 19 cached files = 16 target pages + 3 esummary pages, each sha256 in `data/processed/external_bindingdb/external_manifest.json → raw_cache_sha256` — all 19 recomputed on 2026-11-07 and every one matches. Re-check (only in the workspace that fetched them — the pages themselves are git-ignored, so a fresh clone gets the hashes but not the bytes): `python -c "import json,hashlib;M=json.load(open('data/processed/external_bindingdb/external_manifest.json'));print(all(hashlib.sha256(open(e.split(': ')[0],'rb').read()).hexdigest()==e.split(': ')[1] for e in M['raw_cache_sha256']))"` |
| wire vs disk | the ledger counts **41,948,907 B downloaded on this run** while the cache holds **55,254,947 B**; the 13,306,040 B difference decomposes exactly: `uniprot_O60674` (7,282,973) + `uniprot_P23458` (5,920,535) were **cache re-reads** (`transfer.cached_responses_reused = 2`), and `esummary_e50cb9fa…` (102,532) is a **stale page from an earlier partial fetch** — it is not one of the two batches this run consumed, which are reproducible from the committed 298-pmid list at `--batch 200` (`esummary_191751fa…` = 200 ids, `esummary_800191f6…` = 98 ids; the third key does not appear at any batch size). `bytes_downloaded` therefore measures this run's network traffic and `raw_cache_bytes_on_disk` measures the whole pinned directory — different quantities, deliberately both recorded, and the manifest hashes every file in the directory rather than only the ones this run needed |

The output tables and their sha256 are in
`data/processed/external_bindingdb/external_manifest.json → files` (`bindingdb_ki.csv`,
`ligands.csv`, `targets.csv`, `documents.csv`); derived features and the five splits carry their
own manifests under the same tree.

Two limits specific to this table, recorded in the manifest rather than hidden: BindingDB's REST
endpoint exposes **no release version**, so this snapshot is pinned by fetch time + the 100,000 nM
cutoff + per-page sha256; and its rows are mostly DOI-only with an empty `pmid`, so only
**3,572 of 18,478 pairs (19.3 %)** resolve to a PubMed year — the `temporal` protocol runs on that
dated subset (n_test 770), the same exclude-undated-and-count-them treatment ChEMBL uses (README
§9b). The evaluation itself re-runs script 14's five stages through `scripts/15_external_bindingdb.py`
(in-process, `jobs=1` because a spawned bootstrap child cannot inherit the path redirect);
script 14 and the committed ChEMBL tree are byte-unchanged. That invariance is a command rather
than an assertion: `git status --porcelain data/processed/external results/external_chembl scripts/14_external_chembl.py`
prints nothing when run after the BindingDB grid, so no BindingDB stage touched either tree.

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
