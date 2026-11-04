# Protein–ligand affinity benchmark: the split protocol moves the number more than the model does

DAVIS (30,056 kinase–inhibitor $K_d$ rows) and KIBA (118,254 rows) scored with
RDKit ECFP4 fingerprints + amino-acid/dipeptide composition, evaluated under **five
different split protocols** with RMSE and both concordance calibres, plus an
AutoDock Vina arm on the same compounds — and then re-checked on a **third,
independently curated table** (a ChEMBL kinase Ki panel, §9) to make sure the
headline is not a property of the two shipped benchmarks.

Every table and figure here is produced by the code in this repository from committed
inputs — `results/report.md`, `results/summary.json` and
`notebook/dti_affinity_bench.ipynb` (executed, with outputs) are the same numbers you see below.

## Headline

Every number below is a key in `results/summary.json` or `results/readme_numbers.json`, so the
claim and its evidence are one `jq` apart. The one exception is the external-validation block
(§9), which lives in its own tree — `results/external_chembl/paired_bootstrap.json` — because
it must not be able to touch the shipped grid.

1. **The split protocol moves the result more than the model does — for the models that
   work.** With identical features, code and seeds, the best achievable DAVIS test RMSE runs
   from **0.5697** (`random`) to **1.0520** (`cold_target`) — a range of **0.4823 pKd**,
   more than half a label standard deviation — while inside one protocol the three trained
   models differ by at most **0.1864** (KIBA: range 0.2101, spread ≤ 0.2554). A two-way
   variance decomposition puts **91.3 %** of DAVIS RMSE variance on protocol and **6.8 %** on
   model (KIBA 61.5 % / 30.7 %); adding the collapsed MLP raises the model share to
   45.7 % / 47.3 %, so the "model matters little" reading is a statement about the trained
   tabular models, not about architecture in general.
   `readme_numbers.json → protocol_spread_of_the_best_model`,
   `model_spread_within_protocol_trained3`, `variance_decomposition`.
   The whole grid is now bootstrapped (script 12, 2000 reps over the stored test predictions, no
   refitting): the protocol contrast survives on every model × dataset cell — hardest-minus-easiest
   protocol RMSE spread runs **0.2171–1.4746** and **100 %** of those intervals exclude zero under
   row resampling — while 55 of the 60 within-protocol learned-vs-learned gaps are also significant
   under target-cluster resampling, because n ≈ 6–24 k test rows resolves differences as small as
   0.005 pKd. Significance is not the same as magnitude: the gaps that clear Holm are routinely
   an order of magnitude below the protocol effect they are being compared to.
2. **A two-line lookup beats a trained model, in the cell nobody checks.** On DAVIS `scaffold`
   the *per-target train mean* scores **0.7783** and beats every learned model in the grid,
   XGBoost by 0.032; Ridge is worse than the global-mean null on DAVIS `scaffold` (0.8786 vs
   0.8522), DAVIS `temporal` (1.0442 vs 1.0030) and KIBA `temporal` (0.9798 vs 0.8149) — but
   the paired bootstrap resolves only the last two (Holm p = 0.010 each); the `scaffold` gap
   of −0.0264 does not clear Holm (p = 0.79), so it is a point estimate, not a shown loss.
   `learned_models_worse_than_the_global_mean` lists **10 of 40** learned cells that lose to a
   mean on a point estimate — and 9 of the 10 are significant after Holm.
3. **The neural baseline is the least robust thing here.** A DeepDTA-style MLP is worse than
   "predict the training mean" on **3 of 5** DAVIS protocols and **4 of 5** KIBA protocols
   (worst 2.0822 vs 1.0030 pKd), and unlike the Ridge case above all **7** of those cells clear
   Holm in the paired bootstrap (ΔRMSE −0.068 to −1.079, Holm p ≤ 0.046) — the collapse is not
   sampling noise. Kept in the tables, not deleted.
4. **Concordance has to be reported twice or it is inflated.** 69.64 % of DAVIS labels sit on
   the assay's censoring ceiling, which makes **36.4–48.7 % of the label pairs inside each
   DAVIS test block exact ties** (KIBA: 5.1–6.7 %) — and the two standard calibres differ by
   up to **0.1819** on identical predictions. Both are in every table, and the per-block tie
   counts are their own file.
   `ci_caliber_gap_max`, `data_summary.json → davis.fraction_at_censor_ceiling_pKd_5`,
   `results/label_tie_measures.csv`.
5. **Leakage is measured, not asserted.** **34.98 %** of KIBA `random` test rows have an ECFP4
   Tanimoto ≥ 0.7 ligand twin already in training; a scaffold split cuts that to **10.98 %**,
   and `cold_drug` turns out *not* to be a scaffold split — **48.79 %** of its held-out drugs
   keep a scaffold training already saw. `summary.json → headline_leakage`.
6. **Uncertainty is measured per protocol and it is ugly.** Split-conformal 90 % intervals
   cover 0.791–0.926 of the test block (mean 0.877) but are **2.68 training label sds wide** on
   average; at nominal 50 % the same interval over-covers to **0.700** on DAVIS `scaffold` (the
   ceiling spike) and under-covers to **0.288** on DAVIS `temporal` (drift). The adaptive
   variant made calibration worse in half the cells and wider in 15 of 20.
   `readme_numbers.json → conformal`.
7. **The physics arm beats the collapse, not the learners.** AutoDock Vina on 140 docked
   pairs ranks at CI 0.6583 pooled. On point estimates the trained model wins **11 of 16**
   head-to-head cells, Vina **4**, one tie — and **two** of Vina's four wins are against the
   MLP §4 already flags as collapsed. Paired bootstraps on identical pairs (rows, and targets
   as clusters) leave **none of the four** Vina ranking wins standing under row resampling;
   the only ranking difference that survives in Vina's favour is `cold_drug`/MLP, where the
   model's calibration is broken. In the other direction, on `random` the trained models'
   ranking advantage does survive (XGBoost ΔCI **+0.34**, cluster CI **[0.15, 0.54]**). Vina's
   only RMSE wins with an interval excluding zero are against the MLP on `scaffold`,
   `cold_drug` and `temporal`. On `scaffold` its CI falls to 0.421, *below* the constant
   predictor's 0.500. `readme_numbers.json → docking`, `docking_paired`.
8. **The headline reproduces on a third table that nobody in this repo had seen.** DAVIS and KIBA
   are both small, both kinase-centred and both widely re-used — exactly the setting in which a
   protocol artefact can look like a property of the field. So the same five protocols, three
   nulls, four models, three seeds, two concordance calibres and the same bootstrap families were
   re-run on a **ChEMBL kinase Ki panel** (ChEMBL_37, 19,088 pairs, 14,713 ligands, 16 human
   kinases, pKi = 9 − log₁₀(Ki/nM)), fetched and hashed by `13_fetch_chembl.py` and scored by
   `14_external_chembl.py` in a self-contained tree that does not touch `results/metrics.csv`:
   **η²(protocol) = 0.8657** (seed bootstrap CI [0.8606, 0.8702]) against **η²(model) = 0.0197**
   ([0.0172, 0.0223]) — a *smaller* model share than on DAVIS (6.8 %) or KIBA (30.7 %); the best
   achievable test RMSE runs **0.6467** (`random`) to **1.3514** (`temporal`), a range of
   **0.7047 pKi**, larger than DAVIS's 0.4823 and KIBA's 0.2101, while the three trained models
   differ by at most 0.3515 inside one protocol; and **15 of 15** per-arm hardest-minus-easiest
   protocol contrasts exclude zero under row resampling. The negative results reproduce too, and
   sharpen: under `temporal`, XGBoost beats the global-mean null by **−0.0064**
   (95 % CI [−0.0239, 0.0117], Holm p = 1.0) and random forest by **−0.0050** ([−0.0216, 0.0120],
   p = 1.0) — *statistically indistinguishable from predicting the training mean* — while only
   Ridge is significantly worse (+0.0933). The MLP collapses on `cold_target` (3.2429 vs a 1.6165
   null) and `temporal` (2.8400 vs 1.3576), as it does on DAVIS and KIBA. pKi, pKd and KIBA's
   consensus score are different scales, so **RMSE magnitudes are not compared across tables —
   only the protocol-vs-model structure is.** `results/external_chembl/paired_bootstrap.json`,
   §9.

The audit trail is part of the result: `04_uncertainty.py` recomputes `03_models.py`'s point
predictions and asserts the RMSE match (40/40 rows pass on the shipped grid), and that assert is
what caught an incomplete preprocessing fix that had quietly put two code versions in one table
— §8 tells that story, and `scripts/09_preproc_sensitivity.py` +
`results/preproc_sensitivity.csv` commit the three-way ablation as evidence.

## 1. What is measured, and against what

Every cell of the grid is scored with the same `evaluate()` in `scripts/dti_core.py`:
RMSE and MAE in label units (pKd / KIBA score — **not comparable across datasets**),
Pearson *r*, Spearman ρ, and **two concordance calibres reported side by side**:

| column | denominator | label tie | prediction tie |
|---|---|---|---|
| `ci_harrell` | comparable pairs only (unequal labels) | dropped | 0.5 credit |
| `ci_sklearn_like` | every pair | 0.5 credit | 0.5 credit |

Three different things get called "ties", and conflating them is how a concordance index gets
mis-read, so all three are measured per test block in `results/label_tie_measures.csv` and
plotted in `fig4_label_ties.png`:

| measure | what it counts | DAVIS range | KIBA range |
|---|---|---|---|
| `% label pairs tied` | the pairs `ci_harrell` throws away | 36.4 – 48.7 % | 5.1 – 6.7 % |
| `% rows on the modal label` | the censoring share (pKd 5.0) | 60.3 – 69.7 % | 11.8 – 15.2 % |
| `1 − distinct labels / rows` | pure label redundancy | 92.8 – 93.9 % | 94.0 – 97.5 % |

DAVIS puts roughly two-thirds of its rows on the censoring ceiling, which is enough to make
nearly half of the pairs inside a test block tied, and the two calibres disagree by up to
~0.18 on identical predictions. Reporting one of them silently is the single easiest way to
inflate an affinity benchmark, so both are in `results/metrics.csv` and both appear in every
table.

Against **three null baselines that learn nothing** (train means only): the global train
mean, the *per-ligand* train mean, and the *per-target* train mean, falling back to the
global mean for an unseen entity. On a scaffold split the per-target mean is the number a
"which kinase is this?" shortcut would achieve, and it is the floor the models have to beat.

Models (3 seeds each; features standardised on **training rows only**, with columns that are
constant on the training rows dropped and the surviving |z| clipped to 10 — that choice is not
cosmetic, see §8 and `results/preproc_sensitivity.csv`):

| model | features | notes |
|---|---|---|
| Ridge | 693-dim dense block | α ∈ {1, 3, 10, 30, 100, 300} chosen on the **validation** block |
| XGBoost | same | `hist`, 400 trees, depth 6, lr 0.08, early stopping (40 rounds) on validation |
| RandomForest | same | 150 trees, min_samples_leaf 20, max_features 0.3 |
| MLP (DeepDTA-style) | 949-dim dense block | 256/128/64, dropout 0.25, AdamW, cosine schedule, ≤ 25 epochs with validation early stop |
| mean baselines | — | see above |

The feature block is the conventional one: ECFP4 (Morgan radius 2) counts for the ligand,
plus 13 RDKit descriptors, and amino-acid + dipeptide composition for the target sequence —
i.e. deliberately *not* a SOTA architecture, so that the spread in the tables below is
attributable to the split protocol rather than to model engineering.

## 2. Data

| | DAVIS | KIBA |
|---|---|---|
| rows | 30,056 | 118,254 |
| ligands (distinct canonical SMILES) | 68 | 2,068 |
| targets by sequence / by name | 379 / 442 | 229 / 229 |
| label | pKd = 9 − log₁₀(Kd nM), Kd clipped to [0.016, 10 000] | KIBA score, 0 – 17.2 |
| rows sitting on the censoring ceiling | **69.6 % at pKd 5.0** | 1 row at the floor |
| grid fill | 68 × 442 complete | 118,254 / 473,572 = 25.0 % |

`00_fetch_data.py` runs 13 explicit PASS/FAIL integrity checks before anything is modelled
(row count = the exact sum of the five published DAVIS files, every label inside its physical
range, no unparseable SMILES, ids unique, …). Two of them are worth stating up front:

* **442 DAVIS target names collapse onto 379 sequences.** The same protein features appear
  under two different names, so a split done on `target_name` leaks by construction. Every
  split here keys on `target_id` (= sequence), and the label attached to a duplicated
  sequence is treated as interchangeable — see the caveat recorded in
  `data/processed/data_summary.json`.
* **The two benchmarks share 179 of their 379 DAVIS target sequences**, so
  "train on DAVIS, test on KIBA" is *not* target-disjoint.

Raw downloads are hash-pinned in [`docs/data_sources.md`](docs/data_sources.md); the tidy
tables and split assignments are committed, so the reported numbers can be re-derived
without network access.

## 3. Five split protocols, and the leakage each one admits

`02_splits.py` builds the protocols and then measures what they let through — an
ECFP4 Tanimoto nearest-neighbour search of every test ligand against the whole training
ligand set, Murcko-scaffold membership, and a 400-dim dipeptide-composition cosine for
target families:

| protocol | what it holds out | measured contamination |
|---|---|---|
| `random` | nothing | KIBA: **35.0 %** of test rows have a ≥0.7 Tanimoto twin in train (0.5 % at ≥0.95); 100 % of test scaffolds already in train |
| `scaffold` | generic Murcko scaffolds | KIBA: still **11.0 %** near-duplicate ligands ≥ 0.7 — scaffold-disjoint is not ligand-disjoint |
| `cold_drug` | ligands absent from train | KIBA: 26.2 % near-duplicate ≥ 0.7 and **48.8 % of its test scaffolds are present in train** — a "cold drug" split that is not scaffold-disjoint |
| `cold_target` | target sequences absent from train | the only protocol where the protein-similarity metric carries information (mean max cosine 0.81 – 0.85, 0 % of pairs ≥ 0.95) |
| `temporal` | a forward-looking date cut | DAVIS keeps 25,194 of 30,056 dated rows; KIBA keeps 44,691 of 118,254 (**37.8 %**) — the price of a look-ahead protocol is that it silently changes the dataset |

The published DAVIS cold-start blocks are cross-checked against these: in the shipped
partition `cold_drug` really is ligand-disjoint but every one of its targets is warm, and
the shipped `val` block is not entity-disjoint at all (100 % of its ligands *and* targets
also appear in train). A number quoted as "DAVIS cold-drug" therefore depends on which
shipped block the author used.

## 4. Model × protocol grid

`03_models.py` fits four learned models (Ridge with α searched on the validation block,
XGBoost 400 trees with early stopping, random forest 150 trees, a DeepDTA-style MLP) under
all five protocols on both datasets, **3 seeds each = 120 fitted runs**, and every run also
carries three deterministic nulls that need no learning: the global training mean, the
per-ligand training mean, and the per-target training mean. Test RMSE, mean of 3 seeds
(seeds are deterministic; the spread is in §10):

**DAVIS — RMSE in pKd units** (the training block's label sd is 0.79–0.91 depending on
protocol, so RMSE 0.85 ≈ "predict the mean")

| protocol | global-mean null | Ridge | XGBoost | Rand. forest | MLP | best learned |
|---|---|---|---|---|---|---|
| `random` | 0.9224 | 0.7561 | **0.5697** | 0.6558 | 0.7251 | XGBoost 0.5697 |
| `scaffold` | 0.8522 | 0.8786 | **0.8104** | 0.8110 | 1.6124 | XGBoost 0.8104 |
| `cold_drug` | 0.9173 | 0.8479 | **0.7872** | 0.7877 | 1.4814 | XGBoost 0.7872 |
| `cold_target` | 1.2116 | 1.1393 | **1.0520** | 1.0907 | 1.2061 | XGBoost 1.0520 |
| `temporal` | 1.0030 | 1.0442 | 0.9692 | **0.9458** | 2.0822 | RF 0.9458 |

**KIBA — RMSE in KIBA-score units** (not comparable with the DAVIS column above)

| protocol | global-mean null | Ridge | XGBoost | Rand. forest | MLP | best learned |
|---|---|---|---|---|---|---|
| `random` | 0.8343 | 0.6906 | **0.5143** | 0.5854 | 0.7630 | XGBoost 0.5143 |
| `scaffold` | 0.8701 | 0.7252 | 0.6193 | **0.6145** | 0.9385 | RF 0.6145 |
| `cold_drug` | 0.7225 | 0.6695 | 0.6025 | **0.5697** | 0.9598 | RF 0.5697 |
| `cold_target` | 0.9396 | 0.9241 | **0.7052** | 0.8214 | 1.1147 | XGBoost 0.7052 |
| `temporal` | 0.8149 | 0.9798 | 0.7583 | **0.7244** | 1.7832 | RF 0.7244 |

Four readings, each traceable to `results/metrics_by_protocol.csv` and
`results/readme_numbers.json`:

1. **The protocol is the biggest single effect.** Follow the `best learned` column: on DAVIS
   the best achievable RMSE moves from 0.5697 (`random`) to 1.0520 (`cold_target`) — a range
   of **0.4823 pKd**, i.e. more than half a label standard deviation bought or lost purely by
   how the rows are partitioned. On KIBA the same column moves **0.2101** (0.5143 → 0.7244,
   and there the hardest protocol is `temporal`, not `cold_target`).
2. **Model choice, among models that work, is smaller than that.** Within one protocol the
   three non-collapsed models differ by at most **0.1864** (DAVIS `random`) and usually by
   ≲ 0.10; on KIBA by at most 0.2554 (`temporal`). So on DAVIS, *changing the split costs about
   2.6–7.9× what changing the model buys.* The per-protocol winner also flips, and some flips
   are not resolvable: XGBoost wins four of five DAVIS protocols, but on `cold_drug` it beats
   random forest by 0.0005 while its own three seeds span 0.7395–0.8304 — at 3 seeds that
   "win" is noise. Script 12's paired bootstrap puts an interval on it directly:
   ΔRMSE = −0.0005, 95 % CI [−0.0309, 0.0220], Holm p = 0.93 — unresolved, as claimed. It also
   shows the converse: 6 of the 10 cells' *closest* pairs do clear Holm, at gaps of
   0.0233–0.1027 RMSE, because n ≈ 6–24 k test rows is a lot of resolving power
   (results/report.md §7.1). Of the 4 that stay unresolved, 3 are gaps of ≤ 0.005 and the
   fourth is DAVIS `random`'s Ridge–MLP pair at 0.0310 — a gap of the same size as the ones
   that cleared, wide interval because the MLP's own seed span is 0.18. So "not resolvable" is
   a property of the gap size and of the noisier member of the pair, not of the sample size.
   Random forest wins DAVIS `temporal` and three of five KIBA protocols. There
   is no "best model" row in this table, only best-model-per-protocol rows, and the
   best-model-per-protocol rows are sometimes ties.
3. **Two honest negative results — and the bootstrap now says which of them are shown.**
   (a) The **MLP is worse than predicting the training mean**
   on 3 of 5 DAVIS protocols and 4 of 5 KIBA protocols (DAVIS 1.6124 / 1.4814 / 2.0822 vs
   nulls 0.8522 / 0.9173 / 1.0030; KIBA 0.9385 / 0.9598 / 1.1147 / 1.7832 vs 0.8701 / 0.7225 /
   0.9396 / 0.8149) — only `random` rescues it, in both datasets. A DeepDTA-style network on
   hand-made features is not a free win; it is the least robust entry in the grid, and also
   the least reproducible one (seed span up to 0.179 vs ≤ 0.013 for random forest, §10).
   Script 12's paired bootstrap resolves **all seven** of those cells: ΔRMSE −0.0684 to
   −1.0793, every one significant after Holm (worst p = 0.046).
   (b) **Ridge loses to the global-mean null** on DAVIS `scaffold` (0.8786 vs 0.8522) and
   `temporal` (1.0442 vs 1.0030), and on KIBA `temporal` (0.9798 vs 0.8149) — a linear model
   fitted to the older decade of deposition dates does not extrapolate forward, and the tree
   models that do beat the null there buy only 0.03–0.09. Only two of the three losses are
   resolvable: `temporal` clears Holm in both datasets (ΔRMSE −0.0412 and −0.1649, p = 0.010),
   while the DAVIS `scaffold` gap of −0.0264 does **not** (p = 0.79). The same cell's
   "wins" are equally unresolved — XGBoost (+0.0419) and random forest (+0.0412) over the null
   both sit at p = 0.57 — so the honest reading of the whole DAVIS `scaffold` column is that
   the null and five of six learned models are inside each other's noise, and only the MLP's
   −0.76 is outside it.
4. **The entity nulls degenerate exactly where the protocols say they should**, which doubles
   as a check that the splits are what they claim. The per-ligand train mean is bit-identical
   to the global mean on `scaffold`, `cold_drug` *and* `temporal`, in **both** datasets — 6 of
   the 10 cells — which is itself a finding: on these two tables a Murcko-scaffold split holds
   out whole ligands, so "scaffold-disjoint" already implies "drug-unseen" for the test block.
   The per-target train mean likewise collapses to the global mean on `cold_target` in both
   datasets (KIBA 0.9396 for both rows). And **several of these nulls are competitive with, or
   better than, the learned models**: on DAVIS `scaffold` the per-target mean (0.7783) beats
   *every* learned model in the grid, including the best one by 0.032 — the only cell in 20
   where a two-line lookup is the state of the art; on KIBA `cold_target` the per-ligand mean
   (0.7131) beats Ridge, random forest and the MLP and loses to XGBoost by 0.008; on KIBA
   `cold_drug` the per-target mean (0.6538) beats Ridge (0.6695); on KIBA `random` the
   per-ligand mean (0.6481) beats Ridge (0.6906). **Knowing which kinase you are assaying — or
   which drug you are holding — beats a trained linear model.** A paper that reports a
   protocol number without the matching entity null next to it is hiding the number's most
   important rival.
   Script 12 puts that on a formal footing, in a form worth reading carefully because it is
   easy to misreport: of the 40 trained-model × cell contrasts against the global-mean null
   (4 models × 10 cells), 30 favour the model on a point estimate and **33 clear Holm** — but a
   two-sided test does not say *whose* side the rejection is on. Split by sign, 24 of the 33
   favour the model and **9 favour the null**, i.e. nine learned models are significantly
   *worse* than predicting the training mean. Printing "33/40 significant" as a win tally would
   mis-state 9 of 33 of them.

**A variance decomposition, not two ranges.** `11_variance_decomposition.py` fits
`metric ~ protocol + model + protocol×model` on the cell means (interaction as residual,
seeds as replication) and bootstraps seeds within each cell. For the three trained models,
protocol accounts for **91.3 %** of DAVIS RMSE variance (seed-bootstrap CI 90.7–91.7) against
**6.8 %** for model; on KIBA the split is **61.5 % / 30.7 %**. Add the MLP and the model share
jumps to **45.7 %** on DAVIS and **47.3 %** on KIBA, with a 24.1 % / 21.1 % interaction —
i.e. the "model hardly matters" claim is true of the tabular learners and false once a
collapsed architecture is allowed to interact with protocol. The intervals are seed-only: they
say how stable these 20 cells are to refitting, not how they would move under a different test
sample. `results/variance_decomposition.json`; the all-model spread and the trained-three
spread are separate keys so the two cannot be quoted interchangeably.

Concordance tells the same story with different units (`ci_harrell`, mean of 3 seeds): DAVIS
`random` peaks at 0.873 (XGBoost) while `temporal` falls to 0.666 and `cold_drug` to 0.692;
on KIBA, 0.815 → 0.688 from `random` to `temporal`. The nulls make the same point in the other
direction: the global-mean null is pinned at exactly 0.500 (all predictions identical, so no
pair can be ordered at all), yet a per-ligand mean reaches 0.755 and a per-target mean 0.656
on DAVIS `random` with no features fitted whatsoever. Remember §1: those DAVIS columns are
computed on a label distribution where ~two-thirds of the rows sit on the assay ceiling and
**41.4–48.7 % of the pairs in the block are exact label ties**, so `ci_sklearn_like` for the
same cell can sit ~0.18 away, and both are in the table.


## 5. Uncertainty: split conformal with measured coverage

RMSE is a property of a table; the question a user asks is how wrong **one** predicted pKd is.
`04_uncertainty.py` wraps the two cheapest models (Ridge, XGBoost — `--models` also accepts
random forest and the MLP) in split-conformal intervals: the protocol's validation block
supplies the nonconformity scores `|y − ŷ|`, the half-width is the `ceil((n+1)·level)`-th
smallest score (exact split conformal, no normality assumption), and the coverage on the *test*
block is then **measured**, not asserted. 20 configurations × 2 interval variants = 40 rows in
`results/uncertainty/conformal.csv`. Every row re-fits through `03`'s own `fit_predict` and its
test RMSE is checked against `results/metrics.csv`: **40/40 match to 1e-5**
(`conformal.rmse_all_match_03: true` in `results/readme_numbers.json`) — the same assert that
caught the §8 preprocessing bug, and the reason §8's story was catchable at all.

Empirical coverage at the middle and the top of the curve, mean absolute coverage error over the
six nominal levels 0.50–0.95, and the 90 % interval width (constant-width variant, seed 0,
`ridge / xgboost`):

| dataset × protocol | cov@0.50 Ridge | cov@0.50 XGB | cov@0.90 Ridge | cov@0.90 XGB | coverage error R / X | width90 R / X | width90 / sd R / X |
|---|---|---|---|---|---|---|---|
| davis `random` | 0.513 | 0.515 | 0.896 | 0.895 | 0.006 / 0.006 | 2.17 / 1.74 | 2.45 / 1.96 |
| davis `scaffold` | **0.700** | 0.511 | 0.884 | 0.881 | 0.081 / 0.021 | 2.75 / 2.39 | 3.03 / 2.64 |
| davis `cold_drug` | **0.366** | 0.438 | 0.878 | 0.905 | 0.098 / 0.052 | 2.43 / 2.78 | 2.73 / 3.12 |
| davis `cold_target` | 0.497 | **0.464** | **0.830** | **0.791** | 0.046 / 0.077 | 2.48 / 1.94 | 3.09 / 2.41 |
| davis `temporal` | **0.346** | **0.288** | 0.889 | 0.881 | 0.076 / **0.114** | 3.23 / 3.00 | **4.09** / 3.79 |
| kiba `random` | 0.491 | 0.497 | 0.897 | 0.900 | 0.005 / **0.002** | 2.13 / 1.58 | 2.55 / 1.89 |
| kiba `scaffold` | 0.488 | 0.504 | 0.894 | 0.886 | 0.015 / 0.005 | 2.21 / 1.90 | 2.67 / 2.29 |
| kiba `cold_drug` | 0.435 | **0.401** | **0.831** | **0.819** | 0.071 / 0.091 | 1.61 / 1.35 | 1.80 / 1.50 |
| kiba `cold_target` | 0.415 | **0.407** | 0.852 | 0.886 | 0.075 / 0.057 | 2.50 / 2.10 | 3.10 / 2.61 |
| kiba `temporal` | 0.496 | 0.581 | 0.916 | 0.926 | 0.020 / 0.051 | 3.32 / 2.89 | 3.18 / 2.77 |

Bold marks a cell ≥ 0.05 away from its nominal level at 0.50, or < 0.85 at 0.90. Five readings:

1. **The machinery is calibrated exactly where exchangeability holds** — and that is what makes
   the rest readable as protocol effects rather than implementation bugs. `random` traces the
   diagonal: KIBA/XGBoost goes 0.50→0.497, 0.80→0.801, 0.90→0.900, 0.95→0.949, coverage error
   0.0016; DAVIS/Ridge 0.50→0.513 … 0.95→0.945, error 0.0062.
2. **Over-coverage in the middle is the censoring signature.** On DAVIS `scaffold` a nominal
   50 % interval covers **70.0 %** of test rows (Ridge). With 68.8 % of this block's rows on the
   assay ceiling — 47.3 % of its label pairs tied — the residual distribution is a spike near
   zero plus a tail; one width that must reach the tail covers the spike twice over. The same
   cell's Ridge curve rises 0.700 → 0.738 → 0.778 → 0.823 → 0.884 → 0.921, i.e. it starts 0.20
   above nominal and converges to it.
3. **Under-coverage in the middle is the cold/drift signature.** 8 of 20 cells cover less than
   0.45 at nominal 0.50, and they are exactly the `cold_drug`, `cold_target` and DAVIS `temporal`
   cells (worst: DAVIS `temporal`/XGBoost **0.288**). The calibration block's central spread is
   narrower than the test block's — new targets and later assay batches are harder than the
   validation rows — so the interval is confident about the wrong middle. Under `temporal` that
   is not a surprise but a construction: the calibration rows *precede* the test rows in time.
4. **The 90 % tail mostly holds, and it holds by getting fat.** Mean cov90 is 0.877 (range
   0.791–0.926); the mean coverage error over all six levels is 0.048, worst cell DAVIS
   `temporal`/XGBoost 0.114. The cost is width: mean width90 = **2.68 training label sds**
   (range 1.50–4.09). In pKd units the interval is ±0.87–1.08 on DAVIS `random` (≈ a factor of
   7–12 in Kd), ±1.24–1.62 under `temporal` (≈ 32–42×), ±0.97–1.24 on DAVIS `cold_target`
   (≈ 9–17×).
   **A single affinity prediction from these features is worth a log-unit, not a decimal**, and
   any downstream use (ranking a compound series, deciding go/no-go) has to survive that.
5. **The adaptive (normalised) variant did not earn its complexity.** σ̂ = a Ridge on the first
   half of the calibration residuals, applied to the second half and to the test rows: it
   improved |cov90 − 0.90| in only **4 of 20** cells, improved the six-level coverage error in
   exactly **10 of 20** (a coin flip), and cost mean width in **15 of 20** (mean width90 2.68 vs
   2.33 label units, +15 %). Heteroscedasticity is real — corr(|residual|, |ŷ|) averages 0.32
   and reaches 0.57 — but σ̂ itself correlates only 0.10–0.40 with the true error, so dividing by
   it imports more variance than it removes. Worst case KIBA `scaffold`/Ridge collapses to
   **0.598** coverage at nominal 0.90 (curve 0.426 → 0.661). The shipped recommendation is the
   constant-width interval, and the width *is* the finding.

Two caveats are recorded in `results/uncertainty/uncertainty_summary.json` rather than hidden:
the validation block does double duty (`03` uses it for the α search / early stopping, `04` uses
it for calibration), so the 1−α statement is approximate, not a guarantee; and under `temporal`
exchangeability fails by construction, so its coverage number is a measurement of drift.

Figures: `results/figures/fig5_conformal_coverage.png` (empirical vs nominal, all 40 curves) and
`fig4_label_ties.png` (the censoring that produces reading 2).

## 6. The physics arm: AutoDock Vina on the same compounds

`05_docking.py` builds a structural panel from the DAVIS target names — gene → UniProt
(reviewed, human) → PDBe SIFTS `best_structures` (X-ray, ≤ 2.6 Å, ≥ 0.35 chain coverage) →
RCSB coordinates → gemmi chain trim → meeko → `vina.exe` (`--exhaustiveness 3`,
`--num_modes 2`, `--cpu 1`, `--seed 42+pair_id`) — and docks 20 pKd-stratified DAVIS
compounds prepared with RDKit ETKDGv3 + MMFF. **140/140 pairs scored, 0 failures, 70.4 s
per pair mean (8.7–353.2), 41.3 min wall-clock on 4 workers.** Every panel receptor passed
a Smith–Waterman local-identity gate ≥ 0.90 against the *DAVIS* sequence, so the assay's
own protein is what got docked:

| target | gene | PDB | Å | PDBe cov. | id vs DAVIS | co-crystal | box (Å) | pairs | mean Vina |
|---|---|---|---|---|---|---|---|---|---|
| T0004 | CDC2L1 | 7ukz_A | 2.60 | 0.389 | 0.9767 | NK3 (28 heavy) | 19.3×18.6×19.3 | 20 | −8.46 |
| T0006 | ERK2 | 8aoj_A | 1.12 | 1.000 | 0.9798 | N8L (21) | 19.9×15.9×17.1 | 20 | −7.38 |
| T0007 | ERK1 | 4qtb_A | 1.40 | 1.000 | 1.0000 | 38Z (44) | 26.7×20.4×24.9 | 20 | −9.56 |
| T0012 | ADCK3 | 7udq_A | 1.90 | 0.606 | 0.9560 | MVS (25) | 15.7×18.4×24.2 | 20 | −9.78 |
| T0013 | BRAF | 8c7x_A | 1.65 | 0.363 | 0.9487 | TXV (34) | 21.9×16.1×22.0 | 40 | −9.50 |
| T0015 | MKK7 | 6yz4_A | 1.70 | 0.730 | 0.9539 | 1E8 (33) | 19.7×20.2×19.3 | 20 | −8.34 |

**Vina as a ranker.** Pooled over all 140 docked rows, `−Vina` gives concordance
**0.6583** (Harrell) / 0.5421 (all-pairs caliber) on **2,590 comparable pairs** of 9,730.
Per target the honest reading needs the denominator, which `comparison.csv` now prints:
four of the six blocks contain only 19–37 comparable pairs, and **ERK1/ERK2 report
CI = 1.0000 on 19 pairs — because those 20 compounds are 19 ceiling labels plus a single
detected binder, so the "perfect" ranking is one compound beating 19 undetectable ones.**
Only CDC2L1 (70 pairs) and BRAF (374 pairs) have enough usable pairs to mean anything, and
there Vina scores **0.543** and **0.644** — close to chance. The pooled 0.658 is largely a
*between-target* effect: mean scores per pocket span −7.38 (ERK2) to −9.78 (ADCK3) kcal/mol,
and a physics score carries no knowledge of the assay's per-protein offset.

**Vina as a regressor** (`pKd ≈ a + b·(−Vina)`, two parameters fitted on the protocol's
*train* block and evaluated on its *test* block, so no label is used for the fit):

| protocol | test rows | fit rows | slope (pKd / kcal) | intercept | test RMSE | CI | comparable pairs |
|---|---|---|---|---|---|---|---|
| random | 26 | 105 | 0.068 | 4.57 | 1.019 | 0.638 | 94 |
| scaffold | 21 | 91 | 0.120 | 4.10 | 1.460 | 0.421 | 57 |
| cold_drug | 28 | 105 | 0.044 | 4.84 | 0.482 | 0.667 | 78 |
| cold_target | 80 | **0** | — | — | — | — | — |
| temporal | 28 | 70 | 0.093 | 4.30 | 1.377 | 0.559 | 188 |

Two things to read off this table. First the slopes: **0.04–0.12 pKd per kcal/mol**, so even
a 10 kcal/mol swing in score moves the prediction by less than the label spacing of the
assay — the map from docking energy to pKd is shallow and unstable across protocols. Second,
`cold_target` is **not computable**: every panel kinase is held out by construction, so the
calibration has zero training rows and `comparison.csv` records "too few docked pairs (train
0, test 80)" rather than inventing a number.

**Head-to-head on identical rows** (`results/docking/comparison.csv`, seed 0, the same pairs
the trained models predict, and for Vina the same two-parameter map fitted on the same
train block): of the 16 learned-model comparisons, **the trained model wins 11, Vina wins 4,
and 1 is an exact tie** — DAVIS `temporal`/Ridge scores CI 0.5585 and so does the two-parameter
Vina map, to four decimals, which is why `delta_ci_favours` says `tie` instead of rounding a
verdict. Vina's four wins are three inside `cold_drug` (CI 0.667 vs ridge 0.596, XGBoost 0.660,
MLP 0.436) and one inside `temporal` (0.559 vs the MLP's 0.436). Two caveats on those wins, both
of which the table already shows: against XGBoost on `cold_drug` the margin is **0.006 CI on 78
comparable pairs**, which is not a result and is not presented as one; and two of the four
wins are against the MLP, so the honest phrasing is *a rigid-pocket physics score beats the
baseline §4 already flagged as collapsed*, not *physics beats learning*. On `cold_drug` random
forest (0.801) and even the per-target train mean (0.821) stay clearly ahead of Vina. On
`random` Vina loses to all four learned models (0.638 vs 0.809–0.979) though it beats the
global-mean null, and on `scaffold` it lands at **0.421 — below the constant predictor's 0.500**
on 57 comparable pairs, i.e. the pocket ranking actively inverts the assay order there. What the
arm does establish is the *pattern*: physics is competitive exactly on the protocol that forbids
the model from having memorised the drug, and uninformative on the one that removes the scaffold
class the compounds were docked into.

**The point estimates above are not a test.** `10_paired_headtohead.py` keeps the identical
pairing (same test `pair_id`s, same Vina map fitted on the same train block, same seed-0
predictions) and puts a paired interval on both differences, resampling rows and — separately —
kinase targets, because pairs inside one target are not independent. The result is narrower
than the table suggests: **none of Vina's four ranking wins survives the row bootstrap**; the
target-cluster bootstrap keeps only `cold_drug`/MLP (ΔCI −0.231, [−0.478, −0.038]), which is a
cell where the MLP's calibration is already broken, not evidence that docking knows the
affinity order. In the other direction the three trained models' ranking advantage on `random`
does survive (XGBoost ΔCI **+0.340**, row [0.020, 0.789], cluster [0.154, 0.535];
random forest and MLP likewise under the cluster bootstrap). For RMSE, Vina is reliably
better only against the MLP on `scaffold` (−0.631, [−0.887, −0.463]), `cold_drug`
(−0.919, [−1.066, −0.786]) and `temporal` (−0.760, [−1.093, −0.576]); no trained model vs Vina
RMSE difference excludes zero. So the defensible claim is "Vina beats the collapsed neural
baseline and ties the tabular learners at n = 21–28", not "Vina beats learning".
`readme_numbers.json → docking_paired`; the full 16-row table is
`results/docking/paired_headtohead.csv`.

The pKd-stratified 20-compound docked subset is small by design (26–28 test rows per protocol,
80 under `cold_target`), so the per-protocol point estimates are fragile; the paired intervals
above are the error bar, and they leave most learned-vs-Vina ranking gaps unresolved at this n.
§10 says so explicitly.

**Specificity control.** Re-docking 10 sampled pairs with the search box translated +30 Å
along x into bulk solvent: mean fitted strength falls from **−8.478 to −6.414 kcal/mol**
(the shifted box finds nothing comparable), which is the signal that the pocket, not the
molecule, was doing the work. The CI of that control is *not* reported as evidence: 10
ceiling-heavy rows leave **9 comparable pairs of 45**, so both the on-site (0.222) and
shifted (0.333) CIs are noise, and `comparison.csv` says so in its `note` column.

Reproduce: `python run_all.py --with-docking --stage docking` then `--stage dockstats`, or
`--stage paired` for the bootstrap table alone. `11_variance_decomposition.py` needs only
`results/metrics.csv`. `vina.exe` is not committed (see `.gitignore`); `docs/data_sources.md`
records its version, size and sha256, and the script refuses to start without it.

## 7. Layout and how to run it

```
data/processed/    committed tidy tables (davis.csv, kiba.csv, ligands.csv, targets.csv)
                   + the five split files per dataset + feature arrays
                   + external/ : the ChEMBL kinase Ki table of §9 (tables, hashes, features,
                     splits) — a self-contained tree that the main pipeline never reads
results/           committed metrics, leakage audit, docking tables, bootstrap tables
                   (testset_bootstrap_ci/models/pairs + .json), figures, report.md
results/external_chembl/  §9's own results (metrics, leakage audit, Holm contrasts,
                   bootstrap json, run manifest) — never written by scripts 00…12
scripts/00…12      one stage per script, run in order
scripts/13, 14     the external table: fetch + hash-pin (13), then features/splits/audit/
                   grid/analyse on it (14); imported by the same modules, not re-implemented
run_all.py         one-command reproduction
docs/data_sources.md  endpoints, sha256 of every raw file, vina binary hash, versions
```

```bash
pip install -r requirements.txt
python run_all.py                       # order of 2 h on an 8-core laptop; the measured part is
                                        # 03's 69.8 min of summed fit time (metrics.csv `seconds`)
python run_all.py --quick               # ~7 min smoke test
python run_all.py --with-docking        # adds the Vina arm (needs tools/vina.exe)
python run_all.py --with-sensitivity    # adds 09: the three standardisation guards re-fitted
python run_all.py --stage bootstrap     # adds 12: intervals for all 150 fits, no refitting.
                                        # Measured 82 min wall (4,936 s) at reps 2000 /
                                        # ci-reps 600 / 6 parallel cells on the same laptop;
                                        # needs results/predictions/*.npz from a prior 03 run
                                        # (that directory is gitignored, so a fresh clone must
                                        # run the grid before this stage alone will work)
python run_all.py --stage report        # re-render tables + figures from committed CSVs
python run_all.py --stage external      # §9: fetch + hash-pin the ChEMBL panel (network),
                                        # then features/splits/audit/grid/analyse on it.
                                        # Measured: fetch 77.5 MB transferred, features 38 s,
                                        # audit 50 s, grid 291 s, analyse 766 s (2,000 reps)
python run_all.py --stage external --quick   # smoke version of the same two stages; runs in
                                        # data/processed/external_quick/ +
                                        # results/external_chembl_quick/ so it cannot overwrite
                                        # the committed table or grid
```

`run_all.py` recomputes the whole grid by default; `--resume-grid` is opt-in, because keeping
old rows is exactly how `results/metrics.csv` once ended up mixing two code versions (§8).

## 8. What could go wrong, and what catches it

| risk | what the code does about it |
|---|---|
| a fast concordance index silently disagreeing with the definition | `dti_core.selftest()` compares the Fenwick-tree CI against an O(n²) brute force on synthetic data (perfect / reversed / ties / random) and `03_models.py` refuses to start until it matches to 1e-9 |
| mis-assigned protein sequence in the docking panel | `sw_align()` is Smith–Waterman with a sign convention that is unit-tested four ways (identical / substring / one-mismatch / disjoint); the script exits non-zero if any case fails, and every panel target is then rejected unless its **local identity against the DAVIS sequence** is ≥ 0.90 |
| a search box that is not a binding site | the box is derived from a co-crystal ligand with ≥ 15 heavy atoms, and any dimension outside 10–30 Å rejects the structure; a **decoy control** re-docks sampled pairs with the box translated +30 Å into solvent |
| labels recovered from feature-only files being wrong | `00_fetch_data.py` cross-aligns the canonical DAVIS tables against the published Kd table row by row and requires agreement (max abs Δ pKd recorded, 0 disagreeing rows) |
| test information reaching the scaler | `standardize()` computes mean/sd from training rows only, drops the columns that are constant on training rows, and clips the rest — the drop mask and the clip are both train-only |
| a metric table that silently drifts from the code that made it | `04_uncertainty.py` re-fits the point predictions and asserts its own test RMSE against `results/metrics.csv` to 1e-5; `08_notebook.py` executes the notebook and fails the build on any error output. **The assert has fired for real**: it reported davis/scaffold/ridge as 1.4973 recomputed vs 0.8409 stored, which is how a table holding rows from two code versions was caught — the grid was then re-run from scratch instead of patched with `--resume` |
| a long grid run dying half-way | `03_models.py` rewrites `results/metrics.csv` after **every** seed and supports `--resume`, so a crash costs one row, not two hours |
| a bootstrap reporting "no significant difference" that is only an artefact of its own rep count | a percentile-bootstrap two-sided p cannot fall below `2/(B+1)`, and Holm over `m` tests needs `0.05/m`; over the 105 arm pairs in a cell that requires `B > 2095`, so at B=2000 **nothing can ever be rejected**. `12_testset_bootstrap.py` computes this check (`holm_resolvable`), stamps `holm_underpowered=True` on every arm-level row instead of a verdict, and takes its claims from a model-level family of 10 (needs only `B > 398`). The first version of that script printed "0/660 model-vs-model RMSE gaps significant" and the number was arithmetic, not data |

Two bugs found this way are documented rather than hidden — and the second one is a
preprocessing decision that changed a headline number, so it gets its own ablation
(`scripts/09_preproc_sensitivity.py` → `results/preproc_sensitivity.csv`):

* **Train-constant fingerprint columns destroyed the neural model, and fixing them moved the
  linear model.** The first version of `standardize()` divided by `max(sd, 1e-6)`. A bit that
  never appears in the training rows but appears in a test row then gets z ≈ 10⁶ — measured
  **|z|max = 2,000,000** on the DAVIS `scaffold` / `cold_drug` / `temporal` test blocks (13.2 on
  `random`, which is why `random` looked fine). The DeepDTA-style MLP "converged" to test RMSE
  **116,142** on DAVIS `scaffold`. The first fix left such columns at scale 1 and clipped every
  z to ±10; that repaired the MLP (116,142 → 1.611) but pushed **Ridge from 0.8805 to 1.4973**
  — worse than the 0.8522 null. The shipped guard instead *drops* columns whose training sd is
  below 10⁻³ (a column with no training variance carries no training information for a linear
  or neural model; its test value is pure out-of-support extrapolation), then z-scores and
  clips what survives: Ridge back to **0.8786**, MLP 1.6165, XGBoost 0.8199. Measured on the
  same rows, same seed:

  | DAVIS `scaffold`, seed 0 | floor 1e-6 (bug) | clip-only fix | **drop + clip (shipped)** |
  |---|---|---|---|
  | Ridge | 0.8805 | **1.4973** | 0.8786 |
  | XGBoost | 0.7933 | 0.7933 | 0.8199 |
  | MLP | **116,142** | 1.6110 | 1.6165 |
  | (global-mean null) | 0.8522 | 0.8522 | 0.8522 |

  Three things follow, and all three are in the repo rather than in a footnote. (i) A bug can be
  *model-specific*: the 10⁶ blow-up is fatal to an untrained weight times a huge input (the MLP),
  harmless to a ridge coefficient that is exactly zero on a zero training column, and harmless to
  a split threshold. Attributing "the linear and neural models were broken" to one cause, as my
  first version of this section did, was wrong. (ii) The guard is a real decision, not a free
  repair: dropping columns costs XGBoost 0.0266 RMSE here and buys Ridge 0.62 — so cold-protocol
  numbers are partly a preprocessing choice, which is why the ablation is a committed script and
  not a sentence. (iii) The MLP's collapse to ~1.6 against a 0.85 null **survives both repairs**,
  so §4's negative result about it is a property of the model under cold-drug protocols, not an
  artifact of my scaling. `03_models.py` also stopped re-standardising its input a second time
  inside `_mlp()`.

  The ablation is not one cell — it is the whole DAVIS grid (5 protocols × 4 models × 3 guards,
  seed 0, `results/preproc_sensitivity.csv`), and it puts a number on how much of §4 is a
  preprocessing decision:

  | arm | largest RMSE swing across the three guards | where |
  |---|---|---|
  | RandomForest | 0.0166 | `cold_drug` |
  | XGBoost | 0.0266 | `scaffold` |
  | Ridge | 0.6187 | `scaffold` |
  | MLP | 154,527 | `temporal` (154,529 → 2.13) |

  The trees are the invariance control and they behave like one: a split threshold does not care
  whether an absent fingerprint bit sits at 10⁶, at 10, or is gone. The two arms that do care are
  exactly the two whose cold-protocol conclusions §4 draws. One cell flips a conclusion outright
  — DAVIS `cold_target`/MLP scores 1.1621 (floor), 1.2134 (clip-only) and 1.2020 (shipped)
  against a 1.2116 null, so "the MLP beats the training mean" is true under two of the three
  guards and false under the third, and the shipped margin (0.0096) is a fifth of the guard swing
  on that same cell (`preproc_sensitivity.json → null_flips_caught_by_the_guard`, one entry in
  60 fits). §4's only MLP win outside `random` should be read as noise, not as a result.

  And the guard is now pinned to the shipped table: `09` recomputes every DAVIS seed-0 cell under
  `drop_clip10` and compares it against `results/metrics.csv` — **20 of 20 rows agree to 1e-5**
  (`preproc_sensitivity.json → shipped_guard_reproduces_stored_grid`), so the committed numbers
  can be regenerated from the committed code instead of taken on trust.
* **A duplicated keyword argument** (`np.savez_compressed(..., y=…, y=…)`) made
  `03_models.py` a `SyntaxError` for every *new* process while an already-running process
  kept working from the pre-edit bytecode — which is how a crash got misread as "stale
  cache". It is fixed, and it is the reason the per-row `metrics.csv` flush exists.

## 9. External validation on a third table (ChEMBL kinase Ki)

**Why a third table at all.** DAVIS and KIBA share exactly the properties that make a
protocol artefact look like a property of the field: both are kinase-centred, both are small
(68 ligands / 379 sequences, and a 25 %-filled matrix), and both are re-used by so many
papers that a protocol tuned on them is already tuned on the literature. §1–§8 therefore rest on
*two tables* — and two tables that agree can still agree for a shared reason. The test here is
whether the *structure* of the headline survives on a table whose curation pipeline, label
definition, temporal field and split sizes share none of those properties.

**The table.** `scripts/13_fetch_chembl.py` builds a 16-target human kinase panel from the
ChEMBL REST API (ChEMBL_37, release date 2026-05-01, version read live at run time): human
SINGLE PROTEIN targets whose preferred name contains "kinase", exact-standard Ki rows
(`standard_type=Ki`, `standard_flag=1`, `standard_relation='='`), the top 16 by that count with
≥ 300 such rows each. Every (ligand, target) pair is labelled **pKi = 9 − log₁₀(median
Ki / nM)** over its curated rows: **19,088 pairs, 14,713 ligands, 16 targets, 398 source
documents**, pKi 4.0–11.0 (mean 7.786, sd 1.476). Everything is hashed before it is modelled —
`data/processed/external/external_manifest.json` carries the sha256 of each output table and of
all 1,123 cached raw API pages.

**The checks that had to pass before the table was allowed near a model.**

| check | measured result |
|---|---|
| every API filter is discriminating, not silently ignored | `molecule_molecule_chembl_id__in` reports `total_count` 3 for 3 requested ids; an ignored filter would report the whole compound table |
| protein sequences are the ones ChEMBL claimed | 16 requested / **16 verified** (fasta header accession equal to the ChEMBL accession *and* `OS=` equal to "Homo sapiens") / 0 rejected |
| the structure attached to an activity is that molecule's structure | 300 sampled rows, **300/300 exact string matches** between `activity.canonical_smiles` and `molecule_structures.canonical_smiles` |
| the year attached to an activity is the year of its document | 18,978 rows carrying both, **18,978 agree, 0 disagree** |
| the local filters actually bite | 4,197 potential duplicates dropped, 229 data-validity comments, 22 rows without a canonical SMILES |

**The sampling caveat, and the repair that came out of it.** The download cap means the fetcher
cannot read every Ki row of every target, so it samples pages (page size 1,000, up to 3,000
distinct rows per target) and spreads them across that target's own offset range instead of
reading the head. The first version sampled offsets 0/1000/2000 for every target; because
ChEMBL paginates by activity id and each target's Ki rows are *not* contiguous, that returned
the same activity rows under two targets and inflated the pair count. The repair was measured,
not assumed: the offsets are now per-target and listed in the manifest
(`row_sampling.offsets_per_target`), the re-fetch re-hashed every table, and the manifest now
reports per target how many rows the API claimed, how many *distinct* rows were fetched and how
many survived the local filters (`coverage_per_target`, with a `complete` flag). Three of the
sixteen targets are still `complete: false` — that is the cap, not a bug, and it is labelled as
such rather than hidden.

**Ordering probe.** If the API returns rows in deposition order, a head-only sample would
silently over-represent old measurements and bias the temporal protocol. Measured on the live
API (`data/processed/external/probe_page_year_ordering.json`): for CHEMBL262 the head page has
median year 2011 and the deep page at offset 269 has 2021 (`later_at_depth: true`), while for
CHEMBL1862 both pages have median 2014 (`false`). The ordering is therefore *not* uniform across
targets — which is exactly why the sampler spreads pages instead of reading the head.

**Download footprint** against the ≤ 200 MB the run was authorised under: 59.2 MB + 18.3 MB +
0.0 MB over the three fetch runs = **77.5 MB transferred**; the on-disk raw cache is
104,814,223 bytes; the script's own budget is 175 MB and `budget_exceeded` is `false`.

**What was reused, and what was re-derived.** Reused *by import, not by copy*:
`01_features.build_ligand_features / build_protein_features / scaffold_table / bit_vectors`,
`02_splits.random_split / scaffold_split / cold_split / temporal_split / audit`,
`03_models.entity_mean / standardize / fit_predict`, and `dti_core.evaluate / ci_harrell`.
Re-derived for this table: every feature array, every split, every fit, every bootstrap draw.
The isolation is structural rather than a promise — `dti_core.Featuriser` reads the module
globals `PROC`/`FEAT` at call time, so script 14 points them at `data/processed/external/`.
The consequence is that the 256 top ECFP4 bits are selected over the *external* ligand set
only, and the shipped DAVIS/KIBA arrays — hence every number in `results/metrics.csv` — cannot
move. Checked after the run: `git diff` over `results/metrics.csv`, `results/readme_numbers.json`,
`results/variance_decomposition.json` and `data/processed/` is empty. Script 14 writes only to
`data/processed/external/` and `results/external_chembl/`.

**The grid** (75 rows: 4 learned models × 3 seeds + 3 nulls × 5 protocols). RMSE in pKi units,
3-seed mean; the training label sd is 1.35–1.49 per protocol, so "predict the mean" costs
about 1.35–1.62.

| protocol | global-mean null | per-ligand null | per-target null | Ridge | XGBoost | Rand. forest | MLP | best learned |
|---|---|---|---|---|---|---|---|---|
| random | 1.4780 | 1.3022 | 1.2634 | 0.8988 | **0.6467** | 0.7519 | 0.9795 | XGBoost 0.6467 |
| scaffold | 1.4937 | 1.4937 | 1.2818 | 0.9113 | **0.7315** | 0.7670 | 1.0065 | XGBoost 0.7315 |
| cold_drug | 1.6209 | 1.6209 | 1.7731 | 1.4828 | **1.1415** | 1.2117 | 1.3366 | XGBoost 1.1415 |
| cold_target | 1.6165 | 1.2957 | 1.6165 | **1.2129** | 1.5644 | 1.5136 | 3.2429 | Ridge 1.2129 |
| temporal | 1.3576 | 1.3588 | **1.2397** | 1.4506 | 1.3514 | 1.3528 | 2.8400 | XGBoost 1.3514 |

**What the third table adds.**

1. **The protocol effect is larger here, not smaller.** The best achievable RMSE runs 0.6467
   (`random`) → 1.3514 (`temporal`), a range of **0.7047 pKi**, against DAVIS's 0.4823 and
   KIBA's 0.2101. The two-way decomposition gives **η²(protocol) = 0.8657** (seed bootstrap CI
   [0.8606, 0.8702]) against **η²(model) = 0.0197** ([0.0172, 0.0223]) — a *smaller* model share
   than on either shipped table. Adding the collapsed MLP (all4) gives η²(protocol) 0.4735 /
   η²(model) 0.2599 / interaction 0.2666, so once again "the model barely matters" is a
   statement about the trained tabular models, not about architecture.
2. **Every per-arm protocol contrast survives.** Hardest-minus-easiest RMSE spread runs
   **0.2633–2.4121** over the 15 arms and **15 of 15** intervals exclude zero under row
   resampling (same conservative treatment of the two protocols' draws as script 12).
3. **The negative results reproduce and sharpen.** Under `temporal` the trees are
   *statistically indistinguishable from predicting the training mean*: XGBoost − mean =
   **−0.0064** [−0.0239, 0.0117] and random forest − mean = **−0.0050** [−0.0216, 0.0120],
   both Holm p = 1.0, while only Ridge is significantly worse (+0.0933, p = 0.01). XGBoost vs
   random forest on `temporal` is −0.0014 [−0.0079, 0.0054], Holm p = 1.0 — the same
   "the gap is smaller than the noise" cell that DAVIS `cold_drug` (0.0005) provides on the
   other table. The MLP collapses on `cold_target` (3.2429 vs a 1.6165 null) and `temporal`
   (2.8400 vs 1.3576), as it does on DAVIS and KIBA.
4. **A two-line lookup wins a cell here too.** Under `temporal` the **per-target training mean
   (1.2397) beats every learned model** (best 1.3514). Two honest caveats: 35.69 % of test rows
   have a target absent from training, so this null is a partial lookup with a fallback; and it
   is not a member of the Holm family above, which is defined against the *global* mean, so it
   is reported as a point estimate rather than as a tested win.
5. **Leakage is again a property of the protocol, but the ordering differs from the shipped
   tables.** `random` leaks *more* here — 89.79 % of test rows have an ECFP4 ≥ 0.7 ligand twin
   in train (40.65 % at ≥ 0.95) against DAVIS/KIBA's ~35 % — and `temporal` is the cleanest
   (6.44 %). `cold_target` is not clean either: 56.78 % of its test rows still have a
   near-duplicate ligand in train, because that protocol hides *targets*, not ligands.

| protocol | test rows with a ligand ≥ 0.7 twin | ≥ 0.95 | scaffold already in train | rows whose ligand is unseen | rows whose target is unseen | rows with target family ≥ 0.95 |
|---|---|---|---|---|---|---|
| random | 89.79 % | 40.65 % | 80.02 % | 69.25 % | 0.00 % | 100 % |
| scaffold | 66.87 % | 0.45 % | 0.00 % | 100.00 % | 0.00 % | 100 % |
| cold_drug | 18.91 % | 1.15 % | 30.15 % | 100.00 % | 0.00 % | 100 % |
| cold_target | 56.78 % | 42.67 % | 51.61 % | 57.59 % | 100.00 % | 0.00 % |
| temporal | 6.44 % | 0.66 % | 4.17 % | 99.89 % | 35.69 % | 64.31 % |

6. **The scale caveat, stated once and applied everywhere.** pKi shares the 9 − log₁₀(nM)
   *definition* with DAVIS pKd but not the measurement process; KIBA's 0–17.2 consensus score
   shares neither. **RMSE magnitudes are therefore not comparable across the three tables**,
   and nothing above compares them. What is comparable is the *structure* — the protocol-vs-model
   variance share, the sign and resolvability of paired contrasts, which nulls win which cells
   — and that structure is what was tested.

**Scope of the external arm.** It reproduces the *protocol-vs-model* headline only. The
conformal arm (script 04) and the docking arm (scripts 05/06) were **not** re-run on this
table: the docking panel is DAVIS-specific by construction (it needs DAVIS compounds, DAVIS
sequences and a per-target Vina calibration), and the conformal arm is a property of the
protocol, not of the headline being tested. Both remain as they are on the shipped grid.

**Where the numbers live.** `results/external_chembl/` — `metrics.csv` (75 rows),
`metrics_by_protocol.csv`, `leakage_audit.csv`/`.json`, `model_contrasts_holm.csv` (50 rows:
5 protocols × 10 contrasts), `arm_marginal_intervals.csv`, `paired_bootstrap.json` (variance
decomposition, protocol contrasts, cross-table block) and `external_run_manifest.json` (every
output file with its sha256). The bootstrap's point estimates reproduce `metrics.csv` exactly
(max absolute deviation 0.0 over all five cells), so the intervals provably come from this
table's own predictions. `results/external_chembl/predictions/` is gitignored like
`results/predictions/`: it is regenerated by `--stage grid`, and the analyse stage reads it.

**How to run it.**

```bash
python scripts/13_fetch_chembl.py --budget-mb 175      # network; caches every page it reads
python scripts/14_external_chembl.py --stage all        # features → splits → audit → grid → analyse
python scripts/14_external_chembl.py --stage analyse --reps 2000 --ci-reps 600 --jobs 5
```

or `python run_all.py --stage external`, which runs both. Measured wall time on the same
8-core laptop: features 38.0 s, audit 50.3 s (202,980,503 similarity entries computed), grid
291.0 s, analyse **765.5 s** (2,000 RMSE/MAE reps + 600 Harrell-CI reps, 5 protocol cells in
parallel; the committed `results/external_chembl/` is the output of one such run).

**A smoke run cannot overwrite the committed artifacts.** `13`/`14` take an `--out-tag` that
moves the whole tree — table, features, splits, grid and results — to a sibling directory, and
`--quick` implies `--out-tag _quick`, so `python run_all.py --stage external --quick` works in
`data/processed/external_quick/` and `results/external_chembl_quick/` (both gitignored). This
was not cosmetic: the untagged quick run originally wrote into the same directories as the
19,088-pair table and the 75-row grid, and the bootstrap pool's worker processes re-imported
the module in a fresh interpreter, so a child could read the *committed* prediction files while
its parent wrote the smoke ones. Both are fixed — the tag is now applied inside the worker too —
and the isolation is checked, not asserted: after a full quick run, all 27 committed files
under `data/processed/external/` and `results/external_chembl/` are byte-identical.

## 10. Known limitations / next steps

**The label is censored, and nothing here fixes that.** 69.64 % of DAVIS pKd values sit on
the assay ceiling (pKd 5.0 = "Kd ≥ 10 000 nM, the limit of detection"). A model that
predicts the ceiling for everything therefore gets a respectable RMSE while having
concordance 0.5, and the residual distribution is a mixture of "true affinity" and
"undetected", which is why constant-width conformal intervals over-cover. The statistically
correct treatment is a censored (Tobit) likelihood or a two-stage classifier-then-regressor;
this repo uses ordinary squared error and *reports* the consequence (both CI calibers, the three
separate tie measures in §1 and `results/label_tie_measures.csv`, the ceiling share in §2)
instead of modelling around it.

**Ligand identity is RDKit canonical SMILES only** (`canonical_smiles()` in
`00_fetch_data.py`: `MolFromSmiles` → `MolToSmiles`). No tautomer, protonation-state or
salt-fragment enumeration (no Epik/SaltBreak), so the same compound measured in two
ionisation states is two ligands, and the ≥0.7 Tanimoto near-duplicate counts in §3 are a
lower bound on real chemical duplication.

**DAVIS is small, and the protocols are not independent.** 68 ligands against 379 distinct
target sequences (442 names). Both `scaffold` and `cold_drug` hold out 13 of those 68
ligands, and only 3 ligands are shared between the two held-out sets, so a one-point RMSE
difference between the two protocols is a difference between *which* 13 drugs were hidden,
not a proven property of the split family. Script 12 now puts that worry on a measured
footing rather than an asserted one: every cell in the grid carries marginal intervals under
three resamples — test `row`, `target` cluster and `ligand` cluster — for RMSE and MAE
(1,200 rows in `results/testset_bootstrap_ci.csv`). Resampling *compounds* instead of rows
widens those intervals by a median factor of **5.11 on DAVIS and 3.10 on KIBA** for RMSE
(6.82 and 4.04 for MAE), which is what 68 ligands costs you. Applied to the exact contrast
above — DAVIS `scaffold` vs `cold_drug` — the point difference is small (−0.085 to +0.179
RMSE) *and flips sign with the model*: `scaffold` comes out the easier protocol for the three
nulls and one XGBoost seed, and the harder one for all three seeds of Ridge, random forest and
the MLP plus the other two XGBoost seeds.
Under row resampling the two protocols' 95 % intervals are disjoint for 7 of those 15
model × seed arms; under ligand resampling they are disjoint for **0 of 15**. The biggest-
spread contrast does survive either resample: the hardest-minus-easiest protocol difference
per arm (0.2171–1.4746 RMSE) excludes zero in 30 of 30 dataset × arm contrasts under row
resampling (`protocol_spread_excludes_zero_frac = 1.0` in `results/testset_bootstrap.json`),
and those same 30 easiest-vs-hardest pairs stay disjoint under ligand resampling too. Both
checks compare *marginal* intervals, not a paired ligand-level contrast; disjointness is
sufficient but not necessary for separation, so "0 of 15" records the absence of unpaired
evidence under compound resampling, not a tested null. The paired ligand-level protocol
contrast is not computed.

**KIBA is a consensus score, not a physical unit.** Its 0–17.2 scale is a
kinase-inhibitor activity consensus, not a pKd, and the matrix is only 24.97 % filled. One
row sits on the score floor. KIBA numbers are comparable to other KIBA numbers and to
nothing else.

**Conformal coverage is measured, not guaranteed.** `04_uncertainty.py` calibrates on the
same validation block that `03_models.py` used for the ridge α search and the XGBoost early
stop, so the 1−α coverage statement does not strictly hold; a fourth split (or a nested
resample) would be needed for that. The point predictions now come from a *single* fit
shared by the calibration and test rows — the earlier version of this script fitted twice,
which made the two sides' scores come from two different models. Under `temporal`,
exchangeability fails by construction (calibration precedes test in time), so the coverage
there is a measurement of drift, not a property of the method.

**The docking arm is a handful of rigid pockets.** 6 kinase targets, one X-ray structure
each, box from a co-crystal ligand, rigid receptor, no induced fit and no explicit solvent;
`--exhaustiveness 3`, `--num_modes 2`, 70.4 s per pair on average (range 8.7–353.2 s, 4
workers, 41.3 min wall-clock for all 140 pairs — a production setting would be 8-32
exhaustiveness and several minutes per pair regardless). Compounds are redocked from
ETKDG+MMFF conformers, so the reference
pose is not the crystal pose. Per-target concordance is therefore estimated from ~20
points. Under `cold_target` the panel targets are *entirely* held out, so the two-parameter
Vina→pKd calibration has zero training rows there and `comparison.csv` records that as
"too few docked pairs" rather than inventing a number.

**ML and physics are not information-matched.** The ML arm sees ECFP4 count bits plus
amino-acid/dipeptide composition: it has no notion of pocket geometry, which is precisely
what Vina sees, while Vina has no notion of the rest of the assay table, which is what the
ML arm exploits. §6 compares them on identical *rows* with identical *metrics*, which is
the fair part; it is not an apples-to-apples information comparison, and no "docking loses
to ML" conclusion should be drawn from it.

**The paired docking test is low-powered, and the variance decomposition is cell-level.**
Fifteen of the sixteen learned-vs-Vina intervals in §6 include zero under row resampling
(twelve under target clustering), which is the honest reading at 21–28 test rows, but it is an
absence of evidence, not evidence of equivalence; the cells that do exclude zero are the ones
§6 names. The two-way
decomposition in §4 treats the 20 protocol × model cells as fixed levels and resamples only
the three seeds within each cell, so its intervals describe refit stability, not sampling
variability over ligands, targets or assay batches. Script 12 now supplies the test-set
bootstrap over the full grid that decomposition was not a substitute for — but the two measure
different noise and neither retires the other. Script 12's intervals carry test-set sampling
only, not refit randomness: every contrast in it is between two *fixed* fitted models, so an
unresolved gap there is unresolved for these fits on these rows, not for the method. The §4
seed intervals do the opposite — hold the test set fixed and move the fit. Reading either one
alone would overstate what it licenses.

**Seed spread is real only for the stochastic models, and it is not uniform across them.**
Ridge's three seeds are bit-identical rows in all 10 dataset × protocol cells, so `rmse_sd = 0`
there is a property of the estimator, not evidence of stability. XGBoost and random forest are
seeded but not bit-reproducible across environments or code versions, and their seed noise is
comparable to the differences §4 reads as a winner. Measured on the shipped grid (span =
max−min RMSE over the 3 seeds within one cell, seed order in parentheses): random forest ≤
**0.0125** (worst davis `temporal`, 0.9413 / 0.9425 / 0.9538), XGBoost ≤ **0.0909** (worst
davis `cold_drug`, 0.7916 / 0.7395 / 0.8304 — the spread is 180× the 0.0005 by which XGBoost
"beats" random forest in that cell), the MLP ≤ **0.1785** (worst davis `temporal`,
2.0175 / 2.0331 / 2.1961; 0.169 on KIBA `temporal`). Read together with §4: the tree-model
differences sit at or below their own seed noise, while the MLP's collapse sits far above it,
so §4's negative result for the MLP is not a seeding artifact. Per-seed rows are in
`results/metrics.csv`. That file is the record of the run that produced the figures and the
tables above, on Python 3.13.7 / i7-9700 / 32 GB with the versions in `requirements.txt`.

**The external table has three limitations of its own, and they are not the same as the
shipped ones.** (i) *Scale*: pKi = 9 − log₁₀(median Ki / nM) shares the **definition** of
DAVIS pKd but not the measurement process, and KIBA's 0–17.2 consensus score shares neither —
so RMSE magnitudes are comparable **within** the ChEMBL table and **not across** the three
tables; §9 compares structure, never magnitude. (ii) *The temporal protocol is a proxy*: the
public ChEMBL API exposes no per-activity deposition date, so the temporal split uses the
**source publication year**, at one-year resolution, and the 2,829 pairs with no year at all
(14.8 % of the table) are excluded from that protocol — 16,259 of 19,088 pairs enter it, and
the cutoff is 2021-01-01. That is a documented substitute for DAVIS's `deposited` date, not
the same quantity. (iii) *Row sampling*: the download cap forces the fetcher to sample activity
pages, and three of the sixteen targets are still incompletely covered
(`complete: false` in the manifest). The panel is also a **single chemical family** (human
kinases), so §9 tests whether the protocol effect is a property of *affinity prediction* or
only of *kinases*; the latter is not excluded by this table.

Next steps that would actually change the conclusions: a censored likelihood for DAVIS; a
held-out fourth split so the conformal guarantee is real; ≥3 structures per target with
side-chain ensembles, plus a second chemistry set (pKi, or METLI-style multi-target data)
so the physics arm is not kinases-only; and pocket-level target features (residue identity
in the box) so the two arms see comparable information.
