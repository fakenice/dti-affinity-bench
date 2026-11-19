# Changelog

All notable changes to this benchmark are documented here. Dates are git commit
dates and map 1:1 to `git log`.

## [Unreleased]

- 2026-11-19 Delivered the Tobit/censored-likelihood baseline that was listed here as
  planned — the "first next step" README §10 had named since v1.0 and only asserted.
  `scripts/17_tobit_baseline.py` fits a one-sided left Tobit by MLE (analytic gradient
  self-checked against a finite difference, 3.0e-10) on the same features, standardise
  guard, split files and `evaluate()` as `03_models.py`, with λ chosen on the valid
  block's censored NLL over 03's own alpha grid, and the point forecast is E[observed
  y | x]. New README §4b records the verdict, which is negative: DAVIS RMSE improves in
  4 of 5 protocols by at most 0.0226 pKd, gets worse on `cold_drug` (+0.0220), Spearman
  splits 2-up/3-down, KIBA (≈0 % censored) stays within 0.0120 RMSE of the ridge row as
  the negative control, and only 26.7–47.5 % of test rows observed at the floor get a
  forecast within 0.1 of it. λ sensitivity is measured rather than claimed: in 10 of 10
  cells the NLL-selected λ is also the test-optimal λ (selection regret ≤0.0051 % of
  RMSE), so the row is not a hyper-parameter artefact. Adds
  `results/metrics_tobit.csv`, `results/tobit_baseline.json`, `results/run_17_tobit.log`
  and per-block prediction npz files (gitignored, like the other 03-era dumps),
  a `run_all.py --stage tobit` entry (11.2 min measured), §10 rewritten from
  "nothing here fixes that" to the measured result, §7 layout notes that the generated
  leaderboard deliberately does not rank the Tobit row, and `07_report.py` now recomputes
  every §4b figure into `results/readme_numbers.json` under a `tobit_baseline` key (deltas
  vs the ridge rows, the 10-of-10 λ agreement, the floor-recovery range) so the prose
  cannot drift from the table. New `tests/test_tobit_baseline.py` (14 tests, suite 39 OK)
  pins the NLL against a hand-computed log-sum, the dropped density constant, the penalty
  exemptions for intercept and sigma, the degeneration to least squares when nothing is
  censored, the truncnorm construction (above-floor, monotone-in-μ) of the reported
  forecast, and re-counts the committed §4b numbers from the artifacts.
- 2026-11-18 Added `docs/external_baselines.md`: source-verified published
  random-split baselines (DeepDTA/GraphDTA-preprint/MGraphDTA tables), external
  protocol-sensitivity evidence (MGraphDTA 0.207→0.572→0.654 MSE), CI-caliber
  source chain, and a do-not-cite register (unverifiable leads + injected fake
  identifiers); README gained section 11 (later sections renumbered)
- 2026-11-18 Added `docs/DECISIONS.md`: caliber/protocol decisions logged only where
  the repo itself carries written evidence (commit dates cited)

## [v1.0] - 2026-11-18

Release freeze: versioned, cited, and DOI-ready.

### Added

- `20b9e81` generated leaderboard (`scripts/16_leaderboard.py`, byte-stable for a
  given `results/metrics.csv`), Benchmark-study block and AE-style checklist in the
  README, `CITATION.cff`
- `7ea8987` `.zenodo.json` release metadata

## 2026-11-07

- `e330859` headline re-checked on a fourth table: 16 BindingDB kinases, same code

## 2026-11-04

- `cce2252` headline re-checked on a third table: 16 ChEMBL kinases, same code, same protocols
- `4d237c7` smoke runs kept out of the committed external table and grid
- `72d9f32` interval on every grid claim; significance split by direction

## 2026-11-03

- `f456664` docking and protocol claims tested with intervals instead of point estimates

## 2026-10-14

- `c3601d6` initial benchmark: DAVIS/KIBA affinity, 5 split protocols x 4 models x
  seeds; headline finding that the split protocol moves the number more than the
  model does
