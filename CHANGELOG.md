# Changelog

All notable changes to this benchmark are documented here. Dates are git commit
dates and map 1:1 to `git log`.

## [Unreleased]

- Planned: Tobit/censored-likelihood baseline (DAVIS/KIBA label censoring handled
  explicitly instead of dropped) — CPU only
- Planned: external-baseline table with published numbers (Tafei, DeepDTA, and the
  Davis/KIBA re-benchmark papers), each row with source and protocol-caveat
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
