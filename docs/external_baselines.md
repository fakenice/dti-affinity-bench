# External baselines (published numbers, each re-checked against the source)

Verified 2026-11-18 against full texts (Europe PMC fullTextXML, PMC author
manuscripts, PubMed eutils, Crossref, bioRxiv PDFs). Every number maps to a
citation and a table; rows that are not comparable with this benchmark are
flagged in §5 instead of being silently averaged together.

## 1. This benchmark's own anchors

Headline is protocol sensitivity, not leaderboard position (README §Headline).
The only cells that mean anything next to the published literature are the
**random-split** ones:

- DAVIS random: best RMSE 0.5697 (XGBoost); `ci_harrell` 0.8738 vs
  `ci_sklearn_like` 0.6919 on the same run — the 0.18 gap is a caliber choice,
  not performance (§4).
- Cross-protocol spread of the best model: 0.4823 pKd units on DAVIS
  (random → cold_target), η²(protocol) 0.9134 vs η²(model) 6.8%.
- Random-split leakage measured: 34.98% (→10.98% de-duplicated); cold_drug
  48.79% of test compounds reappear in training.

## 2. Published baselines on the same data (all random splits)

| Paper (source) | Split | DAVIS CI / MSE | KIBA CI / MSE | Notes |
|---|---|---|---|---|
| DeepDTA — Öztürk et al., *Bioinformatics* 2018;34(17):i821–i829, doi:10.1093/bioinformatics/bty593, Tables 3–4 (full text verified) | six equal random parts, one held out; 5-fold CV on the rest for hyper-parameters | CNN\|CNN 0.878 (std 0.004) / 0.261 | 0.863 (0.002) / 0.194 | arXiv:1801.10193v2 tables match the journal version (checked side by side); SimBoost 0.872/0.282 and KronRLS 0.871/0.379 are **as reported in DeepDTA's Table 3** — original texts not accessed |
| GraphDTA — Nguyen et al., preprint bioRxiv doi:10.1101/684662 **v3** (2019-07-02, PDF verified), Tables 1–2 | 80/20 random, "same train/test splits with DeepDTA and WideDTA" (§4.1 verbatim) | GAT 0.892/0.232, GIN 0.893/0.229, GCN 0.880/0.254 | GCN 0.889/0.139, GAT_GCN 0.891/0.139 | journal version (*Bioinformatics* 2021;37(8):1140–1147, doi:10.1093/bioinformatics/btaa921) exists (metadata verified) but its table numbers were **not** accessible (OUP 403) — quote the preprint with its version, do not claim journal numbers |
| MGraphDTA — Yang et al., *Chemical Science* 2022;13:816–833, doi:10.1039/D1SC05180F, Tables 1, 3, 5 (full text verified) | "same training and test set as DeepAffinity and GraphDTA" for Tables 3–4 | 0.900 (0.004) / 0.207 (0.001) | 0.902 (0.001) / 0.128 (0.001) | also the only verified source for the **filtered-Davis** comparison (see §3, §5) |

External random-split CI clusters at 0.87–0.90 with MSE 0.207–0.282
(RMSE ≈ 0.455–0.531 in pKd units); this repo's random-best RMSE 0.5697 is
slightly worse than that range. That gap is *not* a failure to reproduce:
external folds keep the ~69.6% pKd-5.0 censored rows, report no bootstrap
intervals, and use fold std of 5 training splits as their only uncertainty.
"Replication" cannot be claimed or denied at this caliber (§5).

## 3. Published evidence that the split protocol moves the number (external support for the headline)

MGraphDTA §3.3, verbatim (full text, 2026-11-18):

> "However, it is noteworthy that the random split setting can lead to
> over-optimistic results because it results in information leakage (e.g., drug
> or target information) to the test set."

Same model, same code, only the protocol changed (MGraphDTA on DAVIS,
MSE): random **0.207** → orphan–drug split **0.572 ± 0.088** → cluster-based
**0.654 ± 0.207** — a 0.35–0.45 MSE swing, while the spread across published
models under random split is ≈ 0.05. The paper also states the graph methods'
random-split superiority "was less obvious in the orphan-based and
cluster-based split settings". This is the published counterpart of this
repo's variance decomposition (η² protocol 0.91 ≫ η² model 0.07); the two
agree in direction, use different statistics (paired MSE deltas vs variance
share), and must be reported as agreement-in-direction, not identical numbers.

Censoring is the third axis: DeepDTA scores 0.878 CI on the full (censored)
Davis table but 0.653 CI / RMSE 0.931 on filtered Davis — as reported in
MGraphDTA's Table 5 (MGraphDTA itself: 0.740 CI there). Label caliber, not the
model, moved that 0.22.

## 4. Concordance-index caliber and its sources

The CI formula used by DeepDTA (verified verbatim, §3.2): comparable pairs are
those with **unequal labels** (δi > δj); tied predictions score 0.5 — this
matches this repo's `ci_harrell` caliber; the repo additionally reports
`ci_sklearn_like` (all pairs, both ties = 0.5), which on DAVIS differs by up
to 0.18 because 36.4–48.7% of label pairs are tied. Source chain, as cited by
the papers themselves: the formula is attributed to Gönen & Heller 2005,
*Biometrika* 92(4):965–970, doi:10.1093/biomet/92.4.965 (DOI verified via
Crossref; definition text not independently accessed), with the step-function
presentation following Pahikkala et al. 2014, *Brief Bioinform*,
doi:10.1093/bib/bbu010 (as it appears in DeepDTA's reference list). Harrell
1982, *JAMA* 247(17):2543–6 (PMID 7069920, verified via PubMed eutils), is
citable as historical origin only — its verbatim definition was not accessed
(JAMA paywall), so do not describe anything as "Harrell's definition as
printed in 1982".

## 5. Caliber differences (read before citing §2)

1. **Protocols**: external numbers are all random splits and benchmark only
   this repo's `*/random/*` cells; this repo's headline (protocol spread
   0.4823) has no same-caliber external counterpart — MGraphDTA §3.3 is the
   nearest published support.
2. **Error units**: external papers report MSE (pKd²); this repo reports RMSE
   (pKd). √0.261 ≈ 0.511; do not compare 0.261 and 0.5697 directly.
3. **Censoring**: external folds retain the pKd = 5.0 plateau (~69.6% of
   DAVIS); MGraphDTA Table 5 shows the same models lose 0.15–0.23 CI when
   censored rows are dropped. Any cross-paper CI comparison must state which
   Davis table it uses ("Davis" 379 sequences / 442 target names vs "filtered
   davis" 9,125 rows — README §1 distinction, mirrored by MGraphDTA Table 1).
4. **Uncertainty kind**: fold std (external) vs bootstrap CI + Holm (this
   repo). Fold std of 5 splits is not a 95% interval.
5. **Tie handling in CI**: see §4; `ci_sklearn_like` numbers are not
   comparable to any published CI.
6. **GraphDTA version**: preprint v3 numbers only; journal numbers unverified.

## 6. What this supports — and what it does not

Supports: the random-split cell being in-family with published numbers
(0.87–0.90 CI); the headline direction (protocol ≫ model) having independent
published evidence (§3); the tie/censoring caliber effects being real and
documented elsewhere.

Does not support: "we reproduced DeepDTA/MGraphDTA" (folds, censoring and
uncertainty calibers differ); "our random number is worse, therefore the
implementation is wrong" (same reason, opposite direction); any leaderboard
comparison against §2 rows.

## 7. Citation register (all accessed 2026-11-18)

Citable: DeepDTA doi:10.1093/bioinformatics/bty593 (Tables 3–4, split and CI
formula text); GraphDTA bioRxiv doi:10.1101/684662v3 (Tables 1–2, §4.1; cite
version); MGraphDTA doi:10.1039/D1SC05180F (Tables 1/3/5, §3.3 text);
Gönen & Heller doi:10.1093/biomet/92.4.965 (title-level); Harrell 1982
PMID 7069920 (title-level, historical origin only); Davis 2011
doi:10.1038/nbt.1990 and KIBA 2014 doi:10.1021/ci400709d (data provenance).

Not citable: "Merino et al. 2022 Nature Communications pitfalls paper" —
searched six routes, no such record (the PMID given in the task brief leads to
an unrelated sperm-evolution paper); OnDTA / TransDTA — no publication record
found; SimBoost/KronRLS/WideDTA original texts (numbers available only as
"as reported in DeepDTA/GraphDTA tables"); GraphDTA journal-table numbers;
Harrell verbatim definition; and the mid-research fabricated identifiers
(PMIDs 39526122 / 36357384 / 39789094, DOI 10.1038/s42256-024-00908-x) that
appeared in an injected message and were each checked and rejected.
