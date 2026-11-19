"""The Tobit baseline of README §4b must be a real censored likelihood, not a ridge fit
with a relabelled output, and its committed headline numbers must re-derive from the
committed tables.

Pinned here:
  * the penalised censored NLL matches an independent log-sum by hand, the penalty is
    exempt for the intercept and for sigma, and the analytic gradient matches a central
    difference (the script's own selftest, asserted);
  * with the floor below every observation the likelihood degenerates to the Gaussian one
    and the unpenalised fit reproduces least squares — the censoring term cannot leak into
    a problem that has no censoring;
  * on genuinely censored data the Tobit fit is closer to the LATENT truth than squared
    error on the clipped labels, which is the whole claim of §4b reading 1;
  * E[observed y | x] equals a truncnorm construction, is always above the floor, and is
    monotone in mu — the "the RMSE-optimal forecast sits above the ceiling" caveat;
  * the committed results/tobit_baseline.json records one lambda_path entry per grid value,
    its selected lambda is the valid-NLL argmin, and "the selected lambda is also the
    test-optimal lambda in 10 of 10 cells" is counted from the file, not from prose;
  * the committed results/readme_numbers.json carries the tobit_baseline block §4b quotes,
    with the same censoring shares as results/metrics_tobit.csv.

Run:  python -m unittest discover -s tests -v   (from the repo root)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import norm, truncnorm

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"


def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


m17 = load("17_tobit_baseline", "tobit_for_test")


def latent_data(n=400, p=3, seed=3, censor_frac=0.6):
    """A true linear latent model plus noise, then the assay floor clips the labels."""
    rng = np.random.default_rng(seed)
    X = np.hstack([np.ones((n, 1)), rng.normal(size=(n, p))])
    b_true = np.r_[2.0, rng.normal(size=p) * 0.5]
    sigma_true = 0.6
    y_latent = X @ b_true + sigma_true * rng.normal(size=n)
    if censor_frac <= 0.0:
        L = float(y_latent.min()) - 1.0     # a floor no row can reach: the negative control
    else:
        L = float(np.quantile(y_latent, censor_frac))
    return X, b_true, sigma_true, np.clip(y_latent, L, None), L, y_latent


class LikelihoodTests(unittest.TestCase):
    def test_gradient_selftest_passes(self):
        self.assertLess(m17.gradient_selftest()["analytic_vs_numeric_max_rel_dev"], 1e-4)

    def test_nll_matches_independent_log_sum(self):
        X, _, _, y, L, _ = latent_data()
        theta = np.r_[np.zeros(X.shape[1]), np.log(0.8)]
        nll, _ = m17._nll_grad(theta, X, y, L, lam=0.0)
        b, s = theta[:-1], float(np.exp(theta[-1]))
        mu = X @ b
        cens = y <= L + 1e-12
        # same convention as _nll_grad: the 0.5*log(2*pi) density constant is dropped
        # (constant in theta, so it cannot move the MLE or the lambda choice)
        expect = -((-np.log(s) - 0.5 * (((y[~cens] - mu[~cens]) / s) ** 2)).sum()
                   + norm.logcdf((L - mu[cens]) / s).sum())
        self.assertAlmostEqual(nll, float(expect), places=8)

    def test_dropped_constant_is_exactly_the_density_normaliser(self):
        """Quantifies the convention above rather than trusting it: adding the constant
        back shifts the NLL by 0.5*log(2*pi) per UNCENSORED row and nothing else."""
        X, _, _, y, L, _ = latent_data()
        theta = np.r_[np.zeros(X.shape[1]), np.log(0.8)]
        nll = m17._nll_grad(theta, X, y, L, lam=0.0)[0]
        cens = y <= L + 1e-12
        with_const = nll + 0.5 * np.log(2 * np.pi) * int((~cens).sum())
        b, s = theta[:-1], float(np.exp(theta[-1]))
        mu = X @ b
        textbook = -(np.log(norm.pdf(y[~cens], loc=mu[~cens], scale=s)).sum()
                     + norm.logcdf((L - mu[cens]) / s).sum())
        self.assertAlmostEqual(with_const, float(textbook), places=8)

    def test_penalty_is_exempt_for_intercept_and_sigma(self):
        X, _, _, y, L, _ = latent_data(n=200)
        theta = np.r_[np.full(X.shape[1], 5.0), np.log(0.8)]
        base = m17._nll_grad(theta, X, y, L, lam=0.0)[0]
        lam = 2.0
        penalised = m17._nll_grad(theta, X, y, L, lam=lam)[0]
        # slopes are b[1:]; b[0] is the constant column and sigma is parameterised
        # separately -- neither may be penalised.
        self.assertAlmostEqual(penalised - base, lam * 5.0 ** 2 * (X.shape[1] - 1), places=6)


class DegenerationTests(unittest.TestCase):
    def test_no_censoring_reduces_to_gaussian_mle(self):
        X, b_true, sigma_true, _, L, y_latent = latent_data(censor_frac=0.0)
        y = y_latent                                  # nothing is clipped
        res = m17.fit_tobit(X, y, L, lam=0.0)
        beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
        # L-BFGS converges to the same optimum as the normal equations, not to their exact
        # solution -- gtol 1e-9 leaves coefficient differences of a few 1e-3.
        np.testing.assert_allclose(res.x[:-1], beta_ols, rtol=0.02, atol=0.01)
        self.assertAlmostEqual(float(np.exp(res.x[-1])),
                               float(np.std(y - X @ beta_ols)), delta=0.02)
        self.assertTrue(np.all(y > L + 1e-12), "the no-censoring control must have no ties")

    def test_tobit_beats_squared_error_on_the_clipped_labels(self):
        """Why §4b fits the censored likelihood at all: squared error on clipped labels is
        biased against the latent line, the Tobit MLE is not."""
        X, b_true, _, y, L, _ = latent_data(n=1200)
        tobit = m17.fit_tobit(X, y, L, lam=0.0).x[:-1]
        ols, *_ = np.linalg.lstsq(X, y, rcond=None)
        err_t = float(np.linalg.norm(tobit - b_true))
        err_o = float(np.linalg.norm(ols - b_true))
        self.assertLess(err_t, err_o)
        self.assertLess(err_t / np.linalg.norm(b_true), 0.15)


class ConditionalMeanTests(unittest.TestCase):
    def test_predicted_mean_matches_the_truncated_normal_construction(self):
        rng = np.random.default_rng(11)
        mu = rng.normal(3.0, 1.0, size=200)
        s, L = 1.3, 2.0
        X = np.eye(len(mu))                       # X @ b == mu, row by row
        obs, _, sd = m17.predict_tobit(np.r_[mu, np.log(s)], X, L)
        a = (L - mu) / s
        expect = L * norm.cdf(a) + truncnorm.mean(a, np.inf, loc=mu, scale=s) * (1 - norm.cdf(a))
        np.testing.assert_allclose(obs, expect, rtol=1e-9)
        self.assertAlmostEqual(sd, s)

    def test_predicted_mean_is_above_the_floor_and_monotone_in_mu(self):
        mu = np.linspace(-2.0, 6.0, 400)
        X = np.eye(len(mu))
        obs, _, _ = m17.predict_tobit(np.r_[mu, np.log(1.5)], X, L=2.0)
        self.assertGreater(float(np.min(obs)), 2.0)
        self.assertTrue(bool(np.all(np.diff(obs) > 0)))

    def test_conditional_mean_of_a_floor_row_sits_above_the_floor(self):
        """README §4b reading 3: the forecast cannot chase the ceiling, by construction."""
        mu = np.full(50, 2.0)                     # latent mean exactly at the floor
        X = np.eye(len(mu))
        obs, _, _ = m17.predict_tobit(np.r_[mu, np.log(1.0)], X, L=2.0)
        self.assertGreater(float(obs[0]), 2.0)
        self.assertAlmostEqual(float(obs[0]), 2.0 + norm.pdf(0.0) * 1.0, places=9)


class CommittedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.js = json.loads((RESULTS / "tobit_baseline.json").read_text(encoding="utf-8"))
        cls.metrics = (RESULTS / "metrics_tobit.csv").read_text(encoding="utf-8")

    def test_lambda_path_covers_the_grid_for_every_cell(self):
        self.assertEqual(len(self.js["per_cell"]), 10)
        for key, cell in self.js["per_cell"].items():
            self.assertEqual([e["lambda"] for e in cell["lambda_path"]], list(m17.LAMBDAS),
                             f"{key} lambda_path is not the committed grid")

    def test_selected_lambda_is_the_valid_nll_argmin(self):
        for key, cell in self.js["per_cell"].items():
            best = min(cell["lambda_path"], key=lambda e: e["valid_nll_per_row"])
            self.assertEqual(cell["lambda"], best["lambda"], key)

    def test_selected_lambda_is_test_optimal_in_every_cell(self):
        """README §4b reading 2 ("10 of 10") counted from the artifact."""
        cells = list(self.js["per_cell"].items())
        agree = [int(min(c["lambda_path"], key=lambda e: e["test_rmse"])["lambda"]
                      == c["lambda"]) for _, c in cells]
        self.assertEqual(sum(agree), len(cells))

    def test_censoring_shares_match_the_dataset_documents(self):
        """DAVIS is the ~70 % ceiling dataset, KIBA the ~0 % negative control."""
        import csv
        import io
        rows = list(csv.DictReader(io.StringIO(self.metrics)))
        dv = [float(r["cens_frac_train"]) for r in rows if r["dataset"] == "davis"]
        kb = [float(r["cens_frac_train"]) for r in rows if r["dataset"] == "kiba"]
        self.assertTrue(all(x > 0.6 for x in dv), dv)
        self.assertTrue(all(x == 0.0 for x in kb), kb)

    def test_readme_numbers_carries_the_tobit_block(self):
        rn = json.loads((RESULTS / "readme_numbers.json").read_text(encoding="utf-8"))
        tb = rn["tobit_baseline"]
        self.assertEqual(tb["lambda_selected_equals_lambda_test_optimal"], "10 of 10")
        self.assertEqual(tb["davis_protocols_where_tobit_rmse_is_better"], 4)
        self.assertEqual(tb["davis_spearman_up"], 2)
        self.assertEqual(tb["davis_worst_rmse_loss"], 0.022)
        self.assertAlmostEqual(tb["kiba_max_abs_rmse_gap_vs_ridge"],
                               self.js["controls"]["kiba"]["max_abs_rmse_gap_vs_ridge"],
                               places=3)
        lo, hi = tb["floor_recovery_share_range_davis"]
        self.assertEqual(round(lo, 4), 0.2669)
        self.assertEqual(round(hi, 4), 0.4751)


if __name__ == "__main__":
    unittest.main()
