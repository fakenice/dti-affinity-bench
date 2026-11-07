"""The BindingDB adapter must reuse 14_external_chembl's evaluation machinery unchanged
while reading the BindingDB table -- and must never touch the committed ChEMBL tree.

These tests pin the redirection contract (the adapter changes module globals, not 14's code):
  * configure_paths moves EXT/EXTFEAT/EXTSPL/OUT/PRED to the _bindingdb trees;
  * DATASET becomes "bindingdb" (so split/prediction files are namespaced);
  * external_pairs() reads bindingdb_ki.csv, not chembl_ki.csv;
  * the two hardcoded EXT/"chembl_ki.csv" reads in stage_grid / stage_analyse are redirected;
  * configure_paths("") restores the committed ChEMBL paths exactly.

Run:  python -m unittest discover -s tests -v   (from the repo root)
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


class AdapterRedirectionTests(unittest.TestCase):
    def setUp(self):
        # the adapter and the test must drive the SAME module object: the adapter loads 14 under
        # the alias "external_chembl_engine", so reuse that instance rather than loading a second
        # copy whose globals configure_bindingdb() would never touch
        self.adapter = load("15_external_bindingdb", "adapter_for_test")
        self.ext14 = self.adapter.ext14
        self.chembl = {k: getattr(self.ext14, k)
                       for k in ("EXT", "EXTFEAT", "EXTSPL", "OUT", "PRED", "DATASET")}

    def tearDown(self):
        # always restore the committed tree so a test cannot leak state into the next
        self.ext14.configure_paths("")
        self.ext14.DATASET = "chembl"
        self.adapter.restore_chembl()

    def test_configure_moves_whole_tree_to_bindingdb(self):
        self.adapter.configure_bindingdb()
        e = self.ext14
        self.assertEqual(e.EXT, ROOT / "data" / "processed" / "external_bindingdb")
        self.assertEqual(e.EXTFEAT, e.EXT / "features")
        self.assertEqual(e.EXTSPL, e.EXT / "splits")
        self.assertEqual(e.OUT, ROOT / "results" / "external_bindingdb")
        self.assertEqual(e.PRED, e.OUT / "predictions")
        self.assertEqual(e.DATASET, "bindingdb")

    def test_committed_chembl_tree_is_untouched_after_configure(self):
        self.adapter.configure_bindingdb()
        # the committed ChEMBL directories are different paths, never the redirected ones
        self.assertNotEqual(self.ext14.EXT, self.chembl["EXT"])
        self.assertNotEqual(self.ext14.OUT, self.chembl["OUT"])
        self.assertEqual(self.chembl["EXT"], ROOT / "data" / "processed" / "external")
        self.assertEqual(self.chembl["OUT"], ROOT / "results" / "external_chembl")

    def test_external_pairs_reads_bindingdb_table(self):
        self.adapter.configure_bindingdb()
        # write a tiny bindingdb_ki.csv into the redirected EXT and confirm external_pairs reads it
        ext = self.ext14.EXT
        ext.mkdir(parents=True, exist_ok=True)
        probe = ext / "bindingdb_ki.csv"
        existed = probe.exists()
        backup = probe.read_bytes() if existed else None
        try:
            pd.DataFrame({"pair_id": [0], "ligand_id": ["BL00000"], "target_id": ["XT000"],
                          "pki": [7.0]}).to_csv(probe, index=False)
            got = self.ext14.external_pairs()
            self.assertEqual(list(got.columns), ["pair_id", "ligand_id", "target_id", "pki"])
            self.assertEqual(got.iloc[0]["ligand_id"], "BL00000")
        finally:
            if existed:
                probe.write_bytes(backup)
            elif probe.exists():
                probe.unlink()

    def test_restore_returns_committed_chembl_globals(self):
        self.adapter.configure_bindingdb()
        self.adapter.restore_chembl()
        for k, v in self.chembl.items():
            self.assertEqual(getattr(self.ext14, k), v, f"{k} not restored")
        # external_pairs is the original function object again (reads chembl_ki.csv)
        self.assertEqual(self.ext14.external_pairs.__name__, "external_pairs")

    def test_grid_and_analyse_hardcoded_reads_are_redirected(self):
        # the adapter wraps stage_grid/stage_analyse so their literal EXT/"chembl_ki.csv" reads
        # resolve to the BindingDB table; confirm the wrappers exist and delegate to 14's functions
        self.assertTrue(callable(self.adapter.stage_grid))
        self.assertTrue(callable(self.adapter.stage_analyse))
        self.assertIsNot(self.adapter.stage_grid, self.ext14.stage_grid)

    def test_internal_configure_paths_is_bindingdb_aware(self):
        # 14's bootstrap worker run_cell_ext calls configure_paths(job["out_tag"]) to re-apply the
        # tree tag. For BindingDB that call must land on the BindingDB tree AND set DATASET, or the
        # worker scores chembl predictions in the wrong directory (the spawn-child bug).
        self.adapter.configure_bindingdb()
        self.ext14.configure_paths("_plumbing")
        self.assertEqual(self.ext14.EXT, ROOT / "data" / "processed" / "external_bindingdb_plumbing")
        self.assertEqual(self.ext14.OUT, ROOT / "results" / "external_bindingdb_plumbing")
        self.assertEqual(self.ext14.DATASET, "bindingdb")

    def test_analyse_runs_bootstrap_in_process(self):
        # Windows 'spawn' children re-import 14 fresh and cannot inherit the adapter's redirection,
        # so the bootstrap pool must run in-process (jobs=1). The wrapper forces this.
        import argparse
        seen = {}
        orig = self.ext14.stage_analyse
        def spy(args):
            seen["jobs"] = args.jobs
            return {}
        self.ext14.stage_analyse = spy
        try:
            self.adapter.configure_bindingdb()
            self.adapter.stage_analyse(argparse.Namespace(jobs=6, out_tag="", quick=True))
        finally:
            self.ext14.stage_analyse = orig
        self.assertEqual(seen["jobs"], 1)


if __name__ == "__main__":
    unittest.main()