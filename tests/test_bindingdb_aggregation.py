"""Aggregation logic for the BindingDB external table (scripts/15_fetch_bindingdb.py).

These tests pin the parts that decide whether the fourth table is honest:
  * only exact Ki (nM) survives -- censored '<'/'>' bounds and IC50/Kd/EC50 are dropped+counted;
  * one ligand is one RDKit InChIKey, so a duplicated structure cannot sit in train and test;
  * a pair label is the median Ki, pKi = 9 - log10(median Ki / nM);
  * a row is attributed to the target it actually names (query == UniProt name), because a
    single-UniProt BindingDB response also carries selectivity rows for *other* proteins;
  * temporal years come from PubMed pubdate, and an undated pair stays undated (excluded by the
    temporal arm downstream) rather than being given an invented date.

Run:  python -m unittest discover -s tests -v   (from the repo root)
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_fetch():
    spec = importlib.util.spec_from_file_location("fetch_bindingdb", SCRIPTS / "15_fetch_bindingdb.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_bindingdb"] = mod
    spec.loader.exec_module(mod)
    return mod


fb = load_fetch()


def row(smi="CCO", ki="100", typ="Ki", query="Tyrosine-protein kinase JAK2",
        pmid="", doi="", mon="1"):
    return {"smile": smi, "affinity": ki, "affinity_type": typ, "query": query,
            "pmid": pmid, "doi": doi, "monomerid": mon}


class KiFilterTests(unittest.TestCase):
    def test_exact_numeric_ki_kept(self):
        keep, drop = fb.filter_ki_rows([row(ki="100"), row(ki="0.55")])
        self.assertEqual(len(keep), 2)
        self.assertAlmostEqual(keep[0]["ki_nM"], 100.0)
        self.assertAlmostEqual(keep[1]["ki_nM"], 0.55)

    def test_censored_bounds_dropped_and_counted(self):
        keep, drop = fb.filter_ki_rows([row(ki="<0.1"), row(ki=">10000"), row(ki="~50")])
        self.assertEqual(len(keep), 0)
        self.assertEqual(drop["censored_relation"], 3)

    def test_non_ki_types_dropped(self):
        keep, drop = fb.filter_ki_rows([row(typ="IC50"), row(typ="Kd"), row(typ="EC50")])
        self.assertEqual(len(keep), 0)
        self.assertEqual(drop["wrong_affinity_type"], 3)

    def test_non_positive_or_non_numeric_dropped(self):
        keep, drop = fb.filter_ki_rows([row(ki="0"), row(ki="-5"), row(ki="abc"), row(ki="")])
        self.assertEqual(len(keep), 0)
        self.assertEqual(drop["non_numeric"], 2)
        self.assertEqual(drop["non_positive_or_nan"], 2)

    def test_missing_smiles_dropped(self):
        keep, drop = fb.filter_ki_rows([row(smi=""), row(smi="   ")])
        self.assertEqual(len(keep), 0)
        self.assertEqual(drop["no_smiles"], 2)


class TargetAttributionTests(unittest.TestCase):
    def test_only_rows_naming_the_queried_target_are_kept(self):
        name = "Tyrosine-protein kinase JAK2"
        rows = [row(query=name, ki="10"),
                row(query="Non-receptor tyrosine-protein kinase TYK2", ki="20"),
                row(query="Tyrosine-protein kinase JAK1", ki="30")]
        keep, drop = fb.filter_ki_rows(rows, target_name=name)
        self.assertEqual(len(keep), 1)
        self.assertAlmostEqual(keep[0]["ki_nM"], 10.0)
        self.assertEqual(drop["off_target_query"], 2)

    def test_without_target_name_no_query_filter(self):
        rows = [row(query="A", ki="10"), row(query="B", ki="20")]
        keep, drop = fb.filter_ki_rows(rows)
        self.assertEqual(len(keep), 2)
        self.assertEqual(drop.get("off_target_query", 0), 0)


class InchiDedupTests(unittest.TestCase):
    def test_same_structure_different_strings_share_one_key(self):
        # ethanol written two ways -> one InChIKey
        k1 = fb.inchikey("CCO")
        k2 = fb.inchikey("OCC")
        self.assertIsNotNone(k1)
        self.assertEqual(k1, k2)

    def test_distinct_structures_get_distinct_keys(self):
        self.assertNotEqual(fb.inchikey("CCO"), fb.inchikey("c1ccccc1"))

    def test_unparseable_smiles_returns_none(self):
        self.assertIsNone(fb.inchikey("this is not a molecule ((("))

    def test_build_ligand_index_groups_by_key_and_counts_monomers(self):
        recs = [{"smiles": "CCO", "inchikey": "K1", "monomerid": "1"},
                {"smiles": "OCC", "inchikey": "K1", "monomerid": "2"},
                {"smiles": "c1ccccc1", "inchikey": "K2", "monomerid": "3"}]
        index, ligands = fb.build_ligand_index(recs)
        self.assertEqual(len(ligands), 2)
        self.assertEqual(index["K1"], index["K1"])
        self.assertNotEqual(index["K1"], index["K2"])
        by_key = {l["inchikey"]: l for l in ligands}
        self.assertEqual(by_key["K1"]["n_monomer_ids"], 2)
        self.assertEqual(by_key["K2"]["n_monomer_ids"], 1)


class AggregatePairsTests(unittest.TestCase):
    def test_median_ki_and_pki_definition(self):
        recs = [{"inchikey": "K1", "ki_nM": 10.0, "pmid": "111", "doi": "d1", "monomerid": "1"},
                {"inchikey": "K1", "ki_nM": 100.0, "pmid": "111", "doi": "d1", "monomerid": "1"},
                {"inchikey": "K1", "ki_nM": 1000.0, "pmid": "222", "doi": "d2", "monomerid": "2"}]
        index = {"K1": "BL00000"}
        pairs = fb.aggregate_pairs(recs, index, "XT000", {"111": 2006, "222": 2013})
        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertAlmostEqual(p["ki_nM"], 100.0)
        self.assertAlmostEqual(p["pki"], 9.0 - np.log10(100.0), places=5)
        self.assertEqual(p["n_rows"], 3)
        self.assertEqual(p["n_monomer_ids"], 2)
        # n_docs counts distinct source documents: 2 pmids, each with its own doi -> 2
        self.assertEqual(p["n_docs"], 2)
        self.assertEqual(p["first_year"], 2006)
        self.assertEqual(p["last_year"], 2013)
        self.assertEqual(p["target_id"], "XT000")
        self.assertEqual(p["ligand_id"], "BL00000")

    def test_undated_pair_has_no_year_not_an_invented_one(self):
        recs = [{"inchikey": "K1", "ki_nM": 50.0, "pmid": "", "doi": "10.7270/x", "monomerid": "1"}]
        pairs = fb.aggregate_pairs(recs, {"K1": "BL00000"}, "XT000", {})
        self.assertIsNone(pairs[0]["first_year"])
        self.assertIsNone(pairs[0]["last_year"])

    def test_single_row_iqr_is_zero(self):
        recs = [{"inchikey": "K1", "ki_nM": 50.0, "pmid": "1", "doi": "d", "monomerid": "1"}]
        pairs = fb.aggregate_pairs(recs, {"K1": "BL00000"}, "XT000", {"1": 2010})
        self.assertEqual(pairs[0]["ki_iqr_nM"], 0.0)
        self.assertEqual(pairs[0]["n_rows"], 1)


class YearTests(unittest.TestCase):
    def test_pubdate_year_parsed(self):
        self.assertEqual(fb.year_from_pubdate("2006 Feb 28"), 2006)
        self.assertEqual(fb.year_from_pubdate("2013 Nov"), 2013)
        self.assertEqual(fb.year_from_pubdate("1998"), 1998)

    def test_unparseable_pubdate_is_none(self):
        self.assertIsNone(fb.year_from_pubdate(""))
        self.assertIsNone(fb.year_from_pubdate("in press"))
        self.assertIsNone(fb.year_from_pubdate(None))

    def test_esummary_result_to_year_map(self):
        result = {"uids": ["111", "222"],
                  "111": {"pubdate": "2006 Feb 28", "title": "x"},
                  "222": {"pubdate": "2013 Nov 1", "title": "y"}}
        years = fb.years_from_esummary({"result": result})
        self.assertEqual(years, {"111": 2006, "222": 2013})

    def test_missing_pubdate_yields_no_entry(self):
        result = {"uids": ["111"], "111": {"title": "no date"}}
        years = fb.years_from_esummary({"result": result})
        self.assertEqual(years, {})


if __name__ == "__main__":
    unittest.main()