#!/usr/bin/env python3
"""
15_fetch_bindingdb.py -- a *fourth*, independently curated affinity table, for external
validation of the headline claim ("the split protocol moves the number more than the model
does").

Why BindingDB after all: an earlier note in 13_fetch_chembl.py claimed BindingDB's REST
endpoint "requires an emailed API key". That is FALSE and was disproven against the live
service on 2026-11-07 (raw evidence under results/bindingdb_probe/): the official RESTful
API at bindingdb.org/rest is open and needs no key. getLigandsByUniprots?uniprot=...&cutoff=
...&response=application/json returns JSON directly; a single O60674 query at cutoff=100000
returned 25,079 rows (7.3 MB). This script pulls the same 16 human kinases the ChEMBL table
uses (data/processed/external/targets.csv), so the two external tables share a target panel
but differ in curation source, label aggregation and publication history.

Two measured properties of the response shape drive the design (both verified on the JAK2
probe, results/bindingdb_probe/jak2.json):

  1. A single-UniProt response is NOT pure: querying O60674 also returns selectivity rows
     whose `query` names TYK2, JAK1 and PDGFR-beta. The response carries no per-row UniProt,
     only the protein *name* in `query`. So each target is fetched on its own and a row is
     attributed to that target only when `query` equals the target's UniProt name (the same
     name carried in targets.csv). Off-name rows are dropped and counted, never silently
     mixed into the queried target's label.

  2. `affinity` is a string and is frequently a censored bound: of the 4,126 JAK2 Ki rows in the
     committed probe (results/bindingdb_probe/jak2.json, 25,079 rows), 1,030 begin with '<' or '>'
     (570 + 460; 0 with '~'), e.g. '<0.1', '>10000'. A bound is not an exact Ki, so only pure
     non-negative numbers are kept (the same exact-measurement rule ChEMBL's
     standard_relation='=' enforces). IC50/Kd/EC50 rows are dropped -- this is a Ki table.

Label convention: pKi = 9 - log10(median Ki / nM), identical to the ChEMBL table and to the
9 - log10(nM) scale DAVIS's pKd uses. One ligand is one RDKit InChIKey, so a duplicated
structure (different SMILES strings or BindingDB monomer ids for the same molecule) can never
sit in train and test at the same time.

Temporal proxy: the publication year of the row's PubMed record (esummary pubdate), looked up
by the `pmid` the row carries. Most BindingDB rows are doi-only with an empty pmid (3,662 of
4,126 JAK2 Ki rows), and a doi-only row has no PubMed year here -- it stays undated, the
downstream temporal arm excludes undated pairs and reports how many it dropped (the same
handling ChEMBL uses for its undated document). No date is invented. The manifest records the
pmid coverage so this limitation is visible, not hidden.

Budget: every download is byte-counted against --budget-mb (default 175 MB, under the
user-approved 200 MB ceiling); each per-target response is additionally capped at
--per-file-mb (default 50 MB) by a streamed read that aborts if the body would exceed it.
When the budget would be exceeded the script stops early, writes completed:false into the
manifest, and refuses to pretend the table is complete.

Outputs (data/processed/external_bindingdb/, plus raw response caches under
data/raw/bindingdb/ which is gitignored -- the manifest records the sha256 of every cached
response so a reviewer can re-verify what was fetched):

    bindingdb_ki.csv  pair_id, ligand_id, target_id, pki, ki_nM, ki_min_nM, ki_max_nM,
                      ki_iqr_nM, n_rows, n_monomer_ids, n_docs, first_year, last_year
    ligands.csv       ligand_id, inchikey, canonical_smiles, n_monomer_ids, monomer_ids
    targets.csv       target_id, uniprot, target_name, chembl_target_id, organism, seq_len,
                      sequence, n_pairs_with_data       (copied from the ChEMBL panel)
    documents.csv     pmid, year, doi_seen
    external_manifest.json

The output schema deliberately matches what 14_external_chembl.py consumes (counts.pairs,
chembl_release, label_definition in the manifest; ligands/targets/<table>_ki.csv), so the
evaluation machinery is reused unchanged via a tagged adapter (15_external_bindingdb.py).
The committed ChEMBL tree (data/processed/external/) is never touched.

Usage:
    python scripts/15_fetch_bindingdb.py --max-targets 2 --budget-mb 30 --out-tag _plumbing
    python scripts/15_fetch_bindingdb.py                       # the real pull (background it)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bindingdb"
PROC = ROOT / "data" / "processed"
CHEMBL_TARGETS = PROC / "external" / "targets.csv"
REST = "https://bindingdb.org/rest/getLigandsByUniprots"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UA = "dti-affinity-bench/1.0 (BindingDB external validation fetch; local script)"

_NUM = re.compile(r"^\s*[+-]?\d+(?:\.\d+)?\s*$")
_YEAR = re.compile(r"(18|19|20)\d{2}")


# --------------------------------------------------------------------------- pure logic (tested)

def year_from_pubdate(s) -> int | None:
    """First four-digit year in a PubMed pubdate string ('2006 Feb 28', '2013 Nov', '1998').
    Returns None for empty / unparseable input -- an undated record stays undated."""
    if not s:
        return None
    m = _YEAR.search(str(s))
    return int(m.group(0)) if m else None


def years_from_esummary(d: dict) -> dict[str, int]:
    """{pmid: year} from an esummary JSON response. Records without a parseable pubdate are
    omitted (not given a fabricated year)."""
    res = (d or {}).get("result") or {}
    out = {}
    for uid in res.get("uids", []):
        rec = res.get(uid) or {}
        y = year_from_pubdate(rec.get("pubdate") or rec.get("epubdate"))
        if y is not None:
            out[str(uid)] = y
    return out


def filter_ki_rows(rows: list[dict], target_name: str | None = None
                   ) -> tuple[list[dict], dict]:
    """Keep exact, positive, numeric Ki (nM) rows that name the queried target.

    Every rejection is counted by reason so 'how much did we throw away' is visible. When
    target_name is given, a row whose `query` differs is an off-target selectivity row and is
    dropped (the single-UniProt response is not pure -- measured on the JAK2 probe)."""
    drop = {"wrong_affinity_type": 0, "off_target_query": 0, "censored_relation": 0,
            "non_numeric": 0, "non_positive_or_nan": 0, "no_smiles": 0}
    keep = []
    for a in rows:
        if a.get("affinity_type") != "Ki":
            drop["wrong_affinity_type"] += 1
            continue
        if target_name is not None and (a.get("query") or "").strip() != target_name.strip():
            drop["off_target_query"] += 1
            continue
        raw = a.get("affinity")
        s = "" if raw is None else str(raw)
        if not _NUM.match(s):
            # a leading '<'/'>'/'~' is a censored bound, not an exact measurement; an empty
            # value is a missing measurement (non_numeric), not a bound
            if s.strip() and s.strip()[:1] in ("<", ">", "~"):
                drop["censored_relation"] += 1
            else:
                drop["non_numeric"] += 1
            continue
        ki = float(s)
        if not (ki > 0) or not np.isfinite(ki):
            drop["non_positive_or_nan"] += 1
            continue
        smi = (a.get("smile") or "").strip()
        if not smi:
            drop["no_smiles"] += 1
            continue
        keep.append({"smiles": smi, "ki_nM": ki,
                     "pmid": str(a.get("pmid") or "").strip(),
                     "doi": str(a.get("doi") or "").strip(),
                     "monomerid": str(a.get("monomerid") or "").strip(),
                     "query": (a.get("query") or "").strip()})
    return keep, drop


def inchikey(smiles: str) -> str | None:
    """RDKit standard InChIKey for a SMILES string, or None if it cannot be parsed. Two
    different SMILES for the same molecule share one key, which is what de-duplicates ligands
    across train/test."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.MolToInchiKey(m)


def build_ligand_index(recs: list[dict]) -> tuple[dict[str, str], list[dict]]:
    """Group kept records by InChIKey. recs carry smiles/inchikey/monomerid. Returns
    (index {inchikey: ligand_id}, ligand rows). The canonical SMILES stored is RDKit's
    canonical form of the first record seen for that key (deterministic given input order)."""
    from rdkit import Chem
    by_key: dict[str, list[dict]] = {}
    for r in recs:
        by_key.setdefault(r["inchikey"], []).append(r)
    index, ligands = {}, []
    for i, key in enumerate(sorted(by_key)):
        lid = f"BL{i:05d}"
        index[key] = lid
        group = by_key[key]
        m = Chem.MolFromSmiles(group[0]["smiles"])
        canon = Chem.MolToSmiles(m) if m is not None else group[0]["smiles"]
        mons = sorted({g["monomerid"] for g in group if g.get("monomerid")})
        ligands.append({"ligand_id": lid, "inchikey": key, "canonical_smiles": canon,
                        "n_monomer_ids": len(mons), "monomer_ids": "|".join(mons)})
    return index, ligands


def aggregate_pairs(recs: list[dict], index: dict[str, str], target_id: str,
                    years: dict[str, int]) -> list[dict]:
    """Median Ki per (ligand, target). recs carry inchikey/ki_nM/pmid/doi/monomerid. A pair's
    years come from the pmids of its rows; a row with no pmid (or no resolved year) contributes
    no year, and a pair with no dated row has first_year/last_year = None (excluded by the
    temporal arm downstream, never given an invented date)."""
    by_lig: dict[str, list[dict]] = {}
    for r in recs:
        by_lig.setdefault(r["inchikey"], []).append(r)
    out = []
    for key, ks in by_lig.items():
        kis = np.array([k["ki_nM"] for k in ks], dtype=float)
        med = float(np.median(kis))
        pmids = {k["pmid"] for k in ks if k.get("pmid")}
        # a source document is a pmid where present, else the row's BindingDB doi (doi-only
        # rows are deposited records, e.g. 10.7270/Q2...); the two namespaces are kept apart
        doi_only = {k["doi"] for k in ks if k.get("doi") and not k.get("pmid")}
        yrs = [years[p] for p in pmids if p in years]
        out.append({"ligand_id": index[key], "target_id": target_id,
                    "ki_nM": round(med, 6), "pki": round(9.0 - np.log10(med), 6),
                    "n_rows": len(ks),
                    "n_monomer_ids": len({k["monomerid"] for k in ks if k.get("monomerid")}),
                    "n_docs": len(pmids) + len(doi_only),
                    "first_year": min(yrs) if yrs else None,
                    "last_year": max(yrs) if yrs else None,
                    "ki_min_nM": round(float(kis.min()), 6),
                    "ki_max_nM": round(float(kis.max()), 6),
                    "ki_iqr_nM": (round(float(np.subtract(*np.percentile(kis, [75, 25]))), 6)
                                  if len(ks) > 1 else 0.0)})
    return out


# --------------------------------------------------------------------------- network plumbing

class Ledger:
    """Counts requests and bytes so the budget is enforced and reported."""

    def __init__(self, budget_mb: float):
        self.bytes = 0
        self.requests = 0
        self.cached = 0
        self.retries = 0
        self.limit = budget_mb * 1e6
        self.exceeded = False

    def over_budget(self, extra: int) -> bool:
        return self.bytes + extra > self.limit


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch_target(uniprot: str, cutoff: int, ledger: Ledger, per_file_mb: float,
                 sleep: float = 0.4, tries: int = 5) -> list[dict] | None:
    """One getLigandsByUniprots response for a single UniProt, cached on disk by accession.
    The body is streamed and aborted if it would exceed per_file_mb or the run budget, so a
    runaway target cannot blow the cap. Returns the affinities list, or None on failure.

    Measured quirk (2026-11-07): the endpoint intermittently answers a large query with
    `200 OK, Content-Length: 0` and an empty body (a cold-backend / rate signal -- the same
    query succeeds on retry, and a small query succeeds while the large one is empty). An
    empty body is therefore treated as a retryable failure, not parsed as JSON and not cached;
    only a non-empty, parseable response is written to the cache."""
    cf = RAW / f"uniprot_{uniprot}_cutoff{cutoff}.json"
    if cf.exists():
        try:
            d = json.loads(cf.read_text(encoding="utf-8"))
            ledger.cached += 1
            return d["getLindsByUniprotsResponse"]["affinities"]
        except (json.JSONDecodeError, KeyError):
            cf.unlink()
    url = (f"{REST}?uniprot={urllib.parse.quote(uniprot)}&cutoff={cutoff}"
           f"&response=application/json")
    cap = int(per_file_mb * 1e6)
    for attempt in range(tries):
        if ledger.exceeded:
            return None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                length = r.headers.get("Content-Length")
                length = int(length) if length else 0
                # a positive Content-Length over the cap is refused before reading; a zero or
                # absent length (chunked) is not trusted -- the streamed read below enforces the
                # cap and an empty body is handled as a retryable failure after the read
                if length > 0 and (length > cap or ledger.over_budget(length)):
                    ledger.exceeded = True
                    print(f"  BUDGET: refusing {uniprot} (Content-Length {length / 1e6:.1f} MB)",
                          file=sys.stderr)
                    return None
                chunks, total = [], 0
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    total += len(block)
                    if total > cap or ledger.over_budget(total):
                        ledger.exceeded = True
                        print(f"  BUDGET: aborting {uniprot} mid-stream at {total / 1e6:.1f} MB",
                              file=sys.stderr)
                        return None
                    chunks.append(block)
                body = b"".join(chunks)
            if not body.strip():
                # intermittent empty 200 (cold backend / rate signal) -- retry, never cache
                ledger.requests += 1
                if attempt + 1 < tries:
                    ledger.retries += 1
                    print(f"  empty 200 for {uniprot}; retry {attempt + 1}/{tries - 1}",
                          file=sys.stderr)
                    time.sleep(3.0 * (attempt + 1))
                    continue
                print(f"  FETCH FAIL {uniprot}: empty body after {tries} tries", file=sys.stderr)
                return None
            ledger.bytes += len(body)
            ledger.requests += 1
            d = json.loads(body.decode("utf-8"))
            rows = d["getLindsByUniprotsResponse"]["affinities"]
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_bytes(body)
            return rows
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError,
                KeyError, TimeoutError) as e:
            code = getattr(e, "code", None)
            ledger.requests += 1
            if code in (429, 500, 502, 503) and attempt + 1 < tries:
                ledger.retries += 1
                time.sleep(2.0 * (attempt + 1))
                continue
            print(f"  FETCH FAIL {uniprot}: {type(e).__name__} {e}", file=sys.stderr)
            return None
        finally:
            time.sleep(sleep)
    return None


def fetch_years(pmids: list[str], ledger: Ledger, batch: int = 200,
                sleep: float = 0.34) -> dict[str, int]:
    """PubMed publication years via esummary, batched. Cached per batch on disk. NCBI's
    no-key limit is ~3 requests/s, so sleep keeps us under it."""
    out: dict[str, int] = {}
    ids = sorted({p for p in pmids if p})
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        key = "esummary_" + hashlib.sha1(",".join(chunk).encode()).hexdigest()[:16]
        cf = RAW / f"{key}.json"
        if cf.exists():
            try:
                d = json.loads(cf.read_text(encoding="utf-8"))
                ledger.cached += 1
                out.update(years_from_esummary(d))
                continue
            except json.JSONDecodeError:
                cf.unlink()
        if ledger.over_budget(50_000):
            ledger.exceeded = True
            break
        url = (f"{EUTILS}/esummary.fcgi?db=pubmed&id={urllib.parse.quote(','.join(chunk))}"
               f"&retmode=json")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read()
            ledger.bytes += len(body)
            ledger.requests += 1
            d = json.loads(body.decode("utf-8"))
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_bytes(body)
            out.update(years_from_esummary(d))
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError,
                TimeoutError) as e:
            ledger.requests += 1
            print(f"  esummary fail ({len(chunk)} ids): {type(e).__name__} {e}", file=sys.stderr)
        time.sleep(sleep)
    return out


# --------------------------------------------------------------------------- main

def main() -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-mb", type=float, default=175.0,
                    help="hard cap on bytes downloaded this run (cached responses are free); "
                         "the user-approved ceiling for this task was 200 MB")
    ap.add_argument("--per-file-mb", type=float, default=50.0,
                    help="hard cap on a single per-target response body (streamed, aborted if exceeded)")
    ap.add_argument("--cutoff", type=int, default=100000,
                    help="affinity cutoff in nM passed to getLigandsByUniprots")
    ap.add_argument("--max-targets", type=int, default=16,
                    help="how many of the ChEMBL panel's targets to pull (default: all 16)")
    ap.add_argument("--min-rows", type=int, default=50,
                    help="drop a target below this many kept exact-Ki rows")
    ap.add_argument("--out-tag", default="",
                    help="suffix for the output directory; a smoke run uses a separate tag so "
                         "it cannot be mistaken for the real table")
    args = ap.parse_args()

    ledger = Ledger(args.budget_mb)
    t0 = time.time()
    out = PROC / ("external_bindingdb" + args.out_tag)
    out.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    tg = pd.read_csv(CHEMBL_TARGETS).head(args.max_targets)
    print(f"panel: {len(tg)} targets from the ChEMBL external panel (same UniProt accessions)")

    # 1. fetch + filter per target
    per_target: dict[str, list[dict]] = {}
    drop_total = {"wrong_affinity_type": 0, "off_target_query": 0, "censored_relation": 0,
                  "non_numeric": 0, "non_positive_or_nan": 0, "no_smiles": 0}
    fetched_rows = {}
    kept_rows = {}
    failed = []
    tgt_rows = []
    for i, t in enumerate(tg.to_dict("records")):
        acc = t["uniprot"]
        name = t["target_name"]
        rows = fetch_target(acc, args.cutoff, ledger, args.per_file_mb)
        if rows is None:
            failed.append(acc)
            print(f"  {acc} ({name[:36]}): FETCH FAILED -- excluded", file=sys.stderr)
            continue
        keep, drop = filter_ki_rows(rows, target_name=name)
        for k, v in drop.items():
            drop_total[k] += v
        fetched_rows[acc] = len(rows)
        kept_rows[acc] = len(keep)
        print(f"  {acc} ({name[:36]}): {len(rows)} rows fetched, {len(keep)} exact-Ki kept "
              f"(off-target {drop['off_target_query']}, censored {drop['censored_relation']}, "
              f"non-Ki {drop['wrong_affinity_type']})")
        if len(keep) < args.min_rows:
            print(f"    below --min-rows {args.min_rows}; excluded from the table", file=sys.stderr)
            continue
        # attach InChIKey now so ligands can be de-duplicated across the whole panel
        bad_smi = 0
        keyed = []
        for k in keep:
            ik = inchikey(k["smiles"])
            if ik is None:
                bad_smi += 1
                continue
            keyed.append({**k, "inchikey": ik})
        drop_total.setdefault("unparseable_smiles", 0)
        drop_total["unparseable_smiles"] = drop_total.get("unparseable_smiles", 0) + bad_smi
        per_target[acc] = keyed
        trow = len(tgt_rows)
        tgt_rows.append({**t, "target_id": f"XT{trow:03d}"})
    print(f"dropped rows (all targets): {drop_total}")
    if ledger.exceeded:
        print("BUDGET EXCEEDED during fetch -- table will be marked incomplete", file=sys.stderr)

    # 2. ligand index over all kept records (InChIKey de-dup across targets)
    all_recs = [r for recs in per_target.values() for r in recs]
    index, ligands = build_ligand_index(all_recs)
    print(f"ligands: {len(ligands)} unique InChIKeys from {len(all_recs)} kept rows")

    # 3. PubMed years for every pmid that appears on a kept row
    pmids = sorted({r["pmid"] for r in all_recs if r["pmid"]})
    years = fetch_years(pmids, ledger)
    print(f"years: {len(years)}/{len(pmids)} distinct pmids resolved to a publication year")

    # 4. assemble pairs + per-target counts
    pairs = []
    for t in tgt_rows:
        acc = t["uniprot"]
        recs = per_target.get(acc, [])
        prows = aggregate_pairs(recs, index, t["target_id"], years)
        t["n_pairs_with_data"] = len(prows)
        pairs.extend(prows)
    pdf = pd.DataFrame(pairs)
    if pdf.empty:
        print("no pairs assembled", file=sys.stderr)
        (out / "external_manifest.json").write_text(json.dumps(
            {"completed": False, "error": "no pairs assembled", "failed_targets": failed},
            indent=2), encoding="utf-8")
        return 1
    pdf = pdf.sort_values(["target_id", "ligand_id"]).reset_index(drop=True)
    pdf.insert(0, "pair_id", np.arange(len(pdf)))
    used = sorted(set(pdf.ligand_id))
    ldf = pd.DataFrame([l for l in ligands if l["ligand_id"] in used])
    tdf = pd.DataFrame(tgt_rows)[["target_id", "uniprot", "target_name", "chembl_target_id",
                                  "organism", "seq_len", "sequence", "n_pairs_with_data"]]
    ddf = pd.DataFrame([{"pmid": p, "year": years.get(p),
                         "doi_seen": "|".join(sorted({r["doi"] for r in all_recs
                                                      if r["pmid"] == p and r["doi"]}))}
                        for p in pmids])

    wall = round(time.time() - t0, 1)
    files = {}
    for name, df in (("bindingdb_ki.csv", pdf), ("ligands.csv", ldf),
                     ("targets.csv", tdf), ("documents.csv", ddf)):
        p = out / name
        df.to_csv(p, index=False)
        files[name] = {"path": p.relative_to(ROOT).as_posix(), "rows": int(len(df)),
                       "sha256": sha256_file(p)}

    n_undated = int(pdf.first_year.isna().sum())
    yrs_all = [y for y in pdf.first_year.dropna().tolist()]
    manifest = {
        "generated_by": "scripts/15_fetch_bindingdb.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": not ledger.exceeded and not failed,
        "source": "BindingDB RESTful API (bindingdb.org/rest/getLigandsByUniprots), open, no key "
                  "+ PubMed E-utilities esummary for publication years",
        # 14_external_chembl reads this key for provenance; BindingDB's REST carries no release
        # version, so the fetch timestamp + cutoff stand in (recorded honestly, not a ChEMBL tag)
        "chembl_release": {"ok": not ledger.exceeded, "source": "BindingDB",
                           "bindingdb_release": None,
                           "note": "BindingDB REST exposes no version stamp; fetch time + cutoff "
                                   "identify this snapshot",
                           "cutoff_nM": args.cutoff,
                           "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "params": vars(args),
        "selection_rule": "the 16 human kinases of the ChEMBL external panel "
                          "(data/processed/external/targets.csv), same UniProt accessions; "
                          "a target is kept if it has >= --min-rows exact-Ki rows naming it",
        "label_definition": "pKi = 9 - log10(median Ki / nM); same 9-log10(nM) scale as "
                            "DAVIS pKd and the ChEMBL table, not comparable to KIBA's 0-17.2 score",
        "aggregation": "median of the exact Ki rows for one (InChIKey, target) pair; "
                       "n_rows / ki_iqr_nM carried alongside so the aggregation is auditable",
        "ligand_identity": "RDKit standard InChIKey; different SMILES/monomer ids for one "
                           "structure are one ligand, so a duplicate cannot straddle train/test",
        "target_attribution": "a single-UniProt BindingDB response also carries selectivity rows "
                              "for other proteins (measured on O60674: TYK2/JAK1/PDGFR-beta rows); "
                              "a row is kept only when query == the target's UniProt name",
        "temporal_proxy": "PubMed publication year (esummary pubdate) of the row's pmid. Most "
                          "BindingDB rows are doi-only with empty pmid and stay undated; the "
                          "temporal arm excludes undated pairs and reports the count. A "
                          "substitution, not the same quantity as a deposition date.",
        "row_filters_applied": drop_total,
        "coverage_per_target": {acc: {"rows_fetched": fetched_rows.get(acc),
                                      "exact_ki_kept": kept_rows.get(acc),
                                      "in_table": acc in per_target}
                                for acc in [t["uniprot"] for t in tg.to_dict("records")]},
        "failed_targets": failed,
        "pmid_coverage": {"distinct_pmids_on_kept_rows": len(pmids),
                          "resolved_to_year": len(years),
                          "kept_rows_with_pmid": sum(1 for r in all_recs if r["pmid"]),
                          "kept_rows_total": len(all_recs),
                          "pairs_without_any_year": n_undated},
        "counts": {"pairs": int(len(pdf)), "ligands": int(len(ldf)), "targets": int(len(tdf)),
                   "documents": int(len(ddf)),
                   "rows_fetched": int(sum(fetched_rows.values())),
                   "rows_kept": int(len(all_recs)),
                   "pairs_per_target": {t.target_id: int((pdf.target_id == t.target_id).sum())
                                        for t in tdf.itertuples()},
                   "pairs_with_multiple_rows": int((pdf.n_rows > 1).sum()),
                   "pairs_without_any_year": n_undated,
                   "year_min": int(min(yrs_all)) if yrs_all else None,
                   "year_max": int(max(yrs_all)) if yrs_all else None,
                   "pki_min": float(pdf.pki.min()), "pki_max": float(pdf.pki.max()),
                   "pki_mean": float(pdf.pki.mean()), "pki_sd": float(pdf.pki.std())},
        "transfer": {"http_requests": ledger.requests, "bytes_downloaded": ledger.bytes,
                     "cached_responses_reused": ledger.cached, "retries": ledger.retries,
                     "budget_mb": args.budget_mb, "per_file_mb": args.per_file_mb,
                     "budget_exceeded": ledger.exceeded, "wall_seconds": wall,
                     "raw_cache_bytes_on_disk": sum(f.stat().st_size for f in RAW.glob("*")
                                                    if f.is_file())},
        "files": files,
        "raw_cache_sha256": sorted(f"data/raw/bindingdb/{c.name}: {sha256_file(c)}"
                                   for c in RAW.glob("*") if c.is_file()),
    }
    (out / "external_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\nwrote {out}  (completed={manifest['completed']})")
    print(f"pairs {len(pdf)}  ligands {len(ldf)}  targets {len(tdf)}  "
          f"undated pairs {n_undated}  requests {ledger.requests}  "
          f"{ledger.bytes / 1e6:.1f} MB  {wall} s")
    return 0 if manifest["completed"] else 3


if __name__ == "__main__":
    sys.exit(main())