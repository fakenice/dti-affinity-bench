#!/usr/bin/env python3
"""
13_fetch_chembl.py -- a *third*, independently curated affinity table, for external
validation of the headline claim ("the split protocol moves the number more than the
model does").

Why ChEMBL and not BindingDB: BindingDB's REST endpoint requires an emailed API key and
its bulk TSV paths return 404 from this host, so it cannot be pulled reproducibly here.
ChEMBL's REST API is open, filterable and versioned. This script verifies every filter it
relies on against the live API and records the verified response shapes in the manifest --
nothing here is written against a guessed schema.

What is pulled (and the budget, which is enforced, not aspirational):

  1. kinase-named targets            GET /target?pref_name__icontains=kinase
  2. per-target usable Ki row count  GET /activity.json?target_chembl_id=X
                                     &standard_type=Ki&standard_flag=1
                                     &standard_relation=%3D&limit=1
     (every screened candidate is counted -- `--screen` defaults above the size of the
     kinase-named human set, so the panel is the top `--max-targets` over the whole named-kinase
     population rather than over an arbitrary head slice of the API's id order)
  3. the Ki rows themselves          GET /activity.json?...&limit=1000&offset=...  (<= cap,
     non-overlapping pages; rows additionally de-duplicated on activity_id -- see plan_offsets)
  4. a 300-molecule spot check       GET /molecule.json?molecule_chembl_id__in=...&limit=1000
     -- to verify the canonical_smiles carried on the activity rows against the structure table
  5. source publication years        GET /document/{id}.json  (the temporal proxy; see below)
  6. protein sequences               UniProt REST, fasta, by the accession ChEMBL reports
  7. release stamp                   GET /status/ (a 362-byte XML carrying chembl_db_version;
     the run recorded here read ChEMBL_37, released 2026-05-01, over 24,527,044 activities)

Canonical SMILES and the source publication year both arrive on the activity row, which is what
keeps this pull inside a browser-sized budget: the /molecule endpoint costs ~3.7 kB per structure
measured (a 20-molecule page was 75 kB), so fetching a structure record for every ligand of the
panel would have cost more than the activity rows themselves. Item 4 is what stops that shortcut
being taken on trust -- the manifest records the exact-match rate, and a mismatch would be visible
instead of silent. (In the smoke run all 300 sampled structures matched their activity row's
canonical_smiles exactly, and all 459 rows that carried a year agreed with /document/.)

Label convention: pKi = 9 - log10(Ki / nM), the same "9 - log10(nM)" scale DAVIS's pKd uses,
so the RMSE units are comparable to the DAVIS column (they are *not* comparable to KIBA's
0-17.2 consensus score, and nothing here claims they are).  Only rows with standard_relation
'=' are kept: a 'Ki > 10 mM' row is a censored measurement, not an exact one, and averaging it
into a median would bias the label.  Two chembl ids that share a canonical SMILES are one
ligand here, so a duplicated structure can never sit in train and test at the same time.

The temporal protocol uses the **publication year** of the source document, not ChEMBL's
deposition date, because the public API does not expose a per-activity deposition date. That
is a documented substitution, not the same quantity: `external_manifest.json` records it in
`temporal_proxy` together with a cross-check of the activity-carried year against /document/,
and the README carries it as a limitation. A related measured gap: one ChEMBL document
(CHEMBL1201862) has no year at all in either place, and in the pre-relation-filter smoke run its
rows were 406 of the 965 kept -- so `chembl_ki.csv` carries `first_year` as NaN for those pairs
and the downstream script excludes undated pairs from the temporal arm and reports how many it
dropped, rather than inventing a date for them.

Every download is byte-counted against `--budget-mb`. When the budget would be exceeded the
script stops early, writes `completed: false` into the manifest, and refuses to pretend the
table is complete.

Outputs (data/processed/external/, plus raw page caches under data/raw/chembl/ which is
gitignored -- the manifest therefore records the sha256 of every cached page so a reviewer
can re-verify what was fetched):

    chembl_ki.csv     pair_id, ligand_id, target_id, pki, ki_nM, ki_min_nM, ki_max_nM,
                      ki_iqr_nM, n_rows, n_chembl_ids, n_docs, first_year, last_year
    ligands.csv       ligand_id, canonical_smiles, n_chembl_ids, chembl_ids
    targets.csv       target_id, uniprot, target_name, chembl_target_id, organism, seq_len,
                      sequence, n_pairs_with_data
    documents.csv     document_chembl_id, year, journal, doi, publication_year
    external_manifest.json

Usage:
    python scripts/13_fetch_chembl.py --screen 20 --max-targets 3 --rows-per-target 400 \
        --min-rows 50 --budget-mb 40 --out-tag _plumbing      # smoke test, separate directory
    python scripts/13_fetch_chembl.py                          # the real pull (background it)
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
RAW = ROOT / "data" / "raw" / "chembl"
OUT = ROOT / "data" / "processed" / "external"
API = "https://www.ebi.ac.uk/chembl/api/data"
UP = "https://rest.uniprot.org/uniprotkb"
UA = "dti-affinity-bench/1.0 (external validation fetch; local script)"

# the four filters below were each verified against the live API before this script was
# written (unfiltered ChEMBL list endpoints silently return the whole table, so "total_count
# changed" is the test that a filter actually applied -- see verify_filters())
KINASE_FILTER = "pref_name__icontains=kinase"
# standard_relation '=' is the exact-measurement filter: measured 2026-11-04 on CHEMBL258,
# standard_type=Ki&standard_flag=1 gives 1052 rows and adding standard_relation='=' gives 415.
# The other 637 are '>'/'<' bounds, which are not Ki values and would bias a median, so they are
# excluded server-side (cheaper per usable row) and re-checked locally (counted if any slip in).
ACT_FILTER = "standard_type=Ki&standard_flag=1&standard_relation=%3D"
ACT_FILTER_PLAIN = "standard_type=Ki&standard_flag=1"

def act_url(target_id: str, limit: int = 1000, offset: int = 0,
            relation: bool = True) -> str:
    return (f"{API}/activity.json?target_chembl_id={target_id}"
            f"&{ACT_FILTER if relation else ACT_FILTER_PLAIN}"
            f"&limit={limit}&offset={offset}")


class Ledger:
    """Counts requests and bytes so the budget is enforced, and so the manifest can say
    exactly how much was fetched."""

    def __init__(self, budget_mb: float):
        self.bytes = 0
        self.requests = 0
        self.cached = 0
        self.retries = 0
        self.limit = budget_mb * 1e6
        self.exceeded = False

    def over_budget(self, extra: int) -> bool:
        return self.bytes + extra > self.limit


def get_json(url: str, ledger: Ledger, cache_key: str | None = None,
             sleep: float = 0.35, tries: int = 3) -> dict | None:
    """GET with a disk cache (cache hits cost no bytes against the network budget but are
    still hashed and recorded), 429/5xx backoff, and a hard budget check on the response
    length header before the body is read.

    The cache filename carries a fingerprint of the full URL, so a page cached under an older
    query can never be served for a new one -- the query for activity pages changed mid-
    development (standard_flag was added), and a name-only key would have quietly served the
    pre-filter page as if it were the filtered one."""
    if cache_key:
        fp = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        cf = RAW / f"{cache_key}--{fp}.json"
    else:
        cf = None
    if cf is not None and cf.exists():
        try:
            ledger.cached += 1
            return json.loads(cf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cf.unlink()          # a truncated cache file is not a cache
    for attempt in range(tries):
        if ledger.exceeded:
            return None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                length = int(r.headers.get("Content-Length") or 0)
                if ledger.over_budget(length or 200_000):
                    ledger.exceeded = True
                    print(f"  BUDGET: refusing {url} "
                          f"({(ledger.bytes + (length or 200_000)) / 1e6:.1f} MB > "
                          f"{ledger.limit / 1e6:.0f} MB)", file=sys.stderr)
                    return None
                body = r.read()
            ledger.bytes += len(body)
            ledger.requests += 1
            data = json.loads(body.decode("utf-8"))
            if cf is not None:
                cf.parent.mkdir(parents=True, exist_ok=True)
                cf.write_bytes(body)
            return data
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError,
                TimeoutError) as e:
            code = getattr(e, "code", None)
            ledger.requests += 1
            if code in (429, 500, 502, 503) and attempt + 1 < tries:
                ledger.retries += 1
                time.sleep(2.0 * (attempt + 1))
                continue
            print(f"  FETCH FAIL {url}: {type(e).__name__} {e}", file=sys.stderr)
            return None
        finally:
            time.sleep(sleep)
    return None


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def total_count(url: str, ledger: Ledger, key: str, sleep: float = 0.35) -> int | None:
    """page_meta.total_count for a query, at minimum cost. `limit` is only appended when the
    caller's URL does not already set it, so the two are never in conflict."""
    extra = "&format=json" + ("" if "limit=" in url else "&limit=1")
    d = get_json(url + extra, ledger, key, sleep=sleep)
    if not d:
        return None
    pm = d.get("page_meta") or {}
    tc = pm.get("total_count")
    return int(tc) if tc is not None else None


def verify_filters(ledger: Ledger) -> dict:
    """A ChEMBL filter that is silently ignored returns the *unfiltered* table. So the check
    for "does this filter work" is that the filtered total differs from the unfiltered one,
    and that the returned records match the filter textually."""
    checks = {}
    unfiltered = total_count(f"{API}/target?target_type=SINGLE PROTEIN".replace(" ", "%20"),
                             ledger, "filtercheck_target_all")
    kin = f"{API}/target?{KINASE_FILTER}"
    n_kin = total_count(kin, ledger, "filtercheck_target_kinase")
    d = get_json(kin + "&limit=3&format=json", ledger, "filtercheck_target_kinase_rows")
    names = [t.get("pref_name") for t in (d or {}).get("targets", [])]
    checks["target_pref_name__icontains=kinase"] = {
        "unfiltered_total": unfiltered, "filtered_total": n_kin,
        "first_three_names": names,
        "applies": bool(n_kin and unfiltered and n_kin < unfiltered) and
                   all("kinase" in (x or "").lower() for x in names)}
    tid = "CHEMBL258"                                 # Lck: a target with both Ki and non-Ki rows
    d2 = get_json(f"{API}/activity.json?target_chembl_id={tid}&standard_type=Ki&limit=100",
                  ledger, "filtercheck_activity_ki_only")
    a2 = (d2 or {}).get("activities") or []
    types = sorted({a.get("standard_type") for a in a2})
    checks["activity_standard_type=Ki"] = {
        "n_rows": len(a2), "standard_types_seen": types,
        "null_standard_values": sum(1 for a in a2 if a.get("standard_value") is None),
        "total_count": ((d2 or {}).get("page_meta") or {}).get("total_count"),
        "applies": bool(a2) and types == ["Ki"]}
    d3 = get_json(act_url(tid, limit=100, relation=False), ledger, "filtercheck_activity_ki_flag")
    a3 = (d3 or {}).get("activities") or []
    flags = sorted({str(a.get("standard_flag")) for a in a3})
    nulls = sum(1 for a in a3 if a.get("standard_value") is None)
    n3 = ((d3 or {}).get("page_meta") or {}).get("total_count")
    checks["activity_standard_flag=1"] = {
        "n_rows": len(a3), "standard_flags_seen": flags, "null_standard_values": nulls,
        "total_count": n3,
        "applies": bool(a3) and flags == ["1"] and nulls == 0}
    d4 = get_json(act_url(tid, limit=100), ledger, "filtercheck_activity_all_three")
    a4 = (d4 or {}).get("activities") or []
    rel = sorted({str(a.get("standard_relation")) for a in a4})
    n4 = ((d4 or {}).get("page_meta") or {}).get("total_count")
    checks["activity_standard_relation=exact"] = {
        "n_rows": len(a4), "standard_relations_seen": rel,
        "total_count_without_relation_filter": n3, "total_count": n4,
        "applies": bool(a4) and rel == ["="] and n4 is not None and n3 is not None
                   and int(n4) < int(n3)}
    # the __in lookup paginates at limit=20 by default -- measured on 2026-11-04, where a
    # 150-id batch returned 20 molecules and page_meta.total_count said 150. An explicit limit
    # is therefore part of the filter contract, not an optimisation.
    probe_ids = ["CHEMBL25", "CHEMBL121", "CHEMBL1991143"]
    mol = get_json(f"{API}/molecule.json?molecule_chembl_id__in={urllib.parse.quote(','.join(probe_ids))}"
                   f"&limit=1000&format=json", ledger, "filtercheck_molecule_in")
    ids = [m.get("molecule_chembl_id") for m in (mol or {}).get("molecules", [])]
    tc_mol = ((mol or {}).get("page_meta") or {}).get("total_count")
    checks["molecule_molecule_chembl_id__in"] = {
        "requested": probe_ids, "ids": ids, "total_count": tc_mol,
        "note": ("the discriminator is total_count: an ignored __in filter would report the whole "
                 "compound table (millions), a working one reports at most the ids asked for. "
                 "An id in `requested` that no longer exists is therefore not a filter failure."),
        "applies": bool(ids) and set(ids) <= set(probe_ids) and tc_mol is not None
                   and int(tc_mol) <= len(probe_ids)}
    return checks


def chembl_status(ledger: Ledger) -> dict:
    """GET /status/ is a 362-byte XML document that carries the database release name and date.
    Parsed with a regex (it is not JSON); a failure is recorded as ok:false, never guessed."""
    url = f"{API}/status/"
    cf = RAW / "status.json"
    if cf.exists():
        ledger.cached += 1
        text = cf.read_text(encoding="utf-8")
    else:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            ledger.bytes += len(body)
            ledger.requests += 1
            text = body.decode("utf-8")
            cf.write_text(text, encoding="utf-8")
        except Exception as e:                        # noqa: BLE001 - recorded, not fatal
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "url": url}
        time.sleep(0.2)

    def field(name):
        m = re.search(rf"<{name}>([^<]*)</{name}>", text)
        return m.group(1) if m else None
    return {"ok": bool(field("chembl_db_version")),
            "chembl_db_version": field("chembl_db_version"),
            "chembl_release_date": field("chembl_release_date"),
            "activities_in_db": field("activities"),
            "publications_in_db": field("publications"),
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# --------------------------------------------------------------------------- panel choice

def kinase_candidates(ledger: Ledger, screen: int, organism: str,
                      stats: dict) -> list[dict]:
    """All human, single-protein, kinase-named targets, in the order ChEMBL returns them,
    truncated to `screen` for the counting pass. A candidate without a component accession
    cannot ever enter the panel (no sequence), so it is dropped here and counted, rather than
    silently costing a panel slot at assembly time."""
    out, offset = [], 0
    while len(out) < screen:
        d = get_json(f"{API}/target?{KINASE_FILTER}&target_type=SINGLE%20PROTEIN"
                     f"&limit=1000&offset={offset}&format=json",
                     ledger, f"targets_kinase_off{offset}")
        if not d:
            break
        rows = d.get("targets") or []
        if not rows:
            break
        last_page = len(rows) < 1000          # a short page is the end of the result set, and
        # each page costs ~7 MB measured, so stopping here rather than asking for the next one
        # keeps screening from paging past the whole kinase-named table
        for t in rows:
            if organism and (t.get("organism") or "").lower() != organism.lower():
                continue
            if t.get("target_type") != "SINGLE PROTEIN":
                continue
            acc = None
            comps = t.get("target_components") or []
            if comps:
                acc = comps[0].get("accession")
            if not acc:
                stats["no_component_accession"] += 1
                continue
            out.append({"chembl_target_id": t["target_chembl_id"],
                        "pref_name": t.get("pref_name"), "accession": acc})
            if len(out) >= screen:
                break
        offset += 1000
        if last_page or offset > 5000:        # short page, or the deep-paging cap
            break
    return out


def count_ki_rows(cands: list[dict], ledger: Ledger, t0: float, stats: dict) -> list[dict]:
    """One limit=1 request per candidate: its exact-standard Ki row count. Cheap in bytes
    (~1.5 kB measured) but not in wall time, so progress is printed for the background log."""
    for i, c in enumerate(cands, 1):
        n = total_count(act_url(c["chembl_target_id"], limit=1),
                        ledger, f"count_{c['chembl_target_id']}", sleep=0.2)
        c["n_ki_rows"] = int(n) if n is not None else -1
        if n is None:
            stats["count_requests_failed"] += 1
        if i % 100 == 0 or i == len(cands):
            print(f"  counted {i}/{len(cands)} candidates "
                  f"({ledger.bytes / 1e6:.1f} MB, {round(time.time() - t0, 0)} s)", flush=True)
    return cands


# --------------------------------------------------------------------------- activities

def plan_offsets(total: int, cap: int, page: int, mode: str = "spread") -> list[int]:
    """Page starts for one target's Ki rows, chosen so that no two pages can overlap.

    The first version of this function spaced `n_pages` pages evenly across [0, total-page],
    which overlaps whenever a target is *slightly* bigger than the cap: measured on the live
    API 2026-11-04, CHEMBL2971 has 3333 usable rows against cap=3000 and page=1000, so the
    spacing came out at 778 < 1000 and 445 of the 3000 fetched rows were repeats of rows
    already in the list (CHEMBL2835: 509, CHEMBL2147: 516 -- all three confirmed against the
    page cache by re-deriving each cached URL). A repeated activity row double-weights one
    measurement inside a pair median, which is exactly the kind of silent defect an external
    validation table must not have. Spacing is therefore floored at one full page, and the
    last page is anchored at the end of the range so the sample still spans the history."""
    n_pages = max(1, -(-cap // page))
    if mode == "head" or not total or total <= cap or n_pages < 2:
        return [i * page for i in range(n_pages)]
    last = max(0, total - page)                     # deepest start that still returns rows
    step = max(float(page), last / (n_pages - 1))
    offs = [int(round(i * step)) for i in range(n_pages)]
    offs[-1] = last
    return sorted({min(o, last) for o in offs})


def fetch_activities(target_id: str, cap: int, total: int, ledger: Ledger,
                     page: int = 1000, mode: str = "spread"
                     ) -> tuple[list[dict], list[int], int]:
    """Up to `cap` *distinct* rows for one target, sampled as evenly spaced, non-overlapping
    pages across the target's whole result range rather than as its first `cap` rows.

    Why spread at all: in the smoke run a head-only slice (first 600 rows of each of three
    kinases) spanned publication years 2003-2012 only, which is too little history for a
    temporal protocol to split on. Whether deeper offsets actually carry later literature is
    testable and was tested by --probe-year-spread (result: later at depth for CHEMBL262,
    a tie for CHEMBL1862 -- so "deeper pages are newer" is a tendency, not a guarantee, and
    the manifest records the offsets used rather than implying coverage).

    Rows are additionally de-duplicated on `activity_id` as a hard safety net, and the number
    removed is returned so the manifest can state it instead of hiding it."""
    offsets = plan_offsets(total, cap, page, mode)
    rows: list[dict] = []
    seen: set = set()
    n_dupe = 0
    for off in offsets:
        d = get_json(act_url(target_id, limit=page, offset=off),
                     ledger, f"act_{target_id}_off{off}")
        if not d:
            continue
        for a in (d.get("activities") or []):
            aid = a.get("activity_id")
            if aid is not None:
                if aid in seen:
                    n_dupe += 1
                    continue
                seen.add(aid)
            rows.append(a)
    return rows[:cap], offsets, n_dupe


# --------------------------------------------------------------------------- molecules

def fetch_molecules(ids: list[str], ledger: Ledger, batch: int = 100) -> dict[str, dict]:
    """Spot-check source: the /molecule record for a sample of the ligands we kept, so the
    denormalised canonical_smiles on the activity row can be verified against the authoritative
    structure table rather than taken on trust. Returns {chembl_id: {smiles, ...}}."""
    out: dict[str, dict] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        key = "mol_" + hashlib.sha1(",".join(sorted(chunk)).encode()).hexdigest()[:16]
        q = urllib.parse.quote(",".join(chunk))
        d = get_json(f"{API}/molecule.json?molecule_chembl_id__in={q}&limit=1000&format=json",
                     ledger, key)
        if not d:
            continue
        pm = d.get("page_meta") or {}
        got = d.get("molecules") or []
        if len(got) < int(pm.get("total_count") or 0):
            print(f"  molecule page truncated: {len(got)} of {pm.get('total_count')}",
                  file=sys.stderr)
        for m in got:
            st = m.get("molecule_structures") or {}
            out[m["molecule_chembl_id"]] = {
                "canonical_smiles": st.get("canonical_smiles"),
                "pref_name": m.get("pref_name"),
                "molecule_type": m.get("molecule_type"),
                "full_mwt": (m.get("molecule_properties") or {}).get("full_mwt")}
    return out


def _fasta_accession(header: str) -> str | None:
    """UniProt fasta headers are '>sp|P12345|KPA1_HUMAN ...' (reviewed) or '>tr|A0A...|...'
    (unreviewed); the bare '>P12345 ...' form also exists. Returns the accession field, or None
    if the header is not parseable -- so a mis-mapped record fails loudly rather than silently."""
    h = (header or "").strip().lstrip(">").strip()
    if not h:
        return None
    parts = h.split("|")
    if len(parts) >= 3 and parts[0] in ("sp", "tr", "Swiss-Prot", "TrEMBL"):
        return parts[1] or None
    return h.split()[0].split("-")[0] or None


def fetch_uniprot(accessions: list[str], ledger: Ledger) -> dict[str, dict]:
    """Sequence straight from UniProt, fasta format -- and the record is checked to be the
    human reviewed entry ChEMBL claims it is, so a mis-mapped accession cannot slip in."""
    out: dict[str, dict] = {}
    for acc in accessions:
        if not acc:
            continue
        url = f"{UP}/{acc}?format=fasta&includeIsoform=false"
        if ledger.over_budget(20_000):
            ledger.exceeded = True
            break
        cf = RAW / f"uniprot_{acc}.fasta"
        if cf.exists():
            text = cf.read_text(encoding="utf-8")
            ledger.cached += 1
        else:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                ledger.bytes += len(body)
                ledger.requests += 1
                text = body.decode("utf-8")
                cf.parent.mkdir(parents=True, exist_ok=True)
                cf.write_text(text, encoding="utf-8")
            except Exception as e:                       # noqa: BLE001 - logged and skipped
                print(f"  uniprot fail {acc}: {e}", file=sys.stderr)
                out[acc] = {"error": str(e)}
                continue
            time.sleep(0.2)
        lines = text.strip().splitlines()
        head = lines[0] if lines else ""
        seq = "".join(l.strip() for l in lines[1:] if not l.startswith(">"))
        # UniProt fasta headers are '>sp|P00519|ABL1_HUMAN Tyrosine-protein kinase ABL1
        #  OS=Homo sapiens OX=9606 GN=ABL1 PE=1 SV=4' -- unbracketed (measured on the cached
        #  records; the bracketed [OS=...] form is the NCBI style, so both are tolerated here)
        m = re.search(r"OS=(.+?)(?=\s+(?:OX=|GN=|PE=|SV=|$))", head)
        organism = m.group(1).strip().strip("[]") if m else None
        out[acc] = {"header": head, "sequence": seq, "seq_len": len(seq),
                    "organism": organism,
                    "reviewed": head.startswith(">sp|"),
                    "fasta_accession": _fasta_accession(head),
                    "matches_requested_accession": _fasta_accession(head) == acc}
    return out


# --------------------------------------------------------------------------- main

def year_spread_probe(panel: list[dict], ledger: Ledger, out: Path) -> dict:
    """Falsifiable test for --slice spread: for each of the first two panel targets, read one
    page at offset 0 and one page at the deepest offset the target's total allows, and compare
    the publication-year distributions of the two. If ordering by activity id really does put
    recent literature at the end, the deep page's median year is later; if not, spread sampling
    buys nothing and --slice head is the honest choice."""
    res = {}
    for c in panel[:2]:
        total = c["n_ki_rows"]
        pages = {"head": 0}
        if total > 1000:
            pages["deep"] = max(0, total - 1000)
        per = {}
        for name, off in pages.items():
            rows = get_json(act_url(c["chembl_target_id"], limit=1000, offset=off),
                            ledger, f"probe_{c['chembl_target_id']}_off{off}")
            a = (rows or {}).get("activities") or []
            yr = [k.get("document_year") for k in a if k.get("document_year")]
            if not yr:                              # the row field is often empty; ask the docs
                dids = sorted({k.get("document_chembl_id") for k in a
                               if k.get("document_chembl_id")})[:40]
                for did in dids:
                    d = get_json(f"{API}/document/{did}.json", ledger, f"doc_{did}")
                    if d and d.get("year"):
                        yr.append(int(d["year"]))
            act_ids = [k.get("activity_id") for k in a if k.get("activity_id")]
            per[name] = {"offset": off, "rows": len(a), "n_years": len(yr),
                         "year_min": min(yr) if yr else None,
                         "year_max": max(yr) if yr else None,
                         "year_median": int(np.median(yr)) if yr else None,
                         "activity_id_min": min(act_ids) if act_ids else None,
                         "activity_id_max": max(act_ids) if act_ids else None}
        spread = (per.get("deep", {}).get("year_median") is not None
                  and per["head"]["year_median"] is not None
                  and per["deep"]["year_median"] > per["head"]["year_median"])
        res[c["chembl_target_id"]] = {"pages": per, "later_at_depth": spread,
                                      "n_ki_rows_total": total}
        print(f"  probe {c['chembl_target_id']}: head median year "
              f"{per['head']['year_median']} vs deep "
              f"{per.get('deep', {}).get('year_median')} -> later at depth = {spread}")
    (out / "probe_year_spread.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res

def main() -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-mb", type=float, default=175.0,
                    help="hard cap on bytes downloaded in this run (cached pages are free); "
                         "the user-approved ceiling for this task was 200 MB")
    ap.add_argument("--screen", type=int, default=2000,
                    help="how many kinase-named human targets to count exact Ki rows for; the "
                         "whole named-kinase set is ~1.4 k, so the default screens all of them "
                         "rather than an arbitrary head slice of the API's id order")
    ap.add_argument("--max-targets", type=int, default=16, help="panel size")
    ap.add_argument("--rows-per-target", type=int, default=3000,
                    help="cap on exact-standard Ki rows fetched per target; where a target has "
                         "fewer rows than this the whole of it is taken, so the table is the "
                         "complete curated set for most panel members rather than a sample")
    ap.add_argument("--organism", default="Homo sapiens")
    ap.add_argument("--min-rows", type=int, default=300,
                    help="drop a panel target below this many usable rows")
    ap.add_argument("--skip-filter-verify", action="store_true",
                    help="reuse the cached filter-check pages only (never silently skips it)")
    ap.add_argument("--out-tag", default="",
                    help="suffix for the output directory, e.g. '--out-tag _plumbing' writes to "
                         "data/processed/external_plumbing/ so a smoke test cannot be mistaken "
                         "for the real table")
    ap.add_argument("--slice", choices=["head", "spread"], default="spread",
                    help="which rows of a target's result range to take when the cap is below "
                         "the total: 'head' = the first cap rows, 'spread' = evenly spaced "
                         "offsets across the whole range (see --probe-year-spread)")
    ap.add_argument("--probe-year-spread", action="store_true",
                    help="run the head-vs-deep publication-year test on the panel, write "
                         "probe_year_spread.json and stop without building the table")
    args = ap.parse_args()

    ledger = Ledger(args.budget_mb)
    t0 = time.time()
    out = ROOT / "data" / "processed" / ("external" + args.out_tag)
    out.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    # 1. prove the filters work, or refuse to continue: a silently-ignored ChEMBL filter
    #    returns the whole table, which would quietly turn "12 kinases" into "all of ChEMBL"
    if args.skip_filter_verify:
        cf = RAW / "filter_checks.json"
        if not cf.exists():
            print("filter checks were never cached; refusing to skip them", file=sys.stderr)
            return 2
        checks = json.loads(cf.read_text(encoding="utf-8"))
    else:
        checks = verify_filters(ledger)
    bad = [k for k, v in checks.items() if not v.get("applies")]
    (RAW / "filter_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("filter verification:")
    for k, v in checks.items():
        print(f"  {'OK  ' if v.get('applies') else 'FAIL'} {k}")
    if bad:
        print(f"refusing to continue; these filters did not verify: {bad}", file=sys.stderr)
        return 1

    # 2. candidates -> panel
    release = chembl_status(ledger)
    print(f"chembl release stamp: {release}")
    screen_stats = {"no_component_accession": 0, "count_requests_failed": 0}
    cands = kinase_candidates(ledger, args.screen, args.organism, screen_stats)
    print(f"kinase-named human single-protein targets screened: {len(cands)} "
          f"(dropped for no component accession: "
          f"{screen_stats['no_component_accession']})")
    cands = count_ki_rows(cands, ledger, t0, screen_stats)
    print(f"count requests that failed (those candidates are excluded, n_ki_rows=-1): "
          f"{screen_stats['count_requests_failed']}")
    usable = [c for c in cands if c["n_ki_rows"] >= args.min_rows]
    panel = sorted(usable, key=lambda c: -c["n_ki_rows"])[:args.max_targets]
    print(f"panel ({len(panel)} of {len(usable)} candidates with >= {args.min_rows} Ki rows):")
    planned_rows = 0
    for c in panel:
        planned_rows += min(c["n_ki_rows"], args.rows_per_target)
        print(f"  {c['chembl_target_id']:>14}  {(c['pref_name'] or '-')[:42]:44s} "
              f"{c['accession'] or '-':>10}  {c['n_ki_rows']:>7} Ki rows")
    if len(panel) < 3:
        print("need >= 3 targets for cold_target to mean anything; aborting", file=sys.stderr)
        return 1
    # a projection, not a promise: the hard cap is the ledger, which checks every response's
    # Content-Length before reading it. planned_rows is printed so an over-sized pull is
    # visible before it happens rather than as a truncated run afterwards.
    print(f"planned activity rows: {planned_rows}  slice={args.slice}  "
          f"(budget {args.budget_mb:.0f} MB, spent so far {ledger.bytes / 1e6:.2f} MB)")

    if args.probe_year_spread:
        probe = year_spread_probe(panel, ledger, out)
        print(f"probe written to {out / 'probe_year_spread.json'}; no table built "
              f"({ledger.requests} requests, {ledger.bytes / 1e6:.1f} MB, "
              f"{round(time.time() - t0, 1)} s)")
        return 0

    # 3. activities.  The row filters below are all local, on fields measured as present in the
    #    activity JSON; each one is counted, because "how much did we throw away" is the first
    #    thing a reviewer of an external validation set should be able to see.
    drop = {"data_validity": 0, "wrong_std_type": 0, "wrong_flag": 0, "censored_relation": 0,
            "non_numeric": 0, "non_positive_or_nan": 0, "wrong_units": 0, "dup_flagged": 0,
            "no_doc": 0, "no_smiles": 0, "row_no_smiles": 0, "bad_mol_id": 0}
    per_target: dict[str, list[dict]] = {}
    offsets_used: dict[str, list[int]] = {}
    dupe_removed: dict[str, int] = {}
    fetched_per_target: dict[str, int] = {}
    fetched_rows = 0
    for c in panel:
        rows, offs, n_dupe = fetch_activities(c["chembl_target_id"], args.rows_per_target,
                                              c["n_ki_rows"], ledger, mode=args.slice)
        offsets_used[c["chembl_target_id"]] = offs
        dupe_removed[c["chembl_target_id"]] = n_dupe
        fetched_per_target[c["chembl_target_id"]] = len(rows)
        fetched_rows += len(rows)
        keep = []
        for a in rows:
            if a.get("data_validity_comment"):
                drop["data_validity"] += 1
                continue
            if a.get("standard_type") != "Ki":
                # both filters were verified up front; a non-Ki row here means the page lied, so
                # drop it and count it rather than letting it into a Ki-only label
                drop["wrong_std_type"] += 1
                continue
            if str(a.get("standard_flag")) != "1":
                drop["wrong_flag"] += 1
                continue
            if a.get("standard_units") != "nM":
                drop["wrong_units"] += 1        # pKi = 9 - log10(nM) is only right on nM
                continue
            if a.get("standard_relation") not in (None, "="):
                # '>' / '<' is a censored measurement (Ki > 10000 is not Ki = 10000); treating it
                # as exact would bias the median, so it is dropped and counted
                drop["censored_relation"] += 1
                continue
            if a.get("potential_duplicate"):
                drop["dup_flagged"] += 1
                continue
            v = a.get("standard_value")
            try:
                ki = float(v)
            except (TypeError, ValueError):
                drop["non_numeric"] += 1
                continue
            if not (ki > 0):        # also catches NaN, inf and 0 -- pKi would be undefined
                drop["non_positive_or_nan"] += 1
                continue
            if not a.get("document_chembl_id"):
                drop["no_doc"] += 1
                continue
            smi = (a.get("canonical_smiles") or "").strip()
            if not smi:
                drop["no_smiles"] += 1
                continue
            if not a.get("molecule_chembl_id"):
                drop["bad_mol_id"] += 1
                continue
            keep.append({"mol": a["molecule_chembl_id"], "smiles": smi, "ki_nM": ki,
                         "doc": a["document_chembl_id"], "doc_year_row": a.get("document_year"),
                         "assay": a.get("assay_chembl_id"),
                         "std_type": a.get("standard_type"),
                         "act_id": a.get("activity_id")})
        per_target[c["chembl_target_id"]] = keep
        frac = (len(rows) / c["n_ki_rows"]) if c["n_ki_rows"] else float("nan")
        print(f"  {c['chembl_target_id']}: {len(rows)} distinct fetched of {c['n_ki_rows']} usable "
              f"({frac:.1%}){' COMPLETE' if len(rows) >= c['n_ki_rows'] else ' SAMPLED'}, "
              f"{len(keep)} kept"
              + (f", {n_dupe} duplicate rows dropped" if n_dupe else ""))
    print(f"dropped rows: {drop}")
    if ledger.exceeded:
        print("BUDGET EXCEEDED during activity fetch -- writing nothing complete",
              file=sys.stderr)

    # 4. ligands come from the canonical_smiles carried on each activity row (measured present
    #    on 1000/1000 rows).  A sample is then cross-checked against the authoritative
    #    /molecule records; agreement is recorded in the manifest rather than assumed.
    by_smiles: dict[str, list[str]] = {}
    for rows in per_target.values():
        for k in rows:
            by_smiles.setdefault(k["smiles"], []).append(k["mol"])
    smiles_list = sorted(by_smiles)
    lig_index = {s: f"XL{i:05d}" for i, s in enumerate(smiles_list)}
    chembl_per_smiles = {s: sorted(set(v)) for s, v in by_smiles.items()}
    print(f"ligands: {len(by_smiles)} unique canonical SMILES carried by "
          f"{sum(len(v) for v in by_smiles.values())} kept rows (chembl ids sharing one "
          f"canonical SMILES are one ligand here, so a duplicated structure cannot sit in "
          f"train and test at the same time)")

    all_mol_ids = sorted({k["mol"] for r in per_target.values() for k in r})
    rng = np.random.default_rng(20261104)
    sample_ids = sorted(rng.choice(all_mol_ids, size=min(300, len(all_mol_ids)),
                                   replace=False).tolist())
    molcheck = fetch_molecules(sample_ids, ledger)
    row_smi = {k["mol"]: k["smiles"] for r in per_target.values() for k in r}
    agree = disagree = missing_struct = 0
    for mid, rec in molcheck.items():
        auth = rec.get("canonical_smiles")
        if auth is None:
            missing_struct += 1
            continue
        if auth == row_smi.get(mid):
            agree += 1
        else:
            disagree += 1
    smiles_check = {"sampled": len(sample_ids), "records_returned": len(molcheck),
                    "authoritative_structures_present": len(molcheck) - missing_struct,
                    "exact_string_matches": agree, "mismatches": disagree,
                    "match_rate": (round(agree / max(1, agree + disagree), 4)
                                   if (agree + disagree) else None),
                    "note": "activity.canonical_smiles vs molecule_structures.canonical_smiles"}
    print(f"smiles cross-check: {agree} exact matches / {disagree} mismatches "
          f"of {len(molcheck)} records")

    # 5. documents (the temporal proxy).  Every distinct source document is fetched from
    #    /document/{id}; where the activity row also carried a denormalised document_year the two
    #    are compared, so the proxy is verified rather than trusted.
    docs = {}
    for rows in per_target.values():
        for k in rows:
            docs.setdefault(k["doc"], None)
    doc_ids = sorted(docs)
    for did in doc_ids:
        d = get_json(f"{API}/document/{did}.json", ledger, f"doc_{did}")
        if d:
            docs[did] = {"year": d.get("year"), "journal": d.get("journal"),
                         "doi": d.get("doi"), "publication_year": d.get("publication_year")}
    years = [v["year"] for v in docs.values() if v and v.get("year")]
    year_pairs = [(k["doc_year_row"], (docs.get(k["doc"]) or {}).get("year"))
                  for rows in per_target.values() for k in rows
                  if k["doc_year_row"] and (docs.get(k["doc"]) or {}).get("year")]
    year_agree = sum(1 for a, b in year_pairs if int(a) == int(b))
    year_check = {"rows_with_both_years": len(year_pairs), "agree": year_agree,
                  "disagree": len(year_pairs) - year_agree,
                  "note": "activity.document_year vs /document/{id}.year"}
    print(f"documents: {len(doc_ids)} requested, {len(years)} with a year; "
          f"year cross-check {year_agree}/{len(year_pairs)} agree")

    # 6. sequences -- a record counts only if it is the accession we asked for AND the organism
    #    we asked for; the manifest keeps the reject reasons so a silent mis-mapping is visible
    accs = sorted({c["accession"] for c in panel if c["accession"]})
    uni = fetch_uniprot(accs, ledger)
    seq_ok, seq_bad = {}, {}
    for a, v in uni.items():
        why = None
        if not v.get("sequence"):
            why = "no_sequence"
        elif not v.get("matches_requested_accession"):
            why = f"header_accession_is_{v.get('fasta_accession')}"
        elif args.organism and (v.get("organism") or "").lower() != args.organism.lower():
            why = f"organism_is_{v.get('organism')}"
        else:
            seq_ok[a] = v
            continue
        seq_bad[a] = why
    print(f"sequences: {len(seq_ok)}/{len(accs)} verified as {args.organism}")
    if seq_bad:
        print(f"  rejected: {seq_bad}")

    # ---- assemble the pair table (median Ki per ligand x target, as in the DAVIS columns)
    tgt_rows, pairs = [], []
    for c in panel:
        tid = c["accession"] if c["accession"] in seq_ok else None
        if tid is None:
            print(f"  skip {c['chembl_target_id']}: no verified sequence", file=sys.stderr)
            continue
        keep = per_target[c["chembl_target_id"]]
        by_lig: dict[str, list[dict]] = {}
        for k in keep:
            by_lig.setdefault(k["smiles"], []).append(k)
        rows = []
        for smi, ks in by_lig.items():
            kis = np.array([k["ki_nM"] for k in ks], dtype=float)
            # a document that failed to fetch is stored as None, so `or {}` -- `.get(k, {})`
            # would hand back that None and blow up on .get("year")
            yrs = [int(y) for y in ((docs.get(k["doc"]) or {}).get("year")
                                    or k["doc_year_row"] for k in ks) if y]
            med = float(np.median(kis))
            rows.append({"ligand_id": lig_index[smi], "ki_nM": round(med, 6),
                         "pki": round(9.0 - np.log10(med), 6), "n_rows": len(ks),
                         "n_chembl_ids": len({k["mol"] for k in ks}),
                         "n_docs": len({k["doc"] for k in ks}),
                         "first_year": min(yrs) if yrs else None,
                         "last_year": max(yrs) if yrs else None,
                         "ki_min_nM": round(float(kis.min()), 6),
                         "ki_max_nM": round(float(kis.max()), 6),
                         "ki_iqr_nM": round(float(np.subtract(*np.percentile(kis, [75, 25]))), 6)
                         if len(ks) > 1 else 0.0})
        tgt = {"uniprot": c["accession"], "chembl_target_id": c["chembl_target_id"],
               "target_name": c["pref_name"], "n_pairs_with_data": len(rows),
               "sequence": seq_ok[c["accession"]]["sequence"],
               "seq_len": seq_ok[c["accession"]]["seq_len"],
               "organism": seq_ok[c["accession"]]["organism"]}
        trow = len(tgt_rows)
        tgt["target_id"] = f"XT{trow:03d}"
        tgt_rows.append(tgt)
        for r in rows:
            r["target_id"] = tgt["target_id"]
            pairs.append(r)
    pdf = pd.DataFrame(pairs)
    if pdf.empty:
        print("no pairs assembled", file=sys.stderr)
        return 1
    pdf = pdf.sort_values(["target_id", "ligand_id"]).reset_index(drop=True)
    pdf.insert(0, "pair_id", np.arange(len(pdf)))

    used = sorted(set(pdf.ligand_id))
    inv = {v: k for k, v in lig_index.items()}
    ldf = pd.DataFrame([{"ligand_id": u, "canonical_smiles": inv[u],
                         "n_chembl_ids": len(chembl_per_smiles[inv[u]]),
                         "chembl_ids": "|".join(chembl_per_smiles[inv[u]])} for u in used])
    n_lig_orphans = len(lig_index) - len(used)
    tdf = pd.DataFrame(tgt_rows)[["target_id", "uniprot", "target_name", "chembl_target_id",
                                  "organism", "seq_len", "sequence", "n_pairs_with_data"]]
    ddf = pd.DataFrame([{"document_chembl_id": k, **(v or {})} for k, v in docs.items()])

    wall = round(time.time() - t0, 1)
    files = {}
    for name, df in (("chembl_ki.csv", pdf), ("ligands.csv", ldf),
                     ("targets.csv", tdf), ("documents.csv", ddf)):
        p = out / name
        df.to_csv(p, index=False)
        files[name] = {"path": p.relative_to(ROOT).as_posix(), "rows": int(len(df)),
                       "sha256": sha256_file(p)}

    in_panel = {c["chembl_target_id"] for c in panel}
    manifest = {
        "generated_by": "scripts/13_fetch_chembl.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": not ledger.exceeded,
        "source": "ChEMBL REST API (www.ebi.ac.uk/chembl/api/data) + UniProt REST fasta",
        "chembl_release": release,
        "params": vars(args),
        "filter_verification": checks,
        "selection_rule": (f"human SINGLE PROTEIN targets whose pref_name contains 'kinase', "
                           f"first {args.screen} in API order counted for exact-standard Ki rows "
                           f"(standard_type=Ki, standard_flag=1, standard_relation='='), "
                           f"panel = top {args.max_targets} by that count, "
                           f">= {args.min_rows} such rows required"),
        "row_sampling": {"mode": args.slice, "offsets_per_target": offsets_used,
                         "page": 1000,
                         "duplicate_activity_ids_removed": dupe_removed,
                         "coverage_per_target": {
                             c["chembl_target_id"]: {
                                 "usable_rows_reported": c["n_ki_rows"],
                                 "distinct_rows_fetched": fetched_per_target[c["chembl_target_id"]],
                                 "rows_kept_after_local_filters": len(per_target[c["chembl_target_id"]]),
                                 "complete": fetched_per_target[c["chembl_target_id"]] >= c["n_ki_rows"]}
                             for c in panel},
                         "note": "'spread' takes evenly spaced 1000-row pages across the "
                                 "target's own result range with spacing floored at one page so "
                                 "pages cannot overlap; 'head' takes the first pages. A target is "
                                 "'complete' when every usable row was fetched, otherwise the "
                                 "fetched/usable ratio is the sample fraction."},
        "screening": {"candidates_with_accession": len(cands),
                      "dropped_no_component_accession": screen_stats["no_component_accession"],
                      "count_requests_failed": screen_stats["count_requests_failed"],
                      "with_enough_ki_rows": len(usable),
                      "planned_activity_rows": planned_rows},
        "panel": [{"target_id": f"XT{i:03d}", "chembl_target_id": r["chembl_target_id"],
                   "pref_name": r["target_name"], "accession": r["uniprot"],
                   "seq_len": r["seq_len"], "organism": r["organism"],
                   "n_pairs_with_data": r["n_pairs_with_data"]}
                  for i, r in enumerate(tdf.to_dict("records"))],
        "panel_candidates": [{"chembl_target_id": c["chembl_target_id"],
                              "pref_name": c["pref_name"], "accession": c["accession"],
                              "n_ki_rows_reported": c["n_ki_rows"],
                              "in_panel": c["chembl_target_id"] in in_panel} for c in usable[:40]],
        "label_definition": "pKi = 9 - log10(median Ki / nM); same 9-log10(nM) scale as "
                            "DAVIS pKd, not comparable to KIBA's 0-17.2 consensus score",
        "aggregation": "median of the curated Ki rows for one (ligand, target) pair; "
                       "n_rows / ki_iqr_nM carried alongside so the aggregation is auditable",
        "temporal_proxy": "document year from /document/{id}.json (ChEMBL's public API has no "
                          "per-activity deposition date). This is a substitution, not the same "
                          "quantity as DAVIS's deposited date.",
        "row_filters_applied": {
            "data_validity_comment_non_null": drop["data_validity"],
            "standard_type_not_Ki": drop["wrong_std_type"],
            "standard_flag_not_1": drop["wrong_flag"],
            "standard_relation_censored": drop["censored_relation"],
            "standard_units_not_nM": drop["wrong_units"],
            "potential_duplicate": drop["dup_flagged"],
            "non_numeric_standard_value": drop["non_numeric"],
            "non_positive_or_nan_standard_value": drop["non_positive_or_nan"],
            "no_document_chembl_id": drop["no_doc"],
            "no_canonical_smiles_on_row": drop["no_smiles"],
            "no_molecule_chembl_id": drop["bad_mol_id"]},
        "sequence_verification": {"requested": len(accs), "verified": len(seq_ok),
                                  "rejected": seq_bad,
                                  "note": "a UniProt record is used only if the fasta header "
                                          "accession equals the one ChEMBL reported and the "
                                          "OS= field equals --organism"},
        "cross_checks": {"activity_smiles_vs_molecule_table": smiles_check,
                         "activity_year_vs_document_table": year_check},
        "counts": {"pairs": int(len(pdf)), "ligands": int(len(ldf)),
                   "targets": int(len(tdf)), "documents": int(len(ddf)),
                   "activity_rows_fetched_distinct": int(fetched_rows),
                   "activity_rows_kept": int(sum(len(v) for v in per_target.values())),
                   "distinct_chembl_ids_on_kept_rows": len(all_mol_ids),
                   "unique_canonical_smiles_on_kept_rows": len(lig_index),
                   "ligands_with_no_kept_target": int(n_lig_orphans),
                   "pairs_per_target": {r.target_id: int((pdf.target_id == r.target_id).sum())
                                        for r in tdf.itertuples()},
                   "targets_skipped_no_sequence": [c["chembl_target_id"] for c in panel
                                                   if c["accession"] not in seq_ok],
                   "pairs_with_multiple_rows": int((pdf.n_rows > 1).sum()),
                   "pairs_without_any_year": int(pdf.first_year.isna().sum()),
                   "year_min": int(min(years)) if years else None,
                   "year_max": int(max(years)) if years else None,
                   "pki_min": float(pdf.pki.min()), "pki_max": float(pdf.pki.max()),
                   "pki_mean": float(pdf.pki.mean()), "pki_sd": float(pdf.pki.std())},
        "transfer": {"http_requests": ledger.requests, "bytes_downloaded": ledger.bytes,
                     "cached_pages_reused": ledger.cached, "retries": ledger.retries,
                     "budget_mb": args.budget_mb, "budget_exceeded": ledger.exceeded,
                     "wall_seconds": wall,
                     "bytes_this_process_only": (
                         "bytes_downloaded counts this process only: pages already in "
                         "data/raw/chembl/ are served from the disk cache and cost nothing here, "
                         "so a re-run after a sampler change is nearly free while the first pull "
                         "of the panel paid the bytes. The download cap the user authorised is on "
                         "the whole job, so the size of everything actually held on disk is "
                         "reported next to it."),
                     "raw_cache_bytes_on_disk": sum(f.stat().st_size for f in RAW.glob("*")
                                                    if f.is_file())},
        "files": files,
        "raw_page_cache": "data/raw/chembl/ (gitignored); every page this run read is cached "
                          "and its sha256 is listed in raw_cache_sha256 below",
    }
    cache_files = sorted(RAW.glob("*"))
    manifest["raw_cache_sha256"] = {f"data/raw/chembl/{c.name}: {sha256_file(c)}"
                                    for c in cache_files if c.is_file()}
    manifest["raw_cache_sha256"] = sorted(manifest["raw_cache_sha256"])
    manifest["raw_cache_n_files"] = len(cache_files)
    (out / "external_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\nwrote {out}  (completed={not ledger.exceeded})")
    print(f"pairs {len(pdf)}  ligands {len(ldf)}  targets {len(tdf)}  "
          f"requests {ledger.requests}  {ledger.bytes / 1e6:.1f} MB  {wall} s")
    return 0 if not ledger.exceeded else 3


if __name__ == "__main__":
    sys.exit(main())
