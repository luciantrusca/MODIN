"""
Literature-anchored RWR seed selection, replacing the hardcoded rwr_seeds
literal in network_metrics.ipynb cell 1 (uncited, silently edited on reruns).

METHOD
------
A network node is admitted as a seed only with evidence of IBD association
independent of this cohort:

  transcripts  Open Targets Platform association score for IBD (MONDO_0005265),
               aggregating GWAS/ChEMBL/expression/literature evidence across
               20+ sources. Seeds are the top-scoring network nodes.
  microbiome   PubMed co-occurrence screen over all network species, then
               evidence review of the ranked candidates; admitted only with
               direct IBD-specific experimental evidence (MGX_EVIDENCE below).
  metabolites  19/25 metabolite nodes have no confirmed chemical identity to
               look up at all. Of the ~6 with a putative name (MACARRoN match,
               Data/mbx_node_annotations.csv "Anchor" column), only arachidonate
               has direct compound-specific IBD literature and admits
               (MBX_EVIDENCE); N-acetylhistamine, malate, eicosatrienoate,
               eicosenoate and homovanillate were checked and found none.
               mbx_candidates() re-derives this against whichever network is
               loaded, since the metabolite node set can change on a rerun.

Every network node was selected using the IBD/non-IBD labels, so node-derived
seeds would make SQ2 circular in the same way node selection is -- external
seeds are not derived from this cohort at all.

Run from the pipeline/ directory:
    python 02_seed_selection.py            # screen only, prints tables
    python 02_seed_selection.py --write    # also writes seed_selection.csv
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import networkx as nx
import pandas as pd

GRAPHML = Path(__file__).parent.parent / "GLASSO/results/ibd/ibd_joint_network.graphml"
OUT = Path(__file__).parent.parent / "GLASSO/results/ibd/seed_selection.csv"
MBX_ANNOTATIONS = Path(__file__).parent / "Data/mbx_node_annotations.csv"

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"
IBD_MONDO = "MONDO_0005265"  # Open Targets migrated off EFO_0003767; that ID now returns null.

IBD_TIAB = ('("inflammatory bowel disease"[tiab] OR "Crohn disease"[tiab] OR '
            '"Crohn\'s disease"[tiab] OR "ulcerative colitis"[tiab] OR IBD[tiab])')

# Microbiome seeds admitted after evidence review of the PubMed screen. Direction of the
# association is recorded but does not gate admission: a species depleted in disease is as
# informative an anchor for network proximity as one that is enriched.
MGX_EVIDENCE = {
    "Clostridium_innocuum": {
        "direction": "enriched",
        "pmids": ["32991841", "34963635"],
        "note": "Signature organism translocating to mesenteric adipose and driving creeping "
                "fat in Crohn's disease (Ha et al. 2020, Cell); associated with reduced "
                "clinical remission in UC, 50% vs 87.5%, p=0.044 (Le et al. 2021, J Infect).",
    },
    "Roseburia_hominis": {
        "direction": "depleted",
        "pmids": ["24021287", "29018440"],
        "note": "Significantly reduced in UC versus controls, p<0.0001, inversely correlated "
                "with disease activity (Machiels et al. 2013, Gut); protective against DSS "
                "colitis on mono-colonization (Patterson et al. 2017, Front Immunol).",
    },
}

# Metabolite tier 1 (see THE RULE above): the compound itself, not just its chemical family,
# has direct IBD literature. Keyed by the putative name in the annotation crosswalk's "Anchor"
# column, not by feature ID, since feature IDs are cohort-specific and the annotation is
# regenerated on every rerun -- a name here matches whichever feature ID currently carries it.
MBX_EVIDENCE = {
    "arachidonate": {
        "direction": "enriched",
        "pmids": ["3665357", "40362272"],
        "note": "Elevated in colonic mucosal phospholipids in both UC (19+/-4) and Crohn's "
                "(20+/-3) versus controls (13+/-5 ug/mg protein) (Pacheco et al. 1987, "
                "Clinical Science, PMID 3665357); fecal arachidonic acid positively "
                "correlates with fecal calprotectin, the standard clinical disease-activity "
                "marker (Huss et al. 2025, Int J Mol Sci, PMID 40362272) -- the more directly "
                "relevant of the two, since this cohort's MBX layer is also fecal.",
    },
}
# Checked, found NO compound-specific IBD evidence: N-acetylhistamine (only
# pathway-level histamine-metabolism evidence, and for a different histamine metabolite,
# N-methylhistamine, not this one), malate (mentioned only as one of many generic TCA-cycle
# intermediates altered in IBD, no compound-specific finding), eicosatrienoate and eicosenoate
# (only the unrelated, better-known fatty acid EPA has direct evidence), homovanillate (no IBD
# link found at all -- its documented gut associations are to depression and IBS, not IBD).


def gql(query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(OT_API, data=body,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def open_targets_scores(symbols):
    """IBD association score per gene symbol. Absent from the result means no recorded
    association, which is a meaningful negative, not a lookup failure."""
    q = ("query($d:String!,$i:Int!,$s:Int!){ disease(efoId:$d){ "
         "associatedTargets(page:{index:$i,size:$s}){ count rows{ target{approvedSymbol} score } } } }")
    total = gql(q, {"d": IBD_MONDO, "i": 0, "s": 1})["data"]["disease"]["associatedTargets"]["count"]
    wanted, found, idx, size = set(symbols), {}, 0, 500
    while idx * size < total and len(found) < len(wanted):
        rows = gql(q, {"d": IBD_MONDO, "i": idx, "s": size})["data"]["disease"]["associatedTargets"]["rows"]
        if not rows:
            break
        for offset, row in enumerate(rows):
            sym = row["target"]["approvedSymbol"]
            if sym in wanted and sym not in found:
                found[sym] = (row["score"], idx * size + offset + 1)
        idx += 1
        time.sleep(0.2)
    return found, total


def pubmed_count(term):
    p = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmode": "json", "retmax": 0})
    u = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + p
    return int(json.load(urllib.request.urlopen(u, timeout=45))["esearchresult"]["count"])


def pubmed_screen(species):
    rows = []
    for s in species:
        name = s.replace("_", " ")
        total = pubmed_count(f'"{name}"[tiab]')
        time.sleep(0.4)
        ibd = pubmed_count(f'"{name}"[tiab] AND {IBD_TIAB}')
        time.sleep(0.4)
        rows.append({"node": s, "ibd_papers": ibd, "total_papers": total,
                     "ibd_fraction": round(ibd / total, 3) if total else 0.0})
    return pd.DataFrame(rows).sort_values("ibd_papers", ascending=False)


def mbx_candidates(G):
    """Cross-references the network's metabolite nodes against the annotation crosswalk's
    putative names (Anchor column), then against MBX_EVIDENCE. Returns a DataFrame with one
    row per network metabolite node that has a putative name at all -- most will show
    admitted=False, tier=NaN, since only arachidonate currently has direct evidence (see
    MBX_EVIDENCE for the five checked-and-rejected names)."""
    mbx = sorted(n for n in G if nx.get_node_attributes(G, "modality").get(n) == "metabolite")
    if not MBX_ANNOTATIONS.exists():
        print(f"\n{MBX_ANNOTATIONS} not found -- metabolite tier skipped (annotation crosswalk "
              "must be regenerated by 01a_mbx_annotation.ipynb before this can run).")
        return pd.DataFrame(columns=["node", "putative_name", "admitted", "tier", "pmids", "note"])

    anno = pd.read_csv(MBX_ANNOTATIONS, index_col="feature_id")
    rows = []
    for node in mbx:
        name = anno["Anchor"].get(node) if node in anno.index else None
        if pd.isna(name):
            continue
        ev = MBX_EVIDENCE.get(name)
        rows.append({
            "node": node, "putative_name": name, "admitted": ev is not None,
            "tier": 1 if ev else None,
            "pmids": ";".join(ev["pmids"]) if ev else "",
            "direction": ev["direction"] if ev else "",
            "note": ev["note"] if ev else "checked, no compound-specific IBD evidence found",
        })
    return pd.DataFrame(rows)


def main(write):
    G = nx.read_graphml(GRAPHML)
    mod = nx.get_node_attributes(G, "modality")
    htx = sorted(n for n in G if mod.get(n) == "transcript")
    mgx = sorted(n for n in G if mod.get(n) == "microbiome")

    scores, total = open_targets_scores(htx)
    ot = (pd.DataFrame([{"node": g, "ot_score": v[0], "ot_rank": v[1]} for g, v in scores.items()])
          .sort_values("ot_score", ascending=False))
    print(f"\nOpen Targets, IBD ({IBD_MONDO}): {total} associated targets; "
          f"{len(ot)}/{len(htx)} transcript nodes have a recorded association")
    print(ot.head(8).to_string(index=False))
    print("no recorded association:", sorted(set(htx) - set(scores)) or "none")

    screen = pubmed_screen(mgx)
    print(f"\nPubMed IBD co-occurrence screen, {len(mgx)} microbiome nodes (top 8):")
    print(screen.head(8).to_string(index=False))
    print("\nRaw counts do not rank on their own: a widely studied organism accumulates IBD "
          "co-mentions in proportion to its overall literature. Admission is by evidence "
          "review of the ranked candidates, recorded in MGX_EVIDENCE.")

    mbx = mbx_candidates(G)
    if len(mbx):
        print(f"\nMetabolite tier: {len(mbx)}/{sum(1 for n in G if mod.get(n)=='metabolite')} "
              "network nodes have a putative name to check at all")
        print(mbx[["node", "putative_name", "admitted", "tier"]].to_string(index=False))
    mbx_seeds = list(mbx.loc[mbx.admitted, "node"]) if len(mbx) else []

    seeds = list(ot.head(2).node) + list(MGX_EVIDENCE) + mbx_seeds
    missing = [s for s in seeds if s not in G]
    assert not missing, f"selected seed absent from network: {missing}"
    print(f"\nSEEDS ({len(seeds)}, within the three-to-five rule): {seeds}")

    out = pd.concat([
        ot.assign(layer="transcript", admitted=ot.node.isin(seeds), basis="open_targets_score"),
        screen.rename(columns={"node": "node"}).assign(
            layer="microbiome", admitted=screen.node.isin(seeds),
            basis=screen.node.map(lambda n: "pubmed_screen+evidence_review" if n in MGX_EVIDENCE else "pubmed_screen"),
            evidence_pmids=screen.node.map(lambda n: ";".join(MGX_EVIDENCE[n]["pmids"]) if n in MGX_EVIDENCE else ""),
            direction=screen.node.map(lambda n: MGX_EVIDENCE[n]["direction"] if n in MGX_EVIDENCE else ""),
        ),
        mbx.assign(
            layer="metabolite",
            basis=mbx["tier"].map(lambda t: "literature_review_tier1" if t == 1 else "checked_not_admitted"),
            evidence_pmids=mbx["pmids"],
        ) if len(mbx) else pd.DataFrame(),
    ], ignore_index=True)
    if write:
        out.to_csv(OUT, index=False)
        print(f"wrote {OUT}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
