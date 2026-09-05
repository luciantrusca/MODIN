"""
IBD vs non-IBD comparison on the single shared topology (SQ1 centrality, SQ2 diffusion).

Replaces a two-network design (a separate graph inferred for the 22 non-IBD
participants) that was rejected: at n=22 against p=75, the two graphs would
differ mostly through estimation noise, making every downstream difference
uninterpretable.

METHOD
------
Topology is fixed -- both subgroups use the same 454 edges, inferred once on
the pooled 90 participants. Only EDGE WEIGHTS are subgroup-specific
(|Pearson r| between endpoint features within that subgroup). These are
MARGINAL correlations, not partial ones -- a subgroup-specific precision
matrix is not estimable at n=22 with 75 features even with the sparsity
pattern fixed, so state this as a limitation; do not describe subgroup
weights as conditional dependencies. The question becomes "given the same
wiring, is it arranged differently in the two groups", which is what SQ1
and SQ2 actually ask.

Significance: permutation test, diagnosis labels shuffled across all 90
participants with the 68/22 split preserved, whole comparison recomputed
each time. Per-node p-values get Benjamini-Hochberg correction.

LIMITATIONS
-----------
All 75 nodes were selected because they differ between these same two
groups, so a subgroup difference here is expected by construction -- this
describes how already disease-selected features are organized differently
by group, not independent evidence that they are IBD-relevant. (The RWR
seeds are the exception -- they come from outside the cohort.)

Run from the pipeline/ directory:
    python 04_subgroup_comparison.py [--perms 1000]
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).parent.parent
GRAPHML = BASE / "GLASSO/results/ibd/ibd_joint_network.graphml"
LABELS = BASE / "GLASSO/data/ibd_diagnosis_labels.csv"
LAYERS = ["mgx_for_pig_ibd.csv", "mbx_for_pig_ibd.csv", "rna_for_pig_ibd.csv"]
SEEDS_CSV = BASE / "GLASSO/results/ibd/seed_selection.csv"
OUT_NODES = BASE / "GLASSO/results/ibd/subgroup_node_comparison.csv"
OUT_SUMMARY = BASE / "GLASSO/results/ibd/subgroup_comparison.json"
SEED = 42
RESTART = 0.3


def load_data():
    X = pd.concat([pd.read_csv(BASE / "GLASSO/data" / f, index_col=0) for f in LAYERS], axis=1)
    y = pd.read_csv(LABELS, index_col=0)["diagnosis"]
    assert list(X.index) == list(y.index), "participant order differs between matrix and labels"
    G = nx.read_graphml(GRAPHML)
    missing = [n for n in G.nodes() if n not in X.columns]
    assert not missing, f"network nodes absent from the feature matrix: {missing[:5]}"
    seeds = sorted(pd.read_csv(SEEDS_CSV).query("admitted")["node"])
    return X, y, G, seeds


def weighted_copy(G, X, rows):
    """Same edges, weights re-estimated as |Pearson r| within the given participants."""
    sub = X.loc[rows]
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v in G.edges():
        r = np.corrcoef(sub[u].values, sub[v].values)[0, 1]
        H.add_edge(u, v, weight=0.0 if np.isnan(r) else abs(float(r)))
    return H


def composite_rank(H):
    """Mean of degree / betweenness / eigenvector rank. Lower = more central."""
    dc = nx.degree_centrality(H)
    dist = nx.Graph()
    dist.add_nodes_from(H.nodes())
    for u, v, d in H.edges(data=True):
        w = d["weight"]
        dist.add_edge(u, v, weight=(1.0 / w) if w > 0 else 1e6)
    bc = nx.betweenness_centrality(dist, weight="weight", normalized=True)
    try:
        ec = nx.eigenvector_centrality(H, weight="weight", max_iter=2000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        ec = nx.eigenvector_centrality_numpy(H, weight="weight")
    df = pd.DataFrame({"dc": dc, "bc": bc, "ec": ec})
    return sum(df[c].rank(ascending=False) for c in df.columns) / 3.0


def rwr_rank(H, seeds):
    p = {n: (1.0 / len(seeds) if n in seeds else 0.0) for n in H.nodes()}
    s = pd.Series(nx.pagerank(H, alpha=1 - RESTART, personalization=p, weight="weight"))
    return s.rank(ascending=False)


def contrast(X, y, G, seeds, ibd_rows, non_rows):
    Hi, Hn = weighted_copy(G, X, ibd_rows), weighted_copy(G, X, non_rows)
    ci, cn = composite_rank(Hi), composite_rank(Hn)
    ri, rn = rwr_rank(Hi, seeds), rwr_rank(Hn, seeds)
    return ci, cn, ri, rn


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, p[i] * m / k)
        q[i] = prev
    return q


def main(perms):
    X, y, G, seeds = load_data()
    ibd = list(y.index[y == "IBD"])
    non = list(y.index[y == "nonIBD"])
    print(f"nodes {G.number_of_nodes()}, edges {G.number_of_edges()}, "
          f"IBD {len(ibd)}, non-IBD {len(non)}, seeds {seeds}\n")

    ci, cn, ri, rn = contrast(X, y, G, seeds, ibd, non)
    obs_cent_shift = (ci - cn).abs()
    obs_rwr_shift = (ri - rn).abs()
    obs_cent_rho = spearmanr(ci, cn).statistic
    obs_rwr_rho = spearmanr(ri, rn).statistic
    print(f"centrality rank correlation IBD vs non-IBD : rho = {obs_cent_rho:.4f}")
    print(f"diffusion  rank correlation IBD vs non-IBD : rho = {obs_rwr_rho:.4f}\n")

    rng = np.random.default_rng(SEED)
    allp = list(X.index)
    null_cent_rho, null_rwr_rho = [], []
    ge_cent = np.zeros(len(ci))
    ge_rwr = np.zeros(len(ri))
    for i in range(perms):
        perm = list(rng.permutation(allp))
        pi, pn = perm[:len(ibd)], perm[len(ibd):]
        a, b, c, d = contrast(X, y, G, seeds, pi, pn)
        null_cent_rho.append(spearmanr(a, b).statistic)
        null_rwr_rho.append(spearmanr(c, d).statistic)
        ge_cent += ((a - b).abs().reindex(ci.index).values >= obs_cent_shift.values)
        ge_rwr += ((c - d).abs().reindex(ri.index).values >= obs_rwr_shift.values)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{perms} permutations")

    p_cent = (ge_cent + 1) / (perms + 1)
    p_rwr = (ge_rwr + 1) / (perms + 1)
    nodes = pd.DataFrame({
        "modality": pd.Series(nx.get_node_attributes(G, "modality")).reindex(ci.index),
        "cent_rank_IBD": ci.round(2), "cent_rank_nonIBD": cn.round(2),
        "cent_shift": obs_cent_shift.round(2), "cent_p": p_cent.round(4),
        "cent_q": bh(p_cent).round(4),
        "rwr_rank_IBD": ri.astype(int), "rwr_rank_nonIBD": rn.astype(int),
        "rwr_shift": obs_rwr_shift.astype(int), "rwr_p": p_rwr.round(4),
        "rwr_q": bh(p_rwr).round(4),
        "is_seed": [n in seeds for n in ci.index],
    }).sort_values("cent_q")

    def emp_p(null, obs):
        null = np.asarray(null)
        return (np.sum(null <= obs) + 1) / (len(null) + 1)

    summary = {
        "n_ibd": len(ibd), "n_nonibd": len(non), "permutations": perms,
        "edges": G.number_of_edges(), "seeds": seeds,
        "centrality_rho": round(float(obs_cent_rho), 4),
        "centrality_null_rho_mean": round(float(np.mean(null_cent_rho)), 4),
        "centrality_p": round(float(emp_p(null_cent_rho, obs_cent_rho)), 5),
        "diffusion_rho": round(float(obs_rwr_rho), 4),
        "diffusion_null_rho_mean": round(float(np.mean(null_rwr_rho)), 4),
        "diffusion_p": round(float(emp_p(null_rwr_rho, obs_rwr_rho)), 5),
        "n_nodes_cent_q_lt_05": int((nodes.cent_q < 0.05).sum()),
        "n_nodes_rwr_q_lt_05": int((nodes.rwr_q < 0.05).sum()),
    }

    print("\n=== GLOBAL ===")
    for k, v in summary.items():
        if k not in ("seeds",):
            print(f"  {k:28s} {v}")
    print("\n=== TOP 12 NODES BY CENTRALITY REORGANIZATION ===")
    print(nodes.head(12).to_string())

    nodes.to_csv(OUT_NODES)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_NODES}\nwrote {OUT_SUMMARY}")
    return nodes, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    main(ap.parse_args().perms)
