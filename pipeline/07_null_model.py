"""
Degree-preserving null model for the reported network statistics.

The Guidelines require a variance, confidence interval or significance test on every
reported metric. Modularity Q and the cross-layer edge fraction were reported as bare
point estimates, which says nothing about whether either is unusual: a modularity of
0.61 is unremarkable in some degree sequences and extreme in others.

METHOD
------
Rewire the observed graph with networkx's double_edge_swap, which preserves every node's
degree exactly while randomizing which nodes are connected. Node modality labels stay
fixed, so the cross-layer fraction is free to vary. For each of N rewirings, recompute
Louvain modularity (same random seed as the main pipeline) and the cross-layer edge
fraction, then report where the observed value falls in that null distribution.

Two-sided empirical p, with the +1 correction so p is never reported as exactly zero.

Run from the pipeline/ directory:
    python 07_null_model.py [--draws 1000]
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
import networkx.algorithms.community as nx_comm

GRAPHML = Path(__file__).parent.parent / "GLASSO/results/ibd/ibd_joint_network.graphml"
OUT = Path(__file__).parent.parent / "GLASSO/results/ibd/null_model.json"
SEED = 42
CROSS = {"mgx-mbx", "mgx-rna", "mbx-rna"}


def abs_graph(H):
    A = nx.Graph()
    A.add_nodes_from(H.nodes(data=True))
    for u, v, d in H.edges(data=True):
        A.add_edge(u, v, weight=abs(float(d.get("weight", 1.0))))
    return A


def modularity_q(H):
    A = abs_graph(H)
    return nx_comm.modularity(A, louvain_communities(A, weight="weight", seed=SEED),
                              weight="weight")


def cross_fraction(H, mod):
    """Fraction of edges joining two different modalities. Recomputed from the modality
    labels rather than the stored edge_type attribute, which does not survive rewiring."""
    cross = sum(1 for u, v in H.edges() if mod[u] != mod[v])
    return cross / H.number_of_edges()


def two_sided_p(null, obs):
    null = np.asarray(null)
    centre = null.mean()
    return (np.sum(np.abs(null - centre) >= abs(obs - centre)) + 1) / (len(null) + 1)


def main(draws):
    G = nx.read_graphml(GRAPHML)
    mod = nx.get_node_attributes(G, "modality")
    obs_q = modularity_q(G)
    obs_x = cross_fraction(G, mod)
    print(f"observed: Q = {obs_q:.4f}, cross-layer fraction = {obs_x:.4f} "
          f"({int(round(obs_x * G.number_of_edges()))}/{G.number_of_edges()} edges)\n")

    rng = np.random.default_rng(SEED)
    qs, xs = [], []
    for i in range(draws):
        H = G.copy()
        # nswap = 10x edges is the usual rule of thumb for mixing; max_tries guards against
        # graphs where swaps are hard to find.
        nx.double_edge_swap(H, nswap=10 * H.number_of_edges(),
                            max_tries=200 * H.number_of_edges(),
                            seed=int(rng.integers(1e9)))
        qs.append(modularity_q(H))
        xs.append(cross_fraction(H, mod))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{draws} rewirings")

    res = {}
    for name, obs, null in [("modularity_Q", obs_q, qs), ("cross_layer_fraction", obs_x, xs)]:
        null = np.asarray(null)
        res[name] = {
            "observed": round(float(obs), 4),
            "null_mean": round(float(null.mean()), 4),
            "null_sd": round(float(null.std(ddof=1)), 4),
            "null_ci95": [round(float(np.percentile(null, 2.5)), 4),
                          round(float(np.percentile(null, 97.5)), 4)],
            "z": round(float((obs - null.mean()) / null.std(ddof=1)), 2),
            "p_two_sided": round(float(two_sided_p(null, obs)), 5),
            "draws": int(draws),
        }
        r = res[name]
        print(f"\n{name}")
        print(f"  observed   : {r['observed']}")
        print(f"  null       : {r['null_mean']} (sd {r['null_sd']}), "
              f"95% CI [{r['null_ci95'][0]}, {r['null_ci95'][1]}]")
        print(f"  z = {r['z']},  two-sided empirical p = {r['p_two_sided']}")

    # The check: a degree-preserving null must not change the degree sequence.
    H = G.copy()
    nx.double_edge_swap(H, nswap=10 * H.number_of_edges(),
                        max_tries=200 * H.number_of_edges(), seed=SEED)
    assert sorted(d for _, d in H.degree()) == sorted(d for _, d in G.degree()), \
        "rewiring changed the degree sequence"
    assert H.number_of_edges() == G.number_of_edges(), "rewiring changed the edge count"
    print("\ndegree sequence and edge count preserved under rewiring: OK")

    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    main(ap.parse_args().draws)
