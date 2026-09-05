"""
Permutation control for H1b.

Question it answers: the light-linked panel correlates strongly with the community
eigengene scores -- but is that special to the panel, or would ANY set of transcripts
do the same? The suspicion is that most of the signal is shared technical structure:
communities 3 and 4 are transcript-dominated, so a panel transcript is being correlated
against a summary of other transcripts from the same RNA-seq run on the same biopsy.

METHOD
------
Draw N_DRAWS random panels of the same size from the transcripts that were NOT
selected as network nodes, run the identical Spearman-against-eigengene test, and record
each panel's median |rho|. The real panel's median |rho| is then read off that null
distribution as an empirical p-value.

Also reports the same comparison restricted to the non-transcript communities, where the
same-assay artifact cannot operate.

Run from the pipeline/ directory:
    python 06_permutation_control.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import analysis_05_exposome as ex

N_DRAWS = 1000
RNG = np.random.default_rng(0)


def median_abs_rho(genes, rna, eig, communities):
    rhos = []
    for g in genes:
        for c in communities:
            paired = pd.concat([rna[g], eig[c]], axis=1).dropna()
            if paired.iloc[:, 0].std() == 0:
                continue
            rhos.append(abs(spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])[0]))
    return float(np.median(rhos)) if rhos else np.nan


def main():
    G, communities = ex.load_communities()
    modality_frames = {
        "microbiome": pd.read_csv(ex.MGX_PATH, index_col=0),
        "metabolite": pd.read_csv(ex.MBX_PATH, index_col=0),
        "transcript": pd.read_csv(ex.RNA_PATH, index_col=0),
    }
    eig = ex.community_eigengenes(G, communities, modality_frames)
    rna = pd.read_csv(ex.RNA_FULL_POOL_PATH, index_col=0).reindex(eig.index)

    # which communities are transcript-dominated?
    comp = []
    for i, nodes in enumerate(communities):
        mods = [G.nodes[n].get("modality") for n in nodes]
        frac = mods.count("transcript") / len(mods)
        comp.append((f"community_{i}", len(nodes), frac))
    print("Community composition (fraction transcript):")
    for name, size, frac in comp:
        print(f"  {name}: {size:3d} nodes, {frac:5.0%} transcript")

    tx_comms  = [n for n, _, f in comp if f >= 0.5]
    non_comms = [n for n, _, f in comp if f < 0.5]

    panel = [g for g in ex.LIGHT_LINKED_PANEL if g in rna.columns]
    network_nodes = set(G.nodes())
    candidates = [c for c in rna.columns if c not in network_nodes and c not in panel]
    print(f"\nPanel: {len(panel)} genes {panel}")
    print(f"Null pool: {len(candidates):,} transcripts not in the network and not in the panel")

    for label, comms in [("ALL communities", list(eig.columns)),
                         ("TRANSCRIPT-dominated only", tx_comms),
                         ("NON-transcript communities only", non_comms)]:
        if not comms:
            continue
        obs = median_abs_rho(panel, rna, eig, comms)
        null = np.array([
            median_abs_rho(RNG.choice(candidates, size=len(panel), replace=False),
                           rna, eig, comms)
            for _ in range(N_DRAWS)
        ])
        p = (np.sum(null >= obs) + 1) / (N_DRAWS + 1)
        print(f"\n=== {label} ({len(comms)} communities) ===")
        print(f"  real panel median |rho| : {obs:.3f}")
        print(f"  random panels           : median {np.median(null):.3f}, "
              f"90th pct {np.percentile(null, 90):.3f}, max {null.max():.3f}")
        print(f"  empirical p             : {p:.4f}   "
              f"({'panel beats random' if p < 0.05 else 'PANEL IS NOT SPECIAL'})")


if __name__ == "__main__":
    main()
