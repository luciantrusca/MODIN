"""
run_ibd.py — GLASSO network inference on IBD microbiome + metabolomics +
host transcriptomics data.

Runs piglasso (stability-based graphical LASSO) on the MGX, MBX, and RNA
layers with a zero prior matrix (no STRING priors; features are species +
metabolites + transcripts, not genes). Saves precision matrices, edge lists,
and a combined NetworkX graph.

Run from the GLASSO root directory with the monika conda environment:
    cd /path/to/GLASSO
    conda run -n monika python src/run_ibd.py
"""

import os
import sys
import rpy2.situation

# Must set R_HOME before any rpy2 import so it uses the conda R (which has glasso)
os.environ["R_HOME"] = rpy2.situation.get_r_home()

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
os.chdir(project_dir)

import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx

from piglasso import QJSweeper
from estimate_lambdas import estimate_lambda_np, find_all_knee_points
from evaluation_of_graph import optimize_graph

# All IBD outputs go under results/ibd/ — keeps them separate from the CRC/glioma runs
RESULTS_DIR = "results/ibd"
os.makedirs(f"{RESULTS_DIR}/inferred_adjacencies", exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
Q       = 200    # sub-samples (lower than default 1000 given n=90)
B_PERC  = 0.65   # sub-sample size as fraction of n
LLO     = 0.01   # lambda range lower bound
LHI     = 1.5    # lambda range upper bound
LAMLEN  = 500    # number of lambda values
END_SLICE = 250  # trims high-lambda tail (low-density regime)
SEED    = 42
# Stability-based knee-point selection (StARS-adapted): summed binomial
# edge-stability score over the Q x LAMLEN subsample sweep, lambda taken at
# the knee of the stability curve. A fixed override is not used here since
# any single value would be tuned to one n, p regime and this cohort's size
# has changed across reruns.
LAMBDA_NP_OVERRIDE = None


def load_and_scale(path):
    # Data is already z-score standardized (ddof=1, mean 0 / var 1 per
    # feature) by pipeline.ipynb Step 5. Re-standardizing here with ddof=0
    # was a redundant, uniform sqrt((n-1)/n) rescale of every entry, which
    # shrinks the empirical covariance by (n-1)/n and so raises the
    # *effective* lambda_np by a factor of n/(n-1) relative to the fixed
    # LAMBDA_NP_OVERRIDE above — for n=90 this measurably shifted which
    # edges cleared the threshold (verified: same total edge count, but 4
    # edges swapped support, and shared-edge weights shifted ~2% on average).
    df = pd.read_csv(path, index_col=0)
    arr = df.values.astype(float)
    return df, arr


def run_layer(name, arr, feature_names):
    n, p = arr.shape
    b = int(B_PERC * n)
    prior_matrix = np.zeros((p, p))

    print(f"\n{'='*50}")
    print(f"Layer: {name}  |  n={n}, p={p}, Q={Q}, b={b}")
    print(f"{'='*50}")

    if LAMBDA_NP_OVERRIDE is not None:
        # Edge-count sweep (Q x LAMLEN graphical LASSO fits) is only needed for
        # knee-point selection, so it is skipped entirely when lambda is fixed.
        lambda_np = LAMBDA_NP_OVERRIDE
        print(f"Using manual λ = {lambda_np} (knee-point bypassed, edge-count sweep skipped)")
    else:
        lambda_range = np.linspace(LLO, LHI, LAMLEN)
        cache_file = f"{RESULTS_DIR}/{name}_edge_counts_Q{Q}.pkl"

        if os.path.exists(cache_file):
            print(f"Loading cached edge counts from {cache_file}")
            with open(cache_file, "rb") as f:
                edge_counts_all = pickle.load(f)
        else:
            sweeper = QJSweeper(arr, prior_matrix, b, Q, rank=1, size=1, seed=SEED)
            edge_counts_all, _ = sweeper.run_subsample_optimization(lambda_range)
            with open(cache_file, "wb") as f:
                pickle.dump(edge_counts_all, f)
            print(f"Edge counts saved to {cache_file}")

        sliced = edge_counts_all[:, :, :-END_SLICE]
        new_lamlen = sliced.shape[2]
        new_lhi = LLO + (LHI - LLO) * (new_lamlen - 1) / (LAMLEN - 1)
        lam_range = np.linspace(LLO, new_lhi, new_lamlen)
        (_, _, _, l_lo_idx, knee_idx, l_hi_idx) = find_all_knee_points(lam_range, sliced)
        select_lam = lam_range[l_lo_idx:l_hi_idx]
        select_ec  = sliced[:, :, l_lo_idx:l_hi_idx]
        lambda_np, _ = estimate_lambda_np(select_ec, Q, select_lam)
        print(f"Knee-point λ = {lambda_np:.4f}")

    lambda_wp = 0  # no prior

    # Persist the selected lambda, both so a rerun does not have to re-derive it
    # from the knee-point sweep and so the value actually used is recorded
    # somewhere other than stdout.
    lambda_path = f"{RESULTS_DIR}/{name}_selected_lambda.json"
    with open(lambda_path, "w") as f:
        json.dump({"layer": name, "lambda_np": float(lambda_np),
                    "lambda_wp": float(lambda_wp)}, f, indent=2)
    print(f"Selected lambda saved to {lambda_path}")

    precision_mat, edge_count, density = optimize_graph(
        arr, prior_matrix, lambda_np, lambda_wp, verbose=True
    )

    print(f"Edges inferred: {edge_count}  |  Density: {density:.4f}")

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(feature_names)
    for i in range(p):
        for j in range(i + 1, p):
            if precision_mat[i, j] != 0:
                G.add_edge(feature_names[i], feature_names[j],
                           weight=float(precision_mat[i, j]))

    # Save outputs
    np.save(f"{RESULTS_DIR}/inferred_adjacencies/{name}_precision.npy", precision_mat)
    edges_df = nx.to_pandas_edgelist(G)
    edges_df.to_csv(f"{RESULTS_DIR}/{name}_edges.csv", index=False)
    print(f"Saved precision matrix and edge list for {name}")

    return G, precision_mat, feature_names


def main():
    mgx_df, mgx_arr = load_and_scale("data/mgx_for_pig_ibd.csv")
    mbx_df, mbx_arr = load_and_scale("data/mbx_for_pig_ibd.csv")
    rna_df, rna_arr = load_and_scale("data/rna_for_pig_ibd.csv")

    mgx_features = mgx_df.columns.tolist()
    mbx_features = mbx_df.columns.tolist()
    rna_features = rna_df.columns.tolist()
    all_features  = mgx_features + mbx_features + rna_features

    # ── Joint run on concatenated MGX + MBX + RNA matrix ──────────────────────
    # Running on the joint matrix produces a 75×75 precision matrix whose
    # off-diagonal blocks are the interlayer species–metabolite–transcript
    # conditional dependencies. Samples are matched at the participant level
    # (n=90 > p=75), the margin the 25-per-layer node budget was chosen to keep.
    joint_arr = np.hstack([mgx_arr, mbx_arr, rna_arr])
    print(f"Joint matrix shape: {joint_arr.shape}  (samples × [MGX + MBX + RNA features])")

    G_joint, prec_joint, _ = run_layer("joint", joint_arr, all_features)

    # ── Tag nodes and edges by type ───────────────────────────────────────────
    for node in mgx_features:
        G_joint.nodes[node]["modality"] = "microbiome"
    for node in mbx_features:
        G_joint.nodes[node]["modality"] = "metabolite"
    for node in rna_features:
        G_joint.nodes[node]["modality"] = "transcript"

    mgx_set = set(mgx_features)
    mbx_set = set(mbx_features)
    rna_set = set(rna_features)
    for u, v in G_joint.edges():
        if u in mgx_set and v in mgx_set:
            G_joint[u][v]["edge_type"] = "mgx-mgx"
        elif u in mbx_set and v in mbx_set:
            G_joint[u][v]["edge_type"] = "mbx-mbx"
        elif u in rna_set and v in rna_set:
            G_joint[u][v]["edge_type"] = "rna-rna"
        elif (u in mgx_set and v in rna_set) or (u in rna_set and v in mgx_set):
            G_joint[u][v]["edge_type"] = "mgx-rna"
        elif (u in mbx_set and v in rna_set) or (u in rna_set and v in mbx_set):
            G_joint[u][v]["edge_type"] = "mbx-rna"
        else:
            G_joint[u][v]["edge_type"] = "mgx-mbx"

    # ── Summary ───────────────────────────────────────────────────────────────
    edge_types = nx.get_edge_attributes(G_joint, "edge_type")
    from collections import Counter
    counts = Counter(edge_types.values())
    print(f"\nEdge breakdown:")
    for etype, n in sorted(counts.items()):
        print(f"  {etype}: {n}")

    nx.write_graphml(G_joint, f"{RESULTS_DIR}/ibd_joint_network.graphml")
    edges_df = nx.to_pandas_edgelist(G_joint)
    edges_df.to_csv(f"{RESULTS_DIR}/ibd_joint_edges.csv", index=False)

    print(f"\nJoint network: {G_joint.number_of_nodes()} nodes, "
          f"{G_joint.number_of_edges()} edges")
    print(f"Saved: {RESULTS_DIR}/ibd_joint_network.graphml")
    print(f"Saved: {RESULTS_DIR}/ibd_joint_edges.csv")


if __name__ == "__main__":
    main()
