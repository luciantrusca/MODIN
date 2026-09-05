"""
Feature-selection stability check (rank-and-cap robustness).

Does NOT fix the double-dipping problem (Methodology, Limitations of feature
selection): the same IBD/non-IBD labels still drive both selection and every
downstream group comparison. This instead answers a separate question: is
the top-25-per-layer list itself stable, or would a slightly different set
of 90 participants have produced a different list?

METHOD
------
Re-run the same differential ranking used in pipeline.ipynb on many random
80% subsamples of the participants (stratified to keep roughly the real
68/22 IBD/non-IBD split), and record how often each feature lands in the
top 25. A feature selected in most resamples is stable; one that only shows
up occasionally was borderline in the original ranking too.

MGX and MBX: the identical two-stage selection used in pipeline.ipynb --
Mann-Whitney U + BH-FDR, gate on padj<0.05, rank the gated set by absolute
rank-biserial effect size (falling back to plain padj ranking if fewer than
the node budget clear the gate). Earlier versions ranked by padj alone; that
mismatched pipeline.ipynb once thousands of MBX features became significant
simultaneously, where padj ordering is dominated by sampling noise rather
than effect size -- producing a spurious 0/25 stability result that was
diagnosing the wrong selection rule, not the data.

HTX: the real selection step fits DESeq2 (~1 hour per fit; 200 resamples of
that is infeasible). This ranks HTX genes by Mann-Whitney U on the already
variance-stabilized (VST) values instead, as a fast proxy for the DESeq2
Wald-test ranking -- an approximation, not a re-run of the original test,
and should be reported as such if these numbers make it into the thesis.

Run from the pipeline/ directory:
    python 09_feature_selection_stability.py            # run on real data
    python 09_feature_selection_stability.py --self-test  # verify logic only
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

N_RESAMPLES = 200          # matches the StARS subsample count used for lambda selection
SUBSAMPLE_FRAC = 0.8
NODE_BUDGET = 25
STABLE_THRESHOLD = 0.7      # re-selected in >=70% of resamples counts as stable
SEED = 0

META_PATH = "Data/hmp2_metadata_2018-08-20.csv"
GROUP_MAP = {"CD": "IBD", "UC": "IBD", "nonIBD": "nonIBD"}

LAYERS = {
    "MGX": "Data/mgx_species_clr_full_pool.csv",
    "MBX": "Data/mbx_metabolomics_log_full_pool.csv",
    "HTX": "Data/rna_transcriptomics_vst_full_pool.csv",
}


def load_diagnosis_groups(participant_ids):
    meta = pd.read_csv(META_PATH, usecols=["Participant ID", "diagnosis"])
    pid_diag = meta.drop_duplicates("Participant ID").set_index("Participant ID")["diagnosis"]
    return pid_diag.reindex(participant_ids).map(GROUP_MAP)


def mannwhitney_padj_and_effect(pool_df, ibd_group):
    """Mann-Whitney U + BH-FDR per feature, plus the absolute rank-biserial
    correlation: effect = |2*U/(n1*n2) - 1|, the effect size native to the
    Mann-Whitney U statistic, bounded [0, 1] and independent of sample size.
    Returns (padj series, effect series)."""
    ibd_ids = ibd_group[ibd_group == "IBD"].index
    non_ids = ibd_group[ibd_group == "nonIBD"].index
    n1, n2 = len(ibd_ids), len(non_ids)
    pvals, effects = [], []
    for col in pool_df.columns:
        a = pool_df.loc[ibd_ids, col].values
        b = pool_df.loc[non_ids, col].values
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        pvals.append(p)
        effects.append(abs(2.0 * u / (n1 * n2) - 1.0))
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")
    return (pd.Series(padj, index=pool_df.columns),
            pd.Series(effects, index=pool_df.columns))


def rank_by_mannwhitney(pool_df, ibd_group):
    """Ascending-padj ranking only (drops the effect size). Used as a fast
    proxy for the HTX DESeq2 Wald test (see module docstring) -- DESeq2's Wald
    p-value is already a ratio of log2 fold change to its standard error, so it
    incorporates effect size directly and needs no separate effect-size stage,
    unlike the rank-based Mann-Whitney p-value used for MGX/MBX (see
    two_stage_top below). Thin wrapper over mannwhitney_padj_and_effect so the
    two functions share one Mann-Whitney + BH-FDR loop rather than each
    re-running it."""
    padj, _ = mannwhitney_padj_and_effect(pool_df, ibd_group)
    return padj.sort_values()


def two_stage_top(pool_df, ibd_group, budget, padj_gate=0.05):
    """Two-stage rank and cap, matching pipeline.ipynb's MGX/MBX selection cells
    (pipeline.ipynb cell 5 for MGX, cell 9 for MBX -- kept in sync by hand since
    a notebook cell can't be imported from a .py module): gate on padj<padj_gate, rank the gated set
    by |rank-biserial effect size| descending, take the top `budget`. Falls back
    to plain ascending-padj ranking (the old, single-stage rule) if fewer than
    `budget` features clear the gate -- same fallback pipeline.ipynb uses."""
    padj, effect = mannwhitney_padj_and_effect(pool_df, ibd_group)
    sig = padj[padj < padj_gate].index
    if len(sig) < budget:
        return padj.sort_values().index[:budget]
    return effect.loc[sig].sort_values(ascending=False).index[:budget]


def stratified_subsample(ibd_group, frac, rng):
    ibd_ids = ibd_group[ibd_group == "IBD"].index
    non_ids = ibd_group[ibd_group == "nonIBD"].index
    n_ibd = max(2, round(len(ibd_ids) * frac))
    n_non = max(2, round(len(non_ids) * frac))
    picked_ibd = rng.choice(ibd_ids, size=n_ibd, replace=False)
    picked_non = rng.choice(non_ids, size=n_non, replace=False)
    return list(picked_ibd) + list(picked_non)


def _padj_only_select(pool_df, ibd_group, budget):
    """Wraps rank_by_mannwhitney to the same (pool, group, budget) -> selected
    columns interface as two_stage_top, for the HTX proxy."""
    return rank_by_mannwhitney(pool_df, ibd_group).index[:budget]


def selection_stability(pool_df, ibd_group, select_fn=two_stage_top, n_resamples=N_RESAMPLES,
                         frac=SUBSAMPLE_FRAC, budget=NODE_BUDGET, seed=SEED):
    """select_fn(pool_df, ibd_group, budget) -> selected column index, applied
    identically inside every resample and to the full pool. Pass two_stage_top
    (default) for MGX/MBX, _padj_only_select for the HTX proxy."""
    rng = np.random.default_rng(seed)
    counts = pd.Series(0, index=pool_df.columns, dtype=int)

    for _ in range(n_resamples):
        subset_ids = stratified_subsample(ibd_group, frac, rng)
        selected = select_fn(pool_df.loc[subset_ids], ibd_group.loc[subset_ids], budget)
        counts.loc[selected] += 1

    freq = (counts / n_resamples).sort_values(ascending=False)

    original_top = select_fn(pool_df, ibd_group, budget)

    result = pd.DataFrame({
        "selection_frequency": freq.reindex(pool_df.columns),
    })
    result["in_original_top25"] = result.index.isin(original_top)
    result["stable"] = result["selection_frequency"] >= STABLE_THRESHOLD
    return result.loc[original_top].sort_values("selection_frequency", ascending=False)


SELECT_FN = {"MGX": two_stage_top, "MBX": two_stage_top, "HTX": _padj_only_select}


def run_real_data():
    for layer, path in LAYERS.items():
        print(f"\n=== {layer} ===")
        pool = pd.read_csv(path, index_col=0)
        ibd_group = load_diagnosis_groups(pool.index)
        result = selection_stability(pool, ibd_group, select_fn=SELECT_FN[layer])
        n_stable = result["stable"].sum()
        print(f"{n_stable}/{len(result)} of the original top-{NODE_BUDGET} are "
              f"stable (selected in >={STABLE_THRESHOLD:.0%} of {N_RESAMPLES} resamples)")
        out_path = f"Data/feature_selection_stability_{layer}.csv"
        result.to_csv(out_path)
        print(f"Saved to {out_path}")
        fragile = result[~result["stable"]]
        if len(fragile):
            print(f"Fragile features (selected but below {STABLE_THRESHOLD:.0%}):")
            print(fragile["selection_frequency"].round(3).to_string())


def self_test():
    """Synthetic sanity check: one feature perfectly separates the groups
    (should end up stable=True), one feature is pure noise (should not)."""
    rng = np.random.default_rng(1)
    n_ibd, n_non = 24, 16
    ids = [f"P{i}" for i in range(n_ibd + n_non)]
    ibd_group = pd.Series(["IBD"] * n_ibd + ["nonIBD"] * n_non, index=ids)

    strong = pd.Series(
        list(rng.normal(5, 0.3, n_ibd)) + list(rng.normal(0, 0.3, n_non)), index=ids
    )
    noise = pd.Series(rng.normal(0, 1, n_ibd + n_non), index=ids)
    pool = pd.DataFrame({"strong_signal": strong, "pure_noise": noise})

    result = selection_stability(pool, ibd_group, n_resamples=30, budget=1, seed=2)

    assert "strong_signal" in result.index, "the separating feature should make the top-1 cut at all"
    assert result.loc["strong_signal", "selection_frequency"] >= 0.9, (
        f"separating feature should be selected almost every resample, got "
        f"{result.loc['strong_signal', 'selection_frequency']}"
    )
    print("self-test passed:", result["selection_frequency"].round(3).to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run_real_data()
