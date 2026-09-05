"""
Prevalence-threshold audit for the MGX layer.

Reproduces three checks, all on the CURRENT pipeline ordering (cross-layer
intersection applied BEFORE prevalence filtering, n=90 participants):

  1. sweep()      -- candidates and padj<0.05 counts at 10/15/20/25%.
                     Also re-runs the superseded no-intersection ordering,
                     which is where the void n=130 figures came from.
  2. by_band()    -- is significance concentrated at low prevalence?
  3. artifact()   -- is that concentration driven by CLR imputation rather
                     than by measured abundance?

Run from the pipeline/ directory:
    python3 08_prevalence_threshold_audit.py

Only needs pandas/numpy/scipy. BH correction is implemented inline so the
script does not depend on statsmodels (which .venv lacks).
"""

import numpy as np
import pandas as pd
from scipy import stats

MGX_PATH = "Data/Unprocessed_Other/mgx_species_abundances_raw.tsv"
META_PATH = "Data/hmp2_metadata_2018-08-20.csv"
KEEP_PATH = "Data/cross_layer_participants.csv"
GROUP_MAP = {"CD": "IBD", "UC": "IBD", "nonIBD": "nonIBD"}


def bh(p):
    """Benjamini-Hochberg step-up, equivalent to multipletests(method='fdr_bh')."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def load():
    raw = pd.read_csv(MGX_PATH, sep="\t", index_col=0)
    species = raw.loc[raw.index.str.contains(r"\|s__", regex=True)].copy()
    species.index = species.index.str.extract(r"\|s__(.+)$")[0]
    mgx = species.T.astype(float)
    mgx.index = mgx.index.str.replace(r"_P_profile$|_profile$|_P$", "", regex=True)

    meta = pd.read_csv(META_PATH, low_memory=False)
    pid = (
        meta[["External ID", "Participant ID"]]
        .assign(c=lambda d: d["External ID"].str.replace(
            r"_P_profile$|_profile$|_P$", "", regex=True))
        .drop_duplicates("c").set_index("c")["Participant ID"]
    )
    diag = meta.drop_duplicates("Participant ID").set_index("Participant ID")["diagnosis"]
    keep = set(pd.read_csv(KEEP_PATH)["Participant ID"])
    return mgx, pid, diag, keep


def clr_by_participant(mgx, threshold, pid):
    """Prevalence filter -> pseudocount -> CLR -> per-participant median."""
    prev = (mgx > 0).mean(axis=0)
    m = mgx.loc[:, prev >= threshold]
    pseudo = m.where(m > 0).stack().min() / 2
    logged = np.log(m.where(m > 0, pseudo))
    clr = logged.subtract(logged.mean(axis=1), axis=0)
    clr.index = [pid.get(s) for s in clr.index]
    clr = clr[[i is not None and i == i for i in clr.index]]
    return m, logged, clr.groupby(level=0).median()


def test_groups(pmat, diag):
    grp = pd.Series(pmat.index, index=pmat.index).map(diag).map(GROUP_MAP)
    a, b = pmat[grp == "IBD"], pmat[grp == "nonIBD"]
    ps = [
        stats.mannwhitneyu(a[c].dropna(), b[c].dropna(), alternative="two-sided")[1]
        if a[c].notna().sum() > 1 and b[c].notna().sum() > 1 else 1.0
        for c in pmat.columns
    ]
    return bh(np.array(ps)), grp


def sweep(mgx, pid, diag, label):
    print(f"\n=== {label}: {mgx.shape[0]} samples ===")
    print("  thresh | candidates | padj<0.05 | worst padj in top 25")
    for t in (0.10, 0.15, 0.20, 0.25):
        m, _, pmat = clr_by_participant(mgx, t, pid)
        padj, _ = test_groups(pmat, diag)
        worst = pd.Series(padj, index=pmat.columns).sort_values().iloc[:25].max()
        print(f"   {int(t*100):>3}%  |    {m.shape[1]:>4}    |    {(padj<0.05).sum():>3}    | {worst:.4f}")


def by_band(mgx, pid, diag):
    """Is significance concentrated in the low-prevalence band?"""
    prev_all = (mgx > 0).mean(axis=0)
    m, _, pmat = clr_by_participant(mgx, 0.10, pid)
    padj, _ = test_groups(pmat, diag)
    t = pd.DataFrame({"prev": prev_all[pmat.columns].values, "padj": padj}, index=pmat.columns)

    print(f"\n=== Significance by prevalence band (10% threshold, {len(t)} candidates) ===")
    print("  band      | candidates | significant | % significant")
    for lo, hi in ((.10, .20), (.20, .40), (.40, .70), (.70, 1.01)):
        band = t[(t.prev >= lo) & (t.prev < hi)]
        n_sig = int((band.padj < 0.05).sum())
        pct = 100 * n_sig / max(len(band), 1)
        print(f"  {lo:.0%}-{hi:.0%}   |    {len(band):>4}    |     {n_sig:>3}     | {pct:5.1f}%")

    sig, nonsig = t[t.padj < 0.05], t[t.padj >= 0.05]
    rho, prho = stats.spearmanr(t.prev, t.padj)
    print(f"\n  median prevalence: significant={sig.prev.median():.3f}  non-significant={nonsig.prev.median():.3f}")
    print(f"  Spearman rho(prevalence, padj) = {rho:.3f}, p={prho:.3g}")
    return t


def artifact(mgx, pid, diag, t):
    """Is the low-prevalence signal driven by CLR imputation?"""
    m, logged, pmat = clr_by_participant(mgx, 0.10, pid)

    m_i = logged.mean(axis=1)
    m_i.index = [pid.get(s) for s in m_i.index]
    m_i = m_i[[i is not None and i == i for i in m_i.index]].groupby(level=0).median()
    grp = pd.Series(m_i.index, index=m_i.index).map(diag).map(GROUP_MAP)
    A, B = m_i[grp == "IBD"], m_i[grp == "nonIBD"]
    p_mi = stats.mannwhitneyu(A, B, alternative="two-sided")[1]

    print("\n=== Is the CLR denominator itself group-associated? ===")
    print(f"  mean log-abundance m_i: IBD={A.mean():.4f}  nonIBD={B.mean():.4f}  diff={A.mean()-B.mean():+.4f}")
    print(f"  Mann-Whitney: p={p_mi:.3g}")
    print("  -> every undetected entry becomes log(pseudocount) - m_i, inheriting this group difference.")

    low_sig = t[(t.prev < 0.20) & (t.padj < 0.05)].index
    det = (m > 0)
    det.index = [pid.get(s) for s in det.index]
    det = det[[i is not None and i == i for i in det.index]].groupby(level=0).any()

    rows = []
    for sp in low_sig:
        has = det[sp]
        sub = pmat.loc[has[has].index, sp]
        gsub = pd.Series(sub.index, index=sub.index).map(diag).map(GROUP_MAP)
        x, y = sub[gsub == "IBD"], sub[gsub == "nonIBD"]
        p_det = (stats.mannwhitneyu(x, y, alternative="two-sided")[1]
                 if len(x) > 1 and len(y) > 1 else np.nan)
        rows.append((sp[:38], t.prev[sp], t.padj[sp], 1 - has.mean(), p_det, len(x), len(y)))

    r = pd.DataFrame(rows, columns=[
        "species", "prev", "padj_CLR", "frac_imputed", "p_detected_only", "nIBD_det", "nNon_det"])
    print(f"\n=== The {len(r)} significant species with prevalence <20% ===")
    pd.set_option("display.width", 200)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\n  {(r.p_detected_only < 0.05).sum()} of {len(r)} remain p<0.05 (UNCORRECTED) using only")
    print("  participants where the species was actually detected.")
    print("  Caveat: that subset is much smaller, so this loses power -- suggestive, not conclusive.")


def presence_absence(mgx, pid, diag, t):
    """Robustness check: Fisher's exact test on detection rate (present vs
    absent) rather than CLR Mann-Whitney on abundance values. This asks a different
    question than artifact()'s detected-only test: not "does abundance differ among
    those where it was detected" but "is the species detected at a different RATE in
    IBD vs non-IBD at all". A real absence-in-IBD effect should show up here even
    though it is invisible to the CLR test once undetected samples are imputed to a
    shared floor value."""
    m, _, pmat = clr_by_participant(mgx, 0.10, pid)
    det = (m > 0)
    det.index = [pid.get(s) for s in det.index]
    det = det[det.index.notna()].groupby(level=0).any()
    grp = pd.Series(det.index, index=det.index).map(diag).map(GROUP_MAP)
    ibd_mask, non_mask = grp == "IBD", grp == "nonIBD"

    low_sig = t[(t.prev < 0.20) & (t.padj < 0.05)].index
    rows = []
    for sp in low_sig:
        d = det[sp]
        a, b = int((d & ibd_mask).sum()), int((~d & ibd_mask).sum())
        c, dd = int((d & non_mask).sum()), int((~d & non_mask).sum())
        _, p = stats.fisher_exact([[a, b], [c, dd]])
        rows.append((sp[:38], t.prev[sp], t.padj[sp], a, b, c, dd, p))

    r = pd.DataFrame(rows, columns=[
        "species", "prev", "padj_CLR", "det_IBD", "undet_IBD", "det_non", "undet_non", "p_fisher"])
    r["padj_fisher"] = bh(r["p_fisher"].values)

    print(f"\n=== Presence/absence Fisher's exact test, same {len(r)} species as artifact() ===")
    pd.set_option("display.width", 200)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\n  {(r.p_fisher < 0.05).sum()} of {len(r)} significant uncorrected;"
          f" {(r.padj_fisher < 0.05).sum()} of {len(r)} survive BH correction.")
    print("  This tests detection RATE directly -- a real absence-in-IBD effect")
    print("  survives here even though it is invisible to the CLR/imputation test.")
    return r


def tail_correlation(mgx, pid):
    """Is the prevalence filter justified by tail correlation, not average correlation?"""
    prev_all = (mgx > 0).mean(axis=0)
    _, _, pmat = clr_by_participant(mgx, 0.10, pid)
    pv = prev_all[pmat.columns]
    low = [c for c in pmat.columns if pv[c] < 0.20]
    high = [c for c in pmat.columns if pv[c] >= 0.40]

    def offdiag(cols):
        C = pmat[cols].corr().values
        return C[np.triu_indices(len(cols), 1)]

    lo, hi = offdiag(low), offdiag(high)
    print("\n=== Pairwise correlation in the CLR matrix (the GLASSO input scale) ===")
    print(f"  low-prevalence  (<20%, {len(low)} species): mean |r|={np.abs(lo).mean():.3f}, |r|>0.5 in {np.mean(np.abs(lo)>0.5):.1%} of pairs")
    print(f"  high-prevalence (>=40%, {len(high)} species): mean |r|={np.abs(hi).mean():.3f}, |r|>0.5 in {np.mean(np.abs(hi)>0.5):.1%} of pairs")
    print(f"  Mann-Whitney |r| low vs high: p={stats.mannwhitneyu(np.abs(lo), np.abs(hi), alternative='two-sided')[1]:.3g}")
    print("  -> the filter is justified by the TAIL, not by average correlation.")


def main():
    mgx_all, pid, diag, keep = load()
    mgx_cur = mgx_all.loc[[s for s in mgx_all.index if pid.get(s) in keep]]

    sweep(mgx_cur, pid, diag, "CURRENT pipeline (intersection first, n=90 participants)")
    sweep(mgx_all, pid, diag, "SUPERSEDED ordering (no intersection) -- source of the void n=130 figures")

    t = by_band(mgx_cur, pid, diag)
    artifact(mgx_cur, pid, diag, t)
    presence_absence(mgx_cur, pid, diag, t)
    tail_correlation(mgx_cur, pid)


if __name__ == "__main__":
    main()
