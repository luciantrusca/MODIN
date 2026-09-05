r"""
Figure for SQ1: how node centrality is redistributed between IBD and non-IBD.

Shows the most extreme nodes among the 36 whose composite centrality rank shifts
significantly (BH q < 0.05) between diagnosis groups, as a signed rank shift coloured by
omics layer. The point of the figure is that the sign separates perfectly by layer, which a
table does not convey.

Showing all 36 individually made the figure tall without adding to the point it makes, so
it shows the N_PER_SIDE=9 most extreme nodes on each side (18 individual bars, the
sign-separates-by-layer claim is already fully visible from the extremes) and aggregates
the remaining, smaller-magnitude nodes into one summary bar per omics layer present in that
remainder (mean signed shift, hatched to mark it as an aggregate, labelled with its node
count). Aggregate bars land in the middle of the chart by construction, since they are
built from the smallest-magnitude nodes and the chart is still sorted by signed value. The
all-36 invariants (transcripts always IBD-side, microbiome/metabolite always non-IBD-side)
are still asserted on the full 36 before any aggregation, so the underlying claim is
checked at full strength even though only a subset is drawn individually.

Sized to the exact column width of the thesis, so LaTeX includes it at 1:1 and the font
sizes below are the sizes that appear in print. The width is not a guess: see COLWIDTH_IN
below for the current measured \columnwidth. Re-measure it if the document class or
geometry ever changes.

Run from the pipeline/ directory:
    python 11_make_subgroup_fig.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).parent.parent
SRC = BASE / "GLASSO/results/ibd/subgroup_node_comparison.csv"
OUT = BASE / "images/subgroup_centrality_shift.pdf"

COLOURS = {"microbiome": "#4C9BE8", "metabolite": "#E8834C", "transcript": "#5CB85C"}
# acmart migration done 2026-08-31 (mscthesis[ds]/acmart[sigconf]): re-measured,
# \columnwidth is now 241.147pt, not 236.85pt.
COLWIDTH_IN = 241.147 / 72.27   # \columnwidth in pt -> inches; 3.339 in

N_PER_SIDE = 9  # individual nodes kept on each side of zero; rest aggregated per layer

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 5.5, "legend.fontsize": 6,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


def prettify(name):
    """Abbreviate the genus but keep every remaining token, so strain identifiers survive:
    Veillonella_sp_T11011_6 -> V. sp T11011 6, not V. sp."""
    if "_" in name and name[0].isupper() and not name.startswith(("C18n", "HILn", "C8p")):
        parts = name.split("_")
        if len(parts) >= 2 and parts[1][:1].islower():
            return f"{parts[0][0]}. " + " ".join(parts[1:])
    return name.replace("_", " ")


def main():
    d = pd.read_csv(SRC, index_col=0)
    sig = d[d.cent_q < 0.05].copy()
    # Signed shift: negative = more central in IBD (lower rank number is more central).
    sig["signed"] = sig.cent_rank_IBD - sig.cent_rank_nonIBD
    sig = sig.sort_values("signed")
    assert len(sig) == 36, f"expected 36 significant nodes, got {len(sig)}"
    # The claim the figure makes -- verify it on the FULL set, before any aggregation.
    t = sig[sig.modality == "transcript"]
    o = sig[sig.modality != "transcript"]
    assert (t.signed < 0).all(), "a transcript did not gain centrality in IBD"
    assert (o.signed > 0).all(), "a non-transcript did not gain centrality in non-IBD"

    neg = sig[sig.signed < 0]
    pos = sig[sig.signed > 0]
    keep_neg = neg.iloc[:N_PER_SIDE]          # most extreme (most negative first)
    keep_pos = pos.iloc[-N_PER_SIDE:]         # most extreme (most positive last)
    agg_neg = neg.iloc[N_PER_SIDE:]
    agg_pos = pos.iloc[:-N_PER_SIDE]

    # Build the aggregate rows: one per modality present in each remainder, at that
    # modality's mean signed shift, which -- since these are the smallest-magnitude nodes --
    # lands each aggregate bar close to zero, in the middle of the chart.
    agg_rows = []
    for remainder in (agg_neg, agg_pos):
        for modality, grp in remainder.groupby("modality"):
            label = f"{len(grp)} more {modality}{'s' if len(grp) != 1 else ''}"
            agg_rows.append({"label": label, "modality": modality,
                              "signed": grp.signed.mean(), "is_agg": True})

    plot_rows = (
        [{"label": prettify(n), "modality": m, "signed": s, "is_agg": False}
         for n, m, s in zip(keep_neg.index, keep_neg.modality, keep_neg.signed)]
        + agg_rows
        + [{"label": prettify(n), "modality": m, "signed": s, "is_agg": False}
           for n, m, s in zip(keep_pos.index, keep_pos.modality, keep_pos.signed)]
    )
    plot_df = pd.DataFrame(plot_rows).sort_values("signed").reset_index(drop=True)
    n_rows = len(plot_df)

    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 4.7 * n_rows / 36))
    bars = ax.barh(range(n_rows), plot_df.signed,
                    color=[COLOURS[m] for m in plot_df.modality], height=0.72,
                    edgecolor="none")
    for bar, is_agg in zip(bars, plot_df.is_agg):
        if is_agg:
            bar.set_alpha(0.55)
            bar.set_hatch("///")
    ax.set_yticks(range(n_rows))
    labels = ax.set_yticklabels(plot_df.label)
    for lbl, is_agg in zip(labels, plot_df.is_agg):
        if is_agg:
            lbl.set_style("italic")
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_ylim(-0.8, n_rows - 0.2)
    ax.set_xlabel("composite centrality rank shift\n(negative = more central in IBD)")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # Sorted ascending, so the most negative (IBD-gaining) rows sit at the BOTTOM and their
    # bars extend left; the non-IBD rows sit at the top and extend right. That leaves the
    # top-left and bottom-right corners empty, which is where the annotations go. The legend
    # goes above the axes entirely, since both free corners are now occupied.
    lim = max(abs(plot_df.signed)) * 1.18
    ax.set_xlim(-lim, lim)
    ax.text(-lim * 0.96, n_rows - 1.0, "more central\nin non-IBD $\\rightarrow$",
            fontsize=6, style="italic", va="top", ha="left", color="0.3")
    ax.text(lim * 0.96, 0.8, "$\\leftarrow$ more central\nin IBD",
            fontsize=6, style="italic", va="bottom", ha="right", color="0.3")
    legend_handles = [mpatches.Patch(color=c, label=m) for m, c in COLOURS.items()]
    legend_handles.append(mpatches.Patch(facecolor="0.6", hatch="///", alpha=0.55,
                                          edgecolor="0.6", label="aggregate (mean)"))
    ax.legend(handles=legend_handles,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, handlelength=1.1, borderpad=0.2, columnspacing=1.2)

    # NOTE: no bbox_inches="tight". Tight bbox trims to the drawn content, which made the
    # saved file 247pt wide against a 237pt column, so \includegraphics[width=\linewidth]
    # scaled it DOWN by 0.958 and the authored 5.2pt labels printed at 4.98pt. Letting
    # tight_layout handle padding keeps the saved width equal to figsize, so the scale is
    # exactly 1.0 and authored point sizes are printed point sizes.
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT)

    import re
    import subprocess
    info = subprocess.run(["pdfinfo", str(OUT)], capture_output=True, text=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info)
    print(f"wrote {OUT}")
    print(f"  {n_rows} rows drawn ({n_rows - len(agg_rows)} individual + {len(agg_rows)} "
          f"aggregate), representing all 36 significant nodes")
    print(f"  {len(t)} transcripts all negative, {len(o)} non-transcripts all positive (full 36)")
    if m:
        w = float(m.group(1))
        scale = COLWIDTH_IN * 72.27 / w
        print(f"  saved width {w:.1f}pt vs column 241.147pt -> LaTeX scale {scale:.3f}")
        print(f"  smallest label prints at {5.5 * scale:.2f}pt")
        assert 0.99 <= scale <= 1.01, f"figure will be rescaled by {scale:.3f}; fonts will shift"


if __name__ == "__main__":
    main()
