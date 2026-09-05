r"""
Figure for SQ2: diffusion ranking from the literature-anchored seeds.

The existing metric_rwr_scores.png is from the OLD hand-picked seed set (SAA2, DUOXA2,
K. pneumoniae) and must not be used. This regenerates the ranking from the current
seed_selection.csv against the current graphml.

The point of the figure is that all three omics layers appear near the top, which the
earlier seed set never achieved -- its top 10 non-seed nodes contained no metabolite at all.
Seeds are drawn hatched so a reader can see the ranking is not simply the seeds and their
immediate neighbours.

Sized to \columnwidth (241.147pt = 3.336in) and saved WITHOUT bbox_inches="tight", so the
saved width equals figsize and LaTeX includes it at 1:1. See 11_make_subgroup_fig.py for why
that matters: tight bbox trims to content and silently rescales the fonts.

Run from the pipeline/ directory:
    python 10_make_rwr_fig.py
"""

import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

BASE = Path(__file__).parent.parent
GRAPHML = BASE / "GLASSO/results/ibd/ibd_joint_network.graphml"
SEEDS_CSV = BASE / "GLASSO/results/ibd/seed_selection.csv"
OUT = BASE / "images/rwr_ranking.pdf"

COLOURS = {"microbiome": "#4C9BE8", "metabolite": "#E8834C", "transcript": "#5CB85C"}
# acmart migration, 2026-08-31: \columnwidth under mscthesis[ds]/acmart[sigconf]
# measures 241.147pt (was 236.85pt under the old plain-article preamble).
COLWIDTH_IN = 241.147 / 72.27
RESTART = 0.3
TOP_N = 15

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 5.5, "legend.fontsize": 5.8,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


def prettify(name):
    if "_" in name and name[0].isupper() and not name.startswith(("C18n", "HILn", "C8p")):
        parts = name.split("_")
        if len(parts) >= 2 and parts[1][:1].islower():
            return f"{parts[0][0]}. " + " ".join(parts[1:])
    return name.replace("_", " ")


def main():
    G = nx.read_graphml(GRAPHML)
    for _, _, d in G.edges(data=True):
        d["w"] = abs(float(d.get("weight", 0.0)))
    seeds = sorted(pd.read_csv(SEEDS_CSV).query("admitted")["node"])
    assert all(s in G for s in seeds), "a selected seed is absent from the network"

    p = {n: (1.0 / len(seeds) if n in seeds else 0.0) for n in G.nodes()}
    score = pd.Series(nx.pagerank(G, alpha=1 - RESTART, personalization=p, weight="w"))
    mod = nx.get_node_attributes(G, "modality")

    # Seeds are excluded from the plot. They hold ranks 1-4 by construction and their scores
    # (0.077-0.104) are three times the leading non-seed, so including them compresses the
    # entire informative comparison into the left third of the axis. The caption names them.
    assert list(score.nlargest(len(seeds)).index.sort_values()) == seeds, \
        "seeds are not the top-ranked nodes; the exclusion below would drop real results"
    non_seed = score.drop(labels=seeds)
    top = non_seed.nlargest(TOP_N)[::-1]

    # The claim the caption makes: all three layers present among the top non-seed nodes.
    layers = {mod[n] for n in non_seed.nlargest(10).index}
    assert layers == set(COLOURS), f"top 10 non-seeds cover only {layers}"

    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.9))
    for i, (name, val) in enumerate(top.items()):
        ax.barh(i, val, color=COLOURS[mod[name]], height=0.72, edgecolor="none")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([prettify(n) for n in top.index])
    ax.set_xlabel("random walk with restart score")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    handles = [mpatches.Patch(color=c, label=m) for m, c in COLOURS.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, handlelength=1.1, borderpad=0.2, columnspacing=1.2)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT)

    info = subprocess.run(["pdfinfo", str(OUT)], capture_output=True, text=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info)
    print(f"wrote {OUT}")
    print(f"  seeds: {seeds}")
    print(f"  top 10 non-seed layers: {sorted(layers)}")
    if m:
        w = float(m.group(1))
        scale = COLWIDTH_IN * 72.27 / w
        print(f"  saved width {w:.1f}pt vs column 241.147pt -> LaTeX scale {scale:.3f}")
        print(f"  smallest label prints at {5.5 * scale:.2f}pt")
        assert 0.99 <= scale <= 1.01, f"figure will be rescaled by {scale:.3f}"


if __name__ == "__main__":
    main()
