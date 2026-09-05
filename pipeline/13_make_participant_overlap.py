"""Participant coverage across the three omics layers (Figure: fig:venn-overlap).

Replaces the original three-circle Venn, which drew four regions containing nobody,
and the nested Euler diagram, which compressed a 31% drop in count into a 17% change
in radius. Bar LENGTH is proportional to participant count, so the differences read
directly. Built at exactly \\linewidth -- place with width=\\linewidth and do not
rescale, or the label sizes stop being true.

Run from the pipeline/ directory:
    python 13_make_participant_overlap.py
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TOTAL = {"MGX": 130, "MBX": 106, "HTX": 90}
# Hand-copied from 01_run_pipeline.py's console output (Stage 1, cross_layer_participants.csv
# generation). Nothing enforces these stay in sync -- re-copy after any cohort-size change.
CORE, MID, OUT = 90, 16, 24
# acmart migration, 2026-08-31: \columnwidth under mscthesis[ds]/acmart[sigconf]
# measures 241.147pt (was 236.85pt under the old plain-article preamble).
W = 241.147 / 72.27
INK, MUTE = "#173241", "#5c7c8c"
C_CORE, C_MID, C_OUT = "#2f6690", "#8ab4d0", "#cbdff0"

X0, XSPAN, XTOT = 0.48, 2.30, 2.86
u = XSPAN / TOTAL["MGX"]
BAR_H, GAP, TOP, FOOT = 0.235, 0.205, 0.07, 0.20
H = TOP + 3 * BAR_H + 2 * GAP + FOOT

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

rows = [("MGX", [(CORE, C_CORE), (MID, C_MID), (OUT, C_OUT)]),
        ("MBX", [(CORE, C_CORE), (MID, C_MID)]),
        ("HTX", [(CORE, C_CORE)])]

ybs = []
for i, (layer, segs) in enumerate(rows):
    yb = H - TOP - BAR_H - i * (BAR_H + GAP); ybs.append(yb)
    x = X0
    for n, color in segs:
        w = n * u
        ax.add_patch(Rectangle((x, yb), w, BAR_H, facecolor=color,
                               edgecolor="white", linewidth=0.9))
        ax.text(x + w / 2, yb + BAR_H / 2, str(n), ha="center", va="center",
                fontsize=8.5 if color == C_CORE else 8,
                color="white" if color == C_CORE else INK,
                fontweight="bold" if color == C_CORE else "normal")
        x += w
    ax.text(X0 - 0.07, yb + BAR_H / 2, layer, ha="right", va="center",
            fontsize=9.5, color=INK)
    ax.text(XTOT, yb + BAR_H / 2, str(TOTAL[layer]), ha="left", va="center",
            fontsize=9, color=INK)

# Why each band drops out, written in the gap directly under the band that vanishes.
for i, (drop, label) in enumerate(((OUT, "no metabolomics sample"),
                                   (MID, "no baseline biopsy"))):
    right = X0 + TOTAL[["MGX", "MBX"][i]] * u
    ax.text(right, ybs[i] - GAP / 2, label, ha="right", va="center",
            fontsize=7, color=MUTE)

ax.text(X0, ybs[2] - FOOT / 2 - 0.01, "analysis cohort", ha="left", va="center",
        fontsize=7.5, color=MUTE, style="italic")

assert CORE + MID + OUT == TOTAL["MGX"]
assert TOTAL["MGX"] - TOTAL["MBX"] == OUT and TOTAL["MBX"] - TOTAL["HTX"] == MID
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "participant_overlap.pdf"), format="pdf", transparent=True)
print(f"ok; {H*72:.1f}pt tall (euler was 228.9pt)")
