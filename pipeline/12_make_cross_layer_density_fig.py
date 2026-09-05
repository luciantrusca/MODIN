"""
Figure: cross-layer edge count/density heatmap (3x3, one cell per omics-layer pair).

Run from the pipeline/ directory:
    python 12_make_cross_layer_density_fig.py
"""
import pandas as pd, numpy as np, matplotlib, networkx as nx
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R="../GLASSO/results/ibd/"
edges=pd.read_csv(R+"ibd_joint_edges.csv")
# Repointed 2026-08-31 (acmart migration, item 7 cleanup): network_metrics_summary.csv
# was deleted as stale (carried the rejected seed set). Node modality is unaffected by
# the seed-rule change and lives on the graph itself, so read it from there directly --
# no pipeline rerun needed.
G=nx.read_graphml(R+"ibd_joint_network.graphml")
modality=nx.get_node_attributes(G,"modality")

MAP={"microbiome":"mgx","metabolite":"mbx","transcript":"rna"}
LAB={"mgx":"Microbiome\n(MGX)","mbx":"Metabolite\n(MBX)","rna":"Transcript\n(HTX)"}
ORD=["mgx","mbx","rna"]
n={k:sum(1 for v in modality.values() if MAP.get(v)==k) for k in ORD}

# normalise edge_type to unordered pair
def pair(t):
    a,b=t.split("-"); return tuple(sorted((a,b)))
cnt=edges["edge_type"].map(pair).value_counts().to_dict()

C=np.zeros((3,3)); D=np.zeros((3,3))
for i,a in enumerate(ORD):
    for j,b in enumerate(ORD):
        k=tuple(sorted((a,b))); c=cnt.get(k,0)
        poss = n[a]*(n[a]-1)/2 if a==b else n[a]*n[b]
        C[i,j]=c; D[i,j]=c/poss if poss else 0

# acmart migration, 2026-08-31: \columnwidth is 241.147pt = 3.337in (was 3.35in,
# an unlabelled approximation under the old class). Uses bbox_inches="tight" below,
# so this is a canvas size, not an exact 1:1 calibration like the other figure scripts.
fig,ax=plt.subplots(figsize=(241.147/72.27,2.75))
im=ax.imshow(D,cmap="viridis",vmin=0,vmax=D.max())
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([LAB[k] for k in ORD],fontsize=7)
ax.set_yticklabels([LAB[k] for k in ORD],fontsize=7)
ax.tick_params(length=0)
for i in range(3):
    for j in range(3):
        v=D[i,j]
        ax.text(j,i,f"{int(C[i,j])}\n{v:.3f}",ha="center",va="center",fontsize=7.5,
                color="white" if v<D.max()*0.6 else "black")
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
cb.set_label("edge density",fontsize=7); cb.ax.tick_params(labelsize=6.5)
ax.set_title("Edge count and density by layer pair",fontsize=8,pad=6)
fig.tight_layout()
fig.savefig("../images/cross_layer_density.pdf",bbox_inches="tight")
print("counts\n",C.astype(int))
print("density\n",np.round(D,4))
print("nodes per layer:",n,"total edges:",len(edges))
