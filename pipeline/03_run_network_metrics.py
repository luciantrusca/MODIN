"""
Headless runner for network_metrics.ipynb (topology, centrality, community
detection, and RWR diffusion scores on the GLASSO-inferred joint network).

Requires GLASSO/results/ibd/seed_selection.csv to already exist (produced by
02_seed_selection.py --write) -- the notebook raises if it's missing.

Run from the pipeline/ directory with the monika conda environment:
    python 03_run_network_metrics.py

Executes the notebook in place and overwrites network_metrics.ipynb with the run outputs.
"""
import nbformat
from nbclient import NotebookClient
import time

nb = nbformat.read("network_metrics.ipynb", as_version=4)
client = NotebookClient(nb, timeout=1200, kernel_name="monika")

print(f"[{time.strftime('%H:%M:%S')}] Starting execution...", flush=True)
client.execute()
print(f"[{time.strftime('%H:%M:%S')}] Execution complete.", flush=True)

nbformat.write(nb, "network_metrics.ipynb")
print(f"[{time.strftime('%H:%M:%S')}] Notebook saved.", flush=True)
