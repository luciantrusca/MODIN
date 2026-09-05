"""
Stage 1: headless runner for pipeline.ipynb (raw data -> analysis-ready matrices).

Run from the pipeline/ directory with the monika conda environment:
    python 01_run_pipeline.py

Executes the notebook in place and overwrites pipeline.ipynb with the run outputs.
"""
import nbformat
from nbclient import NotebookClient
import sys, time

nb = nbformat.read("pipeline.ipynb", as_version=4)
full_cells = nb.cells
# Run cells 0-11 only; cell 12 (network visualization) needs the graphml that GLASSO
# hasn't produced yet at this point, so it's excluded here and reattached unexecuted below
nb.cells = full_cells[:12]

client = NotebookClient(nb, timeout=7200, kernel_name="monika")

print(f"[{time.strftime('%H:%M:%S')}] Starting execution of cells 0-11...", flush=True)
try:
    client.execute()
    print(f"[{time.strftime('%H:%M:%S')}] Execution complete.", flush=True)
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ERROR during execution: {e}", flush=True)
    # Still save whatever got executed so we can inspect partial progress
    nb.cells = list(nb.cells) + full_cells[12:]
    nbformat.write(nb, "pipeline.ipynb")
    sys.exit(1)

# Reattach cell 12 (unexecuted) so the notebook file still contains it
nb.cells = list(nb.cells) + full_cells[12:]
nbformat.write(nb, "pipeline.ipynb")
print(f"[{time.strftime('%H:%M:%S')}] Notebook saved.", flush=True)
