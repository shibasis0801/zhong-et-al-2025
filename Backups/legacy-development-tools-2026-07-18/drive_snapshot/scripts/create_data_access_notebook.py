#!/usr/bin/env python3
"""Generate a concise Janelia data-access notebook."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/00_use_janelia_drive_colab.ipynb")


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def py(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "accelerator": "CPU",
        "colab": {
            "name": NOTEBOOK.name,
            "private_outputs": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook.cells = [
        md(
            """
# Janelia data access

A visual picker and concise Python examples provide access to the complete
Zhong et al. release. Drive paths, caching, checksums, and file loading are
handled by `drive.py`.

Select **Runtime → Run all** after opening the notebook. A CPU runtime is
sufficient.
"""
        ),
        md(
            """
## Data connection

A shortcut named **Zhong et al. 2025 - Neuromatch Team Workspace** is required
in **My Drive**. The connection cell uses the shared read-only workspace.
"""
        ),
        py(
            """
#@title Connect to the shared dataset { display-mode: "form" }
import importlib
import sys

from google.colab import drive as google_drive

google_drive.mount("/content/drive", force_remount=False)
workspace = (
    "/content/drive/MyDrive/"
    "Zhong et al. 2025 - Neuromatch Team Workspace"
)
if workspace not in sys.path:
    sys.path.insert(0, workspace)
sys.modules.pop("drive", None)
drive = importlib.import_module("drive")
data = drive.setup(mount=False)
"""
        ),
        md(
            """
## Visual file selection

**By recording** selects a known experiment/session; **Any file** exposes the
complete 297-file catalog. **Load selected data** begins loading after a
selection is made.
"""
        ),
        py(
            """
picker = data.picker()
picker
"""
        ),
        md(
            """
`picker.value` returns an ordinary Python dictionary or NumPy array after
loading. Rerun the value cell after changing the selected file.
"""
        ),
        py(
            """
sample = picker.value
if sample is None:
    print("No data loaded; a picker selection must be loaded before rerunning this cell.")
elif isinstance(sample, dict):
    print("Available keys:", ", ".join(map(str, sample.keys())))
else:
    print(type(sample).__name__, getattr(sample, "shape", ""))
"""
        ),
        md(
            """
## Programmatic file access

The example searches the catalog and loads one small real file from the
421 GiB release. It is descriptive and does not test a hypothesis.
"""
        ),
        py(
            """
import matplotlib.pyplot as plt

matches = data.find(category="retinotopy", contains="TX119_2023_12_24")
retinotopy = data.load(matches[0])

xy = retinotopy["xy_t"]
area = retinotopy["iarea"]
fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
points = ax.scatter(xy[:, 0], -xy[:, 1], c=area, s=2, cmap="tab20", rasterized=True)
ax.set(title=matches[0].name, xlabel="retinotopy x", ylabel="retinotopy y")
ax.set_aspect("equal")
fig.colorbar(points, ax=ax, label="published area ID")
plt.show()
"""
        ),
        md(
            """
## Recording-level access

`data.recording(...)` resolves the behavior, reduced-neural, full-neural, and
retinotopy files associated with a session. Inspection alone does not download
data.
"""
        ),
        py(
            """
session = data.recording("TX119_2023_12_24_1")
session
"""
        ),
        md(
            """
Only layers required by an experiment should be loaded:

```python
svd = session.load("reduced_neural")
behavior = session.load("behavior", experiment="unsup_test1")
retinotopy = session.load("retinotopy")
```

Reduced neural data are the practical default for typical Colab analyses.
Full-neural files are deliberately not loaded by the visual picker because an
individual file can be several GiB.
"""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"data-access-{index:03d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
    print(NOTEBOOK)
