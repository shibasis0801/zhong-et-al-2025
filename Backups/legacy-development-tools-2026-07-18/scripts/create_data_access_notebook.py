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

Sources: [Nature paper — data availability](https://www.nature.com/articles/s41586-025-09180-y#data-availability) ·
[Figshare release v2](https://doi.org/10.25378/janelia.28811129.v2).

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

The example selects one exact retinotopy file from the 297-file, 452,233,500,962-byte
Figshare v2 release. It displays the pinned catalog row before loading the file.
The resulting map is a released-data inspection, not a statistical result.

The coordinate orientation and area groups follow the paper's
[Figure 1i](https://www.nature.com/articles/s41586-025-09180-y#Fig1),
[neural-selectivity method](https://www.nature.com/articles/s41586-025-09180-y#Sec20),
and [`Get_density_map` implementation](https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416).
"""
        ),
        py(
            """
import matplotlib.pyplot as plt
import numpy as np

matches = data.find(category="retinotopy", contains="TX119_2023_12_24")
if len(matches) != 1:
    raise ValueError(f"Expected one pinned retinotopy file, found {len(matches)}")
selected = matches[0]
source_url = f"https://ndownloader.figshare.com/files/{selected.id}"
print({
    "name": selected.name,
    "file_id": selected.id,
    "category": selected.category,
    "size_bytes": selected.size_bytes,
    "md5": selected.md5,
    "release_path": selected.relative_path,
    "source_url": source_url,
})
retinotopy = data.load(selected)

xy = retinotopy["xy_t"]
area = retinotopy["iarea"]
if np.asarray(xy).shape != (len(area), 2):
    raise ValueError("xy_t and iarea do not share one neuron axis")

# Exact paper orientation: x = -xy_t[:, 1], y = xy_t[:, 0].
x, y = -np.asarray(xy)[:, 1], np.asarray(xy)[:, 0]
groups = {
    "V1 · iarea 8": np.asarray(area) == 8,
    "Medial · iarea 0,1,2,9": np.isin(area, [0, 1, 2, 9]),
    "Lateral · iarea 5,6": np.isin(area, [5, 6]),
    "Anterior · iarea 3,4": np.isin(area, [3, 4]),
}
assigned = np.logical_or.reduce(tuple(groups.values()))
groups["Excluded / unassigned"] = ~assigned
colors = ["#4c78a8", "#f28e2b", "#59a14f", "#e15759", "#a0a0a0"]

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
for (label, mask), color in zip(groups.items(), colors):
    ax.scatter(
        x[mask], y[mask], s=2, alpha=0.65, color=color,
        label=f"{label} · n={int(mask.sum()):,}", rasterized=True,
    )
ax.set(
    title=selected.name,
    xlabel="paper cortical x (release coordinate units)",
    ylabel="paper cortical y (release coordinate units)",
)
ax.set_aspect("equal")
ax.legend(markerscale=4, fontsize=8, frameon=False)
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

The released reduced-neural layer contains 400-component `U`/`V` factors; the
full-neural layer contains the Suite2p-derived deconvolved traces described in
[Methods: processing of calcium imaging data](https://www.nature.com/articles/s41586-025-09180-y#Sec19).
Inspect `session.file(<layer>)` for the exact file size before choosing a layer.
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
