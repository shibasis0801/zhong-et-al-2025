#!/usr/bin/env python3
"""Generate a source-linked walkthrough of the Zhong et al. public release."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/03_dataset_walkthrough_colab.ipynb")
PAPER = "https://www.nature.com/articles/s41586-025-09180-y"
FIGSHARE = "https://doi.org/10.25378/janelia.28811129.v2"


def md(cell_id, text):
    cell = nbf.v4.new_markdown_cell(text.strip() + "\n")
    cell.id = cell_id
    return cell


def py(cell_id, text):
    cell = nbf.v4.new_code_cell(text.strip() + "\n")
    cell.id = cell_id
    cell.execution_count = None
    cell.outputs = []
    return cell


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "accelerator": "CPU",
        "colab": {"name": NOTEBOOK.name, "private_outputs": True},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook.cells = [
        md(
            "walkthrough-title",
            f"""
# Zhong et al. (2025): released-data walkthrough

This notebook reports three things directly from the pinned release: its file
and experiment inventory, the authors' cortical-coordinate transform for one
retinotopy file, and quality-control views of one disclosed release derivative.
It does not add a d′ analysis or make a new biological inference.

Primary sources: [Nature Figure 1]({PAPER}#Fig1),
[Results: supervised and unsupervised plasticity]({PAPER}#Sec2),
[processing of calcium-imaging data]({PAPER}#Sec19),
[data availability]({PAPER}#data-availability), and
[Figshare v2]({FIGSHARE}).
""",
        ),
        md(
            "walkthrough-protocol",
            f"""
## Published experiment represented by the release

The paper reports head-fixed mice running through 4 m virtual corridors that
contained naturalistic textures and were separated by 2 m grey intervals. Task
mice could receive water after a random sound cue in the rewarded corridor;
unrewarded-exposure mice encountered the stimuli without water reward. The
training timeline and cohort definitions are in
[Figure 1a–b]({PAPER}#Fig1) and its
[Results subsection]({PAPER}#Sec2).

The inventory counts below are release-availability counts. Experiment
memberships overlap, so they are not substituted for panel-specific sample
sizes in the paper's figure captions.
""",
        ),
        md(
            "walkthrough-setup-intro",
            """
## Connect to the fixed team workspace

Add **Zhong et al. 2025 - Neuromatch Team Workspace** as a shortcut in
**My Drive**, then run the next cell. It imports the release utilities from
that one path and reads the bundled, pinned metadata.
""",
        ),
        py(
            "walkthrough-setup",
            """
# @title Connect to the release
import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from google.colab import drive as google_drive
except ImportError:
    pass
else:
    google_drive.mount("/content/drive", force_remount=False)
    workspace = (
        "/content/drive/MyDrive/"
        "Zhong et al. 2025 - Neuromatch Team Workspace"
    )
    if workspace not in sys.path:
        sys.path.insert(0, workspace)

for name in tuple(sys.modules):
    if name == "drive" or name.startswith("zhong2025"):
        sys.modules.pop(name, None)

drive = importlib.import_module("drive")
data = drive.setup()
from zhong2025 import experiment_rows, load_atlas_demo, load_experiment_index
""",
        ),
        md(
            "walkthrough-release-intro",
            f"""
## Release and experiment inventory

The file inventory is the pinned [Figshare version-2 record]({FIGSHARE}). The
experiment index is the released
[`Imaging_Exp_info.npy`](https://ndownloader.figshare.com/files/54183854).
The next cell calculates all counts from those records and keeps both raw
metadata-row and deduplicated experiment–recording membership counts visible.
""",
        ),
        py(
            "walkthrough-release",
            """
# @title Verify the release and experiment membership counts
release = data.figshare(live=drive.is_colab())
rows = experiment_rows(load_experiment_index())
membership_pairs = {
    (row["experiment"], row["recording_id"])
    for row in rows
}
acquisitions = {row["recording_id"] for row in rows}

release_summary = {
    "figshare_id": release["id"],
    "figshare_version": release["version"],
    "doi": release["doi"],
    "published_files": len(data.files),
    "published_bytes": sum(item.size_bytes for item in data.files),
    "metadata_rows": len(rows),
    "unique_experiment_recording_memberships": len(membership_pairs),
    "unique_acquisitions": len(acquisitions),
}
print(release_summary)

experiment_membership_counts = {
    experiment: len({
        row["recording_id"] for row in rows if row["experiment"] == experiment
    })
    for experiment in sorted({row["experiment"] for row in rows})
}
for experiment, count in experiment_membership_counts.items():
    print(f"{experiment:28s} {count:2d} unique recording memberships")
""",
        ),
        md(
            "walkthrough-retinotopy-intro",
            f"""
## One exact retinotopy file

This section opens only
[`TX119_2023_12_24_trans.npz`](https://ndownloader.figshare.com/files/54184070)
(983,358 bytes; MD5 `ddda2db80ae338435ffa73b289690ae0`). The paper code
plots `x = -xy_t[:, 1]` and `y = xy_t[:, 0]` and groups `iarea` as V1 (`8`),
medial (`0,1,2,9`), lateral (`5,6`), anterior (`3,4`), with `-1` and `7`
excluded from visual-area analyses. See the exact
[density-map implementation](https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416)
and [Nature Figure 1]({PAPER}#Fig1).
""",
        ),
        py(
            "walkthrough-retinotopy",
            """
# @title Load and plot the released retinotopy with the paper transform
RETINOTOPY_FILE = "TX119_2023_12_24_trans.npz"
matches = [item for item in data.files if item.name == RETINOTOPY_FILE]
if len(matches) != 1:
    raise ValueError(f"Expected one catalog row; found {len(matches)}")
retinotopy_file = matches[0]
expected_retinotopy = (
    54184070,
    983358,
    "ddda2db80ae338435ffa73b289690ae0",
)
if (
    retinotopy_file.id,
    retinotopy_file.size_bytes,
    retinotopy_file.md5,
) != expected_retinotopy:
    raise ValueError("Pinned retinotopy metadata changed")
print({
    "name": retinotopy_file.name,
    "id": retinotopy_file.id,
    "size_bytes": retinotopy_file.size_bytes,
    "md5": retinotopy_file.md5,
    "direct_url": f"https://ndownloader.figshare.com/files/{retinotopy_file.id}",
})


def prepare_retinotopy(retinotopy):
    xy_t = np.asarray(retinotopy["xy_t"], dtype=float)
    iarea = np.asarray(retinotopy["iarea"])
    if xy_t.ndim != 2 or xy_t.shape[1] != 2 or iarea.shape != (len(xy_t),):
        raise ValueError("xy_t and iarea do not share one neuron axis")
    groups = {
        "V1": iarea == 8,
        "Medial": np.isin(iarea, [0, 1, 2, 9]),
        "Lateral": np.isin(iarea, [5, 6]),
        "Anterior": np.isin(iarea, [3, 4]),
    }
    assigned = np.logical_or.reduce(list(groups.values()))
    groups["Excluded / unassigned"] = ~assigned
    return {
        "x": -xy_t[:, 1],
        "y": xy_t[:, 0],
        "iarea": iarea,
        "groups": groups,
    }


def plot_retinotopy(prepared):
    colors = {
        "V1": "#4C78A8",
        "Medial": "#F28E2B",
        "Lateral": "#59A14F",
        "Anterior": "#E15759",
        "Excluded / unassigned": "#A0A0A0",
    }
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for label, mask in prepared["groups"].items():
        axis.scatter(
            prepared["x"][mask], prepared["y"][mask],
            s=2, alpha=0.55, color=colors[label],
            label=f"{label} ({int(mask.sum()):,})", rasterized=True,
        )
    axis.set(
        title=RETINOTOPY_FILE,
        xlabel="paper cortical x = -xy_t[:, 1]",
        ylabel="paper cortical y = xy_t[:, 0]",
    )
    axis.set_aspect("equal")
    axis.legend(markerscale=4, fontsize=8)
    return figure


released_retinotopy = data.load(RETINOTOPY_FILE)
prepared_retinotopy = prepare_retinotopy(released_retinotopy)
retinotopy_figure = plot_retinotopy(prepared_retinotopy)
retinotopy_figure
""",
        ),
        md(
            "walkthrough-derivative-intro",
            f"""
## One disclosed release derivative

The bundled derivative uses session `TX119_2023_12_24_1`, condition
`unsup_test1`, and exactly three release files:

- [behaviour, file 54183911](https://ndownloader.figshare.com/files/54183911)
- [reduced-neural SVD, file 54866057](https://ndownloader.figshare.com/files/54866057)
- [retinotopy, file 54184070](https://ndownloader.figshare.com/files/54184070)

Its [deterministic builder](https://github.com/shibasis0801/zhong-et-al-2025/blob/main/zhong2025/demo.py#L80-L235)
verifies the source checksums, retains moving frames, bins each trial into 18
fixed position bins from 0 to 6 m, and retains the first 48 released SVD
component scores plus mean running speed. This is a project QC derivative, not
a paper figure or result. The released preprocessing is described in
[Methods]({PAPER}#Sec19).
""",
        ),
        py(
            "walkthrough-derivative",
            """
# @title Inspect the derivative provenance and arrays
demo = load_atlas_demo()
metadata = demo["metadata"]
assert metadata["session"] == "TX119_2023_12_24_1"
assert metadata["source_file_ids"] == [54183911, 54866057, 54184070]
assert demo["population_features"].shape == (452, 18, 48)
assert demo["mean_run_speed"].shape == (452, 18)
assert demo["frame_counts"].shape == (452, 18)

print({
    "session": metadata["session"],
    "condition": metadata["condition"],
    "source_file_ids": metadata["source_file_ids"],
    "source_sha256": metadata["source_sha256"],
    "binning": metadata["binning"],
    "population_representation": metadata["population_representation"],
    "population_shape": demo["population_features"].shape,
})
""",
        ),
        py(
            "walkthrough-derivative-plot",
            """
# @title Plot observed quantities in the disclosed derivative
def plot_derivative_qc(demo, component_index=0):
    population = np.asarray(demo["population_features"], dtype=float)
    speed = np.asarray(demo["mean_run_speed"], dtype=float)
    position = np.asarray(demo["position_centers_m"], dtype=float)
    trial_id = np.asarray(demo["trial_id"])
    wall_name = np.asarray(demo["wall_name"])
    if not 0 <= int(component_index) < population.shape[-1]:
        raise ValueError("component_index is outside the released derivative")

    component = population[:, :, int(component_index)]
    limit = max(float(np.nanmax(np.abs(component))), 1e-12)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    walls = sorted(set(wall_name.tolist()))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(walls)))
    for wall, color in zip(walls, colors):
        mask = wall_name == wall
        axes[0, 0].scatter(
            trial_id[mask], np.full(mask.sum(), walls.index(wall)),
            s=9, color=color, label=wall,
        )
    axes[0, 0].set(
        title="Raw wall identity by trial",
        xlabel="released trial ID", ylabel="raw wall",
        yticks=range(len(walls)), yticklabels=walls,
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    image = axes[0, 1].imshow(
        component, aspect="auto", interpolation="nearest", cmap="coolwarm",
        vmin=-limit, vmax=limit,
        extent=[position[0], position[-1], len(component), 0],
    )
    axes[0, 1].set(
        title=f"Binned released SVD component {int(component_index) + 1}",
        xlabel="position (m)", ylabel="trial row",
    )
    figure.colorbar(image, ax=axes[0, 1], label="released SVD score")

    axes[1, 0].plot(position, np.nanmean(component, axis=0), marker="o")
    axes[1, 0].set(
        title="Across-trial mean SVD score",
        xlabel="position (m)", ylabel="released SVD score",
    )

    axes[1, 1].plot(position, np.nanmean(speed, axis=0), marker="s", color="#F28E2B")
    axes[1, 1].set(
        title="Across-trial mean running speed",
        xlabel="position (m)", ylabel="release speed units",
    )
    figure.suptitle(
        f"{demo['metadata']['session']} · {len(trial_id)} trials · "
        "18 fixed position bins"
    )
    return figure


derivative_figure = plot_derivative_qc(demo)
derivative_figure
""",
        ),
        md(
            "walkthrough-scope",
            f"""
## Scope of these views

The release inventory establishes file availability; the retinotopy panel
applies the authors' published coordinate transform; and the final panels show
only values contained in the disclosed derivative. Biological conclusions and
panel-specific sample sizes must be read from the corresponding published
[figure caption]({PAPER}#Fig1), [Results]({PAPER}#Sec2), and
[Methods]({PAPER}#Sec19).
""",
        ),
    ]

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nbf.validate(notebook)
    return notebook


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
    print(NOTEBOOK)
