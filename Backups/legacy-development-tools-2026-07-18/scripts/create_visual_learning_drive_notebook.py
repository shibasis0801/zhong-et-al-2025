#!/usr/bin/env python3
"""Build the source-linked Drive edition of the Neuromatch visual-learning notebook."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import nbformat as nbf


UPSTREAM_URL = (
    "https://github.com/NeuromatchAcademy/course-content/blob/main/"
    "projects/neurons/visual_learning_80k_neurons.ipynb"
)
REFERENCE = Path("notebooks/05_neuromatch_visual_learning_project.ipynb")
NOTEBOOK = Path("notebooks/02_released_example_dprime_walkthrough.ipynb")

PAPER = "https://www.nature.com/articles/s41586-025-09180-y"
FIGURE_1_IMAGE = (
    "https://media.springernature.com/full/springer-static/image/"
    "art%3A10.1038%2Fs41586-025-09180-y/MediaObjects/"
    "41586_2025_9180_Fig1_HTML.png"
)


def _source(text: str) -> str:
    return text.strip() + "\n"


def _markdown(cell_id: str, text: str):
    cell = nbf.v4.new_markdown_cell(_source(text))
    cell.id = cell_id
    return cell


def _code(cell_id: str, text: str):
    cell = nbf.v4.new_code_cell(_source(text))
    cell.id = cell_id
    cell.execution_count = None
    cell.outputs = []
    return cell


def _strip_outputs(notebook):
    clean = deepcopy(notebook)
    for cell in clean.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return clean


def prepare_reference(source: Path, destination: Path = REFERENCE):
    """Store an output-free snapshot of the upstream teaching notebook."""

    upstream = _strip_outputs(nbf.read(source, as_version=4))
    upstream.metadata.setdefault("zhong2025_reference", {})
    upstream.metadata["zhong2025_reference"] = {
        "source": UPSTREAM_URL,
        "purpose": "unaltered scientific-source reference for the Drive edition",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(upstream, destination)
    return upstream


def build_notebook(reference: Path = REFERENCE):
    # Validate that the preserved source snapshot is present without modifying it.
    nbf.validate(nbf.read(reference, as_version=4))

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
        "zhong2025_conversion": {
            "upstream_reference": str(reference),
            "upstream_url": UPSTREAM_URL,
            "data_release": "10.25378/janelia.28811129.v2",
            "scope": "one released raw-trace example using the paper's whole-session d-prime estimator",
        },
    }
    notebook.cells = [
        _markdown(
            "visual-learning-title",
            f"""
# Visual learning across 80,000 neurons: released-example walkthrough

This notebook has one bounded purpose: inspect the released
`VR2_2021_03_20_1` example traces with the paper's signed, whole-session
per-neuron d′ calculation. The complete Neuromatch teaching notebook is preserved
separately as [`neuromatch_visual_learning_80k_neurons.ipynb`]({UPSTREAM_URL}); it
is not rewritten here.

Primary sources: [Nature Figure 1]({PAPER}#Fig1),
[selectivity analysis in Methods]({PAPER}#Sec20),
[data availability]({PAPER}#data-availability), and
[Figshare v2](https://doi.org/10.25378/janelia.28811129.v2).
""",
        ),
        _markdown(
            "nature-figure-1",
            f"""
## Complete published figure

**Zhong et al. (2025), Nature Figure 1.**
[Open the anchored figure and caption]({PAPER}#Fig1).

![Complete Nature Figure 1]({FIGURE_1_IMAGE})
""",
        ),
        _markdown(
            "drive-setup-intro",
            """
## Connect to the released files

Add **Zhong et al. 2025 - Neuromatch Team Workspace** as a shortcut in
**My Drive**, then run the next cell. `drive.setup()` reads the pinned 297-file
catalog. The later load cell requests only the three files named below.
""",
        ),
        _code(
            "drive-setup",
            """
# @title Connect to the shared Drive and load the pinned catalog
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
""",
        ),
        _markdown(
            "released-inputs",
            """
## Exact released inputs

| Use | Released filename | Figshare file ID | Bytes | Published MD5 | Direct file |
|---|---|---:|---:|---|---|
| 1,000-neuron deconvolved example | `VR2_2021_03_20_1_example_raw_spk.npy` | 54866153 | 97,192,128 | `7e341d96305e3a235213b419f71c576d` | [file](https://ndownloader.figshare.com/files/54866153) |
| supervised Train 1 before-learning behaviour | `Beh_sup_train1_before_learning.npy` | 54183863 | 124,559,852 | `75169b8c4c02f5ed9af3fd492e93b9bd` | [file](https://ndownloader.figshare.com/files/54183863) |
| matching retinotopy | `VR2_2021_03_20_trans.npz` | 54184211 | 2,934,270 | `f8fbb33ee2c9461011306c5072d0b06e` | [file](https://ndownloader.figshare.com/files/54184211) |

The next cell checks every value against the pinned
[Figshare v2 release](https://doi.org/10.25378/janelia.28811129.v2) before any
array is loaded.
""",
        ),
        _code(
            "verify-released-inputs",
            """
# @title Verify exact filenames, IDs, byte counts, and checksums
RECORDING_ID = "VR2_2021_03_20_1"
EXPERIMENT = "sup_train1_before_learning"
RAW_FILE = "VR2_2021_03_20_1_example_raw_spk.npy"

EXPECTED_FILES = {
    RAW_FILE: (54866153, 97192128, "7e341d96305e3a235213b419f71c576d"),
    "Beh_sup_train1_before_learning.npy": (
        54183863,
        124559852,
        "75169b8c4c02f5ed9af3fd492e93b9bd",
    ),
    "VR2_2021_03_20_trans.npz": (
        54184211,
        2934270,
        "f8fbb33ee2c9461011306c5072d0b06e",
    ),
}

verified_files = {}
for filename, expected in EXPECTED_FILES.items():
    matches = [item for item in data.files if item.name == filename]
    if len(matches) != 1:
        raise ValueError(f"Expected one catalog row for {filename}; found {len(matches)}")
    item = matches[0]
    actual = (item.id, item.size_bytes, item.md5)
    if actual != expected:
        raise ValueError(f"Pinned metadata changed for {filename}: {actual!r}")
    verified_files[filename] = item
    print({
        "name": item.name,
        "id": item.id,
        "size_bytes": item.size_bytes,
        "md5": item.md5,
        "direct_url": f"https://ndownloader.figshare.com/files/{item.id}",
    })
""",
        ),
        _markdown(
            "paper-estimator",
            f"""
## Paper estimator, unchanged

The authors' implementation defines, for each neuron,

`d′ = 2 × (mean(leaf1) − mean(circle1)) / (std(leaf1) + std(circle1))`.

It uses every frame in the recording for which `ft_move > 0` and
`ft_CorrSpc` is true, maps stimulus ID 2 to `leaf1` and ID 0 to `circle1`, and
classifies signed selectivity at ±0.3 for the Figure 1 density map. See the
authors' exact [d′ function and valid-frame selection](https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L370-L443),
[0.3 threshold and density map](https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416),
and [Methods: selectivity analysis]({PAPER}#Sec20).

The paper's result uses the full deconvolved population for each recording. This
notebook applies the same estimator to the released 1,000-neuron raw example
from one recording. Its output is therefore an inspection of that
released example, not a reproduction of the paper's cohort estimate.
""",
        ),
        _code(
            "paper-dprime-functions",
            """
# @title Exact whole-session signed d-prime and paper-coordinate transform
DP_THRESHOLD = 0.3


def load_released_example():
    session = data.recording(RECORDING_ID)
    return {
        "activity": data.load(RAW_FILE, max_gib=0.2),
        "behavior": session.load("behavior", experiment=EXPERIMENT),
        "retinotopy": session.load("retinotopy"),
    }


def paper_dprime_for_released_example(activity, behavior, retinotopy):
    activity = np.asarray(activity, dtype=float)
    if activity.ndim != 2:
        raise ValueError("Released activity must have shape neurons × frames")

    frame_wall = np.asarray(behavior["ft_WallID"])
    frame_move = np.asarray(behavior["ft_move"]) > 0
    frame_corridor = np.asarray(behavior["ft_CorrSpc"], dtype=bool)
    n_frames = min(activity.shape[1], len(frame_wall), len(frame_move), len(frame_corridor))
    activity = activity[:, :n_frames]
    valid = frame_move[:n_frames] & frame_corridor[:n_frames]

    stimulus_id = np.asarray(behavior["stim_id"])
    unique_walls = np.asarray(behavior["UniqWalls"])
    leaf_matches = unique_walls[stimulus_id == 2]
    circle_matches = unique_walls[stimulus_id == 0]
    if len(leaf_matches) == 0 or len(circle_matches) == 0:
        raise ValueError("Released behavior is missing leaf1 or circle1")

    leaf_mask = (frame_wall[:n_frames] == leaf_matches[0]) & valid
    circle_mask = (frame_wall[:n_frames] == circle_matches[0]) & valid
    if min(int(leaf_mask.sum()), int(circle_mask.sum())) < 2:
        raise ValueError("Fewer than two valid frames remain for one stimulus")

    leaf = activity[:, leaf_mask]
    circle = activity[:, circle_mask]
    mean_leaf = np.nanmean(leaf, axis=1)
    mean_circle = np.nanmean(circle, axis=1)
    spread = np.nanstd(leaf, axis=1) + np.nanstd(circle, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dprime = 2.0 * (mean_leaf - mean_circle) / spread

    xy_t = np.asarray(retinotopy["xy_t"], dtype=float)[: activity.shape[0]]
    iarea = np.asarray(retinotopy["iarea"])[: activity.shape[0]]
    if len(xy_t) != activity.shape[0] or len(iarea) != activity.shape[0]:
        raise ValueError("Retinotopy does not cover every released example neuron")
    cortical_x = -xy_t[:, 1]
    cortical_y = xy_t[:, 0]
    mapped = (iarea != -1) & (iarea != 7)
    leaf_selective = mapped & np.isfinite(dprime) & (dprime >= DP_THRESHOLD)
    circle_selective = mapped & np.isfinite(dprime) & (dprime <= -DP_THRESHOLD)
    return {
        "dprime": dprime,
        "mean_leaf": mean_leaf,
        "mean_circle": mean_circle,
        "cortical_x": cortical_x,
        "cortical_y": cortical_y,
        "iarea": iarea,
        "mapped": mapped,
        "leaf_selective": leaf_selective,
        "circle_selective": circle_selective,
        "leaf_frames": int(leaf_mask.sum()),
        "circle_frames": int(circle_mask.sum()),
    }
""",
        ),
        _code(
            "run-paper-dprime",
            """
# @title Load the three verified files and compute the released-example result
released_example = load_released_example()
example_result = paper_dprime_for_released_example(**released_example)
print({
    "recording_id": RECORDING_ID,
    "experiment": EXPERIMENT,
    "neurons": int(len(example_result["dprime"])),
    "leaf1_valid_frames": example_result["leaf_frames"],
    "circle1_valid_frames": example_result["circle_frames"],
    "mapped_neurons": int(example_result["mapped"].sum()),
    "leaf1_selective_at_0.3": int(example_result["leaf_selective"].sum()),
    "circle1_selective_at_-0.3": int(example_result["circle_selective"].sum()),
})
""",
        ),
        _code(
            "plot-paper-dprime",
            """
# @title Plot the released-example frame counts, signed d-prime, and locations
def plot_released_example(result):
    leaf_color = "#2E8B57"
    circle_color = "#7A5195"
    neutral = "#9AA2AD"
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    bars = axes[0, 0].bar(
        ["leaf1", "circle1"],
        [result["leaf_frames"], result["circle_frames"]],
        color=[leaf_color, circle_color],
    )
    axes[0, 0].bar_label(bars)
    axes[0, 0].set(title="Valid whole-session frames", ylabel="frames")

    finite = result["mapped"] & np.isfinite(result["dprime"])
    axes[0, 1].hist(result["dprime"][finite], bins=40, color=neutral)
    axes[0, 1].axvline(DP_THRESHOLD, color=leaf_color, linestyle="--")
    axes[0, 1].axvline(-DP_THRESHOLD, color=circle_color, linestyle="--")
    axes[0, 1].set(title="Signed per-neuron d′", xlabel="d′: leaf1 − circle1", ylabel="neurons")

    axes[1, 0].scatter(
        result["mean_circle"][finite], result["mean_leaf"][finite],
        s=8, color=neutral, alpha=0.4,
    )
    axes[1, 0].scatter(
        result["mean_circle"][result["leaf_selective"]],
        result["mean_leaf"][result["leaf_selective"]],
        s=14, color=leaf_color, label="d′ ≥ 0.3",
    )
    axes[1, 0].scatter(
        result["mean_circle"][result["circle_selective"]],
        result["mean_leaf"][result["circle_selective"]],
        s=14, color=circle_color, label="d′ ≤ −0.3",
    )
    axes[1, 0].set(
        title="Whole-session mean responses",
        xlabel="circle1 mean", ylabel="leaf1 mean",
    )
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(
        result["cortical_x"][result["mapped"]],
        result["cortical_y"][result["mapped"]],
        s=4, color=neutral, alpha=0.25,
    )
    axes[1, 1].scatter(
        result["cortical_x"][result["leaf_selective"]],
        result["cortical_y"][result["leaf_selective"]],
        s=14, color=leaf_color, label="leaf1",
    )
    axes[1, 1].scatter(
        result["cortical_x"][result["circle_selective"]],
        result["cortical_y"][result["circle_selective"]],
        s=14, color=circle_color, label="circle1",
    )
    axes[1, 1].set(
        title="Paper-coordinate locations",
        xlabel="cortical x = -xy_t[:, 1]",
        ylabel="cortical y = xy_t[:, 0]",
    )
    axes[1, 1].set_aspect("equal")
    axes[1, 1].legend(frameon=False)
    figure.suptitle(f"{RECORDING_ID} · released 1,000-neuron example")
    return figure


example_figure = plot_released_example(example_result)
example_figure
""",
        ),
        _markdown(
            "scope-note",
            f"""
## What this output establishes

The output reports the released example's frame support, signed d′ values, and
paper-coordinate neuron locations. It does not estimate learning, reward effects,
or population prevalence because it contains one example recording and 1,000
released neurons. The paper's cohort-level result and sample sizes are in
[Figure 1]({PAPER}#Fig1) and the associated
[Results subsection]({PAPER}#Sec2).
""",
        ),
    ]

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nbf.validate(notebook)
    return notebook


def write_notebook(reference: Path = REFERENCE, destination: Path = NOTEBOOK):
    destination.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(reference)
    nbf.write(notebook, destination)
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="optional downloaded upstream notebook to refresh the preserved reference",
    )
    args = parser.parse_args()
    if args.source is not None:
        prepare_reference(args.source)
    write_notebook()
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
