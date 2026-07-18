#!/usr/bin/env python3
"""Generate the concise team dataset walkthrough."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/03_dataset_walkthrough_colab.ipynb")


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
            # Saved widget views are disconnected from a newly opened runtime.
            # Colab's private-output setting keeps every shared graph genuinely live.
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
# Zhong et al. (2025): explore the dataset as visible flows

Three small editable graphs move from the release map to one real cortical
file and then one compact recording. Change a hollow input port, press
**Run flow**, and follow the blue wires to the figure.

The views are descriptive. They do not calculate d-prime, fit learning curves,
or compare rewarded and unrewarded neural outcomes.
"""
        ),
        md(
            """
## Connect once

Add **Zhong et al. 2025 - Neuromatch Team Workspace** as a shortcut in
**My Drive**. The short cell below refreshes the team modules. Paths, caching,
file-size limits, and checksum verification remain inside drive.py.

Each time you reopen this notebook, choose **Runtime → Run all** once. Graph
outputs are deliberately not saved, so every visible port belongs to your own
live Colab session.
"""
        ),
        py(
            """
import importlib
import sys

try:
    from google.colab import drive as google_drive
except ImportError:
    pass
else:
    google_drive.mount("/content/drive", force_remount=False)
    from google.colab import output as colab_output

    colab_output.enable_custom_widget_manager()
    workspace = (
        "/content/drive/MyDrive/"
        "Zhong et al. 2025 - Neuromatch Team Workspace"
    )
    if workspace not in sys.path:
        sys.path.insert(0, workspace)

for name in tuple(sys.modules):
    if name in {"drive", "graph", "zhong2025"} or name.startswith("zhong2025."):
        sys.modules.pop(name, None)

drive = importlib.import_module("drive")
graph = importlib.import_module("graph")
data = drive.setup()
"""
        ),
        md(
            """
## The public release

The Figshare response is an ordinary dictionary. The high-level data object
searches the 297-row catalog and fetches only a selected file.
"""
        ),
        py(
            """
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from zhong2025 import (
    experiment_rows,
    format_bytes,
    load_atlas_demo,
    load_experiment_index,
)

figshare_api = data.figshare(live=drive.is_colab())
recording_rows = experiment_rows(load_experiment_index())
print({key: figshare_api.get(key) for key in ("id", "title", "version", "doi")})
print(f"{len(data.files)} published files; {len(recording_rows)} experiment memberships")
"""
        ),
        md(
            """
## Reading the graphs

A box is one ordinary Python function. Filled ports receive wired values;
hollow ports are editable choices. Numbered nodes always run sequentially,
even when two boxes sit side by side.
"""
        ),
        md(
            """
## Graph 1 — Release map and file picker

Search filenames or choose a published category. The plots describe catalog
storage and dataset availability; they are not biological results.
"""
        ),
        py(
            """
#@title Release-map transformations { display-mode: "form" }
@graph.node(outputs="release")
def load_release():
    return {"dataset": data, "recordings": recording_rows}


@graph.node(outputs="quality")
def check_release(release):
    files = release["dataset"].files
    if len(files) != 297:
        raise ValueError("Expected the pinned 297-file release")
    return {"files": len(files), "gib": sum(x.size_bytes for x in files) / 2**30}


@graph.node(outputs="selection")
def select_files(release, category="all", filename_contains=""):
    selected = release["dataset"].find(
        category=None if category == "all" else category,
        contains=filename_contains,
    )
    if not selected:
        raise ValueError("No filenames match these ports")
    return tuple(selected)


@graph.node(outputs="summary")
def summarize_release(
    release,
    quality,
    selection,
    measure="storage",
    availability="recordings",
):
    if measure not in {"storage", "count"}:
        raise ValueError("Release measure must be 'storage' or 'count'")
    if availability not in {"recordings", "mice"}:
        raise ValueError("Availability must be 'recordings' or 'mice'")
    groups = {}
    for item in selection:
        groups.setdefault(item.category, []).append(item)
    names = sorted(groups)
    group_bytes = [sum(x.size_bytes for x in groups[name]) for name in names]
    group_counts = [len(groups[name]) for name in names]
    values = (
        [size / 2**30 for size in group_bytes]
        if measure == "storage"
        else group_counts
    )
    value_labels = (
        [format_bytes(size) for size in group_bytes]
        if measure == "storage"
        else [f"{count} file{'s' if count != 1 else ''}" for count in group_counts]
    )

    def cohort(experiment):
        if experiment.startswith("sup_"):
            return "Task"
        if experiment.startswith("unsup_"):
            return "Unrewarded exposure"
        if experiment.startswith("naive_"):
            return "Naive"
        return "Grating control"

    cohorts = ["Task", "Unrewarded exposure", "Naive", "Grating control"]
    field = "recording_id" if availability == "recordings" else "mouse"
    counts = [
        len({row[field] for row in release["recordings"] if cohort(row["experiment"]) == name})
        for name in cohorts
    ]
    return {
        "categories": [name.replace("_", " ") for name in names],
        "values": values,
        "value_labels": value_labels,
        "measure": measure,
        "measure_label": "Storage" if measure == "storage" else "File count",
        "unit": "GiB" if measure == "storage" else "files",
        "matched": len(selection),
        "matched_gib": sum(x.size_bytes for x in selection) / 2**30,
        "cohorts": cohorts,
        "counts": counts,
        "availability": availability,
        "quality": quality,
    }


@graph.node(outputs="figure")
def plot_release(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    if summary["measure"] == "storage":
        positions = np.arange(len(summary["categories"]))
        baseline = min(summary["values"]) / 2
        axes[0].hlines(
            positions, baseline, summary["values"],
            color="#4c78a8", linewidth=2,
        )
        axes[0].scatter(summary["values"], positions, color="#4c78a8", s=42)
        axes[0].set_yticks(positions, labels=summary["categories"])
        axes[0].set_xscale("log")
        axes[0].set_xlim(baseline, max(summary["values"]) * 4)
        for position, value, label in zip(
            positions, summary["values"], summary["value_labels"]
        ):
            axes[0].annotate(
                label, (value, position), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8,
            )
    else:
        bars = axes[0].barh(
            summary["categories"], summary["values"], color="#4c78a8"
        )
        axes[0].bar_label(
            bars, labels=summary["value_labels"], padding=4, fontsize=8
        )
        axes[0].set_xlim(0, max(summary["values"]) * 1.3)
    axes[0].invert_yaxis()
    axes[0].set(
        title=f"{summary['measure_label']} by matching file category",
        xlabel=summary["unit"],
    )
    bars = axes[1].bar(summary["cohorts"], summary["counts"], color="#59a14f")
    axes[1].set(
        title=f"Whole-release availability: {summary['availability']}",
        ylabel=summary["availability"],
    )
    axes[1].tick_params(axis="x", rotation=15)
    for bar, count in zip(bars, summary["counts"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, count + 0.3, str(count), ha="center")
    fig.suptitle(
        f"{summary['matched']} matching files · {summary['matched_gib']:.1f} GiB"
    )
    plt.close(fig)
    return fig
"""
        ),
        py(
            """
#@title Edit the hollow ports, then run { display-mode: "form" }
release_graph = graph.Graph(
    "Release map and file picker",
    load_release,
    check_release,
    select_files,
    summarize_release,
    plot_release,
)
release_panel = release_graph.widget(
    controls={
        "category": widgets.Dropdown(
            description="File category",
            options=[
                ("All categories", "all"),
                *[(x.replace("_", " ").title(), x) for x in sorted({f.category for f in data.files})],
            ],
            value="all",
        ),
        "filename_contains": widgets.Text(
            description="Filename contains", value="", placeholder="Try TX119"
        ),
        "measure": widgets.Dropdown(
            description="Release measure",
            options=[("Storage", "storage"), ("File count", "count")],
            value="storage",
        ),
        "availability": widgets.Dropdown(
            description="Availability",
            options=[("Recordings", "recordings"), ("Mice", "mice")],
            value="recordings",
        ),
    },
    show="figure",
)
release_panel
"""
        ),
        md(
            """
## Graph 2 — One real cortical file from the 421 GiB release

The filename port comes from the 89 retinotopy files. Running the graph calls
the high-level data loader, which verifies and opens only the selected arrays.
"""
        ),
        py(
            """
#@title Cortical-map transformations { display-mode: "form" }
@graph.node(outputs="retinotopy")
def load_retinotopy(retinotopy_file="TX119_2023_12_24_trans.npz"):
    return {"file_name": retinotopy_file, **data.load(retinotopy_file)}


@graph.node(outputs="quality")
def check_retinotopy(retinotopy):
    xy = np.asarray(retinotopy["xy_t"])
    areas = np.asarray(retinotopy["iarea"])
    if xy.ndim != 2 or xy.shape[1] != 2 or areas.shape != (len(xy),):
        raise ValueError("Coordinates and area labels do not align")
    return {"points": len(xy)}


@graph.node(outputs="selection")
def select_area(retinotopy, cortical_area="all"):
    area_id = np.asarray(retinotopy["iarea"])
    masks = {
        "V1": area_id == 8,
        "mHV": np.isin(area_id, [0, 1, 2, 9]),
        "lHV": np.isin(area_id, [5, 6]),
        "aHV": np.isin(area_id, [3, 4]),
    }
    masks["Other"] = ~np.logical_or.reduce(list(masks.values()))
    if cortical_area != "all" and cortical_area not in masks:
        raise ValueError("Choose a visible cortical area")
    return {"area": cortical_area, "masks": masks}


@graph.node(outputs="summary")
def summarize_retinotopy(retinotopy, quality, selection):
    xy = np.asarray(retinotopy["xy_t"])
    selected = (
        np.ones(len(xy), dtype=bool)
        if selection["area"] == "all"
        else selection["masks"][selection["area"]]
    )
    x_span = float(np.ptp(xy[:, 0]))
    y_values = -xy[:, 1]
    y_span = float(np.ptp(y_values))
    return {
        "file_name": retinotopy["file_name"],
        "xy": xy,
        "points": quality["points"],
        "selected_points": int(selected.sum()),
        "x_limits": (
            float(xy[:, 0].min() - max(x_span * 0.03, 1e-9)),
            float(xy[:, 0].max() + max(x_span * 0.03, 1e-9)),
        ),
        "y_limits": (
            float(y_values.min() - max(y_span * 0.03, 1e-9)),
            float(y_values.max() + max(y_span * 0.03, 1e-9)),
        ),
        **selection,
    }


@graph.node(outputs="figure")
def plot_retinotopy(summary, point_size=2):
    if point_size <= 0:
        raise ValueError("Point size must be positive")
    colors = {
        "V1": "#4c78a8", "mHV": "#f28e2b", "lHV": "#59a14f",
        "aHV": "#e15759", "Other": "#aaaaaa",
    }
    xy = summary["xy"]
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for name, mask in summary["masks"].items():
        if summary["area"] not in {"all", name}:
            continue
        ax.scatter(
            xy[mask, 0], -xy[mask, 1], s=point_size, alpha=0.6,
            color=colors[name], label=f"{name} ({mask.sum():,})", rasterized=True,
        )
    ax.set(
        title=(
            f"{summary['file_name']} · {summary['area']} · "
            f"{summary['selected_points']:,}/{summary['points']:,} points"
        ),
        xlabel="retinotopy x (release units)",
        ylabel="retinotopy y (release units)",
        xlim=summary["x_limits"],
        ylim=summary["y_limits"],
    )
    ax.set_aspect("equal")
    ax.legend(markerscale=4, fontsize=8)
    plt.close(fig)
    return fig
"""
        ),
        py(
            """
#@title Edit the hollow ports, then run { display-mode: "form" }
retinotopy_files = [item.name for item in data.find(category="retinotopy")]
retinotopy_graph = graph.Graph(
    "Cortical map from one Drive file",
    load_retinotopy,
    check_retinotopy,
    select_area,
    summarize_retinotopy,
    plot_retinotopy,
)
retinotopy_panel = retinotopy_graph.widget(
    controls={
        "retinotopy_file": widgets.Combobox(
            description="Published file",
            options=retinotopy_files,
            value="TX119_2023_12_24_trans.npz",
            ensure_option=True,
        ),
        "cortical_area": widgets.Dropdown(
            description="Cortical area",
            options=[
                ("All areas", "all"),
                ("V1", "V1"),
                ("mHV", "mHV"),
                ("lHV", "lHV"),
                ("aHV", "aHV"),
                ("Other", "Other"),
            ],
            value="all",
        ),
        "point_size": widgets.IntSlider(
            description="Point size", value=2, min=1, max=8, step=1
        ),
    },
    show="figure",
)
retinotopy_panel
"""
        ),
        md(
            """
## Graph 3 — Trials, neural activity, and corridor position

One selection drives four factual views of the same trials. PC numbers are
session-specific; there is no fitted trend or reward comparison.
"""
        ),
        py(
            """
#@title Session transformations { display-mode: "form" }
@graph.node(outputs="demo")
def load_compact_recording():
    return load_atlas_demo()


@graph.node(outputs="quality")
def check_recording(demo):
    neural = np.asarray(demo["population_features"])
    speed = np.asarray(demo["mean_run_speed"])
    frames = np.asarray(demo["frame_counts"])
    if neural.shape[:2] != speed.shape or speed.shape != frames.shape:
        raise ValueError("Neural, speed, and support axes do not align")
    if np.any(frames <= 0):
        raise ValueError("A trial-position bin has no moving-frame support")
    return {"session": demo["metadata"]["session"], "shape": neural.shape}


@graph.node(outputs="selection")
def select_trials(demo, stimulus_id="all", corridor="full", trial_range=(0, 452)):
    start, stop = map(int, trial_range)
    trial_id = np.asarray(demo["trial_id"])
    if not 0 <= start < stop <= len(trial_id):
        raise ValueError("Choose 0 ≤ start < stop ≤ 452")
    trials = (trial_id >= start) & (trial_id < stop)
    valid_stimuli = {"all", *np.asarray(demo["stimulus_id"]).tolist()}
    if stimulus_id not in valid_stimuli:
        raise ValueError("Choose a published stimulus ID or 'all'")
    if stimulus_id != "all":
        trials &= np.asarray(demo["stimulus_id"]) == stimulus_id
    stimulus_label = (
        "all canonical roles"
        if stimulus_id == "all"
        else f"canonical role {stimulus_id}"
    )
    texture = np.asarray(demo["texture_bin_mask"], dtype=bool)
    regions = {"full": np.ones_like(texture), "texture": texture, "gray": ~texture}
    if corridor not in regions:
        raise ValueError("Corridor must be 'full', 'texture', or 'gray'")
    bins = regions[corridor]
    if not trials.any():
        raise ValueError("No trials match these ports")
    return {
        "trials": trials,
        "bins": bins,
        "stimulus_id": stimulus_id,
        "stimulus_label": stimulus_label,
        "corridor": corridor,
        "trial_range": (start, stop),
    }


@graph.node(outputs="summary")
def summarize_session(demo, quality, selection, pc_index=0, statistic="mean"):
    if statistic not in {"mean", "median"}:
        raise ValueError("Summary must be 'mean' or 'median'")
    trials, bins = selection["trials"], selection["bins"]
    population = np.asarray(demo["population_features"])
    if (
        not isinstance(pc_index, (int, np.integer))
        or not 0 <= int(pc_index) < population.shape[-1]
    ):
        raise ValueError(
            f"Published component must be between 0 and {population.shape[-1] - 1}"
        )
    pc_index = int(pc_index)
    neural = population[trials][:, bins, pc_index]
    speed = np.asarray(demo["mean_run_speed"])[trials][:, bins]
    reducer = np.mean if statistic == "mean" else np.median
    return {
        "session": quality["session"],
        "trial_id": np.asarray(demo["trial_id"])[trials],
        "stimulus": np.asarray(demo["stimulus_id"])[trials],
        "wall": np.asarray(demo["wall_name"])[trials],
        "position": np.asarray(demo["position_centers_m"])[bins],
        "heatmap": neural,
        "neural": reducer(neural, axis=0),
        "speed": reducer(speed, axis=0),
        "pc_index": pc_index,
        "statistic": statistic,
        "selection": selection,
    }


@graph.node(outputs="figure")
def plot_session(summary):
    colors = {
        "rock1": "#4c78a8", "rock2": "#72b7b2",
        "wood1": "#f28e2b", "wood2": "#e15759",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for wall, color in colors.items():
        mask = summary["wall"] == wall
        axes[0, 0].scatter(
            summary["trial_id"][mask], summary["stimulus"][mask],
            s=12, color=color, label=wall,
        )
    axes[0, 0].set(
        title="Physical texture in raw trial order",
        xlabel="trial number", ylabel="stimulus ID", yticks=[0, 1, 2, 3],
    )
    axes[0, 0].legend(ncol=2, fontsize=7)
    heatmap = summary["heatmap"]
    limit = max(float(np.max(np.abs(heatmap))), 1e-12)
    image = axes[0, 1].imshow(
        heatmap, aspect="auto", interpolation="nearest", cmap="coolwarm",
        vmin=-limit, vmax=limit,
        extent=[summary["position"][0], summary["position"][-1], len(heatmap), 0],
    )
    axes[0, 1].set(
        title=f"Published PC {summary['pc_index'] + 1}: trial × position",
        xlabel="corridor position (m)", ylabel="selected-trial row",
    )
    fig.colorbar(image, ax=axes[0, 1], label="PC score")
    axes[1, 0].plot(summary["position"], summary["neural"], marker="o")
    axes[1, 0].set(
        title=f"PC {summary['pc_index'] + 1} across-trial {summary['statistic']}",
        xlabel="corridor position (m)", ylabel="PC score",
    )
    axes[1, 1].plot(summary["position"], summary["speed"], marker="s", color="#f28e2b")
    axes[1, 1].set(
        title=f"Running speed across-trial {summary['statistic']}",
        xlabel="corridor position (m)", ylabel="release speed units",
    )
    fig.suptitle(
        f"{summary['session']} · {len(summary['trial_id'])} trials · "
        f"{summary['selection']['stimulus_label']} · "
        f"trials {summary['selection']['trial_range'][0]}–"
        f"{summary['selection']['trial_range'][1]} · "
        f"{summary['selection']['corridor']} corridor"
    )
    plt.close(fig)
    return fig
"""
        ),
        py(
            """
#@title Edit the hollow ports, then run { display-mode: "form" }
demo_preview = load_atlas_demo()
session_graph = graph.Graph(
    "Trials, neural activity, and corridor position",
    load_compact_recording,
    check_recording,
    select_trials,
    summarize_session,
    plot_session,
)
session_panel = session_graph.widget(
    controls={
        "stimulus_id": widgets.Dropdown(
            description="Stimulus role",
            options=[
                ("All canonical roles", "all"),
                ("Role 0 · circle1", 0),
                ("Role 1 · circle2", 1),
                ("Role 2 · leaf1", 2),
                ("Role 3 · leaf2", 3),
            ],
            value="all",
        ),
        "corridor": widgets.Dropdown(
            description="Corridor region",
            options=[("Whole corridor", "full"), ("Texture region", "texture"), ("Gray region", "gray")],
            value="full",
        ),
        "trial_range": widgets.IntRangeSlider(
            description="Trial range",
            value=(0, len(demo_preview["trial_id"])),
            min=0, max=len(demo_preview["trial_id"]), step=1,
            continuous_update=False,
        ),
        "pc_index": widgets.Dropdown(
            description="Published component",
            options=[(f"PC {i + 1}", i) for i in range(demo_preview["population_features"].shape[-1])],
            value=0,
        ),
        "statistic": widgets.Dropdown(
            description="Summary",
            options=[("Mean", "mean"), ("Median", "median")],
            value="mean",
        ),
    },
    show="figure",
)
session_panel
"""
        ),
        md(
            """
## Stop before inference

The graphs establish release composition, file selection, cortical coordinates,
trial order, session-specific PC structure, and corridor-position structure.
The team must still agree on the d-prime definition, reward definition,
early-trial window, curve model, and independent unit of inference.
"""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"walkthrough-{index:03d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
    print(NOTEBOOK)
