#!/usr/bin/env python3
"""Generate notebook 07: every imaging mouse and its released data journey."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/07_complete_mouse_journeys_colab.ipynb")
CANONICAL_MAP = Path("zhong2025/assets/canonical_map.json")
NATURE = "https://www.nature.com/articles/s41586-025-09180-y"
FIGSHARE = "https://doi.org/10.25378/janelia.28811129.v2"
INDEX_FILE = "https://ndownloader.figshare.com/files/54183854"
ATLAS = "https://shibasis.dev/neuromatch/"

COHORT_ORDER = {"sup": 0, "unsup": 1, "grating": 2, "naive": 3}
COHORT_LABELS = {
    "sup": "Task / rewarded",
    "unsup": "Unrewarded natural texture",
    "grating": "Grating exposure control",
    "naive": "Naive only",
}


def md(cell_id: str, text: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(text.strip() + "\n")
    cell.id = cell_id
    return cell


def py(cell_id: str, text: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(text.strip() + "\n")
    cell.id = cell_id
    cell.execution_count = None
    cell.outputs = []
    return cell


def _membership_text(recording: dict) -> str:
    labels = []
    seen = set()
    for membership in recording["memberships"]:
        key = (
            membership["cohort"], membership["phase"],
            membership.get("moment"), membership["label"],
        )
        if key in seen:
            continue
        seen.add(key)
        moment = f" {membership['moment']}" if membership.get("moment") else ""
        labels.append(
            f"{membership['cohort']} {membership['phase']}{moment}"
        )
    return "; ".join(labels)


def _coverage_text(recordings: list[dict]) -> str:
    memberships = [
        membership
        for recording in recordings
        for membership in recording["memberships"]
    ]

    def has(phase: str, moment: str | None = None) -> bool:
        return any(
            membership["phase"] == phase
            and (moment is None or membership.get("moment") == moment)
            for membership in memberships
        )

    parts = []
    if any(membership["cohort"] == "naive" for membership in memberships):
        parts.append("naive")
    if has("Train 1", "before") and has("Train 1", "after"):
        parts.append("Train 1 B/A")
    elif has("Train 1"):
        parts.append("Train 1 partial")
    if has("Test 1"):
        parts.append("Test 1")
    if has("Train 2", "before") and has("Train 2", "after"):
        parts.append("Train 2 B/A")
    elif has("Train 2"):
        parts.append("Train 2 partial")
    if has("Test 2"):
        parts.append("Test 2")
    if has("Test 3"):
        parts.append("Test 3")
    return ", ".join(parts)


def _all_mouse_table() -> str:
    canonical = json.loads(CANONICAL_MAP.read_text(encoding="utf-8"))
    rows = []
    mice = sorted(
        canonical["mice"].values(),
        key=lambda mouse: (
            COHORT_ORDER.get(mouse["primary_cohort"], 9),
            mouse["mouse"],
        ),
    )
    for mouse in mice:
        recordings = [
            canonical["recordings"][recording_id]
            for recording_id in mouse["recordings"]
        ]
        recordings.sort(key=lambda row: (row["date"], int(row["block"])))
        start = recordings[0]["date"].replace("_", "-")
        stop = recordings[-1]["date"].replace("_", "-")
        span = f"{start} → {stop}" if start != stop else start
        journey = "<br>".join(
            f"<code>{recording['date'].replace('_', '-')}</code> · "
            f"{_membership_text(recording)}"
            for recording in recordings
        )
        mouse_id = mouse["mouse"]
        rows.append(
            "| "
            f"[{mouse_id}]({ATLAS}#mouse-{mouse_id}) | "
            f"{COHORT_LABELS[mouse['primary_cohort']]} | "
            f"{len(recordings)} | {span} | {journey} | "
            f"{_coverage_text(recordings)} |"
        )
    assert len(rows) == 19
    return "\n".join(rows)


def build_notebook() -> nbf.NotebookNode:
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
            "journeys-title",
            f"""
# Every imaging mouse: complete released-data journeys

This notebook follows all **19 imaging mice** through their **89 unique neural
acquisitions**. It keeps physical acquisitions separate from the 23 analysis
labels that reuse some acquisitions in more than one role.

It has three applied parts:

1. use the `Dataset` class to find every mouse, acquisition, analysis
   membership, and exact published file;
2. use `graph` to select a mouse or an exact acquisition–membership pair, load
   behavior + reduced neural activity + retinotopy, and draw a data dashboard;
3. reduce loaded sessions to acquisition, mouse, and cohort summaries before
   making aggregates.

Primary sources: [Animals]({NATURE}#Sec11), [experimental sequence]({NATURE}#Fig1),
[released imaging index]({INDEX_FILE}), [Figshare version 2]({FIGSHARE}), and the
[live analysis atlas]({ATLAS}#mice).
""",
        ),
        md(
            "journeys-boundaries",
            f"""
## What the release can and cannot show

Before the indexed imaging timeline, the paper reports surgery and recovery,
at least 3 days of handling, at least 3 days of head-fixation acclimation, and
at least 5 days of ball-running training
([Behavioural training]({NATURE}#Sec17)). Exact mouse-level dates for those
preparatory steps are not in the released imaging index, so this notebook does
not invent them.

The indexed sequence is `before Train 1 → after Train 1 → Test 1 / before
Train 2 → after Train 2 → Test 2 → Test 3`, with naive and grating-control
views where present. A date appears only when a neural acquisition exists.

Hard limits:

- there is no released cross-day cell registration; longitudinal plots compare
  acquisition- or mouse-level summaries, not the same neuron across days;
- imaging occurs at recorded endpoints, not on every training day;
- SVD components are fitted per acquisition, so component identities are not
  tracked across dates;
- `stim_id` defines functional stimulus role; a physical texture name alone
  does not;
- Figure 5's 23 behavior-only mice are not part of these 19 imaging journeys.

Relevant methods: [calcium processing]({NATURE}#Sec19),
[neural selectivity]({NATURE}#Sec20), [coding direction]({NATURE}#Sec21),
[reward prediction]({NATURE}#Sec22), and [retinotopy]({NATURE}#Sec25).
""",
        ),
        md(
            "journeys-plot-map",
            f"""
## Which plots are relevant at each recorded stage

| Indexed stage | Direct descriptive views in this notebook | Published analysis to consult before interpreting |
|---|---|---|
| Every acquisition | dated timeline, exact file manifest, lick raster, running speed, released SVD time courses, cortical area map | [imaging]({NATURE}#Sec13), [calcium processing]({NATURE}#Sec19), [retinotopy]({NATURE}#Sec25) |
| Train 1 before/after | the same QC views in paired acquisition context | [Figure 1]({NATURE}#Fig1), [neural-selectivity method]({NATURE}#Sec20) |
| Test 1 / before Train 2 | four-role behavior and population-session views | [Figure 2]({NATURE}#Fig2), [coding-direction method]({NATURE}#Sec21) |
| Train 2 after | post-training behavior and population-session views | [Figure 3]({NATURE}#Fig3), [Results: familiarity]({NATURE}#Sec4) |
| Test 2 | leaf3 behavior and population-session views | [Extended Data Figure 6]({NATURE}/figures/11), [Results: recognition memory]({NATURE}#Sec5) |
| Test 3 | keep swap1 and swap2 analysis memberships separate | [Extended Data Figure 7]({NATURE}/figures/12), [Results: recognition memory]({NATURE}#Sec5) |
| Task-mouse Test 1 and later tests | cue, reward, and lick-aligned extensions | [Figure 4]({NATURE}#Fig4), [reward-prediction method]({NATURE}#Sec22) |

The notebook's plots are release-data QC and orientation. They do not replace
the paper's estimators or establish a training, reward, or cohort effect.
""",
        ),
        md(
            "journeys-all-mice",
            f"""
## The 19 complete indexed journeys

Each dated line is one physical acquisition followed by every analysis
membership assigned to it in the released index. `B/A` in the final column
means both before and after endpoints exist; it does not imply daily neural
measurements between them.

| Mouse | Primary cohort | Acquisitions | Indexed date span | Every acquisition and membership | Indexed endpoints |
|---|---|---:|---|---|---|
{_all_mouse_table()}

Source: [exact released `Imaging_Exp_info.npy`]({INDEX_FILE}). Counts and joins
are checked below against the pinned [Figshare v2 inventory]({FIGSHARE}).
""",
        ),
        md(
            "journeys-setup-intro",
            """
## Connect to the shared release

Add **Zhong et al. 2025 - Neuromatch Team Workspace** as a shortcut in
**My Drive**. The next cell imports `drive.py`, `graph.py`, and the canonical
map from that exact workspace. The committed notebook contains no saved data
outputs or disconnected widgets.
""",
        ),
        py(
            "journeys-setup",
            """
# @title Connect to Dataset and graph
import importlib
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

IN_COLAB = False
try:
    from google.colab import drive as google_drive
except ImportError:
    WORKSPACE = Path.cwd()
else:
    IN_COLAB = True
    google_drive.mount("/content/drive", force_remount=False)
    from google.colab import output as colab_output
    colab_output.enable_custom_widget_manager()
    WORKSPACE = Path(
        "/content/drive/MyDrive/Zhong et al. 2025 - Neuromatch Team Workspace"
    )

required = (
    WORKSPACE / "drive.py",
    WORKSPACE / "graph.py",
    WORKSPACE / "zhong2025/assets/canonical_map.json",
    WORKSPACE / "zhong2025/assets/imaging-experiment-index.json",
)
missing = [str(path) for path in required if not path.is_file()]
assert not missing, "Missing required workspace files:\\n" + "\\n".join(missing)

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
for module_name in tuple(sys.modules):
    if module_name in {"drive", "graph", "zhong2025"} or module_name.startswith("zhong2025."):
        sys.modules.pop(module_name, None)

drive = importlib.import_module("drive")
graph = importlib.import_module("graph")
if IN_COLAB:
    assert Path(drive.__file__).resolve() == (WORKSPACE / "drive.py").resolve()
    assert Path(graph.__file__).resolve() == (WORKSPACE / "graph.py").resolve()

data = drive.setup()
from zhong2025.catalog import load_map

canonical = load_map()
assert len(canonical["mice"]) == 19
assert len(canonical["recordings"]) == 89
assert len(data.files) == 297
assert sum(item.size_bytes for item in data.files) == 452_233_500_962
data
""",
        ),
        md(
            "journeys-dataset-intro",
            f"""
# Tutorial 1 — apply the `Dataset` class to every mouse

The `data` object above displays its variables and public functions. The core
path is:

`data.recordings(mouse=...) → Recording → Recording.file(...) → Recording.load(...)`

The next cell builds the 19 journeys by joining the canonical projection of
the [released imaging index]({INDEX_FILE}) to `Dataset.recording()`. It verifies
that the two sources resolve the same 89 physical acquisitions before plotting
anything.
""",
        ),
        py(
            "journeys-build",
            """
# @title Build and validate every mouse journey
COHORT_ORDER = {"sup": 0, "unsup": 1, "grating": 2, "naive": 3}
COHORT_LABELS = {
    "sup": "Task / rewarded",
    "unsup": "Unrewarded natural texture",
    "grating": "Grating exposure control",
    "naive": "Naive only",
}
ATLAS = "https://shibasis.dev/neuromatch/"


def membership_label(membership):
    moment = f" {membership['moment']}" if membership.get("moment") else ""
    return f"{membership['cohort']} · {membership['phase']}{moment}"


def build_journeys(canonical_map, dataset):
    canonical_ids = set(canonical_map["recordings"])
    dataset_ids = {recording.recording_id for recording in dataset.recordings()}
    if canonical_ids != dataset_ids:
        raise ValueError(
            f"Recording join differs: canonical-only={sorted(canonical_ids-dataset_ids)}, "
            f"dataset-only={sorted(dataset_ids-canonical_ids)}"
        )

    journeys = {}
    for mouse_id, mouse in canonical_map["mice"].items():
        acquisitions = []
        for recording_id in mouse["recordings"]:
            metadata = canonical_map["recordings"][recording_id]
            recording = dataset.recording(recording_id)
            unique_memberships = tuple({
                membership["label"]: membership
                for membership in metadata["memberships"]
            }.values())
            acquisitions.append({
                "recording_id": recording_id,
                "mouse": mouse_id,
                "date": date.fromisoformat(metadata["date"].replace("_", "-")),
                "block": int(metadata["block"]),
                "cohort": metadata["cohort"],
                "stage": metadata["stage"],
                "memberships": unique_memberships,
                "experiments": recording.experiments,
                "layers": recording.layers,
                "retinotopy_id": recording.retinotopy_id,
            })
        acquisitions.sort(key=lambda row: (row["date"], row["block"]))
        journeys[mouse_id] = {
            "mouse": mouse_id,
            "primary_cohort": mouse["primary_cohort"],
            "also": tuple(mouse["also"]),
            "acquisitions": tuple(acquisitions),
        }
    return journeys


journeys = build_journeys(canonical, data)
print({
    "mice": len(journeys),
    "physical_acquisitions": sum(len(row["acquisitions"]) for row in journeys.values()),
    "unique_experiment_recording_memberships": sum(
        len(acquisition["memberships"])
        for row in journeys.values()
        for acquisition in row["acquisitions"]
    ),
})
""",
        ),
        py(
            "journeys-dataset-example",
            """
# @title Select one mouse and inspect exact Recording objects and files
SELECTED_MOUSE = "TX119"
selected_sessions = data.recordings(mouse=SELECTED_MOUSE)
selected_journey = journeys[SELECTED_MOUSE]

print(f"{SELECTED_MOUSE}: {len(selected_sessions)} physical acquisitions")
for session in selected_sessions:
    print(session.recording_id, session.experiments, session.layers)

selected_session = data.recording("TX119_2023_12_24_1")
rows = []
for item in selected_session.files:
    rows.append(
        f"| `{item.category}` | `{item.name}` | {item.id} | "
        f"{item.size_bytes / 2**20:,.1f} MiB | `{item.md5}` |"
    )
display(Markdown(
    "| Layer | Exact published file | Figshare file id | Size | MD5 |\\n"
    "|---|---|---:|---:|---|\\n" + "\\n".join(rows)
))
""",
        ),
        py(
            "journeys-metadata-plots",
            """
# @title Plot all 19 journeys and their indexed endpoint coverage
STAGE_COLUMNS = (
    ("Naive", "Naive", None),
    ("Train 1 B", "Train 1", "before"),
    ("Train 1 A", "Train 1", "after"),
    ("Test 1", "Test 1", None),
    ("Train 2 B", "Train 2", "before"),
    ("Train 2 A", "Train 2", "after"),
    ("Test 2", "Test 2", None),
    ("Test 3", "Test 3", None),
)
COHORT_COLOURS = {
    "sup": "#2f855a",
    "unsup": "#3182bd",
    "grating": "#d97706",
    "naive": "#7c3aed",
}


def has_stage(journey, phase, moment=None):
    if phase == "Naive":
        return any(
            membership["cohort"] == "naive"
            for acquisition in journey["acquisitions"]
            for membership in acquisition["memberships"]
        )
    return any(
        membership["phase"] == phase
        and (moment is None or membership.get("moment") == moment)
        for acquisition in journey["acquisitions"]
        for membership in acquisition["memberships"]
    )


def plot_all_journeys(journey_map):
    order = sorted(
        journey_map,
        key=lambda mouse: (
            COHORT_ORDER[journey_map[mouse]["primary_cohort"]], mouse
        ),
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(17, 10), gridspec_kw={"width_ratios": [1.35, 1]},
        constrained_layout=True,
    )

    coverage = np.zeros((len(order), len(STAGE_COLUMNS)), dtype=int)
    for y, mouse in enumerate(order):
        journey = journey_map[mouse]
        start = journey["acquisitions"][0]["date"]
        days = np.array([(row["date"] - start).days for row in journey["acquisitions"]])
        colour = COHORT_COLOURS[journey["primary_cohort"]]
        axes[0].plot(days, np.full_like(days, y), color=colour, alpha=0.45)
        axes[0].scatter(days, np.full_like(days, y), color=colour, s=38)
        for x, acquisition in zip(days, journey["acquisitions"]):
            axes[0].annotate(
                acquisition["date"].strftime("%m-%d"), (x, y), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=6.5,
            )
        for column, (_, phase, moment) in enumerate(STAGE_COLUMNS):
            coverage[y, column] = has_stage(journey, phase, moment)

    axes[0].set(
        yticks=np.arange(len(order)),
        yticklabels=[
            f"{mouse} · {journey_map[mouse]['acquisitions'][0]['date'].isoformat()}"
            for mouse in order
        ],
        xlabel="days since that mouse's first indexed neural acquisition",
        title="True within-mouse acquisition spacing (dates label each point)",
    )
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].imshow(coverage, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    axes[1].set(
        xticks=np.arange(len(STAGE_COLUMNS)),
        xticklabels=[row[0] for row in STAGE_COLUMNS],
        yticks=np.arange(len(order)),
        yticklabels=order,
        title="Indexed endpoint availability (1 = present)",
    )
    axes[1].tick_params(axis="x", rotation=45)
    figure.suptitle("Released imaging journeys: 19 mice, 89 physical acquisitions")
    return figure


all_journeys_figure = plot_all_journeys(journeys)
all_journeys_figure
""",
        ),
        py(
            "journeys-preflight",
            """
# @title Preflight a complete mouse before loading any array
DEFAULT_LAYERS = ("behavior", "reduced_neural", "retinotopy")


def mouse_file_plan(mouse, layers=DEFAULT_LAYERS):
    planned = {}
    for recording in data.recordings(mouse=mouse):
        for layer in layers:
            if layer == "behavior":
                for experiment in recording.experiments:
                    item = recording.file("behavior", experiment=experiment)
                    planned[item.id] = item
            else:
                item = recording.file(layer)
                planned[item.id] = item
    rows = sorted(planned.values(), key=lambda item: (item.category, item.name))
    return {
        "mouse": mouse,
        "layers": tuple(layers),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(item.size_bytes for item in rows),
    }


selected_plan = mouse_file_plan(SELECTED_MOUSE)
by_layer = defaultdict(lambda: {"files": 0, "bytes": 0})
for item in selected_plan["files"]:
    by_layer[item.category]["files"] += 1
    by_layer[item.category]["bytes"] += item.size_bytes

print(
    SELECTED_MOUSE,
    f"{selected_plan['file_count']} unique published files",
    f"{selected_plan['total_bytes'] / 2**30:,.3f} GiB",
)
for layer, row in sorted(by_layer.items()):
    print(f"{layer:24s} {row['files']:2d} files  {row['bytes'] / 2**30:7.3f} GiB")
""",
        ),
        md(
            "journeys-load-intro",
            f"""
## Load one exact acquisition–membership pair

The compact applied path loads three release layers:

- `behavior` for one exact experiment membership;
- `reduced_neural`, the released per-acquisition SVD (`U`, `V`);
- `retinotopy` (`xy_t`, `iarea`).

The cortical plot uses the paper code's coordinate transform
`x = -xy_t[:, 1]`, `y = xy_t[:, 0]`
([exact implementation](https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416)).
The SVD panel is an orientation/QC view of released `V` rows; component
identity is not compared across dates. Full neural arrays are deliberately not
loaded by this tutorial; publication-faithful single-neuron selectivity belongs
with [the authors' estimator]({NATURE}#Sec20).
""",
        ),
        py(
            "journeys-session-tools",
            """
# @title Define exact loading, summaries, and one-session plots
AREA_GROUPS = {
    "V1": (8,),
    "Medial": (0, 1, 2, 9),
    "Lateral": (5, 6),
    "Anterior": (3, 4),
}
AREA_COLOURS = {
    "V1": "#4c78a8",
    "Medial": "#59a14f",
    "Lateral": "#f28e2b",
    "Anterior": "#e15759",
    "Excluded / unassigned": "#9aa2ad",
}


def resolve_selection(selection):
    recording_id, experiment = selection.split("|", 1)
    recording = data.recording(recording_id)
    if experiment not in recording.experiments:
        raise ValueError(
            f"{recording_id} has no {experiment!r} membership; "
            f"choose one of {recording.experiments}"
        )
    metadata = canonical["recordings"][recording_id]
    return {
        "selection": selection,
        "recording_id": recording_id,
        "experiment": experiment,
        "mouse": recording.mouse,
        "date": recording.date,
        "stage": metadata["stage"],
        "recording": recording,
    }


def session_file_plan(session_spec):
    recording = session_spec["recording"]
    files = (
        recording.file("behavior", experiment=session_spec["experiment"]),
        recording.file("reduced_neural"),
        recording.file("retinotopy"),
    )
    return {
        "files": files,
        "total_bytes": sum(item.size_bytes for item in files),
        "file_ids": tuple(item.id for item in files),
    }


def load_session_bundle(session_spec, load_arrays=False, max_gib=10.0):
    plan = session_file_plan(session_spec)
    bundle = {"loaded": False, "plan": plan, "spec": session_spec}
    if not load_arrays:
        return bundle
    recording = session_spec["recording"]
    bundle.update({
        "loaded": True,
        "behavior": recording.load(
            "behavior", experiment=session_spec["experiment"], max_gib=max_gib
        ),
        "svd": recording.load("reduced_neural", max_gib=max_gib),
        "retinotopy": recording.load("retinotopy", max_gib=max_gib),
    })
    return bundle


def summarize_behavior(behavior):
    frame_position = np.asarray(behavior.get("ft_Pos", ()), dtype=float)
    frame_speed = np.asarray(behavior.get("ft_RunSpeed", ()), dtype=float)
    trial_id = np.asarray(behavior.get("ft_trInd", ()), dtype=float)
    n_trials = int(behavior.get("ntrials", 0) or 0)
    if n_trials == 0 and trial_id.size:
        finite_trials = trial_id[np.isfinite(trial_id)]
        n_trials = int(np.unique(finite_trials).size)
    lick_count = int(np.asarray(behavior.get("LickPos", ())).size)
    return {
        "frames": int(frame_position.size),
        "trials": n_trials,
        "licks": lick_count,
        "licks_per_trial": lick_count / n_trials if n_trials else np.nan,
        "mean_run_speed": float(np.nanmean(frame_speed)) if frame_speed.size else np.nan,
    }


def summarize_svd(svd):
    U = np.asarray(svd["U"])
    V = np.asarray(svd["V"])
    if U.ndim != 2 or V.ndim != 2 or U.shape[0] != V.shape[0]:
        raise ValueError("Expected released U and V to share a component axis")
    return {
        "components": int(U.shape[0]),
        "neurons": int(U.shape[1]),
        "frames": int(V.shape[1]),
    }


def summarize_retinotopy(retinotopy):
    xy_t = np.asarray(retinotopy["xy_t"], dtype=float)
    iarea = np.asarray(retinotopy["iarea"])
    if xy_t.ndim != 2 or xy_t.shape[1] != 2 or iarea.shape != (len(xy_t),):
        raise ValueError("xy_t and iarea must share one neuron axis")
    groups = {
        label: int(np.isin(iarea, codes).sum())
        for label, codes in AREA_GROUPS.items()
    }
    groups["Excluded / unassigned"] = int((~np.isin(
        iarea, [code for codes in AREA_GROUPS.values() for code in codes]
    )).sum())
    return {"neurons": int(len(iarea)), "area_counts": groups}


def _plot_one_mouse_timeline(axis, mouse, selected_recording_id=None):
    journey = journeys[mouse]
    start = journey["acquisitions"][0]["date"]
    days = np.array([(row["date"] - start).days for row in journey["acquisitions"]])
    colours = [
        "#dc2626" if row["recording_id"] == selected_recording_id else
        COHORT_COLOURS[journey["primary_cohort"]]
        for row in journey["acquisitions"]
    ]
    axis.plot(days, np.zeros_like(days), color="#94a3b8", linewidth=1.5)
    axis.scatter(days, np.zeros_like(days), c=colours, s=55, zorder=3)
    for x, row in zip(days, journey["acquisitions"]):
        axis.annotate(
            f"{row['date'].isoformat()}\\n{row['stage']}", (x, 0), xytext=(0, 9),
            textcoords="offset points", rotation=35, ha="left", va="bottom", fontsize=7,
        )
    axis.set(
        yticks=[], xlabel="days since first indexed acquisition",
        title=f"{mouse}: every physical acquisition (red = selected)",
    )
    axis.grid(axis="x", alpha=0.2)


def plot_session_dashboard(bundle, session_spec):
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    _plot_one_mouse_timeline(
        axes[0, 0], session_spec["mouse"], session_spec["recording_id"]
    )
    plan = bundle["plan"]
    if not bundle["loaded"]:
        categories = [item.category for item in plan["files"]]
        sizes = [item.size_bytes / 2**20 for item in plan["files"]]
        axes[0, 1].bar(categories, sizes, color="#4c78a8")
        axes[0, 1].set(title="Exact selected files", ylabel="MiB")
        for axis in axes.flat[2:]:
            axis.axis("off")
        axes[0, 2].axis("off")
        axes[0, 2].text(
            0, 1,
            "Metadata-only preview\\n\\n"
            f"recording: {session_spec['recording_id']}\\n"
            f"membership: {session_spec['experiment']}\\n"
            f"files: {plan['file_ids']}\\n"
            f"total: {plan['total_bytes']/2**20:,.1f} MiB\\n\\n"
            "Set load_arrays=True in the graph and run again.",
            transform=axes[0, 2].transAxes, va="top", family="monospace",
        )
        figure.suptitle("Released-data dashboard · no arrays loaded")
        return figure

    behavior = bundle["behavior"]
    svd = bundle["svd"]
    retinotopy = bundle["retinotopy"]
    behavior_qc = summarize_behavior(behavior)
    svd_qc = summarize_svd(svd)
    retinotopy_qc = summarize_retinotopy(retinotopy)

    lick_position = np.asarray(behavior.get("LickPos", ()), dtype=float) / 10.0
    lick_trial = np.asarray(behavior.get("LickTrind", ()), dtype=float)
    if lick_position.size and lick_trial.size:
        n = min(lick_position.size, lick_trial.size)
        axes[0, 1].scatter(lick_position[:n], lick_trial[:n], s=5, alpha=0.45)
    sound_position = np.asarray(behavior.get("SoundPos", ()), dtype=float) / 10.0
    if sound_position.size:
        axes[0, 1].scatter(
            sound_position, np.arange(sound_position.size), marker="|", s=45,
            color="#7c3aed", label="sound cue",
        )
    axes[0, 1].axvline(4, color="#64748b", linestyle="--", linewidth=1)
    axes[0, 1].set(
        title="Behavior: lick raster and cue positions",
        xlabel="corridor position (m)", ylabel="trial index", xlim=(0, 6),
    )
    if sound_position.size:
        axes[0, 1].legend(fontsize=7)

    frame_position = np.asarray(behavior.get("ft_Pos", ()), dtype=float) / 10.0
    frame_speed = np.asarray(behavior.get("ft_RunSpeed", ()), dtype=float)
    frame_n = min(frame_position.size, frame_speed.size)
    bins = np.linspace(0, 6, 31)
    centers = (bins[:-1] + bins[1:]) / 2
    speed_profile = np.full(len(centers), np.nan)
    if frame_n:
        bin_id = np.digitize(frame_position[:frame_n], bins) - 1
        for index in range(len(centers)):
            selected = bin_id == index
            if np.any(selected):
                speed_profile[index] = np.nanmean(frame_speed[:frame_n][selected])
    axes[0, 2].plot(centers, speed_profile, color="#4c78a8")
    axes[0, 2].axvline(4, color="#64748b", linestyle="--", linewidth=1)
    axes[0, 2].set(
        title="Behavior: mean running speed by position",
        xlabel="corridor position (m)", ylabel="released speed units", xlim=(0, 6),
    )

    V = np.asarray(svd["V"], dtype=float)
    display_components = min(4, V.shape[0])
    display_frames = min(1800, V.shape[1])
    for component in range(display_components):
        values = V[component, :display_frames]
        scale = np.nanstd(values)
        z = (values - np.nanmean(values)) / scale if scale > 0 else values * 0
        axes[1, 0].plot(z + component * 5, linewidth=0.7, label=f"V[{component}]")
    axes[1, 0].set(
        title="Reduced neural: first released V rows",
        xlabel="neural frame (first 1,800)", ylabel="display z-score + offset",
    )

    xy_t = np.asarray(retinotopy["xy_t"], dtype=float)
    iarea = np.asarray(retinotopy["iarea"])
    assigned = np.zeros(len(iarea), dtype=bool)
    for label, codes in AREA_GROUPS.items():
        selected = np.isin(iarea, codes)
        assigned |= selected
        axes[1, 1].scatter(
            -xy_t[selected, 1], xy_t[selected, 0], s=2, alpha=0.35,
            color=AREA_COLOURS[label], label=label,
        )
    axes[1, 1].scatter(
        -xy_t[~assigned, 1], xy_t[~assigned, 0], s=2, alpha=0.2,
        color=AREA_COLOURS["Excluded / unassigned"], label="Excluded / unassigned",
    )
    axes[1, 1].set(
        title="Retinotopy: paper-coordinate neuron map",
        xlabel="-xy_t[:, 1]", ylabel="xy_t[:, 0]", aspect="equal",
    )
    axes[1, 1].legend(markerscale=4, fontsize=6, ncol=2)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0, 1,
        f"recording: {session_spec['recording_id']}\\n"
        f"membership: {session_spec['experiment']}\\n"
        f"stage: {session_spec['stage']}\\n\\n"
        f"SVD: {svd_qc['components']} components\\n"
        f"neurons: {svd_qc['neurons']:,}\\n"
        f"neural frames: {svd_qc['frames']:,}\\n"
        f"behavior trials: {behavior_qc['trials']:,}\\n"
        f"licks: {behavior_qc['licks']:,}\\n"
        f"retinotopy neurons: {retinotopy_qc['neurons']:,}\\n\\n"
        "QC/orientation only; no biological effect estimated.",
        transform=axes[1, 2].transAxes, va="top", family="monospace",
    )
    figure.suptitle(
        f"{session_spec['mouse']} · {session_spec['date']} · {session_spec['experiment']}"
    )
    return figure
""",
        ),
        py(
            "journeys-load-example",
            """
# @title Preview one exact selection; change to True to load and plot real arrays
DEFAULT_SELECTION = "TX119_2023_12_24_1|unsup_test1"
RUN_ONE_SESSION = False

example_spec = resolve_selection(DEFAULT_SELECTION)
example_bundle = load_session_bundle(
    example_spec, load_arrays=RUN_ONE_SESSION, max_gib=10.0
)
example_dashboard = plot_session_dashboard(example_bundle, example_spec)
example_dashboard
""",
        ),
        md(
            "journeys-graph-intro",
            """
# Tutorial 2 — express the same work as `graph` flows

A graph node is an ordinary Python function. Function parameters are input
ports; names declared in `outputs=` are output ports. Matching names wire
automatically. Unconnected inputs become controls.

The first graph is metadata-only and selects among all 19 mice. The second
graph selects an exact acquisition–membership pair and uses the same Dataset
loading functions defined above. The graph does not create a second data model;
it makes the existing selection, file plan, load, and plot steps visible.
""",
        ),
        py(
            "journeys-graph-metadata",
            """
# @title Build a mouse-selection graph (no data arrays)
@graph.node(outputs="mouse_id")
def choose_mouse(mouse="TX119"):
    if mouse not in journeys:
        raise ValueError(f"Unknown imaging mouse {mouse!r}")
    return mouse


@graph.node(outputs="journey")
def resolve_journey(mouse_id):
    return journeys[mouse_id]


@graph.node(outputs="file_plan")
def plan_journey_files(mouse_id):
    return mouse_file_plan(mouse_id)


@graph.node(outputs="journey_plot", cache=False)
def draw_journey(journey, file_plan):
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    _plot_one_mouse_timeline(axes[0], journey["mouse"])
    by_layer = defaultdict(int)
    for item in file_plan["files"]:
        by_layer[item.category] += item.size_bytes
    labels = sorted(by_layer)
    axes[1].bar(labels, [by_layer[label] / 2**30 for label in labels], color="#4c78a8")
    axes[1].set(
        title=f"Exact unique-file preflight · {file_plan['total_bytes']/2**30:,.3f} GiB",
        ylabel="GiB", xlabel="published layer",
    )
    axes[1].tick_params(axis="x", rotation=25)
    figure.suptitle(f"{journey['mouse']} · metadata journey and release-file plan")
    return figure


journey_graph = graph.Graph(
    "Choose one imaging mouse",
    choose_mouse,
    resolve_journey,
    plan_journey_files,
    draw_journey,
)
journey_graph.describe()
""",
        ),
        py(
            "journeys-graph-metadata-widget",
            """
# @title Run the interactive 19-mouse journey graph
MOUSE_CHOICES = [
    (f"{mouse} · {COHORT_LABELS[journeys[mouse]['primary_cohort']]}", mouse)
    for mouse in sorted(
        journeys,
        key=lambda current: (
            COHORT_ORDER[journeys[current]["primary_cohort"]], current
        ),
    )
]
journey_panel = journey_graph.widget(
    controls={"mouse": MOUSE_CHOICES},
    show="journey_plot",
    auto_run=True,
)
journey_panel
""",
        ),
        py(
            "journeys-graph-data",
            """
# @title Build an exact acquisition-membership loading graph
@graph.node(outputs="selected_membership")
def choose_membership(selection=DEFAULT_SELECTION):
    return selection


@graph.node(outputs="session_spec")
def resolve_membership(selected_membership):
    return resolve_selection(selected_membership)


@graph.node(outputs="file_plan")
def plan_session(session_spec):
    return session_file_plan(session_spec)


@graph.node(outputs="bundle", cache=False)
def load_selected_layers(session_spec, load_arrays=False, max_gib=10.0):
    return load_session_bundle(
        session_spec, load_arrays=load_arrays, max_gib=max_gib
    )


@graph.node(outputs="session_dashboard", cache=False)
def draw_selected_session(bundle, session_spec):
    return plot_session_dashboard(bundle, session_spec)


session_graph = graph.Graph(
    "Load and inspect one exact released acquisition",
    choose_membership,
    resolve_membership,
    plan_session,
    load_selected_layers,
    draw_selected_session,
)
session_graph.diagram()
""",
        ),
        py(
            "journeys-graph-data-widget",
            """
# @title Select a membership; choose Real arrays; then Run flow
MEMBERSHIP_CHOICES = []
for mouse in sorted(journeys):
    for acquisition in journeys[mouse]["acquisitions"]:
        for experiment in acquisition["experiments"]:
            label = (
                f"{mouse} · {acquisition['date'].isoformat()} · "
                f"{acquisition['stage']} · {experiment}"
            )
            MEMBERSHIP_CHOICES.append(
                (label, f"{acquisition['recording_id']}|{experiment}")
            )

session_panel = session_graph.widget(
    controls={
        "selection": MEMBERSHIP_CHOICES,
        "load_arrays": [("Metadata only", False), ("Real arrays", True)],
        "max_gib": 10.0,
    },
    show="session_dashboard",
    auto_run=False,
)
session_panel
""",
        ),
        md(
            "journeys-batch-intro",
            """
## Process a complete mouse without retaining every large array

The batch loader below is sequential and reduction-first:

1. load each physical acquisition's SVD and retinotopy once;
2. reduce them immediately to shapes and cortical area counts;
3. load each behavior file once per experiment label;
4. reduce every acquisition–membership view to trial, frame, speed, and lick
   counts;
5. discard the source arrays before continuing.

This preserves all analysis memberships—including separate Test 3 swap views—
while keeping acquisition summaries distinct from behavior-view summaries.
Run the file preflight before enabling it.
""",
        ),
        py(
            "journeys-batch-tools",
            """
# @title Define sequential reduction for one or many mice
def load_many_mouse_journeys(mouse_ids, max_gib=10.0):
    mouse_ids = tuple(dict.fromkeys(mouse_ids))
    unknown = sorted(set(mouse_ids) - set(journeys))
    if unknown:
        raise ValueError(f"Unknown imaging mice: {unknown}")

    selected_recordings = {
        recording.recording_id: recording
        for mouse in mouse_ids
        for recording in data.recordings(mouse=mouse)
    }
    acquisition_rows = []
    for index, recording in enumerate(selected_recordings.values(), start=1):
        print(f"neural/retinotopy {index}/{len(selected_recordings)}: {recording.recording_id}")
        svd = recording.load("reduced_neural", max_gib=max_gib)
        retinotopy = recording.load("retinotopy", max_gib=max_gib)
        metadata = canonical["recordings"][recording.recording_id]
        svd_summary = summarize_svd(svd)
        retinotopy_summary = summarize_retinotopy(retinotopy)
        if svd_summary["neurons"] != retinotopy_summary["neurons"]:
            raise ValueError(
                f"Neuron-axis mismatch for {recording.recording_id}: "
                f"SVD={svd_summary['neurons']}, "
                f"retinotopy={retinotopy_summary['neurons']}"
            )
        acquisition_rows.append({
            "mouse": recording.mouse,
            "recording_id": recording.recording_id,
            "date": date.fromisoformat(recording.date),
            "stage": metadata["stage"],
            **svd_summary,
            "retinotopy_neurons": retinotopy_summary["neurons"],
            "area_counts": retinotopy_summary["area_counts"],
        })
        del svd, retinotopy

    experiment_recordings = defaultdict(list)
    for recording in selected_recordings.values():
        for experiment in recording.experiments:
            experiment_recordings[experiment].append(recording.recording_id)

    behavior_rows = []
    for index, experiment in enumerate(sorted(experiment_recordings), start=1):
        print(f"behavior {index}/{len(experiment_recordings)}: {experiment}")
        published = data.load(f"Beh_{experiment}.npy", max_gib=max_gib)
        for recording_id in experiment_recordings[experiment]:
            if recording_id not in published:
                raise ValueError(f"Beh_{experiment}.npy lacks {recording_id}")
            metadata = canonical["recordings"][recording_id]
            behavior_rows.append({
                "mouse": metadata["mouse"],
                "recording_id": recording_id,
                "date": date.fromisoformat(metadata["date"].replace("_", "-")),
                "stage": metadata["stage"],
                "experiment": experiment,
                **summarize_behavior(published[recording_id]),
            })
        del published

    return {
        mouse: {
            "mouse": mouse,
            "cohort": journeys[mouse]["primary_cohort"],
            "acquisitions": [row for row in acquisition_rows if row["mouse"] == mouse],
            "behavior_views": [row for row in behavior_rows if row["mouse"] == mouse],
        }
        for mouse in mouse_ids
    }


def plot_loaded_mouse_journey(result):
    acquisitions = sorted(result["acquisitions"], key=lambda row: row["date"])
    behavior = sorted(result["behavior_views"], key=lambda row: (row["date"], row["experiment"]))
    start = acquisitions[0]["date"]
    acquisition_day = np.array([(row["date"] - start).days for row in acquisitions])
    behavior_day = np.array([(row["date"] - start).days for row in behavior])

    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    _plot_one_mouse_timeline(axes[0, 0], result["mouse"])
    axes[0, 1].plot(acquisition_day, [row["neurons"] for row in acquisitions], "o-")
    axes[0, 1].set(title="SVD neuron axis by acquisition", xlabel="day", ylabel="neurons")
    axes[0, 2].plot(acquisition_day, [row["frames"] for row in acquisitions], "o-")
    axes[0, 2].set(title="SVD neural frames by acquisition", xlabel="day", ylabel="frames")

    for area in (*AREA_GROUPS, "Excluded / unassigned"):
        fraction = [row["area_counts"][area] / row["neurons"] for row in acquisitions]
        axes[1, 0].plot(acquisition_day, fraction, "o-", label=area)
    axes[1, 0].set(
        title="Retinotopy area fractions", xlabel="day", ylabel="fraction of neurons"
    )
    axes[1, 0].legend(fontsize=6, ncol=2)

    axes[1, 1].scatter(behavior_day, [row["trials"] for row in behavior], s=28)
    axes[1, 1].set(
        title="Trials in every behavior membership view", xlabel="day", ylabel="trials"
    )
    axes[1, 2].scatter(behavior_day, [row["licks_per_trial"] for row in behavior], s=28)
    axes[1, 2].set(
        title="Licks per trial in every behavior view", xlabel="day", ylabel="licks / trial"
    )
    figure.suptitle(
        f"{result['mouse']} · descriptive release QC across the complete indexed journey"
    )
    return figure
""",
        ),
        py(
            "journeys-batch-run",
            """
# @title Set True only after reading the exact file preflight above
RUN_SELECTED_MOUSE = False

loaded_mouse_results = {}
if RUN_SELECTED_MOUSE:
    loaded_mouse_results = load_many_mouse_journeys([SELECTED_MOUSE], max_gib=10.0)
    loaded_mouse_figure = plot_loaded_mouse_journey(
        loaded_mouse_results[SELECTED_MOUSE]
    )
    display(loaded_mouse_figure)
else:
    print(
        f"Not loaded. {SELECTED_MOUSE} preflight: "
        f"{selected_plan['file_count']} unique files, "
        f"{selected_plan['total_bytes']/2**30:,.3f} GiB."
    )
""",
        ),
        md(
            "journeys-aggregate-intro",
            """
## Aggregates at three explicit units

1. **Acquisition inventory:** all 89 dated neural acquisitions; immediate and
   metadata-only.
2. **Behavior membership views:** one row for each released experiment–recording
   membership; do not call these independent neural acquisitions.
3. **Mouse summaries:** reduce acquisitions or behavior views within each mouse
   before comparing cohorts. Neurons, frames, and trials are repeated
   measurements, not the cohort sample size.

The following functions create descriptive QC aggregates. They do not perform
the paper's selectivity, coding-direction, reward-prediction, or inferential
tests.
""",
        ),
        py(
            "journeys-aggregates",
            """
# @title Metadata aggregates now; optional data aggregates after batch loading
def metadata_aggregate(journey_map):
    rows = []
    for mouse, journey in journey_map.items():
        dates = [row["date"] for row in journey["acquisitions"]]
        gaps = np.diff(sorted(dates)).astype("timedelta64[D]").astype(int) if len(dates) > 1 else np.array([])
        rows.append({
            "mouse": mouse,
            "cohort": journey["primary_cohort"],
            "acquisitions": len(dates),
            "span_days": (max(dates) - min(dates)).days,
            "median_gap_days": float(np.median(gaps)) if len(gaps) else np.nan,
            "memberships": sum(len(row["memberships"]) for row in journey["acquisitions"]),
        })
    return rows


def aggregate_loaded_mice(mouse_results):
    rows = []
    for mouse, result in mouse_results.items():
        acquisitions = result["acquisitions"]
        behavior = result["behavior_views"]
        rows.append({
            "mouse": mouse,
            "cohort": result["cohort"],
            "acquisitions": len(acquisitions),
            "behavior_views": len(behavior),
            "median_neurons": float(np.median([row["neurons"] for row in acquisitions])),
            "median_neural_frames": float(np.median([row["frames"] for row in acquisitions])),
            "median_trials": float(np.median([row["trials"] for row in behavior])),
            "median_licks_per_trial": float(np.nanmedian([
                row["licks_per_trial"] for row in behavior
            ])),
        })
    return rows


def plot_aggregate_rows(rows, title):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    order = sorted(rows, key=lambda row: (COHORT_ORDER[row["cohort"]], row["mouse"]))
    colours = [COHORT_COLOURS[row["cohort"]] for row in order]
    axes[0].bar([row["mouse"] for row in order], [row["acquisitions"] for row in order], color=colours)
    axes[0].set(title="Physical acquisitions", ylabel="count")
    axes[1].bar([row["mouse"] for row in order], [row["span_days"] for row in order], color=colours)
    axes[1].set(title="Indexed date span", ylabel="days")
    axes[2].bar([row["mouse"] for row in order], [row["memberships"] for row in order], color=colours)
    axes[2].set(title="Analysis memberships", ylabel="count")
    for axis in axes:
        axis.tick_params(axis="x", rotation=70)
    figure.suptitle(title)
    return figure


inventory_rows = metadata_aggregate(journeys)
inventory_aggregate_figure = plot_aggregate_rows(
    inventory_rows, "All 19 mice · release inventory aggregates"
)
inventory_aggregate_figure

# Optional all-mouse real-data reduction. This is intentionally off.
RUN_ALL_MICE = False
all_mouse_results = {}
if RUN_ALL_MICE:
    all_mouse_results = load_many_mouse_journeys(sorted(journeys), max_gib=10.0)
    mouse_qc_rows = aggregate_loaded_mice(all_mouse_results)
    display(mouse_qc_rows)
else:
    unique_files = {}
    for mouse in journeys:
        for item in mouse_file_plan(mouse)["files"]:
            unique_files[item.id] = item
    print(
        "All-mouse SVD + behavior + retinotopy preflight:",
        len(unique_files), "unique files,",
        f"{sum(item.size_bytes for item in unique_files.values())/2**30:,.3f} GiB",
    )
""",
        ),
        md(
            "journeys-limits",
            f"""
## Interpretation and next analysis

Use this notebook to answer: *which mouse, which date, which analysis role,
which exact files, and what descriptive support is available?*

Do not infer learning from a gap with no imaging, pair neurons across dates, or
treat 133 experiment–recording memberships as 133 independent acquisitions.
For biological endpoints, move from the exact selection here to the appropriate
paper-backed estimator:

- familiar-texture selectivity: [Figure 1]({NATURE}#Fig1) and
  [Methods]({NATURE}#Sec20);
- held-out sequence/coding geometry: [Figure 2]({NATURE}#Fig2) and
  [Methods]({NATURE}#Sec21);
- Train 2 representation: [Figure 3]({NATURE}#Fig3);
- reward-prediction response: [Figure 4]({NATURE}#Fig4) and
  [Methods]({NATURE}#Sec22).

Every generated table and graph can be traced back to the
[released imaging index]({INDEX_FILE}), the exact files in
[Figshare version 2]({FIGSHARE}), and the per-mouse entries in the
[live atlas]({ATLAS}#mice).
""",
        ),
    ]

    return notebook


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(f"wrote {NOTEBOOK} ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
