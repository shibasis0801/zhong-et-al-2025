#!/usr/bin/env python3
"""Generate the reward-versus-exposure d-prime dynamics notebook."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/05_reward_dprime_dynamics_colab.ipynb")


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def py(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


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
    cells = []

    cells.extend(
        [
            md(
                r"""
# Released-data plan for within-session visual discriminability

This notebook specifies a new, descriptive project analysis:

> **Do held-out within-session $d'$ trajectories differ between the
> passive-reward and no-reward Train 1 before-learning cohorts?**

This question is not a result reported by Zhong et al. The paper reports
before/after selectivity endpoints and a separate late-versus-early
cue-position statistic ([Results](https://www.nature.com/articles/s41586-025-09180-y#Sec2),
[reward-prediction Results](https://www.nature.com/articles/s41586-025-09180-y#Sec6),
[Methods](https://www.nature.com/articles/s41586-025-09180-y#Sec22)).

Every major analysis is an **executable node-and-wire workflow graph**:
labelled functions are nodes, named values are ports, blue wires show the data
flow, and hollow ports expose editable controls. **Run flow** executes the
workflow and returns its plots; folded code contains the exact calculation.

The notebook keeps the time-block trajectory and its position/behavior support
checks together. Results remain cohort-associated and descriptive: this design
does not isolate reward from all other task and acquisition differences.
"""
            ),
            md(
                r"""
## Evidence and analysis decisions

| Status | Fixed by | Decision and exact source |
|---|---|---|
| **Paper-defined** | Zhong et al. | Per-neuron sensory $d'$ uses running frames in the 0–4 m texture and the published formula; $|d'|\geq0.3$ defines selective neurons ([neural-selectivity Methods](https://www.nature.com/articles/s41586-025-09180-y#Sec20)). The paper's group tests are two-sided ([statistics](https://www.nature.com/articles/s41586-025-09180-y#Sec24)). |
| **Release-defined** | Deposited index | Train 1 before-learning contains 4 passive-reward task mice and 9 no-reward mice; recording IDs, reward labels, files, byte sizes and MD5 values come from [`Imaging_Exp_info.npy`](https://ndownloader.figshare.com/files/54183854) and [Figshare v2](https://doi.org/10.25378/janelia.28811129.v2). |
| **Project-defined** | This notebook | SVD representation, visual area, block width, early horizon, fold count, support rule and group summary are explicit editable specifications. They are not paper constants or preregistered confirmatory choices. The default group contrast is two-sided. |

The after-learning task set mixes passive and active-after-cue reward modes in
the release index, so it is not treated as a clean active-reward comparison.

### Descriptive mouse-level estimand

For each mouse, divide the session into **fixed, non-overlapping trial blocks**.
Within every block, fit a stimulus coding direction on some trials and score
only held-out trials. Estimate one early slope per mouse:

$$d'_{m,b} = \frac{2(\mu_{A,m,b}-\mu_{B,m,b})}
{\sigma_{A,m,b}+\sigma_{B,m,b}},\qquad
\Delta_{rate}=\overline{slope}_{rewarded}-\overline{slope}_{unrewarded}.$$

The reported difference is two-sided by default. Mice—not trials, bins,
components, or neurons—are the independent units in the cohort summary.
"""
            ),
            md(
                """
## Choose a path

**Quick path:** run setup, Graph 1, and Graph 2. They show the exact files and
how one released recording becomes analysis-ready without making a group claim.

**Programmer path:** inspect Graph 3's complete-cohort plan and Graph 4's
position/support diagnostics, then save the specification, manifest, excluded
recordings, environment and mouse-level results together.

## Four released-data workflows

| Graph | Playable structure | What it answers | Can it make a group claim? |
|---|---:|---|---|
| **1 · Cohort and fetch plan** | 11 nodes · 11 controls | Which mice and exact Drive files will be used, and how large are they? | No - provenance only |
| **2 · One real recording and d′ lab** | 14 nodes · 15 controls | How do trials, positions, folds, held-out scores, speed and support fit together? | No - mechanics/QC |
| **3 · Cohort trajectory plan/run** | exact real-data nodes | Plan the 4+9 mouse run; load only when explicitly enabled | Descriptive only after every eligible mouse is processed |
| **4 · Position and support** | exact real-data nodes | Where are the estimated changes and where is support missing? | Diagnostic whole-curve summary after a real two-group run |

The graphs stay separate on purpose. Exploring one recording must not silently
turn into a thirteen-mouse inference, and changing a plot-display control must not
quietly reload two gigabytes of Drive data.

Every scientific setting is recorded. Editable controls are for transparent
sensitivity analysis, not for selecting a preferred-looking result.
"""
            ),
            md(
                """
## Connect to the shared Drive workspace once

1. Add **Zhong et al. 2025 - Neuromatch Team Workspace** as a shortcut in
   **My Drive**. Do not copy the 421 GiB release.
2. Choose **Runtime -> Run all** in Colab.
3. The high-level `drive.py` interface resolves scientific recording IDs to
   exact files, copies only selected files, checks their size and MD5, and then
   returns ordinary NumPy arrays.

The setup works locally in metadata-only mode, so Graphs 1–2 and the bundled
real example remain useful without a mounted Drive.
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
            py(
                """
import warnings

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from zhong2025 import experiment_rows, load_atlas_demo, load_experiment_index
from zhong2025.learning import (
    AREA_IDS,
    blockwise_dprime,
    bootstrap_group_difference,
    cross_temporal_dprime,
    crossvalidated_scores,
    exact_group_permutation,
    fit_early_slope,
    fit_saturation_curve,
    position_dprime_surface,
    prepare_session_trials,
    trial_responses,
)

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.dpi": 110,
})

REWARDED = "rewarded"
UNREWARDED = "unrewarded"
GROUP_COLOURS = {REWARDED: "#2f855a", UNREWARDED: "#7b4f9d"}
AREA_LABELS = {"V1": "V1", "mHV": "medial HVA", "lHV": "lateral HVA", "aHV": "anterior HVA"}
ROLE_LABELS = {0: "circle1 / non-reward identity", 2: "leaf1 / reward identity"}
STIMULUS_PAIRS = {
    "leaf1_vs_circle1": (2, 0, "leaf1 vs circle1"),
    "leaf2_vs_circle2": (3, 1, "leaf2 vs circle2"),
}
recording_rows = experiment_rows(load_experiment_index())
"""
            ),
            md(
                """
## Analysis-methods source used here

Stringer and Pachitariu's *Analysis methods for large-scale neuronal
recordings* is in the shared workspace. Its published Figure 3 diagrams fitting
on training data and scoring on separate test data
([Drive PDF, page 5](https://drive.google.com/file/d/1DlmPeyaHn-thn9ILrt-rAXP96-y3IMU7/view#page=5));
Figure 4 separates encoding, decoding and dimensionality-reduction frameworks
([Drive PDF, page 6](https://drive.google.com/file/d/1DlmPeyaHn-thn9ILrt-rAXP96-y3IMU7/view#page=6)).

This notebook's trial blocks, contiguous folds, SVD representation, position
support rules and mouse summaries are **project-defined choices**, not values
reported by either paper. The published Zhong estimator and its frame selection
are documented separately in
[Neural selectivity](https://www.nature.com/articles/s41586-025-09180-y#Sec20).
Released file identities and checksums come from
[Figshare v2](https://doi.org/10.25378/janelia.28811129.v2).
"""
            ),
            md(
                """
## How to read a graph

- A **box** is one ordinary Python function.
- A **filled port** receives a named value from an earlier box.
- A **hollow port** is an editable setting.
- Branches are explicit, but nodes run sequentially for predictable Colab
  memory use.
- **Run to** stops after any chosen node; **Inspect port** opens any value from
  the latest run without recomputing the graph.
- The plots are returned through the last port. After a run, every intermediate
  result remains inspectable as `panel.last_run["port_name"]`.
"""
            ),
        ]
    )

    cells.extend(
        [
            md(
                """
## Graph 1 - Cohort, provenance, and Drive fetch plan

This graph is safe to run first: it reads only the pinned catalog and experiment
index. It shows the exact eligible recordings, reward metadata, deduplicated
files, and download budget **before** opening any neural array.

Use reduced neural (SVD) for interactive population exploration. The complete
before-learning plan is about **0.67 GiB** for 4 rewarded mice and **1.27 GiB**
for 9 unrewarded mice. Full-neural replication is roughly **62 GiB** across the
same cohorts and is an explicit heavy-data path.

The published release-wide SVD and area basis is **transductive**: downstream
trial scores are held out, but the upstream representation has seen the full
session. Treat it as an exploratory population read-out, not a prospective
online decoder.
"""
            ),
            py(
                """
#@title Graph 1 transformations { display-mode: "form" }
@graph.node(outputs="release_catalog")
def read_release_catalog():
    return {"dataset": data, "rows": recording_rows, "file_count": len(data.files)}


@graph.node(outputs="experiment_pair")
def define_experiment_pair(
    release_catalog,
    training_set="train1",
    stage="before_learning",
):
    if training_set not in {"train1", "train2"}:
        raise ValueError("training_set must be train1 or train2")
    if stage not in {"before_learning", "after_learning"}:
        raise ValueError("stage must be before_learning or after_learning")
    labels = {
        REWARDED: f"sup_{training_set}_{stage}",
        UNREWARDED: f"unsup_{training_set}_{stage}",
    }
    available = {row["experiment"] for row in release_catalog["rows"]}
    missing = sorted(set(labels.values()) - available)
    if missing:
        raise ValueError(f"release is missing experiment labels: {missing}")
    return {
        "training_set": training_set,
        "stage": stage,
        "labels": labels,
        "question": "passive reward present vs no reward",
    }


@graph.node(outputs="candidate_sessions")
def enumerate_candidate_sessions(release_catalog, experiment_pair):
    sessions = []
    for group, experiment in experiment_pair["labels"].items():
        for recording in release_catalog["dataset"].recordings(experiment=experiment):
            row = next(
                item for item in release_catalog["rows"]
                if item["experiment"] == experiment
                and item["recording_id"] == recording.recording_id
            )
            sessions.append({
                "group": group,
                "experiment": experiment,
                "recording": recording,
                "recording_id": recording.recording_id,
                "mouse": recording.mouse,
                "date": str(row["date"]),
                "block": str(row["block"]),
                "reward_type": str(row["reward_type"]),
                "stimulus_ids": tuple(row["stimulus_ids"]),
            })
    return sessions


@graph.node(outputs="cohort_policy")
def define_cohort_policy(
    experiment_pair,
    mice_per_group=0,
    preview_sampling="spread_dates",
):
    limit = int(mice_per_group)
    if limit < 0:
        raise ValueError("mice_per_group must be 0 (all) or positive")
    if preview_sampling not in {"first_catalog", "spread_dates", "latest"}:
        raise ValueError("unknown preview_sampling")
    return {
        "limit": limit,
        "sampling": preview_sampling,
        "complete": limit == 0,
        "stage": experiment_pair["stage"],
    }


@graph.node(outputs="selected_sessions")
def select_cohort_sessions(candidate_sessions, cohort_policy):
    selected = []
    for group in (REWARDED, UNREWARDED):
        rows = sorted(
            [row for row in candidate_sessions if row["group"] == group],
            key=lambda row: (row["date"], row["mouse"], row["recording_id"]),
        )
        limit = cohort_policy["limit"]
        if limit and len(rows) > limit:
            if cohort_policy["sampling"] == "latest":
                rows = rows[-limit:]
            elif cohort_policy["sampling"] == "spread_dates":
                indices = np.linspace(0, len(rows) - 1, limit).round().astype(int)
                rows = [rows[index] for index in indices]
            else:
                rows = rows[:limit]
        selected.extend(rows)
    return selected


@graph.node(outputs="layer_plan")
def define_data_layers(
    experiment_pair,
    neural_layer="reduced_neural",
    include_behavior=True,
    include_retinotopy=True,
):
    if neural_layer not in {"reduced_neural", "full_neural"}:
        raise ValueError("neural_layer must be reduced_neural or full_neural")
    layers = []
    if bool(include_behavior):
        layers.append("behavior")
    layers.append(neural_layer)
    if bool(include_retinotopy):
        layers.append("retinotopy")
    return {
        "layers": tuple(layers),
        "neural_layer": neural_layer,
        "population_exploration": neural_layer == "reduced_neural",
    }


@graph.node(outputs="drive_manifest")
def resolve_drive_manifest(release_catalog, selected_sessions, layer_plan):
    rows = []
    for session in selected_sessions:
        for layer in layer_plan["layers"]:
            item = session["recording"].file(
                layer,
                experiment=session["experiment"] if layer == "behavior" else None,
            )
            rows.append({
                "group": session["group"],
                "mouse": session["mouse"],
                "date": session["date"],
                "recording_id": session["recording_id"],
                "layer": layer,
                "name": item.name,
                "size_bytes": item.size_bytes,
                "md5": item.md5,
            })
    return rows


@graph.node(outputs="unique_manifest")
def deduplicate_drive_manifest(drive_manifest, deduplicate_files=True):
    if not bool(deduplicate_files):
        rows = list(drive_manifest)
    else:
        unique = {}
        for row in drive_manifest:
            unique.setdefault(row["name"], row)
        rows = list(unique.values())
    return {
        "rows": rows,
        "raw_rows": drive_manifest,
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "deduplicated": bool(deduplicate_files),
    }


@graph.node(outputs="cohort_audit")
def audit_cohort_semantics(candidate_sessions, selected_sessions, experiment_pair):
    by_group = {}
    for group in (REWARDED, UNREWARDED):
        all_rows = [row for row in candidate_sessions if row["group"] == group]
        rows = [row for row in selected_sessions if row["group"] == group]
        by_group[group] = {
            "eligible_mice": len({row["mouse"] for row in all_rows}),
            "selected_mice": len({row["mouse"] for row in rows}),
            "reward_types": sorted({row["reward_type"] for row in rows}),
            "dates": sorted({row["date"] for row in rows}),
            "recordings": [row["recording_id"] for row in rows],
        }
    return {
        "by_group": by_group,
        "mixed_reward": len(by_group[REWARDED]["reward_types"]) > 1,
        "exchangeability_warning": "cohorts differ in acquisition year/batch",
        "question": experiment_pair["question"],
    }


@graph.node(outputs="budget_audit")
def validate_download_budget(
    unique_manifest,
    max_download_gib=3.0,
    max_single_file_gib=1.0,
):
    total_gib = unique_manifest["total_bytes"] / 2**30
    largest_gib = max(
        (row["size_bytes"] / 2**30 for row in unique_manifest["rows"]),
        default=0.0,
    )
    return {
        "total_gib": total_gib,
        "largest_gib": largest_gib,
        "within_total_limit": total_gib <= float(max_download_gib),
        "within_file_limit": largest_gib <= float(max_single_file_gib),
        "max_download_gib": float(max_download_gib),
        "max_single_file_gib": float(max_single_file_gib),
    }


@graph.node(outputs="plots")
def make_fetch_plots(
    experiment_pair,
    selected_sessions,
    layer_plan,
    unique_manifest,
    cohort_audit,
    budget_audit,
    preview_file_rows=10,
):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 7.5), constrained_layout=True)
    groups = [REWARDED, UNREWARDED]
    labels = ["Rewarded", "Unrewarded"]
    counts = [cohort_audit["by_group"][group]["selected_mice"] for group in groups]
    bars = axes[0, 0].bar(labels, counts, color=[GROUP_COLOURS[group] for group in groups])
    axes[0, 0].bar_label(bars)
    axes[0, 0].set(title="Selected independent mice", ylabel="mice")

    for offset, group in enumerate(groups):
        rows = sorted(
            [row for row in selected_sessions if row["group"] == group],
            key=lambda row: (row["date"], row["mouse"]),
        )
        dates = []
        for row in rows:
            parts = row["date"].replace("-", "_").split("_")
            year, month, day = map(int, parts[:3])
            dates.append(year + (month - 1) / 12 + (day - 1) / 365)
        dates = np.asarray(dates, dtype=float)
        y = offset + np.linspace(-0.14, 0.14, len(rows))
        axes[0, 1].scatter(
            dates, y,
            color=GROUP_COLOURS[group],
            s=55,
            label=group,
        )
        for date, y_value, row in zip(dates, y, rows):
            axes[0, 1].annotate(
                row["mouse"], (date, y_value), xytext=(2, 5),
                textcoords="offset points", ha="left", fontsize=5.8, rotation=25,
            )
    axes[0, 1].set_yticks([0, 1], labels)
    axes[0, 1].set_ylim(-0.35, 1.35)
    axes[0, 1].set(title="Acquisition date/batch structure", xlabel="calendar year")

    layers = list(layer_plan["layers"])
    bottom = np.zeros(2)
    layer_colours = ["#4c78a8", "#f28e2b", "#9aa2ad"]
    for layer, colour in zip(layers, layer_colours):
        values = []
        for group in groups:
            names = {
                row["name"] for row in unique_manifest["raw_rows"]
                if row["group"] == group and row["layer"] == layer
            }
            values.append(sum(
                row["size_bytes"] for row in unique_manifest["rows"]
                if row["name"] in names
            ) / 2**30)
        axes[0, 2].bar(labels, values, bottom=bottom, label=layer, color=colour)
        bottom += values
    axes[0, 2].set(title="Drive budget by layer", ylabel="GiB")
    axes[0, 2].legend(fontsize=7)

    session_totals = []
    session_labels = []
    session_colours = []
    for session in selected_sessions:
        names = {
            row["name"] for row in unique_manifest["raw_rows"]
            if row["recording_id"] == session["recording_id"]
        }
        session_totals.append(sum(
            row["size_bytes"] for row in unique_manifest["rows"] if row["name"] in names
        ) / 2**30)
        session_labels.append(session["mouse"])
        session_colours.append(GROUP_COLOURS[session["group"]])
    axes[1, 0].barh(session_labels, session_totals, color=session_colours)
    axes[1, 0].set(title="Per-recording staged size", xlabel="GiB")

    axes[1, 1].axis("off")
    preview_n = max(1, int(preview_file_rows))
    preview = unique_manifest["rows"][:preview_n]
    lines = [f"Exact catalog files (first {len(preview)}):"]
    lines += [
        f"{row['layer']:<15} {row['size_bytes']/2**20:7.1f} MiB  {row['name']}"
        for row in preview
    ]
    remaining = len(unique_manifest["rows"]) - len(preview)
    if remaining > 0:
        lines.append(f"... {remaining} more file(s)")
    axes[1, 1].text(0, 1, "\\n".join(lines), va="top", family="monospace", fontsize=7.3)

    axes[1, 2].axis("off")
    status = "WITHIN DECLARED LIMITS" if (
        budget_audit["within_total_limit"] and budget_audit["within_file_limit"]
    ) else "EXCEEDS A DECLARED LIMIT - PLAN ONLY"
    message = [
        status,
        "",
        f"Protocol: {experiment_pair['training_set']} · {experiment_pair['stage'].replace('_', ' ')}",
        f"Question: {cohort_audit['question']}",
        f"Reward metadata: {cohort_audit['by_group'][REWARDED]['reward_types']}",
        f"No-reward metadata: {cohort_audit['by_group'][UNREWARDED]['reward_types']}",
        f"Unique files: {len(unique_manifest['rows'])}",
        f"Total: {budget_audit['total_gib']:.2f} / {budget_audit['max_download_gib']:.2f} GiB",
        f"Largest: {budget_audit['largest_gib']:.2f} / {budget_audit['max_single_file_gib']:.2f} GiB",
        "",
        "SVD = transductive population exploration.",
        "Full neural = heavy per-neuron replication path.",
        "Acquisition batches limit causal/exchangeable claims.",
    ]
    axes[1, 2].text(0, 1, "\\n".join(message), va="top", fontsize=8.5)
    fig.suptitle("Graph 1 · cohort truth, acquisition structure, and Drive plan")
    plt.close(fig)
    return fig


"""
            ),
            py(
                """
#@title Choose the cohort and data representation { display-mode: "form" }
fetch_graph = graph.Graph(
    "Cohort, acquisition structure, and Drive fetch plan",
    read_release_catalog,
    define_experiment_pair,
    enumerate_candidate_sessions,
    define_cohort_policy,
    select_cohort_sessions,
    define_data_layers,
    resolve_drive_manifest,
    deduplicate_drive_manifest,
    audit_cohort_semantics,
    validate_download_budget,
    make_fetch_plots,
)
fetch_panel = fetch_graph.widget(
    controls={
        "training_set": widgets.Dropdown(
            description="Training set",
            options=[("Train 1", "train1"), ("Train 2", "train2")],
            value="train1",
        ),
        "stage": widgets.Dropdown(
            description="Training stage",
            options=[("Before learning", "before_learning"), ("After learning", "after_learning")],
            value="before_learning",
        ),
        "mice_per_group": widgets.Dropdown(
            description="Cohort size",
            options=[("All eligible mice", 0), ("Two per group preview", 2), ("One per group smoke test", 1)],
            value=0,
        ),
        "preview_sampling": widgets.Dropdown(
            description="Preview sample",
            options=[("Spread across dates", "spread_dates"), ("First catalog rows", "first_catalog"), ("Latest dates", "latest")],
            value="spread_dates",
        ),
        "neural_layer": widgets.Dropdown(
            description="Neural layer",
            options=[("Reduced neural / SVD (interactive)", "reduced_neural"), ("Full neural (heavy replication)", "full_neural")],
            value="reduced_neural",
        ),
        "include_behavior": widgets.Checkbox(description="Include behavior", value=True),
        "include_retinotopy": widgets.Checkbox(description="Include retinotopy", value=True),
        "deduplicate_files": widgets.Checkbox(description="Deduplicate files", value=True),
        "max_download_gib": widgets.FloatSlider(description="Total GiB limit", value=3.0, min=0.5, max=70.0, step=0.5),
        "max_single_file_gib": widgets.FloatSlider(description="File GiB limit", value=1.0, min=0.25, max=10.0, step=0.25),
        "preview_file_rows": widgets.IntSlider(description="Manifest rows", value=10, min=4, max=20, step=2),
    },
    show="plots",
)
fetch_panel
"""
            ),
            md(
                """
### The exact Drive fetch - no folder guessing

The graph above resolves file names without downloading. Selecting the
real-data path uses only the following calls:
"""
            ),
            py(
                """
#@title Read this fetch recipe; set LOAD_ONE_SESSION=True when ready { display-mode: "form" }
LOAD_ONE_SESSION = False  # change to True in Colab
experiment = "sup_train1_before_learning"
recording = data.recordings(experiment=experiment)[0]

print("Scientific selection:", experiment, "->", recording.recording_id)
for layer in ("behavior", "reduced_neural", "retinotopy"):
    item = recording.file(layer, experiment=experiment if layer == "behavior" else None)
    print(f"{layer:15s} {item.size_mib:7.1f} MiB  {item.name}  MD5 {item.md5}")

if LOAD_ONE_SESSION:
    behavior = recording.load("behavior", experiment=experiment)
    svd = recording.load("reduced_neural")
    retinotopy = recording.load("retinotopy")
    print("Behavior fields:", sorted(behavior)[:20], "...")
    print("SVD U/V:", svd["U"].shape, svd["V"].shape)
    print("Retinotopy iarea:", retinotopy["iarea"].shape)
else:
    print("Nothing downloaded. Set LOAD_ONE_SESSION=True when the Drive is mounted.")
"""
            ),
            md(
                """
### Arrays used in the analysis

| Layer | Fields | Meaning |
|---|---|---|
| Behaviour | `ft_trInd`, `ft_Pos`, `ft_isMoving`, `ft_RunSpeed` | frame -> trial, position (decimetres), movement, speed |
| Behaviour | `WallName`, `UniqWalls`, `stim_id` | physical wall -> canonical stimulus role |
| Behaviour | `Reward_Mode`, `isRew`, cue/reward/lick positions | reward truth and behavioral diagnostics |
| SVD | `U` (components x neurons), `V` (components x frames) | compact session population representation |
| Retinotopy | `iarea` | V1=8; medial=0/1/2/9; lateral=5/6; anterior=3/4 |

Positions are converted from 0-60 decimetres to 0-6 metres. Every frame is
kept inside its own trial and position bin; there is no interpolation across a
trial boundary. The main sensory region is 0-4 m, while 4-6 m grey is a useful
negative/control region.
"""
            ),
        ]
    )

    cells.extend(
        [
            md(
                """
## Graph 2 - How one real recording becomes analysis-ready

This graph uses the bundled, checksum-derived TX119 compact recording. It is
**real released data**, but it is an `unsup_test1` session - not a Train 1
reward comparison. Its job is to make shapes, labels, trial order, position,
speed, support and visual-area features tangible before the expensive group
run.

The plot is deliberately labelled **mechanics only**. It cannot answer whether
reward accelerates learning.
"""
            ),
            py(
                """
#@title Graph 2 transformations { display-mode: "form" }
@graph.node(outputs="compact_source")
def load_compact_source():
    return load_atlas_demo()


@graph.node(outputs="trial_slice")
def select_recording_window(compact_source, trial_start=0, trial_count=120):
    start = int(trial_start)
    count = int(trial_count)
    if start < 0 or count < 40:
        raise ValueError("trial_start must be non-negative and trial_count at least 40")
    stop = min(start + count, len(compact_source["trial_id"]))
    if stop - start < 40:
        raise ValueError("the selected trial range is too short")
    return {"start": start, "stop": stop, "count": stop - start}


@graph.node(outputs="area_view")
def select_visual_area(compact_source, cortical_area="V1"):
    areas = [str(name) for name in compact_source["area_name"]]
    if cortical_area not in areas:
        raise ValueError(f"cortical_area must be one of {areas}")
    index = areas.index(cortical_area)
    return {
        "area": cortical_area,
        "features": np.asarray(compact_source["area_features"][index], dtype=float),
    }


@graph.node(outputs="contrast_spec")
def select_stimulus_contrast(stimulus_pair="leaf1_vs_circle1"):
    if stimulus_pair not in STIMULUS_PAIRS:
        raise ValueError(f"stimulus_pair must be one of {list(STIMULUS_PAIRS)}")
    role_a, role_b, label = STIMULUS_PAIRS[stimulus_pair]
    return {"key": stimulus_pair, "role_a": role_a, "role_b": role_b, "label": label}


@graph.node(outputs="corridor_spec")
def define_corridor_view(
    compact_source,
    position_start_m=0.0,
    position_end_m=6.0,
):
    start, stop = float(position_start_m), float(position_end_m)
    if not 0 <= start < stop <= 6:
        raise ValueError("corridor window must satisfy 0 <= start < end <= 6 m")
    position = np.asarray(compact_source["position_centers_m"], dtype=float)
    mask = (position >= start) & (position < stop)
    if not np.any(mask):
        raise ValueError("corridor window selects no position bins")
    return {"start": start, "stop": stop, "mask": mask, "position": position}


@graph.node(outputs="trial_tensor")
def slice_trial_tensor(compact_source, trial_slice, area_view):
    section = slice(trial_slice["start"], trial_slice["stop"])
    return {
        "features": area_view["features"][section],
        "labels": np.asarray(compact_source["stimulus_id"])[section],
        "trial_id": np.asarray(compact_source["trial_id"])[section],
        "speed": np.asarray(compact_source["mean_run_speed"])[section],
        "counts": np.asarray(compact_source["frame_counts"])[section],
        "position": np.asarray(compact_source["position_centers_m"]),
        "wall": np.asarray(compact_source["wall_name"])[section],
        "metadata": compact_source["metadata"],
        "area": area_view["area"],
    }


@graph.node(outputs="coverage_qc")
def measure_position_coverage(
    trial_tensor,
    corridor_spec,
    minimum_frames_per_bin=1,
    coverage_rule="complete",
):
    minimum = int(minimum_frames_per_bin)
    if minimum < 1:
        raise ValueError("minimum_frames_per_bin must be positive")
    if coverage_rule not in {"complete", "available"}:
        raise ValueError("coverage_rule must be complete or available")
    enough = trial_tensor["counts"][:, corridor_spec["mask"]] >= minimum
    valid_trial = np.all(enough, axis=1) if coverage_rule == "complete" else np.any(enough, axis=1)
    return {
        "valid_trial": valid_trial,
        "bin_support": np.mean(enough, axis=0),
        "rule": coverage_rule,
        "minimum": minimum,
        "valid_fraction": float(np.mean(valid_trial)),
    }


@graph.node(outputs="analysis_slice")
def filter_supported_trials(
    trial_tensor,
    contrast_spec,
    coverage_qc,
    role_filter_mode="pair_only",
):
    if role_filter_mode not in {"pair_only", "all_roles"}:
        raise ValueError("role_filter_mode must be pair_only or all_roles")
    pair = np.isin(trial_tensor["labels"], [contrast_spec["role_a"], contrast_spec["role_b"]])
    display = coverage_qc["valid_trial"] & (pair if role_filter_mode == "pair_only" else True)
    return {
        **trial_tensor,
        "pair_trial": pair,
        "display_trial": np.asarray(display, dtype=bool),
        "valid_trial": coverage_qc["valid_trial"],
        "role_filter_mode": role_filter_mode,
    }


@graph.node(outputs="activity_readout")
def construct_activity_readout(
    analysis_slice,
    activity_metric="feature_norm",
    feature_index=0,
    display_normalization="raw",
):
    features = np.asarray(analysis_slice["features"], dtype=float)
    if activity_metric == "feature_norm":
        matrix = np.linalg.norm(features, axis=2)
        label = "population-feature norm"
    elif activity_metric == "feature_mean":
        matrix = np.nanmean(features, axis=2)
        label = "mean area feature"
    elif activity_metric == "single_feature":
        index = int(feature_index)
        if not 0 <= index < features.shape[2]:
            raise ValueError(f"feature_index must be between 0 and {features.shape[2]-1}")
        matrix = features[:, :, index]
        label = f"area feature {index}"
    else:
        raise ValueError("unknown activity_metric")
    if display_normalization == "zscore_by_position":
        mean = np.nanmean(matrix, axis=0)
        scale = np.nanstd(matrix, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
        matrix = (matrix - mean) / scale
        label += " (display z-score)"
    elif display_normalization != "raw":
        raise ValueError("display_normalization must be raw or zscore_by_position")
    return {"matrix": matrix, "label": label, "normalization": display_normalization}


@graph.node(outputs="behavior_qc")
def summarize_trial_behavior(analysis_slice, corridor_spec, coverage_qc):
    return {
        "mean_speed": np.nanmean(analysis_slice["speed"], axis=0),
        "mean_count": np.mean(analysis_slice["counts"], axis=0),
        "missing_fraction": np.mean(analysis_slice["counts"] == 0, axis=0),
        "selected_support": coverage_qc["bin_support"],
        "role_counts": {
            int(role): int(np.count_nonzero(analysis_slice["labels"] == role))
            for role in np.unique(analysis_slice["labels"])
        },
        "corridor_bins": int(np.count_nonzero(corridor_spec["mask"])),
    }


@graph.node(outputs="neural_qc")
def summarize_neural_activity(analysis_slice, activity_readout, coverage_qc):
    matrix = activity_readout["matrix"]
    included = analysis_slice["display_trial"]
    return {
        "matrix": matrix[included],
        "trial_id": analysis_slice["trial_id"][included],
        "finite_fraction": float(np.isfinite(analysis_slice["features"]).mean()),
        "shape": analysis_slice["features"].shape,
        "included_trials": int(np.count_nonzero(included)),
        "excluded_trials": int(len(included) - np.count_nonzero(included)),
        "coverage_rule": coverage_qc["rule"],
    }


@graph.node(outputs="role_profiles")
def build_role_profiles(analysis_slice, activity_readout, contrast_spec):
    profiles = {}
    matrix = activity_readout["matrix"]
    for role in (contrast_spec["role_a"], contrast_spec["role_b"]):
        selected = (analysis_slice["labels"] == role) & analysis_slice["valid_trial"]
        values = matrix[selected]
        count = np.sum(np.isfinite(values), axis=0)
        profiles[role] = np.divide(
            np.nansum(values, axis=0), count,
            out=np.full(matrix.shape[1], np.nan), where=count > 0,
        )
    descriptive_dprime = np.full(matrix.shape[1], np.nan)
    for bin_index in range(matrix.shape[1]):
        a = matrix[(analysis_slice["labels"] == contrast_spec["role_a"]) & analysis_slice["valid_trial"], bin_index]
        b = matrix[(analysis_slice["labels"] == contrast_spec["role_b"]) & analysis_slice["valid_trial"], bin_index]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if min(len(a), len(b)) >= 2:
            spread = np.std(a, ddof=1) + np.std(b, ddof=1)
            if spread > 0:
                descriptive_dprime[bin_index] = 2 * (np.mean(a) - np.mean(b)) / spread
    return {"profiles": profiles, "descriptive_dprime": descriptive_dprime}


@graph.node(outputs="heldout_lab")
def measure_heldout_discriminability(
    analysis_slice,
    corridor_spec,
    contrast_spec,
    coverage_qc,
    crossvalidation_folds=4,
    minimum_trials_per_role=4,
):
    responses = trial_responses(
        analysis_slice["features"],
        corridor_spec["mask"],
        require_complete_position_coverage=coverage_qc["rule"] == "complete",
    )
    responses[~analysis_slice["valid_trial"]] = np.nan
    local = crossvalidated_scores(
        responses,
        analysis_slice["labels"],
        role_a=contrast_spec["role_a"],
        role_b=contrast_spec["role_b"],
        n_folds=int(crossvalidation_folds),
        min_per_role=int(minimum_trials_per_role),
    )
    return {**local, "responses": responses, "contrast": contrast_spec["label"]}


@graph.node(outputs="plots")
def make_recording_plots(
    analysis_slice,
    corridor_spec,
    contrast_spec,
    activity_readout,
    coverage_qc,
    behavior_qc,
    neural_qc,
    role_profiles,
    heldout_lab,
    heatmap_percentile=99,
):
    percentile = float(heatmap_percentile)
    if not 80 <= percentile <= 100:
        raise ValueError("heatmap_percentile must be between 80 and 100")
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 7.6), constrained_layout=True)
    trial_number = analysis_slice["trial_id"]
    position = analysis_slice["position"]
    role_a, role_b = contrast_spec["role_a"], contrast_spec["role_b"]

    colours = [
        "#2f855a" if role == role_a else "#7b4f9d" if role == role_b else "#9aa2ad"
        for role in analysis_slice["labels"]
    ]
    axes[0, 0].scatter(trial_number, analysis_slice["labels"], c=colours, marker="|", s=110)
    excluded = ~analysis_slice["valid_trial"]
    axes[0, 0].scatter(trial_number[excluded], analysis_slice["labels"][excluded], facecolors="none", edgecolors="#e15759", s=25, label="coverage excluded")
    axes[0, 0].set(title="Physical trial order and role", xlabel="published trial index", ylabel="canonical role")
    if np.any(excluded):
        axes[0, 0].legend(fontsize=6.5)

    matrix = neural_qc["matrix"]
    finite_matrix = matrix[np.isfinite(matrix)]
    if matrix.shape[0] and len(finite_matrix):
        limit = max(float(np.percentile(np.abs(finite_matrix), percentile)), 1e-6)
        image = axes[0, 1].imshow(
            matrix, aspect="auto", origin="upper", cmap="viridis",
            vmin=float(np.min(finite_matrix)), vmax=limit,
            extent=[position[0], position[-1], neural_qc["trial_id"][-1], neural_qc["trial_id"][0]],
        )
        axes[0, 1].axvspan(corridor_spec["start"], corridor_spec["stop"], color="white", alpha=0.08)
        axes[0, 1].axvline(4, color="white", linestyle="--", linewidth=1)
        fig.colorbar(image, ax=axes[0, 1], label=activity_readout["label"])
    else:
        axes[0, 1].text(
            0.5, 0.5,
            "No trials satisfy the selected\\nposition-coverage rule.",
            ha="center", va="center", transform=axes[0, 1].transAxes,
        )
    axes[0, 1].set(title=f"Trial x position · {AREA_LABELS[analysis_slice['area']]}", xlabel="position (m)", ylabel="included trial")

    drew_profile = False
    for role, colour in ((role_a, "#2f855a"), (role_b, "#7b4f9d")):
        profile = role_profiles["profiles"][role]
        if np.any(np.isfinite(profile)):
            axes[0, 2].plot(position, profile, color=colour, label=f"role {role}")
            drew_profile = True
    axes[0, 2].axvspan(4, 6, color="#999999", alpha=0.12, label="grey")
    axes[0, 2].set(title=f"Mean activity · {contrast_spec['label']}", xlabel="position (m)", ylabel=activity_readout["label"])
    if drew_profile:
        axes[0, 2].legend(fontsize=6.5)
    else:
        axes[0, 2].text(0.5, 0.5, "No supported role profiles", ha="center", va="center", transform=axes[0, 2].transAxes)

    if np.any(np.isfinite(role_profiles["descriptive_dprime"])):
        axes[0, 3].plot(position, role_profiles["descriptive_dprime"], color="#4c78a8", marker="o", markersize=3)
    else:
        axes[0, 3].text(0.5, 0.5, "Descriptive d′ unavailable", ha="center", va="center", transform=axes[0, 3].transAxes)
    axes[0, 3].axhline(0, color="#999", linewidth=1)
    axes[0, 3].axvline(4, color="#777", linestyle="--")
    axes[0, 3].set(title="Descriptive d′ along position", xlabel="position (m)", ylabel="predefined-readout d′")

    axes[1, 0].plot(position, behavior_qc["mean_speed"], color="#4c78a8")
    axes[1, 0].axvline(4, color="#777", linestyle="--")
    axes[1, 0].set(title="Running-speed control", xlabel="position (m)", ylabel="release speed units")

    axes[1, 1].plot(position, behavior_qc["mean_count"], color="#f28e2b", label="frames/bin")
    axes[1, 1].plot(position, behavior_qc["missing_fraction"] * np.nanmax(behavior_qc["mean_count"]), color="#e15759", linestyle="--", label="missing fraction (scaled)")
    axes[1, 1].axvspan(corridor_spec["start"], corridor_spec["stop"], color="#4c78a8", alpha=0.06)
    axes[1, 1].set(title="Sampling support and chosen window", xlabel="position (m)", ylabel="frames")
    axes[1, 1].legend(fontsize=6.5)

    if heldout_lab["fold_metrics"]:
        fold_ids = [int(row["fold"]) for row in heldout_lab["fold_metrics"]]
        fold_dprime = [row["dprime"] for row in heldout_lab["fold_metrics"]]
        bars = axes[1, 2].bar(fold_ids, fold_dprime, color="#59a14f")
        axes[1, 2].bar_label(bars, fmt="%.2f", fontsize=7)
        axes[1, 2].axhline(0, color="#999", linewidth=1)
    else:
        axes[1, 2].text(0.5, 0.5, heldout_lab["invalid_reason"], ha="center", va="center", transform=axes[1, 2].transAxes)
    axes[1, 2].set(title="Held-out d′ inside each physical-time fold", xlabel="fold", ylabel="within-fold d′")

    axes[1, 3].axis("off")
    lines = [
        "REAL COMPACT RECORDING - MECHANICS ONLY",
        f"Session: {analysis_slice['metadata']['session']}",
        f"Condition: {analysis_slice['metadata']['condition']}",
        f"Tensor: {neural_qc['shape'][0]} x {neural_qc['shape'][1]} x {neural_qc['shape'][2]}",
        f"Contrast: {heldout_lab['contrast']}",
        f"Coverage: {coverage_qc['rule']} · {100*coverage_qc['valid_fraction']:.1f}% valid trials",
        f"Displayed: {neural_qc['included_trials']} · excluded: {neural_qc['excluded_trials']}",
        f"CV folds valid: {heldout_lab['valid_folds']}/{heldout_lab['required_folds']}",
        "",
        "Trials are observations; frames/bins are repeated measurements.",
        "This one unsupervised session teaches the estimator—it is not a group result.",
    ]
    axes[1, 3].text(0, 1, "\\n".join(lines), va="top", fontsize=8.3)
    fig.suptitle("Graph 2 · real recording anatomy and held-out d′ laboratory")
    plt.close(fig)
    return fig
"""
            ),
            py(
                """
#@title Choose a visual area and trial range { display-mode: "form" }
recording_graph = graph.Graph(
    "One real recording: anatomy, coverage, folds, and d-prime",
    load_compact_source,
    select_recording_window,
    select_visual_area,
    select_stimulus_contrast,
    define_corridor_view,
    slice_trial_tensor,
    measure_position_coverage,
    filter_supported_trials,
    construct_activity_readout,
    summarize_trial_behavior,
    summarize_neural_activity,
    build_role_profiles,
    measure_heldout_discriminability,
    make_recording_plots,
)
recording_panel = recording_graph.widget(
    controls={
        "trial_start": widgets.IntSlider(description="First trial", value=0, min=0, max=320, step=20),
        "trial_count": widgets.IntSlider(description="Trial count", value=120, min=40, max=240, step=20),
        "cortical_area": widgets.Dropdown(
            description="Cortical area",
            options=[("V1", "V1"), ("Medial HVA", "mHV"), ("Lateral HVA", "lHV"), ("Anterior HVA", "aHV")],
            value="V1",
        ),
        "stimulus_pair": widgets.Dropdown(
            description="Stimulus pair",
            options=[("Leaf 1 vs circle 1", "leaf1_vs_circle1"), ("Leaf 2 vs circle 2", "leaf2_vs_circle2")],
            value="leaf1_vs_circle1",
        ),
        "position_start_m": widgets.FloatSlider(description="Position start m", value=0.0, min=0.0, max=5.5, step=0.5),
        "position_end_m": widgets.FloatSlider(description="Position end m", value=6.0, min=0.5, max=6.0, step=0.5),
        "minimum_frames_per_bin": widgets.IntSlider(description="Min frames/bin", value=1, min=1, max=4, step=1),
        "coverage_rule": widgets.Dropdown(
            description="Coverage rule",
            options=[("Complete window", "complete"), ("Any available bin", "available")],
            value="complete",
        ),
        "role_filter_mode": widgets.Dropdown(
            description="Trial display",
            options=[("Selected pair only", "pair_only"), ("All stimulus roles", "all_roles")],
            value="pair_only",
        ),
        "activity_metric": widgets.Dropdown(
            description="Activity readout",
            options=[("Feature norm", "feature_norm"), ("Mean feature", "feature_mean"), ("One feature", "single_feature")],
            value="feature_norm",
        ),
        "feature_index": widgets.IntSlider(description="Feature index", value=0, min=0, max=11, step=1),
        "display_normalization": widgets.Dropdown(
            description="Plot normalization",
            options=[("Raw readout", "raw"), ("Z-score by position", "zscore_by_position")],
            value="raw",
        ),
        "crossvalidation_folds": widgets.Dropdown(description="CV folds", options=[2, 4], value=4),
        "minimum_trials_per_role": widgets.IntSlider(description="Min trials/role", value=4, min=2, max=8, step=1),
        "heatmap_percentile": widgets.IntSlider(description="Heatmap percentile", value=99, min=90, max=100, step=1),
    },
    show="plots",
)
recording_panel
"""
            ),
        ]
    )

    cells.extend(
        [
            md(
                r"""
## Graph 4 - Released-data cohort trajectories

This graph specifies a new descriptive analysis of the released recordings.
Start in **Plan only** to see the exact sessions and budget. Then choose
**Load and analyse** in Colab.

The default two-mice-per-group setting is a pipeline preview. It must not be
reported as the cohort result. For a complete descriptive summary choose **All
eligible mice**: 4 passive-reward and 9 no-reward before-learning sessions.

### What happens inside the expanded graph

1. Define the cohort, representation, stimulus contrast, corridor window,
   temporal estimator and execution policy in separate nodes.
2. Compare every selected field with the displayed reference specification;
   differences are retained in the result package rather than hidden.
3. Resolve exact files and show acquisition dates and storage before opening an array.
4. Load verified behavior, SVD and retinotopy layers sequentially; align neural
   and behavior frames explicitly; apply the named movement and coverage rules.
5. In each fixed trial block, fit a coding direction on contiguous
   training folds and score held-out trials only.
6. Decompose $d'$ into stimulus-mean separation and held-out spread, fit the
   early rate, fit the secondary saturation model, and branch to cross-temporal,
   position, speed, support, and event diagnostics.
7. Reduce to one rate per mouse. Preview inference is suppressed by default;
   a complete-cohort run or an explicit exploratory override is required.
8. Bootstrap mice, enumerate exact labels, and leave one mouse out in separate
   inspectable nodes.

The default reference specification requires complete coverage of every selected
corridor bin for each trial. The release-wide SVD/area basis is
**transductive**: trial scoring is held out, but the upstream basis is not. The
result is therefore an exploratory population read-out, not a prospective
online decoder.

The **Full-neuron replication** control exposes the correct heavy file plan and
keeps the estimand distinction visible. Its interactive estimator is not
implemented here, so the graph refuses to present an SVD result under a
full-neuron label.

The cross-temporal panel trains the coding axis in one time block and tests it
in another. A stable axis with larger separation looks different from axis
rotation/reorganization; this is exploratory, but closer to the proposal's
mechanistic wording than a single d′ curve.
"""
            ),
            py(
                """
#@title Graph 4 transformations { display-mode: "form" }
@graph.node(outputs="analysis_dataset")
def connect_real_analysis_release():
    return data


@graph.node(outputs="real_cohort_policy")
def define_real_cohort_policy(
    analysis_dataset,
    real_training_set="train1",
    training_stage="before_learning",
    cohort_size=2,
    cohort_sampling="spread_dates",
):
    if real_training_set not in {"train1", "train2"}:
        raise ValueError("real_training_set must be train1 or train2")
    if training_stage not in {"before_learning", "after_learning"}:
        raise ValueError("training_stage must be before_learning or after_learning")
    if int(cohort_size) not in {0, 1, 2, 3}:
        raise ValueError("cohort_size must be 0, 1, 2, or 3")
    if cohort_sampling not in {"spread_dates", "first_catalog", "latest"}:
        raise ValueError("unknown cohort_sampling")
    return {
        "training_set": real_training_set,
        "stage": training_stage,
        "limit": int(cohort_size),
        "sampling": cohort_sampling,
        "dataset_connected": bool(analysis_dataset.connected),
    }


@graph.node(outputs="representation_spec")
def define_neural_representation(
    representation_mode="svd_population",
    cortical_area="V1",
    feature_count=12,
    movement_rule="moving_only",
):
    if representation_mode not in {"svd_population", "full_neuron_replication"}:
        raise ValueError("unknown representation_mode")
    if cortical_area not in AREA_IDS:
        raise ValueError(f"cortical_area must be one of {list(AREA_IDS)}")
    if int(feature_count) not in {8, 12, 24}:
        raise ValueError("feature_count must be 8, 12, or 24")
    if movement_rule not in {"moving_only", "all_valid_frames"}:
        raise ValueError("unknown movement_rule")
    return {
        "mode": representation_mode,
        "layer": "reduced_neural" if representation_mode == "svd_population" else "full_neural",
        "area": cortical_area,
        "n_features": int(feature_count),
        "movement_rule": movement_rule,
        "estimand": (
            "cross-fitted population d-prime in the published SVD/area basis"
            if representation_mode == "svd_population"
            else "paper-style full-neuron selectivity replication (heavy plan only here)"
        ),
        "transductive": representation_mode == "svd_population",
    }


@graph.node(outputs="real_contrast_spec")
def define_real_stimulus_and_corridor_contrast(
    stimulus_pair="leaf1_vs_circle1",
    corridor_region="texture_0_4",
    custom_position_start_m=0.0,
    custom_position_end_m=4.0,
    position_bin_count=18,
    real_coverage_rule="complete",
):
    if stimulus_pair not in STIMULUS_PAIRS:
        raise ValueError(f"stimulus_pair must be one of {list(STIMULUS_PAIRS)}")
    if corridor_region == "texture_0_4":
        start, stop = 0.0, 4.0
    elif corridor_region == "full_corridor_0_6":
        start, stop = 0.0, 6.0
    elif corridor_region == "custom":
        start, stop = float(custom_position_start_m), float(custom_position_end_m)
    else:
        raise ValueError("unknown corridor_region")
    if not 0 <= start < stop <= 6:
        raise ValueError("corridor bounds must satisfy 0 <= start < end <= 6")
    if int(position_bin_count) not in {12, 18, 24}:
        raise ValueError("position_bin_count must be 12, 18, or 24")
    if real_coverage_rule not in {"complete", "available"}:
        raise ValueError("real_coverage_rule must be complete or available")
    role_a, role_b, label = STIMULUS_PAIRS[stimulus_pair]
    return {
        "pair": stimulus_pair, "role_a": role_a, "role_b": role_b, "label": label,
        "region": corridor_region, "start": start, "stop": stop,
        "n_position_bins": int(position_bin_count), "coverage_rule": real_coverage_rule,
    }


@graph.node(outputs="time_estimator_spec")
def define_time_and_estimator_rules(
    trial_block_width=40,
    block_stride_mode="nonoverlap",
    early_trial_horizon=140,
    real_crossvalidation_folds=4,
    real_minimum_trials_per_role=4,
    minimum_valid_early_blocks=3,
):
    block, horizon = int(trial_block_width), int(early_trial_horizon)
    if block not in {32, 40, 48, 60}:
        raise ValueError("trial_block_width must be 32, 40, 48, or 60")
    if horizon < 2 * block:
        raise ValueError("early_trial_horizon must span at least two blocks")
    if block_stride_mode not in {"nonoverlap", "half_block_overlap"}:
        raise ValueError("unknown block_stride_mode")
    return {
        "block_trials": block,
        "stride_trials": block if block_stride_mode == "nonoverlap" else block // 2,
        "stride_mode": block_stride_mode,
        "early_horizon": horizon,
        "n_folds": int(real_crossvalidation_folds),
        "min_per_role": int(real_minimum_trials_per_role),
        "minimum_early_blocks": int(minimum_valid_early_blocks),
    }


@graph.node(outputs="execution_spec")
def define_real_execution_policy(
    claim_status="pipeline_preview",
    load_mode="plan_only",
    compute_cross_temporal=True,
    compute_position_surface=True,
    real_inference_alternative="two-sided",
    real_bootstrap_draws=4000,
    allow_exploratory_inference=False,
):
    if claim_status not in {"pipeline_preview", "complete_cohort", "sensitivity"}:
        raise ValueError("unknown claim_status")
    if load_mode not in {"plan_only", "load_and_analyse"}:
        raise ValueError("unknown load_mode")
    if real_inference_alternative not in {"greater", "two-sided"}:
        raise ValueError("unknown inference alternative")
    return {
        "claim_status": claim_status,
        "load_mode": load_mode,
        "cross_temporal": bool(compute_cross_temporal),
        "position_surface": bool(compute_position_surface),
        "alternative": real_inference_alternative,
        "bootstrap_draws": int(real_bootstrap_draws),
        "allow_exploratory_inference": bool(allow_exploratory_inference),
    }


@graph.node(outputs="eligible_sessions")
def enumerate_real_eligible_sessions(analysis_dataset, real_cohort_policy):
    sessions = []
    for group, prefix in ((REWARDED, "sup"), (UNREWARDED, "unsup")):
        experiment = f"{prefix}_{real_cohort_policy['training_set']}_{real_cohort_policy['stage']}"
        for recording in analysis_dataset.recordings(experiment=experiment):
            metadata = next(
                row for row in recording_rows
                if row["experiment"] == experiment
                and row["recording_id"] == recording.recording_id
            )
            sessions.append({
                "group": group, "experiment": experiment, "recording": recording,
                "recording_id": recording.recording_id, "mouse": recording.mouse,
                "date": str(metadata["date"]), "block": str(metadata["block"]),
                "reward_type": str(metadata["reward_type"]),
            })
    return sessions


@graph.node(outputs="selected_real_sessions")
def select_real_sessions(eligible_sessions, real_cohort_policy):
    selected = []
    for group in (REWARDED, UNREWARDED):
        rows = sorted(
            [row for row in eligible_sessions if row["group"] == group],
            key=lambda row: (row["date"], row["mouse"], row["recording_id"]),
        )
        limit = real_cohort_policy["limit"]
        if limit and len(rows) > limit:
            if real_cohort_policy["sampling"] == "latest":
                rows = rows[-limit:]
            elif real_cohort_policy["sampling"] == "spread_dates":
                indices = np.linspace(0, len(rows)-1, limit).round().astype(int)
                rows = [rows[index] for index in indices]
            else:
                rows = rows[:limit]
        selected.extend(rows)
    return selected


@graph.node(outputs="analysis_spec")
def assemble_analysis_spec(
    real_cohort_policy,
    representation_spec,
    real_contrast_spec,
    time_estimator_spec,
    execution_spec,
    selected_real_sessions,
):
    observed = {
        "training_set": real_cohort_policy["training_set"],
        "stage": real_cohort_policy["stage"],
        "cohort_size": real_cohort_policy["limit"],
        "representation": representation_spec["mode"],
        "area": representation_spec["area"],
        "features": representation_spec["n_features"],
        "movement": representation_spec["movement_rule"],
        "stimulus_pair": real_contrast_spec["pair"],
        "position_start": real_contrast_spec["start"],
        "position_stop": real_contrast_spec["stop"],
        "position_bins": real_contrast_spec["n_position_bins"],
        "coverage": real_contrast_spec["coverage_rule"],
        "block_trials": time_estimator_spec["block_trials"],
        "stride_mode": time_estimator_spec["stride_mode"],
        "early_horizon": time_estimator_spec["early_horizon"],
        "folds": time_estimator_spec["n_folds"],
        "min_role": time_estimator_spec["min_per_role"],
        "min_early_blocks": time_estimator_spec["minimum_early_blocks"],
    }
    reference = {
        "training_set": "train1", "stage": "before_learning", "cohort_size": 0,
        "representation": "svd_population", "area": "V1", "features": 12,
        "movement": "moving_only", "stimulus_pair": "leaf1_vs_circle1",
        "position_start": 0.0, "position_stop": 4.0, "position_bins": 18,
        "coverage": "complete", "block_trials": 40, "stride_mode": "nonoverlap",
        "early_horizon": 140, "folds": 4, "min_role": 4,
        "min_early_blocks": 3,
    }
    deviations = {
        name: {"reference": reference[name], "selected": observed[name]}
        for name in reference if observed[name] != reference[name]
    }
    complete = (
        execution_spec["claim_status"] == "complete_cohort"
        and real_cohort_policy["limit"] == 0
    )
    return {
        **observed,
        "cohort_policy": real_cohort_policy,
        "representation": representation_spec,
        "contrast": real_contrast_spec,
        "time": time_estimator_spec,
        "execution": execution_spec,
        "sessions": selected_real_sessions,
        "reference_specification": reference,
        "deviations": deviations,
        "complete_cohort": complete,
        "result_label": (
            "COMPLETE COHORT DESCRIPTIVE" if complete
            else "PIPELINE PREVIEW" if execution_spec["claim_status"] == "pipeline_preview"
            else "SENSITIVITY / EXPLORATORY"
        ),
    }


@graph.node(outputs="analysis_spec")
def choose_real_hypothesis_analysis(
    analysis_dataset,
    training_stage="before_learning",
    cortical_area="V1",
    cohort_size=2,
    trial_block_width=40,
    early_trial_horizon=140,
    feature_count=12,
    load_mode="plan_only",
):
    if training_stage not in {"before_learning", "after_learning"}:
        raise ValueError("training_stage must be before_learning or after_learning")
    if cortical_area not in AREA_IDS:
        raise ValueError(f"cortical_area must be one of {list(AREA_IDS)}")
    if int(cohort_size) not in {0, 1, 2}:
        raise ValueError("cohort_size must be 0, 1, or 2")
    if int(trial_block_width) < 24:
        raise ValueError("trial_block_width must be at least 24")
    if int(early_trial_horizon) < 2 * int(trial_block_width):
        raise ValueError("early_trial_horizon must cover at least two blocks")
    if int(feature_count) not in {8, 12, 24}:
        raise ValueError("feature_count must be 8, 12, or 24")
    if load_mode not in {"plan_only", "load_and_analyse"}:
        raise ValueError("load_mode must be plan_only or load_and_analyse")

    experiment_by_group = {
        REWARDED: f"sup_train1_{training_stage}",
        UNREWARDED: f"unsup_train1_{training_stage}",
    }
    sessions = []
    for group, experiment in experiment_by_group.items():
        recordings = analysis_dataset.recordings(experiment=experiment)
        if int(cohort_size):
            recordings = recordings[: int(cohort_size)]
        for recording in recordings:
            sessions.append({
                "group": group,
                "experiment": experiment,
                "recording": recording,
                "mouse": recording.mouse,
                "recording_id": recording.recording_id,
            })
    return {
        "stage": training_stage,
        "area": cortical_area,
        "cohort_size": int(cohort_size),
        "block_trials": int(trial_block_width),
        "early_horizon": int(early_trial_horizon),
        "n_features": int(feature_count),
        "load_mode": load_mode,
        "sessions": sessions,
        "complete_cohort": int(cohort_size) == 0,
    }


@graph.node(outputs="file_preflight")
def resolve_real_analysis_files(analysis_spec):
    rows = []
    for session in analysis_spec["sessions"]:
        layers = ("behavior", analysis_spec["representation"]["layer"], "retinotopy")
        files = []
        for layer in layers:
            item = session["recording"].file(
                layer,
                experiment=session["experiment"] if layer == "behavior" else None,
            )
            files.append({
                "layer": layer, "name": item.name,
                "size_bytes": item.size_bytes, "md5": item.md5,
            })
        rows.append({**session, "files": files, "bytes": sum(item["size_bytes"] for item in files)})
    return rows


@graph.node(outputs="analysis_budget_audit")
def audit_real_analysis_budget(file_preflight, maximum_analysis_gib=4.0):
    unique = {}
    for row in file_preflight:
        for item in row["files"]:
            unique.setdefault(item["name"], item)
    total = sum(item["size_bytes"] for item in unique.values()) / 2**30
    largest = max((item["size_bytes"] for item in unique.values()), default=0) / 2**30
    return {
        "total_gib": total,
        "largest_gib": largest,
        "limit_gib": float(maximum_analysis_gib),
        "approved": total <= float(maximum_analysis_gib),
        "unique_files": len(unique),
    }


@graph.node(outputs="session_tensors")
def load_verified_trial_tensors_expanded(
    analysis_dataset,
    analysis_spec,
    file_preflight,
    analysis_budget_audit,
):
    if analysis_spec["execution"]["load_mode"] == "plan_only":
        return [
            {
                **{key: row[key] for key in (
                    "group", "experiment", "recording", "recording_id", "mouse",
                    "date", "block", "reward_type",
                )},
                "loaded": False,
                "files": [item["name"] for item in row["files"]],
                "bytes": row["bytes"],
            }
            for row in file_preflight
        ]
    if analysis_spec["representation"]["mode"] == "full_neuron_replication":
        raise ValueError(
            "Full-neuron replication is exposed as a heavy fetch plan, but this "
            "interactive graph currently implements only the SVD population estimator."
        )
    if not analysis_budget_audit["approved"]:
        raise ValueError(
            f"Selected files total {analysis_budget_audit['total_gib']:.2f} GiB, above "
            f"the declared {analysis_budget_audit['limit_gib']:.2f} GiB limit."
        )
    if not analysis_dataset.connected:
        raise ValueError(
            "Load and analyse requires the mounted shared Drive. "
            "Run in Colab or set ZHONG2025_DATASET_ROOT."
        )
    tensors = []
    for offset, session in enumerate(analysis_spec["sessions"], start=1):
        recording = session["recording"]
        print(f"[{offset}/{len(analysis_spec['sessions'])}] {session['group']} · {recording.recording_id}")
        behavior = recording.load("behavior", experiment=session["experiment"])
        svd = recording.load("reduced_neural")
        retinotopy = recording.load("retinotopy")
        tensor = prepare_session_trials(
            behavior, svd, retinotopy,
            area=analysis_spec["representation"]["area"],
            n_features=analysis_spec["representation"]["n_features"],
            n_position_bins=analysis_spec["contrast"]["n_position_bins"],
            movement_rule=analysis_spec["representation"]["movement_rule"],
            mouse_id=session["mouse"], recording_id=session["recording_id"],
        )
        tensors.append({
            **session, **tensor, "loaded": True,
            "reward_mode_behavior": str(behavior.get("Reward_Mode", "unknown")),
        })
        del behavior, svd, retinotopy
    return tensors


@graph.node(outputs="tensor_qc")
def audit_loaded_session_tensors(session_tensors, analysis_spec):
    if not session_tensors or not session_tensors[0]["loaded"]:
        return {
            "loaded": False,
            "selected_sessions": len(session_tensors),
            "selected_mice": len({row["mouse"] for row in session_tensors}),
            "status": "plan only; no arrays opened",
        }
    rows = []
    for session in session_tensors:
        roles, counts = np.unique(session["labels"], return_counts=True)
        rows.append({
            "mouse": session["mouse"], "group": session["group"],
            "trials": len(session["trial_id"]),
            "shape": session["trial_features"].shape,
            "finite_fraction": float(np.isfinite(session["trial_features"]).mean()),
            "role_counts": {int(role): int(count) for role, count in zip(roles, counts)},
            "alignment": session["alignment"],
            "movement_rule": session["movement_rule"],
        })
    return {"loaded": True, "rows": rows, "status": "verified and aligned"}


# Compact predecessor retained for notebook readers; the graph uses the expanded loader.
@graph.node(outputs="session_tensors_compact")
def load_verified_trial_tensors(analysis_dataset, analysis_spec):
    tensors = []
    if analysis_spec["load_mode"] == "plan_only":
        for session in analysis_spec["sessions"]:
            recording = session["recording"]
            files = [
                recording.file("behavior", experiment=session["experiment"]),
                recording.file("reduced_neural"),
                recording.file("retinotopy"),
            ]
            tensors.append({
                **session,
                "loaded": False,
                "files": [item.name for item in files],
                "bytes": sum(item.size_bytes for item in files),
            })
        return tensors

    if not analysis_dataset.connected:
        raise ValueError(
            "Load and analyse requires the mounted shared Drive. "
            "Run in Colab or set ZHONG2025_DATASET_ROOT."
        )
    for offset, session in enumerate(analysis_spec["sessions"], start=1):
        recording = session["recording"]
        print(
            f"[{offset}/{len(analysis_spec['sessions'])}] "
            f"{session['group']} · {recording.recording_id}"
        )
        behavior = recording.load("behavior", experiment=session["experiment"])
        svd = recording.load("reduced_neural")
        retinotopy = recording.load("retinotopy")
        tensor = prepare_session_trials(
            behavior,
            svd,
            retinotopy,
            area=analysis_spec["area"],
            n_features=analysis_spec["n_features"],
            n_position_bins=18,
            mouse_id=session["mouse"],
            recording_id=session["recording_id"],
        )
        tensors.append({
            **session,
            **tensor,
            "loaded": True,
            "reward_mode_behavior": str(behavior.get("Reward_Mode", "unknown")),
        })
        del behavior, svd, retinotopy
    return tensors


@graph.node(outputs="session_position_masks")
def construct_real_position_masks(session_tensors, analysis_spec):
    if not session_tensors or not session_tensors[0]["loaded"]:
        return []
    rows = []
    for session in session_tensors:
        position = np.asarray(session["position_centers_m"], dtype=float)
        mask = (
            (position >= analysis_spec["contrast"]["start"])
            & (position < analysis_spec["contrast"]["stop"])
        )
        if not np.any(mask):
            raise ValueError(f"corridor window selects no bins for {session['recording_id']}")
        rows.append({"mouse": session["mouse"], "mask": mask, "position": position})
    return rows


@graph.node(outputs="real_block_curves")
def estimate_real_blockwise_dprime(session_tensors, session_position_masks, analysis_spec):
    if not session_position_masks:
        return []
    masks = {row["mouse"]: row["mask"] for row in session_position_masks}
    rows = []
    for session in session_tensors:
        curve = blockwise_dprime(
            session["trial_features"], session["labels"], session["trial_id"],
            position_mask=masks[session["mouse"]],
            role_a=analysis_spec["contrast"]["role_a"],
            role_b=analysis_spec["contrast"]["role_b"],
            block_trials=analysis_spec["time"]["block_trials"],
            stride_trials=analysis_spec["time"]["stride_trials"],
            n_folds=analysis_spec["time"]["n_folds"],
            min_per_role=analysis_spec["time"]["min_per_role"],
            require_complete_position_coverage=analysis_spec["contrast"]["coverage_rule"] == "complete",
        )
        rows.append({"mouse": session["mouse"], "group": session["group"], "curve": curve})
    return rows


@graph.node(outputs="real_rate_summaries")
def fit_real_early_rates(real_block_curves, analysis_spec):
    rows = []
    for row in real_block_curves:
        slope = fit_early_slope(
            row["curve"]["midpoint"], row["curve"]["dprime"],
            early_horizon=analysis_spec["time"]["early_horizon"],
        )
        sufficient = slope["n_blocks"] >= analysis_spec["time"]["minimum_early_blocks"]
        if not sufficient:
            slope = {**slope, "slope": float("nan"), "intercept": float("nan")}
        rows.append({**row, "slope": slope, "sufficient_early_support": bool(sufficient)})
    return rows


@graph.node(outputs="real_saturation_summaries")
def fit_real_saturation_models(real_rate_summaries):
    return [
        {
            **row,
            "saturation": fit_saturation_curve(row["curve"]["midpoint"], row["curve"]["dprime"]),
        }
        for row in real_rate_summaries
    ]


@graph.node(outputs="cross_temporal_diagnostics")
def compute_real_cross_temporal_diagnostics(
    session_tensors,
    session_position_masks,
    analysis_spec,
):
    if not session_position_masks or not analysis_spec["execution"]["cross_temporal"]:
        return {"available": False, "rows": [], "reason": "disabled or arrays not loaded"}
    masks = {row["mouse"]: row["mask"] for row in session_position_masks}
    rows = []
    for session in session_tensors:
        result = cross_temporal_dprime(
            session["trial_features"], session["labels"], session["trial_id"],
            position_mask=masks[session["mouse"]],
            role_a=analysis_spec["contrast"]["role_a"],
            role_b=analysis_spec["contrast"]["role_b"],
            block_trials=analysis_spec["time"]["block_trials"],
            min_per_role=analysis_spec["time"]["min_per_role"],
            require_complete_position_coverage=analysis_spec["contrast"]["coverage_rule"] == "complete",
        )
        rows.append({"mouse": session["mouse"], "group": session["group"], "result": result})
    return {"available": True, "rows": rows}


@graph.node(outputs="position_diagnostics")
def compute_real_position_diagnostics(session_tensors, analysis_spec):
    if not session_tensors or not session_tensors[0]["loaded"] or not analysis_spec["execution"]["position_surface"]:
        return {"available": False, "rows": [], "reason": "disabled or arrays not loaded"}
    rows = []
    for session in session_tensors:
        result = position_dprime_surface(
            session["trial_features"], session["labels"], session["trial_id"],
            role_a=analysis_spec["contrast"]["role_a"],
            role_b=analysis_spec["contrast"]["role_b"],
            block_trials=analysis_spec["time"]["block_trials"],
            n_folds=analysis_spec["time"]["n_folds"],
            min_per_role=analysis_spec["time"]["min_per_role"],
            require_complete_position_coverage=analysis_spec["contrast"]["coverage_rule"] == "complete",
        )
        rows.append({"mouse": session["mouse"], "group": session["group"], "result": result})
    return {"available": True, "rows": rows}


@graph.node(outputs="behavior_diagnostics")
def summarize_real_behavior_and_support(session_tensors, session_position_masks, analysis_spec):
    if not session_position_masks:
        return []
    masks = {row["mouse"]: row["mask"] for row in session_position_masks}
    block = analysis_spec["time"]["block_trials"]
    stride = analysis_spec["time"]["stride_trials"]
    rows = []
    for session in session_tensors:
        mask = masks[session["mouse"]]
        with np.errstate(invalid="ignore"):
            speed_by_trial = np.nanmean(session["run_speed"][:, mask], axis=1)
        support_by_trial = np.mean(session["frame_counts"][:, mask], axis=1)
        starts = range(0, max(len(speed_by_trial)-block+1, 0), stride)
        speed_blocks = np.asarray([np.nanmean(speed_by_trial[start:start+block]) for start in starts])
        support_blocks = np.asarray([np.mean(support_by_trial[start:start+block]) for start in starts])
        rows.append({
            "mouse": session["mouse"], "group": session["group"],
            "speed_blocks": speed_blocks, "support_blocks": support_blocks,
            "mean_speed_by_position": np.nanmean(session["run_speed"], axis=0),
            "mean_support_by_position": np.mean(session["frame_counts"], axis=0),
            "cue_position_m": session["cue_position_m"],
            "reward_position_m": session["reward_position_m"],
            "first_lick_position_m": session["first_lick_position_m"],
        })
    return rows


@graph.node(outputs="session_curves")
def assemble_real_session_results(
    session_tensors,
    real_saturation_summaries,
    cross_temporal_diagnostics,
    position_diagnostics,
    behavior_diagnostics,
):
    if not real_saturation_summaries:
        return []
    tensor = {row["mouse"]: row for row in session_tensors}
    temporal = {row["mouse"]: row["result"] for row in cross_temporal_diagnostics["rows"]}
    position = {row["mouse"]: row["result"] for row in position_diagnostics["rows"]}
    behavior = {row["mouse"]: row for row in behavior_diagnostics}
    rows = []
    for summary in real_saturation_summaries:
        mouse = summary["mouse"]
        session, diagnostics = tensor[mouse], behavior[mouse]
        rows.append({
            **summary,
            "recording_id": session["recording_id"],
            "date": session["date"], "reward_mode": session["reward_mode_behavior"],
            "n_trials": len(session["trial_id"]),
            "cross_temporal": temporal.get(mouse, {"dprime": np.empty((0, 0))}),
            "position_surface": position.get(mouse, {"midpoint": np.array([]), "dprime": np.empty((0, len(session["position_centers_m"]))) }),
            "position": session["position_centers_m"],
            **{key: diagnostics[key] for key in (
                "speed_blocks", "support_blocks", "mean_speed_by_position",
                "mean_support_by_position", "cue_position_m", "reward_position_m",
                "first_lick_position_m",
            )},
        })
    return rows


# Compact predecessor retained for notebook readers; not wired into the expanded graph.
@graph.node(outputs="session_curves_compact")
def estimate_real_session_curves(session_tensors, analysis_spec):
    if not session_tensors or not session_tensors[0]["loaded"]:
        return []
    rows = []
    for session in session_tensors:
        curve = blockwise_dprime(
            session["trial_features"],
            session["labels"],
            session["trial_id"],
            position_mask=session["texture_mask"],
            role_a=2,
            role_b=0,
            block_trials=analysis_spec["block_trials"],
            n_folds=4,
            min_per_role=4,
        )
        slope = fit_early_slope(
            curve["midpoint"], curve["dprime"],
            early_horizon=analysis_spec["early_horizon"],
        )
        saturation = fit_saturation_curve(curve["midpoint"], curve["dprime"])
        temporal = cross_temporal_dprime(
            session["trial_features"],
            session["labels"],
            session["trial_id"],
            position_mask=session["texture_mask"],
            role_a=2,
            role_b=0,
            block_trials=analysis_spec["block_trials"],
            min_per_role=4,
        )
        position = position_dprime_surface(
            session["trial_features"],
            session["labels"],
            session["trial_id"],
            role_a=2,
            role_b=0,
            block_trials=analysis_spec["block_trials"],
            n_folds=4,
            min_per_role=4,
        )

        block = analysis_spec["block_trials"]
        texture = session["texture_mask"]
        with np.errstate(invalid="ignore"):
            speed_by_trial = np.nanmean(session["run_speed"][:, texture], axis=1)
            support_by_trial = np.mean(session["frame_counts"][:, texture], axis=1)
        speed_blocks, support_blocks = [], []
        for start in range(0, max(len(speed_by_trial)-block+1, 0), block):
            speed_blocks.append(float(np.nanmean(speed_by_trial[start:start+block])))
            support_blocks.append(float(np.nanmean(support_by_trial[start:start+block])))
        rows.append({
            "mouse": session["mouse"],
            "recording_id": session["recording_id"],
            "group": session["group"],
            "reward_mode": session["reward_mode_behavior"],
            "n_trials": len(session["trial_id"]),
            "curve": curve,
            "slope": slope,
            "saturation": saturation,
            "cross_temporal": temporal,
            "position_surface": position,
            "position": session["position_centers_m"],
            "speed_blocks": np.asarray(speed_blocks),
            "support_blocks": np.asarray(support_blocks),
            "mean_speed_by_position": np.nanmean(session["run_speed"], axis=0),
            "mean_support_by_position": np.mean(session["frame_counts"], axis=0),
            "cue_position_m": session["cue_position_m"],
            "reward_position_m": session["reward_position_m"],
            "first_lick_position_m": session["first_lick_position_m"],
        })
    return rows


@graph.node(outputs="mouse_statistics")
def summarize_real_mouse_statistics_expanded(session_curves, analysis_spec):
    return {
        "rows": session_curves,
        "slopes": np.asarray([row["slope"]["slope"] for row in session_curves], dtype=float),
        "groups": np.asarray([row["group"] for row in session_curves]),
        "mice": np.asarray([row["mouse"] for row in session_curves]),
        "complete_cohort": analysis_spec["complete_cohort"],
        "result_label": analysis_spec["result_label"],
    }


@graph.node(outputs="group_inference")
def infer_real_reward_rate_difference_expanded(mouse_statistics, analysis_spec):
    if not mouse_statistics["rows"]:
        return {"available": False, "reason": "no real session curves; run is preflight only"}
    allowed = (
        analysis_spec["complete_cohort"]
        or analysis_spec["execution"]["allow_exploratory_inference"]
    )
    if not allowed:
        return {
            "available": False,
            "reason": "inference suppressed for previews/sensitivities; inspect QC or enable the explicit exploratory override",
        }
    slopes = mouse_statistics["slopes"]
    groups = mouse_statistics["groups"]
    mice = mouse_statistics["mice"]
    valid_counts = {
        group: int(np.count_nonzero((groups == group) & np.isfinite(slopes)))
        for group in (REWARDED, UNREWARDED)
    }
    if min(valid_counts.values()) < 2:
        return {"available": False, "reason": "fewer than two valid mice in a group", "valid_counts": valid_counts}
    rows = mouse_statistics["rows"]
    max_blocks = max(len(row["curve"]["dprime"]) for row in rows)
    group_curves = {}
    for group in (REWARDED, UNREWARDED):
        selected = [row for row in rows if row["group"] == group]
        values = np.full((len(selected), max_blocks), np.nan)
        xvalues = np.full((len(selected), max_blocks), np.nan)
        for index, row in enumerate(selected):
            n = len(row["curve"]["dprime"])
            values[index, :n] = row["curve"]["dprime"]
            xvalues[index, :n] = row["curve"]["midpoint"]
        count = np.sum(np.isfinite(values), axis=0)
        mean = np.divide(np.nansum(values, axis=0), count, out=np.full(max_blocks, np.nan), where=count > 0)
        xcount = np.sum(np.isfinite(xvalues), axis=0)
        xmean = np.divide(np.nansum(xvalues, axis=0), xcount, out=np.full(max_blocks, np.nan), where=xcount > 0)
        sem = np.full(max_blocks, np.nan)
        for block_index in range(max_blocks):
            finite = values[:, block_index][np.isfinite(values[:, block_index])]
            if len(finite) > 1:
                sem[block_index] = np.std(finite, ddof=1) / np.sqrt(len(finite))
        group_curves[group] = {"x": xmean, "mean": mean, "sem": sem, "values": values, "valid_mice": count}
    return {
        "available": True,
        "valid_counts": valid_counts,
        "permutation": exact_group_permutation(
            slopes, groups, mouse_ids=mice,
            alternative=analysis_spec["execution"]["alternative"],
        ),
        "bootstrap": bootstrap_group_difference(
            slopes, groups, mouse_ids=mice,
            n_boot=analysis_spec["execution"]["bootstrap_draws"], seed=23,
        ),
        "group_curves": group_curves,
        "exploratory_override": not analysis_spec["complete_cohort"],
    }


@graph.node(outputs="real_leave_one_out")
def leave_one_mouse_out_real_group(mouse_statistics, group_inference):
    if not group_inference.get("available"):
        return []
    slopes, groups, mice = (
        mouse_statistics["slopes"], mouse_statistics["groups"], mouse_statistics["mice"]
    )
    rows = []
    for index, mouse in enumerate(mice):
        keep = (np.arange(len(mice)) != index) & np.isfinite(slopes)
        if min(np.count_nonzero(keep & (groups == group)) for group in (REWARDED, UNREWARDED)) < 1:
            continue
        rows.append({
            "omitted_mouse": str(mouse), "omitted_group": str(groups[index]),
            "difference": float(
                np.mean(slopes[keep & (groups == REWARDED)])
                - np.mean(slopes[keep & (groups == UNREWARDED)])
            ),
        })
    return rows


@graph.node(outputs="mouse_statistics_compact")
def summarize_real_mouse_statistics(session_curves, analysis_spec):
    return {
        "rows": session_curves,
        "slopes": np.asarray([row["slope"]["slope"] for row in session_curves], dtype=float),
        "groups": [row["group"] for row in session_curves],
        "mice": [row["mouse"] for row in session_curves],
        "complete_cohort": analysis_spec["complete_cohort"],
    }


@graph.node(outputs="group_inference_compact")
def infer_real_reward_rate_difference(mouse_statistics):
    rows = mouse_statistics["rows"]
    if not rows:
        return {"available": False}
    slopes = mouse_statistics["slopes"]
    groups = mouse_statistics["groups"]
    mice = mouse_statistics["mice"]
    valid_counts = {
        group: len({
            mouse for mouse, observed_group, slope in zip(mice, groups, slopes)
            if observed_group == group and np.isfinite(slope)
        })
        for group in (REWARDED, UNREWARDED)
    }
    if min(valid_counts.values()) < 2:
        return {"available": False, "valid_counts": valid_counts}

    max_blocks = max(len(row["curve"]["dprime"]) for row in rows)
    group_curves = {}
    for group in (REWARDED, UNREWARDED):
        selected = [row for row in rows if row["group"] == group]
        values = np.full((len(selected), max_blocks), np.nan)
        xvalues = np.full((len(selected), max_blocks), np.nan)
        for index, row in enumerate(selected):
            n = len(row["curve"]["dprime"])
            values[index, :n] = row["curve"]["dprime"]
            xvalues[index, :n] = row["curve"]["midpoint"]
        group_curves[group] = {
            "x": np.nanmean(xvalues, axis=0),
            "mean": np.nanmean(values, axis=0),
            "sem": np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(values), axis=0)),
            "values": values,
        }

    return {
        "available": True,
        "valid_counts": valid_counts,
        "permutation": exact_group_permutation(slopes, groups, mouse_ids=mice),
        "bootstrap": bootstrap_group_difference(
            slopes, groups, mouse_ids=mice, n_boot=4000, seed=23
        ),
        "group_curves": group_curves,
    }


@graph.node(outputs="legacy_plots")
def plot_real_reward_hypothesis(
    analysis_spec,
    session_tensors,
    session_curves,
    mouse_statistics,
    group_inference,
):
    if analysis_spec["load_mode"] == "plan_only":
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
        groups = [REWARDED, UNREWARDED]
        counts = [sum(row["group"] == group for row in session_tensors) for group in groups]
        bars = axes[0].bar(["Rewarded", "Unrewarded"], counts, color=[GROUP_COLOURS[g] for g in groups])
        axes[0].bar_label(bars)
        axes[0].set(title="Sessions selected for the next run", ylabel="independent mice")
        axes[1].axis("off")
        total = sum(row["bytes"] for row in session_tensors)
        axes[1].text(
            0, 1,
            "PLAN ONLY - NO ARRAYS LOADED\\n\\n"
            f"Stage: {analysis_spec['stage'].replace('_', ' ')}\\n"
            f"Area: {AREA_LABELS[analysis_spec['area']]}\\n"
            f"Feature count: {analysis_spec['n_features']}\\n"
            f"Block width: {analysis_spec['block_trials']} trials\\n"
            f"Early horizon: {analysis_spec['early_horizon']} trials\\n"
            f"Selected recording-file sum: {total/2**30:.2f} GiB\\n\\n"
            "Next: choose Load and analyse in the hollow port.\\n"
            "Use the two-per-group preview to test the pipeline.\\n"
            "Use all eligible mice for a complete-cohort descriptive result.",
            va="top", fontsize=10,
        )
        fig.suptitle("Graph 4 · real hypothesis analysis preflight")
        plt.close(fig)
        return fig

    fig, axes = plt.subplots(2, 4, figsize=(16.2, 8.0), constrained_layout=True)
    for row in session_curves:
        axes[0, 0].plot(
            row["curve"]["midpoint"], row["curve"]["dprime"],
            color=GROUP_COLOURS[row["group"]], alpha=0.46, linewidth=1.1,
        )
    axes[0, 0].axvline(analysis_spec["early_horizon"], color="#f28e2b", linestyle="--", label="early horizon")
    axes[0, 0].set(title="Every mouse (fixed blocks)", xlabel="trial number", ylabel="held-out d′")
    axes[0, 0].legend(fontsize=7)

    if group_inference.get("available"):
        for group in (REWARDED, UNREWARDED):
            summary = group_inference["group_curves"][group]
            axes[0, 1].plot(summary["x"], summary["mean"], marker="o", color=GROUP_COLOURS[group], label=group)
            axes[0, 1].fill_between(summary["x"], summary["mean"]-summary["sem"], summary["mean"]+summary["sem"], color=GROUP_COLOURS[group], alpha=0.16)
    axes[0, 1].set(title="Group mean ± mouse SEM", xlabel="trial number", ylabel="held-out d′")
    axes[0, 1].legend(fontsize=8)

    rng = np.random.default_rng(13)
    slopes = mouse_statistics["slopes"]
    groups = np.asarray(mouse_statistics["groups"])
    for offset, group in enumerate((REWARDED, UNREWARDED)):
        selected = slopes[(groups == group) & np.isfinite(slopes)]
        axes[0, 2].scatter(offset + rng.normal(0, 0.035, len(selected)), selected, color=GROUP_COLOURS[group], s=38)
        if len(selected):
            axes[0, 2].plot([offset-0.16, offset+0.16], [np.mean(selected)]*2, color="black", linewidth=2)
    axes[0, 2].axhline(0, color="#999", linewidth=1)
    axes[0, 2].set_xticks([0, 1], ["Rewarded", "Unrewarded"])
    axes[0, 2].set(title="Primary: one early slope per mouse", ylabel="Δd′ per trial")

    axes[0, 3].axis("off")
    if group_inference.get("available"):
        boot = group_inference["bootstrap"]
        perm = group_inference["permutation"]
        lo, hi = boot["ci"]
        completeness = "COMPLETE COHORT" if analysis_spec["complete_cohort"] else "PREVIEW / SENSITIVITY"
        plateau_observed = sum(bool(row["saturation"]["plateau_observed"]) for row in session_curves)
        axes[0, 3].text(
            0, 1,
            f"{completeness}\\n\\n"
            f"Rewarded - unrewarded slope: {boot['difference']:.5f} d′/trial\\n"
            f"Mouse-bootstrap 95% interval: [{lo:.5f}, {hi:.5f}]\\n"
            f"Exact {analysis_spec['execution']['alternative']} label-permutation p: "
            f"{perm['pvalue']:.4f}\\n"
            f"Label permutations: {int(perm['permutations'])}\\n"
            f"Valid mice: {group_inference['valid_counts']}\\n"
            f"Plateau observed in {plateau_observed}/{len(session_curves)} fits\\n\\n"
            "Report the effect and interval before the p-value.\\n"
            "Permutation validity assumes cohort labels are exchangeable.",
            va="top", fontsize=9,
        )
    else:
        axes[0, 3].text(0, 1, "Not enough valid mice for a group contrast.", va="top")

    for group in (REWARDED, UNREWARDED):
        selected = [row for row in session_curves if row["group"] == group]
        max_blocks = max((len(row["curve"]["dprime"]) for row in selected), default=0)
        separation = np.full((len(selected), max_blocks), np.nan)
        spread = np.full((len(selected), max_blocks), np.nan)
        x = np.full((len(selected), max_blocks), np.nan)
        for index, row in enumerate(selected):
            n = len(row["curve"]["dprime"])
            separation[index, :n] = row["curve"]["separation"]
            spread[index, :n] = row["curve"]["spread"]
            x[index, :n] = row["curve"]["midpoint"]
        if max_blocks:
            xm = np.nanmean(x, axis=0)
            axes[1, 0].plot(xm, np.nanmean(separation, axis=0), color=GROUP_COLOURS[group], label=f"{group} mean separation")
            axes[1, 0].plot(xm, np.nanmean(spread, axis=0), color=GROUP_COLOURS[group], linestyle="--", label=f"{group} spread")
    axes[1, 0].set(title="Why d′ changed: numerator vs spread", xlabel="trial number", ylabel="held-out score units")
    axes[1, 0].legend(fontsize=6.8)

    min_blocks = min((row["cross_temporal"]["dprime"].shape[0] for row in session_curves), default=0)
    if min_blocks:
        matrices = {}
        for group in (REWARDED, UNREWARDED):
            selected = [row["cross_temporal"]["dprime"][:min_blocks, :min_blocks] for row in session_curves if row["group"] == group]
            matrices[group] = np.nanmean(np.stack(selected), axis=0)
        difference = matrices[REWARDED] - matrices[UNREWARDED]
        image = axes[1, 1].imshow(difference, origin="lower", aspect="equal", cmap="coolwarm")
        fig.colorbar(image, ax=axes[1, 1], label="rewarded - unrewarded d′")
    axes[1, 1].set(title="Cross-temporal axis generalization (exploratory)", xlabel="test block", ylabel="train block")

    for group in (REWARDED, UNREWARDED):
        selected = [row for row in session_curves if row["group"] == group]
        max_blocks = max((len(row["speed_blocks"]) for row in selected), default=0)
        values = np.full((len(selected), max_blocks), np.nan)
        for index, row in enumerate(selected):
            values[index, :len(row["speed_blocks"])] = row["speed_blocks"]
        if max_blocks:
            x = (np.arange(max_blocks)+0.5) * analysis_spec["block_trials"]
            axes[1, 2].plot(x, np.nanmean(values, axis=0), color=GROUP_COLOURS[group], marker="o", label=group)
    axes[1, 2].set(title="Behavioral control: running speed", xlabel="trial number", ylabel="mean speed in texture")
    axes[1, 2].legend(fontsize=8)

    for group in (REWARDED, UNREWARDED):
        selected = [row for row in session_curves if row["group"] == group]
        for role, linestyle in (("n_a", "-"), ("n_b", "--")):
            max_blocks = max((len(row["curve"][role]) for row in selected), default=0)
            values = np.full((len(selected), max_blocks), np.nan)
            xvalues = np.full((len(selected), max_blocks), np.nan)
            for index, row in enumerate(selected):
                n = len(row["curve"][role])
                values[index, :n] = row["curve"][role]
                xvalues[index, :n] = row["curve"]["midpoint"]
            if max_blocks:
                axes[1, 3].plot(np.nanmean(xvalues, axis=0), np.nanmean(values, axis=0), color=GROUP_COLOURS[group], linestyle=linestyle, label=f"{group} {'leaf' if role == 'n_a' else 'circle'}")
    axes[1, 3].set(title="Held-out trial support per block", xlabel="trial number", ylabel="scored trials")
    axes[1, 3].legend(fontsize=6.8)

    fig.suptitle(
        f"Graph 4 · REAL DATA · {AREA_LABELS[analysis_spec['area']]} · "
        f"Train 1 {analysis_spec['stage'].replace('_', ' ')}"
    )
    plt.close(fig)
    return fig


@graph.node(outputs="plots")
def make_real_plan_a_plots(
    analysis_spec,
    file_preflight,
    analysis_budget_audit,
    tensor_qc,
    session_curves,
    mouse_statistics,
    group_inference,
    real_leave_one_out,
    real_plot_percentile=98,
):
    percentile = float(real_plot_percentile)
    if not 80 <= percentile <= 100:
        raise ValueError("real_plot_percentile must be between 80 and 100")

    if analysis_spec["execution"]["load_mode"] == "plan_only":
        fig, axes = plt.subplots(2, 4, figsize=(17.0, 7.4), constrained_layout=True)
        groups = (REWARDED, UNREWARDED)
        counts = [sum(row["group"] == group for row in file_preflight) for group in groups]
        bars = axes[0, 0].bar(["Rewarded", "Unrewarded"], counts, color=[GROUP_COLOURS[group] for group in groups])
        axes[0, 0].bar_label(bars)
        axes[0, 0].set(title="Independent mice selected", ylabel="mice")

        for offset, group in enumerate(groups):
            rows = [row for row in file_preflight if row["group"] == group]
            years = [int(row["date"][:4]) for row in rows]
            axes[0, 1].scatter(years, np.full(len(years), offset), color=GROUP_COLOURS[group], s=48, alpha=0.8)
        axes[0, 1].set_yticks([0, 1], ["Rewarded", "Unrewarded"])
        axes[0, 1].set(title="Acquisition year / batch warning", xlabel="year")

        labels = [row["mouse"] for row in file_preflight]
        budgets = [row["bytes"] / 2**30 for row in file_preflight]
        colours = [GROUP_COLOURS[row["group"]] for row in file_preflight]
        axes[0, 2].bar(np.arange(len(labels)), budgets, color=colours)
        axes[0, 2].set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=6.5)
        axes[0, 2].set(title="Per-mouse verified file budget", ylabel="GiB")

        layer_totals = {}
        seen = set()
        for row in file_preflight:
            for item in row["files"]:
                if item["name"] not in seen:
                    seen.add(item["name"])
                    layer_totals[item["layer"]] = layer_totals.get(item["layer"], 0) + item["size_bytes"] / 2**30
        bars = axes[0, 3].bar(list(layer_totals), list(layer_totals.values()), color="#4c78a8")
        axes[0, 3].bar_label(bars, fmt="%.2f", fontsize=7)
        axes[0, 3].tick_params(axis="x", rotation=25)
        axes[0, 3].set(title="Unique storage by data layer", ylabel="GiB")

        axes[1, 0].axis("off")
        spec_lines = [
            analysis_spec["result_label"],
            "",
            f"Estimand: {analysis_spec['representation']['estimand']}",
            f"Area / movement: {AREA_LABELS[analysis_spec['area']]} / {analysis_spec['movement']}",
            f"Contrast: {analysis_spec['contrast']['label']} · {analysis_spec['position_start']:.1f}-{analysis_spec['position_stop']:.1f} m",
            f"Coverage: {analysis_spec['coverage']} · {analysis_spec['position_bins']} position bins",
            f"Time: {analysis_spec['block_trials']}-trial {analysis_spec['stride_mode']} blocks",
            f"Early rule: ≥{analysis_spec['min_early_blocks']} valid blocks through trial {analysis_spec['early_horizon']}",
            f"CV: {analysis_spec['folds']} physical-time folds · min {analysis_spec['min_role']} train trials/role",
        ]
        axes[1, 0].text(0, 1, "\\n".join(spec_lines), va="top", fontsize=8.0)

        axes[1, 1].axis("off")
        deviation_lines = ["REFERENCE SPECIFICATION"]
        if analysis_spec["deviations"]:
            deviation_lines += [
                f"{name}: {value['selected']} (reference {value['reference']})"
                for name, value in analysis_spec["deviations"].items()
            ]
        else:
            deviation_lines += ["No scientific deviations."]
        deviation_lines += ["", "Every changed field is retained in the result package."]
        axes[1, 1].text(0, 1, "\\n".join(deviation_lines), va="top", fontsize=7.7)

        axes[1, 2].axis("off")
        file_lines = ["EXACT FILES (first 10)"]
        file_lines += [item["name"] for row in file_preflight for item in row["files"]][:10]
        if sum(len(row["files"]) for row in file_preflight) > 10:
            file_lines.append("… inspect file_preflight for the complete manifest")
        axes[1, 2].text(0, 1, "\\n".join(file_lines), va="top", family="monospace", fontsize=6.3)

        axes[1, 3].axis("off")
        status = [
            "PREFLIGHT ONLY - NO ARRAYS LOADED",
            "",
            f"Dataset: {'Drive connected' if analysis_spec['cohort_policy']['dataset_connected'] else 'metadata only'}",
            f"Unique files: {analysis_budget_audit['unique_files']}",
            f"Budget: {analysis_budget_audit['total_gib']:.2f} / {analysis_budget_audit['limit_gib']:.2f} GiB",
            f"Budget approved: {analysis_budget_audit['approved']}",
            f"Tensor audit: {tensor_qc['status']}",
            "",
            "Preview mode suppresses inferential output.",
            "Acquisition year is visible because label exchangeability is an assumption.",
            "Use Run to + Inspect port to audit each node before loading.",
        ]
        axes[1, 3].text(0, 1, "\\n".join(status), va="top", fontsize=8.2)
        fig.suptitle("Graph 4 · real Plan A preflight, locked specification, and file audit")
        plt.close(fig)
        return fig

    def finite_column_mean(values):
        values = np.asarray(values, dtype=float)
        count = np.sum(np.isfinite(values), axis=0)
        return np.divide(np.nansum(values, axis=0), count, out=np.full(values.shape[1:], np.nan), where=count > 0)

    fig, axes = plt.subplots(2, 4, figsize=(17.0, 7.8), constrained_layout=True)
    for row in session_curves:
        axes[0, 0].plot(row["curve"]["midpoint"], row["curve"]["dprime"], color=GROUP_COLOURS[row["group"]], alpha=0.48)
    axes[0, 0].axvline(analysis_spec["early_horizon"], color="#f28e2b", linestyle="--", label="early horizon")
    axes[0, 0].set(title="Every mouse and every valid block", xlabel="trial number", ylabel="held-out d′")
    axes[0, 0].legend(fontsize=7)

    for group in (REWARDED, UNREWARDED):
        selected = [row for row in session_curves if row["group"] == group]
        max_blocks = max((len(row["curve"]["dprime"]) for row in selected), default=0)
        values = np.full((len(selected), max_blocks), np.nan)
        xvalues = np.full((len(selected), max_blocks), np.nan)
        for index, row in enumerate(selected):
            n = len(row["curve"]["dprime"])
            values[index, :n] = row["curve"]["dprime"]
            xvalues[index, :n] = row["curve"]["midpoint"]
        if max_blocks:
            axes[0, 1].plot(finite_column_mean(xvalues), finite_column_mean(values), marker="o", color=GROUP_COLOURS[group], label=group)
    axes[0, 1].set(title="Descriptive group mean (mouse weighted)", xlabel="trial number", ylabel="held-out d′")
    axes[0, 1].legend(fontsize=7)

    rng = np.random.default_rng(13)
    for offset, group in enumerate((REWARDED, UNREWARDED)):
        selected = mouse_statistics["slopes"][(mouse_statistics["groups"] == group) & np.isfinite(mouse_statistics["slopes"])]
        axes[0, 2].scatter(offset + rng.normal(0, 0.035, len(selected)), selected, color=GROUP_COLOURS[group], s=38)
        if len(selected):
            axes[0, 2].plot([offset-0.16, offset+0.16], [np.mean(selected)]*2, color="black", linewidth=2)
    axes[0, 2].axhline(0, color="#999", linewidth=1)
    axes[0, 2].set_xticks([0, 1], ["Rewarded", "Unrewarded"])
    axes[0, 2].set(title="Primary estimand: one slope per mouse", ylabel="Δd′ per trial")

    axes[0, 3].axis("off")
    inference_lines = [analysis_spec["result_label"], ""]
    if group_inference.get("available"):
        boot, perm = group_inference["bootstrap"], group_inference["permutation"]
        inference_lines += [
            f"Rewarded - unrewarded: {boot['difference']:.5f} d′/trial",
            f"Mouse-bootstrap 95% interval: [{boot['ci'][0]:.5f}, {boot['ci'][1]:.5f}]",
            f"Exact {analysis_spec['execution']['alternative']} p: {perm['pvalue']:.4f}",
            f"Label allocations: {int(perm['permutations'])}",
            f"Valid mice: {group_inference['valid_counts']}",
            f"Exploratory override: {group_inference['exploratory_override']}",
        ]
    else:
        inference_lines += ["INFERENCE NOT SHOWN", group_inference.get("reason", "not available")]
    inference_lines += ["", "Effect and mouse interval come before the p-value.", "Cohort-label exchangeability is assumed, not guaranteed."]
    axes[0, 3].text(0, 1, "\\n".join(inference_lines), va="top", fontsize=8.3)

    for group in (REWARDED, UNREWARDED):
        selected = [row for row in session_curves if row["group"] == group]
        max_blocks = max((len(row["curve"]["dprime"]) for row in selected), default=0)
        if not max_blocks:
            continue
        for metric, linestyle in (("separation", "-"), ("spread", "--")):
            values = np.full((len(selected), max_blocks), np.nan)
            xvalues = np.full((len(selected), max_blocks), np.nan)
            for index, row in enumerate(selected):
                n = len(row["curve"][metric])
                values[index, :n] = row["curve"][metric]
                xvalues[index, :n] = row["curve"]["midpoint"]
            axes[1, 0].plot(finite_column_mean(xvalues), finite_column_mean(values), color=GROUP_COLOURS[group], linestyle=linestyle, label=f"{group} {metric}")
    axes[1, 0].set(title="Why d′ changed: separation vs spread", xlabel="trial number", ylabel="held-out score units")
    axes[1, 0].legend(fontsize=6.3)

    temporal_rows = [row for row in session_curves if row["cross_temporal"]["dprime"].size]
    if temporal_rows and {row["group"] for row in temporal_rows} == {REWARDED, UNREWARDED}:
        size = min(row["cross_temporal"]["dprime"].shape[0] for row in temporal_rows)
        matrices = {}
        for group in (REWARDED, UNREWARDED):
            stack = np.stack([row["cross_temporal"]["dprime"][:size, :size] for row in temporal_rows if row["group"] == group])
            matrices[group] = finite_column_mean(stack)
        difference = matrices[REWARDED] - matrices[UNREWARDED]
        finite = np.abs(difference[np.isfinite(difference)])
        limit = max(float(np.percentile(finite, percentile)) if len(finite) else 0.0, 0.1)
        image = axes[1, 1].imshow(difference, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        fig.colorbar(image, ax=axes[1, 1], label="rewarded - unrewarded d′")
    else:
        axes[1, 1].text(0.5, 0.5, "Cross-temporal diagnostic disabled or unavailable", ha="center", va="center", transform=axes[1, 1].transAxes)
    axes[1, 1].set(title="Coding-axis generalization", xlabel="test block", ylabel="train block")

    for group in (REWARDED, UNREWARDED):
        selected = [row for row in session_curves if row["group"] == group]
        max_blocks = max((len(row["speed_blocks"]) for row in selected), default=0)
        if max_blocks:
            values = np.full((len(selected), max_blocks), np.nan)
            for index, row in enumerate(selected):
                values[index, :len(row["speed_blocks"])] = row["speed_blocks"]
            x = (np.arange(max_blocks)+0.5) * analysis_spec["time"]["stride_trials"]
            axes[1, 2].plot(x, finite_column_mean(values), marker="o", color=GROUP_COLOURS[group], label=group)
    axes[1, 2].set(title="Behavioral control: speed", xlabel="trial scale", ylabel="mean speed in chosen window")
    axes[1, 2].legend(fontsize=7)

    if real_leave_one_out:
        differences = [row["difference"] for row in real_leave_one_out]
        colours = [GROUP_COLOURS[row["omitted_group"]] for row in real_leave_one_out]
        axes[1, 3].scatter(differences, np.arange(len(differences)), c=colours, s=24)
        axes[1, 3].axvline(0, color="#e15759", linestyle="--")
        axes[1, 3].set_yticks(np.arange(len(differences)), [row["omitted_mouse"] for row in real_leave_one_out], fontsize=6)
    else:
        axes[1, 3].text(0.5, 0.5, "Leave-one-out follows an allowed group inference", ha="center", va="center", transform=axes[1, 3].transAxes)
    axes[1, 3].set(title="Leave one mouse out", xlabel="remaining slope difference")

    fig.suptitle(
        f"Graph 4 · REAL DATA · {analysis_spec['result_label']} · "
        f"{AREA_LABELS[analysis_spec['area']]} · {analysis_spec['contrast']['label']}"
    )
    plt.close(fig)
    return fig
"""
            ),
            py(
                """
#@title Plan first; then load the selected cohort in Colab { display-mode: "form" }
hypothesis_graph = graph.Graph(
    "Real Plan A: reward and within-session d-prime dynamics",
    connect_real_analysis_release,
    define_real_cohort_policy,
    define_neural_representation,
    define_real_stimulus_and_corridor_contrast,
    define_time_and_estimator_rules,
    define_real_execution_policy,
    enumerate_real_eligible_sessions,
    select_real_sessions,
    assemble_analysis_spec,
    resolve_real_analysis_files,
    audit_real_analysis_budget,
    load_verified_trial_tensors_expanded,
    audit_loaded_session_tensors,
    construct_real_position_masks,
    estimate_real_blockwise_dprime,
    fit_real_early_rates,
    fit_real_saturation_models,
    compute_real_cross_temporal_diagnostics,
    compute_real_position_diagnostics,
    summarize_real_behavior_and_support,
    assemble_real_session_results,
    summarize_real_mouse_statistics_expanded,
    infer_real_reward_rate_difference_expanded,
    leave_one_mouse_out_real_group,
    make_real_plan_a_plots,
)
hypothesis_panel = hypothesis_graph.widget(
    controls={
        "real_training_set": widgets.Dropdown(
            description="Training set",
            options=[("Train 1", "train1"), ("Train 2", "train2")],
            value="train1",
        ),
        "training_stage": widgets.Dropdown(
            description="Training stage",
            options=[("Before learning", "before_learning"), ("After learning (mixed reward modes)", "after_learning")],
            value="before_learning",
        ),
        "cohort_size": widgets.Dropdown(
            description="Cohort size",
            options=[("Two per group preview", 2), ("All eligible mice", 0), ("One per group smoke test", 1), ("Three per group preview", 3)],
            value=2,
        ),
        "cohort_sampling": widgets.Dropdown(
            description="Preview sample",
            options=[("Spread across dates", "spread_dates"), ("First catalog rows", "first_catalog"), ("Latest dates", "latest")],
            value="spread_dates",
        ),
        "representation_mode": widgets.Dropdown(
            description="Neural estimand",
            options=[("SVD population d′ (implemented)", "svd_population"), ("Full-neuron replication (plan only)", "full_neuron_replication")],
            value="svd_population",
        ),
        "cortical_area": widgets.Dropdown(
            description="Cortical area",
            options=[("V1", "V1"), ("Medial HVA", "mHV"), ("Lateral HVA", "lHV"), ("Anterior HVA", "aHV")],
            value="V1",
        ),
        "feature_count": widgets.Dropdown(description="Area features", options=[8, 12, 24], value=12),
        "movement_rule": widgets.Dropdown(
            description="Movement rule",
            options=[("Moving frames only", "moving_only"), ("All valid frames", "all_valid_frames")],
            value="moving_only",
        ),
        "stimulus_pair": widgets.Dropdown(
            description="Stimulus pair",
            options=[("Leaf 1 vs circle 1", "leaf1_vs_circle1"), ("Leaf 2 vs circle 2", "leaf2_vs_circle2")],
            value="leaf1_vs_circle1",
        ),
        "corridor_region": widgets.Dropdown(
            description="Corridor window",
            options=[("Texture 0-4 m", "texture_0_4"), ("Full corridor 0-6 m", "full_corridor_0_6"), ("Custom bounds", "custom")],
            value="texture_0_4",
        ),
        "custom_position_start_m": widgets.FloatSlider(description="Custom start m", value=0.0, min=0.0, max=5.5, step=0.5),
        "custom_position_end_m": widgets.FloatSlider(description="Custom end m", value=4.0, min=0.5, max=6.0, step=0.5),
        "position_bin_count": widgets.Dropdown(description="Position bins", options=[12, 18, 24], value=18),
        "real_coverage_rule": widgets.Dropdown(
            description="Coverage rule",
            options=[("Complete window", "complete"), ("Available bins", "available")],
            value="complete",
        ),
        "trial_block_width": widgets.Dropdown(description="Block width", options=[32, 40, 48, 60], value=40),
        "block_stride_mode": widgets.Dropdown(
            description="Block stride",
            options=[("Non-overlapping", "nonoverlap"), ("Half-block overlap", "half_block_overlap")],
            value="nonoverlap",
        ),
        "early_trial_horizon": widgets.Dropdown(description="Early horizon", options=[120, 140, 180, 220], value=140),
        "real_crossvalidation_folds": widgets.Dropdown(description="CV folds", options=[2, 4], value=4),
        "real_minimum_trials_per_role": widgets.Dropdown(description="Min train/role", options=[3, 4, 5], value=4),
        "minimum_valid_early_blocks": widgets.Dropdown(description="Min early blocks", options=[2, 3, 4], value=3),
        "claim_status": widgets.Dropdown(
            description="Claim label",
            options=[("Pipeline preview (default)", "pipeline_preview"), ("Complete eligible cohort", "complete_cohort"), ("Sensitivity / exploratory", "sensitivity")],
            value="pipeline_preview",
        ),
        "load_mode": widgets.Dropdown(
            description="Data action",
            options=[("Plan only (no download)", "plan_only"), ("Load and analyse real data", "load_and_analyse")],
            value="plan_only",
        ),
        "compute_cross_temporal": widgets.Checkbox(description="Cross-temporal diagnostic", value=True),
        "compute_position_surface": widgets.Checkbox(description="Position surface for Plan B", value=True),
        "real_inference_alternative": widgets.Dropdown(
            description="Inference alternative",
            options=[("Two-sided", "two-sided"), ("Rewarded > unrewarded", "greater")],
            value="two-sided",
        ),
        "real_bootstrap_draws": widgets.Dropdown(description="Bootstrap draws", options=[1000, 4000, 10000], value=4000),
        "allow_exploratory_inference": widgets.Checkbox(description="Show preview inference", value=False),
        "maximum_analysis_gib": widgets.FloatSlider(description="Download limit GiB", value=4.0, min=1.0, max=80.0, step=1.0),
        "real_plot_percentile": widgets.IntSlider(description="Colour percentile", value=98, min=90, max=100, step=1),
    },
    show="plots",
)
hypothesis_panel
"""
            ),
            md(
                """
### Required sensitivity runs

After one complete-cohort run, use `hypothesis_graph.run_many(...)` for a
small, declared grid and report every result in that grid:
"""
            ),
            py(
                """
#@title Sensitivity grid template (does not run automatically) { display-mode: "form" }
sensitivity_settings = [
    {
        "real_training_set": "train1",
        "training_stage": "before_learning",
        "cohort_sampling": "spread_dates",
        "cortical_area": area,
        "cohort_size": 0,
        "representation_mode": "svd_population",
        "trial_block_width": block,
        "early_trial_horizon": horizon,
        "feature_count": 12,
        "claim_status": "sensitivity",
        "allow_exploratory_inference": True,
        "load_mode": "load_and_analyse",
    }
    for area in ("V1", "mHV", "lHV", "aHV")
    for block in (32, 40, 60)
    for horizon in (140, 180, 220)
]
print(f"{len(sensitivity_settings)} declared runs; files are cached after the first verified copy.")
print("Run deliberately with: sensitivity_runs = hypothesis_graph.run_many(sensitivity_settings)")
"""
            ),
        ]
    )
    cells.extend(
        [
            md(
                """
## Graph 5 - Position and support diagnostics

This graph displays position-resolved held-out scores together with measured
frame support, speed, cue, reward, and lick locations. It does not assign a
mechanism to a within-session change.

The default source is the bundled real unrewarded recording, so the graph runs
without a large download. After Graph 4 has completed a real cohort run, choose
**Reuse last real time-block analysis run**. This passes the documented
`hypothesis_panel.last_run` outputs into Graph 5 and does not reload Drive data.

The graph now computes one early-to-late position-change profile per mouse,
then the rewarded-minus-unrewarded difference-in-change and a mouse bootstrap
band that controls the whole displayed position curve. It also averages cue,
reward and first-lick histograms **within mouse before group averaging**, so a
long session cannot dominate the event plot. With only the bundled one-mouse
demo, inferential nodes remain explicitly unavailable. Do not replace this
with 18 independent position-wise t-tests.
"""
            ),
            py(
                """
#@title Graph 5 transformations { display-mode: "form" }
@graph.node(outputs="position_source")
def choose_position_analysis_source_expanded(
    position_source_mode="bundled_demo",
    position_area="V1",
    position_block_width=80,
    position_cv_folds=4,
    position_min_trials_per_role=4,
    position_stimulus_pair="leaf1_vs_circle1",
):
    if position_source_mode == "last_real_plan_a":
        last = getattr(hypothesis_panel, "last_run", None)
        if last is None or not last.get("session_curves"):
            raise ValueError(
                "Run Graph 4 with Load and analyse first, then choose Reuse last real Plan A run."
            )
        usable = [row for row in last["session_curves"] if row["position_surface"]["dprime"].size]
        if not usable:
            raise ValueError("The last Plan A run did not compute position surfaces.")
        return {
            "kind": "real_group", "rows": usable,
            "area": last["analysis_spec"]["area"],
            "block_trials": last["analysis_spec"]["block_trials"],
            "label": "last completed real Plan A cohort run",
            "contrast": last["analysis_spec"]["contrast"]["label"],
        }
    if position_source_mode != "bundled_demo":
        raise ValueError("unknown position_source_mode")
    if position_area not in AREA_IDS:
        raise ValueError(f"position_area must be one of {list(AREA_IDS)}")
    if position_stimulus_pair not in STIMULUS_PAIRS:
        raise ValueError("unknown position_stimulus_pair")
    role_a, role_b, contrast_label = STIMULUS_PAIRS[position_stimulus_pair]
    demo = load_atlas_demo()
    area_names = [str(name) for name in demo["area_name"]]
    features = np.asarray(demo["area_features"][area_names.index(position_area)], dtype=float)
    surface = position_dprime_surface(
        features, demo["stimulus_id"], demo["trial_id"],
        role_a=role_a, role_b=role_b,
        block_trials=int(position_block_width),
        n_folds=int(position_cv_folds),
        min_per_role=int(position_min_trials_per_role),
    )
    row = {
        "mouse": "TX119", "recording_id": demo["metadata"]["session"],
        "group": UNREWARDED, "position_surface": surface,
        "position": np.asarray(demo["position_centers_m"]),
        "mean_speed_by_position": np.nanmean(demo["mean_run_speed"], axis=0),
        "mean_support_by_position": np.mean(demo["frame_counts"], axis=0),
        "cue_position_m": np.array([]), "reward_position_m": np.array([]),
        "first_lick_position_m": np.array([]),
    }
    return {
        "kind": "bundled_demo", "rows": [row], "area": position_area,
        "block_trials": int(position_block_width),
        "label": "TX119 unsup_test1 · real compact mechanics example",
        "contrast": contrast_label,
    }


@graph.node(outputs="position_grid")
def define_common_corridor_grid(
    position_source,
    corridor_plot_start_m=0.0,
    corridor_plot_end_m=6.0,
    texture_boundary_m=4.0,
    position_smoothing_bins=1,
):
    start, stop, boundary = map(float, (corridor_plot_start_m, corridor_plot_end_m, texture_boundary_m))
    if not 0 <= start < stop <= 6:
        raise ValueError("corridor plot bounds must satisfy 0 <= start < end <= 6")
    if not start < boundary < stop:
        raise ValueError("texture_boundary_m must lie inside the displayed corridor")
    if int(position_smoothing_bins) not in {1, 3, 5}:
        raise ValueError("position_smoothing_bins must be 1, 3, or 5")
    return {
        "start": start, "stop": stop, "texture_boundary": boundary,
        "smoothing_bins": int(position_smoothing_bins),
        "source_label": position_source["label"],
    }


@graph.node(outputs="position_window")
def define_position_time_windows(
    position_source,
    early_block_count=2,
    late_block_count=2,
    minimum_common_time_blocks=4,
):
    early, late, minimum = map(int, (early_block_count, late_block_count, minimum_common_time_blocks))
    if min(early, late, minimum) < 1:
        raise ValueError("early, late, and minimum common block counts must be positive")
    return {
        "early": early, "late": late,
        "minimum_common_requested": minimum,
        "minimum_common": max(minimum, early + late),
        "block_trials": position_source["block_trials"],
    }


@graph.node(outputs="event_spec")
def define_position_event_display(
    position_grid,
    show_cue_events=True,
    show_reward_events=True,
    show_lick_events=True,
    event_histogram_bins=24,
    minimum_events_per_mouse=3,
):
    return {
        "show": {
            "cue": bool(show_cue_events),
            "reward": bool(show_reward_events),
            "first lick": bool(show_lick_events),
        },
        "bins": int(event_histogram_bins),
        "minimum_events": int(minimum_events_per_mouse),
        "range": (position_grid["start"], position_grid["stop"]),
    }


@graph.node(outputs="position_source")
def choose_position_analysis_source(
    position_source_mode="bundled_demo",
    position_area="V1",
    position_block_width=80,
):
    if position_source_mode == "last_real_plan_a":
        last = getattr(hypothesis_panel, "last_run", None)
        if last is None or not last["session_curves"]:
            raise ValueError(
                "Run Graph 4 with Load and analyse first, then choose "
                "Reuse last real Plan A run."
            )
        return {
            "kind": "real_group",
            "rows": last["session_curves"],
            "area": last["analysis_spec"]["area"],
            "block_trials": last["analysis_spec"]["block_trials"],
            "label": "last completed real Plan A cohort run",
        }
    if position_source_mode != "bundled_demo":
        raise ValueError("position_source_mode must be bundled_demo or last_real_plan_a")
    if position_area not in AREA_IDS:
        raise ValueError(f"position_area must be one of {list(AREA_IDS)}")
    demo = load_atlas_demo()
    area_names = [str(name) for name in demo["area_name"]]
    area_index = area_names.index(position_area)
    features = np.asarray(demo["area_features"][area_index], dtype=float)
    surface = position_dprime_surface(
        features,
        demo["stimulus_id"],
        demo["trial_id"],
        role_a=2,
        role_b=0,
        block_trials=int(position_block_width),
        n_folds=4,
        min_per_role=4,
    )
    row = {
        "mouse": "TX119",
        "recording_id": demo["metadata"]["session"],
        "group": UNREWARDED,
        "position_surface": surface,
        "position": np.asarray(demo["position_centers_m"]),
        "mean_speed_by_position": np.nanmean(demo["mean_run_speed"], axis=0),
        "mean_support_by_position": np.mean(demo["frame_counts"], axis=0),
        "cue_position_m": np.array([]),
        "reward_position_m": np.array([]),
        "first_lick_position_m": np.array([]),
    }
    return {
        "kind": "bundled_demo",
        "rows": [row],
        "area": position_area,
        "block_trials": int(position_block_width),
        "label": "TX119 unsup_test1 · real compact mechanics example",
    }


@graph.node(outputs="position_window")
def choose_early_and_late_blocks(position_source, early_block_count=2, late_block_count=2):
    early = int(early_block_count)
    late = int(late_block_count)
    if early < 1 or late < 1:
        raise ValueError("early and late block counts must be positive")
    return {"early": early, "late": late}


@graph.node(outputs="corridor_summary")
def summarize_position_dynamics(position_source, position_window):
    rows = position_source["rows"]
    position = np.asarray(rows[0]["position"], dtype=float)
    groups_present = [group for group in (REWARDED, UNREWARDED) if any(row["group"] == group for row in rows)]
    by_group = {}
    for group in groups_present:
        selected = [row for row in rows if row["group"] == group]
        common_blocks = min(row["position_surface"]["dprime"].shape[0] for row in selected)
        surfaces = np.stack([row["position_surface"]["dprime"][:common_blocks] for row in selected])
        midpoints = np.stack([row["position_surface"]["midpoint"][:common_blocks] for row in selected])
        early_n = min(position_window["early"], common_blocks)
        late_n = min(position_window["late"], common_blocks)
        cues = np.concatenate([np.asarray(row["cue_position_m"], dtype=float) for row in selected])
        rewards = np.concatenate([np.asarray(row["reward_position_m"], dtype=float) for row in selected])
        licks = np.concatenate([np.asarray(row["first_lick_position_m"], dtype=float) for row in selected])
        by_group[group] = {
            "surface": np.nanmean(surfaces, axis=0),
            "surface_mouse_values": surfaces,
            "midpoint": np.nanmean(midpoints, axis=0),
            "early_profile": np.nanmean(surfaces[:, :early_n], axis=(0, 1)),
            "late_profile": np.nanmean(surfaces[:, -late_n:], axis=(0, 1)),
            "speed": np.nanmean(np.stack([row["mean_speed_by_position"] for row in selected]), axis=0),
            "support": np.nanmean(np.stack([row["mean_support_by_position"] for row in selected]), axis=0),
            "cue": cues[np.isfinite(cues)],
            "reward": rewards[np.isfinite(rewards)],
            "lick": licks[np.isfinite(licks)],
            "n_mice": len(selected),
        }
    difference = None
    if set(by_group) == {REWARDED, UNREWARDED}:
        common = min(len(by_group[REWARDED]["surface"]), len(by_group[UNREWARDED]["surface"]))
        difference = by_group[REWARDED]["surface"][:common] - by_group[UNREWARDED]["surface"][:common]
    return {
        "kind": position_source["kind"],
        "label": position_source["label"],
        "area": position_source["area"],
        "block_trials": position_source["block_trials"],
        "position": position,
        "by_group": by_group,
        "difference": difference,
        "window": position_window,
    }


@graph.node(outputs="mouse_position_surfaces")
def collect_mouse_position_surfaces(position_source, position_grid):
    def smooth_matrix(values, width):
        values = np.asarray(values, dtype=float)
        if width == 1:
            return values.copy()
        output = np.full_like(values, np.nan, dtype=float)
        radius = width // 2
        for index in range(values.shape[-1]):
            chunk = values[..., max(0, index-radius):min(values.shape[-1], index+radius+1)]
            count = np.sum(np.isfinite(chunk), axis=-1)
            output[..., index] = np.divide(
                np.nansum(chunk, axis=-1), count,
                out=np.full(count.shape, np.nan, dtype=float), where=count > 0,
            )
        return output

    rows = []
    common_position = None
    for source_row in position_source["rows"]:
        position = np.asarray(source_row["position"], dtype=float)
        mask = (position >= position_grid["start"]) & (position <= position_grid["stop"])
        selected_position = position[mask]
        if common_position is None:
            common_position = selected_position
        elif not np.allclose(common_position, selected_position):
            raise ValueError("all mice must share the same physical position grid")
        rows.append({
            **source_row,
            "position": selected_position,
            "surface": smooth_matrix(
                np.asarray(source_row["position_surface"]["dprime"], dtype=float)[:, mask],
                position_grid["smoothing_bins"],
            ),
            "midpoint": np.asarray(source_row["position_surface"]["midpoint"], dtype=float),
            "speed": smooth_matrix(np.asarray(source_row["mean_speed_by_position"], dtype=float)[None, mask], position_grid["smoothing_bins"])[0],
            "support": smooth_matrix(np.asarray(source_row["mean_support_by_position"], dtype=float)[None, mask], position_grid["smoothing_bins"])[0],
        })
    return {
        "kind": position_source["kind"], "label": position_source["label"],
        "area": position_source["area"], "contrast": position_source["contrast"],
        "position": np.asarray([] if common_position is None else common_position),
        "rows": rows,
    }


@graph.node(outputs="position_support_audit")
def audit_mouse_position_support(
    mouse_position_surfaces,
    position_window,
    minimum_mouse_position_coverage=0.60,
):
    threshold = float(minimum_mouse_position_coverage)
    if not 0 < threshold <= 1:
        raise ValueError("minimum_mouse_position_coverage must be in (0, 1]")
    rows = []
    for row in mouse_position_surfaces["rows"]:
        finite_fraction = float(np.isfinite(row["surface"]).mean()) if row["surface"].size else 0.0
        enough_blocks = row["surface"].shape[0] >= position_window["minimum_common"]
        rows.append({
            "mouse": row["mouse"], "group": row["group"],
            "finite_fraction": finite_fraction,
            "time_blocks": int(row["surface"].shape[0]),
            "included": bool(finite_fraction >= threshold and enough_blocks),
        })
    included_mice = {row["mouse"] for row in rows if row["included"]}
    return {
        "rows": rows,
        "included_rows": [row for row in mouse_position_surfaces["rows"] if row["mouse"] in included_mice],
        "threshold": threshold,
    }


@graph.node(outputs="mouse_position_changes")
def compute_mouse_early_late_position_changes(position_support_audit, position_window):
    def finite_mean(values, axis):
        count = np.sum(np.isfinite(values), axis=axis)
        return np.divide(np.nansum(values, axis=axis), count, out=np.full(count.shape, np.nan), where=count > 0)

    rows = []
    for row in position_support_audit["included_rows"]:
        surface = row["surface"]
        early = finite_mean(surface[:position_window["early"]], axis=0)
        late = finite_mean(surface[-position_window["late"]:], axis=0)
        rows.append({
            "mouse": row["mouse"], "group": row["group"],
            "early": early, "late": late, "change": late - early,
            "effective_early": position_window["early"],
            "effective_late": position_window["late"],
        })
    return rows


@graph.node(outputs="group_position_surfaces")
def aggregate_mouse_weighted_position_surfaces(mouse_position_surfaces, position_support_audit, mouse_position_changes):
    def finite_mean(values):
        values = np.asarray(values, dtype=float)
        count = np.sum(np.isfinite(values), axis=0)
        return np.divide(np.nansum(values, axis=0), count, out=np.full(values.shape[1:], np.nan), where=count > 0)

    included = position_support_audit["included_rows"]
    by_group = {}
    for group in (REWARDED, UNREWARDED):
        surface_rows = [row for row in included if row["group"] == group]
        change_rows = [row for row in mouse_position_changes if row["group"] == group]
        if not surface_rows:
            continue
        common = min(row["surface"].shape[0] for row in surface_rows)
        surfaces = np.stack([row["surface"][:common] for row in surface_rows])
        changes = np.stack([row["change"] for row in change_rows])
        by_group[group] = {
            "surface": finite_mean(surfaces),
            "midpoint": finite_mean(np.stack([row["midpoint"][:common] for row in surface_rows])),
            "change": finite_mean(changes),
            "mouse_changes": changes,
            "n_mice": len(surface_rows),
        }
    return {
        "position": mouse_position_surfaces["position"],
        "by_group": by_group,
        "label": mouse_position_surfaces["label"],
        "area": mouse_position_surfaces["area"],
        "contrast": mouse_position_surfaces["contrast"],
    }


@graph.node(outputs="position_group_contrast")
def estimate_position_difference_in_change(group_position_surfaces):
    by_group = group_position_surfaces["by_group"]
    if set(by_group) != {REWARDED, UNREWARDED}:
        return {"available": False, "reason": "both groups require a completed real Plan A run"}
    common = min(by_group[group]["surface"].shape[0] for group in (REWARDED, UNREWARDED))
    return {
        "available": True,
        "surface_difference": by_group[REWARDED]["surface"][:common] - by_group[UNREWARDED]["surface"][:common],
        "change_difference": by_group[REWARDED]["change"] - by_group[UNREWARDED]["change"],
        "rewarded_mouse_changes": by_group[REWARDED]["mouse_changes"],
        "unrewarded_mouse_changes": by_group[UNREWARDED]["mouse_changes"],
    }


@graph.node(outputs="simultaneous_position_inference")
def bootstrap_simultaneous_position_band(
    position_group_contrast,
    position_bootstrap_draws=2000,
    simultaneous_confidence_level=0.95,
    position_bootstrap_seed=31,
):
    if not position_group_contrast["available"]:
        return {"available": False, "reason": position_group_contrast["reason"]}
    rewarded = position_group_contrast["rewarded_mouse_changes"]
    unrewarded = position_group_contrast["unrewarded_mouse_changes"]
    if min(len(rewarded), len(unrewarded)) < 2:
        return {"available": False, "reason": "simultaneous uncertainty needs at least two mice per group"}
    draws = int(position_bootstrap_draws)
    confidence = float(simultaneous_confidence_level)
    rng = np.random.default_rng(int(position_bootstrap_seed))
    samples = np.full((draws, rewarded.shape[1]), np.nan)
    def finite_mean(values):
        count = np.sum(np.isfinite(values), axis=0)
        return np.divide(
            np.nansum(values, axis=0), count,
            out=np.full(values.shape[1], np.nan), where=count > 0,
        )
    for index in range(draws):
        r = rewarded[rng.integers(0, len(rewarded), len(rewarded))]
        u = unrewarded[rng.integers(0, len(unrewarded), len(unrewarded))]
        samples[index] = finite_mean(r) - finite_mean(u)
    observed = position_group_contrast["change_difference"]
    valid = np.isfinite(observed) & np.all(np.isfinite(samples), axis=0)
    if not np.any(valid):
        return {"available": False, "reason": "no position bin has complete mouse-bootstrap support"}
    pointwise = np.quantile(samples[:, valid], [(1-confidence)/2, 1-(1-confidence)/2], axis=0)
    max_deviation = np.max(np.abs(samples[:, valid] - observed[valid]), axis=1)
    critical = float(np.quantile(max_deviation, confidence))
    lower = np.full_like(observed, np.nan)
    upper = np.full_like(observed, np.nan)
    point_lower = np.full_like(observed, np.nan)
    point_upper = np.full_like(observed, np.nan)
    lower[valid], upper[valid] = observed[valid] - critical, observed[valid] + critical
    point_lower[valid], point_upper[valid] = pointwise[0], pointwise[1]
    return {
        "available": True, "observed": observed,
        "simultaneous_lower": lower, "simultaneous_upper": upper,
        "pointwise_lower": point_lower, "pointwise_upper": point_upper,
        "critical_max_deviation": critical, "confidence": confidence,
        "draws": draws, "valid_position": valid,
    }


@graph.node(outputs="behavior_position_profiles")
def summarize_mouse_weighted_position_behavior(mouse_position_surfaces, position_support_audit):
    def finite_mean(values):
        values = np.asarray(values, dtype=float)
        count = np.sum(np.isfinite(values), axis=0)
        return np.divide(np.nansum(values, axis=0), count, out=np.full(values.shape[1:], np.nan), where=count > 0)
    by_group = {}
    for group in (REWARDED, UNREWARDED):
        rows = [row for row in position_support_audit["included_rows"] if row["group"] == group]
        if rows:
            by_group[group] = {
                "speed": finite_mean(np.stack([row["speed"] for row in rows])),
                "support": finite_mean(np.stack([row["support"] for row in rows])),
                "n_mice": len(rows),
            }
    return {"position": mouse_position_surfaces["position"], "by_group": by_group}


@graph.node(outputs="mouse_event_profiles")
def summarize_mouse_level_event_positions(position_support_audit, event_spec):
    event_keys = {
        "cue": "cue_position_m",
        "reward": "reward_position_m",
        "first lick": "first_lick_position_m",
    }
    edges = np.linspace(event_spec["range"][0], event_spec["range"][1], event_spec["bins"] + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    by_group = {}
    for group in (REWARDED, UNREWARDED):
        rows = [row for row in position_support_audit["included_rows"] if row["group"] == group]
        group_events = {}
        for name, key in event_keys.items():
            if not event_spec["show"][name]:
                continue
            mouse_histograms = []
            for row in rows:
                values = np.asarray(row[key], dtype=float)
                values = values[np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])]
                if len(values) >= event_spec["minimum_events"]:
                    hist, _ = np.histogram(values, bins=edges, density=True)
                    mouse_histograms.append(hist)
            if mouse_histograms:
                group_events[name] = np.mean(np.stack(mouse_histograms), axis=0)
        if group_events:
            by_group[group] = group_events
    return {"centers": centers, "by_group": by_group, "mouse_weighted": True}


@graph.node(outputs="position_region_contrasts")
def summarize_texture_and_gray_changes(mouse_position_surfaces, mouse_position_changes, position_grid):
    position = mouse_position_surfaces["position"]
    texture = position < position_grid["texture_boundary"]
    gray = ~texture
    def finite_scalar(values):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(np.mean(finite)) if len(finite) else float("nan")
    rows = []
    for row in mouse_position_changes:
        rows.append({
            "mouse": row["mouse"], "group": row["group"],
            "texture_change": finite_scalar(row["change"][texture]),
            "gray_change": finite_scalar(row["change"][gray]),
        })
    return rows


@graph.node(outputs="legacy_plots")
def plot_corridor_dynamics(corridor_summary):
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.5), constrained_layout=True)
    position = corridor_summary["position"]
    by_group = corridor_summary["by_group"]

    available_surfaces = [summary["surface"] for summary in by_group.values()]
    limit = max((float(np.nanpercentile(np.abs(surface), 95)) for surface in available_surfaces), default=1.0)
    limit = max(limit, 0.1)
    for axis, group, title in (
        (axes[0, 0], REWARDED, "Rewarded d′ over trial x position"),
        (axes[0, 1], UNREWARDED, "Unrewarded d′ over trial x position"),
    ):
        if group not in by_group:
            axis.axis("off")
            axis.text(0.5, 0.5, f"{title}\\nnot available in this source", ha="center", va="center")
            continue
        summary = by_group[group]
        image = axis.imshow(
            summary["surface"], origin="lower", aspect="auto", cmap="coolwarm",
            vmin=-limit, vmax=limit,
            extent=[position[0], position[-1], summary["midpoint"][0], summary["midpoint"][-1]],
        )
        axis.axvline(4, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"{title} · n={summary['n_mice']} mouse(s)", xlabel="corridor position (m)", ylabel="trial-block midpoint")
        fig.colorbar(image, ax=axis, label="held-out d′")

    difference = corridor_summary["difference"]
    if difference is None:
        axes[0, 2].axis("off")
        axes[0, 2].text(
            0.5, 0.5,
            "Group-difference surface requires\\na completed real Plan A cohort run.",
            ha="center", va="center",
        )
    else:
        diff_limit = max(float(np.nanpercentile(np.abs(difference), 95)), 0.1)
        image = axes[0, 2].imshow(
            difference, origin="lower", aspect="auto", cmap="coolwarm",
            vmin=-diff_limit, vmax=diff_limit,
            extent=[position[0], position[-1], 1, difference.shape[0]],
        )
        axes[0, 2].axvline(4, color="black", linestyle="--", linewidth=1)
        axes[0, 2].set(title="Rewarded - unrewarded surface", xlabel="corridor position (m)", ylabel="common time block")
        fig.colorbar(image, ax=axes[0, 2], label="Δ held-out d′")

    for group, summary in by_group.items():
        colour = GROUP_COLOURS[group]
        axes[1, 0].plot(position, summary["early_profile"], color=colour, label=f"{group} early")
        axes[1, 0].plot(position, summary["late_profile"], color=colour, linestyle="--", label=f"{group} late")
    axes[1, 0].axvline(4, color="#777", linestyle="--")
    axes[1, 0].axhline(0, color="#aaa", linewidth=1)
    axes[1, 0].set(title="Early vs late position profiles", xlabel="corridor position (m)", ylabel="held-out d′")
    axes[1, 0].legend(fontsize=7)

    support_axis = axes[1, 1].twinx()
    for group, summary in by_group.items():
        colour = GROUP_COLOURS[group]
        axes[1, 1].plot(position, summary["speed"], color=colour, label=f"{group} speed")
        support_axis.plot(position, summary["support"], color=colour, linestyle=":", alpha=0.7, label=f"{group} frames")
    axes[1, 1].axvline(4, color="#777", linestyle="--")
    axes[1, 1].set(title="Speed (solid) and sampling support (dotted)", xlabel="corridor position (m)", ylabel="speed")
    support_axis.set_ylabel("frames per position bin")

    axes[1, 2].set(title="Cue, reward, and first-lick locations", xlabel="corridor position (m)", ylabel="density")
    bins = np.linspace(0, 6, 25)
    drew_event = False
    for group, summary in by_group.items():
        colour = GROUP_COLOURS[group]
        for name, values, linestyle in (
            ("cue", summary["cue"], "-"),
            ("reward", summary["reward"], "--"),
            ("first lick", summary["lick"], ":"),
        ):
            if len(values):
                hist, edges = np.histogram(values, bins=bins, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                axes[1, 2].plot(centers, hist, color=colour, linestyle=linestyle, label=f"{group} {name}")
                drew_event = True
    axes[1, 2].axvline(4, color="#777", linestyle="--")
    if drew_event:
        axes[1, 2].legend(fontsize=6.3)
    else:
        axes[1, 2].text(0.5, 0.5, "Event positions are not included\\nin the compact derivative.", ha="center", va="center", transform=axes[1, 2].transAxes)

    fig.suptitle(
        f"Graph 5 · {corridor_summary['label']} · {AREA_LABELS[corridor_summary['area']]} · "
        f"texture 0-4 m | grey 4-6 m"
    )
    plt.close(fig)
    return fig


@graph.node(outputs="plots")
def make_plan_b_plots(
    mouse_position_surfaces,
    position_window,
    position_support_audit,
    mouse_position_changes,
    group_position_surfaces,
    position_group_contrast,
    simultaneous_position_inference,
    behavior_position_profiles,
    mouse_event_profiles,
    position_region_contrasts,
    position_grid,
    position_plot_percentile=97,
    show_individual_mouse_profiles=True,
):
    percentile = float(position_plot_percentile)
    if not 80 <= percentile <= 100:
        raise ValueError("position_plot_percentile must be between 80 and 100")
    position = mouse_position_surfaces["position"]
    by_group = group_position_surfaces["by_group"]
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 7.8), constrained_layout=True)

    all_finite = np.concatenate([
        summary["surface"][np.isfinite(summary["surface"])] for summary in by_group.values()
    ]) if by_group else np.array([])
    limit = max(float(np.percentile(np.abs(all_finite), percentile)) if len(all_finite) else 0.0, 0.1)
    for axis, group, title in (
        (axes[0, 0], REWARDED, "Rewarded mouse-weighted surface"),
        (axes[0, 1], UNREWARDED, "Unrewarded mouse-weighted surface"),
    ):
        if group not in by_group or not by_group[group]["surface"].size:
            axis.axis("off")
            axis.text(0.5, 0.5, f"{title}\\nnot available in this source", ha="center", va="center")
            continue
        summary = by_group[group]
        image = axis.imshow(
            summary["surface"], origin="lower", aspect="auto", cmap="coolwarm",
            vmin=-limit, vmax=limit,
            extent=[position[0], position[-1], summary["midpoint"][0], summary["midpoint"][-1]],
        )
        axis.axvline(position_grid["texture_boundary"], color="black", linestyle="--", linewidth=1)
        axis.set(title=f"{title} · n={summary['n_mice']}", xlabel="corridor position (m)", ylabel="trial-block midpoint")
        fig.colorbar(image, ax=axis, label="held-out d′")

    if position_group_contrast["available"]:
        difference = position_group_contrast["surface_difference"]
        finite = np.abs(difference[np.isfinite(difference)])
        diff_limit = max(float(np.percentile(finite, percentile)) if len(finite) else 0.0, 0.1)
        image = axes[0, 2].imshow(
            difference, origin="lower", aspect="auto", cmap="coolwarm",
            vmin=-diff_limit, vmax=diff_limit,
            extent=[position[0], position[-1], 1, difference.shape[0]],
        )
        axes[0, 2].axvline(position_grid["texture_boundary"], color="black", linestyle="--")
        fig.colorbar(image, ax=axes[0, 2], label="rewarded - unrewarded d′")
    else:
        axes[0, 2].axis("off")
        axes[0, 2].text(0.5, 0.5, "Difference surface requires both groups", ha="center", va="center")
    axes[0, 2].set(title="Group difference surface", xlabel="corridor position (m)", ylabel="common time block")

    if bool(show_individual_mouse_profiles):
        for row in mouse_position_changes:
            axes[0, 3].plot(position, row["change"], color=GROUP_COLOURS[row["group"]], alpha=0.45, linewidth=1)
    for group, summary in by_group.items():
        axes[0, 3].plot(position, summary["change"], color=GROUP_COLOURS[group], linewidth=2.5, label=f"{group} mean")
    axes[0, 3].axhline(0, color="#aaa", linewidth=1)
    axes[0, 3].axvline(position_grid["texture_boundary"], color="#777", linestyle="--")
    axes[0, 3].set(title="Mouse early-to-late profiles", xlabel="corridor position (m)", ylabel="late - early d′")
    if by_group:
        axes[0, 3].legend(fontsize=6.5)

    if position_group_contrast["available"]:
        observed = position_group_contrast["change_difference"]
        axes[1, 0].plot(position, observed, color="#111827", linewidth=2, label="difference-in-change")
        if simultaneous_position_inference["available"]:
            axes[1, 0].fill_between(
                position,
                simultaneous_position_inference["simultaneous_lower"],
                simultaneous_position_inference["simultaneous_upper"],
                color="#4c78a8", alpha=0.18,
                label=f"{100*simultaneous_position_inference['confidence']:.0f}% simultaneous band",
            )
            axes[1, 0].fill_between(
                position,
                simultaneous_position_inference["pointwise_lower"],
                simultaneous_position_inference["pointwise_upper"],
                color="#4c78a8", alpha=0.12, label="pointwise band",
            )
    else:
        axes[1, 0].text(0.5, 0.5, "Formal difference-in-change awaits both groups", ha="center", va="center", transform=axes[1, 0].transAxes)
    axes[1, 0].axhline(0, color="#999", linewidth=1)
    axes[1, 0].axvline(position_grid["texture_boundary"], color="#777", linestyle="--")
    axes[1, 0].set_xlim(position_grid["start"], position_grid["stop"])
    axes[1, 0].set(title="Simultaneous whole-curve uncertainty", xlabel="corridor position (m)", ylabel="rewarded - unrewarded change")
    if position_group_contrast["available"]:
        axes[1, 0].legend(fontsize=6.2)

    support_axis = axes[1, 1].twinx()
    for group, summary in behavior_position_profiles["by_group"].items():
        axes[1, 1].plot(position, summary["speed"], color=GROUP_COLOURS[group], label=f"{group} speed")
        support_axis.plot(position, summary["support"], color=GROUP_COLOURS[group], linestyle=":", alpha=0.8, label=f"{group} support")
    axes[1, 1].axvline(position_grid["texture_boundary"], color="#777", linestyle="--")
    axes[1, 1].set(title="Speed and sampling support", xlabel="corridor position (m)", ylabel="speed")
    support_axis.set_ylabel("frames / position bin")

    styles = {"cue": "-", "reward": "--", "first lick": ":"}
    drew_event = False
    for group, events in mouse_event_profiles["by_group"].items():
        for name, density in events.items():
            axes[1, 2].plot(mouse_event_profiles["centers"], density, color=GROUP_COLOURS[group], linestyle=styles[name], label=f"{group} {name}")
            drew_event = True
    axes[1, 2].axvline(position_grid["texture_boundary"], color="#777", linestyle="--")
    axes[1, 2].set_xlim(position_grid["start"], position_grid["stop"])
    axes[1, 2].set(title="Mouse-weighted event locations", xlabel="corridor position (m)", ylabel="mean mouse density")
    if drew_event:
        axes[1, 2].legend(fontsize=6.0)
    else:
        axes[1, 2].text(0.5, 0.5, "Event arrays are absent in the compact demo", ha="center", va="center", transform=axes[1, 2].transAxes)

    axes[1, 3].axis("off")
    included = sum(row["included"] for row in position_support_audit["rows"])
    lines = [
        "POSITION IS A REPEATED MEASURE WITHIN MOUSE",
        "",
        f"Source: {mouse_position_surfaces['label']}",
        f"Area / contrast: {AREA_LABELS[mouse_position_surfaces['area']]} / {mouse_position_surfaces['contrast']}",
        f"Included mice: {included}/{len(position_support_audit['rows'])}",
        f"Coverage threshold: {position_support_audit['threshold']:.0%}",
        f"Early / late blocks: {position_window['early']} / {position_window['late']}",
        f"Minimum common blocks: {position_window['minimum_common']} (requested {position_window['minimum_common_requested']})",
        f"Smoothing: {position_grid['smoothing_bins']} position bin(s)",
        f"Mouse-weighted events: {mouse_event_profiles['mouse_weighted']}",
    ]
    if simultaneous_position_inference["available"]:
        lines += [
            f"Simultaneous confidence: {simultaneous_position_inference['confidence']:.0%}",
            f"Mouse-bootstrap draws: {simultaneous_position_inference['draws']}",
        ]
    else:
        lines += [f"Simultaneous band: {simultaneous_position_inference.get('reason', 'not available')}"]
    for group in (REWARDED, UNREWARDED):
        rows = [row for row in position_region_contrasts if row["group"] == group]
        if rows:
            lines.append(
                f"{group} texture / grey Δ: "
                f"{np.mean([row['texture_change'] for row in rows]):.3f} / "
                f"{np.mean([row['gray_change'] for row in rows]):.3f}"
            )
    lines += ["", "Do not run an uncorrected test at every position bin."]
    axes[1, 3].text(0, 1, "\\n".join(lines), va="top", fontsize=8.0)

    fig.suptitle(
        f"Graph 5 · Plan B · {mouse_position_surfaces['label']} · "
        f"texture < {position_grid['texture_boundary']:.1f} m"
    )
    plt.close(fig)
    return fig
"""
            ),
            py(
                """
#@title Explore position; reuse the last real cohort run when available { display-mode: "form" }
position_graph = graph.Graph(
    "Plan B: trial, corridor position, behavior, and d-prime",
    choose_position_analysis_source_expanded,
    define_common_corridor_grid,
    define_position_time_windows,
    define_position_event_display,
    collect_mouse_position_surfaces,
    audit_mouse_position_support,
    compute_mouse_early_late_position_changes,
    aggregate_mouse_weighted_position_surfaces,
    estimate_position_difference_in_change,
    bootstrap_simultaneous_position_band,
    summarize_mouse_weighted_position_behavior,
    summarize_mouse_level_event_positions,
    summarize_texture_and_gray_changes,
    make_plan_b_plots,
)
position_panel = position_graph.widget(
    controls={
        "position_source_mode": widgets.Dropdown(
            description="Position source",
            options=[("Bundled real recording (mechanics)", "bundled_demo"), ("Reuse last real Plan A run", "last_real_plan_a")],
            value="bundled_demo",
        ),
        "position_area": widgets.Dropdown(
            description="Demo area",
            options=[("V1", "V1"), ("Medial HVA", "mHV"), ("Lateral HVA", "lHV"), ("Anterior HVA", "aHV")],
            value="V1",
        ),
        "position_block_width": widgets.Dropdown(description="Demo time block", options=[80, 100], value=80),
        "position_cv_folds": widgets.Dropdown(description="Demo CV folds", options=[2, 4], value=4),
        "position_min_trials_per_role": widgets.Dropdown(description="Demo min train/role", options=[3, 4], value=4),
        "position_stimulus_pair": widgets.Dropdown(
            description="Demo stimulus pair",
            options=[("Leaf 1 vs circle 1", "leaf1_vs_circle1"), ("Leaf 2 vs circle 2", "leaf2_vs_circle2")],
            value="leaf1_vs_circle1",
        ),
        "corridor_plot_start_m": widgets.FloatSlider(description="Plot start m", value=0.0, min=0.0, max=5.0, step=0.5),
        "corridor_plot_end_m": widgets.FloatSlider(description="Plot end m", value=6.0, min=1.0, max=6.0, step=0.5),
        "texture_boundary_m": widgets.FloatSlider(description="Texture boundary m", value=4.0, min=1.0, max=5.5, step=0.5),
        "position_smoothing_bins": widgets.Dropdown(description="Position smoothing", options=[1, 3, 5], value=1),
        "early_block_count": widgets.IntSlider(description="Early blocks", value=2, min=1, max=3, step=1),
        "late_block_count": widgets.IntSlider(description="Late blocks", value=2, min=1, max=3, step=1),
        "minimum_common_time_blocks": widgets.Dropdown(description="Min common blocks", options=[3, 4, 5, 6], value=4),
        "minimum_mouse_position_coverage": widgets.FloatSlider(description="Min surface coverage", value=0.60, min=0.40, max=0.95, step=0.05, readout_format=".2f"),
        "show_cue_events": widgets.Checkbox(description="Show cue events", value=True),
        "show_reward_events": widgets.Checkbox(description="Show reward events", value=True),
        "show_lick_events": widgets.Checkbox(description="Show first licks", value=True),
        "event_histogram_bins": widgets.Dropdown(description="Event position bins", options=[12, 24, 36], value=24),
        "minimum_events_per_mouse": widgets.Dropdown(description="Min events/mouse", options=[1, 3, 5], value=3),
        "position_bootstrap_draws": widgets.Dropdown(description="Mouse bootstrap", options=[500, 2000, 5000], value=2000),
        "simultaneous_confidence_level": widgets.Dropdown(description="Simultaneous level", options=[0.90, 0.95, 0.99], value=0.95),
        "position_bootstrap_seed": widgets.IntSlider(description="Bootstrap seed", value=31, min=1, max=100, step=1),
        "position_plot_percentile": widgets.IntSlider(description="Surface colour %", value=97, min=90, max=100, step=1),
        "show_individual_mouse_profiles": widgets.Checkbox(description="Show each mouse", value=True),
    },
    show="plots",
)
position_panel
"""
            ),
            md(
                """
### How to read the diagnostics

- Position plots describe where the held-out estimate is supported by released
  frames; they do not identify a biological mechanism.
- Speed, occupancy, cue, reward and lick panels report measured covariates from
  the same released behavior arrays. They do not by themselves establish or
  remove confounding.
- Cross-temporal plots are project-defined summaries of coding-axis transfer;
  they are not reproductions of a Zhong et al. panel.
- Position values repeat within mouse. The group display therefore preserves
  mouse identity and marks bins without sufficient support.
"""
            ),
        ]
    )

    cells.extend(
        [
            md(
                """
## What counts as evidence

Before reporting a descriptive cohort difference, check all of these:

- the selected cortical area and every project-defined setting are saved with
  the result;
- all eligible mice were included, or exclusions are listed by recording ID;
- every $d'$ point uses held-out **trials**, not held-out frames from the same trial;
- block width, early horizon and feature dimension are fixed across mice;
- the result survives reasonable block-width and early-window sensitivity checks;
- individual mouse curves agree with the group summary;
- speed, position occupancy, missing bins, cue/reward/lick timing and global
  drift do not trivially reproduce the group-by-time interaction;
- every analyzed cortical area is reported; if inferential tests are added,
  the family of tested contrasts and its multiplicity handling are stated;
- an SVD population result is not described as an exact replication of the
  paper's full-neuron statistic or as a prospective decoder—the upstream basis
  is transductive;
- “decodable” is not translated into “causal” without perturbation evidence.

Report the mouse-level values, included and excluded recording IDs, exact
specification, and diagnostics together. Describe association only; these
cohorts differ in more than reward delivery.
"""
            ),
        ]
    )

    cleaned_cells = []
    for cell in cells:
        cell.source = (
            cell.source
            .replace("Graph 4", "Graph 3")
            .replace("Graph 5", "Graph 4")
            .replace("Plan A", "time-block analysis")
            .replace("Plan B", "position/support analysis")
            .replace(
                "inspect Graph 3's complete-cohort plan and Graph 3's",
                "inspect Graph 3's complete-cohort plan and Graph 4's",
            )
        )
        cleaned_cells.append(cell)

    notebook.cells = cleaned_cells
    for index, cell in enumerate(notebook.cells):
        cell.id = f"reward-dynamics-{index:03d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
    print(NOTEBOOK)
