#!/usr/bin/env python3
"""Generate the complete, output-free plotting recipe gallery."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/08_plotting_gallery_colab.ipynb")


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
    notebook.cells = [
        md(
            """
# Complete plotting gallery

This notebook exercises every public recipe in `plot.py`. It is both a visual
catalog for non-programmers and an executable integration test for the plotting
API.

The first half uses the bundled, checksum-pinned compact recording and the
canonical release metadata. It is descriptive: the compact neural arrays are
published SVD/area factors, not the full deconvolved single-neuron traces used
for the Nature paper's exact d-prime distributions.

The section headed **Illustrative inputs for unavailable layers** is visibly
separate. The bundled example contains no lick/cue/reward events and no cortical
ROI coordinates, and the current analysis result does not expose a permutation
null distribution. Fixed-seed examples demonstrate those renderers without
presenting synthetic values as observations.

Sources: [Zhong et al., Nature (2025)](https://doi.org/10.1038/s41586-025-09180-y) ·
[Stringer & Pachitariu, Science (2024)](https://doi.org/10.1126/science.adp7429) ·
[Figshare release v2](https://doi.org/10.25378/janelia.28811129.v2).
"""
        ),
        md(
            """
## Load the shared workspace

In Colab, the cell mounts Drive and uses the standard team-workspace shortcut.
Locally it finds the repository from the current directory. Every plot call
returns a fresh figure; `show(...)` displays and then explicitly releases it.
"""
        ),
        py(
            """
from pathlib import Path
import importlib
import sys

import numpy as np
from IPython.display import display

try:
    from google.colab import drive as google_drive
except ImportError:
    google_drive = None

if google_drive is not None:
    google_drive.mount("/content/drive", force_remount=False)
    workspace = Path(
        "/content/drive/MyDrive/"
        "Zhong et al. 2025 - Neuromatch Team Workspace"
    )
    if not workspace.exists():
        raise FileNotFoundError(
            "Add the team workspace as a My Drive shortcut with its existing name."
        )
    sys.path.insert(0, str(workspace))
else:
    workspace = None
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "plot.py").is_file():
            workspace = candidate
            break
    if workspace is None:
        raise FileNotFoundError(
            "Run this notebook from the Zhong 2025 workspace, or add that "
            "workspace as a Colab Drive shortcut."
        )
    sys.path.insert(0, str(workspace))

plot = importlib.import_module("plot")
if Path(plot.__file__).resolve() != (workspace / "plot.py").resolve():
    raise ImportError("Imported an unrelated plot module instead of this workspace's plot.py")
from zhong2025 import load_atlas_demo
from zhong2025.atlas import load_file_inventory
from zhong2025.catalog import load_map
from zhong2025.learning import (
    blockwise_dprime,
    cross_temporal_dprime,
    position_dprime_surface,
)

rendered_recipes = set()
render_log = []

def show(label, figure):
    # Display, verify, record, and explicitly release one figure.
    figure.canvas.draw()
    details = plot.info(figure)
    recipe = details.recipe
    rendered_recipes.add(recipe)
    render_log.append((label, recipe, len(figure.axes)))
    display(figure)
    for warning in details.warnings:
        print(f"CAUTION - {label}: {warning}")
    plot.close(figure)
"""
        ),
        md(
            """
## Choose a plot by the question

Run `plot.guide()` whenever you do not know which function to use. Inputs are
ordinary arrays, dictionaries, and lists; pandas and seaborn are not required.
"""
        ),
        py(
            """
plot.guide()
"""
        ),
        md(
            """
## Published visual references

These are packaged reference images, not recomputed results. They make the
visual vocabulary of both papers available beside the API.
"""
        ),
        py(
            """
show("Nature main reference gallery", plot.reference_gallery(paper="nature"))
show(
    "Nature Extended Data reference gallery",
    plot.reference_gallery(paper="nature", extended=True),
)
show("Science methods reference gallery", plot.reference_gallery(paper="science"))
show("One reference figure", plot.reference_figure(1, paper="nature"))
"""
        ),
        md(
            """
## Bundled released recording

The archive is a compact derivative of TX119's `unsup_test1` recording. The
dashboard exposes trial order, neural-factor activity, corridor profiles,
running speed, and frame support. The second view shows role contrasts across
SVD and area factors and labels their representational boundary explicitly.
"""
        ),
        py(
            """
demo = load_atlas_demo()
position = np.asarray(demo["position_centers_m"])
texture = np.asarray(demo["texture_bin_mask"], dtype=bool)
stimulus = np.asarray(demo["stimulus_id"])
trial_id = np.asarray(demo["trial_id"])
population = np.asarray(demo["population_features"], dtype=float)

show("Released recording dashboard", plot.recording(demo, feature=0))
show("Released feature summary", plot.released_example(demo))
"""
        ),
        md(
            """
## Held-out d-prime over trial progress and position

These calls use the tested contiguous-fold functions in `zhong2025.learning`.
Rows of the cross-temporal matrix are train blocks and columns are test blocks.
The current result schema does not retain class support for each matrix cell;
`plot.info(...)` preserves that limitation where applicable.
"""
        ),
        py(
            """
mhv_index = list(demo["area_name"]).index("mHV")
mhv_features = np.asarray(demo["area_features"][mhv_index], dtype=float)

block_result = blockwise_dprime(
    mhv_features,
    stimulus,
    trial_id,
    position_mask=texture,
    block_trials=80,
    stride_trials=80,
    n_folds=4,
    min_per_role=4,
)
surface_result = position_dprime_surface(
    mhv_features,
    stimulus,
    trial_id,
    block_trials=80,
    n_folds=4,
    min_per_role=4,
)
cross_result = cross_temporal_dprime(
    mhv_features,
    stimulus,
    trial_id,
    position_mask=texture,
    block_trials=80,
    min_per_role=4,
)

show("Blockwise held-out dashboard", plot.blockwise(block_result))
show(
    "Trial-progress by position surface",
    plot.position_surface(
        surface_result,
        position=position,
        title="mHV: trial progress x corridor position",
    ),
)
show(
    "Cross-temporal matrix",
    plot.cross_temporal(
        cross_result,
        title="mHV: train block x test block",
    ),
)
"""
        ),
        md(
            """
## Signed distributions, tails, raw units, and trajectories

ECDF is the stable distribution default. KDE, histogram, violin, and ridgeline
are alternative views with a common axis; none replaces numerical estimates of
spread, skew, kurtosis, or tails. Thin lines remain individual trials/units and
bold lines are means with the named uncertainty band.
"""
        ),
        py(
            """
def component_dprime(mask):
    local_response = np.nanmean(population[mask][:, texture, :], axis=1)
    local_labels = stimulus[mask]
    role_2 = local_response[local_labels == 2]
    role_0 = local_response[local_labels == 0]
    numerator = np.nanmean(role_2, axis=0) - np.nanmean(role_0, axis=0)
    denominator = (
        np.nanstd(role_2, axis=0, ddof=0)
        + np.nanstd(role_0, axis=0, ddof=0)
    ) / 2
    output = np.full(population.shape[2], np.nan)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output

half = len(trial_id) // 2
early = component_dprime(np.arange(len(trial_id)) < half)
late = component_dprime(np.arange(len(trial_id)) >= half)
dprime_groups = {"early half": early, "late half": late}

for kind in ("ecdf", "density", "histogram", "violin", "ridge"):
    show(
        f"Distribution - {kind}",
        plot.distribution(
            dprime_groups,
            kind=kind,
            references=(-0.3, 0, 0.3),
            show_quantiles=(kind == "ecdf"),
            title=f"Compact signed feature contrasts - {kind}",
            xlabel="Role 2 - role 0 d-prime",
        ),
    )

show("Signed tails", plot.signed_tails(dprime_groups, reference=0.3))

texture_response = np.nanmean(population[:, texture, 0], axis=1)
show(
    "D-prime explainer",
    plot.dprime(
        texture_response[stimulus == 2],
        texture_response[stimulus == 0],
        title="Compact feature 0: paper-specific role contrast",
    ),
)
"""
        ),
        py(
            """
wall = np.asarray(demo["wall_name"])
role_profiles = {
    name: population[wall == name, :, 0]
    for name in ("rock1", "wood1")
}

show(
    "Raw role trajectories",
    plot.curve(
        role_profiles,
        x=position,
        band="sem",
        individuals=True,
        regions={"grey": (4, 6)},
        title="Feature 0 by physical wall",
        xlabel="Position (m)",
        ylabel="Feature activity",
    ),
)
show(
    "Metric small multiples",
    plot.small_multiples(
        {
            "running speed": {"all trials": demo["mean_run_speed"]},
            "frame support": {"all trials": demo["frame_counts"]},
        },
        x=position,
        title="Adjacent behavioral and support trajectories",
        xlabel="Position (m)",
    ),
)
show(
    "Paired feature comparison",
    plot.comparison(
        {"early half": early, "late half": late},
        paired=True,
        pair_ids={
            "early half": np.arange(len(early)),
            "late half": np.arange(len(late)),
        },
        summary="mean",
        error="sem",
        reference=0,
        unit="feature",
        title="Same SVD features in two physical trial halves",
        ylabel="Role 2 - role 0 d-prime",
    ),
)
show(
    "Corridor profile",
    plot.corridor(
        role_profiles,
        position=position,
        events={"cue range": (0.5, 3.5)},
        title="Corridor-aware role profiles",
        ylabel="Feature activity",
    ),
)
"""
        ),
        md(
            """
## Matrices, held-out sorting, and large supplied orderings

`train_test` computes row order from the train matrix only and applies it
unchanged to held-out data. `rastermap` renders a supplied ordering; it does not
fit Rastermap. Here the compact SVD features are simply kept in published order.
"""
        ),
        py(
            """
feature_tuning_train = np.nanmean(population[::2], axis=0).T
feature_tuning_test = np.nanmean(population[1::2], axis=0).T
feature_by_sample = population.transpose(2, 0, 1).reshape(population.shape[2], -1)
speed_by_sample = np.asarray(demo["mean_run_speed"]).reshape(-1)

show(
    "Activity matrix",
    plot.activity(
        population[:, :, 0],
        samples=position,
        sort_by="peak",
        center=0,
        regions={"grey": (4, 6)},
        colorbar="Feature 0",
        title="Trials sorted by their own peak - descriptive only",
        xlabel="Position (m)",
        ylabel="Trial",
    ),
)
show(
    "Train-test activity",
    plot.train_test(
        feature_tuning_train,
        feature_tuning_test,
        samples=position,
        sort_by="peak",
        center=0,
        labels=("Even trials: choose order", "Odd trials: held-out display"),
        title="Train-derived feature ordering survives or fails on held-out trials",
    ),
)

correlation = np.corrcoef(feature_by_sample)
show(
    "Representation-similarity matrix",
    plot.matrix(
        correlation,
        center=0,
        robust=False,
        vmin=-1,
        vmax=1,
        colorbar="Correlation",
        title="Compact feature correlation",
        xlabel="Feature",
        ylabel="Feature",
    ),
)
show(
    "Large ordered activity overview",
    plot.rastermap(
        feature_by_sample,
        tracks={"running speed": speed_by_sample},
        events={"25-trial boundary": np.arange(0, feature_by_sample.shape[1], len(position) * 25)},
        title="Published SVD-feature order - no Rastermap fit",
        colorbar="Feature activity",
    ),
)
"""
        ),
        md(
            """
## Population-property relationships and geometry

These views stay on the compact released representation: response means,
component properties, early/late density changes, eigenspectrum, and supplied
three-component trajectories. No PCA, UMAP, topology, or dynamical-system model
is fitted inside the plotting layer.
"""
        ),
        py(
            """
all_response = np.nanmean(population[:, texture, :], axis=1)
mean_role_2 = np.nanmean(all_response[stimulus == 2], axis=0)
mean_role_0 = np.nanmean(all_response[stimulus == 0], axis=0)
variance = np.nanvar(all_response, axis=0)
mean_tuning = np.nanmean(population, axis=0).T
peak_position = position[np.nanargmax(np.abs(mean_tuning), axis=1)]

show(
    "Response relationship",
    plot.relationship(
        mean_role_0,
        mean_role_2,
        identity=True,
        fit=True,
        title="Component response means",
        xlabel="Role 0 mean",
        ylabel="Role 2 mean",
    ),
)
show(
    "Property matrix",
    plot.pairwise(
        {
            "signed d-prime": component_dprime(np.ones(len(trial_id), dtype=bool)),
            "response variance": variance,
            "peak position": peak_position,
        },
        title="Compact component properties",
    ),
)

early_points = np.column_stack([early, variance])
late_points = np.column_stack([late, variance])
show(
    "Density difference",
    plot.density_difference(
        early_points,
        late_points,
        labels=("Early half", "Late half"),
        title="Feature-property density change",
        xlabel="Signed d-prime",
        ylabel="Response variance",
    ),
)

flattened = population.reshape(-1, population.shape[-1])
covariance = np.cov(flattened, rowvar=False)
eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
show("Eigenspectrum", plot.spectrum(eigenvalues, title="Compact population-feature spectrum"))

role_trajectories = {
    "leaf1": np.nanmean(population[stimulus == 2, :, :3], axis=0),
    "circle1": np.nanmean(population[stimulus == 0, :, :3], axis=0),
}
show(
    "Component trajectories",
    plot.trajectory(
        role_trajectories,
        dimensions=(0, 1, 2),
        title="Supplied compact feature trajectories over corridor position",
    ),
)
"""
        ),
        md(
            """
## Illustrative inputs for unavailable layers

Everything below is deterministic and visibly labeled **illustrative**. These
calls prove that the API can render event rasters, ROI maps, model predictions,
agreement, inference outputs, filter banks, and annotations when a teammate
supplies the required prepared arrays. They are not released observations and
must not be cited as Zhong et al. results.
"""
        ),
        py(
            """
rng = np.random.default_rng(2025)

illustrative_events = {
    "lick": [np.sort(rng.uniform(0.2, 5.8, size=2 + trial % 5)) for trial in range(28)],
    "cue": np.linspace(0.7, 3.3, 28),
    "reward": np.where(np.arange(28) % 2 == 0, 3.6, np.nan),
}

roi_angle = np.linspace(0, 2 * np.pi, 450, endpoint=False)
roi_radius = 1.0 + 0.18 * np.sin(5 * roi_angle) + rng.normal(0, 0.06, len(roi_angle))
roi_x = roi_radius * np.cos(roi_angle) + rng.normal(0, 0.12, len(roi_angle))
roi_y = 0.72 * roi_radius * np.sin(roi_angle) + rng.normal(0, 0.10, len(roi_angle))
roi_group = np.where(roi_x < -0.3, "V1", np.where(roi_y > 0.15, "medial", "lateral"))
roi_selected = (roi_x + 0.7 * roi_y + rng.normal(0, 0.45, len(roi_x))) > 0.45
outline_angle = np.linspace(0, 2 * np.pi, 160)
outline = np.column_stack([1.38 * np.cos(outline_angle), 1.05 * np.sin(outline_angle)])

show(
    "Illustrative event raster",
    plot.event_raster(
        illustrative_events,
        regions={"texture": (0, 4), "grey": (4, 6)},
        title="Illustrative events - not released observations",
        xlabel="Position (m)",
    ),
)
show(
    "Illustrative cortical points",
    plot.cortical_map(
        roi_x,
        roi_y,
        groups=roi_group,
        outlines=[outline],
        title="Illustrative ROI categories - no atlas inference",
    ),
)
show(
    "Illustrative cortical density",
    plot.cortical_density(
        roi_x,
        roi_y,
        selected=roi_selected,
        outlines=[outline],
        bins=70,
        sigma=2.5,
        title="Illustrative selected-ROI density - not released data",
    ),
)
"""
        ),
        py(
            """
sample_time = np.linspace(0, 12, 180)
truth = np.sin(sample_time) + 0.25 * np.cos(2.4 * sample_time)
predicted = 0.92 * np.sin(sample_time - 0.15) + 0.20 * np.cos(2.4 * sample_time)
show(
    "Illustrative prediction",
    plot.prediction(
        truth,
        predicted,
        x=sample_time,
        title="Illustrative held-out prediction arrays",
    ),
)
show(
    "Illustrative representation agreement",
    plot.agreement(
        truth,
        predicted,
        labels=("Reference representation", "Approximation"),
        title="Illustrative approximation agreement",
    ),
)

null = rng.normal(0, 0.11, size=715)
observed = 0.24
pvalue = (np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)
show(
    "Illustrative permutation",
    plot.permutation(
        null,
        observed,
        pvalue=pvalue,
        title="Illustrative supplied permutation null",
    ),
)
forest_estimates = np.array([0.24, 0.21, 0.27, 0.19, 0.25])
forest_half_width = np.array([0.08, 0.10, 0.09, 0.12, 0.08])
show(
    "Illustrative forest",
    plot.forest(
        ["All mice", "Leave out A", "Leave out B", "Leave out C", "Leave out D"],
        forest_estimates,
        np.column_stack(
            [forest_estimates - forest_half_width, forest_estimates + forest_half_width]
        ),
        title="Illustrative leave-one-mouse-out estimates",
    ),
)

show(
    "Illustrative adjusted annotations",
    plot.comparison(
        {
            "condition A": rng.normal(0.0, 0.35, 12),
            "condition B": rng.normal(0.3, 0.35, 12),
            "condition C": rng.normal(0.6, 0.35, 12),
        },
        comparisons=[("condition A", "condition B", 0.041), ("condition A", "condition C", 0.004)],
        correction="holm",
        title="Illustrative supplied p-values with explicit Holm correction",
        ylabel="Illustrative value",
    ),
)

grid = np.linspace(-2.5, 2.5, 40)
xx, yy = np.meshgrid(grid, grid)
filters = []
filter_labels = []
for index, angle in enumerate(np.linspace(0, np.pi, 8, endpoint=False)):
    rotated = xx * np.cos(angle) + yy * np.sin(angle)
    envelope = np.exp(-(xx**2 + yy**2) / 2.2)
    filters.append(envelope * np.cos(4.5 * rotated))
    filter_labels.append(f"orientation {index + 1}")
show(
    "Illustrative filter bank",
    plot.image_grid(
        filters,
        labels=filter_labels,
        columns=4,
        cmap="coolwarm",
        colorbar="Illustrative filter weight",
        title="Illustrative supplied filter bank",
    ),
)
"""
        ),
        md(
            """
## Timelines, totals, images, journeys, and release preflight

These views use the paper sequence and the repository's canonical 89-recording,
297-file metadata. Bars are reserved for counts, storage, and other totals—not
as a substitute for raw experimental-unit comparisons.
"""
        ),
        py(
            """
inventory = load_file_inventory()
canonical = load_map()
files = inventory["files"]

category_counts = {}
category_bytes = {}
for row in files:
    category = row["category"]
    category_counts[category] = category_counts.get(category, 0) + 1
    category_bytes[category] = category_bytes.get(category, 0) + row["size_bytes"]

show(
    "Release file counts",
    plot.bars(
        category_counts,
        horizontal=True,
        title="Figshare v2 file counts by layer",
        xlabel="Files",
    ),
)
show(
    "Release bytes by layer",
    plot.stacked_bars(
        {"GiB": {key: value / 1024**3 for key, value in category_bytes.items()}},
        categories=list(category_bytes),
        title="Figshare v2 storage by layer",
        ylabel="GiB",
    ),
)
show(
    "Paper timeline",
    plot.timeline(
        [
            "Acclimation",
            "Train 1 before",
            "Train 1 after",
            "Test 1",
            "Train 2",
            "Test 2",
        ],
        groups=["setup", "train", "train", "test", "train", "test"],
        title="Imaging-study sequence",
    ),
)

"""
        ),
        py(
            """
journeys = {}
for recording in canonical["recordings"].values():
    journeys.setdefault(recording["mouse"], []).append(
        {
            "mouse": recording["mouse"],
            "date": recording["date"],
            "stage": recording["stage"],
            "memberships": len(recording["memberships"]),
        }
    )
for mouse in journeys:
    journeys[mouse] = sorted(journeys[mouse], key=lambda row: row["date"])

selected_mouse = max(journeys, key=lambda mouse: len(journeys[mouse]))
show(
    "One real indexed mouse journey",
    plot.mouse_journey(
        journeys[selected_mouse],
        metrics={"Analysis memberships": "memberships"},
    ),
)
show("All real indexed mouse journeys", plot.all_mouse_journeys(journeys))

size_by_recording = {}
retinotopy_bytes_by_id = {}
for file_row in files:
    recording_id = file_row.get("recording_id")
    if recording_id:
        size_by_recording.setdefault(recording_id, {})[
            f"{file_row['category']}_bytes"
        ] = file_row["size_bytes"]
    if file_row["category"] == "retinotopy" and file_row.get("retinotopy_id"):
        retinotopy_bytes_by_id[file_row["retinotopy_id"]] = file_row["size_bytes"]

manifest_rows = []
for recording_id, recording in canonical["recordings"].items():
    row = {
        "mouse": recording["mouse"],
        "group": recording["cohort"],
        "date": recording["date"],
    }
    row.update(size_by_recording.get(recording_id, {}))
    retinotopy_id = recording.get("retinotopy_id")
    if retinotopy_id in retinotopy_bytes_by_id:
        row["retinotopy_bytes"] = retinotopy_bytes_by_id[retinotopy_id]
    manifest_rows.append(row)

show(
    "Real release preflight",
    plot.cohort_preflight(
        manifest_rows,
        layers={
            "full neural": "full_neural_bytes",
            "reduced neural": "reduced_neural_bytes",
            "retinotopy": "retinotopy_bytes",
        },
        title="All 89 indexed recordings - metadata-only preflight",
    ),
)
"""
        ),
        md(
            """
## Adjacent quality control

Support and behavior stay beside the scientific view. Missing bins and unequal
role counts are visible rather than silently filtered from the story.
"""
        ),
        py(
            """
show(
    "Released recording QC",
    plot.qc(
        demo,
        position=position,
        labels=stimulus,
        title="TX119 compact derivative - support and behavior QC",
    ),
)
"""
        ),
        md(
            """
## Coverage check

The assertion below fails whenever a public recipe is added without a working
example in this notebook. This keeps the gallery and the API synchronized.
"""
        ),
        py(
            """
expected_recipes = set(plot.recipes())
missing_recipes = sorted(expected_recipes - rendered_recipes)
unexpected_recipes = sorted(rendered_recipes - expected_recipes)
if missing_recipes or unexpected_recipes:
    raise AssertionError(
        {"missing": missing_recipes, "unexpected": unexpected_recipes}
    )

print(f"Rendered all {len(expected_recipes)} plotting recipes successfully.")
for label, recipe, axes in render_log:
    print(f"- {recipe:20s} | {axes:2d} axes | {label}")
"""
        ),
        md(
            """
## Use the API in team analysis

The usual pattern is one named recipe and ordinary arrays:

```python
import plot

figure = plot.comparison(
    {"before": before_by_mouse, "after": after_by_mouse},
    paired=True,
    pair_ids={"before": mouse_ids, "after": mouse_ids},
    unit="mouse",
    ylabel="Selective neurons (%)",
)
plot.save(figure, "results/selectivity.png")
plot.close(figure)
```

Use full neural traces for exact paper-style per-neuron distributions, keep the
sign as role 2 minus role 0, and make mouse/session/trial/neuron identity explicit
before plotting. A polished figure cannot repair an invalid experimental unit.

Friendly aliases make analysis prose readable without introducing new behavior:
`learning_curve` and `population_trace` call `curve`; `paired_summary` calls
`comparison`; `cross_temporal`, `position_surface`, and
`representation_similarity` call `matrix`; `retinotopy_map` calls
`cortical_map`; and `within_session_dprime` calls `blockwise`.
"""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"plot-gallery-{index:03d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
