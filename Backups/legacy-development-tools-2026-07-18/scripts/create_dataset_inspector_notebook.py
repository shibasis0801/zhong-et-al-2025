#!/usr/bin/env python3
"""Generate the complete Colab dataset inspector notebook."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/12_complete_dataset_inspector_colab.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def py(text: str):
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
# Zhong et al. (2025): complete dataset inspector

This notebook gives one coherent access surface for the entire Figshare v2
release:

- **Pandas** holds tidy file, mouse, experiment, acquisition, frame, trial, and
  neuron tables.
- **DuckDB SQL** resolves identities and joins those compact tables.
- **NumPy** keeps the dense neural matrices dense.
- **Google Drive** is touched only when a selected array is explicitly loaded.

The complete release is visible immediately as **297 file rows**, **19 imaging
mice**, **89 physical acquisitions**, **23 imaging experiment labels**, **142
source experiment rows**, and **133 unique experiment–acquisition
memberships**. A separate Figure 5 cohort contributes **23 behavior-only mice**
and has no neural or retinotopy join.

The notebook never expands every neuron × frame combination into SQL. That
would create billions of rows. Instead, metadata is exhaustive and array access
is lazy, bounded, and validated.
"""
        ),
        md(
            """
## The join model

| Grain | Identifier | What it connects |
|---|---|---|
| biological subject | `mouse` | repeated acquisitions and longitudinal summaries |
| physical acquisition | `recording_id = mouse_date_block` | full neural, SVD, experiment membership, ordinary behavior session |
| retinotopy session | `retinotopy_id = mouse_date` | neuron-indexed retinotopy to the acquisition's neuron axis |
| logical protocol | `experiment` | selects one `Beh_{experiment}.npy` bundle |
| behavior instance | `behavior_key` | usually `recording_id`; Test 3 uses explicit `_swap1` / `_swap2` keys |
| neural sample | `frame_id` | `V[:, frame_id]` or `N[:, frame_id]` to every `ft_*[frame_id]` field |
| detected cell | `neuron_id` | `U[:, neuron_id]` or `N[neuron_id, :]` to `iarea[neuron_id]` and `xy_t[neuron_id]` |
| traversal | `trial_id` | frame rows to trial wall, stimulus role, and trial summaries |

`U` and `V` share a 400-component coordinate system. `U[:, neuron_id]` is the
selected neuron's loading vector; `V[:, frame_id]` is the population state at
the selected frame. Approximate activity is reconstructed only for requested
subsets with `U[:, neurons].T @ V[:, frames]`.
"""
        ),
        py(
            r'''
#@title 1 · Mount Drive, load the workspace helpers, and connect { display-mode: "form" }
import json
import importlib
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys
import types


def ensure_package(distribution, requirement, *, exact=None, minimum=None):
    try:
        installed = version(distribution)
    except PackageNotFoundError:
        installed = None
    numeric = tuple(int(part) for part in installed.split(".")[:2]) if installed else None
    acceptable = installed is not None
    acceptable &= exact is None or installed == exact
    acceptable &= minimum is None or numeric >= minimum
    if not acceptable:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", requirement])


ensure_package("duckdb", "duckdb>=1.4,<2", minimum=(1, 4))
ensure_package("scikit-learn", "scikit-learn==1.6.0", exact="1.6.0")

from google.colab import drive as google_drive

google_drive.mount("/content/drive", force_remount=False)

MY_DRIVE = Path("/content/drive/MyDrive")
WORKSPACE = Path(os.environ.get(
    "ZHONG2025_WORKSPACE",
    MY_DRIVE / "Zhong et al. 2025 - Neuromatch Team Workspace",
)).expanduser()
DATA_ROOT = Path(os.environ.get(
    "ZHONG2025_DATASET_ROOT",
    WORKSPACE / "Janelia dataset - Zhong et al. 2025 (Figshare v2)",
)).expanduser()

required_paths = [WORKSPACE / "sql.py", WORKSPACE / "drive.py", DATA_ROOT / "VERIFIED.json"]
missing = [path for path in required_paths if not path.exists()]
if missing:
    tried = "\n".join(f"- {path}" for path in missing)
    raise FileNotFoundError(
        "The team workspace or verified release is missing. Add it to My Drive, "
        f"then rerun. Missing:\n{tried}"
    )

sys.path.insert(0, str(WORKSPACE))

# The shared workspace can contain a newer ``zhong2025/__init__.py`` beside an
# older, incomplete helper-package copy.  Import only the two small modules that
# ``drive.py`` needs instead of executing the package's broad convenience API.
package_dir = WORKSPACE / "zhong2025"
if package_dir.is_dir():
    package = types.ModuleType("zhong2025")
    package.__path__ = [str(package_dir)]
    package.__package__ = "zhong2025"
    sys.modules["zhong2025"] = package
    atlas_module = importlib.import_module("zhong2025.atlas")
    data_module = importlib.import_module("zhong2025.data")
    for name in (
        "experiment_rows",
        "format_bytes",
        "load_experiment_index",
        "load_file_inventory",
        "recording_bundle",
    ):
        setattr(package, name, getattr(atlas_module, name))
    package.fetch_figshare_article = data_module.fetch_figshare_article

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
import sql as release_sql

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 100)
plt.style.use("seaborn-v0_8-whitegrid")

db = release_sql.setup(root=DATA_ROOT, report=False)
verified = json.loads((DATA_ROOT / "VERIFIED.json").read_text())
assert verified["state"] == "complete"
assert verified["article_id"] == 28811129 and verified["version"] == 2
assert verified["file_count"] == 297
assert verified["total_bytes"] == 452_233_500_962

print("Workspace:", WORKSPACE)
print("Release:", DATA_ROOT)
display(db.schema())
'''
        ),
        md(
            """
## Make Test 3 behavior instances explicit

The release index contains 142 source rows but only 133 unique
experiment–recording memberships. The difference is meaningful: nine Test 3
memberships contain separate `swap1` and `swap2` behavior instances. We retain
all source rows and derive an explicit `behavior_key` before doing any joins.
"""
        ),
        py(
            r'''
experiment_rows = db.table("experiment_rows")
has_suffix = experiment_rows["stimulus_type"].fillna("").astype(str).str.len().gt(0)
behavior_instances = experiment_rows.copy()
behavior_instances["behavior_key"] = np.where(
    has_suffix,
    behavior_instances["recording_id"] + "_" + behavior_instances["stimulus_type"].astype(str),
    behavior_instances["recording_id"],
)

assert len(behavior_instances) == 142
assert behavior_instances[["experiment", "behavior_key"]].drop_duplicates().shape[0] == 142
assert behavior_instances[["experiment", "recording_id"]].drop_duplicates().shape[0] == 133
db.register("behavior_instances", behavior_instances)

display(
    behavior_instances.loc[has_suffix, [
        "experiment", "recording_id", "behavior_key", "stimulus_type", "mouse", "date"
    ]].sort_values(["experiment", "recording_id", "behavior_key"])
)
'''
        ),
        md(
            """
## Release-wide overview

These cells inspect the complete compact catalog. No neural array is loaded.
The totals are asserted so a partial mount or stale release cannot masquerade
as the complete dataset.
"""
        ),
        py(
            r'''
overview = db.query("""
SELECT
    (SELECT count(*) FROM files) AS files,
    (SELECT sum(size_bytes) FROM files) AS bytes,
    (SELECT count(*) FROM mice) AS imaging_mice,
    (SELECT count(*) FROM recordings) AS physical_acquisitions,
    (SELECT count(*) FROM experiments) AS imaging_experiments,
    (SELECT count(*) FROM experiment_rows) AS source_rows,
    (SELECT count(*) FROM memberships) AS memberships,
    (SELECT count(*) FROM behavior_instances) AS behavior_instances
""")

expected = {
    "files": 297,
    "bytes": 452_233_500_962,
    "imaging_mice": 19,
    "physical_acquisitions": 89,
    "imaging_experiments": 23,
    "source_rows": 142,
    "memberships": 133,
    "behavior_instances": 142,
}
assert overview.iloc[0].to_dict() == expected
display(overview.style.format({"bytes": "{:,}"}))
'''
        ),
        py(
            r'''
# Every deposited file family: count, exact size, and share of the release.
file_families = db.query("""
SELECT
    category,
    count(*) AS files,
    sum(size_bytes) AS bytes,
    round(sum(size_bytes) / pow(1024, 3), 3) AS gib,
    round(100 * sum(size_bytes) / (SELECT sum(size_bytes) FROM files), 3) AS percent_of_release
FROM files
GROUP BY category
ORDER BY bytes DESC
""")
display(file_families.style.format({"bytes": "{:,}", "gib": "{:.3f}", "percent_of_release": "{:.3f}"}))

ax = file_families.sort_values("gib").plot.barh(
    x="category", y="gib", legend=False, figsize=(9, 4.8), color="#2a9d8f"
)
ax.set(xlabel="GiB", ylabel="", title="Complete Figshare v2 release by file family")
plt.tight_layout()
'''
        ),
        py(
            r'''
# All imaging mice, cohorts, and physical acquisition counts.
mice = db.query("""
SELECT mouse, primary_cohort, recording_count, cohorts_json,
       has_full_neural, has_reduced_neural
FROM mice
ORDER BY primary_cohort, mouse
""")
display(mice)

cohort_summary = (
    mice.groupby("primary_cohort", as_index=False)
    .agg(mice=("mouse", "nunique"), acquisitions=("recording_count", "sum"))
    .sort_values("primary_cohort")
)
display(cohort_summary)
'''
        ),
        py(
            r'''
# Every imaging experiment label and its sample size.
experiments = db.query("""
SELECT experiment, cohort, stage, moment, recording_count, mouse_count
FROM experiments
ORDER BY cohort, stage, moment, experiment
""")
display(experiments)

coverage = experiments.pivot_table(
    index="experiment", columns="cohort", values="recording_count", fill_value=0
)
fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(coverage))))
image = ax.imshow(coverage.to_numpy(), aspect="auto", cmap="YlGn")
ax.set_xticks(range(len(coverage.columns)), coverage.columns, rotation=30, ha="right")
ax.set_yticks(range(len(coverage.index)), coverage.index)
ax.set_title("Physical acquisitions per experiment label")
fig.colorbar(image, ax=ax, label="recordings")
plt.tight_layout()
'''
        ),
        py(
            r'''
# All 89 physical acquisitions. experiments_json preserves multiple logical memberships.
recordings = db.query("""
SELECT recording_id, mouse, date, block, experiment_count, experiments_json,
       has_behavior, has_reduced_neural, has_full_neural, has_retinotopy,
       linked_file_count, round(linked_gib, 3) AS linked_gib
FROM recordings
ORDER BY mouse, date, block
""")
assert len(recordings) == 89
display(recordings)
'''
        ),
        py(
            r'''
# Acquisitions reused under more than one logical experiment label.
reused_acquisitions = db.query("""
SELECT recording_id, mouse, date, block, experiment_count, experiments_json
FROM recordings
WHERE experiment_count > 1
ORDER BY experiment_count DESC, mouse, date, block
""")
display(reused_acquisitions)

# Train 1 before/after acquisition sets, aligned within mouse and cohort.
train1_pairs = db.query("""
WITH stages AS (
    SELECT m.cohort, m.mouse, m.date, m.block, m.recording_id, e.moment
    FROM memberships AS m
    JOIN experiments AS e USING (experiment)
    WHERE e.stage = 'train1' AND e.moment IN ('before', 'after')
)
SELECT
    cohort,
    mouse,
    list(recording_id ORDER BY date, block) FILTER (WHERE moment = 'before') AS before_recordings,
    list(recording_id ORDER BY date, block) FILTER (WHERE moment = 'after') AS after_recordings,
    count(*) FILTER (WHERE moment = 'before') AS before_count,
    count(*) FILTER (WHERE moment = 'after') AS after_count
FROM stages
GROUP BY cohort, mouse
HAVING before_count > 0 AND after_count > 0
ORDER BY cohort, mouse
""")
display(train1_pairs)
'''
        ),
        md(
            """
## One elegant access object

`DatasetInspector` keeps the global catalog exhaustive and the arrays lazy.
`JoinedRecording` exposes three explicitly validated coordinate tables:

- `session.frames`: one row per neural frame, with aligned behavior labels;
- `session.trials`: one row per corridor traversal;
- `session.neurons`: one row per neuron, with retinotopy and area labels.

Dense activity remains in `session.U` and `session.V`. The `activity(...)`
method reconstructs only requested subsets and refuses oversized accidental
expansions.
"""
        ),
        py(
            r'''
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Mapping


AREA_GROUP = {
    -1: "excluded", 0: "mHV", 1: "mHV", 2: "mHV", 3: "aHV", 4: "aHV",
    5: "lHV", 6: "lHV", 7: "excluded", 8: "V1", 9: "mHV",
}
MAX_TRAILING_BEHAVIOR_FRAMES = 3
MAX_ACTIVITY_VALUES = 2_000_000


def one_dimensional(name, values):
    array = np.asarray(values).squeeze()
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


def bounded_indices(values: Iterable[int] | None, size: int, default_count: int):
    if values is None:
        if size <= default_count:
            return np.arange(size, dtype=np.int64)
        return np.linspace(0, size - 1, default_count, dtype=np.int64)
    result = np.asarray(list(values), dtype=np.int64)
    if result.ndim != 1 or np.any(result < 0) or np.any(result >= size):
        raise IndexError(f"indices must be one-dimensional and inside [0, {size})")
    return result


@dataclass
class JoinedRecording:
    metadata: Mapping
    behavior: Mapping
    U: np.ndarray
    V: np.ndarray
    retinotopy: Mapping
    full_payload: Mapping | None = None

    def __post_init__(self):
        self.U = np.asarray(self.U)
        self.V = np.asarray(self.V)
        self.iarea = one_dimensional("iarea", self.retinotopy["iarea"])
        self.xy_t = np.asarray(self.retinotopy["xy_t"])
        if self.U.ndim != 2 or self.V.ndim != 2:
            raise ValueError("U and V must both be two-dimensional")
        if self.U.shape[0] != self.V.shape[0]:
            raise ValueError("U and V do not share one component axis")
        if self.U.shape[0] != 400:
            raise ValueError(f"Expected 400 retained components, found {self.U.shape[0]}")
        if self.xy_t.ndim != 2 or self.xy_t.shape[1] != 2:
            raise ValueError(f"xy_t must be neurons × 2, got {self.xy_t.shape}")
        if self.U.shape[1] != len(self.iarea) or self.U.shape[1] != len(self.xy_t):
            raise ValueError("U, iarea, and xy_t do not share one neuron axis")

    @property
    def recording_id(self):
        return str(self.metadata["recording_id"])

    @property
    def experiment(self):
        return str(self.metadata["experiment"])

    @cached_property
    def frames(self):
        fields = {
            "trial_raw": one_dimensional("ft_trInd", self.behavior["ft_trInd"]),
            "wall_at_frame": one_dimensional("ft_WallID", self.behavior["ft_WallID"]),
            "position_dm": one_dimensional("ft_Pos", self.behavior["ft_Pos"]),
            "move": one_dimensional(
                "ft_move",
                self.behavior["ft_move"] if "ft_move" in self.behavior else self.behavior["ft_isMoving"],
            ),
            "in_texture": one_dimensional("ft_CorrSpc", self.behavior["ft_CorrSpc"]),
            "run_speed": one_dimensional("ft_RunSpeed", self.behavior["ft_RunSpeed"]),
        }
        if "ft" in self.behavior:
            fields["time"] = one_dimensional("ft", self.behavior["ft"])

        lengths = {name: len(values) for name, values in fields.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Behavior frame fields disagree: {lengths}")
        neural_frames = self.V.shape[1]
        trailing_excess = next(iter(lengths.values())) - neural_frames
        if not 0 <= trailing_excess <= MAX_TRAILING_BEHAVIOR_FRAMES:
            raise ValueError(
                f"Behavior/neural frame mismatch is {trailing_excess}; allowed trailing excess is "
                f"0..{MAX_TRAILING_BEHAVIOR_FRAMES}"
            )
        aligned = {name: values[:neural_frames] for name, values in fields.items()}

        wall_by_trial = one_dimensional("WallName", self.behavior["WallName"]).astype(str)
        trial_raw = np.asarray(aligned["trial_raw"], dtype=float)
        integer_trial = np.isfinite(trial_raw) & np.isclose(trial_raw, np.round(trial_raw))
        trial_integer = np.full(neural_frames, -1, dtype=np.int64)
        trial_integer[integer_trial] = np.round(trial_raw[integer_trial]).astype(np.int64)
        valid_trial = integer_trial & (trial_integer >= 0) & (trial_integer < len(wall_by_trial))
        trial_nullable = pd.array(np.where(valid_trial, trial_integer, np.nan), dtype="Int64")

        unique_walls = one_dimensional("UniqWalls", self.behavior["UniqWalls"]).astype(str)
        stimulus_ids = one_dimensional("stim_id", self.behavior["stim_id"])
        if len(unique_walls) != len(stimulus_ids):
            raise ValueError("UniqWalls and stim_id must be parallel arrays")
        role_by_wall = {}
        for wall, role in zip(unique_walls, stimulus_ids):
            try:
                if np.isfinite(role):
                    role_by_wall[str(wall)] = int(role)
            except TypeError:
                pass

        trial_wall = np.full(neural_frames, None, dtype=object)
        trial_wall[valid_trial] = wall_by_trial[trial_integer[valid_trial]]
        frame = pd.DataFrame({
            "recording_id": self.recording_id,
            "experiment": self.experiment,
            "frame_id": np.arange(neural_frames, dtype=np.int64),
            "trial_id": trial_nullable,
            "valid_trial": valid_trial,
            "wall_at_frame": np.asarray(aligned["wall_at_frame"]).astype(str),
            "trial_wall": trial_wall,
            "stimulus_role": pd.Series(trial_wall).map(role_by_wall).astype("Int64"),
            "position_m": np.asarray(aligned["position_dm"], dtype=float) / 10.0,
            "is_moving": np.asarray(aligned["move"]) > 0,
            "in_texture": np.asarray(aligned["in_texture"], dtype=bool),
            "run_speed": np.asarray(aligned["run_speed"], dtype=float),
        })
        if "time" in aligned:
            frame["time"] = np.asarray(aligned["time"])
        frame.attrs["trailing_behavior_frames_dropped"] = int(trailing_excess)
        return frame

    @cached_property
    def trials(self):
        valid = self.frames.dropna(subset=["trial_id"]).copy()
        return (
            valid.groupby("trial_id", as_index=False, observed=True)
            .agg(
                frame_start=("frame_id", "min"),
                frame_stop=("frame_id", "max"),
                n_frames=("frame_id", "size"),
                wall_name=("trial_wall", "first"),
                stimulus_role=("stimulus_role", "first"),
                mean_position_m=("position_m", "mean"),
                mean_speed=("run_speed", "mean"),
                moving_fraction=("is_moving", "mean"),
                texture_fraction=("in_texture", "mean"),
            )
            .assign(recording_id=self.recording_id, experiment=self.experiment)
        )

    @cached_property
    def neurons(self):
        frame = pd.DataFrame({
            "recording_id": self.recording_id,
            "neuron_id": np.arange(self.U.shape[1], dtype=np.int64),
            "area_id": self.iarea.astype(np.int16, copy=False),
            "area_group": pd.Series(self.iarea).map(AREA_GROUP).fillna("excluded"),
            "cortical_x": -self.xy_t[:, 1],
            "cortical_y": self.xy_t[:, 0],
        })
        for name in ("xpos", "ypos"):
            if name in self.retinotopy:
                values = one_dimensional(name, self.retinotopy[name])
                if len(values) == len(frame):
                    frame[name] = values
        return frame

    def describe(self):
        return pd.DataFrame([
            ("recording_id", self.recording_id),
            ("experiment", self.experiment),
            ("behavior_key", self.metadata["behavior_key"]),
            ("components", self.U.shape[0]),
            ("neurons", self.U.shape[1]),
            ("frames", self.V.shape[1]),
            ("trials", len(self.trials)),
            ("retinotopy_neurons", len(self.iarea)),
            ("trailing_behavior_frames_dropped", self.frames.attrs["trailing_behavior_frames_dropped"]),
            ("full_neural_loaded", self.full_payload is not None),
        ], columns=["field", "value"])

    def activity(self, neuron_ids=None, frame_ids=None, *, max_values=MAX_ACTIVITY_VALUES):
        neurons = bounded_indices(neuron_ids, self.U.shape[1], default_count=40)
        frames = bounded_indices(frame_ids, self.V.shape[1], default_count=1500)
        values = len(neurons) * len(frames)
        if values > max_values:
            raise ValueError(
                f"Requested {values:,} values; choose fewer neurons/frames or raise max_values explicitly"
            )
        return self.U[:, neurons].T @ self.V[:, frames], neurons, frames

    def activity_long(self, neuron_ids=None, frame_ids=None, *, max_values=100_000):
        activity, neurons, frames = self.activity(
            neuron_ids=neuron_ids, frame_ids=frame_ids, max_values=max_values
        )
        return pd.DataFrame({
            "recording_id": self.recording_id,
            "neuron_id": np.repeat(neurons, len(frames)),
            "frame_id": np.tile(frames, len(neurons)),
            "activity": activity.reshape(-1),
        })

    def full_matrix(self):
        if self.full_payload is None:
            raise RuntimeError("Reload with include_full=True to access exact full-neural traces")
        planes = self.full_payload["spks"]
        arrays = [np.asarray(plane) for plane in planes]
        matrix = np.concatenate(arrays, axis=0)
        if matrix.shape != (self.U.shape[1], self.V.shape[1]):
            raise ValueError(
                f"Full neural shape {matrix.shape} does not match SVD axes "
                f"{(self.U.shape[1], self.V.shape[1])}"
            )
        return matrix


class DatasetInspector:
    def __init__(self, db):
        self.db = db

    def sql(self, statement, parameters=None):
        return self.db.query(statement, parameters)

    def table(self, name):
        return self.db.table(name)

    def manifest(self, recording_id=None, experiment=None, behavior_key=None):
        conditions, parameters = [], []
        for column, value in (
            ("bi.recording_id", recording_id),
            ("bi.experiment", experiment),
            ("bi.behavior_key", behavior_key),
        ):
            if value:
                conditions.append(f"{column} = ?")
                parameters.append(value)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        return self.db.query(f"""
            SELECT
                bi.experiment, bi.source_row, bi.mouse, bi.date, bi.block,
                bi.recording_id, bi.retinotopy_id, bi.behavior_key,
                bi.stimulus_type, bi.reward_type, bi.stimulus,
                max(CASE WHEN rf.layer = 'behavior' THEN rf.filename END) AS behavior_filename,
                max(CASE WHEN rf.layer = 'behavior' THEN rf.size_bytes END) AS behavior_bytes,
                max(CASE WHEN rf.layer = 'reduced_neural' THEN rf.filename END) AS svd_filename,
                max(CASE WHEN rf.layer = 'reduced_neural' THEN rf.size_bytes END) AS svd_bytes,
                max(CASE WHEN rf.layer = 'full_neural' THEN rf.filename END) AS full_neural_filename,
                max(CASE WHEN rf.layer = 'full_neural' THEN rf.size_bytes END) AS full_neural_bytes,
                max(CASE WHEN rf.layer = 'retinotopy' THEN rf.filename END) AS retinotopy_filename,
                max(CASE WHEN rf.layer = 'retinotopy' THEN rf.size_bytes END) AS retinotopy_bytes
            FROM behavior_instances AS bi
            LEFT JOIN recording_files AS rf
              ON rf.recording_id = bi.recording_id
             AND (rf.layer <> 'behavior' OR rf.experiment = bi.experiment)
            {where}
            GROUP BY ALL
            ORDER BY bi.mouse, bi.date, bi.block, bi.experiment, bi.behavior_key
        """, parameters)

    def recording(self, recording_id):
        return self.db.query("""
            SELECT r.recording_id, r.mouse, r.date, r.block, r.retinotopy_id,
                   m.cohort, m.experiment, e.stage, e.moment,
                   rf.layer, rf.filename, rf.size_bytes, rf.md5, rf.relative_path
            FROM recordings AS r
            LEFT JOIN memberships AS m USING (recording_id)
            LEFT JOIN experiments AS e USING (experiment)
            LEFT JOIN recording_files AS rf
              ON rf.recording_id = r.recording_id
             AND (rf.layer <> 'behavior' OR rf.experiment = m.experiment)
            WHERE r.recording_id = ?
            ORDER BY m.experiment, rf.layer
        """, [recording_id])

    def download_plan(self, recording_id, experiment=None):
        manifest = self.manifest(recording_id=recording_id, experiment=experiment)
        columns = [
            "behavior_filename", "behavior_bytes", "svd_filename", "svd_bytes",
            "retinotopy_filename", "retinotopy_bytes", "full_neural_filename", "full_neural_bytes",
        ]
        rows = []
        for row in manifest[columns].drop_duplicates().itertuples(index=False):
            for layer, filename, size in (
                ("behavior", row.behavior_filename, row.behavior_bytes),
                ("svd", row.svd_filename, row.svd_bytes),
                ("retinotopy", row.retinotopy_filename, row.retinotopy_bytes),
                ("full_neural", row.full_neural_filename, row.full_neural_bytes),
            ):
                rows.append({"layer": layer, "filename": filename, "size_bytes": size})
        plan = pd.DataFrame(rows).dropna().drop_duplicates().sort_values("size_bytes", ascending=False)
        plan["size_gib"] = plan["size_bytes"] / 2**30
        return plan.reset_index(drop=True)

    def load(
        self,
        recording_id,
        experiment,
        *,
        behavior_key=None,
        include_full=False,
        max_gib=1.0,
        max_full_gib=8.0,
    ):
        manifest = self.manifest(
            recording_id=recording_id, experiment=experiment, behavior_key=behavior_key
        )
        if len(manifest) != 1:
            choices = manifest[["experiment", "recording_id", "behavior_key", "stimulus_type"]]
            raise ValueError(
                f"Expected one behavior instance, found {len(manifest)}. "
                f"Choose behavior_key explicitly from:\n{choices.to_string(index=False)}"
            )
        selected = manifest.iloc[0]
        source = self.db.source
        behavior_bundle = source.load(str(selected["behavior_filename"]), max_gib=max_gib)
        key = str(selected["behavior_key"])
        if key not in behavior_bundle:
            raise KeyError(f"Behavior key {key!r} is absent from {selected['behavior_filename']}")
        svd = source.load(str(selected["svd_filename"]), max_gib=max_gib)
        retinotopy = source.load(str(selected["retinotopy_filename"]), max_gib=max_gib)
        full_payload = None
        if include_full:
            full_payload = source.load(str(selected["full_neural_filename"]), max_gib=max_full_gib)
        return JoinedRecording(
            metadata=selected.to_dict(),
            behavior=behavior_bundle[key],
            U=np.asarray(svd["U"]),
            V=np.asarray(svd["V"]),
            retinotopy=retinotopy,
            full_payload=full_payload,
        )


inspector = DatasetInspector(db)
print("DatasetInspector ready")
'''
        ),
        md(
            """
## Exhaustive joined manifest

This is the complete file-level access surface: one row per behavior instance,
including all Test 3 swaps, with the four imaging layers resolved. It is small
enough to inspect, filter, export, or query directly.
"""
        ),
        py(
            r'''
acquisition_manifest = inspector.manifest()
assert len(acquisition_manifest) == 142
for column in (
    "behavior_filename", "svd_filename", "full_neural_filename", "retinotopy_filename"
):
    assert acquisition_manifest[column].notna().all(), column
db.register("acquisition_manifest", acquisition_manifest)
display(acquisition_manifest)
'''
        ),
        py(
            r'''
# Every deposited file is inspectable here. Filter in Pandas or SQL before loading.
files = db.table("files")
assert len(files) == 297
display(files)

# The three behavior-only Figure 5 bundles have no imaging join by design.
behavior_only = db.query("""
SELECT filename, experiment, size_bytes, round(size_gib, 3) AS size_gib, md5, relative_path
FROM files
WHERE category = 'faster_learning_behavior'
ORDER BY experiment
""")
display(behavior_only)
'''
        ),
        md(
            """
## Interactive metadata browser

The controls below browse all cohorts, mice, acquisitions, experiment labels,
and resolved files without loading any arrays. Select a row here, then copy its
identifiers into the loading cell in the next section.
"""
        ),
        py(
            r'''
import ipywidgets as widgets

cohort_control = widgets.Dropdown(
    options=["all"] + sorted(mice["primary_cohort"].unique().tolist()),
    description="Cohort",
)
mouse_control = widgets.Dropdown(description="Mouse")
recording_control = widgets.Dropdown(description="Recording", layout=widgets.Layout(width="520px"))
output = widgets.Output()


def update_mice(*_):
    subset = mice if cohort_control.value == "all" else mice[mice["primary_cohort"].eq(cohort_control.value)]
    mouse_control.options = subset["mouse"].tolist()


def update_recordings(*_):
    subset = recordings[recordings["mouse"].eq(mouse_control.value)]
    recording_control.options = subset["recording_id"].tolist()


def refresh_browser(*_):
    with output:
        clear_output(wait=True)
        if not recording_control.value:
            return
        manifest = inspector.manifest(recording_id=recording_control.value)
        display(manifest)
        display(inspector.download_plan(recording_control.value).style.format({"size_bytes": "{:,}", "size_gib": "{:.3f}"}))


cohort_control.observe(update_mice, names="value")
mouse_control.observe(update_recordings, names="value")
recording_control.observe(refresh_browser, names="value")
update_mice()
update_recordings()
refresh_browser()
display(widgets.VBox([widgets.HBox([cohort_control, mouse_control]), recording_control, output]))
'''
        ),
        md(
            """
## Load one joined recording

The default loads behavior, SVD, and retinotopy only. The exact full-neural file
remains visible in the download plan and is loaded only when
`INCLUDE_FULL_NEURAL = True`.

For Test 3 swap sessions, set `BEHAVIOR_KEY` explicitly to the `_swap1` or
`_swap2` key shown by `inspector.manifest(...)`.
"""
        ),
        py(
            r'''
#@title 2 · Choose one behavior instance { display-mode: "form" }
RECORDING_ID = "TX108_2023_03_13_1"  #@param {type:"string"}
EXPERIMENT = "sup_train1_before_learning"  #@param {type:"string"}
BEHAVIOR_KEY = ""  #@param {type:"string"}
INCLUDE_FULL_NEURAL = False  #@param {type:"boolean"}

selected_manifest = inspector.manifest(
    recording_id=RECORDING_ID,
    experiment=EXPERIMENT,
    behavior_key=BEHAVIOR_KEY or None,
)
display(selected_manifest)
display(
    inspector.download_plan(RECORDING_ID, EXPERIMENT)
    .style.format({"size_bytes": "{:,}", "size_gib": "{:.3f}"})
)
'''
        ),
        py(
            r'''
#@title 3 · Load and validate the selected arrays { display-mode: "form" }
session = inspector.load(
    RECORDING_ID,
    EXPERIMENT,
    behavior_key=BEHAVIOR_KEY or None,
    include_full=INCLUDE_FULL_NEURAL,
)
display(session.describe())
'''
        ),
        py(
            r'''
# The three tidy coordinate tables are now ordinary DataFrames.
frames = session.frames
trials = session.trials
neurons = session.neurons

assert len(frames) == session.V.shape[1]
assert len(neurons) == session.U.shape[1] == len(session.iarea) == len(session.xy_t)

for name, frame in {
    "frames_current": frames,
    "trials_current": trials,
    "neurons_current": neurons,
}.items():
    db.register(name, frame)

display(frames.head(12))
display(trials.head(12))
display(neurons.head(12))
'''
        ),
        md(
            """
## Inspect the selected acquisition

The following views stay bounded: they show behavior on the frame axis,
retinotopy on the neuron axis, and a small SVD reconstruction connecting both.
"""
        ),
        py(
            r'''
# Behavior timeline: neural frame index is the shared coordinate.
window = frames.iloc[: min(5000, len(frames))]
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
axes[0].plot(window["frame_id"], window["position_m"], lw=1, color="#264653")
axes[0].set_ylabel("position (m)")
axes[1].plot(window["frame_id"], window["run_speed"], lw=1, color="#e76f51")
axes[1].set_ylabel("run speed")
axes[2].scatter(
    window["frame_id"], window["stimulus_role"],
    c=window["stimulus_role"], cmap="viridis", s=5,
)
axes[2].set(ylabel="stimulus role", xlabel="neural frame_id")
fig.suptitle(f"{session.recording_id} · frame-aligned behavior")
plt.tight_layout()
'''
        ),
        py(
            r'''
# Retinotopy: one dot per neuron; neuron_id is shared with the U axis.
area_order = ["V1", "mHV", "lHV", "aHV", "excluded"]
area_colors = {
    "V1": "#e76f51", "mHV": "#2a9d8f", "lHV": "#457b9d",
    "aHV": "#e9c46a", "excluded": "#adb5bd",
}
fig, ax = plt.subplots(figsize=(7, 6))
for area in area_order:
    subset = neurons[neurons["area_group"].eq(area)]
    if subset.empty:
        continue
    ax.scatter(
        subset["cortical_x"], subset["cortical_y"], s=2, alpha=0.45,
        label=f"{area} ({len(subset):,})", color=area_colors[area], rasterized=True,
    )
ax.set(xlabel="cortical x", ylabel="cortical y", title=f"{session.recording_id} · retinotopy")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(markerscale=4, frameon=False)
plt.tight_layout()
'''
        ),
        py(
            r'''
# SVD state and a bounded reconstruction.
frame_ids = np.arange(min(2500, session.V.shape[1]), dtype=np.int64)
v1_ids = neurons.loc[neurons["area_group"].eq("V1"), "neuron_id"].head(24).to_numpy()
if len(v1_ids) == 0:
    v1_ids = neurons["neuron_id"].head(24).to_numpy()

activity, used_neurons, used_frames = session.activity(v1_ids, frame_ids)
activity_z = (activity - activity.mean(axis=1, keepdims=True)) / np.maximum(
    activity.std(axis=1, keepdims=True), 1e-8
)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].imshow(session.V[:20, frame_ids], aspect="auto", cmap="coolwarm", interpolation="nearest")
axes[0].set(ylabel="PC", title="First 20 component amplitudes V[:, frame]")
axes[1].imshow(activity_z, aspect="auto", cmap="magma", interpolation="nearest")
axes[1].set(ylabel="selected neurons", xlabel="frame", title="U[:, neurons].T @ V[:, frames] (row z-score)")
plt.tight_layout()
'''
        ),
        py(
            r'''
# A literal all-modality join, deliberately limited to a small activity subset.
labeled_frame_ids = frames.loc[
    frames["stimulus_role"].isin([0, 2]) & frames["is_moving"] & frames["in_texture"],
    "frame_id",
].head(80).to_numpy(dtype=np.int64)
small_neuron_ids = neurons.loc[neurons["area_group"].eq("V1"), "neuron_id"].head(8).to_numpy()
if len(small_neuron_ids) == 0:
    small_neuron_ids = neurons["neuron_id"].head(8).to_numpy()

activity_long = session.activity_long(small_neuron_ids, labeled_frame_ids)
db.register("activity_current", activity_long)

all_modalities = db.query("""
SELECT
    a.recording_id, a.neuron_id, a.frame_id, a.activity,
    n.area_id, n.area_group, n.cortical_x, n.cortical_y,
    f.trial_id, f.position_m, f.is_moving, f.in_texture,
    f.trial_wall, f.stimulus_role, f.run_speed
FROM activity_current AS a
JOIN neurons_current AS n USING (recording_id, neuron_id)
JOIN frames_current AS f USING (recording_id, frame_id)
ORDER BY a.neuron_id, a.frame_id
""")
display(all_modalities.head(40))
'''
        ),
        md(
            """
## Practical query gallery

The examples below demonstrate the intended division of labor: SQL chooses
identities and compact labels, Pandas reshapes coordinate tables, and NumPy
computes dense activity.
"""
        ),
        py(
            r'''
# 1. Schema; 2. largest files; 3. files for the current acquisition.
display(db.schema())
display(db.query("SELECT filename, category, round(size_gib, 3) AS size_gib FROM files ORDER BY size_bytes DESC LIMIT 15"))
display(inspector.recording(RECORDING_ID))
'''
        ),
        py(
            r'''
# 4. Moving leaf1 frames between 2 and 3 metres; 5. trial summaries.
leaf_mid = frames.loc[
    frames["is_moving"]
    & frames["in_texture"]
    & frames["stimulus_role"].eq(2)
    & frames["position_m"].between(2.0, 3.0)
]
display(leaf_mid.head(20))
display(
    trials.groupby("stimulus_role", dropna=False)
    .agg(trials=("trial_id", "nunique"), mean_frames=("n_frames", "mean"), mean_speed=("mean_speed", "mean"))
    .reset_index()
)
'''
        ),
        py(
            r'''
# 6. Neurons by area; 7. frame/trial coverage by role; 8. SQL over registered tables.
display(neurons["area_group"].value_counts(dropna=False).rename_axis("area_group").reset_index(name="neurons"))
display(db.query("""
SELECT stimulus_role, count(*) AS frames, count(DISTINCT trial_id) AS trials,
       avg(run_speed) AS mean_speed, avg(CAST(is_moving AS INTEGER)) AS moving_fraction
FROM frames_current
GROUP BY stimulus_role
ORDER BY stimulus_role
"""))
display(db.query("""
SELECT n.area_group, count(*) AS neurons
FROM neurons_current AS n
GROUP BY n.area_group
ORDER BY neurons DESC
"""))
'''
        ),
        py(
            r'''
# 9. One frame's population state; 10. selected V1 activity at that frame.
frame_id = min(1000, session.V.shape[1] - 1)
population_state = session.V[:, frame_id]
v1_all = neurons.loc[neurons["area_group"].eq("V1"), "neuron_id"].to_numpy()
v1_at_frame = session.U[:, v1_all].T @ population_state

display(pd.Series(population_state, name="component_amplitude").to_frame().head(20))
display(pd.DataFrame({"neuron_id": v1_all, "activity": v1_at_frame}).head(20))
'''
        ),
        py(
            r'''
# 11. Circle1/leaf1 masks for a later d-prime analysis; 12. a download budget.
usable = frames["is_moving"] & frames["in_texture"]
circle1_mask = usable & frames["stimulus_role"].eq(0)
leaf1_mask = usable & frames["stimulus_role"].eq(2)
display(pd.DataFrame({
    "role": ["circle1", "leaf1"],
    "frames": [int(circle1_mask.sum()), int(leaf1_mask.sum())],
    "trials": [
        frames.loc[circle1_mask, "trial_id"].nunique(),
        frames.loc[leaf1_mask, "trial_id"].nunique(),
    ],
}))
display(inspector.download_plan(RECORDING_ID, EXPERIMENT).style.format({"size_bytes": "{:,}", "size_gib": "{:.3f}"}))
'''
        ),
        md(
            """
## Optional exact full-neural access

Only use this after checking `inspector.download_plan(...)`. Reload with
`include_full=True`, then call `full_matrix()` to concatenate the released
plane-wise `spks` arrays along the neuron axis. This can require several GiB of
RAM and is intentionally not part of the default run.
"""
        ),
        py(
            r'''
# Deliberately disabled. Set to True only after checking the file size above.
LOAD_EXACT_FULL_NEURAL = False

if LOAD_EXACT_FULL_NEURAL:
    exact_session = inspector.load(
        RECORDING_ID,
        EXPERIMENT,
        behavior_key=BEHAVIOR_KEY or None,
        include_full=True,
        max_full_gib=8.0,
    )
    N = exact_session.full_matrix()  # neurons × frames
    print("Exact full-neural matrix:", N.shape, N.dtype, f"{N.nbytes / 2**30:.2f} GiB in memory")
else:
    print("Full neural remains unloaded. The complete metadata catalog is still available.")
'''
        ),
        md(
            """
## What is now available

- `db`: the DuckDB/Pandas release catalog.
- `inspector`: exhaustive file/acquisition resolution plus lazy loading.
- `acquisition_manifest`: all 142 imaging behavior instances with every layer resolved.
- `files`: all 297 deposited files.
- `session`: one validated behavior + SVD + retinotopy acquisition.
- `frames`, `trials`, `neurons`: tidy coordinate tables for the selected acquisition.
- `session.U`, `session.V`: the dense 400-component neural representation.
- `all_modalities`: a safe, bounded example of neural activity joined to both axes.

Use `mouse` for longitudinal inference, `recording_id` for physical acquisition
identity, `frame_id` for behavior/neural alignment, and `neuron_id` for
retinotopy/neural alignment. Neuron identities do not persist across dates.
"""
        ),
    ]

    for index, cell in enumerate(notebook.cells):
        cell.id = f"dataset-inspector-{index:03d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


def main():
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
