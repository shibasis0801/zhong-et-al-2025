#!/usr/bin/env python3
"""Generate the filesystem-only neural/behavior/retinotopy join notebook."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/11_join_neural_behavior_retinotopy_colab.ipynb")


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
# Join neural, behavioral, and retinotopy data—without hidden APIs

This notebook treats Google Drive as a filesystem. It uses only:

- `google.colab.drive.mount(...)` and `pathlib.Path` for files;
- Pandas and DuckDB for catalogs, keys, and compact joins;
- NumPy for the dense neural matrices and d-prime calculations.

It deliberately does **not** import the workspace `drive.py` or `sql.py`, and
does not create `Dataset`, `Recording`, `Experiment`, or `DataFile` objects.

The central idea is small: behavior annotates the neural **frame axis**, while
retinotopy annotates the neural **neuron axis**. There is no reason to explode
the complete neuron × frame matrix into a gigantic SQL table.

The final figures keep neural frames on their acquisition timeline, draw the
exact circle1/leaf1 trial spans, give every usable trial one held-out neural
evidence observation, and attach d-prime only to segments with repeated trials
of both roles.
"""
        ),
        md(
            """
## The four grains and their keys

| Table / array | One row/value means | Join key |
|---|---|---|
| `experiment_rows` | one released experiment-to-acquisition association | `experiment`, `recording_id`, `behavior_key` |
| `frames` | one behavior/neural frame inside one acquisition | `recording_id`, `frame_id` |
| `trials` | one corridor traversal | `recording_id`, `trial_id` |
| `neurons` | one detected neuron inside one acquisition | `recording_id`, `neuron_id` |
| `V[:, frame_id]` | the 400-component neural state at a frame | frame axis |
| `U[:, neuron_id]` | the 400-component weights for a neuron | neuron axis |

At the file level, behavior joins by `experiment`, SVD/full neural joins by
`recording_id = mouse_date_block`, and retinotopy joins by
`retinotopy_id = mouse_date`. Test 3 has a crucial exception inside behavior
bundles: `behavior_key = recording_id + '_' + stimtype` for `swap1`/`swap2`.
The raw 142 source rows must therefore be preserved; deduplicating to the 133
experiment–recording pairs loses real behavior instances.
"""
        ),
        py(
            r'''
#@title Mount Drive and locate the verified release { display-mode: "form" }
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    import duckdb
    import pandas as pd
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "duckdb>=1.4,<2", "pandas>=2.2,<3",
    ])
    import duckdb
    import pandas as pd

# The SVD .npy contains a trusted serialized sklearn object even though this
# notebook reads only U and V. Match the environment that wrote the release.
try:
    import sklearn
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "scikit-learn==1.6.0",
    ])

override = os.environ.get("ZHONG2025_DATA_ROOT")
if override:
    DATA_ROOT = Path(override).expanduser().resolve()
else:
    from google.colab import drive as google_drive

    google_drive.mount("/content/drive", force_remount=False)
    my_drive = Path("/content/drive/MyDrive")
    candidates = [
        my_drive / "Zhong2025_Janelia_v2",
        my_drive
        / "Zhong et al. 2025 - Neuromatch Team Workspace"
        / "Janelia dataset - Zhong et al. 2025 (Figshare v2)",
    ]
    DATA_ROOT = next(
        (path for path in candidates if (path / "metadata/catalog.csv").is_file()),
        None,
    )
    if DATA_ROOT is None:
        tried = "\n".join(f"- {path}" for path in candidates)
        raise FileNotFoundError(
            "Add the team workspace (or Zhong2025_Janelia_v2) as a My Drive "
            f"shortcut. Tried:\n{tried}"
        )

verified = json.loads((DATA_ROOT / "VERIFIED.json").read_text())
assert verified["state"] == "complete"
assert verified["article_id"] == 28811129 and verified["version"] == 2
assert verified["file_count"] == 297
assert verified["total_bytes"] == 452_233_500_962

catalog = pd.read_csv(
    DATA_ROOT / "metadata/catalog.csv",
    keep_default_na=False,
    dtype={
        "name": "string", "category": "string", "experiment": "string",
        "recording_id": "string", "retinotopy_id": "string",
        "relative_path": "string", "md5": "string", "url": "string",
    },
)
catalog["size_bytes"] = pd.to_numeric(catalog["size_bytes"], downcast=None)
assert len(catalog) == 297
assert int(catalog["size_bytes"].sum()) == 452_233_500_962

print("Dataset root:", DATA_ROOT)
print("Verified files:", len(catalog), "· bytes:", int(catalog["size_bytes"].sum()))
'''
        ),
        md(
            """
## Normalize the released experiment index

`Imaging_Exp_info.npy` is small metadata, but it is the bridge from scientific
experiment labels to physical acquisitions. We load it directly by its catalog
path and make one ordinary DataFrame row per source association.
"""
        ),
        py(
            r'''
import numpy as np


def plain(value):
    """Convert NumPy scalars/containers to ordinary Python values."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return plain(value.item())
        return [plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def optional_text(value):
    value = plain(value)
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value).strip()
    return text or None


def block_text(value):
    value = plain(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return str(int(number)) if number.is_integer() else str(value).strip()


index_row = catalog.loc[catalog["name"].eq("Imaging_Exp_info.npy")]
assert len(index_row) == 1
index_path = DATA_ROOT / index_row.iloc[0]["relative_path"]
raw_index = np.load(index_path, allow_pickle=True).item()

rows = []
for experiment, entries in raw_index.items():
    entries = [entries] if isinstance(entries, dict) else list(entries)
    for source_row, source in enumerate(entries):
        source = plain(source)
        mouse = str(source["mname"])
        date = str(source["datexp"])
        block = block_text(source["blk"])
        recording_id = f"{mouse}_{date}_{block}"
        retinotopy_id = f"{mouse}_{date}"
        stimulus_type = optional_text(source.get("stimtype"))
        behavior_key = (
            f"{recording_id}_{stimulus_type}" if stimulus_type else recording_id
        )
        rows.append({
            "experiment": str(experiment),
            "source_row": int(source_row),
            "mouse": mouse,
            "date": date,
            "block": block,
            "recording_id": recording_id,
            "retinotopy_id": retinotopy_id,
            "behavior_key": behavior_key,
            "stimulus_type": stimulus_type,
            "reward_type": optional_text(source.get("rewType")),
            "stimulus_ids_json": json.dumps(plain(source.get("stim_id", []))),
        })

experiment_rows = pd.DataFrame(rows).sort_values(
    ["experiment", "recording_id", "source_row"], ignore_index=True
)

assert len(experiment_rows) == 142
assert len(experiment_rows.drop_duplicates(["experiment", "recording_id"])) == 133
assert len(experiment_rows.drop_duplicates(["experiment", "behavior_key"])) == 142

experiment_rows.groupby("experiment", sort=True).agg(
    source_rows=("behavior_key", "size"),
    behavior_instances=("behavior_key", "nunique"),
    recordings=("recording_id", "nunique"),
).reset_index()
'''
        ),
        py(
            r'''
# DuckDB sees the same ordinary Pandas tables shown above.
db = duckdb.connect(database=":memory:")
db.register("catalog", catalog)
db.register("experiment_rows", experiment_rows)

db.execute("""
    SELECT
        count(*) AS source_rows,
        count(DISTINCT experiment) AS experiments,
        count(DISTINCT mouse) AS mice,
        count(DISTINCT recording_id) AS physical_recordings,
        count(DISTINCT experiment || '|' || recording_id) AS memberships,
        count(DISTINCT experiment || '|' || behavior_key) AS behavior_instances
    FROM experiment_rows
""").df()
'''
        ),
        md(
            """
## File-level join: one SQL manifest row

This is the only file-resolution query needed for a recording:

- behavior catalog row: `catalog.experiment = experiment_rows.experiment`;
- reduced/full neural rows: `catalog.recording_id = recording_id`;
- retinotopy row: `catalog.retinotopy_id = retinotopy_id`.

The default below is the same real session used in the exploratory notebook.
Change all three selectors together for a Test-3 swap instance.
"""
        ),
        py(
            r'''
#@title Choose one experiment/acquisition { display-mode: "form" }
EXPERIMENT = "sup_train1_before_learning"  #@param {type:"string"}
RECORDING_ID = "TX108_2023_03_13_1"  #@param {type:"string"}
BEHAVIOR_KEY = "TX108_2023_03_13_1"  #@param {type:"string"}

manifest_sql = """
    SELECT
        e.experiment, e.source_row, e.mouse, e.date, e.block,
        e.recording_id, e.retinotopy_id, e.behavior_key, e.stimulus_type,
        b.relative_path AS behavior_path,
        s.relative_path AS svd_path,
        n.relative_path AS full_neural_path,
        r.relative_path AS retinotopy_path,
        b.size_bytes AS behavior_bytes,
        s.size_bytes AS svd_bytes,
        n.size_bytes AS full_neural_bytes,
        r.size_bytes AS retinotopy_bytes
    FROM experiment_rows AS e
    JOIN catalog AS b
      ON b.category = 'imaging_behavior'
     AND b.experiment = e.experiment
    JOIN catalog AS s
      ON s.category = 'reduced_neural'
     AND s.recording_id = e.recording_id
    JOIN catalog AS n
      ON n.category = 'full_neural'
     AND n.recording_id = e.recording_id
    JOIN catalog AS r
      ON r.category = 'retinotopy'
     AND r.retinotopy_id = e.retinotopy_id
    WHERE e.experiment = ?
      AND e.recording_id = ?
      AND e.behavior_key = ?
"""

manifest = db.execute(
    manifest_sql, [EXPERIMENT, RECORDING_ID, BEHAVIOR_KEY]
).df()
if len(manifest) != 1:
    alternatives = db.execute("""
        SELECT experiment, recording_id, behavior_key, stimulus_type
        FROM experiment_rows
        WHERE experiment = ? AND recording_id = ?
        ORDER BY source_row
    """, [EXPERIMENT, RECORDING_ID]).df()
    raise ValueError(
        f"Expected one manifest row, found {len(manifest)}. "
        f"Valid behavior keys:\n{alternatives.to_string(index=False)}"
    )

manifest.assign(
    behavior_gib=manifest["behavior_bytes"] / 2**30,
    svd_gib=manifest["svd_bytes"] / 2**30,
    full_neural_gib=manifest["full_neural_bytes"] / 2**30,
    retinotopy_gib=manifest["retinotopy_bytes"] / 2**30,
)
'''
        ),
        md(
            """
## Load the selected files directly

Only the behavior, SVD, and retinotopy files are opened. The multi-GiB full
neural file stays visible in the manifest but is not loaded. `allow_pickle=True`
is used only for these checksum-verified release files.
"""
        ),
        py(
            r'''
selected = manifest.iloc[0]


def release_path(column):
    path = DATA_ROOT / str(selected[column])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


behavior_bundle = np.load(
    release_path("behavior_path"), allow_pickle=True
).item()
if selected["behavior_key"] not in behavior_bundle:
    raise KeyError(
        f"{selected['behavior_key']!r} is not in the selected behavior bundle"
    )
behavior = behavior_bundle[selected["behavior_key"]]
svd = np.load(release_path("svd_path"), allow_pickle=True).item()
with np.load(release_path("retinotopy_path"), allow_pickle=True) as archive:
    retinotopy = {name: archive[name] for name in archive.files}

U = np.asarray(svd["U"])
V = np.asarray(svd["V"])
iarea = np.asarray(retinotopy["iarea"]).squeeze()
xy_t = np.asarray(retinotopy["xy_t"])

{
    "behavior_fields": sorted(behavior),
    "U_components_x_neurons": U.shape,
    "V_components_x_frames": V.shape,
    "iarea_neurons": iarea.shape,
    "xy_t_neurons_x_2": xy_t.shape,
}
'''
        ),
        md(
            """
## Build the frame, trial, role, and neuron tables

The join is positional only after explicit validation. Neural frames are never
discarded. We permit at most three extra **trailing behavior frames**, matching
the bounded release mismatch already observed in this workspace.
"""
        ),
        py(
            r'''
MAX_TRAILING_BEHAVIOR_FRAMES = 3


def one_dimensional(name, values):
    array = np.asarray(values).squeeze()
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


frame_fields = {
    "trial_raw": one_dimensional("ft_trInd", behavior["ft_trInd"]),
    "wall_at_frame": one_dimensional("ft_WallID", behavior["ft_WallID"]),
    "position_dm": one_dimensional("ft_Pos", behavior["ft_Pos"]),
    "move": one_dimensional(
        "ft_move",
        behavior["ft_move"] if "ft_move" in behavior else behavior["ft_isMoving"],
    ),
    "in_texture": one_dimensional("ft_CorrSpc", behavior["ft_CorrSpc"]),
    "run_speed": one_dimensional("ft_RunSpeed", behavior["ft_RunSpeed"]),
}
if "ft" in behavior:
    frame_fields["time"] = one_dimensional("ft", behavior["ft"])

behavior_lengths = {name: len(values) for name, values in frame_fields.items()}
if len(set(behavior_lengths.values())) != 1:
    raise ValueError(f"Behavior frame fields disagree: {behavior_lengths}")

if U.ndim != 2 or V.ndim != 2 or U.shape[0] != V.shape[0]:
    raise ValueError("U and V must be components×neurons and components×frames")
if xy_t.ndim != 2 or xy_t.shape[1] != 2:
    raise ValueError("xy_t must have shape neurons×2")
if U.shape[1] != len(iarea) or U.shape[1] != xy_t.shape[0]:
    raise ValueError("U, iarea, and xy_t do not share one neuron axis")

behavior_frames = next(iter(behavior_lengths.values()))
neural_frames = V.shape[1]
trailing_excess = behavior_frames - neural_frames
if trailing_excess < 0 or trailing_excess > MAX_TRAILING_BEHAVIOR_FRAMES:
    raise ValueError(
        f"Behavior has {behavior_frames} frames and V has {neural_frames}; "
        f"allowed trailing behavior excess is 0..{MAX_TRAILING_BEHAVIOR_FRAMES}"
    )
aligned = {name: values[:neural_frames] for name, values in frame_fields.items()}

wall_by_trial = one_dimensional("WallName", behavior["WallName"])
trial_raw = np.asarray(aligned["trial_raw"], dtype=float)
integer_trial = np.isfinite(trial_raw) & np.isclose(trial_raw, np.round(trial_raw))
trial_integer = np.full(neural_frames, -1, dtype=np.int64)
trial_integer[integer_trial] = np.round(trial_raw[integer_trial]).astype(np.int64)
valid_trial = integer_trial & (trial_integer >= 0) & (trial_integer < len(wall_by_trial))
trial_nullable = pd.array(
    np.where(valid_trial, trial_integer, np.nan), dtype="Int64"
)

frames = pd.DataFrame({
    "recording_id": selected["recording_id"],
    "experiment": selected["experiment"],
    "frame_id": np.arange(neural_frames, dtype=np.int64),
    "trial_id": trial_nullable,
    "valid_trial": valid_trial,
    "wall_at_frame": np.asarray(aligned["wall_at_frame"]).astype(str),
    "position_m": np.asarray(aligned["position_dm"], dtype=float) / 10.0,
    "is_moving": np.asarray(aligned["move"]) > 0,
    "in_texture": np.asarray(aligned["in_texture"], dtype=bool),
    "run_speed": np.asarray(aligned["run_speed"], dtype=float),
})
if "time" in aligned:
    frames["time"] = np.asarray(aligned["time"])

trials = pd.DataFrame({
    "recording_id": selected["recording_id"],
    "trial_id": np.arange(len(wall_by_trial), dtype=np.int64),
    "wall_name": np.asarray(wall_by_trial).astype(str),
})

unique_walls = one_dimensional("UniqWalls", behavior["UniqWalls"])
stimulus_ids = one_dimensional("stim_id", behavior["stim_id"])
if len(unique_walls) != len(stimulus_ids):
    raise ValueError("UniqWalls and stim_id must be parallel arrays")
wall_role_rows = []
for wall, role in zip(unique_walls, stimulus_ids):
    try:
        finite = bool(np.isfinite(role))
    except TypeError:
        finite = False
    if finite:
        wall_role_rows.append({
            "recording_id": selected["recording_id"],
            "wall_name": str(wall),
            "stimulus_role": int(role),
        })
wall_roles = pd.DataFrame(wall_role_rows)


def area_group(code):
    if code == 8:
        return "V1"
    if code in {0, 1, 2, 9}:
        return "mHV"
    if code in {5, 6}:
        return "lHV"
    if code in {3, 4}:
        return "aHV"
    return "excluded"


neurons = pd.DataFrame({
    "recording_id": selected["recording_id"],
    "neuron_id": np.arange(U.shape[1], dtype=np.int64),
    "area_id": iarea.astype(np.int16, copy=False),
    "area_group": [area_group(int(code)) for code in iarea],
    "cortical_x": -xy_t[:, 1],
    "cortical_y": xy_t[:, 0],
})

for name, table in {
    "frames": frames,
    "trials": trials,
    "wall_roles": wall_roles,
    "neurons": neurons,
}.items():
    db.register(name, table)

{
    "behavior_frames_before_alignment": behavior_frames,
    "neural_frames": neural_frames,
    "dropped_trailing_behavior_frames": trailing_excess,
    "invalid_or_unassigned_frames": int((~valid_trial).sum()),
    "trials": len(trials),
    "neurons": len(neurons),
}
'''
        ),
        py(
            r'''
# Frame → trial → functional stimulus role is a compact SQL join.
frame_observations = db.execute("""
    SELECT
        f.recording_id, f.experiment, f.frame_id, f.trial_id,
        f.position_m, f.is_moving, f.in_texture, f.run_speed,
        f.wall_at_frame, t.wall_name AS trial_wall,
        wr.stimulus_role
    FROM frames AS f
    LEFT JOIN trials AS t
      ON t.recording_id = f.recording_id
     AND t.trial_id = f.trial_id
    LEFT JOIN wall_roles AS wr
      ON wr.recording_id = t.recording_id
     AND wr.wall_name = t.wall_name
    ORDER BY f.frame_id
""").df()
db.register("frame_observations", frame_observations)

db.execute("""
    SELECT stimulus_role,
           count(*) AS frames,
           count(DISTINCT trial_id) AS trials,
           avg(run_speed) AS mean_speed
    FROM frame_observations
    WHERE is_moving AND in_texture
    GROUP BY stimulus_role
    ORDER BY stimulus_role
""").df()
'''
        ),
        md(
            """
## A literal all-modality join—kept deliberately small

Retinotopy cannot be joined directly onto frame rows: it lives on the neuron
axis. The bridge is neural activity. Below we reconstruct only four V1 neurons
and 25 labeled frames, then SQL joins `(neuron_id, frame_id)` activity to both
the frame/behavior and neuron/retinotopy tables.

Doing this for all neurons and all frames would recreate billions of rows. Keep
that dense object as `U.T @ V` (or the original full-neural array), not SQL.
"""
        ),
        py(
            r'''
v1_ids = neurons.loc[neurons["area_group"].eq("V1"), "neuron_id"].head(4).to_numpy()
if len(v1_ids) < 4:
    v1_ids = neurons["neuron_id"].head(4).to_numpy()
labeled_frames = frame_observations.loc[
    frame_observations["stimulus_role"].isin([0, 2]), "frame_id"
].head(25).to_numpy(dtype=np.int64)

small_activity = U[:, v1_ids].T @ V[:, labeled_frames]
activity_long = pd.DataFrame({
    "recording_id": selected["recording_id"],
    "neuron_id": np.repeat(v1_ids, len(labeled_frames)),
    "frame_id": np.tile(labeled_frames, len(v1_ids)),
    "activity": small_activity.reshape(-1),
})
db.register("activity_long", activity_long)

all_modalities = db.execute("""
    SELECT
        a.recording_id, a.neuron_id, a.frame_id, a.activity,
        n.area_id, n.area_group, n.cortical_x, n.cortical_y,
        f.trial_id, f.position_m, f.is_moving, f.in_texture,
        f.trial_wall, f.stimulus_role, f.run_speed
    FROM activity_long AS a
    JOIN neurons AS n
      ON n.recording_id = a.recording_id
     AND n.neuron_id = a.neuron_id
    JOIN frame_observations AS f
      ON f.recording_id = a.recording_id
     AND f.frame_id = a.frame_id
    ORDER BY a.neuron_id, a.frame_id
""").df()

assert len(all_modalities) == len(v1_ids) * len(labeled_frames)
all_modalities.head(12)
'''
        ),
        md(
            """
## Reduce frames to equal-weight trial × position responses

Raw frame pooling gives slow trials more weight. For a trial-indexed analysis,
we first average the 400-component `V` state within each trial and fixed
position bin. Empty bins remain missing; no interpolation crosses a trial
boundary. Retinotopy selects an area on the `U` neuron axis, then a small
area-specific transform turns the binned component states into 12 features.
"""
        ),
        py(
            r'''
N_POSITION_BINS = 18
AREA = "V1"  # one of V1, mHV, lHV, aHV
N_FEATURES = 12
AREA_IDS = {
    "V1": (8,), "mHV": (0, 1, 2, 9), "lHV": (5, 6), "aHV": (3, 4),
}

position = frame_observations["position_m"].to_numpy(dtype=float)
trial_values = frame_observations["trial_id"].astype("Float64").to_numpy(
    dtype=float, na_value=np.nan
)
valid = (
    frame_observations["is_moving"].to_numpy(dtype=bool)
    & frame_observations["in_texture"].to_numpy(dtype=bool)
    & np.isfinite(position)
    & np.isfinite(trial_values)
    & (position >= 0.0)
    & (position <= 6.0)
)

edges_m = np.linspace(0.0, 6.0, N_POSITION_BINS + 1)
selected_frames = np.flatnonzero(valid)
selected_trials = trial_values[valid].astype(np.int64)
selected_positions = position[valid]
bin_id = np.searchsorted(edges_m, selected_positions, side="right") - 1
bin_id[selected_positions == edges_m[-1]] = N_POSITION_BINS - 1
if np.any((bin_id < 0) | (bin_id >= N_POSITION_BINS)):
    raise ValueError("A selected frame fell outside the position bins")

trial_ids = np.unique(selected_trials)
trial_offset = np.searchsorted(trial_ids, selected_trials)
group_id = trial_offset * N_POSITION_BINS + bin_id
n_groups = len(trial_ids) * N_POSITION_BINS

component_sums = np.zeros((n_groups, V.shape[0]), dtype=np.float64)
np.add.at(component_sums, group_id, V[:, selected_frames].T)
frame_counts = np.bincount(group_id, minlength=n_groups).reshape(
    len(trial_ids), N_POSITION_BINS
)
component_counts = frame_counts.reshape(-1, 1)
binned_v = np.full_like(component_sums, np.nan)
np.divide(
    component_sums, component_counts,
    out=binned_v, where=component_counts > 0,
)
binned_v = binned_v.reshape(
    len(trial_ids), N_POSITION_BINS, V.shape[0]
).astype(np.float32)

area_mask = np.isin(iarea, AREA_IDS[AREA])
if not np.any(area_mask):
    raise ValueError(f"No neurons found for {AREA}")
area_weights = np.asarray(U[:, area_mask], dtype=np.float64)
gram = area_weights @ area_weights.T
eigenvalues, eigenvectors = np.linalg.eigh(gram)
keep = np.argsort(eigenvalues)[::-1][:N_FEATURES]
area_transform = eigenvectors[:, keep] * np.sqrt(
    np.maximum(eigenvalues[keep], 0.0)
)[None, :]
trial_features = np.einsum(
    "tbp,pk->tbk", binned_v, area_transform, optimize=True
).astype(np.float32)

trial_labels = db.execute("""
    SELECT t.trial_id, t.wall_name, wr.stimulus_role
    FROM trials AS t
    LEFT JOIN wall_roles AS wr
      ON wr.recording_id = t.recording_id
     AND wr.wall_name = t.wall_name
    ORDER BY t.trial_id
""").df().set_index("trial_id")
labels = trial_labels.reindex(trial_ids)["stimulus_role"].fillna(-1).to_numpy(
    dtype=np.int16
)

texture_bins = edges_m[1:] <= 4.0
chosen = trial_features[:, texture_bins, :]
complete_coverage = np.isfinite(chosen).all(axis=(1, 2))
responses = np.full((len(trial_ids), N_FEATURES), np.nan, dtype=np.float64)
responses[complete_coverage] = np.mean(chosen[complete_coverage], axis=1)

trial_summary = pd.DataFrame({
    "recording_id": selected["recording_id"],
    "trial_id": trial_ids,
    "stimulus_role": labels,
    "complete_texture_coverage": complete_coverage,
    "selected_frame_count": frame_counts.sum(axis=1),
})
db.register("trial_summary", trial_summary)

{
    "binned_V_trials_x_position_x_components": binned_v.shape,
    "trial_features_trials_x_position_x_features": trial_features.shape,
    "complete_texture_trials": int(complete_coverage.sum()),
    "role_counts": trial_summary.groupby("stimulus_role").size().to_dict(),
}
'''
        ),
        md(
            """
## Put neural frames back underneath their physical trials

The analysis below keeps the acquisition timeline visible. Each colored span is
one physical trial; vertical lines are the exact frame boundaries from
`ft_trInd`. The heatmap reconstructs a deterministic, label-free sample of
neurons from the selected retinotopic area:

`U[:, neuron_ids].T @ V[:, frame_ids]`

Rows are z-scored only to make the heatmap readable. The d-prime calculation
does **not** use these display z-scores or cherry-pick neurons. A separate
support band marks frames satisfying both `(ft_move > 0)` and `ft_CorrSpc`.
Those are the frames allowed into the trial response; grey frames remain in the
timeline but not in the statistic.
"""
        ),
        py(
            r'''
# SQL makes one row per physical trial segment on the frame axis.
trial_segments = db.execute("""
    WITH labeled_frames AS (
        SELECT
            recording_id,
            CAST(trial_id AS BIGINT) AS trial_id,
            frame_id,
            stimulus_role,
            is_moving,
            in_texture
        FROM frame_observations
        WHERE trial_id IS NOT NULL
    ),
    segments AS (
        SELECT
            recording_id,
            trial_id,
            max(stimulus_role) AS stimulus_role,
            min(frame_id) AS start_frame,
            max(frame_id) AS stop_frame,
            avg(frame_id) AS mid_frame,
            count(*) AS total_frames,
            sum(CASE WHEN is_moving AND in_texture THEN 1 ELSE 0 END)
                AS valid_texture_frames
        FROM labeled_frames
        GROUP BY recording_id, trial_id
    )
    SELECT
        s.*,
        coalesce(t.complete_texture_coverage, false)
            AS complete_texture_coverage,
        coalesce(t.selected_frame_count, 0) AS selected_frame_count
    FROM segments AS s
    LEFT JOIN trial_summary AS t
      ON t.recording_id = s.recording_id
     AND t.trial_id = s.trial_id
    ORDER BY s.trial_id
""").df()
db.register("trial_segments", trial_segments)

# A trial must occupy one unbroken run of acquisition frames.
expected_lengths = (
    trial_segments["stop_frame"] - trial_segments["start_frame"] + 1
)
if not np.array_equal(
    expected_lengths.to_numpy(dtype=np.int64),
    trial_segments["total_frames"].to_numpy(dtype=np.int64),
):
    raise ValueError("At least one trial is split into non-contiguous frame runs")

# Display the first 12 chronological physical trials beginning at the first
# circle1/leaf1 trial. Do not search for an especially clean alternation.
role_is_target = trial_segments["stimulus_role"].isin([0, 2]).to_numpy()
target_positions = np.flatnonzero(role_is_target)
if len(target_positions) == 0:
    raise ValueError("No role-0/role-2 trials are available for the timeline")
DISPLAY_TRIALS = 12
display_start_row = int(target_positions[0])
display_segments = trial_segments.iloc[
    display_start_row:display_start_row + DISPLAY_TRIALS
].copy()

timeline_start_frame = int(display_segments["start_frame"].min())
timeline_stop_frame = int(display_segments["stop_frame"].max())
timeline_frame_ids = np.arange(
    timeline_start_frame, timeline_stop_frame + 1, dtype=np.int64
)
assert np.array_equal(
    timeline_frame_ids,
    frame_observations.loc[
        frame_observations["frame_id"].between(
            timeline_start_frame, timeline_stop_frame
        ),
        "frame_id",
    ].to_numpy(dtype=np.int64),
)

timeline_rows = (
    frame_observations.set_index("frame_id")
    .reindex(timeline_frame_ids)
    .reset_index()
)
timeline_role_values = timeline_rows["stimulus_role"].to_numpy(dtype=float)
timeline_role_code = np.where(
    timeline_role_values == 0, 1,
    np.where(timeline_role_values == 2, 2, 0),
).astype(np.int8)
timeline_analysis_valid = (
    timeline_rows["is_moving"].fillna(False).to_numpy(dtype=bool)
    & timeline_rows["in_texture"].fillna(False).to_numpy(dtype=bool)
    & np.isin(timeline_role_values, [0, 2])
)

# Deterministic, label-free neuron sample: evenly spaced IDs in the chosen area.
area_neuron_ids = np.flatnonzero(area_mask)
DISPLAY_NEURONS = min(24, len(area_neuron_ids))
display_neuron_positions = np.unique(np.linspace(
    0, len(area_neuron_ids) - 1, DISPLAY_NEURONS, dtype=np.int64
))
timeline_neuron_ids = area_neuron_ids[display_neuron_positions]
timeline_activity_raw = (
    U[:, timeline_neuron_ids].T @ V[:, timeline_frame_ids]
)
timeline_activity_mean = np.mean(timeline_activity_raw, axis=1, keepdims=True)
timeline_activity_sd = np.std(
    timeline_activity_raw, axis=1, ddof=1, keepdims=True
)
timeline_activity_sd[
    ~np.isfinite(timeline_activity_sd) | (timeline_activity_sd < 1e-9)
] = 1.0
timeline_activity_z = (
    timeline_activity_raw - timeline_activity_mean
) / timeline_activity_sd

trial_segments.head(), display_segments[
    ["trial_id", "stimulus_role", "start_frame", "stop_frame",
     "valid_texture_frames", "complete_texture_coverage"]
]
'''
        ),
        md(
            """
## One held-out observation per trial; one d-prime per multi-trial segment

A single trial contains one stimulus role, so it cannot have a classical
d-prime by itself. The honest one-point-per-trial result is **held-out neural
evidence**: positive is leaf1-like, negative is circle1-like, and the trial's
entire frame run is kept out while its coding direction is fitted.

Actual d-prime is calculated from a predeclared 40-trial segment containing
repeated role-2 (leaf1) and role-0 (circle1) trials. Each segment is divided into
four contiguous folds. Standardization and the coding direction are fitted on
the other folds, d-prime is calculated separately from each fold's held-out
test scores with sample SD (`ddof=1`), and the four fold d-primes are averaged.
Raw scores from separately fitted folds are never pooled.

Sliding the 40-trial segment by one trial produces a trailing d-prime value at
each physical trial after the first 39. Adjacent values share 39/40 trials, so
the line is descriptive and highly correlated. Non-overlapping 40-trial
segments are the safer inferential summaries. Both estimands are distinct from
the paper's whole-session, per-neuron, frame-pooled formula, which uses
`ddof=0` and `(ft_move > 0) & ft_CorrSpc`.

Do not compute primary d-prime from all frames in one circle trial versus all
frames in one leaf trial. Those frames are autocorrelated, ordered by corridor
position, and would give slow trials more weight.
"""
        ),
        py(
            r'''
def score_dprime(scores, score_labels, role_a=2, role_b=0):
    scores = np.asarray(scores, dtype=float)
    score_labels = np.asarray(score_labels)
    a = scores[score_labels == role_a]
    b = scores[score_labels == role_b]
    if min(len(a), len(b)) < 2:
        return np.nan
    spread = np.std(a, ddof=1) + np.std(b, ddof=1)
    return np.nan if not np.isfinite(spread) or spread <= 0 else float(
        2.0 * (np.mean(a) - np.mean(b)) / spread
    )


def heldout_trial_evidence(
    values, value_labels, physical_trials, n_folds=4, min_per_role=4
):
    """Return one score per held-out trial plus one d-prime per test fold."""
    values = np.asarray(values, dtype=float)
    value_labels = np.asarray(value_labels)
    physical_trials = np.asarray(physical_trials)
    if not (len(values) == len(value_labels) == len(physical_trials)):
        raise ValueError("values, labels, and physical trial IDs must align")

    fold = np.empty(len(values), dtype=np.int16)
    for fold_id, indices in enumerate(np.array_split(np.arange(len(values)), n_folds)):
        fold[indices] = fold_id

    usable = np.isin(value_labels, [0, 2]) & np.isfinite(values).all(axis=1)
    detail_rows = []
    fold_dprimes = []
    for fold_id in range(n_folds):
        train = usable & (fold != fold_id)
        test = usable & (fold == fold_id)
        train_labels = value_labels[train]
        test_labels = value_labels[test]
        if min(
            np.count_nonzero(train_labels == 2),
            np.count_nonzero(train_labels == 0),
        ) < min_per_role:
            fold_dprimes.append(np.nan)
            continue

        mean = np.mean(values[train], axis=0)
        scale = np.std(values[train], axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
        train_z = (values[train] - mean) / scale
        direction = (
            np.mean(train_z[train_labels == 2], axis=0)
            - np.mean(train_z[train_labels == 0], axis=0)
        )
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm <= 0:
            fold_dprimes.append(np.nan)
            continue

        unit_direction = direction / norm
        train_scores = train_z @ unit_direction
        train_role_2 = train_scores[train_labels == 2]
        train_role_0 = train_scores[train_labels == 0]
        midpoint = 0.5 * (np.mean(train_role_2) + np.mean(train_role_0))
        train_spread = (
            np.std(train_role_2, ddof=1) + np.std(train_role_0, ddof=1)
        )
        if not np.isfinite(train_spread) or train_spread <= 0:
            fold_dprimes.append(np.nan)
            continue

        test_indices = np.flatnonzero(test)
        test_scores = ((values[test] - mean) / scale) @ unit_direction
        # Dimensionless signed evidence uses training data only. It is one
        # observation, not a d-prime estimate for the individual trial.
        test_evidence = 2.0 * (test_scores - midpoint) / train_spread
        fold_dprimes.append(score_dprime(test_scores, test_labels))
        for local_index, score, evidence in zip(
            test_indices, test_scores, test_evidence
        ):
            detail_rows.append({
                "local_index": int(local_index),
                "trial_id": int(physical_trials[local_index]),
                "stimulus_role": int(value_labels[local_index]),
                "fold_id": int(fold_id),
                "heldout_score": float(score),
                "heldout_evidence": float(evidence),
                "is_held_out": True,
                "training_role_2_trials": int(
                    np.count_nonzero(train_labels == 2)
                ),
                "training_role_0_trials": int(
                    np.count_nonzero(train_labels == 0)
                ),
            })

    detail_columns = [
        "local_index", "trial_id", "stimulus_role", "fold_id",
        "heldout_score", "heldout_evidence", "is_held_out",
        "training_role_2_trials", "training_role_0_trials",
    ]
    return (
        pd.DataFrame(detail_rows, columns=detail_columns),
        np.asarray(fold_dprimes, dtype=float),
    )


def heldout_window(
    values, window_labels, physical_trials, n_folds=4, min_per_role=4
):
    details, fold_dprimes = heldout_trial_evidence(
        values,
        window_labels,
        physical_trials,
        n_folds=n_folds,
        min_per_role=min_per_role,
    )
    valid_folds = int(np.count_nonzero(np.isfinite(fold_dprimes)))
    if valid_folds != n_folds:
        return np.nan, valid_folds, details, fold_dprimes
    return (
        float(np.mean(fold_dprimes)),
        valid_folds,
        details,
        fold_dprimes,
    )


def dprime_trajectory(values, value_labels, physical_trials, window_trials, stride):
    rows = []
    fold_rows = []
    for segment_id, start in enumerate(
        range(0, len(physical_trials) - window_trials + 1, stride)
    ):
        stop = start + window_trials
        local_labels = value_labels[start:stop]
        local_trials = physical_trials[start:stop]
        dprime, valid_folds, _, fold_dprimes = heldout_window(
            values[start:stop], local_labels, local_trials
        )
        local_usable = (
            np.isin(local_labels, [0, 2])
            & np.isfinite(values[start:stop]).all(axis=1)
        )
        rows.append({
            "segment_id": int(segment_id),
            "start_trial": int(physical_trials[start]),
            "stop_trial": int(physical_trials[stop - 1]),
            "midpoint_trial": float(np.mean(physical_trials[start:stop])),
            "dprime": dprime,
            "role_2_trials": int(
                np.count_nonzero((local_labels == 2) & local_usable)
            ),
            "role_0_trials": int(
                np.count_nonzero((local_labels == 0) & local_usable)
            ),
            "valid_folds": int(valid_folds),
            "required_folds": 4,
        })
        for fold_id, fold_dprime in enumerate(fold_dprimes):
            fold_rows.append({
                "segment_id": int(segment_id),
                "fold_id": int(fold_id),
                "fold_dprime": float(fold_dprime),
                "start_trial": int(physical_trials[start]),
                "stop_trial": int(physical_trials[stop - 1]),
            })
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


WINDOW_TRIALS = 40
session_evidence, session_fold_dprimes = heldout_trial_evidence(
    responses, labels, trial_ids, n_folds=4
)
assert not session_evidence["trial_id"].duplicated().any()
assert session_evidence["is_held_out"].all()
db.register("trial_evidence", session_evidence)

trial_results = db.execute("""
    SELECT
        s.recording_id,
        s.trial_id,
        s.stimulus_role,
        CASE s.stimulus_role
            WHEN 0 THEN 'circle1'
            WHEN 2 THEN 'leaf1'
            ELSE 'other'
        END AS role_name,
        s.start_frame,
        s.stop_frame,
        s.mid_frame,
        s.total_frames,
        s.valid_texture_frames,
        s.complete_texture_coverage,
        e.fold_id,
        e.heldout_score,
        e.heldout_evidence,
        e.is_held_out
    FROM trial_segments AS s
    LEFT JOIN trial_evidence AS e
      ON e.trial_id = s.trial_id
    ORDER BY s.trial_id
""").df()
db.register("trial_results", trial_results)

exploratory, exploratory_folds = dprime_trajectory(
    responses, labels, trial_ids, window_trials=WINDOW_TRIALS, stride=1
)
nonoverlapping, nonoverlapping_folds = dprime_trajectory(
    responses, labels, trial_ids, window_trials=WINDOW_TRIALS, stride=WINDOW_TRIALS
)
db.register("dprime_trajectory", exploratory)
db.register("dprime_segments", nonoverlapping)
db.register("dprime_segment_folds", nonoverlapping_folds)

trial_results.head(), exploratory.head(), nonoverlapping
'''
        ),
        py(
            r'''
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Stable role encoding for both figures.
ROLE_COLORS = {0: "#6f58a8", 2: "#2f9e73"}
ROLE_NAMES = {0: "circle1", 2: "leaf1"}

# Figure 1: exactly the requested neural-frame → trial alignment.
fig = plt.figure(figsize=(14, 8), constrained_layout=True)
grid = fig.add_gridspec(4, 1, height_ratios=[0.65, 0.32, 4.8, 1.5])
ax_role = fig.add_subplot(grid[0])
ax_support = fig.add_subplot(grid[1], sharex=ax_role)
ax_neural = fig.add_subplot(grid[2], sharex=ax_role)
ax_trial = fig.add_subplot(grid[3], sharex=ax_role)
timeline_x_extent = [timeline_start_frame - 0.5, timeline_stop_frame + 0.5]

ax_role.imshow(
    timeline_role_code[np.newaxis, :],
    aspect="auto", interpolation="nearest", vmin=0, vmax=2,
    cmap=ListedColormap(["#c9c9c9", ROLE_COLORS[0], ROLE_COLORS[2]]),
    extent=[*timeline_x_extent, 0.5, -0.5],
)
ax_support.imshow(
    timeline_analysis_valid[np.newaxis, :].astype(np.int8),
    aspect="auto", interpolation="nearest", vmin=0, vmax=1,
    cmap=ListedColormap(["#e4e4e4", "#202020"]),
    extent=[*timeline_x_extent, 0.5, -0.5],
)
heat = ax_neural.imshow(
    np.clip(timeline_activity_z, -3.0, 3.0),
    aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-3, vmax=3,
    extent=[
        *timeline_x_extent,
        len(timeline_neuron_ids) - 0.5,
        -0.5,
    ],
)

for _, segment in display_segments.iterrows():
    start_x = float(segment["start_frame"]) - 0.5
    mid_x = float(segment["mid_frame"])
    for axis in (ax_role, ax_support, ax_neural, ax_trial):
        axis.axvline(start_x, color="black", linewidth=0.55, alpha=0.55)
    role = int(segment["stimulus_role"]) if np.isfinite(
        segment["stimulus_role"]
    ) else -1
    role_letter = {0: "C", 2: "L"}.get(role, "–")
    ax_role.text(
        mid_x, 0, f"{role_letter} {int(segment['trial_id'])}",
        ha="center", va="center", color="white" if role in (0, 2) else "black",
        fontsize=8, fontweight="bold",
    )

timeline_trial_results = trial_results.loc[
    trial_results["mid_frame"].between(
        timeline_start_frame, timeline_stop_frame
    )
].copy()
timeline_trial_results["plot_x"] = timeline_trial_results["mid_frame"]
valid_evidence = timeline_trial_results["heldout_evidence"].notna()
ax_trial.plot(
    timeline_trial_results.loc[valid_evidence, "plot_x"],
    timeline_trial_results.loc[valid_evidence, "heldout_evidence"],
    color="#777777", linewidth=0.8, alpha=0.5,
)
for role in (0, 2):
    subset = timeline_trial_results.loc[
        valid_evidence & timeline_trial_results["stimulus_role"].eq(role)
    ]
    ax_trial.scatter(
        subset["plot_x"], subset["heldout_evidence"],
        color=ROLE_COLORS[role], edgecolor="white", linewidth=0.5,
        s=48, zorder=3, label=f"role {role} = {ROLE_NAMES[role]}",
    )
ax_trial.axhline(0, color="black", linewidth=0.8)

ax_role.set_ylabel("trial role", rotation=0, ha="right", va="center")
ax_support.set_ylabel("valid", rotation=0, ha="right", va="center")
ax_neural.set_ylabel(f"{AREA} neurons\n(label-free sample)")
ax_trial.set_ylabel("held-out\nevidence")
ax_trial.set_xlabel("acquisition frame (vertical lines are physical trial boundaries)")
for axis in (ax_role, ax_support):
    axis.set_yticks([])
    axis.tick_params(axis="x", labelbottom=False)
ax_neural.tick_params(axis="x", labelbottom=False)
ax_neural.set_yticks(
    np.linspace(0, len(timeline_neuron_ids) - 1, min(5, len(timeline_neuron_ids))).astype(int)
)
ax_neural.set_yticklabels(
    timeline_neuron_ids[
        np.linspace(
            0, len(timeline_neuron_ids) - 1, min(5, len(timeline_neuron_ids))
        ).astype(int)
    ]
)
fig.colorbar(
    heat, ax=ax_neural, pad=0.01, shrink=0.8,
    label="within-neuron z-score (display only)",
)
ax_role.legend(
    handles=[
        Patch(color=ROLE_COLORS[0], label="role 0 = circle1"),
        Patch(color=ROLE_COLORS[2], label="role 2 = leaf1"),
        Patch(color="#c9c9c9", label="other / unassigned"),
    ],
    loc="upper left", bbox_to_anchor=(1.005, 1.2), frameon=False,
)
ax_trial.legend(frameon=False, ncol=2, loc="upper right")
ax_trial.text(
    0.005, 0.97, "one dot = one held-out trial; this is evidence, not d′",
    transform=ax_trial.transAxes, va="top", fontsize=9,
)
fig.suptitle(
    f"{selected['recording_id']} · neural frames aligned to physical trials",
    fontsize=14,
)
plt.show()

# Figure 2: one held-out observation per trial, and actual d-prime only where
# repeated trials of both roles make it defined.
evidence_rows = trial_results.loc[
    trial_results["stimulus_role"].isin([0, 2])
    & trial_results["heldout_evidence"].notna()
].copy()
fig, (ax_evidence, ax_dprime) = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True,
    gridspec_kw={"height_ratios": [1.5, 1.0]}, constrained_layout=True,
)
ax_evidence.plot(
    evidence_rows["trial_id"], evidence_rows["heldout_evidence"],
    color="#777777", linewidth=0.7, alpha=0.45,
)
for role in (0, 2):
    subset = evidence_rows.loc[evidence_rows["stimulus_role"].eq(role)]
    ax_evidence.scatter(
        subset["trial_id"], subset["heldout_evidence"],
        color=ROLE_COLORS[role], s=24, alpha=0.85,
        label=f"role {role} = {ROLE_NAMES[role]}",
    )
ax_evidence.axhline(0, color="black", linewidth=0.8)
ax_evidence.set_ylabel("held-out neural evidence\n(one observation per trial)")
ax_evidence.set_title(
    f"{selected['recording_id']} · {AREA} · trial evidence and segment d′"
)
ax_evidence.legend(frameon=False, ncol=2)

ax_dprime.plot(
    exploratory["stop_trial"], exploratory["dprime"],
    color="#4c78a8", alpha=0.55,
    label="trailing 40-trial segment, updated every trial (39/40 overlap)",
)
finite_blocks = nonoverlapping.loc[nonoverlapping["dprime"].notna()]
for _, segment in finite_blocks.iterrows():
    ax_dprime.hlines(
        segment["dprime"], segment["start_trial"], segment["stop_trial"],
        color="#e45756", linewidth=4.0, alpha=0.9,
    )
    ax_dprime.scatter(
        segment["midpoint_trial"], segment["dprime"],
        color="#e45756", edgecolor="white", linewidth=0.6, s=55, zorder=4,
    )

# Small x marks show the four held-out fold d-primes supporting each red segment.
fold_plot = nonoverlapping_folds.merge(
    nonoverlapping[["segment_id", "start_trial", "stop_trial"]],
    on=["segment_id", "start_trial", "stop_trial"], how="left",
)
fold_plot["fold_x"] = fold_plot["start_trial"] + (
    fold_plot["fold_id"] + 0.5
) * (fold_plot["stop_trial"] - fold_plot["start_trial"] + 1) / 4.0
ax_dprime.scatter(
    fold_plot["fold_x"], fold_plot["fold_dprime"],
    marker="x", color="#9c2f2f", s=26, alpha=0.75,
    label="constituent held-out fold d′",
)
ax_dprime.axhline(0, color="black", linewidth=0.8)
ax_dprime.set(
    xlabel="physical trial ID (blue value is attached to the segment endpoint)",
    ylabel="held-out population d′\n(role 2 − role 0)",
)
ax_dprime.legend(frameon=False, ncol=1, loc="best")
plt.show()

segment_report = db.execute("""
    SELECT
        s.segment_id, s.start_trial, s.stop_trial, s.dprime,
        s.role_2_trials, s.role_0_trials,
        s.valid_folds, s.required_folds,
        f.fold_id, f.fold_dprime
    FROM dprime_segments AS s
    LEFT JOIN dprime_segment_folds AS f
      ON f.segment_id = s.segment_id
    ORDER BY s.segment_id, f.fold_id
""").df()
segment_report
'''
        ),
        md(
            """
## What to carry into the all-mouse analysis

1. Run the same parameterized manifest query for each intended experiment row.
2. Keep one result row per physical `recording_id`; do not double-count an
   acquisition that has several experiment memberships.
3. Preserve the window settings, frame mask, role counts, missing-bin counts,
   speed/occupancy summaries, and exact file paths/MD5 values with every curve.
   Export `trial_results` for the one-row-per-trial evidence sequence and
   `segment_report` for the d-prime estimates and their four held-out fold
   values. Never rename `heldout_evidence` to one-trial d-prime.
4. Reduce repeated recordings to mouse-stage summaries, then compare stages
   within mouse. Neuron `n` and PC `k` are recording-specific and must not be
   joined across dates.
5. Treat the stride-1 curve as descriptive smoothing. Mouse-level inference
   must use predeclared non-overlapping summaries, not overlapping windows as
   independent samples.

If the scientific target is the paper endpoint instead, load the explicitly
listed full-neural file only after checking its size and compute per-neuron
whole-session d-prime:

`2 * (mean(role2) - mean(role0)) / (nanstd(role2, ddof=0) + nanstd(role0, ddof=0))`

That is a different estimand from the held-out trial-level population curve
above. It should be named and reported separately.
"""
        ),
        py(
            r'''
# Final join audit: fail loudly if an axis or key silently drifted.
assert len(frames) == V.shape[1]
assert len(neurons) == U.shape[1] == len(iarea) == xy_t.shape[0]
assert U.shape[0] == V.shape[0]
assert set(wall_roles["stimulus_role"]) >= {0, 2}
assert len(all_modalities) > 0
assert len(trial_ids) == len(responses) == len(labels)
assert len(timeline_frame_ids) == timeline_activity_raw.shape[1]
assert timeline_activity_raw.shape[0] == len(timeline_neuron_ids)
assert not session_evidence["trial_id"].duplicated().any()
assert session_evidence["is_held_out"].all()
finite_segments = nonoverlapping.loc[nonoverlapping["dprime"].notna()]
assert (finite_segments["role_2_trials"] >= 2).all()
assert (finite_segments["role_0_trials"] >= 2).all()
assert (finite_segments["valid_folds"] == 4).all()

audit = {
    "status": "ALL JOIN CHECKS PASSED",
    "experiment": selected["experiment"],
    "recording_id": selected["recording_id"],
    "behavior_key": selected["behavior_key"],
    "behavior_frames": behavior_frames,
    "neural_frames": V.shape[1],
    "trailing_behavior_frames_removed": trailing_excess,
    "neurons": U.shape[1],
    "trials_in_tensor": len(trial_ids),
    "position_bins": N_POSITION_BINS,
    "area": AREA,
    "features": N_FEATURES,
    "timeline_trials": len(display_segments),
    "timeline_neurons": len(timeline_neuron_ids),
    "heldout_trial_evidence_rows": len(session_evidence),
    "finite_stride_1_windows": int(exploratory["dprime"].notna().sum()),
    "finite_nonoverlapping_segments": int(
        nonoverlapping["dprime"].notna().sum()
    ),
}
audit
'''
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"joined-data-{index:03d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
    print(NOTEBOOK)
