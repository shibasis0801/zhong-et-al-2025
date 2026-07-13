#!/usr/bin/env python3
"""Generate the neutral, team-facing Zhong et al. data-atlas notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "CPU",
        "colab": {"name": "Zhong et al. (2025): complete data atlas"},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown(
            r"""
# Zhong et al. (2025): complete data atlas

**A neutral orientation notebook for the Neuromatch team**

This notebook is for understanding the release before choosing or implementing
a research question. It does not fit a model, recommend a hypothesis, or rank
one experiment above another.

By the end, every teammate should be able to explain:

1. what the 297 published files contain;
2. how the two studies differ;
3. how 23 imaging experiment labels relate to 89 physical recordings;
4. how behavior, neural frames, SVD components, trials, stimuli, and retinotopy
   join to one another;
5. which relationships are absent and must not be assumed; and
6. what a proposed data slice would cost before downloading it.

The default run is CPU-only and uses a committed, pickle-free metadata catalog
plus a 2.9 MB real-data example. It downloads **none of the 421 GiB release**.
"""
        ),
        markdown(
            r"""
## 0. Reproducible setup

In Colab, the next cell checks out the requested repository revision and installs
the small helper package. Use `main` while the atlas is changing and pin a commit
SHA when the team wants a fixed reference. Local execution reuses the current
checkout. The only first-run network transfer is the Git repository itself.
"""
        ),
        code(
            r"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

try:
    import google.colab  # type: ignore[import-not-found]  # noqa: F401
except ImportError:
    IN_COLAB = False
else:
    IN_COLAB = True

REPO_URL = "https://github.com/shibasis0801/zhong-et-al-2025.git"
REPO_REF = "main"  # @param {type:"string"}

if IN_COLAB:
    REPO_ROOT = Path("/content/zhong-et-al-2025")
    fresh_clone = False
    if not (REPO_ROOT / ".git").exists():
        if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-Git directory {REPO_ROOT}")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(REPO_ROOT)],
            check=True,
        )
        fresh_clone = True
    remote_url = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remote_url.rstrip("/").removesuffix(".git") != REPO_URL.rstrip("/").removesuffix(".git"):
        raise RuntimeError(f"Unexpected origin in {REPO_ROOT}: {remote_url}")
    if not fresh_clone and subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise RuntimeError("Existing Colab checkout has local changes; use a fresh runtime")
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", REPO_REF],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", "FETCH_HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_ROOT)],
        check=True,
    )
else:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    REPO_ROOT = next(
        (candidate for candidate in candidates if (candidate / "pyproject.toml").exists()),
        None,
    )
    if REPO_ROOT is None:
        raise RuntimeError("Run this notebook from inside the repository checkout")
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)
HELPER_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO_ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
print(f"Repository: {REPO_ROOT}")
print(f"Revision: {HELPER_COMMIT} (requested ref: {REPO_REF})")
print(f"Python: {sys.version.split()[0]} | Colab: {IN_COLAB}")
"""
        ),
        code(
            r"""
from collections import Counter, defaultdict
import html
import json

from IPython.display import HTML, display
import matplotlib.pyplot as plt
import numpy as np

from zhong2025 import (
    experiment_recordings,
    experiment_rows,
    experiment_semantics,
    filter_inventory,
    format_bytes,
    inventory_summary,
    load_atlas_demo,
    load_experiment_index,
    load_file_inventory,
    recording_bundle,
)


def show_table(rows, columns, *, max_rows=30):
    # Display a dependency-free HTML table and report omitted rows.
    rows = list(rows)
    shown = rows[:max_rows]
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in shown:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, (list, tuple, dict)):
                value = json.dumps(value, ensure_ascii=False)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    display(HTML(
        "<div style='overflow-x:auto'><table style='border-collapse:collapse'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        "<style>th,td{border:1px solid #ddd;padding:5px 8px;text-align:left}"
        "th{background:#f4f4f4}</style>"
    ))
    if len(rows) > max_rows:
        print(f"Showing {max_rows} of {len(rows)} rows.")


inventory = load_file_inventory()
experiment_index = load_experiment_index()
demo = load_atlas_demo()
article = inventory["article"]

assert article["file_count"] == 297
assert article["total_size_bytes"] == 452_233_500_962
assert experiment_index["summary"] == {
    "associations": 142,
    "experiment_labels": 23,
    "unique_mice": 19,
    "unique_recordings": 89,
}
print(
    f"Loaded the complete v{article['version']} catalog and "
    f"compact session {demo['metadata']['session']} without network data access."
)
"""
        ),
        markdown(
            r"""
## 1. The mental model

The Figshare API is a **warehouse listing files**, not the semantic experiment
database. Meaning emerges by combining five layers:

```text
Figshare file inventory
        +
Imaging_Exp_info.npy        experiment label ↔ recording identity
        +
Beh_<experiment>.npy       trials, events, physical stimuli, frame labels
        +
*_neural_data.npy / SVD    neural activity on the frame axis
        +
*_trans.npz                neuron coordinates and visual-area labels
```

The canonical imaging recording key is `mouse_date_block`, for example
`TX119_2023_12_24_1`. Retinotopy joins on `mouse_date` because its filename does
not contain the block. Behavior files are experiment-level dictionaries that
contain many recording keys; they are not one behavior file per recording.
"""
        ),
        code(
            r"""
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

nodes = {
    "Figshare v2\n297-file warehouse": (0.50, 0.90),
    "Imaging study\n19 mice · 89 recordings": (0.27, 0.69),
    "Faster-learning study\n23 different mice · behavior only": (0.77, 0.69),
    "Experiment index\n23 labels · 142 memberships": (0.10, 0.43),
    "Behavior bundles\ntrials · events · frame labels": (0.34, 0.43),
    "Recording ID\nmouse_date_block": (0.34, 0.18),
    "Neural / SVD\nneurons or PCs × frames": (0.62, 0.28),
    "Retinotopy\nneuron ↔ area": (0.86, 0.28),
}
for label, (x, y) in nodes.items():
    ax.text(
        x, y, label, ha="center", va="center", fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#eef5ff", "edgecolor": "#5277a3"},
    )

edges = [
    ("Figshare v2\n297-file warehouse", "Imaging study\n19 mice · 89 recordings"),
    ("Figshare v2\n297-file warehouse", "Faster-learning study\n23 different mice · behavior only"),
    ("Imaging study\n19 mice · 89 recordings", "Experiment index\n23 labels · 142 memberships"),
    ("Imaging study\n19 mice · 89 recordings", "Behavior bundles\ntrials · events · frame labels"),
    ("Experiment index\n23 labels · 142 memberships", "Recording ID\nmouse_date_block"),
    ("Behavior bundles\ntrials · events · frame labels", "Recording ID\nmouse_date_block"),
    ("Recording ID\nmouse_date_block", "Neural / SVD\nneurons or PCs × frames"),
    ("Neural / SVD\nneurons or PCs × frames", "Retinotopy\nneuron ↔ area"),
]
for source, target in edges:
    x0, y0 = nodes[source]
    x1, y1 = nodes[target]
    ax.annotate("", xy=(x1, y1 + 0.055), xytext=(x0, y0 - 0.055), arrowprops={"arrowstyle": "->", "color": "#555"})

ax.set_title("How the released data layers relate", fontsize=13)
plt.show()
"""
        ),
        markdown(
            r"""
## 2. The complete release at a glance

Version 2 is pinned by article ID, version, file IDs, byte counts, and checksums.
The catalog below is committed as JSON, so it can be inspected offline. The full
release is **452,233,500,962 bytes (421.175 GiB)**; raw neural arrays dominate
storage. The compact example is not a replacement for the release—it is only a
safe object for learning its axes and labels.
"""
        ),
        code(
            r"""
summary_rows = []
for row in inventory_summary(inventory):
    summary_rows.append({
        **row,
        "readable_size": format_bytes(row["size_bytes"]),
        "percent": f"{100 * row['size_bytes'] / article['total_size_bytes']:.2f}%",
    })
show_table(
    summary_rows,
    [
        ("label", "File family"),
        ("file_count", "Files"),
        ("readable_size", "Storage"),
        ("percent", "% of release"),
    ],
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
labels = [row["category"] for row in summary_rows]
axes[0].barh(labels, [row["file_count"] for row in summary_rows], color="#4c78a8")
axes[0].invert_yaxis()
axes[0].set(xlabel="file count", title="Inventory count")
axes[1].barh(labels, [row["size_bytes"] / 1024**3 for row in summary_rows], color="#f58518")
axes[1].set_xscale("log")
axes[1].invert_yaxis()
axes[1].set(xlabel="GiB (log scale)", title="Storage footprint")
plt.show()
"""
        ),
        markdown(
            r"""
## 3. Two related but separate studies

| Study | Animals and design | Released modalities | Important boundary |
|---|---|---|---|
| Imaging study | 19 mice; task, unrewarded natural-image exposure, naive, and grating-control conditions; 89 two-photon recordings at selected stages | Behavior, full neural activity, 400-PC SVD, retinotopy | Stage snapshots, not continuous imaging on every training day |
| Faster-learning study | 23 different, non-imaged mice: 11 natural-pretrained, 7 grating-pretrained, 5 no-pretraining; then five task days | Three behavior bundles | There is no same-mouse neural join to the imaging study |

“Unsupervised” here means active, closed-loop VR exposure without water reward;
it does not mean passive viewing. The faster-learning study is a distinct cohort,
not an additional modality for the 19 imaging mice.

Its three behavior bundles contain mouse/session/day identity, physical textures,
trial and reward status, lick time and position, first-lick position, running,
reward delivery, trials completed, and performance over days. The simplified
task used in that study omitted the imaging task's sound cue.
"""
        ),
        code(
            r"""
faster_files = [
    entry for entry in inventory["files"]
    if entry["category"] == "faster_learning_behavior"
]
show_table(
    sorted(faster_files, key=lambda row: row["name"]),
    [("name", "Behavior-only file"), ("size_bytes", "Exact bytes"), ("id", "Figshare file ID")],
)
print("These files have no recording_id, neural_data, SVD, or retinotopy partner.")
"""
        ),
        markdown(
            r"""
## 4. Imaging experiment timeline and vocabulary

The experiment index contains **23 labels**. The natural-exposure lanes follow
Train 1 → Test 1 → Train 2 → Test 2 → Test 3. Grating controls use paired
before/after labels for Train 1, Test 1, and Test 2.

| Stage | Physical event | Task | Unrewarded exposure | Naive | Grating control |
|---|---|---|---|---|---|
| Train 1, before | initial familiar A1/B1 pair | `sup_train1_before_learning` | `unsup_train1_before_learning` | — | `train1_before_grating` |
| Train 1, after | A1/B1 after repeated experience | `sup_train1_after_learning` | `unsup_train1_after_learning` | — | `train1_after_grating` |
| Test 1 | familiar A1/B1 plus new A2/B2 | `sup_test1` | `unsup_test1` | `naive_test1` | `test1_before_grating`, `test1_after_grating` |
| Train 2, before | A2 initially novel | `sup_train2_before_learning` | `unsup_train2_before_learning` | — | — |
| Train 2, after | A2 after repeated presentation | `sup_train2_after_learning` | `unsup_train2_after_learning` | — | — |
| Test 2 | A1, trained A2, new A3, and B1 | `sup_test2` | `unsup_test2` | `naive_test2` | `test2_before_grating`, `test2_after_grating` |
| Test 3 | A1, A2, B1, and spatial A1 swaps | `sup_test3` | `unsup_test3` | `naive_test3` | — |

These are semantic roles. A label can reuse a recording already referenced by
another label, so an experiment membership is not automatically a new acquisition.
"""
        ),
        code(
            r"""
association_rows = experiment_rows(experiment_index)
experiment_table = []
for experiment, entries in experiment_index["experiments"].items():
    semantics = experiment_semantics(experiment)
    experiment_table.append({
        "experiment": experiment,
        **semantics,
        "membership_rows": len(entries),
        "unique_recordings": len({entry["recording_id"] for entry in entries}),
    })
stage_order = {"Train 1": 0, "Test 1": 1, "Train 2": 2, "Test 2": 3, "Test 3": 4}
cohort_order = {"task": 0, "unrewarded exposure": 1, "naive": 2, "grating control": 3}
experiment_table.sort(key=lambda row: (stage_order[row["stage"]], cohort_order[row["cohort"]], row["moment"]))
show_table(
    experiment_table,
    [
        ("experiment", "Experiment label"),
        ("cohort", "Cohort"),
        ("stage", "Stage"),
        ("moment", "Moment"),
        ("stimulus_roles", "Canonical stimulus IDs"),
        ("membership_rows", "Membership rows"),
        ("unique_recordings", "Unique recordings"),
    ],
    max_rows=30,
)
"""
        ),
        markdown(
            r"""
### Why 142 memberships are not 142 recordings

`Imaging_Exp_info.npy` has 142 descriptors but only 89 unique
`mouse_date_block` acquisition keys. Reuse occurs for two reasons:

- one physical session can serve more than one stage label (for example, a Test
  1 session can also be the “before” snapshot for Train 2);
- Test 3 may describe `swap1` and `swap2` as separate membership rows while they
  share one physical recording.

Every one of the 89 physical acquisitions has exactly one full-neural file, one
SVD file, and one date-level retinotopy file in the release.
"""
        ),
        code(
            r"""
by_recording = defaultdict(list)
for row in association_rows:
    by_recording[row["recording_id"]].append(row)

reused = []
for recording_id, rows in by_recording.items():
    if len(rows) > 1:
        reused.append({
            "recording_id": recording_id,
            "association_rows": len(rows),
            "experiment_labels": sorted({row["experiment"] for row in rows}),
            "stimulus_variants": sorted({str(row["stimulus_type"]) for row in rows if row["stimulus_type"]}),
        })
reused.sort(key=lambda row: (-row["association_rows"], row["recording_id"]))

print(f"Associations: {len(association_rows)}")
print(f"Physical recordings: {len(by_recording)}")
print(f"Recordings with more than one association row: {len(reused)}")
show_table(
    reused,
    [
        ("recording_id", "Recording"),
        ("association_rows", "Rows"),
        ("experiment_labels", "Experiment labels"),
        ("stimulus_variants", "Test-3 variants"),
    ],
    max_rows=15,
)
"""
        ),
        markdown(
            r"""
## 5. Canonical stimulus roles versus physical textures

`stim_id` is the experiment-wide role; `WallName` and `UniqWalls` preserve the
literal texture shown in a particular session. Always keep both.

| `stim_id` | Canonical role | Meaning |
|---:|---|---|
| 0 | `circle1` | familiar B exemplar |
| 1 | `circle2` | new B exemplar |
| 2 | `leaf1` | familiar A exemplar; rewarded in task mice |
| 3 | `leaf2` | new A exemplar |
| 4 | `leaf3` | third A exemplar |
| 5 | `leaf1_swap1` | one spatial rearrangement of A1 |
| 6 | `leaf1_swap2` | a second spatial rearrangement of A1 |

The physical families can be leaf/circle or brick/rock. The compact example's
release strings are `rock*` and `wood*`; the paper describes the `wood*` family
as brick. Never infer a global role from a string prefix—resolve it through
`zip(beh["UniqWalls"], beh["stim_id"])` for that session.
"""
        ),
        code(
            r"""
wall_names = np.asarray(demo["wall_name"])
stimulus_ids = np.asarray(demo["stimulus_id"])
mapping_rows = []
for wall in sorted(np.unique(wall_names)):
    selected = wall_names == wall
    ids = np.unique(stimulus_ids[selected])
    mapping_rows.append({
        "physical_wall_name": wall,
        "canonical_stim_id": int(ids[0]),
        "trials": int(selected.sum()),
    })
show_table(
    mapping_rows,
    [
        ("physical_wall_name", "Physical WallName"),
        ("canonical_stim_id", "Canonical stim_id"),
        ("trials", "Trials in compact example"),
    ],
)
"""
        ),
        markdown(
            r"""
## 6. Join anatomy for one recording

For recording `MOUSE_DATE_BLOCK`:

| Layer | File/key | Main axes | Join rule |
|---|---|---|---|
| Experiment membership | `Imaging_Exp_info.npy` | descriptors | construct `mname_datexp_blk`; keep optional `stimtype` |
| Behavior | `Beh_<experiment>.npy[recording_id]` | trials and imaging frames | `ft_*[t]` labels neural frame `t` |
| Full neural | `<recording_id>_neural_data.npy["spks"]` | neurons × frames | concatenate imaging-plane arrays on the neuron axis |
| Reduced neural | `<recording_id>_SVD_dec.npy` | `U`: 400 × neurons; `V`: 400 × frames | `U.T @ V` approximates full neural activity |
| Retinotopy | `<mouse_date>_trans.npz` | one row per neuron | retinotopy row `i` ↔ neural row `i` ↔ `U` column `i` |

The SVD basis is fitted separately per session. “PC 1” is therefore not a
shared biological axis across dates. Retinotopy labels simultaneous areas within
a recording; it is not a longitudinal cell-registration map.
"""
        ),
        code(
            r"""
SELECTED_RECORDING = "TX119_2023_12_24_1"  # @param {type:"string"}
bundle = recording_bundle(
    SELECTED_RECORDING,
    inventory=inventory,
    index=experiment_index,
)
print(f"Recording: {bundle['recording_id']}")
print(f"Retinotopy join key: {bundle['retinotopy_id']}")
print(f"Experiment memberships: {bundle['experiments']}")
print(f"All referenced files: {format_bytes(bundle['total_bytes'])}")
show_table(
    bundle["files"],
    [
        ("category", "Layer"),
        ("name", "Published filename"),
        ("size_bytes", "Exact bytes"),
        ("id", "Figshare file ID"),
    ],
)

membership_details = []
for experiment, entries in experiment_index["experiments"].items():
    for entry in entries:
        if entry["recording_id"] != SELECTED_RECORDING:
            continue
        source = entry["source"]
        membership_details.append({
            "experiment": experiment,
            "sess#": source.get("sess#"),
            "days": source.get("days"),
            "reward": source.get("rewType"),
            "stim_id": source.get("stim_id"),
            "stimtype": source.get("stimtype"),
            "depth": source.get("depth"),
            "exptype": source.get("exptype"),
            "isDR": source.get("isDR"),
            "gender": source.get("Gender"),
            "note": source.get("Note") or source.get("stim"),
        })
print("Exact experiment-index descriptors for this recording:")
show_table(
    membership_details,
    [
        ("experiment", "Experiment"),
        ("sess#", "sess#"),
        ("days", "days"),
        ("reward", "Reward type"),
        ("stim_id", "stim_id"),
        ("stimtype", "stimtype"),
        ("depth", "Depth"),
        ("exptype", "exptype"),
        ("isDR", "isDR"),
        ("gender", "Gender"),
        ("note", "Note / stimulus description"),
    ],
)
"""
        ),
        markdown(
            r"""
## 7. What is inside each data layer

### Experiment-index descriptors

Every descriptor contains `mname`, `datexp`, `blk`, `is2p`, `rewType`,
`stim_id`, and `ROIdir`. Depending on the recording it can also contain
`sess#`, `days`, `depth`, `Gender`, `Note`, `exptype`, `isDR`, `artLick`,
`2pblk`, `stim`, or `stimtype`. Preserve missing and inconsistent values rather
than inventing defaults. `stimtype` distinguishes Test-3 swap memberships and
is not part of the physical acquisition key.

### Behavior dictionaries

The 23 imaging `Beh_*` files are pickled dictionaries of session dictionaries.
Their fields fall into these groups:

| Group | Important fields | Meaning |
|---|---|---|
| Session/trials | `ntrials`, `trInd`, `Trial_start_time`, `Trial_end_time` | trial identity and boundaries |
| Stimuli | `WallName`, `UniqWalls`, `stim_id`, `isRew` | physical texture, canonical role, reward status |
| Raw movement/VR | `SubjMove`, `VRpos`, `VRposCum`, `VRposTime` | treadmill and virtual-corridor trajectory |
| Frame-aligned behavior | `ft`, `ft_trInd`, `ft_Pos`, `ft_PosCum`, `ft_RunSpeed`, `ft_isMoving`, `ft_CorrSpc`, `ft_GraySpc`, `ft_WallID` | behavior at every neural frame |
| Events | `SoundPos`, `SoundTime`, `RewPos`, `RewTime`, `LickPos`, `LickTime` | cue, reward, and licking |
| Event frames | `StartFr`, `GrayFr`, `EndFr`, `SoundFr`, `RewardFr`, `LickFr`, `BefCueFr`, `AftCueFr` | events mapped onto the neural-frame axis |
| Lookup/settings | `TrialStim`, `StimTrial`, `StimFrame`, `Corridor_Length`, `Gray_Space_length`, `Texture_Length`, `Reward_Mode` | convenience indices and session configuration |

`WallType` and `WallIsProbe` are explicitly marked as unsuitable for these
experiments in the released processing notebook. Use `WallName` plus `stim_id`.

### Neural and spatial files

- Full neural files contain Suite2p-deconvolved fluorescence under `spks`.
- SVD files contain numeric `U`, numeric `V`, and a pickled scikit-learn
  `SVD_model`; the exact source file must be checksum-verified before unpickling.
- Per-session retinotopy files are numeric NPZ archives containing `xy_t`,
  `iarea`, `xpos`, `ypos`, and `A`.
- `areas.npz` contains object-typed shared area outlines and also requires trusted
  pickle loading.
- Area IDs used by the released code are V1 = 8; mHV = 0/1/2/9; lHV = 5/6;
  aHV = 3/4.

For the compact example's verified source, `U` has shape `(400, 27,281)`, `V`
has shape `(400, 14,570)`, and retinotopy `iarea` has length 27,281. Thus `V`
shares the frame axis with behavior, while `U` shares the neuron axis with
retinotopy.

This atlas never unpickles those source files. It uses normalized JSON metadata
and the pickle-free compact example instead.
"""
        ),
        markdown(
            r"""
## 8. Inspect one real compact example

The example comes from `TX119_2023_12_24_1`. Its catalog memberships are
`unsup_test1` and `unsup_train2_before_learning`. It exists only to make the
abstract axes tangible; it is not a recommended analysis cohort.

The source behavior had 14,571 frame labels and the source SVD had 14,570 neural
frames; the builder records and removes the one trailing behavior frame. It then
stores means within each trial and 18 fixed position bins. No interpolation
occurs across trial boundaries.
"""
        ),
        code(
            r"""
axis_meanings = {
    "population_features": "trial × position bin × published PC",
    "area_features": "area × trial × position bin × compact metric feature",
    "mean_run_speed": "trial × position bin",
    "frame_counts": "trial × position bin",
    "trial_id": "trial",
    "wall_name": "trial",
    "stimulus_id": "trial",
    "texture_family": "trial",
    "exemplar": "trial",
    "area_name": "area",
    "position_edges_m": "position-bin edges",
    "position_centers_m": "position bins",
    "texture_bin_mask": "position bins",
}
schema_rows = []
for name, meaning in axis_meanings.items():
    value = np.asarray(demo[name])
    schema_rows.append({
        "name": name,
        "shape": value.shape,
        "dtype": value.dtype,
        "axes": meaning,
    })
show_table(schema_rows, [("name", "Array"), ("shape", "Shape"), ("dtype", "Dtype"), ("axes", "Axes")])

population = np.asarray(demo["population_features"])
speed = np.asarray(demo["mean_run_speed"])
centers = np.asarray(demo["position_centers_m"])
frame_counts = np.asarray(demo["frame_counts"])
assert population.shape[:2] == speed.shape == frame_counts.shape == (452, 18)
assert np.array_equal(np.asarray(demo["trial_id"]), np.arange(452))
assert np.all(frame_counts > 0)
"""
        ),
        code(
            r"""
trial_order = np.argsort(wall_names)
unique_walls, wall_counts = np.unique(wall_names, return_counts=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
axes[0, 0].bar(unique_walls, wall_counts, color="#4c78a8")
axes[0, 0].set(title="Trials by physical texture", ylabel="trials")

image = axes[0, 1].imshow(
    population[trial_order, :, 0], aspect="auto", interpolation="nearest",
    extent=[centers[0], centers[-1], len(trial_order), 0], cmap="coolwarm",
)
axes[0, 1].set(title="Published PC 1 across trials", xlabel="corridor position (m)", ylabel="trials sorted by WallName")
fig.colorbar(image, ax=axes[0, 1], label="PC score")

for wall in unique_walls:
    selected = wall_names == wall
    axes[1, 0].plot(centers, population[selected, :, 0].mean(axis=0), label=wall)
    axes[1, 1].plot(centers, speed[selected].mean(axis=0), label=wall)
axes[1, 0].set(title="Mean PC 1 trajectory", xlabel="corridor position (m)", ylabel="PC score")
axes[1, 1].set(title="Mean running-speed field", xlabel="corridor position (m)", ylabel="release speed units")
axes[1, 0].legend(fontsize=8)
axes[1, 1].legend(fontsize=8)
plt.show()
"""
        ),
        code(
            r"""
area_features = np.asarray(demo["area_features"])
area_names = np.asarray(demo["area_name"])
mean_area_magnitude = np.linalg.norm(area_features, axis=-1).mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
for area_index, area_name in enumerate(area_names):
    axes[0].plot(centers, mean_area_magnitude[area_index], marker="o", label=area_name)
axes[0].set(title="Compact area representations", xlabel="corridor position (m)", ylabel="mean feature-vector norm")
axes[0].legend()
axes[1].boxplot(frame_counts, positions=centers, widths=0.18, showfliers=False)
axes[1].set(title="Frames contributing to each trial/bin", xlabel="corridor position (m)", ylabel="frames", xticks=centers[::3])
plt.show()
print("Area feature shape:", area_features.shape)
print("Frame-count range:", int(frame_counts.min()), "to", int(frame_counts.max()))
"""
        ),
        markdown(
            r"""
### What the compact example preserves and omits

| Preserves | Omits |
|---|---|
| exact trial IDs, physical labels, canonical stimulus IDs | raw per-frame event arrays |
| fixed corridor positions and frame counts | individual-neuron activity |
| first 48 session PC time courses after trial/bin averaging | the full 400-PC source |
| compact simultaneous views of V1, mHV, lHV, and aHV | neuron coordinates and original retinotopy arrays |
| running speed and source provenance | other recordings, mice, and the behavior-only cohort |

It is suitable for schema learning and plotting checks, not for claims about the
full study.
"""
        ),
        markdown(
            r"""
## 9. Browse all published files without downloading them

Set `FILE_CATEGORY` or `FILE_SEARCH` and re-run the cell. Searches operate on the
offline 297-file catalog. The result reports exact storage but performs no file
download.
"""
        ),
        code(
            r"""
FILE_CATEGORY = "all"  # @param ["all", "imaging_behavior", "faster_learning_behavior", "full_neural", "reduced_neural", "retinotopy", "imaging_experiment_index", "area_outlines", "behavior_example", "neural_example"]
FILE_SEARCH = ""  # @param {type:"string"}

matches = filter_inventory(
    inventory["files"],
    category=FILE_CATEGORY,
    search=FILE_SEARCH,
)
matches = sorted(matches, key=lambda row: (row["category"], row["name"]))
print(f"Matched {len(matches)} files totaling {format_bytes(sum(row['size_bytes'] for row in matches))}.")
show_table(
    matches,
    [
        ("category", "Category"),
        ("name", "Filename"),
        ("size_bytes", "Exact bytes"),
        ("recording_id", "Recording ID"),
        ("id", "Figshare ID"),
    ],
    max_rows=40,
)
"""
        ),
        markdown(
            r"""
## 10. Preview an experiment slice before any download

Leave `SELECTED_EXPERIMENT` blank for the neutral default. To inspect one of the
23 labels, enter it exactly as shown in section 4. The planner counts each shared
behavior bundle once and each selected recording modality once. It still does
not download anything.
"""
        ),
        code(
            r"""
SELECTED_EXPERIMENT = ""  # @param {type:"string"}
INCLUDE_FULL_NEURAL = False  # @param {type:"boolean"}
ALLOW_DATA_DOWNLOADS = False

if ALLOW_DATA_DOWNLOADS:
    raise RuntimeError("The data atlas is intentionally planning-only; use the checksum-pinned downloader separately.")

if not SELECTED_EXPERIMENT:
    print("No experiment selected. Available labels:")
    print("\n".join(sorted(experiment_index["experiments"])))
else:
    recording_ids = experiment_recordings(SELECTED_EXPERIMENT, experiment_index)
    names = {f"Beh_{SELECTED_EXPERIMENT}.npy"}
    for recording_id in recording_ids:
        retinotopy_id = recording_id.rsplit("_", 1)[0]
        names.add(f"{recording_id}_SVD_dec.npy")
        names.add(f"{retinotopy_id}_trans.npz")
        if INCLUDE_FULL_NEURAL:
            names.add(f"{recording_id}_neural_data.npy")
    entries_by_name = {entry["name"]: entry for entry in inventory["files"]}
    selected_files = [entries_by_name[name] for name in sorted(names)]
    layer_counts = Counter(entry["category"] for entry in selected_files)
    print(f"Experiment: {SELECTED_EXPERIMENT}")
    print(f"Physical recordings: {len(recording_ids)}")
    print(f"Files: {len(selected_files)} | storage: {format_bytes(sum(row['size_bytes'] for row in selected_files))}")
    show_table(
        [
            {"category": category, "files": count}
            for category, count in sorted(layer_counts.items())
        ],
        [("category", "Layer"), ("files", "Files")],
    )
"""
        ),
        markdown(
            r"""
## 11. Locally derived processing files

The released processing notebook in `original/data_process_script.ipynb` creates
additional files. They are **derived products**, not additional recordings and
not part of the 297-file Figshare inventory.

| Derived family | Built from | What it stores |
|---|---|---|
| `*_interpolate_spk.npy` | behavior + full neural | neuron × trial × 60 one-decimeter bins |
| `*_dprime.npy` | stimulus-selected frames + neural + retinotopy | per-neuron selectivity and spatial metadata |
| `*_dprime_distribution.npy` | d-prime + retinotopy | cortical density maps |
| `*_dprime_frac.npy` | d-prime + area labels | selective-neuron fractions by area and threshold |
| `*_coding_direction.npy` | split trials + position-aligned activity | population projections by stimulus and area |
| `*_sort_spk.npy` | position-aligned activity | stimulus responses sorted by peak position |
| `*_reward_response.npy` / `*_rew_frac.npy` | task events + neural activity | cue/reward-aligned summaries |

The original full processing path is Windows-oriented, unpickles source objects,
and estimates roughly 900 GB of derived output in addition to the release. It is
preserved for provenance, not used by this Colab atlas.
"""
        ),
        markdown(
            r"""
## 12. Relationships that exist—and ones that do not

### Supported within an imaging recording

- neural frame ↔ trial, physical texture, canonical stimulus role, corridor
  position, running, cue, licking, and reward fields;
- neural row ↔ retinotopy coordinate and cortical-area assignment;
- full neural activity ↔ its own session's 400-PC representation;
- simultaneous V1 ↔ HVA activity;
- experiment label ↔ physical recording through the experiment index.

### Not supplied by this release

- neural activity and faster-learning behavior in the same mouse—the cohorts are
  separate;
- a clean map tracking the same neuron across dates;
- a shared PC axis across sessions—each SVD basis is session-specific;
- continuous daily neural measurements throughout learning;
- causal influence from one cortical area to another;
- passive-viewing data for the “unsupervised” group—the animals ran in closed-loop
  VR.

Before any analysis, identify the physical acquisition unit, the frame and neuron
axes, the session-specific label mapping, and the missing relationships. Do not
treat membership rows or trials as independent mice.
"""
        ),
        markdown(
            r"""
## 13. Team comprehension check

Before the team defines an analysis, everyone should be able to answer these
without guessing:

- [ ] Which of the two studies contains the record of interest?
- [ ] What is the canonical `mouse_date_block` acquisition key?
- [ ] Which experiment labels point to it, and are any rows stimulus variants?
- [ ] Which `Beh_*` bundle contains its frame labels?
- [ ] Which axes in behavior, neural/SVD, and retinotopy must agree?
- [ ] What are the canonical stimulus roles and the session's physical names?
- [ ] Which released layers are required, and how many exact bytes are they?
- [ ] Is the intended comparison within a recording, across recordings, or
      across mice?
- [ ] Which desired relationship is not present in the release?
- [ ] Is any source file pickled, and has it been checksum-verified before load?

Once these answers are shared, the team can choose its research question without
the setup repository steering that decision.

Sources: [Zhong et al., Nature (2025)](https://doi.org/10.1038/s41586-025-09180-y),
[Figshare dataset v2](https://doi.org/10.25378/janelia.28811129.v2), and the
[released processing code](https://github.com/MouseLand/zhong-et-al-2025).
"""
        ),
    ]
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"atlas-{index:03d}"
    return notebook


def main() -> None:
    output = Path("notebooks/zhong2025_data_atlas_colab.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.validate(notebook)
    nbf.write(notebook, output)
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
