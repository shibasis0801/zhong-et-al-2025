#!/usr/bin/env python3
"""Generate the Pandas counterpart to the canonical Drive d-prime notebook."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/10_dprime_pandas_colab.ipynb")


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

    cells = [
        md(
            """
# D-prime exploration — direct Pandas counterpart

This notebook follows the current Drive notebook `dprime.ipynb` in the same
order. Each code cell names the source cell it replaces. The nested Python
dictionaries and lists are replaced by ordinary `pandas.DataFrame` objects;
the resulting tables are easier to filter, join, group, and inspect.

The source notebook currently inventories experiments, mice, recordings,
files, Train 1 labels, and an interactive selector. It does **not yet compute
d-prime**, so neither does this counterpart. Metadata construction does not
download neural arrays.
"""
        ),
        py(
            """
# Main dprime cell 0: connect to the release.
#@title Connect to Drive { display-mode: "form" }
import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "pandas>=2.2,<3",
    ])
    import pandas as pd

try:
    from google.colab import drive as google_drive
except ImportError:
    workspace = None
else:
    google_drive.mount("/content/drive", force_remount=False)
    workspace = (
        "/content/drive/MyDrive/"
        "Zhong et al. 2025 - Neuromatch Team Workspace"
    )
    if workspace not in sys.path:
        sys.path.insert(0, workspace)

for module_name in tuple(sys.modules):
    if module_name == "drive" or module_name == "zhong2025" or module_name.startswith("zhong2025."):
        sys.modules.pop(module_name, None)

# The shared Drive workspace intentionally contains only the data-access
# package subset. Register that subset without importing optional launchers.
if workspace is not None:
    package = ModuleType("zhong2025")
    package.__path__ = [str(Path(workspace) / "zhong2025")]
    package.__package__ = "zhong2025"
    sys.modules["zhong2025"] = package

    atlas = importlib.import_module("zhong2025.atlas")
    data_support = importlib.import_module("zhong2025.data")
    for export in (
        "experiment_rows",
        "format_bytes",
        "load_experiment_index",
        "load_file_inventory",
        "recording_bundle",
    ):
        setattr(package, export, getattr(atlas, export))
    package.fetch_figshare_article = data_support.fetch_figshare_article
else:
    atlas = importlib.import_module("zhong2025.atlas")

drive = importlib.import_module("drive")
data = drive.setup(mount=False, report=False)

assert data.release["article_id"] == 28811129
assert data.release["version"] == 2
assert len(data.files) == 297
assert sum(item.size_bytes for item in data.files) == 452_233_500_962
data
"""
        ),
        py(
            '''
# Main dprime cell 1: info.keys(), represented as a Pandas catalog.
# atlas.load_experiment_index() is the pinned, normalized equivalent of
# Imaging_Exp_info.npy used by drive.py; it avoids loading any neural arrays.
COHORT_ORDER = ("supervised", "unsupervised", "grating", "naive")
LAYER_FOR_CATEGORY = {
    "imaging_behavior": "behavior",
    "reduced_neural": "reduced_neural",
    "full_neural": "full_neural",
    "retinotopy": "retinotopy",
}


def cohort_for(experiment):
    if experiment.startswith("sup_"):
        return "supervised"
    if experiment.startswith("unsup_"):
        return "unsupervised"
    if experiment.startswith("naive_"):
        return "naive"
    if experiment.endswith("_grating"):
        return "grating"
    return "other"


def stage_for(experiment):
    return next(
        (stage for stage in ("train1", "test1", "train2", "test2", "test3")
         if stage in experiment),
        None,
    )


def moment_for(experiment):
    if "_before_" in experiment:
        return "before"
    if "_after_" in experiment:
        return "after"
    return "snapshot"


index = atlas.load_experiment_index()
source_rows = []
for experiment, entries in index["experiments"].items():
    for source_row, entry in enumerate(entries):
        source = entry["source"]
        source_rows.append({
            "experiment": experiment,
            "source_row": source_row,
            "recording_id": entry["recording_id"],
            "retinotopy_id": entry["retinotopy_id"],
            "mouse": source["mname"],
            "date": source["datexp"],
            "block": str(source["blk"]),
            "session_number": source.get("sess#"),
            "reward_type": source.get("rewType"),
            "stimulus": source.get("stim"),
            "stimulus_ids": source.get("stim_id", []),
            "note": source.get("Note"),
        })

experiment_rows = (
    pd.DataFrame(source_rows)
    .sort_values(["experiment", "recording_id", "source_row"], ignore_index=True)
)
membership_keys = [
    "experiment", "recording_id", "retinotopy_id", "mouse", "date", "block"
]
memberships = (
    experiment_rows.groupby(membership_keys, dropna=False, sort=True)
    .size()
    .rename("source_row_count")
    .reset_index()
)
memberships.insert(1, "cohort", memberships["experiment"].map(cohort_for))

experiment_catalog = (
    memberships.groupby("experiment", sort=True)
    .agg(
        recording_count=("recording_id", "nunique"),
        mouse_count=("mouse", "nunique"),
    )
    .reset_index()
)
experiment_catalog.insert(1, "cohort", experiment_catalog["experiment"].map(cohort_for))
experiment_catalog.insert(2, "stage", experiment_catalog["experiment"].map(stage_for))
experiment_catalog.insert(3, "moment", experiment_catalog["experiment"].map(moment_for))

file_catalog = pd.DataFrame([
    {
        "filename": item.name,
        "figshare_file_id": item.id,
        "category": item.category,
        "experiment": item.experiment,
        "recording_id": item.recording_id,
        "retinotopy_id": item.retinotopy_id,
        "size_bytes": item.size_bytes,
        "size_mib": item.size_mib,
        "size_gib": item.size_gib,
        "md5": item.md5,
        "relative_path": item.relative_path,
    }
    for item in data.files
]).sort_values("filename", ignore_index=True)

file_columns = [
    "recording_id", "file_experiment", "layer", "filename",
    "figshare_file_id", "category", "size_bytes", "size_mib", "size_gib",
    "md5", "relative_path",
]

direct_files = file_catalog.loc[
    file_catalog["category"].isin(["reduced_neural", "full_neural"])
].copy()
direct_files["layer"] = direct_files["category"].map(LAYER_FOR_CATEGORY)
direct_files["file_experiment"] = None

behavior_files = memberships[["experiment", "recording_id"]].merge(
    file_catalog.loc[file_catalog["category"].eq("imaging_behavior")],
    on="experiment",
    how="inner",
    validate="many_to_one",
    suffixes=("", "_file"),
)
behavior_files["layer"] = "behavior"
behavior_files = behavior_files.rename(columns={"experiment": "file_experiment"})

recording_keys = memberships[["recording_id", "retinotopy_id"]].drop_duplicates()
retinotopy_files = recording_keys.merge(
    file_catalog.loc[file_catalog["category"].eq("retinotopy")],
    on="retinotopy_id",
    how="inner",
    validate="many_to_one",
    suffixes=("", "_file"),
)
retinotopy_files["layer"] = "retinotopy"
retinotopy_files["file_experiment"] = None

recording_files = (
    pd.concat(
        [
            direct_files[file_columns],
            behavior_files[file_columns],
            retinotopy_files[file_columns],
        ],
        ignore_index=True,
    )
    .drop_duplicates(["recording_id", "file_experiment", "layer", "filename"])
    .sort_values(
        ["recording_id", "layer", "file_experiment"],
        na_position="first",
        ignore_index=True,
    )
)

identity = memberships[
    ["recording_id", "retinotopy_id", "mouse", "date", "block"]
].drop_duplicates()
experiment_summary = (
    memberships.groupby("recording_id")["experiment"]
    .agg(
        experiment_count="nunique",
        experiments=lambda values: tuple(sorted(set(values))),
    )
    .reset_index()
)
layer_presence = (
    recording_files.assign(present=True)
    .pivot_table(
        index="recording_id",
        columns="layer",
        values="present",
        aggfunc="any",
        fill_value=False,
    )
    .rename_axis(columns=None)
    .reset_index()
    .rename(columns={layer: f"has_{layer}" for layer in LAYER_FOR_CATEGORY.values()})
)

recordings = (
    identity.merge(experiment_summary, on="recording_id", validate="one_to_one")
    .merge(layer_presence, on="recording_id", validate="one_to_one")
    .sort_values(["mouse", "date", "block"], ignore_index=True)
)
for layer in LAYER_FOR_CATEGORY.values():
    column = f"has_{layer}"
    if column not in recordings:
        recordings[column] = False


def primary_cohort(values):
    present = set(values)
    return next((name for name in COHORT_ORDER if name in present), "other")


mouse_cohorts = (
    memberships[["mouse", "cohort"]]
    .drop_duplicates()
    .groupby("mouse")["cohort"]
    .agg(
        primary_cohort=primary_cohort,
        cohorts=lambda values: tuple(sorted(set(values))),
    )
    .reset_index()
)
recording_counts = (
    recordings.groupby("mouse")
    .agg(
        physical_recordings=("recording_id", "nunique"),
        has_full_neural=("has_full_neural", "any"),
        has_reduced_neural=("has_reduced_neural", "any"),
    )
    .reset_index()
)
mice = (
    mouse_cohorts.merge(recording_counts, on="mouse", validate="one_to_one")
    .loc[lambda frame: frame["has_full_neural"]]
    .sort_values(["primary_cohort", "mouse"], ignore_index=True)
)
mice_recordings = (
    recordings.merge(
        mice[["mouse", "primary_cohort"]],
        on="mouse",
        how="inner",
        validate="many_to_one",
    )
    .sort_values(["primary_cohort", "mouse", "date", "block"], ignore_index=True)
)

experiment_catalog["experiment"].tolist()
'''
        ),
        py(
            """
# Main dprime cell 2: info['naive_test1'].
experiment_rows.loc[
    experiment_rows["experiment"].eq("naive_test1"),
    [
        "recording_id", "mouse", "date", "block", "session_number",
        "reward_type", "stimulus", "stimulus_ids", "note",
    ],
].sort_values(["mouse", "date", "block"], ignore_index=True)
"""
        ),
        py(
            """
# Main dprime cell 3: every experiment label, grouped without hardcoding.
experiments = experiment_catalog.loc[
    :,
    ["cohort", "stage", "moment", "experiment", "recording_count", "mouse_count"],
].sort_values(["cohort", "stage", "moment", "experiment"], ignore_index=True)
experiments
"""
        ),
        py(
            """
# Main dprime cell 4: mice with full-neural recordings, derived from the files.
mice.loc[
    :,
    ["primary_cohort", "mouse", "physical_recordings", "cohorts"],
]
"""
        ),
        py(
            """
# Main dprime cell 5: all physical recordings for TX108.
mice_recordings.loc[
    mice_recordings["mouse"].eq("TX108"),
    [
        "recording_id", "date", "block", "experiments", "has_behavior",
        "has_reduced_neural", "has_full_neural", "has_retinotopy",
    ],
].reset_index(drop=True)
"""
        ),
        py(
            """
# Main dprime cell 6: the mice_recordings inventory and cohort counts.
mice_recording_summary = (
    mice_recordings.groupby("primary_cohort", sort=False)
    .agg(
        mice=("mouse", "nunique"),
        physical_recordings=("recording_id", "nunique"),
    )
    .reindex(COHORT_ORDER)
    .dropna(how="all")
    .reset_index()
)
mice_recording_summary
"""
        ),
        py(
            """
# Main dprime cell 7: supervised/TX108 slice.
mice_recordings.loc[
    mice_recordings["primary_cohort"].eq("supervised")
    & mice_recordings["mouse"].eq("TX108")
].reset_index(drop=True)
"""
        ),
        py(
            """
# Main dprime cell 8: file lookup, now returning a tidy DataFrame.
def files(recording_rows):
    selected = recording_rows[["recording_id", "mouse"]].drop_duplicates()
    return (
        selected.merge(
            recording_files,
            on="recording_id",
            how="inner",
            validate="one_to_many",
        )
        [[
            "mouse", "recording_id", "file_experiment", "layer",
            "filename", "size_gib",
        ]]
        .sort_values(
            ["mouse", "recording_id", "layer", "file_experiment", "filename"],
            na_position="first",
            ignore_index=True,
        )
    )
"""
        ),
        py(
            """
# Main dprime cell 9: every linked file for supervised/TX108.
files(mice_recordings.loc[
    mice_recordings["primary_cohort"].eq("supervised")
    & mice_recordings["mouse"].eq("TX108")
])
"""
        ),
        py(
            """
# Main dprime cell 10: every linked file for naive/TX124.
files(mice_recordings.loc[
    mice_recordings["primary_cohort"].eq("naive")
    & mice_recordings["mouse"].eq("TX124")
])
"""
        ),
        py(
            """
# Main dprime cell 11: every linked file for naive/TX140.
files(mice_recordings.loc[
    mice_recordings["primary_cohort"].eq("naive")
    & mice_recordings["mouse"].eq("TX140")
])
"""
        ),
        py(
            """
# Main dprime cell 12: Train 1 labels, represented as seven rows.
train1_labels = pd.DataFrame([
    ("supervised", "before", "sup_train1_before_learning"),
    ("supervised", "after", "sup_train1_after_learning"),
    ("unsupervised", "before", "unsup_train1_before_learning"),
    ("unsupervised", "after", "unsup_train1_after_learning"),
    ("grating", "before", "train1_before_grating"),
    ("grating", "after", "train1_after_grating"),
    ("naive", "reference", "naive_test1"),
], columns=["cohort", "phase", "experiment"])
train1_labels
"""
        ),
        py(
            """
# Main dprime cell 13: full mice_recordings table.
mice_recordings
"""
        ),
        py(
            """
# Main dprime cell 14: select intended-cohort, full-neural recordings.
def select_recordings(cohort, label):
    return (
        memberships.loc[memberships["experiment"].eq(label)]
        .drop(columns="cohort")
        .merge(
            recordings[["recording_id", "experiments", "has_full_neural"]],
            on="recording_id",
            how="inner",
            validate="many_to_one",
        )
        .merge(
            mice[["mouse", "primary_cohort"]],
            on="mouse",
            how="inner",
            validate="many_to_one",
        )
        .loc[lambda frame: (
            frame["has_full_neural"]
            & frame["primary_cohort"].eq(cohort)
        )]
        [[
            "experiment", "mouse", "recording_id", "date", "block",
            "experiments",
        ]]
        .sort_values(["mouse", "date", "block"], ignore_index=True)
    )


train1_recordings = pd.concat(
    [
        select_recordings(row.cohort, row.experiment).assign(
            cohort=row.cohort,
            phase=row.phase,
        )
        for row in train1_labels.itertuples(index=False)
    ],
    ignore_index=True,
)[[
    "cohort", "phase", "experiment", "mouse", "recording_id",
    "date", "block", "experiments",
]].sort_values(
    ["cohort", "mouse", "phase", "date", "block"],
    ignore_index=True,
)

train1_recordings
"""
        ),
        py(
            """
# Main dprime cell 15: labels present under each cohort.
train1_recordings.groupby("cohort", sort=False)["experiment"].agg(
    lambda values: tuple(dict.fromkeys(values))
).reindex(COHORT_ORDER).dropna()
"""
        ),
        py(
            """
# Main dprime cell 16: files for the naive Train 1 reference.
files(train1_recordings.loc[
    train1_recordings["cohort"].eq("naive")
    & train1_recordings["experiment"].eq("naive_test1")
])
"""
        ),
        py(
            """
# Main dprime cell 17: file_map as one tidy row per selected file.
file_map = (
    train1_recordings.merge(
        recording_files,
        on="recording_id",
        how="inner",
        validate="many_to_many",
    )
    [[
        "cohort", "phase", "experiment", "mouse", "recording_id",
        "file_experiment", "layer", "filename", "size_gib",
    ]]
    .drop_duplicates()
    .sort_values(
        ["cohort", "experiment", "mouse", "recording_id", "layer", "filename"],
        ignore_index=True,
    )
)
file_map
"""
        ),
        py(
            """
# Main dprime cell 18: supervised file map.
file_map.loc[file_map["cohort"].eq("supervised")].reset_index(drop=True)
"""
        ),
        py(
            """
# Main dprime cell 19: unsupervised file map.
file_map.loc[file_map["cohort"].eq("unsupervised")].reset_index(drop=True)
"""
        ),
        py(
            """
# Main dprime cell 20: grating file map.
file_map.loc[file_map["cohort"].eq("grating")].reset_index(drop=True)
"""
        ),
        py(
            """
# Main dprime cell 21: naive file map.
file_map.loc[file_map["cohort"].eq("naive")].reset_index(drop=True)
"""
        ),
        py(
            """
# Main dprime cell 22: Train 1 labels again, matching the source checkpoint.
train1_labels
"""
        ),
        py(
            """
# Main dprime cell 23: experiment -> recording selector backed by a DataFrame.
import ipywidgets as widgets
from IPython.display import display


class RecordingSelector(widgets.VBox):
    def __init__(self, frame):
        self.frame = frame.copy().reset_index(drop=True)
        experiments = tuple(self.frame["experiment"].drop_duplicates())
        if not experiments:
            raise ValueError("The selector needs at least one experiment row")

        self.experiment = widgets.Dropdown(
            options=experiments,
            value=experiments[0],
            description="Experiment:",
        )
        self.recording = widgets.Dropdown(description="Recording:")
        self.output = widgets.Output()
        super().__init__([self.experiment, self.recording, self.output])

        self.experiment.observe(self._on_experiment, names="value")
        self.recording.observe(self._on_recording, names="value")
        self._refresh_recordings()

    def _on_experiment(self, change):
        if change.get("name") == "value":
            self._refresh_recordings()

    def _on_recording(self, change):
        if change.get("name") == "value" and change.get("new") is not None:
            self._render()

    def _refresh_recordings(self):
        options = tuple(
            self.frame.loc[
                self.frame["experiment"].eq(self.experiment.value),
                "recording_id",
            ].drop_duplicates()
        )
        self.recording.options = options
        self.recording.value = options[0] if options else None
        self._render()

    @property
    def selected(self):
        return self.frame.loc[
            self.frame["experiment"].eq(self.experiment.value)
            & self.frame["recording_id"].eq(self.recording.value)
        ].reset_index(drop=True)

    @property
    def value(self):
        return {
            "experiment": self.experiment.value,
            "recording_id": self.recording.value,
        }

    def _render(self):
        self.output.clear_output(wait=True)
        with self.output:
            display(self.selected[[
                "mouse", "recording_id", "layer", "filename", "size_gib"
            ]])


def dataframe_recording_selector(frame):
    return RecordingSelector(frame)
"""
        ),
        py(
            """
# Main dprime cell 24: supervised selector.
sup_train1 = dataframe_recording_selector(
    file_map.loc[file_map["cohort"].eq("supervised")]
)
sup_train1
"""
        ),
        py(
            """
# Main dprime cell 25 intended result: read the current selection directly.
# This replaces the unfinished `dict_ara` expression in the source notebook.
sup_train1.value, sup_train1.selected
"""
        ),
        py(
            """
# Main dprime cell 26 was empty. Use it for parity checks.
expected_rows = {
    "experiment_rows": 142,
    "experiment_catalog": 23,
    "file_catalog": 297,
    "memberships": 133,
    "mice": 19,
    "recording_files": 400,
    "recordings": 89,
    "train1_recordings": 38,
}
actual_rows = {
    "experiment_rows": len(experiment_rows),
    "experiment_catalog": len(experiment_catalog),
    "file_catalog": len(file_catalog),
    "memberships": len(memberships),
    "mice": len(mice),
    "recording_files": len(recording_files),
    "recordings": len(recordings),
    "train1_recordings": len(train1_recordings),
}
assert actual_rows == expected_rows
assert set(train1_labels["experiment"]) == set(train1_recordings["experiment"])

actual_rows
"""
        ),
        md(
            """
## Load arrays only after selecting a recording

The selector returns the exact experiment and physical `recording_id`. Load
only the layers needed by the next d-prime analysis step:

```python
choice = sup_train1.value
behavior = data.load(
    recording=choice["recording_id"],
    layer="behavior",
    experiment=choice["experiment"],
)
svd = data.load(recording=choice["recording_id"], layer="reduced_neural")
```

Inspect `sup_train1.selected["size_gib"]` before loading `full_neural` files.
"""
        ),
    ]

    source_index = 0
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            cell.id = f"dprime-pandas-source-{source_index:02d}"
            source_index += 1
            cell.execution_count = None
            cell.outputs = []
        else:
            cell.id = f"dprime-pandas-note-{index:02d}"

    notebook.cells = cells
    return notebook


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
    print(NOTEBOOK)
