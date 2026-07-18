#!/usr/bin/env python3
"""Generate a SQL-first replacement for the team's d-prime data inventory."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/03_sql_dataset_selection_for_dprime.ipynb")


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
# D-prime data selection with SQL

This notebook repeats the metadata work from the team's exploratory `dprime`
notebook using flat tables rather than nested `Dataset → Recording → DataFile`
objects. `drive.py` is only the validated filesystem/loading boundary;
`sql.py` turns the supplied catalog and `Imaging_Exp_info.npy` into Pandas
DataFrames and registers them in DuckDB.

The queries below do **not** download neural arrays. They establish exactly
which physical recordings and files an analysis would load.
"""
        ),
        py(
            """
#@title Connect to Drive and build the SQL catalog { display-mode: "form" }
import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType

try:
    import duckdb
    import pandas as pd
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "pandas>=2.2,<3", "duckdb>=1.4,<2",
    ])

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
    if (
        module_name in {"drive", "sql", "zhong2025"}
        or module_name.startswith("zhong2025.")
    ):
        sys.modules.pop(module_name, None)

# The shared workspace intentionally contains only the small analysis subset of
# zhong2025.  Build the package surface drive.py needs without executing the
# development checkout's broader __init__.py (which imports optional modules).
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

drive = importlib.import_module("drive")
sql = importlib.import_module("sql")
data = drive.setup(mount=False, report=False)
db = sql.setup(source=data)
"""
        ),
        md(
            """
## What can be queried?

`schema()` is the only inventory to remember. The most useful joins are:

- `experiments → memberships → recordings → mice`
- `recordings → recording_files → files`
- `experiment_rows` when the original fields from `Imaging_Exp_info.npy` matter
"""
        ),
        py(
            """
db.schema()
"""
        ),
        py(
            '''
# The 23 supplied experiment labels (the equivalent of info.keys()).
db.query("""
    SELECT cohort, stage, moment, experiment, recording_count, mouse_count
    FROM experiments
    ORDER BY cohort, stage, moment, experiment
""")
'''
        ),
        py(
            '''
# The raw supplied rows behind one label (the equivalent of info['naive_test1']).
db.query("""
    SELECT recording_id, mouse, date, block, session_number,
           reward_type, stimulus, stimulus_ids_json, note
    FROM experiment_rows
    WHERE experiment = ?
    ORDER BY mouse, date, block, source_row
""", ["naive_test1"])
'''
        ),
        md(
            """
## Mice with neural recordings

The old notebook hardcoded four mouse lists. Here they are derived from the
supplied experiment memberships and restricted to recordings that have a
published full-neural file. `primary_cohort` uses the mutually exclusive study
grouping: supervised, then unsupervised, then grating, then naive-only.
"""
        ),
        py(
            '''
db.query("""
    SELECT m.primary_cohort,
           string_agg(m.mouse, ', ' ORDER BY m.mouse) AS mice,
           count(*) AS mouse_count,
           sum(m.recording_count) AS physical_recordings
    FROM mice AS m
    WHERE m.has_full_neural
    GROUP BY m.primary_cohort
    ORDER BY CASE m.primary_cohort
        WHEN 'supervised' THEN 1 WHEN 'unsupervised' THEN 2
        WHEN 'grating' THEN 3 WHEN 'naive' THEN 4 END
""")
'''
        ),
        md(
            """
## Everything for one mouse

One row below is one physical imaging acquisition. `experiments_json` can hold
multiple labels because labels describe analytical roles; it does not mean the
neural file contains multiple sessions.
"""
        ),
        py(
            '''
mouse = "TX108"

db.query("""
    SELECT recording_id, date, block, experiments_json,
           has_behavior, has_reduced_neural, has_full_neural, has_retinotopy
    FROM recordings
    WHERE mouse = ?
    ORDER BY date, block
""", [mouse])
'''
        ),
        py(
            '''
# Every supplied file linked to that mouse's physical recordings.
db.query("""
    SELECT rf.recording_id, rf.experiment, rf.layer,
           rf.filename, round(rf.size_gib, 3) AS size_gib
    FROM recording_files AS rf
    JOIN recordings AS r USING (recording_id)
    WHERE r.mouse = ?
    ORDER BY rf.recording_id, rf.layer, rf.experiment
""", [mouse])
'''
        ),
        py(
            '''
# The same file inspection for the two naive-only mice from the old notebook.
db.query("""
    SELECT r.mouse, rf.recording_id, rf.experiment, rf.layer,
           rf.filename, round(rf.size_gib, 3) AS size_gib
    FROM recording_files AS rf
    JOIN recordings AS r USING (recording_id)
    WHERE r.mouse IN ('TX124', 'TX140')
    ORDER BY r.mouse, rf.recording_id, rf.layer, rf.experiment
""")
'''
        ),
        md(
            """
## Train 1 selection

This little DataFrame replaces the nested `train_1_labels` dictionary. It is
registered as a DuckDB table, so the selection is one readable join. The naive
snapshot is a reference, not a before/after pair.
"""
        ),
        py(
            """
train1_labels = pd.DataFrame([
    ("supervised", "before", "sup_train1_before_learning"),
    ("supervised", "after", "sup_train1_after_learning"),
    ("unsupervised", "before", "unsup_train1_before_learning"),
    ("unsupervised", "after", "unsup_train1_after_learning"),
    ("grating", "before", "train1_before_grating"),
    ("grating", "after", "train1_after_grating"),
    ("naive", "reference", "naive_test1"),
], columns=["cohort", "phase", "experiment"])

db.register("train1_labels", train1_labels)
train1_labels
"""
        ),
        py(
            '''
# Complete Train 1 recording selection, restricted to the intended cohort and
# to physical acquisitions with full neural data.
train1_recordings = db.query("""
    SELECT l.cohort, l.phase, l.experiment,
           m.mouse, m.recording_id, m.date, m.block,
           r.experiments_json
    FROM train1_labels AS l
    JOIN memberships AS m USING (experiment)
    JOIN recordings AS r USING (recording_id)
    JOIN mice AS mouse_info ON mouse_info.mouse = m.mouse
    WHERE r.has_full_neural
      AND mouse_info.primary_cohort = l.cohort
    ORDER BY l.cohort, m.mouse, l.phase, m.date, m.block
""")

train1_recordings
'''
        ),
        md(
            """
`train1_recordings` is itself registered because `db.query(...)` returns a
DataFrame; register it before the summary query:
"""
        ),
        py(
            '''
db.register("train1_recordings", train1_recordings)

db.query("""
    SELECT cohort, phase,
           count(DISTINCT mouse) AS mice,
           count(DISTINCT recording_id) AS physical_recordings
    FROM train1_recordings
    GROUP BY cohort, phase
    ORDER BY cohort, phase
""")
'''
        ),
        md(
            """
## Load only after selecting

The SQL catalog resolves exact filenames without downloading arrays. Once a
row has been chosen, `db.load` delegates one safe load to `drive.py`:

```python
recording_id = "TX108_2023_03_13_1"
behavior = db.load(
    recording_id, "behavior",
    experiment="sup_train1_before_learning",
)
svd = db.load(recording_id, "reduced_neural")
retinotopy = db.load(recording_id, "retinotopy")
```

Use `full_neural` only after checking `recording_files.size_gib`; those files
are several GiB. No `Recording` object is involved in these calls.
"""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"dprime-sql-{index:03d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
    print(NOTEBOOK)
