# Zhong et al. (2025): Neuromatch data workspace

[Open the shared Neuromatch workspace in Google Drive](https://drive.google.com/drive/folders/1jKMIEf2srnu_Dg_TP6NKk7XBIDxYB4EN)

[01 — Pandas/SQL atlas and graph tutorials](https://colab.research.google.com/drive/1VqEQNouYY3PAhe0NTHF2ynr_fbJtQBga) ·
[02 — Released-example d-prime walkthrough](https://colab.research.google.com/drive/1YvuuZPrkPNoMFCu15yfBU_V0zeMBwsRz) ·
[03 — SQL dataset selection for d-prime](https://colab.research.google.com/drive/150-BVYjFZbWWvJUqTYGSqm6C53SNHPq6) ·
[04 — Joined-data and d-prime workspace](https://colab.research.google.com/drive/1MYg6FaXJlWvI_vvNxGs-xLw31u2v4oH2) ·
[05 — Neuromatch visual-learning project](https://colab.research.google.com/drive/1gfx81il2wj5A1b15VpqGMTe2CM213cSn)

[Archived notebooks](https://drive.google.com/drive/folders/1ExkCpjfloETZsKFlaj5EvmoHnJ-DzDl7)

The five active notebooks now form one sequence: understand the dataset and
analysis graphs, reproduce the released d-prime example, select candidate data
with SQL, inspect joined neural/behavioral/retinotopy data, and compare the work
with the original Neuromatch project. Earlier exploratory and generated
notebooks remain available under `notebooks/archived/` for provenance.

Workspace terminology is fixed: an **analysis graph** is an interactive
node-and-wire workflow made from connected analysis steps and editable
controls. A **plot** is a 2D Matplotlib visual produced by that workflow. Run
analysis graphs, read plots, and inspect intermediate ports.

The intended audience is an interdisciplinary research team, including
clinicians, PhD researchers, neuroscientists, and quantitative or computational
team members. No single contributor is expected to supply every kind of
expertise.

## Start here

1. In Google Drive, add **Zhong et al. 2025 - Neuromatch Team Workspace** as a
   shortcut in **My Drive**. Do not make a copy.
2. Start with `notebooks/01_dataset_atlas_and_graph_tutorials.ipynb` to learn
   the released dataset structure and the analysis-graph interface.
3. Use `notebooks/02_released_example_dprime_walkthrough.ipynb` for the released
   example and `notebooks/03_sql_dataset_selection_for_dprime.ipynb` to select
   candidate sessions and trials.
4. Continue in `notebooks/04_joined_data_and_dprime_workspace.ipynb` for the
   neural, behavioral, and retinotopy join and d-prime analysis surface.
5. Use `notebooks/05_neuromatch_visual_learning_project.ipynb` as the upstream
   scientific reference and comparison point.
6. Record the team's analysis decisions in the private shared workspace after
   establishing a shared understanding of the data structure.

## Find and fetch data

`sql.setup()` reads the Drive release and `Imaging_Exp_info.npy`, creates eight
ordinary Pandas tables, and writes the same tables to `catalog.duckdb`:

```python
import sql

db = sql.setup()

files = db.table("files")
recordings = db.table("recordings")

selected = db.query("""
    SELECT recording_id, experiment
    FROM memberships
    WHERE mouse = ? AND experiment = ?
""", ["TX119", "unsup_test1"])

print(db.database_path)
db.export("/path/to/shibasis.dev/targets/appWeb/public/catalog.duckdb")
```

The browser artifact can also be rebuilt directly:

```bash
python code/database.py /path/to/shibasis.dev/targets/appWeb/public/catalog.duckdb
```

Arrays remain lazy. Load only after selecting a recording or exact catalog file:

```python
recording_id = selected.iloc[0]["recording_id"]
svd = db.load(recording_id, "reduced_neural")
behavior = db.load(recording_id, "behavior", experiment="unsup_test1")
retinotopy = db.load(recording_id, "retinotopy")
areas = db.load_file("areas.npz")
```

There are no custom file or recording objects. `recording_files` is the
relational join table connecting recording IDs to behavior, full-neural, SVD,
and retinotopy files. The private filesystem layer copies and verifies only the
chosen file. The default 10 GiB per-file limit covers every released file;
loading a full-neural file remains an explicit large-data choice.

## Notebook 03's walkthrough analysis graphs

Each analysis graph has Colab controls above its code. Change a control and
rerun that cell, then read the resulting plots:

1. release composition, file selection, and imaging availability;
2. cortical locations from one selected retinotopy file;
3. trial order, trial-by-position activity, and corridor profiles;
4. descriptive held-out d-prime by visual area in one compact recording.

These are factual orientation views. The fourth analysis graph introduces the
project metric, but none fits a learning curve or tests the
rewarded-versus-unrewarded hypothesis.

## Notebook 05's five interactive analysis graphs

Notebook 05 is an interactive analysis map rather than a sequence of
disconnected plots. Its five workflow graphs are:

1. **Cohort, provenance, and Drive fetch plan** — select the experiment pair,
   session policy, and data layers; inspect acquisition structure, deduplicate
   the exact manifest, and check per-file and total storage limits without
   downloading data.
2. **One real recording and held-out d-prime laboratory** — choose trials,
   visual area, stimulus contrast, corridor interval, support rules, activity
   readout, and cross-validation settings; compare the neural, behavioral,
   coverage, and held-out-discriminability plots.
3. **Simulation-only Plan A sandbox** — vary learning rate, plateau,
   between-mouse variation, drift, confounding, missingness, and uncertainty to
   see what the proposed estimand can and cannot recover. Its output is a
   design aid, never evidence about the released cohort.
4. **Real Plan A reward analysis** — assemble the cohort, representation,
   contrast, time-estimator, and execution specifications; audit exact files
   and storage first, then optionally reconstruct session curves and compare
   early rates, saturation, behavior, sensitivity, and mouse-level inference.
5. **Plan B corridor analysis** — explore where activity and discriminability
   change along the corridor, alongside speed, support, and event profiles.
   It can use the bundled real example or reuse the last completed real Plan A
   run without loading the Drive files again.

The safe real-data path is explicit: analysis graph 1 establishes cohort and
file truth; analysis graph 4 starts in **Plan only (no download)** mode and
reports its immutable analysis specification, exact files, and GiB preflight
budget. Select **Load and analyse real data** with the shared Drive mounted only
after the team accepts that preflight. The workflow graph then retains
intermediate ports for inspection, while its final port contains only the
explanatory plots.

## What the workspace covers

- all **297 files** and their exact **452,233,500,962-byte** footprint;
- the 19-mouse imaging study and separate 23-mouse behavior-only study;
- all 23 imaging experiment labels, 142 membership rows, and 89 physical
  recordings;
- canonical stimulus roles versus session-specific physical texture names;
- behavior, trial, frame, full-neural, SVD, and retinotopy schemas;
- the joins between experiment labels, `mouse_date_block`, behavior bundles,
  neural frames, neurons, and cortical areas;
- locally generated processing products and how they differ from recordings;
- relationships that exist and relationships the release cannot provide;
- offline catalog filtering and storage previews before any optional download.

## Workspace map

- [`notebooks/01_dataset_atlas_and_graph_tutorials.ipynb`](notebooks/01_dataset_atlas_and_graph_tutorials.ipynb): dataset atlas and analysis-graph tutorials
- [`notebooks/02_released_example_dprime_walkthrough.ipynb`](notebooks/02_released_example_dprime_walkthrough.ipynb): released-example d-prime walkthrough
- [`notebooks/03_sql_dataset_selection_for_dprime.ipynb`](notebooks/03_sql_dataset_selection_for_dprime.ipynb): Pandas inventory queried through DuckDB for d-prime dataset selection
- [`notebooks/04_joined_data_and_dprime_workspace.ipynb`](notebooks/04_joined_data_and_dprime_workspace.ipynb): joined neural, behavioral, and retinotopy data with the working d-prime analysis surface
- [`notebooks/05_neuromatch_visual_learning_project.ipynb`](notebooks/05_neuromatch_visual_learning_project.ipynb): original Neuromatch visual-learning project reference
- [`notebooks/archived/`](notebooks/archived): earlier generated and exploratory notebooks retained for provenance
- [`code/database.py`](code/database.py): the readable `ZhongDB` class that creates the Pandas tables and native DuckDB file
- [`code/dprime.py`](code/dprime.py): trial summaries, held-out d-prime, learning curves, and mouse-level inference
- [`code/drive.py`](code/drive.py): one-call Colab setup plus mounting, catalog reading, checksummed copying, and NumPy loading
- [`code/graph.py`](code/graph.py): the complete notebook-only analysis-graph runner and widget surface
- [`code/position.py`](code/position.py): behavior–neural frame alignment and trial-by-position binning
- [`code/sql.py`](code/sql.py): small `connect()` and `setup()` entry points for `ZhongDB`
- [`code/metadata/`](code/metadata): the two small validated snapshots that let ZhongDB expose catalog tables without mounting the release
- [`notebooks/05_neuromatch_visual_learning_project.ipynb`](notebooks/05_neuromatch_visual_learning_project.ipynb): upstream project notebook used to verify plot-source parity offline
- [`code/zhong/`](code/zhong): untouched 12-file snapshot of upstream commit `ba64ac697f5d9914926baac79399e80707a5f3a6`
- [`references/`](references): paper, analysis-methods review, supplementary movie, and derived reference maps
- [`Backups/legacy-development-tools-2026-07-18/`](Backups/legacy-development-tools-2026-07-18): retired generators, tests, and prior Drive snapshot

All upstream paper-reproduction notebooks and plotting scripts live only in
`code/zhong/`; duplicate root copies were removed. Do not edit that folder.

## Workflow graph interface

The `graph` module keeps repeated notebook analysis understandable. Ordinary
Python functions are nodes, their arguments and declared returns are named
ports, and matching names show the data flow. Here, **analysis graph** means
this node-and-wire workflow; its Matplotlib results are **plots**. The notebook
UI is one curved-wire flow canvas with native input controls placed directly on
the hollow ports inside their nodes. Filled and hollow sockets distinguish
wired data, editable inputs, unused outputs, and the displayed plots. After a
run, the same canvas shows completion timings and safe shape-aware previews.
The resulting run remains available as `panel.last_run` and records settings,
intermediate outputs, execution order, terminal branches, and timings.
Results travel through a synchronized HTML value rather than callback output
capture, so HTML cards and bounded Matplotlib PNGs update reliably in Colab.
Map-rendering failures are isolated from execution: a valid result is still
shown with an explicit map warning.

One output may feed several independent branches—for example, selection and
data-quality checks can inspect the same loaded slice before rejoining. These
branches are visual and logical, not concurrent: nodes execute one at a time in
numbered declaration order for predictable Colab memory use and reproducible
errors. Nodes should treat shared arrays and dictionaries as read-only.

The module intentionally provides only:

- `@graph.node(outputs=...)`;
- `graph.Graph(...)`;
- `run`, `run_many`, `diagram`, and `widget`.

There is no draggable desktop canvas, node palette, scheduler, persistence
layer, parallel runtime, or cross-run cache. Notebooks 00, 03, and 04 use it for
orientation without inference. Notebook 05 provides the separate, explicit
inferential analysis graphs for the team-selected scientific comparison.

## Google Drive collaboration

Google Drive is the team's entry point and the canonical shared workspace. Code
lives in `code/`; active team notebooks live in `notebooks/`, while
superseded notebooks live in `notebooks/archived/`. Add the workspace shortcut
with its existing name so Colab can import `code/sql.py`. The data API then finds
the shared dataset through either
of these locations:

- the workspace shortcut with its existing name; or
- the dataset shortcut renamed `Zhong2025_Janelia_v2`.

The shared dataset remains read-only. Store individual outputs in
`MyDrive/Zhong2025_personal_results`, and put only deliberately shared results
in the team's agreed results area. A source checkout and a separate copy of the
421 GiB release are not required.

## Dataset safety

The full release is approximately 421.175 GiB. Source behavior and neural NPY
files are pickled, SVD files include a serialized scikit-learn object, and
`areas.npz` contains an object array. `db.load()` and `db.load_file()` accept only cataloged files
and verifies size and MD5 before any pickle-enabled load. Numeric retinotopy
archives are opened without pickle.

The separate optional Figshare downloader still exposes only two small,
declared profiles:

- `metadata`: the experiment index and shared area outlines;
- `atlas_demo_source`: the three checksum-pinned files needed to reproduce the
  compact example.

There is no raw-neural or full-release network-download profile. For the
already verified shared Drive copy, query the `files` table to describe any
published file and use `db.fetch()` or `db.fetch_file()` to stage one selection. Copies are
streamed, size- and MD5-checked, and atomically renamed. Pickle-enabled demo
rebuilding also verifies exact SHA-256 values before loading.

Install the optional `rebuild` extra only when regenerating the compact example;
it supplies the scikit-learn version used by the published SVD pickle:

```bash
python -m pip install -e 'code[rebuild]'
```

## Local verification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e 'code[dev]'
PYTHONPATH=code python -c "import database, dprime, drive, graph, position, sql"
```

The former notebook generators and broad test suite are preserved only under
`Backups/legacy-development-tools-2026-07-18/`; they are not part of the active
Colab workspace.

## Sources and license

- Zhong et al., “Unsupervised pretraining in biological neural networks,”
  *Nature* (2025), [DOI 10.1038/s41586-025-09180-y](https://doi.org/10.1038/s41586-025-09180-y)
- Published data, Figshare v2,
  [DOI 10.25378/janelia.28811129.v2](https://doi.org/10.25378/janelia.28811129.v2), CC BY 4.0
- Neuromatch Academy,
  [`visual_learning_80k_neurons.ipynb`](https://github.com/NeuromatchAcademy/course-content/blob/main/projects/neurons/visual_learning_80k_neurons.ipynb), CC BY 4.0 / BSD-3-Clause

Repository code is distributed under GNU GPL v3. The compact derivative retains
the source dataset's CC BY 4.0 terms; cite Zhong et al. and the Figshare dataset.
