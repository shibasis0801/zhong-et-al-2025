# Zhong et al. (2025): Neuromatch data workspace

[Open the shared Neuromatch workspace in Google Drive](https://drive.google.com/drive/folders/1jKMIEf2srnu_Dg_TP6NKk7XBIDxYB4EN)

[00 — Connect to the shared data](https://colab.research.google.com/drive/1b2GEoNKX578kYF1xbteSTtajiXit7212) ·
[01 — Understand the complete dataset](https://colab.research.google.com/drive/1W7r6ZNp-R2h1LOd_WBzuNAsCiysyLUc1) ·
[02 — Run visible graph experiments](https://colab.research.google.com/drive/1GIYknr_LrG3q2Xhd8s9_PlyJIS3TwO_9)

This repository gives the Neuromatch team a shared, neutral understanding of the
Zhong et al. release before the team chooses an analysis. It focuses on setup,
data exploration, experimental design, schemas, and relationships. It does not
propose a research question, hypothesis, decoder, or preferred result.

The intended audience is an interdisciplinary research team, including
clinicians, PhD researchers, neuroscientists, and quantitative or computational
teammates. No one person is expected to supply every kind of expertise.

## Start here

1. In Google Drive, add **Zhong et al. 2025 - Neuromatch Team Workspace** as a
   shortcut in **My Drive**. Do not make a copy.
2. Open `01_understand_the_dataset_colab.ipynb` in the workspace root and choose
   **Runtime → Run all**. A CPU runtime is enough.
3. Work through the inventory, experiment timeline, join rules, schemas, and
   compact real-data example together.
4. Open `02_graph_experiments_colab.ipynb` in the workspace root to see the same small
   recording as a visible, rerunnable flow.
5. Record the team's eventual question and decisions in the private shared
   workspace, after everyone has the same data model.

The default run downloads no published data. It uses a bundled, pickle-free
catalog of the complete Figshare v2 release and a 2.9 MB real-data example.

The public Figshare article metadata is also available as a normal dictionary:

```python
from zhong2025 import fetch_figshare_article

figshare_api = fetch_figshare_article()
```

This reads `https://api.figshare.com/v2/articles/28811129` without credentials.
The pinned local catalog remains the reproducible source for analysis.

## What the atlas covers

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

## Repository map

- [`notebooks/zhong2025_data_atlas_colab.ipynb`](notebooks/zhong2025_data_atlas_colab.ipynb): complete data and experiment orientation
- [`notebooks/zhong2025_graph_experiments_colab.ipynb`](notebooks/zhong2025_graph_experiments_colab.ipynb): one neutral sample flow with interactive variations
- [`graph.py`](graph.py): the complete notebook-only graph runner and widget surface
- [`zhong2025/atlas.py`](zhong2025/atlas.py): validated catalog and relationship helpers
- [`zhong2025/assets/`](zhong2025/assets): one canonical copy of the safe catalog, experiment index, curated download manifest, and compact example
- [`scripts/create_data_atlas_notebook.py`](scripts/create_data_atlas_notebook.py): reviewable notebook source
- [`scripts/build_atlas_assets.py`](scripts/build_atlas_assets.py): rebuilds normalized JSON from pinned Figshare metadata
- [`scripts/build_atlas_demo.py`](scripts/build_atlas_demo.py): rebuilds the compact example from checksum-verified sources
- [`drive-sync.json`](drive-sync.json): non-secret Drive folder IDs and import-safety policy
- [`scripts/drive_sync.py`](scripts/drive_sync.py): committed-snapshot publishing and reviewed Drive imports
- [`original/`](original): untouched byte-for-byte snapshot of upstream commit `ba64ac697f5d9914926baac79399e80707a5f3a6`
- [`tests/`](tests): catalog, download-safety, notebook, packaging-data, and upstream-integrity checks

All upstream paper-reproduction notebooks and figure scripts live only in
`original/`; duplicate root copies were removed. Do not edit that folder.

## Graph experiments

The `graph` module keeps repeated notebook analysis understandable without
introducing a workflow platform. Ordinary Python functions are nodes, their
arguments and declared returns are named ports, and matching names show the
data flow. The notebook UI combines a curved-wire flow map with native input
controls inside the corresponding node cards. Filled and hollow sockets
distinguish wired data, editable inputs, unused outputs, and the displayed
result. After a run, the map and cards show completion timings and safe shape-
aware previews. The resulting run remains available as `panel.last_run` and
records settings, intermediate outputs, execution order, terminal branches,
and timings.

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
layer, parallel runtime, or cross-run cache. The included experiment summarizes
one recording under different descriptive settings and explicitly avoids
inference or a preferred scientific comparison.

## Google Drive collaboration

Google Drive is the team's entry point. Every notebook mounts each teammate's
own Drive and accepts either of these shortcuts:

- the workspace shortcut with its existing name; or
- the dataset shortcut renamed `Zhong2025_Janelia_v2`.

The shared dataset remains read-only. Save personal outputs in
`MyDrive/Zhong2025_personal_results`, and put only deliberately shared results
in the team's agreed results area. Team members do not need a source checkout
or a separate copy of the 421 GiB release.

## Dataset safety

The full release is approximately 421.175 GiB. Source behavior and neural NPY
files are pickled, SVD files include a serialized scikit-learn object, and
`areas.npz` contains an object array. The atlas does not load them.

The optional downloader exposes only two small, declared profiles:

- `metadata`: the experiment index and shared area outlines;
- `atlas_demo_source`: the three checksum-pinned files needed to reproduce the
  compact example.

There is no raw-neural or full-release download profile. Downloads are streamed,
size- and MD5-checked, and atomically renamed. Pickle-enabled demo rebuilding
also verifies exact SHA-256 values before loading.

Install the optional `rebuild` extra only when regenerating the compact example;
it supplies the scikit-learn version used by the published SVD pickle:

```bash
python -m pip install -e '.[rebuild]'
```

## Local verification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Regenerate the notebook after editing its source:

```bash
python scripts/create_data_atlas_notebook.py
```

Rebuild the two normalized metadata assets from previously downloaded, pinned
inputs:

```bash
python scripts/build_atlas_assets.py \
  --article-json /path/to/figshare-v2.json \
  --experiment-index /path/to/Imaging_Exp_info.npy
```

## Sources and license

- Zhong et al., “Unsupervised pretraining in biological neural networks,”
  *Nature* (2025), [DOI 10.1038/s41586-025-09180-y](https://doi.org/10.1038/s41586-025-09180-y)
- Published data, Figshare v2,
  [DOI 10.25378/janelia.28811129.v2](https://doi.org/10.25378/janelia.28811129.v2), CC BY 4.0

Repository code is distributed under GNU GPL v3. The compact derivative retains
the source dataset's CC BY 4.0 terms; cite Zhong et al. and the Figshare dataset.
