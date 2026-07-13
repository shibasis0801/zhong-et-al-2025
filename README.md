# Zhong et al. (2025): Neuromatch data workspace

[![Open the complete data atlas in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibasis0801/zhong-et-al-2025/blob/main/notebooks/zhong2025_data_atlas_colab.ipynb)

This repository gives the Neuromatch team a shared, neutral understanding of the
Zhong et al. release before the team chooses an analysis. It focuses on setup,
data exploration, experimental design, schemas, and relationships. It does not
propose a research question, hypothesis, decoder, or preferred result.

The intended audience is an interdisciplinary research team, including
clinicians, PhD researchers, neuroscientists, and quantitative or computational
teammates. No one person is expected to supply every kind of expertise.

## Start here

1. Open the Colab notebook with the badge above. The link will work after these
   changes are pushed to GitHub.
2. Choose **Runtime → Run all**. A CPU runtime is enough.
3. Work through the inventory, experiment timeline, join rules, schemas, and
   compact real-data example together.
4. Record the team's eventual question and decisions in the private shared
   workspace, after everyone has the same data model.

The default run downloads no published data. It uses a committed, pickle-free
catalog of the complete Figshare v2 release and a 2.9 MB real-data example.

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

- [`notebooks/zhong2025_data_atlas_colab.ipynb`](notebooks/zhong2025_data_atlas_colab.ipynb): the single team-facing notebook
- [`zhong2025/atlas.py`](zhong2025/atlas.py): validated catalog and relationship helpers
- [`zhong2025/assets/`](zhong2025/assets): one canonical copy of the safe catalog, experiment index, curated download manifest, and compact example
- [`scripts/create_data_atlas_notebook.py`](scripts/create_data_atlas_notebook.py): reviewable notebook source
- [`scripts/build_atlas_assets.py`](scripts/build_atlas_assets.py): rebuilds normalized JSON from pinned Figshare metadata
- [`scripts/build_atlas_demo.py`](scripts/build_atlas_demo.py): rebuilds the compact example from checksum-verified sources
- [`original/`](original): untouched byte-for-byte snapshot of upstream commit `ba64ac697f5d9914926baac79399e80707a5f3a6`
- [`tests/`](tests): catalog, download-safety, notebook, packaging-data, and upstream-integrity checks

All upstream paper-reproduction notebooks and figure scripts live only in
`original/`; duplicate root copies were removed. Do not edit that folder.

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
- [Released paper-reproduction code](https://github.com/MouseLand/zhong-et-al-2025)

Repository code is distributed under GNU GPL v3. The compact derivative retains
the source dataset's CC BY 4.0 terms; cite Zhong et al. and the Figshare dataset.
