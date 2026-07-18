#!/usr/bin/env python3
"""Generate notebook 01: two cited tutorials for the release Dataset and Graph APIs.

The notebook teaches data access and workflow mechanics against the pinned
Figshare v2 release. It does not simulate figures or calculate a scientific
result.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/01_dataset_atlas_and_graph_tutorials.ipynb")
NATURE = "https://www.nature.com/articles/s41586-025-09180-y"
FIGSHARE = "https://doi.org/10.25378/janelia.28811129.v2"
FIGSHARE_API = "https://api.figshare.com/v2/articles/28811129/versions/2"
PAPER_CODE = "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def py(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "accelerator": "CPU",
        "colab": {
            "name": NOTEBOOK.name,
            "private_outputs": True,
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }

    notebook.cells = [
        md(
            f"""
# Atlas companion: Dataset and Graph tutorials

This notebook teaches the same release-inspection task in two forms:

1. **Dataset tutorial** — discover the `data` API, filter the pinned catalog,
   resolve one recording, inspect its exact published files, and load its small
   retinotopy layer.
2. **Graph tutorial** — express those same operations as named functions with
   visible input/output ports, inspect the wiring, run the flow, and use an
   interactive recording selector.

Every example uses the published Zhong et al. release. It does not simulate a
figure or calculate a scientific result.

- [Article main text]({NATURE}#Sec1) · [Methods]({NATURE}#Sec9) ·
  [Data availability]({NATURE}#data-availability) ·
  [Code availability]({NATURE}#code-availability)
- [Published dataset, version 2]({FIGSHARE}) ·
  [versioned Figshare API record]({FIGSHARE_API})
"""
        ),
        md(
            f"""
## Published-figure and method index

Use these links when moving from release mechanics to scientific analysis.

| Published item | Contents stated in the paper's figure caption | Analysis method |
|---|---|---|
| [Figure 1]({NATURE}#Fig1) | VR task and training timeline; licking; imaging; neural selectivity; cortical distributions and regional fractions | [Neural selectivity]({NATURE}#Sec20), [retinotopy]({NATURE}#Sec25) |
| [Figure 2]({NATURE}#Fig2) | Test-1 stimuli and licking; spatial sequences; coding-direction projections and similarity index | [Coding direction and similarity index]({NATURE}#Sec21) |
| [Figure 3]({NATURE}#Fig3) | Novel and adapted stimuli; selective-neuron distributions; coding-direction projections | [Neural selectivity]({NATURE}#Sec20), [coding direction and similarity index]({NATURE}#Sec21) |
| [Figure 4]({NATURE}#Fig4) | Rastermap view and the late-cue-versus-early-cue reward-prediction analysis | [Reward-prediction neurons]({NATURE}#Sec22) |
| [Figure 5]({NATURE}#Fig5) | Behavioural learning after natural-image, grating, or no pretraining | [Faster task learning after pretraining]({NATURE}#Sec7) |

Protocol details are in [imaging acquisition]({NATURE}#Sec13),
[visual stimuli]({NATURE}#Sec14), and
[behavioural training]({NATURE}#Sec17). Processing and inference details are in
[processing of calcium-imaging data]({NATURE}#Sec19) and
[statistics and reproducibility]({NATURE}#Sec24).
"""
        ),
        md(
            f"""
# Tutorial 1 — Dataset: find, inspect, and load published data

## 1. Connect to the fixed team workspace

The setup imports `drive.py` and `graph.py` from the exact shared-workspace
path below. It also checks the bundled, version-2 inventory before continuing.
The final line is intentionally just `data`: its notebook card lists the
variables available on the connected object and every public function you can
call.

Release provenance: [Figshare version 2]({FIGSHARE}) and the paper's
[Data availability section]({NATURE}#data-availability).
"""
        ),
        py(
            """
#@title Connect to the release { display-mode: "form" }
import importlib
import sys
from pathlib import Path

from google.colab import drive as google_drive

google_drive.mount("/content/drive", force_remount=False)
WORKSPACE = Path(
    "/content/drive/MyDrive/Zhong et al. 2025 - Neuromatch Team Workspace"
)
required = (
    WORKSPACE / "drive.py",
    WORKSPACE / "graph.py",
    WORKSPACE / "zhong2025/assets/figshare-v2-inventory.json",
    WORKSPACE / "zhong2025/assets/imaging-experiment-index.json",
)
missing = [str(path) for path in required if not path.is_file()]
assert not missing, "Missing required workspace files:\\n" + "\\n".join(missing)

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
for module_name in tuple(sys.modules):
    if (
        module_name in {"drive", "graph", "zhong2025"}
        or module_name.startswith("zhong2025.")
    ):
        sys.modules.pop(module_name, None)
drive = importlib.import_module("drive")
graph = importlib.import_module("graph")
assert Path(drive.__file__).resolve() == (WORKSPACE / "drive.py").resolve()
assert Path(graph.__file__).resolve() == (WORKSPACE / "graph.py").resolve()
assert drive.REPRESENTATION_API_VERSION >= 1, (
    "The shared workspace has a stale drive.py. Replace it with the current "
    "workspace source, then rerun this cell."
)
assert callable(getattr(drive.Recording, "to_dict", None))

data = drive.setup(mount=False, report=False)
assert data.release["article_id"] == 28811129
assert data.release["version"] == 2
assert len(data.files) == 297
assert sum(item.size_bytes for item in data.files) == 452_233_500_962
data
"""
        ),
        md(
            f"""
## 2. Read the release-level variables

`data.connected`, `data.files`, `data.experiments`, and `data.folders` are
variables; `data.figshare(...)` is a function. Here `live=False` reads the
bundled snapshot of the [versioned Figshare API record]({FIGSHARE_API}), so this
cell makes no network request and downloads no release file.
"""
        ),
        py(
            """
release = data.figshare(live=False)
dataset_summary = {
    "connected": data.connected,
    "mounted_folders": data.folders,
    "published_files": len(data.files),
    "published_bytes": sum(item.size_bytes for item in data.files),
    "experiment_labels": len(data.experiments),
    "article_id": release["id"],
    "version": release["version"],
    "doi": release["doi"],
    "license": release["license"],
}
dataset_summary
"""
        ),
        md(
            f"""
## 3. Filter recordings by experiment or mouse

`data.recordings(...)` joins the released experiment index to the pinned file
inventory. The experiment labels come from `Imaging_Exp_info.npy`
([exact Figshare file 54183854](https://ndownloader.figshare.com/files/54183854)).
The paper defines the experimental sequence in [Figure 1]({NATURE}#Fig1) and
[behavioural training]({NATURE}#Sec17).
"""
        ),
        py(
            """
supervised_test1 = data.recordings(experiment="sup_test1")
tx119_sessions = data.recordings(mouse="TX119")

{
    "sup_test1_recording_ids": [
        session.recording_id for session in supervised_test1
    ],
    "TX119_recording_ids": [
        session.recording_id for session in tx119_sessions
    ],
}
"""
        ),
        md(
            f"""
## 4. Resolve one recording and inspect its exact files

`data.recording(...)` returns one `Recording`. Its `mouse`, `date`, `block`,
`experiments`, `layers`, and `files` fields describe release identity; they do
not infer experimental meaning. The table links each row to its exact Figshare
file and prints the published byte count and MD5 checksum.

The selected recording, `TX119_2023_12_24_1`, is associated with two experiment
labels in the released index. File identity comes from
[Figshare version 2]({FIGSHARE_API}).
"""
        ),
        py(
            """
from IPython.display import Markdown, display

session = data.recording("TX119_2023_12_24_1")
print(
    session.recording_id,
    session.mouse,
    session.date,
    session.block,
    session.experiments,
    session.layers,
    sep="\\n",
)

rows = []
for item in session.files:
    link = f"https://ndownloader.figshare.com/files/{item.id}"
    rows.append(
        f"| `{item.category}` | [{item.name}]({link}) | "
        f"{item.size_bytes:,} | `{item.md5}` |"
    )

display(Markdown(
    "| Layer | Exact published file | Bytes | MD5 |\\n"
    "|---|---|---:|---|\\n" + "\\n".join(rows)
))
"""
        ),
        md(
            f"""
## 5. Load one small data layer and inspect its variables

`session.load("retinotopy")` resolves the exact related file, verifies its MD5,
and opens the NumPy archive. For this example it loads
[`TX119_2023_12_24_trans.npz`](https://ndownloader.figshare.com/files/54184070),
a 983,358-byte file. The cell prints only array names, shapes, and dtypes; it
does not display or interpret values. Retinotopy acquisition and processing are
described in [Methods: retinotopy]({NATURE}#Sec25).
"""
        ),
        py(
            """
retinotopy_file = session.file("retinotopy")
retinotopy = session.load("retinotopy")

{
    "published_file": retinotopy_file.name,
    "figshare_file_id": retinotopy_file.id,
    "bytes": retinotopy_file.size_bytes,
    "md5": retinotopy_file.md5,
    "variables": {
        name: {"shape": tuple(value.shape), "dtype": str(value.dtype)}
        for name, value in retinotopy.items()
    },
}
"""
        ),
        md(
            f"""
## 6. Load another layer only when you need it

The other `Recording.load(...)` calls use the same interface, but their files
are larger. Choose an experiment explicitly when one recording belongs to more
than one experiment.

```python
# 66,967,262-byte reduced-neural file for this recording
reduced = session.load("reduced_neural")

# Behaviour is stored per experiment; this recording has two experiment labels
behaviour = session.load("behavior", experiment="unsup_test1")

# Full neural data is 1,589,937,153 bytes here; raise the guard deliberately
full = session.load("full_neural", max_gib=2.0)
```

Array definitions and preprocessing belong to
[processing of calcium-imaging data]({NATURE}#Sec19), not to this access layer.
"""
        ),
        md(
            f"""
# Tutorial 2 — Graph: run the same task as visible nodes and ports

The graph below performs the same four operations as Tutorial 1:

1. resolve one recording with `data.recording(...)`;
2. summarize its exact published files;
3. load the related retinotopy archive with `Recording.load(...)`;
4. summarize the loaded variable names, shapes, and dtypes.

A node is an ordinary Python function. Function parameters are input ports;
`outputs=...` names output ports. When an earlier output and a later input have
the same name, `graph.Graph(...)` wires them together. An input with no upstream
producer—`recording_id` here—is a run setting and can become a widget control.

This changes orchestration only. The data source remains the exact
[Figshare version-2 release]({FIGSHARE_API}).
"""
        ),
        py(
            """
@graph.node(outputs="recording")
def resolve_recording(recording_id="TX119_2023_12_24_1"):
    return data.recording(recording_id)


@graph.node(outputs="file_inventory")
def summarize_files(recording):
    return tuple(
        {
            "category": item.category,
            "name": item.name,
            "figshare_file_id": item.id,
            "bytes": item.size_bytes,
            "md5": item.md5,
        }
        for item in recording.files
    )


@graph.node(outputs="retinotopy")
def load_retinotopy(recording):
    return recording.load("retinotopy")


@graph.node(outputs="retinotopy_inventory")
def summarize_retinotopy(retinotopy):
    return {
        name: {"shape": tuple(value.shape), "dtype": str(value.dtype)}
        for name, value in retinotopy.items()
    }


atlas_flow = graph.Graph(
    "Inspect one released recording",
    resolve_recording,
    summarize_files,
    load_retinotopy,
    summarize_retinotopy,
)
"""
        ),
        md(
            """
## 1. Inspect the graph before running it

`describe()` returns a plain dictionary of nodes, input ports, output ports,
settings, and connections. `diagram()` renders that same contract. Neither call
loads a release file.
"""
        ),
        py(
            """
atlas_flow.describe()
"""
        ),
        py(
            """
atlas_flow.diagram()
"""
        ),
        md(
            """
## 2. Run the graph and inspect named outputs

`run()` executes the nodes once in their declared order. The returned `Run`
keeps every named port, the actual setting, execution order, and per-node
timings. This run loads the same small retinotopy file used in Tutorial 1.
"""
        ),
        py(
            """
atlas_run = atlas_flow.run()

{
    "settings": dict(atlas_run.settings),
    "execution_order": atlas_run.order,
    "seconds_by_node": dict(atlas_run.timings),
    "exact_files": atlas_run["file_inventory"],
    "retinotopy_variables": atlas_run["retinotopy_inventory"],
}
"""
        ),
        md(
            """
## 3. Use the same graph interactively

The dropdown is populated from `data.recordings(mouse="TX119")`. Changing it
feeds the unconnected `recording_id` port. Press **Run** to execute the flow;
use **Run to** to stop at a selected node, and inspect any produced output port
without rerunning. The widget does not run automatically when it is created.
"""
        ),
        py(
            """
tx119_recording_ids = [
    current.recording_id for current in data.recordings(mouse="TX119")
]
atlas_panel = atlas_flow.widget(
    controls={"recording_id": tx119_recording_ids},
    show="retinotopy_inventory",
)
atlas_panel
"""
        ),
        md(
            f"""
## Authors' analysis-code index

The tutorials above stop at data discovery and loading. For scientific
analysis, use the authors' `paper` branch and the corresponding Methods section:

- [d-prime definition]({PAPER_CODE}#L370-L374)
- [selectivity threshold, cortical masking, coordinate transform, and density normalization]({PAPER_CODE}#L394-L416)
- [frame selection and per-neuron selectivity calculation]({PAPER_CODE}#L418-L441)
- [held-out sorting and display-trial split for sequence plots]({PAPER_CODE}#L599-L703)
- [ten-fold reward-response selection and held-out evaluation]({PAPER_CODE}#L814-L882)

The corresponding paper methods are
[neural selectivity]({NATURE}#Sec20),
[coding direction and similarity index]({NATURE}#Sec21), and
[reward-prediction neurons]({NATURE}#Sec22).
"""
        ),
        md(
            f"""
## Continue with the analysis notebooks

- `archived/03_dataset_walkthrough_colab.ipynb` inspects selected released arrays.
- `archived/04_paper_companion_colab.ipynb` follows the paper's experimental sequence.
- `archived/05_reward_dprime_dynamics_colab.ipynb` and
  `archived/06_within_session_dprime_colab.ipynb` address reward-related analyses.

Interpret any computed result against the relevant published figure and method
above, the paper's [Statistics and reproducibility]({NATURE}#Sec24), and the
exact release-file records used by that computation.
"""
        ),
    ]

    for index, cell in enumerate(notebook.cells):
        cell.id = f"companion-{index:03d}"
        if cell.cell_type == "code":
            cell.metadata = {"id": cell.id}
            cell.execution_count = None
            cell.outputs = []
    return notebook


def main() -> None:
    notebook = build()
    nbf.validate(notebook)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
