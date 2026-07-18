#!/usr/bin/env python3
"""Generate the concise, top-down companion to the Zhong et al. paper."""

import base64
from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/04_paper_companion_colab.ipynb")
FIGURE_DIR = Path("zhong2025/assets/reference_figures")


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def py(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def published_figure(number, description):
    """Embed the complete published Nature figure, not a redrawn surrogate."""

    filename = f"nature-main-{number}.png"
    cell = md(
        f"""
### Published Figure {number}

![{description}](attachment:{filename})

**Source:** Zhong et al., *Nature* (2025),
[Figure {number}](https://www.nature.com/articles/s41586-025-09180-y#Fig{number}).
The complete published figure is shown; labels and plotted values are unchanged.
"""
    )
    cell.attachments = {
        filename: {
            "image/png": base64.b64encode((FIGURE_DIR / filename).read_bytes()).decode()
        }
    }
    return cell


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "accelerator": "CPU",
        "colab": {
            "name": NOTEBOOK.name,
            "private_outputs": True,
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
            r"""
# Zhong et al. (2025): study and data companion

This companion follows the study sequence: **experience → plasticity →
generalization → familiarity → recognition → reward prediction → later
learning**.

Reported findings are identified as the authors' conclusions. The companion
maps the study and its released data; it does not reproduce or independently
test the reported analyses.
"""
        ),
        md(
            """
## Data connection

A shortcut named **Zhong et al. 2025 - Neuromatch Team Workspace** is required
in **My Drive**. **Runtime → Run all** mounts Drive and imports the workspace
data and graph interfaces. Paths, caching, checksums, and safe loading are
handled by `drive.py`.
"""
        ),
        py(
            """
import importlib
import sys

try:
    from google.colab import drive as google_drive
except ImportError:
    pass
else:
    google_drive.mount("/content/drive", force_remount=False)
    from google.colab import output as colab_output

    colab_output.enable_custom_widget_manager()
    workspace = (
        "/content/drive/MyDrive/"
        "Zhong et al. 2025 - Neuromatch Team Workspace"
    )
    if workspace not in sys.path:
        sys.path.insert(0, workspace)

for name in tuple(sys.modules):
    if name in {"drive", "graph", "zhong2025"} or name.startswith("zhong2025."):
        sys.modules.pop(name, None)

drive = importlib.import_module("drive")
graph = importlib.import_module("graph")
data = drive.setup()
"""
        ),
        md(
            """
## 1. One paper, two linked studies

- **Imaging study:** 19 GCaMP6s mice, 89 physical recordings and 20,547–89,577
  detected neurons per recording
  ([Animals](https://www.nature.com/articles/s41586-025-09180-y#Sec11),
  [first Results section](https://www.nature.com/articles/s41586-025-09180-y#Sec2)).
  Rewarded task mice were compared with unrewarded, naive and grating-exposure
  cohorts ([Figure 1](https://www.nature.com/articles/s41586-025-09180-y#Fig1)).
- **Behaviour-only validation:** 23 different, non-imaged mice: 11 pretrained
  on natural textures, 7 pretrained on gratings and 5 without pretraining
  ([Figure 5](https://www.nature.com/articles/s41586-025-09180-y#Fig5),
  [Animals](https://www.nature.com/articles/s41586-025-09180-y#Sec11)).

The same imaging mouse can appear at several stages, and one recording can
belong to more than one analysis label. The deposited index contains 142 source
metadata rows but 133 unique experiment–recording pairs. Neither count is a
count of independent recordings, and figure sample sizes are not additive.
"""
        ),
        py(
            """
import textwrap

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from zhong2025 import experiment_rows, format_bytes, load_experiment_index

article = data.figshare(live=drive.is_colab())
experiment_index = load_experiment_index()
index_summary = experiment_index["summary"]
index_rows = experiment_rows(experiment_index)
unique_memberships = {
    (row["experiment"], row["recording_id"])
    for row in index_rows
}

print(article["title"])
print(
    f"{index_summary['unique_mice']} imaging mice · "
    f"{index_summary['unique_recordings']} recordings · "
    f"{index_summary['experiment_labels']} experiment labels · "
    f"{index_summary['associations']} source metadata rows · "
    f"{len(unique_memberships)} unique experiment-recording pairs"
)
print("Separate behaviour-only study: 23 mice (11 natural, 7 grating, 5 none)")
"""
        ),
        md(
            """
## 2. Imaging-trial structure

The head-fixed mouse ran on an air-floating ball surrounded by three screens.
Each randomized trial contained a **4 m textured corridor** followed by **2 m
of grey**. Virtual motion was held at 60 cm/s while the mouse ran above 6 cm/s
([Visual stimuli](https://www.nature.com/articles/s41586-025-09180-y#Sec14)).

In imaging sessions, a sound cue appeared at a random position from 0.5–3.5 m.
Task mice could receive water after the cue in the rewarded corridor;
unrewarded mice heard the cue but received no task reward. Position, running,
licks, cues, rewards and calcium activity were aligned in time
([Behavioural training](https://www.nature.com/articles/s41586-025-09180-y#Sec17)).
"""
        ),
        published_figure(
            1,
            "Supervised and unsupervised plasticity after exposure to visual textures",
        ),
        md(
            r"""
## 3. The study sequence

| Paper step | What changed | Main readout | Authors' reported conclusion |
|---|---|---|---|
| **[Fig. 1 — Train 1](https://www.nature.com/articles/s41586-025-09180-y#Fig1)** | Rewarded training or matched unrewarded exposure to familiar textures; released matched imaging gaps span 9–24 and 3–12 days, respectively | Per-neuron corridor selectivity, $d'$, mapped across visual areas | Medial-HVA selectivity increased in both task and unrewarded mice; the strongest reward-specific difference was anterior ([Results](https://www.nature.com/articles/s41586-025-09180-y#Sec2)) |
| **[Fig. 2 — Test 1](https://www.nature.com/articles/s41586-025-09180-y#Fig2)** | New crops from the same categories (`leaf2`, `circle2`) | Position sequences, held-out coding direction and similarity | Generalization followed visual features rather than absolute corridor position ([Results](https://www.nature.com/articles/s41586-025-09180-y#Sec3)) |
| **[Fig. 3 — Train 2](https://www.nature.com/articles/s41586-025-09180-y#Fig3)** | `leaf2` changed from novel to familiar | Novelty responses, `leaf1`–`leaf2` selectivity and orthogonalization | Novelty responses declined in V1/lateral areas; fine discrimination emerged strongly in medial areas with or without reward ([Results](https://www.nature.com/articles/s41586-025-09180-y#Sec4)) |
| **Extended Data 6 — Test 2** | A third leaf exemplar (`leaf3`) was introduced | Licking and projection on the `leaf1`–`leaf2` axis | Both behaviour and population activity treated `leaf3` more like the unrewarded `leaf2` exemplar |
| **Extended Data 6–7 — Test 3** | Familiar `leaf1` patches were spatially rearranged | Licking and location-specific neural sequences | Mice still recognized the corridor; responses followed visual patches rather than memorized absolute positions |
| **[Fig. 4 — Reward signal](https://www.nature.com/articles/s41586-025-09180-y#Fig4)** | Early versus late cues, licked versus unlicked trials, and later tests | Rastermap discovery followed by a late-versus-early-cue $d'$ | The reported late-cue population was concentrated mainly in anterior HVAs of task mice ([Results](https://www.nature.com/articles/s41586-025-09180-y#Sec6); [estimator](https://www.nature.com/articles/s41586-025-09180-y#Sec22)) |
| **[Fig. 5 — Behavioural validation](https://www.nature.com/articles/s41586-025-09180-y#Fig5)** | 10 days of natural-texture, grating or no pretraining before a five-day task | Lick discrimination, first licks and trials per day | Natural-texture pretraining accelerated early rewarded learning in this separate behaviour-only cohort ([Results](https://www.nature.com/articles/s41586-025-09180-y#Sec7)) |

`leaf` and `circle` are canonical shorthand. Some mice saw rock/brick-like
textures. Physical identity is stored in each session's `WallName`; experimental
role is stored in `stim_id`. In an `unsup_*_after_learning` filename, “learning”
means the matched study stage; those mice did not receive task rewards.
"""
        ),
        published_figure(
            2,
            "Visual rather than spatial coding of familiar and novel textures",
        ),
        published_figure(
            3,
            "Plasticity produced by fine visual discrimination training",
        ),
        published_figure(
            4,
            "Reward-prediction responses in anterior higher visual areas",
        ),
        published_figure(
            5,
            "Effect of visual pretraining on later task learning",
        ),
        md(
            """
## 4. Paper-step explorer

The paper-step control selects an experiment stage. **Run flow** displays its
place in the experiment, supporting release labels, controls, measurements, and
the authors' reported conclusion. Counts are deduplicated by physical recording.
"""
        ),
        py(
            r"""
#@title Paper-step transformations { display-mode: "form" }
PAPER_STEPS = {
    "fig1": {
        "label": "Fig. 1 · familiar-stimulus plasticity",
        "title": "Supervised and unsupervised plasticity",
        "question": "Does familiar-stimulus plasticity require task feedback?",
        "changed": "Released matched imaging gaps span 9–24 days for task mice and 3–12 days for unrewarded mice.",
        "controlled": "The visual textures and running-only analysis periods were matched; gratings controlled for generic VR exposure.",
        "measured": "Corridor d′ per neuron, |d′| ≥ 0.3, then density and fraction by cortical region.",
        "reported": "Medial-HVA selectivity increased after both task training and unrewarded natural-texture exposure, but not grating exposure.",
        "labels": [
            "sup_train1_before_learning", "sup_train1_after_learning",
            "unsup_train1_before_learning", "unsup_train1_after_learning",
            "train1_before_grating", "train1_after_grating",
        ],
        "highlight": [0, 1],
    },
    "fig2": {
        "label": "Fig. 2 · visual or spatial coding",
        "title": "New category exemplars in Test 1",
        "question": "Did the learned representation follow visual features or corridor position?",
        "changed": "New spatial crops leaf2 and circle2 appeared without reward.",
        "controlled": "The category source was preserved while the spatial arrangement changed.",
        "measured": "Held-out position-sequence correlations, coding-direction projections and similarity indices.",
        "reported": "Category readout generalized by visual identity; absolute position sequences did not.",
        "labels": ["sup_test1", "unsup_test1", "naive_test1"],
        "highlight": [2],
    },
    "fig3": {
        "label": "Fig. 3 · novelty and orthogonalization",
        "title": "Leaf2 becomes familiar in Train 2",
        "question": "How does a novel exemplar change with repeated experience?",
        "changed": "Leaf2 was measured when new and again after continued exposure.",
        "controlled": "Task and unrewarded groups experienced the same exemplar; naive and grating cohorts supplied references.",
        "measured": "Novelty-selective fractions, leaf1–leaf2 d′ and projection on the leaf1–circle1 coding axis.",
        "reported": "V1/lateral novelty responses declined, while leaf1 and leaf2 became more distinct, especially in medial HVAs, in both main cohorts.",
        "labels": [
            "sup_train2_before_learning", "sup_train2_after_learning",
            "unsup_train2_before_learning", "unsup_train2_after_learning",
        ],
        "highlight": [3, 4],
    },
    "test2": {
        "label": "Test 2 · recognition of leaf3",
        "title": "A third exemplar tests recognition memory",
        "question": "Was familiar leaf1 represented as a specific exemplar, not only a category?",
        "changed": "Novel leaf3 was shown beside familiar leaf1, trained leaf2 and circle1.",
        "controlled": "The new stimulus belonged to the same visual category but had never been seen.",
        "measured": "Licking and projection on the leaf1–leaf2 coding axis.",
        "reported": "Leaf3 was treated more like unrewarded leaf2 than familiar leaf1 in behaviour and population activity.",
        "labels": ["sup_test2", "unsup_test2", "naive_test2", "test2_after_grating"],
        "highlight": [5],
    },
    "test3": {
        "label": "Test 3 · spatially swapped leaf1",
        "title": "Rearranging the familiar corridor",
        "question": "Did recognition depend on fixed absolute positions?",
        "changed": "Familiar leaf1 patches were presented in two spatially swapped layouts.",
        "controlled": "Visual patches were preserved while their corridor locations changed.",
        "measured": "Licking, first-lick locations and position-specific neural sequence correspondence.",
        "reported": "Mice recognized swapped corridors as leaf1 and neural responses tracked the relocated visual patches.",
        "labels": ["sup_test3", "unsup_test3", "naive_test3"],
        "highlight": [6],
    },
    "fig4": {
        "label": "Fig. 4 · reward prediction",
        "title": "What reward added in anterior HVAs",
        "question": "Which population signal distinguished task learning from unrewarded exposure?",
        "changed": "Cue timing, reward delivery and the animal's choice varied across otherwise matched trials.",
        "controlled": "Early- and late-cue trials matched stimulus and reward while changing anticipatory duration.",
        "measured": "Rastermap-guided discovery, then cross-validated late-versus-early-cue d′ and cue/lick alignment.",
        "reported": "Anterior-HVA activity ramped before licking, was suppressed by reward and tracked reward expectation only in task mice.",
        "labels": [
            "sup_train1_before_learning", "sup_train1_after_learning",
            "unsup_train1_before_learning", "unsup_train1_after_learning",
        ],
        "highlight": [1, 2, 5, 6],
    },
    "fig5": {
        "label": "Fig. 5 · faster task learning",
        "title": "Separate behaviour-only validation",
        "question": "Does relevant unrewarded pretraining help later rewarded learning?",
        "changed": "Ten days of natural-texture, grating or no pretraining preceded the same task.",
        "controlled": "All cohorts then received one passive-reward day and four active-reward days and ran similar trial counts.",
        "measured": "Lick discrimination, first-lick distributions and trials per day.",
        "reported": "Natural-texture pretraining accelerated early discrimination; all groups approached strong performance by the fifth day.",
        "labels": [],
        "behavior_counts": {"Natural textures": 11, "Gratings": 7, "No pretraining": 5},
        "highlight": [7],
    },
}

STUDY_SEQUENCE = [
    "Train 1\nbefore", "Train 1\nafter", "Test 1",
    "Train 2\nbefore", "Train 2\nafter", "Test 2\nleaf3",
    "Test 3\nswaps", "Separate\nbehaviour study",
]


@graph.node(outputs="paper")
def load_paper():
    return {
        "steps": PAPER_STEPS,
        "rows": experiment_rows(experiment_index),
        "sequence": STUDY_SEQUENCE,
    }


@graph.node(outputs="chapter")
def choose_paper_step(paper, paper_step="fig1"):
    if paper_step not in paper["steps"]:
        raise ValueError("paper_step must name an available paper step")
    return {
        "key": paper_step,
        "content": paper["steps"][paper_step],
        "rows": paper["rows"],
        "sequence": paper["sequence"],
    }


@graph.node(outputs="evidence")
def map_released_evidence(chapter):
    content = chapter["content"]
    if "behavior_counts" in content:
        counts = content["behavior_counts"]
        return {
            **chapter,
            "count_labels": list(counts),
            "counts": list(counts.values()),
            "count_unit": "mice (separate, non-imaged cohort)",
        }

    count_labels, counts = [], []
    for experiment in content["labels"]:
        recordings = {
            row["recording_id"]
            for row in chapter["rows"]
            if row["experiment"] == experiment
        }
        count_labels.append(experiment)
        counts.append(len(recordings))
    return {
        **chapter,
        "count_labels": count_labels,
        "counts": counts,
        "count_unit": "unique physical recordings",
    }


@graph.node(outputs="figure")
def plot_paper_step(evidence):
    content = evidence["content"]
    fig = plt.figure(figsize=(13, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.8, 2.2], width_ratios=[1.05, 1.35])
    timeline = fig.add_subplot(grid[0, :])
    counts = fig.add_subplot(grid[1, 0])
    explanation = fig.add_subplot(grid[1, 1])

    x = np.arange(len(evidence["sequence"]))
    timeline.plot(x[:7], np.zeros(7), color="#777777", linewidth=2, zorder=1)
    timeline.plot([6, 7], [0, 0], color="#aaaaaa", linewidth=1.5, linestyle="--")
    selected = set(content["highlight"])
    colors = ["#f28e2b" if index in selected else "#c7c7c7" for index in x]
    timeline.scatter(x, np.zeros_like(x), s=160, color=colors, edgecolor="#555555", zorder=2)
    for index, label in enumerate(evidence["sequence"]):
        timeline.text(index, -0.13, label, ha="center", va="top", fontsize=9)
    timeline.set(xlim=(-0.45, 7.45), ylim=(-0.55, 0.28), title=content["title"])
    timeline.axis("off")

    y = np.arange(len(evidence["counts"]))
    bars = counts.barh(y, evidence["counts"], color="#4c78a8")
    counts.set_yticks(y, labels=evidence["count_labels"])
    counts.invert_yaxis()
    counts.set(title="Released evidence used at this step", xlabel=evidence["count_unit"])
    counts.bar_label(bars, padding=4)
    counts.set_xlim(0, max(evidence["counts"]) * 1.28)
    counts.tick_params(axis="y", labelsize=8)

    lines = [
        ("Question", content["question"]),
        ("What changed", content["changed"]),
        ("What stayed controlled", content["controlled"]),
        ("What was measured", content["measured"]),
        ("Authors reported", content["reported"]),
    ]
    y_text = 0.98
    for heading, body in lines:
        explanation.text(0, y_text, heading, weight="bold", va="top", fontsize=10)
        y_text -= 0.075
        explanation.text(
            0, y_text, textwrap.fill(body, 70), va="top", fontsize=9, linespacing=1.35
        )
        y_text -= 0.19
    explanation.set_xlim(0, 1)
    explanation.set_ylim(0, 1)
    explanation.axis("off")
    plt.close(fig)
    return fig
"""
        ),
        py(
            """
#@title Paper-step explorer { display-mode: "form" }
paper_graph = graph.Graph(
    "The paper from question to evidence",
    load_paper,
    choose_paper_step,
    map_released_evidence,
    plot_paper_step,
)
paper_panel = paper_graph.widget(
    controls={
        "paper_step": widgets.Dropdown(
            description="Paper step",
            options=[(step["label"], key) for key, step in PAPER_STEPS.items()],
            value="fig1",
        ),
    },
    show="figure",
)
paper_panel
"""
        ),
        md(
            r"""
## 5. Analysis pipeline from acquisition to inference

1. **Acquire:** the two-photon mesoscope and ScanImage recorded broad visual
   cortex while behaviour was logged.
2. **Process:** Suite2p performed motion correction, ROI and cell detection,
   neuropil correction and spike deconvolution (0.75 s decay). The paper's
   reported neural analyses used these deconvolved traces.
3. **Match behaviour:** the principal selectivity analysis kept moving frames
   in the 0–4 m textured region, excluding reward stops and the grey corridor.
4. **Calculate selectivity:** for two corridors the paper defined

   $$d' = \frac{\mu_1-\mu_2}{\sigma_1/2+\sigma_2/2}
        = \frac{2(\mu_1-\mu_2)}{\sigma_1+\sigma_2},$$

   with $|d'|\geq0.3$ used as a selectivity threshold.
5. **Test generalization:** neuron selection used training trials; sequence and
   coding-direction results were evaluated on held-out or new-stimulus trials.
6. **Locate effects:** retinotopy grouped cells into V1, medial, lateral and
   anterior visual regions.
7. **Quantify the task-only signal:** Rastermap revealed a candidate population;
   a cross-validated late-versus-early-cue $d'$ then tested it across mice.
8. **Infer at the right level:** trials and neurons are nested measurements.
   Group conclusions depend on mice or recording sessions, not tens of
   thousands of cells treated as independent animals.

The release also contains 400-component SVD files for efficient population
exploration. Those PCs are session-specific conveniences, not replacements for
the deconvolved neuron traces used in the paper's reported statistics.
"""
        ),
        md(
            """
## 6. Released files by paper label

The release-mapping graph does not load a large array. It resolves a scientific
experiment label to physical recordings and one selected data layer, then shows
the exact files and sizes. Hollow ports specify the label and layer.
"""
        ),
        py(
            """
#@title Release-mapping transformations { display-mode: "form" }
@graph.node(outputs="catalog")
def load_release_catalog():
    return {"dataset": data}


@graph.node(outputs="selection")
def choose_released_layer(
    catalog,
    experiment_label="sup_train1_before_learning",
    data_layer="behavior",
):
    if experiment_label not in catalog["dataset"].experiments:
        raise ValueError("experiment_label must name a published imaging experiment")
    if data_layer not in {"behavior", "reduced_neural", "full_neural", "retinotopy"}:
        raise ValueError("data_layer must name an available data layer")
    return {
        "dataset": catalog["dataset"],
        "experiment": experiment_label,
        "layer": data_layer,
    }


@graph.node(outputs="summary")
def resolve_released_files(selection):
    recordings = selection["dataset"].recordings(experiment=selection["experiment"])
    files = {}
    for recording in recordings:
        item = recording.file(
            selection["layer"],
            experiment=selection["experiment"] if selection["layer"] == "behavior" else None,
        )
        files[item.name] = item
    files = sorted(files.values(), key=lambda item: item.size_bytes, reverse=True)
    return {
        **selection,
        "recordings": recordings,
        "mice": sorted({recording.mouse for recording in recordings}),
        "files": files,
        "total_bytes": sum(item.size_bytes for item in files),
    }


@graph.node(outputs="figure")
def plot_released_files(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    files = summary["files"]
    labels = [
        item.recording_id or item.name.replace("Beh_", "").replace(".npy", "")
        for item in files
    ]
    values = [item.size_bytes / 2**30 for item in files]
    bars = axes[0].barh(labels, values, color="#59a14f")
    axes[0].invert_yaxis()
    axes[0].set(title=f"{summary['layer'].replace('_', ' ').title()} files", xlabel="GiB")
    axes[0].bar_label(
        bars, labels=[format_bytes(item.size_bytes) for item in files],
        padding=4, fontsize=8,
    )
    axes[0].set_xlim(0, max(values) * 1.32)
    axes[0].tick_params(axis="y", labelsize=8)

    preview = "\\n".join(f"• {item.name}" for item in files[:6])
    if len(files) > 6:
        preview += f"\\n• … {len(files) - 6} more"
    details = (
        f"Experiment\\n{summary['experiment']}\\n\\n"
        f"{len(summary['mice'])} mice\\n"
        f"{len(summary['recordings'])} physical recordings\\n"
        f"{len(files)} unique files · {format_bytes(summary['total_bytes'])}\\n\\n"
        f"Selected files (metadata only)\\n{preview}"
    )
    axes[1].text(0, 1, details, va="top", fontsize=10, linespacing=1.35)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")
    fig.suptitle("Paper label → recording → released data layer")
    plt.close(fig)
    return fig
"""
        ),
        py(
            """
#@title Release mapping { display-mode: "form" }
release_graph = graph.Graph(
    "Find the data behind a paper stage",
    load_release_catalog,
    choose_released_layer,
    resolve_released_files,
    plot_released_files,
)
release_panel = release_graph.widget(
    controls={
        "experiment_label": widgets.Dropdown(
            description="Experiment",
            options=list(data.experiments),
            value="sup_train1_before_learning",
        ),
        "data_layer": widgets.Dropdown(
            description="Data layer",
            options=[
                ("Behaviour", "behavior"),
                ("Reduced neural (SVD)", "reduced_neural"),
                ("Full neural", "full_neural"),
                ("Retinotopy", "retinotopy"),
            ],
            value="behavior",
        ),
    },
    show="figure",
)
release_panel
"""
        ),
        md(
            """
## 7. What the release can and cannot answer

| Layer | What it contributes |
|---|---|
| `Beh_*.npy` | Trial identity, physical `WallName`, canonical `stim_id`, position, running, licks, cues and rewards |
| `*_neural_data.npy` | Full deconvolved neuron activity used for neuron-level analyses |
| `*_SVD_dec.npy` | About 400 components for efficient population exploration |
| `*_trans.npz` | Neuron coordinates and cortical-area assignments |
| Three behaviour-only files | The separate natural-pretraining, grating-pretraining and no-pretraining cohorts in Fig. 5 |

Trial counts vary by session and stimulus, so `ntrials` and `WallName` should be
read directly rather than treated as fixed. The release does not establish
longitudinal registration of the same individual neurons across days.
Population representations should be compared across stages unless explicit
cell-registration evidence is available.

`03_dataset_walkthrough_colab.ipynb` provides hands-on data exploration;
`00_use_janelia_drive_colab.ipynb` provides released-file selection and loading.

**Exact sources:** [Nature Results: supervised and unsupervised
plasticity](https://www.nature.com/articles/s41586-025-09180-y#Sec2),
[Neural-selectivity Methods](https://www.nature.com/articles/s41586-025-09180-y#Sec20),
[Statistics and reproducibility](https://www.nature.com/articles/s41586-025-09180-y#Sec24),
and [Janelia Figshare v2](https://doi.org/10.25378/janelia.28811129.v2).
"""
        ),
    ]

    for index, cell in enumerate(notebook.cells):
        cell.id = f"paper-companion-{index:03d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
    print(NOTEBOOK)
