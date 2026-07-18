#!/usr/bin/env python3
"""Generate notebook 06: one-recording held-out population discriminability.

The notebook deliberately does not reproduce or redraw a paper panel.  It
loads one verified recording from the released dataset, bins the released SVD
features by trial and corridor position, and computes a project-authored
descriptive curve with non-overlapping trial blocks and label-held-out folds.
"""

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("notebooks/archived/06_within_session_dprime_colab.ipynb")
NATURE = "https://www.nature.com/articles/s41586-025-09180-y"
SCIENCE_PDF = "https://mouseland.github.io/research/science.adp7429.pdf"
SCIENCE_DRIVE = (
    "https://drive.google.com/file/d/"
    "1DlmPeyaHn-thn9ILrt-rAXP96-y3IMU7/view"
)
FIGSHARE = "https://doi.org/10.6084/m9.figshare.28811129.v2"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def py(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


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
            f"""
# One-recording held-out visual discriminability

This notebook makes one compact, descriptive plot from **one released imaging
recording**. It asks whether a trial-level population coding score for the
paper's `leaf1` and `circle1` roles differs across successive, non-overlapping
blocks inside that recording.

> **Scope contract.** This is a project-authored analysis of released data. It
> is **not a reproduction or reparametrization of a Nature figure**, and one
> recording cannot establish a training, reward, cohort, or mouse-level effect.

## Exact sources

| Source | What it supports here |
|---|---|
| [Nature Figure 1]({NATURE}#Fig1) | Published corridor roles, before/after design, cellular selectivity examples, maps, and regional fractions. |
| [Nature Results: supervised and unsupervised plasticity]({NATURE}#Sec2) | The paper's reported before/after group result and cohort interpretation. |
| [Nature Methods: neural selectivity]({NATURE}#Sec20) | The published per-neuron signed $d'$ calculation on running, non-interpolated frames in the 0–4 m texture and the $|d'| ≥ 0.3$ selectivity threshold. |
| [Nature Methods: calcium processing]({NATURE}#Sec19) | Processing of the Suite2p-derived activity released by the authors. |
| [Nature Methods: retinotopy]({NATURE}#Sec25) | How recording-specific retinotopic maps were aligned and used for cortical area assignment. |
| Stringer & Pachitariu, *Science*, Figure 3: [open PDF, p. 5]({SCIENCE_PDF}#page=5) · [Drive copy, p. 5]({SCIENCE_DRIVE}#page=5) | Why a fitted coding direction must be evaluated on data not used to fit it. No artwork from that figure is redrawn here. |
| [Released dataset, Figshare v2]({FIGSHARE}) | Every plotted value and recording/file identifier used below. |
"""
        ),
        md(
            """
## Beginner path

1. Add **Zhong et al. 2025 - Neuromatch Team Workspace** as a shortcut in
   **My Drive**.
2. Choose **Runtime → Run all** once.
3. Pick one release experiment, one recording, and one visual-area grouping.
4. Press **Run one recording**.

The first chart is intentionally simple: each point is the held-out $d'$ for
one fixed, non-overlapping 40-trial block. A point is omitted when any required
fold lacks both stimulus roles. Connecting the observed blocks is a reading aid,
not a fitted learning curve.
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
    if name == "drive" or name == "zhong2025" or name.startswith("zhong2025."):
        sys.modules.pop(name, None)

drive = importlib.import_module("drive")
data = drive.setup(report=False)

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, clear_output, display
from zhong2025.learning import blockwise_dprime, prepare_session_trials
"""
        ),
        md(
            f"""
## What is held out—and what is not

The paper's [neural-selectivity method]({NATURE}#Sec20) computes a signed
**per-neuron** response contrast from frames and Figure 1 summarizes whole-session
selective-neuron fractions before and after training. This notebook computes a
different quantity: a **population coding score** from the released 400-component
SVD representation.

Within each 40-trial block, four contiguous physical-time folds are assigned
before unusable trials are filtered. For each fold, feature standardization and
the `leaf1`–`circle1` coding direction are fit on the other three folds; $d'$ is
computed only from scores in the held-out fold. The four within-fold contrasts
are averaged. Scores from different fitted directions are never pooled.

The 40-trial width, four folds, 12 retained area features, 18 position bins, and
area grouping are **project settings**, not constants reported by Zhong et al.
The SVD and area projection were derived using the complete recording, so this
is label-held-out but not a prospective online decoder. The cross-validation
principle is illustrated in [Stringer & Pachitariu Figure 3, PDF p. 5]({SCIENCE_PDF}#page=5).
"""
        ),
        py(
            """
#@title Released-data preparation and held-out analysis { display-mode: "form" }
EXPERIMENT_LABELS = {
    "sup_train1_before_learning": "Rewarded cohort · Train 1 before learning",
    "sup_train1_after_learning": "Rewarded cohort · Train 1 after learning",
    "unsup_train1_before_learning": "Unrewarded cohort · Train 1 before exposure",
    "unsup_train1_after_learning": "Unrewarded cohort · Train 1 after exposure",
}
AREA_LABELS = {
    "V1": "V1",
    "mHV": "Medial group",
    "lHV": "Lateral group",
    "aHV": "Anterior group",
}
DEFAULT_EXPERIMENT = "sup_train1_before_learning"
DEFAULT_BLOCK_TRIALS = 40
DEFAULT_FOLDS = 4
DEFAULT_FEATURES = 12
DEFAULT_POSITION_BINS = 18


def recording_options(experiment):
    # Deterministic recording labels from the pinned release index.
    return [
        (f"{recording.mouse} · {recording.date} · block {recording.block}",
         recording.recording_id)
        for recording in data.recordings(experiment=experiment)
    ]


def load_prepared(recording_id, experiment, area):
    # Load three exact release layers and construct trial-position features.
    session = data.recording(recording_id)
    source_files = {
        "behavior": session.file("behavior", experiment=experiment),
        "reduced_neural": session.file("reduced_neural"),
        "retinotopy": session.file("retinotopy"),
    }
    behavior = session.load("behavior", experiment=experiment)
    reduced = session.load("reduced_neural")
    retinotopy = session.load("retinotopy")
    prepared = prepare_session_trials(
        behavior,
        reduced,
        retinotopy,
        area=area,
        n_features=DEFAULT_FEATURES,
        n_position_bins=DEFAULT_POSITION_BINS,
        movement_rule="moving_only",
        mouse_id=session.mouse,
        recording_id=recording_id,
    )
    return prepared, source_files


def analyse_prepared(prepared, *, block_trials=DEFAULT_BLOCK_TRIALS,
                     n_folds=DEFAULT_FOLDS):
    # Compute label-held-out d′ in fixed, non-overlapping trial blocks.
    curve = blockwise_dprime(
        prepared["trial_features"],
        prepared["labels"],
        prepared["trial_id"],
        position_mask=prepared["texture_mask"],
        role_a=2,
        role_b=0,
        block_trials=block_trials,
        stride_trials=block_trials,
        n_folds=n_folds,
        min_per_role=4,
        require_complete_position_coverage=True,
    )
    return {
        "recording_id": prepared["recording_id"],
        "mouse_id": prepared["mouse_id"],
        "area": prepared["area"],
        "movement_rule": prepared["movement_rule"],
        "n_trials": int(len(prepared["trial_id"])),
        "block_trials": int(block_trials),
        "stride_trials": int(block_trials),
        "n_folds": int(n_folds),
        "n_features": int(prepared["n_features"]),
        "position_edges_m": np.asarray(prepared["position_edges_m"]),
        "curve": curve,
    }
"""
        ),
        py(
            """
#@title Beginner chart and programmer diagnostics { display-mode: "form" }
def plot_beginner(result):
    # All marks come from the selected release recording.
    curve = result["curve"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="#8d9690", linewidth=1)
    ax.plot(
        curve["midpoint"],
        curve["dprime"],
        color="#2f9e79",
        marker="o",
        linewidth=2,
    )
    ax.set(
        title=(f"Held-out leaf1–circle1 population discriminability · "
               f"{result['recording_id']} · {AREA_LABELS[result['area']]}"),
        xlabel="Trial index at block midpoint",
        ylabel="Mean held-out d′ across folds",
    )
    ax.grid(axis="y", alpha=0.2)
    plt.close(fig)
    return fig


def plot_programmer_diagnostics(result):
    # Expose support and numerator/denominator components for every block.
    curve = result["curve"]
    x = curve["midpoint"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)

    axes[0, 0].axhline(0.0, color="#8d9690", linewidth=1)
    axes[0, 0].plot(x, curve["dprime"], "o-", color="#2f9e79")
    axes[0, 0].set(title="Held-out block d′", ylabel="d′")

    axes[0, 1].plot(x, curve["separation"], "o-", label="held-out mean separation")
    axes[0, 1].plot(x, curve["spread"], "o-", label="mean within-role spread")
    axes[0, 1].set(title="d′ components")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x, curve["n_a"], "o-", label="leaf1 held-out scores")
    axes[1, 0].plot(x, curve["n_b"], "o-", label="circle1 held-out scores")
    axes[1, 0].set(title="Held-out support", xlabel="Trial midpoint", ylabel="scores")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, curve["valid_folds"], "o-", label="valid folds")
    axes[1, 1].plot(x, curve["required_folds"], "--", label="required folds")
    axes[1, 1].set(title="Fold completeness", xlabel="Trial midpoint", ylabel="folds")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Programmer diagnostics · no additional observations or fitted curve")
    plt.close(fig)
    return fig


def provenance_html(result, source_files):
    rows = "".join(
        "<tr>"
        f"<td><code>{layer}</code></td>"
        f"<td><code>{item.name}</code></td>"
        f"<td>{item.size_mib:,.1f} MiB</td>"
        f"<td><code>{item.md5}</code></td>"
        "</tr>"
        for layer, item in source_files.items()
    )
    valid = int(np.count_nonzero(np.isfinite(result["curve"]["dprime"])))
    total = int(len(result["curve"]["dprime"]))
    return HTML(
        f"<p><b>{result['recording_id']}</b> · mouse {result['mouse_id']} · "
        f"{result['area']} · {result['n_trials']} binned trials · "
        f"{valid}/{total} valid non-overlapping blocks</p>"
        "<table><thead><tr><th>layer</th><th>published file</th>"
        "<th>size</th><th>MD5</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
"""
        ),
        md(
            """
## Run one verified recording

The controls alter only explicit project settings. Changing experiment refreshes
the recording list from the pinned release index. Data are fetched and checksum
verified only after the button is pressed. **Programmer diagnostics** replot the
same held-out block summaries; they do not add samples or fit a trend.
"""
        ),
        py(
            """
#@title Choose one release recording, then run { display-mode: "form" }
experiment_control = widgets.Dropdown(
    description="Experiment",
    options=[(label, name) for name, label in EXPERIMENT_LABELS.items()],
    value=DEFAULT_EXPERIMENT,
    layout=widgets.Layout(width="680px"),
)
recording_control = widgets.Dropdown(
    description="Recording",
    options=recording_options(DEFAULT_EXPERIMENT),
    layout=widgets.Layout(width="680px"),
)
area_control = widgets.Dropdown(
    description="Area",
    options=[(label, name) for name, label in AREA_LABELS.items()],
    value="mHV",
    layout=widgets.Layout(width="420px"),
)
diagnostics_control = widgets.Checkbox(
    description="Show programmer diagnostics",
    value=False,
    indent=False,
)
run_button = widgets.Button(
    description="Run one recording",
    button_style="primary",
    icon="play",
)
run_output = widgets.Output()


def refresh_recordings(change=None):
    options = recording_options(experiment_control.value)
    recording_control.options = options
    recording_control.value = options[0][1] if options else None


def run_selected(_button=None):
    with run_output:
        clear_output(wait=True)
        if not recording_control.value:
            print("No recording is available for this experiment.")
            return None
        try:
            prepared, source_files = load_prepared(
                recording_control.value,
                experiment_control.value,
                area_control.value,
            )
            result = analyse_prepared(prepared)
        except Exception as error:
            print(f"Analysis stopped: {type(error).__name__}: {error}")
            return None
        display(plot_beginner(result))
        display(provenance_html(result, source_files))
        if diagnostics_control.value:
            display(plot_programmer_diagnostics(result))
        return result


experiment_control.observe(refresh_recordings, names="value")
run_button.on_click(run_selected)
run_panel = widgets.VBox([
    experiment_control,
    recording_control,
    area_control,
    diagnostics_control,
    run_button,
    run_output,
])
run_panel
"""
        ),
        md(
            f"""
## Reading the output without overclaiming

- A positive point means `leaf1` and `circle1` held-out scores differed in the
  positive coding-direction orientation for that block. It is not the paper's
  per-neuron $|d'| ≥ 0.3$ fraction ([published method]({NATURE}#Sec20)).
- Every point summarizes one physical-time block. The points do not constitute
  independent mice, and the line is not a statistical model.
- Missing points expose insufficient fold support; they are not interpolated.
- A within-recording change can reflect neural response, movement/coverage,
  recording drift, or the transductive representation. This notebook does not
  assign a mechanism.
- Reward or training comparisons require the eligible cohort and mouse-level
  unit used by the published design ([Nature Figure 1]({NATURE}#Fig1);
  [statistics and reproducibility]({NATURE}#Sec24)).

### Programmer result contract

`run_selected()` returns a dictionary whose `curve` contains one aligned array
per block: `start_trial`, `stop_trial`, `midpoint`, `dprime`, `separation`,
`spread`, `n_a`, `n_b`, `valid_folds`, and `required_folds`. The provenance table
records the exact three release filenames, byte-derived sizes, and pinned MD5
checksums used for the run. No output is saved automatically.
"""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"project-{index:03d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK)
    print(NOTEBOOK)
