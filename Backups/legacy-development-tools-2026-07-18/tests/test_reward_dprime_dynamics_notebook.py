from __future__ import annotations

from io import BytesIO
from pathlib import Path
import runpy
import time
import warnings

import nbformat
import numpy as np
import pytest


NOTEBOOK = Path("notebooks/archived/05_reward_dprime_dynamics_colab.ipynb")
GENERATOR = Path("scripts/create_reward_dprime_dynamics_notebook.py")
GRAPH_NAMES = ("fetch_graph", "recording_graph", "hypothesis_graph", "position_graph")
PANEL_NAMES = ("fetch_panel", "recording_panel", "hypothesis_panel", "position_panel")


def _generated_notebook():
    return runpy.run_path(str(GENERATOR))["build_notebook"]()


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def _execute_generated(monkeypatch, mpl_config_dir):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_config_dir))
    monkeypatch.delenv("ZHONG2025_DATASET_ROOT", raising=False)
    monkeypatch.syspath_prepend(str(Path.cwd()))
    namespace = {"__name__": "__reward_dynamics_smoke__"}
    started = time.monotonic()
    for index, cell in enumerate(_generated_notebook().cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"reward-dynamics-cell-{index}", "exec"), namespace)
    return namespace, time.monotonic() - started


def _assert_valid_plot(figure):
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    assert isinstance(figure, Figure)
    assert figure.axes
    finite_artist = False
    for axis in figure.axes:
        assert np.isfinite((*axis.get_xlim(), *axis.get_ylim())).all()
        for line in axis.lines:
            x = np.asarray(line.get_xdata(orig=False), dtype=float).reshape(-1)
            y = np.asarray(line.get_ydata(orig=False), dtype=float).reshape(-1)
            assert not np.isinf(x).any() and not np.isinf(y).any()
            finite_artist |= bool(x.size and y.size and np.any(np.isfinite(x) & np.isfinite(y)))
        for image in axis.images:
            values = np.asarray(image.get_array(), dtype=float)
            assert not np.isinf(values).any()
            finite_artist |= bool(np.any(np.isfinite(values)))
        for collection in axis.collections:
            offsets = np.ma.asarray(collection.get_offsets(), dtype=float)
            if offsets.size > 2:
                values = np.asarray(offsets.filled(np.nan), dtype=float)
                assert not np.isinf(values).any()
                finite_artist |= bool(np.any(np.isfinite(values)))
        finite_artist |= bool(axis.patches)
    assert finite_artist
    figure.canvas.draw()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=48)
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.tell() > 1_000
    plt.close(figure)


@pytest.fixture(scope="module")
def executed_notebook(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    try:
        yield _execute_generated(
            monkeypatch,
            tmp_path_factory.mktemp("reward-dynamics-matplotlib"),
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close("all")
        monkeypatch.undo()


def test_reward_dynamics_notebook_is_valid_and_exactly_reproducible():
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = _generated_notebook()
    nbformat.validate(committed)
    nbformat.validate(generated)
    assert committed.metadata == generated.metadata
    assert committed.metadata["colab"]["private_outputs"] is True
    assert len(committed.cells) == len(generated.cells)
    for index, (actual, expected) in enumerate(zip(committed.cells, generated.cells)):
        assert actual.cell_type == expected.cell_type
        assert actual.id == expected.id == f"reward-dynamics-{index:03d}"
        assert actual.metadata == expected.metadata
        assert actual.source == expected.source
        if actual.cell_type == "code":
            compile(actual.source, f"reward-dynamics-cell-{index}", "exec")
            assert actual.get("execution_count") is None
            assert not actual.get("outputs")


def test_notebook_contains_only_released_data_workflows_and_precise_sources():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    lowered = source.lower()
    for forbidden in (
        "simulation",
        "synthetic",
        "immutable primary",
        "primary locked",
        "the graph that answers",
        "exact replication of figure 1j",
    ):
        assert forbidden not in lowered

    for exact_source in (
        "https://www.nature.com/articles/s41586-025-09180-y#Sec20",
        "https://www.nature.com/articles/s41586-025-09180-y#Sec24",
        "https://drive.google.com/file/d/1DlmPeyaHn-thn9ILrt-rAXP96-y3IMU7/view#page=5",
        "https://drive.google.com/file/d/1DlmPeyaHn-thn9ILrt-rAXP96-y3IMU7/view#page=6",
        "https://ndownloader.figshare.com/files/54183854",
    ):
        assert exact_source in source

    assert "project-defined choices" in lowered
    assert "new, descriptive project analysis" in lowered
    assert "does not isolate reward" in lowered
    assert "4 passive-reward" in lowered and "9 no-reward" in lowered
    assert "mice—not trials" in lowered


def test_notebook_uses_the_shared_drive_api_without_reimplementing_it():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    assert "data = drive.setup()" in source
    assert "Zhong et al. 2025 - Neuromatch Team Workspace" in source
    assert 'load_mode="plan_only"' in source
    assert "LOAD_ONE_SESSION = False" in source
    for infrastructure in (
        "np.load(",
        "allow_pickle",
        "import csv",
        "import hashlib",
        "requests.get(",
        "os.walk(",
    ):
        assert infrastructure not in source


def test_four_graphs_expose_a_beginner_to_programmer_progression(executed_notebook):
    namespace, elapsed = executed_notebook
    graph_module = namespace["graph"]
    assert elapsed < 35
    assert all(isinstance(namespace[name], graph_module.Graph) for name in GRAPH_NAMES)
    assert all(hasattr(namespace[name], "describe") for name in GRAPH_NAMES)
    assert "sandbox_graph" not in namespace

    minimum_nodes = {
        "fetch_graph": 10,
        "recording_graph": 12,
        "hypothesis_graph": 20,
        "position_graph": 12,
    }
    for name, minimum in minimum_nodes.items():
        description = namespace[name].describe()
        assert len(description["nodes"]) >= minimum
        assert description["nodes"][-1]["outputs"] == ["plots"]


def test_all_default_graphs_run_offline_and_preserve_observation_units(executed_notebook):
    namespace, _ = executed_notebook
    widgets = __import__("ipywidgets")
    assert all(isinstance(namespace[name], widgets.VBox) for name in PANEL_NAMES)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = {name: namespace[name].run() for name in GRAPH_NAMES}
        for result in results.values():
            _assert_valid_plot(result["plots"])
    assert not caught

    fetch = results["fetch_graph"]
    assert len(fetch["selected_sessions"]) == 13
    assert fetch["cohort_audit"]["by_group"]["rewarded"]["selected_mice"] == 4
    assert fetch["cohort_audit"]["by_group"]["unrewarded"]["selected_mice"] == 9

    recording = results["recording_graph"]
    assert recording["neural_qc"]["shape"][0] == 120
    assert recording["heldout_lab"]["valid_folds"] >= 2

    plan = results["hypothesis_graph"]
    assert plan["analysis_spec"]["execution"]["load_mode"] == "plan_only"
    assert plan["analysis_spec"]["execution"]["alternative"] == "two-sided"
    assert plan["analysis_spec"]["complete_cohort"] is False
    assert plan["group_inference"]["available"] is False

    position = results["position_graph"]
    assert position["position_source"]["kind"] == "bundled_demo"
    assert position["mouse_position_changes"]


def test_complete_cohort_mode_is_descriptive_and_records_deviations(executed_notebook):
    namespace, _ = executed_notebook
    result = namespace["hypothesis_graph"].run(
        until="analysis_spec",
        claim_status="complete_cohort",
        cohort_size=0,
        cortical_area="mHV",
    )
    spec = result["analysis_spec"]
    assert spec["complete_cohort"] is True
    assert spec["result_label"] == "COMPLETE COHORT DESCRIPTIVE"
    assert len(spec["sessions"]) == 13
    assert spec["deviations"]["area"] == {"reference": "V1", "selected": "mHV"}
    assert spec["reference_specification"]["area"] == "V1"


def test_real_data_switch_without_drive_is_actionable(executed_notebook):
    namespace, _ = executed_notebook
    with pytest.raises(
        namespace["graph"].NodeError,
        match="Load and analyse requires the mounted shared Drive",
    ):
        namespace["hypothesis_graph"].run(cohort_size=1, load_mode="load_and_analyse")
