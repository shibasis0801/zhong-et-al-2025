from __future__ import annotations

from pathlib import Path
import runpy
import time

import nbformat
import numpy as np
import pytest

NOTEBOOK = Path("notebooks/archived/03_dataset_walkthrough_colab.ipynb")
GENERATOR = Path("scripts/create_team_walkthrough_notebook.py")


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def _generated_notebook():
    return runpy.run_path(str(GENERATOR))["build_notebook"]()


def _walk_widgets(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _walk_widgets(child)


def _execute_generated(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.syspath_prepend(str(Path.cwd()))
    namespace = {"__name__": "__walkthrough_smoke__"}
    started = time.monotonic()
    for index, cell in enumerate(_generated_notebook().cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"walkthrough-cell-{index}", "exec"), namespace)
    return namespace, time.monotonic() - started


def test_walkthrough_is_valid_and_exactly_reproducible():
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = _generated_notebook()
    nbformat.validate(committed)
    nbformat.validate(generated)

    assert committed.metadata == generated.metadata
    assert committed.metadata["colab"]["private_outputs"] is True
    assert "widgets" not in committed.metadata
    assert len(committed.cells) == len(generated.cells)
    for index, (actual, expected) in enumerate(zip(committed.cells, generated.cells)):
        assert actual.cell_type == expected.cell_type
        assert actual.id == expected.id == f"walkthrough-{index:03d}"
        assert actual.metadata == expected.metadata
        assert actual.source == expected.source
        if actual.cell_type == "code":
            compile(actual.source, f"walkthrough-cell-{index}", "exec")
            assert actual.get("execution_count") is None
            assert not actual.get("outputs")


def test_walkthrough_uses_only_the_high_level_drive_api():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    assert 'importlib.import_module("drive")' in source
    assert 'sys.modules.pop(name, None)' in source
    assert "data = drive.setup()" in source
    assert "data.figshare(" in source
    assert "data.find(" in source
    assert "data.load(" in source
    assert "Zhong et al. 2025 - Neuromatch Team Workspace" in source
    assert "Runtime → Run all" in source

    for infrastructure in (
        "import csv",
        "csv.DictReader",
        "import hashlib",
        "import shutil",
        "import subprocess",
        "from pathlib import Path",
        "Path.cwd()",
        "metadata/catalog.csv",
        "MAX_STAGE_BYTES",
        "def file_md5",
        "def stage_file",
        "shutil.disk_usage",
        ".partial",
        "np.load(",
        "allow_pickle",
    ):
        assert infrastructure not in source

    lowered = source.lower()
    assert "github" not in lowered
    assert "git clone" not in lowered
    assert source.count('mount("/content/drive"') == 1
    assert "rglob(" not in source
    assert "os.walk(" not in source
    assert "requests.get(" not in source


def test_setup_cell_replaces_a_stale_drive_module(monkeypatch):
    setup_cell = next(
        cell
        for cell in _generated_notebook().cells
        if cell.cell_type == "code" and "data = drive.setup()" in cell.source
    )
    monkeypatch.syspath_prepend(str(Path.cwd()))
    import drive as stale_drive

    monkeypatch.delattr(stale_drive, "setup")
    namespace = {"__name__": "__walkthrough_stale_module_test__"}
    exec(compile(setup_cell.source, "walkthrough-setup", "exec"), namespace)

    assert callable(namespace["drive"].setup)
    assert len(namespace["data"].files) == 297


def test_walkthrough_has_exactly_three_editable_graph_canvases():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    assert 'importlib.import_module("graph")' in source
    assert "import ipywidgets as widgets" in source
    assert source.count("graph.Graph(") == 3
    assert source.count(".widget(") == 3
    assert source.count('show="figure"') == 3
    assert "output.enable_custom_widget_manager()" in source
    assert "# @param" not in source
    assert "plt.show()" not in source
    assert ".diagram(" not in source

    # The controls belong on hollow ports inside the three main canvases.
    assert source.count("widgets.Dropdown(") >= 6
    assert "widgets.IntSlider(" in source
    for description in (
        "File category",
        "Release measure",
        "Availability",
        "Published file",
        "Cortical area",
        "Point size",
        "Stimulus role",
        "Corridor region",
        "Trial range",
        "Published component",
        "Summary",
    ):
        assert f'description="{description}"' in source


def test_walkthrough_remains_descriptive_and_neutral():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    prose = " ".join(source.split())
    for forbidden in (
        "curve_fit(",
        "dprime(",
        "ttest_",
        "polyfit(",
        "LogisticRegression",
        "download_profile(",
    ):
        assert forbidden not in source

    assert "The views are descriptive" in prose
    assert "do not calculate d-prime" in prose
    assert "compare rewarded and unrewarded neural outcomes" in prose
    assert "dataset availability; they are not biological results" in prose
    assert "no fitted trend or reward comparison" in prose
    assert "independent unit of inference" in prose
    assert "PC numbers are session-specific" in prose


def test_walkthrough_builds_three_live_panels_and_smokes_local_flows(monkeypatch):
    namespace, elapsed = _execute_generated(monkeypatch)
    widgets = __import__("ipywidgets")

    graph_module = namespace["graph"]
    flows = {
        id(value): value
        for value in namespace.values()
        if isinstance(value, graph_module.Graph)
    }
    panels = {
        id(value): value
        for value in namespace.values()
        if isinstance(value, widgets.VBox)
        and hasattr(value, "last_run")
        and hasattr(value, "last_error")
    }
    assert elapsed < 30
    assert len(flows) == 3
    assert len(panels) == 3
    assert len(namespace["data"].files) == 297
    assert namespace["figshare_api"]["id"] == 28811129

    for panel in panels.values():
        descendants = list(_walk_widgets(panel))
        diagrams = [
            item
            for item in descendants
            if isinstance(item, widgets.HTML) and "<svg" in item.value
        ]
        controls = [
            item
            for item in descendants
            if getattr(getattr(item, "layout", None), "grid_area", None) == "canvas"
        ]
        buttons = [
            item
            for item in descendants
            if isinstance(item, widgets.Button) and item.description == "Run flow"
        ]
        assert len(diagrams) == 1
        assert controls
        assert len(buttons) == 1

    # The release and compact-session paths are fully local. The cortical-map
    # path intentionally waits for a selected file in the mounted Drive release.
    runnable = [
        (name, value)
        for name, value in namespace.items()
        if name in {"release_panel", "session_panel"}
    ]
    assert len(runnable) == 2
    for _, panel in runnable:
        button = next(
            item
            for item in _walk_widgets(panel)
            if isinstance(item, widgets.Button) and item.description == "Run flow"
        )
        button.click()
        assert panel.last_error is None
        assert panel.last_run is not None
        assert panel.run_count == 1

    # A changed hollow port must reach the Python run and change the result.
    release_panel = namespace["release_panel"]
    release_widgets = list(_walk_widgets(release_panel))
    release_controls = {
        item.description: item
        for item in release_widgets
        if isinstance(item, widgets.ValueWidget) and getattr(item, "description", "")
    }
    first_release = release_panel.last_run
    release_controls["File category"].value = "retinotopy"
    release_controls["Release measure"].value = "count"
    release_controls["Availability"].value = "mice"
    release_button = next(
        item for item in release_widgets
        if isinstance(item, widgets.Button) and item.description == "Run flow"
    )
    release_button.click()
    second_release = release_panel.last_run
    assert second_release.settings["category"] == "retinotopy"
    assert second_release.settings["measure"] == "count"
    assert second_release.settings["availability"] == "mice"
    assert second_release["summary"]["matched"] < first_release["summary"]["matched"]
    assert second_release["summary"]["unit"] == "files"
    assert second_release["summary"]["availability"] == "mice"
    assert release_panel.run_count == 2

    session_panel = namespace["session_panel"]
    session_widgets = list(_walk_widgets(session_panel))
    session_controls = {
        item.description: item
        for item in session_widgets
        if isinstance(item, widgets.ValueWidget) and getattr(item, "description", "")
    }
    first_session = session_panel.last_run
    session_controls["Stimulus role"].value = 1
    session_controls["Corridor region"].value = "texture"
    session_controls["Trial range"].value = (0, 100)
    session_controls["Published component"].value = 1
    session_controls["Summary"].value = "median"
    session_button = next(
        item for item in session_widgets
        if isinstance(item, widgets.Button) and item.description == "Run flow"
    )
    session_button.click()
    second_session = session_panel.last_run
    assert second_session.settings["stimulus_id"] == 1
    assert second_session.settings["corridor"] == "texture"
    assert second_session.settings["trial_range"] == (0, 100)
    assert second_session.settings["pc_index"] == 1
    assert second_session.settings["statistic"] == "median"
    assert len(second_session["summary"]["trial_id"]) < len(first_session["summary"]["trial_id"])
    assert len(second_session["summary"]["position"]) < len(first_session["summary"]["position"])
    assert session_panel.run_count == 2

    import matplotlib.pyplot as plt

    assert not plt.get_fignums()


def test_each_release_control_changes_its_intended_data_or_visible_labels(monkeypatch):
    namespace, _ = _execute_generated(monkeypatch)
    release = namespace["release_graph"]

    storage = release.run(measure="storage")
    count = release.run(measure="count")
    retinotopy = release.run(category="retinotopy")
    search = release.run(filename_contains="TX119")
    mice = release.run(availability="mice")

    assert storage["summary"]["measure"] == "storage"
    assert storage["summary"]["unit"] == "GiB"
    assert count["summary"]["measure"] == "count"
    assert count["summary"]["unit"] == "files"
    assert storage["summary"]["values"] != count["summary"]["values"]

    storage_axis = storage["figure"].axes[0]
    count_axis = count["figure"].axes[0]
    assert storage_axis.get_xscale() == "log"
    assert count_axis.get_xscale() == "linear"
    assert storage_axis.get_title() == "Storage by matching file category"
    assert count_axis.get_title() == "File count by matching file category"
    assert set(storage["summary"]["value_labels"]) <= {
        text.get_text() for text in storage_axis.texts
    }
    assert set(count["summary"]["value_labels"]) <= {
        text.get_text() for text in count_axis.texts
    }
    assert not storage_axis.patches
    assert len(count_axis.patches) == len(count["summary"]["categories"])
    assert storage["figure"].axes[1].get_title().startswith(
        "Whole-release availability"
    )
    assert retinotopy["summary"]["categories"] == ["retinotopy"]
    assert retinotopy["summary"]["matched"] == 89
    assert 0 < search["summary"]["matched"] < storage["summary"]["matched"]
    assert mice["summary"]["counts"] != storage["summary"]["counts"]
    assert [bar.get_height() for bar in mice["figure"].axes[1].patches] == mice[
        "summary"
    ]["counts"]

    with pytest.raises(namespace["graph"].NodeError, match="storage.*count"):
        release.run(measure="unsupported")
    with pytest.raises(namespace["graph"].NodeError, match="recordings.*mice"):
        release.run(availability="unsupported")


def test_cortical_controls_change_selection_without_rescaling_the_map(monkeypatch):
    namespace, _ = _execute_generated(monkeypatch)
    xy = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 1.5], [4.0, 1.0], [5.0, 2.0]]
    )
    area_ids = np.array([8, 8, 0, 5, 3, 99])
    retinotopy = {"file_name": "synthetic_trans.npz", "xy_t": xy, "iarea": area_ids}

    quality = namespace["check_retinotopy"](retinotopy)
    all_summary = namespace["summarize_retinotopy"](
        retinotopy, quality, namespace["select_area"](retinotopy, "all")
    )
    v1_summary = namespace["summarize_retinotopy"](
        retinotopy, quality, namespace["select_area"](retinotopy, "V1")
    )
    all_figure = namespace["plot_retinotopy"](all_summary, point_size=2)
    v1_figure = namespace["plot_retinotopy"](v1_summary, point_size=6)

    assert all_summary["selected_points"] == 6
    assert v1_summary["selected_points"] == 2
    assert all_figure.axes[0].get_xlim() == v1_figure.axes[0].get_xlim()
    assert all_figure.axes[0].get_ylim() == v1_figure.axes[0].get_ylim()
    assert "V1 · 2/6 points" in v1_figure.axes[0].get_title()
    np.testing.assert_array_equal(v1_figure.axes[0].collections[0].get_sizes(), [6])

    monkeypatch.setattr(
        namespace["data"],
        "load",
        lambda filename: {"xy_t": xy, "iarea": area_ids},
    )
    loaded = namespace["load_retinotopy"]("another_trans.npz")
    assert loaded["file_name"] == "another_trans.npz"
    with pytest.raises(ValueError, match="positive"):
        namespace["plot_retinotopy"](v1_summary, point_size=0)


def test_each_session_control_changes_its_intended_summary(monkeypatch):
    namespace, _ = _execute_generated(monkeypatch)
    session = namespace["session_graph"]
    baseline = session.run()
    stimulus = session.run(stimulus_id=1)
    corridor = session.run(corridor="texture")
    trial_range = session.run(trial_range=(0, 100))
    component = session.run(pc_index=1)
    median = session.run(statistic="median")

    assert len(stimulus["summary"]["trial_id"]) < len(
        baseline["summary"]["trial_id"]
    )
    assert len(corridor["summary"]["position"]) < len(
        baseline["summary"]["position"]
    )
    assert len(trial_range["summary"]["trial_id"]) == 100
    assert not np.array_equal(
        component["summary"]["heatmap"], baseline["summary"]["heatmap"]
    )
    assert not np.allclose(
        median["summary"]["neural"], baseline["summary"]["neural"]
    )
    assert "canonical role 1" in stimulus["figure"]._suptitle.get_text()
    assert "trials 0–100" in trial_range["figure"]._suptitle.get_text()

    with pytest.raises(namespace["graph"].NodeError, match="mean.*median"):
        session.run(statistic="unsupported")
    with pytest.raises(namespace["graph"].NodeError, match="full.*texture.*gray"):
        session.run(corridor="unsupported")
    with pytest.raises(namespace["graph"].NodeError, match="Published component"):
        session.run(pc_index=999)
