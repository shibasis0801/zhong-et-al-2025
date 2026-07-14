from pathlib import Path
import time

import nbformat

import graph


NOTEBOOK = Path("notebooks/zhong2025_graph_experiments_colab.ipynb")


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def _walk_widgets(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _walk_widgets(child)


def _execute_default_path(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "__graph_notebook_smoke__"}
    started = time.monotonic()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"graph-cell-{index}", "exec"), namespace)
    return namespace, time.monotonic() - started


def test_graph_notebook_is_clean_valid_and_compilable():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    ids = [cell["id"] for cell in notebook.cells]
    assert ids == [f"graph-{index:03d}" for index in range(len(ids))]
    assert len(ids) == len(set(ids))
    for index, cell in enumerate(notebook.cells):
        assert cell.get("execution_count") is None
        assert not cell.get("outputs")
        if cell.cell_type == "code":
            compile(cell.source, f"graph-cell-{index}", "exec")


def test_graph_notebook_stays_small_neutral_and_offline_for_data():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = _source(notebook)
    normalized = " ".join(source.split())
    for forbidden in (
        "allow_pickle=True",
        "download_profile(",
        "requests.get(",
        "LogisticRegression",
        "Ridge(",
        ".fit(",
        ".predict(",
        "p-value",
        "networkx",
        "cytoscape",
        "reactflow",
    ):
        assert forbidden not in source
    assert "Dataset downloads: 0" in source
    assert "descriptive" in source.lower()
    assert "moving frames only" in source.lower()
    assert "contributing moving frames" in source.lower()
    assert "not a statistical comparison or a biological conclusion" in normalized
    assert ("rea" + "ktor") not in source.lower()


def test_graph_colab_setup_uses_the_shared_drive_only():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    setup = next(cell for cell in notebook.cells if cell.id == "graph-002")
    assert "drive.mount('/content/drive', force_remount=False)" in setup.source
    assert "Zhong et al. 2025 - Neuromatch Team Workspace" in setup.source
    assert "Zhong2025_Janelia_v2" in setup.source
    assert "team_tools/packages" in setup.source
    assert "module_name == 'graph'" in setup.source
    assert "module_name == 'zhong2025'" in setup.source
    assert "importlib.invalidate_caches()" in setup.source
    source = _source(notebook).lower()
    assert "github" not in source
    assert "git clone" not in source
    assert "git+https" not in source


def test_graph_notebook_teaches_the_complete_small_surface():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = _source(notebook)
    for required in (
        "The four-word mental model",
        "The sample experiment",
        "Build the flow",
        "Configure and run",
        "Optional: run the same flow in Python",
        "Run explicit variations",
        "Inspect a run instead of trusting hidden state",
        "Write another sequential flow",
        "Boundaries of this example",
        "Hollow input",
        "Stimulus role",
        "Published component",
        "@graph.node(outputs=\"demo\")",
        "@graph.node(outputs=\"quality\")",
        "independent branches",
        "run_panel.last_run",
        "train/test split",
        "read-only",
        "Role 0 — familiar B exemplar",
        "#@title Configure the visible ports",
        "across-trial {summary['statistic']}",
        "experiment = graph.Graph(",
        "experiment.widget(",
        "experiment.run_many(",
    ):
        assert required in source


def test_reusable_markdown_example_runs_as_written():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = _source(notebook)
    section = source.split("## Write another sequential flow", 1)[1]
    example = section.split("```python\n", 1)[1].split("\n```", 1)[0]
    namespace = {"graph": graph}

    exec(compile(example, "graph-markdown-example", "exec"), namespace)

    assert namespace["my_flow"].run().order == ("load", "clean", "summarize")


def test_graph_notebook_executes_quickly_with_real_compact_data(monkeypatch):
    namespace, elapsed = _execute_default_path(monkeypatch)

    assert elapsed < 30
    assert namespace["baseline"]["summary"]["trial_count"] == 452
    assert namespace["baseline"]["demo"]["population_features"].shape == (452, 18, 48)
    assert [row["trials"] for row in namespace["variation_table"]] == [
        110,
        116,
        116,
        110,
    ]
    assert {row["position_bins"] for row in namespace["variation_table"]} == {18}
    assert namespace["baseline"].order == (
        "load_compact_recording",
        "check_recording",
        "select_trials",
        "summarize_position_profiles",
    )
    assert namespace["baseline"]["quality"]["zero_frame_trial_bins"] == 0
    assert namespace["baseline"]["quality"]["finite_population_fraction"] == 1.0
    assert namespace["baseline"]["summary"]["minimum_valid_trials_per_bin"] == 452
    panel = namespace["run_panel"]
    toolbar, surface, _ = panel.children
    assert toolbar.children[0].description == "Run flow"
    scroller = surface.children[1]
    canvas = scroller.children[0]
    diagram = canvas.children[0]
    controls = canvas.children[1:]
    assert "<svg" in diagram.value
    assert "data-source='load_compact_recording.demo'" in diagram.value
    assert len(controls) == 4
    assert all(control.layout.grid_area == "canvas" for control in controls)
    assert "Input controls and port details" not in diagram.value
    descriptions = {
        widget.description
        for widget in _walk_widgets(surface)
        if hasattr(widget, "description") and widget.description
    }
    assert descriptions == {
        "Stimulus role",
        "Corridor region",
        "Published component",
        "Summary",
    }
    assert "data-source='check_recording.quality'" in diagram.value
    assert "data-source='select_trials.selection'" in diagram.value
    assert "data-source='summarize_position_profiles.summary'" in diagram.value

    median = namespace["experiment"].run(statistic="median")
    assert "across-trial median" in median["figure"].axes[0].get_ylabel()
    assert median["summary"]["zero_frame_trial_bins"] == 0
    namespace["plt"].close(median["figure"])


def test_wheel_configuration_includes_the_single_graph_module():
    pyproject = Path("pyproject.toml").read_text()
    assert 'py-modules = ["graph"]' in pyproject
    assert '"ipywidgets>=8.1,<9"' in pyproject
