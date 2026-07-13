from pathlib import Path
import subprocess
import sys
import time

import nbformat

import graph


NOTEBOOK = Path("notebooks/zhong2025_graph_experiments_colab.ipynb")


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


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


def test_setup_detects_a_fresh_colab_runtime_before_it_is_imported(
    tmp_path, monkeypatch
):
    package = tmp_path / "google" / "colab"
    package.mkdir(parents=True)
    (tmp_path / "google" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "output.py").write_text(
        "def enable_custom_widget_manager():\n    pass\n"
    )
    for module_name in ("google.colab.output", "google.colab", "google"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    installs = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, check: installs.append((command, check)),
    )
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    setup = next(cell for cell in notebook.cells if cell.id == "graph-002")
    namespace = {}
    exec(compile(setup.source, "graph-colab-setup", "exec"), namespace)

    assert namespace["IN_COLAB"] is True
    assert len(installs) == 1
    assert installs[0][1] is True
    assert "git+https://github.com/shibasis0801/zhong-et-al-2025.git@main" in installs[0][0]


def test_graph_notebook_teaches_the_complete_small_surface():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = _source(notebook)
    for required in (
        "The four-word mental model",
        "The sample experiment",
        "Build and inspect the flow",
        "Run once",
        "Change settings without changing the flow",
        "Run explicit variations",
        "Inspect a run instead of trusting hidden state",
        "Write another sequential flow",
        "Boundaries of this example",
        "@graph.node(outputs=\"demo\")",
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
        "select_trials",
        "summarize_position_profiles",
        "plot_position_profiles",
    )
    panel = namespace["run_panel"]
    assert panel.children[2].description == "Run flow"
    assert len(panel.children[1].children) == 4


def test_wheel_configuration_includes_the_single_graph_module():
    pyproject = Path("pyproject.toml").read_text()
    assert 'py-modules = ["graph"]' in pyproject
    assert '"ipywidgets>=8.1,<9"' in pyproject
