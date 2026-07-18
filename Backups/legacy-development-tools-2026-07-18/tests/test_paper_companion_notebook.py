from __future__ import annotations

import base64
from pathlib import Path
import runpy

import nbformat


NOTEBOOK = Path("notebooks/archived/04_paper_companion_colab.ipynb")
GENERATOR = Path("scripts/create_paper_companion_notebook.py")


def _generated_notebook():
    return runpy.run_path(str(GENERATOR))["build_notebook"]()


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def _execute(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/zhong-matplotlib")
    monkeypatch.syspath_prepend(str(Path.cwd()))
    namespace = {"__name__": "__paper_companion_test__"}
    for index, cell in enumerate(_generated_notebook().cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"paper-companion-cell-{index}", "exec"), namespace)
    return namespace


def test_paper_companion_is_valid_and_reproducible():
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
        assert actual.id == expected.id == f"paper-companion-{index:03d}"
        assert actual.source == expected.source
        assert actual.metadata == expected.metadata
        assert actual.get("attachments", {}) == expected.get("attachments", {})
        if actual.cell_type == "code":
            compile(actual.source, f"paper-companion-cell-{index}", "exec")
            assert actual.get("execution_count") is None
            assert not actual.get("outputs")


def test_complete_published_figures_are_embedded_byte_for_byte_with_exact_links():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    figure_cells = [cell for cell in notebook.cells if cell.get("attachments")]
    assert len(figure_cells) == 5

    for number, cell in enumerate(figure_cells, start=1):
        filename = f"nature-main-{number}.png"
        assert set(cell.attachments) == {filename}
        embedded = base64.b64decode(cell.attachments[filename]["image/png"])
        original = Path("zhong2025/assets/reference_figures", filename).read_bytes()
        assert embedded == original
        assert len(embedded) > 100_000
        assert f"https://www.nature.com/articles/s41586-025-09180-y#Fig{number}" in cell.source
        assert "complete published figure" in cell.source.lower()

    source = _source(notebook)
    assert "# A schematic, not measured data" not in source
    assert "ax.barh(0, 4" not in source


def test_paper_companion_covers_the_full_study_without_infrastructure_bloat():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    compact = " ".join(source.split())

    for fact in (
        "19 GCaMP6s mice",
        "89 physical recordings",
        "20,547–89,577",
        "23 different, non-imaged mice",
        "4 m textured corridor",
        "2 m of grey",
        "60 cm/s",
        "6 cm/s",
        "0.75 s decay",
        "|d'|\\geq0.3",
        "one passive-reward day and four active-reward days",
        "does not establish longitudinal registration",
    ):
        assert fact in compact

    for step in (
        "Fig. 1 — Train 1",
        "Fig. 2 — Test 1",
        "Fig. 3 — Train 2",
        "Extended Data 6 — Test 2",
        "Extended Data 6–7 — Test 3",
        "Fig. 4 — Reward signal",
        "Fig. 5 — Behavioural validation",
    ):
        assert step in source

    assert "data = drive.setup()" in source
    assert "data.figshare(" in source
    assert ".recordings(experiment=" in source
    assert ".file(" in source
    assert "03_dataset_walkthrough_colab.ipynb" in source
    assert "00_use_janelia_drive_colab.ipynb" in source
    assert "01_understand_the_dataset_colab.ipynb" not in source
    assert "02_graph_experiments_colab.ipynb" not in source

    lowered = source.lower()
    assert "github" not in lowered
    for infrastructure in (
        "import csv",
        "import hashlib",
        "import shutil",
        "from pathlib import Path",
        "requests.get(",
        "np.load(",
        "allow_pickle",
        "metadata/catalog.csv",
    ):
        assert infrastructure not in source


def test_paper_companion_has_two_linear_editable_graphs():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    assert source.count("graph.Graph(") == 2
    assert source.count(".widget(") == 2
    assert source.count('show="figure"') == 2
    assert source.count("widgets.Dropdown(") == 3
    assert 'description="Paper step"' in source
    assert 'description="Experiment"' in source
    assert 'description="Data layer"' in source
    assert ".diagram(" not in source

    # Both canvases are intentionally linear so wires do not cross nodes.
    assert "map_released_evidence(chapter)" in source
    assert "resolve_released_files(selection)" in source


def test_all_paper_steps_run_and_map_to_deduplicated_release_evidence(monkeypatch):
    namespace = _execute(monkeypatch)
    paper_graph = namespace["paper_graph"]

    expected = {
        "fig1": [4, 4, 9, 9, 5, 5],
        "fig2": [5, 7, 11],
        "fig3": [5, 5, 6, 6],
        "test2": [5, 6, 7, 5],
        "test3": [5, 4, 5],
        "fig4": [4, 4, 9, 9],
        "fig5": [11, 7, 5],
    }
    for key, counts in expected.items():
        result = paper_graph.run(paper_step=key)
        assert result["evidence"]["counts"] == counts
        assert result["chapter"]["key"] == key
        assert result["figure"].axes

    test3 = paper_graph.run(paper_step="test3")["evidence"]
    assert test3["counts"] == [5, 4, 5]
    assert test3["count_unit"] == "unique physical recordings"


def test_release_graph_changes_with_experiment_and_layer(monkeypatch):
    namespace = _execute(monkeypatch)
    release_graph = namespace["release_graph"]

    behavior = release_graph.run(
        experiment_label="sup_train1_before_learning",
        data_layer="behavior",
    )["summary"]
    reduced = release_graph.run(
        experiment_label="sup_train1_before_learning",
        data_layer="reduced_neural",
    )["summary"]
    other_experiment = release_graph.run(
        experiment_label="unsup_test3",
        data_layer="reduced_neural",
    )["summary"]

    assert len(behavior["recordings"]) == 4
    assert len(behavior["files"]) == 1
    assert behavior["files"][0].name == "Beh_sup_train1_before_learning.npy"
    assert len(reduced["recordings"]) == 4
    assert len(reduced["files"]) == 4
    assert all(item.category == "reduced_neural" for item in reduced["files"])
    assert len(other_experiment["recordings"]) == 4
    assert len(other_experiment["files"]) == 4
    assert {item.name for item in reduced["files"]} != {
        item.name for item in other_experiment["files"]
    }


def test_companion_never_loads_the_large_release_during_run_all():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    assert "data.load(" not in source
    assert ".load(\"full_neural\")" not in source
    assert ".fetch(" not in source
    assert "metadata only" in source
