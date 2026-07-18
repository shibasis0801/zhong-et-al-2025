from __future__ import annotations

from pathlib import Path
import runpy

import nbformat


NOTEBOOK = Path("notebooks/archived/10_dprime_pandas_colab.ipynb")
GENERATOR = Path("scripts/create_pandas_dprime_notebook.py")


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def test_pandas_dprime_notebook_is_valid_output_free_and_reproducible():
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = runpy.run_path(str(GENERATOR))["build_notebook"]()
    nbformat.validate(committed)
    nbformat.validate(generated)
    assert committed.metadata == generated.metadata
    assert len(committed.cells) == len(generated.cells)
    for actual, expected in zip(committed.cells, generated.cells):
        assert actual.id == expected.id
        assert actual.source == expected.source
        if actual.cell_type == "code":
            compile(actual.source, actual.id, "exec")
            assert actual.execution_count is None
            assert actual.outputs == []


def test_pandas_notebook_matches_every_meaningful_main_dprime_cell():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = _source(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == 27
    for index, cell in enumerate(code_cells):
        assert f"Main dprime cell {index}" in cell.source

    assert "import pandas as pd" in source
    assert "import duckdb" not in source
    assert "db.query(" not in source
    assert "sql.setup(" not in source
    assert ".merge(" in source
    assert ".groupby(" in source
    assert ".sort_values(" in source
    assert "experiment_catalog" in source
    assert "experiment_rows" in source
    assert "recording_files" in source
    assert "mice_recordings" in source
    assert "train1_recordings" in source
    assert "file_map" in source
    assert "RecordingSelector" in source
    assert "sup_train1.value, sup_train1.selected" in source
    assert '"train1_recordings": 38' in source


def test_pandas_notebook_documents_lazy_exact_loading():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    assert "does not\ndownload neural arrays" in source
    assert 'recording=choice["recording_id"]' in source
    assert 'experiment=choice["experiment"]' in source
    assert 'data.load(recording=choice["recording_id"], layer="reduced_neural")' in source
    assert "before loading `full_neural` files" in source
