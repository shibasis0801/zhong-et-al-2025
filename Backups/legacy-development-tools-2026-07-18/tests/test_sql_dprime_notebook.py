from __future__ import annotations

from pathlib import Path
import runpy

import nbformat


NOTEBOOK = Path("notebooks/03_sql_dataset_selection_for_dprime.ipynb")
GENERATOR = Path("scripts/create_sql_dprime_notebook.py")


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def test_sql_dprime_notebook_is_valid_output_free_and_reproducible():
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


def test_sql_dprime_notebook_replaces_nested_objects_with_queries():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    assert "sql.setup(source=data)" in source
    assert "db.schema()" in source
    assert "FROM experiment_rows" in source
    assert "FROM recordings" in source
    assert "FROM recording_files" in source
    assert "JOIN memberships" in source
    assert 'db.register("train1_labels"' in source
    assert 'db.register("train1_recordings"' in source
    assert "primary_cohort = l.cohort" in source
    assert "data.recordings(" not in source
    assert "data.recording(" not in source
    assert "mice_recordings =" not in source


def test_sql_dprime_notebook_documents_lazy_exact_loading():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    assert 'db.load(\n    recording_id, "behavior"' in source
    assert 'db.load(recording_id, "reduced_neural")' in source
    assert 'db.load(recording_id, "retinotopy")' in source
    assert "do **not** download neural arrays" in source
    assert "Use `full_neural` only after checking" in source
