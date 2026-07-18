from pathlib import Path
import runpy

import nbformat


NOTEBOOK = Path("notebooks/archived/12_complete_dataset_inspector_colab.ipynb")
GENERATOR = Path("scripts/create_dataset_inspector_notebook.py")


def source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def test_dataset_inspector_notebook_is_valid_clean_and_reproducible():
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = runpy.run_path(str(GENERATOR))["build_notebook"]()
    nbformat.validate(committed)
    nbformat.validate(generated)
    assert committed.metadata == generated.metadata
    assert len(committed.cells) == len(generated.cells)
    for index, (actual, expected) in enumerate(zip(committed.cells, generated.cells)):
        assert actual.id == expected.id == f"dataset-inspector-{index:03d}"
        assert actual.source == expected.source
        if actual.cell_type == "code":
            compile(actual.source, actual.id, "exec")
            assert actual.execution_count is None
            assert actual.outputs == []


def test_notebook_surfaces_the_complete_release_and_preserves_join_grains():
    text = source(nbformat.read(NOTEBOOK, as_version=4))
    prose = " ".join(text.replace("**", "").split())
    for fact in (
        "297 file rows",
        "19 imaging mice",
        "89 physical acquisitions",
        "23 imaging experiment labels",
        "142 source experiment rows",
        "133 unique experiment–acquisition",
        "23 behavior-only mice",
    ):
        assert fact in prose
    for key in (
        "recording_id = mouse_date_block",
        "retinotopy_id = mouse_date",
        "behavior_key",
        "frame_id",
        "neuron_id",
        "trial_id",
    ):
        assert key in text
    assert 'db.register("behavior_instances", behavior_instances)' in text
    assert 'db.register("acquisition_manifest", acquisition_manifest)' in text
    assert "len(acquisition_manifest) == 142" in text
    assert "len(files) == 297" in text
    assert 'types.ModuleType("zhong2025")' in text
    assert 'importlib.import_module("zhong2025.atlas")' in text


def test_notebook_keeps_arrays_lazy_bounded_and_axis_checked():
    text = source(nbformat.read(NOTEBOOK, as_version=4))
    assert "class DatasetInspector" in text
    assert "class JoinedRecording" in text
    assert "U and V do not share one component axis" in text
    assert "U, iarea, and xy_t do not share one neuron axis" in text
    assert "MAX_TRAILING_BEHAVIOR_FRAMES = 3" in text
    assert "MAX_ACTIVITY_VALUES = 2_000_000" in text
    assert "self.U[:, neurons].T @ self.V[:, frames]" in text
    assert "Requested {values:,} values" in text
    assert "include_full=False" in text
    assert "LOAD_EXACT_FULL_NEURAL = False" in text
    assert "np.concatenate(arrays, axis=0)" in text
    assert "billions of rows" in text


def test_notebook_includes_global_browser_joined_tables_and_query_gallery():
    text = source(nbformat.read(NOTEBOOK, as_version=4))
    assert "Interactive metadata browser" in text
    assert "cohort_control = widgets.Dropdown" in text
    assert "inspector.download_plan" in text
    assert 'db.register("frames_current", frames)' not in text
    assert '"frames_current": frames' in text
    assert '"trials_current": trials' in text
    assert '"neurons_current": neurons' in text
    assert 'db.register("activity_current", activity_long)' in text
    assert "all_modalities = db.query" in text
    assert "Practical query gallery" in text
