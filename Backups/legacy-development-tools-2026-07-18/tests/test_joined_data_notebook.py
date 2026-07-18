from __future__ import annotations

from pathlib import Path
import runpy

import nbformat


NOTEBOOK = Path("notebooks/archived/11_join_neural_behavior_retinotopy_colab.ipynb")
GENERATOR = Path("scripts/create_joined_data_notebook.py")


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def test_joined_data_notebook_is_valid_output_free_and_reproducible():
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


def test_notebook_uses_drive_only_as_a_filesystem():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    assert "from google.colab import drive as google_drive" in source
    assert 'google_drive.mount("/content/drive"' in source
    assert 'DATA_ROOT / "metadata/catalog.csv"' in source
    assert "np.load(index_path, allow_pickle=True)" in source
    assert 'release_path("behavior_path")' in source
    assert 'release_path("svd_path")' in source
    assert 'release_path("retinotopy_path")' in source
    for forbidden in (
        "import drive\n", "import sql\n", "drive.setup(", "sql.setup(",
        "data.recording(", "data.recordings(", "Dataset(", "Recording(",
        "DataFile(", ".load(\"behavior\"", ".load(\"retinotopy\"",
    ):
        assert forbidden not in source


def test_notebook_preserves_exact_file_and_axis_join_keys():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    assert 'f"{recording_id}_{stimulus_type}"' in source
    assert 'len(experiment_rows) == 142' in source
    assert 'drop_duplicates(["experiment", "recording_id"])' in source
    assert 'drop_duplicates(["experiment", "behavior_key"])' in source
    assert "b.experiment = e.experiment" in source
    assert "s.recording_id = e.recording_id" in source
    assert "n.recording_id = e.recording_id" in source
    assert "r.retinotopy_id = e.retinotopy_id" in source
    assert "f.frame_id = a.frame_id" in source
    assert "n.neuron_id = a.neuron_id" in source
    assert "U.shape[1] != len(iarea)" in source
    assert "behavior_frames - neural_frames" in source
    assert "MAX_TRAILING_BEHAVIOR_FRAMES = 3" in source


def test_notebook_uses_sql_for_tables_and_numpy_for_dense_neural_data():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    assert 'duckdb.connect(database=":memory:")' in source
    assert 'db.register("catalog", catalog)' in source
    assert 'db.register("experiment_rows", experiment_rows)' in source
    assert "frame_observations = db.execute" in source
    assert "all_modalities = db.execute" in source
    assert "small_activity = U[:, v1_ids].T @ V[:, labeled_frames]" in source
    assert "billions of rows" in source
    assert "component_sums" in source
    assert "area_weights @ area_weights.T" in source


def test_notebook_defines_honest_trial_indexed_dprime():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    assert "A single trial contains one stimulus role" in source
    assert "held-out neural evidence" in source
    assert "evidence, not d′" in source
    assert "one d-prime per multi-trial segment" in source
    assert "WINDOW_TRIALS = 40" in source
    assert "heldout_window" in source
    assert "heldout_trial_evidence" in source
    assert '"trial_id": int(physical_trials[local_index])' in source
    assert '"fold_id": int(fold_id)' in source
    assert '"is_held_out": True' in source
    assert "fold_dprimes.append(score_dprime(test_scores, test_labels))" in source
    assert "Raw scores from separately fitted folds are never pooled" in source
    assert "ddof=1" in source
    assert "ddof=0" in source
    assert "stride=1" in source
    assert "stride=WINDOW_TRIALS" in source
    assert "correlated" in source
    assert "whole-session, per-neuron, frame-pooled formula" in source
    assert "ALL JOIN CHECKS PASSED" in source


def test_notebook_aligns_neural_frames_trials_and_analysis_support():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    assert 'db.register("trial_segments", trial_segments)' in source
    assert "At least one trial is split into non-contiguous frame runs" in source
    assert "timeline_frame_ids = np.arange" in source
    assert "U[:, timeline_neuron_ids].T @ V[:, timeline_frame_ids]" in source
    assert "timeline_activity_z" in source
    assert 'frame_observations["in_texture"].to_numpy(dtype=bool)' in source
    assert "timeline_analysis_valid" in source
    assert "role 0 = circle1" in source
    assert "role 2 = leaf1" in source
    assert "vertical lines are physical trial boundaries" in source
    assert "one dot = one held-out trial; this is evidence, not d′" in source
    assert 'db.register("trial_results", trial_results)' in source
    assert 'db.register("dprime_segments", nonoverlapping)' in source
    assert 'db.register("dprime_segment_folds", nonoverlapping_folds)' in source
    assert 'segment["start_trial"], segment["stop_trial"]' in source
