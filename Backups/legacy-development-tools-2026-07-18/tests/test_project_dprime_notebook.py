from __future__ import annotations

from pathlib import Path
import runpy

import nbformat
import numpy as np


NOTEBOOK = Path("notebooks/archived/06_within_session_dprime_colab.ipynb")
GENERATOR = Path("scripts/create_project_dprime_notebook.py")


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def _generated_notebook():
    return runpy.run_path(str(GENERATOR))["build_notebook"]()


def _execute_generated(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.syspath_prepend(str(Path.cwd()))
    namespace = {"__name__": "__project_smoke__"}
    for index, cell in enumerate(_generated_notebook().cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"project-cell-{index}", "exec"), namespace)
    return namespace


def _prepared_release_shape():
    """Small deterministic shape fixture for the released-data mechanics."""
    n_trials, n_bins, n_features = 80, 4, 3
    trial_id = np.arange(n_trials)
    labels = np.tile([2, 0], n_trials // 2)
    role_sign = np.where(labels == 2, 1.0, -1.0)
    features = np.empty((n_trials, n_bins, n_features), dtype=np.float32)
    for trial in range(n_trials):
        for position in range(n_bins):
            features[trial, position, 0] = role_sign[trial] * (0.5 + trial / 400)
            features[trial, position, 1] = (trial % 7) / 20 + position / 50
            features[trial, position, 2] = role_sign[trial] * 0.1 + (trial % 5) / 25
    return {
        "trial_features": features,
        "labels": labels,
        "trial_id": trial_id,
        "texture_mask": np.array([True, True, False, False]),
        "recording_id": "fixture_recording",
        "mouse_id": "fixture_mouse",
        "area": "mHV",
        "movement_rule": "moving_only",
        "n_features": n_features,
        "position_edges_m": np.linspace(0, 6, n_bins + 1),
    }


def test_project_notebook_is_valid_output_free_and_reproducible():
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = _generated_notebook()
    nbformat.validate(committed)
    nbformat.validate(generated)
    assert committed.metadata == generated.metadata
    assert committed.metadata["colab"]["private_outputs"] is True
    assert len(committed.cells) == len(generated.cells)
    for index, (actual, expected) in enumerate(zip(committed.cells, generated.cells)):
        assert actual.cell_type == expected.cell_type
        assert actual.id == expected.id == f"project-{index:03d}"
        assert actual.source == expected.source
        if actual.cell_type == "code":
            compile(actual.source, f"project-cell-{index}", "exec")
            assert actual.execution_count is None
            assert actual.outputs == []


def test_project_notebook_has_exact_sources_and_an_honest_scope_contract():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    normalized = " ".join(source.split())
    for anchor in ("#Fig1", "#Sec2", "#Sec19", "#Sec20", "#Sec24", "#Sec25"):
        assert f"https://www.nature.com/articles/s41586-025-09180-y{anchor}" in source
    assert "science.adp7429.pdf#page=5" in source
    assert "1DlmPeyaHn-thn9ILrt-rAXP96-y3IMU7/view#page=5" in source
    assert "https://doi.org/10.6084/m9.figshare.28811129.v2" in source
    assert "not a reproduction or reparametrization of a Nature figure" in normalized
    assert "project-authored analysis of released data" in normalized
    assert "recording cannot establish" in normalized
    assert "label-held-out but not a prospective online decoder" in normalized

    lowered = source.lower()
    for forbidden in (
        "exactly the fig",
        "figure 1j reparametrized",
        "reward accelerates",
        "same d′ formula",
        "simulate_mouse",
        "np.random",
        "data:image/",
        "<svg",
    ):
        assert forbidden not in lowered


def test_project_notebook_uses_verified_release_layers_and_nonoverlap():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    assert 'importlib.import_module("drive")' in source
    assert "data = drive.setup(report=False)" in source
    assert "data.recording(recording_id)" in source
    assert "data.recordings(experiment=experiment)" in source
    assert 'session.load("behavior", experiment=experiment)' in source
    assert 'session.load("reduced_neural")' in source
    assert 'session.load("retinotopy")' in source
    assert "prepare_session_trials(" in source
    assert "blockwise_dprime(" in source
    assert "stride_trials=block_trials" in source
    assert "require_complete_position_coverage=True" in source
    assert 'movement_rule="moving_only"' in source
    assert "plot_beginner" in source
    assert "plot_programmer_diagnostics" in source
    assert "U.T @ V" not in source
    assert "graph.Graph(" not in source


def test_analysis_uses_two_nonoverlapping_held_out_blocks(monkeypatch):
    namespace = _execute_generated(monkeypatch)
    result = namespace["analyse_prepared"](
        _prepared_release_shape(), block_trials=40, n_folds=4
    )
    curve = result["curve"]
    np.testing.assert_array_equal(curve["start_trial"], [0, 40])
    np.testing.assert_array_equal(curve["stop_trial"], [39, 79])
    assert result["block_trials"] == result["stride_trials"] == 40
    assert result["n_folds"] == 4
    assert np.isfinite(curve["dprime"]).all()
    np.testing.assert_array_equal(curve["valid_folds"], [4, 4])
    np.testing.assert_array_equal(curve["required_folds"], [4, 4])


def test_simple_and_programmer_figures_only_display_computed_arrays(monkeypatch):
    namespace = _execute_generated(monkeypatch)
    result = namespace["analyse_prepared"](_prepared_release_shape())
    simple = namespace["plot_beginner"](result)
    detailed = namespace["plot_programmer_diagnostics"](result)
    assert len(simple.axes) == 1
    assert "Held-out" in simple.axes[0].get_title()
    np.testing.assert_array_equal(
        simple.axes[0].lines[1].get_xdata(), result["curve"]["midpoint"]
    )
    np.testing.assert_array_equal(
        simple.axes[0].lines[1].get_ydata(), result["curve"]["dprime"]
    )
    assert len(detailed.axes) == 4
    assert {axis.get_title() for axis in detailed.axes} == {
        "Held-out block d′",
        "d′ components",
        "Held-out support",
        "Fold completeness",
    }

    import matplotlib.pyplot as plt

    plt.close(simple)
    plt.close(detailed)


def test_controls_are_deterministic_and_do_not_fetch_on_construction(monkeypatch):
    namespace = _execute_generated(monkeypatch)
    options = list(namespace["recording_control"].options)
    expected = namespace["recording_options"](namespace["DEFAULT_EXPERIMENT"])
    assert options == expected
    assert namespace["recording_control"].value == expected[0][1]
    assert namespace["area_control"].value == "mHV"
    assert namespace["diagnostics_control"].value is False
    assert len(namespace["run_output"].outputs) == 0
    assert isinstance(namespace["run_panel"], namespace["widgets"].VBox)
