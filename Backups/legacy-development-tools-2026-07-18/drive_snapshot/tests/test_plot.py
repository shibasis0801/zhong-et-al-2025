from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from zhong2025 import plot


def _provenance(figure):
    return dict(plot.info(figure).provenance)


def test_root_facade_and_recipe_catalog_expose_the_same_stable_api(monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd()))
    facade = importlib.import_module("plot")

    assert facade.recipes() == plot.recipes()
    assert len(plot.recipes()) == len(set(plot.recipes())) == 36
    assert set(plot.recipes()) <= set(plot.__all__)
    assert all(callable(getattr(facade, name)) for name in plot.recipes())
    assert "comparison" in repr(plot.guide())
    assert facade.colors("stimulus") == plot.colors("stimulus")


def test_recipes_return_fresh_tagged_figures_without_mutating_rcparams():
    before = matplotlib.rcParams.copy()
    first = plot.curve({"mice": np.arange(12, dtype=float).reshape(3, 4)})
    second = plot.curve({"mice": np.arange(12, dtype=float).reshape(3, 4)})
    try:
        assert first is not second
        assert plot.info(first).recipe == "curve"
        assert first.axes and second.axes
        for key, value in before.items():
            assert matplotlib.rcParams[key] == value
    finally:
        plot.close(first)
        plot.close(second)


def test_comparison_requires_a_declared_multiplicity_policy_and_aligns_ids():
    groups = {"before": [1.0, 2.0, 3.0], "after": [4.0, 5.0, 6.0]}
    open_before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="explicit correction policy"):
        plot.comparison(groups, comparisons=[("before", "after", 0.02)])
    with pytest.raises(ValueError, match="requires pair_ids"):
        plot.comparison(groups, paired=True)
    with pytest.raises(ValueError, match="identical pair_ids"):
        plot.comparison(
            groups,
            paired=True,
            pair_ids={"before": ["a", "b", "c"], "after": ["a", "b", "d"]},
        )
    assert set(plt.get_fignums()) == open_before

    figure = plot.comparison(
        groups,
        paired=True,
        pair_ids={"before": ["a", "b", "c"], "after": ["c", "a", "b"]},
        comparisons=[("before", "after", 0.02)],
        correction="holm",
        unit="mouse",
    )
    try:
        assert plot.info(figure).recipe == "comparison"
        assert _provenance(figure)["paired"] == "True"
        expected_pairs = ([1.0, 5.0], [2.0, 6.0], [3.0, 4.0])
        for connector, expected in zip(figure.axes[0].lines[:3], expected_pairs):
            np.testing.assert_array_equal(connector.get_ydata(), expected)
    finally:
        plot.close(figure)


def test_event_raster_accepts_ragged_per_trial_event_lists():
    figure = plot.event_raster(
        {
            "lick": [np.array([0.2, 0.4]), np.array([]), np.array([0.8])],
            "reward": [None, 0.7, 0.9],
        }
    )
    try:
        assert plot.info(figure).recipe == "event_raster"
        assert sum(len(collection.get_offsets()) for collection in figure.axes[0].collections) == 5
    finally:
        plot.close(figure)

    open_before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="would clip"):
        plot.event_raster({"lick": [[0.1], [0.2], [0.3]]}, trial_count=2)
    assert set(plt.get_fignums()) == open_before


def test_centered_matrix_uses_symmetric_limits_and_train_order_is_reused():
    centered = plot.matrix([[-1.0, 2.0], [0.5, -0.25]], center=0.0, robust=False)
    train = np.array([[0.0, 3.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    test = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]])
    held_out = plot.train_test(train, test, sort_by="peak")
    try:
        norm = centered.axes[0].images[0].norm
        assert norm.vcenter == 0.0
        assert norm.vmin == -norm.vmax == -2.0

        expected_order = np.array([1, 0, 2])
        np.testing.assert_array_equal(
            np.asarray(held_out.axes[0].images[0].get_array()), train[expected_order]
        )
        np.testing.assert_array_equal(
            np.asarray(held_out.axes[1].images[0].get_array()), test[expected_order]
        )
        assert "train" in plot.info(held_out).caption.lower()
    finally:
        plot.close(centered)
        plot.close(held_out)


def test_activity_ordering_rejects_float_indices_with_a_clear_error():
    with pytest.raises(ValueError, match="integer row indices"):
        plot.activity([[1, 2], [3, 4]], order=[0.0, 1.0])
    with pytest.raises(ValueError, match="integer row indices"):
        plot.rastermap([[1, 2], [3, 4]], order=[0.0, 1.0])


def test_heatmaps_reject_irregular_coordinates_instead_of_distorting_them():
    open_before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="evenly spaced"):
        plot.activity([[1, 2, 3]], samples=[0.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="evenly spaced"):
        plot.train_test([[1, 2, 3]], [[3, 2, 1]], samples=[0.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="evenly spaced"):
        plot.rastermap([[1, 2, 3]], time=[0.0, 1.0, 3.0])
    assert set(plt.get_fignums()) == open_before


def test_dprime_uses_the_paper_formula_and_records_the_sign():
    role_2 = np.array([2.0, 4.0, 6.0])
    role_0 = np.array([1.0, 2.0, 3.0])
    expected = 2 * (role_2.mean() - role_0.mean()) / (
        role_2.std(ddof=0) + role_0.std(ddof=0)
    )
    figure = plot.dprime(role_2, role_0)
    try:
        provenance = _provenance(figure)
        assert float(provenance["dprime"]) == pytest.approx(expected)
        assert provenance["sign"] == "role_2_minus_role_0"
        assert "population SD" in plot.info(figure).caption
    finally:
        plot.close(figure)


def test_infinite_dprime_is_selective_and_threshold_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        plot.dprime([1, 2], [2, 3], threshold=-0.1)

    figure = plot.dprime([2, 2], [1, 1], threshold=0.3)
    try:
        annotation = "\n".join(text.get_text() for text in figure.axes[0].texts)
        assert "d-prime = inf" in annotation
        assert ": selective" in annotation
    finally:
        plot.close(figure)


def test_numeric_series_keys_keep_their_supplied_x_coordinates():
    figure = plot.curve({1: [2.0, 3.0]}, x={1: [10.0, 20.0]})
    try:
        np.testing.assert_array_equal(figure.axes[0].lines[-1].get_xdata(), [10.0, 20.0])
    finally:
        plot.close(figure)


def test_matrix_annotations_and_labels_follow_supplied_coordinates():
    figure = plot.matrix(
        [[1.0, 2.0], [3.0, 4.0]],
        x=[10.0, 20.0],
        y=[100.0, 200.0],
        column_labels=["left", "right"],
        row_labels=["early", "late"],
        annotate=True,
    )
    try:
        assert [text.get_position() for text in figure.axes[0].texts] == [
            (10.0, 100.0),
            (20.0, 100.0),
            (10.0, 200.0),
            (20.0, 200.0),
        ]
        np.testing.assert_array_equal(figure.axes[0].get_xticks(), [10.0, 20.0])
        np.testing.assert_array_equal(figure.axes[0].get_yticks(), [100.0, 200.0])
    finally:
        plot.close(figure)


def test_grouped_relationship_reports_within_group_correlations_only():
    figure = plot.relationship(
        [0, 1, 10, 11],
        [0, 1, 10, 9],
        group=["supervised", "supervised", "unsupervised", "unsupervised"],
    )
    try:
        annotations = [text.get_text() for text in figure.axes[0].texts]
        assert any(text.startswith("supervised:") for text in annotations)
        assert any(text.startswith("unsupervised:") for text in annotations)
        assert not any(text.startswith("r =") for text in annotations)
    finally:
        plot.close(figure)


def test_qc_counts_nonfinite_support_as_missing_and_uses_finite_means():
    figure = plot.qc(
        {
            "mean_run_speed": [[1.0, np.nan], [3.0, 5.0], [np.nan, 7.0]],
            "frame_counts": [[1.0, np.nan], [0.0, 2.0], [2.0, 4.0]],
        }
    )
    try:
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [2.0, 6.0])
        np.testing.assert_allclose(figure.axes[2].lines[0].get_ydata(), [1.0, 3.0])
        missing_line = next(
            line
            for axis in figure.axes
            for line in axis.lines
            if line.get_label() == "missing fraction"
        )
        np.testing.assert_allclose(missing_line.get_ydata(), [1 / 3, 1 / 3])
    finally:
        plot.close(figure)


def test_recording_heatmap_uses_position_bin_edges_not_centers():
    from zhong2025 import load_atlas_demo

    demo = load_atlas_demo()
    position = np.asarray(demo["position_centers_m"], dtype=float)
    step = position[1] - position[0]
    figure = plot.recording(demo)
    try:
        expected = [position[0] - step / 2, position[-1] + step / 2]
        np.testing.assert_allclose(figure.axes[1].images[0].get_extent()[:2], expected)
        np.testing.assert_allclose(figure.axes[4].images[0].get_extent()[:2], expected)
    finally:
        plot.close(figure)


def test_style_is_deterministic_even_when_callers_customize_rcparams():
    with matplotlib.rc_context({"text.color": "magenta", "axes.titlecolor": "magenta"}):
        figure = plot.curve([1.0, 2.0], title="Stable title")
        try:
            assert figure.axes[0].title.get_color() == "#111827"
            assert matplotlib.rcParams["text.color"] == "magenta"
        finally:
            plot.close(figure)


def test_domain_wrappers_accept_analysis_results_and_cortex_guards_weights():
    result = {"midpoint": np.array([20.0, 60.0]), "dprime": np.eye(2)}
    cross = plot.cross_temporal(result)
    surface = plot.position_surface(result, position=[1.0, 2.0])
    empty_cortex = plot.cortical_map([0, 1], [1, 2], values=[np.nan, np.nan], center=0)
    try:
        assert plot.info(cross).recipe == "matrix"
        assert plot.info(surface).recipe == "matrix"
        assert "No finite cortical values" in empty_cortex.axes[0].texts[0].get_text()
    finally:
        plot.close(cross)
        plot.close(surface)
        plot.close(empty_cortex)

    with pytest.raises(ValueError, match="non-negative"):
        plot.cortical_density([0, 1], [0, 1], selected=[1, -1])


def test_spectrum_sorts_rank_and_trajectory_breaks_at_missing_samples():
    spectrum = plot.spectrum([1.0, 4.0, 2.0])
    trajectory = plot.trajectory([[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan], [3.0, 3.0]])
    try:
        np.testing.assert_array_equal(spectrum.axes[0].lines[0].get_ydata(), [4.0, 2.0, 1.0])
        assert np.isnan(trajectory.axes[0].lines[0].get_ydata()[2])
    finally:
        plot.close(spectrum)
        plot.close(trajectory)


def test_composition_and_empty_journey_inputs_fail_or_render_cleanly():
    with pytest.raises(ValueError, match="all stacked segments"):
        plot.stacked_bars({"mapping": {"x": 1}, "array": [2]})

    figure = plot.all_mouse_journeys({"mouse": []})
    try:
        assert plot.info(figure).recipe == "all_mouse_journeys"
        assert "No acquisition rows" in figure.axes[0].texts[0].get_text()
    finally:
        plot.close(figure)


def test_cohort_preflight_records_global_mice_and_cross_group_overlap():
    records = [
        {"mouse": "A", "group": "sup", "date": "1", "neural_bytes": 10},
        {"mouse": "A", "group": "unsup", "date": "2", "neural_bytes": 20},
        {"mouse": "B", "group": "sup", "date": "3", "neural_bytes": 30},
    ]
    figure = plot.cohort_preflight(records)
    try:
        provenance = _provenance(figure)
        assert provenance["unique_mice"] == "2"
        assert provenance["group_mouse_memberships"] == "3"
        assert provenance["overlapping_mice"] == "A"
        assert "not independent" in plot.info(figure).warnings[0]
        assert figure.axes[2].get_title(loc="left") == "Reported layer totals"
        assert matplotlib.colors.to_hex(figure.axes[0].patches[0].get_facecolor()) == plot.colors("cohort")["sup"].lower()
        assert matplotlib.colors.to_hex(figure.axes[0].patches[1].get_facecolor()) == plot.colors("cohort")["unsup"].lower()
    finally:
        plot.close(figure)


def test_blockwise_discloses_missing_invalid_reason_schema():
    result = {
        "midpoint": np.array([10.0, 20.0]),
        "dprime": np.array([0.2, np.nan]),
        "separation": np.array([0.3, np.nan]),
        "spread": np.array([1.0, np.nan]),
        "n_a": np.array([8, 3]),
        "n_b": np.array([8, 2]),
        "valid_folds": np.array([4, 1]),
        "required_folds": np.array([4, 4]),
    }
    figure = plot.blockwise(result)
    try:
        assert plot.info(figure).warnings == (
            "The current blockwise result schema does not retain per-block invalid reasons.",
        )
    finally:
        plot.close(figure)


def test_save_adds_a_png_suffix_and_close_releases_the_figure(tmp_path):
    figure = plot.bars({"files": 297})
    number = figure.number
    output = plot.save(figure, tmp_path / "release-counts", dpi=80)
    plot.close(figure)

    assert output == (tmp_path / "release-counts.png").resolve()
    assert output.is_file() and output.stat().st_size > 1_000
    assert number not in plt.get_fignums()
