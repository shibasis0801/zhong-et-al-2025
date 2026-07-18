from __future__ import annotations

import numpy as np
import pytest

from zhong2025.learning import (
    area_transform,
    blockwise_dprime,
    bootstrap_group_difference,
    cross_temporal_dprime,
    crossvalidated_scores,
    exact_group_permutation,
    fit_early_slope,
    fit_saturation_curve,
    position_dprime_surface,
    prepare_session_trials,
    simulate_mouse_curves,
    svd_dprime,
    svd_dprime_contrasts,
    trial_responses,
)


def _learning_tensor(seed=4):
    rng = np.random.default_rng(seed)
    n_trials, n_bins, n_features = 160, 6, 5
    labels = np.tile([2, 0], n_trials // 2)
    features = rng.normal(0, 0.7, size=(n_trials, n_bins, n_features))
    for trial in range(n_trials):
        block = trial // 40
        sign = 1 if labels[trial] == 2 else -1
        features[trial, :4, 0] += sign * (0.12 + 0.10 * block)
    return features, labels, np.arange(n_trials)


def _session_inputs():
    positions = np.tile([5.0, 15.0, 45.0, 55.0], 2)
    behavior = {
        "ft_Pos": positions,
        "ft_trInd": np.repeat([0, 1], 4),
        "ft_isMoving": np.tile([True, False, True, False], 2),
        "ft_RunSpeed": np.arange(1.0, 9.0),
        "WallName": np.array(["leaf", "circle"]),
        "UniqWalls": np.array(["leaf", "circle"]),
        "stim_id": np.array([2, 0]),
    }
    svd = {
        "U": np.eye(2),
        "V": np.vstack([np.arange(8.0), np.arange(8.0) + 10.0]),
    }
    retinotopy = {"iarea": np.array([8, 8])}
    return behavior, svd, retinotopy


def test_svd_dprime_matches_direct_reconstruction_without_epsilon():
    rng = np.random.default_rng(12)
    components, neurons, frames = 7, 31, 83
    u = rng.normal(size=(components, neurons))
    v = rng.normal(size=(components, frames))
    frames_a = np.arange(frames) % 3 == 0
    frames_b = np.flatnonzero(np.arange(frames) % 3 == 1)

    with np.errstate(all="ignore"):
        reconstructed = u.T @ v
        expected = 2.0 * (
            reconstructed[:, frames_a].mean(axis=1)
            - reconstructed[:, frames_b].mean(axis=1)
        ) / (
            reconstructed[:, frames_a].std(axis=1)
            + reconstructed[:, frames_b].std(axis=1)
        )
    actual = svd_dprime(
        u,
        v,
        frames_a,
        frames_b,
        neuron_chunk_size=5,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_svd_dprime_contrasts_reuses_groups_and_preserves_zero_spread():
    rng = np.random.default_rng(19)
    u = rng.normal(size=(4, 13))
    v = rng.normal(size=(4, 30))
    groups = {
        "circle1": np.arange(0, 10),
        "leaf1": np.arange(10, 20),
        "leaf2": np.arange(20, 30),
    }
    contrasts = svd_dprime_contrasts(
        u,
        v,
        groups,
        {
            "learned": ("leaf1", "circle1"),
            "novel": ("leaf2", "circle1"),
        },
        neuron_chunk_size=3,
    )
    with np.errstate(all="ignore"):
        reconstructed = u.T @ v
        for name, stimulus in (("learned", "leaf1"), ("novel", "leaf2")):
            a = reconstructed[:, groups[stimulus]]
            b = reconstructed[:, groups["circle1"]]
            expected = 2.0 * (a.mean(1) - b.mean(1)) / (a.std(1) + b.std(1))
            np.testing.assert_allclose(
                contrasts[name], expected, rtol=1e-12, atol=1e-12
            )

    constant_v = np.vstack([np.zeros(4), np.array([0.0, 0.0, 1.0, 1.0])])
    with np.errstate(all="ignore"):
        zero_spread = svd_dprime(
            np.array([[0.0], [1.0]]),
            constant_v,
            [0, 1],
            [2, 3],
        )
    assert np.isneginf(zero_spread[0])


def test_svd_dprime_validates_factor_shapes_and_frame_selectors():
    u = np.ones((2, 3))
    v = np.ones((2, 5))
    with pytest.raises(ValueError, match="same component axis"):
        svd_dprime(np.ones((3, 3)), v, [0], [1])
    with pytest.raises(ValueError, match="one value per V frame"):
        svd_dprime(u, v, [True, False], [1])
    with pytest.raises(ValueError, match="out-of-range"):
        svd_dprime(u, v, [0], [5])
    with pytest.raises(ValueError, match="at least one frame"):
        svd_dprime(u, v, [], [1])


def test_prepare_session_trials_defaults_to_moving_frames_and_records_rule():
    behavior, svd, retinotopy = _session_inputs()
    default = prepare_session_trials(
        behavior, svd, retinotopy, n_features=2, n_position_bins=2
    )
    explicit = prepare_session_trials(
        behavior,
        svd,
        retinotopy,
        n_features=2,
        n_position_bins=2,
        movement_rule="moving_only",
    )

    assert default["movement_rule"] == "moving_only"
    np.testing.assert_array_equal(default["frame_counts"], [[1, 1], [1, 1]])
    np.testing.assert_array_equal(default["frame_counts"], explicit["frame_counts"])
    np.testing.assert_allclose(
        default["trial_features"], explicit["trial_features"], equal_nan=True
    )


def test_prepare_session_trials_can_include_stationary_valid_frames():
    behavior, svd, retinotopy = _session_inputs()
    result = prepare_session_trials(
        behavior,
        svd,
        retinotopy,
        n_features=2,
        n_position_bins=2,
        movement_rule="all_valid_frames",
    )

    assert result["movement_rule"] == "all_valid_frames"
    np.testing.assert_array_equal(result["frame_counts"], [[2, 2], [2, 2]])


def test_prepare_session_trials_rejects_unknown_movement_rule():
    behavior, svd, retinotopy = _session_inputs()
    with pytest.raises(ValueError, match="movement_rule must be one of"):
        prepare_session_trials(
            behavior,
            svd,
            retinotopy,
            movement_rule="moving_and_fast",
        )


def test_trial_level_blocked_dprime_tracks_increasing_separation():
    features, labels, trials = _learning_tensor()
    texture = np.array([True, True, True, True, False, False])
    response = trial_responses(features, texture)
    assert response.shape == (160, 5)

    curve = blockwise_dprime(
        features,
        labels,
        trials,
        position_mask=texture,
        block_trials=40,
        n_folds=4,
    )
    assert len(curve["dprime"]) == 4
    assert np.isfinite(curve["dprime"]).all()
    assert curve["dprime"][-1] > curve["dprime"][0]
    assert np.all(curve["n_a"] >= 4)
    assert np.all(curve["n_b"] >= 4)


def test_position_surface_and_cross_temporal_matrix_have_expected_shapes():
    features, labels, trials = _learning_tensor()
    surface = position_dprime_surface(
        features,
        labels,
        trials,
        block_trials=40,
    )
    assert surface["dprime"].shape == (4, 6)
    assert surface["midpoint"].shape == (4,)

    generalization = cross_temporal_dprime(
        features,
        labels,
        trials,
        position_mask=np.arange(6) < 4,
        block_trials=40,
    )
    assert generalization["dprime"].shape == (4, 4)
    assert np.isfinite(np.diag(generalization["dprime"])).all()


def test_curve_summaries_report_rate_and_observed_plateau():
    midpoint = np.arange(20, 341, 40, dtype=float)
    expected = 0.2 + 1.1 * (1.0 - np.exp(-0.018 * (midpoint - midpoint.min())))
    slope = fit_early_slope(midpoint, expected, early_horizon=140)
    fit = fit_saturation_curve(midpoint, expected)

    assert slope["slope"] > 0
    assert slope["n_blocks"] == 4
    assert np.isclose(fit["plateau"], 1.3, atol=0.04)
    assert np.isclose(fit["k"], 0.018, rtol=0.2)
    assert fit["initial_rate"] > 0
    assert fit["plateau_observed"] is True


def test_mouse_level_inference_resamples_mice_and_exactly_enumerates_labels():
    values = np.array([0.030, 0.026, 0.021, 0.020, *([0.004] * 9)])
    groups = ["rewarded"] * 4 + ["unrewarded"] * 9
    mice = [f"mouse-{index:02d}" for index in range(13)]
    permutation = exact_group_permutation(values, groups, mouse_ids=mice)
    bootstrap = bootstrap_group_difference(
        values, groups, mouse_ids=mice, n_boot=300, seed=3
    )

    assert permutation["permutations"] == 715
    assert permutation["difference"] > 0
    assert permutation["pvalue"] <= 1 / 715 + 1e-12
    assert bootstrap["difference"] > 0
    assert bootstrap["ci"][0] > 0
    assert bootstrap["samples"].shape == (300,)


def test_incomplete_corridor_coverage_cannot_create_a_label_difference():
    labels = np.tile([2, 0], 20)
    features = np.full((40, 2, 1), np.nan)
    features[labels == 2, 0, 0] = 0.0
    features[labels == 0, 1, 0] = 10.0

    strict = trial_responses(features)
    sensitivity = trial_responses(
        features, require_complete_position_coverage=False
    )
    assert np.isnan(strict).all()
    assert np.isfinite(sensitivity).all()
    curve = blockwise_dprime(features, labels, np.arange(40), block_trials=40)
    assert np.isnan(curve["dprime"][0])
    assert curve["valid_folds"][0] == 0


def test_fold_offsets_are_not_pooled_into_spurious_dprime():
    # Class proportions differ by physical-time fold, but within each fold the
    # two roles have exactly the same feature mean. Pooling scores from four
    # separately normalized models turns fold offsets into a class effect; a
    # within-fold contrast is exactly zero.
    responses = []
    labels = []
    for fold, n_a in enumerate([8, 8, 2, 2]):
        n_b = 10 - n_a
        responses.extend(
            [[float(fold) + offset] for offset in np.linspace(-0.1, 0.1, n_a)]
        )
        labels.extend([2] * n_a)
        responses.extend(
            [[float(fold) + offset] for offset in np.linspace(-0.1, 0.1, n_b)]
        )
        labels.extend([0] * n_b)
    local = crossvalidated_scores(
        np.asarray(responses), np.asarray(labels), n_folds=4, min_per_role=4
    )
    assert local["valid_folds"] == local["required_folds"] == 4
    assert all(abs(row["dprime"]) < 1e-12 for row in local["fold_metrics"])

    features = np.asarray(responses)[:, None, :]
    curve = blockwise_dprime(
        features, np.asarray(labels), np.arange(40), block_trials=40
    )
    assert abs(curve["dprime"][0]) < 1e-12
    assert abs(curve["separation"][0]) < 1e-12


def test_missing_class_in_one_physical_fold_invalidates_whole_block():
    rng = np.random.default_rng(31)
    responses = rng.normal(size=(40, 3))
    labels = np.tile([2, 0], 20)
    labels[:10] = 2
    local = crossvalidated_scores(
        responses, labels, n_folds=4, min_per_role=4
    )
    assert local["valid_folds"] == 0
    assert local["scores"].size == 0
    assert "fold 0" in local["invalid_reason"]


def test_half_time_is_not_misreported_as_an_observed_plateau():
    midpoint = np.linspace(20.0, 120.0, 6)
    expected = 0.2 + 1.1 * (
        1.0 - np.exp(-0.01 * (midpoint - midpoint.min()))
    )
    fit = fit_saturation_curve(midpoint, expected)
    assert 0.55 < fit["fraction_of_amplitude_observed"] < 0.70
    assert fit["plateau_observed"] is False
    assert fit["reference_trial"] == 20.0


def test_repeated_sessions_collapse_to_unique_mice_and_groups_are_strict():
    values = [0.03, 0.01, 0.02, 0.00, 0.00]
    groups = ["rewarded", "rewarded", "rewarded", "unrewarded", "unrewarded"]
    mice = ["R1", "R1", "R2", "U1", "U2"]
    result = exact_group_permutation(values, groups, mouse_ids=mice)
    bootstrap = bootstrap_group_difference(
        values, groups, mouse_ids=mice, n_boot=50, seed=2
    )
    assert result["n_mice"] == 4
    assert result["permutations"] == 6
    assert np.isclose(result["difference"], 0.02)
    assert bootstrap["n_mice"] == 4

    with pytest.raises(ValueError, match="unexpected group"):
        exact_group_permutation(
            [0.1, 0.0, 0.2],
            ["rewarded", "unrewarded", "rewraded"],
            mouse_ids=["R1", "U1", "X1"],
        )
    with pytest.raises(ValueError, match="more than one group"):
        exact_group_permutation(
            [0.1, 0.2, 0.0],
            ["rewarded", "unrewarded", "unrewarded"],
            mouse_ids=["same", "same", "U1"],
        )


def test_public_shape_and_count_validation_is_explicit():
    features, labels, trials = _learning_tensor()
    with pytest.raises(ValueError, match="must align"):
        cross_temporal_dprime(features, labels[:-1], trials)
    with pytest.raises(ValueError, match="n_features"):
        area_transform(np.ones((3, 4)), np.full(4, 8), "V1", n_features=-1)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_group_difference(
            [0.1, 0.0],
            ["rewarded", "unrewarded"],
            mouse_ids=["R1", "U1"],
            n_boot=0,
        )


def test_area_transform_and_simulator_keep_feature_and_mouse_counts_fixed():
    rng = np.random.default_rng(8)
    u = rng.normal(size=(7, 12))
    area_id = np.array([8] * 6 + [0, 1, 2, 9, 3, 4])
    transform = area_transform(u, area_id, "V1", n_features=4)
    assert transform.shape == (7, 4)

    rows = simulate_mouse_curves(rewarded_mice=4, unrewarded_mice=9, seed=9)
    assert len(rows) == 13
    assert {row["group"] for row in rows} == {"rewarded", "unrewarded"}
    assert all(len(row["midpoint"]) == 8 for row in rows)
