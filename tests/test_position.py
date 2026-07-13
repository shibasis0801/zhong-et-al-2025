import numpy as np
import pytest

from zhong2025.position import (
    align_trailing_behavior_frames,
    bin_trial_features,
    decimeters_to_meters,
    position_bin_indices,
)


def test_decimeter_to_meter_units():
    np.testing.assert_allclose(
        decimeters_to_meters([0, 20, 40, 60]), [0, 2, 4, 6]
    )


def test_bin_edges_are_left_closed_right_open_and_last_inclusive():
    edges = np.array([0.0, 1.0, 2.0])
    result = position_bin_indices([0.0, np.nextafter(1.0, 0.0), 1.0, 2.0], edges)
    np.testing.assert_array_equal(result, [0, 0, 1, 1])


def test_out_of_range_and_nonfinite_positions_raise():
    with pytest.raises(ValueError, match="must lie"):
        position_bin_indices([-0.01, 0.5], [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        position_bin_indices([np.nan], [0.0, 1.0])


def test_frame_mismatch_requires_explicit_trailing_allowance():
    neural = np.zeros((3, 2))
    behavior = {"position": np.arange(4), "trial": np.arange(4)}
    with pytest.raises(ValueError, match="extra frame"):
        align_trailing_behavior_frames(neural, behavior)
    _, aligned, report = align_trailing_behavior_frames(
        neural, behavior, max_trailing_behavior_frames=1
    )
    assert report.dropped_trailing_behavior_frames == 1
    assert all(len(values) == 3 for values in aligned.values())


def test_behavior_fields_must_remain_frame_aligned():
    with pytest.raises(ValueError, match="lengths disagree"):
        align_trailing_behavior_frames(
            np.zeros((3, 1)), {"position": np.arange(3), "trial": np.arange(4)}
        )


def test_binning_does_not_interpolate_across_trial_reset():
    features = np.array([[1.0], [3.0], [10.0], [14.0]])
    position = np.array([0.1, 0.9, 0.1, 0.9])
    trial = np.array([0, 0, 1, 1])
    trial_ids, binned, counts = bin_trial_features(
        features, position, trial, [0.0, 0.5, 1.0]
    )
    np.testing.assert_array_equal(trial_ids, [0, 1])
    np.testing.assert_array_equal(counts, [[1, 1], [1, 1]])
    np.testing.assert_allclose(binned[:, :, 0], [[1, 3], [10, 14]])


def test_empty_bins_are_nan_with_zero_count():
    _, binned, counts = bin_trial_features(
        [[5.0]], [0.1], [7], [0.0, 0.5, 1.0]
    )
    assert counts[0, 1] == 0
    assert np.isnan(binned[0, 1, 0])


def test_impulse_keeps_same_frame_position():
    features = np.zeros((4, 1))
    features[2, 0] = 9.0
    _, binned, _ = bin_trial_features(
        features, [0.1, 0.2, 0.7, 0.8], [0, 0, 0, 0], [0.0, 0.5, 1.0]
    )
    assert binned[0, 0, 0] == 0.0
    assert binned[0, 1, 0] == 4.5

