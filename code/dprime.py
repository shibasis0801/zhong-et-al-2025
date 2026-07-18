"""Trial-level helpers for the Zhong reward-versus-exposure project.

The functions in this module deliberately keep inference at the recording/mouse
level.  Neural frames are first averaged within trial and corridor-position
bins; they are never treated as independent observations in a group test.
NumPy is the only numerical dependency so the helpers run in a basic Colab CPU
runtime and in the repository's small test environment.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from position import (
    align_trailing_behavior_frames,
    bin_trial_features,
    decimeters_to_meters,
)


AREA_IDS: Mapping[str, tuple[int, ...]] = {
    "V1": (8,),
    "mHV": (0, 1, 2, 9),
    "lHV": (5, 6),
    "aHV": (3, 4),
}


def _finite_rows(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    return np.isfinite(values).all(axis=1)


def _positive_integer(name: str, value: Any, *, minimum: int = 1) -> int:
    """Validate count-like public arguments without silently truncating them."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer of at least {minimum}") from error
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(numeric)


def _dprime(scores: ArrayLike, labels: ArrayLike, role_a: int, role_b: int) -> dict[str, float]:
    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels)
    a = score[label == role_a]
    b = score[label == role_b]
    if min(len(a), len(b)) < 2:
        return {
            "dprime": float("nan"),
            "mean_a": float("nan"),
            "mean_b": float("nan"),
            "sd_a": float("nan"),
            "sd_b": float("nan"),
            "n_a": float(len(a)),
            "n_b": float(len(b)),
        }
    sd_a = float(np.std(a, ddof=1))
    sd_b = float(np.std(b, ddof=1))
    spread = sd_a + sd_b
    value = float("nan") if spread <= 0 else float(2.0 * (np.mean(a) - np.mean(b)) / spread)
    return {
        "dprime": value,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "sd_a": sd_a,
        "sd_b": sd_b,
        "n_a": float(len(a)),
        "n_b": float(len(b)),
    }


def _frame_indices(
    selector: ArrayLike,
    *,
    n_frames: int,
    name: str,
) -> NDArray[np.intp]:
    """Normalize one boolean mask or integer index vector."""

    values = np.asarray(selector)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional frame selector")
    if len(values) == 0:
        raise ValueError(f"{name} must select at least one frame")
    if np.issubdtype(values.dtype, np.bool_):
        if len(values) != n_frames:
            raise ValueError(f"{name} boolean mask must have one value per V frame")
        indices = np.flatnonzero(values)
    elif np.issubdtype(values.dtype, np.integer):
        indices = values.astype(np.intp, copy=False)
        if np.any(indices < 0) or np.any(indices >= n_frames):
            raise ValueError(f"{name} contains an out-of-range frame index")
    else:
        raise ValueError(f"{name} must be a boolean mask or integer frame indices")
    if len(indices) == 0:
        raise ValueError(f"{name} must select at least one frame")
    return indices


def _svd_projected_group_moments(
    u: NDArray,
    v: NDArray,
    frame_indices: NDArray[np.intp],
    *,
    neuron_chunk_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-neuron mean and population SD for one frame group."""

    # Advanced indexing gives us an owned components x selected-frames block,
    # so centering it in-place cannot modify the caller's factor matrix.  NaN
    # component frames would reconstruct to NaN for every neuron and are
    # therefore omitted, matching nanmean/nanstd on the reconstructed matrix.
    block = np.asarray(v[:, frame_indices], dtype=np.float64)
    valid_frames = ~np.isnan(block).any(axis=0)
    if not np.all(valid_frames):
        block = block[:, valid_frames]
    if block.shape[1] == 0:
        return (
            np.full(u.shape[1], np.nan, dtype=np.float64),
            np.full(u.shape[1], np.nan, dtype=np.float64),
        )

    component_mean = np.mean(block, axis=1)
    block -= component_mean[:, None]
    with np.errstate(all="ignore"):
        component_covariance = (block @ block.T) / block.shape[1]

    n_neurons = u.shape[1]
    mean = np.empty(n_neurons, dtype=np.float64)
    standard_deviation = np.empty(n_neurons, dtype=np.float64)
    for start in range(0, n_neurons, neuron_chunk_size):
        stop = min(start + neuron_chunk_size, n_neurons)
        weights = np.asarray(u[:, start:stop], dtype=np.float64)
        with np.errstate(all="ignore"):
            mean[start:stop] = weights.T @ component_mean
            covariance_times_weights = component_covariance @ weights
            variance = np.sum(weights * covariance_times_weights, axis=0)
        # Roundoff can make a mathematically non-negative quadratic form a few
        # ulps negative.  Clipping only that numerical residue preserves the
        # population-SD definition used by np.nanstd(..., ddof=0).
        standard_deviation[start:stop] = np.sqrt(np.maximum(variance, 0.0))
    return mean, standard_deviation


def svd_dprime_contrasts(
    u_components_by_neuron: ArrayLike,
    v_components_by_frame: ArrayLike,
    frame_groups: Mapping[Any, ArrayLike],
    contrasts: Mapping[str, tuple[Any, Any]],
    *,
    neuron_chunk_size: int = 4096,
) -> dict[str, NDArray[np.float64]]:
    """Compute several per-neuron d-prime contrasts from SVD factors.

    ``U`` must be ``components x neurons`` and ``V`` must be
    ``components x frames``.  Frame groups may be boolean masks or integer
    index vectors.  Every group's mean and population standard deviation is
    computed once, so a shared reference group can be reused across contrasts.

    This is algebraically equivalent to reconstructing ``U.T @ V`` and using
    ``np.nanmean``/``np.nanstd(..., ddof=0)`` on each selected frame group, but
    it never materializes the ``neurons x frames`` matrix.  It also deliberately
    adds no denominator epsilon, preserving the original notebook's NaN/Inf
    behavior when both within-group standard deviations are zero.

    If ``C`` is the number of components, ``N`` the number of neurons, ``B``
    the neuron chunk size, and ``F`` the largest selected frame group, extra
    working memory is ``O(C*F + C**2 + C*B + N*G)`` for ``G`` unique groups,
    rather than ``O(N*T)`` for a full reconstruction.
    """

    u = np.asarray(u_components_by_neuron)
    v = np.asarray(v_components_by_frame)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("U and V must both be two-dimensional")
    if u.shape[0] != v.shape[0]:
        raise ValueError("U and V must have the same component axis")
    if (
        not np.issubdtype(u.dtype, np.number)
        or not np.issubdtype(v.dtype, np.number)
        or np.issubdtype(u.dtype, np.complexfloating)
        or np.issubdtype(v.dtype, np.complexfloating)
    ):
        raise ValueError("U and V must contain real numeric values")
    neuron_chunk_size = _positive_integer("neuron_chunk_size", neuron_chunk_size)
    if not frame_groups:
        raise ValueError("frame_groups must contain at least one group")
    if not contrasts:
        raise ValueError("contrasts must contain at least one contrast")

    required_groups: list[Any] = []
    for contrast_name, pair in contrasts.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                f"contrast {contrast_name!r} must name exactly two frame groups"
            )
        for group_name in pair:
            if group_name not in frame_groups:
                raise ValueError(
                    f"contrast {contrast_name!r} references unknown group {group_name!r}"
                )
            if group_name not in required_groups:
                required_groups.append(group_name)

    moments: dict[Any, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for group_name in required_groups:
        indices = _frame_indices(
            frame_groups[group_name],
            n_frames=v.shape[1],
            name=f"frame_groups[{group_name!r}]",
        )
        moments[group_name] = _svd_projected_group_moments(
            u,
            v,
            indices,
            neuron_chunk_size=neuron_chunk_size,
        )

    result: dict[str, NDArray[np.float64]] = {}
    for contrast_name, (group_a, group_b) in contrasts.items():
        mean_a, sd_a = moments[group_a]
        mean_b, sd_b = moments[group_b]
        with np.errstate(all="ignore"):
            result[contrast_name] = 2.0 * (mean_a - mean_b) / (sd_a + sd_b)
    return result


def svd_dprime(
    u_components_by_neuron: ArrayLike,
    v_components_by_frame: ArrayLike,
    frames_a: ArrayLike,
    frames_b: ArrayLike,
    *,
    neuron_chunk_size: int = 4096,
) -> NDArray[np.float64]:
    """Compute one per-neuron d-prime contrast without reconstructing frames."""

    return svd_dprime_contrasts(
        u_components_by_neuron,
        v_components_by_frame,
        {"a": frames_a, "b": frames_b},
        {"dprime": ("a", "b")},
        neuron_chunk_size=neuron_chunk_size,
    )["dprime"]


def trial_responses(
    trial_features: ArrayLike,
    position_mask: ArrayLike | None = None,
    *,
    require_complete_position_coverage: bool = True,
) -> NDArray[np.float64]:
    """Average position-binned features once per trial.

    ``trial_features`` must be ``trials x position bins x features``.  The
    default requires every chosen position bin to be populated for a trial, so
    label- or time-dependent corridor coverage cannot masquerade as a neural
    response difference.  Relaxing this rule is an explicitly named
    sensitivity analysis, never the primary path.
    """

    features = np.asarray(trial_features, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError("trial_features must have shape (trials, position, features)")
    if position_mask is None:
        mask = np.ones(features.shape[1], dtype=bool)
    else:
        mask = np.asarray(position_mask, dtype=bool)
        if mask.shape != (features.shape[1],):
            raise ValueError("position_mask must have one value per position bin")
    if not np.any(mask):
        raise ValueError("position_mask must select at least one bin")
    selected = features[:, mask, :]
    complete = np.isfinite(selected).all(axis=(1, 2))
    finite = np.isfinite(selected)
    counts = np.sum(finite, axis=1)
    response = np.full((len(selected), selected.shape[2]), np.nan, dtype=np.float64)
    np.divide(
        np.nansum(selected, axis=1),
        counts,
        out=response,
        where=counts > 0,
    )
    if require_complete_position_coverage:
        response[~complete] = np.nan
    return response


def crossvalidated_scores(
    responses: ArrayLike,
    labels: ArrayLike,
    *,
    role_a: int = 2,
    role_b: int = 0,
    n_folds: int = 4,
    min_per_role: int = 4,
) -> dict[str, Any]:
    """Return held-out coding scores and a contrast for every physical-time fold.

    Fold boundaries are assigned *before* trials with other roles or missing
    responses are filtered.  This preserves physical time: an early and late
    fold cannot collapse together simply because one contains unusable trials.
    Every requested fold must contain both roles (at least two test trials per
    role), otherwise the whole block is marked invalid rather than being
    estimated from a convenient subset of time.

    The coding direction and feature standardization are fit only on the other
    contiguous folds.  D-prime is then computed separately inside each held-out
    fold; callers must not pool raw scores from independently fitted folds.

    These local splits make the score held out with respect to trials.  If the
    input features came from the release-wide SVD/area basis, that upstream
    representation remains transductive and this is not a prospective online
    decoder.
    """

    values = np.asarray(responses, dtype=np.float64)
    label = np.asarray(labels)
    if values.ndim != 2 or label.ndim != 1 or len(values) != len(label):
        raise ValueError("responses and labels must align as trials x features")
    if role_a == role_b:
        raise ValueError("role_a and role_b must be distinct")
    n_folds = _positive_integer("n_folds", n_folds, minimum=2)
    min_per_role = _positive_integer("min_per_role", min_per_role, minimum=2)
    if len(values) < n_folds:
        return {
            "scores": np.array([]), "labels": np.array([]),
            "fold": np.array([], dtype=np.int16), "valid_folds": 0,
            "required_folds": int(n_folds), "fold_metrics": [],
            "invalid_reason": "fewer physical trials than requested folds",
        }

    # Assign folds on the original local block, before any role/coverage filter.
    fold = np.empty(len(values), dtype=np.int16)
    for fold_id, indices in enumerate(np.array_split(np.arange(len(values)), n_folds)):
        fold[indices] = fold_id
    selected = np.isin(label, [role_a, role_b]) & _finite_rows(values)

    # Validate every physical-time fold up front, so no partial time support is
    # silently reported as a complete block estimate.
    for fold_id in range(n_folds):
        test = selected & (fold == fold_id)
        train = selected & (fold != fold_id)
        train_labels = label[train]
        test_labels = label[test]
        if min(
            np.count_nonzero(train_labels == role_a),
            np.count_nonzero(train_labels == role_b),
        ) < min_per_role or min(
            np.count_nonzero(test_labels == role_a),
            np.count_nonzero(test_labels == role_b),
        ) < 2:
            return {
                "scores": np.array([]), "labels": np.array([]),
                "fold": np.array([], dtype=np.int16), "valid_folds": 0,
                "required_folds": int(n_folds), "fold_metrics": [],
                "invalid_reason": (
                    f"fold {fold_id} lacks the predeclared train/test class support"
                ),
            }

    scores = np.full(len(values), np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, float]] = []
    for fold_id in range(n_folds):
        test = selected & (fold == fold_id)
        train = selected & (fold != fold_id)
        train_labels = label[train]
        mean = np.mean(values[train], axis=0)
        scale = np.std(values[train], axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
        train_z = (values[train] - mean) / scale
        direction = (
            np.mean(train_z[train_labels == role_a], axis=0)
            - np.mean(train_z[train_labels == role_b], axis=0)
        )
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0:
            return {
                "scores": np.array([]), "labels": np.array([]),
                "fold": np.array([], dtype=np.int16), "valid_folds": 0,
                "required_folds": int(n_folds), "fold_metrics": [],
                "invalid_reason": f"fold {fold_id} has no trainable coding direction",
            }
        scores[test] = ((values[test] - mean) / scale) @ (direction / norm)
        metrics = _dprime(scores[test], label[test], role_a, role_b)
        if not np.isfinite(metrics["dprime"]):
            return {
                "scores": np.array([]), "labels": np.array([]),
                "fold": np.array([], dtype=np.int16), "valid_folds": 0,
                "required_folds": int(n_folds), "fold_metrics": [],
                "invalid_reason": f"fold {fold_id} has an undefined held-out d-prime",
            }
        fold_metrics.append({"fold": float(fold_id), **metrics})
    keep = np.isfinite(scores)
    return {
        "scores": scores[keep],
        "labels": label[keep],
        "fold": fold[keep],
        "valid_folds": int(n_folds),
        "required_folds": int(n_folds),
        "fold_metrics": fold_metrics,
        "invalid_reason": None,
    }


def _summarize_fold_metrics(local: Mapping[str, Any]) -> dict[str, float]:
    """Combine only within-fold contrasts; never pool fold-specific scores."""

    metrics = list(local.get("fold_metrics", []))
    required = int(local.get("required_folds", 0))
    if len(metrics) != required or int(local.get("valid_folds", 0)) != required:
        return {
            "dprime": float("nan"), "mean_a": float("nan"),
            "mean_b": float("nan"), "sd_a": float("nan"),
            "sd_b": float("nan"), "separation": float("nan"),
            "spread": float("nan"), "n_a": 0.0, "n_b": 0.0,
        }
    return {
        "dprime": float(np.mean([row["dprime"] for row in metrics])),
        "mean_a": float(np.mean([row["mean_a"] for row in metrics])),
        "mean_b": float(np.mean([row["mean_b"] for row in metrics])),
        "sd_a": float(np.mean([row["sd_a"] for row in metrics])),
        "sd_b": float(np.mean([row["sd_b"] for row in metrics])),
        "separation": float(np.mean([
            row["mean_a"] - row["mean_b"] for row in metrics
        ])),
        "spread": float(np.mean([
            (row["sd_a"] + row["sd_b"]) / 2.0 for row in metrics
        ])),
        "n_a": float(np.sum([row["n_a"] for row in metrics])),
        "n_b": float(np.sum([row["n_b"] for row in metrics])),
    }


def blockwise_dprime(
    trial_features: ArrayLike,
    labels: ArrayLike,
    trial_id: ArrayLike,
    *,
    position_mask: ArrayLike | None = None,
    role_a: int = 2,
    role_b: int = 0,
    block_trials: int = 40,
    stride_trials: int | None = None,
    n_folds: int = 4,
    min_per_role: int = 4,
    require_complete_position_coverage: bool = True,
) -> dict[str, NDArray]:
    """Estimate local d-prime in fixed-width trial blocks.

    Non-overlapping blocks are the default.  A smaller stride is allowed for a
    dotted exploratory display, but overlapping points must not be treated as
    independent observations.
    """

    features = np.asarray(trial_features, dtype=np.float64)
    label = np.asarray(labels)
    trials = np.asarray(trial_id, dtype=np.float64)
    if features.ndim != 3 or label.ndim != 1 or trials.ndim != 1:
        raise ValueError("trial_features, labels, and trial_id must be 3D, 1D, and 1D")
    if len(features) != len(label) or len(features) != len(trials):
        raise ValueError("trial_features, labels, and trial_id must align")
    if not np.isfinite(trials).all():
        raise ValueError("trial_id must contain only finite values")
    if role_a == role_b:
        raise ValueError("role_a and role_b must be distinct")
    block_trials = _positive_integer("block_trials", block_trials, minimum=8)
    n_folds = _positive_integer("n_folds", n_folds, minimum=2)
    min_per_role = _positive_integer("min_per_role", min_per_role, minimum=2)
    stride = block_trials if stride_trials is None else _positive_integer(
        "stride_trials", stride_trials
    )
    order = np.argsort(trials)
    features, label, trials = features[order], label[order], trials[order]
    response = trial_responses(
        features,
        position_mask,
        require_complete_position_coverage=require_complete_position_coverage,
    )

    rows: list[dict[str, float]] = []
    starts = range(0, max(len(trials) - block_trials + 1, 0), stride)
    for start in starts:
        stop = start + block_trials
        local = crossvalidated_scores(
            response[start:stop],
            label[start:stop],
            role_a=role_a,
            role_b=role_b,
            n_folds=n_folds,
            min_per_role=min_per_role,
        )
        metrics = _summarize_fold_metrics(local)
        rows.append(
            {
                "start_trial": float(trials[start]),
                "stop_trial": float(trials[stop - 1]),
                "midpoint": float(np.mean(trials[start:stop])),
                "valid_folds": float(local["valid_folds"]),
                "required_folds": float(local["required_folds"]),
                **metrics,
            }
        )
    names = (
        "start_trial", "stop_trial", "midpoint", "dprime", "mean_a",
        "mean_b", "sd_a", "sd_b", "separation", "spread", "n_a", "n_b",
        "valid_folds", "required_folds",
    )
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in names
    }


def position_dprime_surface(
    trial_features: ArrayLike,
    labels: ArrayLike,
    trial_id: ArrayLike,
    *,
    role_a: int = 2,
    role_b: int = 0,
    block_trials: int = 40,
    n_folds: int = 4,
    min_per_role: int = 4,
    require_complete_position_coverage: bool = True,
) -> dict[str, NDArray]:
    """Estimate held-out d-prime for every time block and corridor bin."""

    features = np.asarray(trial_features, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError("trial_features must have shape (trials, position, features)")
    values = []
    midpoint = None
    for bin_index in range(features.shape[1]):
        mask = np.zeros(features.shape[1], dtype=bool)
        mask[bin_index] = True
        curve = blockwise_dprime(
            features,
            labels,
            trial_id,
            position_mask=mask,
            role_a=role_a,
            role_b=role_b,
            block_trials=block_trials,
            n_folds=n_folds,
            min_per_role=min_per_role,
            require_complete_position_coverage=require_complete_position_coverage,
        )
        midpoint = curve["midpoint"] if midpoint is None else midpoint
        values.append(curve["dprime"])
    return {
        "midpoint": np.asarray([] if midpoint is None else midpoint),
        "dprime": np.asarray(values, dtype=np.float64).T,
    }


def fit_early_slope(
    midpoint: ArrayLike,
    dprime: ArrayLike,
    *,
    early_horizon: float,
) -> dict[str, float]:
    """Fit a per-trial linear rate within the predeclared early horizon."""

    x = np.asarray(midpoint, dtype=np.float64)
    y = np.asarray(dprime, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("midpoint and dprime must be aligned one-dimensional arrays")
    if not np.isfinite(float(early_horizon)):
        raise ValueError("early_horizon must be finite")
    keep = np.isfinite(x) & np.isfinite(y) & (x <= float(early_horizon))
    if np.count_nonzero(keep) < 2 or np.ptp(x[keep]) <= 0:
        return {"slope": float("nan"), "intercept": float("nan"), "n_blocks": float(np.count_nonzero(keep))}
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    return {"slope": float(slope), "intercept": float(intercept), "n_blocks": float(np.count_nonzero(keep))}


def fit_saturation_curve(midpoint: ArrayLike, dprime: ArrayLike) -> dict[str, Any]:
    """Fit ``baseline + amplitude * (1 - exp(-k * trial))`` by grid search.

    The baseline and ``initial_rate`` are referenced to the first observed
    block midpoint, not trial zero.  The amplitude is not forced positive.
    ``plateau_observed`` requires at least 90% of the fitted amplitude to be
    attained within the data *and* a non-boundary grid optimum.  This keeps a
    half-completed rise or a grid-limited extrapolation from being described as
    an observed plateau.
    """

    x = np.asarray(midpoint, dtype=np.float64)
    y = np.asarray(dprime, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("midpoint and dprime must be aligned one-dimensional arrays")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 4 or np.ptp(x) <= 0:
        return {
            "baseline": float("nan"), "amplitude": float("nan"),
            "plateau": float("nan"), "k": float("nan"),
            "initial_rate": float("nan"), "half_time": float("nan"),
            "r2": float("nan"), "plateau_observed": False,
            "fraction_of_amplitude_observed": float("nan"),
            "grid_boundary_hit": False, "reference_trial": float("nan"),
            "x_fitted": np.array([]), "y_fitted": np.array([]),
        }
    shifted = x - np.min(x)
    span = float(np.ptp(shifted))
    k_grid = np.geomspace(0.05 / span, 20.0 / span, 300)
    best = None
    for grid_index, k in enumerate(k_grid):
        basis = np.column_stack([np.ones(len(x)), 1.0 - np.exp(-k * shifted)])
        coefficient, *_ = np.linalg.lstsq(basis, y, rcond=None)
        predicted = basis @ coefficient
        sse = float(np.sum((y - predicted) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(k), coefficient, predicted, grid_index)
    assert best is not None
    sse, rate, coefficient, predicted, grid_index = best
    baseline, amplitude = map(float, coefficient)
    total = float(np.sum((y - np.mean(y)) ** 2))
    half_time = float(np.log(2.0) / rate)
    fraction_observed = float(1.0 - np.exp(-rate * span))
    grid_boundary_hit = bool(grid_index in (0, len(k_grid) - 1))
    order = np.argsort(x)
    return {
        "baseline": baseline,
        "amplitude": amplitude,
        "plateau": baseline + amplitude,
        "k": rate,
        "initial_rate": amplitude * rate,
        "half_time": half_time,
        "r2": float("nan") if total <= 0 else 1.0 - sse / total,
        "plateau_observed": bool(fraction_observed >= 0.90 and not grid_boundary_hit),
        "fraction_of_amplitude_observed": fraction_observed,
        "grid_boundary_hit": grid_boundary_hit,
        "reference_trial": float(np.min(x)),
        "x_fitted": x[order],
        "y_fitted": predicted[order],
    }


def cross_temporal_dprime(
    trial_features: ArrayLike,
    labels: ArrayLike,
    trial_id: ArrayLike,
    *,
    position_mask: ArrayLike | None = None,
    role_a: int = 2,
    role_b: int = 0,
    block_trials: int = 40,
    min_per_role: int = 4,
    require_complete_position_coverage: bool = True,
) -> dict[str, NDArray]:
    """Train a coding axis in one block and test it in every other block.

    Off-diagonal cells use disjoint train and test trials.  Diagonal cells use
    the contiguous-fold cross-validation from :func:`crossvalidated_scores`.
    """

    features = np.asarray(trial_features, dtype=np.float64)
    label = np.asarray(labels)
    trials = np.asarray(trial_id, dtype=np.float64)
    if features.ndim != 3 or label.ndim != 1 or trials.ndim != 1:
        raise ValueError("trial_features, labels, and trial_id must be 3D, 1D, and 1D")
    if len(features) != len(label) or len(features) != len(trials):
        raise ValueError("trial_features, labels, and trial_id must align")
    if not np.isfinite(trials).all():
        raise ValueError("trial_id must contain only finite values")
    if role_a == role_b:
        raise ValueError("role_a and role_b must be distinct")
    block_trials = _positive_integer("block_trials", block_trials, minimum=8)
    min_per_role = _positive_integer("min_per_role", min_per_role, minimum=2)
    order = np.argsort(trials)
    response = trial_responses(
        features[order],
        position_mask,
        require_complete_position_coverage=require_complete_position_coverage,
    )
    label, trials = label[order], trials[order]
    blocks = [
        np.arange(start, start + block_trials)
        for start in range(0, max(len(trials) - block_trials + 1, 0), block_trials)
    ]
    matrix = np.full((len(blocks), len(blocks)), np.nan, dtype=np.float64)
    for train_id, train_idx in enumerate(blocks):
        train_values, train_labels = response[train_idx], label[train_idx]
        selected_train = np.isin(train_labels, [role_a, role_b]) & _finite_rows(train_values)
        train_values, train_labels = train_values[selected_train], train_labels[selected_train]
        if min(np.count_nonzero(train_labels == role_a), np.count_nonzero(train_labels == role_b)) < min_per_role:
            continue
        mean = np.mean(train_values, axis=0)
        scale = np.std(train_values, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
        train_z = (train_values - mean) / scale
        direction = np.mean(train_z[train_labels == role_a], axis=0) - np.mean(train_z[train_labels == role_b], axis=0)
        norm = float(np.linalg.norm(direction))
        if norm <= 0 or not np.isfinite(norm):
            continue
        direction /= norm
        for test_id, test_idx in enumerate(blocks):
            if train_id == test_id:
                local = crossvalidated_scores(
                    response[test_idx], label[test_idx], role_a=role_a,
                    role_b=role_b, n_folds=4, min_per_role=min_per_role,
                )
                matrix[train_id, test_id] = _summarize_fold_metrics(local)["dprime"]
                continue
            test_values, test_labels = response[test_idx], label[test_idx]
            selected_test = np.isin(test_labels, [role_a, role_b]) & _finite_rows(test_values)
            score = ((test_values[selected_test] - mean) / scale) @ direction
            matrix[train_id, test_id] = _dprime(score, test_labels[selected_test], role_a, role_b)["dprime"]
    midpoint = np.asarray([float(np.mean(trials[index])) for index in blocks])
    return {"midpoint": midpoint, "dprime": matrix}


def _mouse_level_observations(
    values: ArrayLike,
    groups: Sequence[str],
    *,
    mouse_ids: Sequence[str],
    rewarded_label: str = "rewarded",
    unrewarded_label: str = "unrewarded",
) -> tuple[NDArray[np.float64], NDArray, NDArray]:
    """Collapse repeated sessions to one value per mouse and validate labels."""

    value = np.asarray(values, dtype=np.float64)
    group = np.asarray(groups, dtype=object)
    mouse = np.asarray(mouse_ids, dtype=object)
    if value.ndim != 1 or group.ndim != 1 or mouse.ndim != 1:
        raise ValueError("values, groups, and mouse_ids must be one-dimensional")
    if not (len(value) == len(group) == len(mouse)):
        raise ValueError("values, groups, and mouse_ids must align")
    if rewarded_label == unrewarded_label:
        raise ValueError("rewarded_label and unrewarded_label must be distinct")
    allowed = {rewarded_label, unrewarded_label}
    unexpected = set(group.tolist()) - allowed
    if unexpected:
        raise ValueError(f"unexpected group labels: {sorted(map(str, unexpected))}")
    if any(
        not isinstance(identifier, (str, np.str_)) or str(identifier).strip() == ""
        for identifier in mouse
    ):
        raise ValueError("every observation must have a non-empty string mouse_id")

    ordered_mice = list(dict.fromkeys(mouse.tolist()))
    collapsed_values: list[float] = []
    collapsed_groups: list[str] = []
    collapsed_mice: list[Any] = []
    for identifier in ordered_mice:
        selected = mouse == identifier
        labels = set(group[selected].tolist())
        if len(labels) != 1:
            raise ValueError(f"mouse {identifier!r} appears in more than one group")
        finite = value[selected & np.isfinite(value)]
        if len(finite) == 0:
            continue
        collapsed_values.append(float(np.mean(finite)))
        collapsed_groups.append(next(iter(labels)))
        collapsed_mice.append(identifier)
    observed_groups = set(collapsed_groups)
    if observed_groups != allowed:
        raise ValueError("both rewarded and unrewarded groups need a finite mouse-level value")
    return (
        np.asarray(collapsed_values, dtype=np.float64),
        np.asarray(collapsed_groups, dtype=object),
        np.asarray(collapsed_mice, dtype=object),
    )


def exact_group_permutation(
    values: ArrayLike,
    groups: Sequence[str],
    *,
    mouse_ids: Sequence[str],
    rewarded_label: str = "rewarded",
    unrewarded_label: str = "unrewarded",
    alternative: str = "greater",
) -> dict[str, Any]:
    """Exact label permutation after reducing repeated sessions to mice."""

    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be 'greater' or 'two-sided'")
    value, group, mouse = _mouse_level_observations(
        values,
        groups,
        mouse_ids=mouse_ids,
        rewarded_label=rewarded_label,
        unrewarded_label=unrewarded_label,
    )
    rewarded = group == rewarded_label
    n_rewarded = int(np.count_nonzero(rewarded))
    observed = float(np.mean(value[rewarded]) - np.mean(value[~rewarded]))
    null = []
    all_indices = np.arange(len(value))
    for chosen in combinations(all_indices, n_rewarded):
        mask = np.zeros(len(value), dtype=bool)
        mask[list(chosen)] = True
        null.append(float(np.mean(value[mask]) - np.mean(value[~mask])))
    null_values = np.asarray(null)
    if alternative == "greater":
        pvalue = float(np.mean(null_values >= observed - 1e-12))
    else:
        pvalue = float(np.mean(np.abs(null_values) >= abs(observed) - 1e-12))
    return {
        "difference": observed,
        "pvalue": pvalue,
        "permutations": float(len(null_values)),
        "n_mice": float(len(value)),
        "mouse_ids": mouse,
        "mouse_values": value,
        "mouse_groups": group,
    }


def bootstrap_group_difference(
    values: ArrayLike,
    groups: Sequence[str],
    *,
    mouse_ids: Sequence[str],
    rewarded_label: str = "rewarded",
    unrewarded_label: str = "unrewarded",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap a mean difference by resampling unique mice within group."""

    n_boot = _positive_integer("n_boot", n_boot)
    value, group, mouse = _mouse_level_observations(
        values,
        groups,
        mouse_ids=mouse_ids,
        rewarded_label=rewarded_label,
        unrewarded_label=unrewarded_label,
    )
    rewarded = value[group == rewarded_label]
    unrewarded = value[group == unrewarded_label]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for index in range(len(samples)):
        samples[index] = (
            np.mean(rng.choice(rewarded, size=len(rewarded), replace=True))
            - np.mean(rng.choice(unrewarded, size=len(unrewarded), replace=True))
        )
    return {
        "difference": float(np.mean(rewarded) - np.mean(unrewarded)),
        "ci": tuple(np.quantile(samples, [0.025, 0.975]).tolist()),
        "samples": samples,
        "n_mice": float(len(value)),
        "mouse_ids": mouse,
        "mouse_values": value,
        "mouse_groups": group,
    }


def area_transform(
    u_components_by_neuron: ArrayLike,
    area_id: ArrayLike,
    area: str,
    *,
    n_features: int = 12,
) -> NDArray[np.float64]:
    """Factor the reconstructed-neuron distance metric for one visual area."""

    if area not in AREA_IDS:
        raise ValueError(f"area must be one of {list(AREA_IDS)}")
    n_features = _positive_integer("n_features", n_features)
    u = np.asarray(u_components_by_neuron, dtype=np.float64)
    ids = np.asarray(area_id)
    if u.ndim != 2 or u.shape[1] != len(ids):
        raise ValueError("U must have shape components x neurons aligned to iarea")
    mask = np.isin(ids, AREA_IDS[area])
    if not np.any(mask):
        raise ValueError(f"no neurons found for area {area}")
    weights = u[:, mask]
    gram = weights @ weights.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    keep = order[: min(n_features, len(order))]
    positive = np.maximum(eigenvalues[keep], 0.0)
    return eigenvectors[:, keep] * np.sqrt(positive)[None, :]


def prepare_session_trials(
    behavior: Mapping[str, Any],
    svd: Mapping[str, Any],
    retinotopy: Mapping[str, Any],
    *,
    area: str = "V1",
    n_features: int = 12,
    n_position_bins: int = 18,
    max_trailing_behavior_frames: int = 3,
    movement_rule: str = "moving_only",
    mouse_id: str | None = None,
    recording_id: str | None = None,
) -> dict[str, Any]:
    """Reconstruct one verified release session as trial x position x feature."""

    n_features = _positive_integer("n_features", n_features)
    n_position_bins = _positive_integer("n_position_bins", n_position_bins, minimum=2)
    max_trailing_behavior_frames = _positive_integer(
        "max_trailing_behavior_frames", max_trailing_behavior_frames, minimum=0
    )
    if movement_rule not in {"moving_only", "all_valid_frames"}:
        raise ValueError(
            "movement_rule must be one of ['moving_only', 'all_valid_frames']"
        )
    u = np.asarray(svd["U"], dtype=np.float64)
    v_by_frame = np.asarray(svd["V"], dtype=np.float64).T
    transform = area_transform(u, retinotopy["iarea"], area, n_features=n_features)
    moving = (
        np.asarray(behavior["ft_isMoving"])
        if "ft_isMoving" in behavior
        else np.asarray(behavior["ft_move"]) > 0
    )
    behavior_fields = {
        "position_dm": behavior["ft_Pos"],
        "trial_id": behavior["ft_trInd"],
        "is_moving": moving,
        "run_speed": behavior["ft_RunSpeed"],
    }
    v_by_frame, aligned, report = align_trailing_behavior_frames(
        v_by_frame,
        behavior_fields,
        max_trailing_behavior_frames=max_trailing_behavior_frames,
    )
    position_m = decimeters_to_meters(aligned["position_dm"])
    movement_mask = (
        np.asarray(aligned["is_moving"], dtype=bool)
        if movement_rule == "moving_only"
        else np.ones(len(position_m), dtype=bool)
    )
    valid = (
        movement_mask
        & np.isfinite(position_m)
        & np.isfinite(aligned["trial_id"])
        & (position_m >= 0.0)
        & (position_m <= 6.0)
    )
    edges_m = np.linspace(0.0, 6.0, n_position_bins + 1)
    trial_id, binned_v, frame_counts = bin_trial_features(
        v_by_frame,
        position_m,
        aligned["trial_id"],
        edges_m,
        valid_mask=valid,
    )
    speed_trial_id, speed, speed_counts = bin_trial_features(
        aligned["run_speed"],
        position_m,
        aligned["trial_id"],
        edges_m,
        valid_mask=valid,
    )
    if not np.array_equal(trial_id, speed_trial_id) or not np.array_equal(frame_counts, speed_counts):
        raise RuntimeError("neural and speed binning lost frame alignment")
    features = np.einsum("tbp,pk->tbk", binned_v, transform, optimize=True)
    wall_by_trial = np.asarray(behavior["WallName"])[trial_id]
    wall_to_role = {}
    for wall, role in zip(np.asarray(behavior["UniqWalls"]), np.asarray(behavior["stim_id"])):
        try:
            finite = bool(np.isfinite(role))
        except TypeError:
            finite = False
        if finite:
            wall_to_role[str(wall)] = int(role)
    labels = np.asarray([wall_to_role.get(str(wall), -1) for wall in wall_by_trial], dtype=np.int16)
    n_selected_trials = len(trial_id)

    def selected_trial_values(name: str) -> NDArray[np.float64]:
        values = np.asarray(behavior.get(name, np.array([])))
        output = np.full(n_selected_trials, np.nan, dtype=np.float64)
        if values.ndim == 1 and len(values) > int(np.max(trial_id)):
            try:
                output = np.asarray(values[trial_id], dtype=np.float64)
            except (TypeError, ValueError):
                pass
        return output

    if "SoundDelPos" in behavior:
        cue_position_dm = selected_trial_values("SoundDelPos")
        corridor_length = float(behavior.get("Corridor_Length", 60.0))
        cue_position_dm = np.mod(cue_position_dm, corridor_length)
    else:
        cue_position_dm = selected_trial_values("SoundPos")
    reward_position_dm = selected_trial_values("RewPos")
    rewarded_trial = selected_trial_values("isRew")
    first_lick_dm = np.full(n_selected_trials, np.nan, dtype=np.float64)
    if "LickPos" in behavior and "LickTrind" in behavior:
        lick_position = np.asarray(behavior["LickPos"], dtype=np.float64)
        lick_trial = np.asarray(behavior["LickTrind"])
        for offset, current_trial in enumerate(trial_id):
            current = lick_position[lick_trial == current_trial]
            if len(current):
                first_lick_dm[offset] = float(current[0])
    centers_m = (edges_m[:-1] + edges_m[1:]) / 2.0
    return {
        "trial_features": features.astype(np.float32),
        "run_speed": speed[:, :, 0].astype(np.float32),
        "frame_counts": frame_counts,
        "trial_id": trial_id,
        "labels": labels,
        "wall_name": wall_by_trial,
        "position_edges_m": edges_m,
        "position_centers_m": centers_m,
        "texture_mask": edges_m[1:] <= 4.0,
        "gray_mask": edges_m[:-1] >= 4.0,
        "area": area,
        "movement_rule": movement_rule,
        "n_features": int(features.shape[-1]),
        "alignment": report,
        "reward_mode": str(behavior.get("Reward_Mode", "unknown")),
        "rewarded_trial": np.isfinite(rewarded_trial) & (rewarded_trial > 0),
        "cue_position_m": cue_position_dm / 10.0,
        "reward_position_m": reward_position_dm / 10.0,
        "first_lick_position_m": first_lick_dm / 10.0,
        "mouse_id": None if mouse_id is None else str(mouse_id),
        "recording_id": None if recording_id is None else str(recording_id),
    }


def simulate_mouse_curves(
    *,
    rewarded_mice: int = 4,
    unrewarded_mice: int = 9,
    rewarded_rate: float = 0.018,
    unrewarded_rate: float = 0.008,
    rewarded_plateau: float = 1.25,
    unrewarded_plateau: float = 1.05,
    baseline: float = 0.15,
    noise: float = 0.10,
    block_trials: int = 40,
    n_blocks: int = 8,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Create clearly synthetic mouse curves for the notebook design sandbox."""

    rewarded_mice = _positive_integer("rewarded_mice", rewarded_mice)
    unrewarded_mice = _positive_integer("unrewarded_mice", unrewarded_mice)
    block_trials = _positive_integer("block_trials", block_trials, minimum=8)
    n_blocks = _positive_integer("n_blocks", n_blocks, minimum=2)
    if not np.isfinite(float(noise)) or float(noise) < 0:
        raise ValueError("noise must be finite and non-negative")
    rng = np.random.default_rng(seed)
    midpoint = (np.arange(n_blocks) + 0.5) * block_trials
    rows = []
    for group, count, rate, plateau in (
        ("rewarded", rewarded_mice, float(rewarded_rate), float(rewarded_plateau)),
        ("unrewarded", unrewarded_mice, float(unrewarded_rate), float(unrewarded_plateau)),
    ):
        amplitude = plateau - baseline
        k = rate / amplitude
        for mouse_index in range(count):
            mouse_baseline = baseline + rng.normal(0, noise * 0.35)
            mouse_amplitude = amplitude * rng.normal(1.0, 0.12)
            mouse_k = max(k * rng.normal(1.0, 0.15), 1e-6)
            dprime = mouse_baseline + mouse_amplitude * (1.0 - np.exp(-mouse_k * midpoint))
            dprime += rng.normal(0.0, noise, size=len(midpoint))
            rows.append(
                {
                    "mouse": f"{group[:1].upper()}{mouse_index + 1:02d}",
                    "group": group,
                    "midpoint": midpoint.astype(np.float64),
                    "dprime": dprime.astype(np.float64),
                }
            )
    return rows


__all__ = [
    "AREA_IDS",
    "area_transform",
    "blockwise_dprime",
    "bootstrap_group_difference",
    "cross_temporal_dprime",
    "crossvalidated_scores",
    "exact_group_permutation",
    "fit_early_slope",
    "fit_saturation_curve",
    "position_dprime_surface",
    "prepare_session_trials",
    "simulate_mouse_curves",
    "svd_dprime",
    "svd_dprime_contrasts",
    "trial_responses",
]
