"""Build the compact real-data example used by the data-atlas notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .position import (
    align_trailing_behavior_frames,
    bin_trial_features,
    decimeters_to_meters,
)

AREA_IDS = {
    "V1": (8,),
    "mHV": (0, 1, 2, 9),
    "lHV": (5, 6),
    "aHV": (3, 4),
}

TX119_SOURCE_SPECS = {
    "behavior": {
        "size_bytes": 270_497_670,
        "sha256": "0111710701d5d71740743d2402b08ee36332af2c511f34d011a4bef4c0ac9edd",
    },
    "svd": {
        "size_bytes": 66_967_262,
        "sha256": "e0e4b78272b05398e451cc79c75aeb778564b72c051c674efc13e66bae4fa47a",
    },
    "retinotopy": {
        "size_bytes": 983_358,
        "sha256": "767de68b4c8928ec244f4c3133c42311622bd80a6e57d0f36462c445b3ddbbed",
    },
}


def _verify_source(path: str | Path, spec: dict[str, int | str]) -> Path:
    """Verify an exact trusted source before any pickle-enabled load."""

    source = Path(path)
    expected_size = int(spec["size_bytes"])
    if source.stat().st_size != expected_size:
        raise ValueError(
            f"source size mismatch for {source}: expected {expected_size:,} bytes"
        )
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"source SHA-256 mismatch for {source}")
    return source


def _area_transform(
    u_components_by_neuron: NDArray[np.float32],
    area_id: NDArray[np.int32],
    selected_ids: tuple[int, ...],
    n_features: int,
) -> NDArray[np.float64]:
    """Factor the area-specific reconstructed-neuron distance metric."""

    mask = np.isin(area_id, selected_ids)
    if not np.any(mask):
        raise ValueError(f"no neurons found for area ids {selected_ids}")
    weights = np.asarray(u_components_by_neuron[:, mask], dtype=np.float64)
    gram = weights @ weights.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    keep = order[:n_features]
    positive = np.maximum(eigenvalues[keep], 0.0)
    return eigenvectors[:, keep] * np.sqrt(positive)[None, :]


def build_atlas_demo(
    behavior_path: str | Path,
    svd_path: str | Path,
    retinotopy_path: str | Path,
    *,
    session_name: str = "TX119_2023_12_24_1",
    n_bins: int = 18,
    n_population_features: int = 48,
    n_area_features: int = 12,
) -> dict[str, Any]:
    """Create a small trial-by-position example from checksum-verified sources.

    The source `.npy` files contain trusted pickled Python objects.  Only use
    files downloaded from the pinned manifest and verified by `download_profile`.
    """

    behavior_path = _verify_source(behavior_path, TX119_SOURCE_SPECS["behavior"])
    svd_path = _verify_source(svd_path, TX119_SOURCE_SPECS["svd"])
    retinotopy_path = _verify_source(
        retinotopy_path, TX119_SOURCE_SPECS["retinotopy"]
    )
    behavior_all = np.load(behavior_path, allow_pickle=True).item()
    if session_name not in behavior_all:
        raise KeyError(f"session {session_name!r} is not in {behavior_path}")
    behavior = behavior_all[session_name]
    svd = np.load(svd_path, allow_pickle=True).item()
    u = np.asarray(svd["U"], dtype=np.float32)
    v_by_frame = np.asarray(svd["V"], dtype=np.float32).T
    with np.load(retinotopy_path, allow_pickle=False) as retinotopy:
        area_id = np.asarray(retinotopy["iarea"], dtype=np.int32)
    if u.shape[1] != len(area_id):
        raise ValueError("U neuron axis and retinotopy iarea length disagree")
    if n_population_features > v_by_frame.shape[1]:
        raise ValueError("requested more population features than published PCs")

    behavior_fields = {
        "position_dm": behavior["ft_Pos"],
        "trial_id": behavior["ft_trInd"],
        "is_moving": behavior["ft_isMoving"],
        "run_speed": behavior["ft_RunSpeed"],
    }
    v_by_frame, aligned, report = align_trailing_behavior_frames(
        v_by_frame,
        behavior_fields,
        max_trailing_behavior_frames=3,
    )
    position_m = decimeters_to_meters(aligned["position_dm"])
    valid = (
        aligned["is_moving"].astype(bool)
        & np.isfinite(position_m)
        & np.isfinite(aligned["trial_id"])
        & (position_m >= 0.0)
        & (position_m <= 6.0)
    )
    edges_m = np.linspace(0.0, 6.0, n_bins + 1, dtype=np.float64)
    trial_ids, binned_v, frame_counts = bin_trial_features(
        v_by_frame,
        position_m,
        aligned["trial_id"],
        edges_m,
        valid_mask=valid,
    )
    speed_trials, mean_speed, speed_counts = bin_trial_features(
        aligned["run_speed"],
        position_m,
        aligned["trial_id"],
        edges_m,
        valid_mask=valid,
    )
    if not np.array_equal(trial_ids, speed_trials) or not np.array_equal(
        frame_counts, speed_counts
    ):
        raise RuntimeError("neural and speed binning lost frame alignment")
    if np.any(frame_counts == 0):
        raise ValueError("chosen session/binning unexpectedly contains empty bins")

    wall_by_trial = np.asarray(behavior["WallName"])[trial_ids]
    unexpected_walls = set(wall_by_trial) - {"rock1", "rock2", "wood1", "wood2"}
    if unexpected_walls:
        raise ValueError(f"unexpected WallName values: {sorted(unexpected_walls)}")
    release_walls = np.asarray(behavior["UniqWalls"])
    release_ids = np.asarray(behavior["stim_id"])
    wall_to_id = {str(wall): int(stimulus) for wall, stimulus in zip(release_walls, release_ids)}
    stimulus_id = np.array([wall_to_id[str(wall)] for wall in wall_by_trial], dtype=np.int8)
    family = np.array(
        [0 if str(wall).startswith("rock") else 1 for wall in wall_by_trial],
        dtype=np.int8,
    )
    exemplar = np.array([int(str(wall)[-1]) for wall in wall_by_trial], dtype=np.int8)

    area_names = np.array(list(AREA_IDS), dtype="U3")
    area_features = np.empty(
        (len(area_names), len(trial_ids), n_bins, n_area_features),
        dtype=np.float32,
    )
    for area_offset, area_name in enumerate(area_names):
        transform = _area_transform(
            u, area_id, AREA_IDS[str(area_name)], n_area_features
        )
        area_features[area_offset] = np.einsum(
            "tbp,pk->tbk", binned_v, transform, optimize=True
        ).astype(np.float32)

    metadata = {
        "schema_version": 1,
        "source": "Zhong et al. (2025) Figshare v2",
        "doi": "10.25378/janelia.28811129.v2",
        "license": "CC BY 4.0",
        "session": session_name,
        "condition": "unsup_test1 (unsupervised cohort, test 1)",
        "source_file_ids": [54183911, 54866057, 54184070],
        "source_sha256": {
            name: str(spec["sha256"]) for name, spec in TX119_SOURCE_SPECS.items()
        },
        "source_behavior_frames": report.behavior_frames,
        "source_neural_frames": report.neural_frames,
        "dropped_trailing_behavior_frames": report.dropped_trailing_behavior_frames,
        "position_units_in_release": "decimetres",
        "corridor_m": 6.0,
        "texture_region_m": [0.0, 4.0],
        "gray_region_m": [4.0, 6.0],
        "binning": "mean of moving frames within trial and fixed position bin; no interpolation",
        "population_representation": f"first {n_population_features} published session PCs",
        "area_representation": (
            f"top {n_area_features} factors of each area's reconstructed-neuron "
            "Euclidean metric"
        ),
        "label_note": (
            "Raw WallName values are preserved. The release says wood1/wood2; "
            "the paper describes this texture family as brick."
        ),
        "single_session_warning": (
            "Use this derivative for pipeline checks and exploration only; mice or "
            "sessions, not trials, are the inferential unit for group claims."
        ),
        "svd_warning": (
            "The published session SVD basis was fit before trial splitting. It is "
            "label-free but transductive."
        ),
    }
    return {
        "population_features": binned_v[:, :, :n_population_features].astype(np.float32),
        "area_features": area_features,
        "mean_run_speed": mean_speed[:, :, 0].astype(np.float32),
        "frame_counts": frame_counts,
        "trial_id": trial_ids.astype(np.int16),
        "wall_name": wall_by_trial.astype("U5"),
        "stimulus_id": stimulus_id,
        "texture_family": family,
        "exemplar": exemplar,
        "area_name": area_names,
        "position_edges_m": edges_m.astype(np.float32),
        "position_centers_m": ((edges_m[:-1] + edges_m[1:]) / 2).astype(np.float32),
        "texture_bin_mask": (edges_m[1:] <= 4.0).astype(bool),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }


def save_atlas_demo(data: dict[str, Any], output_path: str | Path) -> Path:
    """Write a pickle-free, compressed archive atomically."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(output)
    return output
