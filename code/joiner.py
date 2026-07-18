from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from database import ZhongDB
from position import AlignmentReport, align_trailing_behavior_frames


AREA_GROUPS = {
    0: "mHV",
    1: "mHV",
    2: "mHV",
    3: "aHV",
    4: "aHV",
    5: "lHV",
    6: "lHV",
    8: "V1",
    9: "mHV",
}


class Joiner:
    def __init__(
        self,
        database: ZhongDB,
        recording_id: str,
        *,
        experiment: str | None = None,
        max_gib: float = 10.0,
        max_trailing_behavior_frames: int = 3,
    ) -> None:
        self.recording_id = recording_id
        self.experiment = experiment

        behavior = database.load(
            recording_id,
            "behavior",
            experiment=experiment,
            max_gib=max_gib,
        )
        neural = database.load(recording_id, "reduced_neural", max_gib=max_gib)
        retinotopy = database.load(recording_id, "retinotopy", max_gib=max_gib)

        self.U, self.V = neural_factors(neural)
        behavior_frames, self.alignment = aligned_behavior(
            behavior,
            self.V.shape[1],
            max_trailing_behavior_frames,
        )
        self.trials = trial_table(behavior, recording_id)
        self.stimuli = stimulus_table(behavior, recording_id)
        self.frames = frame_table(
            behavior_frames,
            self.trials,
            self.stimuli,
            recording_id,
            experiment,
        )
        self.neurons = neuron_table(retinotopy, self.U.shape[1], recording_id)

        self._connection = duckdb.connect(":memory:")
        self._tables: dict[str, pd.DataFrame] = {}
        self.register("frames", self.frames)
        self.register("neurons", self.neurons)
        self.register("trials", self.trials)
        self.register("stimuli", self.stimuli)

    @property
    def tables(self) -> tuple[str, ...]:
        return tuple(sorted(self._tables))

    def register(self, name: str, frame: pd.DataFrame) -> pd.DataFrame:
        if not name.isidentifier():
            raise ValueError("Table name must be a Python and SQL identifier")
        if name in self._tables:
            self._connection.unregister(name)
        stored = frame.copy()
        self._tables[name] = stored
        self._connection.register(name, stored)
        return stored

    def query(
        self,
        statement: str,
        parameters: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        relation = (
            self._connection.execute(statement, parameters)
            if parameters is not None
            else self._connection.execute(statement)
        )
        return relation.fetchdf()

    def join(
        self,
        frames: pd.DataFrame | pd.Series | Iterable[int] | None = None,
        neurons: pd.DataFrame | pd.Series | Iterable[int] | None = None,
        *,
        max_cells: int = 1_000_000,
    ) -> pd.DataFrame:
        frame_ids = selected_ids(frames, "frame_id", len(self.frames))
        neuron_ids = selected_ids(neurons, "neuron_id", len(self.neurons))
        cell_count = len(frame_ids) * len(neuron_ids)

        if cell_count > max_cells:
            raise ValueError(
                f"The selection contains {cell_count:,} neuron-frame pairs; "
                f"raise max_cells or select fewer than {max_cells:,}"
            )

        values = self.U[:, neuron_ids].T @ self.V[:, frame_ids]
        activity = pd.DataFrame(
            {
                "neuron_id": np.repeat(neuron_ids, len(frame_ids)),
                "frame_id": np.tile(frame_ids, len(neuron_ids)),
                "activity": values.reshape(-1),
            }
        )
        activity = activity.merge(
            self.neurons,
            on="neuron_id",
            how="left",
            validate="many_to_one",
        )
        activity = activity.merge(
            self.frames,
            on=["recording_id", "frame_id"],
            how="left",
            validate="many_to_one",
        )
        return self.register("activity", activity)

    def schema(self, table: str | None = None) -> pd.DataFrame:
        if table is None:
            return pd.DataFrame(
                [
                    {"table": name, "rows": len(frame), "columns": len(frame.columns)}
                    for name, frame in sorted(self._tables.items())
                ]
            )
        if table not in self._tables:
            raise KeyError(f"Unknown table {table!r}; choose from {self.tables}")
        return self.query(f'DESCRIBE SELECT * FROM "{table}"')

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "Joiner":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Joiner({self.recording_id!r}; {len(self.neurons):,} neurons; "
            f"{len(self.frames):,} frames; {self.U.shape[0]} components)"
        )


def neural_factors(neural: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(neural, Mapping) or not {"U", "V"}.issubset(neural):
        raise ValueError("Reduced neural data must contain U and V")

    U = np.asarray(neural["U"])
    V = np.asarray(neural["V"])
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must both be two-dimensional")
    if U.shape[0] != V.shape[0]:
        raise ValueError(f"U and V component counts disagree: {U.shape}, {V.shape}")
    return U, V


def aligned_behavior(
    behavior: Any,
    neural_frames: int,
    max_trailing_behavior_frames: int,
) -> tuple[dict[str, np.ndarray], AlignmentReport]:
    if not isinstance(behavior, Mapping):
        raise ValueError("Behavior data must be a mapping")

    names = [
        "ft_trInd",
        "ft_Pos",
        "ft_isMoving" if "ft_isMoving" in behavior else "ft_move",
        "ft_RunSpeed",
        "ft_WallID",
        "ft_CorrSpc",
    ]
    if "ft" in behavior:
        names.append("ft")
    missing = [name for name in names if name not in behavior]
    if missing:
        raise ValueError(f"Behavior data is missing {missing}")

    arrays = {name: vector(behavior[name], name) for name in names}
    _, aligned, report = align_trailing_behavior_frames(
        np.empty((neural_frames, 0)),
        arrays,
        max_trailing_behavior_frames=max_trailing_behavior_frames,
    )
    return aligned, report


def frame_table(
    behavior: Mapping[str, np.ndarray],
    trials: pd.DataFrame,
    stimuli: pd.DataFrame,
    recording_id: str,
    experiment: str | None,
) -> pd.DataFrame:
    frame_count = len(behavior["ft_Pos"])
    position_dm = behavior["ft_Pos"].astype(float)
    raw_trial = behavior["ft_trInd"].astype(float)
    rounded_trial = np.rint(raw_trial)
    integer_trial = np.isfinite(raw_trial) & np.isclose(raw_trial, rounded_trial)
    valid_trial = integer_trial & (rounded_trial >= 0) & (rounded_trial < len(trials))
    trial_id = pd.array(
        np.where(valid_trial, rounded_trial, np.nan),
        dtype="Int64",
    )

    movement_name = "ft_isMoving" if "ft_isMoving" in behavior else "ft_move"

    columns: dict[str, Any] = {
        "recording_id": recording_id,
        "experiment": experiment,
        "frame_id": np.arange(frame_count, dtype=np.int64),
        "trial_id": trial_id,
        "valid_trial": valid_trial,
        "position_dm": position_dm,
        "position_m": position_dm / 10.0,
        "is_moving": behavior[movement_name] > 0,
        "run_speed": behavior["ft_RunSpeed"].astype(float),
        "wall_at_frame": behavior["ft_WallID"].astype(str),
        "in_texture": behavior["ft_CorrSpc"].astype(bool),
    }
    if "ft" in behavior:
        columns["time"] = behavior["ft"]

    frames = pd.DataFrame(columns)
    frames = frames.merge(
        trials,
        on=["recording_id", "trial_id"],
        how="left",
        validate="many_to_one",
    )
    return frames.merge(
        stimuli,
        on=["recording_id", "wall_name"],
        how="left",
        validate="many_to_one",
    )


def trial_table(behavior: Mapping[str, Any], recording_id: str) -> pd.DataFrame:
    if "WallName" not in behavior:
        raise ValueError("Behavior data is missing WallName")
    walls = vector(behavior["WallName"], "WallName")
    return pd.DataFrame(
        {
            "recording_id": recording_id,
            "trial_id": np.arange(len(walls), dtype=np.int64),
            "wall_name": walls.astype(str),
        }
    )


def stimulus_table(behavior: Mapping[str, Any], recording_id: str) -> pd.DataFrame:
    missing = [name for name in ["UniqWalls", "stim_id"] if name not in behavior]
    if missing:
        raise ValueError(f"Behavior data is missing {missing}")
    walls = vector(behavior["UniqWalls"], "UniqWalls")
    roles = vector(behavior["stim_id"], "stim_id")
    if len(walls) != len(roles):
        raise ValueError("UniqWalls and stim_id must have the same length")
    return pd.DataFrame(
        {
            "recording_id": recording_id,
            "wall_name": walls.astype(str),
            "stimulus_role": pd.to_numeric(
                pd.Series(roles),
                errors="coerce",
            ).astype("Int64"),
        }
    )


def neuron_table(
    retinotopy: Any,
    neuron_count: int,
    recording_id: str,
) -> pd.DataFrame:
    if not isinstance(retinotopy, Mapping):
        raise ValueError("Retinotopy data must be a mapping")
    areas = vector(retinotopy.get("iarea", []), "iarea").astype(np.int64)
    coordinates = np.asarray(retinotopy.get("xy_t", []))
    if len(areas) != neuron_count or coordinates.shape != (neuron_count, 2):
        raise ValueError(
            "Retinotopy must provide one iarea value and one xy_t pair per neuron"
        )
    return pd.DataFrame(
        {
            "recording_id": recording_id,
            "neuron_id": np.arange(neuron_count, dtype=np.int64),
            "area_id": areas,
            "area_group": [AREA_GROUPS.get(int(area), "excluded") for area in areas],
            "cortical_x": -coordinates[:, 1],
            "cortical_y": coordinates[:, 0],
        }
    )


def selected_ids(
    selection: pd.DataFrame | pd.Series | Iterable[int] | None,
    column: str,
    size: int,
) -> np.ndarray:
    if selection is None:
        values = np.arange(size)
    elif isinstance(selection, pd.DataFrame):
        if column not in selection:
            raise KeyError(f"Selection must contain {column!r}")
        values = selection[column].to_numpy()
    elif isinstance(selection, pd.Series):
        values = selection.to_numpy()
    else:
        values = np.asarray(list(selection))

    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{column} values must be a one-dimensional number sequence")
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"{column} values must be finite integers")

    ids = pd.unique(values.astype(np.int64))
    if np.any(ids < 0) or np.any(ids >= size):
        raise IndexError(f"{column} values must be between 0 and {size - 1}")
    return np.asarray(ids, dtype=np.int64)


def vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


__all__ = ["Joiner"]
