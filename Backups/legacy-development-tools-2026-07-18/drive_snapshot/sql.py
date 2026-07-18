"""Flat Pandas and DuckDB access to the Zhong et al. (2025) Drive release.

``drive.py`` remains the filesystem boundary: it validates the mounted release,
checks files before opening them, and owns the cache.  This module deliberately
does not expose ``Dataset``, ``Recording``, ``Experiment``, or ``DataFile``
objects to an analysis.  It projects their metadata into ordinary DataFrames and
registers those frames as DuckDB tables instead::

    import sql

    db = sql.setup()
    db.query(
        "SELECT mouse, recording_id, experiment "
        "FROM memberships WHERE experiment LIKE '%train1%' "
        "ORDER BY mouse, recording_id"
    )

Arrays stay lazy.  Query metadata first, then explicitly load one selected
recording layer with ``db.load(...)``.
"""

from __future__ import annotations

import json
from pathlib import Path
import pprint
import re
from typing import Any, Iterable, Mapping

import duckdb
import numpy as np
import pandas as pd

import drive
from zhong2025.atlas import load_experiment_index


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LAYER_FOR_CATEGORY = {
    "imaging_behavior": "behavior",
    "reduced_neural": "reduced_neural",
    "full_neural": "full_neural",
    "retinotopy": "retinotopy",
}
_COHORT_ORDER = ("supervised", "unsupervised", "grating", "naive")


def _plain(value: Any) -> Any:
    """Convert NumPy containers/scalars into JSON-safe Python values."""

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _plain(value.item())
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(_plain(value)).strip()
    return text or None


def _block_text(value: Any) -> str:
    plain = _plain(value)
    if isinstance(plain, float) and plain.is_integer():
        return str(int(plain))
    return str(plain).strip()


def _cohort(experiment: str) -> str:
    if experiment.startswith("sup_"):
        return "supervised"
    if experiment.startswith("unsup_"):
        return "unsupervised"
    if experiment.startswith("naive_"):
        return "naive"
    if experiment.endswith("_grating"):
        return "grating"
    return "other"


def _stage(experiment: str) -> str | None:
    for token in ("train1", "test1", "train2", "test2", "test3"):
        if token in experiment:
            return token
    return None


def _moment(experiment: str) -> str:
    if "_before_" in experiment:
        return "before"
    if "_after_" in experiment:
        return "after"
    return "snapshot"


def _index_rows_from_projection() -> list[dict[str, Any]]:
    index = load_experiment_index()
    rows: list[dict[str, Any]] = []
    for experiment, entries in index["experiments"].items():
        for source_index, entry in enumerate(entries):
            rows.append(
                _experiment_row(
                    str(experiment),
                    _plain(entry["source"]),
                    source_index=source_index,
                    recording_id=str(entry["recording_id"]),
                    retinotopy_id=str(entry["retinotopy_id"]),
                )
            )
    return rows


def _index_rows_from_drive(source: drive.Dataset) -> list[dict[str, Any]]:
    raw = _plain(source.load("Imaging_Exp_info.npy", max_gib=1.0))
    if not isinstance(raw, Mapping):
        raise drive.DriveDataError("Imaging_Exp_info.npy should contain a dictionary")

    rows: list[dict[str, Any]] = []
    for experiment, entries in raw.items():
        plain_entries = _plain(entries)
        if isinstance(plain_entries, Mapping):
            plain_entries = [plain_entries]
        if not isinstance(plain_entries, list):
            raise drive.DriveDataError(
                f"Imaging_Exp_info.npy[{experiment!r}] should contain rows"
            )
        for source_index, entry in enumerate(plain_entries):
            if not isinstance(entry, Mapping):
                raise drive.DriveDataError(
                    f"Imaging_Exp_info.npy[{experiment!r}][{source_index}] is not a row"
                )
            rows.append(
                _experiment_row(
                    str(experiment), _plain(entry), source_index=source_index
                )
            )
    return rows


def _experiment_row(
    experiment: str,
    source: Mapping[str, Any],
    *,
    source_index: int,
    recording_id: str | None = None,
    retinotopy_id: str | None = None,
) -> dict[str, Any]:
    mouse = str(source["mname"])
    date = str(source["datexp"])
    block = _block_text(source["blk"])
    recording_id = recording_id or f"{mouse}_{date}_{block}"
    retinotopy_id = retinotopy_id or f"{mouse}_{date}"
    return {
        "experiment": experiment,
        "source_row": int(source_index),
        "recording_id": recording_id,
        "retinotopy_id": retinotopy_id,
        "mouse": mouse,
        "date": date,
        "block": block,
        "session_number": _plain(source.get("sess#")),
        "is_2p": bool(source.get("is2p")) if source.get("is2p") is not None else None,
        "gender": _optional_text(source.get("Gender")),
        "reward_type": _optional_text(source.get("rewType")),
        "stimulus": _optional_text(source.get("stim")),
        "stimulus_type": _optional_text(source.get("stimtype")),
        "depth_json": json.dumps(_plain(source.get("depth", []))),
        "stimulus_ids_json": json.dumps(_plain(source.get("stim_id", []))),
        "note": _optional_text(source.get("Note")),
        "source_json": json.dumps(_plain(source), sort_keys=True),
    }


def _files_frame(source: drive.Dataset) -> pd.DataFrame:
    rows = [
        {
            "filename": item.name,
            "figshare_file_id": int(item.id),
            "category": item.category,
            "experiment": item.experiment,
            "recording_id": item.recording_id,
            "retinotopy_id": item.retinotopy_id,
            "size_bytes": int(item.size_bytes),
            "size_mib": item.size_bytes / 2**20,
            "size_gib": item.size_bytes / 2**30,
            "md5": item.md5,
            "relative_path": item.relative_path,
        }
        for item in source.files
    ]
    return pd.DataFrame(rows).sort_values("filename", ignore_index=True)


def _experiment_frames(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiment_rows = pd.DataFrame(rows).sort_values(
        ["experiment", "recording_id", "source_row"], ignore_index=True
    )
    keys = ["experiment", "recording_id", "retinotopy_id", "mouse", "date", "block"]
    memberships = (
        experiment_rows.groupby(keys, dropna=False, sort=True)
        .size()
        .rename("source_row_count")
        .reset_index()
    )
    memberships.insert(1, "cohort", memberships["experiment"].map(_cohort))

    experiments = (
        memberships.groupby("experiment", sort=True)
        .agg(recording_count=("recording_id", "nunique"), mouse_count=("mouse", "nunique"))
        .reset_index()
    )
    experiments.insert(1, "cohort", experiments["experiment"].map(_cohort))
    experiments.insert(2, "stage", experiments["experiment"].map(_stage))
    experiments.insert(3, "moment", experiments["experiment"].map(_moment))
    return experiment_rows, memberships, experiments


def _recording_files_frame(
    files: pd.DataFrame, memberships: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "recording_id",
        "experiment",
        "layer",
        "filename",
        "figshare_file_id",
        "category",
        "size_bytes",
        "size_mib",
        "size_gib",
        "md5",
        "relative_path",
    ]
    parts: list[pd.DataFrame] = []

    direct = files[files["category"].isin(("reduced_neural", "full_neural"))].copy()
    direct["layer"] = direct["category"].map(_LAYER_FOR_CATEGORY)
    direct["experiment"] = None
    parts.append(direct[columns])

    behavior = memberships[["experiment", "recording_id"]].merge(
        files[files["category"] == "imaging_behavior"],
        on="experiment",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_file"),
    )
    behavior["layer"] = "behavior"
    parts.append(behavior[columns])

    recording_keys = memberships[["recording_id", "retinotopy_id"]].drop_duplicates()
    retinotopy = recording_keys.merge(
        files[files["category"] == "retinotopy"],
        on="retinotopy_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_file"),
    )
    retinotopy["layer"] = "retinotopy"
    retinotopy["experiment"] = None
    parts.append(retinotopy[columns])

    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(["recording_id", "experiment", "layer", "filename"])
        .sort_values(["recording_id", "layer", "experiment"], na_position="first")
        .reset_index(drop=True)
    )


def _recordings_frame(
    memberships: pd.DataFrame, recording_files: pd.DataFrame
) -> pd.DataFrame:
    identity = memberships[
        ["recording_id", "retinotopy_id", "mouse", "date", "block"]
    ].drop_duplicates()
    experiment_lists = (
        memberships.groupby("recording_id")["experiment"]
        .agg(lambda values: json.dumps(sorted(set(values))))
        .rename("experiments_json")
    )
    experiment_counts = memberships.groupby("recording_id")["experiment"].nunique()
    layers = (
        recording_files.assign(present=True)
        .pivot_table(
            index="recording_id", columns="layer", values="present", aggfunc="any", fill_value=False
        )
        .rename_axis(columns=None)
    )
    totals = recording_files.groupby("recording_id").agg(
        linked_file_count=("filename", "nunique"), linked_bytes=("size_bytes", "sum")
    )
    result = identity.set_index("recording_id")
    result["experiment_count"] = experiment_counts
    result["experiments_json"] = experiment_lists
    for layer in _LAYER_FOR_CATEGORY.values():
        result[f"has_{layer}"] = layers[layer] if layer in layers else False
    result = result.join(totals).reset_index()
    result["linked_gib"] = result["linked_bytes"] / 2**30
    return result.sort_values(["mouse", "date", "block"], ignore_index=True)


def _mice_frame(memberships: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    pairs = memberships[["mouse", "cohort"]].drop_duplicates()

    def primary(values: pd.Series) -> str:
        present = set(values)
        return next((name for name in _COHORT_ORDER if name in present), "other")

    cohorts = pairs.groupby("mouse")["cohort"].agg(
        primary_cohort=primary,
        cohorts_json=lambda values: json.dumps(sorted(set(values))),
    )
    summary = recordings.groupby("mouse").agg(
        recording_count=("recording_id", "nunique"),
        has_full_neural=("has_full_neural", "any"),
        has_reduced_neural=("has_reduced_neural", "any"),
    )
    return cohorts.join(summary).reset_index().sort_values("mouse", ignore_index=True)


def _release_frame(source: drive.Dataset) -> pd.DataFrame:
    row = {str(key): _plain(value) for key, value in source.release.items()}
    row["connected"] = source.connected
    row["root"] = None if source.root is None else str(source.root)
    row["cache"] = str(source.cache)
    return pd.DataFrame([row])


def _build_tables(source: drive.Dataset) -> dict[str, pd.DataFrame]:
    index_rows = (
        _index_rows_from_drive(source) if source.connected else _index_rows_from_projection()
    )
    files = _files_frame(source)
    experiment_rows, memberships, experiments = _experiment_frames(index_rows)
    recording_files = _recording_files_frame(files, memberships)
    recordings = _recordings_frame(memberships, recording_files)
    mice = _mice_frame(memberships, recordings)
    return {
        "release": _release_frame(source),
        "files": files,
        "experiment_rows": experiment_rows,
        "memberships": memberships,
        "experiments": experiments,
        "recording_files": recording_files,
        "recordings": recordings,
        "mice": mice,
    }


class SQLData:
    """A small relational facade over ordinary DataFrames and one Drive handle."""

    def __init__(self, source: drive.Dataset) -> None:
        self._source = source
        self._connection = duckdb.connect(database=":memory:")
        self._tables = _build_tables(source)
        for name, frame in self._tables.items():
            self._connection.register(name, frame)

    @property
    def source(self) -> drive.Dataset:
        """The low-level, safe Drive handle used only for explicit file loading."""

        return self._source

    @property
    def tables(self) -> tuple[str, ...]:
        return tuple(sorted(self._tables))

    def table(self, name: str) -> pd.DataFrame:
        """Return a copy of one registered Pandas table."""

        if name not in self._tables:
            raise KeyError(f"unknown table {name!r}; choose from {self.tables}")
        return self._tables[name].copy()

    def register(self, name: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Register an analysis DataFrame so later SQL can join against it."""

        if not _IDENTIFIER.fullmatch(name):
            raise ValueError("table name must be a simple SQL identifier")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas.DataFrame")
        stored = frame.copy()
        self._tables[name] = stored
        self._connection.register(name, stored)
        return stored

    def query(
        self, statement: str, parameters: Iterable[Any] | Mapping[str, Any] | None = None
    ) -> pd.DataFrame:
        """Execute DuckDB SQL and return the result as an ordinary DataFrame."""

        cursor = self._connection.execute(statement, parameters) if parameters is not None else self._connection.execute(statement)
        return cursor.fetchdf()

    def schema(self, table: str | None = None) -> pd.DataFrame:
        """Describe every table, or the columns of one table."""

        if table is None:
            rows = [
                {"table": name, "rows": len(frame), "columns": len(frame.columns)}
                for name, frame in sorted(self._tables.items())
            ]
            return pd.DataFrame(rows)
        if table not in self._tables:
            raise KeyError(f"unknown table {table!r}; choose from {self.tables}")
        return self.query(f'DESCRIBE SELECT * FROM "{table}"')

    def _one_file(
        self, recording_id: str, layer: str, experiment: str | None
    ) -> Mapping[str, Any]:
        normalized = "reduced_neural" if layer == "svd" else layer
        if normalized not in set(_LAYER_FOR_CATEGORY.values()):
            raise ValueError(
                "layer must be behavior, reduced_neural (or svd), full_neural, or retinotopy"
            )
        matches = self._tables["recording_files"]
        matches = matches[
            (matches["recording_id"] == recording_id) & (matches["layer"] == normalized)
        ]
        if normalized == "behavior" and experiment is not None:
            matches = matches[matches["experiment"] == experiment]
        if matches.empty:
            detail = f" and experiment {experiment!r}" if experiment else ""
            raise drive.DriveDataError(
                f"No {normalized!r} file for recording {recording_id!r}{detail}"
            )
        if len(matches) != 1:
            choices = matches["experiment"].dropna().sort_values().tolist()
            raise drive.DriveDataError(
                f"Choose experiment= for {recording_id!r}; behavior labels are {choices}"
            )
        return matches.iloc[0].to_dict()

    def fetch(
        self,
        recording_id: str,
        layer: str,
        *,
        experiment: str | None = None,
        max_gib: float = drive.DEFAULT_MAX_GIB,
    ) -> Path:
        """Fetch and verify the one file selected by relational identity."""

        row = self._one_file(recording_id, layer, experiment)
        return self._source.fetch(str(row["filename"]), max_gib=max_gib)

    def load(
        self,
        recording_id: str,
        layer: str,
        *,
        experiment: str | None = None,
        max_gib: float = drive.DEFAULT_MAX_GIB,
    ) -> Any:
        """Load one selected layer without constructing a ``Recording`` object."""

        row = self._one_file(recording_id, layer, experiment)
        value = self._source.load(str(row["filename"]), max_gib=max_gib)
        if row["layer"] == "behavior":
            if not isinstance(value, Mapping) or recording_id not in value:
                raise drive.DriveDataError(
                    f"Behavior file {row['filename']} has no session {recording_id!r}"
                )
            return value[recording_id]
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "availability": "mounted" if self._source.connected else "metadata_only",
            "tables": {
                name: {"rows": len(frame), "columns": list(frame.columns)}
                for name, frame in sorted(self._tables.items())
            },
            "query": 'db.query("SELECT * FROM recordings LIMIT 5")',
            "load": 'db.load(recording_id, "reduced_neural")',
        }

    def __repr__(self) -> str:
        return f"SQLData({pprint.pformat(self.to_dict(), width=100, sort_dicts=False)})"

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLData":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def connect(
    *,
    source: drive.Dataset | None = None,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    mount: bool = True,
) -> SQLData:
    """Create flat tables from a supplied or newly connected Drive release."""

    if source is not None and (root is not None or cache is not None):
        raise ValueError("source cannot be combined with root= or cache=")
    handle = source or drive.connect(root=root, cache=cache, mount=mount)
    return SQLData(handle)


def setup(
    *,
    source: drive.Dataset | None = None,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    mount: bool = True,
    report: bool = True,
) -> SQLData:
    """Connect to Drive, build the relational catalog, and optionally print it."""

    db = connect(source=source, root=root, cache=cache, mount=mount)
    if report:
        print(db.schema().to_string(index=False))
    return db


__all__ = ["SQLData", "connect", "setup"]
