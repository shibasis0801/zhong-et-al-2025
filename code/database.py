"""One readable Pandas/DuckDB interface to the Zhong et al. dataset."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

import drive


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LAYERS = {
    "imaging_behavior": "behavior",
    "reduced_neural": "reduced_neural",
    "full_neural": "full_neural",
    "retinotopy": "retinotopy",
}
_COHORT_ORDER = ("supervised", "unsupervised", "grating", "naive")
_DATABASE_NAME = "catalog.duckdb"


class ZhongDB:
    """Read the dataset and expose its catalog through Pandas and SQL."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        cache: str | Path | None = None,
        database: str | Path | None = None,
        mount: bool = True,
    ) -> None:
        self.root, self.cache = drive.locate_release(root=root, cache=cache, mount=mount)
        self.database_path = _database_path(database, self.cache)
        release, catalog = self._read_catalog()
        release = {**release, "root": str(self.root) if self.root is not None else None}
        files = _files_frame(catalog)
        experiment_rows = _experiment_rows(self._read_imaging_index(files))
        self._tables = _build_tables(release, files, experiment_rows)
        self._connection = _create_database(self.database_path, self._tables)

    @property
    def connected(self) -> bool:
        return self.root is not None

    @property
    def tables(self) -> tuple[str, ...]:
        return tuple(sorted(self._tables))

    def table(self, name: str) -> pd.DataFrame:
        """Return an independent copy of one table."""

        if name not in self._tables:
            raise KeyError(f"Unknown table {name!r}; choose from {self.tables}")
        return self._tables[name].copy()

    def query(
        self,
        statement: str,
        parameters: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run DuckDB SQL and return the result as a DataFrame."""

        cursor = (
            self._connection.execute(statement, parameters)
            if parameters is not None
            else self._connection.execute(statement)
        )
        return cursor.fetchdf()

    def register(self, name: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Register an analysis DataFrame for later SQL joins."""

        if not _IDENTIFIER.fullmatch(name):
            raise ValueError("Table name must be a simple SQL identifier")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas.DataFrame")
        if name in self._tables:
            raise ValueError(f"Table already exists: {name}")
        stored = frame.copy()
        self._tables[name] = stored
        self._connection.register(name, stored)
        return stored

    def export(self, path: str | Path) -> Path:
        """Copy the native DuckDB database to a browser or deployment folder."""

        destination = Path(path).expanduser().resolve()
        if destination == self.database_path:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f".{destination.name}.partial")
        partial.unlink(missing_ok=True)
        self._connection.execute("CHECKPOINT")
        try:
            shutil.copy2(self.database_path, partial)
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return destination

    def schema(self, table: str | None = None) -> pd.DataFrame:
        """Summarize all tables, or describe the columns of one table."""

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

    def fetch_file(
        self,
        filename: str,
        *,
        max_gib: float = drive.DEFAULT_MAX_GIB,
    ) -> Path:
        """Fetch one exact filename selected from the ``files`` table."""

        return drive.fetch_file(
            self._file(filename), root=self.root, cache=self.cache, max_gib=max_gib
        )

    def load_file(
        self,
        filename: str,
        *,
        max_gib: float = drive.DEFAULT_MAX_GIB,
    ) -> Any:
        """Load one exact filename selected from the ``files`` table."""

        return self._load_row(self._file(filename), max_gib=max_gib)

    def fetch(
        self,
        recording_id: str,
        layer: str,
        *,
        experiment: str | None = None,
        max_gib: float = drive.DEFAULT_MAX_GIB,
    ) -> Path:
        """Fetch a file selected by recording, layer, and optional experiment."""

        return drive.fetch_file(
            self._recording_file(recording_id, layer, experiment),
            root=self.root,
            cache=self.cache,
            max_gib=max_gib,
        )

    def load(
        self,
        recording_id: str,
        layer: str,
        *,
        experiment: str | None = None,
        max_gib: float = drive.DEFAULT_MAX_GIB,
    ) -> Any:
        """Load a file selected by recording, layer, and optional experiment."""

        row = self._recording_file(recording_id, layer, experiment)
        value = self._load_row(row, max_gib=max_gib)
        if row["layer"] == "behavior":
            if not isinstance(value, Mapping) or recording_id not in value:
                raise drive.DriveDataError(
                    f"Behavior file {row['filename']} has no session {recording_id!r}"
                )
            return value[recording_id]
        return value

    def close(self) -> None:
        self._connection.execute("CHECKPOINT")
        self._connection.close()

    def __enter__(self) -> "ZhongDB":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        mode = "mounted" if self.connected else "metadata only"
        file_count = len(self._tables["files"])
        recording_count = len(self._tables["recordings"])
        return (
            f"ZhongDB({mode}; {file_count} files; {recording_count} recordings; "
            f"{self.database_path})"
        )

    def _read_catalog(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self.root is not None:
            return drive.read_release(self.root)
        return drive.read_snapshot()

    def _read_imaging_index(self, files: pd.DataFrame) -> Mapping[str, Any]:
        if self.root is None:
            return drive.read_experiment_index()
        row = files.loc[files["filename"] == "Imaging_Exp_info.npy"]
        if len(row) != 1:
            raise drive.DriveDataError("The release must contain Imaging_Exp_info.npy")
        value = self._load_row(_series_row(row.iloc[0]), max_gib=1.0)
        if not isinstance(value, Mapping):
            raise drive.DriveDataError("Imaging_Exp_info.npy should contain a dictionary")
        return value

    def _load_row(self, row: Mapping[str, Any], *, max_gib: float) -> Any:
        filename = str(row["filename"])
        allow_pickle = filename.endswith(".npy") or row["category"] == "area_outlines"
        value = drive.load_numpy(
            row,
            root=self.root,
            cache=self.cache,
            max_gib=max_gib,
            allow_pickle=allow_pickle,
        )
        if row["category"] != "reduced_neural":
            return value
        if not isinstance(value, Mapping):
            raise drive.DriveDataError(f"Reduced-neural file is not a mapping: {filename}")
        arrays = {
            str(name): array
            for name, array in value.items()
            if isinstance(array, np.ndarray)
        }
        if not {"U", "V"}.issubset(arrays):
            raise drive.DriveDataError(f"Reduced-neural file has no U/V arrays: {filename}")
        return arrays

    def _file(self, filename: str) -> dict[str, Any]:
        if Path(filename).name != filename:
            raise drive.DriveDataError("Choose an exact filename, not a path")
        matches = self._tables["files"]
        matches = matches[matches["filename"] == filename]
        if len(matches) != 1:
            raise drive.DriveDataError(f"File is not in the pinned release: {filename!r}")
        return _series_row(matches.iloc[0])

    def _recording_file(
        self, recording_id: str, layer: str, experiment: str | None
    ) -> dict[str, Any]:
        layer = "reduced_neural" if layer == "svd" else layer
        if layer not in set(_LAYERS.values()):
            raise ValueError(
                "layer must be behavior, reduced_neural (or svd), full_neural, or retinotopy"
            )
        matches = self._tables["recording_files"]
        matches = matches[
            (matches["recording_id"] == recording_id) & (matches["layer"] == layer)
        ]
        if layer == "behavior" and experiment is not None:
            matches = matches[matches["experiment"] == experiment]
        if matches.empty:
            detail = f" and experiment {experiment!r}" if experiment else ""
            raise drive.DriveDataError(
                f"No {layer!r} file for recording {recording_id!r}{detail}"
            )
        if len(matches) != 1:
            choices = matches["experiment"].dropna().sort_values().tolist()
            raise drive.DriveDataError(
                f"Choose experiment= for {recording_id!r}; behavior labels are {choices}"
            )
        return _series_row(matches.iloc[0])


def _plain(value: Any) -> Any:
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


def _series_row(series: pd.Series) -> dict[str, Any]:
    return {str(name): value for name, value in series.items()}


def _database_path(database: str | Path | None, cache: Path) -> Path:
    path = Path(database).expanduser() if database is not None else cache / _DATABASE_NAME
    return path.resolve()


def _create_database(
    path: Path, tables: Mapping[str, pd.DataFrame]
) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    connection = duckdb.connect(str(partial))
    try:
        try:
            for name, frame in tables.items():
                connection.register("_frame", frame)
                connection.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _frame')
                connection.unregister("_frame")
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return duckdb.connect(str(path))


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(_plain(value)).strip()
    return text or None


def _block(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _cohort(experiment: str) -> str:
    if experiment.startswith("sup_"):
        return "supervised"
    if experiment.startswith("unsup_"):
        return "unsupervised"
    if experiment.startswith("naive_"):
        return "naive"
    return "grating" if experiment.endswith("_grating") else "other"


def _stage(experiment: str) -> str | None:
    stages = ("train1", "test1", "train2", "test2", "test3")
    return next((stage for stage in stages if stage in experiment), None)


def _moment(experiment: str) -> str:
    if "_before_" in experiment:
        return "before"
    if "_after_" in experiment:
        return "after"
    return "snapshot"


def _experiment_rows(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment, entries in index.items():
        entries = _plain(entries)
        entries = [entries] if isinstance(entries, Mapping) else entries
        if not isinstance(entries, list):
            raise drive.DriveDataError(f"Imaging index {experiment!r} should contain rows")
        for source_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise drive.DriveDataError(
                    f"Imaging index {experiment!r}[{source_index}] is not a row"
                )
            source = _plain(entry.get("source", entry))
            if not isinstance(source, Mapping):
                raise drive.DriveDataError(
                    f"Imaging index {experiment!r}[{source_index}] has no source row"
                )
            mouse, date, block = str(source["mname"]), str(source["datexp"]), _block(source["blk"])
            rows.append(
                {
                    "experiment": str(experiment),
                    "source_row": source_index,
                    "recording_id": str(entry.get("recording_id", f"{mouse}_{date}_{block}")),
                    "retinotopy_id": str(entry.get("retinotopy_id", f"{mouse}_{date}")),
                    "mouse": mouse,
                    "date": date,
                    "block": block,
                    "session_number": _plain(source.get("sess#")),
                    "is_2p": bool(source.get("is2p")) if source.get("is2p") is not None else None,
                    "gender": _text(source.get("Gender")),
                    "reward_type": _text(source.get("rewType")),
                    "stimulus": _text(source.get("stim")),
                    "stimulus_type": _text(source.get("stimtype")),
                    "depth_json": json.dumps(_plain(source.get("depth", []))),
                    "stimulus_ids_json": json.dumps(_plain(source.get("stim_id", []))),
                    "note": _text(source.get("Note")),
                    "source_json": json.dumps(_plain(source), sort_keys=True),
                }
            )
    return rows


def _files_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        name, size = str(row["name"]), int(row["size_bytes"])
        records.append(
            {
                "filename": name,
                "figshare_file_id": int(row["id"]),
                "category": str(row["category"]),
                "experiment": _text(row.get("experiment")),
                "recording_id": _text(row.get("recording_id")),
                "retinotopy_id": _text(row.get("retinotopy_id")),
                "size_bytes": size,
                "size_mib": size / 2**20,
                "size_gib": size / 2**30,
                "md5": str(row["md5"]).lower(),
                "relative_path": str(row.get("relative_path") or f"data/{name}"),
            }
        )
    return pd.DataFrame(records).sort_values("filename", ignore_index=True)


def _build_tables(
    release: Mapping[str, Any],
    files: pd.DataFrame,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
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
    recording_files = _recording_files(files, memberships)
    recordings = _recordings(memberships, recording_files)
    mice = _mice(memberships, recordings)
    release_row = {str(key): _plain(value) for key, value in release.items()}
    return {
        "release": pd.DataFrame([release_row]),
        "files": files,
        "experiment_rows": experiment_rows,
        "memberships": memberships,
        "experiments": experiments,
        "recording_files": recording_files,
        "recordings": recordings,
        "mice": mice,
    }


def _recording_files(files: pd.DataFrame, memberships: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "recording_id", "experiment", "layer", "filename", "figshare_file_id",
        "category", "size_bytes", "size_mib", "size_gib", "md5", "relative_path",
    ]
    direct = files[files["category"].isin(("reduced_neural", "full_neural"))].copy()
    direct["layer"] = direct["category"].map(_LAYERS)
    direct["experiment"] = None
    behavior = memberships[["experiment", "recording_id"]].merge(
        files[files["category"] == "imaging_behavior"],
        on="experiment", how="inner", validate="many_to_one", suffixes=("", "_file"),
    )
    behavior["layer"] = "behavior"
    retinotopy = memberships[["recording_id", "retinotopy_id"]].drop_duplicates().merge(
        files[files["category"] == "retinotopy"],
        on="retinotopy_id", how="inner", validate="many_to_one", suffixes=("", "_file"),
    )
    retinotopy["layer"], retinotopy["experiment"] = "retinotopy", None
    return (
        pd.concat([direct[columns], behavior[columns], retinotopy[columns]], ignore_index=True)
        .drop_duplicates(["recording_id", "experiment", "layer", "filename"])
        .sort_values(["recording_id", "layer", "experiment"], na_position="first")
        .reset_index(drop=True)
    )


def _recordings(memberships: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    identity = memberships[
        ["recording_id", "retinotopy_id", "mouse", "date", "block"]
    ].drop_duplicates()
    result = identity.set_index("recording_id")
    result["experiment_count"] = memberships.groupby("recording_id")["experiment"].nunique()
    result["experiments_json"] = memberships.groupby("recording_id")["experiment"].agg(
        lambda values: json.dumps(sorted(set(values)))
    )
    layers = files.assign(present=True).pivot_table(
        index="recording_id", columns="layer", values="present", aggfunc="any", fill_value=False
    )
    for layer in _LAYERS.values():
        result[f"has_{layer}"] = layers[layer] if layer in layers else False
    result = result.join(
        files.groupby("recording_id").agg(
            linked_file_count=("filename", "nunique"), linked_bytes=("size_bytes", "sum")
        )
    ).reset_index()
    result["linked_gib"] = result["linked_bytes"] / 2**30
    return result.sort_values(["mouse", "date", "block"], ignore_index=True)


def _mice(memberships: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    def primary(values: pd.Series) -> str:
        present = set(values)
        return next((name for name in _COHORT_ORDER if name in present), "other")

    cohorts = memberships[["mouse", "cohort"]].drop_duplicates().groupby("mouse")["cohort"].agg(
        primary_cohort=primary,
        cohorts_json=lambda values: json.dumps(sorted(set(values))),
    )
    summary = recordings.groupby("mouse").agg(
        recording_count=("recording_id", "nunique"),
        has_full_neural=("has_full_neural", "any"),
        has_reduced_neural=("has_reduced_neural", "any"),
    )
    return cohorts.join(summary).reset_index().sort_values("mouse", ignore_index=True)


def main() -> None:
    parser = ArgumentParser(description="Build catalog.duckdb from the Zhong dataset")
    parser.add_argument("database", nargs="?", help="output DuckDB path")
    options = parser.parse_args()
    with ZhongDB(database=options.database) as db:
        print(db.database_path)


__all__ = ["ZhongDB"]


if __name__ == "__main__":
    main()
