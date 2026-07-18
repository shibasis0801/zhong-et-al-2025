from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

import arrays
import catalog
import drive
import release
import warehouse


class ZhongDB:
    def __init__(
        self,
        *,
        root: str | Path | None = None,
        cache: str | Path | None = None,
        database: str | Path | None = None,
        mount: bool = True,
    ) -> None:
        self.root, self.cache = drive.locate_release(root=root, cache=cache, mount=mount)

        release_metadata, file_records = release.read(self.root)
        files = catalog.files(file_records)
        imaging_index = self._read_imaging_index(files)
        experiment_rows = catalog.experiment_rows(imaging_index)

        release_metadata["root"] = str(self.root) if self.root is not None else None
        self._tables = catalog.tables(release_metadata, files, experiment_rows)

        self.database_path = warehouse.database_path(database, self.cache)
        self._connection = warehouse.create(self.database_path, self._tables)

    @property
    def connected(self) -> bool:
        return self.root is not None

    @property
    def tables(self) -> tuple[str, ...]:
        return tuple(sorted(self._tables))

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            raise KeyError(f"Unknown table {name!r}; choose from {self.tables}")
        return self._tables[name].copy()

    def query(
        self,
        statement: str,
        parameters: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        return warehouse.query(self._connection, statement, parameters)

    def register(self, name: str, frame: pd.DataFrame) -> pd.DataFrame:
        stored = warehouse.registration(name, frame, self.tables)
        self._tables[name] = stored
        self._connection.register(name, stored)
        return stored

    def export(self, path: str | Path) -> Path:
        return warehouse.export(self._connection, self.database_path, path)

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

    def fetch_file(
        self,
        filename: str,
        *,
        max_gib: float = drive.DATASET["default_max_gib"],
    ) -> Path:
        return self._fetch(self._file(filename), max_gib)

    def load_file(
        self,
        filename: str,
        *,
        max_gib: float = drive.DATASET["default_max_gib"],
    ) -> Any:
        return self._load(self._file(filename), max_gib)

    def fetch(
        self,
        recording_id: str,
        layer: str,
        *,
        experiment: str | None = None,
        max_gib: float = drive.DATASET["default_max_gib"],
    ) -> Path:
        return self._fetch(self._recording_file(recording_id, layer, experiment), max_gib)

    def load(
        self,
        recording_id: str,
        layer: str,
        *,
        experiment: str | None = None,
        max_gib: float = drive.DATASET["default_max_gib"],
    ) -> Any:
        row = self._recording_file(recording_id, layer, experiment)
        value = self._load(row, max_gib)
        if row["layer"] == "behavior":
            return arrays.session_behavior(value, row["filename"], recording_id)
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
        files = len(self._tables["files"])
        recordings = len(self._tables["recordings"])
        return f"ZhongDB({mode}; {files} files; {recordings} recordings; {self.database_path})"

    def _read_imaging_index(self, files: pd.DataFrame) -> Mapping[str, Any]:
        if self.root is None:
            return release.read_experiment_index()

        matches = files.loc[files["filename"] == "Imaging_Exp_info.npy"]
        if len(matches) != 1:
            raise drive.DriveDataError("The release must contain Imaging_Exp_info.npy")
        value = self._load(row_dict(matches.iloc[0]), max_gib=1.0)
        if not isinstance(value, Mapping):
            raise drive.DriveDataError("Imaging_Exp_info.npy should contain a dictionary")
        return value

    def _fetch(self, row: Mapping[str, Any], max_gib: float) -> Path:
        return drive.fetch_file(row, root=self.root, cache=self.cache, max_gib=max_gib)

    def _load(self, row: Mapping[str, Any], max_gib: float) -> Any:
        return arrays.load(row, root=self.root, cache=self.cache, max_gib=max_gib)

    def _file(self, filename: str) -> dict[str, Any]:
        if Path(filename).name != filename:
            raise drive.DriveDataError("Choose an exact filename, not a path")
        matches = self._tables["files"]
        matches = matches[matches["filename"] == filename]
        if len(matches) != 1:
            raise drive.DriveDataError(f"File is not in the pinned release: {filename!r}")
        return row_dict(matches.iloc[0])

    def _recording_file(
        self,
        recording_id: str,
        layer: str,
        experiment: str | None,
    ) -> dict[str, Any]:
        selected_layer = "reduced_neural" if layer == "svd" else layer
        if selected_layer not in set(catalog.LAYERS.values()):
            raise ValueError(
                "layer must be behavior, reduced_neural (or svd), full_neural, or retinotopy"
            )

        matches = self._tables["recording_files"]
        matches = matches[
            (matches["recording_id"] == recording_id)
            & (matches["layer"] == selected_layer)
        ]
        if selected_layer == "behavior" and experiment is not None:
            matches = matches[matches["experiment"] == experiment]

        if matches.empty:
            detail = f" and experiment {experiment!r}" if experiment else ""
            raise drive.DriveDataError(
                f"No {selected_layer!r} file for recording {recording_id!r}{detail}"
            )
        if len(matches) != 1:
            choices = matches["experiment"].dropna().sort_values().tolist()
            raise drive.DriveDataError(
                f"Choose experiment= for {recording_id!r}; behavior labels are {choices}"
            )
        return row_dict(matches.iloc[0])


def row_dict(series: pd.Series) -> dict[str, Any]:
    return {str(name): value for name, value in series.items()}


def main() -> None:
    parser = ArgumentParser(description="Build catalog.duckdb from the Zhong dataset")
    parser.add_argument("database", nargs="?", help="output DuckDB path")
    options = parser.parse_args()
    with ZhongDB(database=options.database) as database:
        print(database.database_path)


__all__ = ["ZhongDB"]


if __name__ == "__main__":
    main()
