from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from drive import (
    ARTICLE_ID,
    ARTICLE_VERSION,
    EXPERIMENT_INDEX_SNAPSHOT,
    EXPECTED_FILE_COUNT,
    EXPECTED_IMAGING_MICE,
    EXPECTED_IMAGING_RECORDINGS,
    EXPECTED_TOTAL_BYTES,
    INVENTORY_SNAPSHOT,
    DriveDataError,
    catalog_filename,
    catalog_path,
)


RELEASE_FIELDS = {
    "article_id": ARTICLE_ID,
    "version": ARTICLE_VERSION,
    "file_count": EXPECTED_FILE_COUNT,
    "total_bytes": EXPECTED_TOTAL_BYTES,
}
TRANSFER_FIELDS = {
    "state": "complete",
    "expected_files": EXPECTED_FILE_COUNT,
    "verified_files": EXPECTED_FILE_COUNT,
    "expected_bytes": EXPECTED_TOTAL_BYTES,
    "verified_bytes": EXPECTED_TOTAL_BYTES,
}


def read(root: str | Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return read_snapshot() if root is None else read_release(root)


def read_release(root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    release_root = Path(root)
    status, metadata, files = read_shared_metadata(release_root)
    require_fields(metadata, RELEASE_FIELDS, "release metadata")
    require_fields(status, TRANSFER_FIELDS, "transfer status")
    validate_catalog(files)
    return metadata, files


def read_shared_metadata(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    try:
        status = read_json(root / "TRANSFER_STATUS.json")
        release = read_json(root / "metadata/RELEASE.json")
        with (root / "metadata/catalog.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            files = [catalog_row(row) for row in csv.DictReader(stream)]
        return status, release, files
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DriveDataError(f"Shared dataset metadata is incomplete: {error}") from error


def read_snapshot() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        inventory = read_json(INVENTORY_SNAPSHOT)
        article = inventory["article"]
        files = [
            catalog_row(
                {
                    **row,
                    "relative_path": row.get("relative_path", f"data/{row['name']}"),
                }
            )
            for row in inventory["files"]
        ]
        metadata = {
            "article_id": int(article["id"]),
            "version": int(article["version"]),
            "file_count": int(article["file_count"]),
            "total_bytes": int(article["total_size_bytes"]),
            "doi": str(article["doi"]),
            "title": str(article["title"]),
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DriveDataError(f"Bundled release metadata is incomplete: {error}") from error

    require_fields(metadata, RELEASE_FIELDS, "snapshot")
    validate_catalog(files)
    return metadata, files


def read_experiment_index() -> Mapping[str, Any]:
    try:
        index = read_json(EXPERIMENT_INDEX_SNAPSHOT)
        source = index["source"]
        summary = index["summary"]
        experiments = index["experiments"]
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DriveDataError(f"Bundled imaging index is incomplete: {error}") from error

    require_fields(
        source,
        {"article_id": ARTICLE_ID, "article_version": ARTICLE_VERSION},
        "imaging-index source",
    )
    if summary.get("unique_recordings") != EXPECTED_IMAGING_RECORDINGS:
        raise DriveDataError("Imaging index should describe 89 recordings")
    if summary.get("unique_mice") != EXPECTED_IMAGING_MICE:
        raise DriveDataError("Imaging index should describe 19 mice")
    if not isinstance(experiments, Mapping):
        raise DriveDataError("Imaging index experiments must be a mapping")
    return experiments


def require_fields(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for name, value in expected.items():
        if actual.get(name) != value:
            raise DriveDataError(f"Unexpected {label} field {name!r}")


def validate_catalog(files: Sequence[Mapping[str, Any]]) -> None:
    if len(files) != EXPECTED_FILE_COUNT:
        raise DriveDataError(f"Catalog has {len(files)} rows; expected {EXPECTED_FILE_COUNT}")
    if sum(int(row["size_bytes"]) for row in files) != EXPECTED_TOTAL_BYTES:
        raise DriveDataError("Catalog byte total does not match Figshare v2")

    names = [catalog_filename(row) for row in files]
    ids = [int(row["id"]) for row in files]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        raise DriveDataError("Catalog file names and IDs must be unique")

    for row in files:
        name = catalog_filename(row)
        if int(row["size_bytes"]) < 1:
            raise DriveDataError(f"Catalog size must be positive: {name}")
        md5 = str(row["md5"])
        if len(md5) != 32 or any(character not in "0123456789abcdef" for character in md5):
            raise DriveDataError(f"Invalid MD5 in catalog: {name}")
        catalog_path(str(row["relative_path"]))


def catalog_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(row),
        "name": str(row["name"]),
        "id": int(row["id"]),
        "category": str(row["category"]),
        "size_bytes": int(row["size_bytes"]),
        "md5": str(row["md5"]).lower(),
        "relative_path": str(row["relative_path"]),
        "experiment": optional_text(row.get("experiment", "")),
        "recording_id": optional_text(row.get("recording_id", "")),
        "retinotopy_id": optional_text(row.get("retinotopy_id", "")),
    }


def optional_text(value: Any) -> str | None:
    return str(value).strip() or None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["read", "read_experiment_index", "read_release", "read_snapshot"]
