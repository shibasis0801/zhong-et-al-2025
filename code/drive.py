"""Filesystem access and one-call setup for the Zhong et al. dataset."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ARTICLE_ID = 28811129
ARTICLE_VERSION = 2
EXPECTED_FILE_COUNT = 297
EXPECTED_TOTAL_BYTES = 452_233_500_962
EXPECTED_IMAGING_RECORDINGS = 89
EXPECTED_IMAGING_MICE = 19
DEFAULT_MAX_GIB = 10.0
WORKSPACE_NAME = "Zhong et al. 2025 - Neuromatch Team Workspace"
DATASET_NAME = "Janelia dataset - Zhong et al. 2025 (Figshare v2)"
DATASET_SHORTCUT = "Zhong2025_Janelia_v2"
METADATA_ROOT = Path(__file__).resolve().parent / "metadata"
INVENTORY_SNAPSHOT = METADATA_ROOT / "figshare-v2-inventory.json"
EXPERIMENT_INDEX_SNAPSHOT = METADATA_ROOT / "imaging-experiment-index.json"


class DriveDataError(RuntimeError):
    """Raised when release files cannot be accessed safely."""


def is_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def setup(
    *,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    database: str | Path | None = None,
    mount: bool = True,
    report: bool = False,
) -> Any:
    """Prepare Colab and return the dataset's Pandas/SQL interface."""

    if mount and is_colab():
        _mount_colab_drive()
    _install_catalog_dependencies()
    code = Path(__file__).resolve().parent
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    importlib.invalidate_caches()
    for name in ("database", "sql"):
        sys.modules.pop(name, None)
    sql = importlib.import_module("sql")
    return sql.setup(
        root=root,
        cache=cache,
        database=database,
        mount=False,
        report=report,
    )


def locate_release(
    *,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    mount: bool = True,
) -> tuple[Path | None, Path]:
    """Return the mounted release root (if available) and local file cache."""

    cache_path = (
        Path(cache) if cache is not None else Path(tempfile.gettempdir()) / "zhongdb-cache"
    ).resolve()
    explicit = root or os.environ.get("ZHONGDB_DATASET_ROOT")
    if explicit is not None:
        dataset_root = Path(explicit)
    elif not is_colab():
        return None, cache_path
    else:
        if mount:
            _mount_colab_drive()
        dataset_root = _discover_root()

    if not dataset_root.is_dir():
        raise DriveDataError(f"Dataset root does not exist: {dataset_root}")
    return dataset_root.resolve(), cache_path


def read_release(root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and validate the release marker and its complete file catalog."""

    root = Path(root)
    try:
        status = _read_json(root / "TRANSFER_STATUS.json")
        release = _read_json(root / "metadata/RELEASE.json")
        with (root / "metadata/catalog.csv").open(newline="", encoding="utf-8") as stream:
            files = [_catalog_row(row) for row in csv.DictReader(stream)]
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DriveDataError(f"Shared dataset metadata is incomplete: {error}") from error

    expected_release = {
        "article_id": ARTICLE_ID,
        "version": ARTICLE_VERSION,
        "file_count": EXPECTED_FILE_COUNT,
        "total_bytes": EXPECTED_TOTAL_BYTES,
    }
    expected_status = {
        "state": "complete",
        "expected_files": EXPECTED_FILE_COUNT,
        "verified_files": EXPECTED_FILE_COUNT,
        "expected_bytes": EXPECTED_TOTAL_BYTES,
        "verified_bytes": EXPECTED_TOTAL_BYTES,
    }
    for actual, expected, label in (
        (release, expected_release, "release metadata"),
        (status, expected_status, "transfer status"),
    ):
        for key, value in expected.items():
            if actual.get(key) != value:
                raise DriveDataError(f"Unexpected {label} field {key!r}")

    _validate_catalog(files)
    return release, files


def read_snapshot() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the bundled metadata-only snapshot of the pinned release."""

    try:
        inventory = _read_json(INVENTORY_SNAPSHOT)
        article = inventory["article"]
        files = [
            _catalog_row(
                {
                    **row,
                    "relative_path": row.get("relative_path", f"data/{row['name']}"),
                }
            )
            for row in inventory["files"]
        ]
        release = {
            "article_id": int(article["id"]),
            "version": int(article["version"]),
            "file_count": int(article["file_count"]),
            "total_bytes": int(article["total_size_bytes"]),
            "doi": str(article["doi"]),
            "title": str(article["title"]),
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DriveDataError(f"Bundled release metadata is incomplete: {error}") from error

    expected = {
        "article_id": ARTICLE_ID,
        "version": ARTICLE_VERSION,
        "file_count": EXPECTED_FILE_COUNT,
        "total_bytes": EXPECTED_TOTAL_BYTES,
    }
    for key, value in expected.items():
        if release[key] != value:
            raise DriveDataError(f"Unexpected snapshot field {key!r}")
    _validate_catalog(files)
    return release, files


def read_experiment_index() -> Mapping[str, Any]:
    """Read the bundled JSON projection of Imaging_Exp_info.npy."""

    try:
        index = _read_json(EXPERIMENT_INDEX_SNAPSHOT)
        source = index["source"]
        summary = index["summary"]
        experiments = index["experiments"]
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DriveDataError(f"Bundled imaging index is incomplete: {error}") from error

    expected = {
        "article_id": ARTICLE_ID,
        "article_version": ARTICLE_VERSION,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise DriveDataError(f"Unexpected imaging-index source field {key!r}")
    if summary.get("unique_recordings") != EXPECTED_IMAGING_RECORDINGS:
        raise DriveDataError("Imaging index should describe 89 recordings")
    if summary.get("unique_mice") != EXPECTED_IMAGING_MICE:
        raise DriveDataError("Imaging index should describe 19 mice")
    if not isinstance(experiments, Mapping):
        raise DriveDataError("Imaging index experiments must be a mapping")
    return experiments


def fetch_file(
    row: Mapping[str, Any],
    *,
    root: str | Path | None,
    cache: str | Path,
    max_gib: float = DEFAULT_MAX_GIB,
) -> Path:
    """Copy one catalog-selected file locally and verify its size and MD5."""

    if root is None:
        raise DriveDataError("Fetching requires a mounted or explicit dataset root")
    if not isinstance(max_gib, (int, float)) or max_gib <= 0:
        raise ValueError("max_gib must be positive")

    name = _filename(row)
    size = int(row["size_bytes"])
    md5 = str(row["md5"]).lower()
    if size > int(float(max_gib) * 2**30):
        raise DriveDataError(
            f"{name} is {size / 2**30:.2f} GiB; increase max_gib only after "
            "checking the available disk and memory"
        )

    root = Path(root).resolve()
    relative = _safe_relative_path(str(row["relative_path"]))
    source = root.joinpath(*relative.parts).resolve()
    if source != root and root not in source.parents:
        raise DriveDataError(f"Catalog path escapes the dataset root: {name}")

    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / name
    if destination.is_file():
        if destination.stat().st_size == size and _md5(destination) == md5:
            return destination
        destination.unlink()

    if not source.is_file():
        raise DriveDataError(f"Selected Drive file is unavailable: {source}")
    if source.stat().st_size != size:
        raise DriveDataError(f"Drive file size does not match the catalog: {name}")
    if shutil.disk_usage(cache).free < size * 1.2:
        raise DriveDataError("Not enough local disk space for the selected file")

    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.md5()
    copied = 0
    try:
        with source.open("rb") as src, partial.open("wb") as dst:
            while block := src.read(8 * 2**20):
                dst.write(block)
                digest.update(block)
                copied += len(block)
        if copied != size or digest.hexdigest() != md5:
            raise DriveDataError(f"Copied file did not match the release catalog: {name}")
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def load_numpy(
    row: Mapping[str, Any],
    *,
    root: str | Path | None,
    cache: str | Path,
    max_gib: float = DEFAULT_MAX_GIB,
    allow_pickle: bool = False,
) -> Any:
    """Fetch and open one catalog-selected ``.npy`` or ``.npz`` file."""

    path = fetch_file(row, root=root, cache=cache, max_gib=max_gib)
    name = _filename(row)
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=allow_pickle) as archive:
                return {key: archive[key] for key in archive.files}
        if path.suffix.lower() == ".npy":
            loaded = np.load(path, allow_pickle=allow_pickle)
            return loaded.item() if loaded.shape == () else loaded
    except ModuleNotFoundError as error:
        raise DriveDataError(f"{name} needs a missing scientific Python package") from error
    except (OSError, ValueError, TypeError) as error:
        raise DriveDataError(f"Could not load {name}: {error}") from error
    raise DriveDataError(f"Unsupported published file type: {name}")


def _install_catalog_dependencies() -> None:
    try:
        importlib.import_module("duckdb")
        importlib.import_module("pandas")
    except ModuleNotFoundError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "pandas>=2.2,<3",
                "duckdb>=1.4,<2",
            ]
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog_row(row: Mapping[str, Any]) -> dict[str, Any]:
    optional = lambda value: str(value).strip() or None
    return {
        **dict(row),
        "name": str(row["name"]),
        "id": int(row["id"]),
        "category": str(row["category"]),
        "size_bytes": int(row["size_bytes"]),
        "md5": str(row["md5"]).lower(),
        "relative_path": str(row["relative_path"]),
        "experiment": optional(row.get("experiment", "")),
        "recording_id": optional(row.get("recording_id", "")),
        "retinotopy_id": optional(row.get("retinotopy_id", "")),
    }


def _validate_catalog(files: Sequence[Mapping[str, Any]]) -> None:
    if len(files) != EXPECTED_FILE_COUNT:
        raise DriveDataError(f"Catalog has {len(files)} rows; expected {EXPECTED_FILE_COUNT}")
    if sum(int(row["size_bytes"]) for row in files) != EXPECTED_TOTAL_BYTES:
        raise DriveDataError("Catalog byte total does not match Figshare v2")
    names = [_filename(row) for row in files]
    ids = [int(row["id"]) for row in files]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        raise DriveDataError("Catalog file names and IDs must be unique")
    for row in files:
        name = _filename(row)
        if Path(name).name != name:
            raise DriveDataError(f"Catalog name must not contain a path: {name!r}")
        if int(row["size_bytes"]) < 1:
            raise DriveDataError(f"Catalog size must be positive: {name}")
        md5 = str(row["md5"])
        if len(md5) != 32 or any(char not in "0123456789abcdef" for char in md5):
            raise DriveDataError(f"Invalid MD5 in catalog: {name}")
        _safe_relative_path(str(row["relative_path"]))


def _filename(row: Mapping[str, Any]) -> str:
    value = row.get("filename", row.get("name"))
    if value is None or Path(str(value)).name != str(value):
        raise DriveDataError("Choose an exact catalog filename, not a path")
    return str(value)


def _safe_relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        not value
        or relative == PurePosixPath(".")
        or relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "data"
    ):
        raise DriveDataError(f"Unsafe catalog path: {value!r}")
    return relative


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mount_colab_drive() -> None:
    if Path("/content/drive/MyDrive").is_dir():
        return
    from google.colab import drive as colab_drive

    colab_drive.mount("/content/drive", force_remount=False)


def _discover_root() -> Path:
    my_drive = Path("/content/drive/MyDrive")
    choices = (
        Path(__file__).resolve().parent / DATASET_NAME,
        my_drive / DATASET_SHORTCUT,
        my_drive / WORKSPACE_NAME / DATASET_NAME,
    )
    match = next((path for path in choices if path.is_dir()), None)
    if match is None:
        raise DriveDataError(
            f"Shared dataset not found. Add {DATASET_SHORTCUT!r} or the team workspace "
            "to My Drive, then rerun."
        )
    return match


__all__ = [
    "DEFAULT_MAX_GIB",
    "DriveDataError",
    "fetch_file",
    "is_colab",
    "load_numpy",
    "locate_release",
    "read_experiment_index",
    "read_release",
    "read_snapshot",
    "setup",
]
