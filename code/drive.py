from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

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
    pass


def setup(
    *,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    database: str | Path | None = None,
    mount: bool = True,
    report: bool = False,
) -> Any:
    prepare_colab(mount)
    install_dependencies()
    expose_code_folder()
    reload_workspace_modules()

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
    cache_path = local_cache(cache)
    explicit_root = root or os.environ.get("ZHONGDB_DATASET_ROOT")

    if explicit_root is not None:
        dataset_root = Path(explicit_root)
    elif not is_colab():
        return None, cache_path
    else:
        prepare_colab(mount)
        dataset_root = discover_release()

    if not dataset_root.is_dir():
        raise DriveDataError(f"Dataset root does not exist: {dataset_root}")
    return dataset_root.resolve(), cache_path


def fetch_file(
    row: Mapping[str, Any],
    *,
    root: str | Path | None,
    cache: str | Path,
    max_gib: float = DEFAULT_MAX_GIB,
) -> Path:
    selection = selected_file(row, max_gib)
    source = source_path(root, selection)
    destination = cache_path(cache, selection["name"])

    if verified_file(destination, selection):
        return destination

    discard(destination)
    validate_source(source, selection, destination.parent)
    return copy_verified(source, destination, selection)


def load_numpy(
    row: Mapping[str, Any],
    *,
    root: str | Path | None,
    cache: str | Path,
    max_gib: float = DEFAULT_MAX_GIB,
    allow_pickle: bool = False,
) -> Any:
    path = fetch_file(row, root=root, cache=cache, max_gib=max_gib)
    name = catalog_filename(row)

    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=allow_pickle) as archive:
                return {key: archive[key] for key in archive.files}
        if path.suffix.lower() == ".npy":
            value = np.load(path, allow_pickle=allow_pickle)
            return value.item() if value.shape == () else value
    except ModuleNotFoundError as error:
        raise DriveDataError(f"{name} needs a missing scientific Python package") from error
    except (OSError, ValueError, TypeError) as error:
        raise DriveDataError(f"Could not load {name}: {error}") from error

    raise DriveDataError(f"Unsupported published file type: {name}")


def selected_file(row: Mapping[str, Any], max_gib: float) -> dict[str, Any]:
    if not isinstance(max_gib, (int, float)) or max_gib <= 0:
        raise ValueError("max_gib must be positive")

    name = catalog_filename(row)
    size = int(row["size_bytes"])
    if size > int(float(max_gib) * 2**30):
        raise DriveDataError(
            f"{name} is {size / 2**30:.2f} GiB; increase max_gib only after "
            "checking the available disk and memory"
        )
    return {
        "name": name,
        "relative_path": catalog_path(str(row["relative_path"])),
        "size": size,
        "md5": str(row["md5"]).lower(),
    }


def source_path(
    root: str | Path | None,
    selection: Mapping[str, Any],
) -> Path:
    if root is None:
        raise DriveDataError("Fetching requires a mounted or explicit dataset root")

    dataset_root = Path(root).resolve()
    source = dataset_root.joinpath(*selection["relative_path"].parts).resolve()
    if source != dataset_root and dataset_root not in source.parents:
        raise DriveDataError(
            f"Catalog path escapes the dataset root: {selection['name']}"
        )
    return source


def cache_path(cache: str | Path, name: str) -> Path:
    folder = Path(cache).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def verified_file(path: Path, selection: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == selection["size"]
        and file_md5(path) == selection["md5"]
    )


def validate_source(
    source: Path,
    selection: Mapping[str, Any],
    cache: Path,
) -> None:
    if not source.is_file():
        raise DriveDataError(f"Selected Drive file is unavailable: {source}")
    if source.stat().st_size != selection["size"]:
        raise DriveDataError(
            f"Drive file size does not match the catalog: {selection['name']}"
        )
    if shutil.disk_usage(cache).free < selection["size"] * 1.2:
        raise DriveDataError("Not enough local disk space for the selected file")


def copy_verified(
    source: Path,
    destination: Path,
    selection: Mapping[str, Any],
) -> Path:
    partial = destination.with_suffix(destination.suffix + ".partial")
    discard(partial)
    digest = hashlib.md5()
    copied = 0

    try:
        with source.open("rb") as input_stream, partial.open("wb") as output_stream:
            while block := input_stream.read(8 * 2**20):
                output_stream.write(block)
                digest.update(block)
                copied += len(block)
        if copied != selection["size"] or digest.hexdigest() != selection["md5"]:
            raise DriveDataError(
                f"Copied file did not match the release catalog: {selection['name']}"
            )
        partial.replace(destination)
    except BaseException:
        discard(partial)
        raise

    return destination


def atomic_copy(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source).resolve()
    target = Path(destination).expanduser().resolve()
    if target == source_path:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    discard(partial)
    try:
        shutil.copy2(source_path, partial)
        partial.replace(target)
    except BaseException:
        discard(partial)
        raise
    return target


def catalog_filename(row: Mapping[str, Any]) -> str:
    value = row.get("filename", row.get("name"))
    if value is None or Path(str(value)).name != str(value):
        raise DriveDataError("Choose an exact catalog filename, not a path")
    return str(value)


def catalog_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    invalid = (
        not value
        or relative == PurePosixPath(".")
        or relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "data"
    )
    if invalid:
        raise DriveDataError(f"Unsafe catalog path: {value!r}")
    return relative


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_cache(cache: str | Path | None) -> Path:
    selected = Path(cache) if cache is not None else Path(tempfile.gettempdir()) / "zhongdb-cache"
    return selected.resolve()


def is_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def prepare_colab(mount: bool) -> None:
    if mount and is_colab():
        mount_colab_drive()


def mount_colab_drive() -> None:
    if Path("/content/drive/MyDrive").is_dir():
        return
    from google.colab import drive as colab_drive

    colab_drive.mount("/content/drive", force_remount=False)


def discover_release() -> Path:
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


def install_dependencies() -> None:
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


def expose_code_folder() -> None:
    code = str(Path(__file__).resolve().parent)
    if code not in sys.path:
        sys.path.insert(0, code)
    importlib.invalidate_caches()


def reload_workspace_modules() -> None:
    modules = (
        "arrays",
        "catalog",
        "database",
        "dprime",
        "dprime.evaluation",
        "dprime.inference",
        "dprime.trials",
        "graph",
        "graph.context",
        "graph.display",
        "graph.execution",
        "graph.types",
        "graph.diagram",
        "graph.widget",
        "joiner",
        "position",
        "release",
        "sql",
        "warehouse",
    )
    for name in modules:
        sys.modules.pop(name, None)


def discard(path: Path) -> None:
    path.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_MAX_GIB",
    "DriveDataError",
    "atomic_copy",
    "catalog_filename",
    "catalog_path",
    "fetch_file",
    "is_colab",
    "load_numpy",
    "locate_release",
    "setup",
]
