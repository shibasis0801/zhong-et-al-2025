"""Small, safe access layer for the team's shared Zhong et al. dataset.

The public workflow is intentionally short::

    import drive
    data = drive.setup()
    picker = data.picker()
    session = data.recording("TX119_2023_12_24_1")
    reduced = session.load("reduced_neural")

Only selected files are copied from Google Drive. Copies are checked against
the pinned release catalog before loading; paths, checksums, pickle handling,
and cache management remain behind this module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import html
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import pprint
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping
import warnings

import numpy as np

from zhong2025 import (
    experiment_rows,
    fetch_figshare_article,
    format_bytes,
    load_experiment_index,
    load_file_inventory,
    recording_bundle,
)
from zhong2025.atlas import CATEGORY_LABELS


ARTICLE_ID = 28811129
ARTICLE_VERSION = 2
REPRESENTATION_API_VERSION = 1
EXPECTED_FILE_COUNT = 297
EXPECTED_TOTAL_BYTES = 452_233_500_962
WORKSPACE_NAME = "Zhong et al. 2025 - Neuromatch Team Workspace"
DATASET_NAME = "Janelia dataset - Zhong et al. 2025 (Figshare v2)"
DATASET_SHORTCUT = "Zhong2025_Janelia_v2"
DEFAULT_MAX_GIB = 10.0
_MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_RECORDING_PATTERN = re.compile(
    r"^(?P<mouse>.+)_(?P<date>\d{4}_\d{2}_\d{2})_(?P<block>\d+)$"
)
LAYER_LABELS = {
    "behavior": "Behavior",
    "reduced_neural": "Reduced neural (SVD)",
    "full_neural": "Full neural",
    "retinotopy": "Retinotopy",
}


class DriveDataError(RuntimeError):
    """Raised when the shared release cannot be accessed safely."""


def _dict_html(value: Mapping[str, Any]) -> str:
    """Render a real mapping as copyable, escaped notebook text."""

    rendered = pprint.pformat(dict(value), width=100, sort_dicts=False)
    return (
        "<pre style='margin:.3rem 0;padding:.7rem .8rem;max-width:1100px;"
        "max-height:34rem;overflow:auto;border:1px solid #7775;border-radius:7px;"
        "white-space:pre-wrap;overflow-wrap:anywhere'>"
        f"{html.escape(rendered)}</pre>"
    )


@dataclass(frozen=True)
class DataFile:
    """One lightweight row from the 297-file release catalog."""

    name: str
    id: int
    category: str
    size_bytes: int
    md5: str
    relative_path: str
    experiment: str | None = None
    recording_id: str | None = None
    retinotopy_id: str | None = None

    @property
    def size_mib(self) -> float:
        return self.size_bytes / 2**20

    @property
    def size_gib(self) -> float:
        return self.size_bytes / 2**30

    @property
    def label(self) -> str:
        """Readable file label used by notebook controls."""

        return f"{self.name} · {format_bytes(self.size_bytes)}"

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix.lower()

    def to_dict(self) -> dict[str, Any]:
        """Return exact catalog metadata as an ordinary dictionary."""

        return {
            "name": self.name,
            "figshare_file_id": self.id,
            "category": self.category,
            "size_bytes": self.size_bytes,
            "size": format_bytes(self.size_bytes),
            "size_mib": self.size_mib,
            "size_gib": self.size_gib,
            "md5": self.md5,
            "relative_path": self.relative_path,
            "experiment": self.experiment,
            "recording_id": self.recording_id,
            "retinotopy_id": self.retinotopy_id,
            "label": self.label,
            "suffix": self.suffix,
            "load": f'data.load("{self.name}")',
        }

    def __repr__(self) -> str:
        return f"DataFile({pprint.pformat(self.to_dict(), sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


def _optional(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _data_file(row: Mapping[str, Any]) -> DataFile:
    return DataFile(
        name=str(row["name"]),
        id=int(row["id"]),
        category=str(row["category"]),
        size_bytes=int(row["size_bytes"]),
        md5=str(row["md5"]).lower(),
        relative_path=str(row.get("relative_path", "")),
        experiment=_optional(row.get("experiment")),
        recording_id=_optional(row.get("recording_id")),
        retinotopy_id=_optional(row.get("retinotopy_id")),
    )


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
        raise DriveDataError(f"unsafe catalog path: {value!r}")
    return relative


def _validate_catalog(files: Iterable[DataFile]) -> tuple[DataFile, ...]:
    files = tuple(files)
    if len(files) != EXPECTED_FILE_COUNT:
        raise DriveDataError(
            f"catalog has {len(files)} rows; expected {EXPECTED_FILE_COUNT}"
        )
    if sum(item.size_bytes for item in files) != EXPECTED_TOTAL_BYTES:
        raise DriveDataError("catalog byte total does not match Figshare v2")
    names = [item.name for item in files]
    ids = [item.id for item in files]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        raise DriveDataError("catalog file names and IDs must be unique")
    for item in files:
        if Path(item.name).name != item.name:
            raise DriveDataError(f"catalog name must not contain a path: {item.name!r}")
        if item.size_bytes < 1:
            raise DriveDataError(f"catalog size must be positive: {item.name}")
        if not _MD5_PATTERN.fullmatch(item.md5):
            raise DriveDataError(f"invalid MD5 in catalog: {item.name}")
        _safe_relative_path(item.relative_path)
    return files


def _read_connected_release(root: Path) -> tuple[dict[str, Any], tuple[DataFile, ...]]:
    status_path = root / "TRANSFER_STATUS.json"
    release_path = root / "metadata/RELEASE.json"
    catalog_path = root / "metadata/catalog.csv"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        release = json.loads(release_path.read_text(encoding="utf-8"))
        with catalog_path.open(newline="", encoding="utf-8") as stream:
            files = _validate_catalog(_data_file(row) for row in csv.DictReader(stream))
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DriveDataError(f"shared dataset metadata is incomplete: {error}") from error

    expected_release = {
        "article_id": ARTICLE_ID,
        "version": ARTICLE_VERSION,
        "file_count": EXPECTED_FILE_COUNT,
        "total_bytes": EXPECTED_TOTAL_BYTES,
    }
    for key, expected in expected_release.items():
        if release.get(key) != expected:
            raise DriveDataError(f"release metadata has unexpected {key!r}")
    expected_status = {
        "state": "complete",
        "expected_files": EXPECTED_FILE_COUNT,
        "verified_files": EXPECTED_FILE_COUNT,
        "expected_bytes": EXPECTED_TOTAL_BYTES,
        "verified_bytes": EXPECTED_TOTAL_BYTES,
    }
    for key, expected in expected_status.items():
        if status.get(key) != expected:
            raise DriveDataError(f"dataset transfer is not fully verified ({key})")
    return release, files


def _bundled_release() -> tuple[dict[str, Any], tuple[DataFile, ...]]:
    inventory = load_file_inventory()
    article = dict(inventory["article"])
    release = {
        "article_id": article["id"],
        "version": article["version"],
        "file_count": article["file_count"],
        "total_bytes": article["total_size_bytes"],
        "doi": article["doi"],
        "title": article["title"],
    }
    files = tuple(
        _data_file({**row, "relative_path": f"data/{row['name']}"})
        for row in inventory["files"]
    )
    return release, files


def _default_cache() -> Path:
    if Path("/content").is_dir():
        return Path("/content/janelia_cache")
    return Path(tempfile.gettempdir()) / "zhong2025-cache"


def is_colab() -> bool:
    """Return whether this code is running inside Google Colab."""

    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _mount_colab_drive() -> None:
    """Mount Drive only when the Colab filesystem is not mounted already."""

    if Path("/content/drive/MyDrive").is_dir():
        return
    from google.colab import drive as _colab_drive

    _colab_drive.mount("/content/drive", force_remount=False)


def _enable_colab_widgets() -> None:
    if not is_colab():
        return
    from google.colab import output as _colab_output

    _colab_output.enable_custom_widget_manager()


def _discover_root() -> Path:
    my_drive = Path("/content/drive/MyDrive")
    module_folder = Path(__file__).resolve().parent
    choices = (
        module_folder / DATASET_NAME,
        my_drive / DATASET_SHORTCUT,
        my_drive / WORKSPACE_NAME / DATASET_NAME,
    )
    match = next((path for path in choices if path.is_dir()), None)
    if match is None:
        raise DriveDataError(
            "Shared dataset not found. Add the team workspace or "
            f"{DATASET_SHORTCUT} shortcut to My Drive, then rerun."
        )
    return match


def _md5(path: Path, block_size: int = 8 * 2**20) -> str:
    digest = hashlib.md5()  # noqa: S324 - the published catalog uses MD5
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_verified_numpy(path: Path, item: DataFile) -> Any:
    """Load one checksum-verified release file into ordinary Python values."""

    try:
        if item.suffix == ".npz":
            allow_pickle = item.category == "area_outlines"
            with np.load(path, allow_pickle=allow_pickle) as archive:
                return {name: archive[name] for name in archive.files}
        if item.suffix == ".npy":
            loaded = np.load(path, allow_pickle=True)
            value = loaded.item() if loaded.shape == () else loaded
            if item.category == "reduced_neural" and isinstance(value, Mapping):
                numeric = {
                    str(name): array
                    for name, array in value.items()
                    if isinstance(array, np.ndarray)
                }
                if not {"U", "V"}.issubset(numeric):
                    raise DriveDataError(
                        f"Reduced-neural file has no U/V arrays: {item.name}"
                    )
                return numeric
            return value
    except ModuleNotFoundError as error:
        raise DriveDataError(
            f"{item.name} needs a standard Colab scientific package that is "
            "not available in this runtime. Reconnect with a normal Colab CPU "
            "runtime and try again."
        ) from error
    except (OSError, ValueError, TypeError) as error:
        raise DriveDataError(f"Could not load {item.name}: {error}") from error
    raise DriveDataError(f"Unsupported published file type: {item.name}")


def _value_summary(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"array {tuple(value.shape)} · {value.dtype}"
    if isinstance(value, Mapping):
        return f"dictionary · {len(value)} keys"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} · {len(value)} items"
    return type(value).__name__


def _loaded_html(value: Any) -> str:
    """Return a small shape-only summary; never render a large data object."""

    if isinstance(value, Mapping):
        rows = []
        for index, (name, child) in enumerate(value.items()):
            if index == 8:
                rows.append(
                    f"<tr><td colspan='2' style='opacity:.7'>"
                    f"… {len(value) - index} more keys</td></tr>"
                )
                break
            rows.append(
                "<tr>"
                f"<td style='padding-right:1rem'><code>{html.escape(str(name))}</code></td>"
                f"<td>{html.escape(_value_summary(child))}</td>"
                "</tr>"
            )
        return (
            "<div><b>Loaded dictionary</b>"
            "<table style='margin-top:.35rem'>" + "".join(rows) + "</table></div>"
        )
    return f"<div><b>Loaded</b> · {html.escape(_value_summary(value))}</div>"


@dataclass(frozen=True)
class Recording:
    """One imaging recording and its related published data layers."""

    recording_id: str
    retinotopy_id: str
    experiments: tuple[str, ...]
    files: tuple[DataFile, ...]
    dataset: "Dataset" = field(repr=False, compare=False)

    def _identity(self) -> re.Match[str]:
        match = _RECORDING_PATTERN.fullmatch(self.recording_id)
        if match is None:
            raise DriveDataError(f"Invalid recording identity: {self.recording_id!r}")
        return match

    @property
    def mouse(self) -> str:
        return self._identity().group("mouse")

    @property
    def date(self) -> str:
        return self._identity().group("date").replace("_", "-")

    @property
    def block(self) -> str:
        return self._identity().group("block")

    @property
    def layers(self) -> tuple[str, ...]:
        present = {item.category for item in self.files}
        return tuple(
            layer
            for layer in LAYER_LABELS
            if layer == "behavior" or layer in present
        )

    def file(self, layer: str, *, experiment: str | None = None) -> DataFile:
        """Resolve a scientific layer to its exact published file."""

        layer = "reduced_neural" if layer == "svd" else layer
        if layer not in LAYER_LABELS:
            choices = ", ".join(LAYER_LABELS)
            raise DriveDataError(f"Unknown data layer {layer!r}; choose: {choices}")
        if layer == "behavior":
            if experiment is None:
                if len(self.experiments) != 1:
                    choices = ", ".join(self.experiments)
                    raise DriveDataError(
                        "This recording belongs to several experiments. Choose "
                        f"experiment= from: {choices}"
                    )
                experiment = self.experiments[0]
            if experiment not in self.experiments:
                raise DriveDataError(
                    f"{self.recording_id} is not part of experiment {experiment!r}"
                )
            return self.dataset.file(f"Beh_{experiment}.npy")
        candidates = [item for item in self.files if item.category == layer]
        if len(candidates) != 1:
            raise DriveDataError(
                f"Expected one {layer} file for {self.recording_id}; "
                f"found {len(candidates)}"
            )
        return candidates[0]

    def load(
        self,
        layer: str,
        *,
        experiment: str | None = None,
        max_gib: float = DEFAULT_MAX_GIB,
    ) -> Any:
        """Load one related layer; paths and verification stay hidden."""

        return self.dataset.load(
            recording=self.recording_id,
            layer=layer,
            experiment=experiment,
            max_gib=max_gib,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return identity, provenance, files, and exact access calls."""

        load_calls: dict[str, str] = {
            f"behavior:{experiment}": (
                'session.load("behavior", '
                f'experiment="{experiment}")'
            )
            for experiment in self.experiments
        }
        for layer in ("reduced_neural", "full_neural", "retinotopy"):
            if layer in self.layers:
                load_calls[layer] = f'session.load("{layer}")'

        ordered_files = sorted(
            self.files,
            key=lambda item: (
                item.category != "imaging_behavior",
                item.category,
                item.name,
            ),
        )
        return {
            "recording_id": self.recording_id,
            "mouse": self.mouse,
            "date": self.date,
            "block": self.block,
            "retinotopy_id": self.retinotopy_id,
            "experiments": self.experiments,
            "layers": self.layers,
            "published_file_count": len(self.files),
            "published_bytes": sum(item.size_bytes for item in self.files),
            "published_size": format_bytes(
                sum(item.size_bytes for item in self.files)
            ),
            "files": [item.to_dict() for item in ordered_files],
            "load": load_calls,
            "dataset": {
                "type": type(self.dataset).__name__,
                "connected": self.dataset.connected,
                "access": "session.dataset",
            },
            "availability": (
                "mounted"
                if self.dataset.connected
                else "metadata_only_mount_required_for_arrays"
            ),
        }

    def __repr__(self) -> str:
        return f"Recording({pprint.pformat(self.to_dict(), width=100, sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


class Dataset:
    """Connected release metadata plus safe, selected-file fetching."""

    def __init__(
        self,
        *,
        root: Path | None,
        cache: Path,
        release: Mapping[str, Any],
        files: Iterable[DataFile],
    ) -> None:
        self.root = None if root is None else root.resolve()
        self.cache = cache.resolve()
        self.release = dict(release)
        self.files = tuple(files)
        self._by_name = {item.name: item for item in self.files}
        self._experiment_index = load_experiment_index()
        self._experiment_rows = tuple(experiment_rows(self._experiment_index))

    @property
    def connected(self) -> bool:
        return self.root is not None

    @property
    def folders(self) -> tuple[str, ...]:
        if self.root is None:
            return ()
        return tuple(sorted(path.name for path in self.root.iterdir() if path.is_dir()))

    @property
    def experiments(self) -> tuple[str, ...]:
        """Published imaging experiment labels in the pinned index."""

        return tuple(sorted(self._experiment_index["experiments"]))

    def _public_variables(self) -> tuple[tuple[str, str, str], ...]:
        """Return the small, intentional notebook-facing data surface."""

        location = str(self.root) if self.root is not None else "metadata only"
        folders = ", ".join(self.folders) if self.folders else "none"
        return (
            ("connected", repr(self.connected), "whether release files are mounted"),
            ("root", location, "mounted read-only dataset root"),
            ("cache", str(self.cache), "verified local file cache"),
            ("release", f"dict ({len(self.release)} fields)", "pinned release metadata"),
            (
                "files",
                f"tuple[DataFile] ({len(self.files)} rows)",
                "complete file catalog",
            ),
            (
                "experiments",
                f"tuple[str] ({len(self.experiments)} labels)",
                "published analysis labels",
            ),
            ("folders", folders, "top-level mounted folders"),
        )

    def _public_functions(self) -> tuple[tuple[str, str], ...]:
        """Discover the real public methods instead of maintaining a fake API list."""

        rows: list[tuple[str, str]] = []
        for name, member in inspect.getmembers(type(self), inspect.isfunction):
            if name.startswith("_"):
                continue
            method = getattr(self, name)
            signature = inspect.signature(method)
            parameters = tuple(
                parameter.replace(annotation=inspect.Signature.empty)
                for parameter in signature.parameters.values()
            )
            signature = signature.replace(
                parameters=parameters,
                return_annotation=inspect.Signature.empty,
            )
            summary = inspect.getdoc(member) or ""
            rows.append(
                (f"{name}{signature}", summary.splitlines()[0] if summary else "")
            )
        return tuple(rows)

    def to_dict(self) -> dict[str, Any]:
        """Return a compact, copyable description of the dataset API."""

        total_bytes = sum(item.size_bytes for item in self.files)
        return {
            "connected": self.connected,
            "availability": "mounted" if self.connected else "metadata_only",
            "root": None if self.root is None else str(self.root),
            "cache": str(self.cache),
            "release": dict(self.release),
            "published_file_count": len(self.files),
            "published_bytes": total_bytes,
            "published_size": format_bytes(total_bytes),
            "experiment_count": len(self.experiments),
            "experiments": self.experiments,
            "folders": self.folders,
            "public_attributes": {
                f"data.{name}": {"value": value, "description": description}
                for name, value, description in self._public_variables()
            },
            "api": {
                f"data.{signature}": description
                for signature, description in self._public_functions()
            },
            "examples": {
                "choose": "picker = data.picker()",
                "find_recordings": 'sessions = data.recordings(mouse="TX119")',
                "select_recording": (
                    'session = data.recording("TX119_2023_12_24_1")'
                ),
                "load_layer": 'sample = session.load("reduced_neural")',
            },
        }

    def __repr__(self) -> str:
        return f"Dataset({pprint.pformat(self.to_dict(), width=100, sort_dicts=False)})"

    def _repr_html_(self) -> str:
        """Render the same ordinary dictionary exposed by :meth:`to_dict`."""

        return _dict_html(self.to_dict())

    def figshare(self, *, live: bool = True) -> dict[str, Any]:
        """Return the public Figshare API dictionary or the pinned summary."""

        if live:
            try:
                return fetch_figshare_article()
            except Exception as error:
                warnings.warn(
                    f"Live Figshare API unavailable ({type(error).__name__}); "
                    "using the pinned v2 summary.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        inventory = load_file_inventory()
        return dict(inventory["article"])

    def find(
        self,
        *,
        category: str | None = None,
        contains: str = "",
        experiment: str | None = None,
        recording_id: str | None = None,
        retinotopy_id: str | None = None,
    ) -> list[DataFile]:
        """Filter the catalog without scanning or opening release files."""

        query = contains.casefold().strip()
        return sorted(
            (
                item
                for item in self.files
                if (category is None or item.category == category)
                and (not query or query in item.name.casefold())
                and (experiment is None or item.experiment == experiment)
                and (recording_id is None or item.recording_id == recording_id)
                and (retinotopy_id is None or item.retinotopy_id == retinotopy_id)
            ),
            key=lambda item: (item.size_bytes, item.name),
        )

    def file(self, name: str) -> DataFile:
        """Resolve one exact published filename."""

        if Path(name).name != name:
            raise DriveDataError("Choose a filename from the catalog, not a path")
        try:
            return self._by_name[name]
        except KeyError as error:
            raise DriveDataError(f"File is not in the pinned release: {name!r}") from error

    def recording(self, recording_id: str) -> Recording:
        """Return one recording with its behavior, neural, and cortical files."""

        inventory = {
            "files": [
                {"name": item.name, "size_bytes": item.size_bytes}
                for item in self.files
            ]
        }
        try:
            bundle = recording_bundle(
                recording_id,
                inventory=inventory,
                index=self._experiment_index,
            )
        except (KeyError, ValueError) as error:
            raise DriveDataError(str(error)) from error
        return Recording(
            recording_id=str(bundle["recording_id"]),
            retinotopy_id=str(bundle["retinotopy_id"]),
            experiments=tuple(str(name) for name in bundle["experiments"]),
            files=tuple(self.file(str(row["name"])) for row in bundle["files"]),
            dataset=self,
        )

    def recordings(
        self,
        *,
        experiment: str | None = None,
        mouse: str | None = None,
        contains: str = "",
    ) -> list[Recording]:
        """Find imaging recordings by scientific identity, not Drive paths."""

        if experiment is not None and experiment not in self.experiments:
            raise DriveDataError(f"Unknown experiment: {experiment!r}")
        query = contains.casefold().strip()
        mouse_query = None if mouse is None else mouse.casefold().strip()
        recording_ids = sorted(
            {
                str(row["recording_id"])
                for row in self._experiment_rows
                if (experiment is None or row["experiment"] == experiment)
                and (
                    mouse_query is None
                    or str(row["mouse"]).casefold() == mouse_query
                )
                and (
                    not query
                    or query in str(row["recording_id"]).casefold()
                )
            }
        )
        return [self.recording(recording_id) for recording_id in recording_ids]

    def _load_item(
        self,
        item: DataFile,
        *,
        max_gib: float,
        recording_id: str | None = None,
        layer: str | None = None,
    ) -> tuple[Any, Path]:
        path = self.fetch(item, max_gib=max_gib)
        value = _load_verified_numpy(path, item)
        if layer == "behavior":
            if not isinstance(value, Mapping) or recording_id not in value:
                raise DriveDataError(
                    f"Behavior file {item.name} has no session {recording_id!r}"
                )
            value = value[recording_id]
        return value, path

    def load(
        self,
        selected: DataFile | str | None = None,
        *,
        filename: str | None = None,
        recording: str | None = None,
        layer: str | None = None,
        experiment: str | None = None,
        max_gib: float = DEFAULT_MAX_GIB,
    ) -> Any:
        """Fetch, verify, and load one selected file or recording layer.

        Examples::

            data.load("TX119_2023_12_24_trans.npz")
            data.load(recording="TX119_2023_12_24_1", layer="reduced_neural")
            data.load(
                recording="TX119_2023_12_24_1",
                layer="behavior",
                experiment="unsup_test1",
            )
        """

        direct = selected is not None or filename is not None
        by_recording = recording is not None or layer is not None or experiment is not None
        if direct and by_recording:
            raise DriveDataError(
                "Choose either one filename or recording/layer arguments, not both"
            )
        if not direct and recording is None:
            raise DriveDataError("Choose a filename or recording= with layer=")
        if selected is not None and filename is not None:
            raise DriveDataError("Choose selected or filename, not both")

        recording_id = None
        normalized_layer = None
        if direct:
            item = self.file(
                (selected.name if isinstance(selected, DataFile) else selected)
                if selected is not None
                else str(filename)
            )
        else:
            if layer is None:
                raise DriveDataError("Choose layer= for the selected recording")
            current = self.recording(str(recording))
            normalized_layer = "reduced_neural" if layer == "svd" else layer
            item = current.file(normalized_layer, experiment=experiment)
            recording_id = current.recording_id

        value, _ = self._load_item(
            item,
            max_gib=max_gib,
            recording_id=recording_id,
            layer=normalized_layer,
        )
        return value

    def fetch(
        self,
        selected: DataFile | str,
        *,
        max_gib: float = DEFAULT_MAX_GIB,
    ) -> Path:
        """Copy and verify one selected Drive file into the local cache."""

        if self.root is None:
            raise DriveDataError("Fetching requires a mounted or explicit dataset root")
        if not isinstance(max_gib, (int, float)) or max_gib <= 0:
            raise ValueError("max_gib must be positive")
        item = self.file(selected.name if isinstance(selected, DataFile) else selected)
        max_bytes = int(float(max_gib) * 2**30)
        if item.size_bytes > max_bytes:
            raise DriveDataError(
                f"{item.name} is {item.size_gib:.2f} GiB; increase max_gib "
                "only after checking the Colab disk and memory."
            )

        relative = _safe_relative_path(item.relative_path)
        source = self.root.joinpath(*relative.parts).resolve()
        if source != self.root and self.root not in source.parents:
            raise DriveDataError(f"catalog path escapes the dataset root: {item.name}")
        self.cache.mkdir(parents=True, exist_ok=True)
        destination = self.cache / item.name
        if destination.is_file():
            if destination.stat().st_size == item.size_bytes and _md5(destination) == item.md5:
                return destination
            destination.unlink()

        if not source.is_file():
            raise DriveDataError(f"Selected Drive file is unavailable: {source}")
        if source.stat().st_size != item.size_bytes:
            raise DriveDataError(f"Drive file size does not match the catalog: {item.name}")

        if shutil.disk_usage(self.cache).free < item.size_bytes * 1.2:
            raise DriveDataError("Not enough local disk space for the selected file")
        partial = destination.with_suffix(destination.suffix + ".partial")
        partial.unlink(missing_ok=True)
        digest = hashlib.md5()  # noqa: S324 - the published catalog uses MD5
        copied = 0
        try:
            with source.open("rb") as src, partial.open("wb") as dst:
                while block := src.read(8 * 2**20):
                    dst.write(block)
                    digest.update(block)
                    copied += len(block)
            if copied != item.size_bytes or digest.hexdigest() != item.md5:
                raise DriveDataError(
                    f"Copied file did not match the release catalog: {item.name}"
                )
            os.replace(partial, destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return destination

    def picker(self):
        """Return a compact notebook UI for selecting and loading Janelia data."""

        try:
            import ipywidgets as widgets
        except ImportError as error:  # pragma: no cover - package dependency
            raise DriveDataError("The notebook picker requires ipywidgets") from error

        full = widgets.Layout(width="100%")
        experiment_control = widgets.Dropdown(
            description="Experiment",
            options=[("Choose an experiment", ""), *[(name, name) for name in self.experiments]],
            value="",
            layout=full,
        )
        recording_control = widgets.Dropdown(
            description="Recording",
            options=[],
            disabled=True,
            layout=full,
        )
        layer_control = widgets.Dropdown(
            description="Data layer",
            options=[(label, name) for name, label in LAYER_LABELS.items()],
            value="reduced_neural",
            layout=full,
        )
        session_box = widgets.VBox(
            [experiment_control, recording_control, layer_control],
            layout=widgets.Layout(width="100%", grid_gap="5px", display="flex"),
        )

        categories = sorted({item.category for item in self.files})
        category_control = widgets.Dropdown(
            description="Category",
            options=[
                ("All published files", "all"),
                *[(CATEGORY_LABELS.get(name, name.replace("_", " ")), name) for name in categories],
            ],
            value="all",
            layout=full,
        )
        search_control = widgets.Text(
            description="Find",
            placeholder="mouse, date, experiment, or filename",
            layout=full,
        )
        file_control = widgets.Dropdown(
            description="File",
            options=[],
            layout=full,
        )
        file_box = widgets.VBox(
            [category_control, search_control, file_control],
            layout=widgets.Layout(width="100%", grid_gap="5px", display="none"),
        )
        mode_control = widgets.Dropdown(
            description="Choose by",
            options=[("Recording", "recording"), ("Any file", "file")],
            value="recording",
            layout=full,
        )

        selection = widgets.HTML()
        load_button = widgets.Button(
            description="Load selected data",
            icon="download",
            button_style="primary",
            disabled=True,
            layout=widgets.Layout(width="170px", min_width="170px"),
        )
        status = widgets.HTML(layout=widgets.Layout(flex="1 1 auto", min_width="0"))
        summary = widgets.HTML()
        panel = widgets.VBox(
            [
                widgets.HTML(
                    "<div style='font-weight:700;font-size:1rem'>Janelia data picker</div>"
                    "<div style='opacity:.75'>Choose one recording layer or any published file.</div>"
                ),
                mode_control,
                session_box,
                file_box,
                selection,
                widgets.HBox(
                    [load_button, status],
                    layout=widgets.Layout(width="100%", align_items="center", grid_gap="10px"),
                ),
                summary,
            ],
            layout=widgets.Layout(width="100%", grid_gap="7px"),
        )
        panel.value = None
        panel.path = None
        panel.selected = None
        panel.last_error = None
        panel.controls = {
            "experiment": experiment_control,
            "recording": recording_control,
            "layer": layer_control,
            "category": category_control,
            "search": search_control,
            "file": file_control,
            "mode": mode_control,
            "load": load_button,
        }

        def clear_loaded() -> bool:
            had_value = panel.value is not None or panel.path is not None
            panel.value = None
            panel.path = None
            panel.last_error = None
            summary.value = ""
            return had_value

        def update_recordings(_change: Any = None) -> None:
            experiment = experiment_control.value
            matches = self.recordings(experiment=experiment) if experiment else []
            recording_control.options = [
                (
                    f"{recording.mouse} · {recording.date} · block {recording.block}",
                    recording.recording_id,
                )
                for recording in matches
            ]
            recording_control.value = matches[0].recording_id if matches else None
            recording_control.disabled = not matches

        def update_files(_change: Any = None) -> None:
            category = category_control.value
            matches = self.find(
                category=None if category == "all" else category,
                contains=search_control.value,
            )
            file_control.options = [(item.label, item.name) for item in matches]
            file_control.value = matches[0].name if matches else None
            file_control.disabled = not matches

        def current_selection():
            if mode_control.value == "recording":
                if not experiment_control.value or not recording_control.value:
                    return None
                current = self.recording(recording_control.value)
                item = current.file(
                    layer_control.value,
                    experiment=experiment_control.value,
                )
                kwargs = {
                    "recording": current.recording_id,
                    "layer": layer_control.value,
                    "experiment": experiment_control.value,
                }
                return item, kwargs
            if not file_control.value:
                return None
            return self.file(file_control.value), {"filename": file_control.value}

        def refresh(_change: Any = None) -> None:
            changed = clear_loaded()
            chosen = current_selection()
            if chosen is None:
                panel.selected = None
                load_button.disabled = True
                selection.value = (
                    "<span style='opacity:.7'>Choose an experiment and recording, "
                    "or switch to Any file.</span>"
                )
                status.value = ""
                return
            item, kwargs = chosen
            panel.selected = item
            arguments = ", ".join(
                f"{name}={value!r}" for name, value in kwargs.items()
            )
            too_large = item.size_gib > DEFAULT_MAX_GIB
            load_button.disabled = too_large
            category = CATEGORY_LABELS.get(item.category, item.category.replace("_", " "))
            note = (
                "<br><span style='color:#c77'>Large file: use reduced neural data "
                "for normal Colab exploration, or call data.load with an intentional "
                "max_gib limit.</span>"
                if too_large
                else ""
            )
            selection.value = (
                "<div style='padding:.55rem .7rem;border:1px solid #7775;border-radius:8px'>"
                f"<b>{html.escape(item.name)}</b> · {html.escape(format_bytes(item.size_bytes))}"
                f"<br><span style='opacity:.75'>{html.escape(category)}</span>"
                f"<br><code>sample = data.load({html.escape(arguments)})</code>{note}</div>"
            )
            status.value = (
                "Selection changed — load again."
                if changed
                else "Ready to load one verified file."
            )

        def update_mode(_change: Any = None) -> None:
            by_recording = mode_control.value == "recording"
            session_box.layout.display = "flex" if by_recording else "none"
            file_box.layout.display = "none" if by_recording else "flex"
            if not by_recording and not file_control.options:
                update_files()

        def perform_load() -> Any:
            chosen = current_selection()
            if chosen is None:
                return None
            item, kwargs = chosen
            load_button.disabled = True
            status.value = "Loading and verifying the selected file…"
            panel.last_error = None
            try:
                recording_id = kwargs.get("recording")
                layer = kwargs.get("layer")
                value, path = self._load_item(
                    item,
                    max_gib=DEFAULT_MAX_GIB,
                    recording_id=recording_id,
                    layer=layer,
                )
            except Exception as error:
                panel.value = None
                panel.path = None
                panel.last_error = error
                status.value = (
                    "<span role='alert' style='color:#d66'>"
                    f"{html.escape(str(error))}</span>"
                )
                load_button.disabled = item.size_gib > DEFAULT_MAX_GIB
                return None
            panel.value = value
            panel.path = path
            panel.selected = item
            status.value = "Loaded. The result is available as <code>picker.value</code>."
            summary.value = _loaded_html(value)
            load_button.disabled = False
            return value

        def load_clicked(_button: Any) -> None:
            perform_load()

        experiment_control.observe(update_recordings, names="value")
        experiment_control.observe(refresh, names="value")
        recording_control.observe(refresh, names="value")
        layer_control.observe(refresh, names="value")
        category_control.observe(update_files, names="value")
        category_control.observe(refresh, names="value")
        search_control.observe(update_files, names="value")
        search_control.observe(refresh, names="value")
        file_control.observe(refresh, names="value")
        mode_control.observe(update_mode, names="value")
        mode_control.observe(refresh, names="value")
        load_button.on_click(load_clicked)
        panel.load = perform_load

        update_recordings()
        refresh()
        return panel


def connect(
    *,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    mount: bool = True,
) -> Dataset:
    """Connect to the shared release, or return metadata-only access locally."""

    cache_path = _default_cache() if cache is None else Path(cache)
    explicit = root or os.environ.get("ZHONG2025_DATASET_ROOT")
    if explicit is not None:
        dataset_root = Path(explicit)
    else:
        if not is_colab():
            release, files = _bundled_release()
            return Dataset(root=None, cache=cache_path, release=release, files=files)
        if mount:
            _mount_colab_drive()
        dataset_root = _discover_root()

    if not dataset_root.is_dir():
        raise DriveDataError(f"Dataset root does not exist: {dataset_root}")
    release, files = _read_connected_release(dataset_root)
    return Dataset(root=dataset_root, cache=cache_path, release=release, files=files)


def setup(
    *,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    mount: bool = True,
    report: bool = True,
) -> Dataset:
    """Prepare the shared dataset for a notebook and return its high-level API."""

    data = connect(root=root, cache=cache, mount=mount)
    _enable_colab_widgets()
    if report:
        print(data)
        if data.connected:
            folders = ", ".join(data.folders) or "(none)"
            print(f"Top-level folders: {folders}")
        else:
            print("Metadata-only mode: use Colab with the team Drive to fetch files.")
    return data


__all__ = [
    "DataFile",
    "Dataset",
    "DriveDataError",
    "Recording",
    "REPRESENTATION_API_VERSION",
    "connect",
    "is_colab",
    "setup",
]
