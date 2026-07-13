"""Pinned Figshare manifest, safe downloads, and atlas-example loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ASSET_ROOT = Path(__file__).resolve().parent / "assets"
DEFAULT_MANIFEST = ASSET_ROOT / "figshare-v2-curated.json"
DEFAULT_DEMO = ASSET_ROOT / "tx119_atlas_demo.npz"
MAX_PROFILE_BYTES = 2_000_000_000
ALLOWED_KINDS = {
    "behavior",
    "behavior_metadata",
    "retinotopy",
    "retinotopy_outline",
    "svd_dec",
}


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load the immutable, curated subset manifest."""

    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("article", {}).get("version") != 2:
        raise ValueError("this workflow requires the pinned Figshare v2 manifest")
    return manifest


def _files_by_key(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    files = manifest.get("files", [])
    mapping = {entry["key"]: entry for entry in files}
    if len(mapping) != len(files):
        raise ValueError("manifest file keys must be unique")
    return mapping


def profile_summary(
    profile: str,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return exact files and byte totals before a download begins."""

    manifest = load_manifest() if manifest is None else dict(manifest)
    try:
        keys = manifest["profiles"][profile]
    except KeyError as error:
        available = ", ".join(sorted(manifest.get("profiles", {})))
        raise KeyError(f"unknown profile {profile!r}; choose one of: {available}") from error
    files = _files_by_key(manifest)
    selected = [files[key] for key in keys]
    return {
        "profile": profile,
        "files": selected,
        "total_bytes": sum(int(entry["size_bytes"]) for entry in selected),
    }


def _md5(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - required by the published manifest
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retrying_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_file(
    entry: Mapping[str, Any],
    root: str | Path,
    *,
    session: requests.Session | None = None,
    timeout: tuple[float, float] = (10.0, 120.0),
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Stream one checksum-verified file through a temporary `.part` file."""

    root = Path(root).resolve()
    relative_path = Path(str(entry["path"]))
    if (
        relative_path == Path(".")
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise ValueError(f"download path must stay relative to the cache: {relative_path}")
    destination = (root / relative_path).resolve()
    if root != destination and root not in destination.parents:
        raise ValueError(f"download path escapes the cache root: {relative_path}")
    expected_size = int(entry["size_bytes"])
    expected_md5 = str(entry["md5"]).lower()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if destination.stat().st_size == expected_size and _md5(destination) == expected_md5:
            return destination
        raise ValueError(f"cached file failed size or MD5 validation: {destination}")

    part = destination.with_suffix(destination.suffix + ".part")
    if part.exists():
        part.unlink()
    client = _retrying_session() if session is None else session
    response = None
    digest = hashlib.md5()  # noqa: S324 - required by the published manifest
    received = 0
    try:
        response = client.get(entry["url"], stream=True, timeout=timeout)
        response.raise_for_status()
        with part.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                stream.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if received > expected_size:
                    raise ValueError(f"download exceeded declared size for {entry['key']}")
        if received != expected_size:
            raise ValueError(
                f"downloaded {received} bytes for {entry['key']}; expected {expected_size}"
            )
        if digest.hexdigest().lower() != expected_md5:
            raise ValueError(f"MD5 mismatch for {entry['key']}")
        os.replace(part, destination)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        if response is not None:
            response.close()
        if session is None:
            client.close()
    return destination


def download_profile(
    profile: str,
    root: str | Path,
    *,
    max_total_bytes: int = 2_000_000_000,
    session: requests.Session | None = None,
) -> dict[str, Path]:
    """Download a declared reduced-data profile, never the full article.

    `max_total_bytes` is an explicit guardrail.  This curated manifest contains
    no raw-neural or all-data profile.
    """

    manifest = load_manifest()
    article = manifest.get("article", {})
    if article.get("id") != 28811129 or article.get("version") != 2:
        raise ValueError("refusing a manifest outside pinned Figshare article v2")
    summary = profile_summary(profile, manifest=manifest)
    if max_total_bytes < 1:
        raise ValueError("max_total_bytes must be positive")
    effective_limit = min(max_total_bytes, MAX_PROFILE_BYTES)
    if summary["total_bytes"] > effective_limit:
        raise ValueError(
            f"profile {profile!r} is {summary['total_bytes']:,} bytes, above the "
            f"{effective_limit:,}-byte limit"
        )
    for entry in summary["files"]:
        if entry.get("kind") not in ALLOWED_KINDS:
            raise ValueError(f"refusing unsupported data kind: {entry.get('kind')!r}")
        parsed = urlparse(str(entry.get("url", "")))
        if parsed.scheme != "https" or parsed.hostname != "ndownloader.figshare.com":
            raise ValueError(f"refusing non-Figshare URL for {entry.get('key')!r}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(root).free
    if summary["total_bytes"] > free_bytes:
        raise OSError(
            f"profile needs {summary['total_bytes']:,} bytes but only "
            f"{free_bytes:,} bytes are free at {root}"
        )
    downloaded = {}
    for entry in summary["files"]:
        downloaded[entry["key"]] = download_file(
            entry, root, session=session
        )
    return downloaded


def load_atlas_demo(path: str | Path = DEFAULT_DEMO) -> dict[str, Any]:
    """Load the real, compact atlas example without enabling pickle."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"atlas example not found at {path}; run scripts/build_atlas_demo.py"
        )
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(data.pop("metadata_json").item()))
    data["metadata"] = metadata
    return data
