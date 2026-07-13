"""Safe metadata helpers for understanding the complete Figshare v2 release."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


ASSET_ROOT = Path(__file__).resolve().parent / "assets"
DEFAULT_INVENTORY = ASSET_ROOT / "figshare-v2-inventory.json"
DEFAULT_EXPERIMENT_INDEX = ASSET_ROOT / "imaging-experiment-index.json"

ARTICLE_ID = 28811129
ARTICLE_VERSION = 2
EXPECTED_FILE_COUNT = 297
EXPECTED_TOTAL_BYTES = 452_233_500_962
EXPECTED_IMAGING_RECORDINGS = 89
EXPECTED_IMAGING_MICE = 19

FASTER_LEARNING_BEHAVIOR = {
    "Beh_no_pretrain.npy",
    "Beh_pretrain_on_grat_image.npy",
    "Beh_pretrain_on_nat_image.npy",
}

CATEGORY_LABELS = {
    "imaging_behavior": "Imaging-study behavior bundles",
    "faster_learning_behavior": "Faster-learning behavior bundles",
    "full_neural": "Full deconvolved neural activity",
    "reduced_neural": "Reduced 400-PC neural data",
    "retinotopy": "Per-recording retinotopy",
    "imaging_experiment_index": "Imaging experiment index",
    "area_outlines": "Shared visual-area outlines",
    "behavior_example": "Example behavior derivative",
    "neural_example": "Example raw neural derivative",
}

_RECORDING_SUFFIXES = ("_neural_data.npy", "_SVD_dec.npy", "_example_raw_spk.npy")
_RECORDING_PATTERN = re.compile(r"^(.+_\d{4}_\d{2}_\d{2})_(\d+)$")


def classify_file_name(name: str) -> str:
    """Assign one semantic category to every file in the pinned release."""

    if name == "Imaging_Exp_info.npy":
        return "imaging_experiment_index"
    if name == "areas.npz":
        return "area_outlines"
    if name == "example_bef_and_aft_learning_behavior.npy":
        return "behavior_example"
    if name.endswith("_example_raw_spk.npy"):
        return "neural_example"
    if name in FASTER_LEARNING_BEHAVIOR:
        return "faster_learning_behavior"
    if name.startswith("Beh_") and name.endswith(".npy"):
        return "imaging_behavior"
    if name.endswith("_neural_data.npy"):
        return "full_neural"
    if name.endswith("_SVD_dec.npy"):
        return "reduced_neural"
    if name.endswith("_trans.npz"):
        return "retinotopy"
    raise ValueError(f"unrecognized Figshare v2 filename: {name}")


def recording_id_from_file_name(name: str) -> str | None:
    """Return mouse_date_block for a recording-specific neural file."""

    for suffix in _RECORDING_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def retinotopy_id_for_recording(recording_id: str) -> str:
    """Convert mouse_date_block to the mouse_date retinotopy join key."""

    match = _RECORDING_PATTERN.fullmatch(recording_id)
    if match is None:
        raise ValueError(f"invalid recording identity: {recording_id!r}")
    return match.group(1)


def format_bytes(value: int) -> str:
    """Format exact bytes with a readable binary unit."""

    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or unit == "TiB":
            return f"{amount:,.1f} {unit}" if unit != "B" else f"{int(amount):,} B"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_file_inventory(path: str | Path = DEFAULT_INVENTORY) -> dict[str, Any]:
    """Load and validate the pickle-free snapshot of all 297 published files."""

    inventory = _read_json(path)
    article = inventory.get("article", {})
    files = inventory.get("files", [])
    if article.get("id") != ARTICLE_ID or article.get("version") != ARTICLE_VERSION:
        raise ValueError("inventory is not the pinned Figshare article v2")
    if len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(f"inventory should contain {EXPECTED_FILE_COUNT} files")
    if sum(int(entry["size_bytes"]) for entry in files) != EXPECTED_TOTAL_BYTES:
        raise ValueError("inventory byte total differs from the pinned release")
    names = [str(entry["name"]) for entry in files]
    ids = [int(entry["id"]) for entry in files]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        raise ValueError("inventory file names and IDs must be unique")
    for entry in files:
        expected = classify_file_name(str(entry["name"]))
        if entry.get("category") != expected:
            raise ValueError(f"wrong category for {entry['name']}: {entry.get('category')}")
    return inventory


def load_experiment_index(
    path: str | Path = DEFAULT_EXPERIMENT_INDEX,
) -> dict[str, Any]:
    """Load the safe JSON projection of ``Imaging_Exp_info.npy``."""

    index = _read_json(path)
    source = index.get("source", {})
    if source.get("article_id") != ARTICLE_ID or source.get("article_version") != ARTICLE_VERSION:
        raise ValueError("experiment index is not from the pinned Figshare article v2")
    if index.get("summary", {}).get("unique_recordings") != EXPECTED_IMAGING_RECORDINGS:
        raise ValueError("experiment index should describe 89 unique recordings")
    if index.get("summary", {}).get("unique_mice") != EXPECTED_IMAGING_MICE:
        raise ValueError("experiment index should describe 19 imaging mice")
    return index


def inventory_summary(
    inventory: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Summarize file count and storage by semantic category."""

    inventory = load_file_inventory() if inventory is None else inventory
    grouped: dict[str, dict[str, Any]] = {}
    for entry in inventory["files"]:
        category = str(entry["category"])
        row = grouped.setdefault(
            category,
            {
                "category": category,
                "label": CATEGORY_LABELS[category],
                "file_count": 0,
                "size_bytes": 0,
            },
        )
        row["file_count"] += 1
        row["size_bytes"] += int(entry["size_bytes"])
    return sorted(grouped.values(), key=lambda row: row["size_bytes"], reverse=True)


def experiment_rows(index: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Flatten experiment-to-recording associations into readable rows."""

    index = load_experiment_index() if index is None else index
    rows = []
    for experiment, entries in index["experiments"].items():
        for entry in entries:
            source = entry["source"]
            rows.append(
                {
                    "experiment": experiment,
                    "recording_id": entry["recording_id"],
                    "retinotopy_id": entry["retinotopy_id"],
                    "mouse": source["mname"],
                    "date": source["datexp"],
                    "block": str(source["blk"]),
                    "reward_type": source.get("rewType"),
                    "stimulus_ids": source.get("stim_id", []),
                    "stimulus_type": source.get("stimtype"),
                }
            )
    return rows


def experiment_semantics(experiment: str) -> dict[str, Any]:
    """Decode the controlled experiment label into its intended role."""

    if experiment.startswith("sup_"):
        cohort = "task"
    elif experiment.startswith("unsup_"):
        cohort = "unrewarded exposure"
    elif experiment.startswith("naive_"):
        cohort = "naive"
    elif experiment.endswith("_grating"):
        cohort = "grating control"
    else:
        raise ValueError(f"unknown experiment label: {experiment}")

    stage = next(
        (
            label
            for token, label in (
                ("train1", "Train 1"),
                ("test1", "Test 1"),
                ("train2", "Train 2"),
                ("test2", "Test 2"),
                ("test3", "Test 3"),
            )
            if token in experiment
        ),
        None,
    )
    if stage is None:
        raise ValueError(f"experiment label has no recognized stage: {experiment}")
    if "before" in experiment:
        moment = "before"
    elif "after" in experiment:
        moment = "after"
    else:
        moment = "test snapshot"

    if "test3" in experiment:
        roles = [0, 2, 3, 5, 6]
    elif "test2" in experiment:
        roles = [0, 2, 3, 4]
    elif "train2_after" in experiment:
        roles = [0, 2, 3]
    elif "train2_before" in experiment or "test1" in experiment:
        roles = [0, 1, 2, 3]
    else:
        roles = [0, 2]
    return {"cohort": cohort, "stage": stage, "moment": moment, "stimulus_roles": roles}


def experiment_recordings(
    experiment: str,
    index: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return unique recording IDs associated with one experiment label."""

    index = load_experiment_index() if index is None else index
    try:
        entries = index["experiments"][experiment]
    except KeyError as error:
        raise KeyError(f"unknown imaging experiment: {experiment}") from error
    return sorted({str(entry["recording_id"]) for entry in entries})


def recording_bundle(
    recording_id: str,
    *,
    inventory: Mapping[str, Any] | None = None,
    index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one recording to behavior bundles, neural data, SVD, and retinotopy."""

    retinotopy_id = retinotopy_id_for_recording(recording_id)
    inventory = load_file_inventory() if inventory is None else inventory
    index = load_experiment_index() if index is None else index
    experiments = sorted(
        {
            row["experiment"]
            for row in experiment_rows(index)
            if row["recording_id"] == recording_id
        }
    )
    if not experiments:
        raise KeyError(f"recording is absent from the experiment index: {recording_id}")
    names = {
        f"{recording_id}_neural_data.npy",
        f"{recording_id}_SVD_dec.npy",
        f"{retinotopy_id}_trans.npz",
        *(f"Beh_{experiment}.npy" for experiment in experiments),
    }
    files_by_name = {str(entry["name"]): entry for entry in inventory["files"]}
    missing = names - files_by_name.keys()
    if missing:
        raise KeyError(f"recording bundle is missing published files: {sorted(missing)}")
    files = [files_by_name[name] for name in sorted(names)]
    return {
        "recording_id": recording_id,
        "retinotopy_id": retinotopy_id,
        "experiments": experiments,
        "files": files,
        "total_bytes": sum(int(entry["size_bytes"]) for entry in files),
    }


def filter_inventory(
    entries: Iterable[Mapping[str, Any]],
    *,
    category: str = "all",
    search: str = "",
) -> list[Mapping[str, Any]]:
    """Filter inventory rows for the notebook's lightweight catalog browser."""

    query = search.casefold().strip()
    return [
        entry
        for entry in entries
        if (category == "all" or entry.get("category") == category)
        and (not query or query in str(entry.get("name", "")).casefold())
    ]
