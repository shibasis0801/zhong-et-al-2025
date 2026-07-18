#!/usr/bin/env python3
"""Build safe, compact metadata assets for the dataset-atlas notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from zhong2025.atlas import (
    ARTICLE_ID,
    ARTICLE_VERSION,
    EXPECTED_FILE_COUNT,
    EXPECTED_IMAGING_MICE,
    EXPECTED_IMAGING_RECORDINGS,
    EXPECTED_TOTAL_BYTES,
    classify_file_name,
    recording_id_from_file_name,
)


EXPERIMENT_INDEX_MD5 = "2259b9e5a6cea8987d871c7fbe90a8f9"
EXPERIMENT_INDEX_SIZE = 21_194


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json_atomic(value: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def build_inventory(article_path: Path) -> dict[str, Any]:
    article = json.loads(article_path.read_text(encoding="utf-8"))
    files = article["files"]
    if article.get("id") != ARTICLE_ID or article.get("version") != ARTICLE_VERSION:
        raise ValueError("article metadata is not the pinned Figshare v2 release")
    if len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(f"expected {EXPECTED_FILE_COUNT} Figshare files")
    if sum(int(entry["size"]) for entry in files) != EXPECTED_TOTAL_BYTES:
        raise ValueError("Figshare article byte total differs from the pinned release")

    normalized = []
    for entry in files:
        name = str(entry["name"])
        category = classify_file_name(name)
        recording_id = recording_id_from_file_name(name)
        retinotopy_id = name.removesuffix("_trans.npz") if category == "retinotopy" else None
        experiment = (
            name.removeprefix("Beh_").removesuffix(".npy")
            if category in {"imaging_behavior", "faster_learning_behavior"}
            else None
        )
        normalized.append(
            {
                "id": int(entry["id"]),
                "name": name,
                "size_bytes": int(entry["size"]),
                "md5": str(entry.get("computed_md5") or entry.get("supplied_md5")),
                "url": str(entry["download_url"]),
                "mimetype": str(entry.get("mimetype", "")),
                "category": category,
                "recording_id": recording_id,
                "retinotopy_id": retinotopy_id,
                "experiment": experiment,
            }
        )
    return {
        "schema_version": 1,
        "article": {
            "id": ARTICLE_ID,
            "version": ARTICLE_VERSION,
            "title": article["title"],
            "doi": "10.25378/janelia.28811129.v2",
            "license": article.get("license", {}).get("name", "CC BY 4.0"),
            "published_date": article["published_date"],
            "modified_date": article["modified_date"],
            "file_count": len(normalized),
            "total_size_bytes": EXPECTED_TOTAL_BYTES,
            "api_url": f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{ARTICLE_VERSION}",
            "source_json_sha256": _digest(article_path, "sha256"),
        },
        "files": normalized,
    }


def build_experiment_index(
    source_path: Path,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    if source_path.stat().st_size != EXPERIMENT_INDEX_SIZE:
        raise ValueError("unexpected Imaging_Exp_info.npy size")
    if _digest(source_path, "md5") != EXPERIMENT_INDEX_MD5:
        raise ValueError("unexpected Imaging_Exp_info.npy MD5")
    source = np.load(source_path, allow_pickle=True).item()
    if not isinstance(source, dict) or len(source) != 23:
        raise ValueError("expected 23 imaging experiment labels")

    experiments: dict[str, list[dict[str, Any]]] = {}
    recording_ids: set[str] = set()
    mice: set[str] = set()
    associations = 0
    for experiment, raw_entries in source.items():
        entries = []
        for raw in raw_entries:
            block = str(raw["blk"])
            recording_id = f"{raw['mname']}_{raw['datexp']}_{block}"
            retinotopy_id = f"{raw['mname']}_{raw['datexp']}"
            entries.append(
                {
                    "recording_id": recording_id,
                    "retinotopy_id": retinotopy_id,
                    "source": _json_safe(raw),
                }
            )
            recording_ids.add(recording_id)
            mice.add(str(raw["mname"]))
            associations += 1
        experiments[str(experiment)] = entries

    if associations != 142:
        raise ValueError("expected 142 experiment-to-recording associations")
    if len(recording_ids) != EXPECTED_IMAGING_RECORDINGS:
        raise ValueError("expected 89 unique imaging recordings")
    if len(mice) != EXPECTED_IMAGING_MICE:
        raise ValueError("expected 19 unique imaging mice")

    inventory_recordings = {
        str(entry["recording_id"])
        for entry in inventory["files"]
        if entry["category"] == "full_neural"
    }
    if recording_ids != inventory_recordings:
        raise ValueError("experiment index and full-neural inventory do not agree")

    return {
        "schema_version": 1,
        "source": {
            "article_id": ARTICLE_ID,
            "article_version": ARTICLE_VERSION,
            "file_id": 54_183_854,
            "file_name": "Imaging_Exp_info.npy",
            "size_bytes": EXPERIMENT_INDEX_SIZE,
            "md5": EXPERIMENT_INDEX_MD5,
            "sha256": _digest(source_path, "sha256"),
        },
        "summary": {
            "experiment_labels": len(experiments),
            "associations": associations,
            "unique_recordings": len(recording_ids),
            "unique_mice": len(mice),
        },
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--article-json", type=Path, required=True)
    parser.add_argument("--experiment-index", type=Path, required=True)
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=Path("zhong2025/assets/figshare-v2-inventory.json"),
    )
    parser.add_argument(
        "--experiment-output",
        type=Path,
        default=Path("zhong2025/assets/imaging-experiment-index.json"),
    )
    args = parser.parse_args()
    inventory = build_inventory(args.article_json)
    experiment_index = build_experiment_index(args.experiment_index, inventory)
    _write_json_atomic(inventory, args.inventory_output)
    _write_json_atomic(experiment_index, args.experiment_output)
    print(
        f"wrote {args.inventory_output} ({len(inventory['files'])} files) and "
        f"{args.experiment_output} ({experiment_index['summary']['unique_recordings']} recordings)"
    )


if __name__ == "__main__":
    main()
