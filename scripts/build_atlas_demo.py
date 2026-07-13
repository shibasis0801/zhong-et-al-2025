#!/usr/bin/env python3
"""Rebuild the committed compact atlas example from official Figshare files."""

from __future__ import annotations

import argparse
from pathlib import Path

from zhong2025.demo import build_atlas_demo, save_atlas_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("zhong2025/assets/tx119_atlas_demo.npz"),
    )
    args = parser.parse_args()
    data = build_atlas_demo(
        args.data_root / "beh" / "Beh_unsup_test1.npy",
        args.data_root / "SVD_dec" / "TX119_2023_12_24_1_SVD_dec.npy",
        args.data_root / "retinotopy" / "TX119_2023_12_24_trans.npz",
    )
    output = save_atlas_demo(data, args.output)
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
