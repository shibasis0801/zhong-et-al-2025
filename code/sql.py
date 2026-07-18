"""Convenient entry points for the Zhong et al. Pandas/DuckDB interface."""

from __future__ import annotations

from pathlib import Path

from database import ZhongDB


def connect(
    *,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    database: str | Path | None = None,
    mount: bool = True,
) -> ZhongDB:
    """Read the release and return its unified Pandas/SQL interface."""

    return ZhongDB(root=root, cache=cache, database=database, mount=mount)


def setup(
    *,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    database: str | Path | None = None,
    mount: bool = True,
    report: bool = True,
) -> ZhongDB:
    """Connect to the release and optionally print its table summary."""

    db = connect(root=root, cache=cache, database=database, mount=mount)
    if report:
        print(f"DuckDB: {db.database_path}")
        print(db.schema().to_string(index=False))
    return db


__all__ = ["ZhongDB", "connect", "setup"]
