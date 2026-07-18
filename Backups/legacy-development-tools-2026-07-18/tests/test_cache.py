from __future__ import annotations

from pathlib import Path

import pytest

from zhong2025.cache import Cache


def test_cache_allows_a_nested_relative_namespace(tmp_path: Path) -> None:
    cache = Cache(root=tmp_path / "root", namespace="graph/example", version="v2")

    assert cache.dir == (tmp_path / "root/graph/example/v2").resolve()
    assert cache.dir.is_dir()
    assert cache.to_dict() == cache.info()
    assert "'entries': 0" in repr(cache)
    assert "entries" in cache._repr_html_()


@pytest.mark.parametrize("field", ["namespace", "version"])
def test_cache_rejects_absolute_namespace_and_version(tmp_path: Path, field: str) -> None:
    kwargs = {field: str(tmp_path / "outside")}

    with pytest.raises(ValueError, match=field):
        Cache(root=tmp_path / "root", **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("namespace", "../escape"),
        ("namespace", "safe/../../escape"),
        ("version", "../escape"),
        ("version", "safe/../../escape"),
    ],
)
def test_cache_rejects_parent_traversal(tmp_path: Path, field: str, value: str) -> None:
    kwargs = {field: value}

    with pytest.raises(ValueError, match=f"{field}.*parent traversal"):
        Cache(root=tmp_path / "root", **kwargs)
    assert not (tmp_path / "escape").exists()


def test_cache_rejects_a_namespace_symlink_that_resolves_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes the cache root"):
        Cache(root=root, namespace="linked", version="v1")
    assert not (outside / "v1").exists()
