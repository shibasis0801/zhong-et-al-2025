"""Protect the byte-for-byte upstream repository snapshot."""

from pathlib import Path
import subprocess


UPSTREAM_COMMIT = "ba64ac697f5d9914926baac79399e80707a5f3a6"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = REPOSITORY_ROOT / "original"


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_original_is_exact_upstream_snapshot() -> None:
    upstream_files = _git_bytes(
        "ls-tree", "-r", "--name-only", UPSTREAM_COMMIT
    ).decode().splitlines()
    snapshot_files = sorted(
        str(path.relative_to(ORIGINAL_ROOT))
        for path in ORIGINAL_ROOT.rglob("*")
        if path.is_file()
    )

    assert snapshot_files == sorted(upstream_files)
    for relative_path in upstream_files:
        expected = _git_bytes("show", f"{UPSTREAM_COMMIT}:{relative_path}")
        assert (ORIGINAL_ROOT / relative_path).read_bytes() == expected


def test_upstream_code_has_no_duplicate_root_copy() -> None:
    duplicated_code = {
        "Figures.ipynb",
        "S6.py",
        "data_process_script.ipynb",
        "fig1.py",
        "fig2.py",
        "fig3.py",
        "fig4.py",
        "fig5.py",
        "utils.py",
    }
    assert not any((REPOSITORY_ROOT / name).exists() for name in duplicated_code)
