#!/usr/bin/env python3
"""Publish committed files to Drive and safely review team contributions."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "drive-sync.json"
MANIFEST_NAME = ".drive-mirror-manifest.json"
REPOSITORY = "shibasis0801/zhong-et-al-2025"
PINNED_ORIGINAL_TREE = "4444b06adcda43002721cfd121a696a3b9fab89b"


class SyncError(RuntimeError):
    """Raised when a sync operation would be incomplete or unsafe."""


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise SyncError("unsupported drive-sync.json schema")
    required = {
        "root_folder_id",
        "current_folder_id",
        "incoming_folder_id",
        "default_remote",
        "remote_env",
        "incoming_max_bytes",
        "incoming_max_total_bytes",
        "incoming_max_files",
        "current_max_delete",
        "incoming_extensions",
        "new_file_patterns",
        "deny_pull_patterns",
    }
    missing = sorted(required - set(config))
    if missing:
        raise SyncError(f"drive-sync.json is missing: {', '.join(missing)}")
    return config


def command(
    arguments: Sequence[str],
    *,
    capture: bool = False,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            list(arguments),
            cwd=cwd,
            check=True,
            capture_output=capture,
            text=capture,
        )
    except FileNotFoundError as error:
        raise SyncError(f"required command not found: {arguments[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip() if capture else ""
        suffix = f": {detail}" if detail else ""
        raise SyncError(f"command failed: {' '.join(arguments)}{suffix}") from error


def git(*arguments: str, capture: bool = True, cwd: Path = ROOT) -> str:
    result = command(["git", *arguments], capture=capture, cwd=cwd)
    return result.stdout.strip() if capture else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_git_archive(commit: str, destination: Path) -> str:
    revision = git("rev-parse", "--verify", f"{commit}^{{commit}}")
    try:
        payload = subprocess.run(
            ["git", "archive", "--format=tar", revision],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise SyncError(f"could not archive commit {revision}") from error

    destination_root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as bundle:
        for member in bundle.getmembers():
            if not (member.isfile() or member.isdir()):
                raise SyncError(f"commit contains unsupported link: {member.name}")
            target = (destination / member.name).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise SyncError(f"unsafe archived path: {member.name}")
        bundle.extractall(destination)
    return revision


def create_snapshot(commit: str, destination: Path) -> dict[str, Any]:
    revision = git("rev-parse", "--verify", f"{commit}^{{commit}}")
    original_tree = git("rev-parse", f"{revision}:original")
    if original_tree != PINNED_ORIGINAL_TREE:
        raise SyncError(
            "original/ differs from the pinned upstream snapshot; refusing to publish"
        )
    revision = _extract_git_archive(revision, destination)
    files = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        relative = path.relative_to(destination).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "md5": md5(path),
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "commit": revision,
        "commit_time": git("show", "-s", "--format=%cI", revision),
        "original_tree": original_tree,
        "files": files,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def rclone_binary() -> str:
    executable = os.environ.get("RCLONE_BIN", "rclone")
    if shutil.which(executable) is None:
        raise SyncError(
            "rclone is not installed. See https://rclone.org/install/ and then "
            "run `rclone config` once for Google Drive."
        )
    return executable


def remote_name(config: Mapping[str, Any], override: str | None) -> str:
    name = override or os.environ.get(
        str(config["remote_env"]), str(config["default_remote"])
    )
    return name.removesuffix(":")


def drive_flags(folder_id: str) -> list[str]:
    return [
        "--drive-root-folder-id",
        folder_id,
        "--drive-skip-gdocs",
        "--drive-skip-shortcuts",
    ]


def remote_rows(
    remote: str,
    folder_id: str,
    *,
    directories: bool = False,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    arguments = [
        rclone_binary(),
        "lsjson",
        f"{remote}:",
        *drive_flags(folder_id),
    ]
    arguments.extend(["--recursive"] if recursive else ["--max-depth", "1"])
    if directories:
        arguments.append("--dirs-only")
    else:
        arguments.extend(["--files-only", "--hash"])
    result = command(arguments, capture=True)
    try:
        rows = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as error:
        raise SyncError("rclone returned an invalid Drive listing") from error
    if not isinstance(rows, list):
        raise SyncError("rclone returned an invalid Drive listing")
    return rows


def verify_workspace_targets(config: Mapping[str, Any], remote: str) -> None:
    rows = remote_rows(remote, str(config["root_folder_id"]), directories=True)
    expected = {
        "Current": str(config["current_folder_id"]),
        "Team changes": str(config["incoming_folder_id"]),
    }
    for name, folder_id in expected.items():
        matches = [
            row for row in rows if str(row.get("Name") or row.get("Path")) == name
        ]
        if not matches:
            raise SyncError(f"Drive workspace is missing {name!r}")
        if len(matches) != 1:
            raise SyncError(f"Drive workspace has duplicate {name!r} folders")
        row = matches[0]
        actual_id = str(row.get("ID", ""))
        if not actual_id:
            raise SyncError(f"rclone did not return the Drive ID for {name!r}")
        if actual_id != folder_id:
            raise SyncError(
                f"drive-sync.json points {name!r} at {folder_id}, not {actual_id}"
            )


def read_remote_json(remote: str, folder_id: str, name: str) -> dict[str, Any]:
    result = command(
        [rclone_binary(), "cat", f"{remote}:{name}", *drive_flags(folder_id)],
        capture=True,
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SyncError(f"Drive contains invalid {name}") from error
    if not isinstance(value, dict):
        raise SyncError(f"Drive contains invalid {name}")
    return value


def verify_current(
    config: Mapping[str, Any], remote: str, revision: str
) -> None:
    rows = remote_rows(
        remote, str(config["current_folder_id"]), recursive=True
    )
    if not rows:
        return
    paths = {str(row.get("Path", "")).casefold(): row for row in rows}
    if len(paths) != len(rows):
        raise SyncError("Drive Current contains case-colliding paths")
    manifest_row = paths.get(MANIFEST_NAME.casefold())
    if manifest_row is None:
        raise SyncError(
            "Drive Current is not an initialized mirror; refusing destructive sync"
        )
    manifest = read_remote_json(
        remote, str(config["current_folder_id"]), MANIFEST_NAME
    )
    if (
        manifest.get("repository") != REPOSITORY
        or manifest.get("original_tree") != PINNED_ORIGINAL_TREE
    ):
        raise SyncError("Drive Current belongs to a different or invalid mirror")

    expected = {
        str(item["path"]).casefold(): item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and "path" in item
    }
    expected[MANIFEST_NAME.casefold()] = None
    if set(paths) != set(expected):
        raise SyncError("Drive Current has drifted from its mirror manifest")
    for path, item in expected.items():
        if item is None:
            continue
        if int(paths[path].get("Size", -1)) != int(item.get("bytes", -2)):
            raise SyncError(f"Drive Current has drifted: {item['path']}")
        hashes = paths[path].get("Hashes") or {}
        remote_md5 = hashes.get("md5") or hashes.get("MD5")
        if remote_md5 and str(remote_md5).lower() != str(item.get("md5", "")):
            raise SyncError(f"Drive Current has drifted: {item['path']}")

    previous = str(manifest.get("commit", ""))
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", previous, revision],
        cwd=ROOT,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise SyncError(
            "refusing to replace Drive Current with an older or unrelated commit"
        )


def push(config: Mapping[str, Any], *, commit: str, remote: str, apply: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="zhong-drive-push-") as temporary:
        snapshot = Path(temporary)
        manifest = create_snapshot(commit, snapshot)
        verify_workspace_targets(config, remote)
        verify_current(config, remote, str(manifest["commit"]))
        arguments = [
            rclone_binary(),
            "sync",
            str(snapshot),
            f"{remote}:",
            *drive_flags(str(config["current_folder_id"])),
            "--checksum",
            "--delete-after",
            "--drive-use-trash=true",
            "--max-delete",
            str(config["current_max_delete"]),
            "--create-empty-src-dirs",
        ]
        if not apply:
            arguments.append("--dry-run")
        print(
            f"{'Publishing' if apply else 'Previewing'} "
            f"{len(manifest['files'])} tracked files from {manifest['commit'][:12]}"
        )
        command(arguments, capture=False)
        if not apply:
            print("Preview only. Add --apply to update Drive.")


def _matches(path: PurePosixPath, patterns: Iterable[str]) -> bool:
    value = path.as_posix().lower()
    return any(fnmatch.fnmatch(value, pattern.lower()) for pattern in patterns)


def target_for_remote_path(raw_path: str) -> PurePosixPath:
    if not raw_path or "\\" in raw_path or any(c in raw_path for c in "\0\r\n"):
        raise SyncError(f"unsafe Drive path: {raw_path!r}")
    remote = PurePosixPath(raw_path)
    if remote.is_absolute() or any(part in {"", ".", ".."} for part in remote.parts):
        raise SyncError(f"unsafe Drive path: {raw_path!r}")
    return PurePosixPath("notebooks", "team_changes") / remote


def tracked_paths(repo_root: Path = ROOT) -> set[str]:
    try:
        raw = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SyncError("could not list Git-tracked paths") from error
    values = [item.decode("utf-8") for item in raw.split(b"\0")]
    return {value.casefold() for value in values if value}


def validate_listing(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    repo_root: Path = ROOT,
    tracked: set[str] | None = None,
) -> dict[str, PurePosixPath]:
    mappings: dict[str, PurePosixPath] = {}
    targets: set[str] = set()
    allowed = {str(item).lower() for item in config["incoming_extensions"]}
    denied = [str(item) for item in config["deny_pull_patterns"]]
    new_patterns = [str(item) for item in config["new_file_patterns"]]
    limit = int(config["incoming_max_bytes"])
    max_total = int(config["incoming_max_total_bytes"])
    max_files = int(config["incoming_max_files"])
    tracked_normalized = tracked if tracked is not None else tracked_paths(repo_root)
    tracked_normalized = {item.casefold() for item in tracked_normalized}
    total = 0
    file_count = 0

    for row in rows:
        if row.get("IsDir"):
            continue
        raw_path = str(row.get("Path", ""))
        target = target_for_remote_path(raw_path)
        normalized = target.as_posix().casefold()
        if (
            raw_path.casefold() in {item.casefold() for item in mappings}
            or normalized in targets
        ):
            raise SyncError(f"duplicate Drive path: {raw_path}")
        if _matches(target, denied):
            raise SyncError(f"Drive change targets protected path: {target}")
        if target.suffix.lower() not in allowed:
            raise SyncError(f"Drive change has unsupported type: {target}")
        size = int(row.get("Size", -1))
        if size < 0 or size > limit:
            raise SyncError(f"Drive change exceeds the {limit}-byte limit: {target}")
        file_count += 1
        total += size
        if file_count > max_files:
            raise SyncError(f"Team changes exceeds the {max_files}-file limit")
        if total > max_total:
            raise SyncError(
                f"Team changes exceeds the {max_total}-byte total limit"
            )
        local = repo_root / target.as_posix()
        if not _matches(target, new_patterns):
            raise SyncError(f"Drive contributions must be new notebooks: {target}")
        if normalized in tracked_normalized or local.exists() or local.is_symlink():
            raise SyncError(
                f"Drive contributions cannot overwrite a repository file: {target}"
            )
        parent = local.parent
        while parent != repo_root and repo_root in parent.parents:
            if parent.is_symlink():
                raise SyncError(f"Drive change crosses a local symlink: {target}")
            parent = parent.parent
        mappings[raw_path] = target
        targets.add(normalized)
    return mappings


def list_incoming(
    config: Mapping[str, Any], remote: str
) -> tuple[list[dict[str, Any]], dict[str, PurePosixPath]]:
    rows = remote_rows(
        remote, str(config["incoming_folder_id"]), recursive=True
    )
    return rows, validate_listing(rows, config)


def changed_paths() -> set[str]:
    output = git("status", "--porcelain", "--untracked-files=all")
    return {line[3:].strip() for line in output.splitlines() if len(line) >= 4}


def require_clean_worktree() -> None:
    changed = changed_paths()
    if changed:
        preview = ", ".join(sorted(changed)[:5])
        raise SyncError(
            "pull --apply requires a clean worktree; commit or stash first: " + preview
        )


def pull(config: Mapping[str, Any], *, remote: str, apply: bool) -> None:
    verify_workspace_targets(config, remote)
    rows, mappings = list_incoming(config, remote)
    if not mappings:
        print("Team changes is empty. Nothing to pull.")
        return
    if apply:
        require_clean_worktree()

    with tempfile.TemporaryDirectory(prefix="zhong-drive-pull-") as temporary:
        download = Path(temporary)
        file_list = download / ".rclone-files"
        file_list.write_text("".join(f"{path}\n" for path in mappings), encoding="utf-8")
        arguments = [
            rclone_binary(),
            "copy",
            f"{remote}:",
            str(download),
            *drive_flags(str(config["incoming_folder_id"])),
            "--files-from-raw",
            str(file_list),
            "--max-size",
            str(config["incoming_max_bytes"]),
            "--max-transfer",
            str(config["incoming_max_total_bytes"]),
        ]
        command(arguments, capture=False)

        dirty = changed_paths()
        changes: list[tuple[str, str, Path, Path]] = []
        for raw_path, target in sorted(mappings.items()):
            source = download / raw_path
            destination = ROOT / target.as_posix()
            if not source.is_file():
                raise SyncError(f"rclone did not download expected file: {raw_path}")
            if source.stat().st_size > int(config["incoming_max_bytes"]):
                raise SyncError(f"downloaded file is too large: {raw_path}")
            if destination.exists() and sha256(source) == sha256(destination):
                status = "="
            elif target.as_posix() in dirty:
                status = "!"
            else:
                status = "M" if destination.exists() else "A"
            changes.append((status, target.as_posix(), source, destination))

        for status, path, _, _ in changes:
            label = {"=": "same", "A": "add", "M": "update", "!": "conflict"}[status]
            print(f"{label:8} {path}")
        conflicts = [path for status, path, _, _ in changes if status == "!"]
        if conflicts:
            raise SyncError("local changes conflict with Drive: " + ", ".join(conflicts))

        actionable = [item for item in changes if item[0] in {"A", "M"}]
        if not apply:
            print(f"Preview only: {len(actionable)} file(s) would change. Add --apply.")
            return
        for _, _, source, destination in actionable:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for raw_path in sorted(mappings):
            command(
                [
                    rclone_binary(),
                    "deletefile",
                    f"{remote}:{raw_path}",
                    *drive_flags(str(config["incoming_folder_id"])),
                ],
                capture=False,
            )
        print(
            f"Applied {len(actionable)} file(s) and cleared the Drive inbox. "
            "Review `git diff`, run tests, then commit."
        )


def doctor(config: Mapping[str, Any], *, remote: str) -> None:
    revision = git("rev-parse", "--short", "HEAD")
    print(f"Repository: {ROOT} ({revision})")
    print(f"rclone: {rclone_binary()}")
    print(f"remote: {remote}: (credentials remain outside this repository)")
    verify_workspace_targets(config, remote)
    current = remote_rows(
        remote, str(config["current_folder_id"]), recursive=True
    )
    if current:
        verify_current(config, remote, git("rev-parse", "HEAD"))
    state = "initialized" if current else "empty; first publish will initialize it"
    print(f"Drive: ready (Current is {state}; Team changes is an inbox)")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--remote", help="override the configured rclone remote name")
    subcommands = root.add_subparsers(dest="command", required=True)

    push_parser = subcommands.add_parser("push", help="publish a committed snapshot")
    push_parser.add_argument("--commit", default="HEAD")
    push_parser.add_argument("--apply", action="store_true")

    pull_parser = subcommands.add_parser("pull", help="preview or apply Team changes")
    pull_parser.add_argument("--apply", action="store_true")

    subcommands.add_parser("doctor", help="verify local and Drive setup")
    return root


def main(arguments: Sequence[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    try:
        config = load_config()
        remote = remote_name(config, options.remote)
        if options.command == "push":
            push(config, commit=options.commit, remote=remote, apply=options.apply)
        elif options.command == "pull":
            pull(config, remote=remote, apply=options.apply)
        else:
            doctor(config, remote=remote)
    except SyncError as error:
        print(f"Drive sync stopped: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
