import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SPEC = importlib.util.spec_from_file_location(
    "drive_sync", Path("scripts/drive_sync.py")
)
drive_sync = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(drive_sync)


def run_git(repo, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_drive_config_contains_only_public_ids_and_small_safety_policy():
    config = drive_sync.load_config()

    assert config["workspace_url"].endswith(config["root_folder_id"])
    assert config["current_folder_id"] != config["incoming_folder_id"]
    assert config["incoming_extensions"] == [".ipynb"]
    assert config["incoming_max_files"] == 50
    assert config["current_max_delete"] < 39
    assert config["new_file_patterns"] == ["notebooks/team_changes/*.ipynb"]
    serialized = json.dumps(config).lower()
    for secret_field in ("access_token", "refresh_token", "client_secret"):
        assert secret_field not in serialized


def test_committed_snapshot_is_exactly_git_tracked_files(tmp_path):
    manifest = drive_sync.create_snapshot("HEAD", tmp_path)
    tracked = set(run_git(Path.cwd(), "ls-tree", "-r", "--name-only", "HEAD").splitlines())

    assert {row["path"] for row in manifest["files"]} == tracked
    assert manifest["original_tree"] == drive_sync.PINNED_ORIGINAL_TREE
    assert "original/Figures.ipynb" in tracked
    assert not any(path.startswith((".git/", ".venv/", "data/")) for path in tracked)
    assert (tmp_path / drive_sync.MANIFEST_NAME).is_file()


def test_snapshot_rejects_original_tree_drift(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-q")
    run_git(repo, "config", "user.name", "Test")
    run_git(repo, "config", "user.email", "test@example.com")
    (repo / "original").mkdir()
    (repo / "original" / "source.py").write_text("upstream\n")
    (repo / "README.md").write_text("team\n")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-qm", "base")
    pinned = run_git(repo, "rev-parse", "HEAD:original")

    (repo / "original" / "source.py").write_text("changed\n")
    run_git(repo, "add", "original/source.py")
    run_git(repo, "commit", "-qm", "drift")

    monkeypatch.setattr(drive_sync, "ROOT", repo)
    monkeypatch.setattr(drive_sync, "PINNED_ORIGINAL_TREE", pinned)
    with pytest.raises(drive_sync.SyncError, match="original/"):
        drive_sync.create_snapshot("HEAD", tmp_path / "snapshot")


def test_notebooks_map_to_a_dedicated_new_contributions_folder(tmp_path):
    config = drive_sync.load_config()
    rows = [
        {"Path": "team-experiment.ipynb", "Size": 10, "IsDir": False},
        {"Path": "alice/variation.ipynb", "Size": 10, "IsDir": False},
    ]

    mappings = drive_sync.validate_listing(
        rows, config, repo_root=tmp_path, tracked=set()
    )

    assert mappings["team-experiment.ipynb"].as_posix() == (
        "notebooks/team_changes/team-experiment.ipynb"
    )
    assert mappings["alice/variation.ipynb"].as_posix() == (
        "notebooks/team_changes/alice/variation.ipynb"
    )


@pytest.mark.parametrize(
    "path",
    [
        "original/utils.py",
        ".venv/lib/python/site-packages/pytest/__init__.py",
        "data/generated.py",
        "credentials.json",
        "notes.md",
    ],
)
def test_non_notebook_drive_targets_are_rejected(tmp_path, path):
    config = drive_sync.load_config()
    with pytest.raises(drive_sync.SyncError, match="unsupported type|protected path"):
        drive_sync.validate_listing(
            [{"Path": path, "Size": 10, "IsDir": False}],
            config,
            repo_root=tmp_path,
            tracked=set(),
        )


def test_existing_or_tracked_destination_is_never_overwritten(tmp_path):
    config = drive_sync.load_config()
    target = tmp_path / "notebooks/team_changes/team.ipynb"
    target.parent.mkdir(parents=True)
    target.write_text("local")
    row = [{"Path": "team.ipynb", "Size": 10, "IsDir": False}]

    with pytest.raises(drive_sync.SyncError, match="cannot overwrite"):
        drive_sync.validate_listing(row, config, repo_root=tmp_path, tracked=set())

    target.unlink()
    with pytest.raises(drive_sync.SyncError, match="cannot overwrite"):
        drive_sync.validate_listing(
            row,
            config,
            repo_root=tmp_path,
            tracked={"NOTEBOOKS/TEAM_CHANGES/TEAM.IPYNB"},
        )


def test_case_collisions_size_and_file_count_are_rejected(tmp_path):
    config = drive_sync.load_config()
    with pytest.raises(drive_sync.SyncError, match="duplicate"):
        drive_sync.validate_listing(
            [
                {"Path": "Same.ipynb", "Size": 10, "IsDir": False},
                {"Path": "same.ipynb", "Size": 10, "IsDir": False},
            ],
            config,
            repo_root=tmp_path,
            tracked=set(),
        )
    with pytest.raises(drive_sync.SyncError, match="exceeds"):
        drive_sync.validate_listing(
            [
                {
                    "Path": "large.ipynb",
                    "Size": config["incoming_max_bytes"] + 1,
                    "IsDir": False,
                }
            ],
            config,
            repo_root=tmp_path,
            tracked=set(),
        )
    too_many = [
        {"Path": f"{index}.ipynb", "Size": 1, "IsDir": False}
        for index in range(config["incoming_max_files"] + 1)
    ]
    with pytest.raises(drive_sync.SyncError, match="file limit"):
        drive_sync.validate_listing(
            too_many, config, repo_root=tmp_path, tracked=set()
        )
    too_large_together = [
        {
            "Path": f"large-{index}.ipynb",
            "Size": config["incoming_max_bytes"],
            "IsDir": False,
        }
        for index in range(5)
    ]
    with pytest.raises(drive_sync.SyncError, match="total limit"):
        drive_sync.validate_listing(
            too_large_together, config, repo_root=tmp_path, tracked=set()
        )


def test_workspace_folder_ids_must_match_root_listing(monkeypatch):
    config = drive_sync.load_config()
    rows = [
        {"Name": "Current", "ID": config["current_folder_id"]},
        {"Name": "Team changes", "ID": config["incoming_folder_id"]},
    ]
    monkeypatch.setattr(drive_sync, "remote_rows", lambda *_args, **_kwargs: rows)
    drive_sync.verify_workspace_targets(config, "gdrive")

    rows[0]["ID"] = "wrong"
    with pytest.raises(drive_sync.SyncError, match="points 'Current'"):
        drive_sync.verify_workspace_targets(config, "gdrive")


def test_nonempty_current_requires_repository_manifest(monkeypatch):
    config = drive_sync.load_config()
    monkeypatch.setattr(
        drive_sync,
        "remote_rows",
        lambda *_args, **_kwargs: [{"Path": "unrelated.txt", "Size": 1}],
    )
    with pytest.raises(drive_sync.SyncError, match="not an initialized mirror"):
        drive_sync.verify_current(config, "gdrive", "HEAD")


def test_pull_previews_then_imports_and_clears_inbox(tmp_path, monkeypatch, capsys):
    config = drive_sync.load_config()
    rows = [{"Path": "team.ipynb", "Size": 8, "IsDir": False}]
    mappings = {
        "team.ipynb": drive_sync.target_for_remote_path("team.ipynb")
    }
    calls = []

    monkeypatch.setattr(drive_sync, "ROOT", tmp_path)
    monkeypatch.setattr(drive_sync, "rclone_binary", lambda: "rclone")
    monkeypatch.setattr(drive_sync, "verify_workspace_targets", lambda *_: None)
    monkeypatch.setattr(drive_sync, "list_incoming", lambda *_: (rows, mappings))
    monkeypatch.setattr(drive_sync, "changed_paths", lambda: set())

    def fake_command(arguments, *, capture=False, cwd=None):
        calls.append(arguments)
        if arguments[1] == "copy":
            download = Path(arguments[3])
            (download / "team.ipynb").write_text("notebook")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(drive_sync, "command", fake_command)

    drive_sync.pull(config, remote="gdrive", apply=False)
    destination = tmp_path / "notebooks/team_changes/team.ipynb"
    assert not destination.exists()
    assert "Preview only" in capsys.readouterr().out

    drive_sync.pull(config, remote="gdrive", apply=True)
    assert destination.read_text() == "notebook"
    assert any(call[1] == "deletefile" for call in calls)
    assert "cleared the Drive inbox" in capsys.readouterr().out


def test_hook_is_informational_and_workflow_scopes_credentials():
    hook = Path(".githooks/post-commit").read_text()
    workflow = Path(".github/workflows/drive-mirror.yml").read_text()

    assert "push --commit" not in hook
    assert "pushed to GitHub" in hook
    assert "fetch-depth: 0" in workflow
    assert "GDRIVE_RCLONE_TOKEN" in workflow
    assert "GDRIVE_RCLONE_CLIENT_ID" in workflow
    assert "RCLONE_CONFIG_GDRIVE_TOKEN" not in workflow.split("steps:", 1)[0]
    assert "access_token" not in hook + workflow
    assert "refresh_token" not in hook + workflow
