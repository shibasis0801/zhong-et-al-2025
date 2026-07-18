from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import drive


def _catalog_rows(
    payload: bytes = b"real release bytes",
    *,
    name: str = "sample_trans.npz",
    category: str = "retinotopy",
    relative_path: str = "data/retinotopy/sample_trans.npz",
    retinotopy_id: str = "sample",
):
    rows = [
        {
            "name": name,
            "id": 1,
            "category": category,
            "experiment": "",
            "recording_id": "",
            "retinotopy_id": retinotopy_id,
            "size_bytes": len(payload),
            "md5": hashlib.md5(payload).hexdigest(),
            "relative_path": relative_path,
            "url": "https://ndownloader.figshare.com/files/1",
        }
    ]
    remaining = drive.EXPECTED_TOTAL_BYTES - len(payload)
    for index in range(1, drive.EXPECTED_FILE_COUNT):
        size = remaining - (drive.EXPECTED_FILE_COUNT - 2) if index == 1 else 1
        rows.append(
            {
                "name": f"recording_{index:03d}_neural_data.npy",
                "id": index + 1,
                "category": "full_neural",
                "experiment": "",
                "recording_id": f"recording_{index:03d}",
                "retinotopy_id": "",
                "size_bytes": size,
                "md5": f"{index:032x}",
                "relative_path": f"data/spk/recording_{index:03d}_neural_data.npy",
                "url": f"https://ndownloader.figshare.com/files/{index + 1}",
            }
        )
    assert sum(int(row["size_bytes"]) for row in rows) == drive.EXPECTED_TOTAL_BYTES
    return rows


def _write_release(root: Path, rows, payload: bytes = b"real release bytes"):
    (root / "metadata").mkdir(parents=True)
    source_relative = Path(rows[0]["relative_path"])
    if source_relative.is_absolute() or ".." in source_relative.parts:
        source_relative = Path("data/retinotopy/sample_trans.npz")
    source = root / source_relative
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    status = {
        "state": "complete",
        "expected_files": drive.EXPECTED_FILE_COUNT,
        "verified_files": drive.EXPECTED_FILE_COUNT,
        "expected_bytes": drive.EXPECTED_TOTAL_BYTES,
        "verified_bytes": drive.EXPECTED_TOTAL_BYTES,
    }
    release = {
        "article_id": drive.ARTICLE_ID,
        "version": drive.ARTICLE_VERSION,
        "file_count": drive.EXPECTED_FILE_COUNT,
        "total_bytes": drive.EXPECTED_TOTAL_BYTES,
    }
    (root / "TRANSFER_STATUS.json").write_text(json.dumps(status))
    (root / "metadata/RELEASE.json").write_text(json.dumps(release))
    with (root / "metadata/catalog.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return source


@pytest.fixture
def release_root(tmp_path):
    root = tmp_path / "release"
    payload = b"real release bytes"
    rows = _catalog_rows(payload)
    source = _write_release(root, rows, payload)
    return root, rows, source, payload


def test_connect_reads_only_metadata_and_exposes_a_small_api(release_root, tmp_path):
    root, _, _, _ = release_root
    data = drive.connect(root=root, cache=tmp_path / "cache", mount=False)

    assert data.connected
    assert len(data.files) == 297
    assert data.release["article_id"] == 28811129
    assert data.folders == ("data", "metadata")
    assert "421.2 GiB" in repr(data)
    assert data.file("sample_trans.npz").size_mib < 1
    assert data.find(category="retinotopy", contains="SAMPLE") == [
        data.file("sample_trans.npz")
    ]
    assert not data.find(experiment="not-present")


def test_dataset_display_lists_variables_functions_and_first_steps(
    release_root, tmp_path
):
    root, _, _, _ = release_root
    data = drive.connect(root=root, cache=tmp_path / "cache", mount=False)

    summary = data.to_dict()
    assert summary["connected"] is True
    assert summary["published_file_count"] == 297
    assert summary["published_bytes"] == drive.EXPECTED_TOTAL_BYTES
    assert "data.picker()" in summary["api"]
    assert any(call.startswith("data.recordings(") for call in summary["api"])
    assert any(call.startswith("data.load(") for call in summary["api"])
    assert summary["examples"]["load_layer"] == (
        'sample = session.load("reduced_neural")'
    )

    text = repr(data)
    assert "'published_file_count': 297" in text
    assert "'published_size': '421.2 GiB'" in text
    assert "'public_attributes'" in text
    assert "'api'" in text

    rich = data._repr_html_()
    assert "published_file_count" in rich
    assert "experiments" in rich
    assert "api" in rich
    assert "data.picker()" in rich
    assert "data.recordings(mouse=&quot;TX119&quot;)" in rich
    assert "session.load(&quot;reduced_neural&quot;)" in rich


def test_local_connect_is_metadata_only(tmp_path, monkeypatch):
    monkeypatch.delenv("ZHONG2025_DATASET_ROOT", raising=False)
    data = drive.connect(cache=tmp_path / "cache", mount=False)
    assert not data.connected
    assert len(data.files) == 297
    assert data.folders == ()
    with pytest.raises(drive.DriveDataError, match="requires a mounted"):
        data.fetch("TX119_2023_12_24_trans.npz")


def test_setup_returns_the_dataset_and_reports_the_connection(
    release_root, tmp_path, capsys
):
    root, _, _, _ = release_root

    data = drive.setup(
        root=root,
        cache=tmp_path / "cache",
        mount=False,
    )

    output = capsys.readouterr().out
    assert data.connected
    assert "'published_file_count': 297" in output
    assert "Top-level folders: data, metadata" in output


def test_setup_can_be_quiet(release_root, tmp_path, capsys):
    root, _, _, _ = release_root

    data = drive.setup(
        root=root,
        cache=tmp_path / "cache",
        mount=False,
        report=False,
    )

    assert data.connected
    assert capsys.readouterr().out == ""


def test_setup_explains_metadata_only_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ZHONG2025_DATASET_ROOT", raising=False)

    data = drive.setup(cache=tmp_path / "cache", mount=False)

    assert not data.connected
    assert "Metadata-only mode" in capsys.readouterr().out


def test_discovery_accepts_the_dataset_next_to_drive_module(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    dataset = workspace / drive.DATASET_NAME
    dataset.mkdir(parents=True)
    monkeypatch.setattr(drive, "__file__", str(workspace / "drive.py"))

    assert drive._discover_root() == dataset


def test_connect_with_mount_false_never_mounts_colab(
    release_root, tmp_path, monkeypatch
):
    root, _, _, _ = release_root
    monkeypatch.setattr(drive, "is_colab", lambda: True)
    monkeypatch.setattr(drive, "_discover_root", lambda: root)
    monkeypatch.setattr(
        drive,
        "_mount_colab_drive",
        lambda: (_ for _ in ()).throw(AssertionError("must not mount")),
    )

    data = drive.connect(cache=tmp_path / "cache", mount=False)

    assert data.root == root.resolve()


def test_mount_helper_does_not_remount_an_existing_drive(monkeypatch):
    target = Path("/content/drive/MyDrive")
    original_is_dir = Path.is_dir

    def is_dir(path):
        return True if path == target else original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    drive._mount_colab_drive()


def test_mount_helper_mounts_once_when_drive_is_missing(monkeypatch):
    calls = []
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(
        mount=lambda path, force_remount: calls.append((path, force_remount))
    )
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: False if path == Path("/content/drive/MyDrive") else False,
    )

    drive._mount_colab_drive()

    assert calls == [("/content/drive", False)]


def test_fetch_copies_only_one_selected_file_and_reuses_verified_cache(
    release_root, tmp_path
):
    root, _, source, payload = release_root
    cache = tmp_path / "cache"
    data = drive.connect(root=root, cache=cache, mount=False)
    destination = data.fetch("sample_trans.npz")

    assert destination.read_bytes() == payload
    assert list(cache.iterdir()) == [destination]

    source.unlink()
    assert data.fetch(data.file("sample_trans.npz")) == destination


def test_fetch_repairs_a_corrupt_module_cache(release_root, tmp_path):
    root, _, _, payload = release_root
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "sample_trans.npz").write_bytes(b"bad")
    data = drive.connect(root=root, cache=cache, mount=False)
    assert data.fetch("sample_trans.npz").read_bytes() == payload


def test_fetch_rejects_large_files_before_opening_the_source(release_root, tmp_path):
    root, rows, _, _ = release_root
    large = next(row for row in rows if int(row["size_bytes"]) > 2**30)
    data = drive.connect(root=root, cache=tmp_path / "cache", mount=False)
    with pytest.raises(drive.DriveDataError, match="increase max_gib"):
        data.fetch(large["name"], max_gib=1)


def test_default_file_limit_is_ten_gib():
    assert drive.DEFAULT_MAX_GIB == 10.0


def test_fetch_cleans_partial_file_after_checksum_failure(tmp_path):
    root = tmp_path / "release"
    payload = b"real release bytes"
    rows = _catalog_rows(payload)
    rows[0]["md5"] = "0" * 32
    _write_release(root, rows, payload)
    cache = tmp_path / "cache"
    data = drive.connect(root=root, cache=cache, mount=False)

    with pytest.raises(drive.DriveDataError, match="did not match"):
        data.fetch("sample_trans.npz")
    assert not list(cache.glob("*.partial"))
    assert not (cache / "sample_trans.npz").exists()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "name": rows[0]["name"]}), "unique"),
        (lambda rows: rows.__setitem__(0, {**rows[0], "relative_path": "../escape"}), "unsafe"),
        (lambda rows: rows.__setitem__(0, {**rows[0], "md5": "bad"}), "invalid MD5"),
    ],
)
def test_connect_rejects_malformed_catalog(tmp_path, mutate, message):
    root = tmp_path / "release"
    payload = b"real release bytes"
    rows = _catalog_rows(payload)
    mutate(rows)
    _write_release(root, rows, payload)
    with pytest.raises(drive.DriveDataError, match=message):
        drive.connect(root=root, cache=tmp_path / "cache", mount=False)


def test_fetch_rejects_a_symlink_escape(release_root, tmp_path):
    root, rows, source, _ = release_root
    source.unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sample_trans.npz").write_bytes(b"real release bytes")
    (root / "data/retinotopy").rmdir()
    (root / "data/retinotopy").symlink_to(outside, target_is_directory=True)
    data = drive.connect(root=root, cache=tmp_path / "cache", mount=False)
    with pytest.raises(drive.DriveDataError, match="escapes"):
        data.fetch(rows[0]["name"])


def test_file_requires_an_exact_catalog_name(release_root, tmp_path):
    root, _, _, _ = release_root
    data = drive.connect(root=root, cache=tmp_path / "cache", mount=False)
    with pytest.raises(drive.DriveDataError, match="not a path"):
        data.file("data/retinotopy/sample_trans.npz")
    with pytest.raises(drive.DriveDataError, match="not in the pinned"):
        data.file("missing.npy")


def test_load_returns_a_plain_dictionary_from_one_verified_npz(tmp_path):
    buffer = io.BytesIO()
    np.savez(buffer, xy=np.arange(8).reshape(4, 2), area=np.array([1, 1, 2, 2]))
    payload = buffer.getvalue()
    root = tmp_path / "release"
    rows = _catalog_rows(payload)
    _write_release(root, rows, payload)
    data = drive.connect(root=root, cache=tmp_path / "cache", mount=False)

    loaded = data.load("sample_trans.npz")

    assert set(loaded) == {"xy", "area"}
    np.testing.assert_array_equal(loaded["xy"], np.arange(8).reshape(4, 2))
    assert not isinstance(loaded, np.lib.npyio.NpzFile)


def test_load_unwraps_one_verified_npy_without_exposing_pickle_options(tmp_path):
    buffer = io.BytesIO()
    expected = {"trial": np.arange(5), "label": "published"}
    np.save(buffer, expected, allow_pickle=True)
    payload = buffer.getvalue()
    root = tmp_path / "release"
    rows = _catalog_rows(
        payload,
        name="Beh_demo.npy",
        category="imaging_behavior",
        relative_path="data/beh/Beh_demo.npy",
        retinotopy_id="",
    )
    _write_release(root, rows, payload)
    data = drive.connect(root=root, cache=tmp_path / "cache", mount=False)

    loaded = data.load(filename="Beh_demo.npy")

    assert loaded["label"] == "published"
    np.testing.assert_array_equal(loaded["trial"], np.arange(5))


def test_recording_interface_resolves_related_layers_and_behavior_ambiguity():
    data = drive.connect(mount=False)
    recording = data.recording("TX119_2023_12_24_1")

    assert recording.mouse == "TX119"
    assert recording.date == "2023-12-24"
    assert set(recording.layers) == {
        "behavior", "reduced_neural", "full_neural", "retinotopy"
    }
    assert recording.file("reduced_neural").name == (
        "TX119_2023_12_24_1_SVD_dec.npy"
    )
    assert recording.file("retinotopy").name == "TX119_2023_12_24_trans.npz"
    assert recording.file("behavior", experiment="unsup_test1").name == (
        "Beh_unsup_test1.npy"
    )
    with pytest.raises(drive.DriveDataError, match="several experiments"):
        recording.file("behavior")


def test_recording_display_exposes_files_provenance_and_exact_load_calls():
    data = drive.connect(mount=False)
    recording = data.recording("TX119_2023_12_24_1")

    summary = recording.to_dict()
    assert summary["recording_id"] == recording.recording_id
    assert summary["published_file_count"] == len(recording.files)
    assert summary["published_bytes"] == sum(
        item.size_bytes for item in recording.files
    )
    assert summary["availability"] == (
        "metadata_only_mount_required_for_arrays"
    )
    assert summary["load"]["reduced_neural"] == (
        'session.load("reduced_neural")'
    )
    assert summary["load"]["behavior:unsup_test1"] == (
        'session.load("behavior", experiment="unsup_test1")'
    )

    text = repr(recording)
    assert recording.recording_id in text
    assert f"'published_file_count': {len(recording.files)}" in text
    assert "'files'" in text
    assert "'dataset'" in text

    rich = recording._repr_html_()
    assert recording.recording_id in rich
    assert recording.retinotopy_id in rich
    assert "experiments" in rich
    assert "files" in rich
    assert "load" in rich
    assert "metadata_only_mount_required_for_arrays" in rich
    assert "session.load(&quot;reduced_neural&quot;)" in rich
    assert (
        "session.load(&quot;behavior&quot;, "
        "experiment=&quot;unsup_test1&quot;)"
    ) in rich
    for item in recording.files:
        assert item.name in rich
        assert str(item.id) in rich
        assert item.md5 in rich


def test_recordings_filter_by_experiment_and_mouse():
    data = drive.connect(mount=False)
    recordings = data.recordings(experiment="unsup_test1", mouse="TX119")

    assert recordings
    assert {recording.mouse for recording in recordings} == {"TX119"}
    assert all("unsup_test1" in recording.experiments for recording in recordings)


def test_picker_loads_one_file_and_invalidates_it_when_selection_changes(tmp_path):
    widgets = pytest.importorskip("ipywidgets")
    buffer = io.BytesIO()
    np.savez(buffer, values=np.arange(3))
    payload = buffer.getvalue()
    root = tmp_path / "release"
    rows = _catalog_rows(payload)
    _write_release(root, rows, payload)
    data = drive.connect(root=root, cache=tmp_path / "cache", mount=False)
    picker = data.picker()

    assert isinstance(picker.controls["mode"], widgets.Dropdown)
    assert picker.controls["file"].options == ()
    picker.controls["category"].value = "retinotopy"
    picker.controls["mode"].value = "file"
    assert picker.selected.name == "sample_trans.npz"
    assert picker.controls["load"].disabled is False

    picker.controls["load"].click()
    assert picker.last_error is None
    np.testing.assert_array_equal(picker.value["values"], np.arange(3))
    assert picker.path.name == "sample_trans.npz"

    picker.controls["search"].value = "nothing matches"
    assert picker.value is None
    assert picker.path is None
    assert picker.selected is None
    assert picker.controls["load"].disabled is True


def test_picker_loads_the_new_value_after_a_different_file_is_selected(
    tmp_path, monkeypatch
):
    pytest.importorskip("ipywidgets")
    data = drive.connect(cache=tmp_path / "cache", mount=False)

    def load_selected(item, **_kwargs):
        return {"selected_file": item.name}, tmp_path / item.name

    monkeypatch.setattr(data, "_load_item", load_selected)
    picker = data.picker()
    picker.controls["mode"].value = "file"
    picker.controls["category"].value = "retinotopy"

    choices = [value for _, value in picker.controls["file"].options]
    assert len(choices) > 1
    picker.controls["file"].value = choices[0]
    first = picker.load()

    picker.controls["file"].value = choices[1]
    assert picker.value is None
    assert picker.path is None
    second = picker.load()

    assert first["selected_file"] == choices[0]
    assert second["selected_file"] == choices[1]
    assert second != first
    assert picker.selected.name == choices[1]
