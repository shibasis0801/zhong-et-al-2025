import hashlib
import numpy as np
import pytest

from zhong2025.data import (
    download_file,
    load_atlas_demo,
    profile_summary,
)
from zhong2025.demo import _verify_source


class FakeResponse:
    def __init__(self, payload, *, status_error=False):
        self.payload = payload
        self.closed = False
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise RuntimeError("HTTP failure")

    def iter_content(self, chunk_size):
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, payload, *, status_error=False):
        self.response = FakeResponse(payload, status_error=status_error)
        self.calls = []

    def get(self, url, stream, timeout):
        self.calls.append((url, stream, timeout))
        return self.response


def test_streamed_download_verifies_and_atomically_renames(tmp_path):
    payload = b"published-data"
    entry = {
        "key": "fixture",
        "path": "beh/fixture.bin",
        "url": "https://example.test/fixture",
        "size_bytes": len(payload),
        "md5": hashlib.md5(payload).hexdigest(),
    }
    session = FakeSession(payload)
    result = download_file(entry, tmp_path, session=session, chunk_size=3)
    assert result.read_bytes() == payload
    assert not result.with_suffix(".bin.part").exists()
    assert session.calls[0][1] is True
    assert session.response.closed


def test_corrupt_download_removes_partial_file(tmp_path):
    entry = {
        "key": "fixture",
        "path": "fixture.bin",
        "url": "https://example.test/fixture",
        "size_bytes": 4,
        "md5": "0" * 32,
    }
    with pytest.raises(ValueError, match="MD5"):
        download_file(entry, tmp_path, session=FakeSession(b"data"))
    assert not (tmp_path / "fixture.bin.part").exists()
    assert not (tmp_path / "fixture.bin").exists()


def test_download_rejects_path_escape_before_request(tmp_path):
    entry = {
        "key": "escape",
        "path": "../escape.bin",
        "url": "https://example.test/escape",
        "size_bytes": 1,
        "md5": hashlib.md5(b"x").hexdigest(),
    }
    session = FakeSession(b"x")
    with pytest.raises(ValueError, match="stay relative"):
        download_file(entry, tmp_path, session=session)
    assert not session.calls


def test_http_failure_closes_response_and_removes_partial(tmp_path):
    entry = {
        "key": "failure",
        "path": "failure.bin",
        "url": "https://example.test/failure",
        "size_bytes": 1,
        "md5": hashlib.md5(b"x").hexdigest(),
    }
    session = FakeSession(b"x", status_error=True)
    with pytest.raises(RuntimeError, match="HTTP failure"):
        download_file(entry, tmp_path, session=session)
    assert session.response.closed
    assert not (tmp_path / "failure.bin.part").exists()


def test_profile_reports_exact_pinned_size():
    summary = profile_summary("atlas_demo_source")
    assert summary["total_bytes"] == 338_448_290
    assert {entry["kind"] for entry in summary["files"]} == {
        "behavior",
        "svd_dec",
        "retinotopy",
    }


def test_committed_demo_is_pickle_free_and_consistent():
    demo = load_atlas_demo()
    assert demo["population_features"].shape == (452, 18, 48)
    assert demo["area_features"].shape == (4, 452, 18, 12)
    assert np.all(demo["frame_counts"] > 0)
    assert set(demo["wall_name"]) == {"rock1", "rock2", "wood1", "wood2"}
    assert demo["metadata"]["session"] == "TX119_2023_12_24_1"


def test_pickle_source_verification_rejects_substitution(tmp_path):
    source = tmp_path / "source.npy"
    source.write_bytes(b"trusted")
    trusted_spec = {
        "size_bytes": 7,
        "sha256": hashlib.sha256(b"trusted").hexdigest(),
    }
    assert _verify_source(source, trusted_spec) == source
    source.write_bytes(b"hostile")
    with pytest.raises(ValueError, match="SHA-256"):
        _verify_source(source, trusted_spec)
