from __future__ import annotations

import json

import pandas as pd
import pytest

import drive
import sql


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("ZHONG2025_DATASET_ROOT", raising=False)
    handle = drive.connect(cache=tmp_path / "cache", mount=False)
    value = sql.setup(source=handle, report=False)
    yield value
    value.close()


def test_sql_catalog_has_flat_release_tables(db):
    assert db.tables == (
        "experiment_rows",
        "experiments",
        "files",
        "memberships",
        "mice",
        "recording_files",
        "recordings",
        "release",
    )
    assert {row.table: row.rows for row in db.schema().itertuples()} == {
        "experiment_rows": 142,
        "experiments": 23,
        "files": 297,
        "memberships": 133,
        "mice": 19,
        "recording_files": 400,
        "recordings": 89,
        "release": 1,
    }
    assert isinstance(db.table("recordings"), pd.DataFrame)


def test_duckdb_queries_recover_neural_mice_without_hardcoded_lists(db):
    result = db.query("""
        SELECT primary_cohort,
               list(mouse ORDER BY mouse) AS mice,
               count(*) AS mouse_count
        FROM mice
        WHERE has_full_neural
        GROUP BY primary_cohort
        ORDER BY primary_cohort
    """)
    rows = {row.primary_cohort: (list(row.mice), row.mouse_count) for row in result.itertuples()}
    assert rows == {
        "grating": (["LZ13", "LZ16", "TX139"], 3),
        "naive": (["TX124", "TX140"], 2),
        "supervised": (["TX108", "TX109", "TX60", "TX61", "VR2"], 5),
        "unsupervised": (
            ["DR10", "DR15", "TX104", "TX105", "TX119", "TX123", "TX83", "TX85", "TX88"],
            9,
        ),
    }


def test_recordings_distinguish_one_acquisition_from_multiple_labels(db):
    result = db.query("""
        SELECT experiment_count, experiments_json
        FROM recordings
        WHERE recording_id = 'TX108_2023_03_25_1'
    """).iloc[0]
    assert result["experiment_count"] == 2
    assert json.loads(result["experiments_json"]) == [
        "sup_test1",
        "sup_train2_before_learning",
    ]
    files = db.query("""
        SELECT layer, count(*) AS file_count
        FROM recording_files
        WHERE recording_id = 'TX108_2023_03_25_1'
        GROUP BY layer
        ORDER BY layer
    """)
    assert dict(zip(files["layer"], files["file_count"])) == {
        "behavior": 2,
        "full_neural": 1,
        "reduced_neural": 1,
        "retinotopy": 1,
    }


def test_register_makes_an_analysis_frame_queryable(db):
    labels = pd.DataFrame(
        [("supervised", "sup_train1_before_learning")],
        columns=["cohort", "experiment"],
    )
    returned = db.register("train1_labels", labels)
    assert returned.equals(labels)
    result = db.query("""
        SELECT count(DISTINCT m.recording_id) AS recordings
        FROM train1_labels AS l
        JOIN memberships AS m USING (experiment)
    """)
    assert result.iloc[0]["recordings"] == 4
    with pytest.raises(ValueError, match="simple SQL identifier"):
        db.register("bad table", labels)


def test_behavior_loading_requires_a_label_when_an_acquisition_has_two(db):
    recording_id = "TX108_2023_03_25_1"
    with pytest.raises(drive.DriveDataError, match="Choose experiment"):
        db.load(recording_id, "behavior")

    seen = []

    def fake_load(filename, *, max_gib):
        seen.append((filename, max_gib))
        return {recording_id: {"trial": [1, 2, 3]}}

    db.source.load = fake_load
    loaded = db.load(recording_id, "behavior", experiment="sup_test1", max_gib=0.5)
    assert loaded == {"trial": [1, 2, 3]}
    assert seen == [("Beh_sup_test1.npy", 0.5)]
