from __future__ import annotations

from zhong2025.catalog import catalog, mice, sessions


def _ids(view) -> set[str]:
    return {row[0] for row in view.rows}


def test_naive_mouse_filter_uses_all_canonical_cohorts() -> None:
    view = mice(cohort="naive")

    assert _ids(view) == {
        "LZ13", "LZ16", "TX108", "TX109", "TX119",
        "TX123", "TX124", "TX139", "TX140",
    }
    # The public columns still report the canonical primary cohort and overlap.
    tx108 = next(row for row in view.rows if row[0] == "TX108")
    assert tx108[1:3] == ["sup", "naive"]


def test_catalog_views_export_column_labelled_dictionaries(monkeypatch) -> None:
    view = mice(cohort="naive")
    exported = view.to_dict()

    assert exported["title"]
    assert exported["columns"] == view.columns
    assert exported["rows"][0] == dict(zip(view.columns, view.rows[0]))
    assert catalog().to_dict()["sections"]

    monkeypatch.setattr(type(view), "to_dict", lambda self: {"sentinel": 42})
    assert "'sentinel': 42" in repr(view)
    assert "sentinel" in view._repr_html_()


def test_session_filters_cover_overlapping_naive_and_training_memberships() -> None:
    naive = _ids(sessions(cohort="naive"))
    train2 = _ids(sessions(stage="Train 2"))
    test2 = _ids(sessions(stage="Test 2"))

    assert len(naive) == 15
    assert "LZ13_2024_05_15_1" in naive  # primary cohort is grating

    assert len(train2) == 22
    assert "TX108_2023_03_25_1" in train2  # primary stage is Test 1

    assert len(test2) == 25
    assert "TX119_2023_12_12_1" in test2  # primary stage is naive Test 1
    assert "LZ13_2024_05_28_1" in test2  # primary stage is grating Test 1 after


def test_combined_filters_must_match_the_same_membership() -> None:
    naive_test2 = _ids(sessions(cohort="naive", stage="Test 2"))

    assert len(naive_test2) == 7
    assert "LZ13_2024_05_15_1" in naive_test2
    # This acquisition is naive Test 1 and grating Test 2; it is not naive Test 2.
    assert "LZ13_2024_05_16_2" not in naive_test2
