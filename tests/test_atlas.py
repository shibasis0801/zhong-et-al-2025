from collections import Counter

from zhong2025.atlas import (
    EXPECTED_TOTAL_BYTES,
    experiment_recordings,
    experiment_rows,
    experiment_semantics,
    load_experiment_index,
    load_file_inventory,
    recording_bundle,
)


def test_complete_inventory_counts_and_bytes():
    inventory = load_file_inventory()
    files = inventory["files"]
    counts = Counter(entry["category"] for entry in files)

    assert len(files) == 297
    assert sum(entry["size_bytes"] for entry in files) == EXPECTED_TOTAL_BYTES
    assert counts == {
        "full_neural": 89,
        "reduced_neural": 89,
        "retinotopy": 89,
        "imaging_behavior": 23,
        "faster_learning_behavior": 3,
        "imaging_experiment_index": 1,
        "area_outlines": 1,
        "behavior_example": 1,
        "neural_example": 1,
    }


def test_experiment_index_preserves_membership_and_acquisition_counts():
    index = load_experiment_index()
    rows = experiment_rows(index)

    assert index["summary"] == {
        "associations": 142,
        "experiment_labels": 23,
        "unique_mice": 19,
        "unique_recordings": 89,
    }
    assert len(rows) == 142
    assert len({row["recording_id"] for row in rows}) == 89
    assert len({row["mouse"] for row in rows}) == 19
    assert {stimulus for row in rows for stimulus in row["stimulus_ids"] if stimulus is not None} == set(range(7))


def test_every_physical_recording_resolves_to_the_published_modalities():
    inventory = load_file_inventory()
    index = load_experiment_index()
    recording_ids = {
        entry["recording_id"]
        for entry in inventory["files"]
        if entry["category"] == "full_neural"
    }

    for recording_id in recording_ids:
        bundle = recording_bundle(recording_id, inventory=inventory, index=index)
        categories = Counter(entry["category"] for entry in bundle["files"])
        assert categories["full_neural"] == 1
        assert categories["reduced_neural"] == 1
        assert categories["retinotopy"] == 1
        assert categories["imaging_behavior"] >= 1


def test_membership_reuse_is_explicit_for_compact_example():
    index = load_experiment_index()
    inventory = load_file_inventory()
    recording_id = "TX119_2023_12_24_1"

    assert {
        row["experiment"]
        for row in experiment_rows(index)
        if row["recording_id"] == recording_id
    } == {"unsup_test1", "unsup_train2_before_learning"}
    bundle = recording_bundle(recording_id, inventory=inventory, index=index)
    assert bundle["experiments"] == ["unsup_test1", "unsup_train2_before_learning"]


def test_all_experiment_labels_have_neutral_semantic_dimensions():
    index = load_experiment_index()
    for experiment in index["experiments"]:
        semantics = experiment_semantics(experiment)
        assert semantics["cohort"] in {
            "task",
            "unrewarded exposure",
            "naive",
            "grating control",
        }
        assert semantics["stage"] in {"Train 1", "Test 1", "Train 2", "Test 2", "Test 3"}
        assert set(semantics["stimulus_roles"]) <= set(range(7))
        assert experiment_recordings(experiment, index)
