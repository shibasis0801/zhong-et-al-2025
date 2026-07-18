from __future__ import annotations

from datetime import date
from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import nbformat
import numpy as np


NOTEBOOK = Path("notebooks/archived/07_complete_mouse_journeys_colab.ipynb")
GENERATOR = Path("scripts/create_mouse_journeys_notebook.py")
NATURE = "https://www.nature.com/articles/s41586-025-09180-y"


def _generated():
    return runpy.run_path(str(GENERATOR))["build_notebook"]()


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def _cell(notebook, cell_id):
    matches = [cell for cell in notebook.cells if cell.id == cell_id]
    assert len(matches) == 1
    return matches[0]


def test_mouse_journeys_notebook_is_deterministic_valid_and_output_free():
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = _generated()
    nbformat.validate(committed)
    nbformat.validate(generated)

    assert committed == generated
    assert committed.metadata["colab"]["private_outputs"] is True
    assert len(committed.cells) == 25
    assert len({cell.id for cell in committed.cells}) == len(committed.cells)
    for cell in committed.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
            assert cell.execution_count is None
            assert not cell.outputs


def test_notebook_covers_every_mouse_acquisition_and_membership_without_duplication():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    assert source.count("https://shibasis.dev/neuromatch/#mouse-") == 19
    assert "19 imaging mice" in source
    assert "89 unique neural" in source
    assert 'assert len(canonical["mice"]) == 19' in source
    assert 'assert len(canonical["recordings"]) == 89' in source
    assert '"unique_experiment_recording_memberships"' in source
    assert "133 experiment–recording" in source
    assert "142 source rows" not in source

    for mouse in (
        "VR2", "TX60", "TX61", "TX108", "TX109",
        "TX85", "TX88", "DR10", "TX83", "DR15", "TX104", "TX105",
        "TX119", "TX123", "LZ13", "LZ16", "TX139", "TX124", "TX140",
    ):
        assert f"](https://shibasis.dev/neuromatch/#mouse-{mouse})" in source


def test_notebook_has_precise_primary_sources_and_no_unscoped_paper_link():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    for link in (
        f"{NATURE}#Sec11",
        f"{NATURE}#Sec13",
        f"{NATURE}#Sec17",
        f"{NATURE}#Sec19",
        f"{NATURE}#Sec20",
        f"{NATURE}#Sec21",
        f"{NATURE}#Sec22",
        f"{NATURE}#Sec25",
        f"{NATURE}#Fig1",
        f"{NATURE}#Fig2",
        f"{NATURE}#Fig3",
        f"{NATURE}#Fig4",
        f"{NATURE}/figures/11",
        f"{NATURE}/figures/12",
        "https://ndownloader.figshare.com/files/54183854",
        "https://doi.org/10.25378/janelia.28811129.v2",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416",
    ):
        assert link in source

    assert f"]({NATURE})" not in source


def test_dataset_and_graph_tutorials_are_applied_and_large_runs_are_opt_in():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    for fragment in (
        "data.recordings(mouse=SELECTED_MOUSE)",
        "data.recording(\"TX119_2023_12_24_1\")",
        "recording.file(\"behavior\", experiment=experiment)",
        'recording.load("reduced_neural", max_gib=max_gib)',
        'recording.load("retinotopy", max_gib=max_gib)',
        "@graph.node(outputs=\"mouse_id\")",
        "@graph.node(outputs=\"session_dashboard\"",
        "graph.Graph(",
        "journey_graph.widget(",
        "session_graph.widget(",
        "RUN_ONE_SESSION = False",
        "RUN_SELECTED_MOUSE = False",
        "RUN_ALL_MICE = False",
    ):
        assert fragment in source

    assert 'recording.load("full_neural"' not in source
    assert "synthetic" not in source.lower()
    assert "simulation" not in source.lower()


def test_session_summary_helpers_obey_released_shapes_and_area_codes(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = _generated()
    namespace = {"np": np, "plt": plt}
    exec(_cell(notebook, "journeys-session-tools").source, namespace)

    behavior = {
        "ft_Pos": np.arange(12),
        "ft_RunSpeed": np.array([1.0, 3.0] * 6),
        "ft_trInd": np.repeat(np.arange(3), 4),
        "LickPos": np.array([5.0, 10.0, 20.0, 25.0]),
        "ntrials": 3,
    }
    behavior_qc = namespace["summarize_behavior"](behavior)
    assert behavior_qc == {
        "frames": 12,
        "trials": 3,
        "licks": 4,
        "licks_per_trial": 4 / 3,
        "mean_run_speed": 2.0,
    }

    svd_qc = namespace["summarize_svd"]({
        "U": np.zeros((3, 5)),
        "V": np.zeros((3, 7)),
    })
    assert svd_qc == {"components": 3, "neurons": 5, "frames": 7}

    retinotopy_qc = namespace["summarize_retinotopy"]({
        "xy_t": np.arange(10, dtype=float).reshape(5, 2),
        "iarea": np.array([8, 0, 5, 3, -1]),
    })
    assert retinotopy_qc["neurons"] == 5
    assert retinotopy_qc["area_counts"] == {
        "V1": 1,
        "Medial": 1,
        "Lateral": 1,
        "Anterior": 1,
        "Excluded / unassigned": 1,
    }


def test_real_metadata_build_and_both_graphs_construct(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import drive
    import graph
    from zhong2025.catalog import load_map

    notebook = _generated()
    namespace = {
        "data": drive.setup(),
        "graph": graph,
        "canonical": load_map(),
        "date": date,
        "np": np,
        "plt": plt,
        "defaultdict": __import__("collections").defaultdict,
    }
    for cell_id in (
        "journeys-build",
        "journeys-preflight",
        "journeys-session-tools",
    ):
        if cell_id == "journeys-preflight":
            namespace["SELECTED_MOUSE"] = "TX119"
        exec(_cell(notebook, cell_id).source, namespace)
    namespace["DEFAULT_SELECTION"] = "TX119_2023_12_24_1|unsup_test1"
    exec(_cell(notebook, "journeys-graph-metadata").source, namespace)
    exec(_cell(notebook, "journeys-graph-data").source, namespace)

    journeys = namespace["journeys"]
    assert len(journeys) == 19
    assert sum(len(row["acquisitions"]) for row in journeys.values()) == 89
    assert sum(
        len(acquisition["experiments"])
        for row in journeys.values()
        for acquisition in row["acquisitions"]
    ) == 133
    assert len(namespace["journey_graph"].nodes) == 4
    assert len(namespace["session_graph"].nodes) == 5
    terminal_ports = [
        port["name"]
        for node in namespace["session_graph"].describe()["nodes"]
        for port in node["output_ports"]
        if not port["connected"]
    ]
    assert terminal_ports == ["file_plan", "session_dashboard"]
