from __future__ import annotations

import hashlib
from pathlib import Path
import runpy

import nbformat
import numpy as np

NOTEBOOK = Path("notebooks/02_released_example_dprime_walkthrough.ipynb")
REFERENCE = Path("notebooks/05_neuromatch_visual_learning_project.ipynb")
GENERATOR = Path("scripts/create_visual_learning_drive_notebook.py")


def _module():
    return runpy.run_path(str(GENERATOR))


def _generated():
    return _module()["build_notebook"]()


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def _by_id(notebook, cell_id):
    matches = [cell for cell in notebook.cells if cell.id == cell_id]
    assert len(matches) == 1
    return matches[0]


def test_drive_notebook_is_deterministic_valid_and_output_free():
    reference_before = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = _generated()
    reference_after = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()

    nbformat.validate(committed)
    nbformat.validate(generated)
    assert reference_before == reference_after
    assert committed == generated
    assert len(committed.cells) == 11
    assert committed.metadata["colab"]["private_outputs"] is True
    assert committed.metadata["zhong2025_conversion"]["data_release"] == (
        "10.25378/janelia.28811129.v2"
    )
    assert len({cell.id for cell in committed.cells}) == len(committed.cells)
    for cell in committed.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
            assert cell.execution_count is None
            assert not cell.outputs


def test_notebook_has_the_complete_official_figure_and_precise_sources():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    for exact_link in (
        "https://www.nature.com/articles/s41586-025-09180-y#Fig1",
        "https://www.nature.com/articles/s41586-025-09180-y#Sec2",
        "https://www.nature.com/articles/s41586-025-09180-y#Sec20",
        "https://www.nature.com/articles/s41586-025-09180-y#data-availability",
        "https://doi.org/10.25378/janelia.28811129.v2",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L370-L443",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416",
        "https://media.springernature.com/full/springer-static/image/",
    ):
        assert exact_link in source

    figure_cell = _by_id(nbformat.read(NOTEBOOK, as_version=4), "nature-figure-1")
    assert "Complete published figure" in figure_cell.source
    assert "41586_2025_9180_Fig1_HTML.png" in figure_cell.source
    assert "data:image/" not in source
    assert "attachment:" not in source


def test_notebook_contains_only_the_released_example_and_paper_estimator():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    assert "VR2_2021_03_20_1_example_raw_spk.npy" in source
    assert "Beh_sup_train1_before_learning.npy" in source
    assert "VR2_2021_03_20_trans.npz" in source
    assert "ft_move" in source
    assert "ft_CorrSpc" in source
    assert "stimulus_id == 2" in source
    assert "stimulus_id == 0" in source
    assert "DP_THRESHOLD = 0.3" in source
    assert "cortical_x = -xy_t[:, 1]" in source
    assert "cortical_y = xy_t[:, 0]" in source
    assert "(iarea != -1) & (iarea != 7)" in source
    assert "released 1,000-neuron raw example" in source
    assert "not a reproduction of the paper's cohort estimate" in source

    for removed_noise in (
        "corridor_diagram",
        "paired-trial",
        "descriptive sandbox",
        "Graph 4",
        "workflow map",
        "dprime_playground",
        "np.random",
        "synthetic",
        "toy data",
        "position_start_dm",
        "position_stop_dm",
        "moving_only",
        "components=40",
        "threshold=0.6",
    ):
        assert removed_noise not in source


def test_exact_released_file_metadata_is_verified_against_the_catalog(monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd()))
    import drive

    notebook = _generated()
    namespace = {"data": drive.setup()}
    exec(_by_id(notebook, "verify-released-inputs").source, namespace)
    actual = {
        name: (item.id, item.size_bytes, item.md5)
        for name, item in namespace["verified_files"].items()
    }
    assert actual == {
        "VR2_2021_03_20_1_example_raw_spk.npy": (
            54866153,
            97192128,
            "7e341d96305e3a235213b419f71c576d",
        ),
        "Beh_sup_train1_before_learning.npy": (
            54183863,
            124559852,
            "75169b8c4c02f5ed9af3fd492e93b9bd",
        ),
        "VR2_2021_03_20_trans.npz": (
            54184211,
            2934270,
            "f8fbb33ee2c9461011306c5072d0b06e",
        ),
    }


def test_paper_estimator_matches_the_whole_session_formula_and_xy_t_transform():
    notebook = _generated()
    namespace = {"np": np}
    exec("DP_THRESHOLD = 0.3", namespace)
    exec(_by_id(notebook, "paper-dprime-functions").source, namespace)

    activity = np.array(
        [
            [1.0, 3.0, 100.0, 2.0, 4.0, 100.0],
            [4.0, 4.0, 100.0, 0.0, 0.0, 100.0],
            [3.0, 5.0, 100.0, 1.0, 1.0, 100.0],
        ]
    )
    behavior = {
        "ft_WallID": np.array([10, 10, 10, 20, 20, 20]),
        "ft_move": np.array([1, 1, 0, 1, 1, 1]),
        "ft_CorrSpc": np.array([1, 1, 1, 1, 1, 0], dtype=bool),
        "stim_id": np.array([2, 0]),
        "UniqWalls": np.array([10, 20]),
    }
    xy_t = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    retinotopy = {"xy_t": xy_t, "iarea": np.array([8, -1, 7])}
    result = namespace["paper_dprime_for_released_example"](
        activity, behavior, retinotopy
    )

    leaf = activity[:, [0, 1]]
    circle = activity[:, [3, 4]]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = 2 * (leaf.mean(1) - circle.mean(1)) / (
            leaf.std(1) + circle.std(1)
        )
    np.testing.assert_allclose(result["dprime"], expected, equal_nan=True)
    np.testing.assert_array_equal(result["cortical_x"], -xy_t[:, 1])
    np.testing.assert_array_equal(result["cortical_y"], xy_t[:, 0])
    np.testing.assert_array_equal(result["mapped"], [True, False, False])
    assert result["leaf_frames"] == result["circle_frames"] == 2
    assert bool(result["circle_selective"][0])
    assert not bool(result["leaf_selective"][0])


def test_plot_reports_only_observed_example_quantities(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = _generated()
    namespace = {"np": np}
    exec("DP_THRESHOLD = 0.3", namespace)
    exec(_by_id(notebook, "paper-dprime-functions").source, namespace)

    activity = np.array([[1.0, 3.0, 2.0, 4.0], [5.0, 7.0, 1.0, 3.0]])
    behavior = {
        "ft_WallID": np.array([10, 10, 20, 20]),
        "ft_move": np.ones(4),
        "ft_CorrSpc": np.ones(4, dtype=bool),
        "stim_id": np.array([2, 0]),
        "UniqWalls": np.array([10, 20]),
    }
    result = namespace["paper_dprime_for_released_example"](
        activity,
        behavior,
        {"xy_t": np.array([[1.0, 2.0], [3.0, 4.0]]), "iarea": np.array([8, 0])},
    )
    namespace.update(
        {
            "plt": __import__("matplotlib.pyplot", fromlist=["pyplot"]),
            "example_result": result,
            "RECORDING_ID": "VR2_2021_03_20_1",
        }
    )
    exec(_by_id(notebook, "plot-paper-dprime").source, namespace)
    figure = namespace["example_figure"]
    assert [axis.get_title() for axis in figure.axes] == [
        "Valid whole-session frames",
        "Signed per-neuron d′",
        "Whole-session mean responses",
        "Paper-coordinate locations",
    ]
    assert figure.axes[-1].get_xlabel() == "cortical x = -xy_t[:, 1]"
    assert figure.axes[-1].get_ylabel() == "cortical y = xy_t[:, 0]"
