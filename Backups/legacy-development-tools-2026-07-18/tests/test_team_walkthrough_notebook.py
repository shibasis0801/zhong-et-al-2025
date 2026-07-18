from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace

import nbformat
import numpy as np

NOTEBOOK = Path("notebooks/archived/03_dataset_walkthrough_colab.ipynb")
GENERATOR = Path("scripts/create_team_walkthrough_notebook.py")


def _generated():
    return runpy.run_path(str(GENERATOR))["build_notebook"]()


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def _by_id(notebook, cell_id):
    matches = [cell for cell in notebook.cells if cell.id == cell_id]
    assert len(matches) == 1
    return matches[0]


def test_walkthrough_is_valid_deterministic_and_output_free():
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = _generated()
    nbformat.validate(committed)
    nbformat.validate(generated)

    assert committed == generated
    assert committed.metadata["colab"]["private_outputs"] is True
    assert len(committed.cells) == 12
    assert len({cell.id for cell in committed.cells}) == len(committed.cells)
    for cell in committed.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
            assert cell.execution_count is None
            assert not cell.outputs


def test_walkthrough_uses_precise_paper_release_file_and_builder_links():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    for exact_link in (
        "https://www.nature.com/articles/s41586-025-09180-y#Fig1",
        "https://www.nature.com/articles/s41586-025-09180-y#Sec2",
        "https://www.nature.com/articles/s41586-025-09180-y#Sec19",
        "https://www.nature.com/articles/s41586-025-09180-y#data-availability",
        "https://doi.org/10.25378/janelia.28811129.v2",
        "https://ndownloader.figshare.com/files/54183854",
        "https://ndownloader.figshare.com/files/54183911",
        "https://ndownloader.figshare.com/files/54866057",
        "https://ndownloader.figshare.com/files/54184070",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416",
        "https://github.com/shibasis0801/zhong-et-al-2025/blob/main/zhong2025/demo.py#L80-L235",
    ):
        assert exact_link in source

    for removed_noise in (
        "Graph 4",
        "population_dprime",
        "proposal-primary",
        "early-trial",
        "cross-validated",
        "independent unit of inference",
        "transductive",
        "simulation",
        "synthetic",
        "toy data",
        "graph.Graph(",
        "ipywidgets",
    ):
        assert removed_noise not in source


def test_release_counts_distinguish_rows_memberships_and_acquisitions(monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd()))
    import drive
    from zhong2025 import experiment_rows, load_experiment_index

    namespace = {
        "data": drive.setup(),
        "drive": drive,
        "experiment_rows": experiment_rows,
        "load_experiment_index": load_experiment_index,
    }
    exec(_by_id(_generated(), "walkthrough-release").source, namespace)
    summary = namespace["release_summary"]
    assert summary == {
        "figshare_id": 28811129,
        "figshare_version": 2,
        "doi": "10.25378/janelia.28811129.v2",
        "published_files": 297,
        "published_bytes": 452233500962,
        "metadata_rows": 142,
        "unique_experiment_recording_memberships": 133,
        "unique_acquisitions": 89,
    }
    assert sum(namespace["experiment_membership_counts"].values()) == 133


def test_retinotopy_code_verifies_file_and_uses_the_paper_transform(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    xy_t = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0], [19.0, 23.0]])
    retinotopy = {
        "xy_t": xy_t,
        "iarea": np.array([8, 0, 5, 3, -1]),
    }
    file_row = SimpleNamespace(
        name="TX119_2023_12_24_trans.npz",
        id=54184070,
        size_bytes=983358,
        md5="ddda2db80ae338435ffa73b289690ae0",
    )
    data = SimpleNamespace(files=[file_row], load=lambda _name: retinotopy)
    namespace = {"np": np, "plt": plt, "data": data}
    exec(_by_id(_generated(), "walkthrough-retinotopy").source, namespace)

    prepared = namespace["prepared_retinotopy"]
    np.testing.assert_array_equal(prepared["x"], -xy_t[:, 1])
    np.testing.assert_array_equal(prepared["y"], xy_t[:, 0])
    assert list(prepared["groups"]) == [
        "V1",
        "Medial",
        "Lateral",
        "Anterior",
        "Excluded / unassigned",
    ]
    assert all(int(mask.sum()) == 1 for mask in prepared["groups"].values())
    figure = namespace["retinotopy_figure"]
    assert figure.axes[0].get_xlabel() == "paper cortical x = -xy_t[:, 1]"
    assert figure.axes[0].get_ylabel() == "paper cortical y = xy_t[:, 0]"


def test_disclosed_derivative_is_real_and_qc_plot_contains_only_its_values(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from zhong2025 import load_atlas_demo

    namespace = {"load_atlas_demo": load_atlas_demo}
    exec(_by_id(_generated(), "walkthrough-derivative").source, namespace)
    demo = namespace["demo"]
    assert demo["metadata"]["session"] == "TX119_2023_12_24_1"
    assert demo["metadata"]["source_file_ids"] == [54183911, 54866057, 54184070]
    assert demo["population_features"].shape == (452, 18, 48)
    assert demo["mean_run_speed"].shape == (452, 18)

    namespace.update({"np": np, "plt": plt})
    exec(_by_id(_generated(), "walkthrough-derivative-plot").source, namespace)
    figure = namespace["derivative_figure"]
    assert [axis.get_title() for axis in figure.axes[:2]] == [
        "Raw wall identity by trial",
        "Binned released SVD component 1",
    ]
    assert figure.axes[2].get_title() == "Across-trial mean SVD score"
    assert figure.axes[3].get_title() == "Across-trial mean running speed"
    assert "452 trials" in figure._suptitle.get_text()
    assert "18 fixed position bins" in figure._suptitle.get_text()
