from __future__ import annotations

import json
from pathlib import Path
import runpy

import nbformat


NOTEBOOK = Path("notebooks/01_dataset_atlas_and_graph_tutorials.ipynb")
GENERATOR = Path("scripts/create_companion_notebook.py")
INVENTORY = Path("zhong2025/assets/figshare-v2-inventory.json")
NATURE = "https://www.nature.com/articles/s41586-025-09180-y"


def _generated_notebook():
    return runpy.run_path(str(GENERATOR))["build"]()


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def test_companion_notebook_is_valid_reproducible_and_output_free():
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = _generated_notebook()
    nbformat.validate(committed)
    nbformat.validate(generated)

    assert committed.metadata == generated.metadata
    assert committed.metadata["colab"]["private_outputs"] is True
    assert len(committed.cells) == len(generated.cells) <= 24
    for index, (actual, expected) in enumerate(zip(committed.cells, generated.cells)):
        assert actual.cell_type == expected.cell_type
        assert actual.id == expected.id == f"companion-{index:03d}"
        assert actual.source == expected.source
        assert actual.metadata == expected.metadata
        if actual.cell_type == "code":
            compile(actual.source, f"companion-cell-{index}", "exec")
            assert "cellView" not in actual.metadata
            assert actual.execution_count is None
            assert not actual.outputs


def test_companion_notebook_has_dataset_and_graph_tutorials_without_simulation():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    for fragment in (
        "corridor_diagram",
        "dprime_playground",
        "explain_dprime",
        "explain_figure",
        "z.explain(",
        "z.gotchas(",
        "matplotlib",
        "plt.",
        "np.load(",
        "data.load(",
        "glob.glob",
        ".whl",
        "pip install",
        "recomputes",
        "cached to",
        "Happy exploring",
        "whole picture",
        "one number the whole paper turns on",
    ):
        assert fragment not in source

    assert "# Tutorial 1 — Dataset" in source
    assert "# Tutorial 2 — Graph" in source
    assert "data  #" not in source
    assert source.count("\ndata\n") >= 1
    assert "data.figshare(live=False)" in source
    assert 'data.recordings(experiment="sup_test1")' in source
    assert 'data.recordings(mouse="TX119")' in source
    assert 'data.recording("TX119_2023_12_24_1")' in source
    assert 'session.file("retinotopy")' in source
    assert 'session.load("retinotopy")' in source
    assert "item.size_bytes" in source
    assert "item.md5" in source
    assert "ndownloader.figshare.com/files/{item.id}" in source

    assert "graph = importlib.import_module(\"graph\")" in source
    assert '@graph.node(outputs="recording")' in source
    assert '@graph.node(outputs="file_inventory")' in source
    assert '@graph.node(outputs="retinotopy")' in source
    assert '@graph.node(outputs="retinotopy_inventory")' in source
    assert 'graph.Graph(' in source
    assert "atlas_flow.describe()" in source
    assert "atlas_flow.diagram()" in source
    assert "atlas_flow.run()" in source
    assert "atlas_flow.widget(" in source
    assert 'show="retinotopy_inventory"' in source


def test_companion_notebook_uses_exact_paper_release_and_code_links():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    for anchor in (
        "#Fig1",
        "#Fig2",
        "#Fig3",
        "#Fig4",
        "#Fig5",
        "#Sec7",
        "#Sec9",
        "#Sec13",
        "#Sec14",
        "#Sec17",
        "#Sec19",
        "#Sec20",
        "#Sec21",
        "#Sec22",
        "#Sec24",
        "#Sec25",
        "#data-availability",
        "#code-availability",
    ):
        assert f"{NATURE}{anchor}" in source

    for url in (
        "https://doi.org/10.25378/janelia.28811129.v2",
        "https://api.figshare.com/v2/articles/28811129/versions/2",
        "https://ndownloader.figshare.com/files/54183854",
        "https://ndownloader.figshare.com/files/54184070",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L370-L374",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L418-L441",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L599-L703",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L814-L882",
    ):
        assert url in source


def test_setup_uses_one_fixed_workspace_and_validates_pinned_release():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    assert source.count(
        "/content/drive/MyDrive/Zhong et al. 2025 - Neuromatch Team Workspace"
    ) == 1
    assert 'data.release["article_id"] == 28811129' in source
    assert 'data.release["version"] == 2' in source
    assert "len(data.files) == 297" in source
    assert "452_233_500_962" in source
    assert "Path(drive.__file__).resolve()" in source
    assert "Path(graph.__file__).resolve()" in source
    assert "drive.REPRESENTATION_API_VERSION >= 1" in source
    assert 'getattr(drive.Recording, "to_dict", None)' in source
    assert 'module_name.startswith("zhong2025.")' in source


def test_retinotopy_tutorial_cites_the_exact_selected_release_file():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))
    inventory = json.loads(INVENTORY.read_text())
    selected = next(
        item
        for item in inventory["files"]
        if item["name"] == "TX119_2023_12_24_trans.npz"
    )

    assert selected["id"] == 54184070
    assert selected["size_bytes"] == 983_358
    assert selected["md5"] == "ddda2db80ae338435ffa73b289690ae0"
    assert f"https://ndownloader.figshare.com/files/{selected['id']}" in source
    assert "a 983,358-byte file" in source
