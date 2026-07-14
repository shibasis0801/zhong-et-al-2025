from pathlib import Path
import runpy
import time

import nbformat


NOTEBOOK = Path("notebooks/zhong2025_data_atlas_colab.ipynb")
GENERATOR = Path("scripts/create_data_atlas_notebook.py")


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def test_atlas_notebook_is_clean_valid_and_compilable():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    ids = [cell["id"] for cell in notebook.cells]
    assert len(ids) == len(set(ids))
    assert ids == [f"atlas-{index:03d}" for index in range(len(ids))]
    for index, cell in enumerate(notebook.cells):
        assert cell.get("execution_count") is None
        assert not cell.get("outputs")
        if cell.cell_type == "code":
            compile(cell.source, f"atlas-cell-{index}", "exec")


def test_generator_matches_committed_atlas_notebook():
    generated = runpy.run_path(str(GENERATOR))["build_notebook"]()
    committed = nbformat.read(NOTEBOOK, as_version=4)
    assert generated == committed


def test_atlas_colab_setup_uses_the_shared_drive_only():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    setup_source = notebook.cells[2].source
    assert "drive.mount(\"/content/drive\", force_remount=False)" in setup_source
    assert "Zhong et al. 2025 - Neuromatch Team Workspace" in setup_source
    assert "Zhong2025_Janelia_v2" in setup_source
    assert "team_tools/packages" in setup_source
    source = _source(notebook).lower()
    assert "github" not in source
    assert "git clone" not in source
    assert "git+https" not in source


def test_atlas_is_neutral_and_has_no_data_download_path():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = _source(notebook)
    for forbidden in (
        "LogisticRegression",
        "Ridge(",
        "position_decoding",
        "recommended question",
        "primary hypothesis",
        "download_profile(",
        "requests.get(",
        "allow_pickle=True",
    ):
        assert forbidden not in source
    assert "ALLOW_DATA_DOWNLOADS = False" in source
    assert "421.175 GiB" in source
    assert "142 memberships are not 142 recordings" in source


def test_atlas_contains_the_complete_orientation_path():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = _source(notebook)
    for heading in (
        "The mental model",
        "The complete release at a glance",
        "Two related but separate studies",
        "Imaging experiment timeline and vocabulary",
        "Canonical stimulus roles versus physical textures",
        "Join anatomy for one recording",
        "What is inside each data layer",
        "Inspect one real compact example",
        "Browse all published files without downloading them",
        "Preview an experiment slice before any download",
        "Locally derived processing files",
        "Relationships that exist—and ones that do not",
        "Team comprehension check",
    ):
        assert heading in source


def test_atlas_executes_default_path_quickly(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "__atlas_smoke__"}
    started = time.monotonic()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"atlas-cell-{index}", "exec"), namespace)
    elapsed = time.monotonic() - started

    assert elapsed < 30
    assert namespace["article"]["file_count"] == 297
    assert isinstance(namespace["figshare_api"], dict)
    assert namespace["FETCH_LIVE_FIGSHARE_API"] is False
    assert namespace["experiment_index"]["summary"]["unique_recordings"] == 89
    assert namespace["population"].shape == (452, 18, 48)
    assert namespace["SELECTED_EXPERIMENT"] == ""
    assert namespace["ALLOW_DATA_DOWNLOADS"] is False
