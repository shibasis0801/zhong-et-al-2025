from pathlib import Path
import runpy

import nbformat


NOTEBOOK = Path("notebooks/archived/00_use_janelia_drive_colab.ipynb")
GENERATOR = Path("scripts/create_data_access_notebook.py")


def _generated():
    return runpy.run_path(str(GENERATOR))["build_notebook"]()


def test_data_access_notebook_is_clean_and_reproducible():
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = _generated()
    nbformat.validate(committed)
    nbformat.validate(generated)

    assert committed.metadata == generated.metadata
    assert committed.metadata["colab"]["private_outputs"] is True
    assert "widgets" not in committed.metadata
    assert len(committed.cells) == len(generated.cells)
    for index, (actual, expected) in enumerate(zip(committed.cells, generated.cells)):
        assert actual.id == expected.id == f"data-access-{index:03d}"
        assert actual.source == expected.source
        if actual.cell_type == "code":
            compile(actual.source, f"data-access-cell-{index}", "exec")
            assert actual.execution_count is None
            assert not actual.outputs


def test_notebook_uses_only_the_small_scientist_facing_api():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    prose = "\n".join(cell.source for cell in notebook.cells)

    assert "data = drive.setup(" in code
    assert "data.picker()" in code
    assert "data.find(" in code
    assert "data.load(" in code
    assert "data.recording(" in code
    assert "picker.value" in code
    assert "Runtime → Run all" in prose
    assert "452,233,500,962-byte" in prose
    assert "len(matches) != 1" in code
    assert "selected.relative_path" in code
    assert "selected.md5" in code
    assert "ndownloader.figshare.com/files/{selected.id}" in code
    assert "x, y = -np.asarray(xy)[:, 1], np.asarray(xy)[:, 0]" in code
    assert 'groups["Excluded / unassigned"] = ~assigned' in code
    assert "fig.colorbar" not in code

    for exact_source in (
        "https://www.nature.com/articles/s41586-025-09180-y#data-availability",
        "https://doi.org/10.25378/janelia.28811129.v2",
        "https://www.nature.com/articles/s41586-025-09180-y#Fig1",
        "https://www.nature.com/articles/s41586-025-09180-y#Sec20",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416",
        "https://www.nature.com/articles/s41586-025-09180-y#Sec19",
    ):
        assert exact_source in prose

    for infrastructure in (
        "hashlib",
        "shutil",
        "np.load",
        "allow_pickle",
        "TRANSFER_STATUS",
        "catalog.csv",
        "rglob(",
        "os.walk(",
    ):
        assert infrastructure not in code
