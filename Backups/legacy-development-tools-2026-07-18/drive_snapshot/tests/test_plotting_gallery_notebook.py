from __future__ import annotations

from pathlib import Path
import runpy

import nbformat


NOTEBOOK = Path("notebooks/archived/08_plotting_gallery_colab.ipynb")
GENERATOR = Path("scripts/create_plotting_gallery_notebook.py")


def _generated():
    return runpy.run_path(str(GENERATOR))["build_notebook"]()


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def test_gallery_is_valid_deterministic_output_free_and_colab_ready():
    committed = nbformat.read(NOTEBOOK, as_version=4)
    generated = _generated()
    nbformat.validate(committed)
    nbformat.validate(generated)

    assert committed == generated
    assert committed.metadata["colab"]["private_outputs"] is True
    assert len(committed.cells) == 29
    assert len({cell.id for cell in committed.cells}) == len(committed.cells)
    for index, cell in enumerate(committed.cells):
        assert cell.id == f"plot-gallery-{index:03d}"
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
            assert cell.execution_count is None
            assert not cell.outputs


def test_gallery_labels_unavailable_inputs_and_uses_safe_real_data():
    source = _source(nbformat.read(NOTEBOOK, as_version=4))

    for fragment in (
        "Illustrative inputs for unavailable layers",
        "bundled, checksum-pinned compact recording",
        "not the full deconvolved single-neuron traces",
        "Fixed-seed examples",
        "plot.guide()",
        "load_atlas_demo()",
        "load_file_inventory()",
        "load_map()",
        "expected_recipes = set(plot.recipes())",
        "plot.save(figure",
        "plot.close(figure)",
        "for warning in details.warnings",
        "plot.position_surface(",
        "plot.cross_temporal(",
    ):
        assert fragment in source

    assert "data.fetch(" not in source
    assert "data.load(" not in source
    assert "np.load(" not in source
    assert "plt." not in source


def test_gallery_executes_every_recipe_and_leaves_no_open_figures(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/zhong-plot-gallery-matplotlib")
    monkeypatch.syspath_prepend(str(Path.cwd()))

    import IPython.display as ipython_display
    import matplotlib

    matplotlib.use("Agg", force=True)
    monkeypatch.setattr(ipython_display, "display", lambda *args, **kwargs: None)

    namespace = {"__name__": "__plot_gallery_test__"}
    for cell in _generated().cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, cell.id, "exec"), namespace)

    import matplotlib.pyplot as plt

    assert namespace["rendered_recipes"] == set(namespace["plot"].recipes())
    assert len(namespace["rendered_recipes"]) == 36
    assert len(namespace["render_log"]) >= 36
    assert not plt.get_fignums()
