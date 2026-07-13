import ast
from pathlib import Path

import pytest

import graph


def _walk_widgets(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _walk_widgets(child)


def _html_text(widget):
    return " ".join(
        item.value
        for item in _walk_widgets(widget)
        if hasattr(item, "value") and isinstance(item.value, str)
    )


def _example_graph(calls=None):
    calls = calls if calls is not None else []

    @graph.node(outputs="data")
    def load(scale=2):
        calls.append("load")
        return scale

    @graph.node(outputs="selected")
    def select(data, offset=1):
        calls.append("select")
        return data + offset

    @graph.node(outputs="summary")
    def summarize(data, selected):
        calls.append("summarize")
        return data * selected

    return graph.Graph("example", load, select, summarize)


def test_named_ports_fan_out_and_run_in_declared_order():
    calls = []
    flow = _example_graph(calls)
    run = flow.run(scale=3, offset=4)

    assert calls == ["load", "select", "summarize"]
    assert run.order == ("load", "select", "summarize")
    assert run["data"] == 3
    assert run["selected"] == 7
    assert run["summary"] == 21
    assert run.final == 21
    assert flow.describe()["connections"] == [
        {"from": "load.data", "to": "select.data"},
        {"from": "load.data", "to": "summarize.data"},
        {"from": "select.selected", "to": "summarize.selected"},
    ]


def test_until_stops_at_a_named_node_or_output():
    flow = _example_graph()
    by_node = flow.run(until="select")
    by_port = flow.run(until="selected")

    assert by_node.order == ("load", "select")
    assert by_port.order == by_node.order
    assert set(by_node.outputs) == {"data", "selected"}


def test_until_rejects_a_name_that_points_to_different_steps():
    @graph.node(outputs="later")
    def first():
        return 1

    @graph.node(outputs="value", name="later")
    def second(later):
        return later + 1

    flow = graph.Graph("ambiguous target", first, second)
    with pytest.raises(graph.GraphError, match="ambiguous until"):
        flow.run(until="later")


def test_variations_are_explicit_ordered_and_independent():
    flow = _example_graph()
    runs = flow.run_many([{"scale": 1}, {"scale": 4, "offset": 2}])

    assert [run.settings["scale"] for run in runs] == [1, 4]
    assert [run["summary"] for run in runs] == [2, 24]
    with pytest.raises(TypeError, match="ordered sequence"):
        flow.run_many({"scale": [1, 4]})


def test_graph_rejects_ambiguous_names_and_settings():
    @graph.node(outputs="value")
    def first(shared=1):
        return shared

    @graph.node(outputs="value")
    def second(other=2):
        return other

    with pytest.raises(graph.GraphError, match="output port"):
        graph.Graph("duplicate output", first, second)

    @graph.node(outputs="other")
    def duplicate_setting(shared=3):
        return shared

    with pytest.raises(graph.GraphError, match="setting 'shared'"):
        graph.Graph("duplicate setting", first, duplicate_setting)


def test_output_contract_and_node_failure_keep_context():
    @graph.node(outputs=("left", "right"))
    def malformed():
        return {"left": 1}

    flow = graph.Graph("malformed", malformed)
    with pytest.raises(graph.NodeError, match="wrong ports") as error:
        flow.run()
    assert isinstance(error.value.__cause__, graph.GraphError)

    @graph.node(outputs="value")
    def fails(choice="plain"):
        raise RuntimeError("boom")

    with pytest.raises(graph.NodeError, match="choice.*plain") as error:
        graph.Graph("failure", fails).run()
    assert isinstance(error.value.__cause__, RuntimeError)


def test_multi_output_terminal_node_has_an_unambiguous_final_mapping():
    @graph.node(outputs=("mean", "count"))
    def summarize():
        return {"mean": 2.5, "count": 4}

    run = graph.Graph("two results", summarize).run()

    assert dict(run.final) == {"mean": 2.5, "count": 4}
    assert run.final_ports == ("mean", "count")


def test_unknown_settings_and_targets_fail_clearly():
    flow = _example_graph()
    with pytest.raises(graph.GraphError, match="unknown run settings"):
        flow.run(unknown=3)
    with pytest.raises(graph.GraphError, match="unknown node or output"):
        flow.run(until="missing")


def test_required_external_setting_can_be_validated_and_supplied():
    @graph.node(outputs="value")
    def source(recording):
        return recording

    flow = graph.Graph("external input", source)
    with pytest.raises(graph.GraphError, match="source.recording"):
        flow.validate()
    assert flow.validate(recording="demo") is flow
    assert flow.run(recording="demo")["value"] == "demo"


def test_widget_is_one_node_card_surface_with_controls_and_output():
    widgets = pytest.importorskip("ipywidgets")
    flow = _example_graph()
    panel = flow.widget(controls={"scale": [1, 2, 3]}, show="summary")

    assert isinstance(panel, widgets.VBox)
    surface, button, status, output = panel.children
    assert isinstance(surface, widgets.VBox)
    assert isinstance(surface.children[1], widgets.HBox)
    assert len(surface.children[1].children) == 3
    assert not any(
        "<svg" in item.value
        for item in _walk_widgets(panel)
        if isinstance(item, widgets.HTML)
    )
    assert "<svg" in flow.diagram().value
    controls = list(_walk_widgets(surface))
    assert any(isinstance(item, widgets.Dropdown) for item in controls)
    assert any(isinstance(item, widgets.IntText) for item in controls)
    surface_text = _html_text(surface)
    assert "← load.data" in surface_text
    assert "→ select.data, summarize.data" in surface_text
    assert "← select.selected" in surface_text
    assert isinstance(button, widgets.Button)
    assert isinstance(status, widgets.HTML)
    assert isinstance(output, widgets.Output)
    button.click()
    assert "Ran 3 nodes" in status.value
    assert "role='status'" in status.value
    assert "done" in _html_text(surface)


def test_failed_widget_rerun_clears_the_previous_success_state():
    widgets = pytest.importorskip("ipywidgets")

    @graph.node(outputs="value")
    def maybe_fail(fail=False):
        if fail:
            raise ValueError("requested failure")
        return "ok"

    panel = graph.Graph("rerun", maybe_fail).widget(
        controls={"fail": [False, True]}
    )
    surface, button, status, _ = panel.children
    button.click()
    assert "done" in _html_text(surface)

    fail_control = next(
        item
        for item in _walk_widgets(surface)
        if isinstance(item, widgets.Dropdown) and item.description == "Fail"
    )
    fail_control.value = True
    button.click()
    assert "Could not run" in status.value
    assert "role='alert'" in status.value
    assert "done" not in _html_text(surface)
    assert button.disabled is False


def test_widget_resets_previous_success_before_each_run():
    pytest.importorskip("ipywidgets")
    states_seen_during_run = []
    surface = None

    @graph.node(outputs="value")
    def inspect_surface():
        states_seen_during_run.append("done" in _html_text(surface))
        return 1

    panel = graph.Graph("fresh state", inspect_surface).widget()
    surface, button, _, _ = panel.children

    button.click()
    button.click()

    assert states_seen_during_run == [False, False]


def test_widget_distinguishes_flow_outputs_from_unused_side_outputs():
    @graph.node(outputs=("used", "unused"))
    def source():
        return {"used": 1, "unused": 2}

    @graph.node(outputs="result")
    def finish(used):
        return used

    surface = graph.Graph("labels", source, finish).widget().children[0]
    text = _html_text(surface)

    assert "no downstream connection" in text
    assert "flow output" in text


def test_widget_puts_scalar_value_controls_inside_node_cards():
    widgets = pytest.importorskip("ipywidgets")

    @graph.node(outputs="value")
    def configure(enabled=True, count=2, threshold=0.5, label="all"):
        return enabled, count, threshold, label

    panel = graph.Graph("ports", configure).widget()
    surface = panel.children[0]
    descendants = list(_walk_widgets(surface))

    assert any(isinstance(item, widgets.Checkbox) for item in descendants)
    assert any(isinstance(item, widgets.IntText) for item in descendants)
    assert any(isinstance(item, widgets.FloatText) for item in descendants)
    assert any(isinstance(item, widgets.Text) for item in descendants)
    assert "Hollow sockets are settings" in surface.children[0].value


def test_diagram_exposes_accessible_semantic_port_metadata():
    flow = _example_graph()
    markup = flow._diagram_html()

    assert "role='img'" in markup
    assert "aria-label='example." in markup
    assert "<title>example flow</title>" in markup
    assert "data-endpoint='load.scale'" in markup
    assert "data-direction='input'" in markup
    assert "data-type='int'" in markup
    assert "data-connected='false'" in markup
    assert "data-source='load.data'" in markup
    assert "data-target='select.data'" in markup
    assert "max-width:100%" not in markup


def test_graph_module_has_no_framework_or_scientific_runtime_dependency():
    source = Path("graph.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for item in ast.walk(tree):
        if isinstance(item, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in item.names)
        elif isinstance(item, ast.ImportFrom) and item.module:
            imported.add(item.module.split(".")[0])
    assert imported <= {
        "__future__",
        "collections",
        "dataclasses",
        "html",
        "inspect",
        "time",
        "types",
        "typing",
        "ipywidgets",
        "IPython",
    }
    assert ("rea" + "ktor") not in source.lower()
