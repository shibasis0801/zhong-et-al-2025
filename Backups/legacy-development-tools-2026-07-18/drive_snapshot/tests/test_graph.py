import ast
from html.parser import HTMLParser
from pathlib import Path

import pytest

import graph


class _WireParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.wires = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "path" and "graph-wire" in values.get("class", "").split():
            self.wires[(values["data-source"], values["data-target"])] = values


def _wire_attributes(markup):
    parser = _WireParser()
    parser.feed(markup)
    return parser.wires


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


def _panel_parts(panel):
    toolbar, surface, output = panel.children
    button, status = toolbar.children
    scroller = surface.children[1]
    canvas = scroller.children[0]
    diagram, *controls = canvas.children
    return surface, diagram, controls, button, status, output


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


def test_until_runs_only_the_target_dependency_branch():
    calls = []

    @graph.node(outputs="data")
    def source():
        calls.append("source")
        return 2

    @graph.node(outputs="side")
    def unrelated(label="unused"):
        calls.append("unrelated")
        return label

    @graph.node(outputs="result")
    def target(data):
        calls.append("target")
        return data + 1

    flow = graph.Graph("branches", source, unrelated, target)
    run = flow.run(until="result")

    assert calls == ["source", "target"]
    assert run.order == ("source", "target")
    with pytest.raises(graph.GraphError, match="outside this target"):
        flow.run(until="result", label="ignored")


def test_run_exposes_every_terminal_branch_without_changing_final():
    @graph.node(outputs="data")
    def source():
        return 3

    @graph.node(outputs="left")
    def summarize_left(data):
        return data + 1

    @graph.node(outputs="right")
    def summarize_right(data):
        return data * 2

    run = graph.Graph("fan out", source, summarize_left, summarize_right).run()

    assert run.final == 6
    assert run.terminal_ports == ("left", "right")
    assert dict(run.terminals) == {"left": 4, "right": 6}


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
    assert error.value.node_name == "fails"
    assert error.value.completed == ()
    assert error.value.inputs == {"choice": "plain"}


def test_node_failure_keeps_completed_steps_and_bounded_input_summaries():
    class ArrayLike:
        shape = (452, 18, 48)
        dtype = "float32"

    @graph.node(outputs="data")
    def load():
        return ArrayLike()

    @graph.node(outputs="result")
    def fail(data):
        raise RuntimeError("bad slice")

    with pytest.raises(graph.NodeError, match="ArrayLike.*452×18×48") as error:
        graph.Graph("context", load, fail).run()

    assert error.value.node_name == "fail"
    assert error.value.completed == ("load",)
    assert set(error.value.timings) == {"load"}
    assert error.value.inputs["data"] == "ArrayLike · 452×18×48 · float32"


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


def test_mutable_defaults_are_rejected_before_they_can_leak_between_runs():
    with pytest.raises(graph.GraphError, match="mutable default"):

        @graph.node(outputs="value")
        def unsafe(items=[]):
            items.append(1)
            return items


def test_sync_node_returning_an_awaitable_fails_at_its_own_step():
    @graph.node(outputs="value")
    def returns_awaitable():
        async def later():
            return 1

        return later()

    with pytest.raises(graph.NodeError, match="returned an awaitable") as error:
        graph.Graph("awaitable", returns_awaitable).run()

    assert isinstance(error.value.__cause__, graph.GraphError)


def test_failure_context_survives_hostile_value_and_error_representations():
    class HostileValue:
        @property
        def shape(self):
            raise RuntimeError("shape exploded")

    class HostileError(ValueError):
        def __str__(self):
            raise RuntimeError("message exploded")

    @graph.node(outputs="data")
    def source():
        return HostileValue()

    @graph.node(outputs="result")
    def fail(data):
        raise HostileError()

    with pytest.raises(graph.NodeError, match="message unavailable") as error:
        graph.Graph("safe context", source, fail).run()

    assert isinstance(error.value.__cause__, HostileError)
    assert error.value.inputs == {"data": "HostileValue"}


def test_widget_requires_explicit_controls_for_required_input_ports():
    @graph.node(outputs="value")
    def source(recording):
        return recording

    with pytest.raises(graph.GraphError, match="source.recording"):
        graph.Graph("external input", source).widget()


def test_widget_combines_a_wired_map_with_native_controls_and_output():
    widgets = pytest.importorskip("ipywidgets")
    flow = _example_graph()
    panel = flow.widget(controls={"scale": [1, 2, 3]}, show="summary")

    assert isinstance(panel, widgets.VBox)
    surface, diagram, overlay_controls, button, status, output = _panel_parts(panel)
    assert isinstance(surface, widgets.VBox)
    assert isinstance(diagram, widgets.HTML)
    assert "<svg" in diagram.value
    assert "data-source='load.data'" in diagram.value
    assert "data-target='select.data'" in diagram.value
    assert len(overlay_controls) == 2
    assert "<svg" in flow.diagram().value
    descendants = list(_walk_widgets(surface))
    assert any(isinstance(item, widgets.Dropdown) for item in descendants)
    assert any(isinstance(item, widgets.IntText) for item in descendants)
    assert "Input controls and port details" not in _html_text(surface)
    assert all(
        control.layout.grid_area == "canvas" for control in overlay_controls
    )
    assert "graph-diagram-layer" in diagram._dom_classes
    assert all("graph-control-layer" in control._dom_classes for control in overlay_controls)
    assert ".graph-control-layer{position:relative;z-index:2}" in diagram.value
    assert isinstance(button, widgets.Button)
    assert isinstance(status, widgets.HTML)
    assert isinstance(output, widgets.Output)
    button.click()
    assert "Ran 3 nodes" in status.value
    assert "role='status'" in status.value
    assert "done" in _html_text(surface)
    assert panel.last_run["summary"] == 6
    assert panel.last_error is None
    assert "summary = 6 · result" in diagram.value

    scale = next(
        item
        for item in _walk_widgets(surface)
        if isinstance(item, widgets.Dropdown) and item.description == "Scale"
    )
    scale.value = 3
    assert "Current input values: scale=3" in diagram.value
    assert ">scale = 3<" not in diagram.value
    assert panel.last_run is None
    assert "Input values changed" in status.value


def test_widget_clears_stale_output_before_running_and_after_input_changes(
    monkeypatch,
):
    events = []

    @graph.node(outputs="value")
    def calculate(scale=1):
        events.append("run")
        return scale

    panel = graph.Graph("clear output", calculate).widget(
        controls={"scale": [1, 2]}
    )
    surface, _, _, button, _, output = _panel_parts(panel)
    scale = next(
        item
        for item in _walk_widgets(surface)
        if getattr(item, "description", "") == "Scale"
    )

    def record_clear(*_args, **kwargs):
        events.append(("clear", kwargs.get("wait")))

    monkeypatch.setattr(output, "clear_output", record_clear)
    button.click()
    scale.value = 2

    assert events[:2] == [("clear", False), "run"]
    assert events[-1] == ("clear", False)
    assert panel.last_run is None


def test_widget_blocks_reentrant_runs_and_disables_controls_during_execution():
    calls = []
    button = None
    scale_control = None

    @graph.node(outputs="value")
    def calculate(scale=1):
        calls.append((button.disabled, scale_control.disabled))
        button.click()
        return scale

    panel = graph.Graph("one run", calculate).widget(
        controls={"scale": [1, 2]}
    )
    surface, _, _, button, _, _ = _panel_parts(panel)
    scale_control = next(
        item
        for item in _walk_widgets(surface)
        if getattr(item, "description", "") == "Scale"
    )

    button.click()

    assert calls == [(True, True)]
    assert button.disabled is False
    assert scale_control.disabled is False


def test_widget_marks_a_completed_run_stale_if_a_control_changes_during_it():
    scale_control = None

    @graph.node(outputs="value")
    def calculate(scale=1):
        if scale == 1:
            scale_control.value = 2
        return scale

    panel = graph.Graph("changing input", calculate).widget(
        controls={"scale": [1, 2]}
    )
    surface, _, _, button, status, _ = _panel_parts(panel)
    scale_control = next(
        item
        for item in _walk_widgets(surface)
        if getattr(item, "description", "") == "Scale"
    )

    button.click()

    assert scale_control.value == 2
    assert panel.last_run is None
    assert "Input values changed" in status.value


def test_widget_recovers_if_the_pre_run_diagram_cannot_render(monkeypatch):
    flow = _example_graph()
    panel = flow.widget(controls={"scale": [1, 2]})
    surface, _, controls, button, status, _ = _panel_parts(panel)

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("diagram unavailable")

    monkeypatch.setattr(flow, "_diagram_html", fail_render)
    button.click()

    assert panel.last_run is None
    assert isinstance(panel.last_error, RuntimeError)
    assert "diagram unavailable" in status.value
    assert button.disabled is False
    assert all(control.disabled is False for control in controls)


def test_widget_keeps_a_valid_run_when_only_result_display_fails(monkeypatch):
    display_module = pytest.importorskip("IPython.display")
    panel = _example_graph().widget(show="summary")
    _, _, _, button, status, _ = _panel_parts(panel)

    def fail_display(_value):
        raise RuntimeError("display unavailable")

    monkeypatch.setattr(display_module, "display", fail_display)
    button.click()

    assert panel.last_run["summary"] == 6
    assert isinstance(panel.last_error, RuntimeError)
    assert "Run completed, but the result view failed" in status.value
    assert "display unavailable" in status.value


def test_widget_closes_only_matplotlib_figures_created_by_each_run():
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close("all")
    existing = plt.figure()

    @graph.node(outputs="figure")
    def plot():
        figure, _ = plt.subplots()
        return figure

    panel = graph.Graph("plot", plot).widget(show="figure")
    _, _, _, button, _, _ = _panel_parts(panel)

    button.click()
    first = panel.last_run["figure"]
    button.click()
    second = panel.last_run["figure"]

    assert plt.get_fignums() == [existing.number]
    assert len(first.axes) == 1
    assert len(second.axes) == 1
    plt.close(existing)


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
    surface, _, _, button, status, _ = _panel_parts(panel)
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
    assert "failed" in _html_text(surface)
    assert panel.last_run is None
    assert isinstance(panel.last_error, graph.NodeError)
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
    surface, _, _, button, _, _ = _panel_parts(panel)

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

    surface, _, _, _, _, _ = _panel_parts(
        graph.Graph("labels", source, finish).widget()
    )
    text = _html_text(surface)

    assert "unused" in text
    assert "result" in text


def test_widget_puts_scalar_value_controls_inside_the_wired_canvas():
    widgets = pytest.importorskip("ipywidgets")

    @graph.node(outputs="value")
    def configure(enabled=True, count=2, threshold=0.5, label="all"):
        return enabled, count, threshold, label

    panel = graph.Graph("ports", configure).widget()
    surface, _, _, _, _, _ = _panel_parts(panel)
    descendants = list(_walk_widgets(surface))

    assert any(isinstance(item, widgets.Checkbox) for item in descendants)
    assert any(isinstance(item, widgets.IntText) for item in descendants)
    assert any(isinstance(item, widgets.FloatText) for item in descendants)
    assert any(isinstance(item, widgets.Text) for item in descendants)
    assert "Edit hollow input ports directly inside their nodes" in surface.children[0].value
    controls = [
        item
        for item in descendants
        if isinstance(
            item,
            (widgets.Checkbox, widgets.IntText, widgets.FloatText, widgets.Text),
        )
    ]
    assert all(item.layout.min_width in ("0", "0px") for item in controls)


def test_widget_uses_one_shared_canvas_without_a_duplicate_card_strip():
    widgets = pytest.importorskip("ipywidgets")
    panel = _example_graph().widget(controls={"scale": [1, 2, 3]})
    surface, diagram, controls, _, _, _ = _panel_parts(panel)
    scroller = surface.children[1]
    canvas = scroller.children[0]

    assert isinstance(canvas, widgets.GridBox)
    assert canvas.layout.grid_template_areas == '"canvas"'
    assert diagram.layout.grid_area == "canvas"
    assert scroller.layout.overflow == "auto hidden"
    assert canvas.layout.overflow == "hidden"
    assert len(controls) == 2
    assert all(control.layout.grid_area == "canvas" for control in controls)
    assert diagram.layout.order == "0"
    assert all(control.layout.order == "1" for control in controls)
    assert all(control.layout.min_width in ("0", "0px") for control in controls)
    assert "Input controls and port details" not in _html_text(surface)


def test_widget_rejects_non_value_widgets_with_a_clear_message():
    widgets = pytest.importorskip("ipywidgets")

    @graph.node(outputs="value")
    def configure(scale=1):
        return scale

    with pytest.raises(graph.GraphError, match="must be a value control"):
        graph.Graph("controls", configure).widget(
            controls={"scale": widgets.Button(description="Not a value")}
        )


def test_skip_wire_bypasses_intermediate_nodes_without_complicating_direct_wires():
    flow = _example_graph()
    wires = _wire_attributes(flow._diagram_html(canvas_only=True))
    bypass = wires[("load.data", "summarize.data")]
    direct = wires[("load.data", "select.data")]

    _, node_y, _ = flow._diagram_geometry()["position"]["select"]
    assert bypass["data-route"] == "bypass"
    assert float(bypass["data-lane-y"]) < node_y - 4
    assert "H" in bypass["d"]
    assert direct["data-route"] == "direct"
    assert "H" not in direct["d"]


def test_multiple_skip_wires_use_distinct_clear_lanes():
    @graph.node(outputs=("left", "right", "through", "pad1", "pad2"))
    def source():
        return {
            "left": 2,
            "right": 3,
            "through": 1,
            "pad1": 0,
            "pad2": 0,
        }

    @graph.node(outputs="bridge")
    def middle(through, pad1, pad2):
        return through + pad1 + pad2

    @graph.node(outputs="result")
    def target(bridge, left, right):
        return bridge + left + right

    flow = graph.Graph("two bypasses", source, middle, target)
    wires = _wire_attributes(flow._diagram_html(canvas_only=True))
    lanes = {
        float(wires[("source.left", "target.left")]["data-lane-y"]),
        float(wires[("source.right", "target.right")]["data-lane-y"]),
    }
    _, node_y, _ = flow._diagram_geometry()["position"]["middle"]

    assert len(lanes) == 2
    assert max(lanes) < node_y - 4


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
    assert "tabindex='0'" not in markup


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
        "sys",
        "time",
        "types",
        "typing",
        "ipywidgets",
        "IPython",
    }
    assert ("rea" + "ktor") not in source.lower()


def test_async_nodes_are_rejected_instead_of_returning_coroutines():
    with pytest.raises(graph.GraphError, match="async node"):

        @graph.node(outputs="value")
        async def unsupported():
            return 1
