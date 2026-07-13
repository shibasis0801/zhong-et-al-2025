"""A tiny, notebook-only graph for transparent scientific flows.

The public surface is intentionally small: decorate ordinary Python functions
with :func:`node`, place them in a :class:`Graph`, then run or inspect the flow.
Named function arguments connect to earlier outputs with the same name. Other
arguments are visible run settings.
"""

from __future__ import annotations

import html
import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


__all__ = ["Graph", "GraphError", "Node", "NodeError", "Run", "node"]


class GraphError(ValueError):
    """Raised when a graph or run setting is incomplete or ambiguous."""


class NodeError(RuntimeError):
    """Raised when one node fails while a graph is running."""


@dataclass(frozen=True)
class Node:
    """An ordinary function with named output ports."""

    name: str
    operation: Callable[..., Any]
    outputs: tuple[str, ...]
    signature: inspect.Signature

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").strip().capitalize()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.operation(*args, **kwargs)


@dataclass(frozen=True)
class Run:
    """One read-only top-level record of a flow execution."""

    graph_name: str
    settings: Mapping[str, Any]
    outputs: Mapping[str, Any]
    order: tuple[str, ...]
    timings: Mapping[str, float]
    final_ports: tuple[str, ...]

    def __getitem__(self, port: str) -> Any:
        return self.outputs[port]

    @property
    def final(self) -> Any:
        if not self.final_ports:
            return None
        if len(self.final_ports) == 1:
            return self.outputs[self.final_ports[0]]
        return MappingProxyType(
            {port: self.outputs[port] for port in self.final_ports}
        )

    @property
    def seconds(self) -> float:
        return sum(self.timings.values())


def _output_names(outputs: str | Sequence[str]) -> tuple[str, ...]:
    names = (outputs,) if isinstance(outputs, str) else tuple(outputs)
    if not names or any(not isinstance(name, str) or not name.isidentifier() for name in names):
        raise GraphError("node outputs must be one or more valid Python names")
    if len(names) != len(set(names)):
        raise GraphError("a node cannot declare the same output twice")
    return names


def node(
    function: Callable[..., Any] | None = None,
    *,
    outputs: str | Sequence[str] | None = None,
    name: str | None = None,
) -> Node | Callable[[Callable[..., Any]], Node]:
    """Turn a function into a node.

    Parameters in the function signature become named input ports or settings.
    ``outputs`` must name the value or values returned by the function.
    """

    def decorate(operation: Callable[..., Any]) -> Node:
        if not callable(operation):
            raise TypeError("node expects a callable")
        node_name = name or operation.__name__
        if not node_name.isidentifier():
            raise GraphError(f"invalid node name: {node_name!r}")
        declared_outputs = operation.__name__ if outputs is None else outputs
        signature = inspect.signature(operation)
        unsupported = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in signature.parameters.values():
            if parameter.kind in unsupported:
                raise GraphError(
                    f"node {node_name!r} must use named parameters only; "
                    f"{parameter.name!r} is not supported"
                )
        return Node(
            name=node_name,
            operation=operation,
            outputs=_output_names(declared_outputs),
            signature=signature,
        )

    return decorate(function) if function is not None else decorate


class Graph:
    """A deterministic sequence of named-port nodes for one notebook flow."""

    def __init__(self, name: str, *nodes: Node):
        if not isinstance(name, str) or not name.strip():
            raise GraphError("graph name must be non-empty")
        if not nodes:
            raise GraphError("a graph needs at least one node")
        self.name = name.strip()
        self.nodes = tuple(nodes)
        self._nodes_by_name: dict[str, Node] = {}
        self._producer: dict[str, str] = {}
        self._connections: dict[tuple[str, str], tuple[str, str]] = {}
        self._settings: dict[str, tuple[str, inspect.Parameter]] = {}
        self._build()

    def _build(self) -> None:
        for current in self.nodes:
            if not isinstance(current, Node):
                raise GraphError("Graph accepts only functions decorated with @node")
            if current.name in self._nodes_by_name:
                raise GraphError(f"duplicate node name: {current.name!r}")
            self._nodes_by_name[current.name] = current

            for parameter in current.signature.parameters.values():
                producer_name = self._producer.get(parameter.name)
                if producer_name is not None:
                    self._connections[(current.name, parameter.name)] = (
                        producer_name,
                        parameter.name,
                    )
                    continue
                previous = self._settings.get(parameter.name)
                if previous is not None:
                    raise GraphError(
                        f"setting {parameter.name!r} appears in both "
                        f"{previous[0]!r} and {current.name!r}; rename one for clarity"
                    )
                self._settings[parameter.name] = (current.name, parameter)

            for output in current.outputs:
                previous = self._producer.get(output)
                if previous is not None:
                    raise GraphError(
                        f"output port {output!r} is produced by both "
                        f"{previous!r} and {current.name!r}"
                    )
                earlier_setting = self._settings.get(output)
                if earlier_setting is not None:
                    raise GraphError(
                        f"output port {output!r} is declared after it was used as "
                        f"setting {earlier_setting[0]}.{output}; move its producer earlier"
                    )
                self._producer[output] = current.name

    @property
    def setting_defaults(self) -> Mapping[str, Any]:
        """Defaults for visible settings; required settings map to ``None``."""

        defaults = {}
        for name, (_, parameter) in self._settings.items():
            defaults[name] = (
                None
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
        return MappingProxyType(defaults)

    def validate(self, **settings: Any) -> "Graph":
        """Validate required external settings and return this graph."""

        unknown = sorted(set(settings) - set(self._settings))
        if unknown:
            raise GraphError(f"unknown run settings: {', '.join(unknown)}")
        missing = [
            f"{node_name}.{name}"
            for name, (node_name, parameter) in self._settings.items()
            if parameter.default is inspect.Parameter.empty and name not in settings
        ]
        if missing:
            joined = ", ".join(missing)
            raise GraphError(
                "required run settings have no defaults: "
                f"{joined}. Supply them to run() or add function defaults."
            )
        return self

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe description for inspection or documentation."""

        node_rows = []
        for current in self.nodes:
            inputs = []
            settings = []
            for parameter in current.signature.parameters.values():
                connection = self._connections.get((current.name, parameter.name))
                if connection is None:
                    settings.append(parameter.name)
                else:
                    inputs.append(parameter.name)
            node_rows.append(
                {
                    "name": current.name,
                    "label": current.label,
                    "inputs": inputs,
                    "settings": settings,
                    "outputs": list(current.outputs),
                }
            )
        edge_rows = [
            {
                "from": f"{source_node}.{source_port}",
                "to": f"{target_node}.{target_port}",
            }
            for (target_node, target_port), (source_node, source_port)
            in self._connections.items()
        ]
        return {
            "name": self.name,
            "nodes": node_rows,
            "connections": edge_rows,
            "settings": list(self._settings),
        }

    def _stop_index(self, until: str | None) -> int:
        if until is None:
            return len(self.nodes) - 1
        matches = {
            index
            for index, current in enumerate(self.nodes)
            if until == current.name or until in current.outputs
        }
        if len(matches) == 1:
            return matches.pop()
        if len(matches) > 1:
            raise GraphError(
                f"ambiguous until={until!r}; it names different nodes or outputs"
            )
        raise GraphError(f"unknown node or output for until={until!r}")

    def run(self, *, until: str | None = None, **settings: Any) -> Run:
        """Run nodes once in declaration order.

        ``until`` can name a node or output port and is useful while developing
        a flow. Every run starts fresh; no values are cached between runs.
        """

        unknown = sorted(set(settings) - set(self._settings))
        if unknown:
            raise GraphError(f"unknown run settings: {', '.join(unknown)}")
        stop_index = self._stop_index(until)
        executed_nodes = self.nodes[: stop_index + 1]
        effective: dict[str, Any] = {}
        for current in executed_nodes:
            for parameter in current.signature.parameters.values():
                if (current.name, parameter.name) in self._connections:
                    continue
                if parameter.name in settings:
                    effective[parameter.name] = settings[parameter.name]
                elif parameter.default is not inspect.Parameter.empty:
                    effective[parameter.name] = parameter.default
                else:
                    raise GraphError(
                        f"missing required run setting: {current.name}.{parameter.name}"
                    )

        values: dict[str, Any] = {}
        timings: dict[str, float] = {}
        order: list[str] = []
        for current in executed_nodes:
            arguments: dict[str, Any] = {}
            for parameter in current.signature.parameters.values():
                connection = self._connections.get((current.name, parameter.name))
                if connection is None:
                    arguments[parameter.name] = effective[parameter.name]
                else:
                    arguments[parameter.name] = values[connection[1]]
            started = time.perf_counter()
            try:
                returned = current.operation(**arguments)
                produced = self._normalize_outputs(current, returned)
            except Exception as error:
                raise NodeError(
                    f"node {current.name!r} failed with settings "
                    f"{self._node_settings(current, effective)!r}: {error}"
                ) from error
            timings[current.name] = time.perf_counter() - started
            order.append(current.name)
            values.update(produced)

        return Run(
            graph_name=self.name,
            settings=MappingProxyType(dict(effective)),
            outputs=MappingProxyType(dict(values)),
            order=tuple(order),
            timings=MappingProxyType(dict(timings)),
            final_ports=executed_nodes[-1].outputs,
        )

    @staticmethod
    def _node_settings(current: Node, effective: Mapping[str, Any]) -> dict[str, Any]:
        return {
            parameter.name: effective[parameter.name]
            for parameter in current.signature.parameters.values()
            if parameter.name in effective
        }

    @staticmethod
    def _normalize_outputs(current: Node, returned: Any) -> dict[str, Any]:
        if len(current.outputs) == 1:
            return {current.outputs[0]: returned}
        if not isinstance(returned, Mapping):
            raise GraphError(
                f"node {current.name!r} has multiple outputs and must return a mapping"
            )
        expected = set(current.outputs)
        actual = set(returned)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise GraphError(
                f"node {current.name!r} returned the wrong ports; "
                f"missing={missing}, extra={extra}"
            )
        return {name: returned[name] for name in current.outputs}

    def run_many(
        self,
        variations: Sequence[Mapping[str, Any]],
        *,
        until: str | None = None,
    ) -> tuple[Run, ...]:
        """Run an explicit ordered list of variations.

        The method intentionally does not create a Cartesian parameter grid.
        """

        if isinstance(variations, (str, bytes, Mapping)):
            raise TypeError("variations must be an ordered sequence of mappings")
        runs = []
        for index, variation in enumerate(variations):
            if not isinstance(variation, Mapping):
                raise TypeError(f"variation {index} is not a mapping")
            try:
                runs.append(self.run(until=until, **dict(variation)))
            except Exception as error:
                raise GraphError(f"variation {index} failed: {error}") from error
        return tuple(runs)

    @staticmethod
    def _widgets():
        try:
            import ipywidgets as widgets
        except ImportError as error:  # pragma: no cover - exercised without extras
            raise ImportError(
                "The notebook view needs ipywidgets. Install it with "
                "`python -m pip install ipywidgets`."
            ) from error
        return widgets

    def _diagram_html(self, run: Run | None = None) -> str:
        completed = set(run.order) if run is not None else set()
        node_cards = []
        description = self.describe()
        for row in description["nodes"]:
            state = "done" if row["name"] in completed else "ready"
            timing = ""
            if run is not None and row["name"] in run.timings:
                timing = f" · {run.timings[row['name']] * 1000:.1f} ms"
            inputs = ", ".join(row["inputs"]) or "—"
            outputs = ", ".join(row["outputs"]) or "—"
            settings = ", ".join(row["settings"]) or "—"
            node_cards.append(
                "<div class='tiny-graph-node'>"
                f"<strong>{html.escape(row['label'])}</strong>"
                f"<span>{html.escape(state + timing)}</span>"
                f"<small>in · {html.escape(inputs)}</small>"
                f"<small>out · {html.escape(outputs)}</small>"
                f"<small>settings · {html.escape(settings)}</small>"
                "</div>"
            )
        edge_rows = "".join(
            "<li><code>"
            f"{html.escape(edge['from'])} → {html.escape(edge['to'])}"
            "</code></li>"
            for edge in description["connections"]
        )
        cards = "<span class='tiny-graph-arrow'>→</span>".join(node_cards)
        return (
            "<style>"
            ".tiny-graph-flow{display:flex;align-items:center;gap:.5rem;"
            "flex-wrap:wrap;margin:.25rem 0 .75rem}.tiny-graph-node{display:flex;"
            "flex-direction:column;gap:.2rem;border:1px solid currentColor;"
            "border-radius:.4rem;padding:.55rem .7rem;min-width:9rem}"
            ".tiny-graph-node span,.tiny-graph-node small{opacity:.72}"
            ".tiny-graph-arrow{font-size:1.2rem}.tiny-graph-wires{margin:.25rem 0}"
            "</style>"
            f"<div><b>{html.escape(self.name)}</b>"
            "<div><small>Run order</small></div>"
            f"<div class='tiny-graph-flow'>{cards}</div>"
            "<div><small>Named connections</small></div>"
            f"<ul class='tiny-graph-wires'>{edge_rows}</ul></div>"
        )

    def diagram(self, run: Run | None = None):
        """Return a compact, static ipywidgets view of nodes and ports."""

        widgets = self._widgets()
        return widgets.HTML(value=self._diagram_html(run))

    @staticmethod
    def _control_widget(widgets: Any, name: str, specification: Any, default: Any):
        label = name.replace("_", " ").capitalize()
        if isinstance(specification, widgets.Widget):
            return specification
        if isinstance(specification, range):
            specification = list(specification)
        if isinstance(specification, (list, tuple)):
            options = list(specification)
            if not options:
                raise GraphError(f"widget control {name!r} needs at least one option")
            values = [
                item[1]
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                else item
                for item in options
            ]
            value = default if default in values else values[0]
            return widgets.Dropdown(description=label, options=options, value=value)
        if isinstance(specification, bool):
            return widgets.Checkbox(description=label, value=specification)
        if isinstance(specification, int):
            return widgets.IntText(description=label, value=specification)
        if isinstance(specification, float):
            return widgets.FloatText(description=label, value=specification)
        return widgets.Text(description=label, value=str(specification))

    def widget(
        self,
        *,
        controls: Mapping[str, Any] | None = None,
        show: str | None = None,
    ):
        """Return a small run panel with controls, diagram, status, and output."""

        widgets = self._widgets()
        if show is not None and show not in self._producer:
            raise GraphError(f"unknown output for show={show!r}")
        control_specs = dict(controls or {})
        unknown = sorted(set(control_specs) - set(self._settings))
        if unknown:
            raise GraphError(f"unknown widget controls: {', '.join(unknown)}")
        defaults = self.setting_defaults
        control_widgets = {
            name: self._control_widget(
                widgets,
                name,
                control_specs.get(name, defaults[name]),
                defaults[name],
            )
            for name in self._settings
            if name in control_specs
        }
        diagram = widgets.HTML(value=self._diagram_html())
        controls_row = widgets.HBox(
            list(control_widgets.values()),
            layout=widgets.Layout(display="flex", flex_flow="row wrap"),
        )
        run_button = widgets.Button(description="Run flow", button_style="primary")
        status = widgets.HTML(value="Change a setting or run the defaults.")
        output = widgets.Output()

        def run_clicked(_: Any) -> None:
            selected = {name: control.value for name, control in control_widgets.items()}
            status.value = "Running…"
            try:
                result = self.run(**selected)
                diagram.value = self._diagram_html(result)
                with output:
                    output.clear_output(wait=True)
                    from IPython.display import display

                    display(result[show] if show is not None else result.final)
                status.value = (
                    f"Ran {len(result.order)} nodes in {result.seconds:.3f} seconds."
                )
            except Exception as error:
                diagram.value = self._diagram_html()
                output.clear_output(wait=True)
                status.value = f"<b>Could not run:</b> {html.escape(str(error))}"

        run_button.on_click(run_clicked)
        return widgets.VBox([diagram, controls_row, run_button, status, output])
