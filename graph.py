"""graph -- a tiny, notebook-only flow graph for transparent experiments.

WHAT IT IS
    A minimal port-graph you can read in one sitting. You write ordinary
    Python functions, mark each one as a `node`, and list them in a `Graph`.
    The graph wires itself and runs top to bottom. No hidden state, no
    scheduler, no framework, no external data dependency -- just functions,
    named values, and a small optional ipywidgets view.

THE MENTAL MODEL (four ideas)
    1. A node is a function:            @graph.node(outputs="speed")
    2. What it returns is an OUTPUT port, named by `outputs`.
    3. Each function argument is an INPUT port. If an earlier node produced a
       value with the same name, they connect automatically (a "wire").
    4. An argument that nobody produces is a SETTING -- a dial you feed at run
       time. Give it a default and it becomes a control in the widget.

MINIMAL EXAMPLE
    import graph

    @graph.node(outputs="data")
    def load(scale=2):              # `scale` has no producer -> a setting
        return scale

    @graph.node(outputs="selected")
    def select(data, offset=1):     # `data` wires to load; `offset` is a setting
        return data + offset

    flow = graph.Graph("example", load, select)
    flow.run(scale=3, offset=4)["selected"]          # -> 7

FEEDING VALUES AT THE PORTS
    Yes -- the settings ARE the feedable ports. Supply them three ways:
        flow.run(offset=4)                            # one run
        flow.run_many([{"offset": 1}, {"offset": 9}]) # explicit variations
        flow.widget(controls={"offset": [1, 5, 9]})   # edit the port in its node
    Rule of thumb: a port is feedable only if NO earlier node already produces
    that name. A wired input always takes its upstream value and cannot be
    overridden at run() -- to make something feedable, don't produce it upstream
    (leave it a setting), or stop the flow early with `until=`. To feed data
    INTO the graph, give the first node a plain argument with no producer
    (e.g. `def source(recording): ...`) and pass it: run(recording=...).

EVERYDAY OPERATIONS
    flow.describe()          # plain dict of nodes, ports, and connections
    flow.diagram()           # blueprint-style canvas: nodes, ports, wires, values
    flow.run()               # run once -> a read-only Run record
    flow.run(until="select") # stop early while developing (a node or port name)
    flow.run_many([...])     # an ordered list of setting variations
    flow.widget(controls=..) # node cards with editable ports, Run, and output
    run["port"], run.final, run.settings, run.timings, run.order   # inspect

RULES & GOTCHAS
    - Nodes run in the order you list them; declare a producer before its
      consumer. A later producer that reuses an earlier setting name is rejected.
    - Every output port name is unique across the whole graph.
    - Every setting name is owned by exactly one node; share a value by passing
      it as an output, not by repeating the same setting name in two nodes.
    - Multiple outputs: declare outputs=("a", "b") and return {"a": .., "b": ..}.
    - Each run starts fresh; nothing is cached between runs.
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
            input_ports = []
            for parameter in current.signature.parameters.values():
                connection = self._connections.get((current.name, parameter.name))
                if connection is None:
                    settings.append(parameter.name)
                else:
                    inputs.append(parameter.name)
                input_ports.append(
                    {
                        "name": parameter.name,
                        "endpoint": f"{current.name}.{parameter.name}",
                        "type": self._input_type(current, parameter),
                        "connected": connection is not None,
                        "source": (
                            None
                            if connection is None
                            else f"{connection[0]}.{connection[1]}"
                        ),
                        "editable": connection is None,
                        "required": parameter.default is inspect.Parameter.empty,
                    }
                )
            node_rows.append(
                {
                    "name": current.name,
                    "label": current.label,
                    "inputs": inputs,
                    "settings": settings,
                    "outputs": list(current.outputs),
                    "input_ports": input_ports,
                    "output_ports": [
                        {
                            "name": output,
                            "endpoint": f"{current.name}.{output}",
                            "type": self._output_type(current, output),
                            "connected": any(
                                source == (current.name, output)
                                for source in self._connections.values()
                            ),
                        }
                        for output in current.outputs
                    ],
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

    @staticmethod
    def _type_label(annotation: Any, default: Any = inspect.Parameter.empty) -> str:
        """Return a short, readable type label without enforcing a type system."""

        if annotation is inspect.Parameter.empty:
            if default is inspect.Parameter.empty or default is None:
                return "value"
            annotation = type(default)
        if annotation is Any:
            return "value"
        if isinstance(annotation, str):
            label = annotation
        elif isinstance(annotation, type):
            label = annotation.__name__
        else:
            label = str(annotation).replace("typing.", "")
        return label if len(label) <= 22 else label[:21] + "…"

    @staticmethod
    def _type_colour(type_label: str) -> str:
        """Use a small Blueprint-like colour vocabulary for socket types."""

        lowered = type_label.lower()
        if "bool" in lowered:
            return "#d84a4a"
        if "float" in lowered:
            return "#d8b936"
        if "int" in lowered:
            return "#35b99a"
        if "str" in lowered or "text" in lowered:
            return "#c56bd6"
        if any(word in lowered for word in ("dict", "map", "list", "tuple", "array")):
            return "#2f9fd6"
        return "#5aa7d6"

    def _input_type(self, current: Node, parameter: inspect.Parameter) -> str:
        return self._type_label(parameter.annotation, parameter.default)

    def _output_type(self, current: Node, port: str) -> str:
        annotation = current.signature.return_annotation
        if len(current.outputs) != 1:
            annotation = inspect.Signature.empty
        return self._type_label(annotation)

    @staticmethod
    def _format_setting_value(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, (int, float, str)):
            text = str(value)
        else:
            text = type(value).__name__
        return text if len(text) <= 18 else text[:17] + "…"

    def _diagram_html(
        self,
        run: Run | None = None,
        *,
        preview_settings: Mapping[str, Any] | None = None,
    ) -> str:
        """Render the flow as an SVG node canvas: ports, wires, and values.

        Nodes sit in columns by dependency depth. Input ports are on the left,
        output ports on the right, curved wires connect matching names, and each
        node shows its current setting values. ``preview_settings`` shows pending
        control values before a run; ``run`` shows the values and timings used.
        """

        node_width = 256.0
        header_height = 42.0
        port_height = 26.0
        column_gap = 88.0
        row_gap = 28.0
        margin = 18.0
        port_radius = 5.0

        description = self.describe()
        rows = {row["name"]: row for row in description["nodes"]}
        completed = set(run.order) if run is not None else set()
        if run is not None:
            values: dict[str, Any] = dict(run.settings)
        elif preview_settings is not None:
            values = {**dict(self.setting_defaults), **dict(preview_settings)}
        else:
            values = dict(self.setting_defaults)
        output_values = {} if run is None else dict(run.outputs)

        depth: dict[str, int] = {}
        for current in self.nodes:  # producers are always declared earlier
            sources = [
                source_node
                for (target_node, _), (source_node, _) in self._connections.items()
                if target_node == current.name
            ]
            depth[current.name] = 0 if not sources else 1 + max(depth[s] for s in sources)
        max_depth = max(depth.values()) if depth else 0

        columns: dict[int, list[str]] = {}
        for current in self.nodes:
            columns.setdefault(depth[current.name], []).append(current.name)

        def node_height(row: Mapping[str, Any]) -> float:
            body = max(len(row["input_ports"]), len(row["output_ports"]), 1)
            return header_height + body * port_height + 14

        position: dict[str, tuple[float, float, float]] = {}
        canvas_bottom = margin
        for column in range(max_depth + 1):
            x = margin + column * (node_width + column_gap)
            y = margin
            for name in columns.get(column, []):
                height = node_height(rows[name])
                position[name] = (x, y, height)
                y += height + row_gap
            canvas_bottom = max(canvas_bottom, y)
        width = margin * 2 + (max_depth + 1) * node_width + max_depth * column_gap
        height = canvas_bottom - row_gap + margin

        def in_xy(name: str, index: int) -> tuple[float, float]:
            x, y, _ = position[name]
            return x, y + header_height + index * port_height + port_height / 2

        def out_xy(name: str, index: int) -> tuple[float, float]:
            x, y, _ = position[name]
            return x + node_width, y + header_height + index * port_height + port_height / 2

        wires: list[str] = []
        for (target_node, target_port), (source_node, source_port) in self._connections.items():
            source_rows = rows[source_node]["output_ports"]
            target_rows = rows[target_node]["input_ports"]
            x1, y1 = out_xy(
                source_node,
                next(index for index, item in enumerate(source_rows) if item["name"] == source_port),
            )
            x2, y2 = in_xy(
                target_node,
                next(index for index, item in enumerate(target_rows) if item["name"] == target_port),
            )
            bend = max(40.0, (x2 - x1) * 0.5)
            source_type = next(
                item["type"] for item in source_rows if item["name"] == source_port
            )
            colour = self._type_colour(source_type)
            wires.append(
                f"<path d='M{x1:.1f},{y1:.1f} C{x1 + bend:.1f},{y1:.1f} "
                f"{x2 - bend:.1f},{y2:.1f} {x2:.1f},{y2:.1f}' fill='none' "
                f"stroke='{colour}' stroke-width='2' stroke-opacity='0.9' "
                f"data-source='{html.escape(source_node)}.{html.escape(source_port)}' "
                f"data-target='{html.escape(target_node)}.{html.escape(target_port)}'/>"
            )

        parts: list[str] = []
        for current in self.nodes:
            row = rows[current.name]
            x, y, box_h = position[current.name]
            parts.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{node_width:.0f}' height='{box_h:.1f}' "
                "rx='9' fill='currentColor' fill-opacity='0.04' stroke='currentColor' "
                "stroke-opacity='0.35'/>"
            )
            parts.append(
                f"<rect x='{x + 1:.1f}' y='{y + 1:.1f}' width='{node_width - 2:.0f}' "
                f"height='{header_height - 1:.1f}' rx='8' fill='#2f9fd6' fill-opacity='0.18'/>"
            )
            parts.append(
                f"<path d='M{x:.1f},{y + header_height:.1f} h{node_width:.0f}' "
                "stroke='currentColor' stroke-opacity='0.18'/>"
            )
            parts.append(
                f"<text x='{x + 12:.1f}' y='{y + 19:.1f}' font-size='13' font-weight='600' "
                f"fill='currentColor'>{html.escape(row['label'])}</text>"
            )
            if current.name in completed:
                milliseconds = run.timings.get(current.name, 0.0) * 1000
                state, colour = f"done · {milliseconds:.1f} ms", "fill='hsl(145,58%,46%)'"
            else:
                state, colour = "ready", "fill='currentColor' fill-opacity='0.55'"
            parts.append(
                f"<text x='{x + 12:.1f}' y='{y + 34:.1f}' font-size='10' "
                f"{colour}>{html.escape(state)}</text>"
            )
            for index, port_row in enumerate(row["input_ports"]):
                port = port_row["name"]
                px, py = in_xy(current.name, index)
                colour = self._type_colour(port_row["type"])
                connected = port_row["connected"]
                if connected:
                    label = port
                else:
                    value = "required" if port_row["required"] and port not in values else self._format_setting_value(values.get(port))
                    label = f"{port} = {value}"
                parts.append(
                    f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{port_radius}' "
                    f"fill='{colour if connected else 'none'}' stroke='{colour}' stroke-width='2' "
                    f"data-endpoint='{html.escape(current.name)}.{html.escape(port)}' "
                    f"data-direction='input' data-type='{html.escape(port_row['type'])}' "
                    f"data-connected='{str(connected).lower()}'>"
                    f"<title>{html.escape(port)} input · {html.escape(port_row['type'])}</title>"
                    "</circle>"
                    f"<text x='{px + 9:.1f}' y='{py + 3.5:.1f}' font-size='11' "
                    f"fill='currentColor' fill-opacity='0.88'>{html.escape(label)}</text>"
                )
            for index, port_row in enumerate(row["output_ports"]):
                port = port_row["name"]
                px, py = out_xy(current.name, index)
                colour = self._type_colour(port_row["type"])
                value = output_values.get(port, inspect.Parameter.empty)
                label = port if value is inspect.Parameter.empty else f"{port} = {self._format_setting_value(value)}"
                parts.append(
                    f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{port_radius}' "
                    f"fill='{colour}' stroke='{colour}' stroke-width='2' "
                    f"data-endpoint='{html.escape(current.name)}.{html.escape(port)}' "
                    f"data-direction='output' data-type='{html.escape(port_row['type'])}' "
                    f"data-connected='{str(port_row['connected']).lower()}'>"
                    f"<title>{html.escape(port)} output · {html.escape(port_row['type'])}</title>"
                    "</circle>"
                    f"<text x='{px - 9:.1f}' y='{py + 3.5:.1f}' font-size='11' "
                    f"text-anchor='end' fill='currentColor' fill-opacity='0.85'>"
                    f"{html.escape(label)}</text>"
                )

        if run is not None:
            caption = f"ran {len(completed)} node(s) in {run.seconds:.3f}s · values below are this run"
        else:
            caption = "input sockets are on the left · outputs are on the right · hollow inputs are settings"
        order_text = " then ".join(current.name for current in self.nodes)
        connection_text = ", ".join(
            f"{edge['from']} to {edge['to']}" for edge in description["connections"]
        ) or "no wires"
        setting_text = ", ".join(
            f"{name}={self._format_setting_value(values.get(name))}"
            for name in self._settings
        ) or "no editable inputs"
        accessible = (
            f"{self.name}. Node order: {order_text}. "
            f"Connections: {connection_text}. Current input values: {setting_text}."
        )
        svg = (
            f"<svg xmlns='http://www.w3.org/2000/svg' "
            f"viewBox='0 0 {width:.0f} {height:.0f}' width='{width:.0f}' "
            f"height='{height:.0f}' role='img' aria-label='{html.escape(accessible, quote=True)}' "
            "style='height:auto;font-family:inherit'>"
            f"<title>{html.escape(self.name)} flow</title>"
            f"<desc>{html.escape(accessible)}</desc>"
            + "".join(wires)
            + "".join(parts)
            + "</svg>"
        )
        return (
            "<div style='margin:.25rem 0 .75rem'>"
            f"<div style='font-weight:600'>{html.escape(self.name)}</div>"
            f"<div style='opacity:.6;font-size:.8rem;margin:.1rem 0 .4rem'>"
            f"{html.escape(caption)}</div>"
            f"<div style='overflow-x:auto'>{svg}</div></div>"
        )

    def diagram(self, run: Run | None = None):
        """Return an SVG node-and-wire view of the flow (ports, wires, values)."""

        widgets = self._widgets()
        return widgets.HTML(value=self._diagram_html(run))

    @staticmethod
    def _control_widget(widgets: Any, name: str, specification: Any, default: Any):
        label = name.replace("_", " ").capitalize()
        if isinstance(specification, widgets.Widget):
            if hasattr(specification.style, "description_width"):
                specification.style.description_width = "initial"
            if specification.layout.width is None:
                specification.layout.width = "100%"
            return specification
        if isinstance(specification, range):
            specification = list(specification)
        common = {
            "description": label,
            "style": {"description_width": "initial"},
            "layout": widgets.Layout(width="100%", min_width="210px"),
        }
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
            return widgets.Dropdown(options=options, value=value, **common)
        kind = (
            widgets.Checkbox
            if isinstance(specification, bool)
            else widgets.IntText
            if isinstance(specification, int)
            else widgets.FloatText
            if isinstance(specification, float)
            else widgets.Text
        )
        value = str(specification) if kind is widgets.Text else specification
        return kind(value=value, **common)

    def widget(
        self,
        *,
        controls: Mapping[str, Any] | None = None,
        show: str | None = None,
    ):
        """Return a node-and-port editor with a Run button and visible output.

        Filled input sockets receive an earlier node's output. Hollow sockets
        are editable values. Controls are placed inside the node that owns the
        port so the visual map and the executable settings stay together.
        """

        widgets = self._widgets()
        if show is not None and show not in self._producer:
            raise GraphError(f"unknown output for show={show!r}")
        control_specs = dict(controls or {})
        unknown = sorted(set(control_specs) - set(self._settings))
        if unknown:
            raise GraphError(f"unknown widget controls: {', '.join(unknown)}")
        defaults, control_widgets = self.setting_defaults, {}
        for name, (_, parameter) in self._settings.items():
            explicit = name in control_specs
            default = defaults[name]
            if not explicit and not isinstance(default, (bool, int, float, str)):
                continue
            specification = control_specs[name] if explicit else default
            control_widgets[name] = self._control_widget(
                widgets, name, specification, default
            )

        def _socket_markup(type_label: str, *, connected: bool) -> str:
            colour = self._type_colour(type_label)
            fill = colour if connected else "transparent"
            return (
                f"<span aria-hidden='true' style='display:inline-block;width:10px;"
                f"height:10px;border:2px solid {colour};border-radius:50%;"
                f"background:{fill};box-sizing:border-box'></span>"
            )

        def _socket(type_label: str, *, connected: bool):
            return widgets.HTML(
                value=_socket_markup(type_label, connected=connected),
                layout=widgets.Layout(width="14px", min_width="14px"),
            )

        def _row(*items: Any):
            return widgets.HBox(
                list(items),
                layout=widgets.Layout(
                    width="100%", align_items="center", grid_gap="4px"
                ),
            )

        targets: dict[tuple[str, str], list[str]] = {}
        for (target_node, target_port), source in self._connections.items():
            targets.setdefault(source, []).append(f"{target_node}.{target_port}")

        node_states: dict[str, Any] = {}
        output_labels: dict[tuple[str, str], Any] = {}

        def _state_markup(state: str, *, done: bool = False) -> str:
            colour = "#218653" if done else "currentColor"
            opacity = "1" if done else ".58"
            return (
                f"<span style='color:{colour};opacity:{opacity};font-size:.76rem;"
                "white-space:nowrap'>"
                f"{html.escape(state)}</span>"
            )

        def _output_markup(current: Node, port: str, value: Any = inspect.Parameter.empty) -> str:
            type_label = self._output_type(current, port)
            destination_rows = targets.get((current.name, port), [])
            destination = (
                "→ " + ", ".join(destination_rows)
                if destination_rows
                else (
                    "flow output"
                    if current is self.nodes[-1]
                    else "no downstream connection"
                )
            )
            shown_value = ""
            if value is not inspect.Parameter.empty:
                shown_value = f" = {self._format_setting_value(value)}"
            return (
                "<div style='text-align:right;width:100%'>"
                f"<span style='font-weight:500'>{html.escape(port + shown_value)}</span>"
                f" <span style='opacity:.58;font-size:.76rem'>· {html.escape(type_label)}</span>"
                f"<div style='opacity:.65;font-size:.76rem'>{html.escape(destination)}</div>"
                "</div>"
            )

        node_cards = []
        for current in self.nodes:
            state = widgets.HTML(
                value=_state_markup("ready"),
                layout=widgets.Layout(width="auto", min_width="58px"),
            )
            node_states[current.name] = state
            title = widgets.HTML(
                value=(
                    "<div style='font-weight:650'>"
                    f"{html.escape(current.label)}"
                    f"<div style='font-weight:400;opacity:.62;font-size:.76rem'>"
                    f"{html.escape(current.name)}</div></div>"
                ),
                layout=widgets.Layout(width="100%"),
            )
            rows: list[Any] = [
                widgets.HBox(
                    [title, state],
                    layout=widgets.Layout(
                        width="100%",
                        align_items="flex-start",
                        border_bottom="1px solid var(--jp-border-color2, #6b7280)",
                        padding="2px 2px 7px 2px",
                    ),
                )
            ]
            for parameter in current.signature.parameters.values():
                connection = self._connections.get((current.name, parameter.name))
                type_label = self._input_type(current, parameter)
                socket = _socket(type_label, connected=connection is not None)
                if connection is not None:
                    source = f"{connection[0]}.{connection[1]}"
                    content = widgets.HTML(value=(
                            f"<div><span style='font-weight:500'>"
                            f"{html.escape(parameter.name)}</span>"
                            f"<div style='opacity:.65;font-size:.78rem'>"
                            f"← {html.escape(source)} · {html.escape(type_label)}</div></div>"
                        ),
                        layout=widgets.Layout(width="100%"),
                    )
                elif parameter.name in control_widgets:
                    content = widgets.VBox(
                        [
                            control_widgets[parameter.name],
                            widgets.HTML(
                                value=(
                                    "<div style='opacity:.58;font-size:.74rem;"
                                    "text-align:right'>"
                                    f"{html.escape(type_label)}</div>"
                                )
                            ),
                        ],
                        layout=widgets.Layout(width="100%", grid_gap="1px"),
                    )
                else:
                    if parameter.default is inspect.Parameter.empty:
                        value = "required at run()"
                    else:
                        value = self._format_setting_value(parameter.default)
                    content = widgets.HTML(
                        value=(
                            f"<div><span style='font-weight:500'>"
                            f"{html.escape(parameter.name)}</span>"
                            f"<div style='opacity:.65;font-size:.78rem'>"
                            f"{html.escape(value)} · {html.escape(type_label)}</div></div>"
                        ),
                        layout=widgets.Layout(width="100%"),
                    )
                rows.append(_row(socket, content))

            rows.append(
                widgets.HTML(
                    value="<div style='border-top:1px solid currentColor;opacity:.22'></div>"
                )
            )
            for port in current.outputs:
                label = widgets.HTML(
                    value=_output_markup(current, port),
                    layout=widgets.Layout(width="100%"),
                )
                output_labels[(current.name, port)] = label
                rows.append(
                    widgets.HBox(
                        [
                            label,
                            _socket(
                                self._output_type(current, port),
                                connected=True,
                            ),
                        ],
                        layout=widgets.Layout(
                            width="100%",
                            align_items="center",
                            grid_gap="4px",
                        ),
                    )
                )

            node_cards.append(
                widgets.VBox(
                    rows,
                    layout=widgets.Layout(
                        border="1px solid var(--jp-border-color2, #6b7280)",
                        padding="8px",
                        width="290px",
                        min_width="290px",
                        max_width="320px",
                        grid_gap="5px",
                    ),
                )
            )

        port_editor = widgets.VBox(
            [
                widgets.HTML(
                    value=(
                        f"<div style='font-weight:650'>{html.escape(self.name)}</div>"
                        "<div style='opacity:.65;font-size:.82rem;margin-bottom:.35rem'>"
                        "Read left to right. Hollow sockets are settings; filled sockets "
                        "show values received from another node."
                        "</div>"
                    )
                ),
                widgets.HBox(
                    node_cards,
                    layout=widgets.Layout(
                        flex_flow="row nowrap", align_items="stretch",
                        overflow="auto", grid_gap="12px", width="100%",
                        padding="2px 0 8px 0",
                    ),
                ),
            ],
            layout=widgets.Layout(width="100%"),
        )

        run_button = widgets.Button(
            description="Run flow",
            button_style="primary",
            icon="play",
        )
        status = widgets.HTML()
        output = widgets.Output()

        def _set_status(message: str, *, alert: bool = False) -> None:
            status.value = (
                f"<div role='{'alert' if alert else 'status'}' "
                f"aria-live='{'assertive' if alert else 'polite'}'>{message}</div>"
            )

        _set_status("Choose input values in their nodes, then run the flow.")

        def _reset_surface() -> None:
            for state in node_states.values():
                state.value = _state_markup("ready")
            for current in self.nodes:
                for port in current.outputs:
                    output_labels[(current.name, port)].value = _output_markup(
                        current, port
                    )

        def _preview(_change: Any = None) -> None:
            try:
                _reset_surface()
                output.clear_output(wait=True)
                _set_status("Input values changed. Run the flow to update the result.")
            except Exception as error:
                _set_status(html.escape(str(error)), alert=True)

        for control in control_widgets.values():
            control.observe(_preview, names="value")

        def run_clicked(_: Any) -> None:
            selected = {name: control.value for name, control in control_widgets.items()}
            run_button.disabled = True
            _reset_surface()
            _set_status("Running…")
            try:
                result = self.run(**selected)
                for node_name in result.order:
                    milliseconds = result.timings[node_name] * 1000
                    node_states[node_name].value = _state_markup(
                        f"done · {milliseconds:.1f} ms", done=True
                    )
                for current in self.nodes:
                    for port in current.outputs:
                        value = result.outputs.get(port, inspect.Parameter.empty)
                        output_labels[(current.name, port)].value = _output_markup(
                            current, port, value
                        )
                with output:
                    output.clear_output(wait=True)
                    from IPython.display import display

                    display(result[show] if show is not None else result.final)
                chosen = ", ".join(
                    f"{name.replace('_', ' ')}={self._format_setting_value(value)}"
                    for name, value in selected.items()
                )
                detail = f" Inputs: {html.escape(chosen)}." if chosen else ""
                _set_status(
                    f"Ran {len(result.order)} nodes in {result.seconds:.3f} seconds."
                    f"{detail}"
                )
            except Exception as error:
                _reset_surface()
                output.clear_output(wait=True)
                _set_status(
                    f"<b>Could not run:</b> {html.escape(str(error))}",
                    alert=True,
                )
            finally:
                run_button.disabled = False

        run_button.on_click(run_clicked)
        return widgets.VBox([port_editor, run_button, status, output])
