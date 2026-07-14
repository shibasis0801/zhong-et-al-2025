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
    panel = flow.widget(...) # wired map, editable input ports, Run, and output
    panel.last_run           # the Run made through the UI, available to Python
    run["port"], run.final, run.terminals, run.settings, run.timings

RULES & GOTCHAS
    - Nodes run in the order you list them; declare a producer before its
      consumer. A later producer that reuses an earlier setting name is rejected.
    - Every output port name is unique across the whole graph.
    - Every setting name is owned by exactly one node; share a value by passing
      it as an output, not by repeating the same setting name in two nodes.
    - Multiple outputs: declare outputs=("a", "b") and return {"a": .., "b": ..}.
    - Each run starts fresh; nothing is cached between runs.
    - Fan-out creates visible independent branches. They still execute one at a
      time in declaration order: there is no hidden thread/process scheduler.
    - Treat wired arrays and dictionaries as read-only so sibling branches stay
      deterministic without expensive deep copies.
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
    """Raised when one node fails while a graph is running.

    The compact attributes are useful in notebooks: they identify the failed
    step without printing large arrays or hiding work completed beforehand.
    """

    def __init__(
        self,
        message: str,
        *,
        node_name: str,
        completed: Sequence[str] = (),
        timings: Mapping[str, float] | None = None,
        inputs: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.node_name = node_name
        self.completed = tuple(completed)
        self.timings = MappingProxyType(dict(timings or {}))
        self.inputs = MappingProxyType(dict(inputs or {}))


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
    terminal_ports: tuple[str, ...]

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

    @property
    def terminals(self) -> Mapping[str, Any]:
        """Every output not consumed by another node in this run."""

        return MappingProxyType(
            {port: self.outputs[port] for port in self.terminal_ports}
        )


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
        if inspect.iscoroutinefunction(operation):
            raise GraphError("async node functions are not supported in notebook flows")
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

    def _execution_nodes(self, until: str | None) -> tuple[Node, ...]:
        """Return the target's dependency closure in deterministic order."""

        if until is None:
            return self.nodes
        target = self.nodes[self._stop_index(until)]
        required = {target.name}
        changed = True
        while changed:
            changed = False
            for (target_node, _), (source_node, _) in self._connections.items():
                if target_node in required and source_node not in required:
                    required.add(source_node)
                    changed = True
        return tuple(current for current in self.nodes if current.name in required)

    def run(self, *, until: str | None = None, **settings: Any) -> Run:
        """Run nodes once in declaration order.

        ``until`` can name a node or output port and is useful while developing
        a flow. Every run starts fresh; no values are cached between runs.
        """

        unknown = sorted(set(settings) - set(self._settings))
        if unknown:
            raise GraphError(f"unknown run settings: {', '.join(unknown)}")
        executed_nodes = self._execution_nodes(until)
        active_settings = {
            parameter.name
            for current in executed_nodes
            for parameter in current.signature.parameters.values()
            if (current.name, parameter.name) not in self._connections
        }
        skipped = sorted(set(settings) - active_settings)
        if skipped:
            raise GraphError(
                "run settings belong to nodes outside this target: "
                + ", ".join(skipped)
            )
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
                input_summary = {
                    name: self._value_summary(value)
                    for name, value in arguments.items()
                }
                raise NodeError(
                    f"node {current.name!r} failed with inputs "
                    f"{input_summary!r}: {error}",
                    node_name=current.name,
                    completed=order,
                    timings=timings,
                    inputs=input_summary,
                ) from error
            timings[current.name] = time.perf_counter() - started
            order.append(current.name)
            values.update(produced)

        executed_names = {current.name for current in executed_nodes}
        consumed = {
            source_port
            for (target_node, _), (source_node, source_port)
            in self._connections.items()
            if target_node in executed_names and source_node in executed_names
        }
        terminal_ports = tuple(
            output
            for current in executed_nodes
            for output in current.outputs
            if output not in consumed
        )
        return Run(
            graph_name=self.name,
            settings=MappingProxyType(dict(effective)),
            outputs=MappingProxyType(dict(values)),
            order=tuple(order),
            timings=MappingProxyType(dict(timings)),
            final_ports=executed_nodes[-1].outputs,
            terminal_ports=terminal_ports,
        )

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
    def _value_summary(value: Any) -> str:
        """Describe a value without expanding scientific arrays or tables."""

        if value is None:
            return "—"
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, (int, float, str)):
            text = str(value)
            return text if len(text) <= 32 else text[:31] + "…"
        kind = type(value).__name__
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                dimensions = "×".join(str(int(size)) for size in tuple(shape))
            except (TypeError, ValueError):
                dimensions = str(shape)
            dtype = getattr(value, "dtype", None)
            suffix = f" · {dtype}" if dtype is not None else ""
            return f"{kind} · {dimensions or 'scalar'}{suffix}"
        if isinstance(value, Mapping):
            return f"{kind} · {len(value)} keys"
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return f"{kind} · {len(value)} items"
        return kind

    @staticmethod
    def _format_setting_value(value: Any) -> str:
        text = Graph._value_summary(value)
        return text if len(text) <= 32 else text[:31] + "…"

    def _diagram_geometry(self) -> dict[str, Any]:
        """Return the one compact layout shared by SVG wires and UI controls."""

        node_width = 256.0
        header_height = 42.0
        port_height = 30.0
        column_gap = 88.0
        row_gap = 28.0
        margin = 18.0

        description = self.describe()
        rows = {row["name"]: row for row in description["nodes"]}
        depth: dict[str, int] = {}
        for current in self.nodes:  # producers are always declared earlier
            sources = [
                source_node
                for (target_node, _), (source_node, _) in self._connections.items()
                if target_node == current.name
            ]
            depth[current.name] = (
                0 if not sources else 1 + max(depth[source] for source in sources)
            )
        max_depth = max(depth.values()) if depth else 0

        columns: dict[int, list[str]] = {}
        for current in self.nodes:
            columns.setdefault(depth[current.name], []).append(current.name)

        position: dict[str, tuple[float, float, float]] = {}
        canvas_bottom = margin
        for column in range(max_depth + 1):
            x = margin + column * (node_width + column_gap)
            y = margin
            for name in columns.get(column, []):
                row = rows[name]
                body_rows = max(
                    len(row["input_ports"]), len(row["output_ports"]), 1
                )
                height = header_height + body_rows * port_height + 14
                position[name] = (x, y, height)
                y += height + row_gap
            canvas_bottom = max(canvas_bottom, y)

        width = margin * 2 + (max_depth + 1) * node_width + max_depth * column_gap
        height = canvas_bottom - row_gap + margin
        return {
            "description": description,
            "rows": rows,
            "position": position,
            "node_width": node_width,
            "header_height": header_height,
            "port_height": port_height,
            "margin": margin,
            "width": width,
            "height": height,
        }

    def _diagram_html(
        self,
        run: Run | None = None,
        *,
        preview_settings: Mapping[str, Any] | None = None,
        interactive_settings: Sequence[str] = (),
        error: NodeError | None = None,
        canvas_only: bool = False,
    ) -> str:
        """Render the flow as an SVG node canvas: ports, wires, and values.

        Nodes sit in columns by dependency depth. Input ports are on the left,
        output ports on the right, curved wires connect matching names, and each
        node shows its current setting values. ``preview_settings`` shows pending
        control values before a run; ``run`` shows the values and timings used.
        """

        geometry = self._diagram_geometry()
        description = geometry["description"]
        rows = geometry["rows"]
        position = geometry["position"]
        node_width = geometry["node_width"]
        header_height = geometry["header_height"]
        port_height = geometry["port_height"]
        width = geometry["width"]
        height = geometry["height"]
        port_radius = 5.0
        interactive = set(interactive_settings)

        completed = (
            set(run.order)
            if run is not None
            else set(error.completed)
            if error is not None
            else set()
        )
        timings = (
            run.timings
            if run is not None
            else error.timings
            if error is not None
            else {}
        )
        if run is not None:
            values: dict[str, Any] = dict(run.settings)
        elif preview_settings is not None:
            values = {**dict(self.setting_defaults), **dict(preview_settings)}
        else:
            values = dict(self.setting_defaults)
        output_values = {} if run is None else dict(run.outputs)

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
            wire_label = (
                f"{source_node}.{source_port} to {target_node}.{target_port}"
            )
            wires.append(
                f"<path d='M{x1:.1f},{y1:.1f} C{x1 + bend:.1f},{y1:.1f} "
                f"{x2 - bend:.1f},{y2:.1f} {x2:.1f},{y2:.1f}' fill='none' "
                f"stroke='{colour}' stroke-width='2' stroke-opacity='0.9' "
                "class='graph-wire' tabindex='0' "
                f"aria-label='{html.escape(wire_label, quote=True)}' "
                f"data-source='{html.escape(source_node)}.{html.escape(source_port)}' "
                f"data-target='{html.escape(target_node)}.{html.escape(target_port)}'>"
                f"<title>{html.escape(wire_label)}</title></path>"
            )

        parts: list[str] = []
        for order_number, current in enumerate(self.nodes, start=1):
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
            parts.append(
                f"<circle cx='{x + node_width - 17:.1f}' cy='{y + 17:.1f}' r='10' "
                "fill='currentColor' fill-opacity='0.12' stroke='currentColor' "
                "stroke-opacity='0.32'/>"
                f"<text x='{x + node_width - 17:.1f}' y='{y + 20.5:.1f}' "
                "text-anchor='middle' font-size='10' font-weight='700' "
                f"fill='currentColor'>{order_number}</text>"
            )
            if error is not None and current.name == error.node_name:
                state, colour = "failed", "fill='#d84a4a'"
            elif current.name in completed:
                milliseconds = timings.get(current.name, 0.0) * 1000
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
                if not connected and port in interactive:
                    label = ""
                elif connected:
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
                )
                if label:
                    parts.append(
                        f"<text x='{px + 9:.1f}' y='{py + 3.5:.1f}' font-size='11' "
                        f"fill='currentColor' fill-opacity='0.88'>{html.escape(label)}</text>"
                    )
            for index, port_row in enumerate(row["output_ports"]):
                port = port_row["name"]
                px, py = out_xy(current.name, index)
                colour = self._type_colour(port_row["type"])
                value = output_values.get(port, inspect.Parameter.empty)
                connected = port_row["connected"]
                role = (
                    ""
                    if connected
                    else " · result"
                    if current is self.nodes[-1]
                    else " · unused"
                )
                label = port if value is inspect.Parameter.empty else f"{port} = {self._format_setting_value(value)}"
                label += role
                parts.append(
                    f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{port_radius}' "
                    f"fill='{colour if connected else 'none'}' stroke='{colour}' stroke-width='2' "
                    f"data-endpoint='{html.escape(current.name)}.{html.escape(port)}' "
                    f"data-direction='output' data-type='{html.escape(port_row['type'])}' "
                    f"data-connected='{str(port_row['connected']).lower()}'>"
                    f"<title>{html.escape(port)} output · {html.escape(port_row['type'])}</title>"
                    "</circle>"
                    f"<text x='{px - 9:.1f}' y='{py + 3.5:.1f}' font-size='11' "
                    f"text-anchor='end' fill='currentColor' fill-opacity='0.85'>"
                    f"{html.escape(label)}</text>"
                )

        if error is not None:
            caption = f"stopped at {error.node_name} · completed steps stay visible"
        elif run is not None:
            caption = f"ran {len(completed)} node(s) in {run.seconds:.3f}s · values below are this run"
        else:
            caption = (
                "inputs left · outputs right · hollow sockets are editable or unused · "
                "side-by-side branches still run sequentially in numbered order"
            )
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
            "style='height:auto;font-family:inherit;display:block'>"
            f"<title>{html.escape(self.name)} flow</title>"
            f"<desc>{html.escape(accessible)}</desc>"
            "<style>.graph-wire:hover,.graph-wire:focus{stroke-width:4;outline:none}</style>"
            + "".join(wires)
            + "".join(parts)
            + "</svg>"
        )
        if canvas_only:
            return svg
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
            specification.layout.min_width = "0"
            specification.layout.flex = "1 1 auto"
            return specification
        if isinstance(specification, range):
            specification = list(specification)
        common = {
            "description": label,
            "style": {"description_width": "initial"},
            "layout": widgets.Layout(
                width="100%", min_width="0", flex="1 1 auto"
            ),
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

        interactive_names = tuple(control_widgets)
        geometry = self._diagram_geometry()
        width = geometry["width"]
        height = geometry["height"]
        diagram_view = widgets.HTML(
            value=self._diagram_html(
                preview_settings=self.setting_defaults,
                interactive_settings=interactive_names,
                canvas_only=True,
            ),
            layout=widgets.Layout(
                grid_area="canvas",
                width=f"{width:.0f}px",
                min_width=f"{width:.0f}px",
                height=f"{height:.0f}px",
                overflow="hidden",
            ),
        )
        overlay_controls = []
        for name, control in control_widgets.items():
            node_name, _ = self._settings[name]
            port_rows = geometry["rows"][node_name]["input_ports"]
            index = next(
                index
                for index, port_row in enumerate(port_rows)
                if port_row["name"] == name
            )
            x, y, _ = geometry["position"][node_name]
            output_on_row = index < len(
                geometry["rows"][node_name]["output_ports"]
            )
            control_width = (
                geometry["node_width"] * 0.56 - 14
                if output_on_row
                else geometry["node_width"] - 18
            )
            top = (
                y
                + geometry["header_height"]
                + index * geometry["port_height"]
                + 2
            )
            left = x + 9
            control.layout.grid_area = "canvas"
            control.layout.align_self = "flex-start"
            control.layout.flex = "0 0 auto"
            control.layout.width = f"{control_width:.0f}px"
            control.layout.min_width = "0"
            control.layout.max_width = f"{control_width:.0f}px"
            control.layout.height = f"{geometry['port_height'] - 4:.0f}px"
            control.layout.margin = f"{top:.0f}px 0 0 {left:.0f}px"
            overlay_controls.append(control)

        canvas = widgets.GridBox(
            [diagram_view, *overlay_controls],
            layout=widgets.Layout(
                grid_template_columns=f"{width:.0f}px",
                grid_template_rows=f"{height:.0f}px",
                grid_template_areas='"canvas"',
                flex="0 0 auto",
                width=f"{width:.0f}px",
                min_width=f"{width:.0f}px",
                height=f"{height:.0f}px",
                overflow="hidden",
            ),
        )
        canvas_scroll = widgets.Box(
            [canvas],
            layout=widgets.Layout(
                width="100%", min_width="0", overflow="auto hidden",
                padding="0 0 6px 0",
            ),
        )
        surface = widgets.VBox(
            [
                widgets.HTML(
                    value=(
                        f"<div style='font-weight:700;font-size:1rem'>{html.escape(self.name)}</div>"
                        "<div style='opacity:.78;font-size:.82rem;margin:.15rem 0 .4rem'>"
                        "Edit hollow input ports directly inside their nodes, then run the flow. "
                        "Filled sockets carry wired values; side-by-side branches remain "
                        "deterministic and execute sequentially in numbered order."
                        "</div>"
                    )
                ),
                canvas_scroll,
            ],
            layout=widgets.Layout(width="100%", min_width="0", grid_gap="4px"),
        )
        run_button = widgets.Button(
            description="Run flow",
            button_style="primary",
            icon="play",
            tooltip="Run every numbered step with the visible input values",
            layout=widgets.Layout(width="132px", min_width="132px"),
        )
        status = widgets.HTML(layout=widgets.Layout(flex="1 1 auto", min_width="0"))
        output = widgets.Output()

        def _set_status(message: str, *, alert: bool = False) -> None:
            status.value = (
                f"<div role='{'alert' if alert else 'status'}' "
                f"aria-live='{'assertive' if alert else 'polite'}'>{message}</div>"
            )

        _set_status("Choose input values in their nodes, then run the flow.")

        toolbar = widgets.HBox(
            [run_button, status],
            layout=widgets.Layout(
                width="100%", min_width="0", align_items="center", grid_gap="10px"
            ),
        )
        panel = widgets.VBox(
            [toolbar, surface, output],
            layout=widgets.Layout(width="100%", min_width="0", grid_gap="6px"),
        )
        panel.last_run = None
        panel.last_error = None

        def _selected_settings() -> dict[str, Any]:
            return {
                name: control.value for name, control in control_widgets.items()
            }

        def _preview(_change: Any = None) -> None:
            try:
                selected = _selected_settings()
                diagram_view.value = self._diagram_html(
                    preview_settings=selected,
                    interactive_settings=interactive_names,
                    canvas_only=True,
                )
                panel.last_run = None
                panel.last_error = None
                output.clear_output(wait=True)
                _set_status("Input values changed. Run the flow to update the result.")
            except Exception as error:
                _set_status(html.escape(str(error)), alert=True)

        for control in control_widgets.values():
            control.observe(_preview, names="value")

        def run_clicked(_: Any) -> None:
            selected = _selected_settings()
            run_button.disabled = True
            diagram_view.value = self._diagram_html(
                preview_settings=selected,
                interactive_settings=interactive_names,
                canvas_only=True,
            )
            _set_status("Running…")
            try:
                result = self.run(**selected)
                panel.last_run = result
                panel.last_error = None
                diagram_view.value = self._diagram_html(
                    result,
                    interactive_settings=interactive_names,
                    canvas_only=True,
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
                panel.last_run = None
                panel.last_error = error
                diagram_view.value = self._diagram_html(
                    preview_settings=selected,
                    interactive_settings=interactive_names,
                    error=error if isinstance(error, NodeError) else None,
                    canvas_only=True,
                )
                output.clear_output(wait=True)
                _set_status(
                    f"<b>Could not run:</b> {html.escape(str(error))}",
                    alert=True,
                )
            finally:
                run_button.disabled = False

        run_button.on_click(run_clicked)
        return panel
