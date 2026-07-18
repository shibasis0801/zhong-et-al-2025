"""graph -- a tiny, notebook-only flow graph for transparent experiments.

WHAT IT IS
    A minimal port graph designed for quick inspection. Ordinary Python
    functions become `node` objects and are listed in a `Graph`. The graph
    wires itself and runs top to bottom. No hidden state, no
    scheduler, no framework, no external data dependency -- just functions,
    named values, and a small optional ipywidgets view.

THE MENTAL MODEL (four ideas)
    1. A node is a function:            @graph.node(outputs="speed")
    2. What it returns is an OUTPUT port, named by `outputs`.
    3. Each function argument is an INPUT port. If an earlier node produced a
       value with the same name, they connect automatically (a "wire").
    4. An argument without a producer is a SETTING -- a value supplied at run
       time. A default makes it a control in the widget.

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
    - Nodes run in declared order; place a producer before its
      consumer. A later producer that reuses an earlier setting name is rejected.
    - Every output port name is unique across the whole graph.
    - Every setting name is owned by exactly one node; share a value by passing
      it as an output, not by repeating the same setting name in two nodes.
    - Multiple outputs: declare outputs=("a", "b") and return {"a": .., "b": ..}.
    - Each run starts fresh; nothing is cached between runs.
    - Mutable setting defaults are rejected; use None and build a fresh value
      inside the node.
    - Fan-out creates visible independent branches. They still execute one at a
      time in declaration order: there is no hidden thread/process scheduler.
    - Treat wired arrays and dictionaries as read-only so sibling branches stay
      deterministic without expensive deep copies.
"""

from __future__ import annotations

import base64
import hashlib
import html
import inspect
import io
import pprint
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


__all__ = ["Graph", "GraphError", "Node", "NodeError", "Run", "node"]


def _safe_error_text(error: BaseException) -> str:
    """Return an exception message even when a custom ``__str__`` is broken."""

    try:
        return str(error)
    except Exception:
        return f"{type(error).__name__} (message unavailable)"


_RESULT_TEXT_LIMIT = 12_000
_RESULT_MARKUP_LIMIT = 1_000_000
_RESULT_PNG_LIMIT = 8 * 1024 * 1024


def _oversized_result_html(kind: str, size: int, limit: int) -> str:
    return (
        "<pre style='margin:0;white-space:pre-wrap;overflow-wrap:anywhere'>"
        f"{html.escape(kind)} result omitted: {size:,} bytes exceeds the "
        f"{limit:,}-byte notebook display limit.</pre>"
    )


def _bounded_markup(payload: str, kind: str) -> str:
    size = len(payload.encode("utf-8"))
    if size > _RESULT_MARKUP_LIMIT:
        return _oversized_result_html(kind, size, _RESULT_MARKUP_LIMIT)
    return payload


def _rich_result_payload(value: Any, method_name: str) -> Any:
    """Call one explicitly declared IPython rich-repr method, if present."""

    try:
        inspect.getattr_static(value, method_name)
    except AttributeError:
        return None
    method = getattr(value, method_name)
    if not callable(method):
        return None
    payload = method()
    if isinstance(payload, tuple) and len(payload) == 2:
        payload = payload[0]
    return payload


def _is_matplotlib_figure(value: Any) -> bool:
    """Recognize a Matplotlib Figure without importing the scientific runtime."""

    return any(
        current.__name__ == "Figure"
        and current.__module__.startswith("matplotlib.")
        for current in type(value).__mro__
    ) and callable(getattr(value, "savefig", None))


def _png_result_html(payload: bytes | bytearray | memoryview | str) -> str:
    if isinstance(payload, str):
        decoded = base64.b64decode(payload, validate=True)
    else:
        decoded = bytes(payload)
    if len(decoded) > _RESULT_PNG_LIMIT:
        return _oversized_result_html("PNG", len(decoded), _RESULT_PNG_LIMIT)
    encoded = base64.b64encode(decoded).decode("ascii")
    return (
        "<img alt='Rendered graph result' "
        "style='display:block;max-width:100%;height:auto' "
        f"src='data:image/png;base64,{encoded}'>"
    )


def _result_html(value: Any) -> str:
    """Render one result into HTML that can travel through a widget trait.

    ``widgets.Output`` relies on callback output capture, which is not reliable
    in every Colab frontend. This renderer instead serializes the visible
    result into the ``value`` trait of a plain HTML widget. Rich IPython
    representations are retained, while ordinary values are escaped and
    bounded before they enter the page.
    """

    body: str | None = None
    if _is_matplotlib_figure(value):
        image = io.BytesIO()
        dpi = min(float(getattr(value, "dpi", 100.0)), 150.0)
        value.savefig(image, format="png", bbox_inches="tight", dpi=dpi)
        body = _png_result_html(image.getvalue())
    else:
        rich_html = _rich_result_payload(value, "_repr_html_")
        if isinstance(rich_html, str):
            body = _bounded_markup(rich_html, "HTML")

        if body is None:
            svg = _rich_result_payload(value, "_repr_svg_")
            if isinstance(svg, bytes):
                svg = svg.decode("utf-8", errors="replace")
            if isinstance(svg, str):
                body = _bounded_markup(svg, "SVG")

        if body is None:
            png = _rich_result_payload(value, "_repr_png_")
            if isinstance(png, (bytes, bytearray, memoryview, str)):
                body = _png_result_html(png)

    if body is None:
        try:
            text = repr(value)
            if not isinstance(text, str):
                raise TypeError("repr returned a non-string value")
        except Exception:
            text = f"<{type(value).__name__} (representation unavailable)>"
        if len(text) > _RESULT_TEXT_LIMIT:
            text = text[: _RESULT_TEXT_LIMIT - 1] + "…"
        body = (
            "<pre style='margin:0;white-space:pre-wrap;overflow-wrap:anywhere'>"
            f"{html.escape(text)}</pre>"
        )

    return (
        "<div class='graph-result-view' role='region' aria-label='Graph result' "
        "style='box-sizing:border-box;max-width:100%;max-height:32rem;"
        "overflow:auto;padding:.65rem .75rem;border:1px solid rgba(127,127,127,.28);"
        "border-radius:6px'>"
        f"{body}</div>"
    )


def _dict_html(value: Mapping[str, Any]) -> str:
    """Render an escaped, copyable dictionary in notebook output."""

    rendered = pprint.pformat(dict(value), width=100, sort_dicts=False)
    return (
        "<pre style='margin:.3rem 0;padding:.7rem .8rem;max-width:1100px;"
        "max-height:34rem;overflow:auto;border:1px solid #7775;border-radius:7px;"
        "white-space:pre-wrap;overflow-wrap:anywhere'>"
        f"{html.escape(rendered)}</pre>"
    )


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

    def to_dict(self) -> dict[str, Any]:
        """Return bounded failure context without the traceback machinery."""

        return {
            "type": type(self).__name__,
            "message": _safe_error_text(self),
            "node_name": self.node_name,
            "completed": self.completed,
            "timings_seconds": dict(self.timings),
            "inputs": dict(self.inputs),
        }

    def __repr__(self) -> str:
        return f"NodeError({pprint.pformat(self.to_dict(), width=88, sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


def _public_value(value: Any) -> Any:
    """Keep simple values exact and summarize anything potentially large."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str) and len(value) <= 120:
        return value
    return Graph._value_summary(value)


@dataclass(frozen=True)
class Node:
    """An ordinary function with named output ports."""

    name: str
    operation: Callable[..., Any]
    outputs: tuple[str, ...]
    signature: inspect.Signature
    cache: bool = field(default=True, compare=False)
    knobs: Mapping[str, Any] = field(default=MappingProxyType({}), compare=False)

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").strip().capitalize()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.operation(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return the callable contract without exposing the function object."""

        parameters = []
        for parameter in self.signature.parameters.values():
            parameters.append(
                {
                    "name": parameter.name,
                    "required": parameter.default is inspect.Parameter.empty,
                    "default": (
                        None
                        if parameter.default is inspect.Parameter.empty
                        else _public_value(parameter.default)
                    ),
                    "kind": parameter.kind.name.lower(),
                }
            )
        return {
            "name": self.name,
            "label": self.label,
            "operation": (
                f"{self.operation.__module__}.{self.operation.__qualname__}"
            ),
            "parameters": parameters,
            "outputs": self.outputs,
            "cache": self.cache,
            "knobs": {name: dict(spec) for name, spec in self.knobs.items()},
        }

    def __repr__(self) -> str:
        return f"Node({pprint.pformat(self.to_dict(), width=88, sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


@dataclass(frozen=True)
class Run(Mapping[str, Any]):
    """One read-only top-level record; contained scientific values are not copied."""

    graph_name: str
    settings: Mapping[str, Any]
    outputs: Mapping[str, Any]
    order: tuple[str, ...]
    timings: Mapping[str, float]
    final_ports: tuple[str, ...]
    terminal_ports: tuple[str, ...]

    def __getitem__(self, port: str) -> Any:
        return self.outputs[port]

    def __iter__(self):
        return iter(self.outputs)

    def __len__(self) -> int:
        return len(self.outputs)

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

    def to_dict(self) -> dict[str, Any]:
        """Return bounded run metadata and shape-only value summaries.

        ``dict(run)`` remains the way to obtain the actual output values.
        This method is intentionally safe to print when outputs contain large
        scientific arrays.
        """

        return {
            "graph_name": self.graph_name,
            "settings": {
                name: _public_value(value)
                for name, value in self.settings.items()
            },
            "outputs": {
                name: Graph._value_summary(value)
                for name, value in self.outputs.items()
            },
            "order": self.order,
            "timings_seconds": dict(self.timings),
            "total_seconds": self.seconds,
            "final_ports": self.final_ports,
            "terminal_ports": self.terminal_ports,
            "actual_values": "dict(run)",
        }

    def __repr__(self) -> str:
        return f"Run({pprint.pformat(self.to_dict(), width=88, sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


def _output_names(outputs: str | Sequence[str]) -> tuple[str, ...]:
    names = (outputs,) if isinstance(outputs, str) else tuple(outputs)
    if not names or any(not isinstance(name, str) or not name.isidentifier() for name in names):
        raise GraphError("node outputs must be one or more valid Python names")
    if len(names) != len(set(names)):
        raise GraphError("a node cannot declare the same output twice")
    return names


class _Knob:
    """Sentinel default carrying widget metadata; unwrapped to its real default at node build."""

    __slots__ = ("default", "spec")

    def __init__(self, default: Any, *, min: Any = None, max: Any = None, step: Any = None,
                 tier: str = "beginner", label: str | None = None, help: str | None = None,
                 choices: Sequence[Any] | None = None):
        self.default = default
        spec: dict[str, Any] = {"tier": tier}
        for key, value in (("min", min), ("max", max), ("step", step),
                           ("label", label), ("help", help)):
            if value is not None:
                spec[key] = value
        if choices is not None:
            spec["choices"] = list(choices)
        self.spec = spec


def knob(default: Any, *, min: Any = None, max: Any = None, step: Any = None,
         tier: str = "beginner", label: str | None = None, help: str | None = None,
         choices: Sequence[Any] | None = None) -> Any:
    """Annotate a node setting with widget metadata for ``graph.play`` and ``describe``.

    Use as the default of a setting parameter; ``run`` still sees the real ``default``.
    ``tier`` is ``"beginner"`` (shown by default) or ``"advanced"`` (revealed by a toggle).

        @graph.node(outputs="stats")
        def summarize(alpha=graph.knob(0.05, min=1e-3, max=0.2, tier="advanced",
                                       help="p-value threshold")):
            ...
    """
    return _Knob(default, min=min, max=max, step=step, tier=tier, label=label,
                 help=help, choices=choices)


def node(
    function: Callable[..., Any] | None = None,
    *,
    outputs: str | Sequence[str] | None = None,
    name: str | None = None,
    cache: bool = True,
) -> Node | Callable[[Callable[..., Any]], Node]:
    """Turn a function into a node.

    Parameters in the function signature become named input ports or settings.
    ``outputs`` must name the value or values returned by the function. Set
    ``cache=False`` for nodes whose output should never be reused (e.g. plots).
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
        # Unwrap knob() sentinels: real default stays in the signature, metadata is stashed.
        knob_specs: dict[str, Any] = {}
        rebuilt = []
        for parameter in signature.parameters.values():
            if isinstance(parameter.default, _Knob):
                knob_specs[parameter.name] = dict(parameter.default.spec)
                parameter = parameter.replace(default=parameter.default.default)
            rebuilt.append(parameter)
        signature = signature.replace(parameters=rebuilt)
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
            if isinstance(parameter.default, (dict, list, set, bytearray)):
                raise GraphError(
                    f"node {node_name!r} setting {parameter.name!r} has a mutable "
                    "default; use None and create a fresh value inside the node"
                )
        return Node(
            name=node_name,
            operation=operation,
            outputs=_output_names(declared_outputs),
            signature=signature,
            cache=cache,
            knobs=MappingProxyType(knob_specs),
        )

    return decorate(function) if function is not None else decorate


_DRIVE_STORE_FACTORY: Callable[[str], Any] | None = None


def use_drive_cache(factory: Callable[[str], Any]) -> None:
    """Register a ``factory(graph_name) -> store`` so ``cache="drive"`` persists results.

    Keeps graph.py itself free of any data/framework dependency: the caller (e.g.
    the caller supplies the per-teammate Drive-backed store. Until this is set,
    ``cache="drive"`` degrades gracefully to a per-session in-memory cache.
    """
    global _DRIVE_STORE_FACTORY
    _DRIVE_STORE_FACTORY = factory


class _MemoryStore:
    """A per-session, in-process store used by cache="memory"."""

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}

    def get(self, spec: Any) -> Any:
        return self._entries.get(_spec_key(spec))

    def put(self, spec: Any, value: Any) -> None:
        self._entries[_spec_key(spec)] = value


def _spec_key(spec: Any) -> str:
    return repr(tuple(sorted(spec.items())))


def _make_store(cache: Any, graph_name: str) -> Any:
    """Resolve the cache= option to a store (get/put) or None. Cache is per-teammate."""
    if not cache:
        return None
    if cache == "memory":
        return _MemoryStore()
    if cache is True or cache == "drive":
        if _DRIVE_STORE_FACTORY is not None:
            return _DRIVE_STORE_FACTORY(graph_name)
        return _MemoryStore()
    if hasattr(cache, "get") and hasattr(cache, "put"):
        return cache
    raise GraphError(f"unknown cache option: {cache!r}; use 'drive', 'memory', a store, or None")


def _fingerprint_value(value: Any) -> str | None:
    """A stable token for a JSON-like setting value, or None if it is opaque."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        tokens = [_fingerprint_value(item) for item in value]
        if any(token is None for token in tokens):
            return None
        return "[" + ",".join(token for token in tokens if token is not None) + "]"
    if isinstance(value, Mapping):
        pairs = []
        for key in sorted(value, key=repr):
            token_key = _fingerprint_value(key)
            token_value = _fingerprint_value(value[key])
            if token_key is None or token_value is None:
                return None
            pairs.append(token_key + ":" + token_value)
        return "{" + ",".join(pairs) + "}"
    return None


def _cacheable_result(produced: Mapping[str, Any]) -> bool:
    """Skip caching outputs that are unsafe or cheap to persist (matplotlib figures)."""
    return not any(_is_matplotlib_figure(value) for value in produced.values())


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
                if (
                    type(current).__name__ == "Node"
                    and type(current).__module__ == __name__
                ):
                    raise GraphError(
                        "this node was created by an earlier graph module instance; "
                        "rerun its @graph.node definition cell after the setup cell"
                    )
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
                producer = self._producer.get(output)
                if producer is not None:
                    raise GraphError(
                        f"output port {output!r} is produced by both "
                        f"{producer!r} and {current.name!r}"
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

    def to_dict(self) -> dict[str, Any]:
        """Return the graph structure as an ordinary JSON-safe dictionary."""

        return self.describe()

    def __repr__(self) -> str:
        return f"Graph({pprint.pformat(self.to_dict(), width=100, sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())

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

    def run(self, *, until: str | None = None, cache: Any = None, **settings: Any) -> Run:
        """Run nodes once in declaration order.

        ``until`` can name a node or output port and is useful while developing
        a flow. ``cache`` opts into per-node result reuse: ``"drive"`` persists
        each node's output to your OWN Google Drive (per-teammate), ``"memory"``
        keeps it for this session, or pass a store with get/put. Keys are content
        fingerprints of a node's source + settings + upstream fingerprints, so a
        slider tweak only recomputes what actually changed. Off (None) by default.
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
        store = _make_store(cache, self.name)
        node_fp: dict[str, str | None] = {}
        for current in executed_nodes:
            arguments: dict[str, Any] = {}
            for parameter in current.signature.parameters.values():
                connection = self._connections.get((current.name, parameter.name))
                if connection is None:
                    arguments[parameter.name] = effective[parameter.name]
                else:
                    arguments[parameter.name] = values[connection[1]]
            fingerprint = None
            if store is not None and current.cache:
                fingerprint = self._node_fingerprint(current, arguments, node_fp)
            node_fp[current.name] = fingerprint
            if fingerprint is not None:
                hit = store.get({"node": current.name, "fp": fingerprint})
                if hit is not None:
                    values.update(hit)
                    timings[current.name] = 0.0
                    order.append(current.name)
                    continue
            started = time.perf_counter()
            try:
                returned = current.operation(**arguments)
                if inspect.isawaitable(returned):
                    close = getattr(returned, "close", None)
                    if callable(close):
                        close()
                    raise GraphError(
                        f"node {current.name!r} returned an awaitable; "
                        "async work is not supported in notebook flows"
                    )
                produced = self._normalize_outputs(current, returned)
            except Exception as error:
                input_summary = {
                    name: self._value_summary(value)
                    for name, value in arguments.items()
                }
                raise NodeError(
                    f"node {current.name!r} failed with inputs "
                    f"{input_summary!r}: {_safe_error_text(error)}",
                    node_name=current.name,
                    completed=order,
                    timings=timings,
                    inputs=input_summary,
                ) from error
            timings[current.name] = time.perf_counter() - started
            order.append(current.name)
            values.update(produced)
            if store is not None and fingerprint is not None and _cacheable_result(produced):
                try:
                    store.put({"node": current.name, "fp": fingerprint}, produced)
                except Exception:
                    pass

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

    def _node_fingerprint(self, current: Node, arguments: Mapping[str, Any],
                          node_fp: Mapping[str, str | None]) -> str | None:
        """A content fingerprint for one node, or None if it cannot be cached safely.

        Wired inputs contribute their PRODUCER's fingerprint (never the value), so
        multi-GB arrays are never hashed. Settings contribute a fingerprint of the
        value; an opaque setting makes the node (and everything downstream) uncacheable.
        """
        try:
            source = inspect.getsource(current.operation)
        except (OSError, TypeError):
            return None
        parts = [source, repr(current.outputs)]
        for parameter in current.signature.parameters.values():
            connection = self._connections.get((current.name, parameter.name))
            if connection is None:
                token = _fingerprint_value(arguments[parameter.name])
                if token is None:
                    return None
                parts.append(f"{parameter.name}={token}")
            else:
                upstream = node_fp.get(connection[0])
                if upstream is None:
                    return None
                parts.append(f"{parameter.name}<-{upstream}")
        return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()[:32]

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
                raise GraphError(
                    f"variation {index} failed: {_safe_error_text(error)}"
                ) from error
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

        kind = type(value).__name__
        try:
            if value is None:
                return "—"
            if isinstance(value, bool):
                return "on" if value else "off"
            if isinstance(value, (int, float, str)):
                text = str(value)
                return text if len(text) <= 32 else text[:31] + "…"
            shape = getattr(value, "shape", None)
            if shape is not None:
                try:
                    dimensions = "×".join(
                        str(int(size)) for size in tuple(shape)
                    )
                except Exception:
                    dimensions = "shape unavailable"
                dtype = getattr(value, "dtype", None)
                suffix = f" · {dtype}" if dtype is not None else ""
                return f"{kind} · {dimensions or 'scalar'}{suffix}"
            if isinstance(value, Mapping):
                return f"{kind} · {len(value)} keys"
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return f"{kind} · {len(value)} items"
        except Exception:
            pass
        return kind

    @staticmethod
    def _format_setting_value(value: Any) -> str:
        text = Graph._value_summary(value)
        return text if len(text) <= 32 else text[:31] + "…"

    @staticmethod
    def _fit_port_label(label: str, *, shared_row: bool) -> str:
        """Keep opposing input/output labels inside their side of a node."""

        limit = 22 if shared_row else 38
        return label if len(label) <= limit else label[: limit - 1] + "…"

    def _diagram_geometry(self) -> dict[str, Any]:
        """Return the one compact layout shared by SVG wires and UI controls."""

        node_width = 256.0
        header_height = 42.0
        port_height = 30.0
        column_gap = 88.0
        row_gap = 28.0

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
        bypass_candidates = sum(
            depth[target_node] - depth[source_node] > 1
            for (target_node, _), (source_node, _) in self._connections.items()
        )
        margin = max(18.0, 12.0 + 6.0 * bypass_candidates)

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
            "bypass_candidates": bypass_candidates,
            "width": width,
            "height": height,
        }

    @staticmethod
    def _wire_path(
        source_node: str,
        target_node: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        geometry: Mapping[str, Any],
        lane_index: int = 0,
    ) -> tuple[str, str, float | None]:
        """Route a wire around any node boxes between its endpoints."""

        node_width = float(geometry["node_width"])
        positions = geometry["position"]
        intermediates = [
            (x, y, height)
            for name, (x, y, height) in positions.items()
            if name not in {source_node, target_node}
            and x1 < x + node_width
            and x < x2
        ]
        clearance = 4.0
        low, high = sorted((y1, y2))
        blocked = any(
            low <= y + height + clearance and high >= y - clearance
            for _, y, height in intermediates
        )
        if not blocked:
            bend = max(40.0, (x2 - x1) * 0.5)
            return (
                f"M{x1:.1f},{y1:.1f} C{x1 + bend:.1f},{y1:.1f} "
                f"{x2 - bend:.1f},{y2:.1f} {x2:.1f},{y2:.1f}",
                "direct",
                None,
            )

        lane_y = max(
            4.0,
            min(y for _, y, _ in intermediates) - 12.0 - lane_index * 6.0,
        )
        before = min(x for x, _, _ in intermediates) - 12.0
        after = max(x + node_width for x, _, _ in intermediates) + 12.0
        first_turn = (x1 + before) / 2
        second_turn = (after + x2) / 2
        return (
            f"M{x1:.1f},{y1:.1f} C{first_turn:.1f},{y1:.1f} "
            f"{first_turn:.1f},{lane_y:.1f} {before:.1f},{lane_y:.1f} "
            f"H{after:.1f} C{second_turn:.1f},{lane_y:.1f} "
            f"{second_turn:.1f},{y2:.1f} {x2:.1f},{y2:.1f}",
            "bypass",
            lane_y,
        )

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
        bypass_lane_index = 0
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
            wire_path, route, lane_y = self._wire_path(
                source_node,
                target_node,
                x1,
                y1,
                x2,
                y2,
                geometry,
                bypass_lane_index,
            )
            if route == "bypass":
                bypass_lane_index += 1
            source_type = next(
                item["type"] for item in source_rows if item["name"] == source_port
            )
            colour = self._type_colour(source_type)
            wire_label = (
                f"{source_node}.{source_port} to {target_node}.{target_port}"
            )
            lane_attribute = (
                f"data-lane-y='{lane_y:.1f}' " if lane_y is not None else ""
            )
            wires.append(
                f"<path d='{wire_path}' fill='none' "
                f"stroke='{colour}' stroke-width='2' stroke-opacity='0.9' "
                "class='graph-wire' "
                f"aria-label='{html.escape(wire_label, quote=True)}' "
                f"data-route='{route}' "
                f"{lane_attribute}"
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
                    visible_label = self._fit_port_label(
                        label,
                        shared_row=index < len(row["output_ports"]),
                    )
                    parts.append(
                        f"<text x='{px + 9:.1f}' y='{py + 3.5:.1f}' font-size='11' "
                        "fill='currentColor' fill-opacity='0.88'>"
                        f"<title>{html.escape(label)}</title>"
                        f"{html.escape(visible_label)}</text>"
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
                visible_label = self._fit_port_label(
                    label,
                    shared_row=index < len(row["input_ports"]),
                )
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
                    f"<title>{html.escape(label)}</title>"
                    f"{html.escape(visible_label)}</text>"
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
            "<style>.graph-wire:hover{stroke-width:4}</style>"
            + "".join(wires)
            + "".join(parts)
            + "</svg>"
        )
        if canvas_only:
            return (
                "<style>"
                ".graph-diagram-layer{position:relative;z-index:0}"
                ".graph-control-layer{position:relative;z-index:2}"
                "</style>"
                + svg
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
    def _option_matches(left: Any, right: Any) -> bool:
        if left is right:
            return True
        try:
            return bool(left == right)
        except Exception:
            return False

    @staticmethod
    def _control_widget(widgets: Any, name: str, specification: Any, default: Any):
        label = name.replace("_", " ").capitalize()
        if isinstance(specification, widgets.Widget):
            if not hasattr(specification, "value"):
                raise GraphError(
                    f"widget control {name!r} must be a value control such as "
                    "Dropdown, Checkbox, Text, IntText, or FloatText"
                )
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
            value = next(
                (
                    candidate
                    for candidate in values
                    if Graph._option_matches(candidate, default)
                ),
                values[0],
            )
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
        auto_run: bool = False,
    ):
        """Return a node-and-port editor with targeted runs and port inspection.

        Filled input sockets receive an earlier node's output. Hollow sockets
        are editable values. Controls are placed inside the node that owns the
        port so the visual map and the executable settings stay together. The
        default runs the complete flow; ``Run to`` can stop at one node, and a
        completed run's produced ports can be inspected without rerunning.
        Set ``auto_run=True`` to execute once with the visible default values
        as the widget is built, which is useful for deterministic notebook
        cells that should show a result immediately.
        """

        widgets = self._widgets()
        if show is not None and show not in self._producer:
            raise GraphError(f"unknown output for show={show!r}")
        control_specs = dict(controls or {})
        unknown = sorted(set(control_specs) - set(self._settings))
        if unknown:
            raise GraphError(f"unknown widget controls: {', '.join(unknown)}")
        missing_controls = [
            f"{node_name}.{name}"
            for name, (node_name, parameter) in self._settings.items()
            if parameter.default is inspect.Parameter.empty
            and name not in control_specs
        ]
        if missing_controls:
            raise GraphError(
                "widget needs an explicit value control for required input ports: "
                + ", ".join(missing_controls)
            )
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
        initial_settings = {
            name: control.value for name, control in control_widgets.items()
        }
        geometry = self._diagram_geometry()
        width = geometry["width"]
        height = geometry["height"]
        diagram_view = widgets.HTML(
            value=self._diagram_html(
                preview_settings=initial_settings,
                interactive_settings=interactive_names,
                canvas_only=True,
            ),
            layout=widgets.Layout(
                grid_area="canvas",
                order="0",
                width=f"{width:.0f}px",
                min_width=f"{width:.0f}px",
                height=f"{height:.0f}px",
                overflow="hidden",
            ),
        )
        diagram_view.add_class("graph-diagram-layer")
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
            control.layout.order = "1"
            control.layout.align_self = "flex-start"
            control.layout.flex = "0 0 auto"
            control.layout.width = f"{control_width:.0f}px"
            control.layout.min_width = "0"
            control.layout.max_width = f"{control_width:.0f}px"
            control.layout.height = f"{geometry['port_height'] - 4:.0f}px"
            control.layout.margin = f"{top:.0f}px 0 0 {left:.0f}px"
            control.add_class("graph-control-layer")
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
            tooltip="Run to the selected step with the visible input values",
            layout=widgets.Layout(width="132px", min_width="132px"),
        )
        run_to = widgets.Dropdown(
            options=[
                ("All", None),
                *[
                    (f"{index} · {current.label}", current.name)
                    for index, current in enumerate(self.nodes, start=1)
                ],
            ],
            value=None,
            description="Run to",
            tooltip="Run all nodes, or stop after the selected node",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="190px", min_width="160px"),
        )
        inspect_port = widgets.Dropdown(
            options=[("Run first", None)],
            value=None,
            description="Inspect port",
            disabled=True,
            tooltip="Display a produced output port without rerunning any node",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="230px", min_width="190px"),
        )
        status = widgets.HTML(layout=widgets.Layout(flex="1 1 auto", min_width="0"))
        output = widgets.HTML(
            value="",
            layout=widgets.Layout(width="100%", min_width="0"),
        )

        def _set_status(message: str, *, alert: bool = False) -> None:
            status.value = (
                f"<div role='{'alert' if alert else 'status'}' "
                f"aria-live='{'assertive' if alert else 'polite'}'>{message}</div>"
            )

        _set_status("Choose input values in their nodes, then run the flow.")

        toolbar = widgets.HBox(
            [run_button, run_to, inspect_port, status],
            layout=widgets.Layout(
                width="100%",
                min_width="0",
                align_items="center",
                flex_flow="row wrap",
                grid_gap="8px",
            ),
        )
        panel = widgets.VBox(
            [toolbar, surface, output],
            layout=widgets.Layout(width="100%", min_width="0", grid_gap="6px"),
        )
        panel.last_run = None
        panel.last_error = None
        panel.run_count = 0
        panel.result_view = output
        running = False
        preview_pending = False
        inspector_syncing = False

        def _selected_settings() -> dict[str, Any]:
            return {
                name: control.value for name, control in control_widgets.items()
            }

        def _settings_for_target(
            selected: Mapping[str, Any], target: str | None
        ) -> dict[str, Any]:
            active_nodes = {current.name for current in self._execution_nodes(target)}
            return {
                name: value
                for name, value in selected.items()
                if self._settings[name][0] in active_nodes
            }

        def _reset_inspector() -> None:
            nonlocal inspector_syncing
            inspector_syncing = True
            try:
                inspect_port.options = [("Run first", None)]
                inspect_port.value = None
                inspect_port.disabled = True
            finally:
                inspector_syncing = False

        def _enable_inspector(result: Run) -> None:
            nonlocal inspector_syncing
            labels = {
                port: f"{current.label} · {port}"
                for current in self.nodes
                if current.name in result.order
                for port in current.outputs
                if port in result.outputs
            }
            inspector_syncing = True
            try:
                inspect_port.options = [
                    ("Default result", None),
                    *[(labels[port], port) for port in result.outputs],
                ]
                inspect_port.value = None
                inspect_port.disabled = False
            finally:
                inspector_syncing = False

        def _default_result(result: Run) -> Any:
            if show is not None and show in result.outputs:
                return result[show]
            return result.final

        def _display_result(result: Run, port: str | None = None) -> None:
            value = _default_result(result) if port is None else result[port]
            output.value = _result_html(value)

        def _clear_result() -> None:
            output.value = ""

        def _figure_numbers() -> set[int]:
            pyplot = sys.modules.get("matplotlib.pyplot")
            if pyplot is None:
                return set()
            try:
                return set(pyplot.get_fignums())
            except Exception:
                return set()

        def _close_new_figures(previous: set[int]) -> None:
            pyplot = sys.modules.get("matplotlib.pyplot")
            if pyplot is None:
                return
            try:
                for number in set(pyplot.get_fignums()) - previous:
                    pyplot.close(number)
            except Exception:
                pass

        def _show_failure(error: Exception, selected: Mapping[str, Any]) -> None:
            try:
                diagram_view.value = self._diagram_html(
                    preview_settings=selected,
                    interactive_settings=interactive_names,
                    error=error if isinstance(error, NodeError) else None,
                    canvas_only=True,
                )
            except Exception:
                pass

        def _preview(_change: Any = None) -> None:
            nonlocal preview_pending
            if running:
                preview_pending = True
                return
            panel.last_run = None
            panel.last_error = None
            _reset_inspector()
            _clear_result()
            try:
                selected = _selected_settings()
                diagram_view.value = self._diagram_html(
                    preview_settings=selected,
                    interactive_settings=interactive_names,
                    canvas_only=True,
                )
                _set_status("Input values changed. Run the flow to update the result.")
            except Exception as error:
                panel.last_error = error
                _set_status(html.escape(_safe_error_text(error)), alert=True)

        for control in control_widgets.values():
            control.observe(_preview, names="value")

        def inspect_changed(change: Mapping[str, Any]) -> None:
            if inspector_syncing or running or panel.last_run is None:
                return
            port = change.get("new")
            _clear_result()
            try:
                _display_result(panel.last_run, port)
            except Exception as error:
                panel.last_error = error
                _set_status(
                    "<b>Could not display the selected port:</b> "
                    f"{html.escape(_safe_error_text(error))}",
                    alert=True,
                )
                return
            panel.last_error = None
            label = "the default result" if port is None else f"port {port}"
            _set_status(
                f"Showing {html.escape(label)} from run {panel.run_count}; "
                "no nodes reran."
            )

        inspect_port.observe(inspect_changed, names="value")

        def run_clicked(_: Any) -> None:
            nonlocal running, preview_pending
            if running:
                return
            running = True
            preview_pending = False
            selected: dict[str, Any] = {}
            result: Run | None = None
            figures_before = _figure_numbers()
            disabled_states: dict[str, bool] = {}
            target = run_to.value
            run_button.disabled = True
            run_to.disabled = True
            _reset_inspector()
            for name, control in control_widgets.items():
                if hasattr(control, "disabled"):
                    disabled_states[name] = bool(control.disabled)
                    control.disabled = True
            panel.last_run = None
            panel.last_error = None
            _clear_result()
            try:
                selected = _selected_settings()
                active_selected = _settings_for_target(selected, target)
                map_error: Exception | None = None
                try:
                    diagram_view.value = self._diagram_html(
                        preview_settings=selected,
                        interactive_settings=interactive_names,
                        canvas_only=True,
                    )
                except Exception as error:
                    map_error = error
                _set_status("Running…")
                run_error: Exception | None = None
                try:
                    result = self.run(until=target, **active_selected)
                except Exception as error:
                    run_error = error
                if run_error is not None:
                    panel.last_error = run_error
                    _show_failure(run_error, selected)
                    _set_status(
                        f"<b>Could not run:</b> "
                        f"{html.escape(_safe_error_text(run_error))}",
                        alert=True,
                    )
                    return

                assert result is not None
                panel.last_run = result
                _enable_inspector(result)
                panel.run_count += 1
                try:
                    diagram_view.value = self._diagram_html(
                        result,
                        interactive_settings=interactive_names,
                        canvas_only=True,
                    )
                    map_error = None
                except Exception as error:
                    map_error = error

                result_error: Exception | None = None
                try:
                    _display_result(result)
                except Exception as error:
                    result_error = error
                if result_error is not None:
                    panel.last_error = result_error
                    map_note = (
                        " The node map was also unavailable."
                        if map_error is not None
                        else ""
                    )
                    _set_status(
                        "<b>Run completed, but the result view failed:</b> "
                        f"{html.escape(_safe_error_text(result_error))}"
                        f"{map_note}",
                        alert=True,
                    )
                    return

                if map_error is not None:
                    panel.last_error = map_error
                    _set_status(
                        f"Run {panel.run_count} completed and the result is shown, "
                        "but the node map is unavailable: "
                        f"{html.escape(_safe_error_text(map_error))}",
                        alert=True,
                    )
                    return

                panel.last_error = None
                chosen = ", ".join(
                    f"{name.replace('_', ' ')}={self._format_setting_value(value)}"
                    for name, value in active_selected.items()
                )
                detail = f" Inputs: {html.escape(chosen)}." if chosen else ""
                target_detail = (
                    ""
                    if target is None
                    else f" Stopped after {html.escape(self._nodes_by_name[target].label)}."
                )
                _set_status(
                    f"Run {panel.run_count} complete. Ran {len(result.order)} nodes in "
                    f"{result.seconds:.3f} seconds."
                    f"{target_detail}"
                    f"{detail}"
                )
            except Exception as error:
                panel.last_run = None
                panel.last_error = error
                _show_failure(error, selected)
                _set_status(
                    f"<b>Could not run:</b> "
                    f"{html.escape(_safe_error_text(error))}",
                    alert=True,
                )
            finally:
                _close_new_figures(figures_before)
                for name, disabled in disabled_states.items():
                    control_widgets[name].disabled = disabled
                run_button.disabled = False
                run_to.disabled = False
                running = False
                if preview_pending:
                    preview_pending = False
                    _preview()

        run_button.on_click(run_clicked)
        if auto_run:
            run_clicked(None)
        return panel
