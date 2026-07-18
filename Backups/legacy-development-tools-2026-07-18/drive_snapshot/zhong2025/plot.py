"""Clear, deterministic plotting recipes for the Zhong et al. workspace.

The public functions in this module accept ordinary NumPy arrays, mappings,
and lists.  They deliberately keep analysis separate from presentation: a
plot may summarize already-computed values, but it does not silently fit a
scientific model, choose a statistical test, or reinterpret an experimental
unit.

Every recipe returns a fresh :class:`matplotlib.figure.Figure`, never calls
``show`` or ``close``, and never mutates global Matplotlib ``rcParams``.  This
makes the same call safe in a notebook, a ``graph`` node, a test, or a script.

The shortest useful introduction is::

    from zhong2025 import plot

    plot.guide()                         # one-page recipe index
    fig = plot.curve({"mice": values})  # values: mice x time
    plot.save(fig, "results/curve.png")

The semantic palettes and domain wrappers follow Zhong et al. (2025), while
the distribution, validation, and agreement views follow the safeguards in
Stringer & Pachitariu (2024) and the project's analysis specification.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from importlib import resources
from pathlib import Path
from typing import Any
import html
import math
import pprint

import numpy as np
from numpy.typing import ArrayLike


# Okabe-Ito-derived fallbacks.  Stable semantic palettes are kept separate so
# cohort identity is never accidentally inferred from a stimulus colour.
PALETTE = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky
    "#6B7280",  # grey
    "#111827",  # near black
)

STIMULUS_COLORS = OrderedDict(
    [
        ("circle1", "#D55E00"),
        ("circle2", "#CC79A7"),
        ("leaf2", "#56B4E9"),
        ("leaf1", "#0072B2"),
        ("leaf3", "#E69F00"),
        ("leaf1 swap", "#009E73"),
    ]
)

COHORT_COLORS = OrderedDict(
    [
        ("task", "#009E73"),
        ("sup", "#009E73"),
        ("supervised", "#009E73"),
        ("unsup", "#8E3B68"),
        ("unsupervised", "#8E3B68"),
        ("unrewarded", "#8E3B68"),
        ("naive", "#111827"),
        ("grating", "#6B7280"),
        ("no pretraining", "#111827"),
    ]
)

AREA_COLORS = OrderedDict(
    [
        ("v1", "#0072B2"),
        ("medial", "#E69F00"),
        ("mhv", "#E69F00"),
        ("lateral", "#009E73"),
        ("lhv", "#009E73"),
        ("anterior", "#CC79A7"),
        ("ahv", "#CC79A7"),
        ("excluded", "#9CA3AF"),
    ]
)

ROLE_LABELS = {0: "circle1", 1: "circle2", 2: "leaf1", 3: "leaf2", 4: "leaf3"}


def _dict_repr(name: str, value: dict[str, Any]) -> str:
    return f"{name}({pprint.pformat(value, width=100, sort_dicts=False)})"


def _dict_html(value: dict[str, Any]) -> str:
    return f"<pre>{html.escape(pprint.pformat(value, width=100, sort_dicts=False))}</pre>"


@dataclass(frozen=True)
class PlotInfo:
    """Small provenance record attached to every returned figure."""

    recipe: str
    caption: str = ""
    provenance: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return figure provenance as an ordinary dictionary."""

        return {
            "recipe": self.recipe,
            "caption": self.caption,
            "provenance": dict(self.provenance),
            "warnings": self.warnings,
        }

    def __repr__(self) -> str:
        return _dict_repr(type(self).__name__, self.to_dict())

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


@dataclass(frozen=True)
class Recipe:
    """One row in :func:`guide`."""

    name: str
    use: str
    input: str

    def to_dict(self) -> dict[str, str]:
        """Return one plotting recipe as an ordinary dictionary."""

        return {"name": self.name, "use": self.use, "input": self.input}

    def __repr__(self) -> str:
        return _dict_repr(type(self).__name__, self.to_dict())

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


_RECIPE_ROWS = (
    Recipe("curve", "Time, position, or learning trajectories", "series; optional mice x time"),
    Recipe("small_multiples", "Several trajectory metrics on aligned axes", "metric -> series"),
    Recipe("comparison", "Raw categorical or paired experimental units", "group -> values"),
    Recipe("distribution", "ECDF, KDE, histogram, violin, or ridgeline", "group -> values"),
    Recipe("signed_tails", "Positive and negative d-prime tails over thresholds", "group -> signed values"),
    Recipe("relationship", "Scatter, density, fit, or identity comparison", "x and y"),
    Recipe("pairwise", "Property scatter/density matrix", "column -> values"),
    Recipe("agreement", "Identity scatter plus Bland-Altman", "reference and approximation"),
    Recipe("matrix", "Cross-temporal, similarity, covariance, or support matrix", "rows x columns"),
    Recipe("activity", "Trial/neuron activity heatmap with safe row sorting", "rows x samples"),
    Recipe("train_test", "Train-derived ordering shown separately on held-out data", "train and test matrices"),
    Recipe("event_raster", "Lick, cue, reward, or other events by trial", "event -> trial event lists"),
    Recipe("rastermap", "Large supplied activity ordering plus aligned tracks", "neurons x time"),
    Recipe("cortical_map", "Prepared ROI coordinates, categories, or continuous values", "x and y coordinates"),
    Recipe("cortical_density", "Smoothed selected-neuron density normalized by all neurons", "ROI coordinates + mask"),
    Recipe("density_difference", "Two normalized 2D densities and their difference", "two n x 2 point sets"),
    Recipe("trajectory", "Two- or three-dimensional component trajectories", "label -> samples x dimensions"),
    Recipe("spectrum", "Eigenspectrum and optional power-law reference", "positive eigenvalues"),
    Recipe("prediction", "Prediction versus truth and residual diagnostics", "true and predicted values"),
    Recipe("timeline", "Protocol, acquisition, or mouse-journey stages", "ordered labels"),
    Recipe("corridor", "Position profiles with texture/grey/event regions", "label -> position values"),
    Recipe("dprime", "Paper-specific two-distribution d-prime explainer", "two response samples"),
    Recipe("forest", "Leave-one-out or interval estimates", "labels, estimates, intervals"),
    Recipe("permutation", "Supplied null distribution with observed estimate", "null samples + observed"),
    Recipe("bars", "Counts, bytes, and other non-inferential totals", "label -> total"),
    Recipe("stacked_bars", "Composition or storage totals by layer", "segment -> category -> total"),
    Recipe("image_grid", "Stimuli, filters, FOVs, or packaged references", "sequence of images"),
    Recipe("recording", "Compact released recording dashboard", "load_atlas_demo() mapping"),
    Recipe("released_example", "Compact released SVD-feature orientation summary", "load_atlas_demo() mapping"),
    Recipe("blockwise", "Held-out d-prime and fold-support dashboard", "blockwise_dprime() result"),
    Recipe("qc", "Speed, occupancy, missingness, and role support", "prepared-session mapping"),
    Recipe("mouse_journey", "One mouse's ordered acquisitions and numeric summaries", "list of acquisition rows"),
    Recipe("all_mouse_journeys", "All-mouse acquisition lanes and stage coverage", "mouse -> acquisition rows"),
    Recipe("cohort_preflight", "Mouse counts, acquisition dates, and layer storage", "list of manifest rows"),
    Recipe("reference_figure", "Packaged Nature or Science reference figure", "paper and figure number"),
    Recipe("reference_gallery", "All packaged figures from one paper section", "paper name"),
)


_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "text.color": "#111827",
    "axes.titlesize": 11.0,
    "axes.titleweight": "semibold",
    "axes.titlecolor": "#111827",
    "axes.labelsize": 10.0,
    "axes.labelcolor": "#111827",
    "axes.edgecolor": "#374151",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 9.0,
    "xtick.labelsize": 9.0,
    "xtick.color": "#374151",
    "ytick.labelsize": 9.0,
    "ytick.color": "#374151",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
}


def recipes() -> tuple[str, ...]:
    """Return every stable plotting recipe name in gallery order."""

    return tuple(row.name for row in _RECIPE_ROWS)


class _Guide:
    def to_dict(self) -> dict[str, Any]:
        """Return the complete recipe index in a reusable form."""

        return {
            "recipes": [row.to_dict() for row in _RECIPE_ROWS],
            "save": "plot.save(fig, path)",
        }

    def __repr__(self) -> str:
        return _dict_repr(type(self).__name__, self.to_dict())

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


def guide() -> _Guide:
    """Return the recipe index with matching display and ``to_dict`` views."""

    return _Guide()


def colors(domain: str = "all") -> dict[str, str]:
    """Return a copy of one semantic colour map.

    ``domain`` is ``"stimulus"``, ``"cohort"``, ``"area"``, or ``"all"``.
    """

    tables = {
        "stimulus": STIMULUS_COLORS,
        "cohort": COHORT_COLORS,
        "area": AREA_COLORS,
    }
    if domain == "all":
        merged: dict[str, str] = {}
        for table in tables.values():
            merged.update(table)
        return merged
    if domain not in tables:
        raise ValueError("domain must be 'stimulus', 'cohort', 'area', or 'all'")
    return dict(tables[domain])


def info(figure: Any) -> PlotInfo:
    """Return the :class:`PlotInfo` attached to a figure."""

    value = getattr(figure, "zhong2025_info", None)
    if not isinstance(value, PlotInfo):
        raise TypeError("figure was not created by zhong2025.plot")
    return value


def save(
    figure: Any,
    path: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
) -> Path:
    """Save one returned figure and return its resolved output path."""

    output = Path(path).expanduser()
    if not output.suffix:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), transparent=bool(transparent), bbox_inches="tight")
    return output.resolve()


def close(figure: Any) -> None:
    """Explicitly release one figure after displaying or saving it."""

    _, plt = _mpl()
    plt.close(figure)


def _mpl():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    return mpl, plt


def _tag(
    figure: Any,
    recipe: str,
    *,
    caption: str = "",
    provenance: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
) -> Any:
    items = tuple((str(key), str(value)) for key, value in (provenance or {}).items())
    figure.zhong2025_info = PlotInfo(
        recipe=recipe,
        caption=str(caption),
        provenance=items,
        warnings=tuple(map(str, warnings)),
    )
    return figure


def _subplots(
    recipe: str,
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] | None = None,
    squeeze: bool = True,
    sharex: bool = False,
    sharey: bool = False,
    **kwargs: Any,
):
    mpl, plt = _mpl()
    if figsize is None:
        figsize = (max(6.4, 4.4 * ncols), max(3.8, 3.25 * nrows))
    with mpl.rc_context(_RC):
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            constrained_layout=True,
            squeeze=squeeze,
            sharex=sharex,
            sharey=sharey,
            **kwargs,
        )
    _tag(figure, recipe)
    return figure, axes


def _empty(recipe: str, message: str, *, title: str = ""):
    figure, axis = _subplots(recipe, figsize=(7.0, 3.2))
    axis.set_axis_off()
    if title:
        axis.set_title(title, loc="left")
    axis.text(
        0.5,
        0.5,
        str(message),
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="#6B7280",
        wrap=True,
    )
    return figure


def _clean_label(value: Any) -> str:
    return str(value).replace("_", " ").strip()


def _normal_label(value: Any) -> str:
    return " ".join(_clean_label(value).lower().split())


def _color(label: Any, index: int, palette: Mapping[Any, str] | Sequence[str] | None = None) -> str:
    if isinstance(palette, Mapping):
        if label in palette:
            return str(palette[label])
        normalized = {_normal_label(key): value for key, value in palette.items()}
        if _normal_label(label) in normalized:
            return str(normalized[_normal_label(label)])
    elif palette is not None:
        supplied = tuple(palette)
        if not supplied:
            raise ValueError("palette cannot be empty")
        return str(supplied[index % len(supplied)])

    key = _normal_label(label)
    for table in (STIMULUS_COLORS, COHORT_COLORS, AREA_COLORS):
        if key in table:
            return table[key]
    for semantic, value in {**COHORT_COLORS, **AREA_COLORS, **STIMULUS_COLORS}.items():
        if semantic in key:
            return value
    return PALETTE[index % len(PALETTE)]


def _one_dimensional(values: ArrayLike, name: str, *, allow_empty: bool = True) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if not allow_empty and not len(array):
        raise ValueError(f"{name} cannot be empty")
    return array


def _numeric_1d(values: ArrayLike, name: str, *, allow_empty: bool = True) -> np.ndarray:
    array = _one_dimensional(values, name, allow_empty=allow_empty)
    try:
        return np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error


def _numeric_2d(values: ArrayLike, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric matrix") from error
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}")
    return array


def _uniform_coordinate_step(values: np.ndarray, name: str) -> float:
    """Validate finite monotonic uniform centers and return their signed step."""

    if not np.isfinite(values).all():
        raise ValueError(f"{name} coordinates must be finite")
    differences = np.diff(values)
    if len(differences):
        monotonic = np.all(differences > 0) or np.all(differences < 0)
        if not monotonic or not np.allclose(differences, differences[0]):
            raise ValueError(
                f"{name} coordinates must be monotonic and evenly spaced for a heatmap"
            )
        return float(differences[0])
    return 1.0


def _groups(values: Mapping[Any, ArrayLike] | ArrayLike, *, default: str = "value") -> OrderedDict:
    if isinstance(values, Mapping):
        return OrderedDict((str(label), np.asarray(data)) for label, data in values.items())
    return OrderedDict([(default, np.asarray(values))])


def _finite(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _x_values(x: Any, label: str, length: int) -> np.ndarray:
    if isinstance(x, Mapping):
        if label in x:
            selected = x[label]
        else:
            matches = [value for key, value in x.items() if str(key) == str(label)]
            if len(matches) > 1:
                raise ValueError(f"x contains ambiguous keys for series {label!r}")
            selected = matches[0] if matches else None
    else:
        selected = x
    if selected is None:
        return np.arange(length, dtype=np.float64)
    array = _numeric_1d(selected, f"x for {label}")
    if len(array) != length:
        raise ValueError(f"x for {label!r} has {len(array)} values; expected {length}")
    return array


def _summary(values: np.ndarray, band: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    if values.ndim == 1:
        return values.astype(np.float64, copy=False), None
    if values.ndim != 2:
        raise ValueError("each curve series must be one-dimensional or units x samples")
    finite_count = np.sum(np.isfinite(values), axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(values, axis=0)
        if band in (None, "none"):
            error = None
        elif band == "sd":
            error = np.nanstd(values, axis=0, ddof=1)
        elif band in ("sem", "ci95"):
            error = np.nanstd(values, axis=0, ddof=1)
            np.divide(error, np.sqrt(finite_count), out=error, where=finite_count > 0)
            if band == "ci95":
                from scipy.stats import t as student_t

                error *= student_t.ppf(0.975, finite_count - 1)
        else:
            raise ValueError("band must be 'sem', 'ci95', 'sd', or None")
    if error is not None:
        error = np.asarray(error)
        error[finite_count < 2] = np.nan
    return mean, error


def _column_mean_sem(values: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-value column means and SEMs without a fixed-n bias."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("column summaries require a two-dimensional array")
    finite = np.isfinite(array)
    count = np.sum(finite, axis=0)
    total = np.sum(np.where(finite, array, 0.0), axis=0)
    mean = np.full(array.shape[1], np.nan, dtype=float)
    np.divide(total, count, out=mean, where=count > 0)
    residual = np.where(finite, array - mean, 0.0)
    sum_squares = np.sum(residual**2, axis=0)
    sem = np.full(array.shape[1], np.nan, dtype=float)
    variance = np.full(array.shape[1], np.nan, dtype=float)
    np.divide(sum_squares, count - 1, out=variance, where=count > 1)
    np.divide(np.sqrt(variance), np.sqrt(count), out=sem, where=count > 1)
    return mean, sem


def _draw_curve_series(
    axis: Any,
    series: Mapping[Any, ArrayLike] | ArrayLike,
    *,
    x: Any = None,
    band: str | None = "sem",
    individuals: bool = True,
    max_individuals: int = 60,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    linewidth: float = 2.25,
    marker: str | None = None,
) -> int:
    drawn = 0
    for index, (label, raw) in enumerate(_groups(series).items()):
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim not in (1, 2):
            raise ValueError(f"series {label!r} must be 1D or 2D; got {values.shape}")
        length = values.shape[-1]
        horizontal = _x_values(x, label, length)
        colour = _color(label, index, palette)
        if values.ndim == 2 and individuals:
            if values.shape[0] <= max_individuals:
                chosen = np.arange(values.shape[0])
            else:
                chosen = np.linspace(0, values.shape[0] - 1, max_individuals, dtype=int)
            for row in chosen:
                axis.plot(horizontal, values[row], color=colour, linewidth=0.65, alpha=0.16)
        centre, uncertainty = _summary(values, band)
        if not np.isfinite(centre).any():
            continue
        if uncertainty is not None:
            axis.fill_between(
                horizontal,
                centre - uncertainty,
                centre + uncertainty,
                color=colour,
                alpha=0.18,
                linewidth=0,
            )
        axis.plot(
            horizontal,
            centre,
            color=colour,
            linewidth=linewidth,
            marker=marker,
            markersize=4,
            label=_clean_label(label),
        )
        drawn += 1
    return drawn


def _finish_axis(
    axis: Any,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid: str | None = None,
    legend: bool = False,
) -> None:
    if title:
        axis.set_title(str(title), loc="left")
    if xlabel:
        axis.set_xlabel(str(xlabel))
    if ylabel:
        axis.set_ylabel(str(ylabel))
    if grid:
        axis.grid(axis=grid, color="#D1D5DB", linewidth=0.7, alpha=0.55)
    if legend:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()


def _add_regions(axis: Any, regions: Mapping[str, Any] | Sequence[Any] | None) -> None:
    if regions is None:
        return
    items = regions.items() if isinstance(regions, Mapping) else enumerate(regions)
    for index, (label, value) in enumerate(items):
        if np.isscalar(value):
            axis.axvline(float(value), color=PALETTE[(index + 6) % len(PALETTE)], linestyle="--", linewidth=1.0, label=_clean_label(label))
            continue
        bounds = tuple(value)
        if len(bounds) != 2:
            raise ValueError("each region must be a scalar position or (start, stop)")
        axis.axvspan(float(bounds[0]), float(bounds[1]), color=PALETTE[(index + 5) % len(PALETTE)], alpha=0.10, label=_clean_label(label))


def curve(
    series: Mapping[Any, ArrayLike] | ArrayLike,
    *,
    x: ArrayLike | Mapping[Any, ArrayLike] | None = None,
    band: str | None = "sem",
    individuals: bool = True,
    max_individuals: int = 60,
    reference: float | None = None,
    regions: Mapping[str, Any] | Sequence[Any] | None = None,
    marker: str | None = None,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[float, float] = (8.2, 4.6),
):
    """Plot one or more trajectories, preserving raw units when supplied.

    A one-dimensional series is drawn directly.  A two-dimensional series is
    interpreted as ``independent units x samples``; thin unit paths and a bold
    mean are shown, with ``band`` set to SEM by default.
    """

    figure, axis = _subplots("curve", figsize=figsize)
    drawn = _draw_curve_series(
        axis,
        series,
        x=x,
        band=band,
        individuals=individuals,
        max_individuals=max_individuals,
        palette=palette,
        marker=marker,
    )
    if reference is not None:
        axis.axhline(float(reference), color="#6B7280", linewidth=1.0, linestyle="--")
    _add_regions(axis, regions)
    if not drawn:
        axis.text(0.5, 0.5, "No finite curve values", transform=axis.transAxes, ha="center", va="center", color="#6B7280")
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel=ylabel, grid="y", legend=drawn > 1)
    return figure


def small_multiples(
    metrics: Mapping[Any, Mapping[Any, ArrayLike] | ArrayLike],
    *,
    x: ArrayLike | Mapping[Any, ArrayLike] | None = None,
    band: str | None = "sem",
    individuals: bool = True,
    columns: int = 3,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    reference: float | Mapping[Any, float] | None = None,
    title: str = "",
    xlabel: str = "Progress",
):
    """Plot aligned metric trajectories as visible small multiples."""

    if not isinstance(metrics, Mapping) or not metrics:
        return _empty("small_multiples", "No metrics supplied", title=title)
    columns = max(1, min(int(columns), len(metrics)))
    rows = int(math.ceil(len(metrics) / columns))
    figure, axes = _subplots(
        "small_multiples",
        rows,
        columns,
        figsize=(4.5 * columns, 3.25 * rows),
        squeeze=False,
        sharex=True,
    )
    for axis, (metric, series) in zip(axes.flat, metrics.items()):
        drawn = _draw_curve_series(
            axis,
            series,
            x=x,
            band=band,
            individuals=individuals,
            palette=palette,
        )
        value = reference.get(metric) if isinstance(reference, Mapping) else reference
        if value is not None:
            axis.axhline(float(value), color="#6B7280", linewidth=0.9, linestyle="--")
        _finish_axis(
            axis,
            title=_clean_label(metric),
            xlabel=xlabel if axis in axes[-1, :] else "",
            ylabel=_clean_label(metric),
            grid="y",
            legend=drawn > 1,
        )
    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)
    if title:
        figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    return figure


def _jitter(count: int, width: float = 0.22) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    # Irrational-step sequence: deterministic, seed-free, and less striped than
    # index-linear jitter while remaining exactly reproducible.
    return ((((np.arange(count) * 0.6180339887498949) + 0.5) % 1.0) - 0.5) * width


def _interval(
    values: np.ndarray,
    summary: str,
    error: str | None,
) -> tuple[float, float | tuple[float, float] | None]:
    finite = _finite(values)
    if not len(finite):
        return np.nan, None
    if summary == "mean":
        centre = float(np.mean(finite))
    elif summary == "median":
        centre = float(np.median(finite))
    else:
        raise ValueError("summary must be 'mean' or 'median'")
    if error in (None, "none") or len(finite) < 2:
        return centre, None
    if error == "sd":
        spread = float(np.std(finite, ddof=1))
    elif error in ("sem", "ci95"):
        spread = float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
        if error == "ci95":
            from scipy.stats import t as student_t

            spread *= float(student_t.ppf(0.975, len(finite) - 1))
    elif error == "iqr":
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        spread = (float(q25), float(q75))
    else:
        raise ValueError("error must be 'sem', 'ci95', 'sd', 'iqr', or None")
    return centre, spread


def _adjust_pvalues(values: Sequence[float], correction: str | None) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("comparison p-values must be finite and between 0 and 1")
    if correction is None:
        raise ValueError(
            "comparisons require an explicit correction policy: use correction='none', "
            "'bonferroni', 'holm', or 'fdr_bh'"
        )
    key = str(correction).lower()
    count = len(p)
    if key == "none":
        return p.copy()
    if key == "bonferroni":
        return np.minimum(1.0, p * count)
    order = np.argsort(p)
    if key == "holm":
        adjusted = np.empty(count)
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (count - rank) * p[index])
            adjusted[index] = min(1.0, running)
        return adjusted
    if key == "fdr_bh":
        ranked = p[order] * count / np.arange(1, count + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted = np.empty(count)
        adjusted[order] = np.minimum(1.0, ranked)
        return adjusted
    raise ValueError("correction must be 'none', 'bonferroni', 'holm', or 'fdr_bh'")


def _p_label(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "NS"


def comparison(
    groups: Mapping[Any, ArrayLike],
    *,
    paired: bool = False,
    pair_ids: Mapping[Any, Sequence[Any]] | None = None,
    summary: str = "mean",
    error: str | None = "sem",
    reference: float | None = None,
    comparisons: Sequence[tuple[Any, Any, float]] | None = None,
    correction: str | None = None,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "",
    ylabel: str = "",
    unit: str = "unit",
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Show raw categorical values, optional pairing, and a summary interval.

    Statistical tests are never chosen or run here.  Supply precomputed
    ``comparisons=[(group_a, group_b, pvalue), ...]`` and explicitly state a
    multiplicity ``correction`` policy when annotations are required.
    """

    if not isinstance(groups, Mapping) or not groups:
        return _empty("comparison", "No groups supplied", title=title)
    prepared_groups = OrderedDict((str(label), value) for label, value in groups.items())
    labels = list(prepared_groups)
    values = [_numeric_1d(value, f"values for {label}") for label, value in prepared_groups.items()]
    matrix_values = None
    paired_count = 0
    dropped_incomplete_pairs = 0
    if paired:
        if pair_ids is None:
            raise ValueError(
                "paired=True requires pair_ids for every group; positional pairing "
                "is intentionally not inferred"
            )
        lookups = []
        normalized_ids = {str(label): value for label, value in pair_ids.items()}
        for label, value in zip(labels, values):
            ids = list(normalized_ids.get(label, ()))
            if len(ids) != len(value):
                raise ValueError(f"pair_ids for {label!r} must align with its values")
            if len(ids) != len(set(ids)):
                raise ValueError(f"pair_ids for {label!r} contain duplicates")
            lookups.append(dict(zip(ids, value)))
        expected_ids = set(lookups[0])
        if any(set(lookup) != expected_ids for lookup in lookups[1:]):
            raise ValueError("paired groups must contain identical pair_ids")
        ordered_ids = list(normalized_ids[labels[0]])
        matrix_values = np.asarray([[lookup[item] for lookup in lookups] for item in ordered_ids], dtype=float)
        complete = np.all(np.isfinite(matrix_values), axis=1)
        dropped_incomplete_pairs = int(np.count_nonzero(~complete))
        matrix_values = matrix_values[complete]
        paired_count = int(len(matrix_values))
        values = [matrix_values[:, index] for index in range(len(labels))]

    adjusted = None
    positions = {label: index for index, label in enumerate(labels)}
    if comparisons:
        adjusted = _adjust_pvalues([item[2] for item in comparisons], correction)
        for left, right, _ in comparisons:
            if str(left) not in positions or str(right) not in positions:
                raise ValueError(f"comparison refers to unknown groups {left!r}, {right!r}")

    if figsize is None:
        figsize = (max(6.2, 1.15 * len(labels) + 3.4), 4.8)
    figure, axis = _subplots("comparison", figsize=figsize)

    if matrix_values is not None:
        for row in matrix_values:
            finite = np.isfinite(row)
            if np.count_nonzero(finite) >= 2:
                axis.plot(np.arange(len(labels))[finite], row[finite], color="#9CA3AF", alpha=0.36, linewidth=0.8, zorder=1)

    all_finite: list[float] = []
    for index, (label, value) in enumerate(zip(labels, values)):
        finite = _finite(value)
        colour = _color(label, index, palette)
        axis.scatter(
            index + (_jitter(len(finite)) if not paired else np.zeros(len(finite))),
            finite,
            s=30,
            color=colour,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
            label=f"{_clean_label(label)} (n={len(finite)} {unit}{'' if len(finite) == 1 else 's'})",
        )
        centre, spread = _interval(finite, summary, error)
        if np.isfinite(centre):
            if isinstance(spread, tuple):
                lower, upper = spread
                axis.vlines(index, lower, upper, color="#111827", linewidth=1.7, zorder=3)
                axis.hlines((lower, upper), index - 0.07, index + 0.07, color="#111827", linewidth=1.7, zorder=3)
                axis.plot(index, centre, "D", markersize=6, markerfacecolor=colour, markeredgecolor="#111827", markeredgewidth=0.75, zorder=4)
            else:
                axis.errorbar(
                    index,
                    centre,
                    yerr=spread,
                    fmt="D",
                    markersize=6,
                    markerfacecolor=colour,
                    markeredgecolor="#111827",
                    markeredgewidth=0.75,
                    ecolor="#111827",
                    elinewidth=1.7,
                    capsize=4,
                    zorder=3,
                )
        all_finite.extend(finite.tolist())

    if reference is not None:
        axis.axhline(float(reference), color="#6B7280", linewidth=1.0, linestyle="--")
    axis.set_xticks(np.arange(len(labels)), [_clean_label(label) for label in labels])
    if len(labels) > 5:
        axis.tick_params(axis="x", rotation=30)
    if ylim is not None:
        axis.set_ylim(*ylim)

    if comparisons:
        assert adjusted is not None
        span = np.ptp(all_finite) if len(all_finite) > 1 else 1.0
        base = max(all_finite) if all_finite else 0.0
        step = max(span * 0.09, 0.08)
        for level, ((left, right, _), pvalue) in enumerate(zip(comparisons, adjusted)):
            left_key, right_key = str(left), str(right)
            a, b = sorted((positions[left_key], positions[right_key]))
            y = base + step * (level + 1)
            axis.plot([a, a, b, b], [y - step * 0.2, y, y, y - step * 0.2], color="#374151", linewidth=0.9)
            axis.text((a + b) / 2, y + step * 0.05, _p_label(float(pvalue)), ha="center", va="bottom", fontsize=9)
        axis.margins(y=0.18)

    _finish_axis(axis, title=title, ylabel=ylabel, grid="y", legend=False)
    _tag(
        figure,
        "comparison",
        caption=f"Raw {unit} values with {summary} and {error or 'no'} interval.",
        provenance={
            "paired": paired,
            "complete_pairs": paired_count if paired else "not applicable",
            "dropped_incomplete_pairs": dropped_incomplete_pairs if paired else "not applicable",
            "correction": correction or "not applicable",
        },
        warnings=(
            f"{dropped_incomplete_pairs} incomplete pairs were omitted from points and summaries.",
        ) if dropped_incomplete_pairs else (),
    )
    return figure


def _shared_range(groups: Mapping[str, np.ndarray]) -> tuple[float, float] | None:
    finite = [_finite(value) for value in groups.values()]
    finite = [value for value in finite if len(value)]
    if not finite:
        return None
    joined = np.concatenate(finite)
    low, high = float(np.min(joined)), float(np.max(joined))
    if low == high:
        pad = max(abs(low) * 0.05, 0.5)
        low -= pad
        high += pad
    return low, high


def _density(values: np.ndarray, grid: np.ndarray, bandwidth: Any = None) -> np.ndarray:
    finite = _finite(values)
    if len(finite) < 2 or np.std(finite) <= np.finfo(float).eps:
        return np.zeros_like(grid)
    from scipy.stats import gaussian_kde

    return np.asarray(gaussian_kde(finite, bw_method=bandwidth)(grid))


def distribution(
    groups: Mapping[Any, ArrayLike] | ArrayLike,
    *,
    kind: str = "ecdf",
    bins: int | str = "auto",
    bandwidth: Any = None,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    show_quantiles: bool = False,
    references: Sequence[float] = (),
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "",
    xlabel: str = "Value",
    ylabel: str = "",
    figsize: tuple[float, float] = (8.0, 4.6),
):
    """Plot distributions on a shared scale.

    ``kind`` is ``"ecdf"`` (stable default), ``"density"``, ``"histogram"``,
    ``"violin"``, or ``"ridge"``.  Density bandwidth is presentation only;
    it never substitutes for an estimator of spread or tail shape.
    """

    prepared = OrderedDict((label, _numeric_1d(value, f"values for {label}")) for label, value in _groups(groups).items())
    limits = _shared_range(prepared)
    if limits is None:
        return _empty("distribution", "No finite distribution values", title=title)
    kind = str(kind).lower()
    allowed = {"ecdf", "density", "histogram", "violin", "ridge"}
    if kind not in allowed:
        raise ValueError(f"kind must be one of {sorted(allowed)}")
    figure, axis = _subplots("distribution", figsize=figsize)
    low, high = limits
    grid = np.linspace(low, high, 400)
    degenerate_density_labels = []

    if kind == "violin":
        data = [_finite(value) for value in prepared.values()]
        keep = [(label, value) for label, value in zip(prepared, data) if len(value)]
        if not keep:
            return _empty("distribution", "No finite distribution values", title=title)
        labels, data = zip(*keep)
        parts = axis.violinplot(data, positions=np.arange(len(data)), showmeans=False, showmedians=False, showextrema=False)
        for index, body in enumerate(parts["bodies"]):
            body.set_facecolor(_color(labels[index], index, palette))
            body.set_edgecolor("none")
            body.set_alpha(0.62)
        for index, value in enumerate(data):
            axis.scatter(index + _jitter(len(value), 0.16), value, s=8, alpha=0.16, color=_color(labels[index], index, palette), rasterized=True)
            axis.scatter(index, np.median(value), marker="D", s=28, color="#111827", zorder=3)
        axis.set_xticks(np.arange(len(labels)), [_clean_label(label) for label in labels])
        if not ylabel:
            ylabel = xlabel
        xlabel = ""
    elif kind == "ridge":
        labels = list(prepared)
        for index, (label, value) in enumerate(prepared.items()):
            finite = _finite(value)
            if not len(finite):
                continue
            colour = _color(label, index, palette)
            if len(finite) < 2 or np.std(finite) <= np.finfo(float).eps:
                axis.vlines(finite[0], index, index + 0.55, color=colour, linewidth=2.0)
                degenerate_density_labels.append(_clean_label(label))
            else:
                density_values = _density(finite, grid, bandwidth)
                peak = np.max(density_values)
                if peak > 0:
                    density_values = density_values / peak * 0.82
                axis.fill_between(grid, index, index + density_values, color=colour, alpha=0.45)
                axis.plot(grid, index + density_values, color=colour, linewidth=1.5)
        axis.set_yticks(np.arange(len(labels)), [_clean_label(label) for label in labels])
        ylabel = ""
    else:
        histogram_edges = np.histogram_bin_edges(np.concatenate([_finite(value) for value in prepared.values() if len(_finite(value))]), bins=bins)
        for index, (label, value) in enumerate(prepared.items()):
            finite = _finite(value)
            if not len(finite):
                continue
            colour = _color(label, index, palette)
            if kind == "ecdf":
                ordered = np.sort(finite)
                probability = np.arange(1, len(ordered) + 1) / len(ordered)
                axis.step(ordered, probability, where="post", color=colour, linewidth=2.0, label=_clean_label(label))
                if show_quantiles:
                    q = np.asarray(quantiles, dtype=float)
                    if np.any((q <= 0) | (q >= 1)):
                        raise ValueError("quantiles must lie strictly between 0 and 1")
                    axis.scatter(np.quantile(finite, q), q, s=25, color=colour, edgecolor="white", linewidth=0.4, zorder=3)
            elif kind == "density":
                if len(finite) < 2 or np.std(finite) <= np.finfo(float).eps:
                    axis.axvline(finite[0], color=colour, linewidth=2.0, label=f"{_clean_label(label)} (point mass)")
                    degenerate_density_labels.append(_clean_label(label))
                else:
                    density_values = _density(finite, grid, bandwidth)
                    axis.plot(grid, density_values, color=colour, linewidth=2.0, label=_clean_label(label))
                    axis.fill_between(grid, 0, density_values, color=colour, alpha=0.10)
            else:
                axis.hist(finite, bins=histogram_edges, density=True, histtype="step", linewidth=1.8, color=colour, label=_clean_label(label))

    for value in references:
        axis.axvline(float(value), color="#6B7280", linestyle="--", linewidth=1.0)
    if degenerate_density_labels:
        axis.text(
            0.99,
            0.98,
            "KDE undefined for constant/singleton input; vertical marker shown",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#6B7280",
            fontsize=8,
        )
    default_y = {"ecdf": "Cumulative probability", "density": "Density", "histogram": "Density", "violin": ylabel, "ridge": ""}[kind]
    _finish_axis(
        axis,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel or default_y,
        grid="y" if kind in {"ecdf", "density", "histogram"} else None,
        legend=kind in {"ecdf", "density", "histogram"} and len(prepared) > 1,
    )
    _tag(
        figure,
        "distribution",
        provenance={"kind": kind, "bandwidth": bandwidth or "scipy default"},
        warnings=(
            "KDE is undefined for constant/singleton groups; vertical markers are shown instead: "
            + ", ".join(degenerate_density_labels),
        ) if degenerate_density_labels else (),
    )
    return figure


def signed_tails(
    groups: Mapping[Any, ArrayLike] | ArrayLike,
    *,
    thresholds: ArrayLike | None = None,
    reference: float = 0.3,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "Signed selective tails",
    xlabel: str = "Absolute d-prime threshold",
):
    """Plot positive and negative tail fractions together over thresholds."""

    prepared = OrderedDict((label, _finite(value)) for label, value in _groups(groups).items())
    maximum = max((np.max(np.abs(value)) for value in prepared.values() if len(value)), default=1.0)
    if thresholds is None:
        thresholds_array = np.linspace(0.0, max(float(maximum), float(reference)), 81)
    else:
        thresholds_array = _numeric_1d(thresholds, "thresholds", allow_empty=False)
        if np.any(thresholds_array < 0) or np.any(np.diff(thresholds_array) <= 0):
            raise ValueError("thresholds must be non-negative and strictly increasing")
    figure, axis = _subplots("signed_tails", figsize=(8.2, 4.7))
    drawn = 0
    for index, (label, values) in enumerate(prepared.items()):
        if not len(values):
            continue
        colour = _color(label, index, palette)
        positive = np.array([np.mean(values >= threshold) for threshold in thresholds_array])
        negative = np.array([np.mean(values <= -threshold) for threshold in thresholds_array])
        axis.plot(thresholds_array, positive, color=colour, linewidth=2.1, label=f"{_clean_label(label)}: d-prime >= +t")
        axis.plot(thresholds_array, negative, color=colour, linewidth=2.1, linestyle="--", label=f"{_clean_label(label)}: d-prime <= -t")
        drawn += 1
    axis.axvline(float(reference), color="#6B7280", linestyle=":", linewidth=1.2, label=f"reference {reference:g}")
    axis.set_ylim(-0.02, 1.02)
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel="Fraction of finite values", grid="y", legend=drawn > 0)
    if not drawn:
        axis.text(0.5, 0.5, "No finite signed values", transform=axis.transAxes, ha="center", va="center")
    return figure


def relationship(
    x: ArrayLike,
    y: ArrayLike,
    *,
    group: Sequence[Any] | None = None,
    density: bool = False,
    fit: bool = False,
    identity: bool = False,
    annotate: bool = True,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: tuple[float, float] = (6.2, 5.3),
):
    """Plot a relationship without silently fitting anything unless requested."""

    horizontal = _numeric_1d(x, "x")
    vertical = _numeric_1d(y, "y")
    if len(horizontal) != len(vertical):
        raise ValueError("x and y must have the same number of values")
    if group is None:
        labels = np.full(len(horizontal), "values", dtype=object)
    else:
        labels = _one_dimensional(group, "group")
        if len(labels) != len(horizontal):
            raise ValueError("group must have one label per x/y value")
    figure, axis = _subplots("relationship", figsize=figsize)
    mask = np.isfinite(horizontal) & np.isfinite(vertical)
    horizontal, vertical, labels = horizontal[mask], vertical[mask], labels[mask]
    if not len(horizontal):
        axis.text(0.5, 0.5, "No finite paired values", transform=axis.transAxes, ha="center", va="center")
        _finish_axis(axis, title=title, xlabel=xlabel, ylabel=ylabel)
        return figure
    unique = list(dict.fromkeys(labels.tolist()))
    if density:
        if len(unique) > 1:
            raise ValueError("density=True requires one group; facet grouped densities explicitly")
        artist = axis.hexbin(horizontal, vertical, gridsize=42, mincnt=1, cmap="viridis", bins="log")
        figure.colorbar(artist, ax=axis, label="log10 count")
    for index, label in enumerate(unique):
        selected = labels == label
        colour = _color(label, index, palette)
        if not density:
            axis.scatter(horizontal[selected], vertical[selected], s=22, alpha=0.52, color=colour, edgecolor="none", rasterized=len(horizontal) > 2500, label=_clean_label(label) if len(unique) > 1 else None)
        if fit and np.count_nonzero(selected) >= 2 and np.ptp(horizontal[selected]) > 0:
            slope, intercept = np.polyfit(horizontal[selected], vertical[selected], 1)
            grid = np.linspace(np.min(horizontal[selected]), np.max(horizontal[selected]), 100)
            axis.plot(grid, intercept + slope * grid, color=colour, linewidth=1.8, linestyle="--")
    if identity:
        low = min(np.min(horizontal), np.min(vertical))
        high = max(np.max(horizontal), np.max(vertical))
        axis.plot([low, high], [low, high], color="#6B7280", linestyle=":", linewidth=1.2, label="identity")
    if annotate:
        annotations = []
        for index, label in enumerate(unique):
            selected = labels == label
            selected_x, selected_y = horizontal[selected], vertical[selected]
            if len(selected_x) >= 2 and np.ptp(selected_x) > 0 and np.ptp(selected_y) > 0:
                correlation = np.corrcoef(selected_x, selected_y)[0, 1]
                prefix = f"{_clean_label(label)}: " if group is not None else ""
                annotations.append((f"{prefix}r = {correlation:.2f}, n = {len(selected_x):,}", _color(label, index, palette)))
        for line, (text_value, colour) in enumerate(annotations):
            axis.text(0.03, 0.97 - line * 0.055, text_value, transform=axis.transAxes, va="top", ha="left", color=colour)
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel=ylabel, grid="both", legend=len(unique) > 1 or identity)
    return figure


def pairwise(
    columns: Mapping[Any, ArrayLike],
    *,
    group: Sequence[Any] | None = None,
    kind: str = "scatter",
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "Single-unit property relationships",
):
    """Create a compact scatter/density matrix with diagonal distributions."""

    if not isinstance(columns, Mapping) or not columns:
        return _empty("pairwise", "No property columns supplied", title=title)
    names = [str(name) for name in columns]
    if len(names) > 7:
        raise ValueError("pairwise supports at most seven columns; split larger surveys")
    values = [_numeric_1d(columns[name], f"column {name}") for name in columns]
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError("all pairwise columns must have the same length")
    count = lengths.pop()
    groups_array = np.full(count, "values", dtype=object) if group is None else _one_dimensional(group, "group")
    if len(groups_array) != count:
        raise ValueError("group must align with every property row")
    kind = str(kind).lower()
    if kind not in {"scatter", "density"}:
        raise ValueError("kind must be 'scatter' or 'density'")
    size = max(2.0 * len(names), 5.4)
    figure, axes = _subplots("pairwise", len(names), len(names), figsize=(size, size), squeeze=False)
    unique = list(dict.fromkeys(groups_array.tolist()))
    for row in range(len(names)):
        for column in range(len(names)):
            axis = axes[row, column]
            x_values, y_values = values[column], values[row]
            finite = np.isfinite(x_values) & np.isfinite(y_values)
            if row == column:
                for index, label in enumerate(unique):
                    selected = finite & (groups_array == label)
                    if np.any(selected):
                        axis.hist(x_values[selected], bins="auto", histtype="step", density=True, linewidth=1.2, color=_color(label, index, palette))
            elif row > column:
                if kind == "density" and len(unique) == 1:
                    axis.hexbin(x_values[finite], y_values[finite], gridsize=24, mincnt=1, cmap="viridis")
                else:
                    for index, label in enumerate(unique):
                        selected = finite & (groups_array == label)
                        axis.scatter(x_values[selected], y_values[selected], s=8, alpha=0.25, color=_color(label, index, palette), edgecolor="none", rasterized=True)
            else:
                annotations = []
                for index, label in enumerate(unique):
                    selected = finite & (groups_array == label)
                    selected_x, selected_y = x_values[selected], y_values[selected]
                    if len(selected_x) >= 2 and np.ptp(selected_x) > 0 and np.ptp(selected_y) > 0:
                        correlation = np.corrcoef(selected_x, selected_y)[0, 1]
                        prefix = f"{_clean_label(label)}: " if group is not None else ""
                        annotations.append((f"{prefix}r={correlation:.2f}, n={len(selected_x):,}", _color(label, index, palette)))
                start = 0.5 + 0.09 * (len(annotations) - 1)
                for line, (text_value, colour) in enumerate(annotations):
                    axis.text(0.5, start - line * 0.18, text_value, transform=axis.transAxes, ha="center", va="center", color=colour, fontsize=8)
                axis.set_axis_off()
            if row == len(names) - 1 and column <= row:
                axis.set_xlabel(_clean_label(names[column]))
            else:
                axis.set_xticklabels([])
            if column == 0 and row >= column:
                axis.set_ylabel(_clean_label(names[row]))
            elif row != column:
                axis.set_yticklabels([])
    if title:
        figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    return figure


def agreement(
    reference: ArrayLike,
    approximation: ArrayLike,
    *,
    labels: tuple[str, str] = ("Reference", "Approximation"),
    title: str = "Representation agreement",
):
    """Plot identity agreement and a Bland-Altman diagnostic."""

    first = _numeric_1d(reference, "reference")
    second = _numeric_1d(approximation, "approximation")
    if len(first) != len(second):
        raise ValueError("reference and approximation must align")
    finite = np.isfinite(first) & np.isfinite(second)
    first, second = first[finite], second[finite]
    figure, axes = _subplots("agreement", 1, 2, figsize=(11.5, 4.6))
    if not len(first):
        for axis in axes:
            axis.text(0.5, 0.5, "No finite paired values", transform=axis.transAxes, ha="center", va="center")
        return figure
    axes[0].scatter(first, second, s=20, alpha=0.45, color=PALETTE[0], edgecolor="none", rasterized=len(first) > 2500)
    low, high = min(np.min(first), np.min(second)), max(np.max(first), np.max(second))
    axes[0].plot([low, high], [low, high], color="#6B7280", linestyle="--", linewidth=1.1)
    _finish_axis(axes[0], title="Identity comparison", xlabel=labels[0], ylabel=labels[1], grid="both")

    average = (first + second) / 2.0
    difference = second - first
    bias = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0
    limits = (bias - 1.96 * sd, bias + 1.96 * sd)
    axes[1].scatter(average, difference, s=20, alpha=0.45, color=PALETTE[1], edgecolor="none", rasterized=len(first) > 2500)
    axes[1].axhline(bias, color="#111827", linewidth=1.4, label=f"bias {bias:.3g}")
    axes[1].axhline(limits[0], color="#6B7280", linestyle="--", linewidth=1.0, label="95% limits")
    axes[1].axhline(limits[1], color="#6B7280", linestyle="--", linewidth=1.0)
    _finish_axis(axes[1], title="Bland-Altman", xlabel=f"Mean of {labels[0]} and {labels[1]}", ylabel=f"{labels[1]} - {labels[0]}", grid="y", legend=True)
    if title:
        figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    _tag(figure, "agreement", provenance={"n": len(first), "bias": bias, "lower_limit": limits[0], "upper_limit": limits[1]})
    return figure


def _matrix_limits(values: np.ndarray, *, center: float | None, robust: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0, 1.0
    if robust and len(finite) > 4:
        low, high = np.percentile(finite, [2, 98])
    else:
        low, high = np.min(finite), np.max(finite)
    low, high = float(low), float(high)
    if center is not None:
        radius = max(abs(low - center), abs(high - center))
        if radius == 0:
            radius = 1.0
        return center - radius, center + radius
    if low == high:
        pad = max(abs(low) * 0.05, 0.5)
        return low - pad, high + pad
    return low, high


def _imshow(
    figure: Any,
    axis: Any,
    values: np.ndarray,
    *,
    center: float | None = None,
    robust: bool = True,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: str = "",
    origin: str = "upper",
    aspect: str = "auto",
    extent: Sequence[float] | None = None,
):
    mpl, _ = _mpl()
    automatic_low, automatic_high = _matrix_limits(values, center=center, robust=robust)
    low = automatic_low if vmin is None else float(vmin)
    high = automatic_high if vmax is None else float(vmax)
    if center is not None:
        if not low < center < high:
            raise ValueError("a centered matrix requires vmin < center < vmax")
        norm = mpl.colors.TwoSlopeNorm(vmin=low, vcenter=float(center), vmax=high)
        artist = axis.imshow(values, origin=origin, aspect=aspect, cmap=cmap or "coolwarm", norm=norm, extent=extent, interpolation="nearest", rasterized=True)
    else:
        artist = axis.imshow(values, origin=origin, aspect=aspect, cmap=cmap or "viridis", vmin=low, vmax=high, extent=extent, interpolation="nearest", rasterized=True)
    figure.colorbar(artist, ax=axis, label=colorbar or None, fraction=0.046, pad=0.04)
    return artist


def matrix(
    values: ArrayLike,
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    column_labels: Sequence[Any] | None = None,
    row_labels: Sequence[Any] | None = None,
    center: float | None = None,
    robust: bool = True,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = False,
    colorbar: str = "",
    origin: str = "upper",
    title: str = "",
    xlabel: str = "Column",
    ylabel: str = "Row",
    figsize: tuple[float, float] = (7.2, 5.6),
):
    """Plot a two-dimensional matrix with honest sequential/diverging scaling."""

    array = _numeric_2d(values, "values")
    if not array.size or not np.isfinite(array).any():
        figure, axis = _subplots("matrix", figsize=figsize)
        axis.set_axis_off()
        axis.text(0.5, 0.5, "No finite matrix values", transform=axis.transAxes, ha="center", va="center")
        if title:
            axis.set_title(title, loc="left")
        return figure
    extent = None
    x_values = None
    y_values = None
    if x is not None:
        x_values = _numeric_1d(x, "x")
        if len(x_values) != array.shape[1]:
            raise ValueError("x must have one coordinate per matrix column")
        x_step = _uniform_coordinate_step(x_values, "x")
        extent = [x_values[0] - x_step / 2, x_values[-1] + x_step / 2, -0.5, array.shape[0] - 0.5]
    if y is not None:
        y_values = _numeric_1d(y, "y")
        if len(y_values) != array.shape[0]:
            raise ValueError("y must have one coordinate per matrix row")
        y_step = _uniform_coordinate_step(y_values, "y")
        if extent is None:
            extent = [-0.5, array.shape[1] - 0.5, y_values[0] - y_step / 2, y_values[-1] + y_step / 2]
        else:
            extent[2:] = [y_values[0] - y_step / 2, y_values[-1] + y_step / 2]
        origin = "lower"
    if column_labels is not None and len(column_labels) != array.shape[1]:
        raise ValueError("column_labels must match the matrix column count")
    if row_labels is not None and len(row_labels) != array.shape[0]:
        raise ValueError("row_labels must match the matrix row count")
    if annotate and array.size > 144:
        raise ValueError("annotate=True is limited to matrices with at most 144 cells")

    figure, axis = _subplots("matrix", figsize=figsize)
    _imshow(figure, axis, array, center=center, robust=robust, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=colorbar, origin=origin, extent=extent)
    if column_labels is not None:
        positions = x_values if x_values is not None else np.arange(array.shape[1])
        axis.set_xticks(positions, [_clean_label(value) for value in column_labels], rotation=45, ha="right")
    if row_labels is not None:
        positions = y_values if y_values is not None else np.arange(array.shape[0])
        axis.set_yticks(positions, [_clean_label(value) for value in row_labels])
    if annotate:
        for row, column in np.ndindex(array.shape):
            if np.isfinite(array[row, column]):
                horizontal = x_values[column] if x_values is not None else column
                vertical = y_values[row] if y_values is not None else row
                axis.text(horizontal, vertical, f"{array[row, column]:.2g}", ha="center", va="center", fontsize=8)
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel=ylabel)
    _tag(figure, "matrix", provenance={"rows": array.shape[0], "columns": array.shape[1], "center": center})
    return figure


def activity(
    values: ArrayLike,
    *,
    samples: ArrayLike | None = None,
    sort_by: ArrayLike | str | None = None,
    order: ArrayLike | None = None,
    center: float | None = None,
    robust: bool = True,
    cmap: str | None = None,
    regions: Mapping[str, Any] | Sequence[Any] | None = None,
    colorbar: str = "Activity",
    title: str = "Activity matrix",
    xlabel: str = "Sample",
    ylabel: str = "Trial / neuron",
    invert_rows: bool = True,
    figsize: tuple[float, float] = (8.2, 5.6),
):
    """Plot a supplied trial/neuron matrix with explicit, reproducible sorting."""

    array = _numeric_2d(values, "values")
    if order is not None and sort_by is not None:
        raise ValueError("supply order or sort_by, not both")
    if not array.size or not np.isfinite(array).any():
        return _empty("activity", "No finite activity values", title=title)
    if order is not None:
        selected_order = np.asarray(order)
        if selected_order.ndim != 1 or len(selected_order) != array.shape[0]:
            raise ValueError("order must contain one row index per matrix row")
        if not np.issubdtype(selected_order.dtype, np.integer):
            raise ValueError("order must contain integer row indices")
        if set(selected_order.tolist()) != set(range(array.shape[0])):
            raise ValueError("order must be a permutation of row indices")
    elif isinstance(sort_by, str):
        key = sort_by.lower()
        if key == "peak":
            safe = np.where(np.isfinite(array), array, -np.inf)
            selected_order = np.argsort(np.argmax(safe, axis=1), kind="stable")
        elif key == "mean":
            selected_order = np.argsort(np.nanmean(array, axis=1), kind="stable")
        else:
            raise ValueError("sort_by string must be 'peak' or 'mean'")
    elif sort_by is not None:
        key_values = _numeric_1d(sort_by, "sort_by")
        if len(key_values) != array.shape[0]:
            raise ValueError("sort_by must have one value per row")
        selected_order = np.argsort(key_values, kind="stable")
    else:
        selected_order = np.arange(array.shape[0])
    shown = array[selected_order]
    extent = None
    if samples is not None:
        sample_values = _numeric_1d(samples, "samples")
        if len(sample_values) != shown.shape[1]:
            raise ValueError("samples must have one value per activity column")
        step = _uniform_coordinate_step(sample_values, "samples")
        extent = [sample_values[0] - step / 2, sample_values[-1] + step / 2, -0.5, shown.shape[0] - 0.5]
    figure, axis = _subplots("activity", figsize=figsize)
    _imshow(figure, axis, shown, center=center, robust=robust, cmap=cmap or ("gray_r" if center is None else "coolwarm"), colorbar=colorbar, origin="upper" if invert_rows else "lower", extent=extent)
    _add_regions(axis, regions)
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel=ylabel)
    _tag(figure, "activity", provenance={"rows": array.shape[0], "columns": array.shape[1], "sorting": "supplied" if order is not None else sort_by or "none"})
    return figure


def train_test(
    train: ArrayLike,
    test: ArrayLike,
    *,
    samples: ArrayLike | None = None,
    sort_by: str | ArrayLike = "peak",
    center: float | None = None,
    robust: bool = True,
    cmap: str | None = None,
    labels: tuple[str, str] = ("Train (sorting only)", "Held-out test"),
    colorbar: str = "Response",
    title: str = "Cross-validation: train-derived order applied to test",
):
    """Show train and test matrices using an order learned only on train."""

    train_array = _numeric_2d(train, "train")
    test_array = _numeric_2d(test, "test")
    if train_array.shape != test_array.shape:
        raise ValueError("train and test matrices must have the same shape")
    if not train_array.size or not (np.isfinite(train_array).any() or np.isfinite(test_array).any()):
        return _empty("train_test", "No finite train/test matrix values", title=title)
    if isinstance(sort_by, str):
        if sort_by == "peak":
            safe = np.where(np.isfinite(train_array), train_array, -np.inf)
            ordering = np.argsort(np.argmax(safe, axis=1), kind="stable")
        elif sort_by == "mean":
            ordering = np.argsort(np.nanmean(train_array, axis=1), kind="stable")
        else:
            raise ValueError("sort_by must be 'peak', 'mean', or one value per row")
    else:
        sorter = _numeric_1d(sort_by, "sort_by")
        if len(sorter) != train_array.shape[0]:
            raise ValueError("sort_by must have one value per row")
        ordering = np.argsort(sorter, kind="stable")
    combined = np.concatenate([train_array.ravel(), test_array.ravel()])
    combined = combined[np.isfinite(combined)]
    low, high = _matrix_limits(combined.reshape(1, -1) if len(combined) else np.empty((1, 0)), center=center, robust=robust)
    sample_values = None
    sample_step = None
    if samples is not None:
        sample_values = _numeric_1d(samples, "samples")
        if len(sample_values) != train_array.shape[1]:
            raise ValueError("samples must match matrix columns")
        sample_step = _uniform_coordinate_step(sample_values, "samples")
    figure, axes = _subplots("train_test", 1, 2, figsize=(11.8, 5.2), sharex=True, sharey=True)
    for axis, values, label in zip(axes, (train_array, test_array), labels):
        shown = values[ordering]
        extent = None
        if sample_values is not None and sample_step is not None:
            extent = [sample_values[0] - sample_step / 2, sample_values[-1] + sample_step / 2, -0.5, shown.shape[0] - 0.5]
        _imshow(figure, axis, shown, center=center, robust=False, cmap=cmap or ("gray_r" if center is None else "coolwarm"), vmin=low, vmax=high, colorbar=colorbar, origin="upper", extent=extent)
        _finish_axis(axis, title=label, xlabel="Sample", ylabel="Train-derived row order")
    if title:
        figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    _tag(figure, "train_test", caption="Row selection/order was computed from train and reused unchanged for test.", provenance={"sort_by": sort_by if isinstance(sort_by, str) else "supplied values"})
    return figure


def _event_points(values: Any, name: str) -> tuple[np.ndarray, np.ndarray, int]:
    # NumPy 1.24+ raises for ragged nested lists instead of silently creating
    # an object array.  Ragged lists are the most natural representation for
    # a variable number of events per trial, so keep that form supported.
    try:
        array = np.asarray(values)
    except ValueError:
        array = np.asarray(values, dtype=object)
    if array.ndim == 2 and array.shape[1] == 2 and np.issubdtype(array.dtype, np.number):
        trials = np.asarray(array[:, 0], dtype=float)
        positions = np.asarray(array[:, 1], dtype=float)
        if np.any(~np.isfinite(trials)) or np.any(trials < 0) or np.any(trials != np.floor(trials)):
            raise ValueError(f"trial IDs for {name!r} must be finite non-negative integers")
        finite_positions = np.isfinite(positions)
        count = int(np.max(trials) + 1) if len(trials) else 0
        return positions[finite_positions], trials[finite_positions], count
    if isinstance(values, np.ndarray) and values.ndim == 1 and values.dtype != object:
        # One scalar event per trial; NaN means absent.
        positions = np.asarray(values, dtype=float)
        trials = np.arange(len(positions), dtype=float)
        finite = np.isfinite(positions)
        return positions[finite], trials[finite], len(positions)
    try:
        sequences = list(values)
    except TypeError as error:
        raise ValueError(f"events for {name!r} must be n x 2 points or one sequence per trial") from error
    positions_list: list[float] = []
    trials_list: list[float] = []
    for trial, sequence in enumerate(sequences):
        if sequence is None:
            continue
        if np.isscalar(sequence):
            sequence = [sequence]
        for position in np.asarray(sequence, dtype=float).reshape(-1):
            if np.isfinite(position):
                positions_list.append(float(position))
                trials_list.append(float(trial))
    return np.asarray(positions_list), np.asarray(trials_list), len(sequences)


def event_raster(
    events: Mapping[Any, Any],
    *,
    trial_count: int | None = None,
    regions: Mapping[str, Any] | Sequence[Any] | None = None,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "Event raster",
    xlabel: str = "Position / time",
    ylabel: str = "Trial",
    invert_trials: bool = True,
    figsize: tuple[float, float] = (8.4, 5.3),
):
    """Plot event positions by trial from friendly lists or ``(trial, x)`` pairs."""

    if not isinstance(events, Mapping) or not events:
        return _empty("event_raster", "No events supplied", title=title)
    maximum_trials = 0
    prepared_events = []
    for label, values in events.items():
        positions, trials, count = _event_points(values, str(label))
        maximum_trials = max(maximum_trials, count)
        prepared_events.append((label, positions, trials))
    requested_trials = None
    if trial_count is not None:
        try:
            requested_trials = int(trial_count)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("trial_count must be a positive integer") from error
        if requested_trials <= 0 or float(trial_count) != requested_trials:
            raise ValueError("trial_count must be a positive integer")
        if requested_trials < maximum_trials:
            raise ValueError(
                f"trial_count={requested_trials} would clip events from {maximum_trials} inferred trials"
            )
        maximum_trials = requested_trials

    figure, axis = _subplots("event_raster", figsize=figsize)
    points = 0
    markers = ("o", "|", "s", "^", "D", "x")
    for index, (label, positions, trials) in enumerate(prepared_events):
        if len(positions):
            axis.scatter(positions, trials, s=14 if markers[index % len(markers)] != "|" else 55, marker=markers[index % len(markers)], color=_color(label, index, palette), alpha=0.78, linewidth=0.5, label=_clean_label(label), rasterized=len(positions) > 3000)
            points += len(positions)
    _add_regions(axis, regions)
    if maximum_trials:
        if invert_trials:
            axis.set_ylim(maximum_trials - 0.5, -0.5)
        else:
            axis.set_ylim(-0.5, maximum_trials - 0.5)
    if not points:
        axis.text(0.5, 0.5, "No finite event positions", transform=axis.transAxes, ha="center", va="center")
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel=ylabel, legend=points > 0)
    return figure


def _downsample_rows(values: np.ndarray, max_rows: int) -> tuple[np.ndarray, int]:
    if values.shape[0] <= max_rows:
        return values, 1
    factor = int(math.ceil(values.shape[0] / max_rows))
    rows = []
    with np.errstate(all="ignore"):
        for start in range(0, values.shape[0], factor):
            rows.append(np.nanmean(values[start:start + factor], axis=0))
    return np.asarray(rows), factor


def rastermap(
    values: ArrayLike,
    *,
    time: ArrayLike | None = None,
    order: ArrayLike | None = None,
    tracks: Mapping[Any, ArrayLike] | None = None,
    events: Mapping[Any, ArrayLike] | None = None,
    max_rows: int = 5000,
    robust: bool = True,
    cmap: str = "gray_r",
    title: str = "Supplied population ordering",
    colorbar: str = "Activity",
):
    """Render a supplied large-neuron ordering with aligned event/behavior tracks.

    This function does not run Rastermap.  If the input has more than
    ``max_rows`` rows, adjacent ordered rows are averaged for display and that
    downsampling factor is recorded in :func:`info`.
    """

    array = _numeric_2d(values, "values")
    try:
        max_rows_value = int(max_rows)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("max_rows must be a positive integer") from error
    if max_rows_value <= 0 or float(max_rows) != max_rows_value:
        raise ValueError("max_rows must be a positive integer")
    if not array.size or not np.isfinite(array).any():
        return _empty("rastermap", "No finite ordered activity values", title=title)
    if order is not None:
        ordering = np.asarray(order)
        if ordering.ndim != 1 or len(ordering) != array.shape[0]:
            raise ValueError("order must be a permutation of every activity row")
        if not np.issubdtype(ordering.dtype, np.integer):
            raise ValueError("order must contain integer row indices")
        if set(ordering.tolist()) != set(range(array.shape[0])):
            raise ValueError("order must be a permutation of every activity row")
        array = array[ordering]
    shown, factor = _downsample_rows(array, max_rows_value)
    x = np.arange(array.shape[1], dtype=float) if time is None else _numeric_1d(time, "time")
    if len(x) != array.shape[1]:
        raise ValueError("time must have one value per activity column")
    step = _uniform_coordinate_step(x, "time")
    tracks = OrderedDict(tracks or {})
    events = OrderedDict(events or {})
    for label, track in tracks.items():
        if len(_numeric_1d(track, f"track {label}")) != len(x):
            raise ValueError(f"track {label!r} must align with time")
    rows = 1 + len(tracks) + (1 if events else 0)
    mpl, plt = _mpl()
    with mpl.rc_context(_RC):
        figure = plt.figure(figsize=(12.5, 5.4 + 1.0 * (rows - 1)), constrained_layout=True)
        grid = figure.add_gridspec(rows, 1, height_ratios=[5.0] + [1.0] * (rows - 1))
        axes = [figure.add_subplot(grid[index, 0]) for index in range(rows)]
    _tag(figure, "rastermap")
    extent = [x[0] - step / 2, x[-1] + step / 2, shown.shape[0] - 0.5, -0.5]
    _imshow(figure, axes[0], shown, robust=robust, cmap=cmap, colorbar=colorbar, origin="upper", extent=extent)
    _finish_axis(axes[0], title=title, ylabel=f"Ordered rows (displayed {shown.shape[0]:,} / {array.shape[0]:,})")
    offset = 1
    for index, (label, track) in enumerate(tracks.items()):
        values_track = _numeric_1d(track, f"track {label}")
        axes[offset].plot(x, values_track, color=_color(label, index), linewidth=1.1)
        _finish_axis(axes[offset], ylabel=_clean_label(label), grid="y")
        offset += 1
    if events:
        event_axis = axes[-1]
        for index, (label, positions) in enumerate(events.items()):
            positions_array = _finite(positions)
            event_axis.vlines(positions_array, index - 0.35, index + 0.35, color=_color(label, index), linewidth=0.8)
        event_axis.set_yticks(np.arange(len(events)), [_clean_label(label) for label in events])
        event_axis.set_ylim(-0.7, len(events) - 0.3)
        _finish_axis(event_axis, xlabel="Time", ylabel="Events")
    else:
        axes[-1].set_xlabel("Time")
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    _tag(figure, "rastermap", caption="Ordering is supplied; no Rastermap fit occurs in the plotting layer.", provenance={"input_rows": array.shape[0], "display_rows": shown.shape[0], "row_bin_factor": factor})
    return figure


def _draw_outlines(axis: Any, outlines: Sequence[ArrayLike] | Mapping[Any, ArrayLike] | None) -> None:
    if outlines is None:
        return
    values = outlines.values() if isinstance(outlines, Mapping) else outlines
    for outline in values:
        points = np.asarray(outline, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("each outline must have shape (points, 2)")
        axis.plot(points[:, 0], points[:, 1], color="#111827", linewidth=1.0, alpha=0.9)


def cortical_map(
    x: ArrayLike,
    y: ArrayLike,
    *,
    values: ArrayLike | None = None,
    groups: Sequence[Any] | None = None,
    outlines: Sequence[ArrayLike] | Mapping[Any, ArrayLike] | None = None,
    center: float | None = None,
    robust: bool = True,
    cmap: str | None = None,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    size: float = 5.0,
    colorbar: str = "Value",
    title: str = "Cortical map",
    xlabel: str = "Cortical x",
    ylabel: str = "Cortical y",
):
    """Plot prepared cortical/retinotopy coordinates without fitting an atlas."""

    horizontal = _numeric_1d(x, "x")
    vertical = _numeric_1d(y, "y")
    if len(horizontal) != len(vertical):
        raise ValueError("x and y must align")
    if values is not None and groups is not None:
        raise ValueError("supply continuous values or categorical groups, not both")
    finite = np.isfinite(horizontal) & np.isfinite(vertical)
    coordinate_finite = finite.copy()
    continuous = None
    categorical = None
    if values is not None:
        continuous = _numeric_1d(values, "values")
        if len(continuous) != len(horizontal):
            raise ValueError("values must align with coordinates")
    elif groups is not None:
        categorical = _one_dimensional(groups, "groups")
        if len(categorical) != len(horizontal):
            raise ValueError("groups must align with coordinates")

    figure, axis = _subplots("cortical_map", figsize=(7.2, 6.2))
    if continuous is not None:
        finite_values = finite & np.isfinite(continuous)
        if np.any(finite_values):
            low, high = _matrix_limits(continuous[finite_values].reshape(1, -1), center=center, robust=robust)
            mpl, _ = _mpl()
            norm = mpl.colors.TwoSlopeNorm(vmin=low, vcenter=float(center), vmax=high) if center is not None else mpl.colors.Normalize(vmin=low, vmax=high)
            artist = axis.scatter(horizontal[finite_values], vertical[finite_values], c=continuous[finite_values], cmap=cmap or ("coolwarm" if center is not None else "viridis"), norm=norm, s=size, alpha=0.72, edgecolor="none", rasterized=True)
            figure.colorbar(artist, ax=axis, label=colorbar)
        else:
            axis.scatter(horizontal[finite], vertical[finite], s=size, alpha=0.28, color="#9CA3AF", edgecolor="none", rasterized=True)
            axis.text(0.5, 0.5, "No finite cortical values", transform=axis.transAxes, ha="center", va="center", color="#6B7280")
        finite = finite_values
    elif categorical is not None:
        unique = list(dict.fromkeys(categorical.tolist()))
        for index, label in enumerate(unique):
            selected = finite & (categorical == label)
            axis.scatter(horizontal[selected], vertical[selected], s=size, alpha=0.65, color=_color(label, index, palette), edgecolor="none", rasterized=True, label=f"{_clean_label(label)} (n={np.count_nonzero(selected):,})")
        axis.legend(markerscale=2.5)
    else:
        axis.scatter(horizontal[finite], vertical[finite], s=size, alpha=0.55, color=PALETTE[0], edgecolor="none", rasterized=True)
    _draw_outlines(axis, outlines)
    axis.set_aspect("equal", adjustable="datalim")
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel=ylabel)
    _tag(
        figure,
        "cortical_map",
        caption="Coordinates and atlas outlines are supplied inputs; the plotting layer does not infer retinotopy.",
        provenance={
            "finite_coordinates": np.count_nonzero(coordinate_finite),
            "finite_values": np.count_nonzero(finite) if continuous is not None else "not applicable",
        },
    )
    return figure


def cortical_density(
    x: ArrayLike,
    y: ArrayLike,
    *,
    selected: ArrayLike | None = None,
    bins: int | tuple[int, int] = 120,
    sigma: float = 3.0,
    extent: tuple[float, float, float, float] | None = None,
    coverage: ArrayLike | None = None,
    outlines: Sequence[ArrayLike] | Mapping[Any, ArrayLike] | None = None,
    cmap: str = "magma_r",
    title: str = "Selected-neuron spatial density",
    colorbar: str = "Smoothed selected count / all recorded neurons",
):
    """Render the paper-style normalized selected-neuron density primitive.

    The numerator is a smoothed 2D histogram of selected ROIs.  The denominator
    is the total finite ROI count, not a local pixel-wise fraction.  Cross-mouse
    averaging and atlas registration must be performed upstream.
    """

    horizontal = _numeric_1d(x, "x")
    vertical = _numeric_1d(y, "y")
    if len(horizontal) != len(vertical):
        raise ValueError("x and y must align")
    finite = np.isfinite(horizontal) & np.isfinite(vertical)
    if selected is None:
        weights = np.ones(len(horizontal), dtype=float)
    else:
        supplied = np.asarray(selected)
        if supplied.ndim != 1 or len(supplied) != len(horizontal):
            raise ValueError("selected must have one boolean/weight per coordinate")
        try:
            weights = supplied.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("selected weights must be boolean or numeric") from error
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("selected weights must be finite and non-negative")
    if not np.isfinite(float(sigma)) or float(sigma) < 0:
        raise ValueError("sigma must be finite and non-negative")
    if np.isscalar(bins):
        bin_values = (int(bins), int(bins))
        if float(bins) != int(bins):
            raise ValueError("bins must contain positive integers")
    else:
        if len(bins) != 2:
            raise ValueError("bins must be one integer or a pair of integers")
        bin_values = tuple(int(value) for value in bins)
        if any(float(value) != int(value) for value in bins):
            raise ValueError("bins must contain positive integers")
    if any(value <= 0 for value in bin_values):
        raise ValueError("bins must contain positive integers")
    total = int(np.count_nonzero(finite))
    if not total:
        return _empty("cortical_density", "No finite cortical coordinates", title=title)
    if extent is None:
        x_low, x_high = np.min(horizontal[finite]), np.max(horizontal[finite])
        y_low, y_high = np.min(vertical[finite]), np.max(vertical[finite])
        x_pad = max((x_high - x_low) * 0.03, 1e-6)
        y_pad = max((y_high - y_low) * 0.03, 1e-6)
        extent = (x_low - x_pad, x_high + x_pad, y_low - y_pad, y_high + y_pad)
    else:
        if len(extent) != 4 or not np.isfinite(extent).all() or not (extent[0] < extent[1] and extent[2] < extent[3]):
            raise ValueError("extent must be finite (x_min, x_max, y_min, y_max) bounds")
    x_bins, y_bins = bin_values
    histogram, y_edges, x_edges = np.histogram2d(
        vertical[finite],
        horizontal[finite],
        bins=(int(y_bins), int(x_bins)),
        range=((extent[2], extent[3]), (extent[0], extent[1])),
        weights=weights[finite],
    )
    from scipy.ndimage import gaussian_filter

    density_values = gaussian_filter(histogram, sigma=float(sigma), mode="constant") / total
    if coverage is not None:
        coverage_array = np.asarray(coverage, dtype=bool)
        if coverage_array.shape != density_values.shape:
            raise ValueError("coverage must match the rendered density grid")
        density_values = density_values.copy()
        density_values[~coverage_array] = np.nan
    figure, axis = _subplots("cortical_density", figsize=(7.2, 6.2))
    artist = axis.imshow(density_values, origin="lower", extent=extent, cmap=cmap, aspect="equal", interpolation="nearest", rasterized=True)
    figure.colorbar(artist, ax=axis, label=colorbar)
    _draw_outlines(axis, outlines)
    _finish_axis(axis, title=title, xlabel="Cortical x", ylabel="Cortical y")
    _tag(figure, "cortical_density", caption="Smoothed selected ROI histogram divided by the total finite ROI count.", provenance={"total_neurons": total, "selected_weight": float(np.sum(weights[finite])), "sigma_bins": sigma})
    return figure


def density_difference(
    first: ArrayLike,
    second: ArrayLike,
    *,
    bins: int = 45,
    labels: tuple[str, str] = ("Population 1", "Population 2"),
    title: str = "Property-density comparison",
    xlabel: str = "Parameter 1",
    ylabel: str = "Parameter 2",
):
    """Show two normalized 2D densities and their signed difference."""

    first_array = _numeric_2d(first, "first")
    second_array = _numeric_2d(second, "second")
    if first_array.shape[1] != 2 or second_array.shape[1] != 2:
        raise ValueError("first and second must each have shape (samples, 2)")
    combined = np.vstack([first_array, second_array])
    finite = np.isfinite(combined).all(axis=1)
    if not np.any(finite):
        return _empty("density_difference", "No finite 2D points", title=title)
    combined = combined[finite]
    x_range = (float(np.min(combined[:, 0])), float(np.max(combined[:, 0])))
    y_range = (float(np.min(combined[:, 1])), float(np.max(combined[:, 1])))
    if x_range[0] == x_range[1]:
        x_range = (x_range[0] - 0.5, x_range[1] + 0.5)
    if y_range[0] == y_range[1]:
        y_range = (y_range[0] - 0.5, y_range[1] + 0.5)

    def histogram(points: np.ndarray) -> np.ndarray:
        keep = np.isfinite(points).all(axis=1)
        result, _, _ = np.histogram2d(points[keep, 1], points[keep, 0], bins=int(bins), range=(y_range, x_range))
        total = np.sum(result)
        return result / total if total else result

    first_density = histogram(first_array)
    second_density = histogram(second_array)
    difference = second_density - first_density
    sequential_max = max(float(np.max(first_density)), float(np.max(second_density)), np.finfo(float).eps)
    difference_max = max(float(np.max(np.abs(difference))), np.finfo(float).eps)
    figure, axes = _subplots("density_difference", 1, 3, figsize=(13.8, 4.5), sharex=True, sharey=True)
    for axis, values, label in zip(axes[:2], (first_density, second_density), labels):
        artist = axis.imshow(values, origin="lower", extent=(*x_range, *y_range), aspect="auto", cmap="magma", vmin=0, vmax=sequential_max, rasterized=True)
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
        _finish_axis(axis, title=label, xlabel=xlabel, ylabel=ylabel)
    artist = axes[2].imshow(difference, origin="lower", extent=(*x_range, *y_range), aspect="auto", cmap="coolwarm", vmin=-difference_max, vmax=difference_max, rasterized=True)
    figure.colorbar(artist, ax=axes[2], label=f"{labels[1]} - {labels[0]}", fraction=0.046, pad=0.04)
    _finish_axis(axes[2], title="Difference", xlabel=xlabel, ylabel=ylabel)
    if title:
        figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    return figure


def trajectory(
    paths: Mapping[Any, ArrayLike] | ArrayLike,
    *,
    dimensions: tuple[int, ...] | None = None,
    mark_endpoints: bool = True,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "Component trajectory",
    labels: Sequence[str] | None = None,
):
    """Plot supplied 2D or 3D latent trajectories; no embedding is fitted."""

    prepared = _groups(paths, default="trajectory")
    first = next(iter(prepared.values()), np.empty((0, 2)))
    first_array = _numeric_2d(first, "trajectory")
    if dimensions is None:
        dimensions = (0, 1) if first_array.shape[1] < 3 else (0, 1, 2)
    if len(dimensions) not in (2, 3):
        raise ValueError("dimensions must select two or three columns")
    mpl, plt = _mpl()
    with mpl.rc_context(_RC):
        figure = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d" if len(dimensions) == 3 else None)
    _tag(figure, "trajectory")
    for index, (label, raw) in enumerate(prepared.items()):
        values = _numeric_2d(raw, f"trajectory {label}")
        if max(dimensions) >= values.shape[1]:
            raise ValueError(f"trajectory {label!r} does not contain dimensions {dimensions}")
        selected = values[:, dimensions]
        finite = np.isfinite(selected).all(axis=1)
        finite_index = np.flatnonzero(finite)
        if not len(finite_index):
            continue
        colour = _color(label, index, palette)
        if len(dimensions) == 3:
            axis.plot(selected[:, 0], selected[:, 1], selected[:, 2], color=colour, linewidth=2.0, label=_clean_label(label))
            if mark_endpoints:
                axis.scatter(*selected[finite_index[0]], color=colour, marker="o", s=35)
                axis.scatter(*selected[finite_index[-1]], color=colour, marker="X", s=48)
        else:
            axis.plot(selected[:, 0], selected[:, 1], color=colour, linewidth=2.0, label=_clean_label(label))
            if mark_endpoints:
                axis.scatter(*selected[finite_index[0]], color=colour, marker="o", s=35, zorder=3)
                axis.scatter(*selected[finite_index[-1]], color=colour, marker="X", s=48, zorder=3)
    labels = list(labels or [f"Component {dimension + 1}" for dimension in dimensions])
    if len(labels) != len(dimensions):
        raise ValueError("labels must match the selected dimensions")
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    if len(dimensions) == 3:
        axis.set_zlabel(labels[2])
    axis.set_title(title, loc="left")
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    _tag(figure, "trajectory", caption="Coordinates are supplied; PCA/UMAP/Rastermap fitting is outside the plotting layer.")
    return figure


def spectrum(
    eigenvalues: Mapping[Any, ArrayLike] | ArrayLike,
    *,
    reference_power: float | None = None,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "Population eigenspectrum",
    xlabel: str = "Component rank",
    ylabel: str = "Eigenvalue",
):
    """Plot positive supplied eigenvalues on logarithmic axes."""

    if reference_power is not None and not np.isfinite(float(reference_power)):
        raise ValueError("reference_power must be finite")
    prepared = _groups(eigenvalues, default="eigenvalues")
    figure, axis = _subplots("spectrum", figsize=(7.2, 5.0))
    drawn = 0
    first_values = None
    for index, (label, raw) in enumerate(prepared.items()):
        values = _numeric_1d(raw, f"eigenvalues for {label}")
        finite = np.isfinite(values) & (values > 0)
        values = np.sort(values[finite])[::-1]
        if not len(values):
            continue
        if first_values is None:
            first_values = values
        ranks = np.arange(1, len(values) + 1)
        axis.loglog(ranks, values, color=_color(label, index, palette), linewidth=1.8, label=_clean_label(label))
        drawn += 1
    if reference_power is not None and first_values is not None:
        ranks = np.arange(1, len(first_values) + 1)
        reference = first_values[0] * ranks ** (-float(reference_power))
        axis.loglog(ranks, reference, color="#6B7280", linestyle="--", linewidth=1.2, label=f"rank^-{reference_power:g}")
    if not drawn:
        axis.text(0.5, 0.5, "No finite positive eigenvalues", transform=axis.transAxes, ha="center", va="center")
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel=ylabel, grid="both", legend=drawn > 1 or reference_power is not None)
    _tag(figure, "spectrum", caption="Finite positive eigenvalues are sorted from largest to smallest before plotting by rank.")
    return figure


def prediction(
    truth: ArrayLike,
    predicted: ArrayLike,
    *,
    x: ArrayLike | None = None,
    title: str = "Held-out prediction",
    ylabel: str = "Value",
):
    """Show prediction versus truth and residuals without fitting a model."""

    actual = _numeric_1d(truth, "truth")
    estimate = _numeric_1d(predicted, "predicted")
    if len(actual) != len(estimate):
        raise ValueError("truth and predicted must align")
    horizontal = np.arange(len(actual)) if x is None else _numeric_1d(x, "x")
    if len(horizontal) != len(actual):
        raise ValueError("x must align with truth")
    finite = np.isfinite(actual) & np.isfinite(estimate) & np.isfinite(horizontal)
    actual, estimate, horizontal = actual[finite], estimate[finite], horizontal[finite]
    figure, axes = _subplots("prediction", 1, 2, figsize=(11.7, 4.5))
    if not len(actual):
        for axis in axes:
            axis.text(0.5, 0.5, "No finite prediction pairs", transform=axis.transAxes, ha="center", va="center")
        return figure
    axes[0].plot(horizontal, actual, color="#111827", linewidth=1.6, label="true")
    axes[0].plot(horizontal, estimate, color=PALETTE[0], linewidth=1.4, label="predicted")
    _finish_axis(axes[0], title="Prediction on held-out samples", xlabel="Sample", ylabel=ylabel, legend=True)
    residual = actual - estimate
    axes[1].scatter(estimate, residual, s=22, alpha=0.48, color=PALETTE[1], edgecolor="none")
    axes[1].axhline(0, color="#6B7280", linestyle="--", linewidth=1.0)
    mae = float(np.mean(np.abs(residual)))
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else np.nan
    axes[1].text(0.03, 0.97, f"MAE = {mae:.3g}\nR-squared = {r2:.3g}", transform=axes[1].transAxes, va="top")
    _finish_axis(axes[1], title="Residual diagnostic", xlabel="Predicted", ylabel="True - predicted", grid="y")
    if title:
        figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    _tag(figure, "prediction", provenance={"n": len(actual), "mae": mae, "r_squared": r2})
    return figure


def timeline(
    stages: Sequence[Any],
    *,
    positions: ArrayLike | None = None,
    groups: Sequence[Any] | None = None,
    durations: ArrayLike | None = None,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "Experimental timeline",
    xlabel: str = "Protocol order",
):
    """Plot ordered stages or acquisitions with optional duration spans."""

    labels = list(stages)
    if not labels:
        return _empty("timeline", "No stages supplied", title=title)
    x = np.arange(len(labels), dtype=float) if positions is None else _numeric_1d(positions, "positions")
    if len(x) != len(labels):
        raise ValueError("positions must have one value per stage")
    if np.any(~np.isfinite(x)) or np.any(np.diff(x) < 0):
        raise ValueError("positions must be finite and ordered")
    categories = ["stage"] * len(labels) if groups is None else list(groups)
    if len(categories) != len(labels):
        raise ValueError("groups must have one value per stage")
    spans = None if durations is None else _numeric_1d(durations, "durations")
    if spans is not None and len(spans) != len(labels):
        raise ValueError("durations must have one value per stage")
    figure, axis = _subplots("timeline", figsize=(max(8.0, len(labels) * 1.25), 3.8))
    axis.plot([x[0], x[-1]], [0, 0], color="#9CA3AF", linewidth=2.0, zorder=1)
    unique = list(dict.fromkeys(categories))
    color_lookup = {label: _color(label, index, palette) for index, label in enumerate(unique)}
    seen_categories = set()
    for index, (label, position, category) in enumerate(zip(labels, x, categories)):
        colour = color_lookup[category]
        if spans is not None and spans[index] > 0:
            axis.plot([position, position + spans[index]], [0, 0], color=colour, linewidth=7, alpha=0.28, solid_capstyle="butt")
        legend_label = _clean_label(category) if groups is not None and category not in seen_categories else None
        axis.scatter(position, 0, s=95, color=colour, edgecolor="white", linewidth=1.0, zorder=2, label=legend_label)
        seen_categories.add(category)
        axis.text(position, 0.18 if index % 2 == 0 else -0.18, _clean_label(label), ha="center", va="bottom" if index % 2 == 0 else "top", rotation=22 if len(labels) > 7 else 0)
    axis.set_ylim(-0.65, 0.65)
    axis.set_yticks([])
    _finish_axis(axis, title=title, xlabel=xlabel, legend=groups is not None and len(unique) > 1)
    return figure


def corridor(
    profiles: Mapping[Any, ArrayLike] | ArrayLike,
    *,
    position: ArrayLike | None = None,
    band: str | None = "sem",
    individuals: bool = True,
    texture_end: float = 4.0,
    corridor_end: float = 6.0,
    events: Mapping[str, Any] | None = None,
    reference: float | None = None,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "Corridor profile",
    ylabel: str = "Value",
):
    """Plot position profiles with the 0-4 m texture and 4-6 m grey regions."""

    prepared = _groups(profiles)
    length = np.asarray(next(iter(prepared.values()))).shape[-1] if prepared else 0
    x = np.linspace(0, corridor_end, length) if position is None else _numeric_1d(position, "position")
    regions = OrderedDict([("texture", (0.0, texture_end)), ("grey", (texture_end, corridor_end))])
    if events:
        regions.update(events)
    figure = curve(
        profiles,
        x=x,
        band=band,
        individuals=individuals,
        reference=reference,
        regions=regions,
        palette=palette,
        title=title,
        xlabel="Position (m)",
        ylabel=ylabel,
        figsize=(8.4, 4.7),
    )
    if figure.axes:
        figure.axes[0].legend()
    _tag(figure, "corridor", provenance={"texture_end_m": texture_end, "corridor_end_m": corridor_end})
    return figure


def _paper_dprime(first: np.ndarray, second: np.ndarray) -> float:
    first = _finite(first)
    second = _finite(second)
    if not len(first) or not len(second):
        return np.nan
    denominator = (np.std(first, ddof=0) + np.std(second, ddof=0)) / 2.0
    difference = np.mean(first) - np.mean(second)
    if denominator == 0:
        if difference == 0:
            return np.nan
        return float(np.sign(difference) * np.inf)
    return float(difference / denominator)


def dprime(
    role_2: ArrayLike,
    role_0: ArrayLike,
    *,
    labels: tuple[str, str] = ("leaf1 / role 2", "circle1 / role 0"),
    threshold: float = 0.3,
    bandwidth: Any = None,
    title: str = "Paper-specific signed d-prime",
    xlabel: str = "Response",
):
    """Explain the paper's ``2(mean2-mean0)/(sd2+sd0)`` contrast."""

    if not np.isfinite(float(threshold)) or float(threshold) < 0:
        raise ValueError("threshold must be finite and non-negative")
    first = _numeric_1d(role_2, "role_2")
    second = _numeric_1d(role_0, "role_0")
    value = _paper_dprime(first, second)
    limits = _shared_range(OrderedDict([(labels[0], first), (labels[1], second)]))
    if limits is None:
        return _empty("dprime", "Both response samples are empty", title=title)
    figure, axis = _subplots("dprime", figsize=(8.0, 4.6))
    grid = np.linspace(limits[0], limits[1], 400)
    for index, (label, sample) in enumerate(((labels[0], first), (labels[1], second))):
        finite = _finite(sample)
        if not len(finite):
            continue
        density_values = _density(finite, grid, bandwidth)
        colour = _color(label, index)
        axis.plot(grid, density_values, color=colour, linewidth=2.2, label=_clean_label(label))
        axis.fill_between(grid, 0, density_values, color=colour, alpha=0.16)
        axis.axvline(np.mean(finite), color=colour, linestyle="--", linewidth=1.1)
    status = "selective" if not np.isnan(value) and abs(value) >= threshold else "below threshold"
    axis.text(0.03, 0.96, f"d-prime = {value:.3g}\n|d-prime| {'>=' if status == 'selective' else '<'} {threshold:g}: {status}", transform=axis.transAxes, va="top", ha="left")
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel="Density", legend=True)
    _tag(figure, "dprime", caption="Positive means role 2 exceeds role 0. Formula uses population SD (ddof=0), matching the paper.", provenance={"dprime": value, "threshold": threshold, "sign": "role_2_minus_role_0"})
    return figure


def forest(
    labels: Sequence[Any],
    estimates: ArrayLike,
    intervals: ArrayLike,
    *,
    reference: float = 0.0,
    title: str = "Sensitivity estimates",
    xlabel: str = "Estimate",
):
    """Plot named estimates and intervals, for example leave-one-mouse-out."""

    names = list(labels)
    estimate = _numeric_1d(estimates, "estimates")
    interval = _numeric_2d(intervals, "intervals")
    if len(names) != len(estimate) or interval.shape != (len(estimate), 2):
        raise ValueError("labels, estimates, and (n, 2) intervals must align")
    finite_bounds = np.isfinite(interval).all(axis=1)
    if np.any(finite_bounds & (interval[:, 0] > interval[:, 1])):
        raise ValueError("every finite interval must have lower <= upper")
    figure, axis = _subplots("forest", figsize=(8.2, max(4.0, 0.35 * len(names) + 2.0)))
    y = np.arange(len(names))
    finite = np.isfinite(estimate) & np.isfinite(interval).all(axis=1)
    colors_array = np.where(estimate >= reference, PALETTE[2], PALETTE[4])
    axis.hlines(y[finite], interval[finite, 0], interval[finite, 1], color="#6B7280", linewidth=1.5)
    axis.scatter(estimate[finite], y[finite], c=colors_array[finite], s=38, zorder=3)
    axis.axvline(float(reference), color="#6B7280", linestyle="--", linewidth=1.0)
    axis.set_yticks(y, [_clean_label(label) for label in names])
    axis.invert_yaxis()
    _finish_axis(axis, title=title, xlabel=xlabel, grid="x")
    return figure


def permutation(
    null: ArrayLike,
    observed: float,
    *,
    bins: int | str = "auto",
    pvalue: float | None = None,
    title: str = "Exact-label permutation distribution",
    xlabel: str = "Null estimate",
):
    """Plot a supplied null distribution; no permutations are generated here."""

    values = _finite(null)
    if not np.isfinite(observed):
        raise ValueError("observed must be finite")
    if not len(values):
        return _empty("permutation", "Null samples were not supplied by the analysis result", title=title)
    figure, axis = _subplots("permutation", figsize=(7.6, 4.6))
    axis.hist(values, bins=bins, color=PALETTE[0], alpha=0.55, edgecolor="white")
    axis.axvline(float(observed), color=PALETTE[4], linewidth=2.2, label=f"observed {observed:.3g}")
    if pvalue is not None:
        if not 0 <= pvalue <= 1:
            raise ValueError("pvalue must be between 0 and 1")
        axis.text(0.97, 0.96, f"p = {pvalue:.3g}", transform=axis.transAxes, ha="right", va="top")
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel="Permutations", grid="y", legend=True)
    _tag(figure, "permutation", provenance={"null_samples": len(values), "observed": observed, "pvalue": pvalue})
    return figure


def bars(
    values: Mapping[Any, float] | ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
    errors: ArrayLike | None = None,
    horizontal: bool = False,
    value_labels: bool = True,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
):
    """Plot non-inferential counts, sizes, or totals with optional errors."""

    if isinstance(values, Mapping):
        names = [str(label) for label in values]
        data = np.asarray(list(values.values()), dtype=float)
    else:
        data = _numeric_1d(values, "values")
        names = [str(label) for label in (labels or range(len(data)))]
        if len(names) != len(data):
            raise ValueError("labels must align with values")
    uncertainty = None if errors is None else _numeric_1d(errors, "errors")
    if uncertainty is not None and len(uncertainty) != len(data):
        raise ValueError("errors must align with values")
    figure, axis = _subplots("bars", figsize=(max(6.5, 0.75 * len(data) + 3.0), 4.6))
    positions = np.arange(len(data))
    colour_values = [_color(label, index, palette) for index, label in enumerate(names)]
    if horizontal:
        artists = axis.barh(positions, data, xerr=uncertainty, color=colour_values, alpha=0.82)
        axis.set_yticks(positions, [_clean_label(name) for name in names])
        if value_labels:
            axis.bar_label(artists, fmt="%.3g", padding=3)
    else:
        artists = axis.bar(positions, data, yerr=uncertainty, color=colour_values, alpha=0.82)
        axis.set_xticks(positions, [_clean_label(name) for name in names], rotation=30 if len(names) > 6 else 0, ha="right" if len(names) > 6 else "center")
        if value_labels:
            axis.bar_label(artists, fmt="%.3g", padding=3)
    _finish_axis(axis, title=title, xlabel=xlabel, ylabel=ylabel, grid="x" if horizontal else "y")
    return figure


def stacked_bars(
    segments: Mapping[Any, Mapping[Any, float] | ArrayLike],
    *,
    categories: Sequence[Any] | None = None,
    horizontal: bool = False,
    palette: Mapping[Any, str] | Sequence[str] | None = None,
    title: str = "",
    ylabel: str = "",
):
    """Plot category totals stacked by named segment/layer."""

    if not isinstance(segments, Mapping) or not segments:
        return _empty("stacked_bars", "No segments supplied", title=title)
    mapping_segments = [isinstance(values, Mapping) for values in segments.values()]
    if any(mapping_segments) and not all(mapping_segments):
        raise ValueError("all stacked segments must use mappings or all must use arrays")
    first = next(iter(segments.values()))
    if isinstance(first, Mapping):
        raw_categories = list(categories or first.keys())
        names = [str(value) for value in raw_categories]
        arrays = OrderedDict(
            (
                str(segment),
                np.asarray(
                    [
                        ({str(key): item for key, item in values.items()}).get(str(name), 0.0)
                        for name in raw_categories
                    ],
                    dtype=float,
                ),
            )
            for segment, values in segments.items()
        )
    else:
        arrays = OrderedDict((str(segment), _numeric_1d(values, f"segment {segment}")) for segment, values in segments.items())
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("all stacked segments must have the same category count")
        count = lengths.pop()
        names = [str(value) for value in (categories or range(count))]
        if len(names) != count:
            raise ValueError("categories must align with segment values")
    if any(np.any(~np.isfinite(value)) or np.any(value < 0) for value in arrays.values()):
        raise ValueError("stacked bars require finite non-negative values")
    figure, axis = _subplots("stacked_bars", figsize=(max(7.0, 0.8 * len(names) + 3.0), 4.8))
    positions = np.arange(len(names))
    base = np.zeros(len(names))
    for index, (label, value) in enumerate(arrays.items()):
        if horizontal:
            axis.barh(positions, value, left=base, color=_color(label, index, palette), label=_clean_label(label))
        else:
            axis.bar(positions, value, bottom=base, color=_color(label, index, palette), label=_clean_label(label))
        base += value
    if horizontal:
        axis.set_yticks(positions, [_clean_label(name) for name in names])
    else:
        axis.set_xticks(positions, [_clean_label(name) for name in names], rotation=30 if len(names) > 6 else 0, ha="right" if len(names) > 6 else "center")
    _finish_axis(axis, title=title, ylabel=ylabel, grid="x" if horizontal else "y", legend=True)
    return figure


def image_grid(
    images: Sequence[ArrayLike],
    *,
    labels: Sequence[Any] | None = None,
    columns: int = 4,
    cmap: str = "gray",
    shared_scale: bool = True,
    colorbar: str = "",
    title: str = "",
):
    """Display supplied stimulus, filter, FOV, or reference images."""

    values = [np.asarray(image) for image in images]
    if not values:
        return _empty("image_grid", "No images supplied", title=title)
    names = [str(label) for label in (labels or [""] * len(values))]
    if len(names) != len(values):
        raise ValueError("labels must align with images")
    for index, value in enumerate(values):
        if value.ndim not in (2, 3):
            raise ValueError(f"image {index} must be 2D grayscale or 3D RGB/RGBA")
    columns = max(1, min(int(columns), len(values)))
    rows = int(math.ceil(len(values) / columns))
    figure, axes = _subplots("image_grid", rows, columns, figsize=(3.2 * columns, 3.0 * rows), squeeze=False)
    finite = [value[np.isfinite(value)] for value in values if value.ndim == 2 and np.issubdtype(value.dtype, np.number) and np.isfinite(value).any()]
    limits = None
    if shared_scale and finite:
        joined = np.concatenate(finite)
        limits = np.percentile(joined, [1, 99])
    last_artist = None
    for axis, value, label in zip(axes.flat, values, names):
        kwargs = {}
        if value.ndim == 2:
            kwargs["cmap"] = cmap
            if limits is not None:
                kwargs.update(vmin=limits[0], vmax=limits[1])
        last_artist = axis.imshow(value, interpolation="nearest", **kwargs)
        axis.set_axis_off()
        if label:
            axis.set_title(_clean_label(label))
    for axis in axes.flat[len(values):]:
        axis.set_visible(False)
    if colorbar and last_artist is not None and all(value.ndim == 2 for value in values):
        figure.colorbar(last_artist, ax=list(axes.flat[:len(values)]), label=colorbar, fraction=0.025, pad=0.02)
    if title:
        figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    return figure


def recording(
    data: Mapping[str, Any],
    *,
    feature: int = 0,
    title: str | None = None,
):
    """Plot a six-panel dashboard for ``load_atlas_demo()``-shaped data.

    This preset is descriptive.  Published SVD components are not individual
    neurons, and this figure is not a paper-panel reproduction.
    """

    required = {
        "population_features",
        "mean_run_speed",
        "frame_counts",
        "trial_id",
        "wall_name",
        "position_centers_m",
    }
    missing = sorted(required - set(data))
    if missing:
        raise KeyError(f"recording data is missing: {', '.join(missing)}")
    features = np.asarray(data["population_features"], dtype=float)
    if features.ndim != 3:
        raise ValueError("population_features must have shape trials x position x features")
    if not 0 <= int(feature) < features.shape[2]:
        raise ValueError(f"feature must be between 0 and {features.shape[2] - 1}")
    speed = _numeric_2d(data["mean_run_speed"], "mean_run_speed")
    counts = _numeric_2d(data["frame_counts"], "frame_counts")
    trials = _numeric_1d(data["trial_id"], "trial_id")
    walls = _one_dimensional(data["wall_name"], "wall_name")
    position = _numeric_1d(data["position_centers_m"], "position_centers_m")
    expected = (features.shape[0], features.shape[1])
    if speed.shape != expected or counts.shape != expected or len(trials) != expected[0] or len(walls) != expected[0] or len(position) != expected[1]:
        raise ValueError("recording arrays do not share trial and position axes")
    position_step = _uniform_coordinate_step(position, "position_centers_m")
    heatmap_extent = [
        position[0] - position_step / 2,
        position[-1] + position_step / 2,
        features.shape[0] - 0.5,
        -0.5,
    ]
    figure, axes = _subplots("recording", 2, 3, figsize=(14.5, 8.0), squeeze=False)
    unique_walls = list(dict.fromkeys(walls.tolist()))
    wall_code = np.array([unique_walls.index(value) for value in walls])
    axes[0, 0].scatter(trials, wall_code, c=[_color(value, unique_walls.index(value)) for value in walls], marker="|", s=75)
    axes[0, 0].set_yticks(np.arange(len(unique_walls)), [_clean_label(value) for value in unique_walls])
    _finish_axis(axes[0, 0], title="Physical trial order", xlabel="Trial ID", ylabel="Wall")
    _imshow(figure, axes[0, 1], features[:, :, int(feature)], center=0.0, robust=True, cmap="coolwarm", colorbar=f"Feature {int(feature)}", extent=heatmap_extent)
    axes[0, 1].axvline(4.0, color="#6B7280", linestyle="--", linewidth=1.0)
    _finish_axis(axes[0, 1], title="Trial x position activity", xlabel="Position (m)", ylabel="Trial")
    for index, wall in enumerate(unique_walls):
        selected = walls == wall
        mean, sem = _column_mean_sem(features[selected, :, int(feature)])
        axes[0, 2].plot(position, mean, color=_color(wall, index), linewidth=2, label=_clean_label(wall))
        axes[0, 2].fill_between(position, mean - sem, mean + sem, color=_color(wall, index), alpha=0.15)
    axes[0, 2].axvline(4.0, color="#6B7280", linestyle="--", linewidth=1.0)
    _finish_axis(axes[0, 2], title="Role profiles", xlabel="Position (m)", ylabel=f"Feature {int(feature)}", legend=True)
    speed_mean, speed_sem = _column_mean_sem(speed)
    axes[1, 0].plot(position, speed_mean, color=PALETTE[0], linewidth=2.0)
    axes[1, 0].fill_between(position, speed_mean - speed_sem, speed_mean + speed_sem, color=PALETTE[0], alpha=0.16)
    axes[1, 0].axvline(4.0, color="#6B7280", linestyle="--", linewidth=1.0)
    _finish_axis(axes[1, 0], title="Running-speed QC", xlabel="Position (m)", ylabel="Mean speed", grid="y")
    _imshow(figure, axes[1, 1], counts, robust=True, cmap="viridis", colorbar="Frames / bin", extent=heatmap_extent)
    axes[1, 1].axvline(4.0, color="white", linestyle="--", linewidth=1.0)
    _finish_axis(axes[1, 1], title="Trial x position support", xlabel="Position (m)", ylabel="Trial")
    mean_counts, _ = _column_mean_sem(counts)
    missing_fraction = np.mean((~np.isfinite(counts)) | (counts <= 0), axis=0)
    axes[1, 2].plot(position, mean_counts, color=PALETTE[1], linewidth=2.0, label="frames / bin")
    support_axis = axes[1, 2].twinx()
    support_axis.plot(position, missing_fraction, color=PALETTE[4], linewidth=1.6, linestyle="--", label="missing fraction")
    support_axis.set_ylabel("Missing fraction")
    support_axis.set_ylim(-0.02, 1.02)
    axes[1, 2].axvline(4.0, color="#6B7280", linestyle="--", linewidth=1.0)
    _finish_axis(axes[1, 2], title="Coverage summary", xlabel="Position (m)", ylabel="Mean frames", grid="y")
    handles, labels = axes[1, 2].get_legend_handles_labels()
    other_handles, other_labels = support_axis.get_legend_handles_labels()
    axes[1, 2].legend(handles + other_handles, labels + other_labels)
    display_title = title or f"{data.get('metadata', {}).get('session', 'Compact released recording')} - descriptive feature dashboard"
    figure.suptitle(display_title, x=0.01, ha="left", fontweight="semibold")
    _tag(figure, "recording", caption="Compact published SVD feature dashboard; not single-neuron inference.", provenance={"trials": features.shape[0], "positions": features.shape[1], "feature": int(feature)}, warnings=("Published SVD components are a lossy, transductive representation.",))
    return figure


def released_example(
    data: Mapping[str, Any],
    *,
    role_2: int = 2,
    role_0: int = 0,
    threshold: float = 0.3,
    title: str = "Compact released feature check - not a single-neuron paper result",
):
    """Summarize the bundled released recording without overclaiming its SVD features.

    The compact archive contains population and area factors rather than the
    full deconvolved traces used for paper-style per-neuron distributions.  The
    four panels are therefore an orientation/QC check: physical role counts,
    signed component contrasts, role means, and contrasts by area factor.
    """

    required = {
        "population_features",
        "area_features",
        "stimulus_id",
        "texture_bin_mask",
        "area_name",
    }
    missing = sorted(required - set(data))
    if missing:
        raise KeyError(f"released example data is missing: {', '.join(missing)}")
    if role_2 == role_0:
        raise ValueError("role_2 and role_0 must be distinct")
    if not np.isfinite(float(threshold)) or float(threshold) < 0:
        raise ValueError("threshold must be finite and non-negative")
    population = np.asarray(data["population_features"], dtype=float)
    area_features = np.asarray(data["area_features"], dtype=float)
    labels = _one_dimensional(data["stimulus_id"], "stimulus_id")
    texture = np.asarray(data["texture_bin_mask"], dtype=bool)
    area_names = _one_dimensional(data["area_name"], "area_name")
    if population.ndim != 3 or len(labels) != population.shape[0]:
        raise ValueError("population_features must be trials x position x features and align with stimulus_id")
    if texture.shape != (population.shape[1],):
        raise ValueError("texture_bin_mask must have one value per position bin")
    if area_features.ndim != 4 or area_features.shape[0] != len(area_names) or area_features.shape[1:3] != population.shape[:2]:
        raise ValueError("area_features must be area x trials x position x factors")
    selected = np.isin(labels, [role_2, role_0])
    if not np.any(labels == role_2) or not np.any(labels == role_0):
        raise ValueError("both requested stimulus roles must be present")
    with np.errstate(all="ignore"):
        response = np.nanmean(population[:, texture, :], axis=1)
    first = response[labels == role_2]
    second = response[labels == role_0]
    mean_first, mean_second = np.nanmean(first, axis=0), np.nanmean(second, axis=0)
    denominator = (np.nanstd(first, axis=0, ddof=0) + np.nanstd(second, axis=0, ddof=0)) / 2.0
    component_dprime = np.full(population.shape[2], np.nan)
    np.divide(mean_first - mean_second, denominator, out=component_dprime, where=denominator > 0)

    figure, axes = _subplots("released_example", 2, 2, figsize=(11.8, 8.0), squeeze=False)
    role_names = [ROLE_LABELS.get(role_0, str(role_0)), ROLE_LABELS.get(role_2, str(role_2))]
    counts = [np.count_nonzero(labels == role_0), np.count_nonzero(labels == role_2)]
    artists = axes[0, 0].bar(
        np.arange(2),
        counts,
        color=[_color(role_names[0], 0), _color(role_names[1], 1)],
    )
    axes[0, 0].set_xticks(np.arange(2), role_names)
    axes[0, 0].bar_label(artists)
    _finish_axis(axes[0, 0], title="Physical trial-role support", ylabel="Trials", grid="y")

    finite = _finite(component_dprime)
    axes[0, 1].hist(finite, bins="auto", color=PALETTE[0], alpha=0.65, edgecolor="white")
    axes[0, 1].axvline(threshold, color=_color("leaf1", 0), linestyle="--", linewidth=1.2)
    axes[0, 1].axvline(-threshold, color=_color("circle1", 1), linestyle="--", linewidth=1.2)
    axes[0, 1].axvline(0, color="#6B7280", linewidth=0.9)
    _finish_axis(axes[0, 1], title="Signed SVD-feature contrasts", xlabel="Role 2 - role 0 d-prime", ylabel="Features", grid="y")

    selective = np.isfinite(component_dprime) & (np.abs(component_dprime) >= threshold)
    axes[1, 0].scatter(mean_second, mean_first, s=24, color="#9CA3AF", alpha=0.6, label="all finite features")
    axes[1, 0].scatter(mean_second[selective], mean_first[selective], s=35, c=np.where(component_dprime[selective] >= 0, _color("leaf1", 0), _color("circle1", 1)), edgecolor="white", linewidth=0.5, label=f"|d-prime| >= {threshold:g}")
    paired_values = np.concatenate([mean_first[np.isfinite(mean_first)], mean_second[np.isfinite(mean_second)]])
    if len(paired_values):
        low, high = np.min(paired_values), np.max(paired_values)
        axes[1, 0].plot([low, high], [low, high], color="#6B7280", linestyle="--", linewidth=1.0)
    _finish_axis(axes[1, 0], title="Feature response means", xlabel=f"{role_names[0]} mean", ylabel=f"{role_names[1]} mean", grid="both", legend=True)

    positions = np.arange(len(area_names))
    for index, (area_name, area_values) in enumerate(zip(area_names, area_features)):
        with np.errstate(all="ignore"):
            area_response = np.nanmean(area_values[:, texture, :], axis=1)
        role_first = area_response[labels == role_2]
        role_second = area_response[labels == role_0]
        area_denominator = (np.nanstd(role_first, axis=0, ddof=0) + np.nanstd(role_second, axis=0, ddof=0)) / 2.0
        area_dp = np.full(area_values.shape[-1], np.nan)
        np.divide(
            np.nanmean(role_first, axis=0) - np.nanmean(role_second, axis=0),
            area_denominator,
            out=area_dp,
            where=area_denominator > 0,
        )
        finite_area = _finite(area_dp)
        axes[1, 1].scatter(index + _jitter(len(finite_area), 0.18), finite_area, s=23, color=_color(area_name, index), alpha=0.62)
        if len(finite_area):
            axes[1, 1].scatter(index, np.mean(finite_area), marker="D", s=40, color="#111827", zorder=3)
    axes[1, 1].axhline(0, color="#6B7280", linewidth=1.0)
    axes[1, 1].set_xticks(positions, [_clean_label(value) for value in area_names])
    _finish_axis(axes[1, 1], title="Area-factor contrasts", ylabel="Role 2 - role 0 d-prime", grid="y")
    figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    _tag(
        figure,
        "released_example",
        caption="Descriptive contrasts across published SVD/area factors, not individual neurons.",
        provenance={
            "role_2": role_2,
            "role_0": role_0,
            "threshold": threshold,
            "trials_used": int(np.count_nonzero(selected)),
        },
        warnings=(
            "The compact representation is lossy and transductive.",
            "Exact paper-style single-neuron tails require full deconvolved traces.",
        ),
    )
    return figure


def blockwise(
    result: Mapping[str, Any],
    *,
    title: str = "Held-out within-session discriminability",
):
    """Plot a validated ``blockwise_dprime`` result with visible support."""

    required = {"midpoint", "dprime"}
    missing = sorted(required - set(result))
    if missing:
        raise KeyError(f"blockwise result is missing: {', '.join(missing)}")
    midpoint = _numeric_1d(result["midpoint"], "midpoint")
    dprime_values = _numeric_1d(result["dprime"], "dprime")
    if len(midpoint) != len(dprime_values):
        raise ValueError("midpoint and dprime must align")
    figure, axes = _subplots("blockwise", 2, 2, figsize=(11.6, 7.4), squeeze=False, sharex=True)
    axes[0, 0].axhline(0, color="#6B7280", linewidth=1.0)
    axes[0, 0].plot(midpoint, dprime_values, "o-", color=PALETTE[2], linewidth=2.0)
    _finish_axis(axes[0, 0], title="Held-out block d-prime", ylabel="d-prime", grid="y")
    separation = np.asarray(result.get("separation", np.full(len(midpoint), np.nan)), dtype=float)
    spread = np.asarray(result.get("spread", np.full(len(midpoint), np.nan)), dtype=float)
    if len(separation) != len(midpoint) or len(spread) != len(midpoint):
        raise ValueError("separation and spread must align with midpoint")
    axes[0, 1].plot(midpoint, separation, "o-", color=PALETTE[0], label="mean separation")
    axes[0, 1].plot(midpoint, spread, "o--", color=PALETTE[1], label="within-role spread")
    _finish_axis(axes[0, 1], title="d-prime components", ylabel="Held-out score units", grid="y", legend=True)
    n_a = np.asarray(result.get("n_a", np.full(len(midpoint), np.nan)), dtype=float)
    n_b = np.asarray(result.get("n_b", np.full(len(midpoint), np.nan)), dtype=float)
    if len(n_a) != len(midpoint) or len(n_b) != len(midpoint):
        raise ValueError("n_a and n_b must align with midpoint")
    axes[1, 0].plot(midpoint, n_a, "o-", color=_color("leaf1", 0), label="role 2")
    axes[1, 0].plot(midpoint, n_b, "o-", color=_color("circle1", 1), label="role 0")
    _finish_axis(axes[1, 0], title="Held-out role support", xlabel="Physical trial midpoint", ylabel="Trials", grid="y", legend=True)
    valid = np.asarray(result.get("valid_folds", np.full(len(midpoint), np.nan)), dtype=float)
    required_folds = np.asarray(result.get("required_folds", np.full(len(midpoint), np.nan)), dtype=float)
    if len(valid) != len(midpoint) or len(required_folds) != len(midpoint):
        raise ValueError("valid_folds and required_folds must align with midpoint")
    axes[1, 1].plot(midpoint, required_folds, "o--", color="#6B7280", label="required folds")
    axes[1, 1].plot(midpoint, valid, "o-", color=PALETTE[2], label="valid folds")
    invalid = valid < required_folds
    if np.any(invalid):
        axes[1, 1].scatter(midpoint[invalid], valid[invalid], facecolor="none", edgecolor=PALETTE[4], s=70, linewidth=1.5, label="invalid support")
    _finish_axis(axes[1, 1], title="Fold completeness", xlabel="Physical trial midpoint", ylabel="Folds", grid="y", legend=True)
    figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    warnings = ()
    if "invalid_reason" not in result:
        warnings = ("The current blockwise result schema does not retain per-block invalid reasons.",)
    _tag(figure, "blockwise", caption="Fold-wise held-out contrasts; raw scores from independently fitted folds are not pooled.", provenance={"blocks": len(midpoint)}, warnings=warnings)
    return figure


def qc(
    prepared: Mapping[str, Any],
    *,
    position: ArrayLike | None = None,
    labels: ArrayLike | None = None,
    title: str = "Recording support and behavior QC",
):
    """Plot adjacent speed, support, missingness, and role-count diagnostics."""

    speed_key = "mean_run_speed" if "mean_run_speed" in prepared else "speed"
    count_key = "frame_counts" if "frame_counts" in prepared else "counts"
    if speed_key not in prepared or count_key not in prepared:
        raise KeyError("qc requires mean_run_speed/speed and frame_counts/counts")
    speed = _numeric_2d(prepared[speed_key], speed_key)
    counts = _numeric_2d(prepared[count_key], count_key)
    if speed.shape != counts.shape:
        raise ValueError("speed and frame counts must share trial x position shape")
    x = np.arange(speed.shape[1], dtype=float) if position is None else _numeric_1d(position, "position")
    if len(x) != speed.shape[1]:
        raise ValueError("position must match the position-bin count")
    figure, axes = _subplots("qc", 2, 2, figsize=(11.6, 7.3), squeeze=False)
    speed_mean, speed_sem = _column_mean_sem(speed)
    axes[0, 0].plot(x, speed_mean, color=PALETTE[0], linewidth=2.0)
    axes[0, 0].fill_between(x, speed_mean - speed_sem, speed_mean + speed_sem, color=PALETTE[0], alpha=0.17)
    _finish_axis(axes[0, 0], title="Mean speed", xlabel="Position", ylabel="Speed", grid="y")
    _imshow(figure, axes[0, 1], counts, robust=True, cmap="viridis", colorbar="Frames / bin", origin="upper")
    _finish_axis(axes[0, 1], title="Trial x position support", xlabel="Position bin", ylabel="Trial")
    mean_count, _ = _column_mean_sem(counts)
    missing = np.mean((~np.isfinite(counts)) | (counts <= 0), axis=0)
    axes[1, 0].plot(x, mean_count, color=PALETTE[1], linewidth=2.0, label="mean frames")
    missing_axis = axes[1, 0].twinx()
    missing_axis.plot(x, missing, color=PALETTE[4], linestyle="--", linewidth=1.7, label="missing fraction")
    missing_axis.set_ylabel("Missing fraction")
    missing_axis.set_ylim(-0.02, 1.02)
    handles, legend_labels = axes[1, 0].get_legend_handles_labels()
    other_handles, other_labels = missing_axis.get_legend_handles_labels()
    axes[1, 0].legend(handles + other_handles, legend_labels + other_labels)
    _finish_axis(axes[1, 0], title="Position coverage", xlabel="Position", ylabel="Mean frames", grid="y")
    if labels is None:
        axes[1, 1].set_axis_off()
        axes[1, 1].text(0.5, 0.5, "Role labels not supplied", transform=axes[1, 1].transAxes, ha="center", va="center", color="#6B7280")
    else:
        role = _one_dimensional(labels, "labels")
        if len(role) != speed.shape[0]:
            raise ValueError("labels must have one value per trial")
        unique, role_counts = np.unique(role, return_counts=True)
        role_names = [ROLE_LABELS.get(int(value), str(value)) if np.issubdtype(type(value), np.number) else str(value) for value in unique]
        artists = axes[1, 1].bar(np.arange(len(unique)), role_counts, color=[_color(label, index) for index, label in enumerate(role_names)])
        axes[1, 1].set_xticks(np.arange(len(unique)), role_names)
        axes[1, 1].bar_label(artists)
        _finish_axis(axes[1, 1], title="Physical trial-role counts", ylabel="Trials", grid="y")
    figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    _tag(figure, "qc", provenance={"trials": speed.shape[0], "position_bins": speed.shape[1]})
    return figure


def _rows(records: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    try:
        rows = list(records)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of mapping rows") from error
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"every {name} row must be a mapping")
    return rows


def _record_positions(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[np.ndarray, list[str] | None]:
    raw = [row.get(key, index) for index, row in enumerate(rows)]
    try:
        values = np.asarray(raw, dtype=float)
        if np.isfinite(values).all():
            return values, None
    except (TypeError, ValueError):
        pass
    labels = [str(value) for value in raw]
    unique = list(dict.fromkeys(labels))
    lookup = {value: index for index, value in enumerate(unique)}
    return np.asarray([lookup[value] for value in labels], dtype=float), unique


def mouse_journey(
    records: Sequence[Mapping[str, Any]],
    *,
    stage_key: str = "stage",
    position_key: str = "date",
    metrics: Mapping[str, str] | Sequence[str] | None = None,
    mouse_key: str = "mouse",
    title: str | None = None,
):
    """Plot one mouse's ordered stages plus selected numeric acquisition fields.

    ``metrics`` may map a display label to a row field, or be a list of field
    names.  When omitted, common journey fields are included if present.
    """

    rows = _rows(records, "records")
    if not rows:
        return _empty("mouse_journey", "No acquisition rows supplied", title=title or "Mouse journey")
    if any(stage_key not in row for row in rows):
        raise KeyError(f"every acquisition row must contain {stage_key!r}")
    positions, tick_labels = _record_positions(rows, position_key)
    if metrics is None:
        candidates = OrderedDict(
            [
                ("Neurons", "neurons"),
                ("Neural frames", "neural_frames"),
                ("Behavior trials", "trials"),
                ("Licks / trial", "licks_per_trial"),
            ]
        )
        metric_map = OrderedDict(
            (label, key)
            for label, key in candidates.items()
            if any(key in row and np.isscalar(row[key]) for row in rows)
        )
    elif isinstance(metrics, Mapping):
        metric_map = OrderedDict((str(label), str(key)) for label, key in metrics.items())
    else:
        metric_map = OrderedDict((_clean_label(key), str(key)) for key in metrics)
    panel_count = 1 + len(metric_map)
    figure, axes = _subplots(
        "mouse_journey",
        panel_count,
        1,
        figsize=(10.5, 2.7 + 2.35 * len(metric_map)),
        squeeze=False,
        sharex=True,
    )
    axes = axes[:, 0]
    stages = [str(row[stage_key]) for row in rows]
    stage_types = list(dict.fromkeys(stages))
    axes[0].plot([positions[0], positions[-1]], [0, 0], color="#9CA3AF", linewidth=1.8)
    for index, (position, stage) in enumerate(zip(positions, stages)):
        colour = _color(stage, stage_types.index(stage))
        axes[0].scatter(position, 0, s=74, color=colour, edgecolor="white", linewidth=0.8, zorder=2)
        axes[0].text(position, 0.14 if index % 2 == 0 else -0.14, _clean_label(stage), ha="center", va="bottom" if index % 2 == 0 else "top", fontsize=8)
    axes[0].set_ylim(-0.55, 0.55)
    axes[0].set_yticks([])
    _finish_axis(axes[0], title="Acquisition stages")
    warnings: list[str] = []
    for axis, (label, key) in zip(axes[1:], metric_map.items()):
        values = []
        for row in rows:
            value = row.get(key, np.nan)
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(np.nan)
        values_array = np.asarray(values)
        axis.plot(positions, values_array, "o-", color=_color(label, len(warnings)), linewidth=1.8)
        missing = int(np.count_nonzero(~np.isfinite(values_array)))
        if missing:
            warnings.append(f"{label}: {missing} acquisition(s) missing")
        _finish_axis(axis, title=_clean_label(label), ylabel=_clean_label(label), grid="y")
    if tick_labels is not None:
        axes[-1].set_xticks(np.arange(len(tick_labels)), tick_labels, rotation=35, ha="right")
    axes[-1].set_xlabel(_clean_label(position_key))
    mouse = rows[0].get(mouse_key, "Mouse")
    figure.suptitle(title or f"{mouse} - complete indexed journey", x=0.01, ha="left", fontweight="semibold")
    _tag(figure, "mouse_journey", provenance={"mouse": mouse, "acquisitions": len(rows)}, warnings=warnings)
    return figure


def all_mouse_journeys(
    journeys: Mapping[Any, Sequence[Mapping[str, Any]]],
    *,
    stage_key: str = "stage",
    position_key: str = "date",
    title: str = "All indexed mouse journeys",
):
    """Plot acquisition lanes and a binary mouse-by-stage coverage matrix."""

    if not isinstance(journeys, Mapping) or not journeys:
        return _empty("all_mouse_journeys", "No mouse journeys supplied", title=title)
    prepared = OrderedDict((str(mouse), _rows(records, f"records for {mouse}")) for mouse, records in journeys.items())
    if not any(prepared.values()):
        return _empty("all_mouse_journeys", "No acquisition rows supplied", title=title)
    if any(any(stage_key not in row for row in records) for records in prepared.values()):
        raise KeyError(f"every journey row must contain {stage_key!r}")
    all_position_labels = [str(row.get(position_key, index)) for records in prepared.values() for index, row in enumerate(records)]
    unique_positions = sorted(set(all_position_labels))
    position_lookup = {value: index for index, value in enumerate(unique_positions)}
    stage_names = list(dict.fromkeys(str(row[stage_key]) for records in prepared.values() for row in records))
    stage_lookup = {stage: index for index, stage in enumerate(stage_names)}
    coverage = np.zeros((len(prepared), len(stage_names)), dtype=float)
    figure, axes = _subplots("all_mouse_journeys", 1, 2, figsize=(14.2, max(5.0, 0.32 * len(prepared) + 2.4)))
    for mouse_index, (mouse, records) in enumerate(prepared.items()):
        positions = [position_lookup[str(row.get(position_key, index))] for index, row in enumerate(records)]
        if positions:
            axes[0].plot([min(positions), max(positions)], [mouse_index, mouse_index], color="#D1D5DB", linewidth=1.0)
        for position, row in zip(positions, records):
            stage = str(row[stage_key])
            stage_index = stage_lookup[stage]
            coverage[mouse_index, stage_index] = 1.0
            axes[0].scatter(position, mouse_index, s=36, color=_color(stage, stage_index), edgecolor="white", linewidth=0.45)
    axes[0].set_yticks(np.arange(len(prepared)), list(prepared))
    axes[0].invert_yaxis()
    if len(unique_positions) <= 18:
        axes[0].set_xticks(np.arange(len(unique_positions)), unique_positions, rotation=45, ha="right")
    else:
        tick_index = np.linspace(0, len(unique_positions) - 1, 10, dtype=int)
        axes[0].set_xticks(tick_index, [unique_positions[index] for index in tick_index], rotation=45, ha="right")
    _finish_axis(axes[0], title="Acquisition lanes", xlabel=_clean_label(position_key), ylabel="Mouse")
    image = axes[1].imshow(coverage, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest", rasterized=True)
    axes[1].set_yticks(np.arange(len(prepared)), list(prepared))
    axes[1].set_xticks(np.arange(len(stage_names)), [_clean_label(stage) for stage in stage_names], rotation=45, ha="right")
    _finish_axis(axes[1], title="Indexed stage coverage", xlabel="Stage", ylabel="Mouse")
    figure.colorbar(image, ax=axes[1], ticks=[0, 1], label="Stage represented", fraction=0.046, pad=0.04)
    figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    _tag(figure, "all_mouse_journeys", provenance={"mice": len(prepared), "stages": len(stage_names), "acquisitions": int(np.sum([len(rows) for rows in prepared.values()]))})
    return figure


def cohort_preflight(
    records: Sequence[Mapping[str, Any]],
    *,
    group_key: str = "group",
    mouse_key: str = "mouse",
    date_key: str = "date",
    layers: Mapping[str, str] | None = None,
    title: str = "Cohort and storage preflight",
):
    """Summarize mice by group, acquisition dates, and layer storage.

    ``layers`` maps a display label to a byte-count field.  If omitted, fields
    ending in ``_bytes`` are discovered from the first row.  This is a manifest
    view only; no files are downloaded.
    """

    rows = _rows(records, "records")
    if not rows:
        return _empty("cohort_preflight", "No manifest rows supplied", title=title)
    for key in (group_key, mouse_key):
        if any(key not in row for row in rows):
            raise KeyError(f"every manifest row must contain {key!r}")
    groups = list(dict.fromkeys(str(row[group_key]) for row in rows))
    mice_by_group = OrderedDict(
        (
            group,
            len({str(row[mouse_key]) for row in rows if str(row[group_key]) == group}),
        )
        for group in groups
    )
    if layers is None:
        layer_map = OrderedDict(
            (_clean_label(key.removesuffix("_bytes")), key)
            for key in rows[0]
            if str(key).endswith("_bytes")
        )
    else:
        layer_map = OrderedDict((str(label), str(key)) for label, key in layers.items())
    figure, axes = _subplots("cohort_preflight", 1, 3, figsize=(14.8, 4.8))
    artists = axes[0].bar(
        np.arange(len(groups)),
        list(mice_by_group.values()),
        color=[_color(group, index) for index, group in enumerate(groups)],
    )
    axes[0].set_xticks(np.arange(len(groups)), [_clean_label(group) for group in groups])
    axes[0].bar_label(artists)
    _finish_axis(axes[0], title="Unique mice within each group", ylabel="Mice", grid="y")

    dates = [str(row.get(date_key, index)) for index, row in enumerate(rows)]
    unique_dates = sorted(set(dates))
    date_lookup = {value: index for index, value in enumerate(unique_dates)}
    for group_index, group in enumerate(groups):
        selected_rows = [row for row in rows if str(row[group_key]) == group]
        x = [date_lookup[str(row.get(date_key, 0))] for row in selected_rows]
        y = group_index + _jitter(len(selected_rows), 0.34)
        axes[1].scatter(x, y, s=35, color=_color(group, group_index), alpha=0.75, label=_clean_label(group))
    axes[1].set_yticks(np.arange(len(groups)), [_clean_label(group) for group in groups])
    if len(unique_dates) <= 12:
        axes[1].set_xticks(np.arange(len(unique_dates)), unique_dates, rotation=45, ha="right")
    else:
        tick_index = np.linspace(0, len(unique_dates) - 1, 8, dtype=int)
        axes[1].set_xticks(tick_index, [unique_dates[index] for index in tick_index], rotation=45, ha="right")
    _finish_axis(axes[1], title="Indexed acquisitions", xlabel=_clean_label(date_key), ylabel="Cohort")

    base = np.zeros(len(groups))
    total_bytes = 0.0
    if layer_map:
        for layer_index, (label, key) in enumerate(layer_map.items()):
            values = np.asarray(
                [
                    sum(float(row.get(key, 0) or 0) for row in rows if str(row[group_key]) == group)
                    / (1024 ** 3)
                    for group in groups
                ]
            )
            axes[2].bar(np.arange(len(groups)), values, bottom=base, color=_color(label, layer_index), label=_clean_label(label))
            base += values
            total_bytes += float(np.sum(values))
        axes[2].set_xticks(np.arange(len(groups)), [_clean_label(group) for group in groups])
        _finish_axis(axes[2], title="Reported layer totals", ylabel="GiB", grid="y", legend=True)
    else:
        axes[2].set_axis_off()
        axes[2].text(0.5, 0.5, "No *_bytes layer fields supplied", transform=axes[2].transAxes, ha="center", va="center", color="#6B7280")
    figure.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    unique_mice = len({str(row[mouse_key]) for row in rows})
    group_mouse_memberships = sum(mice_by_group.values())
    memberships_by_mouse: dict[str, set[str]] = {}
    for row in rows:
        memberships_by_mouse.setdefault(str(row[mouse_key]), set()).add(str(row[group_key]))
    overlapping_mice = sorted(mouse for mouse, memberships in memberships_by_mouse.items() if len(memberships) > 1)
    overlap_warnings = (
        f"{len(overlapping_mice)} mice appear in multiple groups; group bars are not independent.",
    ) if overlapping_mice else ()
    _tag(
        figure,
        "cohort_preflight",
        caption=(
            "Manifest-only view; no data files were loaded. Mice are deduplicated "
            "within each group and may contribute to more than one group bar."
        ),
        provenance={
            "rows": len(rows),
            "groups": len(groups),
            "unique_mice": unique_mice,
            "group_mouse_memberships": group_mouse_memberships,
            "overlapping_mice": ", ".join(overlapping_mice) or "none",
            "total_gib": total_bytes,
        },
        warnings=overlap_warnings,
    )
    return figure


def reference_figure(
    number: int,
    *,
    paper: str = "nature",
    extended: bool = False,
    title: str | None = None,
):
    """Display one packaged paper figure as a clearly labeled reference image."""

    number = int(number)
    paper_key = str(paper).lower()
    if paper_key == "nature":
        if extended:
            if not 1 <= number <= 9:
                raise ValueError("Nature extended figure number must be 1 through 9")
            filename = f"nature-ed-{number}.jpg"
            default_title = f"Nature Extended Data Figure {number} - reference image"
        else:
            if not 1 <= number <= 5:
                raise ValueError("Nature main figure number must be 1 through 5")
            filename = f"nature-main-{number}.png"
            default_title = f"Nature Figure {number} - reference image"
    elif paper_key in {"science", "analysis", "methods"}:
        if extended or not 1 <= number <= 4:
            raise ValueError("Science analysis-review figure number must be 1 through 4")
        filename = f"science-methods-fig{number}.jpg"
        default_title = f"Science analysis-methods Figure {number} - reference image"
    else:
        raise ValueError("paper must be 'nature' or 'science'")
    resource = resources.files("zhong2025.assets").joinpath("reference_figures", filename)
    if not resource.is_file():
        raise FileNotFoundError(f"packaged reference figure is missing: {filename}")
    mpl, _ = _mpl()
    with resources.as_file(resource) as path:
        image = mpl.image.imread(path)
    figure = image_grid([image], labels=[""], columns=1, shared_scale=False, title=title or default_title)
    _tag(figure, "reference_figure", caption="Published reference image; not recomputed from data.", provenance={"paper": paper_key, "figure": number, "extended": extended, "asset": filename})
    return figure


def reference_gallery(
    *,
    paper: str = "nature",
    extended: bool = False,
    columns: int | None = None,
    title: str | None = None,
):
    """Display every packaged main/extended figure for one paper section."""

    paper_key = str(paper).lower()
    if paper_key == "nature":
        numbers = range(1, 10) if extended else range(1, 6)
        filenames = [
            f"nature-ed-{number}.jpg" if extended else f"nature-main-{number}.png"
            for number in numbers
        ]
        labels = [
            f"Extended Data Figure {number}" if extended else f"Figure {number}"
            for number in numbers
        ]
        default_title = (
            "Zhong et al. Nature Extended Data figures - references"
            if extended
            else "Zhong et al. Nature main figures - references"
        )
    elif paper_key in {"science", "analysis", "methods"}:
        if extended:
            raise ValueError("the Science analysis review has no extended gallery")
        numbers = range(1, 5)
        filenames = [f"science-methods-fig{number}.jpg" for number in numbers]
        labels = [f"Figure {number}" for number in numbers]
        default_title = "Stringer & Pachitariu Science review figures - references"
    else:
        raise ValueError("paper must be 'nature' or 'science'")
    mpl, _ = _mpl()
    images = []
    for filename in filenames:
        resource = resources.files("zhong2025.assets").joinpath("reference_figures", filename)
        if not resource.is_file():
            raise FileNotFoundError(f"packaged reference figure is missing: {filename}")
        with resources.as_file(resource) as path:
            images.append(mpl.image.imread(path))
    if columns is None:
        columns = 3 if len(images) > 4 else 2
    figure = image_grid(
        images,
        labels=labels,
        columns=columns,
        shared_scale=False,
        title=title or default_title,
    )
    _tag(
        figure,
        "reference_gallery",
        caption="Published reference images; none are recomputed from data.",
        provenance={"paper": paper_key, "extended": extended, "figures": len(images)},
    )
    return figure


# Keep the complete recipe call inside a private style context.  This makes
# later-created annotations and artists deterministic too, while restoring all
# caller rcParams immediately after the figure has been built.
def _styled(function: Any):
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any):
        mpl, _ = _mpl()
        with mpl.rc_context(_RC):
            return function(*args, **kwargs)

    return wrapped


for _recipe_name in recipes():
    globals()[_recipe_name] = _styled(globals()[_recipe_name])
del _recipe_name


# Friendly domain aliases preserve the exact return type of the underlying
# recipes while making notebook prose read naturally.
learning_curve = curve
population_trace = curve
coding_projection = corridor
paired_summary = comparison
categorical_summary = comparison
position_distribution = distribution
preferred_position_scatter = relationship
representation_similarity = matrix
retinotopy_map = cortical_map
component_trajectory = trajectory
eigenspectrum = spectrum
filter_bank = image_grid
stimulus_montage = image_grid
derivative_qc = recording
session_dashboard = recording
within_session_dprime = blockwise
group_dprime_dynamics = small_multiples
position_support_dashboard = qc


def cross_temporal(result: Mapping[str, Any] | ArrayLike, **kwargs: Any):
    """Plot a raw matrix or a ``cross_temporal_dprime`` result directly."""

    if isinstance(result, Mapping):
        missing = {"midpoint", "dprime"} - set(result)
        if missing:
            raise KeyError(f"cross-temporal result is missing: {', '.join(sorted(missing))}")
        values = result["dprime"]
        kwargs.setdefault("x", result["midpoint"])
        kwargs.setdefault("y", result["midpoint"])
    else:
        values = result
    kwargs.setdefault("center", 0.0)
    kwargs.setdefault("cmap", "coolwarm")
    kwargs.setdefault("colorbar", "Held-out d-prime")
    kwargs.setdefault("title", "Cross-temporal generalization")
    kwargs.setdefault("xlabel", "Test-block midpoint")
    kwargs.setdefault("ylabel", "Train-block midpoint")
    return matrix(values, **kwargs)


def position_surface(
    result: Mapping[str, Any] | ArrayLike,
    *,
    position: ArrayLike | None = None,
    **kwargs: Any,
):
    """Plot a raw matrix or ``position_dprime_surface`` result directly."""

    if isinstance(result, Mapping):
        missing = {"midpoint", "dprime"} - set(result)
        if missing:
            raise KeyError(f"position-surface result is missing: {', '.join(sorted(missing))}")
        values = result["dprime"]
        kwargs.setdefault("y", result["midpoint"])
        if position is None and "x" not in kwargs:
            raise ValueError("position is required for a position_dprime_surface result")
        if position is not None:
            kwargs.setdefault("x", position)
    else:
        values = result
        if position is not None:
            kwargs.setdefault("x", position)
    kwargs.setdefault("center", 0.0)
    kwargs.setdefault("cmap", "coolwarm")
    kwargs.setdefault("colorbar", "Held-out d-prime")
    kwargs.setdefault("title", "Trial progress by corridor position")
    kwargs.setdefault("xlabel", "Position (m)")
    kwargs.setdefault("ylabel", "Physical trial midpoint")
    return matrix(values, **kwargs)


__all__ = [
    "AREA_COLORS",
    "COHORT_COLORS",
    "PALETTE",
    "PlotInfo",
    "Recipe",
    "ROLE_LABELS",
    "STIMULUS_COLORS",
    "activity",
    "agreement",
    "bars",
    "blockwise",
    "categorical_summary",
    "coding_projection",
    "close",
    "cohort_preflight",
    "colors",
    "comparison",
    "component_trajectory",
    "corridor",
    "curve",
    "cortical_density",
    "cortical_map",
    "cross_temporal",
    "density_difference",
    "derivative_qc",
    "distribution",
    "dprime",
    "eigenspectrum",
    "event_raster",
    "filter_bank",
    "forest",
    "guide",
    "group_dprime_dynamics",
    "image_grid",
    "info",
    "learning_curve",
    "matrix",
    "paired_summary",
    "pairwise",
    "permutation",
    "population_trace",
    "position_support_dashboard",
    "position_distribution",
    "position_surface",
    "prediction",
    "preferred_position_scatter",
    "qc",
    "rastermap",
    "recipes",
    "recording",
    "released_example",
    "reference_figure",
    "reference_gallery",
    "relationship",
    "representation_similarity",
    "retinotopy_map",
    "save",
    "session_dashboard",
    "signed_tails",
    "small_multiples",
    "spectrum",
    "stacked_bars",
    "stimulus_montage",
    "timeline",
    "train_test",
    "trajectory",
    "within_session_dprime",
    "mouse_journey",
    "all_mouse_journeys",
]
