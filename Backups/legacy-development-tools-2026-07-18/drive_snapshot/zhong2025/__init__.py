"""Colab-safe catalog and exploration helpers for Zhong et al. (2025)."""

from . import plot

from .atlas import (
    experiment_recordings,
    experiment_rows,
    experiment_semantics,
    filter_inventory,
    format_bytes,
    inventory_summary,
    load_experiment_index,
    load_file_inventory,
    recording_bundle,
)
from .data import (
    FIGSHARE_ARTICLE_API_URL,
    download_profile,
    fetch_figshare_article,
    load_atlas_demo,
    load_manifest,
    profile_summary,
)
from .cache import Cache, user_cache, user_cache_root
from .catalog import (
    areas,
    catalog,
    cohorts,
    describe_session,
    friendly,
    load_map,
    mice,
    roles,
    sessions,
)
from .colab import start
from .widgets import corridor_diagram, dprime, dprime_playground
from .explain import (
    explain,
    explain_dprime,
    explain_field,
    explain_figure,
    explain_layer,
    explain_stage,
    explain_units,
    gotchas,
    reading_path,
    start_here,
    topics,
)

__all__ = [
    # consistent plotting recipes (module, to avoid colliding with dprime())
    "plot",
    # one-cell launcher + per-teammate cache
    "start",
    # plain-language clarity helpers
    "explain",
    "explain_dprime",
    "explain_field",
    "explain_figure",
    "explain_layer",
    "explain_stage",
    "explain_units",
    "gotchas",
    "reading_path",
    "start_here",
    "topics",
    # no-data interactive widgets
    "dprime_playground",
    "corridor_diagram",
    "dprime",
    "user_cache",
    "user_cache_root",
    "Cache",
    # friendly catalog (one-call views of the canonical map)
    "catalog",
    "cohorts",
    "areas",
    "roles",
    "mice",
    "sessions",
    "friendly",
    "describe_session",
    "load_map",
    # existing atlas / data helpers
    "download_profile",
    "experiment_recordings",
    "experiment_rows",
    "experiment_semantics",
    "filter_inventory",
    "FIGSHARE_ARTICLE_API_URL",
    "fetch_figshare_article",
    "format_bytes",
    "inventory_summary",
    "load_atlas_demo",
    "load_experiment_index",
    "load_file_inventory",
    "load_manifest",
    "profile_summary",
    "recording_bundle",
]
