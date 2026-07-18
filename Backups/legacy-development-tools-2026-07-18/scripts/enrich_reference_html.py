from __future__ import annotations

import base64
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "zhong2025_reference.html"
INVENTORY_PATH = ROOT / "zhong2025/assets/figshare-v2-inventory.json"
INDEX_PATH = ROOT / "zhong2025/assets/imaging-experiment-index.json"
FIGURE_ASSET_DIR = ROOT / "zhong2025/assets/reference_figures"

NATURE = "https://www.nature.com/articles/s41586-025-09180-y"
NATURE_DOI = "https://doi.org/10.1038/s41586-025-09180-y"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"
FIGSHARE = "https://doi.org/10.25378/janelia.28811129.v2"
IMAGING_INDEX_SOURCE = "https://ndownloader.figshare.com/files/54183854"
SCIENCE = "https://www.science.org/doi/10.1126/science.adp7429"
SCIENCE_DOI = "https://doi.org/10.1126/science.adp7429"
SCIENCE_PDF = "https://mouseland.github.io/research/science.adp7429.pdf"
SCIENCE_SOURCE_SHA256 = "2a92762cee61070b5fbb5bfac780da03ccee5e5d7ea7c74c7be86c969007e511"
PAPER_DRIVE = "https://drive.google.com/file/d/10u0D2bRCScDujpWHFxwAZuvfDIglOW_i/view?usp=drivesdk"
METHODS_DRIVE = "https://drive.google.com/file/d/1DlmPeyaHn-thn9ILrt-rAXP96-y3IMU7/view?usp=drivesdk"
WORKSPACE_DRIVE = "https://drive.google.com/drive/folders/1jKMIEf2srnu_Dg_TP6NKk7XBIDxYB4EN"
DATA_DRIVE = "https://drive.google.com/drive/folders/1NFnt2FS4Dohc3QFvXkHVcl4Z5rP3Dk3Y"
ORIGINAL_DRIVE = "https://drive.google.com/drive/folders/1AZDYl-7QkbtqLGKuOPqtN1AP3DASuH9l"
NEUROMATCH_DOC = "https://docs.google.com/document/d/1iricfodW_F53Lu6Re4uS9v3SvYNXRQ6PbFwBhyullsk/edit?usp=drivesdk"
JOIN_NOTEBOOK = "https://colab.research.google.com/drive/1Xz40c50g5KczU5Rp5Dz_TYH2n21C-abP"

EXPECTED_INVENTORY_SHA256 = "4f498b97501d038a83ab6ba52bfec109e3b540019bbd37602e9382eb600c3d14"
EXPECTED_INDEX_SHA256 = "ad8eaf217b3908976a3f701d6700d9ffd4479c529d8d2eab345919d694b57650"

NOTEBOOKS = [
    ("00", "Data access", "https://colab.research.google.com/drive/1CN2b_NHigbJ4jPd_2FqWPs3mHRhJZRdT", "Mount the shared release, inspect the catalog, choose files, and verify the selective cache. Access/QC only; no scientific estimator."),
    ("02", "Visual learning / RQ1 sandbox", "https://colab.research.google.com/drive/1YvuuZPrkPNoMFCu15yfBU_V0zeMBwsRz", "Graph 4 explores paired leaf–circle d′ distributions and SD/IQR/skew/excess-kurtosis/tails. It runs one of four example sessions at a time and is not the all-mouse confirmatory analysis."),
    ("03", "Dataset walkthrough", "https://colab.research.google.com/drive/1gvK-N_afSSAeNRjvxhm0Swt3s7H2_xm_", "Cohort, retinotopy, alignment, and compact cross-validated population-d′ mechanics. Use for orientation and QC."),
    ("04", "Paper companion", "https://colab.research.google.com/drive/1rtV1WcAZR_pSm--t_yzmy9nnPdSqNNDh", "Maps protocol, paper figures, file semantics, and code provenance. It is a design reference, not an estimator."),
    ("05", "Reward / d′ dynamics", "https://colab.research.google.com/drive/10SDh3byJ_bv48Ob5dNIKL1H3JgBnCzRP", "RQ2 workflow with preflight, held-out blockwise d′, slopes, saturation, cross-temporal and position analyses, exact mouse permutation, bootstrap, and leave-one-mouse-out checks."),
    ("06", "Older within-session demo", "https://drive.google.com/file/d/1fgbSnfLN28cCuPMOg1PVIrT3MP3u5t4J/view?usp=drivesdk", "One rewarded and one unrewarded session with a cumulative in-sample estimator. Keep only as a sanity check; notebook 05 supersedes it."),
    ("11", "Filesystem-only data join", JOIN_NOTEBOOK, "The shortest explicit route from catalog SQL to behavior frames, SVD neural state, retinotopy, trial × position summaries, and an honestly windowed trial-indexed d′ curve. It uses Drive only as a mounted filesystem and exposes every join key."),
    ("Reference", "Upstream Neuromatch notebook", "https://drive.google.com/file/d/1gfx81il2wj5A1b15VpqGMTe2CM213cSn/view?usp=drivesdk", "Provenance/reference material, not the project’s confirmatory workflow."),
]

CODE = [
    ("drive.py", "https://drive.google.com/file/d/1EJHjI8TMDwad2ZcHoepmvR9w_n868Pbw/view?usp=drivesdk", "Canonical 297-file catalog, recording/layer resolution, MD5 and size checks, atomic VM cache copy, disk check, and a 10 GiB default per-file guard. No HTTP/API fallback is implemented."),
    ("graph.py", "https://drive.google.com/file/d/1JhWd-Heaq_Vv5eHA7t8SUmOH9tj9NU6i/view?usp=drivesdk", "Stable sequential node/port execution and notebook widget. It is an orchestration/UI layer, not a statistical method, scheduler, persistent cache, or exporter."),
    ("zhong2025/learning.py", "https://drive.google.com/file/d/1XSh_S8n51DOdLhoDG8GTNqaHlvVCxP09/view?usp=drivesdk", "RQ2 toolbox: SVD contrasts, contiguous cross-validation, blockwise d′, position surfaces, cross-temporal matrices, slopes/saturation, exact permutation, mouse bootstrap, area transforms, and simulations."),
    ("zhong2025/data.py", "https://drive.google.com/file/d/18Tbjmlz5LrjV2HQSDFE3tnAbcu3EKdiC/view?usp=drivesdk", "Public Figshare metadata and declared small download profiles. It deliberately does not provide an unrestricted full-neural network fallback."),
    ("zhong2025/position.py", "https://drive.google.com/file/d/1treoRl9CUCg9_GW8d4DmXr7Mfd59tsNJ/view?usp=drivesdk", "Behavior–neural frame alignment and trial × position binning without interpolation across trials."),
    ("zhong2025/atlas.py", "https://drive.google.com/file/d/1n0l5TIcP8CovPsBlV3z1XbCyUM9LWKDC/view?usp=drivesdk", "Release inventory, experiment semantics, and complete recording-bundle resolution."),
    ("zhong2025/demo.py", "https://drive.google.com/file/d/1zWuCwcIDGx9Af50AQFtknCx_-rgUezQW/view?usp=drivesdk", "Compact TX119 mechanics used for demonstrations, not group evidence."),
    ("Original pipeline", ORIGINAL_DRIVE, "Paper-provenance source, including utils.py and figure scripts. The processing notebook has large workstation-scale storage/runtime requirements and is not the team Colab path."),
]

# The floating navigator intentionally goes deeper than the compact contents
# card at the top of the document. Levels are semantic, not merely visual:
# level 1 is a numbered document section, level 2 is a useful subsection, and
# level 3 is a particularly important code recipe inside a subsection.
TOC_ITEMS = [
    ("paper", "Main paper and findings", 1),
    ("paper-findings", "Main Figures 1–5", 2),
    ("paper-figure-1", "Figure 1 + ED1 · Selectivity", 3),
    ("paper-figure-2", "Figure 2 + ED3 · Visual coding", 3),
    ("paper-figure-3", "Figure 3 + ED5–7 · Adaptation", 3),
    ("paper-figure-4", "Figure 4 + ED4, ED8 · Reward prediction", 3),
    ("paper-figure-5", "Figure 5 + ED9 · Pretraining", 3),
    ("paper-methods", "Methods · Processing and observation units", 3),
    ("experiment", "Mice, cohorts, and stages", 1),
    ("experiment-stage-schedule", "Stage schedule", 2),
    ("stimuli", "Stimuli and role mapping", 1),
    ("data", "Released data layers", 1),
    ("data-neural-file", "Inside one neural file", 2),
    ("data-frame-join", "Frame-by-frame joins", 3),
    ("data-selectivity", "Selectivity from frames", 3),
    ("coverage", "Coverage and longitudinal limits", 1),
    ("coverage-longitudinal", "What day-by-day means", 2),
    ("coverage-download", "Candidate-analysis download footprint", 2),
    ("atlas", "Complete 89-acquisition atlas", 1),
    ("atlas-experiment-labels", "All 23 experiment labels", 2),
    ("atlas-acquisitions", "All 19 mice and 89 acquisitions", 2),
    ("support", "Behavior and retinotopy coverage", 1),
    ("support-duplicates", "Why acquisitions repeat", 2),
    ("support-protocol-snapshots", "Protocol snapshots and missingness", 2),
    ("environment", "Drive, notebooks, and graph", 1),
    ("environment-drive", "Google Drive workspace", 2),
    ("environment-access", "drive.py access layer", 2),
    ("environment-graph", "graph.py analysis widget", 2),
    ("environment-drive-map", "Verified Drive map", 2),
    ("methods-review", "Large-scale analysis methods", 1),
    ("methods-review-figures", "Published review figures", 2),
    ("methods-fig1", "Science Fig. 1 · Single neurons", 3),
    ("methods-fig2", "Science Fig. 2 · Population structure", 3),
    ("methods-fig3", "Science Fig. 3 · Models and validation", 3),
    ("methods-fig4", "Science Fig. 4 · Analysis framework", 3),
    ("methods-review-contract", "Published methods and proposed extensions", 2),
    ("dprime", "Three d′ estimands", 1),
    ("dprime-trial-resolved", "Trial-resolved d′ specification", 2),
    ("within", "Research question 1", 1),
    ("within-hypotheses", "Testable distribution hypotheses", 2),
    ("within-strata", "Four requested strata", 2),
    ("reward-rate", "Research question 2", 1),
    ("reward-hypotheses", "Reward-rate hypotheses", 2),
    ("questions", "Evidence boundaries", 1),
    ("figuremap", "Figure-by-figure recipe map", 1),
    ("complete-evidence-atlas", "Published figure index", 2),
    ("analyses", "RQ1 protocol and inference", 1),
    ("analysis-primary", "1. Lock cohort and manifest", 2),
    ("analysis-alignment", "2. Align frames and trials", 2),
    ("analysis-pairing", "3. Pair trials and form windows", 2),
    ("analysis-distributions", "4. Describe every distribution", 2),
    ("analysis-graph", "5. Interactive graph contract", 2),
    ("analysis-inference", "6. Mouse-level inference", 2),
    ("analysis-validation", "7. Validation and sensitivity", 2),
    ("workflow", "End-to-end runbook", 1),
    ("workflow-order", "Execution order", 2),
    ("workflow-access", "Team data access and fallback", 2),
    ("recipes", "Released recipe reference", 1),
    ("recipe-dprime", "Core d′ recipe", 2),
    ("paper-code-walkthrough", "Paper-code walkthrough", 2),
    ("code-paper-dprime", "Exact paper d′", 3),
    ("code-paper-mask", "Stimulus roles and valid frames", 3),
    ("code-paper-splits", "Selection, ordering, and display", 3),
    ("code-paper-density", "Cortical density maps", 3),
    ("code-paper-coding-direction", "Coding direction", 3),
    ("code-paper-reward", "Reward prediction and RNG", 3),
    ("project-code-extension", "Stable project extensions", 2),
    ("plot-playbook", "Plot playbook", 2),
    ("caveats", "Constraints and interpretation", 1),
]

TOC_SECTIONS = [(target, label) for target, label, level in TOC_ITEMS if level == 1]

# These headings come from both the original reference document and generated
# sections. Applying their anchors after section generation keeps the
# navigator's deep links deterministic without coupling every content builder
# to navigation markup.
TOC_HEADING_TARGETS = {
    "paper-findings": "Main Figures 1&ndash;5",
    "experiment-stage-schedule": "Stage schedule",
    "coverage-longitudinal": "What “day by day” means in this release",
    "coverage-download": "Download footprint for the 13-mouse candidate analysis",
    "atlas-experiment-labels": "All 23 experiment labels",
    "atlas-acquisitions": "All 19 mice and 89 neural acquisitions",
    "support-duplicates": "Why some acquisitions appear more than once",
    "support-protocol-snapshots": "Protocol snapshots are not missing-file errors",
    "environment-drive": "Google Drive workspace",
    "environment-access": "<code>drive.py</code> — access layer",
    "environment-graph": "<code>graph.py</code> — analysis graph and widget",
    "environment-drive-map": "Verified Drive map",
    "dprime-trial-resolved": "Trial-resolved specification",
    "within-hypotheses": "Testable hypotheses",
    "within-strata": "The four requested strata",
    "reward-hypotheses": "Hypotheses and interpretation",
    "analysis-alignment": "2. Frame alignment and trial responses",
    "analysis-pairing": "3. Trial pairing and balanced windows",
    "analysis-distributions": "4. Across-neuron distributions",
    "analysis-graph": "5. Interactive graph",
    "analysis-inference": "6. Mouse-level inference",
    "analysis-validation": "7. Validation and sensitivity",
    "workflow-order": "Execution order",
    "workflow-access": "Team-scale data access and fallback",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _human_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value):,} B"
            return f"{value:,.3f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


@lru_cache(maxsize=None)
def _data_uri(asset_name: str) -> str:
    path = FIGURE_ASSET_DIR / asset_name
    if not path.is_file():
        raise FileNotFoundError(
            f"missing reference-figure asset {path}; run "
            "scripts/build_reference_figure_assets.py after restoring the source figures"
        )
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(f"unsupported figure format: {path.suffix}")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{payload}"


@lru_cache(maxsize=None)
def _image_size(asset_name: str) -> tuple[int, int]:
    with Image.open(FIGURE_ASSET_DIR / asset_name) as image:
        return image.size


def _nature_figure(
    *,
    target: str,
    asset_name: str,
    figure_label: str,
    source_page: int,
    title: str,
    alt: str,
    explanation: str,
    eager: bool = False,
) -> str:
    loading = "eager" if eager else "lazy"
    width, height = _image_size(asset_name)
    source_url = f"{NATURE}#Fig{source_page}" if source_page <= 5 else f"{NATURE}/figures/{source_page}"
    return f'''
    <figure id="{target}" class="paperfig evidence-figure" data-figure-target="{target}" data-figure-kind="nature">
      <img loading="{loading}" decoding="async" width="{width}" height="{height}" alt="{escape(alt)}" src="{_data_uri(asset_name)}">
      <figcaption><b>{escape(figure_label)} &mdash; {escape(title)}</b> {explanation} <a class="figure-source-inline" href="{source_url}" target="_blank" rel="noopener noreferrer">Exact figure on Nature&nbsp;&nearr;</a></figcaption>
    </figure>'''


def _figure_attribution() -> str:
    return f'''
    <p class="figure-attribution"><strong>Figure source.</strong> Figures 1&ndash;5 and Extended Data Figures 1&ndash;9 are reproduced unchanged from <a href="{NATURE_DOI}" target="_blank" rel="noopener noreferrer">Zhong et&nbsp;al., <em>Nature</em> 644, 741&ndash;748 (2025)</a> under <a href="{CC_BY}" target="_blank" rel="noopener noreferrer">CC&nbsp;BY&nbsp;4.0</a>. Every caption links to its exact figure page.</p>'''


def _nature_cite(anchor: str, label: str) -> str:
    """Link a claim to the narrowest matching anchor in the Nature article."""
    return (
        f'<a class="paper-cite" href="{NATURE}#{escape(anchor)}" target="_blank" '
        f'rel="noopener noreferrer">{label}&nbsp;&nearr;</a>'
    )


def _nature_source_for_target(target: str) -> tuple[str, str] | None:
    main = re.match(r"nature-fig([1-5])", target)
    if main:
        number = int(main.group(1))
        return f"{NATURE}#Fig{number}", f"Nature Figure {number}"
    extended = re.fullmatch(r"nature-ed([1-9])", target)
    if extended:
        number = int(extended.group(1))
        return f"{NATURE}/figures/{number + 5}", f"Nature Extended Data Figure {number}"
    methods = re.fullmatch(r"methods-fig([1-4])", target)
    if methods:
        number = int(methods.group(1))
        return f"{SCIENCE_PDF}#page={number + 2}", f"Science Figure {number}"
    return None


def _figure_ref(target: str, label: str) -> str:
    internal = f'<a class="figref-link" href="#{target}" data-figure-ref="{target}">{label}</a>'
    source = _nature_source_for_target(target)
    if source is None:
        return internal
    source_url, source_label = source
    external = (
        f'<a class="figref-source" href="{source_url}" target="_blank" rel="noopener noreferrer" '
        f'aria-label="Open {source_label} in the paper">paper&nbsp;&nearr;</a>'
    )
    return f'<span class="figref-group">{internal}{external}</span>'


def _json_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _file_cell(row: dict, *, compact: bool = False) -> str:
    name = escape(str(row["name"]))
    meta = f'{int(row["size_bytes"]):,} B · {_human_bytes(int(row["size_bytes"]))} · MD5 <code>{escape(str(row["md5"]))}</code>'
    if compact:
        return f'<a href="{escape(str(row["url"]))}" target="_blank" rel="noopener noreferrer"><code>{name}</code></a><span class="filemeta">{meta}</span>'
    return f'<a href="{escape(str(row["url"]))}" target="_blank" rel="noopener noreferrer"><code>{name}</code></a><br><span class="filemeta">{meta}</span>'


def _behavior_key(recording_id: str, source: dict) -> str:
    stimtype = source.get("stimtype")
    return f"{recording_id}_{stimtype}" if stimtype else recording_id


def _metadata_html(source: dict) -> str:
    preferred = [
        "Gender", "mname", "datexp", "blk", "sess#", "stimtype", "exptype",
        "rewType", "stim_id", "stim", "depth", "days", "is2p", "2pblk",
        "isDR", "artLick", "Note", "ROIdir",
    ]
    keys = [key for key in preferred if key in source]
    keys.extend(sorted(key for key in source if key not in keys))
    return "".join(
        f'<span><b>{escape(key)}</b>=<code>{escape(_json_value(source[key]))}</code></span>'
        for key in keys
    )


def _wrap_section(section_id: str, number: int, eyebrow: str, title: str, body: str, *, bleed: bool = False) -> str:
    width = "bleed" if bleed else "wrap"
    return f'''<section id="{section_id}">
  <div class="{width}">
    <div class="sechead"><span class="idx">{number:02d}</span><div class="eyebrow">{eyebrow}</div></div>
    <h2>{title}</h2>
{body}
    <p class="back"><a href="#contents">Back to contents ↑</a></p>
  </div>
</section>'''


def _replace_between_sections(html: str, previous_id: str, next_id: str, content: str) -> str:
    next_tag = f'<section id="{next_id}">'
    pos = html.index(next_tag)
    return html[:pos] + content.rstrip() + "\n\n" + html[pos:]


def _insert_in_section(html: str, section_id: str, marker: str, content: str) -> str:
    start_marker = f"<!-- {marker}:START -->"
    end_marker = f"<!-- {marker}:END -->"
    payload = f"{start_marker}\n{content.rstrip()}\n{end_marker}"
    if start_marker in html:
        pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), re.S)
        return pattern.sub(lambda _: payload, html, count=1)
    section_start = html.index(f'<section id="{section_id}">')
    next_section = html.find("<section id=", section_start + 1)
    if next_section < 0:
        next_section = html.index("</main>", section_start)
    closing = html.rfind("  </div>\n</section>", section_start, next_section)
    if closing < 0:
        raise ValueError(f"could not locate closing wrapper for section {section_id}")
    return html[:closing] + payload + "\n" + html[closing:]


def _legacy_result_screenshot(
    *,
    asset_name: str,
    target: str,
    source_page: int,
    panel_label: str,
    width: int,
    height: int,
    alt: str,
    explanation: str,
) -> str:
    source_url = f"{NATURE}#Fig{source_page}" if source_page <= 5 else f"{NATURE}/figures/{source_page}"
    return f'''
      <figure class="result-shot">
        <a href="#{target}" data-figure-ref="{target}" aria-label="Open the annotated {escape(panel_label)} detail in this document">
          <img loading="lazy" decoding="async" width="{width}" height="{height}" alt="{escape(alt)}" src="{_data_uri(asset_name)}">
        </a>
        <figcaption><strong>{escape(panel_label)}.</strong> {explanation} <span class="result-shot__rights">Adapted from <a href="{NATURE_DOI}" target="_blank" rel="noopener noreferrer">Zhong et&nbsp;al., <em>Nature</em> (2025)</a> under <a href="{CC_BY}" target="_blank" rel="noopener noreferrer">CC&nbsp;BY&nbsp;4.0</a>; cropped only, with scientific labels and plotted values unchanged. <a href="{source_url}" target="_blank" rel="noopener noreferrer">Exact source figure&nbsp;&nearr;</a></span></figcaption>
      </figure>'''


def _legacy_paper_enrichment() -> str:
    return f'''
    <h3>What the complete paper establishes</h3>
    <p class="result-guide__intro">Each numbered figure is a multi-panel argument, not a single result. The cards below keep the decisive subfigures beside the explanation, identify what every panel group measures, and distinguish the authors&rsquo; published conclusion from the new analysis that this reference proposes. Select any screenshot to jump to its larger annotated rendering elsewhere in the document.</p>
    <div class="result-guide" aria-label="Panel-by-panel guide to the paper's main figures">
      <article id="paper-figure-1" class="result-card">
        <header class="result-card__head"><span class="result-card__index">01</span><div><p class="result-card__eyebrow">Nature Figure 1 · panels a–j</p><h4>Plasticity after supervised and unsupervised training</h4><p>The figure moves from task design and behavioral verification to the cellular selectivity definition, then to cortical maps and regional group statistics.</p></div></header>
        <div class="result-card__screens">
{_result_screenshot(asset_name="nature-main-1-panels-a-b.jpg", target="nature-fig1ab", source_page=1, panel_label="Figure 1a–b", width=1200, height=176, alt="Nature Figure 1 panels a and b showing the virtual-reality corridors, cue and reward positions, experimental cohorts, and protocol stages", explanation="The corridor schematic and training timeline define what rewarded task learning, unrewarded natural-texture exposure, and grating exposure actually mean.")}
{_result_screenshot(asset_name="nature-main-1-panels-i-j.jpg", target="nature-fig1ij", source_page=1, panel_label="Figure 1i–j", width=750, height=427, alt="Nature Figure 1 panels i and j showing selective-neuron density maps and regional before-after fractions", explanation="The endpoint evidence: cortical density maps are paired with mouse-level regional fractions before and after learning or exposure.")}
        </div>
        <div class="result-card__body">
          <dl class="result-panel-list">
            <div><dt>{_nature_cite("Fig1", "Figure&nbsp;1a–b")} · design</dt><dd>Panel a defines the circle1 and leaf1 corridors, randomized sound cue, and reward availability. Panel b separates task, unrewarded natural-texture, and unrewarded grating cohorts across Train&nbsp;1/Test&nbsp;1/Train&nbsp;2/Test&nbsp;2/Test&nbsp;3 landmarks.</dd></div>
            <div><dt>{_nature_cite("Fig1", "Figure&nbsp;1c–d")} · behavior</dt><dd>Example lick rasters and anticipatory-lick summaries verify that the task cohort learned the rewarded corridor; these are behavioral checks, not the neural effect itself.</dd></div>
            <div><dt>{_nature_cite("Fig1", "Figure&nbsp;1e–f")} · measurement</dt><dd>Panel e establishes mesoscope coverage and cellular resolution. Panel f defines signed leaf1-versus-circle1 d′ from response distributions inside the two corridors.</dd></div>
            <div><dt>{_nature_cite("Fig1", "Figure&nbsp;1g–h")} · response structure</dt><dd>Single-neuron trials and sorted population sequences show the positive leaf1 and negative circle1 selectivity poles rather than collapsing them into |d′|.</dd></div>
            <div><dt>{_nature_cite("Fig1", "Figure&nbsp;1i–j")} · endpoint</dt><dd>Aligned cortical density maps and regional selective-neuron fractions compare before versus after across visual areas and cohorts; the mouse/session summaries, not the thousands of neurons, carry group replication.</dd></div>
          </dl>
          <aside class="result-card__takeaway"><p class="result-card__kicker">Published result</p><p>Familiar-texture selectivity increased most clearly in medial HVAs after rewarded training <em>and</em> unrewarded natural-texture exposure; grating exposure did not reproduce that pattern. V1 and lateral changes were smaller, while anterior modulation was more task-specific ({_nature_cite("Sec2", "Results: supervised and unsupervised plasticity")}; {_nature_cite("Fig1", "Figure&nbsp;1i–j")}).</p><p class="result-card__kicker">Proposed decision in this reference</p><p>Use medial HVA as RQ1&rsquo;s primary region, keep the other regions as declared contrasts, retain both signed tails, and make mice—not neurons—the inferential units.</p><p class="result-card__boundary"><strong>Boundary:</strong> Figure&nbsp;1 compares whole-session before/after endpoints ({_nature_cite("Fig1", "Figure&nbsp;1i–j")}); it does not report chronological trial-window trajectories.</p></aside>
        </div>
      </article>

      <article id="paper-figure-2" class="result-card">
        <header class="result-card__head"><span class="result-card__index">02</span><div><p class="result-card__eyebrow">Nature Figure 2 · panels a–j</p><h4>Visual identity is retained across corridor position</h4><p>The figure tests whether neural sequences encode the visual stimulus itself, rather than merely repeating a position-locked trajectory.</p></div></header>
        <div class="result-card__screens">
{_result_screenshot(asset_name="nature-main-2-panels-g-j.jpg", target="nature-fig2gj", source_page=2, panel_label="Figure 2g–j", width=1200, height=332, alt="Nature Figure 2 panels g through j showing trial-resolved selective-neuron sequences, coding-direction projections, and regional similarity indices", explanation="Trial-by-position population maps lead into a held-out coding direction and a similarity index for new stimuli.")}
        </div>
        <div class="result-card__body">
          <dl class="result-panel-list">
            <div><dt>{_nature_cite("Fig2", "Figure&nbsp;2a–c")} · stimulus and behavior</dt><dd>Four learned/new corridors are presented in Test&nbsp;1; lick rasters and anticipatory licking establish which stimulus identities the mice behaviorally distinguish.</dd></div>
            <div><dt>{_nature_cite("Fig2", "Figure&nbsp;2d–f")} · sequence reliability</dt><dd>Medial-area responses are sorted by leaf1 preferred position on held-out trials. Preferred-position correlations are then summarized within task mice and across regions and cohorts, including leaf2 versus leaf1.</dd></div>
            <div><dt>{_nature_cite("Fig2", "Figure&nbsp;2g–h")} · trial-resolved populations</dt><dd>Leaf1- and circle1-selective neurons are shown as a population average for each trial, preserving both selectivity poles and the trial-by-position response structure.</dd></div>
            <div><dt>{_nature_cite("Fig2", "Figure&nbsp;2i–j")} · geometry</dt><dd>A coding direction is defined from the leaf1- and circle1-selective populations. Held-out projections yield a similarity index for new stimuli, summarized by region and cohort.</dd></div>
          </dl>
          <aside class="result-card__takeaway"><p class="result-card__kicker">Published result</p><p>Sequence responses followed visual identity more strongly than absolute corridor position, and leaf2 responses generalized along the familiar leaf1–circle1 coding direction ({_nature_cite("Sec3", "Results: visual, not spatial, representations")}; {_nature_cite("Fig2", "Figure&nbsp;2d–j")}).</p><p class="result-card__kicker">Proposed decision in this reference</p><p>Map every released wall label to its stimulus role before analysis; learn selection and ordering on data independent of the trials displayed or scored.</p><p class="result-card__boundary"><strong>Boundary:</strong> Figure&nbsp;2 tests within-session held-out generalization ({_nature_cite("Sec20", "Methods: neural selectivity splits")}); it does not provide cross-date cell identities.</p></aside>
        </div>
      </article>

      <article id="paper-figure-3" class="result-card">
        <header class="result-card__head"><span class="result-card__index">03</span><div><p class="result-card__eyebrow">Nature Figure 3 · panels a–h</p><h4>Fine discrimination reshapes representational geometry</h4><p>The figure asks what happens when mice must distinguish the familiar leaf1 corridor from the visually similar leaf2 corridor.</p></div></header>
        <div class="result-card__screens">
{_result_screenshot(asset_name="nature-main-3-panels-f-h.jpg", target="nature-fig3fh", source_page=3, panel_label="Figure 3f–h", width=815, height=528, alt="Nature Figure 3 panels f through h showing V1 and medial-HVA coding projections and similarity after fine discrimination", explanation="V1 and medial-area projections culminate in the regional similarity summary and the orthogonalization schematic.")}
        </div>
        <div class="result-card__body">
          <dl class="result-panel-list">
            <div><dt>{_nature_cite("Fig3", "Figure&nbsp;3a–b")} · leaf2 selectivity</dt><dd>Maps and regional summaries compare neurons selective for leaf2 versus circle1 when leaf2 is new and after learning, alongside unrewarded and naive cohorts.</dd></div>
            <div><dt>{_nature_cite("Fig3", "Figure&nbsp;3c")} · behavioral learning</dt><dd>Licking to leaf2 before and after training establishes the behavioral fine-discrimination transition.</dd></div>
            <div><dt>{_nature_cite("Fig3", "Figure&nbsp;3d–e")} · leaf1 versus leaf2</dt><dd>Signed selective-neuron distributions and regional fractions directly compare the two similar natural textures across task, unrewarded, naive, and grating-control observations.</dd></div>
            <div><dt>{_nature_cite("Fig3", "Figure&nbsp;3f–g")} · coding projections</dt><dd>Leaf2 population responses are projected onto the familiar leaf1–circle1 coding direction in V1 and medial HVA.</dd></div>
            <div><dt>{_nature_cite("Fig3", "Figure&nbsp;3h")} · orthogonalization</dt><dd>The similarity index summarizes how much leaf2 remains aligned with the familiar axis; the schematic expresses the observed rotation away from leaf1.</dd></div>
          </dl>
          <aside class="result-card__takeaway"><p class="result-card__kicker">Published result</p><p>Fine-discrimination training made the leaf2 representation less leaf1-like, with the strongest orthogonalization in medial HVAs ({_nature_cite("Sec4", "Results: novelty and orthogonalization")}; {_nature_cite("Fig3", "Figure&nbsp;3f–h")}).</p><p class="result-card__kicker">Proposed decision in this reference</p><p>Analyze transfer to similar exemplars only after the simpler familiar leaf1–circle1 distribution and held-out coding axis are locked.</p><p class="result-card__boundary"><strong>Boundary:</strong> The paper defines the similarity index as a population projection ({_nature_cite("Sec21", "Methods: coding direction and similarity index")}); it is not a moment of the across-neuron d′ distribution.</p></aside>
        </div>
      </article>

      <article id="paper-figure-4" class="result-card">
        <header class="result-card__head"><span class="result-card__index">04</span><div><p class="result-card__eyebrow">Nature Figure 4 · panels a–n + Extended Data 8</p><h4>A separate anterior reward-prediction signal</h4><p>This figure defines a late-cue-versus-early-cue estimand, localizes it, and then tests cue, lick, corridor, and movement interpretations.</p></div></header>
        <div class="result-card__screens result-card__screens--three">
{_result_screenshot(asset_name="nature-main-4-panels-e-g.jpg", target="nature-fig4", source_page=4, panel_label="Figure 4e–g", width=1200, height=335, alt="Nature Figure 4 panels e through g showing the late-versus-early cue d-prime definition, density maps, and anterior-region fractions", explanation="A cue-duration discrimination index is defined first, then localized with cortical maps and regional fractions.")}
{_result_screenshot(asset_name="nature-main-4-panels-i-l.jpg", target="nature-fig4il", source_page=4, panel_label="Figure 4i–l", width=705, height=304, alt="Nature Figure 4 panels i through l showing cue-aligned, first-lick-aligned, lick, and no-lick neural controls", explanation="Cue alignment, first-lick alignment, and lick/no-lick trials constrain a purely motor account of the signal.")}
{_result_screenshot(asset_name="nature-ed-8-panels-d-f.jpg", target="nature-ed8df", source_page=13, panel_label="Extended Data 8d–f", width=1260, height=581, alt="Extended Data Figure 8 panels d through f showing cue-aligned neural activity alongside running speed and licking rate", explanation="Neural activity is displayed beside running and licking across all four corridors so state covariates remain visible.")}
        </div>
        <div class="result-card__body">
          <dl class="result-panel-list">
            <div><dt>{_nature_cite("Fig4", "Figure&nbsp;4a–c")} · discovery and location</dt><dd>Rastermap exposes a task-linked population sequence, a selected segment is enlarged, and the selected cells are mapped back onto cortex.</dd></div>
            <div><dt>{_nature_cite("Fig4", "Figure&nbsp;4d–g")} · estimand and regional effect</dt><dd>Trial averages sorted by cue position motivate a late-versus-early cue d′. Density maps and fractions then show the strongest task-specific increase in anterior HVA.</dd></div>
            <div><dt>{_nature_cite("Fig4", "Figure&nbsp;4h–j")} · stimulus and timing</dt><dd>Responses across Test&nbsp;1 corridors are followed by cue-aligned and first-lick-aligned averages, separating cue timing from simple lick timing.</dd></div>
            <div><dt>{_nature_cite("Fig4", "Figure&nbsp;4k–l")} · lick controls</dt><dd>Reward-prediction and medial leaf-selective populations are compared on trials with versus without licks.</dd></div>
            <div><dt>{_nature_cite("Fig4", "Figure&nbsp;4m–n")} · transfer tests</dt><dd>Population responses are repeated in Test&nbsp;2 and Test&nbsp;3 to test how the signal behaves under later stimulus arrangements.</dd></div>
            <div><dt><a class="paper-cite" href="{NATURE}/figures/13" target="_blank" rel="noopener noreferrer">Extended Data Figure&nbsp;8&nbsp;&nearr;</a> · specificity</dt><dd>Panel a compares reward-prediction fractions across regions; b–c define non-reward prediction controls; d–f place neural traces beside corridor-specific running and licking.</dd></div>
          </dl>
          <aside class="result-card__takeaway"><p class="result-card__kicker">Published result</p><p>A late-cue/value-related response increased most clearly in anterior HVAs of task mice. The paper evaluates cue timing, first-lick timing, lick/no-lick trials, non-reward prediction, regions, running, and licking ({_nature_cite("Sec6", "Results: reward prediction in anterior HVAs")}; {_nature_cite("Fig4", "Figure&nbsp;4d–l")}; <a class="paper-cite" href="{NATURE}/figures/13" target="_blank" rel="noopener noreferrer">Extended Data Figure&nbsp;8&nbsp;&nearr;</a>).</p><p class="result-card__kicker">Proposed decision in this reference</p><p>Use the anterior late-versus-early cue statistic as RQ2&rsquo;s separate positive control, not as the sensory endpoint.</p><p class="result-card__boundary"><strong>Boundary:</strong> The paper&rsquo;s reward-prediction estimator contrasts late versus early cue trials ({_nature_cite("Sec22", "Methods: reward-prediction neurons")}); it is neither leaf1-versus-circle1 sensory d′ nor a chronological learning-rate estimate.</p></aside>
        </div>
      </article>

      <article id="paper-figure-5" class="result-card">
        <header class="result-card__head"><span class="result-card__index">05</span><div><p class="result-card__eyebrow">Nature Figure 5 · panels a–h + Extended Data 9</p><h4>Natural-texture pretraining accelerates later behavior</h4><p>A separate behavior-only experiment compares natural-texture, grating, and no-pretraining cohorts across five rewarded training days.</p></div></header>
        <div class="result-card__screens result-card__screens--three">
{_result_screenshot(asset_name="nature-main-5-panels-a-b.jpg", target="nature-fig5", source_page=5, panel_label="Figure 5a–b", width=500, height=225, alt="Nature Figure 5 panels a and b showing the three pretraining cohorts and deterministic rewarded-task structure", explanation="The cohort schedule and altered task structure must be read before comparing this experiment with the imaging protocol.")}
{_result_screenshot(asset_name="nature-main-5-panels-e-h.jpg", target="nature-fig5", source_page=5, panel_label="Figure 5e–h", width=710, height=641, alt="Nature Figure 5 panels e through h showing lick learning curves, performance differences, first-lick locations, and trial counts", explanation="Daily learning curves are accompanied by spatial first-lick distributions and trial-count support, not just a terminal performance bar.")}
{_result_screenshot(asset_name="nature-ed-9-days-1-3.jpg", target="nature-ed9", source_page=14, panel_label="Extended Data 9 · days 1–3", width=1344, height=720, alt="Extended Data Figure 9 crop showing first-half and second-half behavioral performance for the three cohorts over early training days", explanation="First-half versus second-half summaries reveal within-day change during the early part of training.")}
        </div>
        <div class="result-card__body">
          <dl class="result-panel-list">
            <div><dt>{_nature_cite("Fig5", "Figure&nbsp;5a–b")} · independent design</dt><dd>Three new cohorts receive natural-texture, grating, or no unrewarded pretraining, then five rewarded task days. Rewards are deterministic in the second corridor half and the sound cue is absent, unlike the imaging task.</dd></div>
            <div><dt>{_nature_cite("Fig5", "Figure&nbsp;5c–d")} · example behavior</dt><dd>Lick rasters show representative mice on the first active-reward day and the last training day.</dd></div>
            <div><dt>{_nature_cite("Fig5", "Figure&nbsp;5e–f")} · learning trajectory</dt><dd>Mean lick responses and the rewarded-minus-non-rewarded performance difference are tracked across days for all three cohorts.</dd></div>
            <div><dt>{_nature_cite("Fig5", "Figure&nbsp;5g–h")} · spatial and support checks</dt><dd>First-lick locations test where behavioral anticipation emerges; daily trial counts show the observation support behind each curve.</dd></div>
            <div><dt><a class="paper-cite" href="{NATURE}/figures/14" target="_blank" rel="noopener noreferrer">Extended Data Figure&nbsp;9&nbsp;&nearr;</a> · within day</dt><dd>Each day is split into first and second halves, making the early within-session behavioral improvement visible rather than inferred from day averages alone.</dd></div>
          </dl>
          <aside class="result-card__takeaway"><p class="result-card__kicker">Published result</p><p>In 23 additional mice, natural-texture pretraining accelerated later rewarded-task learning relative to grating or no pretraining; the paper&rsquo;s within-day split shows part of the gain emerging inside sessions ({_nature_cite("Sec7", "Results: faster task learning after pretraining")}; {_nature_cite("Fig5", "Figure&nbsp;5e–h")}; <a class="paper-cite" href="{NATURE}/figures/14" target="_blank" rel="noopener noreferrer">Extended Data Figure&nbsp;9&nbsp;&nearr;</a>).</p><p class="result-card__kicker">Proposed decision in this reference</p><p>Use this behavioral result to motivate—not claim—a neural acceleration analysis in the released imaging cohorts.</p><p class="result-card__boundary"><strong>Boundary:</strong> The Animals Methods specify that the 23 behavior-only mice had headbars but no cranial windows ({_nature_cite("Sec11", "Methods: animals")}); Figure&nbsp;5 therefore contains no neural measurements, and its task protocol differs from the imaging task ({_nature_cite("Sec17", "Methods: behavioural training")}).</p></aside>
        </div>
      </article>

      <article id="paper-methods" class="result-card result-card--methods">
        <header class="result-card__head"><span class="result-card__index">M</span><div><p class="result-card__eyebrow">Nature Methods · acquisition and processing</p><h4>What the plotted neural values actually are</h4><p>The figures use processed cellular activity aligned to behavior and visual areas—not raw movies and not a continuous identity-tracked recording across days.</p></div></header>
        <div class="result-card__screens">
{_result_screenshot(asset_name="nature-main-1-panel-f.jpg", target="nature-fig1f", source_page=1, panel_label="Figure 1f", width=305, height=190, alt="Nature Figure 1 panel f showing the signed d-prime formula and the two frame-response distributions", explanation="The displayed formula uses the difference of mean responses divided by the arithmetic mean of their standard deviations.")}
        </div>
        <div class="result-card__body">
          <dl class="result-panel-list">
            <div><dt>Signal</dt><dd>Suite2p-derived deconvolved activity with a 0.75&nbsp;s decay parameter is the paper&rsquo;s neural layer ({_nature_cite("Sec19", "Methods: processing of calcium imaging data")}). The shared Figshare files are processed arrays, not raw fluorescence movies (<a class="paper-cite" href="{NATURE}#data-availability" target="_blank" rel="noopener noreferrer">Data availability&nbsp;&nearr;</a>; <a href="{FIGSHARE}" target="_blank" rel="noopener noreferrer">Figshare v2 inventory&nbsp;&nearr;</a>).</dd></div>
            <div><dt>Valid frames</dt><dd>The paper&rsquo;s selectivity estimator retains original, non-interpolated running frames inside the 0–4&nbsp;m textured corridor ({_nature_cite("Sec20", "Methods: neural selectivity")}). Cue, reward, and lick timing come from the imaging behavioral protocol ({_nature_cite("Sec17", "Methods: behavioural training")}); cortical regions come from retinotopic assignment ({_nature_cite("Sec25", "Methods: retinotopy")}).</dd></div>
            <div><dt>Scale</dt><dd>The paper reports 20,547–89,577 Suite2p traces per recording ({_nature_cite("Sec2", "Results: supervised and unsupervised plasticity")}) and 89 recordings in 19 mice ({_nature_cite("Sec11", "Methods: animals")}). Those are separate descriptive and experimental sample sizes.</dd></div>
            <div><dt>Published observation unit</dt><dd>Figure&nbsp;1f forms d′ from frame-response distributions ({_nature_cite("Fig1", "Figure&nbsp;1f")}; {_nature_cite("Sec20", "Methods: neural selectivity")}). The proposed trial-window analysis preserves the signed algebra but replaces pooled frames with balanced per-trial summaries.</dd></div>
          </dl>
          <aside class="result-card__takeaway"><p class="result-card__kicker">Proposed reproduction contract</p><p>First reproduce the paper&rsquo;s whole-session signed d′, |d′|≥0.3 threshold, cortical density calculation, and regional fractions exactly as specified ({_nature_cite("Sec20", "Methods: neural selectivity")}; {_nature_cite("Fig1", "Figure&nbsp;1f,i–j")}). Only then introduce trial pairing, fixed windows, distribution moments, or population axes.</p><p class="result-card__kicker">Exact paper locations</p><p>{_nature_cite("Sec13", "Imaging acquisition")} · {_nature_cite("Sec19", "Calcium processing")} · {_nature_cite("Sec20", "Neural selectivity")} · {_nature_cite("Sec24", "Statistics and reproducibility")}</p><p class="result-card__boundary"><strong>Boundary:</strong> The paper reports group statistics by mouse/session ({_nature_cite("Sec24", "Statistics and reproducibility")}); the release audit finds no cross-date cell-identity map (<a href="#coverage-longitudinal">release-derived longitudinal audit&nbsp;↓</a>).</p></aside>
        </div>
      </article>
    </div>
    <div class="note angle"><span class="k">Published versus proposed</span><p><strong>Published:</strong> whole-session familiar-stimulus selectivity and regional fractions ({_nature_cite("Sec2", "Results section")}; {_nature_cite("Fig1", "Figure&nbsp;1f,i–j")}), the anterior late-versus-early cue signal ({_nature_cite("Sec6", "Results section")}; {_nature_cite("Fig4", "Figure&nbsp;4")}), and behavior-only within-day learning (<a class="paper-cite" href="{NATURE}/figures/14" target="_blank" rel="noopener noreferrer">Extended Data Figure&nbsp;9&nbsp;&nearr;</a>). <strong>Proposed here:</strong> full-distribution trial trajectories and reward-associated neural acceleration. The <a href="{PAPER_DRIVE}" target="_blank" rel="noopener noreferrer">Drive PDF</a> is a fixed copy; the links above identify the exact public paper locations.</p></div>
'''


def _legacy_extended_figure_atlas() -> str:
    figures = (
        (
            1,
            "Retinotopy, threshold robustness, and signed selectivity poles",
            "Retinotopic area assignment, the cortical field-of-view transform, the full |d′| threshold sweep, and separate positive/negative poles. Use it to audit area masks and to prevent a single 0.3 threshold from carrying the whole RQ1 claim.",
            "Extended Data Figure 1: retinotopy, d-prime threshold sensitivity, and signed selectivity poles",
        ),
        (
            2,
            "Running behaviour before and after learning",
            "Position-resolved running speed and summary comparisons for task and unrewarded mice. This is the paper's principal locomotor control and the template for trial-block speed/occupancy QC.",
            "Extended Data Figure 2: running behaviour and speed controls before and after learning",
        ),
        (
            3,
            "Circle-selective sequences and coding-direction controls",
            "The complementary circle1-selective population sequences, their sorting logic, and coding-direction read-outs. Together with main Fig. 2 this establishes that both selectivity poles must remain signed.",
            "Extended Data Figure 3: circle-selective sequences and coding-direction analyses",
        ),
        (
            4,
            "Licking controls for sequence responses",
            "Trial rasters and population responses split around licking-related events. These panels constrain a simple motor explanation without turning licking into a nuisance variable that can always be regressed away.",
            "Extended Data Figure 4: licking controls for visually driven sequence responses",
        ),
        (
            5,
            "Familiar versus novel stimulus responses",
            "Responses to learned and new exemplars, including controls for novelty and familiarity. This is the closest paper precedent for asking whether distribution change generalizes beyond leaf1 versus circle1.",
            "Extended Data Figure 5: familiar and novel stimulus response controls",
        ),
        (
            6,
            "Exemplar-specific recognition, orthogonalization, and swaps",
            "Coding-direction and similarity analyses for leaf exemplars and spatial rearrangements. These panels motivate the stronger transfer question: does the learned axis generalize when identity or spatial arrangement changes?",
            "Extended Data Figure 6: exemplar-specific representation, orthogonalization, and swap analyses",
        ),
        (
            7,
            "Projection stability, Test 3, and lick-related controls",
            "Additional coding projections, similarity indices, and swap/test controls. Use these as sensitivity analyses after the simpler held-out leaf1–circle1 trajectory is locked.",
            "Extended Data Figure 7: projection stability, Test 3, and control analyses",
        ),
        (
            8,
            "Reward versus non-reward prediction and cue-aligned behaviour",
            "Reward-prediction specificity, cue-position effects, running, and licking. This is essential context for RQ2 because the anterior value/timing signal is distinct from sensory leaf–circle d′.",
            "Extended Data Figure 8: reward-prediction specificity and cue, running, and licking controls",
        ),
        (
            9,
            "Within-day learning in the separate behaviour-only cohort",
            "First-half versus second-half behavioural learning and daily task progression in 23 additional mice. It motivates within-session analysis but cannot fill neural dates because no imaging was collected in this cohort.",
            "Extended Data Figure 9: within-day behavioural learning in the behaviour-only cohorts",
        ),
    )
    main_links = " &middot; ".join(
        _figure_ref(f"nature-fig{number}", f"Fig.&nbsp;{number}") for number in range(1, 6)
    )
    extended_links = " &middot; ".join(
        _figure_ref(f"nature-ed{number}", f"ED&nbsp;{number}") for number in range(1, 10)
    )
    rendered = []
    for number, title, explanation, alt in figures:
        rendered.append(
            _nature_figure(
                target=f"nature-ed{number}",
                asset_name=f"nature-ed-{number}.jpg",
                figure_label=f"Extended Data Figure {number}",
                source_page=number + 5,
                title=title,
                alt=alt,
                explanation=explanation,
            )
        )
    return f'''
    <div class="figure-atlas-intro" id="complete-evidence-atlas">
      <h3>Complete inline evidence atlas</h3>
      <p>Every numbered figure from the Nature paper now appears inside this document. Main figures are embedded where their logic is first used; the full Extended Data sequence is embedded below. Internal links keep the reader in context, while each caption retains the DOI, licence, and exact Nature source page.</p>
      <p class="figure-index"><strong>Main:</strong> {main_links}<br><strong>Extended Data:</strong> {extended_links}</p>
    </div>
{''.join(rendered)}
'''


def _legacy_experiment_figure_repeat() -> str:
    return _nature_figure(
        target="nature-fig1ab",
        asset_name="nature-main-1-panels-a-b.jpg",
        figure_label="Figure 1a–b detail",
        source_page=1,
        title="corridor design, cohort logic, and protocol timeline",
        alt="Cropped Nature Figure 1 panels a and b showing the two corridors, cue and reward, cohorts, and experiment timeline",
        explanation="Read this before interpreting a recording label: Train 1, Test 1, Train 2, Test 2, and Test 3 are sampled protocol stages, not consecutive daily neural recordings.",
        panels="Figure 1 panels a–b",
        eager=True,
    )


def _legacy_dprime_figure_repeat() -> str:
    return _nature_figure(
        target="nature-fig1f",
        asset_name="nature-main-1-panel-f.jpg",
        figure_label="Figure 1f detail",
        source_page=1,
        title="the paper-specific d′ definition",
        alt="Cropped Nature Figure 1 panel f showing the d-prime formula and two response distributions",
        explanation="The denominator is the mean of the two response SDs. The paper's x-axis distributions are frame responses; the project's trial-window estimator preserves the algebra but changes the observation unit to balanced trial summaries.",
        panels="Figure 1 panel f",
        eager=True,
    )


def _legacy_within_figure_repeats() -> str:
    return "\n".join((
        _nature_figure(
            target="nature-fig1ij",
            asset_name="nature-main-1-panels-i-j.jpg",
            figure_label="Figure 1i–j detail",
            source_page=1,
            title="regional endpoint that motivates RQ1",
            alt="Cropped Nature Figure 1 panels i and j showing cortical selective-neuron density and area fractions before and after learning",
            explanation="Medial HVAs show the clearest published before→after increase in selective-neuron density and fraction for both rewarded training and unrewarded exposure. This is the endpoint anchor, not evidence for a trial-by-trial distribution trajectory.",
            panels="Figure 1 panels i–j",
        ),
        _nature_figure(
            target="nature-ed1e",
            asset_name="nature-ed-1-panel-e.jpg",
            figure_label="Extended Data Figure 1e detail",
            source_page=6,
            title="threshold sweep",
            alt="Cropped Extended Data Figure 1 panel e showing selective-neuron fractions across d-prime thresholds by area and cohort",
            explanation="The 0.3 cutoff is highlighted, but the full curve is the appropriate sensitivity analysis. A distribution claim should not depend on one threshold.",
            panels="Extended Data Figure 1 panel e",
        ),
        _nature_figure(
            target="nature-ed1f",
            asset_name="nature-ed-1-panel-f.jpg",
            figure_label="Extended Data Figure 1f detail",
            source_page=6,
            title="signed leaf and circle poles",
            alt="Cropped Extended Data Figure 1 panel f showing leaf1-selective and circle1-selective neuron fractions before and after learning",
            explanation="Both signed poles can grow. That pattern can increase spread while leaving skewness near zero, which is why RQ1 must retain signed d′ and report both tails separately.",
            panels="Extended Data Figure 1 panel f",
        ),
    ))


def _legacy_reward_figure_repeats() -> str:
    return "\n".join((
        _nature_figure(
            target="nature-fig4fg",
            asset_name="nature-main-4-panels-f-g.jpg",
            figure_label="Figure 4f–g detail",
            source_page=4,
            title="anterior reward-prediction localization",
            alt="Cropped Nature Figure 4 panels f and g showing cortical density and before-after fraction of reward-prediction neurons",
            explanation="The task-specific signal localizes most clearly to anterior HVAs. It is a positive control for RQ2, not a substitute for sensory leaf–circle discriminability in medial HVA.",
            panels="Figure 4 panels f–g",
        ),
        _nature_figure(
            target="nature-fig4il",
            asset_name="nature-main-4-panels-i-l.jpg",
            figure_label="Figure 4i–l detail",
            source_page=4,
            title="cue, first-lick, and lick/no-lick controls",
            alt="Cropped Nature Figure 4 panels i through l showing cue-aligned activity, first-lick aligned activity, and lick versus no-lick comparisons",
            explanation="These panels constrain event-timing and licking interpretations. They do not prove that reward, motivation, running, or other state variables are absent from a within-session slope.",
            panels="Figure 4 panels i–l",
        ),
        _nature_figure(
            target="nature-ed8a",
            asset_name="nature-ed-8-panel-a.jpg",
            figure_label="Extended Data Figure 8a detail",
            source_page=13,
            title="reward-prediction fraction across regions",
            alt="Cropped Extended Data Figure 8 panel a showing late-versus-early cue reward-prediction fractions by area and cohort",
            explanation="The effect is an area-specific late-versus-early cue/value fraction, not direct evidence that the complete sensory d′ distribution changed its variance, skewness, or kurtosis.",
            panels="Extended Data Figure 8 panel a",
        ),
        _nature_figure(
            target="nature-ed8df",
            asset_name="nature-ed-8-panels-d-f.jpg",
            figure_label="Extended Data Figure 8d–f detail",
            source_page=13,
            title="neural activity beside running and licking",
            alt="Cropped Extended Data Figure 8 panels d through f showing cue-aligned reward-prediction activity, running, and licking",
            explanation="Read the neural trace together with the motor traces. The three panels are intentionally kept together so the apparent signal is never shown without its state controls.",
            panels="Extended Data Figure 8 panels d–f",
        ),
    ))


def _legacy_question_figure_repeats() -> str:
    return "\n".join((
        _nature_figure(
            target="nature-fig2gj",
            asset_name="nature-main-2-panels-g-j.jpg",
            figure_label="Figure 2g–j detail",
            source_page=2,
            title="sequence populations, coding direction, and similarity",
            alt="Cropped Nature Figure 2 panels g through j showing sorted sequences, coding-direction projections, and similarity indices",
            explanation="These panels motivate a stronger population question beyond single-neuron moments: does a held-out familiar-stimulus axis generalize to novel exemplars and remain stable across trial blocks?",
            panels="Figure 2 panels g–j",
        ),
        _nature_figure(
            target="nature-fig3fh",
            asset_name="nature-main-3-panels-f-h.jpg",
            figure_label="Figure 3f–h detail",
            source_page=3,
            title="representation geometry and orthogonalization",
            alt="Cropped Nature Figure 3 panels f through h showing population projections and similarity changes after fine discrimination",
            explanation="Fine discrimination changes the geometry of similar leaf representations. This is the paper-aligned next question once the simpler familiar leaf1–circle1 distribution is understood.",
            panels="Figure 3 panels f–h",
        ),
    ))


def _legacy_analysis_qc_figure_repeats() -> str:
    return "\n".join((
        _nature_figure(
            target="nature-ed1c",
            asset_name="nature-ed-1-panel-c.jpg",
            figure_label="Extended Data Figure 1c detail",
            source_page=6,
            title="retinotopy map and area boundaries",
            alt="Cropped Extended Data Figure 1 panel c showing horizontal and vertical retinotopy maps with visual-area boundaries",
            explanation="This is the visual check behind the released retinotopy masks. Area assignment is recording-specific and must be joined to the same acquisition as the neural data.",
            panels="Extended Data Figure 1 panel c",
        ),
        _nature_figure(
            target="nature-ed2c",
            asset_name="nature-ed-2-panel-c.jpg",
            figure_label="Extended Data Figure 2c detail",
            source_page=7,
            title="task-mouse running control",
            alt="Cropped Extended Data Figure 2 panel c showing task-mouse running speed before and after learning",
            explanation="Use the same logic at the trial-block level: show speed and position support beside the neural trajectory, with the mouse as the paired unit.",
            panels="Extended Data Figure 2 panel c",
        ),
        _nature_figure(
            target="nature-ed2d",
            asset_name="nature-ed-2-panel-d.jpg",
            figure_label="Extended Data Figure 2d detail",
            source_page=7,
            title="unrewarded-mouse running control",
            alt="Cropped Extended Data Figure 2 panel d showing unrewarded-mouse running speed before and after learning",
            explanation="The same QC is required in the unrewarded cohort so a group difference in neural slope is not merely a difference in locomotor support.",
            panels="Extended Data Figure 2 panel d",
        ),
    ))


def _extended_figure(number: int, title: str, explanation: str, alt: str) -> str:
    return _nature_figure(
        target=f"nature-ed{number}",
        asset_name=f"nature-ed-{number}.jpg",
        figure_label=f"Extended Data Figure {number}",
        source_page=number + 5,
        title=title,
        alt=alt,
        explanation=explanation,
    )


def _main_figure_card(
    *,
    number: int,
    title: str,
    summary: str,
    alt: str,
    panel_rows: str,
    finding: str,
    supporting_figures: str = "",
) -> str:
    return f'''
      <article id="paper-figure-{number}" class="result-card">
        <header class="result-card__head"><span class="result-card__index">{number:02d}</span><div><p class="result-card__eyebrow">Nature Figure {number}</p><h4>{escape(title)}</h4></div></header>
        {_nature_figure(
            target=f"nature-fig{number}",
            asset_name=f"nature-main-{number}.png",
            figure_label=f"Figure {number}",
            source_page=number,
            title=title,
            alt=alt,
            explanation=summary,
            eager=number == 1,
        )}
        {supporting_figures}
        <div class="result-card__body">
          <dl class="result-panel-list">{panel_rows}</dl>
          <aside class="result-card__takeaway"><p class="result-card__kicker">What the paper reports</p><p>{finding}</p></aside>
        </div>
      </article>'''


def _paper_enrichment() -> str:
    ed1 = _extended_figure(
        1,
        "Retinotopy and neural changes after learning for different populations.",
        "Panels a–g show example stimulus crops, retinotopic mapping and atlas alignment, the grating-cohort selective-neuron distribution, threshold sweeps, separate leaf1- and circle1-selective fractions, and medial-versus-lateral V1 summaries.",
        "Extended Data Figure 1 showing stimulus crops, retinotopy, threshold sweeps, signed selective-neuron fractions, and V1 subdivisions",
    )
    ed2 = _extended_figure(
        2,
        "Running behaviors.",
        "Panels a–d show position-resolved running speed in example unrewarded and task mice, before-versus-after speed summaries, and percentage of time running.",
        "Extended Data Figure 2 showing running speed and percentage of time running before and after learning",
    )
    ed3 = _extended_figure(
        3,
        "Sequences in circle1-preferring neurons and average projections on the coding direction.",
        "Panels a–e show circle1-selective sequence analyses and coding-direction projections for naive, unrewarded and task cohorts.",
        "Extended Data Figure 3 showing circle1-selective sequences and coding-direction projections across cohorts",
    )
    ed5 = _extended_figure(
        5,
        "Representations of familiar versus novel stimuli.",
        "Panels a–b show circle2-selective-neuron results corresponding to Figure 3a and leaf1-selective-neuron results corresponding to Figure 3b.",
        "Extended Data Figure 5 showing circle2-selective and leaf1-selective neuron results",
    )
    ed6 = _extended_figure(
        6,
        "Visual recognition memory becomes exemplar-specific after extended training.",
        "Panels a–i show leaf3 behavior and projections on the leaf1–leaf2 axis, the reported de-orthogonalization schematic, and leaf1-swap behavioral and neural analyses.",
        "Extended Data Figure 6 showing leaf3 recognition, de-orthogonalization, and leaf1-swap analyses",
    )
    ed7 = _extended_figure(
        7,
        "Coding direction projections and similarity indices during test3.",
        "Panels a–g show a second swap stimulus, V1 and medial coding-direction examples, similarity indices, and test 3 licking summaries.",
        "Extended Data Figure 7 showing test 3 swap stimuli, coding-direction projections, similarity indices, and licking",
    )
    ed4 = _extended_figure(
        4,
        "Relation between neural activity and licking behaviors.",
        "Panels a–d show pre-cue licking on early- and late-cue trials, coding-direction projections on those trials, and the lick-versus-no-lick comparison corresponding to Figure 4l across regions.",
        "Extended Data Figure 4 showing pre-cue licking, coding-direction projections, and regional lick-versus-no-lick comparisons",
    )
    ed8 = _extended_figure(
        8,
        "Reward and non-reward prediction neurons across areas.",
        "Panels a–f show reward-prediction fractions by region, circle1-defined non-reward-prediction controls, and cue-aligned neural activity, running speed and licking rate across four corridors.",
        "Extended Data Figure 8 showing reward and non-reward prediction neurons, running, and licking across areas",
    )
    ed9 = _extended_figure(
        9,
        "Within-day learning in unsupervised pretraining experiment.",
        "Behavioral performance is split into the first and second halves of each session for 11 naturalistic-pretrained, 7 grating-pretrained and 5 no-pretraining mice.",
        "Extended Data Figure 9 showing within-day behavioral performance by pretraining cohort",
    )

    fig1 = _main_figure_card(
        number=1,
        title="Plasticity in the visual cortex after supervised and unsupervised training.",
        summary="Panels a–j report the task and cohort timeline, anticipatory licking, mesoscope imaging, the signed selectivity index, example selective responses, cortical density maps and regional selective-neuron fractions.",
        alt="Nature Figure 1 showing the task, training cohorts, behavior, imaging, selectivity index, neural responses, cortical maps, and regional fractions",
        panel_rows=f'''
            <div><dt>{_nature_cite("Fig1", "Figure&nbsp;1a–b")}</dt><dd>The virtual-reality corridors contain a random-position sound cue; water is available after the cue in the rewarded corridor. The timeline distinguishes task, unrewarded natural-texture and unrewarded grating cohorts.</dd></div>
            <div><dt>{_nature_cite("Fig1", "Figure&nbsp;1c–d")}</dt><dd>An example lick raster and anticipatory-lick summary report task behavior after learning.</dd></div>
            <div><dt>{_nature_cite("Fig1", "Figure&nbsp;1e–f")}</dt><dd>Mesoscope coverage and cellular resolution are shown beside the signed leaf1-versus-circle1 <em>d</em>&prime; definition.</dd></div>
            <div><dt>{_nature_cite("Fig1", "Figure&nbsp;1g–h")}</dt><dd>Single-trial responses are shown for example circle1-selective and leaf1-selective neurons and their corresponding populations.</dd></div>
            <div><dt>{_nature_cite("Fig1", "Figure&nbsp;1i–j")}</dt><dd>Aligned cortical density maps and regional fractions compare selective neurons before and after learning or exposure.</dd></div>''',
        finding=f'''The medial visual region&mdash;PM, AM, MMA and lateral retrosplenial cortex&mdash;showed similar selectivity changes in task and unrewarded natural-image cohorts, but not after grating exposure. The paper reports no change in lateral regions, little overall change in V1 and anterior modulation only in the supervised condition ({_nature_cite("Sec2", "Results: supervised and unsupervised plasticity")}; {_nature_cite("Fig1", "Figure&nbsp;1i–j")}).''',
        supporting_figures=ed1 + ed2,
    )
    fig2 = _main_figure_card(
        number=2,
        title="Comparing visual and spatial coding on test stimuli.",
        summary="Panels a–j report test 1 stimuli and licking, preferred-position sequence comparisons, leaf1- and circle1-selective populations, coding-direction projections and similarity indices.",
        alt="Nature Figure 2 showing test stimuli, licking, neural sequence comparisons, coding-direction projections, and similarity indices",
        panel_rows=f'''
            <div><dt>{_nature_cite("Fig2", "Figure&nbsp;2a–c")}</dt><dd>Test 1 presents familiar leaf1 and circle1 corridors with new leaf2 and circle2 exemplars; the figure shows an example lick raster and the five-mouse anticipatory-lick summary.</dd></div>
            <div><dt>{_nature_cite("Fig2", "Figure&nbsp;2d–f")}</dt><dd>Responses are sorted by preferred leaf1 position on held-out trials; preferred-position correlations are summarized within task mice and across regions and cohorts.</dd></div>
            <div><dt>{_nature_cite("Fig2", "Figure&nbsp;2g–j")}</dt><dd>Leaf1- and circle1-selective populations define a coding direction. Test-trial projections and a similarity index summarize the four stimuli across regions and cohorts.</dd></div>''',
        finding=f'''Preferred-position sequences in leaf1 and leaf2 were uncorrelated across regions. Separately, a coding direction learned from leaf1- and circle1-selective responses distinguished leaf-category from circle-category test stimuli in task, unrewarded and naive mice; the authors interpret these results as visual rather than spatial coding ({_nature_cite("Sec3", "Results: visual, not spatial, representations")}; {_nature_cite("Sec21", "Methods: coding direction and similarity index")}).''',
        supporting_figures=ed3,
    )
    fig3 = _main_figure_card(
        number=3,
        title="Responses to novel and adapted stimuli and neural orthogonalization.",
        summary="Panels a–h report leaf2 novelty and adaptation, leaf1-versus-leaf2 selectivity, licking, coding-direction projections and similarity indices.",
        alt="Nature Figure 3 showing novel and adapted stimulus responses, licking, leaf1-versus-leaf2 selectivity, projections, and orthogonalization",
        panel_rows=f'''
            <div><dt>{_nature_cite("Fig3", "Figure&nbsp;3a–b")}</dt><dd>Leaf2-versus-circle1 selective-neuron distributions and regional summaries compare leaf2 when new and after learning, with task, unrewarded and naive observations.</dd></div>
            <div><dt>{_nature_cite("Fig3", "Figure&nbsp;3c")}</dt><dd>Licking to leaf2 is compared when it is new and after task training.</dd></div>
            <div><dt>{_nature_cite("Fig3", "Figure&nbsp;3d–e")}</dt><dd>Leaf1-versus-leaf2 selective-neuron distributions and regional fractions compare task, unrewarded, naive and grating-control observations.</dd></div>
            <div><dt>{_nature_cite("Fig3", "Figure&nbsp;3f–h")}</dt><dd>Leaf2 responses are projected onto the leaf1–circle1 coding direction in V1 and the medial region; the similarity index is summarized across regions and cohorts.</dd></div>''',
        finding=f'''When leaf2 was new, leaf2-selective neurons were prominent in V1 and lateral visual areas and declined after an additional week of exposure. Leaf1–leaf2 selectivity increased in medial visual areas after both task training and unrewarded exposure, but not after grating exposure. Relative to naive mice, leaf2 projections onto the leaf1–circle1 coding direction were reduced across regions, most strongly in the medial region; the authors describe this as orthogonalization ({_nature_cite("Sec4", "Results: novelty and orthogonalization")}; {_nature_cite("Sec21", "Methods: coding direction and similarity index")}).''',
        supporting_figures=ed5 + ed6 + ed7,
    )
    fig4 = _main_figure_card(
        number=4,
        title="A reward-prediction signal in supervised training only.",
        summary="Panels a–n report Rastermap-guided discovery, cortical location, a late-versus-early cue-position index, regional distributions and response controls across test sessions.",
        alt="Nature Figure 4 showing Rastermap responses, reward-prediction neuron locations, cue-position selectivity, and behavioral controls",
        panel_rows=f'''
            <div><dt>{_nature_cite("Fig4", "Figure&nbsp;4a–c")}</dt><dd>A Rastermap-ordered population, an enlarged leaf1-active cluster and the selected cells&rsquo; cortical positions are shown.</dd></div>
            <div><dt>{_nature_cite("Fig4", "Figure&nbsp;4d–g")}</dt><dd>Leaf1 trials are split by sound-cue position to define late-versus-early cue-position <em>d</em>&prime;; selected-neuron distributions and anterior-region fractions are compared before and after learning.</dd></div>
            <div><dt>{_nature_cite("Fig4", "Figure&nbsp;4h–l")}</dt><dd>Test 1 responses are shown across corridors, aligned to cue or first lick, and split by lick versus no-lick trials.</dd></div>
            <div><dt>{_nature_cite("Fig4", "Figure&nbsp;4m–n")}</dt><dd>Population responses are shown in test 2 and test 3.</dd></div>''',
        finding=f'''The authors split leaf1 trials by sound-cue position and defined a late-versus-early <em>d</em>&prime;. Neurons meeting the threshold were concentrated primarily in anterior visual areas of task mice after training. The selected population was active before reward, suppressed at reward delivery, began ramping before the first lick and had higher activity on leaf2 trials with licks than on trials without licks. The authors interpret the combined dynamics as indicative of reward expectation ({_nature_cite("Sec6", "Results: reward prediction in anterior HVAs")}; {_nature_cite("Sec22", "Methods: reward-prediction neurons")}).''',
        supporting_figures=ed4 + ed8,
    )
    fig5 = _main_figure_card(
        number=5,
        title="Unsupervised pretraining accelerates subsequent task learning.",
        summary="Panels a–h report the three pretraining cohorts, task design, example licking, group learning curves, first-lick locations and daily trial counts.",
        alt="Nature Figure 5 showing pretraining cohorts, task structure, licking, learning curves, first-lick locations, and trial counts",
        panel_rows=f'''
            <div><dt>{_nature_cite("Fig5", "Figure&nbsp;5a–b")}</dt><dd>Naturalistic-texture, grating and no-pretraining cohorts undergo five task-training days. The sound cue is absent; reward is available in the second half of the rewarded corridor, contingent on licking except during the initial passive-reward day.</dd></div>
            <div><dt>{_nature_cite("Fig5", "Figure&nbsp;5c–d")}</dt><dd>Example lick distributions are shown for the first active-reward day and the last training day.</dd></div>
            <div><dt>{_nature_cite("Fig5", "Figure&nbsp;5e–f")}</dt><dd>Mean lick responses and rewarded-minus-non-rewarded performance are reported across days for 11 naturalistic-pretrained, 7 grating-pretrained and 5 no-pretraining mice.</dd></div>
            <div><dt>{_nature_cite("Fig5", "Figure&nbsp;5g–h")}</dt><dd>First-lick locations and daily trial counts are reported for the three cohorts.</dd></div>''',
        finding=f'''Naturalistic-texture pretraining was followed by faster discrimination learning than grating or no pretraining. All three cohorts reached high performance after five days; most improvement occurred within sessions, and the cohorts had similar first-lick positions and daily trial counts ({_nature_cite("Sec7", "Results: faster task learning after pretraining")}; {_figure_ref("nature-ed9", "Extended Data Figure&nbsp;9")}).''',
        supporting_figures=ed9,
    )

    methods = f'''
      <article id="paper-methods" class="result-card result-card--methods">
        <header class="result-card__head"><span class="result-card__index">M</span><div><p class="result-card__eyebrow">Nature Methods</p><h4>Acquisition, processing and observation units</h4></div></header>
        <div class="result-card__body">
          <dl class="result-panel-list">
            <div><dt>{_nature_cite("Sec13", "Imaging acquisition")}</dt><dd>Two-photon mesoscope recordings sampled multiple visual areas simultaneously.</dd></div>
            <div><dt>{_nature_cite("Sec19", "Calcium processing")}</dt><dd>The paper used Suite2p-derived deconvolved activity with a 0.75&nbsp;s decay parameter.</dd></div>
            <div><dt>{_nature_cite("Sec20", "Neural selectivity")}</dt><dd>The selectivity calculation retained original, non-interpolated running frames in the 0–4&nbsp;m textured segment and defined signed <em>d</em>&prime; from the two corridor-response distributions.</dd></div>
            <div><dt>{_nature_cite("Sec25", "Retinotopy")}</dt><dd>Recording-specific retinotopic maps were aligned to a reference mouse and used to assign cortical regions.</dd></div>
            <div><dt>{_nature_cite("Sec24", "Statistics and reproducibility")}</dt><dd>Panel captions state the mouse or session counts and whether paired or independent two-sided Student&rsquo;s <em>t</em>-tests were used.</dd></div>
          </dl>
          <aside class="result-card__takeaway"><p class="result-card__kicker">Reported scale</p><p>The paper reports 20,547–89,577 Suite2p traces per recording ({_nature_cite("Sec2", "Results")}) and 89 recordings in 19 mice ({_nature_cite("Sec11", "Animals")}). Neuron counts and biological replicate counts are distinct sample sizes.</p></aside>
        </div>
      </article>'''
    return f'''
    <h3>Main Figures 1&ndash;5</h3>
    <p class="result-guide__intro">Complete published figures are shown once, in numerical order. Panel summaries below paraphrase the corresponding Nature captions; each scientific statement links to the exact Results, Methods or figure location.</p>
{_figure_attribution()}
    <div class="result-guide" aria-label="Source-linked guide to the paper's main and Extended Data figures">
{fig1}{fig2}{fig3}{fig4}{fig5}{methods}
    </div>'''


def _extended_figure_atlas() -> str:
    main_links = " &middot; ".join(
        _figure_ref(f"nature-fig{number}", f"Figure&nbsp;{number}") for number in range(1, 6)
    )
    extended_links = " &middot; ".join(
        _figure_ref(f"nature-ed{number}", f"Extended Data&nbsp;{number}") for number in range(1, 10)
    )
    return f'''
    <div class="figure-atlas-intro" id="complete-evidence-atlas">
      <h3>Published figure index</h3>
      <p>Each complete figure appears once beside the main result it supports.</p>
      <p class="figure-index"><strong>Main figures:</strong> {main_links}<br><strong>Extended Data:</strong> {extended_links}</p>
    </div>'''


def _figure_map_section() -> str:
    paper_code = "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py"
    body = f'''
    <p>This table maps published panels to the exact Results or Methods subsection and to the paper-tagged analysis code. It describes the published computation; proposed trial-window extensions remain in the research-question and analysis sections.</p>
    <div class="tablewrap"><table>
      <caption>Published figure, method, and code map</caption>
      <thead><tr><th scope="col">Published panels</th><th scope="col">Reported content</th><th scope="col">Exact method</th><th scope="col">Paper code</th></tr></thead>
      <tbody>
        <tr><td>{_nature_cite("Fig1", "Figure&nbsp;1a–b")}</td><td>Virtual-reality task, cohort structure and training/test timeline.</td><td>{_nature_cite("Sec14", "Visual stimuli")} · {_nature_cite("Sec17", "Behavioral training")}</td><td><a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer"><code>Imaging_Exp_info.npy</code>&nbsp;&nearr;</a> supplies the released session labels.</td></tr>
        <tr><td>{_nature_cite("Fig1", "Figure&nbsp;1f,i–j")}</td><td>Signed leaf1-versus-circle1 <em>d</em>&prime;, selective-neuron density maps and regional fractions.</td><td>{_nature_cite("Sec20", "Neural selectivity")}</td><td><a href="{paper_code}#L370-L416" target="_blank" rel="noopener noreferrer"><code>dprime</code> and density-map helpers&nbsp;&nearr;</a></td></tr>
        <tr><td>{_nature_cite("Fig2", "Figure&nbsp;2d–j")}</td><td>Preferred-position sequence comparisons, coding-direction projections and similarity indices.</td><td>{_nature_cite("Sec20", "Neuron selection and sorting")} · {_nature_cite("Sec21", "Coding direction and similarity index")}</td><td><a href="{paper_code}#L503-L703" target="_blank" rel="noopener noreferrer">coding-direction and split-data helpers&nbsp;&nearr;</a></td></tr>
        <tr><td>{_nature_cite("Fig3", "Figure&nbsp;3a–h")}</td><td>Novel-stimulus selectivity, adaptation, leaf1-versus-leaf2 selectivity and coding-direction projections.</td><td>{_nature_cite("Sec4", "Results: novelty and orthogonalization")} · {_nature_cite("Sec21", "Coding direction")}</td><td><a href="{paper_code}#L503-L703" target="_blank" rel="noopener noreferrer">coding-direction and selective-neuron helpers&nbsp;&nearr;</a></td></tr>
        <tr><td>{_nature_cite("Fig4", "Figure&nbsp;4d–l")}</td><td>Late-versus-early cue-position <em>d</em>&prime;, regional distributions, cue/lick alignment and lick-versus-no-lick comparisons.</td><td>{_nature_cite("Sec6", "Results: reward prediction")} · {_nature_cite("Sec22", "Reward-prediction neurons")}</td><td><a href="{paper_code}#L814-L882" target="_blank" rel="noopener noreferrer">reward-prediction helpers&nbsp;&nearr;</a></td></tr>
        <tr><td>{_nature_cite("Fig5", "Figure&nbsp;5a–h")}</td><td>Behavior-only pretraining cohorts, task structure, learning curves, first-lick positions and trial counts.</td><td>{_nature_cite("Sec7", "Results: faster task learning")} · {_nature_cite("Sec17", "Behavioral training")} · {_nature_cite("Sec11", "Animals")}</td><td>The neural-imaging release does not supply recordings for these 23 behavior-only mice.</td></tr>
      </tbody>
    </table></div>
{_extended_figure_atlas()}
'''
    return _wrap_section(
        "figuremap",
        14,
        "Published figure provenance",
        "Paper panels, Methods, and code locations",
        body,
        bleed=True,
    )


def _experiment_figure_repeat() -> str:
    return ""


def _dprime_figure_repeat() -> str:
    return ""


def _within_figure_repeats() -> str:
    return ""


def _reward_figure_repeats() -> str:
    return ""


def _question_figure_repeats() -> str:
    return ""


def _analysis_qc_figure_repeats() -> str:
    return ""


def _experiment_enrichment() -> str:
    return f'''
    <details class="ref" open>
      <summary><span>Methods detail: corridor, cue, reward, movement, and cohort differences</span></summary>
      <div class="body">
        <ul class="clean">
          <li>The imaging corridor contained 4&nbsp;m of pseudorandom naturalistic texture followed by 2&nbsp;m of grey ({_nature_cite("Sec14", "Methods: visual stimuli")}). The sound cue was sampled from 0.5 to 3.5&nbsp;m in every task and unrewarded imaging trial, while water could follow it only in the rewarded corridor for task mice ({_nature_cite("Sec17", "Methods: behavioural training")}).</li>
          <li>Circle, leaf, rock, and brick are physical texture families in the paper ({_nature_cite("Sec14", "Methods: visual stimuli")}); <code>stim_id</code> and the released <code>Imaging_Exp_info.npy</code> provide the session-specific role mapping (<a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer">exact released index file&nbsp;&nearr;</a>).</li>
          <li>Train&nbsp;1 lasted about two weeks, Test&nbsp;1 introduced new exemplars, Train&nbsp;2 lasted about one week, Test&nbsp;2 introduced leaf3, and Test&nbsp;3 rearranged the familiar texture spatially ({_nature_cite("Fig1", "Figure&nbsp;1b timeline")}).</li>
          <li>The virtual corridor moved at 60&nbsp;cm&nbsp;s<sup>−1</sup> while running exceeded 6&nbsp;cm&nbsp;s<sup>−1</sup> ({_nature_cite("Sec14", "Methods: visual stimuli")}); the paper&rsquo;s selectivity calculation then retained running, non-interpolated frames within the 0–4&nbsp;m textured segment ({_nature_cite("Sec20", "Methods: neural selectivity")}). The broader speed/occupancy/cue/reward/lick QC set is a proposed safeguard here, not a statement that the paper fit all of those variables jointly.</li>
          <li>The paper states that some imaging mice received passive reward after a delay ({_nature_cite("Sec17", "Methods: behavioural training")}); the exact session-level <code>rewType</code> variants and the 4 rewarded versus 9 unrewarded Train&nbsp;1-before rows come from the released index (<a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer">exact released index file&nbsp;&nearr;</a>; <a href="#atlas-experiment-labels">rendered release audit&nbsp;↓</a>). Selecting that comparison for RQ2 is a proposed design choice.</li>
        </ul>
      </div>
    </details>
{_experiment_figure_repeat()}
'''


def _atlas_section(inventory: dict, index: dict) -> str:
    files = inventory["files"]
    full = {row["recording_id"]: row for row in files if row["category"] == "full_neural"}
    reduced = {row["recording_id"]: row for row in files if row["category"] == "reduced_neural"}
    retino = {row["retinotopy_id"]: row for row in files if row["category"] == "retinotopy"}
    behavior = {row["experiment"]: row for row in files if row["category"] == "imaging_behavior"}

    memberships: dict[str, list[dict]] = defaultdict(list)
    for experiment, rows in index["experiments"].items():
        for ordinal, row in enumerate(rows, start=1):
            memberships[row["recording_id"]].append(
                {"experiment": experiment, "ordinal": ordinal, "retinotopy_id": row["retinotopy_id"], "source": row["source"]}
            )

    recording_ids = set(memberships)
    assert recording_ids == set(full) == set(reduced)
    assert {item["retinotopy_id"] for values in memberships.values() for item in values} == set(retino)
    assert set(index["experiments"]) == set(behavior)
    assert len(recording_ids) == 89
    assert sum(len(rows) for rows in index["experiments"].values()) == 142

    exp_rows: list[str] = []
    for experiment in sorted(index["experiments"]):
        rows = index["experiments"][experiment]
        recs = sorted({row["recording_id"] for row in rows})
        mice = sorted({row["source"]["mname"] for row in rows})
        bundle = behavior[experiment]
        rec_list = " · ".join(f"<code>{escape(rec)}</code>" for rec in recs)
        exp_rows.append(
            f'<tr class="experiment-row"><td><code>{escape(experiment)}</code></td><td class="num">{len(rows)}</td>'
            f'<td class="num">{len(recs)}</td><td class="num">{len(mice)}</td>'
            f'<td>{_file_cell(bundle)}</td><td class="recording-list">{rec_list}</td></tr>'
        )

    by_mouse: dict[str, list[str]] = defaultdict(list)
    for recording_id in recording_ids:
        by_mouse[recording_id.split("_")[0]].append(recording_id)

    mouse_blocks: list[str] = []
    for mouse in sorted(by_mouse):
        recs = sorted(by_mouse[mouse], key=lambda rec: tuple(rec.split("_")[1:]))
        sex_values = sorted({
            item["source"]["Gender"]
            for rec in recs for item in memberships[rec]
            if item["source"].get("Gender")
        })
        sex = "/".join(sex_values) if sex_values else "sex not released"
        dates = sorted({"-".join(rec.split("_")[1:4]) for rec in recs})
        rows_html: list[str] = []
        for recording_id in recs:
            items = sorted(memberships[recording_id], key=lambda item: (item["experiment"], item["ordinal"]))
            retinotopy_id = items[0]["retinotopy_id"]
            block = recording_id.rsplit("_", 1)[1]
            date = "-".join(recording_id.split("_")[1:4])
            membership_html: list[str] = []
            for item in items:
                source = item["source"]
                key = _behavior_key(recording_id, source)
                membership_html.append(
                    f'<div class="membership"><div class="membership-title"><code>{escape(item["experiment"])}</code> '
                    f'· behavior key <code>{escape(key)}</code></div><div class="rawmeta">{_metadata_html(source)}</div></div>'
                )
            behavior_experiments = sorted({item["experiment"] for item in items})
            behavior_html = "<hr>".join(_file_cell(behavior[experiment], compact=True) for experiment in behavior_experiments)
            rows_html.append(
                f'<tr class="acquisition-row"><td><strong><code>{escape(recording_id)}</code></strong><span class="filemeta">{date} · block {escape(block)}</span></td>'
                f'<td class="memberships">{"".join(membership_html)}</td>'
                f'<td>{_file_cell(full[recording_id])}</td><td>{_file_cell(reduced[recording_id])}</td>'
                f'<td>{_file_cell(retino[retinotopy_id])}</td><td>{behavior_html}</td></tr>'
            )
        mouse_blocks.append(f'''
    <details class="ref atlas-mouse" open id="mouse-{escape(mouse)}">
      <summary><span>{escape(mouse)} · {len(recs)} {'acquisition' if len(recs) == 1 else 'acquisitions'} · {escape(sex)} · {len(dates)} {'date' if len(dates) == 1 else 'dates'} ({escape(", ".join(dates))})</span></summary>
      <div class="body tablewrap atlas-table"><table>
        <caption>Complete acquisition and logical-membership record for {escape(mouse)}</caption>
        <thead><tr><th scope="col">Acquisition</th><th scope="col">Every released membership and raw metadata</th><th scope="col">Full neural</th><th scope="col">SVD</th><th scope="col">Retinotopy</th><th scope="col">Behavior bundle(s)</th></tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table></div>
    </details>''')

    body = f'''
    <p>The paper reports 89 recordings in 19 imaging mice ({_nature_cite("Sec11", "Methods: animals")}) and deposits the files on Figshare (<a href="{NATURE}#data-availability" target="_blank" rel="noopener noreferrer">Data availability&nbsp;&nearr;</a>). This static atlas is a separate release audit generated from the checksum-pinned <a href="{FIGSHARE}" target="_blank" rel="noopener noreferrer">Figshare v2 inventory</a> and the deposited <a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer"><code>Imaging_Exp_info.npy</code>&nbsp;&nearr;</a>. Its <strong>89 physical acquisitions</strong>, <strong>142 source metadata rows</strong>, and <strong>23 experiment labels</strong> are inventory-derived counts, not numbers copied from a paper paragraph.</p>
    <div class="note angle"><span class="k">File-complete, with one declared manifest limit</span><p>The deposited manifest provides every file, acquisition key, exact byte size, MD5, and direct URL, but not each array&rsquo;s neuron × frame shape. Those shapes must be read from selected SVD/full files after checksum verification (<code>U.shape[1]</code>, <code>V.shape[1]</code>, or concatenated <code>spks.shape</code>) and are not inferred from file size here. The paper reports 20,547–89,577 traces per recording in the first Results section ({_nature_cite("Sec2", "Results: supervised and unsupervised plasticity")}).</p></div>
    <div class="note watch"><span class="k">Three grains—do not count them as one table</span><p>A physical acquisition is identified by <code>recording_id</code>; full and SVD files live at that grain. Retinotopy is acquisition-level but keyed without the block suffix. Behavior is stored once per experiment bundle and contains one or more session keys. The 142 metadata rows reduce to 133 unique experiment–recording memberships and 89 acquisitions; 25 acquisitions are reused under more than one experiment label.</p></div>
    <h3>All 23 experiment labels</h3>
    <div class="tablewrap experiment-table"><table>
      <caption>Association rows, unique physical recordings, mice, exact behavior bundle, and every recording ID</caption>
      <thead><tr><th scope="col">Experiment</th><th scope="col" class="num">Source rows</th><th scope="col" class="num">Recordings</th><th scope="col" class="num">Mice</th><th scope="col">Behavior bundle</th><th scope="col">Recording IDs</th></tr></thead>
      <tbody>{''.join(exp_rows)}</tbody>
    </table></div>
    <h3>All 19 mice and 89 neural acquisitions</h3>
    <p>Every mouse panel is expanded by default. Each acquisition lists its full processed deconvolved activity, 400-component SVD representation, retinotopy transform, linked behavior bundle(s), and every source metadata row. Missing fields remain missing; source spellings and reward-label capitalization are not silently normalized.</p>
{''.join(mouse_blocks)}
    <div class="note angle"><span class="k">Audit provenance</span><p>Inventory JSON SHA-256: <code>{EXPECTED_INVENTORY_SHA256}</code>. Experiment-index JSON SHA-256: <code>{EXPECTED_INDEX_SHA256}</code>. Source <code>Imaging_Exp_info.npy</code> MD5: <code>{escape(index["source"]["md5"])}</code>, SHA-256: <code>{escape(index["source"]["sha256"])}</code>. Direct file URLs, exact bytes, and MD5 values above come from the deposited release manifest.</p></div>
'''
    return _wrap_section("atlas", 6, "Complete release atlas", "Every experiment, mouse, acquisition, membership, and file", body, bleed=True)


def _support_section(inventory: dict, index: dict) -> str:
    duplicate_pairs: list[tuple[str, str, list[str]]] = []
    for experiment, rows in index["experiments"].items():
        grouped: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            grouped[row["recording_id"]].append(str(row["source"].get("stimtype") or "base"))
        for recording_id, variants in grouped.items():
            if len(variants) > 1:
                duplicate_pairs.append((experiment, recording_id, variants))
    assert len(duplicate_pairs) == 9
    swap_rows = "".join(
        f'<tr><td><code>{escape(exp)}</code></td><td><code>{escape(rec)}</code></td><td>{" · ".join(f"<code>{escape(v)}</code>" for v in variants)}</td></tr>'
        for exp, rec, variants in sorted(duplicate_pairs)
    )
    faster = [row for row in inventory["files"] if row["category"] == "faster_learning_behavior"]
    faster_total = sum(int(row["size_bytes"]) for row in faster)
    behavior_files = [row for row in inventory["files"] if row["category"] == "imaging_behavior"]
    behavior_total = sum(int(row["size_bytes"]) for row in behavior_files)
    body = f'''
    <p>Every one of the 89 full-neural acquisitions has a matching SVD file, matching retinotopy, at least one experiment membership, and a resolvable experiment-level behavior bundle in the <a href="{FIGSHARE}" target="_blank" rel="noopener noreferrer">release inventory</a>. The modalities differ in grain and must be joined deliberately.</p>
    <div class="tablewrap"><table>
      <caption>Modality semantics and scientifically relevant fields</caption>
      <thead><tr><th scope="col">Layer / grain</th><th scope="col">Released content</th><th scope="col">How it enters analysis</th></tr></thead>
      <tbody>
        <tr><td><strong>Full neural</strong><br>physical acquisition</td><td>Pickled dict with plane-wise <code>spks</code> arrays concatenated on the neuron axis. These are Suite2p-processed, deconvolved activity traces—not raw movies ({_nature_cite("Sec19", "Methods: processing of calcium imaging data")}).</td><td>Per-neuron paper estimator, distribution shape, thresholds, and full-versus-SVD benchmarks.</td></tr>
        <tr><td><strong>Reduced neural</strong><br>physical acquisition</td><td><code>U</code> is 400 × neuron and <code>V</code> is 400 × frame; reconstruct <code>U.T @ V</code>. It is lossy and the basis sees the full session.</td><td>Fast exploration, population projections, and team-scale prototyping. Validate tails and spread against full traces before confirmation.</td></tr>
        <tr><td><strong>Retinotopy</strong><br>mouse–date acquisition</td><td><code>iarea</code>, <code>xy_t</code>, and position/transform fields map neurons to V1, medial, lateral, anterior, or excluded cortex.</td><td>Area masks and density maps. It does not identify the same neuron across dates.</td></tr>
        <tr><td><strong>Imaging behavior</strong><br>experiment bundle → session key → frame/trial</td><td>Trial IDs/times/walls/stimulus mapping/reward/lick/run and aligned fields including <code>ft_trInd</code>, <code>ft_WallID</code>, <code>ft_Pos</code>, <code>ft_move</code>, <code>ft_CorrSpc</code>, <code>ft_RunSpeed</code>, cue/reward/lick event frames, and position support.</td><td>Role mapping, valid-frame mask, pairing/windowing, movement and occupancy QC, cue/reward/lick controls, and behavior–neural alignment.</td></tr>
      </tbody>
    </table></div>

    <details class="ref" open>
      <summary><span>Complete behavior-field glossary from the released processing notebook</span></summary>
      <div class="body">
        <p>The upstream <a href="{ORIGINAL_DRIVE}" target="_blank" rel="noopener noreferrer"><code>data_process_script.ipynb</code> in Drive</a> documents a calcium-frame rate of 3.17&nbsp;Hz and the schema below. Individual experiment bundles can contain protocol-specific missing or empty fields; the glossary is not permission to impute them.</p>
        <div class="tablewrap"><table><caption>Behavior dictionary keys, grouped without discarding any documented field</caption><thead><tr><th scope="col">Group</th><th scope="col">Fields</th><th scope="col">Meaning / use</th></tr></thead><tbody>
          <tr><td>Trial identity and timing</td><td><code>ntrials</code>, <code>trInd</code>, <code>trInd_odd</code>, <code>trInd_even</code>, <code>Trial_start_time</code>, <code>Trial_end_time</code></td><td>Trial count, ordered indices, odd/even split, and corridor entry/exit times.</td></tr>
          <tr><td>Stimulus identity</td><td><code>stim_id</code>, <code>WallType</code>, <code>WallIsProbe</code>, <code>WallName</code>, <code>UniqWalls</code>, <code>TrialStim</code>, <code>StimTrial</code>, <code>StimFrame</code></td><td><code>stim_id</code> defines roles 0–6. The upstream glossary explicitly says not to use generic <code>WallType</code> for these experiments and marks <code>WallIsProbe</code> as a catch-trial field to ignore here.</td></tr>
          <tr><td>Subject movement</td><td><code>SubjMove</code>: <code>SubjMTime</code>, <code>SubjMPos</code>, <code>SubjMPosCum</code>, <code>SubjMDistCum</code>, <code>SubjM_pitch</code>, <code>SubjM_roll</code>, <code>SubjM_yaw</code>, <code>SubjM_pitch_cum</code></td><td>Raw movement timestamps, VR/cumulative positions, and ball-motion axes. Pitch-cumulative distance is not identical to VR cumulative position because the VR advances at fixed speed after threshold crossing.</td></tr>
          <tr><td>VR position and epochs</td><td><code>Gray_space_time</code>, <code>VRpos</code>, <code>VRposCum</code>, <code>VRposTime</code></td><td>Grey-space entry and continuous VR position/time signals.</td></tr>
          <tr><td>Cue and reward</td><td><code>SoundPos</code>, <code>SoundTime</code>, <code>SoundTimeDelay</code>, <code>RewTime</code>, <code>RewPos</code>, <code>isRew</code></td><td>Sound position/time, delayed sound time, actual reward time/position, and rewarded-trial indicator. Reward values are valid only where the protocol delivered reward.</td></tr>
          <tr><td>Licking</td><td><code>LickTrind</code>, <code>LickTime</code>, <code>LickPos</code>, <code>Lick_wallName</code></td><td>Trial, time, position, and stimulus identity for every lick.</td></tr>
          <tr><td>Neural-frame alignment</td><td><code>ft</code>, <code>ft_trInd</code>, <code>ft_trInd_odd</code>, <code>ft_trInd_even</code>, <code>ft_Pos</code>, <code>ft_PosCum</code>, <code>ft_move</code>, <code>ft_isMoving</code>, <code>ft_GraySpc</code>, <code>ft_CorrSpc</code>, <code>ft_WallID</code>, <code>ft_RunCum</code>, <code>ft_RunSpeed</code>, <code>RunFr</code>, <code>run_pos</code></td><td>Per-neural-frame time, trial, position, movement, corridor/grey mask, wall, and running values; <code>run_pos</code> is the trial × position speed derivative.</td></tr>
          <tr><td>Event-frame indices</td><td><code>RewardFr</code>, <code>StartFr</code>, <code>GrayFr</code>, <code>EndFr</code>, <code>LickFr</code>, <code>SoundFr</code>, <code>SoundDelayFr</code>, <code>SoundDelPos</code>, <code>BefCueFr</code>, <code>AftCueFr</code></td><td>Neural-frame indices/masks for reward, corridor entry, grey entry, exit, licks, cue/delayed cue, and before/after-cue epochs.</td></tr>
          <tr><td>Protocol settings</td><td><code>Corridor_Length</code>, <code>Gray_Space_length</code>, <code>Texture_Length</code>, <code>Reward_Mode</code>, <code>Reward_Delay_ms</code></td><td>Session geometry and configured reward mode/delay. Use <code>RewTime</code> for actual delivery time.</td></tr>
        </tbody></table></div>
      </div>
    </details>

    <details class="ref" open>
      <summary><span>Retinotopy-field glossary and area code</span></summary>
      <div class="body">
        <ul class="clean">
          <li><code>iarea</code>: one integer label per neuron. The released loader maps 8→V1; {{0,1,2,9}}→medial HVA; {{5,6}}→lateral HVA; {{3,4}}→anterior HVA; −1 and 7 are excluded from those visual-cortical groups.</li>
          <li><code>xy_t</code>: transformed cortical coordinates aligned neuron-for-neuron with <code>iarea</code>. The paper density recipe uses <code>x = −xy_t[:,1]</code> and <code>y = xy_t[:,0]</code> before rasterization (<a href="{NATURE}/figures/1" target="_blank" rel="noopener noreferrer">Fig.&nbsp;1g–i</a>).</li>
          <li>Some downstream caches/notebook products expose coordinate aliases <code>xpos</code>/<code>ypos</code>; the canonical paper loader consumes <code>xy_t</code> and <code>iarea</code>, so analysis should not require aliases that are not verified in a selected file.</li>
          <li><code>areas.npz['out']</code> is a separate release-level set of area-outline polygons used for maps; it is not a per-acquisition neuron table.</li>
          <li>Retinotopy assigns location/area inside an acquisition. It supplies no cross-date cell-registration map and cannot support neuron-identity persistence claims.</li>
        </ul>
      </div>
    </details>

    <details class="ref" open>
      <summary><span>Requested missing-modality audit</span></summary>
      <div class="body">
        <div class="grid cols-2">
          <div class="card"><div class="h">Indexed IDs without full neural</div><p><strong>0</strong></p></div>
          <div class="card"><div class="h">Full IDs outside the index</div><p><strong>0</strong></p></div>
          <div class="card"><div class="h">Full IDs without SVD</div><p><strong>0</strong></p></div>
          <div class="card"><div class="h">Indexed dates without retinotopy</div><p><strong>0</strong></p></div>
          <div class="card"><div class="h">Metadata rows without full / retino</div><p><strong>0 / 0</strong></p></div>
          <div class="card"><div class="h">Experiment labels without behavior bundle</div><p><strong>0</strong></p></div>
        </div>
        <p><strong>Therefore item&nbsp;3 has no true imaging instance in the released index:</strong> there is no confirmed later behavioral/retinotopy session from an imaging mouse that lacks full neural data. Apparent extras are logical reuse of an acquisition, not modality-poor acquisitions. The static index proves each membership resolves to a published bundle; it does not claim that every internal pickle key was exhaustively opened.</p>
      </div>
    </details>

    <h3>Why some acquisitions appear more than once</h3>
    <p>Sixty-four acquisitions have one experiment label; 14 have two; 6 have three; 2 have four; and 3 have five. Those 44 extra logical memberships explain the difference between 89 acquisitions and 133 unique experiment–recording pairs. Nine pairs also contain two swap behavior instances:</p>
    <div class="tablewrap"><table><caption>Experiment–recording pairs with separate swap1 and swap2 behavior keys</caption><thead><tr><th scope="col">Experiment</th><th scope="col">Acquisition</th><th scope="col">Session variants</th></tr></thead><tbody>{swap_rows}</tbody></table></div>

    <div class="note watch"><span class="k">Separate behavior-only cohort</span><p>The paper identifies 23 additional behavior-only mice implanted with headbars but no cranial windows ({_nature_cite("Sec11", "Methods: animals")}) and reports their learning result in {_nature_cite("Fig5", "Figure&nbsp;5")}. The three deposited behavior bundles total <strong>{faster_total:,} bytes ({_human_bytes(faster_total)})</strong>; the absence of neural/retinotopy joins is established by the <a href="{FIGSHARE}" target="_blank" rel="noopener noreferrer">Figshare v2 inventory</a>, not inferred from Figure&nbsp;5. The 23 imaging behavior bundles total <strong>{behavior_total:,} bytes ({_human_bytes(behavior_total)})</strong>; repeated atlas links are deduplicated by deposited file ID.</p></div>

    <h3>Protocol snapshots are not missing-file errors</h3>
    <ul class="clean">
      <li>TX108/TX109 have naive baselines before supervised training; TX119/TX123 have naive baselines before unrewarded exposure.</li>
      <li>LZ13/LZ16/TX139 supply grating before/after acquisitions; TX140 and TX124 are naive-only.</li>
      <li>TX61 enters the supervised series at Test&nbsp;1 / Train&nbsp;2-before; TX83 stops at unrewarded Test&nbsp;1; TX104 and TX85 stop after Train&nbsp;1.</li>
      <li>Supervised Test&nbsp;3 may use separate physical sessions for swap variants, whereas several naive/unrewarded Test&nbsp;3 acquisitions hold two logical behavior instances in one neural acquisition.</li>
    </ul>
'''
    return _wrap_section("support", 7, "Behavior and retinotopy coverage", "What accompanies every full-neural acquisition—and what does not", body)


def _neural_frame_detail() -> str:
    return r'''
    <h3 id="data-neural-file">Inside one neural file: every axis and identity</h3>
    <div class="note finding">
      <span class="k">The short answer</span>
      <p><code>{mouse}_{YYYY}_{MM}_{DD}_{block}_neural_data.npy</code> is one continuous physical imaging acquisition. It is not one trial, not one experiment condition, and not the mouse&rsquo;s complete history. After the plane arrays are concatenated, one number <code>full[n, f]</code> is neuron <code>n</code>&rsquo;s non-negative Suite2p-deconvolved relative activity at acquisition frame <code>f</code>. The trial, position, stimulus, movement, cue, lick, reward, experiment, cortical area, session, day, and mouse meanings all come from explicit joins.</p>
    </div>

    <div class="note angle">
      <span class="k">The simplest correct mental model</span>
      <p><strong>One manifest join, two array-axis joins.</strong> SQL first chooses the four files for one experiment/acquisition. Inside those files, behavior labels the neural frame axis and retinotopy labels the neural neuron axis. Trial identity is the bridge from frames to one corridor traversal; the session-specific wall-to-role map is the bridge from physical texture names to leaf1/circle1. The dense neural matrix stays in NumPy.</p>
      <p><a href="https://colab.research.google.com/drive/1Xz40c50g5KczU5Rp5Dz_TYH2n21C-abP" target="_blank" rel="noopener noreferrer"><strong>Open notebook&nbsp;11: filesystem-only neural + behavior + retinotopy join&nbsp;&nearr;</strong></a>. It mounts Drive, reads <code>metadata/catalog.csv</code>, uses the SQL below, loads paths directly with <code>np.load</code>, exposes the frame/trial/neuron tables, and produces a trial-indexed held-out d&prime; curve without using <code>Dataset</code>, <code>Recording</code>, or related helper APIs.</p>
    </div>

    <div class="tablewrap">
      <table>
        <caption>The whole join in four lines</caption>
        <thead><tr><th scope="col">Layer</th><th scope="col">Grain</th><th scope="col">Exact join</th><th scope="col">What arrives</th></tr></thead>
        <tbody>
          <tr><td>File manifest</td><td>one experiment behavior instance</td><td>behavior by <code>experiment</code>; SVD/full by <code>recording_id</code>; retinotopy by <code>retinotopy_id</code></td><td>four verified relative paths</td></tr>
          <tr><td>Behavior &rarr; neural</td><td>one frame</td><td><code>ft_*[frame_id]</code> &harr; <code>V[:, frame_id]</code> or <code>full[:, frame_id]</code></td><td>trial, position, wall, movement, speed, and events</td></tr>
          <tr><td>Frame &rarr; trial &rarr; role</td><td>one traversal</td><td><code>ft_trInd[frame_id]</code> &rarr; <code>WallName[trial_id]</code> &rarr; <code>UniqWalls</code> &harr; <code>stim_id</code></td><td>role&nbsp;2 leaf1 or role&nbsp;0 circle1</td></tr>
          <tr><td>Retinotopy &rarr; neural</td><td>one neuron</td><td><code>iarea[neuron_id]</code>, <code>xy_t[neuron_id]</code> &harr; <code>U[:, neuron_id]</code> or <code>full[neuron_id, :]</code></td><td>area and cortical position</td></tr>
        </tbody>
      </table>
    </div>

    <details class="ref" open>
      <summary><span>The one SQL query that selects every file</span></summary>
      <div class="body">
        <pre><code>SELECT e.experiment, e.recording_id, e.behavior_key,
       b.relative_path AS behavior_path,
       s.relative_path AS svd_path,
       n.relative_path AS full_neural_path,
       r.relative_path AS retinotopy_path
FROM experiment_rows AS e
JOIN catalog AS b
  ON b.category = 'imaging_behavior' AND b.experiment = e.experiment
JOIN catalog AS s
  ON s.category = 'reduced_neural' AND s.recording_id = e.recording_id
JOIN catalog AS n
  ON n.category = 'full_neural' AND n.recording_id = e.recording_id
JOIN catalog AS r
  ON r.category = 'retinotopy' AND r.retinotopy_id = e.retinotopy_id
WHERE e.experiment = ? AND e.recording_id = ? AND e.behavior_key = ?</code></pre>
        <p>Usually <code>behavior_key = recording_id</code>. For the nine Test&nbsp;3 pairs with separate <code>swap1</code>/<code>swap2</code> behavior instances, use <code>behavior_key = recording_id + '_' + stimtype</code>. That is why the 142 raw index rows must not be reduced to only the 133 experiment&ndash;recording pairs before loading behavior.</p>
      </div>
    </details>

    <div class="note finding">
      <span class="k">Concrete alignment check from the linked working session</span>
      <p><code>sup_train1_before_learning</code> / <code>TX108_2023_03_13_1</code> has behavior fields with 22,994 frames and <code>V</code> with 22,992 frames, so exactly two declared trailing behavior frames are removed and no neural frame is truncated. <code>U</code> is 400&nbsp;&times;&nbsp;85,481, retinotopy has the same 85,481-neuron axis, and the validated reduction contains 210 trials &times; 18 position bins &times; 12 V1 features (98 role-0 and 112 role-2 trials). These are join/QC facts, not a d&prime; result.</p>
    </div>

    <div class="note watch">
      <span class="k">Why there is no literal one-trial d&prime;</span>
      <p>One trial supplies one stimulus role, not two response distributions. Notebook&nbsp;11 therefore gives every physical trial a blocked-cross-fitted <strong>held-out neural evidence</strong> value, not a one-trial d&prime;. Actual d&prime; belongs to a multi-trial segment containing repeated role-2 and role-0 trials. The notebook shows a trailing 40-trial held-out estimate at every segment endpoint and marks the stride-1 curve as correlated descriptive smoothing; non-overlapping segments are the safer analysis unit.</p>
    </div>

    <h3 id="data-neural-trial-timeline">Neural frames &rarr; trial evidence &rarr; segment d&prime;</h3>
    <div class="note angle">
      <span class="k">What the new aligned plot means</span>
      <pre><code>neural activity  |████ circle trial ████|████ leaf trial ████|████ circle ████|████ leaf ████|
trial role       | role 0 = circle1     | role 2 = leaf1     | role 0         | role 2       |
held-out result  |          ● evidence  |          ● evidence|       ●        |       ●      |
valid d′ segment |&lt;------------ repeated circle1 and leaf1 trials; four held-out folds --------&gt;|</code></pre>
      <p>The heatmap is a deterministic, label-free sample of retinotopically selected neurons reconstructed at their exact acquisition frames. Trial boundaries come from <code>ft_trInd</code>; the colored ribbon comes from <code>WallName &rarr; UniqWalls &harr; stim_id</code>; and a separate support ribbon marks <code>(ft_move &gt; 0) &amp; ft_CorrSpc</code>. Display rows are z-scored only for color and are not the d&prime; input.</p>
      <p>Each dot is the current trial&rsquo;s signed evidence on a coding direction fitted without that trial&rsquo;s contiguous fold. Positive means leaf1-like and negative means circle1-like. For a valid 40-trial segment, d&prime; is calculated independently on each of four held-out test folds using <code>ddof=1</code>, then those four fold d&prime; values are averaged. Scores produced by separately fitted folds are never pooled.</p>
    </div>

    <div class="tablewrap">
      <table>
        <caption>The complete identity hierarchy, from animal to one matrix value</caption>
        <thead><tr><th scope="col">Level</th><th scope="col">Identifier / axis</th><th scope="col">Exactly what it means</th><th scope="col">What it does not mean</th></tr></thead>
        <tbody>
          <tr><td><strong>Mouse</strong></td><td><code>TX119</code></td><td>The biological subject and independent longitudinal unit.</td><td>Frames, trials, neurons, and repeated sessions are not additional mice.</td></tr>
          <tr><td><strong>Day</strong></td><td><code>2023_12_14</code></td><td>The calendar date sampled at a protocol landmark.</td><td>The release is not continuous daily imaging.</td></tr>
          <tr><td><strong>Block</strong></td><td><code>1</code></td><td>The acquisition block on that date.</td><td>It is not a trial number.</td></tr>
          <tr><td><strong>Recording / session</strong></td><td><code>TX119_2023_12_14_1</code></td><td>One physical continuous calcium acquisition; this key joins full neural, SVD, retinotopy, and behavior.</td><td>An experiment label can point to this same acquisition without creating another session.</td></tr>
          <tr><td><strong>Experiment membership</strong></td><td><code>unsup_train1_before_learning</code></td><td>A logical protocol view selecting the recording&rsquo;s matching dictionary inside <code>Beh_{experiment}.npy</code>.</td><td>It is not stored on the neural matrix and does not create new frames.</td></tr>
          <tr><td><strong>Trial</strong></td><td><code>ft_trInd[f]</code></td><td>One corridor traversal containing a variable number of consecutive frames.</td><td>Trial lengths are not equal because running and stopping differ.</td></tr>
          <tr><td><strong>Frame</strong></td><td><code>f = 0…F−1</code></td><td>One acquisition-wide calcium sample, nominally about 3.17&nbsp;Hz (about 315&nbsp;ms spacing); use <code>ft[f]</code> when an actual released timestamp is needed.</td><td>A frame is not a trial and cannot by itself have d&prime;.</td></tr>
          <tr><td><strong>Neuron</strong></td><td><code>n = 0…N−1</code></td><td>One Suite2p trace in this acquisition; <code>iarea[n]</code> and <code>xy_t[n]</code> describe it.</td><td>Row <code>n</code> is not registered to row <code>n</code> on another date.</td></tr>
          <tr><td><strong>Value</strong></td><td><code>full[n, f]</code></td><td>Relative deconvolved activity of that neuron at that frame.</td><td>Not a raw movie pixel, raw fluorescence, &Delta;F/F, a binary spike, or spikes per second.</td></tr>
        </tbody>
      </table>
    </div>

    <div class="grid cols-2">
      <div class="card"><div class="h">Full released neural</div><p><code>{rec}_neural_data.npy</code> is a pickled dictionary whose <code>spks</code> value is a plane-wise sequence. Each plane is neurons-in-plane &times; the same frame axis. Concatenate only axis&nbsp;0 to obtain <code>N &times; F</code>. These are the processed traces used by the paper&rsquo;s single-neuron selectivity recipe (<a href="https://www.nature.com/articles/s41586-025-09180-y#Sec19" target="_blank" rel="noopener noreferrer">calcium-processing Methods&nbsp;&nearr;</a>; <a href="https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L338-L373" target="_blank" rel="noopener noreferrer">paper loader and d&prime; code&nbsp;&nearr;</a>).</p></div>
      <div class="card"><div class="h">SVD view of the same acquisition</div><p><code>{rec}_SVD_dec.npy</code> contains <code>U</code> with components &times; neurons and <code>V</code> with components &times; frames. <code>U.T @ V</code> is the deposited rank-400 reconstruction. It has the same neuron and frame identities, but it is lossy, is not a second recording, and its component numbers have no stable cross-session identity.</p></div>
    </div>

    <details class="ref" open>
      <summary><span>Load one full file, concatenate planes, and prove every axis</span></summary>
      <div class="body">
        <pre><code>import numpy as np
import drive

data = drive.setup()
session = data.recording("TX119_2023_12_14_1")

payload = session.load("full_neural", max_gib=10.0)
planes = payload["spks"]                    # sequence of 2-D plane arrays
plane_frames = {np.asarray(p).shape[1] for p in planes}
assert len(plane_frames) == 1              # every plane shares frame f
full = np.concatenate([np.asarray(p) for p in planes], axis=0)
n_neurons, n_frames = full.shape           # neurons × acquisition frames

svd = session.load("reduced_neural")
U = np.asarray(svd["U"])                    # 400 × neurons
V = np.asarray(svd["V"])                    # 400 × frames
ret = session.load("retinotopy")
iarea = np.asarray(ret["iarea"])            # neurons
xy_t = np.asarray(ret["xy_t"])              # neurons × 2

assert U.shape[1] == n_neurons == iarea.shape[0] == xy_t.shape[0]
assert V.shape[1] == n_frames
assert xy_t.shape[1] == 2

# Mathematical relationship; avoid materializing this multi-GiB result normally.
# rank400_full = U.T @ V                    # neurons × frames</code></pre>
        <p>Plane order followed by within-plane row order defines the full matrix&rsquo;s neuron axis. That same concatenated order must match <code>U</code>, <code>iarea</code>, and <code>xy_t</code>. Components are fitted separately per acquisition, and Suite2p detects cells separately per acquisition, so neither neuron rows nor component rows may be paired across dates.</p>
      </div>
    </details>

    <h3 id="data-frame-join">Frame by frame: how activity becomes an experiment observation</h3>
    <p>The neural file itself carries activity and axis order. For a chosen experiment membership, <code>Beh_{experiment}.npy[recording_id]</code> supplies the frame and trial annotations. Retinotopy supplies neuron annotations. The exact join is therefore:</p>
    <pre class="tree"><b>mouse</b>
└── <b>date + block = recording_id</b> ───────── one physical acquisition
    ├── full neural: <b>full[n, f]</b> ───────── activity
    ├── reduced neural: <b>U[:, n], V[:, f]</b>  rank-400 view
    ├── retinotopy: <b>iarea[n], xy_t[n]</b> ─── neuron n
    └── experiment membership → behavior
        ├── <b>ft_…[f]</b> ───────────────────── frame f
        ├── <b>WallName[trial]</b> ───────────── trial containing f
        └── cue / lick / reward fields ───────── trial, position, or event frame</pre>

    <div class="tablewrap">
      <table>
        <caption>What can be known about one aligned frame <code>f</code></caption>
        <thead><tr><th scope="col">Question</th><th scope="col">Released field / expression</th><th scope="col">Grain and interpretation</th></tr></thead>
        <tbody>
          <tr><td>What neural population state?</td><td><code>full[:, f]</code> or <code>V[:, f]</code></td><td>All full-neural values or the 400-component compressed state at the same acquisition frame.</td></tr>
          <tr><td>Which trial?</td><td><code>ft_trInd[f]</code></td><td>Integer-like trial identity; it indexes trial-level fields after validation.</td></tr>
          <tr><td>When?</td><td><code>ft[f]</code></td><td>Released frame-time value. The 3.17&nbsp;Hz rate is the nominal sampling rate, not a reason to ignore timestamps.</td></tr>
          <tr><td>Where?</td><td><code>ft_Pos[f] / 10</code></td><td>Position in metres; the release stores decimetres. <code>ft_PosCum</code> is cumulative rather than within-cycle position.</td></tr>
          <tr><td>Moving?</td><td><code>ft_move[f] &gt; 0</code>; also <code>ft_isMoving[f]</code></td><td>Paper-valid running indicator and a documented movement alias. Keep the exact choice in provenance.</td></tr>
          <tr><td>How fast?</td><td><code>ft_RunSpeed[f]</code></td><td>Frame-aligned running speed for confound/QC analysis.</td></tr>
          <tr><td>Texture or grey?</td><td><code>ft_CorrSpc[f]</code>, <code>ft_GraySpc[f]</code></td><td>The textured 0&ndash;4&nbsp;m corridor versus the 4&ndash;6&nbsp;m grey interval.</td></tr>
          <tr><td>Which physical wall?</td><td><code>ft_WallID[f]</code></td><td>The physical rock/wood texture at the frame; do not equate this string with a scientific role globally.</td></tr>
          <tr><td>Which functional role?</td><td><code>UniqWalls</code> joined to <code>stim_id</code></td><td>Session-specific mapping: 0=circle1, 1=circle2, 2=leaf1, 3=leaf2, with later test roles 4&ndash;6.</td></tr>
          <tr><td>Cue / reward / lick?</td><td><code>SoundFr</code>, <code>RewardFr</code>, <code>LickFr</code>; or <code>SoundPos</code>/<code>SoundDelPos</code>, <code>RewPos</code>, <code>LickPos</code> + <code>LickTrind</code></td><td>Event-frame fields can be joined directly; position fields join through trial and position. Missing/empty protocol-specific fields must remain missing.</td></tr>
          <tr><td>Which cortical region?</td><td><code>iarea[n]</code></td><td>Neuron-axis join: V1={8}, medial HVA={0,1,2,9}, lateral HVA={5,6}, anterior HVA={3,4}; exclude −1 and 7.</td></tr>
        </tbody>
      </table>
    </div>

    <div class="note watch">
      <span class="k">Why frame 20 is not comparable across trials</span>
      <p>Imaging is continuous. A fast traversal contributes fewer frames than a slow or stopped traversal, so the ordinal twentieth frame of two trials can represent different corridor positions and events. Join by trial plus position, or align explicitly to cue/lick/reward time. Pooling raw frames also weights slow trials more heavily; trial &times; position averaging is the appropriate sensitivity analysis when equal trial weight is required.</p>
    </div>

    <details class="ref" open>
      <summary><span>Strict alignment and a literal one-frame record</span></summary>
      <div class="body">
        <pre><code>from zhong2025.position import (
    align_trailing_behavior_frames,
    decimeters_to_meters,
)

experiment = "unsup_train1_before_learning"
behavior = session.load("behavior", experiment=experiment)

frame_fields = {
    "time": behavior["ft"],
    "trial_id": behavior["ft_trInd"],
    "wall_id": behavior["ft_WallID"],
    "position_dm": behavior["ft_Pos"],
    "move": behavior["ft_move"],
    "corridor": behavior["ft_CorrSpc"],
    "run_speed": behavior["ft_RunSpeed"],
}

# The helper expects frames × features. It never truncates neural data and
# permits only this explicitly bounded number of trailing behavior frames.
full_by_frame, aligned, report = align_trailing_behavior_frames(
    full.T,
    frame_fields,
    max_trailing_behavior_frames=3,
)
full = full_by_frame.T
position_m = decimeters_to_meters(aligned["position_dm"])
assert full.shape[1] == V.shape[1] == len(position_m)

f = 1372
assert 0 &lt;= f &lt; full.shape[1]
frame_record = {
    "recording_id": session.recording_id,
    "experiment": experiment,             # logical membership, not a new timeline
    "frame": f,
    "time": aligned["time"][f],
    "trial": aligned["trial_id"][f],
    "position_m": position_m[f],
    "wall": aligned["wall_id"][f],
    "moving": aligned["move"][f] &gt; 0,
    "in_texture": bool(aligned["corridor"][f]),
    "run_speed": aligned["run_speed"][f],
    "full_population": full[:, f],         # N values, one per neuron
    "svd_population": V[:, f],             # 400 values, same frame
}</code></pre>
        <p>The original paper recipe slices behavior fields to <code>nfr = full.shape[1]</code>. The helper above is a stricter project safeguard: only a small declared trailing behavior excess is allowed, all frame fields must have one common length, and neural frames are never silently discarded. A larger mismatch means the wrong behavior membership or a damaged selection, not permission to use <code>min(lengths)</code>.</p>
      </div>
    </details>

    <div class="tablewrap">
      <table>
        <caption>How the requested cohorts select before/after files</caption>
        <thead><tr><th scope="col">Case</th><th scope="col">Mice</th><th scope="col">Before membership</th><th scope="col">After membership</th></tr></thead>
        <tbody>
          <tr><td>Supervised</td><td>TX108, TX109, TX60, TX61, VR2</td><td><code>sup_train1_before_learning</code></td><td><code>sup_train1_after_learning</code></td></tr>
          <tr><td>Unsupervised natural textures</td><td>DR10, DR15, TX104, TX105, TX119, TX123, TX83, TX85, TX88</td><td><code>unsup_train1_before_learning</code></td><td><code>unsup_train1_after_learning</code></td></tr>
          <tr><td>Unrewarded gratings</td><td>LZ13, LZ16, TX139</td><td><code>train1_before_grating</code></td><td><code>train1_after_grating</code></td></tr>
          <tr><td>Naive-only reference</td><td>TX124, TX140</td><td colspan="2">No Train&nbsp;1 before/after pair is defined by the requested <code>EXPERIMENT_PAIRS</code>; do not invent an after stage.</td></tr>
        </tbody>
      </table>
    </div>
    <p>Each membership resolves to a recording ID, hence a date and block. &ldquo;Before&rdquo; and &ldquo;after&rdquo; are different acquisitions; they are not early and late frame ranges inside one file. Conversely, one acquisition may carry several valid experiment memberships, so reusing it under another label does not produce an independent recording. Across dates, compare session summaries paired by mouse, never neuron row <code>n</code> paired to neuron row <code>n</code>.</p>

    <h3 id="data-selectivity">Selectivity from frames: exact paper endpoint and trial-resolved alternatives</h3>
    <p>The paper&rsquo;s full-neural single-cell endpoint contrasts all valid role-2 (leaf1) frames against all valid role-0 (circle1) frames inside one acquisition:</p>
    <span class="formula">d&prime;<sub>n</sub> = 2 [mean(full[n, role&nbsp;2 frames]) − mean(full[n, role&nbsp;0 frames])] / [SD(full[n, role&nbsp;2 frames]) + SD(full[n, role&nbsp;0 frames])]</span>
    <p>The paper code uses <code>np.nanstd(...)</code>, hence population SD with <code>ddof=0</code>, no pooled-variance denominator, and no epsilon. Positive values prefer leaf1/role&nbsp;2; negative values prefer circle1/role&nbsp;0. The declared selective threshold is <code>|d&prime;| &ge; 0.3</code>. The exact support is <code>(ft_move &gt; 0) &amp; ft_CorrSpc</code>, then physical walls are resolved from the current session&rsquo;s <code>UniqWalls[stim_id==2]</code> and <code>UniqWalls[stim_id==0]</code> (<a href="https://www.nature.com/articles/s41586-025-09180-y#Sec20" target="_blank" rel="noopener noreferrer">neural-selectivity Methods&nbsp;&nearr;</a>; <a href="https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L418-L447" target="_blank" rel="noopener noreferrer">exact paper function&nbsp;&nearr;</a>).</p>

    <details class="ref" open>
      <summary><span>Paper-style full-neural d&prime;, memory-safe and auditable</span></summary>
      <div class="body">
        <pre><code>uniq_walls = np.asarray(behavior["UniqWalls"])
stim_id = np.asarray(behavior["stim_id"])
leaf1_wall = uniq_walls[stim_id == 2][0]
circle1_wall = uniq_walls[stim_id == 0][0]

valid = (
    (np.asarray(aligned["move"]) &gt; 0)
    &amp; np.asarray(aligned["corridor"]).astype(bool)
)
leaf1_frames = valid &amp; (np.asarray(aligned["wall_id"]) == leaf1_wall)
circle1_frames = valid &amp; (np.asarray(aligned["wall_id"]) == circle1_wall)
assert leaf1_frames.sum() &gt;= 2 and circle1_frames.sum() &gt;= 2

def paper_dprime_full(full, frames_a, frames_b, neuron_chunk=512):
    """Exact frame-pooled formula, evaluated without copying the whole file."""
    result = np.empty(full.shape[0], dtype=np.float64)
    for start in range(0, full.shape[0], neuron_chunk):
        stop = min(start + neuron_chunk, full.shape[0])
        block = np.asarray(full[start:stop])
        a = block[:, frames_a]
        b = block[:, frames_b]
        with np.errstate(divide="ignore", invalid="ignore"):
            result[start:stop] = (
                2.0 * (
                    np.nanmean(a, axis=1, dtype=np.float64)
                    - np.nanmean(b, axis=1, dtype=np.float64)
                )
                / (
                    np.nanstd(a, axis=1, ddof=0, dtype=np.float64)
                    + np.nanstd(b, axis=1, ddof=0, dtype=np.float64)
                )
            )
    return result

dp = paper_dprime_full(full, leaf1_frames, circle1_frames)
finite = np.isfinite(dp)
selective = finite &amp; (np.abs(dp) &gt;= 0.3)
leaf1_selective = finite &amp; (dp &gt;= 0.3)
circle1_selective = finite &amp; (dp &lt;= -0.3)

AREA_CODES = {
    "V1": [8],
    "mHV": [0, 1, 2, 9],
    "lHV": [5, 6],
    "aHV": [3, 4],
}

def selective_fraction(codes):
    eligible = finite &amp; np.isin(iarea, codes)
    return (
        float(np.mean(selective[eligible]))
        if np.any(eligible)
        else float("nan")
    )

fraction_by_area = {
    area: selective_fraction(codes)
    for area, codes in AREA_CODES.items()
}

provenance = {
    "recording_id": session.recording_id,
    "experiment": experiment,
    "n_neurons": int(full.shape[0]),
    "n_frames": int(full.shape[1]),
    "n_leaf1_frames": int(leaf1_frames.sum()),
    "n_circle1_frames": int(circle1_frames.sum()),
    "threshold": 0.3,
    "sd_ddof": 0,
    "frame_mask": "(ft_move &gt; 0) &amp; ft_CorrSpc",
}</code></pre>
        <p>Use finite d&prime; values as the declared denominator when reporting a robust selective fraction; report how many neurons were non-finite rather than silently classifying them as nonselective. Preserve both signed tails: <code>abs(d&prime;)</code> is useful for the total fraction but erases which stimulus a neuron prefers. Also retain the selected frame counts—large imbalance or behavior-dependent occupancy is essential context.</p>
      </div>
    </details>

    <details class="ref" open>
      <summary><span>The same calculation on the rank-400 SVD reconstruction</span></summary>
      <div class="body">
        <pre><code>from zhong2025.learning import svd_dprime

rank400_dp = svd_dprime(
    U,
    V,
    leaf1_frames,
    circle1_frames,
)

# This is algebraically equivalent to:
# reconstructed = U.T @ V
# paper_dprime_full(reconstructed, leaf1_frames, circle1_frames)
# but it does not allocate reconstructed.</code></pre>
        <p><code>rank400_dp</code> should match an explicit <code>U.T @ V</code> calculation to numerical precision. It need not match <code>dp</code> from the full file because the 400-component reconstruction omits residual activity. Use SVD for fast exploration and population dynamics; benchmark distribution spread, tails, signs, and threshold crossings against the full traces before making an exact paper-style claim.</p>
      </div>
    </details>

    <div class="tablewrap">
      <table>
        <caption>Where &ldquo;selectivity&rdquo; can be calculated—and what each number answers</caption>
        <thead><tr><th scope="col">Grain</th><th scope="col">Valid calculation</th><th scope="col">Interpretation</th><th scope="col">Main risk</th></tr></thead>
        <tbody>
          <tr><td>One frame</td><td>None</td><td>A frame supplies activity, label, position, and events, but no two response distributions.</td><td>Calling high activity &ldquo;selectivity&rdquo; without a contrast.</td></tr>
          <tr><td>One trial</td><td>Trial response by neuron and position</td><td>A building block for later contrasts; still only one stimulus role.</td><td>Treating one noisy traversal as d&prime;.</td></tr>
          <tr><td>Trial window containing both roles</td><td>Local per-neuron d&prime; or held-out population d&prime;</td><td>Descriptive within-session change when windows are chronological and predeclared.</td><td>Small counts, unequal occupancy, speed drift, cue/reward contamination, and overlapping-window dependence.</td></tr>
          <tr><td>Whole acquisition</td><td>Paper frame-pooled d&prime; for every neuron</td><td>The published single-neuron selectivity endpoint and selective fraction.</td><td>Slow trials contribute more frames; it is one endpoint, not a learning curve.</td></tr>
          <tr><td>Before versus after</td><td>Difference of session/area summaries paired by mouse</td><td>Learning- or exposure-associated population change.</td><td>Pairing neuron rows across dates or treating neurons/frames as independent animals.</td></tr>
        </tbody>
      </table>
    </div>

    <h4>Can d&prime; change inside one imaging file?</h4>
    <p>Yes, descriptively—but not frame by frame. A local d&prime; needs a chronological window containing repeated role-2 and role-0 trials. For a trial-resolved population trajectory, first average the small <code>V</code> state within each trial and fixed position bin, preserve empty-bin and frame-count information, and then fit/evaluate a coding direction on different trials. This equalizes trial contribution and avoids learning and testing the direction on the same observations.</p>

    <details class="ref" open>
      <summary><span>Held-out blockwise population d&prime; inside one file</span></summary>
      <div class="body">
        <pre><code>from zhong2025.position import bin_trial_features
from zhong2025.learning import blockwise_dprime

trial_raw = np.asarray(aligned["trial_id"], dtype=np.float64)
valid_trial = (
    np.isfinite(trial_raw)
    &amp; np.isclose(trial_raw, np.round(trial_raw))
    &amp; (trial_raw &gt;= 0)
    &amp; (trial_raw &lt; len(np.asarray(behavior["WallName"])))
)

edges_m = np.linspace(0.0, 6.0, 19)
valid_for_binning = (
    valid_trial
    &amp; (np.asarray(aligned["move"]) &gt; 0)
    &amp; np.asarray(aligned["corridor"], dtype=bool)
    &amp; np.isfinite(position_m)
    &amp; (position_m &gt;= 0.0)
    &amp; (position_m &lt;= 6.0)
)
trial_ids, trial_by_position_v, frame_counts = bin_trial_features(
    V.T,
    position_m,
    trial_raw,
    edges_m,
    valid_mask=valid_for_binning,
)

wall_to_role = {
    str(wall): int(role)
    for wall, role in zip(behavior["UniqWalls"], behavior["stim_id"])
    if np.isfinite(role)
}
wall_by_trial = np.asarray(behavior["WallName"])
labels = np.asarray(
    [wall_to_role.get(str(wall_by_trial[t]), -1) for t in trial_ids],
    dtype=np.int16,
)
texture_bins = edges_m[1:] &lt;= 4.0

curve = blockwise_dprime(
    trial_by_position_v,
    labels,
    trial_ids,
    position_mask=texture_bins,
    role_a=2,
    role_b=0,
    block_trials=40,
    stride_trials=40,       # non-overlapping primary display
    n_folds=4,
    min_per_role=4,
    require_complete_position_coverage=True,
)

# Inspect together; d′ alone is not a diagnosis.
for key in (
    "midpoint", "dprime", "mean_a", "mean_b", "sd_a", "sd_b",
    "separation", "spread", "n_a", "n_b", "valid_folds",
):
    print(key, curve[key])</code></pre>
        <p>This is intentionally a different estimand from the paper endpoint. It is held-out population/trial d&prime;, uses sample SD (<code>ddof=1</code>) on each test fold, and reports local chronology; the paper value is per-neuron, frame-pooled, whole-session, and uses <code>ddof=0</code>. A segment value is the mean of its four fold-specific held-out d&prime; values; raw scores from separately fitted folds are not pooled. Alongside the curve, inspect <code>frame_counts</code>, missing position bins, role balance, speed, cue/reward/lick positions, mean separation, within-role spread, and valid folds. Overlapping windows may be plotted only as a correlated exploratory overlay.</p>
      </div>
    </details>

    <div class="note angle">
      <span class="k">Inference boundary</span>
      <p>A rising local curve in one file is a within-acquisition description; it can reflect neural separation, reduced variability, speed/occupancy changes, event signals, drift, or combinations of these. For the requested before/after cases, reduce frames to trials, trials and neurons to an acquisition summary, repeated acquisitions within a stage to one mouse-stage summary, and then compare before versus after within mouse. Cohort evidence is carried by 5 supervised, 9 unsupervised, and 3 grating mice—not by the number of frames or neurons.</p>
    </div>
'''


def _environment_enrichment() -> str:
    notebook_rows = "".join(
        f'<tr><td><code>{escape(number)}</code></td><td><a href="{url}" target="_blank" rel="noopener noreferrer">{escape(title)}</a></td><td>{description}</td></tr>'
        for number, title, url, description in NOTEBOOKS
    )
    code_rows = "".join(
        f'<tr><td><a href="{url}" target="_blank" rel="noopener noreferrer"><code>{escape(name)}</code></a></td><td>{description}</td></tr>'
        for name, url, description in CODE
    )
    return f'''
    <h3>Verified Drive map</h3>
    <p>The current source set is in the <a href="{WORKSPACE_DRIVE}" target="_blank" rel="noopener noreferrer">shared workspace</a>; the 421.175&nbsp;GiB release is in the separate <a href="{DATA_DRIVE}" target="_blank" rel="noopener noreferrer">read-only dataset folder</a>. The two paper PDFs used to substantiate this document are the <a href="{PAPER_DRIVE}" target="_blank" rel="noopener noreferrer">Nature paper in Drive</a> and the <a href="{METHODS_DRIVE}" target="_blank" rel="noopener noreferrer">Science methods review in Drive</a>. The <a href="{NEUROMATCH_DOC}" target="_blank" rel="noopener noreferrer">Neuromatch planning document</a> provides project context but is not primary evidence for scientific claims.</p>
    <div class="tablewrap"><table><caption>All current analysis notebooks in the shared Drive</caption><thead><tr><th scope="col">Notebook</th><th scope="col">Open</th><th scope="col">Correct role and limit</th></tr></thead><tbody>{notebook_rows}</tbody></table></div>
    <div class="tablewrap"><table><caption>Drive code capability map</caption><thead><tr><th scope="col">Module</th><th scope="col">What it provides—and does not provide</th></tr></thead><tbody>{code_rows}</tbody></table></div>
    <div class="note angle"><span class="k">Graph stability contract</span><p><code>graph.py</code> runs declared nodes sequentially and exposes a visible run-state card. Scientific correctness still belongs in pure tested functions. Every graph should show loading, success, empty-result, or the full error; disable duplicate runs; store settings, outputs, and timings in <code>panel.last_run</code>; and provide a non-widget callable path for tests and batch execution. Notebook&nbsp;02 in Drive is a clean output-free copy.</p></div>
'''


def _science_method_figure(number: int, title: str, alt: str) -> str:
    asset_name = f"science-methods-fig{number}.jpg"
    width, height = _image_size(asset_name)
    pdf_page = number + 2
    source_url = f"{SCIENCE_PDF}#page={pdf_page}"
    return f'''
    <figure id="methods-fig{number}" class="paperfig evidence-figure science-methods-figure" data-figure-target="methods-fig{number}" data-figure-kind="science" data-source-pdf-page="{pdf_page}" data-source-sha256="{SCIENCE_SOURCE_SHA256}">
      <a href="{source_url}" target="_blank" rel="noopener noreferrer" aria-label="Open Science Figure {number} on PDF page {pdf_page}">
        <img loading="lazy" decoding="async" width="{width}" height="{height}" alt="{escape(alt)}" src="{_data_uri(asset_name)}">
      </a>
      <figcaption><b>Science Figure {number} &mdash; {escape(title)}.</b> Complete published figure and caption, cropped from PDF page&nbsp;{pdf_page} of the <a href="{METHODS_DRIVE}" target="_blank" rel="noopener noreferrer">reviewed Drive PDF</a>; artwork, labels, and plotted values are unchanged. <a class="figure-source-inline" href="{source_url}" target="_blank" rel="noopener noreferrer">Exact PDF page&nbsp;&nearr;</a></figcaption>
    </figure>'''


def _methods_paper_figures() -> str:
    figures = (
        (1, "Single-neuron analyses at scale", "Science Figure 1, with panels A through C on single-neuron properties, population averaging, and population smoothing, followed by the published caption"),
        (2, "Population vectors and dimensionality reduction", "Science Figure 2, with panels A through D on geometry, topology, dynamics, and exploratory dimensionality reduction, followed by the published caption"),
        (3, "Encoding, decoding, and cross-validation", "Science Figure 3, with panels A through C on encoding, decoding, and cross-validation, followed by the published caption"),
        (4, "Frameworks for analyzing large-scale recordings", "Science Figure 4 showing the review framework for single-neuron, population-structure, encoding, and decoding analyses, followed by the published caption"),
    )
    rendered = "\n".join(_science_method_figure(number, title, alt) for number, title, alt in figures)
    return f'''
    <h3 id="methods-review-figures">Published review figures</h3>
    <p>The four complete figures and their published captions below are cropped from pages&nbsp;3&ndash;6 of <a href="{METHODS_DRIVE}" target="_blank" rel="noopener noreferrer">the Science review PDF in the shared Drive</a>.</p>
    <div class="science-methods-atlas" aria-label="Complete figures from the large-scale neuronal recordings methods review">
{rendered}
    </div>
'''


def _methods_section() -> str:
    body = f'''
    <p>This section separates methods reported by Zhong et&nbsp;al. from safeguards proposed for new analyses. The large-scale-recording principles are sourced to <a href="{SCIENCE}" target="_blank" rel="noopener noreferrer">Stringer &amp; Pachitariu, <em>Science</em> 386, eadp7429 (2024)</a>; the reviewed copy is <a href="{METHODS_DRIVE}" target="_blank" rel="noopener noreferrer">in the shared Drive</a>. The paper-specific estimators remain linked to the exact Zhong Methods subsections.</p>
{_methods_paper_figures()}
    <h3 id="methods-review-contract">Published methods and proposed extensions</h3>
    <div class="tablewrap"><table>
      <caption>Published estimator, proposed extension, and validation source</caption>
      <thead><tr><th scope="col">Stage</th><th scope="col">Analysis</th><th scope="col">Validation criterion</th></tr></thead>
      <tbody>
        <tr><td>1 · Reproduce the published estimator</td><td>Can the signed selectivity and regional fractions be reproduced?</td><td>Use the paper&rsquo;s frame selection, <em>d</em>&prime; definition and threshold exactly as reported ({_nature_cite("Sec20", "Neural selectivity")}; {_nature_cite("Fig1", "Figure&nbsp;1f,i–j")}).</td></tr>
        <tr><td>2 · Single-neuron distributions</td><td>How do selectivity scale, asymmetry and tails vary by area, condition, stage and trial window?</td><td>Label this as a new analysis; preserve the mouse/session hierarchy and use independent observations for learned selection or ordering (<a href="{SCIENCE}" target="_blank" rel="noopener noreferrer">large-scale-recording review&nbsp;&nearr;</a>).</td></tr>
        <tr><td>3 · Population structure</td><td>Does a held-out population axis separate leaf and circle more strongly or earlier?</td><td>Fit normalization and axes on training observations and score held-out temporal blocks; compare invariant summaries across mice (<a href="{SCIENCE}" target="_blank" rel="noopener noreferrer">review&nbsp;&nearr;</a>).</td></tr>
        <tr><td>4 · Behavior and state</td><td>Are neural trajectories associated with licking, running, position, cue or reward timing?</td><td>Keep sensory <em>d</em>&prime;, late-versus-early cue-position <em>d</em>&prime; and chronological slope as distinct estimands ({_nature_cite("Sec20", "Sensory selectivity")}; {_nature_cite("Sec22", "Reward-prediction neurons")}).</td></tr>
      </tbody>
    </table></div>
    <details class="ref" open><summary><span>Analysis safeguards from the methods review</span></summary><div class="body"><ol class="steps" role="list">
      <li><strong>Write the biological estimand first.</strong> Separate sensory leaf–circle d′, late-versus-early cue-position reward-prediction d′, and chronological learning slope; the same symbol does not make them interchangeable.</li>
      <li><strong>Cross-validate every learned choice.</strong> Neuron selection, sorting, coding axes, normalization learned from data, and decoders are fitted on training blocks and scored on unseen blocks.</li>
      <li><strong>Block time.</strong> Slowly varying arousal, pose, running, and neural activity can create nonsense correlations under random-frame splits; use contiguous trial blocks (<a href="{SCIENCE}" target="_blank" rel="noopener noreferrer">review</a>).</li>
      <li><strong>Keep mice as independent units.</strong> Millions of neuron-window values improve descriptive precision but do not turn 4 and 9 mice into a large cohort.</li>
      <li><strong>Do not equate decoding with mechanism.</strong> With large populations many variables are decodable; perturbation or stronger design is needed for causality.</li>
      <li><strong>Label exploration.</strong> The review supports visualization and dimensionality reduction for discovery, but confirmation needs locked metrics, independent scoring, and multiplicity control.</li>
    </ol></div></details>
'''
    return _wrap_section("methods-review", 9, "Analysis methods for large-scale recordings", "A methodological contract for this dataset", body)


def _reward_section() -> str:
    body = f'''
    <div class="note finding"><span class="k">Research question 2</span><p><strong>Does rewarded training change the early within-session rate of cross-validated leaf1-versus-circle1 neural discriminability relative to the unrewarded-exposure cohort, after accounting for trial support, movement, position, cue timing, and licking?</strong></p></div>
    <p><strong>Proposed analysis—not a paper result.</strong> The paper reports familiar-stimulus selectivity increases after both rewarded training and unrewarded natural-texture exposure ({_nature_cite("Sec2", "Results: supervised and unsupervised plasticity")}; {_nature_cite("Fig1", "Figure&nbsp;1i–j")}). It separately reports a late-versus-early cue signal in anterior HVAs of task mice ({_nature_cite("Sec6", "Results: reward prediction")}; {_nature_cite("Fig4", "Figure&nbsp;4e–g")}). RQ2 asks a new chronological-rate question and does not restate either published endpoint.</p>
    <div class="tablewrap"><table><caption>Proposed RQ2 design</caption><thead><tr><th scope="col">Element</th><th scope="col">Primary specification</th><th scope="col">Reason</th></tr></thead><tbody>
      <tr><td>Comparison subset</td><td><code>sup_train1_before_learning</code> (4 passive-reward mice) versus <code>unsup_train1_before_learning</code> (9 no-reward mice)</td><td>Avoids later reward-mode heterogeneity; this remains a cohort comparison, not isolated reward randomization.</td></tr>
      <tr><td>Primary representation</td><td>Temporally blocked, held-out population leaf–circle discriminability in fixed non-overlapping trial blocks</td><td>Avoids same-trial fitting/scoring and overlapping-window pseudo-replication (<a href="{SCIENCE}" target="_blank" rel="noopener noreferrer">methods review</a>).</td></tr>
      <tr><td>Primary area</td><td>Pre-register medial HVA for sensory separation; V1/lateral/anterior are secondary</td><td>Figure&nbsp;1j shows the largest reported familiar-selectivity fraction changes in the medial grouping ({_nature_cite("Fig1", "Figure&nbsp;1j")}); choosing it as the RQ2 primary is a proposed design decision.</td></tr>
      <tr><td>Mouse estimand</td><td>One prespecified early slope per mouse; secondary time-to-threshold and saturation parameters</td><td>Keeps inference at the independent unit and makes &ldquo;faster&rdquo; explicit.</td></tr>
      <tr><td>Group test</td><td>Exact label permutation of mouse slopes; mouse/cluster bootstrap interval; leave-one-mouse-out stability</td><td>Appropriate to n=4 versus n=9 and robust to one influential mouse.</td></tr>
      <tr><td>Paper-reproduction check</td><td>Reproduce the anterior <code>d′late-versus-early</code> reward-prediction result separately</td><td>The exact estimator and held-out selection procedure are in {_nature_cite("Sec22", "Methods: reward-prediction neurons")}; its regional result is in {_nature_cite("Fig4", "Figure&nbsp;4f–g")}.</td></tr>
    </tbody></table></div>
    <h3>Hypotheses and interpretation</h3>
    <ul class="clean">
      <li><strong>H2a · equal rates:</strong> rewarded and unrewarded mice have similar early slopes; this is compatible with exposure-driven medial plasticity.</li>
      <li><strong>H2b · reward-associated acceleration:</strong> rewarded mice have a larger early slope after the locked QC and nuisance sensitivities.</li>
      <li><strong>H2c · proposed regional dissociation:</strong> test whether the new sensory-rate effect concentrates in medial HVA; separately require reproduction of the paper&rsquo;s anterior reward-prediction effect ({_nature_cite("Fig4", "Figure&nbsp;4f–g")}).</li>
      <li><strong>H2d · state explanation:</strong> an apparent group slope attenuates when speed, position support, cue/reward timing, and licking are examined jointly. This does not mean the state variables are mere nuisance if they are on the task pathway.</li>
    </ul>
{_reward_figure_repeats()}
    <div class="note watch"><span class="k">Causal-language limit</span><p>The imaging task protocol couples water restriction, sound cue, licking contingency, and reward delivery ({_nature_cite("Sec15", "Methods: water restriction")}; {_nature_cite("Sec16", "Methods: reward delivery and lick detection")}; {_nature_cite("Sec17", "Methods: behavioural training")}). Therefore a cohort slope difference is reward-associated evidence, not an isolated reward manipulation. Figure&nbsp;5 uses a separate behavior-only design ({_nature_cite("Fig5", "Figure&nbsp;5a–b")}) and cannot supply the missing neural causal contrast.</p></div>
'''
    return _wrap_section("reward-rate", 12, "Research question 2", "Does reward alter within-session neural learning rate?", body)


def _questions_section() -> str:
    body = f'''
    <p>The paper reports endpoint comparisons and a separate task-associated cue-position signal. It does not report the new chronological distribution and learning-rate analyses defined in <a href="#within">Research question&nbsp;1</a> and <a href="#reward-rate">Research question&nbsp;2</a>. The table below keeps those evidential boundaries explicit.</p>
    <div class="tablewrap"><table><caption>Published evidence and scope limits</caption><thead><tr><th scope="col">Topic</th><th scope="col">What the paper reports</th><th scope="col">What is not established</th></tr></thead><tbody>
      <tr><td>Familiar-stimulus selectivity</td><td>After Train&nbsp;1, task and unrewarded natural-texture cohorts increased familiar-stimulus selectivity; the grating cohort did not show a comparable increase ({_nature_cite("Sec2", "Results: supervised and unsupervised plasticity")}; {_nature_cite("Fig1", "Figure&nbsp;1i–j")}).</td><td>The reported panels do not estimate a chronological within-session trajectory of the full signed <em>d</em>&prime; distribution.</td></tr>
      <tr><td>Regional distribution</td><td>The medial grouping&mdash;PM, AM, MMA and lateral retrosplenial cortex&mdash;showed the largest reported familiar-selectivity changes ({_nature_cite("Sec2", "Results")}; {_nature_cite("Fig1", "Figure&nbsp;1i–j")}).</td><td>The endpoint comparison does not by itself identify whether a distribution changed in scale, asymmetry, tails, or mixture structure.</td></tr>
      <tr><td>Task-associated signal</td><td>Neurons meeting the late-versus-early cue-position criterion were concentrated primarily in anterior visual areas of task mice after training; the paper reports cue-, lick-, region-, running- and licking-related controls ({_nature_cite("Sec6", "Results: reward prediction")}; {_nature_cite("Sec22", "Methods: reward-prediction neurons")}; {_nature_cite("Fig4", "Figure&nbsp;4d–l")}).</td><td>This statistic is not leaf1-versus-circle1 sensory <em>d</em>&prime;, and the cohort design does not isolate reward from every task-associated variable.</td></tr>
      <tr><td>Later behavioral learning</td><td>In 23 separate behavior-only mice, natural-texture pretraining was followed by faster discrimination learning than grating or no pretraining ({_nature_cite("Sec7", "Results: faster task learning")}; {_nature_cite("Fig5", "Figure&nbsp;5")}; {_nature_cite("Sec11", "Methods: animals")}).</td><td>Those mice had no cranial windows, so Figure&nbsp;5 does not provide a neural learning-rate measurement.</td></tr>
    </tbody></table></div>
{_question_figure_repeats()}
'''
    return _wrap_section("questions", 13, "Evidence boundaries", "Published results and scope limits", body)


def _workflow_section() -> str:
    notebook_rows = "".join(
        f'<tr><td><a href="{url}" target="_blank" rel="noopener noreferrer"><code>{escape(number)}</code> {escape(title)}</a></td><td>{description}</td></tr>'
        for number, title, url, description in NOTEBOOKS
    )
    body = f'''
    <p>This runbook separates what is executable now from what still needs implementation. Use the <a href="{WORKSPACE_DRIVE}" target="_blank" rel="noopener noreferrer">shared Drive workspace</a> and one shortcut to the <a href="{DATA_DRIVE}" target="_blank" rel="noopener noreferrer">read-only release</a>; do not make a 421&nbsp;GiB copy per teammate.</p>
    <div class="tablewrap"><table><caption>Notebook order and evidential role</caption><thead><tr><th scope="col">Open</th><th scope="col">Use</th></tr></thead><tbody>{notebook_rows}</tbody></table></div>
    <h3>Execution order</h3>
    <ol class="steps" role="list">
      <li><strong>Access and provenance.</strong> Run notebook&nbsp;00; record catalog/release checksums, selected file IDs, sizes, MD5 values, runtime package versions, and the immutable JSON analysis specification.</li>
      <li><strong>Protocol and QC.</strong> Read notebook&nbsp;04, then run notebook&nbsp;03. Verify behavior–neural frame alignment, role mapping, valid-frame mask, trial counts, area counts, speed/position support, cue/reward/lick events, and exclusions by recording ID.</li>
      <li><strong>Reproduce the paper anchor.</strong> On full traces for declared sessions, recover whole-session d′, signed poles, |d′|≥0.3 fractions, and the medial before/after direction from <a href="{NATURE}/figures/1" target="_blank" rel="noopener noreferrer">Nature Fig.&nbsp;1</a>.</li>
      <li><strong>RQ1 exploration.</strong> Use notebook&nbsp;02 Graph&nbsp;4 for a small number of sessions. State that its default 40 PCs / ≤2,000 neurons and frame-pair estimator are exploratory, not the final estimator.</li>
      <li><strong>RQ1 confirmation—implementation still required.</strong> Add a deterministic maximum-lag trial pairer; trial × equal-position neuron summaries; fixed K-pair windows; all-neuron SD/MAD/IQR/skew/excess-kurtosis/quantile/tail metrics; a 4-stratum all-mouse runner; mouse-level inference; area multiplicity; and CSV/Parquet + JSON provenance export.</li>
      <li><strong>RQ2.</strong> In notebook&nbsp;05 run plan/preflight, mechanics, and simulations; explicitly switch from the default 2-per-group preview to all eligible 4+9 mice; run primary slope, exact permutation, clustered/mouse bootstrap, leave-one-mouse-out, cross-temporal, and position/behavior sensitivities.</li>
      <li><strong>Representation validation.</strong> Compare 400-component/all-neuron SVD results with full traces on representative sessions. The paper&rsquo;s single-neuron estimator uses full deconvolved traces; SVD is a computational approximation.</li>
      <li><strong>Persist everything.</strong> Save the locked spec, checksums, per-session QC, per-mouse tidy metrics, model/permutation/bootstrap output, figures, exclusions, and software environment. <code>panel.last_run</code> alone is not a durable scientific artifact.</li>
    </ol>

    <details class="ref" open><summary><span>Exact analysis products for RQ1</span></summary><div class="body"><ul class="clean">
      <li><code>rq1_spec.json</code>: cohort, area primary, role order, K, pair rule/max lag, position bins, valid mask, minimum support, metrics, multiplicity families, and sensitivities.</li>
      <li><code>rq1_window_metrics.parquet</code>: mouse, acquisition, experiment, stage, condition, area, window, progress, counts/gaps/support, SD/MAD/IQR/skew/kurtosis/quantiles/tails, and QC reason.</li>
      <li><code>rq1_mouse_effects.csv</code>: one before→after and within-session effect per mouse/area/metric, with the locked primary medial log-SD effect.</li>
      <li>Figures: ridgeline/ECDF/quantile curves, metric trajectories with mouse lines, signed tail plots, and QC panels. Density bandwidth never substitutes for distribution statistics.</li>
    </ul></div></details>

    <details class="ref" open><summary><span>Exact analysis products for RQ2</span></summary><div class="body"><ul class="clean">
      <li><code>rq2_spec.json</code>: Train&nbsp;1-before cohorts, primary area, contiguous folds, block size/horizon, slope definition, exclusions, exact permutation, bootstrap, LOMO, and nuisance sensitivities.</li>
      <li><code>rq2_mouse_slopes.csv</code>: one primary slope per mouse plus saturation/time-to-threshold and QC summaries.</li>
      <li><code>rq2_cross_temporal.zarr</code> or compact array artifact: training-block × test-block generalization, with position-specific summaries.</li>
      <li>Separate files for the anterior late-versus-early reward-prediction reproduction check; never merge that statistic with sensory leaf–circle d′.</li>
    </ul></div></details>

    <h3>Team-scale data access and fallback</h3>
    <p>The RQ1 26-session SVD/behavior/retinotopy plan is about 3.910&nbsp;GiB; its full-neural sources are 126.080&nbsp;GiB. The RQ2 before-only SVD plan is about 1.94&nbsp;GiB; full sources are about 62&nbsp;GiB. Each Colab runtime currently copies selected Drive files to ephemeral local cache, so concurrent users multiply shared-file reads. Google states that <a href="https://research.google.com/colaboratory/faq.html" target="_blank" rel="noopener noreferrer">Colab resources and limits fluctuate, Drive mounts can hit per-user/per-file operation and bandwidth quotas, and popular shared files are a typical trigger</a>.</p>
    <p>An API fallback should not make every teammate download the same multi-GiB full files again. The stable design is: <strong>mounted Drive → checksum-addressed object-store/CDN derivative → exact public Figshare file → MD5 verification → local VM cache</strong>, with bounded retries and jittered exponential backoff for 403/429 responses (<a href="https://developers.google.com/workspace/drive/api/guides/limits" target="_blank" rel="noopener noreferrer">official Drive API quota guidance</a>). Run full-neural preprocessing centrally once and publish compact trial × position or metric artifacts; keep team Colab fallback to reduced/compact derivatives.</p>
    <div class="note watch"><span class="k">Current implementation limits</span><p>Notebook&nbsp;02 is not yet an all-mouse RQ1 analysis. Notebook&nbsp;05 defaults to a 2-per-group preview, uses a session-wide SVD basis, has no full-neural estimator, treats behavior adjustment as diagnostic rather than causal modeling, and does not persist a complete result package automatically. The Drive workspace is also not a full source mirror; generator scripts and tests can differ from local copies.</p></div>
'''
    return _wrap_section("workflow", 16, "Executable analysis runbook", "How to use every notebook and code layer for both questions", body, bleed=True)


def _paper_code_walkthrough() -> str:
    learning_drive = "https://drive.google.com/file/d/1XSh_S8n51DOdLhoDG8GTNqaHlvVCxP09/view?usp=drivesdk"
    position_drive = "https://drive.google.com/file/d/1treoRl9CUCg9_GW8d4DmXr7Mfd59tsNJ/view?usp=drivesdk"
    return f'''
    <h3 id="paper-code-walkthrough">Read the paper code before extending it</h3>
    <p>The paper-tagged repository is compact enough to audit directly. The snippets below are the parts most likely to change the scientific interpretation of a plot. They are shown with their observation unit, split logic, and project-safe extension rather than as unexplained syntax.</p>

    <div class="code-study" id="code-paper-dprime">
      <div class="code-study-head"><span>1 · The published d′ is frame based</span><a href="https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L370-L374" target="_blank" rel="noopener noreferrer">utils.py 370&ndash;374</a></div>
      <pre><code>def dprime(x1, x2):              # neurons × frames
    u1, u2 = np.nanmean(x1, 1), np.nanmean(x2, 1)
    s1, s2 = np.nanstd(x1, 1), np.nanstd(x2, 1)  # ddof=0
    return 2 * (u1 - u2) / (s1 + s2)</code></pre>
      <p><strong>What it means.</strong> Each neuron gets one signed contrast between all valid leaf-role frames and all valid circle-role frames. The denominator is the arithmetic mean of the two population SDs, not pooled RMS variance and not Cohen's d. Positive means the first argument. Reproduce this exact endpoint before interpreting a trial-resolved extension ({_figure_ref("nature-fig1", "Nature Fig.&nbsp;1f&ndash;j")}).</p>
    </div>

    <div class="code-study" id="code-paper-mask">
      <div class="code-study-head"><span>2 · The caller fixes roles and valid frames</span><a href="https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L418-L441" target="_blank" rel="noopener noreferrer">utils.py 418&ndash;441</a></div>
      <pre><code>stim1 = uniqW[stim_id == 2][0]       # analysis role 2
stim2 = uniqW[stim_id == 0][0]       # analysis role 0
valid = (beh['ft_move'][:nfr] &gt; 0) &amp; beh['ft_CorrSpc'][:nfr]
dp = dprime(spk[:, (ft_WallID == stim1) &amp; valid],
            spk[:, (ft_WallID == stim2) &amp; valid])</code></pre>
      <p><strong>Why it matters.</strong> “Leaf” and “circle” are role names derived from <code>stim_id</code>; physical textures differ between some cohorts. The published comparison also excludes stationary and grey-space frames. A plot that infers roles from filenames, forgets the common frame truncation, or silently includes grey space is a different estimand.</p>
    </div>

    <div class="code-study" id="code-paper-splits">
      <div class="code-study-head"><span>3 · Selection, ordering, and display use different data</span><a href="https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L599-L703" target="_blank" rel="noopener noreferrer">utils.py 599&ndash;703</a></div>
      <pre><code># first trial parity: estimate d′ and select neurons
train_frames = (ft_trInd % 2 == 0) &amp; valid
dp = dprime(spk[:, stim1 &amp; train_frames], spk[:, stim2 &amp; train_frames])

# other parity: split again into ordering and displayed responses
test_spk = selected_position_tensor[:, reference_trials &amp; (trial % 2 == 1)]
order_data, display_data = test_spk[:, ::2], test_spk[:, 1::2]
sort_id = np.argsort(np.argmax(order_data.mean(1)[:, :40], axis=1))</code></pre>
      <p><strong>What the paper protected against.</strong> The first physical-trial parity selects neurons. Within the other parity, one subset determines peak-position order and the complementary subset is displayed. The comments call zero-based parity “odd/even”, so use the literal modulo test, not the comment. This role separation underlies the sorted sequences in {_figure_ref("nature-fig2", "Nature Fig.&nbsp;2")} and is documented in {_nature_cite("Sec20", "Methods: neural selectivity")}.</p>
    </div>

    <div class="code-study" id="code-paper-density">
      <div class="code-study-head"><span>4 · The density map is not a local selective fraction</span><a href="https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L376-L416" target="_blank" rel="noopener noreferrer">utils.py 376&ndash;416</a></div>
      <pre><code>selected = np.abs(dp) &gt;= 0.3
x, y = -xy_t[visual, 1], xy_t[visual, 0]
image = gaussian_filter(raster(selected), sigma=30)
image = image / visual.sum()        # total visual-cortex neurons
group_map = np.nanmean(mouse_images, axis=0)</code></pre>
      <p><strong>Interpretation.</strong> The plot is a smoothed selected-neuron spatial density divided by the total visual-cortex neuron count for each mouse, followed by a mouse mean. It is not <em>selected / all neurons at each pixel</em>. Bandwidth changes appearance but cannot establish that the d′ distribution broadened; use ECDFs, quantiles, and locked moments for that new claim ({_figure_ref("nature-fig1", "Figure&nbsp;1i&ndash;j")}).</p>
    </div>

    <div class="code-study" id="code-paper-coding-direction">
      <div class="code-study-head"><span>5 · Coding direction is a population contrast</span><a href="https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L503-L597" target="_blank" rel="noopener noreferrer">utils.py 503&ndash;597</a></div>
      <pre><code># normalize each neuron relative to grey space and the two stimulus spreads
spk_norm = 2 * (spk_position - grey_mean) / (stim1_sd + stim2_sd)
leaf_axis   = spk_norm[leaf_selective].mean(0)
circle_axis = spk_norm[circle_selective].mean(0)
projection  = leaf_axis - circle_axis</code></pre>
      <p><strong>Proposed extension.</strong> This averages selected populations and asks where each trial lies on a leaf–circle axis. It is complementary to a distribution of per-neuron d′ values: population separation can improve even if only a subset changes. For new analyses, learn normalization, selection and the axis on training blocks and score held-out trials (<a href="#methods-review">analysis safeguards&nbsp;↑</a>).</p>
    </div>

    <div class="code-study" id="code-paper-reward">
      <div class="code-study-head"><span>6 · Reward-prediction panels use held-out neuron selection</span><a href="https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L814-L882" target="_blank" rel="noopener noreferrer">utils.py 814&ndash;882</a></div>
      <pre><code>tr_shuf = np.random.permutation(ntrials)
for fold in range(10):
    dp_stim = dprime(train_circle_cue, train_leaf_cue)
    dp_value = dprime(train_late_cue_leaf, train_early_cue_leaf)
    selected = anterior &amp; (dp_stim &gt; 0.3) &amp; top_five_percent(dp_value)
    held_out_response[test_trials] = activity[selected][:, test_trials].mean(0)</code></pre>
      <p><strong>Important boundary.</strong> This is a late-versus-early cue/value statistic in anterior HVA, not the sensory leaf–circle d′ trajectory. The function calls <code>random.seed(2025)</code> but shuffles with NumPy's RNG, so the published helper is not deterministically seeded by that line. New RQ2 code should use an explicit <code>np.random.default_rng(seed)</code> or, preferably, deterministic contiguous folds. Interpret beside {_figure_ref("nature-fig4", "Nature Fig.&nbsp;4")} and {_figure_ref("nature-ed8", "ED Fig.&nbsp;8")}.</p>
    </div>

    <h3 id="project-code-extension">How the project code turns those recipes into stable analyses</h3>
    <div class="tablewrap"><table><caption>Paper recipe → tested project extension → most useful plot</caption><thead><tr><th scope="col">Need</th><th scope="col">Implementation</th><th scope="col">Plot and decision</th></tr></thead><tbody>
      <tr><td>Trial-level responses without crossing trial boundaries</td><td><a href="{position_drive}" target="_blank" rel="noopener noreferrer"><code>zhong2025/position.py</code></a> bins each physical trial by corridor position and leaves unsupported bins missing; it does not extrapolate across trials.</td><td>Trial × position support heatmap. Stop or mark a window missing when one role lacks the declared position coverage.</td></tr>
      <tr><td>Trial-window d′</td><td><a href="{learning_drive}" target="_blank" rel="noopener noreferrer"><code>learning.py</code> 51&ndash;78</a> uses sample SDs (<code>ddof=1</code>) across trial summaries; 444&ndash;521 builds blockwise sensory contrasts.</td><td>Small-multiple trajectories by area and stratum, with thin mouse lines and a mouse-level summary. Rolling windows are descriptive; non-overlapping blocks support inference.</td></tr>
      <tr><td>Fast SVD moments</td><td><a href="{learning_drive}" target="_blank" rel="noopener noreferrer"><code>learning.py</code> 154&ndash;234</a> obtains per-neuron means/variances from the reduced basis without reconstructing the full matrix.</td><td>SVD-versus-full Bland–Altman/scatter on representative sessions. Do not claim exact single-neuron tails until agreement is measured.</td></tr>
      <tr><td>Held-out population score</td><td><a href="{learning_drive}" target="_blank" rel="noopener noreferrer"><code>learning.py</code> 298&ndash;412</a> assigns contiguous folds before filtering, learns standardization and direction on other folds, and never pools raw scores across independently fit folds.</td><td>Fold-wise d′ trajectory under the <a href="#methods-review">analysis safeguards</a>; invalid folds remain visible with the support reason.</td></tr>
      <tr><td>Stability across time and position</td><td><a href="{learning_drive}" target="_blank" rel="noopener noreferrer"><code>learning.py</code> 524&ndash;647 and 650&ndash;721</a> compute position surfaces, slopes, saturation, and cross-temporal generalization.</td><td>Train-block × test-block heatmap and trial-progress × position surface. Broad off-diagonal generalization supports a stable axis; diagonal-only structure suggests transient state.</td></tr>
      <tr><td>Independent-unit inference</td><td><a href="{learning_drive}" target="_blank" rel="noopener noreferrer"><code>learning.py</code> 724&ndash;860</a> collapses to unique mice, enumerates all 715 assignments for the 4-versus-9 comparison, and bootstraps mice.</td><td>Mouse slope dot plot, exact permutation distribution, bootstrap interval, and leave-one-mouse-out forest plot.</td></tr>
    </tbody></table></div>

    <h3 id="plot-playbook">Plot playbook for the two questions</h3>
    <div class="tablewrap"><table><caption>Use each plot for one clearly stated job</caption><thead><tr><th scope="col">Plot</th><th scope="col">Answers</th><th scope="col">Guardrail</th></tr></thead><tbody>
      <tr><td>ECDF or signed quantile curves</td><td>Did the whole d′ distribution translate, widen, or change asymmetrically?</td><td>Show signed values; same x-axis across strata; add mouse-resampled uncertainty rather than treating neurons as independent.</td></tr>
      <tr><td>Ridgeline / violin / KDE</td><td>Where is the mass, and is a tail or second mode visually plausible?</td><td>Keep one common bandwidth and axis; treat it as visualization, not the estimator of SD or kurtosis.</td></tr>
      <tr><td>Metric trajectory small multiples</td><td>How do log SD, IQR/MAD, skew, excess kurtosis, and signed tail fractions evolve over trial progress?</td><td>One facet per metric, area, and declared stratum; thin mouse paths; never hide a dimension behind a toggle in the confirmatory figure.</td></tr>
      <tr><td>Signed tail plot</td><td>Are leaf- and circle-selective tails recruited symmetrically?</td><td>Plot <code>P(d′≥t)</code> and <code>P(d′≤−t)</code> together over a threshold sweep, anchored to {_figure_ref("nature-ed1", "ED Fig.&nbsp;1e&ndash;f")}.</td></tr>
      <tr><td>Cross-temporal matrix</td><td>Does an axis learned early still separate stimuli later, or only within the same block?</td><td>Fit every row on its training block, test every column without refitting, and annotate class support (<a href="#methods-review">analysis safeguards</a>).</td></tr>
      <tr><td>QC companion panel</td><td>Could speed, occupancy, cue position, reward, licking, or unequal role counts explain the trajectory?</td><td>Show the QC next to the result, not in an appendix. Use {_figure_ref("nature-ed2", "ED Fig.&nbsp;2")} and {_figure_ref("nature-ed8", "ED Fig.&nbsp;8")} as paper precedents.</td></tr>
    </tbody></table></div>

    <details class="ref" open><summary><span>Minimal statistics code for exploration, followed by the inferential boundary</span></summary><div class="body">
      <pre><code>from scipy import stats

summary = {{
    "sd": np.std(dp, ddof=1),
    "iqr": stats.iqr(dp, nan_policy="omit"),
    "skew": stats.skew(dp, bias=False, nan_policy="omit"),
    "excess_kurtosis": stats.kurtosis(dp, fisher=True, bias=False,
                                      nan_policy="omit"),
    "leaf_tail": np.mean(dp &gt;= threshold),
    "circle_tail": np.mean(dp &lt;= -threshold),
}}

# Secondary hierarchical model after one row per mouse/session/window is built.
model = smf.mixedlm(
    "log_sd ~ progress * stage * condition",
    metric_rows,
    groups=metric_rows["mouse"],
    vc_formula={{"session": "0 + C(session)"}},
)</code></pre>
      <p>The SciPy dictionary is descriptive and should be computed for every declared stratum. The mixed model is secondary because the cohort is small and trajectories are correlated. RQ1's primary effect should be a prespecified mouse-level before→after change; RQ2's 4-versus-9 primary comparison should use the exact mouse-label permutation, with bootstrap and leave-one-mouse-out uncertainty. Neuron-window rows are never the independent sample size.</p>
    </div></details>
'''


def _toc() -> str:
    items = "\n".join(f'        <li><a href="#{section_id}">{label}</a></li>' for section_id, label in TOC_SECTIONS)
    return f'''    <nav id="contents" class="toc" aria-label="Contents">
      <ol role="list">
{items}
      </ol>
    </nav>'''


def _numbered_toc_items() -> list[tuple[str, str, int, str]]:
    numbered: list[tuple[str, str, int, str]] = []
    section = subsection = detail = 0
    for target, label, level in TOC_ITEMS:
        if level == 1:
            section += 1
            subsection = detail = 0
            number = f"{section:02d}"
        elif level == 2:
            if not section:
                raise ValueError(f"TOC subsection {target} has no parent section")
            subsection += 1
            detail = 0
            number = f"{section:02d}.{subsection}"
        elif level == 3:
            if not subsection:
                raise ValueError(f"TOC detail {target} has no parent subsection")
            detail += 1
            number = f"{section:02d}.{subsection}.{detail}"
        else:
            raise ValueError(f"unsupported TOC level {level} for {target}")
        numbered.append((target, label, level, number))
    return numbered


def _floating_toc_items() -> str:
    roots: list[dict] = []
    current_root: dict | None = None
    current_subsection: dict | None = None
    for target, label, level, number in _numbered_toc_items():
        node = {
            "target": target,
            "label": label,
            "level": level,
            "number": number,
            "children": [],
        }
        if level == 1:
            roots.append(node)
            current_root = node
            current_subsection = None
        elif level == 2:
            assert current_root is not None
            current_root["children"].append(node)
            current_subsection = node
        else:
            assert current_subsection is not None
            current_subsection["children"].append(node)

    def render(node: dict, indent: int) -> list[str]:
        level = node["level"]
        classes = ["toc-fab__item", f"toc-fab__item--level-{level}"]
        if level == 1:
            classes.append("toc-fab__group")
        prefix = " " * indent
        output = [
            f'{prefix}<li class="{" ".join(classes)}">',
            f'{prefix}  <a class="toc-fab__link toc-fab__link--level-{level}" data-toc-level="{level}" href="#{node["target"]}"><span class="toc-fab__number" aria-hidden="true">{node["number"]}</span><span class="toc-fab__label">{escape(node["label"])}</span></a>',
        ]
        if node["children"]:
            child_level = level + 1
            output.append(f'{prefix}  <ol class="toc-fab__sublist toc-fab__sublist--level-{child_level}" role="list">')
            for child in node["children"]:
                output.extend(render(child, indent + 4))
            output.append(f"{prefix}  </ol>")
        output.append(f"{prefix}</li>")
        return output

    lines: list[str] = []
    for root in roots:
        lines.extend(render(root, 8))
    return "\n".join(lines)


def _floating_toc() -> str:
    items = _floating_toc_items()
    return f'''<!-- FLOATING-TOC:START -->
<div class="toc-fab" data-toc-fab>
  <button class="toc-fab__launcher" type="button" aria-label="Open table of contents" aria-expanded="false" aria-controls="floating-contents">
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <g fill="currentColor"><circle cx="5" cy="7" r="1.4"/><circle cx="5" cy="12" r="1.4"/><circle cx="5" cy="17" r="1.4"/><rect x="9" y="6.1" width="11" height="1.8" rx=".9"/><rect x="9" y="11.1" width="11" height="1.8" rx=".9"/><rect x="9" y="16.1" width="11" height="1.8" rx=".9"/></g>
    </svg>
  </button>
  <nav id="floating-contents" class="toc-fab__panel" aria-labelledby="floating-contents-title" hidden>
    <div class="toc-fab__header">
      <div><span class="toc-fab__eyebrow">On this page</span><strong id="floating-contents-title">Navigator</strong></div>
      <button class="toc-fab__close" type="button" aria-label="Close table of contents">
        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      </button>
    </div>
    <div class="toc-fab__tools">
      <div class="toc-fab__search" role="search">
        <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="m16 16 4 4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
        <input type="search" inputmode="search" autocomplete="off" spellcheck="false" placeholder="Find a section or recipe…" aria-label="Filter navigator destinations">
        <button class="toc-fab__clear" type="button" aria-label="Clear navigator search" hidden>Clear</button>
      </div>
      <div class="toc-fab__status">
        <span data-toc-result-count>{len(TOC_ITEMS)} destinations</span>
        <button class="toc-fab__detail-toggle" type="button" aria-pressed="false">Show all details</button>
      </div>
    </div>
    <ol class="toc-fab__list" role="list">
{items}
    </ol>
  </nav>
</div>
<script>
(() => {{
  const root = document.querySelector('[data-toc-fab]');
  if (!root) return;
  const launcher = root.querySelector('.toc-fab__launcher');
  const panel = root.querySelector('.toc-fab__panel');
  const closeButton = root.querySelector('.toc-fab__close');
  const list = root.querySelector('.toc-fab__list');
  const links = Array.from(root.querySelectorAll('.toc-fab__link'));
  const groups = Array.from(root.querySelectorAll('.toc-fab__group'));
  const searchInput = root.querySelector('.toc-fab__search input');
  const clearSearch = root.querySelector('.toc-fab__clear');
  const detailToggle = root.querySelector('.toc-fab__detail-toggle');
  const resultCount = root.querySelector('[data-toc-result-count]');
  const sections = links.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);
  let isOpen = false;
  let isDetailed = false;
  let framePending = false;

  const normalize = value => value.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLocaleLowerCase();
  const linkMatches = (link, query) => normalize(`${{link.textContent}} ${{link.hash.slice(1).replaceAll('-', ' ')}}`).includes(query);

  const revealActiveLink = () => {{
    const active = root.querySelector('.toc-fab__link[aria-current="location"]');
    if (!active || !list) return;
    const listRect = list.getBoundingClientRect();
    const activeRect = active.getBoundingClientRect();
    list.scrollTop += activeRect.top - listRect.top - (listRect.height - activeRect.height) / 2;
  }};

  const setDetailed = next => {{
    isDetailed = Boolean(next);
    root.classList.toggle('is-detailed', isDetailed);
    detailToggle.setAttribute('aria-pressed', String(isDetailed));
    detailToggle.textContent = isDetailed ? 'Focus current section' : 'Show all details';
  }};

  const applyFilter = () => {{
    const query = normalize(searchInput.value.trim());
    root.classList.toggle('is-filtering', Boolean(query));
    clearSearch.hidden = !query;
    let matches = 0;

    for (const group of groups) {{
      const groupLinks = Array.from(group.querySelectorAll('.toc-fab__link'));
      const matchingLinks = query ? groupLinks.filter(link => linkMatches(link, query)) : groupLinks;
      matches += matchingLinks.length;
      group.hidden = Boolean(query) && matchingLinks.length === 0;

      const nestedItems = Array.from(group.querySelectorAll('.toc-fab__item:not(.toc-fab__group)'));
      for (const item of nestedItems) {{
        if (!query) {{
          item.hidden = false;
          continue;
        }}
        const itemLinks = Array.from(item.querySelectorAll('.toc-fab__link'));
        item.hidden = !itemLinks.some(link => linkMatches(link, query));
      }}
    }}

    resultCount.textContent = query ? `${{matches}} of ${{links.length}} destinations` : `${{links.length}} destinations`;
  }};

  const setOpen = (next, returnFocus = false) => {{
    isOpen = Boolean(next);
    root.classList.toggle('is-open', isOpen);
    launcher.setAttribute('aria-expanded', String(isOpen));
    panel.hidden = !isOpen;
    if (isOpen) {{
      requestAnimationFrame(() => {{ searchInput.focus(); revealActiveLink(); }});
    }} else if (returnFocus) {{
      launcher.focus();
    }}
  }};

  const updateActive = () => {{
    framePending = false;
    const marker = Math.max(96, window.innerHeight * .22);
    let current = sections[0] || null;
    for (const section of sections) {{
      if (section.getBoundingClientRect().top <= marker) current = section;
      else break;
    }}
    for (const link of links) {{
      const active = current && link.getAttribute('href') === `#${{current.id}}`;
      link.classList.toggle('is-active', Boolean(active));
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }}
    const activeLink = root.querySelector('.toc-fab__link[aria-current="location"]');
    for (const group of groups) group.classList.toggle('is-current', Boolean(activeLink && group.contains(activeLink)));
  }};

  const queueActiveUpdate = () => {{
    if (framePending) return;
    framePending = true;
    requestAnimationFrame(updateActive);
  }};

  launcher.addEventListener('click', () => setOpen(true));
  closeButton.addEventListener('click', () => setOpen(false, true));
  detailToggle.addEventListener('click', () => {{ setDetailed(!isDetailed); revealActiveLink(); }});
  searchInput.addEventListener('input', applyFilter);
  clearSearch.addEventListener('click', () => {{
    searchInput.value = '';
    applyFilter();
    searchInput.focus();
  }});
  links.forEach(link => link.addEventListener('click', event => {{
    const target = document.querySelector(link.getAttribute('href'));
    if (!target) return;
    event.preventDefault();
    history.pushState(null, '', link.getAttribute('href'));
    setOpen(false);
    target.scrollIntoView({{block: 'start', behavior: 'instant'}});
    requestAnimationFrame(updateActive);
  }}));
  document.addEventListener('keydown', event => {{
    if (!isOpen) return;
    if (event.key === '/' && event.target !== searchInput) {{
      event.preventDefault();
      searchInput.focus();
    }} else if (event.key === 'Escape' && searchInput.value) {{
      searchInput.value = '';
      applyFilter();
    }} else if (event.key === 'Escape') {{
      setOpen(false, true);
    }}
  }});
  document.addEventListener('pointerdown', event => {{
    if (isOpen && !root.contains(event.target)) setOpen(false);
  }});
  window.addEventListener('scroll', queueActiveUpdate, {{passive: true}});
  window.addEventListener('resize', queueActiveUpdate, {{passive: true}});
  window.addEventListener('hashchange', queueActiveUpdate);
  root.dataset.ready = 'true';
  applyFilter();
  updateActive();
}})();
</script>
<!-- FLOATING-TOC:END -->'''


def _add_toc_heading_ids(html: str) -> str:
    for target, heading in TOC_HEADING_TARGETS.items():
        if re.search(rf'<h3\b[^>]*\bid="{re.escape(target)}"', html):
            continue
        original = f"<h3>{heading}</h3>"
        replacement = f'<h3 id="{target}">{heading}</h3>'
        html, count = re.subn(re.escape(original), lambda _: replacement, html, count=1)
        if count != 1:
            raise ValueError(f"could not add TOC target {target}")
    return html


def _remove_legacy_figure_renderings(html: str) -> str:
    """Remove old full-figure blobs and panel crops before canonical regeneration."""
    html = re.sub(
        r'\s*<figure\b(?=[^>]*\bid="nature-[^"]+")[^>]*>.*?</figure>',
        "",
        html,
        flags=re.S,
    )
    html = re.sub(
        r'\s*<p style="font-size:\.78rem;color:var\(--ink-faint\)">The displayed Figure&nbsp;1j panel.*?</p>',
        "",
        html,
        flags=re.S,
    )
    legacy_targets = {
        "nature-fig1ab": "nature-fig1",
        "nature-fig1f": "nature-fig1",
        "nature-fig1ij": "nature-fig1",
        "nature-fig1j": "nature-fig1",
        "nature-fig2gj": "nature-fig2",
        "nature-fig3fh": "nature-fig3",
        "nature-fig4fg": "nature-fig4",
        "nature-fig4il": "nature-fig4",
        "nature-ed1c": "nature-ed1",
        "nature-ed1e": "nature-ed1",
        "nature-ed1f": "nature-ed1",
        "nature-ed2c": "nature-ed2",
        "nature-ed2d": "nature-ed2",
        "nature-ed8a": "nature-ed8",
        "nature-ed8df": "nature-ed8",
    }
    for legacy, canonical in legacy_targets.items():
        html = html.replace(legacy, canonical)
    return html


def _annotate_existing_figures(html: str) -> str:
    targets = (
        ("nature-fig1", "nature", "Figure 1 from Zhong et al. 2025"),
        ("nature-fig1j", "nature-crop", "Figure 1j: percentage of selective neurons"),
        ("nature-fig4", "nature", "Figure 4 from Zhong et al. 2025"),
        ("nature-fig2", "nature", "Figure 2 from Zhong et al. 2025"),
        ("nature-fig3", "nature", "Figure 3 from Zhong et al. 2025"),
        ("nature-fig5", "nature", "Figure 5 from Zhong et al. 2025"),
    )
    for target, kind, alt_prefix in targets:
        pattern = re.compile(
            r'<figure\b[^>]*>(?=<img\b[^>]*\balt="' + re.escape(alt_prefix) + r')',
            re.S,
        )
        replacement = (
            f'<figure id="{target}" class="paperfig evidence-figure" '
            f'data-figure-target="{target}" data-figure-kind="{kind}">'
        )
        html, count = pattern.subn(replacement, html, count=1)
        if count != 1:
            raise ValueError(f"could not annotate embedded figure {target}: found {count}")
    crop_pattern = re.compile(
        r'(<figure id="nature-fig1j".*?<figcaption>)(.*?)(</figcaption>)',
        re.S,
    )

    def normalize_legacy_crop(match: re.Match[str]) -> str:
        caption = match.group(2).replace("Reproduced from", "Adapted from", 1)
        notice = " Cropped to Figure&nbsp;1 panel&nbsp;j; scientific labels and plotted values are unchanged."
        if notice not in caption:
            caption += notice
        return match.group(1) + caption + match.group(3)

    html, crop_count = crop_pattern.subn(normalize_legacy_crop, html, count=1)
    if crop_count != 1:
        raise ValueError("could not normalize legacy Figure 1j crop caption")
    return html


def _internalize_figure_references(html: str) -> str:
    target_by_page = {
        1: "nature-fig1",
        2: "nature-fig2",
        3: "nature-fig3",
        4: "nature-fig4",
        5: "nature-fig5",
        **{number + 5: f"nature-ed{number}" for number in range(1, 10)},
    }
    pattern = re.compile(
        rf'<a href="{re.escape(NATURE)}/figures/(?P<page>\d+)"[^>]*>(?P<label>.*?)</a>',
        re.S,
    )

    def replace(match: re.Match[str]) -> str:
        label = match.group("label")
        plain = re.sub(r"<[^>]+>", "", label)
        if "Open" in plain or "source figure" in plain:
            return match.group(0)
        page = int(match.group("page"))
        target = target_by_page[page]
        return _figure_ref(target, label)

    return pattern.sub(replace, html)


def _replace_once_or_keep(html: str, old: str, new: str, label: str) -> str:
    if old in html:
        return html.replace(old, new, 1)
    if new in html:
        return html
    raise ValueError(f"could not refine citation block: {label}")


def _refine_paper_citations(html: str) -> str:
    """Remove blanket Nature links and distinguish paper from release evidence."""
    main_intro_old = (
        '<p>Zhong, Baptista, Gattoni, Arnold, Flickinger, Stringer and Pachitariu recorded up to approximately 90,000 neurons simultaneously across primary visual cortex (V1) and the higher visual areas (HVAs) while mice learned a texture-discrimination task, and separately while other mice were exposed to the same textures without reward '
        f'(<a href="{NATURE}" target="_blank" rel="noopener noreferrer">Zhong et&nbsp;al. 2025</a>).</p>'
    )
    main_intro_new = (
        '<p>The paper reports 89 recordings in 19 TetO-GCaMP6s × CaMK2a-tTA mice '
        f'({_nature_cite("Sec11", "Methods: animals")}) and 20,547–89,577 Suite2p-derived traces per recording across V1 and higher visual areas '
        f'({_nature_cite("Sec2", "Results: supervised and unsupervised plasticity")}; {_nature_cite("Fig1", "Figure&nbsp;1e")}). '
        'The rewarded task and unrewarded-exposure cohorts are defined in the same Results section and timeline '
        f'({_nature_cite("Sec2", "Results section")}; {_nature_cite("Fig1", "Figure&nbsp;1a–b")}).</p>'
    )
    main_intro_prior = main_intro_new.replace(
        "The rewarded task and unrewarded-exposure cohorts",
        "The rewarded task and matched unrewarded-exposure cohorts",
    )
    if main_intro_prior in html:
        html = html.replace(main_intro_prior, main_intro_new, 1)
    html = _replace_once_or_keep(html, main_intro_old, main_intro_new, "main paper sample and cohort statement")

    main_claim_old = '<p>Neural changes in the task mice tracked behavioural learning, yet many of the same changes appeared in mice exposed to the stimuli without reward. The authors interpret most measured sensory-representation changes as consistent with unsupervised learning, while identifying a distinct task-specific reward-prediction signal.</p>'
    main_claim_new = (
        '<p>After Train&nbsp;1, the paper reports increased familiar-stimulus selectivity in task and unrewarded natural-texture cohorts, with no comparable increase after grating exposure '
        f'({_nature_cite("Sec2", "Results: supervised and unsupervised plasticity")}; {_nature_cite("Fig1", "Figure&nbsp;1i–j")}). '
        'Separately, it reports a late-versus-early cue-position response concentrated primarily in anterior visual areas of task mice after training '
        f'({_nature_cite("Sec6", "Results: reward prediction in anterior HVAs")}; {_nature_cite("Fig4", "Figure&nbsp;4e–g")}).</p>'
    )
    main_claim_prior = main_claim_new.replace(
        "a late-versus-early cue-position response concentrated primarily in anterior visual areas of task mice after training",
        "a late-versus-early cue response concentrated in anterior HVAs of task mice",
    )
    if main_claim_prior in html:
        html = html.replace(main_claim_prior, main_claim_new, 1)
    html = _replace_once_or_keep(html, main_claim_old, main_claim_new, "main paper result statement")

    source_row_old = (
        f'<tr><td>Publication</td><td><a href="{NATURE}" target="_blank" rel="noopener noreferrer">Nature article</a> · '
        f'<a href="{NATURE_DOI}" target="_blank" rel="noopener noreferrer">DOI</a> · '
        '<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC12367527/" target="_blank" rel="noopener noreferrer">open-access full text</a></td><td>Claims, methods, sample sizes, and statistics</td></tr>'
    )
    source_row_new = (
        '<tr><td>Publication</td><td>'
        f'{_nature_cite("Sec1", "Main text")} · {_nature_cite("Sec9", "Methods")} · {_nature_cite("Sec24", "Statistics and reproducibility")} · '
        f'<a href="{NATURE}#data-availability" target="_blank" rel="noopener noreferrer">Data availability&nbsp;&nearr;</a></td><td>Each claim below links to a narrower figure, Results section, or Methods subsection.</td></tr>'
    )
    source_row_prior = source_row_new.replace(
        '</a></td><td>Each claim below',
        f'</a> · <a href="{NATURE_DOI}" target="_blank" rel="noopener noreferrer">DOI</a></td><td>Each claim below',
    )
    if source_row_prior in html:
        html = html.replace(source_row_prior, source_row_new, 1)
    html = _replace_once_or_keep(html, source_row_old, source_row_new, "primary source table")

    experiment_intro_old = '<p>The imaging study used 19 mice (13 male, 6 female; 2&ndash;11 months; TetO-GCaMP6s &times; CaMK2a-tTA, GCaMP6s in excitatory neurons) across 89 recordings. Each <code>recording_id = {mouse}_{date}_{block}</code> serves as the <code>{rec}</code> key. Reported <em>n</em> varies by panel; for example <a class="figref-link" href="#nature-fig1ij" data-figure-ref="nature-fig1ij">Fig.&nbsp;1j</a> uses n&nbsp;=&nbsp;4 task, 9 unrewarded and 3 grating mice.</p>'
    experiment_intro_new = (
        '<p>The paper&rsquo;s Animals Methods report 89 recordings in 19 TetO-GCaMP6s × CaMK2a-tTA mice (13 male, 6 female; 2–11 months) '
        f'({_nature_cite("Sec11", "Methods: animals")}). The filename key <code>recording_id = {{mouse}}_{{date}}_{{block}}</code> and cohort prefixes come from the deposited '
        f'<a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer"><code>Imaging_Exp_info.npy</code>&nbsp;&nearr;</a>, not from the prose paper. '
        f'Counts are panel-specific: {_nature_cite("Fig1", "Figure&nbsp;1j")} reports <em>n</em>=4 task, 9 unrewarded natural-texture, and 3 grating mice; other panels use the counts shown explicitly below.</p>'
    )
    html = _replace_once_or_keep(html, experiment_intro_old, experiment_intro_new, "experiment source distinction")

    cohort_wrap_old = '<div class="tablewrap">\n      <table>\n        <caption>Imaging cohorts and reward conditions</caption>'
    cohort_wrap_new = '<div class="tablewrap cohort-evidence-table">\n      <table>\n        <caption>Imaging cohorts and reward conditions</caption>'
    html = _replace_once_or_keep(html, cohort_wrap_old, cohort_wrap_new, "wide cohort evidence table")

    cohort_head_old = '<thead><tr><th scope="col">Cohort</th><th scope="col">Prefix</th><th scope="col">Reward</th><th scope="col" class="num">Mice</th><th scope="col">Notes</th></tr></thead>'
    cohort_head_new = '<thead><tr><th scope="col">Paper cohort</th><th scope="col">Release prefix</th><th scope="col">Reward protocol</th><th scope="col">Panel-specific <em>n</em></th><th scope="col">Exact evidence</th></tr></thead>'
    html = _replace_once_or_keep(html, cohort_head_old, cohort_head_new, "cohort table heading")

    cohort_rows = (
        (
            '<tr><td><span class="tag t-task">Task</span></td><td><code>sup_*</code></td><td>Water associated with leaf1</td><td class="num">4–5</td><td>Train&nbsp;1 before-learning: 4 mice, all passive reward. After-learning mixes passive and active-after-cue modes.</td></tr>',
            '<tr><td><span class="tag t-task">Task</span></td><td><code>sup_*</code></td><td>Sound cue marks reward-zone onset; water follows a post-cue lick in the rewarded corridor, with passive-delay variants in some mice.</td><td><em>n</em>=4 in Fig.&nbsp;1j; <em>n</em>=5 in Fig.&nbsp;2e,f,j.</td><td>'
            f'{_nature_cite("Sec17", "Reward protocol")}; {_nature_cite("Fig1", "Figure&nbsp;1j sample size")}; {_nature_cite("Fig2", "Figure&nbsp;2e,f,j sample sizes")}; '
            f'<a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer">prefix and <code>rewType</code> rows&nbsp;&nearr;</a></td></tr>',
        ),
        (
            '<tr><td><span class="tag t-uns">Unrewarded</span></td><td><code>unsup_*</code></td><td>None</td><td class="num">6–9</td><td>Identical corridors, no water; sound cue still presented. leaf1 is the same stimulus, unrewarded</td></tr>',
            '<tr><td><span class="tag t-uns">Unrewarded</span></td><td><code>unsup_*</code></td><td>Same imaging corridors and sound cue; no rewards.</td><td><em>n</em>=9 in Fig.&nbsp;1j; <em>n</em>=7 in Fig.&nbsp;2f; <em>n</em>=6 in Fig.&nbsp;3b,e,h.</td><td>'
            f'{_nature_cite("Sec17", "Unrewarded protocol")}; {_nature_cite("Fig1", "Figure&nbsp;1j sample size")}; {_nature_cite("Fig2", "Figure&nbsp;2f sample size")}; {_nature_cite("Fig3", "Figure&nbsp;3b,e,h sample sizes")}; '
            f'<a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer">release prefix rows&nbsp;&nearr;</a></td></tr>',
        ),
        (
            '<tr><td><span class="tag t-ctrl">Grating control</span></td><td><code>*_grating</code></td><td>None</td><td class="num">3</td><td>Grating walls (0°, 45°); 5–6 sessions</td></tr>',
            '<tr><td><span class="tag t-ctrl">Grating control</span></td><td><code>*_grating</code></td><td>Unrewarded 0°/45° grating exposure; neural responses are tested on naturalistic pairs before and after.</td><td><em>n</em>=3 (5 sessions) in Fig.&nbsp;1j; <em>n</em>=3 (6 sessions) in Fig.&nbsp;3e,h.</td><td>'
            f'{_nature_cite("Sec14", "Grating and naturalistic stimulus protocol")}; {_nature_cite("Fig1", "Figure&nbsp;1j sample size")}; {_nature_cite("Fig3", "Figure&nbsp;3e,h sample sizes")}; '
            f'<a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer">release prefix rows&nbsp;&nearr;</a></td></tr>',
        ),
        (
            '<tr><td><span class="tag t-ctrl">Naive</span></td><td><code>naive_*</code></td><td>Untrained</td><td class="num">7–9</td><td>11 sessions (each mouse sees more than one pair)</td></tr>',
            '<tr><td><span class="tag t-ctrl">Naive</span></td><td><code>naive_*</code></td><td>No training or exposure; more than one naturalistic pair can be imaged per mouse.</td><td><em>n</em>=9 mice (11 sessions) in Fig.&nbsp;2f and Fig.&nbsp;3e,h.</td><td>'
            f'{_nature_cite("Sec14", "Naive stimulus protocol")}; {_nature_cite("Fig2", "Figure&nbsp;2f sample size")}; {_nature_cite("Fig3", "Figure&nbsp;3e,h sample sizes")}; '
            f'<a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer">release prefix rows&nbsp;&nearr;</a></td></tr>',
        ),
    )
    for index, (old, new) in enumerate(cohort_rows, start=1):
        html = _replace_once_or_keep(html, old, new, f"cohort table row {index}")

    imaging_card_old = (
        '<p>Two-photon mesoscope with temporal multiplexing, recording V1 and many higher visual areas simultaneously; 20,547&ndash;89,577 neurons per recording. Processing used Suite2p followed by non-negative deconvolution (decay 0.75&nbsp;s). Neural analyses use the deconvolved traces or their released 400-component reduction.</p>'
    )
    imaging_card_new = (
        '<p>The acquisition used a two-photon random-access mesoscope and temporal multiplexing '
        f'({_nature_cite("Sec13", "Methods: imaging acquisition")}). The first Results section reports simultaneous V1/HVA recordings containing 20,547–89,577 traces '
        f'({_nature_cite("Sec2", "Results: recording scale")}; {_nature_cite("Fig1", "Figure&nbsp;1c–e")}). Suite2p processing and non-negative deconvolution with a 0.75&nbsp;s decay parameter are specified in '
        f'{_nature_cite("Sec19", "Methods: processing of calcium imaging data")}. The 400-component SVD files are a deposited-release representation, not a paper Methods claim '
        f'(<a href="{FIGSHARE}" target="_blank" rel="noopener noreferrer">Figshare v2 release&nbsp;&nearr;</a>).</p>'
    )
    html = _replace_once_or_keep(html, imaging_card_old, imaging_card_new, "imaging data card evidence")

    behavior_card_old = (
        '<p><code>ft_trInd</code> trial, <code>ft_WallID</code> texture, <code>ft_move</code> running, <code>ft_CorrSpc</code> in-texture, <code>ft_Pos</code>/<code>ft_PosCum</code> position, <code>ft_RunSpeed</code>. Per trial: <code>WallName</code>, <code>UniqWalls</code>, <code>stim_id</code>, <code>isRew</code>, <code>ntrials</code>, <code>SoundDelPos</code>.</p>'
    )
    behavior_card_new = (
        '<p><strong>Released-field inventory:</strong> <code>ft_trInd</code> trial, <code>ft_WallID</code> texture, <code>ft_move</code> running, <code>ft_CorrSpc</code> in-texture, <code>ft_Pos</code>/<code>ft_PosCum</code> position, and <code>ft_RunSpeed</code>; per trial, <code>WallName</code>, <code>UniqWalls</code>, <code>stim_id</code>, <code>isRew</code>, <code>ntrials</code>, and <code>SoundDelPos</code>. These field names come from the deposited arrays and the checksum-pinned release audit, not from the article prose '
        f'(<a href="{FIGSHARE}" target="_blank" rel="noopener noreferrer">Figshare v2 release&nbsp;&nearr;</a>; <a href="#atlas-experiment-labels">release atlas&nbsp;↓</a>). The paper defines the corresponding behavioral protocol in {_nature_cite("Sec17", "Methods: behavioural training")}.</p>'
    )
    html = _replace_once_or_keep(html, behavior_card_old, behavior_card_new, "behavior-field provenance")

    area_ids_old = (
        '<p><code>V1 = iarea==8</code>; <code>mHV (medial) = {0,1,2,9}</code>, grouping PM, AM, MMA and lateral retrosplenial cortex; <code>lHV (lateral) = {5,6}</code>; <code>aHV (anterior) = {3,4}</code>. Identifiers <code>-1</code> and <code>7</code> lie outside the included visual-cortical groups. The largest overall selective-fraction change is medial; the whole-V1 fraction changes less, although subregions and stimulus poles can differ.</p>'
    )
    area_ids_new = (
        '<p><strong>Paper-code mapping:</strong> <code>V1 = iarea==8</code>; <code>mHV (medial) = {0,1,2,9}</code>, grouping PM, AM, MMA, and lateral retrosplenial cortex; <code>lHV (lateral) = {5,6}</code>; <code>aHV (anterior) = {3,4}</code>; identifiers <code>-1</code> and <code>7</code> are excluded from those groups '
        '(<a href="https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L312-L324" target="_blank" rel="noopener noreferrer">paper code: <code>neu_area_ID</code>&nbsp;lines&nbsp;312–324&nbsp;&nearr;</a>). '
        'The regional before/after comparison reports the clearest familiar-selective fraction increase in medial HVAs, while the whole-V1 change is smaller '
        f'({_nature_cite("Sec2", "Results: regional plasticity")}; {_nature_cite("Fig1", "Figure&nbsp;1i–j")}).</p>'
    )
    html = _replace_once_or_keep(html, area_ids_old, area_ids_new, "visual-area mapping and result evidence")

    valid_frames_old = (
        '<dt>valid frames</dt><dd><code>(ft_move&gt;0) &amp; ft_CorrSpc</code> &mdash; the paper-compatible running and 0&ndash;4&nbsp;m texture mask '
        f'(<a href="{NATURE}#Sec9" target="_blank" rel="noopener noreferrer">Methods</a>)</dd>'
    )
    valid_frames_new = (
        '<dt>valid frames</dt><dd><code>(ft_move&gt;0) &amp; ft_CorrSpc</code> &mdash; the running, original-trace, 0&ndash;4&nbsp;m texture mask specified in '
        f'{_nature_cite("Sec20", "Methods: neural selectivity")}</dd>'
    )
    html = _replace_once_or_keep(html, valid_frames_old, valid_frames_new, "d-prime valid-frame citation")

    support_mask_old = (
        '<li><strong>Apply the published support mask.</strong> Keep <code>(ft_move&gt;0) &amp; ft_CorrSpc</code>, matching the running, 0&ndash;4&nbsp;m texture analysis in the '
        f'<a href="{NATURE}#Sec9" target="_blank" rel="noopener noreferrer">Nature Methods</a>.</li>'
    )
    support_mask_new = (
        '<li><strong>Apply the published support mask.</strong> Keep <code>(ft_move&gt;0) &amp; ft_CorrSpc</code>, matching the original-trace, running, 0&ndash;4&nbsp;m estimator in '
        f'{_nature_cite("Sec20", "Methods: neural selectivity")}.</li>'
    )
    html = _replace_once_or_keep(html, support_mask_old, support_mask_new, "analysis support-mask citation")

    constants_old = (
        '<p>|d&prime;|≥0.3 selective · reduced release representation: 400 components · deconvolution decay 0.75 s · run threshold 6 cm/s for at least 66 ms · imaging corridor: 4 m texture + 2 m grey with sound cue sampled from 0.5–3.5 m · behaviour-only task: reward-zone start sampled from 2–3 m · 20,547–89,577 neurons per recording. The paper uses paired or independent two-sided Student&rsquo;s t-tests as specified in '
        f'<a href="{NATURE}#Sec9" target="_blank" rel="noopener noreferrer">Methods</a> and reports no multiple-comparison adjustment; the proposed distribution analysis instead uses the locked mouse-level hierarchy in &sect;10.</p>'
    )
    constants_new = (
        '<p><strong>Paper constants, each with its exact location:</strong> |d&prime;|≥0.3 and the running 0–4&nbsp;m selectivity mask '
        f'({_nature_cite("Sec20", "neural-selectivity Methods")}); 0.75&nbsp;s deconvolution decay '
        f'({_nature_cite("Sec19", "calcium-processing Methods")}); 6&nbsp;cm&nbsp;s<sup>−1</sup> run threshold and 4&nbsp;m texture + 2&nbsp;m grey corridor '
        f'({_nature_cite("Sec14", "visual-stimulus Methods")}); 0.5–3.5&nbsp;m imaging cue and 2–3&nbsp;m behavior-only reward-zone start '
        f'({_nature_cite("Sec17", "behavioural-training Methods")}); 20,547–89,577 traces per recording '
        f'({_nature_cite("Sec2", "plasticity Results")}); paired/independent two-sided Student&rsquo;s <em>t</em>-tests and no multiplicity adjustment '
        f'({_nature_cite("Sec24", "statistics and reproducibility")}). The 400-component SVD is a Figshare-v2 release derivative, not a paper Methods constant '
        f'(<a href="{FIGSHARE}" target="_blank" rel="noopener noreferrer">release inventory&nbsp;&nearr;</a>). The proposed distribution analysis instead uses the locked mouse-level hierarchy in &sect;15.</p>'
    )
    html = _replace_once_or_keep(html, constants_old, constants_new, "paper constants source map")

    # Claims that come from specific Methods subsections should never point to
    # the Methods landing heading. These replacements cover the older static
    # sections that predate this generator.
    targeted_links = (
        ("These are processed neural traces used by the publication, not raw two-photon movies", "Sec19", "Methods: processing of calcium imaging data"),
        ("The publication reports 20,547&ndash;89,577 neurons per recording and bases its neural analyses on deconvolved fluorescence traces", "Sec19", "Methods: calcium processing"),
    )
    for claim, anchor, label in targeted_links:
        pattern = re.compile(
            rf'({re.escape(claim)}) \(<a href="{re.escape(NATURE)}" target="_blank" rel="noopener noreferrer">(?:Nature )?Methods</a>\)'
        )
        replacement = rf'\1 ({_nature_cite(anchor, label)})'
        html, count = pattern.subn(replacement, html, count=1)
        if count == 0 and replacement.replace(r"\1", claim) not in html:
            raise ValueError(f"could not refine targeted Methods claim: {claim}")

    # Source links for main figures use the figure anchor in the article. The
    # standalone Extended Data pages remain the narrowest public targets.
    for number in range(1, 6):
        html = html.replace(
            f'href="{NATURE}/figures/{number}"',
            f'href="{NATURE}#Fig{number}"',
        )

    # Normalize every legacy internal Nature-figure reference through the same
    # dual-link helper: one jump to the inline evidence, one exact paper URL.
    # Strip already-generated groups first so reruns remain byte-stable.
    group_pattern = re.compile(
        r'<span class="figref-group">(?P<internal><a class="figref-link" href="#[^"]+" data-figure-ref="[^"]+">.*?</a>)'
        r'<a class="figref-source"[^>]*>.*?</a></span>',
        re.S,
    )
    html = group_pattern.sub(lambda match: match.group("internal"), html)
    legacy_ref_pattern = re.compile(
        r'<a class="figref-link" href="#(?P<target>nature-[^"]+)" data-figure-ref="(?P=target)">(?P<label>.*?)</a>',
        re.S,
    )
    html = legacy_ref_pattern.sub(
        lambda match: _figure_ref(match.group("target"), match.group("label")),
        html,
    )

    # Any remaining bare article link is navigation/bibliography, not support
    # for a specific scientific claim. Give it the narrowest defensible anchor
    # based on its visible label, and fail if a new generic label appears.
    bare_pattern = re.compile(
        rf'<a href="{re.escape(NATURE)}"(?P<attrs>[^>]*)>(?P<label>.*?)</a>',
        re.S,
    )

    def anchor_bare_link(match: re.Match[str]) -> str:
        label_html = match.group("label")
        plain = re.sub(r"<[^>]+>", "", label_html)
        plain = plain.replace("&nbsp;", " ").strip().lower()
        if "data availability" in plain:
            anchor = "data-availability"
        elif "neural selectivity" in plain:
            anchor = "Sec20"
        elif "method" in plain:
            anchor = "Sec9"
        elif "zhong" in plain:
            anchor = "Abs1"
        elif plain in {"nature article", "nature paper", "article"}:
            anchor = "Sec1"
        else:
            raise ValueError(f"unclassified bare Nature link: {plain}")
        return f'<a href="{NATURE}#{anchor}"{match.group("attrs")}>{label_html}</a>'

    html = bare_pattern.sub(anchor_bare_link, html)
    assert f'href="{NATURE}"' not in html
    assert not re.search(rf'href="{re.escape(NATURE)}/figures/[1-5]"', html)
    return html


def _correct_legacy_scientific_copy(html: str) -> str:
    """Replace categorical, ambiguous, or editorial legacy wording."""
    replacements = (
        (
            '<tr><td>Behaviour</td><td><code>Beh_{experiment}.npy</code></td><td>dict: session &rarr; per-frame arrays</td><td class="num">108.1–412.5 MiB</td></tr>',
            '<tr><td>Imaging behaviour</td><td><code>Beh_{experiment}.npy</code></td><td>23 experiment bundles linked to imaging acquisitions; session dictionaries contain the aligned frame/trial fields documented below</td><td class="num">108.1–412.5 MiB</td></tr>'
            '<tr><td>Behaviour-only learning</td><td><code>Beh_no_pretrain.npy</code>, <code>Beh_pretrain_on_grat_image.npy</code>, <code>Beh_pretrain_on_nat_image.npy</code></td><td>3 bundles for the 23 additional mice reported in Figure&nbsp;5; no neural or retinotopy files</td><td class="num">378.7–817.9 MiB</td></tr>',
        ),
        (
            '<tr><td>Index</td><td><code>Imaging_Exp_info.npy</code></td><td>142 experiment&ndash;recording memberships</td><td class="num">21 KB</td></tr>',
            '<tr><td>Index</td><td><code>Imaging_Exp_info.npy</code></td><td>142 source metadata rows; 133 unique experiment&ndash;recording pairs</td><td class="num">21 KB</td></tr>',
        ),
        (
            f'<li><code>xy_t</code>: transformed cortical coordinates aligned neuron-for-neuron with <code>iarea</code>. The paper density recipe uses <code>x = −xy_t[:,1]</code> and <code>y = xy_t[:,0]</code> before rasterization ({_figure_ref("nature-fig1", "Fig.&nbsp;1g–i")}).</li>',
            '<li><code>xy_t</code>: transformed cortical coordinates aligned neuron-for-neuron with <code>iarea</code>. The paper code&rsquo;s <a href="https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416" target="_blank" rel="noopener noreferrer"><code>Get_density_map</code>&nbsp;lines&nbsp;394–416&nbsp;&nearr;</a> uses <code>x = −xy_t[:,1]</code> and <code>y = xy_t[:,0]</code> before rasterization; <a href="https://www.nature.com/articles/s41586-025-09180-y#Fig1" target="_blank" rel="noopener noreferrer">Figure&nbsp;1i&nbsp;&nearr;</a> is the plotted density-map output.</li>',
        ),
        (
            '<p class="shows">Anterior (aHV) reward-prediction fraction rises only in task mice (paired t: task p=0.0069, unrewarded p=0.708). The reward signal is anterior, not medial.</p>',
            f'<p class="shows">Figure&nbsp;4g reports the anterior-region fraction of neurons meeting the late-versus-early cue-position threshold (paired test: task <em>P</em>=0.0069; unrewarded <em>P</em>=0.708). The Results state that selected neurons were distributed primarily in anterior areas of task mice after training ({_nature_cite("Sec6", "Results")}; {_nature_cite("Fig4", "Figure&nbsp;4f–g")}).</p>',
        ),
        (
            '<p class="shows">Density maps place the reward signal in anterior HVAs of task-after only (f); anterior neurons are higher on lick trials in the unrewarded leaf2 corridor (k, p=0.014), tracking expectation.</p>',
            f'<p class="shows">Figure&nbsp;4f shows selected neurons primarily in anterior areas of task mice after training. Figure&nbsp;4k reports higher selected-population activity on leaf2 trials with licks than on trials without licks (<em>P</em>=0.014); the authors interpret this together with the cue- and lick-aligned dynamics as indicative of reward expectation ({_nature_cite("Sec6", "Results")}; {_nature_cite("Fig4", "Figure&nbsp;4f,k")}).</p>',
        ),
        (
            '<p class="shows">Top-5% leaf1-selective (g) and bottom-5% circle1-selective (h) medial populations by position; sequence correlation (f) is high for even-leaf1, low for leaf2 in medial — stimulus-specific, not spatial.</p>',
            f'<p class="shows">Panels g–h show leaf1- and circle1-selective medial populations by position. Panels d–f report low preferred-position correlations between leaf1 and leaf2; panels i–j separately show category separation along the leaf1–circle1 coding direction ({_nature_cite("Sec3", "Results: visual, not spatial, representations")}; {_nature_cite("Fig2", "Figure&nbsp;2d–j")}).</p>',
        ),
    )
    for old, new in replacements:
        html = html.replace(old, new)
    html = re.sub(
        r'<p>The task and unrewarded imaging cohorts follow the staged Train/Test sequence below,.*?</p>',
        f'<p>The task and unrewarded imaging cohorts follow the staged Train/Test sequence shown in {_figure_ref("nature-fig1", "Figure&nbsp;1b")}. The deposited <a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer"><code>Imaging_Exp_info.npy</code> experiment index&nbsp;&nearr;</a> gives matched Train&nbsp;1 acquisition gaps of 3&ndash;24 days overall (supervised: 9&ndash;24 days; unrewarded: 3&ndash;12 days). Naive and grating controls contribute comparison sessions rather than passing through every task stage; the release index contains 23 experiment labels spanning cohort, stage, and learning status.</p>',
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r'<div class="note finding">\s*<span class="k">Two central findings</span>.*?</div>',
        "",
        html,
        count=1,
        flags=re.S,
    )
    replacements = (
        (
            "A complete, source-linked atlas of every released imaging acquisition, behavior and retinotopy relationship, plus rigorous analysis plans for distributional selectivity change and reward-associated within-session dynamics.",
            "Source-linked map of the published results, complete release inventory, data provenance, and two explicitly identified proposed analyses of selectivity distributions and within-session dynamics.",
        ),
        (
            '<meta name="description" content="Complete source-linked Zhong et al. 2025 data atlas: every imaging mouse, experiment, neural acquisition, behavior and retinotopy relationship, both research questions, and an executable analysis plan grounded in the Nature paper and large-scale-recording methods review.">',
            '<meta name="description" content="Source-linked Zhong et al. 2025 paper figures, release inventory, data provenance, and explicit analysis specifications.">',
        ),
        (
            "<title>Zhong et al. 2025 — complete neural data atlas and two-question analysis plan</title>",
            "<title>Zhong et al. 2025 — paper figures, release atlas, and analysis specifications</title>",
        ),
        ("<b>20.5k–89.6k</b> neurons/rec", "<b>20,547–89,577</b> Suite2p traces/recording"),
        ('<div class="eyebrow">The finding</div>', '<div class="eyebrow">Nature paper</div>'),
        (
            "<h2>Most measured visual-cortical plasticity was reproduced without reward</h2>",
            "<h2>Unsupervised pretraining in biological neural networks</h2>",
        ),
        ('<th scope="col">Use here</th>', '<th scope="col">Scope in this reference</th>'),
        (
            '<h3 id="dprime-trial-resolved">Recommended trial-resolved definition</h3>',
            '<h3 id="dprime-trial-resolved">Trial-resolved specification</h3>',
        ),
        (
            '<h3 id="analysis-primary">① Lock the cohort and file manifest <span class="relpill rel-high" style="vertical-align:middle">required first</span></h3>',
            '<h3 id="analysis-primary">1. Cohort and file manifest</h3>',
        ),
        (
            '<h3 id="analysis-alignment">② Align frames and construct trial responses <span class="relpill rel-high" style="vertical-align:middle">paper-anchored</span></h3>',
            '<h3 id="analysis-alignment">2. Frame alignment and trial responses</h3>',
        ),
        (
            '<h3 id="analysis-pairing">③ Pair trials and form balanced windows <span class="relpill rel-high" style="vertical-align:middle">primary estimator</span></h3>',
            '<h3 id="analysis-pairing">3. Trial pairing and balanced windows</h3>',
        ),
        (
            '<h3 id="analysis-distributions">④ Describe every across-neuron distribution <span class="relpill rel-high" style="vertical-align:middle">all four strata</span></h3>',
            '<h3 id="analysis-distributions">4. Across-neuron distributions</h3>',
        ),
        (
            '<h3 id="analysis-graph">⑤ Interactive graph contract <span class="relpill rel-med" style="vertical-align:middle">playground</span></h3>',
            '<h3 id="analysis-graph">5. Interactive graph</h3>',
        ),
        (
            '<h3 id="analysis-inference">⑥ Make the mouse the inferential unit <span class="relpill rel-high" style="vertical-align:middle">confirmatory</span></h3>',
            '<h3 id="analysis-inference">6. Mouse-level inference</h3>',
        ),
        (
            '<h3 id="analysis-validation">⑦ Validation and sensitivity ladder <span class="relpill rel-med" style="vertical-align:middle">required reporting</span></h3>',
            '<h3 id="analysis-validation">7. Validation and sensitivity</h3>',
        ),
        (
            '<h3 id="workflow-order">Recommended execution order</h3>',
            '<h3 id="workflow-order">Execution order</h3>',
        ),
    )
    for old, new in replacements:
        html = html.replace(old, new)
    html = html.replace("publication-faithful", "paper-estimator")
    html = html.replace("Publication-faithful", "Paper-estimator")
    html = re.sub(
        r'<tr><td>Train&nbsp;2</td><td>~1 week</td><td>circle1, leaf1, leaf2</td><td>.*?</td></tr>',
        lambda _: (
            '<tr><td>Train&nbsp;2</td><td>~1 week</td><td>circle1, leaf1, leaf2</td><td>'
            'Fine-discrimination training preceded Test&nbsp;2. After training, the leaf2 representation had a smaller projection onto the leaf1&ndash;circle1 coding direction, with the largest reported change in medial HVAs '
            f'({_nature_cite("Sec4", "Results: novelty and orthogonalization")}; {_nature_cite("Fig3", "Figure&nbsp;3f–h")}).'
            '</td></tr>'
        ),
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r'<p><strong>Primary area:</strong> medial HVAs, chosen in advance because.*?</p>',
        lambda _: (
            '<p><strong>Proposed primary area:</strong> medial HVAs, because Figure&nbsp;1j reports the largest familiar-selectivity changes in that grouping '
            f'({_nature_cite("Sec2", "Results")}; {_nature_cite("Fig1", "Figure&nbsp;1i–j")}). '
            'This is a design choice for the new distribution analysis, not a primary endpoint designated by the paper. V1, lateral HVA, and anterior HVA remain separate regional analyses. The anterior late-versus-early cue-position statistic is evaluated separately '
            f'({_nature_cite("Sec6", "Results: reward prediction")}; {_nature_cite("Fig4", "Figure&nbsp;4")}).</p>'
        ),
        html,
        count=1,
        flags=re.S,
    )
    extended_links = " · ".join(
        _figure_ref(f"nature-ed{number}", f"ED&nbsp;Fig.&nbsp;{number}")
        for number in range(1, 10)
    )
    html = re.sub(
        r'<tr><td>Extended Data</td><td>.*?</td><td>Threshold, locomotion, and within-day behavioural precedents</td></tr>',
        f'<tr><td>Extended Data</td><td>{extended_links}</td><td>Complete supporting figures, placed beside the corresponding main result.</td></tr>',
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r'\s*<p class="figure-source"><a href="https://www\.nature\.com/articles/s41586-025-09180-y#Fig1".*?</p>',
        "",
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r'<p>(?:The paper reports plasticity mainly as a before-versus-after endpoint comparison|The paper reports before-versus-after selectivity and cortical-distribution endpoints).*?<a href="#within">Research question(?:&nbsp;)?1</a>\.</p>',
        f'<p>The paper reports before-versus-after selectivity and cortical-distribution endpoints at the Train&nbsp;1 landmarks ({_figure_ref("nature-fig1", "Figure&nbsp;1b,i,j")}). In the deposited <a href="{IMAGING_INDEX_SOURCE}" target="_blank" rel="noopener noreferrer"><code>Imaging_Exp_info.npy</code> experiment index&nbsp;&nearr;</a>, matched Train&nbsp;1 acquisitions are 3&ndash;24 days apart overall (supervised: 9&ndash;24 days; unrewarded: 3&ndash;12 days). Figure&nbsp;1i&ndash;j summarizes before/after selective-neuron fractions rather than chronological within-session <em>d</em>&prime; distributions; that separate proposed analysis is defined in <a href="#within">Research question&nbsp;1</a>.</p>',
        html,
        count=1,
        flags=re.S,
    )
    return html


def _renumber_sections(html: str) -> str:
    order = [
        "paper", "experiment", "stimuli", "data", "coverage", "atlas", "support",
        "environment", "methods-review", "dprime", "within", "reward-rate", "questions",
        "figuremap", "analyses", "workflow", "recipes", "caveats",
    ]
    for number, section_id in enumerate(order, start=1):
        start = html.index(f'<section id="{section_id}">')
        next_start = html.find("<section id=", start + 1)
        if next_start < 0:
            next_start = html.index("</main>", start)
        chunk = html[start:next_start]
        chunk, count = re.subn(r'<span class="idx">\d+</span>', f'<span class="idx">{number:02d}</span>', chunk, count=1)
        if count != 1:
            raise ValueError(f"could not renumber {section_id}")
        html = html[:start] + chunk + html[next_start:]
    return html


def main() -> None:
    inventory_bytes = INVENTORY_PATH.read_bytes()
    index_bytes = INDEX_PATH.read_bytes()
    assert sha256(inventory_bytes).hexdigest() == EXPECTED_INVENTORY_SHA256
    assert sha256(index_bytes).hexdigest() == EXPECTED_INDEX_SHA256
    inventory = json.loads(inventory_bytes)
    index = json.loads(index_bytes)
    html = HTML_PATH.read_text(encoding="utf-8")
    html = _remove_legacy_figure_renderings(html)

    # Replace previous generated sections if this script is rerun.
    for section_id in ("atlas", "support", "methods-review", "reward-rate", "questions", "workflow"):
        # Consume all inter-section whitespace around generated sections.  A
        # one-newline boundary left older runs accumulating blank lines on
        # every regeneration even though the rendered document was unchanged.
        pattern = re.compile(
            rf'(?:[ \t]*\n)+<section id="{re.escape(section_id)}">.*?</section>(?:[ \t]*\n)+',
            re.S,
        )
        html = pattern.sub("\n\n", html)

    html = _insert_in_section(html, "paper", "PAPER-ENRICHMENT", _paper_enrichment())
    html = _insert_in_section(html, "experiment", "EXPERIMENT-ENRICHMENT", _experiment_enrichment())
    html = _insert_in_section(html, "data", "NEURAL-FRAME-DETAIL", _neural_frame_detail())
    html = _insert_in_section(html, "environment", "ENVIRONMENT-ENRICHMENT", _environment_enrichment())
    html = _insert_in_section(html, "dprime", "DPRIME-FIGURE", _dprime_figure_repeat())
    html = _insert_in_section(html, "within", "WITHIN-FIGURE-REPEATS", _within_figure_repeats())
    html = _insert_in_section(html, "analyses", "ANALYSIS-QC-FIGURES", _analysis_qc_figure_repeats())
    html = _insert_in_section(html, "recipes", "CODE-WALKTHROUGH", _paper_code_walkthrough())

    html = _replace_between_sections(html, "coverage", "environment", _atlas_section(inventory, index) + "\n\n" + _support_section(inventory, index))
    html = _replace_between_sections(html, "environment", "dprime", _methods_section())
    html = _replace_between_sections(html, "within", "figuremap", _reward_section() + "\n\n" + _questions_section())
    html = _replace_between_sections(html, "analyses", "recipes", _workflow_section())
    figuremap_pattern = re.compile(r'<section id="figuremap">.*?</section>', re.S)
    html, figuremap_count = figuremap_pattern.subn(lambda _: _figure_map_section(), html, count=1)
    assert figuremap_count == 1
    html = _add_toc_heading_ids(html)

    toc_pattern = re.compile(r'    <nav(?: id="contents")? class="toc" aria-label="Contents">.*?    </nav>', re.S)
    html, toc_count = toc_pattern.subn(_toc(), html, count=1)
    assert toc_count == 1

    floating_toc = _floating_toc()
    floating_start = "<!-- FLOATING-TOC:START -->"
    floating_end = "<!-- FLOATING-TOC:END -->"
    if floating_start in html:
        html = re.sub(
            re.escape(floating_start) + r".*?" + re.escape(floating_end),
            lambda _: floating_toc,
            html,
            count=1,
            flags=re.S,
        )
    else:
        html = html.replace("\n</body>", "\n" + floating_toc + "\n\n</body>", 1)

    html = _internalize_figure_references(html)

    css = '''
  .result-guide__intro{max-width:62rem;margin-bottom:.75rem;color:var(--ink-soft);font-size:1rem;line-height:1.7}
  .figure-attribution{max-width:62rem;margin:.7rem 0 1.5rem;padding:.8rem 1rem;border-left:2px solid var(--fluoro);background:var(--raise);color:var(--ink-soft);font-size:.82rem;line-height:1.6}
  .result-guide{display:grid;gap:1.45rem;margin:1.25rem 0 2.25rem}
  .result-card{scroll-margin-top:1.1rem;overflow:hidden;border:1px solid var(--line);border-radius:14px;background:var(--surface);box-shadow:0 12px 34px rgba(0,0,0,.09)}
  .result-card__head{display:grid;grid-template-columns:3.1rem minmax(0,1fr);gap:.9rem;padding:1.2rem 1.25rem 1.05rem;border-bottom:1px solid var(--line);background:var(--raise)}
  .result-card__index{display:grid;place-items:center;align-self:start;width:2.7rem;height:2.7rem;border:1px solid color-mix(in srgb,var(--fluoro) 54%,var(--line));border-radius:999px;background:var(--fluoro-soft);color:var(--fluoro-ink);font-family:var(--mono);font-size:.78rem;font-weight:800;letter-spacing:.06em}
  .result-card__head h4{margin:.08rem 0 .35rem;font-family:var(--serif);font-size:clamp(1.25rem,2vw,1.65rem);line-height:1.2;color:var(--ink)}
  .result-card__head p:last-child{max-width:61rem;margin:0;color:var(--ink-soft);font-size:.91rem;line-height:1.6}
  .result-card__eyebrow{margin:0!important;color:var(--fluoro-ink)!important;font-family:var(--mono);font-size:.67rem!important;font-weight:760;letter-spacing:.1em;text-transform:uppercase}
  .result-card>.evidence-figure{margin:0;padding:1rem 1rem .9rem;border-top:1px solid var(--line);background:var(--paper)}
  .result-card>.evidence-figure+.evidence-figure{margin-top:0;border-top:1px solid var(--line)}
  .result-card>.evidence-figure img{width:100%;height:auto;max-height:none;object-fit:contain;border-radius:6px}
  .result-card>.evidence-figure figcaption{max-width:none!important;margin:.7rem 0 0;padding:0;color:var(--ink-soft);font-size:.76rem;line-height:1.55}
  .result-card__body{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(18rem,.8fr);gap:1.2rem;padding:1.15rem 1.25rem 1.3rem}
  .result-panel-list{display:grid;gap:0;margin:0}
  .result-panel-list>div{display:grid;grid-template-columns:minmax(9.8rem,.72fr) minmax(0,1.8fr);gap:.8rem;padding:.68rem 0;border-top:1px solid var(--line-soft)}
  .result-panel-list>div:first-child{padding-top:.15rem;border-top:0}
  .result-panel-list dt{color:var(--ink);font-size:.78rem;font-weight:720;line-height:1.45}
  .result-panel-list dd{margin:0;color:var(--ink-soft);font-size:.82rem;line-height:1.58}
  .result-card__takeaway{align-self:start;padding:1rem;border:1px solid color-mix(in srgb,var(--fluoro) 30%,var(--line));border-radius:10px;background:var(--fluoro-soft)}
  .result-card__takeaway p{margin:.3rem 0 .85rem;font-size:.82rem;line-height:1.6;color:var(--ink-soft)}
  .result-card__takeaway p:last-child{margin-bottom:0}
  .result-card__kicker{margin:.75rem 0 .15rem!important;color:var(--fluoro-ink)!important;font-family:var(--mono);font-size:.64rem!important;font-weight:800;letter-spacing:.1em;text-transform:uppercase}
  .result-card__kicker:first-child{margin-top:0!important}
  .result-card__boundary{padding-top:.75rem;border-top:1px solid color-mix(in srgb,var(--fluoro) 30%,var(--line));color:var(--ink)!important}
  .result-card--methods .result-card__index{border-radius:8px}
  .paper-cite{font-family:var(--mono);font-size:.88em;font-weight:650;text-underline-offset:.18em}
  .figref-group{display:inline-flex;align-items:baseline;flex-wrap:wrap;gap:.22rem;vertical-align:baseline}
  .figref-source{font-family:var(--mono);font-size:.68em;font-weight:700;text-decoration:none;white-space:nowrap}
  .figref-source:hover{text-decoration:underline;text-underline-offset:.18em}
  .result-panel-list dt .paper-cite{font-size:1em}
  .cohort-evidence-table{width:min(64rem,calc(100vw - 2.8rem));margin-left:50%;transform:translateX(-50%)}
  .cohort-evidence-table table{min-width:58rem;table-layout:fixed}
  .cohort-evidence-table th:nth-child(1){width:14%}.cohort-evidence-table th:nth-child(2){width:12%}.cohort-evidence-table th:nth-child(3){width:25%}.cohort-evidence-table th:nth-child(4){width:18%}.cohort-evidence-table th:nth-child(5){width:31%}
  .filemeta{display:block;margin-top:.22rem;font-family:var(--mono);font-size:.66rem;line-height:1.45;color:var(--ink-faint);overflow-wrap:anywhere}
  code{overflow-wrap:anywhere;word-break:break-word}
  .panel .meta>*{min-width:0;overflow-wrap:anywhere;word-break:break-word}.panel .meta .recipe{min-width:0;overflow-wrap:anywhere;word-break:break-word}.kv dd{min-width:0;overflow-wrap:anywhere}
  details.ref>summary,.fig>summary{max-width:100%;overflow:hidden;position:relative;padding-right:2.7rem}details.ref>summary::after,.fig>summary::after{position:absolute;right:1rem;top:50%;transform:translateY(-50%)}details.ref[open]>summary::after,.fig[open]>summary::after{transform:translateY(-50%) rotate(90deg)}details.ref>summary span,.fig>summary .ttl{min-width:0;overflow-wrap:anywhere;word-break:break-word}.recipe-item,.recipe-item h4{min-width:0;overflow-wrap:anywhere;word-break:break-word}details.ref .body{min-width:0}
  .atlas-table{margin:.4rem 0 0;border:0;border-radius:0}.atlas-table table{min-width:112rem;font-size:.78rem}.atlas-table th,.atlas-table td{padding:.55rem .62rem}
  .experiment-table table{min-width:84rem}.recording-list{font-size:.72rem;line-height:1.65}.recording-list code{white-space:nowrap}
  .membership{padding:.45rem 0;border-top:1px solid var(--line-soft)}.membership:first-child{padding-top:0;border-top:0}.membership-title{font-weight:650;margin-bottom:.25rem}
  .rawmeta{display:flex;flex-wrap:wrap;gap:.18rem .5rem;color:var(--ink-soft);font-size:.69rem;line-height:1.45}.rawmeta span{overflow-wrap:anywhere}.rawmeta b{font-family:var(--mono);font-weight:650;color:var(--ink-faint)}.rawmeta code{font-size:.95em;padding:0;border:0;background:transparent}
  .memberships{min-width:29rem}.atlas-mouse>summary span{overflow-wrap:anywhere}.atlas-mouse hr{border:0;border-top:1px solid var(--line-soft);margin:.55rem 0}
  .back{margin:2rem 0 0;font-family:var(--mono);font-size:.72rem}.back a{color:var(--ink-faint)}
  figure[data-figure-target]{scroll-margin-top:1.1rem}
  .figref-link{font-family:var(--mono);font-size:.88em}
  .evidence-figure{padding-top:.2rem}.evidence-figure+.evidence-figure{margin-top:2.8rem}
  .evidence-figure figcaption{max-width:58rem!important}.figure-source-inline{display:inline-block;margin-left:.3rem;font-family:var(--mono);font-size:.92em;white-space:nowrap}
  .science-methods-atlas{display:grid;gap:2rem;width:min(86rem,calc(100vw - 2rem));margin:1rem 0 2rem 50%;transform:translateX(-50%)}
  .science-methods-atlas .science-methods-figure{overflow:hidden;margin:0;padding:0;border:1px solid var(--line);border-radius:12px;background:#fff;box-shadow:0 12px 34px rgba(0,0,0,.09)}
  .science-methods-atlas .science-methods-figure+.science-methods-figure{margin-top:0}
  .science-methods-figure>a{display:block;border:0;background:#fff}
  .science-methods-figure img{display:block;width:100%;height:auto;max-height:none;object-fit:contain;background:#fff}
  .science-methods-figure figcaption{max-width:none!important;margin:0;padding:.8rem 1rem;border-top:1px solid var(--line);background:var(--surface);color:var(--ink-soft);font-size:.78rem;line-height:1.55}
  .figure-atlas-intro{margin:3rem 0 1.5rem;padding-top:1rem;border-top:2px solid var(--line)}.figure-index{line-height:1.9}.figure-index a{white-space:nowrap}
  .code-study{margin:1.25rem 0 1.8rem;border:1px solid var(--line);border-radius:12px;background:var(--surface);overflow:hidden}.code-study-head{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;padding:.7rem .9rem;border-bottom:1px solid var(--line);background:var(--raise);font-weight:650}.code-study-head a{font-family:var(--mono);font-size:.72rem;white-space:nowrap}.code-study pre,details.ref pre{margin:0;padding:1rem;overflow-x:auto;background:var(--paper);border-bottom:1px solid var(--line);font-family:var(--mono);font-size:.75rem;line-height:1.55}.code-study pre code,details.ref pre code{padding:0;border:0;background:transparent;white-space:pre;word-break:normal;overflow-wrap:normal}.code-study p{margin:.85rem 1rem 1rem;font-size:.9rem;color:var(--ink-soft)}
  .toc-fab{--toc-fab-inset:1rem;position:fixed;right:calc(var(--toc-fab-inset) + env(safe-area-inset-right));bottom:calc(var(--toc-fab-inset) + env(safe-area-inset-bottom));z-index:950;pointer-events:none}
  .toc-fab button,.toc-fab a{font-family:var(--sans)}
  .toc-fab__launcher{pointer-events:auto;display:inline-flex;align-items:center;justify-content:center;width:3.3rem;height:3.3rem;padding:0;border:1px solid color-mix(in srgb,var(--fluoro) 55%,var(--line));border-radius:999px;background:var(--fluoro);color:var(--accent-on);box-shadow:0 16px 42px rgba(0,0,0,.32);cursor:pointer;transition:transform .16s ease,box-shadow .16s ease,background .16s ease}
  .toc-fab__launcher:hover{transform:translateY(-2px);box-shadow:0 20px 48px rgba(0,0,0,.36)}
  .toc-fab__launcher:focus-visible,.toc-fab__close:focus-visible{outline:3px solid var(--fluoro);outline-offset:3px}
  .toc-fab.is-open .toc-fab__launcher{display:none}
  .toc-fab__panel{pointer-events:auto;width:min(27rem,calc(100vw - 2rem));max-height:min(78vh,42rem);display:flex;flex-direction:column;gap:.55rem;padding:.9rem .55rem .9rem 1rem;border:1px solid var(--line);border-radius:12px;background:var(--raise);background:color-mix(in srgb,var(--raise) 94%,transparent);color:var(--ink);box-shadow:0 24px 70px rgba(0,0,0,.34);backdrop-filter:saturate(180%) blur(12px);-webkit-backdrop-filter:saturate(180%) blur(12px)}
  .toc-fab__panel[hidden]{display:none}
  .toc-fab__header{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:0 .4rem .2rem 0}
  .toc-fab__header>div{display:grid;gap:.05rem}
  .toc-fab__header strong{font-family:var(--serif);font-size:1.15rem;line-height:1.2}
  .toc-fab__eyebrow{font-family:var(--mono);font-size:.64rem;font-weight:750;letter-spacing:.11em;text-transform:uppercase;color:var(--fluoro-ink)}
  .toc-fab__close{display:inline-flex;align-items:center;justify-content:center;width:2rem;height:2rem;flex:none;padding:0;border:1px solid var(--line);border-radius:7px;background:var(--surface);color:var(--ink);cursor:pointer}
  .toc-fab__close:hover{border-color:var(--fluoro);color:var(--fluoro-ink)}
  .toc-fab__tools{display:grid;gap:.38rem;padding:0 .4rem .2rem 0}
  .toc-fab__search{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:.45rem;min-height:2.35rem;padding:0 .62rem;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink-faint)}
  .toc-fab__search:focus-within{border-color:var(--fluoro);box-shadow:0 0 0 2px var(--fluoro-soft);color:var(--fluoro-ink)}
  .toc-fab__search input{min-width:0;width:100%;padding:.5rem 0;border:0;outline:0;background:transparent;color:var(--ink);font:inherit;font-size:.86rem}
  .toc-fab__search input::placeholder{color:var(--ink-faint)}
  .toc-fab__clear,.toc-fab__detail-toggle{padding:0;border:0;background:transparent;color:var(--fluoro-ink);font-size:.7rem;font-weight:700;cursor:pointer}
  .toc-fab__status{display:flex;align-items:center;justify-content:space-between;gap:1rem;color:var(--ink-faint);font-family:var(--mono);font-size:.63rem}
  .toc-fab__detail-toggle{text-decoration:underline;text-underline-offset:.18em}
  .toc-fab__list,.toc-fab__sublist{list-style:none;margin:0;padding:0}
  .toc-fab__list{min-height:0;overflow-y:auto;overscroll-behavior:contain;-webkit-overflow-scrolling:touch;padding-right:.35rem}
  .toc-fab__item{margin:0}
  .toc-fab__item[hidden]{display:none!important}
  .toc-fab__group+.toc-fab__group{border-top:1px solid var(--line-soft)}
  .toc-fab__sublist{display:none;margin:0 0 .28rem .72rem;padding-left:.5rem;border-left:1px solid var(--line)}
  .toc-fab__group.is-current>.toc-fab__sublist,.toc-fab.is-detailed .toc-fab__sublist,.toc-fab.is-filtering .toc-fab__sublist{display:block}
  .toc-fab__link{display:grid;grid-template-columns:3.15rem minmax(0,1fr);gap:.18rem;align-items:baseline;padding:.4rem .5rem;border:0;border-left:2px solid transparent;border-radius:0 7px 7px 0;color:var(--ink-soft);font-size:.84rem;line-height:1.32;text-decoration:none}
  .toc-fab__link--level-1{padding-top:.5rem;padding-bottom:.5rem;color:var(--ink);font-weight:680}
  .toc-fab__link--level-3{font-size:.78rem;color:var(--ink-faint)}
  .toc-fab__number{font-family:var(--mono);font-size:.62rem;color:var(--ink-faint);white-space:nowrap}
  .toc-fab__label{min-width:0}
  .toc-fab__link:hover{background:var(--surface);color:var(--ink)}
  .toc-fab__link.is-active{border-left-color:var(--fluoro);background:var(--fluoro-soft);color:var(--fluoro-ink);font-weight:680}
  .toc-fab__link.is-active .toc-fab__number{color:var(--fluoro-ink)}
  section[id],article[id],h3[id],h4[id],figure[id]{scroll-margin-top:1.1rem}
  @media(max-width:700px){.result-guide{gap:1rem}.result-card{border-radius:10px}.result-card__head{grid-template-columns:2.5rem minmax(0,1fr);gap:.65rem;padding:.95rem}.result-card__index{width:2.25rem;height:2.25rem}.result-card>.evidence-figure{padding:.65rem}.result-card__body{grid-template-columns:1fr;padding:.9rem}.result-panel-list>div{grid-template-columns:1fr;gap:.18rem;padding:.62rem 0}.result-card__takeaway{display:block}.cohort-evidence-table{width:100%;margin-left:0;transform:none}.cohort-evidence-table table{min-width:58rem}.cohort-evidence-table caption::after{content:" · swipe table →";font-family:var(--mono);font-size:.72em;color:var(--fluoro-ink)}.atlas-table{margin-left:0;margin-right:0;border-radius:0}.science-methods-atlas{width:calc(100vw - 1rem)}.science-methods-figure figcaption{padding:.7rem}.rawmeta{font-size:.67rem}.toc-fab{--toc-fab-inset:.65rem}.toc-fab__panel{width:calc(100vw - 1.3rem);max-height:min(82vh,42rem)}.toc-fab__link{grid-template-columns:3rem minmax(0,1fr);padding-left:.38rem;padding-right:.38rem}.code-study-head{align-items:flex-start;flex-direction:column;gap:.2rem}.code-study pre,details.ref pre{font-size:.69rem}}
  @media print{.toc-fab{display:none!important}.result-card{break-inside:avoid;box-shadow:none}}
'''
    css_marker = "<!-- ATLAS-CSS -->"
    if css_marker in html:
        html = re.sub(re.escape(css_marker) + r".*?<!-- /ATLAS-CSS -->", css_marker + css + "<!-- /ATLAS-CSS -->", html, flags=re.S)
    else:
        html = html.replace("</style>\n</head>", css_marker + css + "<!-- /ATLAS-CSS -->\n</style>\n</head>", 1)

    html = re.sub(r'<details class="(ref|fig)"(?! open)', r'<details class="\1" open', html)
    html = html.replace('<meta name="description" content="Research reference for Zhong et al. (2025): exact Nature links, a verified audit of the released neural data and longitudinal coverage, and a complete trial-window analysis plan for changes in the distribution of neuronal d-prime.">', '<meta name="description" content="Complete source-linked Zhong et al. 2025 data atlas: every imaging mouse, experiment, neural acquisition, behavior and retinotopy relationship, both research questions, and an executable analysis plan grounded in the Nature paper and large-scale-recording methods review.">')
    html = html.replace("<title>Zhong et al. 2025 — neural-data coverage and d-prime distribution study</title>", "<title>Zhong et al. 2025 — complete neural data atlas and two-question analysis plan</title>")
    html = html.replace("A source-linked reference for the experiment, the exact neural-data coverage, and a playable analysis of how leaf-versus-circle d&prime; distributions evolve across trial windows before and after rewarded or unrewarded training.", "A complete, source-linked atlas of every released imaging acquisition, behavior and retinotopy relationship, plus rigorous analysis plans for distributional selectivity change and reward-associated within-session dynamics.")
    html = html.replace(
        f'<a href="{NATURE}" target="_blank" rel="noopener noreferrer">Nature article</a>\n      <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC12367527/"',
        f'<a href="{NATURE}" target="_blank" rel="noopener noreferrer">Nature article</a>\n      <a href="{PAPER_DRIVE}" target="_blank" rel="noopener noreferrer">Nature PDF in Drive</a>\n      <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC12367527/"',
    )
    html = html.replace(
        f'<a href="{SCIENCE_DOI}" target="_blank" rel="noopener noreferrer">Neural-data methods review</a>\n    </div>',
        f'<a href="{SCIENCE_DOI}" target="_blank" rel="noopener noreferrer">Neural-data methods review</a>\n      <a href="{METHODS_DRIVE}" target="_blank" rel="noopener noreferrer">Methods PDF in Drive</a>\n    </div>',
        1,
    )
    html = html.replace(
        f'<a href="https://doi.org/10.1038/s41586-025-09180-y" target="_blank" rel="noopener noreferrer">DOI</a></span>',
        f'<a href="https://doi.org/10.1038/s41586-025-09180-y" target="_blank" rel="noopener noreferrer">DOI</a> · <a href="{PAPER_DRIVE}" target="_blank" rel="noopener noreferrer">Drive PDF</a></span>',
        1,
    )
    html = html.replace(
        '<a href="https://mouseland.github.io/research/science.adp7429.pdf" target="_blank" rel="noopener noreferrer">open PDF</a></span>',
        f'<a href="https://mouseland.github.io/research/science.adp7429.pdf" target="_blank" rel="noopener noreferrer">open PDF</a> · <a href="{METHODS_DRIVE}" target="_blank" rel="noopener noreferrer">Drive PDF</a></span>',
        1,
    )
    html = html.replace("href=\"#within\">&sect;08</a>", "href=\"#within\">&sect;11</a>")
    html = html.replace("href=\"#analyses\">&sect;10</a>", "href=\"#analyses\">&sect;15</a>")
    html = html.replace("href=\"#analysis-primary\">&sect;10</a>", "href=\"#analysis-primary\">&sect;15</a>")
    html = _renumber_sections(html)
    html = _refine_paper_citations(html)
    html = _correct_legacy_scientific_copy(html)

    # Final structural assertions make accidental partial generation fail loudly.
    assert len(re.findall(r'<details\b[^>]*\bopen\b', html)) >= 30
    assert html.count('class="ref atlas-mouse"') == 19
    assert html.count('class="membership"') == 142
    assert html.count('class="toc-fab__link ') == len(TOC_ITEMS)
    assert html.count("<script>") == 1
    assert all(f'id="{target}"' in html for target, _, _ in TOC_ITEMS)
    assert all(f'id="{section_id}"' in html for section_id in ("contents", "atlas", "support", "methods-review", "reward-rate", "questions", "workflow"))
    HTML_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
