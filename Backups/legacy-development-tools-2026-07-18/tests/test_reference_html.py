from __future__ import annotations

import base64
from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image


REFERENCE = Path("zhong2025_reference.html")
INVENTORY = Path("zhong2025/assets/figshare-v2-inventory.json")
NATURE_ROOT = "https://www.nature.com/articles/s41586-025-09180-y"
NATURE_DOI = "https://doi.org/10.1038/s41586-025-09180-y"
IMAGING_INDEX_SOURCE = "https://ndownloader.figshare.com/files/54183854"
SCIENCE_DOI = "https://doi.org/10.1126/science.adp7429"
SCIENCE_PDF = "https://mouseland.github.io/research/science.adp7429.pdf"
SCIENCE_SOURCE_SHA256 = "2a92762cee61070b5fbb5bfac780da03ccee5e5d7ea7c74c7be86c969007e511"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"

NATURE_FULL_TARGETS = (
    *(f"nature-fig{number}" for number in range(1, 6)),
    *(f"nature-ed{number}" for number in range(1, 10)),
)


class _ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.images: list[dict[str, str]] = []
        self.figure_refs: list[dict[str, str]] = []
        self.figure_targets: list[str] = []
        self.target_elements: list[dict[str, str]] = []
        self.figures: list[dict[str, object]] = []
        self.text: list[str] = []
        self._figure_stack: list[dict[str, object]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])
        if values.get("data-figure-target"):
            self.figure_targets.append(values["data-figure-target"])
            self.target_elements.append({"tag": tag, **values})
        if values.get("data-figure-ref"):
            self.figure_refs.append({"tag": tag, **values})
        if tag == "figure":
            self._figure_stack.append(
                {
                    "attrs": values,
                    "text_parts": [],
                    "images": [],
                    "hrefs": [],
                    "figcaptions": 0,
                    "inline_svgs": 0,
                }
            )
        if tag == "a" and values.get("href"):
            self.hrefs.append(values["href"])
            if self._figure_stack:
                self._figure_stack[-1]["hrefs"].append(values["href"])
        if tag == "img":
            self.images.append(values)
            if self._figure_stack:
                self._figure_stack[-1]["images"].append(values)
        if tag == "figcaption" and self._figure_stack:
            self._figure_stack[-1]["figcaptions"] += 1
        if tag == "svg" and self._figure_stack:
            self._figure_stack[-1]["inline_svgs"] += 1

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        if self._figure_stack:
            self._figure_stack[-1]["text_parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "figure" or not self._figure_stack:
            return
        figure = self._figure_stack.pop()
        figure["text"] = " ".join(" ".join(figure.pop("text_parts")).split())
        self.figures.append(figure)


def _parse() -> tuple[str, str, _ReferenceParser]:
    source = REFERENCE.read_text(encoding="utf-8")
    parser = _ReferenceParser()
    parser.feed(source)
    parser.close()
    text = " ".join(" ".join(parser.text).split())
    return source, text, parser


def test_reference_html_has_valid_navigation_and_self_contained_figures():
    source, _, parser = _parse()

    duplicates = [key for key, count in Counter(parser.ids).items() if count > 1]
    assert duplicates == []
    assert all(href[1:] in parser.ids for href in parser.hrefs if href.startswith("#"))
    assert parser.images
    assert all(image.get("alt") for image in parser.images)
    assert all(image.get("src", "").startswith("data:image/") for image in parser.images)
    assert source.count("<script>") == 1
    assert source.count('class="toc-fab__link ') == 69
    assert 'id="floating-contents"' in source
    assert 'aria-label="Open table of contents"' in source
    assert 'aria-label="Close table of contents"' in source
    assert 'aria-label="Filter navigator destinations"' in source
    assert 'class="toc-fab__detail-toggle"' in source
    assert source.count('class="toc-fab__sublist ') == 19
    assert "Focus current section" in source
    assert "applyFilter" in source
    assert "target.scrollIntoView({block: 'start', behavior: 'instant'})" in source
    assert "aria-current', 'location'" in source
    assert "event.key === 'Escape'" in source
    assert all(" open" in tag for tag in re.findall(r"<details\\b[^>]*>", source))

    section_ids = re.findall(r'<section id="([^"]+)">', source)
    assert section_ids == [
        "paper",
        "experiment",
        "stimuli",
        "data",
        "coverage",
        "atlas",
        "support",
        "environment",
        "methods-review",
        "dprime",
        "within",
        "reward-rate",
        "questions",
        "figuremap",
        "analyses",
        "workflow",
        "recipes",
        "caveats",
    ]
    assert parser.hrefs.count("#contents") >= 6


def test_published_findings_are_panel_aware_inline_figure_cards():
    source, text, parser = _parse()

    cards = re.findall(
        r'<article id="paper-(?:figure-[1-5]|methods)" class="result-card(?: result-card--methods)?">',
        source,
    )
    assert len(cards) == 6
    assert source.count('class="result-shot"') == 0
    assert source.count('class="result-panel-list"') == 6
    assert source.count('class="figure-attribution"') == 1

    for target in (
        "paper-figure-1",
        "paper-figure-2",
        "paper-figure-3",
        "paper-figure-4",
        "paper-figure-5",
        "paper-methods",
    ):
        assert target in parser.ids
        assert f"#{target}" in parser.hrefs

    for number in range(1, 6):
        card = re.search(
            rf'<article id="paper-figure-{number}".*?</article>',
            source,
            re.S,
        )
        assert card
        assert f'id="nature-fig{number}"' in card.group(0)

    for phrase in (
        "Plasticity in the visual cortex after supervised and unsupervised training",
        "Comparing visual and spatial coding on test stimuli",
        "Responses to novel and adapted stimuli and neural orthogonalization",
        "A reward-prediction signal in supervised training only",
        "Unsupervised pretraining accelerates subsequent task learning",
        "What the paper reports",
        "Preferred-position sequences in leaf1 and leaf2 were uncorrelated",
        "late-versus-early cue-position",
        "the authors describe this as orthogonalization",
    ):
        assert phrase.lower() in text.lower()
    for noise in (
        "Proposed decision in this reference",
        "Adapted from",
        "cropped only",
        "select any screenshot",
        "Recommended primary pair",
        "Stronger questions that the release can actually answer",
        "Better or complementary questions in the same direction",
        "Most measured visual-cortical plasticity was reproduced without reward",
    ):
        assert noise.lower() not in text.lower()


def test_inline_figure_atlas_has_unique_targets_and_resolvable_internal_links():
    _, _, parser = _parse()
    required_targets = set(NATURE_FULL_TARGETS)
    target_counts = Counter(
        target for target in parser.figure_targets if target.startswith("nature-")
    )

    assert required_targets == target_counts.keys()
    assert all(count == 1 for count in target_counts.values())
    assert all(
        element.get("tag") == "figure"
        and element.get("id") == element.get("data-figure-target")
        for element in parser.target_elements
        if element.get("data-figure-target", "").startswith("nature-")
    )

    for figure_ref in parser.figure_refs:
        target = figure_ref["data-figure-ref"]
        assert figure_ref.get("tag") == "a"
        assert figure_ref.get("href") == f"#{target}"
        assert target_counts[target] == 1


def test_nature_figures_are_complete_unique_source_linked_and_globally_licensed():
    source, text, parser = _parse()
    figures = {
        figure["attrs"]["data-figure-target"]: figure
        for figure in parser.figures
        if figure["attrs"].get("data-figure-kind") == "nature"
    }

    assert set(NATURE_FULL_TARGETS) == figures.keys()
    assert source.count('class="figure-attribution"') == 1
    assert "Zhong et al." in text
    assert "CC BY 4.0" in text
    assert NATURE_DOI in parser.hrefs
    assert CC_BY in parser.hrefs
    assert "Adapted from" not in text
    assert "Cropped to" not in text
    assert "cropped only" not in text.lower()

    payload_hashes: list[str] = []
    for target, figure in figures.items():
        images = figure["images"]
        hrefs = figure["hrefs"]

        assert figure["figcaptions"] == 1, target
        assert len(images) == 1, target
        assert images[0].get("alt"), target
        assert images[0].get("src", "").startswith("data:image/"), target
        assert images[0].get("width"), target
        assert images[0].get("height"), target

        payload = base64.b64decode(images[0]["src"].split(",", 1)[1])
        payload_hashes.append(sha256(payload).hexdigest())
        with Image.open(BytesIO(payload)) as image:
            assert image.size == (int(images[0]["width"]), int(images[0]["height"])), target

        main_match = re.match(r"nature-fig([1-5])", target)
        if main_match:
            assert f"{NATURE_ROOT}#Fig{main_match.group(1)}" in hrefs, target
        else:
            ed_match = re.fullmatch(r"nature-ed([1-9])", target)
            assert ed_match, target
            assert f"{NATURE_ROOT}/figures/{int(ed_match.group(1)) + 5}" in hrefs, target

    assert len(payload_hashes) == len(set(payload_hashes)) == 14


def test_paper_claims_use_precise_article_anchors_and_release_provenance():
    source, text, parser = _parse()

    assert f'href="{NATURE_ROOT}"' not in source
    assert not re.search(rf'href="{re.escape(NATURE_ROOT)}/figures/[1-5]"', source)
    for anchor in (
        "Sec2", "Sec3", "Sec4", "Sec6", "Sec7", "Sec11", "Sec13",
        "Sec14", "Sec15", "Sec16", "Sec17", "Sec19", "Sec20", "Sec21",
        "Sec22", "Sec24", "Sec25", "data-availability",
        "Fig1", "Fig2", "Fig3", "Fig4", "Fig5",
    ):
        assert f"{NATURE_ROOT}#{anchor}" in parser.hrefs, anchor

    nature_figure_refs = [
        ref for ref in parser.figure_refs
        if ref["data-figure-ref"].startswith("nature-")
        and "figref-link" in ref.get("class", "").split()
    ]
    assert source.count('class="figref-source"') == len(nature_figure_refs)
    assert source.count('class="paper-cite"') >= 50

    assert IMAGING_INDEX_SOURCE in parser.hrefs
    for phrase in (
        "Counts are panel-specific",
        "Released-field inventory",
        "Paper-code mapping",
        "The 400-component SVD files are a deposited-release representation",
    ):
        assert phrase.lower() in text.lower(), phrase

    for panel_count in (
        "<em>n</em>=4 task, 9 unrewarded natural-texture, and 3 grating mice",
        "<em>n</em>=5 in Fig.&nbsp;2e,f,j",
        "<em>n</em>=7 in Fig.&nbsp;2f",
        "<em>n</em>=6 in Fig.&nbsp;3b,e,h",
        "<em>n</em>=3 (5 sessions) in Fig.&nbsp;1j",
        "<em>n</em>=3 (6 sessions) in Fig.&nbsp;3e,h",
        "<em>n</em>=9 mice (11 sessions) in Fig.&nbsp;2f and Fig.&nbsp;3e,h",
    ):
        assert panel_count in source, panel_count

    for blanket_range in (">4–5<", ">6–9<", ">7–9<"):
        assert blanket_range not in source


def test_methods_contract_uses_complete_source_crops_without_generated_figures():
    source, text, parser = _parse()

    assert 'class="methods-schematic"' not in source
    assert 'data-figure-kind="methods-summary"' not in source
    assert not any(figure["inline_svgs"] for figure in parser.figures)
    assert SCIENCE_DOI in parser.hrefs
    methods_figures = [
        figure for figure in parser.figures
        if figure["attrs"].get("data-figure-kind") == "science"
    ]
    assert len(methods_figures) == 4
    assert [figure["attrs"]["id"] for figure in methods_figures] == [
        f"methods-fig{number}" for number in range(1, 5)
    ]
    assert [figure["attrs"]["data-source-pdf-page"] for figure in methods_figures] == [
        str(page) for page in range(3, 7)
    ]
    assert all(
        figure["attrs"]["data-source-sha256"] == SCIENCE_SOURCE_SHA256
        for figure in methods_figures
    )
    assert all(len(figure["images"]) == 1 for figure in methods_figures)
    assert all(figure["figcaptions"] == 1 for figure in methods_figures)
    assert {
        href
        for figure in methods_figures
        for href in figure["hrefs"]
        if href.startswith(SCIENCE_PDF)
    } == {f"{SCIENCE_PDF}#page={page}" for page in range(3, 7)}
    for phrase in (
        "This section separates methods reported by Zhong et al. from safeguards proposed for new analyses",
        "The four complete figures and their published captions below are cropped from pages 3–6",
        "Published estimator, proposed extension, and validation source",
        "Reproduce the published estimator",
        "late-versus-early cue-position",
        "Do not equate decoding with mechanism",
    ):
        assert phrase.lower() in text.lower()


def test_paper_code_walkthrough_and_plot_playbook_are_actionable_and_linked():
    _, text, parser = _parse()

    for phrase in (
        "Read the paper code before extending it",
        "The published d′ is frame based",
        "The denominator is the arithmetic mean",
        "not pooled RMS variance",
        "Selection, ordering, and display use different data",
        "The density map is not a local selective fraction",
        "Reward-prediction panels use held-out neuron selection",
        "not deterministically seeded",
        "How the project code turns those recipes into stable analyses",
        "Plot playbook for the two questions",
        "ECDF or signed quantile curves",
        "Ridgeline / violin / KDE",
        "Metric trajectory small multiples",
        "Signed tail plot",
        "Cross-temporal matrix",
        "QC companion panel",
    ):
        assert phrase.lower() in text.lower()

    for link in (
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L370-L374",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L418-L441",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L599-L703",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L376-L416",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L503-L597",
        "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L814-L882",
    ):
        assert link in parser.hrefs


def test_reference_html_covers_release_and_longitudinal_truth():
    source, text, parser = _parse()

    for fact in (
        "297 files and 452,233,500,962 bytes (421.175 GiB)",
        "89/89 indexed recordings",
        "404.356",
        "10.008",
        "Not continuous daily imaging.",
        "1–8 sampled recording dates",
        "3–24 days",
        "no cross-date neuron identity map",
        "Thirteen matched mice and 26 recordings",
        "3.910 GiB",
        "123.038 GiB",
        "126.080 GiB",
        "89 physical acquisitions",
        "142 source metadata rows",
        "133 unique experiment–recording memberships",
        "142 source metadata rows; 133 unique experiment–recording pairs",
        "Imaging behaviour",
        "108.1–412.5 MiB",
        "Behaviour-only learning",
        "378.7–817.9 MiB",
        "no true imaging instance",
        "Inventory JSON SHA-256",
        "ad8eaf217b3908976a3f701d6700d9ffd4479c529d8d2eab345919d694b57650",
        "not each array’s neuron × frame shape",
        "Complete behavior-field glossary",
        "Retinotopy-field glossary",
    ):
        assert fact in text

    assert "142 experiment&ndash;recording memberships" not in source
    assert "Fig.&nbsp;1g–i" not in source
    assert "https://github.com/MouseLand/zhong-et-al-2025/blob/paper/utils.py#L394-L416" in parser.hrefs

    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, int]] = {}
    for entry in inventory["files"]:
        row = rows.setdefault(entry["category"], {"file_count": 0, "size_bytes": 0})
        row["file_count"] += 1
        row["size_bytes"] += int(entry["size_bytes"])
    expected = {
        "full_neural": (89, 434_174_046_325),
        "reduced_neural": (89, 10_746_406_398),
        "retinotopy": (89, 177_062_302),
        "neural_example": (1, 97_192_128),
    }
    for category, (count, size) in expected.items():
        assert rows[category]["file_count"] == count
        assert rows[category]["size_bytes"] == size
        assert f"{size:,}" in text


def test_reference_html_explains_one_neural_file_frame_by_frame():
    source, text, parser = _parse()

    assert "<!-- NEURAL-FRAME-DETAIL:START -->" in source
    assert "<!-- NEURAL-FRAME-DETAIL:END -->" in source
    assert {
        "data-neural-file", "data-neural-trial-timeline",
        "data-frame-join", "data-selectivity",
    } <= set(parser.ids)
    for requirement in (
        "one continuous physical imaging acquisition",
        "full[n, f]",
        "U.T @ V",
        "ft_trInd[f]",
        "ft_Pos[f] / 10",
        "A frame is not a trial and cannot by itself have d′.",
        "sup_train1_before_learning",
        "unsup_train1_after_learning",
        "train1_after_grating",
        "Paper-style full-neural d′",
        "Held-out blockwise population d′ inside one file",
        "One manifest join, two array-axis joins.",
        "filesystem-only neural + behavior + retinotopy join",
        "behavior_key = recording_id + '_' + stimtype",
        "22,994 frames",
        "no literal one-trial d′",
        "held-out neural evidence",
        "not a one-trial d′",
        "multi-trial segment",
        "role 0 = circle1",
        "role 2 = leaf1",
        "four fold-specific held-out d′ values",
        "raw scores from separately fitted folds are not pooled",
        "ddof=0",
        "ddof=1",
    ):
        assert requirement in text


def test_reference_html_contains_complete_distribution_study_design():
    source, text, _ = _parse()

    for requirement in (
        "H1 · broader selectivity",
        "H2 · directional asymmetry",
        "H3 · selective-tail redistribution",
        "Windowed trial-level d′",
        "pairwise frame index",
        "balanced rolling window",
        "bias=False",
        "Fisher excess kurtosis",
        "progress:C(stage):C(condition)",
        "mouse-bootstrap 95% interval",
        "A run-state card must show loading, success, empty-result, or the full error",
        "new analysis",
        "not performed in the paper",
        "Research question 2",
        "Does rewarded training change the early within-session rate",
        "exact label permutation",
        "leave-one-mouse-out",
        "Causal-language limit",
        "Published results and scope limits",
        "Published evidence and scope limits",
    ):
        assert requirement.lower() in text.lower()

    for stale in (
        "40-trial blocks",
        "140-trial early horizon",
    ):
        assert stale not in source


def test_complete_atlas_is_generated_from_every_release_membership():
    source, text, _ = _parse()
    index = json.loads(Path("zhong2025/assets/imaging-experiment-index.json").read_text(encoding="utf-8"))

    assert source.count('class="experiment-row"') == len(index["experiments"]) == 23
    assert source.count('class="acquisition-row"') == index["summary"]["unique_recordings"] == 89
    assert source.count('class="membership"') == index["summary"]["associations"] == 142
    assert source.count('class="ref atlas-mouse"') == index["summary"]["unique_mice"] == 19

    for experiment, rows in index["experiments"].items():
        assert f"Beh_{experiment}.npy" in text
        for row in rows:
            assert row["recording_id"] in text
            assert row["retinotopy_id"] in text


def test_drive_papers_notebooks_and_methods_contract_are_linked():
    source, text, parser = _parse()
    required_links = (
        "https://drive.google.com/file/d/10u0D2bRCScDujpWHFxwAZuvfDIglOW_i/view?usp=drivesdk",
        "https://drive.google.com/file/d/1DlmPeyaHn-thn9ILrt-rAXP96-y3IMU7/view?usp=drivesdk",
        "https://doi.org/10.1126/science.adp7429",
        "https://colab.research.google.com/drive/1CN2b_NHigbJ4jPd_2FqWPs3mHRhJZRdT",
        "https://colab.research.google.com/drive/1YvuuZPrkPNoMFCu15yfBU_V0zeMBwsRz",
        "https://colab.research.google.com/drive/10SDh3byJ_bv48Ob5dNIKL1H3JgBnCzRP",
        "https://colab.research.google.com/drive/1Xz40c50g5KczU5Rp5Dz_TYH2n21C-abP",
        "https://developers.google.com/workspace/drive/api/guides/limits",
        "https://research.google.com/colaboratory/faq.html",
    )
    for link in required_links:
        assert link in parser.hrefs

    for phrase in (
        "score held-out temporal blocks",
        "Do not equate decoding with mechanism",
        "compare invariant summaries across mice",
        "RQ1 confirmation—implementation still required",
        "Persist everything",
        "mounted Drive → checksum-addressed object-store/CDN derivative",
    ):
        assert phrase.lower() in text.lower()


def test_paper_derived_sections_link_to_canonical_or_inline_nature_evidence():
    source, _, parser = _parse()
    assert sum(href.startswith(NATURE_ROOT) for href in parser.hrefs) >= 35

    science_sections = (
        "paper",
        "experiment",
        "stimuli",
        "data",
        "coverage",
        "atlas",
        "support",
        "dprime",
        "within",
        "reward-rate",
        "questions",
        "figuremap",
        "analyses",
        "workflow",
        "recipes",
        "caveats",
    )
    for index, section_id in enumerate(science_sections):
        start = source.index(f'<section id="{section_id}">')
        if index + 1 < len(science_sections):
            end = source.index(f'<section id="{science_sections[index + 1]}">', start)
        else:
            end = source.index("</main>", start)
        section = source[start:end]
        assert NATURE_ROOT in section or 'data-figure-ref="nature-' in section, section_id
