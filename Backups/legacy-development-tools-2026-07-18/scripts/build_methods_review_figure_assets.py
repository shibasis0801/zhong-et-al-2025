from __future__ import annotations

"""Build exact figure crops from the reviewed Science paper PDF.

The source PDF is the file "Analysis methods for large-scale neuronal
recordings.pdf" in the project Drive.  Crops include each complete published
figure and its published caption.  No scientific labels, values, or artwork
are redrawn or altered.
"""

import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "tmp" / "pdfs" / "analysis-methods-large-scale-neuronal-recordings.pdf"
ASSETS = ROOT / "zhong2025" / "assets" / "reference_figures"

SOURCE_SHA256 = "2a92762cee61070b5fbb5bfac780da03ccee5e5d7ea7c74c7be86c969007e511"
RENDER_DPI = 300

# Crop coordinates were measured on the 150-dpi Poppler render.  They retain
# the complete figure and published caption while excluding neighboring body
# text, the page footer, and the download watermark in the outer margin.
# Coordinates are (left, top, right, bottom).
CROPS_150_DPI: dict[str, tuple[int, tuple[int, int, int, int]]] = {
    "science-methods-fig1.jpg": (3, (75, 742, 1165, 1494)),
    "science-methods-fig2.jpg": (4, (75, 92, 1165, 950)),
    "science-methods-fig3.jpg": (5, (75, 94, 1165, 1157)),
    "science-methods-fig4.jpg": (6, (75, 98, 1165, 692)),
}


def _render_pages(source: Path, destination: Path) -> dict[int, Path]:
    prefix = destination / "page"
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            "3",
            "-l",
            "6",
            "-r",
            str(RENDER_DPI),
            "-png",
            str(source),
            str(prefix),
        ],
        check=True,
    )
    rendered = sorted(destination.glob("page-*.png"))
    if len(rendered) != 4:
        raise RuntimeError(f"expected four rendered pages, found {len(rendered)}")
    return {page: path for page, path in zip(range(3, 7), rendered)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"missing reviewed Science PDF: {source}\n"
            "Download Drive file 1DlmPeyaHn-thn9ILrt-rAXP96-y3IMU7 first."
        )
    actual_sha256 = sha256(source.read_bytes()).hexdigest()
    if actual_sha256 != SOURCE_SHA256:
        raise ValueError(
            "Science PDF checksum mismatch: "
            f"expected {SOURCE_SHA256}, found {actual_sha256}"
        )

    ASSETS.mkdir(parents=True, exist_ok=True)
    scale = RENDER_DPI / 150
    with tempfile.TemporaryDirectory(prefix="science-methods-figures-") as tmp:
        pages = _render_pages(source, Path(tmp))
        for output_name, (page, box_150) in CROPS_150_DPI.items():
            box = tuple(round(value * scale) for value in box_150)
            with Image.open(pages[page]) as rendered:
                crop = rendered.convert("RGB").crop(box)
                crop.save(
                    ASSETS / output_name,
                    format="JPEG",
                    quality=94,
                    optimize=True,
                    progressive=True,
                    dpi=(RENDER_DPI, RENDER_DPI),
                )


if __name__ == "__main__":
    main()
