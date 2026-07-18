from __future__ import annotations

"""Build the small, repeated panel crops used by the HTML project guide.

The source images are the open-access figure assets published with Zhong et al.
(Nature 2025).  This script never edits the sources; it only creates explicitly
named JPEG crops so the generated guide stays deterministic and reasonably
small.
"""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "zhong2025" / "assets" / "reference_figures"


# left, top, right, bottom in source-image pixels.  Names include the source
# figure/panels so every adapted crop can be attributed precisely in the HTML.
CROPS: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    "nature-main-1-panels-a-b.jpg": ("nature-main-1.png", (0, 0, 1200, 176)),
    "nature-main-1-panel-f.jpg": ("nature-main-1.png", (895, 155, 1200, 345)),
    "nature-main-1-panels-i-j.jpg": ("nature-main-1.png", (450, 335, 1200, 762)),
    "nature-main-2-panels-g-j.jpg": ("nature-main-2.png", (0, 448, 1200, 780)),
    "nature-main-3-panels-f-h.jpg": ("nature-main-3.png", (385, 250, 1200, 778)),
    "nature-main-4-panel-e.jpg": ("nature-main-4.png", (900, 125, 1200, 365)),
    "nature-main-4-panels-f-g.jpg": ("nature-main-4.png", (0, 280, 650, 590)),
    "nature-main-4-panels-i-l.jpg": ("nature-main-4.png", (0, 465, 705, 769)),
    "nature-main-5-panels-a-b.jpg": ("nature-main-5.png", (0, 0, 500, 225)),
    "nature-main-5-panels-e-h.jpg": ("nature-main-5.png", (490, 0, 1200, 641)),
    "nature-ed-1-panel-c.jpg": ("nature-ed-1.jpg", (1390, 0, 2098, 410)),
    "nature-ed-1-panel-e.jpg": ("nature-ed-1.jpg", (720, 440, 2098, 930)),
    "nature-ed-1-panel-f.jpg": ("nature-ed-1.jpg", (0, 930, 1392, 1453)),
    "nature-ed-2-panel-c.jpg": ("nature-ed-2.jpg", (0, 810, 995, 1459)),
    "nature-ed-2-panel-d.jpg": ("nature-ed-2.jpg", (1135, 810, 2168, 1459)),
    "nature-ed-4-panel-b.jpg": ("nature-ed-4.jpg", (390, 0, 2089, 558)),
    "nature-ed-4-panel-c.jpg": ("nature-ed-4.jpg", (0, 580, 1005, 1142)),
    "nature-ed-4-panel-d.jpg": ("nature-ed-4.jpg", (1020, 580, 2089, 1142)),
    "nature-ed-8-panel-a.jpg": ("nature-ed-8.jpg", (0, 0, 856, 560)),
    "nature-ed-8-panels-d-f.jpg": ("nature-ed-8.jpg", (856, 560, 2116, 1141)),
    "nature-ed-9-days-1-3.jpg": ("nature-ed-9.jpg", (0, 0, 1344, 720)),
}


def main() -> None:
    for output_name, (source_name, box) in CROPS.items():
        source = ASSETS / source_name
        if not source.exists():
            raise FileNotFoundError(f"missing source figure: {source}")
        with Image.open(source) as image:
            crop = image.convert("RGB").crop(box)
            crop.save(
                ASSETS / output_name,
                format="JPEG",
                quality=90,
                optimize=True,
                progressive=True,
            )


if __name__ == "__main__":
    main()
