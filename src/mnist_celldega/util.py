"""Color palettes shared between the JS widget and the Python clustergram.

`DIGIT_COLORS_HEX` mirrors the categorical colors set in the
[MaayanLab Clustergrammer MNIST notebook](https://nbviewer.org/github/MaayanLab/MNIST_heatmaps/blob/master/notebooks/MNIST_Notebook.ipynb)
so the cross-tool color story (heatmap cat strip ↔ MNIST viewer tints) stays
consistent. Every entry is a colorblind-aware contrast against the others.
"""

from __future__ import annotations

# Colors per digit (0..9). Keep in sync with `DIGIT_COLORS_HEX` in
# `js/mnist_viewer_widget.mjs`.
DIGIT_COLORS_HEX: tuple[str, ...] = (
    "#E69F00",  # 0 — orange/yellow (Clustergrammer original used 'yellow')
    "#0072B2",  # 1 — blue
    "#D55E00",  # 2 — vermilion (orange-red)
    "#56B4E9",  # 3 — sky/aqua
    "#009E73",  # 4 — bluish green (lime)
    "#CC79A7",  # 5 — pink/magenta
    "#7030A0",  # 6 — purple
    "#404040",  # 7 — dark gray (so it reads against white digit-7 strokes)
    "#B22222",  # 8 — firebrick red
    "#1c1c1c",  # 9 — near-black
)


def hex_palette_to_rgb(colors: list[str] | tuple[str, ...]) -> list[list[int]]:
    """`#RRGGBB` strings to ``[[r, g, b], ...]`` for the widget's rgb traits."""
    out: list[list[int]] = []
    for h in colors:
        s = str(h).strip()
        if len(s) == 7 and s.startswith("#"):
            out.append([int(s[i : i + 2], 16) for i in (1, 3, 5)])
    return out


def digit_palette_rgb() -> list[list[int]]:
    """RGB triplets for digits 0..9, suitable for the viewer widget trait."""
    return hex_palette_to_rgb(DIGIT_COLORS_HEX)
