from __future__ import annotations

from pathlib import Path

import anywidget
import traitlets

_BUNDLE = Path(__file__).resolve().parent / "bundled" / "widget.js"


class MnistViewerWidget(anywidget.AnyWidget):
    """Custom MNIST viewer linked to a Celldega Clustergram via traitlets.

    Renders one large 28x28 tile via a single deck.gl ``SolidPolygonLayer``
    of 784 pixel quads. The active "image" depends on what the user clicked
    in the paired Clustergram:

    * **default** — average of every cluster (weighted by ``n_images``),
      red-on-white intensity to mirror the heatmap encoding.
    * **cluster** — that single cluster's mean, tinted by its majority digit.
    * **metacluster** — when a column dendrogram selects N >= 2 clusters,
      each contributing cluster keeps its own digit color and the per-pixel
      color is the ink-weighted blend of those colors (shared structure
      stays mono-color, divergent strokes split into multiple hues).
    * **filter-by-digit** — clicking a ``Majority-digit:`` color strip
      averages just the clusters of that digit.

    Row clicks / row-dendrogram selections add **blue outlines** to the
    selected pixels on top of whatever tile is showing.

    Wire format for ``cluster_summaries`` is documented in
    ``data.py::MnistClusters`` — the JS widget consumes it directly.
    ``cluster_examples`` is no longer used by the rendering but is kept on
    the model for backward compatibility.
    """

    _esm = _BUNDLE

    # Heatmap column metadata: one entry per cluster, in display order.
    # [{cluster_id, name, majority_digit, n_images, mean_full: [784 ints], digit_distribution}]
    cluster_summaries = traitlets.List(trait=traitlets.Dict(), default_value=[]).tag(sync=True)
    # Heatmap row metadata: each filtered pixel's index 0..783 (so the JS
    # widget can map "px_NNN" to a (row, col) overlay marker).
    pixel_indices = traitlets.List(trait=traitlets.Int(), default_value=[]).tag(sync=True)
    # cluster_id -> [{image_index, digit, pixels: [784 ints]}]
    cluster_examples = traitlets.Dict(default_value={}).tag(sync=True)
    # 10-color palette aligned with util.DIGIT_COLORS_HEX. Each entry is [r,g,b] 0..255.
    digit_palette_rgb = traitlets.List(default_value=[]).tag(sync=True)

    width = traitlets.Int(520).tag(sync=True)
    height = traitlets.Int(560).tag(sync=True)
    debug = traitlets.Bool(False).tag(sync=True)

    # ------------------------------------------------------------------
    # Linked-with-clustergram traits (mirrors bike_network_traffic).
    # ------------------------------------------------------------------
    click_info = traitlets.Dict(default_value={}).tag(sync=True)
    selected_rows = traitlets.List(default_value=[]).tag(sync=True)
    selected_cols = traitlets.List(default_value=[]).tag(sync=True)
    matrix_axis_slice = traitlets.Dict(default_value={}).tag(sync=True)
    # The widget can ask the Clustergram for a custom slice (kept around for
    # symmetry with bike_network_traffic — currently unused for MNIST).
    matrix_slice_request_out = traitlets.Dict(default_value={}).tag(sync=True)
    cg_row_names = traitlets.List(default_value=[]).tag(sync=True)
    cg_col_names = traitlets.List(default_value=[]).tag(sync=True)
