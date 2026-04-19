"""Higher-level helpers wrapping Celldega + the MNIST viewer widget.

The notebooks should only need :func:`make_mnist_clustergram`,
:func:`make_mnist_viewer_widget`, and :func:`link_viewer_to_clustergram`. The
two-pass clustering trick used by ``bike_network_traffic`` isn't necessary
here because the cluster identity is decided upstream in
:func:`mnist_celldega.cluster_mnist` — we just need the Celldega ``Matrix``
to render in column-id order with the per-digit color strip wired up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mnist_celldega.data import DIGIT_NAMES, MnistClusters, pixel_metadata
from mnist_celldega.util import DIGIT_COLORS_HEX, digit_palette_rgb
from mnist_celldega.widget import MnistViewerWidget

if TYPE_CHECKING:
    pass

__all__ = [
    "make_mnist_clustergram",
    "make_mnist_viewer_widget",
    "link_viewer_to_clustergram",
]


def make_mnist_clustergram(
    clusters: MnistClusters,
) -> tuple[Any, Any]:
    """Build a Celldega ``Matrix`` + ``Clustergram`` from a clustering bundle.

    Adds two metadata strips:

    * **column** ``Majority-digit: <Name>`` — one of "Zero" .. "Nine"
    * **row** ``Center`` — value-based, radial proximity to the image center

    Column colors come from :data:`mnist_celldega.util.DIGIT_COLORS_HEX`, so
    the heatmap legend matches the MNIST viewer's per-digit tints.
    """
    import celldega as dega
    import pandas as pd

    df = clusters.cluster_means

    col_names = list(df.columns)
    col_majority = []
    col_n_images = []
    for name in col_names:
        # name format is "c{cid:03d} | {DigitName}"; we keep both halves.
        try:
            cid = int(str(name).split("|", 1)[0].strip().lstrip("c"))
        except Exception:
            cid = 0
        digit = clusters.majority_digit.get(cid, 0)
        col_majority.append(f"Majority-digit: {DIGIT_NAMES[int(digit)]}")
        col_n_images.append(int(clusters.cluster_size.get(cid, 0)))
    meta_col = pd.DataFrame(
        {
            "Majority-digit": col_majority,
            # Numeric → Celldega renders this as a *value* strip with shading,
            # mirroring the "number in clust" attribute in the Clustergrammer
            # MNIST_heatmaps reference notebook.
            "Number-in-cluster": col_n_images,
        },
        index=df.columns,
    )

    meta_row = pixel_metadata(clusters.pixel_indices)

    # Celldega infers categorical vs value-type from the dtype of the column —
    # `Center` (float) becomes a value-based row strip, `Majority-digit`
    # (string) becomes a categorical column strip, `Number-in-cluster` (int)
    # becomes a value-based column strip.
    mat = dega.clust.Matrix(
        df,
        meta_row=meta_row,
        meta_col=meta_col,
        row_attr=["Center"],
        col_attr=["Majority-digit", "Number-in-cluster"],
    )

    cat_colors: dict[str, str] = {}
    for digit, hex_color in enumerate(DIGIT_COLORS_HEX):
        cat_colors[f"Majority-digit: {DIGIT_NAMES[digit]}"] = hex_color
    mat.set_global_cat_colors(cat_colors)

    mat.cluster(force=True)
    mat.make_viz()

    cgm = dega.viz.Clustergram(matrix=mat)
    return mat, cgm


def make_mnist_viewer_widget(
    clusters: MnistClusters,
    *,
    width: int = 520,
    height: int = 560,
    debug: bool = False,
) -> MnistViewerWidget:
    """Build an ``MnistViewerWidget`` populated from a clustering bundle."""
    summaries = []
    for cid in clusters.cluster_ids:
        summaries.append(
            {
                "cluster_id": int(cid),
                "name": f"c{cid:03d} | {DIGIT_NAMES[clusters.majority_digit[cid]]}",
                "majority_digit": int(clusters.majority_digit[cid]),
                "n_images": int(clusters.cluster_size[cid]),
                "digit_distribution": {
                    str(k): float(v) for k, v in clusters.digit_distribution[cid].items()
                },
                "mean_full": clusters.cluster_mean_full[cid],
            }
        )

    examples_payload = {
        str(cid): [
            {
                "image_index": ex.image_index,
                "digit": ex.digit,
                "pixels": ex.pixels,
            }
            for ex in clusters.cluster_examples[cid]
        ]
        for cid in clusters.cluster_ids
    }

    viewer = MnistViewerWidget(width=width, height=height, debug=debug)
    viewer.cluster_summaries = summaries
    viewer.pixel_indices = list(clusters.pixel_indices)
    viewer.cluster_examples = examples_payload
    viewer.digit_palette_rgb = digit_palette_rgb()
    return viewer


def link_viewer_to_clustergram(viewer: MnistViewerWidget, cgm: Any) -> None:
    """Wire ``jsdlink``s between the MNIST viewer and a Celldega Clustergram.

    All links are frontend-only so the pair survives in static HTML (no kernel needed).
    Mirrors the link set in ``bike_network_traffic.viz.link_flow_to_clustergram``.
    """
    from ipywidgets import jsdlink

    jsdlink((cgm, "click_info"), (viewer, "click_info"))
    jsdlink((cgm, "selected_rows"), (viewer, "selected_rows"))
    jsdlink((cgm, "selected_cols"), (viewer, "selected_cols"))
    jsdlink((cgm, "matrix_slice_result"), (viewer, "matrix_axis_slice"))
    jsdlink((viewer, "matrix_slice_request_out"), (cgm, "matrix_slice_request"))
    jsdlink((cgm, "row_names"), (viewer, "cg_row_names"))
    jsdlink((cgm, "col_names"), (viewer, "cg_col_names"))
