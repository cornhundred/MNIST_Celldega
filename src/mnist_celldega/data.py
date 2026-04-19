"""MNIST loading + Leiden cluster aggregation for the clustergram + viewer pair.

The MNIST dataset (70,000 28x28 grayscale handwritten digits) is fetched via
``sklearn.datasets.fetch_openml`` and cached on disk by sklearn itself. From
there we run a Leiden community-detection clustering (scanpy) — either across
the whole dataset, or restricted to one digit — and bin every original image
into one of those ~100 clusters.

Two artifacts come out, both shaped to drive a Celldega Clustergram + the
custom MNIST viewer widget:

* **`cluster_means`**  ``(n_pixels x n_clusters)`` DataFrame, optionally
  filtered to the top-N pixels by sum so the corner-of-image always-zero
  pixels don't waste rows. Rows: ``"px_{i:03d}"`` style names matching the
  784-pixel layout. Columns: ``"c{cid:03d} | {digit_name}"`` — the cluster id
  plus its majority digit, both used by the JS widget for axis lookups.
* **`cluster_examples`**  ``cluster_id -> list[ImageExample]``, ~16 sample
  images per cluster (capped) so the viewer can show a small grid of
  "what's in this cluster?" thumbnails on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "DIGIT_NAMES",
    "ImageExample",
    "MnistData",
    "MnistClusters",
    "cluster_mnist",
    "get_mnist_data",
    "pixel_metadata",
]


# Spelled-out names match the Clustergrammer MNIST notebook so the legend reads
# nicely (e.g. "Majority-digit: Three" rather than "Majority-digit: 3").
DIGIT_NAMES: tuple[str, ...] = (
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
)


@dataclass
class ImageExample:
    """A single MNIST sample picked as a representative for its cluster."""

    image_index: int
    digit: int
    pixels: list[int]  # length 784, 0..255 ints (compact for JSON wire format)


@dataclass
class MnistData:
    """Raw MNIST: ``(70000, 784)`` float images in ``[0, 1]`` plus integer labels."""

    images: np.ndarray  # shape (n, 784), float32 in [0, 1]
    labels: np.ndarray  # shape (n,), int8 in [0..9]

    @property
    def n_images(self) -> int:
        return int(self.images.shape[0])


@dataclass
class MnistClusters:
    """One Leiden clustering of MNIST plus everything the viewer/Clustergram need."""

    mode: Literal["all", "digit"]
    digit: int | None  # set when mode == 'digit'
    n_clusters: int
    n_top_pixels: int
    # 1-based cluster ids matching the column names of `cluster_means` after sorting.
    cluster_ids: list[int]
    # Filtered DataFrame (rows = filtered pixels, cols = clusters).
    cluster_means: "pd.DataFrame"
    # Per-cluster majority digit (digit 0..9). Aligned with `cluster_ids`.
    majority_digit: dict[int, int]
    # Per-cluster size (# original images that landed in this cluster).
    cluster_size: dict[int, int]
    # Per-cluster digit distribution (digit -> share in 0..1). Aligned with `cluster_ids`.
    digit_distribution: dict[int, dict[int, float]]
    # Examples per cluster: 16 randomly picked images (or fewer if cluster is tiny).
    cluster_examples: dict[int, list[ImageExample]] = field(default_factory=dict)
    # Pixel index (0..783) for every row in `cluster_means`, in row order.
    pixel_indices: list[int] = field(default_factory=list)
    # The full-resolution mean image per cluster (length 784, 0..255 ints) — used
    # by the viewer to render the per-cluster average tile in the overview grid
    # without having to expand the filtered Clustergram matrix back to 784.
    cluster_mean_full: dict[int, list[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MNIST loader
# ---------------------------------------------------------------------------


def get_mnist_data(
    *,
    cache_dir: str | None = None,
    subsample: int | None = None,
    random_state: int = 0,
) -> MnistData:
    """Load MNIST via sklearn's openml cache.

    Parameters
    ----------
    cache_dir
        Forwarded to ``fetch_openml(data_home=...)``. Default uses sklearn's
        usual cache location (``~/scikit_learn_data``).
    subsample
        If set, randomly downsample to this many images (handy for quick
        iteration). Default ``None`` keeps the full 70k.
    random_state
        Seed for the optional subsample.
    """
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=cache_dir,
        parser="auto",
    )
    X = np.asarray(bunch.data, dtype=np.float32) / 255.0  # noqa: N806 — sklearn convention
    y = np.asarray(bunch.target, dtype=np.int16).astype(np.int8)

    if subsample is not None and subsample < X.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=int(subsample), replace=False)
        X = X[idx]
        y = y[idx]

    return MnistData(images=X, labels=y)


# ---------------------------------------------------------------------------
# Leiden clustering + aggregation
# ---------------------------------------------------------------------------


def _run_leiden(
    images: np.ndarray,
    *,
    n_clusters_target: int,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    random_state: int = 0,
    resolution_grid: tuple[float, ...] = (
        0.3, 0.5, 0.8, 1.0, 1.4, 1.8, 2.2, 2.8, 3.5, 4.5, 6.0, 8.0, 10.0,
    ),
) -> np.ndarray:
    """Run scanpy's Leiden community detection, sweeping resolution to land
    near ``n_clusters_target`` clusters.

    We do a coarse sweep over ``resolution_grid`` and pick the resolution
    whose resulting #clusters is closest to the target. PCA is computed once
    and reused across the sweep.
    """
    import anndata as ad
    import scanpy as sc

    adata = ad.AnnData(X=images.astype(np.float32))
    n_pcs_eff = int(min(n_pcs, max(2, min(adata.shape) - 1)))
    sc.pp.pca(adata, n_comps=n_pcs_eff, random_state=random_state)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=random_state)

    best_labels: np.ndarray | None = None
    best_diff = float("inf")
    for res in resolution_grid:
        sc.tl.leiden(
            adata,
            resolution=float(res),
            random_state=random_state,
            key_added="leiden_tmp",
            flavor="igraph",
            directed=False,
            n_iterations=2,
        )
        labels = adata.obs["leiden_tmp"].astype(int).to_numpy()
        n_found = int(labels.max()) + 1
        diff = abs(n_found - n_clusters_target)
        if diff < best_diff:
            best_diff = diff
            best_labels = labels.copy()
        # Stop early if we overshoot; the function is monotone-ish.
        if n_found >= n_clusters_target * 1.4:
            break

    if best_labels is None:
        raise RuntimeError("Leiden sweep produced no clusters")
    return best_labels


def cluster_mnist(
    data: MnistData,
    *,
    mode: Literal["all", "digit"] = "all",
    digit: int | None = None,
    n_clusters: int = 100,
    n_top_pixels: int = 500,
    n_examples: int = 16,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    random_state: int = 0,
) -> MnistClusters:
    """Cluster MNIST with Leiden and bundle everything the visualization needs.

    Parameters
    ----------
    data
        Output of :func:`get_mnist_data`.
    mode
        ``"all"`` clusters the entire dataset; ``"digit"`` filters to a single
        digit type first (set ``digit``).
    digit
        Required when ``mode == "digit"``. Integer in ``0..9``.
    n_clusters
        Target number of Leiden clusters (we sweep resolution to land near it).
    n_top_pixels
        Keep the top-N pixels (by sum across the dataset) as Clustergram rows;
        the rest are dropped. 500 ≈ matches the MaayanLab MNIST notebook.
    n_examples
        Sample this many example images per cluster for the viewer's grid.
        Smaller clusters keep all their members.
    n_neighbors, n_pcs, random_state
        Forwarded to scanpy's neighbor graph + PCA.
    """
    if mode == "digit":
        if digit is None:
            raise ValueError("mode='digit' requires `digit=` (0..9)")
        if not (0 <= int(digit) <= 9):
            raise ValueError("digit must be 0..9")
        sel = data.labels == int(digit)
        images = data.images[sel]
        labels = data.labels[sel]
        # Use sequential indices into the *filtered* arrays; we still record
        # the original MNIST row index in each ImageExample for full reverse-lookup.
        original_indices = np.flatnonzero(sel)
    elif mode == "all":
        images = data.images
        labels = data.labels
        original_indices = np.arange(images.shape[0])
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    leiden_labels = _run_leiden(
        images,
        n_clusters_target=n_clusters,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        random_state=random_state,
    )
    n_found = int(leiden_labels.max()) + 1

    # Build per-cluster arrays.
    rng = np.random.default_rng(random_state)
    cluster_examples: dict[int, list[ImageExample]] = {}
    cluster_size: dict[int, int] = {}
    majority_digit: dict[int, int] = {}
    digit_distribution: dict[int, dict[int, float]] = {}
    cluster_mean_full: dict[int, list[int]] = {}
    cluster_means_full = np.zeros((n_found, 784), dtype=np.float32)

    for cid in range(n_found):
        mask = leiden_labels == cid
        members = np.flatnonzero(mask)
        cluster_size[cid + 1] = int(members.size)
        cluster_means_full[cid] = images[mask].mean(axis=0)
        cluster_mean_full[cid + 1] = (cluster_means_full[cid] * 255).clip(0, 255).astype(np.int16).tolist()

        member_digits = labels[mask]
        # Majority digit: most-common label in cluster (ties broken by smallest digit).
        counts = np.bincount(member_digits, minlength=10)
        majority_digit[cid + 1] = int(counts.argmax())
        total = int(counts.sum()) or 1
        digit_distribution[cid + 1] = {int(d): float(counts[d]) / total for d in range(10) if counts[d] > 0}

        if members.size <= n_examples:
            picks = members
        else:
            picks = rng.choice(members, size=n_examples, replace=False)
        cluster_examples[cid + 1] = [
            ImageExample(
                image_index=int(original_indices[i]),
                digit=int(labels[i]),
                pixels=(images[i] * 255).clip(0, 255).astype(np.int16).tolist(),
            )
            for i in picks
        ]

    # Top-pixel filter (sum over all images, not only cluster means — matches
    # the MaayanLab notebook's `filter_N_top('row', rank_type='sum', N_top=500)`).
    pixel_sum = images.sum(axis=0)
    n_keep = int(min(n_top_pixels, 784))
    # Argsort descending, then sort the resulting indices ascending so pixel
    # row order in the heatmap matches their natural left-to-right top-to-bottom layout.
    top_pixel_idx = np.sort(np.argsort(-pixel_sum)[:n_keep])
    pixel_indices = top_pixel_idx.tolist()

    cluster_means_filt = cluster_means_full[:, top_pixel_idx]  # (n_clusters, n_pixels)

    import pandas as pd

    cluster_ids = list(range(1, n_found + 1))
    col_names = [_format_cluster_name(cid, majority_digit[cid]) for cid in cluster_ids]
    row_names = [f"px_{int(idx):03d}" for idx in pixel_indices]
    cluster_means = pd.DataFrame(cluster_means_filt.T, index=row_names, columns=col_names)

    return MnistClusters(
        mode=mode,
        digit=int(digit) if digit is not None else None,
        n_clusters=n_found,
        n_top_pixels=n_keep,
        cluster_ids=cluster_ids,
        cluster_means=cluster_means,
        majority_digit=majority_digit,
        cluster_size=cluster_size,
        digit_distribution=digit_distribution,
        cluster_examples=cluster_examples,
        pixel_indices=pixel_indices,
        cluster_mean_full=cluster_mean_full,
    )


def _format_cluster_name(cluster_id: int, majority: int) -> str:
    """Heatmap column name. Includes the cluster id (zero-padded for stable
    sort order) and the majority digit so labels read e.g. ``"c007 | Three"``.
    The JS widget parses both halves so either form is recognized in linkages.
    """
    return f"c{cluster_id:03d} | {DIGIT_NAMES[majority]}"


# ---------------------------------------------------------------------------
# Pixel metadata for the Clustergram row axis
# ---------------------------------------------------------------------------


def pixel_metadata(pixel_indices: list[int]) -> "pd.DataFrame":
    """Per-pixel metadata table indexed by row name (``px_NNN``).

    Provides:

    * ``row``, ``col`` — the pixel's grid position (0..27).
    * ``Center`` — radial proximity to the 28x28 image center, in 0..1; matches
      the MaayanLab notebook's ``Center`` value-based category that they use to
      "highlight broad patterns in pixel distributions" by reordering rows.
    """
    import pandas as pd

    rows = []
    for idx in pixel_indices:
        r, c = divmod(int(idx), 28)
        # Distance from the image center (13.5, 13.5), normalized so the corners
        # are 0 and the center is 1.
        dy = (r - 13.5) / 13.5
        dx = (c - 13.5) / 13.5
        d = float((dy * dy + dx * dx) ** 0.5)
        center = max(0.0, 1.0 - d / 2.0**0.5)
        rows.append({"row": int(r), "col": int(c), "Center": round(center, 4)})
    return pd.DataFrame(rows, index=[f"px_{int(i):03d}" for i in pixel_indices])
