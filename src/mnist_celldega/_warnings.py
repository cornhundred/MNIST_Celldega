"""Targeted warning filters for the noisy third-party deps celldega + scanpy pull in.

We only silence specific, well-understood warnings (matched by message + module)
rather than blanket-ignoring categories — that way unrelated warnings still
surface.

Call :func:`silence_warnings` *before* importing celldega / scanpy for the
import-time warnings to be suppressed.
"""

from __future__ import annotations

import warnings

__all__ = ["silence_warnings"]


def silence_warnings(*, also_celldega: bool = True) -> None:
    """Install warning filters for the known-noisy deps. Safe to call repeatedly."""
    warnings.filterwarnings(
        "ignore",
        message=".*legacy Dask DataFrame implementation is deprecated.*",
        category=FutureWarning,
        module=r"dask\.dataframe.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"xarray_schema.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Importing read_text from `anndata` is deprecated.*",
        category=FutureWarning,
        module=r"anndata.*",
    )
    # scanpy emits a flurry of "is_categorical_dtype is deprecated" notices
    # when we feed it AnnData built fresh from numpy arrays.
    warnings.filterwarnings(
        "ignore",
        message=".*is_categorical_dtype is deprecated.*",
        category=FutureWarning,
    )
    if also_celldega:
        warnings.filterwarnings(
            "ignore",
            message=r".*Large matrix .* may cause memory issues.*",
            category=UserWarning,
            module=r"celldega\.clust.*",
        )
