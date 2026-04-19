from __future__ import annotations

from pathlib import Path
from typing import Any

from ipywidgets.embed import embed_minimal_html


def save_minimal_html(
    *views: Any,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Write HTML with embedded widget state (ipywidgets ``embed_minimal_html``).

    Note: nbconvert's HTMLExporter (used in ``build_widget_htmls.ipynb``)
    bundles anywidget's ``_esm`` properly for standalone browser use, while
    ``embed_minimal_html`` relies on requireJS and may fail to load custom
    anywidgets outside JupyterLab. Prefer the nbconvert path for static HTML.
    """
    embed_minimal_html(Path(path), views=views, **kwargs)
