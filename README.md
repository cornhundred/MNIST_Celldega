# MNIST_Celldega

https://cornhundred.github.io/MNIST_Celldega/

Linked **MNIST digit viewer** (custom anywidget) ↔ **Celldega Clustergram** of
Leiden clusters of handwritten digits.

Built in the same style as
[bike_network_traffic](https://github.com/cornhundred/bike_network_traffic):
two anywidgets that talk to each other in the browser via `jsdlink`, packaged
behind a tiny Python API and embedded in a static landing page through iframes.

## What it does

1. Loads the 70,000-image MNIST dataset (via `sklearn.datasets.fetch_openml`).
2. Runs Leiden clustering (scanpy) — either across all digits, or restricted to
   one digit at a time — to produce ~100 image clusters per view.
3. Builds a Celldega Clustergram with:
   - **columns** = image clusters (labeled by majority digit, colored per digit)
   - **rows** = top-N pixels by sum (corner pixels with no signal filtered out)
   - **values** = average pixel intensity inside the cluster
4. Pairs that with a custom MNIST viewer widget that displays:
   - the grid of all cluster averages (default view)
   - sample images from the clicked cluster (column click)
   - the clicked pixel highlighted on every cluster average (row click)
   - the clicked cell magnified, with the pixel marked (matrix click)

The two views are synced live via `ipywidgets.jsdlink` so the static HTML
exports work without a Python kernel.

## Quickstart

```bash
git clone https://github.com/cornhundred/MNIST_Celldega.git
cd MNIST_Celldega

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[notebooks]"

jupyter lab
```

Open `MNIST.ipynb` and run all cells, or run `build_widget_htmls.ipynb` to
regenerate the embedded HTMLs that the landing page (`index.html`) loads.

## API at a glance

```python
from mnist_celldega import (
    silence_warnings, get_mnist_data, cluster_mnist,
    make_mnist_clustergram, make_mnist_viewer_widget, link_viewer_to_clustergram,
)
from ipywidgets import HBox, Layout

silence_warnings()
ds = get_mnist_data()
clusters = cluster_mnist(ds, mode="all", n_clusters=100)
mat, cgm = make_mnist_clustergram(clusters)
viewer = make_mnist_viewer_widget(clusters, height=700)
link_viewer_to_clustergram(viewer, cgm)
viewer.layout = Layout(width="560px", height="700px")
cgm.layout    = Layout(width="720px", height="700px")
HBox([viewer, cgm])
```

`mode="digit"` + `digit=3` restricts clustering to one digit type.
