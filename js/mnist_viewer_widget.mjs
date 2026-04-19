// MNIST viewer anywidget — single 28x28 tile rendered with deck.gl.
//
// Geometry is one `SolidPolygonLayer` of exactly 784 axis-aligned pixel quads
// in world coords (0..28 on both axes). Colors are recomputed on the CPU per
// active selection (cheap: 784 pixels × <=N clusters) and piped to the layer
// via `getFillColor` + `updateTriggers`. A second `PolygonLayer` overlays
// stroked-only blue outlines for any pixels the user has selected via the
// Clustergram's row labels / row dendrogram.
//
// What the user sees, depending on what they clicked in the linked Clustergram:
//
//   * NO selection      -> weighted average of every cluster (red on white,
//                          mirroring the heatmap encoding).
//   * single col label  -> that cluster's mean, tinted by its majority digit
//                          (so a "Three" cluster reads as orange).
//   * col dendrogram of N>=2 clusters
//                       -> per-cluster ink-weighted color blend. Each cluster
//                          keeps its own digit color; pixels shared by all
//                          contributors stay mono-color, pixels claimed by
//                          one cluster pick up that cluster's hue.
//   * cat_value (Majority-digit)
//                       -> average of just the clusters of that digit.
//   * row label         -> blue outline on that single pixel (image stays).
//   * row dendrogram    -> blue outlines on every pixel in the selection.
//   * mat_value         -> cluster tile + single pixel outline.
//
// Wire format mirrors what `mnist_celldega.viz.make_mnist_viewer_widget` writes:
//
//   cluster_summaries  : [{cluster_id, name, majority_digit, n_images,
//                          digit_distribution, mean_full: [784 ints 0..255]}]
//   pixel_indices      : [int]            // every filtered pixel's 0..783 index
//   digit_palette_rgb  : [[r,g,b], ...]   // length 10, indexed by digit 0..9
//
// Linked traits set by the Clustergram via `jsdlink`:
//
//   click_info, selected_rows, selected_cols, matrix_axis_slice,
//   cg_row_names, cg_col_names
//   matrix_slice_request_out  (kept for symmetry with the bikes example)

import {
  COORDINATE_SYSTEM,
  Deck,
  OrthographicView,
  PolygonLayer,
  SolidPolygonLayer,
} from 'deck.gl';

// ---------------------------------------------------------------------------
// Lightweight observable store.
// ---------------------------------------------------------------------------

const Observable = (initialValue) => {
  let value = initialValue;
  const subs = new Set();
  return {
    get: () => value,
    set: (next) => {
      if (value === next) return;
      value = next;
      subs.forEach((fn) => fn(value));
    },
    subscribe: (fn, options = { immediate: true }) => {
      subs.add(fn);
      if (options.immediate) fn(value);
      return () => subs.delete(fn);
    },
  };
};

const createStore = () => ({
  cluster_summaries: Observable([]),
  pixel_indices: Observable([]),
  digit_palette_rgb: Observable([]),
  width: Observable(520),
  height: Observable(560),
  debug: Observable(false),
  click_info: Observable({}),
  selected_rows: Observable([]),
  selected_cols: Observable([]),
  matrix_axis_slice: Observable({}),
  cg_row_names: Observable([]),
  cg_col_names: Observable([]),
  // Active "what's drawn":
  //   active_cluster_ids: int[] (length >=1 => those are blended)
  //                              (empty array  => "default", i.e. all clusters)
  //   selected_pixels:    int[] of pixel indices 0..783 to outline
  //   colorize:           true => each cluster contributes its digit color;
  //                       false => everything red on white intensity
  active_cluster_ids: Observable([]),
  selected_pixels: Observable([]),
  colorize: Observable(false),
});

const log = (store, ...args) => {
  if (store.debug.get()) console.log('[mnist-viewer]', ...args);
};

// ---------------------------------------------------------------------------
// Color palette + name parsing.
// ---------------------------------------------------------------------------

const FALLBACK_DIGIT_PALETTE_HEX = [
  '#E69F00', '#0072B2', '#D55E00', '#56B4E9', '#009E73',
  '#CC79A7', '#7030A0', '#404040', '#B22222', '#1c1c1c',
];

function hexToRgb(hex) {
  const h = String(hex).replace('#', '');
  if (h.length !== 6) return [128, 128, 128];
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
const FALLBACK_DIGIT_PALETTE_RGB = FALLBACK_DIGIT_PALETTE_HEX.map(hexToRgb);

function digitRgb(store, digit) {
  const pal = store.digit_palette_rgb.get();
  if (Array.isArray(pal) && pal.length > digit) {
    const c = pal[digit];
    if (Array.isArray(c) && c.length >= 3) {
      return [Number(c[0]) | 0, Number(c[1]) | 0, Number(c[2]) | 0];
    }
  }
  return FALLBACK_DIGIT_PALETTE_RGB[digit] || [128, 128, 128];
}

// Heatmap-like base color: pure red ink on white.
const HEATMAP_INK_RGB = [220, 20, 30];

const DIGIT_NAMES = [
  'Zero', 'One', 'Two', 'Three', 'Four',
  'Five', 'Six', 'Seven', 'Eight', 'Nine',
];

/** "c007 | Three" -> 7;  bare "7" or "c7" -> 7. */
function parseClusterName(raw) {
  const s = String(raw || '').trim();
  if (!s) return null;
  const head = (s.includes('|') ? s.split('|', 1)[0] : s).trim();
  const m = head.match(/^c?(\d+)$/i);
  if (!m) return null;
  return Number(m[1]);
}

/** "px_123" -> 123. */
function parsePixelName(raw) {
  const s = String(raw || '').trim();
  const m = s.match(/^px_?(\d+)$/i);
  if (!m) return null;
  const n = Number(m[1]);
  if (!Number.isFinite(n) || n < 0 || n > 783) return null;
  return n;
}

function parseMajorityDigit(raw) {
  const s = String(raw || '').trim();
  const idx = s.indexOf(':');
  const name = (idx >= 0 ? s.slice(idx + 1) : s).trim();
  for (let i = 0; i < DIGIT_NAMES.length; i += 1) {
    if (name.toLowerCase() === DIGIT_NAMES[i].toLowerCase()) return i;
  }
  const m = name.match(/^\d$/);
  return m ? Number(m[0]) : null;
}

// ---------------------------------------------------------------------------
// Pre-built pixel polygons (constant — only ever positions, never colors).
// ---------------------------------------------------------------------------

const PIXEL_POLYS = (() => {
  const out = new Array(784);
  for (let r = 0; r < 28; r += 1) {
    for (let c = 0; c < 28; c += 1) {
      out[r * 28 + c] = {
        idx: r * 28 + c,
        polygon: [[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]],
      };
    }
  }
  return out;
})();

/** Outline polygon for a single pixel, slightly inflated so the stroke sits
 *  *around* the cell rather than splitting it. */
function pixelOutlinePoly(pixIdx, pad = 0.06) {
  const r = Math.floor(pixIdx / 28);
  const c = pixIdx % 28;
  return [
    [c - pad, r - pad],
    [c + 1 + pad, r - pad],
    [c + 1 + pad, r + 1 + pad],
    [c - pad, r + 1 + pad],
  ];
}

// ---------------------------------------------------------------------------
// Color computation.
//
// For each pixel i we ask "given the active set of contributing clusters
// (each with its own color C_k and per-pixel intensity I_ki in [0,1]) what
// color do we paint?"  We compute:
//
//   weight  = Σ_k I_ki                       (total ink across contributors)
//   meanI   = weight / N                     (avg ink per contributor)
//   chroma  = Σ_k (C_k * I_ki) / weight      (ink-weighted hue)
//   final   = lerp(white, chroma, meanI)     (lighter where less ink)
//
// When `colorize=false` (default), all C_k collapse to HEATMAP_INK_RGB and
// the formula degenerates to the obvious "red on white" lerp.
// ---------------------------------------------------------------------------

/** Per-pixel weighted-mean ink (0..1) over the active contributors — used by
 *  the hover tooltip so the value matches what the heatmap actually shows. */
function computeMeanInk(contributors) {
  const out = new Float32Array(784);
  if (contributors.length === 0) return out;
  const totalWeight = contributors.reduce((acc, c) => acc + c.weight, 0) || 1;
  for (let i = 0; i < 784; i += 1) {
    let s = 0;
    for (let k = 0; k < contributors.length; k += 1) {
      const c = contributors[k];
      s += ((Number(c.pixels[i]) || 0) / 255) * c.weight;
    }
    out[i] = s / totalWeight;
  }
  return out;
}

function computePixelColors(contributors, colorize) {
  // contributors: [{ color: [r,g,b], pixels: number[] (length 784, 0..255), weight: number }]
  // weight is `n_images` (so summing N small clusters won't outweigh one giant cluster).
  const out = new Uint8ClampedArray(784 * 4);
  const N = contributors.length;
  if (N === 0) {
    for (let i = 0; i < 784; i += 1) {
      out[i * 4] = 255; out[i * 4 + 1] = 255; out[i * 4 + 2] = 255; out[i * 4 + 3] = 255;
    }
    return out;
  }
  const totalImageWeight = contributors.reduce((acc, c) => acc + c.weight, 0) || 1;
  for (let i = 0; i < 784; i += 1) {
    let inkSum = 0; // Σ_k (I_ki * w_k) — total weighted ink at this pixel
    let rNum = 0;
    let gNum = 0;
    let bNum = 0;
    for (let k = 0; k < N; k += 1) {
      const c = contributors[k];
      const I = (Number(c.pixels[i]) || 0) / 255; // 0..1
      const w = c.weight; // n_images for cluster k
      const Iw = I * w;
      inkSum += Iw;
      rNum += c.color[0] * Iw;
      gNum += c.color[1] * Iw;
      bNum += c.color[2] * Iw;
    }
    const meanI = inkSum / totalImageWeight; // 0..1, fraction of "saturation"
    const chromaDenom = Math.max(inkSum, 1e-6);
    const cR = colorize ? rNum / chromaDenom : HEATMAP_INK_RGB[0];
    const cG = colorize ? gNum / chromaDenom : HEATMAP_INK_RGB[1];
    const cB = colorize ? bNum / chromaDenom : HEATMAP_INK_RGB[2];
    const t = Math.min(1, meanI);
    out[i * 4] = Math.round(255 * (1 - t) + cR * t);
    out[i * 4 + 1] = Math.round(255 * (1 - t) + cG * t);
    out[i * 4 + 2] = Math.round(255 * (1 - t) + cB * t);
    out[i * 4 + 3] = 255;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Main render.
// ---------------------------------------------------------------------------

function render({ model, el }) {
  const store = createStore();
  let clusterIndex = new Map(); // cluster_id -> summary

  store.cluster_summaries.subscribe((summaries) => {
    clusterIndex = new Map();
    for (const s of summaries || []) clusterIndex.set(Number(s.cluster_id), s);
  }, { immediate: true });

  // ---- DOM scaffolding ----
  const root = document.createElement('div');
  root.style.cssText =
    'background:#f5f6f8;border-radius:4px;overflow:hidden;display:flex;flex-direction:column;font:12px system-ui,sans-serif;color:#1c1f24;';
  el.appendChild(root);

  const TOPBAR_HEIGHT = 36;
  const topbar = document.createElement('div');
  topbar.style.cssText =
    `flex:0 0 ${TOPBAR_HEIGHT}px;display:flex;align-items:center;padding:0 12px;` +
    'gap:10px;background:#f5f6f8;border-bottom:1px solid #d0d3d8;user-select:none;';
  root.appendChild(topbar);

  const titleEl = document.createElement('div');
  titleEl.style.cssText =
    'flex:1 1 auto;font-weight:600;color:#1c1f24;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
  topbar.appendChild(titleEl);

  const styleToggle = (btn, on) => {
    btn.style.background = on ? '#1f77b4' : '#fff';
    btn.style.color = on ? '#fff' : '#444';
    btn.style.borderColor = on ? '#1f77b4' : '#ccc';
  };
  const colorBtn = document.createElement('button');
  colorBtn.type = 'button';
  colorBtn.textContent = 'Color clusters';
  colorBtn.title = 'Tint each contributing cluster by its majority-digit color';
  colorBtn.style.cssText =
    'flex:0 0 auto;height:24px;padding:0 10px;border:1px solid #ccc;border-radius:4px;' +
    'background:#fff;color:#444;font:inherit;cursor:pointer;transition:all .18s;';
  styleToggle(colorBtn, store.colorize.get());
  colorBtn.addEventListener('click', () => {
    const next = !store.colorize.get();
    store.colorize.set(next);
    styleToggle(colorBtn, next);
  });
  topbar.appendChild(colorBtn);

  const resetBtn = document.createElement('button');
  resetBtn.type = 'button';
  resetBtn.textContent = 'Reset';
  resetBtn.style.cssText =
    'flex:0 0 auto;height:24px;padding:0 10px;border:1px solid #ccc;border-radius:4px;' +
    'background:#fff;color:#444;font:inherit;cursor:pointer;transition:all .18s;';
  resetBtn.addEventListener('click', () => {
    store.active_cluster_ids.set([]);
    store.selected_pixels.set([]);
    refitView();
    try {
      model.set('click_info', {});
      model.save_changes();
    } catch (_e) { /* read-only in some link configs */ }
  });
  topbar.appendChild(resetBtn);

  const mapHolder = document.createElement('div');
  mapHolder.style.cssText =
    'flex:1 1 auto;position:relative;background:#ffffff;display:block;overflow:hidden;';
  root.appendChild(mapHolder);

  // ---- Deck.gl init ----
  let deck = null;
  let raf = 0;

  // Track whether the user has manually panned/zoomed. While `false`, the
  // widget keeps auto-fitting on every paint (so a window resize re-centers).
  // Becomes `true` on the first real interaction; reset by the Refit button.
  let userZoomed = false;
  let currentViewState = { target: [14, 14, 0], zoom: 0, minZoom: -3, maxZoom: 6 };

  const ensureDeck = () => {
    if (deck) return deck;
    deck = new Deck({
      parent: mapHolder,
      views: [new OrthographicView({
        id: 'mnist',
        // `smooth: true` adds a ~200ms tween on every wheel tick which feels
        // stuttery on a tiny 28x28 canvas — keep it off for direct feedback.
        controller: { scrollZoom: { speed: 0.04, smooth: false }, dragPan: true, doubleClickZoom: true, inertia: false },
        flipY: true,
      })],
      initialViewState: currentViewState,
      layers: [],
      style: { background: '#ffffff' },
      getCursor: ({ isDragging }) => (isDragging ? 'grabbing' : 'grab'),
      onViewStateChange: ({ viewState, interactionState }) => {
        currentViewState = viewState;
        // Any of these flags ⇒ a genuine user interaction (not our setProps).
        if (interactionState && (
          interactionState.isDragging
          || interactionState.isPanning
          || interactionState.isZooming
        )) {
          userZoomed = true;
        }
        scheduleRender();
      },
      getTooltip: ({ object, layer }) => {
        if (!object || !layer || layer.id !== 'pixels') return null;
        const idx = object.idx;
        const r = Math.floor(idx / 28);
        const c = idx % 28;
        const meanInk = cachedMeanInk ? cachedMeanInk[idx] : 0;
        const colors = cachedColors;
        let swatch = '';
        if (colors) {
          const k = idx * 4;
          swatch =
            `<span style="display:inline-block;width:10px;height:10px;border:1px solid #999;border-radius:2px;`
            + `background:rgb(${colors[k]},${colors[k + 1]},${colors[k + 2]});vertical-align:middle;margin-right:6px;"></span>`;
        }
        return {
          html:
            `${swatch}<b>(row ${r}, col ${c})</b><br/>`
            + `<span style="color:#555;">mean ink: ${(meanInk * 100).toFixed(0)}%</span>`,
          style: {
            backgroundColor: 'rgba(255,255,255,0.96)',
            color: '#1a1d24',
            border: '1px solid #d0d3d8',
            padding: '5px 9px',
            borderRadius: '4px',
            fontSize: '11px',
            lineHeight: '1.35',
            boxShadow: '0 1px 4px rgba(0,0,0,0.12)',
            whiteSpace: 'nowrap',
          },
        };
      },
    });
    return deck;
  };

  // ---- Active-state derivation ----
  // Translates "active_cluster_ids" + cluster_summaries into the contributor
  // list that `computePixelColors` consumes. Empty active set => all clusters.
  function getContributors() {
    const summaries = store.cluster_summaries.get() || [];
    if (summaries.length === 0) return [];
    const active = store.active_cluster_ids.get() || [];
    const ids = active.length > 0
      ? active.filter((cid) => clusterIndex.has(Number(cid))).map(Number)
      : summaries.map((s) => Number(s.cluster_id));
    const colorize = store.colorize.get();
    return ids.map((cid) => {
      const s = clusterIndex.get(cid);
      return {
        color: colorize ? digitRgb(store, s.majority_digit) : HEATMAP_INK_RGB,
        pixels: s.mean_full,
        weight: Math.max(1, Number(s.n_images) || 1),
      };
    });
  }

  // ---- Layer composition ----
  let cachedColors = null;
  let cachedMeanInk = null; // Float32Array(784): per-pixel weighted-mean ink (0..1)
  let cachedKey = '';

  function buildLayers() {
    const contributors = getContributors();
    const colorize = store.colorize.get();
    const active = store.active_cluster_ids.get() || [];
    const summaries = store.cluster_summaries.get() || [];
    const key = `${active.length === 0 ? 'all' : active.slice().sort().join(',')}|${colorize}|${summaries.length}`;
    if (key !== cachedKey) {
      cachedColors = computePixelColors(contributors, colorize);
      cachedMeanInk = computeMeanInk(contributors);
      cachedKey = key;
    }

    const colors = cachedColors;
    const layers = [];
    layers.push(new SolidPolygonLayer({
      id: 'pixels',
      data: PIXEL_POLYS,
      getPolygon: (d) => d.polygon,
      getFillColor: (d) => {
        const i = d.idx * 4;
        return [colors[i], colors[i + 1], colors[i + 2], colors[i + 3]];
      },
      updateTriggers: { getFillColor: cachedKey },
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      pickable: true,
      stroked: false,
      filled: true,
      autoHighlight: false,
    }));

    const sel = (store.selected_pixels.get() || []).filter((p) => p >= 0 && p <= 783);
    if (sel.length > 0) {
      layers.push(new PolygonLayer({
        id: 'pixel-outlines',
        data: sel,
        getPolygon: (pixIdx) => pixelOutlinePoly(pixIdx),
        getLineColor: [30, 110, 230, 240],
        getLineWidth: 2,
        getFillColor: [0, 0, 0, 0],
        lineWidthUnits: 'pixels',
        stroked: true,
        filled: false,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        pickable: false,
        updateTriggers: { getPolygon: sel.join(',') },
      }));
    }

    return layers;
  }

  // ---- Camera fit ----
  function fitToCanvas(W, H) {
    // World content spans (0..28) on both axes.
    const margin = 8;
    const contentW = 28;
    const contentH = 28;
    const zoomX = Math.log2(Math.max(0.01, (W - margin * 2) / contentW));
    const zoomY = Math.log2(Math.max(0.01, (H - margin * 2) / contentH));
    return {
      target: [14, 14, 0],
      zoom: Math.min(zoomX, zoomY),
      minZoom: -3,
      maxZoom: 6,
    };
  }

  function refitView() {
    const W = mapHolder.clientWidth || store.width.get();
    const H = mapHolder.clientHeight || (store.height.get() - TOPBAR_HEIGHT);
    currentViewState = fitToCanvas(W, H);
    userZoomed = false;
    scheduleRender();
  }

  // ---- Title ----
  function updateTitle() {
    const summaries = store.cluster_summaries.get() || [];
    const active = store.active_cluster_ids.get() || [];
    const sel = store.selected_pixels.get() || [];
    let txt;
    if (summaries.length === 0) {
      txt = 'MNIST viewer (loading…)';
    } else if (active.length === 0) {
      const totalImgs = summaries.reduce((acc, s) => acc + (Number(s.n_images) || 0), 0);
      txt = `Average of all ${summaries.length} clusters · n=${totalImgs.toLocaleString()} images`;
    } else if (active.length === 1) {
      const s = clusterIndex.get(Number(active[0]));
      if (s) {
        const dist = s.digit_distribution || {};
        const pct = Math.round((Number(dist[s.majority_digit]) || 0) * 100);
        txt = `Cluster ${s.cluster_id} · ${DIGIT_NAMES[s.majority_digit]} (${pct}%) · n=${s.n_images}`;
      } else {
        txt = `Cluster ${active[0]}`;
      }
    } else {
      // Group by majority digit so the topbar tells you the metacluster's makeup.
      const counts = new Array(10).fill(0);
      let totalN = 0;
      for (const cid of active) {
        const s = clusterIndex.get(Number(cid));
        if (!s) continue;
        counts[s.majority_digit] += 1;
        totalN += Number(s.n_images) || 0;
      }
      const parts = [];
      for (let d = 0; d < 10; d += 1) {
        if (counts[d] > 0) parts.push(`${counts[d]}×${DIGIT_NAMES[d]}`);
      }
      txt = `Metacluster · ${active.length} clusters (${parts.join(', ')}) · n=${totalN.toLocaleString()}`;
    }
    if (sel.length === 1) {
      const r = Math.floor(sel[0] / 28); const c = sel[0] % 28;
      txt = `${txt}  ·  pixel (${r},${c})`;
    } else if (sel.length > 1) {
      txt = `${txt}  ·  ${sel.length} pixels selected`;
    }
    titleEl.textContent = txt;
  }

  // ---- Render scheduler ----
  function paint() {
    const w = Number(store.width.get() || 520);
    const h = Number(store.height.get() || 560);
    root.style.width = `${w}px`;
    root.style.height = `${h}px`;
    const canvasH = Math.max(120, h - TOPBAR_HEIGHT);
    mapHolder.style.width = `${w}px`;
    mapHolder.style.height = `${canvasH}px`;

    const d = ensureDeck();
    const layers = buildLayers();
    // Auto-fit until the user pans/zooms; after that, respect their viewState.
    if (!userZoomed) currentViewState = fitToCanvas(w, canvasH);
    d.setProps({ width: w, height: canvasH, viewState: currentViewState, layers });
    updateTitle();
  }

  const scheduleRender = () => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      paint();
    });
  };

  // ---- Linked-state derivation ----
  // Track per-event "intent" so re-clicking the same target acts as a toggle.
  let lastActionKey = null;
  let lastActionSeq = -1;
  let interactionSeq = 0;

  function deriveFromClickInfo() {
    const info = store.click_info.get() || {};
    const t = String(info.type || '').replace('-', '_');
    const v = info.value || {};
    const seq = interactionSeq;
    log(store, 'click_info', t, v);
    if (!t) return;

    // BULLETPROOF: any non-row-axis interaction wipes the blue outlines.
    // The row branches below (`row_label`, `row_dendro`, `mat_value` with a
    // pix) re-populate `selected_pixels` after this clear. This guarantees no
    // stale outlines can leak through unhandled event types or race
    // conditions with the Clustergram's own selection traitlets.
    const rowAxisEvent =
      t === 'row_label' || t === 'row_dendro'
      || (t === 'mat_value' && parsePixelName(((v.row || {}).name)) != null);
    if (!rowAxisEvent) {
      store.selected_pixels.set([]);
    }

    if (t === 'col_label') {
      const cid = parseClusterName(v.name);
      if (cid == null) return;
      const k = `col:${cid}`;
      if (k === lastActionKey && seq !== lastActionSeq) {
        store.active_cluster_ids.set([]);
        lastActionKey = null; lastActionSeq = seq;
        return;
      }
      store.active_cluster_ids.set([cid]);
      lastActionKey = k; lastActionSeq = seq;
      return;
    }

    if (t === 'row_label') {
      const pix = parsePixelName(v.name);
      if (pix == null) return;
      const k = `row:${pix}`;
      if (k === lastActionKey && seq !== lastActionSeq) {
        store.selected_pixels.set([]);
        lastActionKey = null; lastActionSeq = seq;
        return;
      }
      store.selected_pixels.set([pix]);
      lastActionKey = k; lastActionSeq = seq;
      return;
    }

    if (t === 'mat_value') {
      const cid = parseClusterName((v.col || {}).name);
      const pix = parsePixelName((v.row || {}).name);
      if (cid == null && pix == null) return;
      const k = `mat:${cid}:${pix}`;
      if (k === lastActionKey && seq !== lastActionSeq) {
        store.active_cluster_ids.set([]);
        store.selected_pixels.set([]);
        lastActionKey = null; lastActionSeq = seq;
        return;
      }
      if (cid != null) store.active_cluster_ids.set([cid]);
      if (pix != null) store.selected_pixels.set([pix]);
      lastActionKey = k; lastActionSeq = seq;
      return;
    }

    if (t === 'cat_value') {
      // Categorical-strip click is column-axis (Majority-digit lives on the
      // column strip). Always clear any pixel outlines.
      const summaries = store.cluster_summaries.get() || [];
      const digit = parseMajorityDigit(v.value);
      let ids;
      if (digit != null) {
        ids = summaries
          .filter((s) => Number(s.majority_digit) === digit)
          .map((s) => Number(s.cluster_id));
      } else if (Array.isArray(v.node_names)) {
        ids = v.node_names.map(parseClusterName).filter((x) => x != null);
      } else {
        ids = [];
      }
      store.selected_pixels.set([]);
      const k = `cat:${v.axis}:${v.attr_index}:${String(v.value)}`;
      if (k === lastActionKey && seq !== lastActionSeq) {
        store.active_cluster_ids.set([]);
        lastActionKey = null; lastActionSeq = seq;
        return;
      }
      if (ids.length > 0) {
        store.active_cluster_ids.set(ids);
        // For multi-cluster filters, default to colorized so the digit is obvious.
        if (ids.length >= 2 && !store.colorize.get()) {
          store.colorize.set(true);
          styleToggle(colorBtn, true);
        }
      }
      lastActionKey = k; lastActionSeq = seq;
      return;
    }

    if (t === 'col_dendro') {
      handleColDendro(v, seq);
      return;
    }
    if (t === 'row_dendro') {
      handleRowDendro(v, seq);
      return;
    }
  }

  function handleColDendro(v, seq) {
    const sel = Array.isArray(v.selected_names) ? v.selected_names : [];
    const fromCols = (store.selected_cols.get() || []).length > 0
      ? (store.selected_cols.get() || [])
      : sel;
    const ids = [...new Set(fromCols.map(parseClusterName).filter((c) => c != null))];
    // Column-axis interaction → always clear pixel outlines.
    store.selected_pixels.set([]);
    if (v.is_unselecting || ids.length === 0) {
      store.active_cluster_ids.set([]);
      lastActionKey = null; lastActionSeq = seq;
      return;
    }
    store.active_cluster_ids.set(ids);
    // Auto-enable per-cluster coloring on multi-select so the metacluster
    // visualization actually shows multiple colors.
    if (ids.length >= 2 && !store.colorize.get()) {
      store.colorize.set(true);
      styleToggle(colorBtn, true);
    }
    lastActionKey = `col_dendro:${ids.slice().sort().join('|')}`;
    lastActionSeq = seq;
  }

  function handleRowDendro(v, seq) {
    const sel = Array.isArray(v.selected_names) ? v.selected_names : [];
    const fromRows = (store.selected_rows.get() || []).length > 0
      ? (store.selected_rows.get() || [])
      : sel;
    const pixIdx = [...new Set(fromRows.map(parsePixelName).filter((p) => p != null))];
    if (v.is_unselecting || pixIdx.length === 0) {
      store.selected_pixels.set([]);
      lastActionKey = null; lastActionSeq = seq;
      return;
    }
    store.selected_pixels.set(pixIdx);
    lastActionKey = `row_dendro:${pixIdx.slice().sort().join('|')}`;
    lastActionSeq = seq;
  }

  // ---- Subscriptions ----
  [
    store.cluster_summaries,
    store.pixel_indices,
    store.digit_palette_rgb,
    store.active_cluster_ids,
    store.selected_pixels,
    store.colorize,
    store.width,
    store.height,
  ].forEach((obs) => obs.subscribe(() => scheduleRender(), { immediate: false }));

  store.click_info.subscribe(() => {
    interactionSeq += 1;
    deriveFromClickInfo();
    scheduleRender();
  }, { immediate: false });

  // NOTE: We deliberately do NOT auto-apply selections from the
  // `selected_rows` / `selected_cols` traitlets. Doing so causes stale row
  // selections to "snap back" the moment the user clicks a column (the
  // Clustergram doesn't clear `selected_rows` on column clicks). All
  // axis-selection state flows through `click_info` instead — col_label,
  // col_dendro, row_label, row_dendro, mat_value, cat_value — which is
  // self-consistent because each of those events knows which axis "owns"
  // pixel outlines.

  // ---- Resize observer ----
  // On resize: if the user hasn't manually zoomed/panned we auto-refit (next
  // paint pass picks up the new dimensions); if they have, we leave their
  // current viewport alone — they can hit the "Reset" button to refit.
  let lastSize = { w: 0, h: 0 };
  const ro = new ResizeObserver(() => {
    const w = mapHolder.clientWidth;
    const h = mapHolder.clientHeight;
    if (w !== lastSize.w || h !== lastSize.h) {
      lastSize = { w, h };
      scheduleRender();
    }
  });
  ro.observe(mapHolder);

  // ---- Sync from model ----
  function syncFromModel() {
    store.cluster_summaries.set(model.get('cluster_summaries') || []);
    store.pixel_indices.set(model.get('pixel_indices') || []);
    store.digit_palette_rgb.set(model.get('digit_palette_rgb') || []);
    store.click_info.set(model.get('click_info') || {});
    store.selected_rows.set(model.get('selected_rows') || []);
    store.selected_cols.set(model.get('selected_cols') || []);
    store.matrix_axis_slice.set(model.get('matrix_axis_slice') || {});
    store.cg_row_names.set(model.get('cg_row_names') || []);
    store.cg_col_names.set(model.get('cg_col_names') || []);
    store.width.set(Number(model.get('width')) || 520);
    store.height.set(Number(model.get('height')) || 560);
    store.debug.set(Boolean(model.get('debug')));
    scheduleRender();
  }

  model.on('change:click_info', () => {
    interactionSeq += 1;
    store.click_info.set(model.get('click_info') || {});
    deriveFromClickInfo();
    scheduleRender();
  });
  [
    'cluster_summaries',
    'pixel_indices',
    'digit_palette_rgb',
    'matrix_axis_slice',
    'cg_row_names',
    'cg_col_names',
    'width',
    'height',
    'debug',
  ].forEach((name) => model.on(`change:${name}`, syncFromModel));
  model.on('change:selected_rows', () => {
    store.selected_rows.set(model.get('selected_rows') || []);
  });
  model.on('change:selected_cols', () => {
    store.selected_cols.set(model.get('selected_cols') || []);
  });

  syncFromModel();

  return () => {
    if (raf) cancelAnimationFrame(raf);
    if (deck) { try { deck.finalize(); } catch (_e) { /* noop */ } }
    ro.disconnect();
  };
}

export default { render };
