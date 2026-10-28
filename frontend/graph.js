// SVG graph renderer, Netron-inspired: compact nodes (op + output tensor),
// click a node to expand its details in place (HTML overlay), click again to
// collapse; edge tensor-name labels; pan / zoom / viewport culling.

const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  return e;
}

function htmlEl(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

export class GraphView {
  constructor(container, opts = {}) {
    this.container = container;
    this.opts = opts; // onSelect, onEmptyClick, onExpand, detailHTML(id)->string|null, showTensorNames
    this.svg = el("svg");
    this.viewport = el("g", { class: "viewport" });
    this.edgeLayer = el("g", { class: "edges" });
    this.edgeLabelLayer = el("g", { class: "edge-labels" });
    this.nodeLayer = el("g", { class: "nodes" });
    this.viewport.appendChild(this.edgeLayer);
    this.viewport.appendChild(this.edgeLabelLayer);
    this.viewport.appendChild(this.nodeLayer);
    this.svg.appendChild(this.viewport);
    container.appendChild(this.svg);

    this.overlay = htmlEl("div", "graph-overlay");
    container.appendChild(this.overlay);

    this.tx = 0; this.ty = 0; this.scale = 1;
    this.boxes = {};
    this.edgePoints = {};
    this.nodeExtra = {};      // id -> {op, outTensors, inTensors}
    this.rendered = new Map();
    this.selected = null;
    this.searchSet = new Set();
    this.styles = {};
    this.expanded = new Set();
    this._overlayCards = new Map(); // id -> element
    this._pending = false;

    this.svg.addEventListener("wheel", (ev) => this._onWheel(ev), { passive: false });
    this.svg.addEventListener("pointerdown", (ev) => this._onPointerDown(ev));
    window.addEventListener("resize", () => this._scheduleRender());
  }

  // ---- data ----
  setData(layout, nodeExtra = {}) {
    this.boxes = layout.boxes || {};
    this.edgePoints = layout.edge_points || {};
    this.nodeExtra = nodeExtra || {};
    this.rendered.clear();
    this.edgeLayer.innerHTML = "";
    this.edgeLabelLayer.innerHTML = "";
    this.nodeLayer.innerHTML = "";
    this.selected = null;
    this.searchSet.clear();
    this.collapseAll();
    this._layoutW = layout.width || 0;
    this._layoutH = layout.height || 0;
    this.fitView();
  }

  clear() {
    this.boxes = {}; this.edgePoints = {}; this.rendered.clear();
    this.edgeLayer.innerHTML = ""; this.edgeLabelLayer.innerHTML = "";
    this.nodeLayer.innerHTML = "";
    this.selected = null; this.searchSet.clear(); this.styles = {};
    this.collapseAll();
  }

  setStyles(styles) {
    this.styles = styles || {};
    this._scheduleRender(true);
  }

  setSearchHighlight(ids) {
    this.searchSet = new Set(ids || []);
    this._scheduleRender(true);
  }

  // ---- expand/collapse (Netron-style detail in place) ----
  toggleExpand(id) {
    if (this.expanded.has(id)) this.expanded.delete(id);
    else this.expanded.add(id);
    this._syncOverlays();
    if (this.opts.onExpand) this.opts.onExpand(id, this.expanded.has(id));
  }

  collapseAll() {
    if (!this.expanded.size) return;
    this.expanded.clear();
    this._syncOverlays();
  }

  _syncOverlays() {
    // remove cards for collapsed/absent nodes
    for (const [id, card] of this._overlayCards) {
      if (!this.expanded.has(id) || !this.boxes[id]) {
        card.remove();
        this._overlayCards.delete(id);
      }
    }
    for (const id of this.expanded) {
      if (!this.boxes[id]) continue;
      let card = this._overlayCards.get(id);
      if (!card) {
        card = htmlEl("div", "detail-card");
        card.addEventListener("click", (ev) => {
          ev.stopPropagation();
          this.toggleExpand(id); // click card body to collapse
        });
        this.overlay.appendChild(card);
        this._overlayCards.set(id, card);
      }
      const body = this.opts.detailHTML ? this.opts.detailHTML(id) : null;
      card.innerHTML = body != null ? body : `<h5>${this._labelOf(id)}</h5><div class="sec">无详情数据</div>`;
      if (body == null || body === "__loading__") {
        card.innerHTML = `<h5>${this._labelOf(id)}</h5><div class="sec">加载中…</div>`;
      }
    }
    this._positionOverlays();
  }

  _labelOf(id) {
    const b = this.boxes[id];
    return b ? b.label : id;
  }

  _positionOverlays() {
    for (const [id, card] of this._overlayCards) {
      const b = this.boxes[id];
      if (!b) continue;
      const x = b.x * this.scale + this.tx;
      const y = (b.y + b.h) * this.scale + this.ty + 2;
      card.style.left = x + "px";
      card.style.top = y + "px";
      card.style.transform = "scale(" + Math.max(this.scale, 0.45) + ")";
      card.style.transformOrigin = "top left";
    }
  }

  // ---- transforms ----
  _applyTransform() {
    this.viewport.setAttribute("transform",
      `translate(${this.tx} ${this.ty}) scale(${this.scale})`);
    this._scheduleRender();
    this._positionOverlays();
    if (this.opts.onTransform) this.opts.onTransform(this);
  }

  fitView() {
    const r = this.svg.getBoundingClientRect();
    const vw = Math.max(r.width, 10), vh = Math.max(r.height, 10);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const id in this.boxes) {
      const b = this.boxes[id];
      x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y);
      x1 = Math.max(x1, b.x + b.w); y1 = Math.max(y1, b.y + b.h);
    }
    if (!isFinite(x0)) { x0 = 0; y0 = 0; x1 = 300; y1 = 200; }
    const w = Math.max(x1 - x0, 300), h = Math.max(y1 - y0, 200);
    const pad = 40;
    let scale = Math.min((vw - pad * 2) / w, (vh - pad * 2) / h, 1.5);
    if (scale < 0.45) {
      // very wide/tall graphs (e.g. deep chains): don't shrink into
      // illegibility — fit the height, anchor top-left, let the user pan.
      scale = 0.45;
      this.scale = scale;
      this.tx = pad - x0 * scale;
      this.ty = pad - y0 * scale;
      this._applyTransform();
      return;
    }
    this.scale = Math.max(scale, 0.02);
    this.tx = (vw - w * this.scale) / 2 - x0 * this.scale;
    this.ty = (vh - h * this.scale) / 2 - y0 * this.scale;
    this._applyTransform();
  }

  focusNode(nodeId, flash = true) {
    const b = this.boxes[nodeId];
    if (!b) return false;
    const r = this.svg.getBoundingClientRect();
    this.scale = Math.max(this.scale, 0.9);
    this.tx = r.width / 2 - (b.x + b.w / 2) * this.scale;
    this.ty = r.height / 2 - (b.y + b.h / 2) * this.scale;
    this._applyTransform();
    this._renderVisible();
    const g = this.rendered.get(nodeId);
    if (g && flash) {
      g.classList.remove("flash");
      void g.getBoundingClientRect();
      g.classList.add("flash");
      setTimeout(() => g.classList.remove("flash"), 2200);
    }
    return true;
  }

  _onWheel(ev) {
    ev.preventDefault();
    const r = this.svg.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
    const ns = Math.min(Math.max(this.scale * factor, 0.02), 3);
    const k = ns / this.scale;
    this.tx = mx - (mx - this.tx) * k;
    this.ty = my - (my - this.ty) * k;
    this.scale = ns;
    this._applyTransform();
  }

  _onPointerDown(ev) {
    if (ev.button !== 0) return;
    const start = { x: ev.clientX, y: ev.clientY, tx: this.tx, ty: this.ty, moved: false };
    this.svg.classList.add("panning");
    const move = (e) => {
      const dx = e.clientX - start.x, dy = e.clientY - start.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) start.moved = true;
      this.tx = start.tx + dx;
      this.ty = start.ty + dy;
      this._applyTransform();
    };
    const up = (e) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      this.svg.classList.remove("panning");
      if (!start.moved && !ev.target.closest(".node-box")) {
        if (this.opts.onEmptyClick) this.opts.onEmptyClick();
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  // ---- culling render ----
  _scheduleRender(force = false) {
    if (force) this._force = true;
    if (this._pending) return;
    this._pending = true;
    requestAnimationFrame(() => {
      this._pending = false;
      const wasForce = this._force; this._force = false;
      this._renderVisible(wasForce);
    });
  }

  _visibleBox() {
    const r = this.svg.getBoundingClientRect();
    const x0 = (0 - this.tx) / this.scale, y0 = (0 - this.ty) / this.scale;
    return { x0, y0, x1: (r.width - this.tx) / this.scale, y1: (r.height - this.ty) / this.scale };
  }

  _renderVisible(force = false) {
    if (!this.boxes) return;
    const vb = this._visibleBox();
    const pad = 60;
    const visible = new Set();
    for (const id in this.boxes) {
      const b = this.boxes[id];
      if (b.x + b.w >= vb.x0 - pad && b.x <= vb.x1 + pad &&
          b.y + b.h >= vb.y0 - pad && b.y <= vb.y1 + pad) visible.add(id);
    }
    for (const [id, g] of this.rendered) {
      if (!visible.has(id)) {
        g.remove();
        this.rendered.delete(id);
      } else if (force) {
        this._styleNode(id, g);
        this._fillNodeText(id, g);
      }
    }
    for (const id of visible) {
      if (!this.rendered.has(id)) {
        const g = this._buildNode(id);
        this.rendered.set(id, g);
        this.nodeLayer.appendChild(g);
      }
    }
    // edges + tensor-name labels
    this.edgeLayer.innerHTML = "";
    this.edgeLabelLayer.innerHTML = "";
    const showNames = this.opts.showTensorNames !== false && this.scale > 0.45;
    for (const eid in this.edgePoints) {
      const pts = this.edgePoints[eid];
      const sx = pts.src[0], sy = pts.src[1], dx = pts.dst[0], dy = pts.dst[1];
      if (Math.max(sx, dx) < vb.x0 - pad || Math.min(sx, dx) > vb.x1 + pad ||
          Math.max(sy, dy) < vb.y0 - pad || Math.min(sy, dy) > vb.y1 + pad) continue;
      const midX = (sx + dx) / 2;
      this.edgeLayer.appendChild(el("path", {
        class: "edge-path",
        d: `M ${sx} ${sy} C ${midX} ${sy}, ${midX} ${dy}, ${dx} ${dy}`,
      }));
      if (showNames && pts.tensor) {
        const lx = (sx + dx) / 2, ly = (sy + dy) / 2 - 3;
        const label = pts.tensor.length > 26 ? pts.tensor.slice(0, 25) + "…" : pts.tensor;
        const w = label.length * 5.6 + 4;
        this.edgeLabelLayer.appendChild(el("rect", {
          class: "edge-label-bg", x: lx - w / 2, y: ly - 8, width: w, height: 12, rx: 2,
        }));
        const t = el("text", {
          class: "edge-label", x: lx, y: ly + 1, "text-anchor": "middle",
        });
        t.textContent = label;
        this.edgeLabelLayer.appendChild(t);
      }
    }
    if (this.searchSet.size) {
      for (const p of this.edgeLayer.children) p.classList.add("edge-dim");
    }
    this._positionOverlays();
  }

  _buildNode(id) {
    const b = this.boxes[id];
    const g = el("g", { class: "node-box", "data-id": id });
    g.setAttribute("transform", `translate(${b.x} ${b.y})`);
    g.appendChild(el("rect", { class: "node-rect", width: b.w, height: b.h, rx: 4 }));
    const opText = el("text", { class: "node-op", x: 9, y: 18 });
    const tensorText = el("text", { class: "node-tensor", x: 9, y: 33 });
    g.appendChild(opText);
    g.appendChild(tensorText);
    this._fillNodeText(id, g);
    this._applyStyleClass(g, this.styles[id] || {});
    g.addEventListener("click", (ev) => {
      ev.stopPropagation();
      this.toggleExpand(id);
      this._select(id);
    });
    return g;
  }

  _fillNodeText(id, g) {
    const b = this.boxes[id];
    const extra = this.nodeExtra[id] || {};
    const opText = g.querySelector(".node-op");
    const tensorText = g.querySelector(".node-tensor");
    if (!opText) return;
    let op = b.kind === "input" || b.kind === "output" ? "Input/Output" : (extra.op || b.label || id);
    if (b.kind === "group") op = b.label;
    op = op.length > 22 ? op.slice(0, 21) + "…" : op;
    opText.textContent = op;
    let tensor = "";
    if (b.kind === "input" || b.kind === "output") {
      tensor = b.label || id.split(":").pop();
    } else {
      const outs = extra.outTensors || [];
      tensor = outs.length ? outs.join(", ") : (b.kind === "group" ? "" : "");
    }
    if (tensor.length > 28) tensor = tensor.slice(0, 27) + "…";
    tensorText.textContent = tensor;
  }

  _applyStyleClass(g, st) {
    g.classList.remove("node-selected", "node-added", "node-removed", "node-modified",
      "node-ambiguous", "node-dim", "io-pill", "group-node");
    const b = this.boxes[g.getAttribute("data-id")];
    if (b && (b.kind === "input" || b.kind === "output")) g.classList.add("io-pill");
    if (b && b.kind === "group") g.classList.add("group-node");
    if (st.cls) g.classList.add(st.cls);
    if (st.dim) g.classList.add("node-dim");
    if (this.selected === g.getAttribute("data-id")) g.classList.add("node-selected");
    const rect = g.querySelector(".node-rect");
    if (rect) {
      if (st.fill) rect.style.fill = st.fill;
      else rect.style.fill = "";
    }
    const dark = st.fill && parseInt(st.fill.slice(1, 3), 16) < 0x80 &&
                 parseInt(st.fill.slice(3, 5), 16) < 0x80;
    for (const t of g.querySelectorAll(".node-op, .node-tensor")) {
      t.style.fill = dark ? "#ffffff" : "";
    }
  }

  _styleNode(id, g) {
    this._applyStyleClass(g, this.styles[id] || {});
  }

  _select(id) {
    if (this.selected) {
      const prev = this.rendered.get(this.selected);
      if (prev) prev.classList.remove("node-selected");
    }
    this.selected = id;
    const g = this.rendered.get(id);
    if (g) g.classList.add("node-selected");
    if (this.opts.onSelect) this.opts.onSelect(id);
  }

  selectExternal(id) {
    if (this.selected) {
      const prev = this.rendered.get(this.selected);
      if (prev) prev.classList.remove("node-selected");
    }
    this.selected = id;
    this._renderVisible();
    const g = this.rendered.get(id);
    if (g) g.classList.add("node-selected");
  }
}
