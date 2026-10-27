// SVG graph renderer: pan / zoom / viewport culling / selection / diff+heat styling.
// Zero dependencies; only renders nodes intersecting the viewport (5k-scale ok).

const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  return e;
}

export class GraphView {
  constructor(container, opts = {}) {
    this.container = container;
    this.opts = opts;
    this.svg = el("svg");
    this.viewport = el("g", { class: "viewport" });
    this.edgeLayer = el("g", { class: "edges" });
    this.nodeLayer = el("g", { class: "nodes" });
    this.viewport.appendChild(this.edgeLayer);
    this.viewport.appendChild(this.nodeLayer);
    this.svg.appendChild(this.viewport);
    container.appendChild(this.svg);

    this.tx = 0; this.ty = 0; this.scale = 1;
    this.boxes = {};          // id -> {x,y,w,h,label,kind}
    this.edgePoints = {};
    this.nodeExtra = {};      // id -> {op}
    this.rendered = new Map(); // id -> <g>
    this.selected = null;
    this.searchSet = new Set();
    this.styles = {};         // id -> {cls, dim, badge, fill}
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
    this.nodeLayer.innerHTML = "";
    this.selected = null;
    this.searchSet.clear();
    this._layoutW = layout.width || 0;
    this._layoutH = layout.height || 0;
    this.fitView();
  }

  clear() {
    this.boxes = {}; this.edgePoints = {}; this.rendered.clear();
    this.edgeLayer.innerHTML = ""; this.nodeLayer.innerHTML = "";
    this.selected = null; this.searchSet.clear(); this.styles = {};
  }

  setStyles(styles) {
    this.styles = styles || {};
    this._scheduleRender(true);
  }

  setSearchHighlight(ids) {
    this.searchSet = new Set(ids || []);
    this._scheduleRender(true);
  }

  // ---- transforms ----
  _applyTransform() {
    this.viewport.setAttribute("transform",
      `translate(${this.tx} ${this.ty}) scale(${this.scale})`);
    this._scheduleRender();
    if (this.opts.onTransform) this.opts.onTransform(this);
  }

  fitView() {
    const r = this.svg.getBoundingClientRect();
    const vw = Math.max(r.width, 10), vh = Math.max(r.height, 10);
    // bbox over all boxes (pseudo io nodes may sit at negative x)
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const id in this.boxes) {
      const b = this.boxes[id];
      x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y);
      x1 = Math.max(x1, b.x + b.w); y1 = Math.max(y1, b.y + b.h);
    }
    if (!isFinite(x0)) { x0 = 0; y0 = 0; x1 = 300; y1 = 200; }
    const w = Math.max(x1 - x0, 300), h = Math.max(y1 - y0, 200);
    const pad = 30;
    this.scale = Math.min((vw - pad * 2) / w, (vh - pad * 2) / h, 1.4);
    this.scale = Math.max(this.scale, 0.02);
    this.tx = (vw - w * this.scale) / 2 - x0 * this.scale;
    this.ty = (vh - h * this.scale) / 2 - y0 * this.scale;
    this._applyTransform();
  }

  focusNode(nodeId, flash = true) {
    const b = this.boxes[nodeId];
    if (!b) return false;
    const r = this.svg.getBoundingClientRect();
    this.scale = Math.max(this.scale, 0.85);
    this.tx = r.width / 2 - (b.x + b.w / 2) * this.scale;
    this.ty = r.height / 2 - (b.y + b.h / 2) * this.scale;
    this._applyTransform();
    this._ensureRendered();
    const g = this.rendered.get(nodeId);
    if (g && flash) {
      g.classList.remove("flash");
      void g.getBoundingClientRect(); // restart animation
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
    // remove out-of-view
    for (const [id, g] of this.rendered) {
      if (!visible.has(id)) {
        g.remove();
        this.rendered.delete(id);
      } else if (force) {
        this._styleNode(id, g);
      }
    }
    // add in-view
    for (const id of visible) {
      if (!this.rendered.has(id)) {
        const g = this._buildNode(id);
        this.rendered.set(id, g);
        this.nodeLayer.appendChild(g);
      }
    }
    // edges: render only edges whose endpoints are visible
    this.edgeLayer.innerHTML = "";
    for (const eid in this.edgePoints) {
      const pts = this.edgePoints[eid];
      const sx = pts.src[0], sy = pts.src[1], dx = pts.dst[0], dy = pts.dst[1];
      if (Math.max(sx, dx) < vb.x0 - pad || Math.min(sx, dx) > vb.x1 + pad ||
          Math.max(sy, dy) < vb.y0 - pad || Math.min(sy, dy) > vb.y1 + pad) continue;
      const midX = (sx + dx) / 2;
      const p = el("path", {
        class: "edge-path",
        d: `M ${sx} ${sy} C ${midX} ${sy}, ${midX} ${dy}, ${dx} ${dy}`,
      });
      this.edgeLayer.appendChild(p);
    }
    // dim edges when a search is active
    if (this.searchSet.size) {
      for (const p of this.edgeLayer.children) p.classList.add("edge-dim");
    }
  }

  _ensureRendered() {
    // after focus, viewport changed; render synchronously for flash target
    this._renderVisible();
  }

  _buildNode(id) {
    const b = this.boxes[id];
    const g = el("g", { class: "node-box", "data-id": id });
    g.setAttribute("transform", `translate(${b.x} ${b.y})`);
    const rect = el("rect", {
      class: "node-rect", width: b.w, height: b.h, rx: 7,
    });
    g.appendChild(rect);
    const label = b.label || id;
    const text = el("text", {
      class: "node-text", x: 10, y: 19,
    });
    text.textContent = label.length > 24 ? label.slice(0, 23) + "…" : label;
    g.appendChild(text);
    const extra = this.nodeExtra[id];
    if (extra && extra.op) {
      const sub = el("text", { class: "node-sub", x: 10, y: 34 });
      sub.textContent = String(extra.op).slice(0, 30);
      g.appendChild(sub);
    } else if (b.kind === "group") {
      const sub = el("text", { class: "node-sub", x: 10, y: 34 });
      sub.textContent = "折叠组";
      g.appendChild(sub);
    }
    // badge (diff/extra marker) top-right: rect + letter so state is never
    // conveyed by color alone
    const st = this.styles[id] || {};
    if (st.badge) {
      const badgeBg = el("rect", {
        class: "node-badge-bg", x: b.w - 26, y: -9, width: 22, height: 16, rx: 3,
        fill: st.badgeColor || "#666",
      });
      const badge = el("text", {
        class: "node-badge", x: b.w - 15, y: 3, "text-anchor": "middle", fill: "#fff",
      });
      badge.textContent = st.badge;
      g.appendChild(badgeBg);
      g.appendChild(badge);
    }
    this._applyStyleClass(g, st);
    g.addEventListener("click", (ev) => {
      ev.stopPropagation();
      this._select(id);
    });
    return g;
  }

  _applyStyleClass(g, st) {
    g.classList.remove("node-selected", "node-added", "node-removed", "node-modified",
      "node-ambiguous", "node-unchanged-dim", "node-dim",
      "node-kind-input", "node-kind-output", "group-node");
    const b = this.boxes[g.getAttribute("data-id")];
    if (b && b.kind === "input") g.classList.add("node-kind-input");
    if (b && b.kind === "output") g.classList.add("node-kind-output");
    if (b && b.kind === "group") g.classList.add("group-node");
    if (st.cls) g.classList.add(st.cls);
    if (st.dim) g.classList.add("node-dim");
    if (this.selected === g.getAttribute("data-id")) g.classList.add("node-selected");
    const rect = g.querySelector(".node-rect");
    if (rect) {
      // inline style beats the CSS `.node-rect { fill: #fff }` rule
      if (st.fill) rect.style.fill = st.fill;
      else rect.style.fill = "";
    }
    // keep text readable on dark heat fills
    const dark = st.fill && parseInt(st.fill.slice(1, 3), 16) < 0x80 &&
                 parseInt(st.fill.slice(3, 5), 16) < 0x80;
    const tcol = dark ? "#ffffff" : "";
    for (const t of g.querySelectorAll(".node-text, .node-sub")) {
      t.style.fill = tcol;
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
    this._ensureRendered();
    const g = this.rendered.get(id);
    if (g) g.classList.add("node-selected");
  }
}
