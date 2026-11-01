// ModelToolbox frontend — canvas-first, no toolbars.
// Interaction model:
//   - welcome card on empty canvas (open / compare / pytorch, drag & drop)
//   - ⌘/Ctrl+K command palette: every action + fuzzy node search & jump
//   - right-click context menus (canvas / node)
//   - floating layer panel (bottom-right): layer toggles + legend
//   - in-place expandable node details (click), properties drawer, status bar
import { apiPost, apiGet, apiDownload } from "./api.js";
import { GraphView } from "./graph.js";

const $ = (id) => document.getElementById(id);

const state = {
  mode: "browse", // browse | compare | pytorch
  browse: { modelId: null, ir: null, meta: null, layout: null, profileId: null, mapped: null },
  compare: {
    diffId: null, diff: null, match: null, folding: null, layouts: null,
    aModelId: null, bModelId: null, nodesA: {}, nodesB: {}, folded: new Set(),
    navIdx: -1, mappedA: null, mappedB: null, nameA: "", nameB: "",
  },
  pytorch: { result: null, view: "overview" },
  useNetron: true, // browse canvas rendered by the vendored official Netron viewer
  detailCache: { a: {}, b: {} },
  pendingCompare: { a: null, b: null },
  layerPanelClosed: false,
  useNetronCompare: false, // dual official Netron viewers for compare mode
};

// ---------- tiny helpers ----------
function toast(msg, ms = 2600) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), ms);
}
function setStatus(el, msg, cls = "") {
  el.textContent = msg || "";
  el.className = "status " + cls;
}
function setStatusText(msg, cls = "") {
  $("sb-left").textContent = msg;
  $("sb-left").style.color = cls === "err" ? "var(--removed)" : cls === "ok" ? "var(--added)" : "";
}
function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function kvTable(pairs) {
  return `<table>${pairs.filter((p) => p[1] !== undefined && p[1] !== null && p[1] !== "")
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${typeof v === "object" ? "<code>" + esc(JSON.stringify(v)) + "</code>" : esc(v)}</td></tr>`).join("")}</table>`;
}
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",", 2)[1]);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

const DIFF_BADGE = {
  added: { letter: "A", cls: "node-added" },
  removed: { letter: "D", cls: "node-removed" },
  op_changed: { letter: "M", cls: "node-modified" },
  attr_changed: { letter: "M", cls: "node-modified" },
  shape_changed: { letter: "S", cls: "node-modified" },
  dtype_changed: { letter: "T", cls: "node-modified" },
  param_changed: { letter: "P", cls: "node-modified" },
  connection_changed: { letter: "C", cls: "node-modified" },
  ambiguous: { letter: "?", cls: "node-ambiguous" },
};
const HEAT_COLORS = ["#31567e", "#2f7fb5", "#2f9e6e", "#c9a227", "#c94f43"];
function heatColor(v, min, max) {
  if (v == null || max == null || max <= min) return HEAT_COLORS[2];
  const t = Math.min(Math.max((v - min) / (max - min || 1), 0), 1);
  return HEAT_COLORS[Math.min(Math.floor(t * 5), 4)];
}

// ---------- graph views ----------
const browseGraph = new GraphView($("browse-graph"), {
  onSelect: (id) => { showProps("browse", id); },
  onEmptyClick: () => { browseGraph.collapseAll(); showProps("browse", null); },
  detailHTML: (id) => browseDetailHTML(id),
});
function netronFrameReadyId(frameId, cb) {
  const f = $(frameId);
  const ready = () => {
    try {
      const w = f.contentWindow;
      if (w && w.__view__ && w.__view__._target &&
          w.__view__._target._table && w.__view__._target._table.size > 0) {
        cb(w);
        return true;
      }
    } catch (e) { /* not ready */ }
    return false;
  };
  if (ready()) return;
  let tries = 0;
  const t = setInterval(() => { tries += 1; if (ready() || tries > 48) clearInterval(t); }, 250);
}

function netronHideChrome(frameId) {
  const f = $(frameId);
  netronFrameReadyId(frameId, (w) => {
    const doc = f.contentDocument;
    if (doc.getElementById("mtbx-style")) return;
    const st = doc.createElement("style");
    st.id = "mtbx-style";
    st.textContent = ".toolbar, #menu-button, #titlebar { display: none !important; }";
    doc.head.appendChild(st);
  });
}

const DIFF_TINT = {
  added: "#1f6f43", removed: "#8f2f2f", op_changed: "#8f4a2f",
  attr_changed: "#7d5a2f", shape_changed: "#7d2f5a", dtype_changed: "#5a2f7d",
  param_changed: "#2f5a7d", connection_changed: "#4a4a2f", ambiguous: "#8a6d1f",
};

let netronFollowLock = false;

function netronFocusId(frameId, nodeName, silent) {
  netronFrameReadyId(frameId, (w) => {
    const t = w.__view__._target;
    for (const [modelNode, gnode] of t._table.entries()) {
      if (modelNode && modelNode.name === nodeName && gnode) {
        t.scrollTo(t.select([modelNode], null));
        return;
      }
    }
    if (!silent) toast("该节点在另一侧无对应（或为歧义）");
  });
}

function netronWireFollow(frameId, side) {
  const f = $(frameId);
  netronFrameReadyId(frameId, () => {
    if (f.__followWired) return;
    f.__followWired = true;
    const doc = f.contentDocument;
    doc.addEventListener("click", (ev) => {
      if (!$("sync-view") || !$("sync-view").checked) return;
      if (state.mode !== "compare" || !state.compare.useNetronCompare) return;
      if (netronFollowLock) return;
      const g = ev.target && ev.target.closest && ev.target.closest("g.node");
      if (!g) return;
      const name = g.getAttribute("id");
      if (!name) return;
      const partner = findPartner(side, name);
      if (!partner) return;
      netronFollowLock = true;
      const other = side === "a" ? "netron-frame-b" : "netron-frame-a";
      netronFocusId(other, partner, true);
      setTimeout(() => { netronFollowLock = false; }, 400);
    }, true);
  });
}

function netronDiffColorMaps() {
  const mapA = {}, mapB = {};
  const dimA = new Set(), dimB = new Set();
  if (!state.compare.diff) return { mapA, mapB, dimA, dimB };
  for (const nd of state.compare.diff.node_diffs) {
    if (nd.kind === "unchanged") {
      if (getLayer("diff-only")) {
        if (nd.a_node_id) dimA.add(nd.a_node_id);
        if (nd.b_node_id) dimB.add(nd.b_node_id);
      }
      continue;
    }
    if (nd.a_node_id) mapA[nd.a_node_id] = nd.kind;
    if (nd.b_node_id) mapB[nd.b_node_id] = nd.kind;
  }
  return { mapA, mapB, dimA, dimB };
}

function netronTintDiff(frameId, side, colorMap, dimIds) {
  netronFrameReadyId(frameId, (w) => {
    const doc = $(frameId).contentDocument;
    doc.querySelectorAll("path.mtbx-heat").forEach((p) => {
      p.classList.remove("mtbx-heat"); p.style.fill = "";
    });
    doc.querySelectorAll("g.mtbx-dim").forEach((g) => g.classList.remove("mtbx-dim"));
    const target = w.__view__._target;
    for (const [modelNode, gnode] of target._table.entries()) {
      const name = modelNode && modelNode.name;
      if (!name) continue;
      const kind = colorMap[name];
      if (gnode && gnode.element) {
        const paths = gnode.element.querySelectorAll("g.node-item path");
        if (kind && DIFF_TINT[kind]) {
          for (const p of paths) { p.style.fill = DIFF_TINT[kind]; p.classList.add("mtbx-heat"); }
        }
        if (dimIds && dimIds.has(name)) gnode.element.classList.add("mtbx-dim");
      }
    }
  });
}


function netronApplyDiffTint() {
  if (!state.compare.useNetronCompare || !state.compare.diff) return;
  const { mapA, mapB, dimA, dimB } = netronDiffColorMaps();
  netronTintDiff("netron-frame-a", "a", mapA, dimA);
  netronTintDiff("netron-frame-b", "b", mapB, dimB);
}

function applyCompareNetronViews() {
  const fa = $("netron-frame-a"), fb = $("netron-frame-b");
  const on = state.compare.useNetronCompare;
  const urlA = "/netron/?model=" + state.compare.aModelId;
  const urlB = "/netron/?model=" + state.compare.bModelId;
  if (on) {
    if (!fa.src.endsWith(urlA)) fa.src = urlA;
    if (!fb.src.endsWith(urlB)) fb.src = urlB;
    fa.classList.remove("hidden"); fb.classList.remove("hidden");
    $("cmp-graph-a").classList.add("hidden");
    $("cmp-graph-b").classList.add("hidden");
    netronHideChrome("netron-frame-a");
    netronHideChrome("netron-frame-b");
    netronWireFollow("netron-frame-a", "a");
    netronWireFollow("netron-frame-b", "b");
    netronApplyDiffTint();
  } else {
    fa.src = "about:blank"; fb.src = "about:blank";
    fa.classList.add("hidden"); fb.classList.add("hidden");
    $("cmp-graph-a").classList.remove("hidden");
    $("cmp-graph-b").classList.remove("hidden");
  }
}

const graphA = new GraphView($("cmp-graph-a"), {
  onSelect: (id) => { showProps("a", id); },
  onEmptyClick: () => { graphA.collapseAll(); showProps("a", null); },
  detailHTML: (id) => compareDetailHTML("a", id),
});
const graphB = new GraphView($("cmp-graph-b"), {
  onSelect: (id) => { showProps("b", id); },
  onEmptyClick: () => { graphB.collapseAll(); showProps("b", null); },
  detailHTML: (id) => compareDetailHTML("b", id),
});
const ptGraph = new GraphView($("pt-graph"), {
  onSelect: (id) => { showPtDetail(id); },
  onEmptyClick: () => { ptGraph.collapseAll(); showPtDetail(null); },
  detailHTML: (id) => ptDetailHTML(id),
});
const activeGraph = () =>
  state.mode === "compare" ? graphB : state.mode === "pytorch" ? ptGraph : browseGraph;

let syncing = false;
graphA.opts.onTransform = (g) => mirrorTransform(graphA, graphB);
graphB.opts.onTransform = (g) => mirrorTransform(graphB, graphA);
function mirrorTransform(src, dst) {
  if (!$("sync-view") || !$("sync-view").checked || syncing) return;
  syncing = true;
  dst.tx = src.tx; dst.ty = src.ty; dst.scale = src.scale;
  dst.viewport.setAttribute("transform", `translate(${dst.tx} ${dst.ty}) scale(${dst.scale})`);
  dst._scheduleRender();
  dst._positionOverlays();
  syncing = false;
}

// ---------- mode & welcome ----------
function hasContent() {
  return state.mode === "compare" ? !!state.compare.diffId
    : state.mode === "pytorch" ? !!(state.pytorch.result && state.pytorch.result.ok)
    : !!state.browse.modelId;
}
function applyBrowseRenderer() {
  const frame = $("netron-frame");
  const graph = $("browse-graph");
  if (!frame || !graph) return;
  if (state.useNetron && state.browse.modelId) {
    const url = "/netron/?model=" + state.browse.modelId;
    if (!frame.src.endsWith(url)) frame.src = url;
    frame.classList.remove("hidden");
    graph.classList.add("hidden");
  } else {
    frame.src = "about:blank";
    frame.classList.add("hidden");
    graph.classList.remove("hidden");
  }
}

function refreshWelcome() {
  const show = !hasContent() && state.mode !== "pytorch";
  $("welcome").classList.toggle("hidden", !show);
  $("statusbar").style.visibility = show ? "hidden" : "visible";
  $("layer-panel").classList.toggle("hidden", show || state.layerPanelClosed);
}
function setMode(mode) {
  state.mode = mode;
  $("view-graph").classList.toggle("hidden", mode === "pytorch");
  $("view-pytorch").classList.toggle("hidden", mode !== "pytorch");
  $("single-wrap").classList.toggle("hidden", mode !== "browse");
  $("dual-wrap").classList.toggle("hidden", mode !== "compare");
  $("compare-header").classList.toggle("hidden", mode !== "compare");
  $("diff-nav").classList.toggle("hidden", mode !== "compare");
  if (mode === "compare") setLayer("layer-diff", true);
  if (mode === "browse") applyBrowseRenderer();
  $("sb-right").textContent =
    mode === "compare" ? `对比 · A ${state.compare.nameA} ⇄ B ${state.compare.nameB}`
    : mode === "pytorch" ? "PyTorch 层级"
    : state.browse.meta ? `${state.browse.meta.name} · ${state.browse.meta.node_count} 节点`
    : "";
  refreshWelcome();
  refreshLayerPanel();
  refreshLayers();
}

// ---------- detail cards ----------
// ---- Netron viewer bridge (iframe is same-origin; official viewer API) ----
function netronFrameReady(cb) {
  const f = $("netron-frame");
  const ready = () => {
    try {
      const w = f.contentWindow;
      if (w && w.__view__ && w.__view__._target &&
          w.__view__._target._table && w.__view__._target._table.size > 0) {
        cb(w);
        return true;
      }
    } catch (e) { /* iframe not ready */ }
    return false;
  };
  if (ready()) return;
  let tries = 0;
  const t = setInterval(() => {
    tries += 1;
    if (ready() || tries > 48) clearInterval(t);
  }, 250);
}

function netronApplyHeat(heatMap) {
  netronFrameReady((w) => {
    const doc = $("netron-frame").contentDocument;
    doc.querySelectorAll("path.mtbx-heat").forEach((p) => {
      p.classList.remove("mtbx-heat");
      p.style.fill = "";
    });
    let hit = 0;
    const target = w.__view__._target;
    // the table is keyed by the ONNX node OBJECT; match by its .name and
    // tint every row of the node (Netron rows are g.node-item > path)
    for (const [modelNode, gnode] of target._table.entries()) {
      const name = modelNode && modelNode.name;
      if (!name) continue;
      const color = heatMap[name];
      if (color && gnode && gnode.element) {
        const paths = gnode.element.querySelectorAll("g.node-item path");
        if (paths.length) {
          for (const p of paths) { p.style.fill = color; p.classList.add("mtbx-heat"); }
          hit += 1;
        }
      }
    }
    if (Object.keys(heatMap).length) {
      setStatusText(hit ? `热力图已应用：${hit}/${Object.keys(heatMap).length} 节点着色（Netron 渲染）` : "热力图：Netron 中未匹配到节点", hit ? "ok" : "err");
    }
  });
}

function netronFocus(nodeName) {
  netronFrameReady((w) => {
    const t = w.__view__._target;
    for (const [modelNode, gnode] of t._table.entries()) {
      if (modelNode && modelNode.name === nodeName && gnode) {
        t.scrollTo(t.select([modelNode], 'sidebar'));
        return;
      }
    }
    toast("该节点在 Netron 中无名称锚点（未命名节点）");
  });
}

function fmtShape(vi) {
  if (!vi) return "";
  const shp = vi.shape === null ? "?" : JSON.stringify(vi.shape).replace(/null/g, "?");
  return `${vi.dtype || "?"} ${shp}`;
}

function browseDetailHTML(nodeId) {
  const n = state.browse.ir && state.browse.ir.nodes.find((x) => x.id === nodeId);
  if (!n) return "__loading__";
  return nodeCardHTML(n, null,
    state.browse.mapped && state.browse.mapped.per_node[nodeId]);
}

function nodeCardHTML(n, diff, perf) {
  const rows = [];
  for (const t of (n.inputs || []).filter(Boolean)) {
    const vi = n.input_info && n.input_info[t];
    rows.push(`<tr><td>← ${esc(t)}</td><td>${esc(fmtShape(vi))}</td></tr>`);
  }
  for (const t of (n.outputs || []).filter(Boolean)) {
    const vi = n.output_info && n.output_info[t];
    rows.push(`<tr><td>→ ${esc(t)}</td><td>${esc(fmtShape(vi))}</td></tr>`);
  }
  let attrRows = "";
  for (const [k, v] of Object.entries(n.attributes || {})) {
    attrRows += `<tr><td>${esc(k)}</td><td>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td></tr>`;
  }
  let paramRows = "";
  for (const p of n.params || []) {
    paramRows += `<tr><td>${esc(p.name)}</td><td>${esc(p.dtype || "?")} ${esc(JSON.stringify(p.shape))}</td></tr>`;
  }
  let chRows = "";
  for (const c of (diff && diff.changes) || []) {
    chRows += `<tr><td>${esc(c.field)}</td><td>${esc(JSON.stringify(c.before))} → ${esc(JSON.stringify(c.after))}</td></tr>`;
  }
  const diffLine = diff && diff.kind !== "unchanged" && diff.kind !== "ambiguous"
    ? `<div class="warnline">${esc(diff.explanation)}（置信度 ${diff.confidence}）</div>`
    : (diff && diff.kind === "ambiguous"
      ? `<div class="warnline">⚠ 存在多个等价候选，对应关系不确定</div>` : "");
  const perfLine = perf
    ? `<div class="perf">⏱ ${perf.duration_us.toFixed(1)} us · ${(perf.share * 100).toFixed(1)}% · ×${perf.calls}</div>` : "";
  const sg = (n.subgraphs || []).length
    ? `<div class="warnline">含嵌套子图 ×${n.subgraphs.length}</div>` : "";
  return `<h5><span>${esc(n.display_name)}</span><span class="op-tag">${esc(n.op_type)}</span></h5>
    <div class="sec">tensors</div><table>${rows.join("")}</table>
    ${attrRows ? `<div class="sec">attributes</div><table>${attrRows}</table>` : ""}
    ${paramRows ? `<div class="sec">weights</div><table>${paramRows}</table>` : ""}
    ${chRows ? `<div class="sec">字段级变化</div><table>${chRows}</table>` : ""}
    ${diffLine}${sg}${perfLine}`;
}

function compareDetailHTML(side, nodeId) {
  const cache = state.detailCache[side][nodeId];
  if (!cache) {
    fetchCompareDetail(side, nodeId);
    return "__loading__";
  }
  return nodeCardHTML(cache, diffByNode(side)[nodeId], null);
}

async function fetchCompareDetail(side, nodeId) {
  const modelId = side === "a" ? state.compare.aModelId : state.compare.bModelId;
  if (!modelId || state.detailCache[side][nodeId]) return;
  try {
    const res = await apiPost("/api/model/nodes", { model_id: modelId, ids: [nodeId] });
    if (res.nodes && res.nodes.length) state.detailCache[side][nodeId] = res.nodes[0];
    else state.detailCache[side][nodeId] = { id: nodeId, display_name: nodeId, inputs: [], outputs: [] };
  } catch (e) {
    state.detailCache[side][nodeId] = { id: nodeId, display_name: nodeId, inputs: [], outputs: [] };
  }
  const graph = side === "a" ? graphA : graphB;
  if (graph.expanded.has(nodeId)) graph._syncOverlays();
}

// ---------- properties drawer ----------
function openProps() { $("props").classList.remove("hidden"); }
function showProps(side, nodeId) {
  const drawer = $("props");
  if (!nodeId) { drawer.classList.add("hidden"); return; }
  drawer.classList.remove("hidden");
  $("props-title").textContent = `属性 — ${side.toUpperCase()}`;
  const body = $("props-body");
  if (side === "browse") {
    const n = state.browse.ir && state.browse.ir.nodes.find((x) => x.id === nodeId);
    body.innerHTML = n ? propsForNode(n, state.browse.mapped && state.browse.mapped.per_node[nodeId]) : "";
  } else {
    const cache = state.detailCache[side][nodeId];
    if (!cache) {
      body.innerHTML = "<p>加载中…</p>";
      fetchCompareDetail(side, nodeId).then(() => showProps(side, nodeId));
      return;
    }
    body.innerHTML = propsForNode(cache, null, diffByNode(side)[nodeId]);
  }
}
function propsForNode(n, perf, diff) {
  const perfHtml = perf
    ? kvTable([["耗时 (us)", perf.duration_us.toFixed(1)], ["占比", (perf.share * 100).toFixed(2) + "%"], ["调用", perf.calls]])
    : "";
  const diffHtml = diff ? kvTable([
    ["差异类型", diff.kind], ["置信度", diff.confidence], ["说明", diff.explanation],
  ]) + ((diff.changes || []).length
    ? "<h4>字段级变化</h4>" + kvTable(diff.changes.map((c) => [c.field, `${JSON.stringify(c.before)} → ${JSON.stringify(c.after)}`]))
    : "") : "";
  return `<h4>${esc(n.display_name)}</h4>
    ${kvTable([["id", n.id], ["op", n.op_type], ["domain", n.domain || "ai.onnx"]])}
    ${diffHtml}
    <h4>输入</h4>${kvTable((n.inputs || []).filter(Boolean).map((t) => {
      const vi = n.input_info && n.input_info[t];
      return [t, vi ? fmtShape(vi) : "无 shape 信息"];
    }))}
    <h4>输出</h4>${kvTable((n.outputs || []).filter(Boolean).map((t) => {
      const vi = n.output_info && n.output_info[t];
      return [t, vi ? fmtShape(vi) : "无 shape 信息"];
    }))}
    ${Object.keys(n.attributes || {}).length ? "<h4>属性</h4>" + kvTable(Object.entries(n.attributes)) : ""}
    ${(n.params || []).length ? "<h4>权重</h4>" + kvTable(n.params.map((p) => [p.name, `${p.dtype || "?"} ${JSON.stringify(p.shape)}`])) : ""}
    ${perfHtml ? "<h4>性能</h4>" + perfHtml : ""}`;
}
$("props-close").addEventListener("click", () => $("props").classList.add("hidden"));

// ---------- model loading (file picker / drag&drop / palette) ----------
async function openModelFile(file) {
  if (!file) return;
  setStatusText(`加载 ${file.name}…`);
  try {
    const b64 = await fileToBase64(file);
    const t0 = performance.now();
    const res = await apiPost("/api/model/upload", { name: file.name, data_base64: b64 });
    const dt = Math.round(performance.now() - t0);
    state.browse = { modelId: res.model_id, ir: res.ir, meta: res.meta, layout: res.layout, profileId: null, mapped: null };
    const extra = {};
    for (const n of res.ir.nodes) extra[n.id] = { op: n.op_type, outTensors: n.outputs.filter(Boolean) };
    browseGraph.setData(res.layout, extra);
    setMode("browse");
    applyBrowseRenderer();
    setStatusText(`${res.meta.name} — ${res.meta.node_count} 节点 / ${res.meta.edge_count} 边 (${dt}ms)`);
    if ((res.meta.warnings || []).length) {
      toast(`⚠ ${res.meta.warnings.length} 条解析警告（详见模型属性）`, 3400);
      showModelProps();
    }
  } catch (e) {
    setStatusText(`加载失败: ${e.message}`, "err");
    toast(`加载失败: ${e.message}`);
  }
}

function showModelProps() {
  const meta = state.browse.meta;
  if (!meta) return;
  openProps();
  $("props-title").textContent = "模型属性";
  $("props-body").innerHTML = kvTable([
    ["名称", meta.name], ["producer", meta.producer || "—"],
    ["opset", JSON.stringify(meta.opsets)],
    ["节点/边", `${meta.node_count} / ${meta.edge_count}`],
    ["参数量", meta.param_count],
    ["文件大小", `${(meta.file_size_bytes / 1024).toFixed(1)} KB`],
    ["输入", meta.inputs.map((v) => `${v.name}${v.shape ? JSON.stringify(v.shape) : " (未知)"}`).join(", ")],
    ["输出", meta.outputs.map((v) => `${v.name}${v.shape ? JSON.stringify(v.shape) : " (未知)"}`).join(", ")],
    ["指纹", meta.fingerprints.content ? meta.fingerprints.content.slice(0, 20) + "…" : "—"],
  ]) + ((meta.warnings || []).length
    ? "<h4>警告</h4>" + meta.warnings.map((w) => `<p style="margin:3px 0;color:var(--ambiguous)">${esc(w)}</p>`).join("")
    : "");
}

async function openProfileFile(file, target) {
  try {
    const b64 = await fileToBase64(file);
    let res;
    if (target === "browse") {
      if (!state.browse.modelId) { toast("请先打开模型"); return; }
      res = await apiPost("/api/profile/upload", { name: file.name, data_base64: b64, model_id: state.browse.modelId });
      state.browse.profileId = res.profile_id;
      state.browse.mapped = res.mapped;
    } else {
      const modelId = target === "a" ? state.compare.aModelId : state.compare.bModelId;
      res = await apiPost("/api/profile/upload", { name: file.name, data_base64: b64, model_id: modelId });
      if (target === "a") state.compare.mappedA = res.mapped;
      else state.compare.mappedB = res.mapped;
    }
    setLayer("layer-heat", true);
    refreshLayers();
    const note = `${file.name}: ${Object.keys(res.mapped.per_node).length} 映射 / ${res.mapped.unmapped.length} 未映射` +
      (res.mapped.model_fingerprint_matched === false ? " · ⚠ 指纹不匹配" : "");
    toast(note, 3400);
    setStatusText(note, res.mapped.model_fingerprint_matched === false ? "err" : "ok");
  } catch (e) {
    toast(`profile 导入失败: ${e.message}`);
  }
}

// ---------- compare ----------
$("w-open").addEventListener("click", () => $("file-open").click());
$("w-compare").addEventListener("click", () => openCompareDialog());
$("w-pytorch").addEventListener("click", () => setMode("pytorch"));

function openCompareDialog() {
  $("dlg-compare").classList.remove("hidden");
  state.pendingCompare = { a: null, b: null };
  $("pick-a-name").textContent = "未选择";
  $("pick-b-name").textContent = "未选择";
  setStatus($("cmp-status"), "");
}
let pickTarget = null;
$("pick-a").addEventListener("click", () => { pickTarget = "a"; $("file-open-b").click(); });
$("pick-b").addEventListener("click", () => { pickTarget = "b"; $("file-open-b").click(); });
$("file-open-b").addEventListener("change", (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f || !pickTarget) return;
  state.pendingCompare[pickTarget] = f;
  $(`pick-${pickTarget}-name`).textContent = f.name;
});
$("cmp-cancel").addEventListener("click", () => $("dlg-compare").classList.add("hidden"));
$("cmp-run").addEventListener("click", runCompare);

async function runCompare() {
  const fa = state.pendingCompare.a;
  const fb = state.pendingCompare.b;
  if (!fa || !fb) { setStatus($("cmp-status"), "请先选择两个模型文件", "err"); return; }
  setStatus($("cmp-status"), "解析与匹配中…");
  try {
    const t0 = performance.now();
    const [aB64, bB64] = [await fileToBase64(fa), await fileToBase64(fb)];
    const res = await apiPost("/api/diff", {
      name_a: fa.name, data_a: aB64, name_b: fb.name, data_b: bB64,
      min_group_size: parseInt($("cmp-min-size").value || "3", 10),
      min_confidence: parseFloat($("cmp-min-conf").value || "0.9"),
    });
    const dt = Math.round(performance.now() - t0);
    $("dlg-compare").classList.add("hidden");
    applyDiffResult(res, dt, fa.name, fb.name);
  } catch (e) {
    setStatus($("cmp-status"), `对比失败: ${e.message}`, "err");
  }
}

function applyDiffResult(res, dt, nameA, nameB) {
  Object.assign(state.compare, {
    diffId: res.diff_id, diff: res.diff, match: res.match,
    folding: res.folding, layouts: res.layouts, folded: new Set(), navIdx: -1,
    aModelId: res.a_model_id, bModelId: res.b_model_id,
    nodesA: res.nodes_a || {}, nodesB: res.nodes_b || {},
    mappedA: null, mappedB: null, nameA, nameB,
  });
  state.detailCache = { a: {}, b: {} };
  $("cmp-head-a").innerHTML = `A: <span>${esc(nameA)}</span>`;
  $("cmp-head-b").innerHTML = `B: <span>${esc(nameB)}</span>`;
  setMode("compare");
  const exA = {}, exB = {};
  for (const [id, n] of Object.entries(state.compare.nodesA)) exA[id] = { op: n.display || n.op, outTensors: n.outputs };
  for (const [id, n] of Object.entries(state.compare.nodesB)) exB[id] = { op: n.display || n.op, outTensors: n.outputs };
  applyCompareNetronViews();
  if (!state.compare.useNetronCompare) {
    graphA.setData(res.layouts.a, exA);
    graphB.setData(res.layouts.b, exB);
  }
  renderDiffSummary();
  renderFoldGroups();
  refreshLayers();
  const s = res.match.summary;
  setStatusText(`匹配 ${s.matched_count} · 歧义 ${s.ambiguous_count} · A独有 ${s.unmatched_a_count} · B独有 ${s.unmatched_b_count} (${dt}ms)`);
}

// ---------- drag & drop ----------
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  e.preventDefault();
  if (e.dataTransfer && [...(e.dataTransfer.types || [])].includes("Files")) {
    dragDepth++;
    $("drop-overlay").classList.remove("hidden");
  }
});
window.addEventListener("dragleave", (e) => {
  e.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("drop-overlay").classList.add("hidden");
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", async (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("drop-overlay").classList.add("hidden");
  const files = [...(e.dataTransfer.files || [])];
  const onnx = files.filter((f) => /\.(onnx|bin)$/i.test(f.name));
  const prof = files.filter((f) => /\.(json|csv|txt)$/i.test(f.name));
  if (onnx.length >= 2) {
    await runCompareWithFiles(onnx[0], onnx[1]);
    return;
  }
  if (onnx.length === 1) { await openModelFile(onnx[0]); return; }
  if (prof.length === 1) {
    const target = state.mode === "compare" ? "a" : "browse";
    await openProfileFile(prof[0], target);
    return;
  }
  if (prof.length >= 2 && state.mode === "compare") {
    await openProfileFile(prof[0], "a");
    await openProfileFile(prof[1], "b");
    return;
  }
  if (files.length) toast("无法识别的文件类型");
});

async function runCompareWithFiles(fa, fb) {
  $("dlg-compare").classList.remove("hidden");
  state.pendingCompare = { a: fa, b: fb };
  $("pick-a-name").textContent = fa.name;
  $("pick-b-name").textContent = fb.name;
  setStatus($("cmp-status"), "解析与匹配中…");
  try {
    const t0 = performance.now();
    const [aB64, bB64] = [await fileToBase64(fa), await fileToBase64(fb)];
    const res = await apiPost("/api/diff", {
      name_a: fa.name, data_a: aB64, name_b: fb.name, data_b: bB64,
      min_group_size: parseInt($("cmp-min-size").value || "3", 10),
      min_confidence: parseFloat($("cmp-min-conf").value || "0.9"),
    });
    const dt = Math.round(performance.now() - t0);
    $("dlg-compare").classList.add("hidden");
    applyDiffResult(res, dt, fa.name, fb.name);
  } catch (e) {
    setStatus($("cmp-status"), `对比失败: ${e.message}`, "err");
  }
}

// ---------- diff UI ----------
function diffByNode(side) {
  const m = {};
  if (!state.compare.diff) return m;
  for (const nd of state.compare.diff.node_diffs) {
    const id = side === "a" ? nd.a_node_id : nd.b_node_id;
    if (id) m[id] = nd;
  }
  return m;
}

function renderDiffSummary() {
  const d = state.compare.diff;
  const s = d.summary;
  const rows = [["相同", s.unchanged], ["新增(B)", s.added], ["删除(A)", s.removed],
    ["算子", s.op_changed], ["属性", s.attr_changed], ["shape", s.shape_changed],
    ["dtype", s.dtype_changed], ["参数", s.param_changed], ["连接", s.connection_changed],
    ["歧义", s.ambiguous]].filter((r) => r[1]);
  const interesting = d.node_diffs.filter((nd) => nd.kind !== "unchanged");
  state.compare.navList = interesting;
  openProps();
  $("props-title").textContent = "差异摘要";
  $("props-body").innerHTML =
    kvTable(rows) + `<p>${d.topology_identical ? "拓扑指纹一致" : "拓扑指纹不一致"}</p>` +
    (interesting.length
      ? "<h4>差异列表</h4><table>" + interesting.map((nd, i) =>
        `<tr style="cursor:pointer" data-i="${i}"><td>${DIFF_BADGE[nd.kind] ? DIFF_BADGE[nd.kind].letter : "M"} ${esc(nd.kind)}</td><td>${esc(nd.a_display || nd.b_display)}</td></tr>`).join("") + "</table>"
      : "<p>无结构差异 🎉</p>");
  $("props-body").querySelectorAll("tr[data-i]").forEach((el) =>
    el.addEventListener("click", () => gotoDiff(parseInt(el.dataset.i, 10))));
  $("diff-nav-pos").textContent = `0/${interesting.length}`;
}

function gotoDiff(i) {
  const list = state.compare.navList || [];
  if (!list.length) return;
  state.compare.navIdx = ((i % list.length) + list.length) % list.length;
  const nd = list[state.compare.navIdx];
  $("diff-nav-pos").textContent = `${state.compare.navIdx + 1}/${list.length}`;
  if (state.compare.useNetronCompare) {
    if (nd.a_node_id) netronFocusId("netron-frame-a", nd.a_node_id, true);
    if (nd.b_node_id) netronFocusId("netron-frame-b", nd.b_node_id, true);
    return;
  }
  if (nd.a_node_id) { graphA.focusNode(nd.a_node_id); graphA.selectExternal(nd.a_node_id); }
  if (nd.b_node_id) { graphB.focusNode(nd.b_node_id); graphB.selectExternal(nd.b_node_id); }
}
$("diff-prev").addEventListener("click", () => gotoDiff((state.compare.navIdx === -1 ? 0 : state.compare.navIdx) - 1));
$("diff-next").addEventListener("click", () => gotoDiff(state.compare.navIdx + 1));

function renderFoldGroups() {
  const groups = state.compare.folding.groups;
  if (state.compare.useNetronCompare) {
    // Netron has no fold concept — group membership is conveyed by tint in
    // a later pass; hide the fold checkboxes here.
    return;
  }
  if (!groups.length) return;
  openProps();
  const body = $("props-body");
  const wrap = document.createElement("div");
  wrap.innerHTML = "<h4>公共子图折叠</h4>" + groups.map((g) => `
    <label style="display:flex;gap:6px;align-items:flex-start;font-size:11.5px;margin:4px 0">
      <input type="checkbox" data-gid="${esc(g.group_id)}">
      <span><b>${esc(g.label)}</b><br><span style="color:var(--muted)">${esc(g.summary)}</span></span>
    </label>`).join("");
  body.appendChild(wrap);
  wrap.querySelectorAll("input[type=checkbox]").forEach((cb) =>
    cb.addEventListener("change", async () => {
      const gid = cb.dataset.gid;
      if (cb.checked) state.compare.folded.add(gid);
      else state.compare.folded.delete(gid);
      await applyCompareViews();
    }));
}

function buildViewForSide(side) {
  const layout = state.compare.layouts[side];
  const folded = state.compare.folded;
  if (!folded.size) {
    return Promise.resolve({ boxes: layout.boxes, edge_points: layout.edge_points,
      width: layout.width, height: layout.height });
  }
  return apiPost("/api/fold", { diff_id: state.compare.diffId, folded_group_ids: [...folded] })
    .then((res) => {
      const view = res.views[side];
      const boxes = {};
      const groupBox = {};
      for (const nid in layout.boxes) {
        const b = layout.boxes[nid];
        if (b.kind === "input" || b.kind === "output") boxes[nid] = b;
      }
      for (const n of view.display_nodes) {
        if (n.kind === "node") { if (layout.boxes[n.id]) boxes[n.id] = layout.boxes[n.id]; }
        else {
          const members = n.orig_node_ids.filter((m) => layout.boxes[m]);
          if (!members.length) continue;
          let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
          for (const m of members) {
            const b = layout.boxes[m];
            x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y);
            x1 = Math.max(x1, b.x + b.w); y1 = Math.max(y1, b.y + b.h);
          }
          groupBox[n.id] = { node_id: n.id, x: x0 - 12, y: y0 - 12,
            w: x1 - x0 + 24, h: y1 - y0 + 24, kind: "group", label: n.label };
        }
      }
      Object.assign(boxes, groupBox);
      const edges = {};
      for (const e of view.display_edges) {
        const sb = boxes[e.src], db = boxes[e.dst];
        if (!sb || !db) continue;
        edges[e.id] = { src: [sb.x + sb.w, sb.y + sb.h / 2], dst: [db.x, db.y + db.h / 2], tensor: e.tensor };
      }
      return { boxes, edge_points: edges, width: layout.width, height: layout.height };
    });
}

async function applyCompareViews() {
  const [la, lb] = await Promise.all([buildViewForSide("a"), buildViewForSide("b")]);
  const exA = {}, exB = {};
  for (const [id, n] of Object.entries(state.compare.nodesA)) exA[id] = { op: n.display || n.op, outTensors: n.outputs };
  for (const [id, n] of Object.entries(state.compare.nodesB)) exB[id] = { op: n.display || n.op, outTensors: n.outputs };
  graphA.setData(la, exA);
  graphB.setData(lb, exB);
  refreshLayers();
}

// ---------- layers & heat ----------
function getLayer(id) {
  const el = $(id);
  return el ? el.checked : false;
}
function setLayer(id, checked) {
  const el = $(id);
  if (el) el.checked = checked;
}
$("heat-mode") && $("heat-mode").addEventListener("change", refreshLayers);

function heatStyles(mapped, mode) {
  if (!mapped) return {};
  const vals = Object.values(mapped.per_node).map((p) => (mode === "share" ? p.share : p.duration_us));
  if (!vals.length) return {};
  const min = Math.min(...vals), max = Math.max(...vals);
  const out = {};
  for (const [nid, p] of Object.entries(mapped.per_node)) {
    out[nid] = heatColor(mode === "share" ? p.share : p.duration_us, min, max);
  }
  $("heat-min").textContent = mode === "share" ? `${(min * 100).toFixed(1)}%` : `${min.toFixed(0)}us`;
  $("heat-max").textContent = mode === "share" ? `${(max * 100).toFixed(1)}%` : `${max.toFixed(0)}us`;
  return out;
}

function refreshLayers() {
  const mode = getLayer("heat-mode") || "duration";
  if (state.mode === "browse") {
    if (!state.browse.ir) { browseGraph.setStyles({}); netronApplyHeat({}); return; }
    const heat = getLayer("layer-heat") ? heatStyles(state.browse.mapped, mode) : {};
    if (state.useNetron) {
      // official Netron viewer: color nodes through the bridge
      const fillMap = {};
      for (const [nid, color] of Object.entries(heat)) fillMap[nid] = color;
      netronApplyHeat(fillMap);
      if (!Object.keys(heat).length) { $("heat-min").textContent = "min"; $("heat-max").textContent = "max"; }
      return;
    }
    const styles = {};
    for (const [nid, fill] of Object.entries(heat)) styles[nid] = { fill };
    browseGraph.setStyles(styles);
  } else if (state.mode === "compare") {
    if (state.compare.useNetronCompare) { netronApplyDiffTint(); return; }
    if (!state.compare.diff) return;
    const useDiff = getLayer("layer-diff");
    const only = getLayer("diff-only");
    const useHeat = getLayer("layer-heat");
    const heatA = useHeat ? heatStyles(state.compare.mappedA, mode) : {};
    const heatB = useHeat ? heatStyles(state.compare.mappedB, mode) : {};
    const stylesA = {}, stylesB = {};
    for (const nd of state.compare.diff.node_diffs) {
      const badge = DIFF_BADGE[nd.kind];
      const style = useDiff && badge ? { cls: badge.cls, badge: badge.letter } : {};
      if (useDiff && only && nd.kind === "unchanged") style.dim = true;
      if (nd.a_node_id) stylesA[nd.a_node_id] = style;
      if (nd.b_node_id) stylesB[nd.b_node_id] = { ...style };
    }
    for (const [nid, fill] of Object.entries(heatA)) stylesA[nid] = { ...(stylesA[nid] || {}), fill };
    for (const [nid, fill] of Object.entries(heatB)) stylesB[nid] = { ...(stylesB[nid] || {}), fill };
    graphA.setStyles(stylesA);
    graphB.setStyles(stylesB);
  } else {
    ptGraph.setStyles({});
  }
}

// ---------- floating layer panel ----------
function refreshLayerPanel() {
  const panel = $("layer-panel");
  if (hasContent() && !state.layerPanelClosed) {
    const mode = state.mode;
    const rows = [];
    if (mode === "browse" && state.browse.modelId)
      rows.push(["layer-netron", "Netron 官方渲染（画布）", state.useNetron]);
    if (mode === "compare") {
      rows.push(["layer-netron-cmp", "Netron 官方渲染（双侧）", state.compare.useNetronCompare]);
      rows.push(["layer-diff", "Diff 染色", getLayer("layer-diff")]);
      if (state.compare.useNetronCompare) {
        rows.push(["sync-view", "点击跟随对侧", getLayer("sync-view") !== false]);
        rows.push(["diff-only", "淡化相同节点", getLayer("diff-only")]);
      } else {
        rows.push(["layer-heat", "性能热力图", getLayer("layer-heat")]);
        rows.push(["show-tensornames", "边显示 Tensor 名", getLayer("show-tensornames") !== false]);
        rows.push(["sync-view", "同步平移缩放", getLayer("sync-view") !== false]);
        rows.push(["diff-only", "仅看差异", getLayer("diff-only")]);
      }
    } else {
      rows.push(["layer-heat", "性能热力图", getLayer("layer-heat")]);
      rows.push(["show-tensornames", "边显示 Tensor 名", getLayer("show-tensornames") !== false]);
    }
    const heatOn = getLayer("layer-heat");
    $("lp-toggles").innerHTML = rows.map(([id, label, checked]) =>
      `<label class="lp-toggle"><input type="checkbox" id="${id}" ${checked ? "checked" : ""}> ${esc(label)}</label>`).join("") +
      (heatOn ? `<label class="lp-toggle" style="color:var(--muted)">着色依据
        <select id="heat-mode" class="menu-select" style="margin-left:auto">
          <option value="duration" ${mode === "duration" ? "selected" : ""}>耗时</option>
          <option value="share" ${mode === "share" ? "selected" : ""}>占比</option>
        </select></label>` : "");
    for (const [id, , ] of rows) {
      const el = $(id);
      if (el) el.addEventListener("change", () => {
        if (id === "layer-netron") {
          state.useNetron = el.checked;
          applyBrowseRenderer();
          return;
        }
        if (id === "layer-netron-cmp") {
          state.compare.useNetronCompare = el.checked;
          applyCompareNetronViews();
          return;
        }
        if (id === "show-tensornames") {
          [browseGraph, graphA, graphB, ptGraph].forEach((g) => {
            g.opts.showTensorNames = el.checked;
            g._scheduleRender(true);
          });
        }
        refreshLayers();
      });
    }
    const hm = $("heat-mode");
    if (hm) hm.addEventListener("change", refreshLayers);
    panel.classList.remove("hidden");
  } else {
    panel.classList.add("hidden");
  }
}
$("lp-close").addEventListener("click", () => {
  state.layerPanelClosed = true;
  $("layer-panel").classList.add("hidden");
});

// ---------- file inputs ----------
$("file-open").addEventListener("change", (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (f) openModelFile(f);
});
$("file-profile").addEventListener("change", (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (f) openProfileFile(f, "browse");
});
$("file-profile-a").addEventListener("change", (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (f) openProfileFile(f, "a");
});
$("file-profile-b").addEventListener("change", (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (f) openProfileFile(f, "b");
});

// ---------- context menu ----------
function closeCtxMenu() { $("ctx-menu").classList.add("hidden"); }
function openCtxMenu(x, y, items) {
  const menu = $("ctx-menu");
  menu.innerHTML = items.map((it) => {
    if (it === "-") return `<div class="ctx-sep"></div>`;
    if (it.checkbox) {
      return `<label class="ctx-item"><input type="checkbox" data-action="${esc(it.action)}" ${it.checked ? "checked" : ""}> ${esc(it.label)}</label>`;
    }
    return `<div class="ctx-item ${it.disabled ? "disabled" : ""}" data-action="${esc(it.action)}">
      <span class="pi-icon">${esc(it.icon || "")}</span>${esc(it.label)}
      ${it.hint ? `<span class="ci-hint">${esc(it.hint)}</span>` : ""}</div>`;
  }).join("");
  menu.classList.remove("hidden");
  menu.style.left = "0px"; menu.style.top = "0px";
  const r = menu.getBoundingClientRect();
  menu.style.left = Math.min(x, window.innerWidth - r.width - 8) + "px";
  menu.style.top = Math.min(y, window.innerHeight - r.height - 8) + "px";
  menu.querySelectorAll("[data-action]").forEach((el) => {
    el.addEventListener("click", (ev) => {
      const act = el.dataset.action;
      closeCtxMenu();
      runAction(act);
    });
    el.addEventListener("change", (ev) => {
      if (el.dataset && el.dataset.action && el.type === "checkbox") {
        runAction("toggle:" + el.dataset.action.replace("toggle:", ""), el.checked);
      }
    });
  });
}

document.addEventListener("click", (e) => {
  if (!e.target.closest("#ctx-menu")) closeCtxMenu();
});
document.addEventListener("contextmenu", (e) => {
  if (e.target.closest("#props") || e.target.closest("#pt-sidebar") ||
      e.target.closest(".detail-card") || e.target.closest(".modal") ||
      e.target.closest("#palette")) return;
  e.preventDefault();
  if (state.mode === "pytorch") { openCtxMenu(e.clientX, e.clientY, pytorchMenuItems()); return; }
  const nodeEl = e.target.closest(".node-box");
  if (nodeEl && state.mode === "browse") {
    openCtxMenu(e.clientX, e.clientY, browseNodeMenuItems(nodeEl.getAttribute("data-id")));
  } else if (nodeEl && state.mode === "compare") {
    const side = e.target.closest("#cmp-graph-a") ? "a" : "b";
    openCtxMenu(e.clientX, e.clientY, compareNodeMenuItems(side, nodeEl.getAttribute("data-id")));
  } else {
    openCtxMenu(e.clientX, e.clientY, canvasMenuItems());
  }
});

function canvasMenuItems() {
  const items = [
    { action: "open", icon: "📄", label: "打开模型…", hint: "O" },
    { action: "compare", icon: "⇆", label: "对比两个模型…", hint: "D" },
    { action: "pytorch", icon: "🐍", label: "PyTorch 项目…" },
    "-",
    { action: "fit", icon: "⤢", label: "适合视图", hint: "F" },
    { action: "props-model", icon: "ⓘ", label: "模型属性", disabled: !state.browse.meta },
    { action: "load-profile", icon: "⏱", label: "加载 Profile…", disabled: !state.browse.modelId },
    "-",
    { action: "panel-layers", icon: "▤", label: "图层面板" },
    { action: "palette", icon: "⌘", label: "命令面板", hint: "⌘K" },
  ];
  if (state.mode === "compare" && state.compare.diffId) {
    items.push("-");
    items.push({ action: "export:json", icon: "↧", label: "导出 Diff JSON" });
    items.push({ action: "export:csv", icon: "↧", label: "导出 Diff CSV" });
    items.push({ action: "export:md", icon: "↧", label: "导出 Diff Markdown" });
    items.push({ action: "export:html", icon: "↧", label: "导出 Diff HTML" });
  }
  if (state.mode === "browse" && state.browse.profileId) {
    items.push({ action: "export:pjson", icon: "↧", label: "导出 Profile JSON" });
    items.push({ action: "export:pcsv", icon: "↧", label: "导出 Profile CSV" });
  }
  return items;
}

function browseNodeMenuItems(nodeId) {
  const n = state.browse.ir && state.browse.ir.nodes.find((x) => x.id === nodeId);
  return [
    { action: "toggle-node", icon: "▦", label: "展开 / 收起详情" },
    { action: "copy-node", icon: "⧉", label: "复制节点信息",
      disabled: !n },
    "-",
    { action: "fit", icon: "⤢", label: "适合视图", hint: "F" },
    { action: "palette", icon: "⌘", label: "搜索节点 / 命令", hint: "⌘K" },
  ];
}

function compareNodeMenuItems(side, nodeId) {
  const other = side === "a" ? "b" : "a";
  const partnerId = findPartner(side, nodeId);
  return [
    { action: "toggle-node", icon: "▦", label: "展开 / 收起详情" },
    { action: `locate:${other}:${partnerId || ""}`, icon: "⇥", label: `在 ${other.toUpperCase()} 侧定位`,
      disabled: !partnerId },
    { action: "copy-node", icon: "⧉", label: "复制节点信息" },
    "-",
    { action: "fit", icon: "⤢", label: "适合视图", hint: "F" },
  ];
}

function pytorchMenuItems() {
  return [
    { action: "pt-toggle", icon: "▦", label: "概览 / 详细切换" },
    { action: "pt-back", icon: "←", label: "返回模型画布" },
    "-",
    { action: "fit", icon: "⤢", label: "适合视图", hint: "F" },
  ];
}

function findPartner(side, nodeId) {
  const m = state.compare.match;
  if (!m) return null;
  for (const match of m.matches) {
    if (side === "a" && match.a_node_id === nodeId) return match.b_node_id;
    if (side === "b" && match.b_node_id === nodeId) return match.a_node_id;
  }
  return null;
}

function copyText(text) {
  navigator.clipboard && navigator.clipboard.writeText(text).then(
    () => toast("已复制到剪贴板"),
    () => toast("复制失败"));
}

function runAction(action, checked) {
  if (action.startsWith("toggle:")) {
    const key = action.slice(7);
    if (key === "layer-diff" || key === "layer-heat" || key === "diff-only") refreshLayers();
    return;
  }
  switch (action) {
    case "open": $("file-open").click(); break;
    case "compare": openCompareDialog(); break;
    case "pytorch": setMode("pytorch"); break;
    case "fit": [browseGraph, graphA, graphB, ptGraph].forEach((g) =>
      document.body.contains(g.container) && !g.container.closest(".hidden") && g.fitView()); break;
    case "props-model": showModelProps(); break;
    case "load-profile": $("file-profile").click(); break;
    case "panel-layers":
      state.layerPanelClosed = false;
      refreshLayerPanel();
      break;
    case "palette": openPalette(); break;
    case "toggle-node": {
      const g = activeGraph();
      if (g.selected) g.toggleExpand(g.selected);
      break;
    }
    case "copy-node": {
      const g = activeGraph();
      if (g.selected) {
        const n = state.browse.ir && state.browse.ir.nodes.find((x) => x.id === g.selected);
        copyText(n ? JSON.stringify(n, null, 2) : g.selected);
      }
      break;
    }
    case "pt-toggle": {
      if (!state.pytorch.result) { toast("请先加载 PyTorch 项目"); break; }
      state.pytorch.view = state.pytorch.view === "overview" ? "detail" : "overview";
      renderPtGraph();
      break;
    }
    case "pt-back": setMode(state.browse.modelId ? "browse" : "browse"); break;
    default:
      if (action.startsWith("export:")) exportAs(action.slice(7));
      else if (action.startsWith("locate:")) {
        const [, side, id] = action.split(":");
        if (id) {
          const g = side === "a" ? graphA : graphB;
          g.focusNode(id); g.selectExternal(id);
        } else toast("该节点在另一侧没有可信对应（或为歧义）");
      }
  }
}

async function exportAs(fmt) {
  try {
    if (fmt.startsWith("p")) {
      if (!state.browse.profileId) { toast("当前没有已映射的 profile"); return; }
      await apiDownload("/api/export/profile", { profile_id: state.browse.profileId, format: fmt.slice(1) },
        `profile_report.${fmt.slice(1)}`);
    } else {
      if (!state.compare.diffId) { toast("当前没有对比结果"); return; }
      await apiDownload("/api/export/diff", { diff_id: state.compare.diffId, format: fmt }, `diff_report.${fmt}`);
    }
    toast("已导出");
  } catch (err) { toast(`导出失败: ${err.message}`); }
}

// ---------- command palette ----------
let paletteSel = 0;
let paletteEntries = [];

function paletteActions() {
  const acts = [
    { icon: "📄", label: "打开模型…", hint: "O", run: () => $("file-open").click() },
    { icon: "⇆", label: "对比两个模型…", hint: "D", run: openCompareDialog },
    { icon: "🐍", label: "PyTorch 项目…", run: () => setMode("pytorch") },
    { icon: "⤢", label: "适合视图", hint: "F", run: () => runAction("fit") },
    { icon: "⌘", label: "命令面板帮助（↑↓ 选择，↵ 执行）", run: () => {} },
  ];
  if (state.browse.meta) acts.push({ icon: "ⓘ", label: "模型属性", run: showModelProps });
  if (state.browse.modelId) acts.push({ icon: "⏱", label: "加载 Profile…", run: () => $("file-profile").click() });
  if (state.mode === "compare" && state.compare.diffId) {
    for (const f of ["json", "csv", "md", "html"]) {
      acts.push({ icon: "↧", label: `导出 Diff ${f.toUpperCase()}`, run: () => exportAs(f) });
    }
    acts.push({ icon: "▤", label: "差异摘要", run: renderDiffSummary });
  }
  if (state.browse.profileId) {
    acts.push({ icon: "↧", label: "导出 Profile JSON", run: () => exportAs("pjson") });
  }
  return acts;
}

function paletteNodeMatches(q) {
  const out = [];
  if (state.mode === "browse" && state.browse.ir) {
    for (const n of state.browse.ir.nodes) {
      const hay = `${n.id} ${n.display_name} ${n.op_type} ${(n.inputs || []).join(" ")} ${(n.outputs || []).join(" ")}`.toLowerCase();
      if (hay.includes(q)) out.push({ icon: "▪", label: `${n.display_name}`, sub: `${n.op_type}`, run: () => {
        setMode("browse");
        if (state.useNetron) {
          netronFocus(n.display_name || n.id);
          return;
        }
        browseGraph.focusNode(n.id); browseGraph.selectExternal(n.id);
        if (!browseGraph.expanded.has(n.id)) browseGraph.toggleExpand(n.id);
        showProps("browse", n.id);
      }});
    }
  } else if (state.mode === "compare") {
    for (const [id, n] of Object.entries(state.compare.nodesA)) {
      if (`${id} ${n.display} ${n.op}`.toLowerCase().includes(q))
        out.push({ icon: "🅐", label: `${n.display || id}`, sub: `${n.op} · A 侧`, run: () => {
          graphA.focusNode(id); graphA.selectExternal(id);
          if (!graphA.expanded.has(id)) graphA.toggleExpand(id);
          showProps("a", id);
        }});
    }
    for (const [id, n] of Object.entries(state.compare.nodesB)) {
      if (`${id} ${n.display} ${n.op}`.toLowerCase().includes(q))
        out.push({ icon: "🅑", label: `${n.display || id}`, sub: `${n.op} · B 侧`, run: () => {
          graphB.focusNode(id); graphB.selectExternal(id);
          if (!graphB.expanded.has(id)) graphB.toggleExpand(id);
          showProps("b", id);
        }});
    }
  } else if (state.mode === "pytorch" && state.pytorch.result && state.pytorch.result.ok) {
    for (const n of flattenClient(state.pytorch.result.module_tree, 0)) {
      if (`${n.path} ${n.class_name}`.toLowerCase().includes(q))
        out.push({ icon: "▪", label: `${n.class_name}`, sub: n.path || "(root)", run: () => {
          ptGraph.focusNode(n.path || "__root__"); ptGraph.selectExternal(n.path || "__root__");
        }});
    }
  }
  return out.slice(0, 40);
}

function openPalette(prefill = "") {
  $("palette").classList.remove("hidden");
  $("palette-input").value = prefill;
  $("palette-input").focus();
  renderPalette();
}
function closePalette() { $("palette").classList.add("hidden"); }

function renderPalette() {
  const q = $("palette-input").value.trim().toLowerCase();
  const acts = paletteActions().filter((a) => !q || a.label.toLowerCase().includes(q));
  const nodes = q ? paletteNodeMatches(q) : [];
  paletteEntries = [
    ...acts.map((a) => ({ ...a, kind: "act" })),
    ...nodes.map((n) => ({ ...n, kind: "node" })),
  ];
  paletteSel = Math.min(paletteSel, Math.max(0, paletteEntries.length - 1));
  $("palette-list").innerHTML = paletteEntries.map((e, i) => `
    <div class="palette-item ${i === paletteSel ? "sel" : ""}" data-i="${i}">
      <span class="pi-icon">${e.icon}</span>
      <span>${esc(e.label)} ${e.sub ? `<span class="pi-sub">— ${esc(e.sub)}</span>` : ""}</span>
      ${e.hint ? `<span class="pi-hint">${esc(e.hint)}</span>` : ""}
    </div>`).join("") || `<div class="palette-item"><span class="pi-sub">无匹配</span></div>`;
  $("palette-list").querySelectorAll(".palette-item[data-i]").forEach((el) => {
    el.addEventListener("click", () => {
      const entry = paletteEntries[parseInt(el.dataset.i, 10)];
      closePalette();
      entry.run();
    });
  });
  const sel = $("palette-list").querySelector(".palette-item.sel");
  if (sel) sel.scrollIntoView({ block: "nearest" });
}

$("palette-input").addEventListener("input", () => { paletteSel = 0; renderPalette(); });
$("palette-input").addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown") { e.preventDefault(); paletteSel = Math.min(paletteSel + 1, paletteEntries.length - 1); renderPalette(); }
  else if (e.key === "ArrowUp") { e.preventDefault(); paletteSel = Math.max(paletteSel - 1, 0); renderPalette(); }
  else if (e.key === "Enter") {
    e.preventDefault();
    const entry = paletteEntries[paletteSel];
    if (entry) { closePalette(); entry.run(); }
  }
});
$("palette").addEventListener("click", (e) => {
  if (e.target === $("palette")) closePalette();
});

// ---------- pytorch ----------
$("pt-load").addEventListener("click", loadPytorch);
$("pt-view-toggle").addEventListener("click", () => runAction("pt-toggle"));
$("pt-back").addEventListener("click", () => runAction("pt-back"));

function parseLeafRules(text) {
  return text.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    if (l.startsWith("class:")) return { class_name: l.slice(6).trim() };
    if (l.startsWith("path:")) return { path_pattern: l.slice(5).trim() };
    return { class_name: l };
  });
}

async function loadPytorch() {
  setStatus($("pt-status"), "子进程加载中…");
  try {
    const res = await apiPost("/api/pytorch/load", {
      project_path: $("pt-project").value.trim(),
      import_path: $("pt-import").value.trim(),
      class_name: $("pt-class").value.trim(),
      leaf_rules: parseLeafRules($("pt-leaves").value),
      trace: true,
    });
    state.pytorch.result = res;
    if (!res.ok) {
      setStatus($("pt-status"), `加载失败: ${res.error}`, "err");
      return;
    }
    const msg = res.degraded
      ? `已加载模块树 (${countModules(res.module_tree)} 模块) · ⚠ FX 追踪失败，仅模块树: ${res.fx_error}`
      : `已加载模块树 (${countModules(res.module_tree)} 模块) · FX 算子图已捕获`;
    setStatus($("pt-status"), msg, res.degraded ? "err" : "ok");
    refreshWelcome();
    refreshLayerPanel();
    renderPtTree();
    renderPtGraph();
  } catch (e) {
    setStatus($("pt-status"), `加载失败: ${e.message}`, "err");
  }
}
function countModules(tree) {
  let n = 1;
  for (const c of tree.children || []) n += countModules(c);
  return n;
}
function flattenClient(tree, depth) {
  const out = [{ path: tree.path, class_name: tree.class_name, depth,
                 param_count: tree.param_count, param_bytes: tree.param_bytes,
                 is_leaf: tree.is_leaf }];
  for (const c of tree.children || []) out.push(...flattenClient(c, depth + 1));
  return out;
}
function renderPtTree() {
  const res = state.pytorch.result;
  if (!res || !res.ok) return;
  const flat = flattenClient(res.module_tree, 0);
  $("pt-tree").innerHTML = flat.map((n) => `
    <div class="pt-row" data-path="${esc(n.path)}">
      <span class="pt-depth">${"· ".repeat(n.depth)}</span>
      <span><b>${esc(n.class_name)}</b> <span style="color:var(--muted)">${esc(n.path || "(root)")}</span>
      ${n.is_leaf ? '<span class="pt-leaf">leaf</span>' : ""}
      ${n.param_count ? `<span style="color:var(--muted)">${n.param_count} params</span>` : ""}</span>
    </div>`).join("");
  $("pt-tree").querySelectorAll(".pt-row").forEach((el) =>
    el.addEventListener("click", () => {
      $("pt-tree").querySelectorAll(".pt-row").forEach((r) => r.classList.remove("sel"));
      el.classList.add("sel");
      showPtDetail(el.dataset.path);
      if (ptGraph.boxes[el.dataset.path]) { ptGraph.focusNode(el.dataset.path); ptGraph.selectExternal(el.dataset.path); }
    }));
}
function ptLayout() {
  const res = state.pytorch.result;
  if (!res || !res.ok) return null;
  const overview = state.pytorch.view === "overview";
  const boxes = {}; const edges = {};
  let y = 0;
  const NODE_W = 190, NODE_H = 40, XG = 64, YG = 18;
  function walk(node, depth) {
    const path = node.path || "__root__";
    const collapsed = overview && (node.is_leaf || !(node.children || []).length);
    boxes[path] = { node_id: path, x: depth * (NODE_W + XG), y: y * (NODE_H + YG),
      w: NODE_W, h: NODE_H, kind: "node", label: node.class_name };
    y += 1;
    if (collapsed) return;
    for (const c of node.children || []) {
      edges[`e:${node.path || "root"}->${c.path}`] = { src: node.path || "__root__", dst: c.path };
      walk(c, depth + 1);
    }
  }
  walk(res.module_tree, 0);
  const edge_points = {};
  for (const [id, e] of Object.entries(edges)) {
    const sb = boxes[e.src], db = boxes[e.dst];
    if (!sb || !db) continue;
    edge_points[id] = { src: [sb.x + sb.w, sb.y + sb.h / 2], dst: [db.x, db.y + db.h / 2], tensor: "" };
  }
  const maxDepth = Math.max(0, ...Object.values(boxes).map((b) => b.x));
  return { boxes, edge_points, width: maxDepth + NODE_W + 40, height: y * (NODE_H + YG) };
}
function renderPtGraph() {
  const layout = ptLayout();
  if (!layout) return;
  const extra = {};
  for (const id in layout.boxes) extra[id] = { op: layout.boxes[id].label, outTensors: [] };
  ptGraph.setData(layout, extra);
}
function ptDetailHTML(path) {
  const res = state.pytorch.result;
  if (!res || !res.ok) return null;
  const flat = flattenClient(res.module_tree, 0);
  const n = flat.find((x) => x.path === path);
  if (!n) return null;
  return `<h5><span>${esc(n.path || "(root)")}</span><span class="op-tag">${esc(n.class_name)}</span></h5>
    <table><tr><td>参数量</td><td>${n.param_count}</td></tr>
    <tr><td>参数字节</td><td>${n.param_bytes || 0}</td></tr>
    <tr><td>leaf</td><td>${n.is_leaf ? "是" : "否"}</td></tr></table>
    ${res.fx_error ? `<div class="warnline">FX 诊断: ${esc(res.fx_error)}</div>` : ""}`;
}
function showPtDetail(path) {
  if (!path) { $("props").classList.add("hidden"); return; }
  openProps();
  $("props-title").textContent = "模块属性";
  $("props-body").innerHTML = ptDetailHTML(path) || "";
}

// ---------- keyboard shortcuts ----------
window.addEventListener("keydown", (e) => {
  const inField = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    openPalette();
    return;
  }
  if (e.key === "Escape") {
    [browseGraph, graphA, graphB, ptGraph].forEach((g) => g.collapseAll());
    closeCtxMenu();
    $("dlg-compare").classList.add("hidden");
    closePalette();
    return;
  }
  if (inField || e.metaKey || e.ctrlKey) return;
  if (e.key === "o" || e.key === "O") { e.preventDefault(); $("file-open").click(); }
  else if (e.key === "d" || e.key === "D") { e.preventDefault(); openCompareDialog(); }
  else if (e.key === "f" || e.key === "F") { runAction("fit"); }
});

// test/debug hook (also used by automation): expose module internals
window.__mtbx = { state, applyCompareNetronViews, netronApplyDiffTint,
  netronFocusId, netronTintDiff, refreshLayers, refreshLayerPanel };

// ---------- init ----------
(async function init() {
  setMode("browse");
  try {
    const h = await apiGet("/api/health");
    $("sb-right").textContent = h.torch_available ? "" : "torch 不可用（PyTorch 功能受限）";
  } catch (e) {
    $("sb-right").textContent = "无法连接后端服务";
  }
})();
