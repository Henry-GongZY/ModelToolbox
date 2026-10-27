// ModelToolbox frontend — Netron-inspired interaction:
// file-picker / drag-and-drop loading, in-place expandable node details,
// edge tensor names, properties drawer. Vanilla ES modules, no build step.
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
  detailCache: { a: {}, b: {} },
  pendingCompare: { a: null, b: null }, // chosen files before running the diff
};

// ---------- tiny helpers ----------
function toast(msg, ms = 2600) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), ms);
}
function setStatus(el, msg, cls = "") { el.textContent = msg || ""; el.className = "status " + cls; }
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
const HEAT_COLORS = ["#e8e6ef", "#b5c2dc", "#6f97c8", "#3a6cb4", "#1d4494"];
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

// synced view (compare)
let syncing = false;
graphA.opts.onTransform = (g) => mirrorTransform(graphA, graphB);
graphB.opts.onTransform = (g) => mirrorTransform(graphB, graphA);
function mirrorTransform(src, dst) {
  if (!$("sync-view").checked || syncing) return;
  syncing = true;
  dst.tx = src.tx; dst.ty = src.ty; dst.scale = src.scale;
  dst.viewport.setAttribute("transform", `translate(${dst.tx} ${dst.ty}) scale(${dst.scale})`);
  dst._scheduleRender();
  dst._positionOverlays();
  syncing = false;
}

// ---------- mode ----------
function setMode(mode) {
  state.mode = mode;
  $("mode-indicator").textContent =
    { browse: "单模型浏览", compare: "双模型对比", pytorch: "PyTorch 层级" }[mode];
  $("view-graph").classList.toggle("hidden", mode === "pytorch");
  $("view-pytorch").classList.toggle("hidden", mode !== "pytorch");
  $("single-wrap").classList.toggle("hidden", mode !== "browse");
  $("dual-wrap").classList.toggle("hidden", mode !== "compare");
  $("compare-header").classList.toggle("hidden", mode !== "compare");
  $("diff-nav").classList.toggle("hidden", mode !== "compare");
  document.querySelectorAll(".cmp-only").forEach((el) =>
    el.classList.toggle("hidden", mode !== "compare"));
  $("btn-export").disabled = !(mode === "compare" && state.compare.diffId) &&
    !(mode === "browse" && state.browse.profileId);
  if (mode === "compare") { $("layer-diff").checked = true; }
  $("legend").classList.toggle("hidden", mode === "pytorch");
  refreshLayers();
}

// ---------- detail cards ----------
function fmtShape(vi) {
  if (!vi) return "";
  const shp = vi.shape === null ? "?" : JSON.stringify(vi.shape);
  return `${vi.dtype || "?"} ${shp}`;
}

function browseDetailHTML(nodeId) {
  const n = state.browse.ir && state.browse.ir.nodes.find((x) => x.id === nodeId);
  if (!n) return "__loading__";
  const rows = [];
  for (const t of (n.inputs || []).filter(Boolean)) {
    const vi = n.input_info[t];
    rows.push(`<tr><td>← ${esc(t)}</td><td>${esc(fmtShape(vi))}</td></tr>`);
  }
  for (const t of (n.outputs || []).filter(Boolean)) {
    const vi = n.output_info[t];
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
  const perf = state.browse.mapped && state.browse.mapped.per_node[nodeId];
  const perfLine = perf
    ? `<div class="perf">⏱ ${perf.duration_us.toFixed(1)} us · ${(perf.share * 100).toFixed(1)}% · ×${perf.calls}</div>` : "";
  const sg = (n.subgraphs || []).length
    ? `<div class="warnline">含嵌套子图 ×${n.subgraphs.length}</div>` : "";
  return `<h5><span>${esc(n.display_name)}</span><span class="op-tag">${esc(n.op_type)}</span></h5>
    <div class="sec">tensors</div><table>${rows.join("")}</table>
    ${attrRows ? `<div class="sec">attributes</div><table>${attrRows}</table>` : ""}
    ${paramRows ? `<div class="sec">weights</div><table>${paramRows}</table>` : ""}
    ${sg}${perfLine}`;
}

function compareDetailHTML(side, nodeId) {
  const cache = state.detailCache[side][nodeId];
  if (!cache) {
    fetchCompareDetail(side, nodeId); // async fill + rerender if still expanded
    return "__loading__";
  }
  const n = cache;
  const diff = diffByNode(side)[nodeId];
  const diffLine = diff && diff.kind !== "unchanged"
    ? `<div class="warnline">${esc(diff.explanation)}（置信度 ${diff.confidence}）</div>`
    : (diff && diff.kind === "ambiguous"
      ? `<div class="warnline">⚠ 存在多个等价候选，对应关系不确定</div>` : "");
  let chRows = "";
  for (const c of (diff && diff.changes) || []) {
    chRows += `<tr><td>${esc(c.field)}</td><td>${esc(JSON.stringify(c.before))} → ${esc(JSON.stringify(c.after))}</td></tr>`;
  }
  const rows = [];
  for (const t of (n.inputs || []).filter(Boolean)) {
    const vi = n.input_info[t];
    rows.push(`<tr><td>← ${esc(t)}</td><td>${esc(fmtShape(vi))}</td></tr>`);
  }
  for (const t of (n.outputs || []).filter(Boolean)) {
    const vi = n.output_info[t];
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
  return `<h5><span>${esc(n.display_name)}</span><span class="op-tag">${esc(n.op_type)}</span></h5>
    <div class="sec">tensors</div><table>${rows.join("")}</table>
    ${attrRows ? `<div class="sec">attributes</div><table>${attrRows}</table>` : ""}
    ${paramRows ? `<div class="sec">weights</div><table>${paramRows}</table>` : ""}
    ${chRows ? `<div class="sec">字段级变化</div><table>${chRows}</table>` : ""}
    ${diffLine}`;
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
    const diff = diffByNode(side)[nodeId];
    if (!cache) {
      body.innerHTML = "<p>加载中…</p>";
      fetchCompareDetail(side, nodeId).then(() => showProps(side, nodeId));
      return;
    }
    body.innerHTML = propsForNode(cache, null, diff);
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

// ---------- file loading (Netron style) ----------
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
    renderPropsModel(res.meta);
    setStatusText(`${res.meta.name} — ${res.meta.node_count} 节点 / ${res.meta.edge_count} 边 (${dt}ms)`);
    $("sb-right").textContent = `${res.meta.name} · ${(res.meta.file_size_bytes / 1024).toFixed(1)} KB`;
    setMode("browse");
    refreshLayers();
  } catch (e) {
    setStatusText(`加载失败: ${e.message}`, "err");
    toast(`加载失败: ${e.message}`);
  }
}

function renderPropsModel(meta) {
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
    ? `<p style="color:var(--ambiguous)">⚠ ${meta.warnings.length} 条警告</p>` : "");
}
function openProps() { $("props").classList.remove("hidden"); }

async function openProfileFile(file, target) {
  // target: "browse" | "a" | "b"
  try {
    const b64 = await fileToBase64(file);
    let res, note = "";
    if (target === "browse") {
      if (!state.browse.modelId) { toast("请先打开模型"); return; }
      res = await apiPost("/api/profile/upload", { name: file.name, data_base64: b64, model_id: state.browse.modelId });
      state.browse.profileId = res.profile_id;
      state.browse.mapped = res.mapped;
      $("layer-heat").checked = true;
      refreshLayers();
    } else {
      const modelId = target === "a" ? state.compare.aModelId : state.compare.bModelId;
      res = await apiPost("/api/profile/upload", { name: file.name, data_base64: b64, model_id: modelId });
      if (target === "a") state.compare.mappedA = res.mapped;
      else state.compare.mappedB = res.mapped;
      $("layer-heat").checked = true;
      refreshLayers();
    }
    const unmapped = res.mapped.unmapped.length;
    note = `${file.name}: ${Object.keys(res.mapped.per_node).length} 映射 / ${unmapped} 未映射` +
      (res.mapped.model_fingerprint_matched === false ? " · ⚠ 指纹不匹配" : "");
    toast(note, 3400);
    setStatusText(note, res.mapped.model_fingerprint_matched === false ? "err" : "ok");
  } catch (e) {
    toast(`profile 导入失败: ${e.message}`);
  }
}

// ---------- file pickers & drag&drop ----------
$("btn-open").addEventListener("click", () => $("file-open").click());
$("file-open").addEventListener("change", (e) => {
  if (e.target.files[0]) openModelFile(e.target.files[0]);
  e.target.value = "";
});
$("btn-panel").addEventListener("click", () => {
  $("props").classList.toggle("hidden");
});
$("btn-zoom-fit").addEventListener("click", () => {
  (state.mode === "compare" ? [graphA, graphB] : state.mode === "pytorch" ? [ptGraph] : [browseGraph])
    .forEach((g) => g.fitView());
});
$("show-tensornames").addEventListener("change", (e) => {
  [browseGraph, graphA, graphB, ptGraph].forEach((g) => {
    g.opts.showTensorNames = e.target.checked;
    g._scheduleRender(true);
  });
});

// compare dialog
$("btn-compare").addEventListener("click", () => {
  $("dlg-compare").classList.remove("hidden");
  state.pendingCompare = { a: null, b: null };
  $("pick-a-name").textContent = "未选择";
  $("pick-b-name").textContent = "未选择";
});
$("cmp-cancel").addEventListener("click", () => $("dlg-compare").classList.add("hidden"));
$("pick-a").addEventListener("click", () => $("file-open-b").click());
$("pick-b").addEventListener("click", () => $("file-open-b").click());
let pickTarget = null;
$("pick-a").addEventListener("click", () => { pickTarget = "a"; });
$("pick-b").addEventListener("click", () => { pickTarget = "b"; });
$("file-open-b").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f || !pickTarget) return;
  state.pendingCompare[pickTarget] = f;
  $(`pick-${pickTarget}-name`).textContent = f.name;
});
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
  graphA.setData(res.layouts.a, exA);
  graphB.setData(res.layouts.b, exB);
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
  if (state.mode === "compare" || onnx.length >= 2) {
    if (onnx.length >= 2) {
      state.pendingCompare = { a: onnx[0], b: onnx[1] };
      await runCompareWithFiles(onnx[0], onnx[1]);
      return;
    }
  }
  if (onnx.length === 1) { await openModelFile(onnx[0]); return; }
  if (prof.length && state.mode === "browse") { await openProfileFile(prof[0], "browse"); return; }
  if (prof.length && state.mode === "compare") {
    if (prof[0]) await openProfileFile(prof[0], "a");
    if (prof[1]) await openProfileFile(prof[1], "b");
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
  if (nd.a_node_id) { graphA.focusNode(nd.a_node_id); graphA.selectExternal(nd.a_node_id); }
  if (nd.b_node_id) { graphB.focusNode(nd.b_node_id); graphB.selectExternal(nd.b_node_id); }
}
$("diff-prev").addEventListener("click", () => gotoDiff((state.compare.navIdx === -1 ? 0 : state.compare.navIdx) - 1));
$("diff-next").addEventListener("click", () => gotoDiff(state.compare.navIdx + 1));

function renderFoldGroups() {
  openProps();
  const groups = state.compare.folding.groups;
  if (!groups.length) return;
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

// ---------- layers / heat ----------
function refreshLayers() {
  if (state.mode === "browse") refreshBrowseStyles();
  else if (state.mode === "compare") refreshCompareStyles();
  else ptGraph.setStyles({});
  $("heat-mode").classList.toggle("hidden", !$("layer-heat").checked);
  $("tg-diff-wrap").classList.toggle("hidden", state.mode === "pytorch");
  $("tg-heat-wrap").classList.toggle("hidden", state.mode === "pytorch");
}

$("layer-diff").addEventListener("change", refreshLayers);
$("layer-heat").addEventListener("change", refreshLayers);
$("heat-mode").addEventListener("change", refreshLayers);
$("diff-only").addEventListener("change", refreshLayers);

function heatStyles(mapped, mode) {
  if (!mapped) return {};
  const vals = Object.values(mapped.per_node).map((p) => (mode === "share" ? p.share : p.duration_us));
  if (!vals.length) return {};
  const min = Math.min(...vals), max = Math.max(...vals);
  const out = {};
  for (const [nid, p] of Object.entries(mapped.per_node)) {
    out[nid] = heatColor(mode === "share" ? p.share : p.duration_us, min, max);
  }
  updateHeatLabels(min, max, mode);
  return out;
}
function updateHeatLabels(min, max, mode) {
  $("heat-min").textContent = mode === "share" ? `${(min * 100).toFixed(1)}%` : `${min.toFixed(0)}us`;
  $("heat-max").textContent = mode === "share" ? `${(max * 100).toFixed(1)}%` : `${max.toFixed(0)}us`;
}

function refreshBrowseStyles() {
  if (!state.browse.ir) return;
  const heat = $("layer-heat").checked ? heatStyles(state.browse.mapped, $("heat-mode").value) : {};
  if (!heat || !Object.keys(heat).length) { browseGraph.setStyles({}); return; }
  const styles = {};
  for (const [nid, fill] of Object.entries(heat)) styles[nid] = { fill };
  browseGraph.setStyles(styles);
  if (!$("layer-heat").checked) { $("heat-min").textContent = "min"; $("heat-max").textContent = "max"; }
}

function refreshCompareStyles() {
  if (!state.compare.diff) return;
  const useDiff = $("layer-diff").checked;
  const only = $("diff-only").checked;
  const useHeat = $("layer-heat").checked;
  const mode = $("heat-mode").value;
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
}

// ---------- profile file pickers (compare + browse via menu) ----------
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
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey) {
    if (e.key === "o") { e.preventDefault(); $("btn-compare").classList.contains("!hidden") || $("dlg-compare").classList.contains("hidden") ? $("file-open").click() : null; }
  }
});

// ---------- export ----------
$("btn-export").addEventListener("click", (e) => {
  e.stopPropagation();
  const canProfile = state.mode === "browse" && state.browse.profileId;
  const canDiff = state.mode === "compare" && state.compare.diffId;
  $("export-menu").querySelectorAll("button").forEach((b) => {
    const fmt = b.dataset.fmt;
    b.style.display = (fmt.startsWith("p") && !canProfile) || (!fmt.startsWith("p") && !canDiff)
      ? "none" : "";
  });
  $("export-menu").classList.toggle("hidden");
});
document.addEventListener("click", (e) => {
  if (!e.target.closest("#export-menu")) $("export-menu").classList.add("hidden");
});
$("export-menu").addEventListener("click", async (e) => {
  const fmt = e.target.dataset && e.target.dataset.fmt;
  if (!fmt) return;
  $("export-menu").classList.add("hidden");
  try {
    if (fmt.startsWith("p")) {
      await apiDownload("/api/export/profile", { profile_id: state.browse.profileId, format: fmt.slice(1) },
        `profile_report.${fmt.slice(1)}`);
    } else {
      await apiDownload("/api/export/diff", { diff_id: state.compare.diffId, format: fmt }, `diff_report.${fmt}`);
    }
    toast("已导出");
  } catch (err) { toast(`导出失败: ${err.message}`); }
});

// ---------- pytorch ----------
$("btn-pytorch").addEventListener("click", () => setMode("pytorch"));
$("pt-load").addEventListener("click", loadPytorch);
$("pt-view-toggle").addEventListener("click", () => {
  if (!state.pytorch.result) { toast("请先加载 PyTorch 项目"); return; }
  state.pytorch.view = state.pytorch.view === "overview" ? "detail" : "overview";
  renderPtGraph();
});

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
  const boxes = {}; const extra = {}; const edges = {};
  let y = 0;
  const NODE_W = 190, NODE_H = 40, XG = 64, YG = 18;
  function walk(node, depth) {
    const path = node.path || "__root__";
    const collapsed = overview && (node.is_leaf || !(node.children || []).length);
    boxes[path] = { node_id: path, x: depth * (NODE_W + XG), y: y * (NODE_H + YG),
      w: NODE_W, h: NODE_H, kind: "node", label: node.class_name };
    extra[path] = { op: `${node.param_count} params`, outTensors: [] };
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
  ptGraph.setData(layout, {});
  for (const id in layout.boxes) ptGraph.nodeExtra[id] = layout.boxes[id].label ? { op: layout.boxes[id].label } : {};
  ptGraph._scheduleRender(true);
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

// ---------- misc ----------
function setStatusText(msg, cls = "") {
  $("sb-left").textContent = msg;
  $("sb-left").style.color = cls === "err" ? "var(--removed)" : cls === "ok" ? "var(--added)" : "";
}

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    [browseGraph, graphA, graphB, ptGraph].forEach((g) => g.collapseAll());
    $("dlg-compare").classList.add("hidden");
    $("export-menu").classList.add("hidden");
  }
});

// ---------- init ----------
(async function init() {
  setMode("browse");
  try {
    const h = await apiGet("/api/health");
    $("sb-right").textContent = h.torch_available ? "torch 可用" : "torch 不可用（PyTorch 功能受限）";
  } catch (e) {
    $("sb-right").textContent = "无法连接后端服务";
  }
})();
