// ModelToolbox frontend application (vanilla ES modules, no build step).
import { apiPost, apiGet, apiDownload } from "./api.js";
import { GraphView } from "./graph.js";

// ---- state -----------------------------------------------------------------
const state = {
  mode: "browse",
  browse: { modelId: null, ir: null, meta: null, layout: null, profile: null, mapped: null },
  compare: {
    diffId: null, diff: null, match: null, folding: null,
    layouts: null, irA: null, irB: null, folded: new Set(),
    navIdx: -1, syncFold: true,
  },
  pytorch: { result: null, view: "overview", selected: null },
};

const $ = (id) => document.getElementById(id);

// ---- small helpers ----------------------------------------------------------
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

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function kvTable(pairs) {
  const rows = pairs.filter((p) => p[1] !== undefined && p[1] !== null)
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${typeof v === "object" ? "<code>" + esc(JSON.stringify(v)) + "</code>" : esc(v)}</td></tr>`);
  return `<table>${rows.join("")}</table>`;
}

const DIFF_BADGE = {
  added: { letter: "A", color: "#0a7d33", cls: "node-added" },
  removed: { letter: "D", color: "#b3261e", cls: "node-removed" },
  op_changed: { letter: "M", color: "#c62828", cls: "node-modified" },
  attr_changed: { letter: "M", color: "#c62828", cls: "node-modified" },
  shape_changed: { letter: "S", color: "#c62828", cls: "node-modified" },
  dtype_changed: { letter: "T", color: "#c62828", cls: "node-modified" },
  param_changed: { letter: "P", color: "#c62828", cls: "node-modified" },
  connection_changed: { letter: "C", color: "#c62828", cls: "node-modified" },
  ambiguous: { letter: "?", color: "#b58a00", cls: "node-ambiguous" },
};

const HEAT_COLORS = ["#e8e6ef", "#b5c2dc", "#6f97c8", "#3a6cb4", "#1d4494"];

function heatColor(v, min, max) {
  if (v == null || max == null || max <= min) return HEAT_COLORS[2];
  const t = Math.min(Math.max((v - min) / (max - min || 1), 0), 1);
  return HEAT_COLORS[Math.min(Math.floor(t * 5), 4)];
}

// ---- mode switching ----------------------------------------------------------
document.querySelectorAll("#mode-tabs .tab").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("#mode-tabs .tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode));
  $("view-browse").classList.toggle("hidden", mode !== "browse");
  $("view-compare").classList.toggle("hidden", mode !== "compare");
  $("view-pytorch").classList.toggle("hidden", mode !== "pytorch");
  $("layers").classList.toggle("hidden", mode === "pytorch");
  $("diff-only-wrap").classList.toggle("hidden", mode !== "compare");
  $("sync-view-wrap").classList.toggle("hidden", mode !== "compare");
  $("heat-mode").classList.toggle("hidden", mode === "pytorch");
  if (mode === "compare") {
    $("layer-diff").checked = true;  // diff view defaults to the diff layer on
    refreshCompareStyles();
  }
  $("legend-heat").classList.toggle("hidden", !heatActive());
  $("legend-diff").classList.toggle("hidden", mode !== "compare" || heatOnly());
  $("sb-left").textContent = `模式: ${{ browse: "单模型浏览", compare: "双模型对比", pytorch: "PyTorch 层级" }[mode]}`;
}

function diffLayerActive() {
  return state.mode === "compare" && $("layer-diff").checked;
}
function heatLayerActive() {
  return $("layer-heat").checked && state.browse.mapped;
}
function heatActive() { return $("layer-heat").checked; }
function heatOnly() { return $("layer-heat").checked && !$("layer-diff").checked; }

$("layer-diff").addEventListener("change", refreshCompareStyles);
$("layer-heat").addEventListener("change", () => {
  $("legend-heat").classList.toggle("hidden", !heatActive());
  $("heat-mode").classList.toggle("hidden", state.mode === "pytorch");
  refreshHeat();
});
$("heat-mode").addEventListener("change", refreshHeat);

// ===========================================================================
// BROWSE MODE
// ===========================================================================
const browseGraph = new GraphView($("browse-graph"), {
  onSelect: (id) => showBrowseDetail(id),
  onEmptyClick: () => showBrowseDetail(null),
});

$("browse-load").addEventListener("click", loadBrowseModel);
$("browse-path").addEventListener("keydown", (e) => { if (e.key === "Enter") loadBrowseModel(); });
$("browse-fit").addEventListener("click", () => browseGraph.fitView());
$("browse-search").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runBrowseSearch($("browse-search").value);
});
$("browse-search-next").addEventListener("click", () => cycleBrowseSearch());
$("browse-profile-load").addEventListener("click", loadBrowseProfile);
$("browse-export").addEventListener("click", () => {
  if (!state.browse.modelId) return;
  apiDownload("/api/export/profile", { profile_id: state.browse.profile }, "profile_report.json")
    .catch((e) => setStatus($("browse-status"), e.message, "err"));
});

async function loadBrowseModel() {
  const path = $("browse-path").value.trim();
  setStatus($("browse-status"), "加载中…");
  try {
    const t0 = performance.now();
    const res = await apiPost("/api/model/load", { path });
    const dt = Math.round(performance.now() - t0);
    state.browse = { modelId: res.model_id, ir: res.ir, meta: res.meta, layout: res.layout, profile: state.browse.profile, mapped: null };
    const extra = {};
    for (const n of res.ir.nodes) extra[n.id] = { op: n.op_type };
    browseGraph.setData(res.layout, extra);
    $("browse-export").disabled = !state.browse.profile;
    renderBrowseMeta(res.meta, res.ir, dt);
    setStatus($("browse-status"),
      `已加载 ${res.meta.node_count} 节点 / ${res.meta.edge_count} 边 (${dt}ms)`, "ok");
    $("sb-right").textContent = `${res.meta.name} · ${res.meta.node_count} 节点 · ${(res.meta.file_size_bytes / 1024).toFixed(1)} KB`;
    // profile loaded earlier for another model is invalidated
    if (state.browse.profile) {
      state.browse.mapped = null;
      setStatus($("browse-profile-status"), "模型已更换，请重新导入 profile", "");
    }
    refreshHeat();
  } catch (e) {
    setStatus($("browse-status"), `加载失败: ${e.message}`, "err");
    $("sb-right").textContent = "";
  }
}

function renderBrowseMeta(meta, ir, dt) {
  const warn = (ir.warnings || []).length
    ? `<p style="color:var(--ambiguous)">⚠ ${ir.warnings.length} 条警告: <code>${esc(ir.warnings.slice(0, 3).join(" | "))}</code></p>`
    : "";
  $("browse-meta").innerHTML =
    kvTable([
      ["名称", meta.name], ["producer", meta.producer || "—"],
      ["opset", JSON.stringify(meta.opsets)], ["节点/边", `${meta.node_count} / ${meta.edge_count}`],
      ["参数量", meta.param_count], ["文件大小", `${(meta.file_size_bytes / 1024).toFixed(1)} KB`],
      ["输入", meta.inputs.map((v) => `${v.name}${v.shape ? JSON.stringify(v.shape) : " (shape未知)"}`).join(", ")],
      ["输出", meta.outputs.map((v) => `${v.name}${v.shape ? JSON.stringify(v.shape) : " (shape未知)"}`).join(", ")],
      ["指纹(content)", meta.fingerprints.content ? meta.fingerprints.content.slice(0, 16) + "…" : "—"],
      ["指纹(topology)", meta.fingerprints.topology ? meta.fingerprints.topology.slice(0, 16) + "…" : "—"],
    ]) + warn;
}

function showBrowseDetail(nodeId) {
  const panel = $("browse-detail");
  if (!nodeId || !state.browse.ir) {
    panel.innerHTML = '<div class="panel"><h3>节点详情</h3><div class="meta">点击节点查看</div></div>';
    return;
  }
  const n = state.browse.ir.nodes.find((x) => x.id === nodeId);
  if (!n) { showBrowseDetail(null); return; }
  const perf = state.browse.mapped && state.browse.mapped.per_node[nodeId];
  const perfHtml = perf
    ? kvTable([["耗时 (us)", perf.duration_us.toFixed(1)], ["占比", (perf.share * 100).toFixed(2) + "%"], ["调用次数", perf.calls]])
    : (state.browse.mapped ? "<p class='muted'>该节点无映射 profile 记录</p>" : "");
  const attrs = Object.keys(n.attributes || {}).length
    ? kvTable(Object.entries(n.attributes)) : "<p class='muted'>无</p>";
  const params = (n.params || []).map((p) =>
    `<tr><td>${esc(p.name)}</td><td><code>${esc(p.dtype || "?")} ${esc(JSON.stringify(p.shape))}</code></td></tr>`).join("");
  const sg = (n.subgraphs || []).map((s) =>
    `<tr><td>${esc(s.attribute)}</td><td><code>${s.node_count} 节点 / ${s.edge_count} 边</code></td></tr>`).join("");
  panel.innerHTML = `<div class="panel">
    <h3>节点详情</h3>
    ${kvTable([
      ["名称", n.display_name], ["id", n.id], ["op", n.op_type],
      ["domain", n.domain || "ai.onnx"],
    ])}
    <h3 style="margin-top:10px">输入 tensors</h3>
    ${kvTable((n.inputs || []).filter(Boolean).map((t) => {
      const vi = n.input_info[t];
      return [t, vi ? `${vi.dtype || "?"} ${vi.shape ? JSON.stringify(vi.shape) : "shape未知"}` : "无shape信息"];
    }))}
    <h3 style="margin-top:10px">输出 tensors</h3>
    ${kvTable((n.outputs || []).filter(Boolean).map((t) => {
      const vi = n.output_info[t];
      return [t, vi ? `${vi.dtype || "?"} ${vi.shape ? JSON.stringify(vi.shape) : "shape未知"}` : "无shape信息"];
    }))}
    <h3 style="margin-top:10px">属性</h3>${attrs}
    <h3 style="margin-top:10px">参数 (initializers)</h3>
    ${params ? `<table>${params}</table>` : "<p class='muted'>无</p>"}
    ${sg ? `<h3 style="margin-top:10px">嵌套子图</h3><table>${sg}</table>` : ""}
    ${perfHtml ? `<h3 style="margin-top:10px">性能</h3>${perfHtml}` : ""}
  </div>`;
}

// ---- search ----
let searchHits = [];
let searchIdx = -1;
function runBrowseSearch(q) {
  searchHits = []; searchIdx = -1;
  if (!q || !state.browse.ir) {
    browseGraph.setSearchHighlight([]);
    $("browse-search-info").textContent = "";
    return;
  }
  const needle = q.toLowerCase();
  for (const n of state.browse.ir.nodes) {
    const hay = `${n.id} ${n.display_name} ${n.op_type} ${(n.inputs || []).join(" ")} ${(n.outputs || []).join(" ")}`.toLowerCase();
    if (hay.includes(needle)) searchHits.push(n.id);
  }
  browseGraph.setSearchHighlight(searchHits);
  $("browse-search-info").textContent = `${searchHits.length} 个匹配`;
  if (searchHits.length) cycleBrowseSearch();
}
function cycleBrowseSearch() {
  if (!searchHits.length) return;
  searchIdx = (searchIdx + 1) % searchHits.length;
  browseGraph.focusNode(searchHits[searchIdx]);
  browseGraph.selectExternal(searchHits[searchIdx]);
  showBrowseDetail(searchHits[searchIdx]);
}

// ---- profile / heatmap ----
async function loadBrowseProfile() {
  if (!state.browse.modelId) { toast("请先加载模型"); return; }
  const path = $("browse-profile-path").value.trim();
  setStatus($("browse-profile-status"), "导入中…");
  try {
    const res = await apiPost("/api/profile/load", { path, model_id: state.browse.modelId });
    state.browse.profile = res.profile_id;
    state.browse.mapped = res.mapped;
    const unmapped = res.mapped.unmapped.length;
    const fp = res.mapped.model_fingerprint_matched;
    let note = `已映射 ${Object.keys(res.mapped.per_node).length} 节点, 未映射 ${unmapped} 条`;
    if (fp === false) note += " · ⚠ 指纹不匹配";
    setStatus($("browse-profile-status"), note, fp === false ? "err" : "ok");
    $("browse-export").disabled = false;
    if (!res.mapped.warnings.length && !unmapped && fp !== false) {
      setStatus($("browse-profile-status"), note, "ok");
    }
    $("layer-heat").checked = true;
    refreshHeat();
  } catch (e) {
    setStatus($("browse-profile-status"), `导入失败: ${e.message}`, "err");
  }
}

function heatValues() {
  const mapped = state.browse.mapped;
  if (!mapped) return null;
  const mode = $("heat-mode").value;
  const vals = Object.values(mapped.per_node)
    .map((p) => (mode === "share" ? p.share : p.duration_us));
  if (!vals.length) return null;
  return { min: Math.min(...vals), max: Math.max(...vals) };
}

function refreshHeat() {
  if (state.mode === "compare") { refreshCompareStyles(); return; }
  if (state.mode !== "browse") { browseGraph.setStyles({}); return; }
  const mapped = state.browse.mapped;
  if (!heatLayerActive() || !mapped) {
    browseGraph.setStyles({});
    $("heat-min").textContent = "min"; $("heat-max").textContent = "max";
    return;
  }
  const mode = $("heat-mode").value;
  const { min, max } = heatValues();
  const styles = {};
  for (const [nid, p] of Object.entries(mapped.per_node)) {
    const v = mode === "share" ? p.share : p.duration_us;
    styles[nid] = { fill: heatColor(v, min, max) };
  }
  browseGraph.setStyles(styles);
  $("heat-min").textContent = mode === "share"
    ? `${(min * 100).toFixed(1)}%` : `${min.toFixed(0)}us`;
  $("heat-max").textContent = mode === "share"
    ? `${(max * 100).toFixed(1)}%` : `${max.toFixed(0)}us`;
}

// ===========================================================================
// COMPARE MODE
// ===========================================================================
const graphA = new GraphView($("cmp-graph-a"), { onSelect: (id) => showCompareDetail("a", id) });
const graphB = new GraphView($("cmp-graph-b"), { onSelect: (id) => showCompareDetail("b", id) });

// synchronized view
let syncing = false;
graphA.opts.onTransform = (g) => mirrorTransform(graphA, graphB);
graphB.opts.onTransform = (g) => mirrorTransform(graphB, graphA);
function mirrorTransform(src, dst) {
  if (!$("sync-view").checked || syncing) return;
  syncing = true;
  dst.tx = src.tx; dst.ty = src.ty; dst.scale = src.scale;
  dst.viewport.setAttribute("transform",
    `translate(${dst.tx} ${dst.ty}) scale(${dst.scale})`);
  dst._scheduleRender();
  syncing = false;
}
$("cmp-fit").addEventListener("click", () => { graphA.fitView(); graphB.fitView(); });
$("sync-view").addEventListener("change", refreshCompareStyles);
$("diff-only").addEventListener("change", refreshCompareStyles);

$("cmp-load").addEventListener("click", loadCompare);
["cmp-path-a", "cmp-path-b"].forEach((id) =>
  $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") loadCompare(); }));
$("cmp-profile-load").addEventListener("click", loadCompareProfiles);
$("cmp-search-a").addEventListener("keydown", (e) => { if (e.key === "Enter") cmpSearch("a"); });
$("cmp-search-b").addEventListener("keydown", (e) => { if (e.key === "Enter") cmpSearch("b"); });
$("diff-prev").addEventListener("click", () => navDiff(-1));
$("diff-next").addEventListener("click", () => navDiff(1));
$("cmp-export-json").addEventListener("click", () => exportDiff("json"));
$("cmp-export-csv").addEventListener("click", () => exportDiff("csv"));
$("cmp-export-md").addEventListener("click", () => exportDiff("md"));
$("cmp-export-html").addEventListener("click", () => exportDiff("html"));

async function loadCompare() {
  const pa = $("cmp-path-a").value.trim();
  const pb = $("cmp-path-b").value.trim();
  setStatus($("cmp-status"), "解析与匹配中…");
  try {
    const t0 = performance.now();
    const res = await apiPost("/api/diff", {
      path_a: pa, path_b: pb,
      min_group_size: parseInt($("cmp-min-size").value || "3", 10),
      min_confidence: parseFloat($("cmp-min-conf").value || "0.9"),
    });
    const dt = Math.round(performance.now() - t0);
    Object.assign(state.compare, {
      diffId: res.diff_id, diff: res.diff, match: res.match,
      folding: res.folding, layouts: res.layouts, folded: new Set(), navIdx: -1,
      aModelId: res.a_model_id, bModelId: res.b_model_id,
      mappedA: null, mappedB: null,
    });
    state.compare.irA = { nodes: indexNodes(res.diff, "a") };
    state.compare.irB = { nodes: indexNodes(res.diff, "b") };
    // store full IRs for details: fetch model IRs is overkill; diff carries displays
    renderCompareSummary(res, dt);
    renderFoldGroups();
    applyCompareViews();
    refreshCompareStyles();
    const btns = ["cmp-export-json", "cmp-export-csv", "cmp-export-md", "cmp-export-html"];
    btns.forEach((b) => { $(b).disabled = false; });
    setStatus($("cmp-status"),
      `匹配 ${res.match.summary.matched_count} · 歧义 ${res.match.summary.ambiguous_count} · A独有 ${res.match.summary.unmatched_a_count} · B独有 ${res.match.summary.unmatched_b_count} (${dt}ms)`, "ok");
    $("cmp-name-a").textContent = res.diff.a_name;
    $("cmp-name-b").textContent = res.diff.b_name;
  } catch (e) {
    setStatus($("cmp-status"), `对比失败: ${e.message}`, "err");
  }
}

function indexNodes(diff, side) {
  // Build a light node index from diff info (enough for search + details).
  const map = {};
  for (const nd of diff.node_diffs) {
    if (side === "a" && nd.a_node_id) {
      map[nd.a_node_id] = { id: nd.a_node_id, display: nd.a_display, diff: nd };
    } else if (side === "b" && nd.b_node_id) {
      map[nd.b_node_id] = { id: nd.b_node_id, display: nd.b_display, diff: nd };
    }
  }
  return map;
}

function diffByNode(side) {
  const m = {};
  for (const nd of state.compare.diff.node_diffs) {
    const id = side === "a" ? nd.a_node_id : nd.b_node_id;
    if (id) m[id] = nd;
  }
  return m;
}

function buildViewForSide(side) {
  // Folded view comes from the server; group node boxes are client-side bboxes.
  const layout = state.compare.layouts[side];
  let boxes = layout.boxes;
  let edgePoints = layout.edge_points;
  const folded = state.compare.folded;
  if (folded.size) {
    return apiPost("/api/fold", {
      diff_id: state.compare.diffId, folded_group_ids: [...folded],
    }).then((res) => {
      const view = res.views[side];
      const newBoxes = {};
      const groupBox = {};
      // keep pseudo io nodes (they are not part of folding)
      for (const nid in boxes) {
        const b = boxes[nid];
        if (b.kind === "input" || b.kind === "output") newBoxes[nid] = b;
      }
      for (const n of view.display_nodes) {
        if (n.kind === "node") {
          if (boxes[n.id]) newBoxes[n.id] = boxes[n.id];
        } else {
          // bbox over members
          const members = n.orig_node_ids.filter((m) => boxes[m]);
          if (!members.length) continue;
          let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
          for (const m of members) {
            const b = boxes[m];
            x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y);
            x1 = Math.max(x1, b.x + b.w); y1 = Math.max(y1, b.y + b.h);
          }
          groupBox[n.id] = { node_id: n.id, x: x0 - 14, y: y0 - 14,
            w: x1 - x0 + 28, h: y1 - y0 + 28, kind: "group", label: n.label };
        }
      }
      Object.assign(newBoxes, groupBox);
      // route edges: endpoints from boxes (pseudo io nodes kept)
      const newEdges = {};
      for (const e of view.display_edges) {
        const sb = newBoxes[e.src], db = newBoxes[e.dst];
        if (!sb || !db) continue;
        newEdges[e.id] = {
          src: [sb.x + sb.w, sb.y + sb.h / 2],
          dst: [db.x, db.y + db.h / 2],
        };
      }
      return { boxes: newBoxes, edge_points: newEdges, width: layout.width, height: layout.height };
    });
  }
  return Promise.resolve({ boxes, edge_points: edgePoints, width: layout.width, height: layout.height });
}

async function applyCompareViews() {
  const [la, lb] = await Promise.all([buildViewForSide("a"), buildViewForSide("b")]);
  const extraA = {}, extraB = {};
  for (const nd of state.compare.diff.node_diffs) {
    if (nd.a_node_id) extraA[nd.a_node_id] = { op: nd.a_display };
    if (nd.b_node_id) extraB[nd.b_node_id] = { op: nd.b_display };
  }
  graphA.setData(la, extraA);
  graphB.setData(lb, extraB);
  refreshCompareStyles();
}

function renderCompareSummary(res, dt) {
  const s = res.diff.summary;
  const rows = [
    ["相同", s.unchanged], ["新增(B)", s.added], ["删除(A)", s.removed],
    ["算子变化", s.op_changed], ["属性", s.attr_changed], ["shape", s.shape_changed],
    ["dtype", s.dtype_changed], ["参数", s.param_changed], ["连接", s.connection_changed],
    ["歧义", s.ambiguous], ["边增/删/变", `${(s.edge_added || 0)}/${(s.edge_removed || 0)}/${(s.edge_changed || 0)}`],
  ].filter((r) => r[1]);
  $("cmp-summary").innerHTML =
    `<table>${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td><b>${esc(v)}</b></td></tr>`).join("")}</table>
     <p class="muted">拓扑指纹${res.diff.topology_identical ? "一致" : "不一致"}</p>`;
  // diff navigation list
  const interesting = res.diff.node_diffs.filter((nd) => nd.kind !== "unchanged");
  const list = $("diff-list");
  if (!interesting.length) {
    list.innerHTML = "<p class='muted'>无结构差异 🎉</p>";
    return;
  }
  list.innerHTML = interesting.map((nd, i) => `
    <div class="diff-item" data-i="${i}">
      <span class="diff-badge badge-${nd.kind}">${DIFF_BADGE[nd.kind] ? DIFF_BADGE[nd.kind].letter : "M"}</span>
      <span>${esc(nd.a_display || nd.b_display)} <span class="muted">${esc(nd.explanation)}</span></span>
    </div>`).join("");
  list.querySelectorAll(".diff-item").forEach((el) =>
    el.addEventListener("click", () => gotoDiff(parseInt(el.dataset.i, 10))));
  state.compare.navList = interesting;
}

function gotoDiff(i) {
  const list = state.compare.navList || [];
  if (!list.length) return;
  state.compare.navIdx = ((i % list.length) + list.length) % list.length;
  const nd = list[state.compare.navIdx];
  $("diff-nav-pos").textContent = `${state.compare.navIdx + 1}/${list.length}`;
  document.querySelectorAll("#diff-list .diff-item").forEach((el) =>
    el.classList.toggle("sel", parseInt(el.dataset.i, 10) === state.compare.navIdx));
  if (nd.a_node_id) { graphA.focusNode(nd.a_node_id); graphA.selectExternal(nd.a_node_id); }
  if (nd.b_node_id) { graphB.focusNode(nd.b_node_id); graphB.selectExternal(nd.b_node_id); }
  showCompareDetail(nd.a_node_id ? "a" : "b", nd.a_node_id || nd.b_node_id);
}
function navDiff(dir) {
  gotoDiff((state.compare.navIdx === -1 ? (dir > 0 ? -1 : 0) : state.compare.navIdx) + dir);
}

function refreshCompareStyles() {
  if (state.mode !== "compare" || !state.compare.diff) return;
  const useDiff = diffLayerActive();
  const only = $("diff-only").checked;
  const useHeat = $("layer-heat").checked;
  const mode = $("heat-mode").value;
  const heatA = scaleHeat(state.compare.mappedA, mode);
  const heatB = scaleHeat(state.compare.mappedB, mode);
  const stylesA = {}, stylesB = {};
  for (const nd of state.compare.diff.node_diffs) {
    const badge = DIFF_BADGE[nd.kind];
    const style = useDiff && badge
      ? { cls: badge.cls, badge: badge.letter, badgeColor: badge.color }
      : {};
    if (useDiff && only && nd.kind === "unchanged") style.dim = true;
    if (nd.a_node_id) stylesA[nd.a_node_id] = style;
    if (nd.b_node_id) stylesB[nd.b_node_id] = { ...style };
  }
  // heat fill overlays on top of diff strokes (independent visual channels)
  if (useHeat) {
    for (const [nid, fill] of Object.entries(heatA)) {
      stylesA[nid] = { ...(stylesA[nid] || {}), fill };
    }
    for (const [nid, fill] of Object.entries(heatB)) {
      stylesB[nid] = { ...(stylesB[nid] || {}), fill };
    }
  }
  graphA.setStyles(stylesA);
  graphB.setStyles(stylesB);
  updateHeatLegend(heatA, heatB, mode);
}

function scaleHeat(mapped, mode) {
  if (!mapped) return {};
  const vals = Object.values(mapped.per_node)
    .map((p) => (mode === "share" ? p.share : p.duration_us));
  if (!vals.length) return {};
  const min = Math.min(...vals), max = Math.max(...vals);
  const out = {};
  for (const [nid, p] of Object.entries(mapped.per_node)) {
    const v = mode === "share" ? p.share : p.duration_us;
    out[nid] = heatColor(v, min, max);
  }
  return out;
}

function updateHeatLegend(heatA, heatB, mode) {
  const all = Object.values({ ...heatA, ...heatB });
  void all;
  const mapped = state.compare.mappedA || state.compare.mappedB;
  if (!mapped || !$("layer-heat").checked) {
    $("heat-min").textContent = "min"; $("heat-max").textContent = "max";
    return;
  }
  const vals = Object.values(mapped.per_node)
    .map((p) => (mode === "share" ? p.share : p.duration_us));
  if (!vals.length) return;
  $("heat-min").textContent = mode === "share"
    ? `${(Math.min(...vals) * 100).toFixed(1)}%` : `${Math.min(...vals).toFixed(0)}us`;
  $("heat-max").textContent = mode === "share"
    ? `${(Math.max(...vals) * 100).toFixed(1)}%` : `${Math.max(...vals).toFixed(0)}us`;
}

async function loadCompareProfiles() {
  if (!state.compare.diffId) { toast("请先执行对比"); return; }
  const pa = $("cmp-profile-a").value.trim();
  const pb = $("cmp-profile-b").value.trim();
  setStatus($("cmp-profile-status"), "导入中…");
  try {
    const resA = pa ? await apiPost("/api/profile/load", {
      path: pa, model_id: state.compare.aModelId }) : null;
    const resB = pb ? await apiPost("/api/profile/load", {
      path: pb, model_id: state.compare.bModelId }) : null;
    state.compare.mappedA = resA ? resA.mapped : null;
    state.compare.mappedB = resB ? resB.mapped : null;
    const notes = [];
    if (resA) notes.push(`A: ${Object.keys(resA.mapped.per_node).length} 映射/${resA.mapped.unmapped.length} 未映射`);
    if (resB) notes.push(`B: ${Object.keys(resB.mapped.per_node).length} 映射/${resB.mapped.unmapped.length} 未映射`);
    const fpBad = [resA, resB].some((r) => r && r.mapped.model_fingerprint_matched === false);
    setStatus($("cmp-profile-status"),
      notes.join(" · ") + (fpBad ? " · ⚠ 指纹不匹配" : ""), fpBad ? "err" : "ok");
    refreshCompareStyles();
  } catch (e) {
    setStatus($("cmp-profile-status"), `导入失败: ${e.message}`, "err");
  }
}

function cmpSearch(side) {
  const q = $(side === "a" ? "cmp-search-a" : "cmp-search-b").value.trim().toLowerCase();
  const graph = side === "a" ? graphA : graphB;
  const info = $(side === "a" ? "cmp-search-info-a" : "cmp-search-info-b");
  if (!q) { graph.setSearchHighlight([]); info.textContent = ""; return; }
  const hits = [];
  for (const id in graph.boxes) {
    const b = graph.boxes[id];
    const extra = graph.nodeExtra[id];
    if (`${id} ${b.label} ${extra ? extra.op : ""}`.toLowerCase().includes(q)) hits.push(id);
  }
  graph.setSearchHighlight(hits);
  info.textContent = `${hits.length} 个匹配`;
  if (hits.length) { graph.focusNode(hits[0]); graph.selectExternal(hits[0]); }
}

function showCompareDetail(side, nodeId) {
  const panel = side === "a" ? $("cmp-detail") : $("cmp-detail");
  const nd = diffByNode(side)[nodeId];
  if (!nd) {
    panel.innerHTML = '<div class="panel"><h3>节点详情</h3><div class="meta">点击节点查看</div></div>';
    return;
  }
  const other = side === "a" ? nd.b_node_id : nd.a_node_id;
  if (other) {
    const gOther = side === "a" ? graphB : graphA;
    gOther.focusNode(other, false);
    gOther.selectExternal(other);
  }
  const changes = (nd.changes || []).map((c) =>
    `<tr><td><code>${esc(c.field)}</code></td><td><code>${esc(JSON.stringify(c.before))}</code> → <code>${esc(JSON.stringify(c.after))}</code></td></tr>`).join("");
  const confNote = nd.confidence < 0.9 && nd.kind !== "unchanged"
    ? `<p style="color:var(--ambiguous)">⚠ 匹配置信度较低 (${nd.confidence})</p>` : "";
  const ambNote = nd.kind === "ambiguous"
    ? "<p style='color:var(--ambiguous)'>⚠ 存在多个等价候选，对应关系不确定</p>" : "";
  panel.innerHTML = `<div class="panel">
    <h3>节点详情 (${side.toUpperCase()} 侧)</h3>
    ${kvTable([
      ["显示名", side === "a" ? nd.a_display : nd.b_display],
      ["node_id", side === "a" ? nd.a_node_id : nd.b_node_id],
      ["对应节点", other || (nd.kind === "added" ? "B 侧独有" : "A 侧独有")],
      ["差异类型", nd.kind], ["匹配置信度", nd.confidence],
    ])}
    ${confNote}${ambNote}
    <p>${esc(nd.explanation)}</p>
    ${changes ? `<h3 style="margin-top:10px">字段级变化</h3><table>${changes}</table>` : ""}
    ${(nd.evidence || []).length ? `<h3 style="margin-top:10px">匹配依据</h3><p class="muted">${esc(nd.evidence.join("; "))}</p>` : ""}
  </div>`;
  // also highlight on the clicked side
  const gSelf = side === "a" ? graphA : graphB;
  gSelf.selectExternal(nodeId);
}

// ---- folding ----
function renderFoldGroups() {
  const wrap = $("fold-groups");
  const groups = state.compare.folding.groups;
  if (!groups.length) {
    wrap.innerHTML = "<p class='muted'>未发现可折叠公共子图（需 ≥ 高置信度匹配的连通区域）</p>";
    return;
  }
  wrap.innerHTML = groups.map((g) => `
    <label class="chk" style="align-items:flex-start">
      <input type="checkbox" data-gid="${esc(g.group_id)}">
      <span><b>${esc(g.label)}</b><br>
      <span class="muted">${esc(g.summary)}</span></span>
    </label>`).join("");
  wrap.querySelectorAll("input[type=checkbox]").forEach((cb) =>
    cb.addEventListener("change", () => toggleFold(cb.dataset.gid, cb.checked)));
}

async function toggleFold(gid, on) {
  if (on) state.compare.folded.add(gid);
  else state.compare.folded.delete(gid);
  if ($("sync-fold").checked) {
    await applyCompareViews();
  } else {
    // fold only the side being toggled is not representable (shared group ids);
    // folding is inherently synchronized in this version — documented.
    await applyCompareViews();
  }
}

// ---- export ----
async function exportDiff(fmt) {
  if (!state.compare.diffId) return;
  try {
    await apiDownload(`/api/export/diff`, { diff_id: state.compare.diffId, format: fmt },
      `diff_report.${fmt}`);
    toast(`已导出 diff_report.${fmt}`);
  } catch (e) {
    toast(`导出失败: ${e.message}`);
  }
}

// ===========================================================================
// PYTORCH MODE
// ===========================================================================
const ptGraph = new GraphView($("pt-graph"), { onSelect: (id) => showPtDetail(id) });

$("pt-load").addEventListener("click", loadPytorch);
$("pt-view-toggle").addEventListener("click", togglePtView);

function parseLeafRules(text) {
  return text.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    if (l.startsWith("class:")) return { class_name: l.slice(6).trim() };
    if (l.startsWith("path:")) return { path_pattern: l.slice(5).trim() };
    return { class_name: l };
  });
}

async function loadPytorch() {
  const project = $("pt-project").value.trim();
  const importPath = $("pt-import").value.trim();
  const className = $("pt-class").value.trim();
  setStatus($("pt-status"), "子进程加载中…");
  try {
    const res = await apiPost("/api/pytorch/load", {
      project_path: project, import_path: importPath, class_name: className,
      leaf_rules: parseLeafRules($("pt-leaves").value), trace: true,
    });
    state.pytorch.result = res;
    if (!res.ok) {
      setStatus($("pt-status"), `加载失败: ${res.error}\n${(res.stderr_tail || "").slice(-300)}`, "err");
      return;
    }
    let msg = `已加载模块树 (${countModules(res.module_tree)} 模块)`;
    if (res.degraded) {
      msg += " · ⚠ FX 追踪失败，仅显示模块树（未得到完整算子图）: " + res.fx_error;
      setStatus($("pt-status"), msg, "err");
    } else {
      msg += " · FX 算子图已捕获";
      setStatus($("pt-status"), msg, "ok");
    }
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

function renderPtTree() {
  const res = state.pytorch.result;
  if (!res || !res.ok) return;
  const flat = flattenClient(res.module_tree, 0);
  $("pt-tree").innerHTML = flat.map((n) => `
    <div class="pt-row" data-path="${esc(n.path)}">
      <span class="pt-depth">${"· ".repeat(n.depth)}</span>
      <span><b>${esc(n.class_name)}</b> <span class="muted">${esc(n.path || "(root)")}</span>
      ${n.is_leaf ? '<span class="pt-leaf">leaf</span>' : ""}
      ${n.param_count ? `<span class="muted">${n.param_count} params</span>` : ""}
      </span>
    </div>`).join("");
  $("pt-tree").querySelectorAll(".pt-row").forEach((el) =>
    el.addEventListener("click", () => {
      $("pt-tree").querySelectorAll(".pt-row").forEach((r) => r.classList.remove("sel"));
      el.classList.add("sel");
      showPtDetail(el.dataset.path);
      if (ptGraph.boxes[el.dataset.path]) {
        ptGraph.focusNode(el.dataset.path);
        ptGraph.selectExternal(el.dataset.path);
      }
    }));
}

function flattenClient(tree, depth) {
  const out = [{ path: tree.path, class_name: tree.class_name, depth,
                 param_count: tree.param_count, is_leaf: tree.is_leaf }];
  for (const c of tree.children || []) out.push(...flattenClient(c, depth + 1));
  return out;
}

function ptLayout() {
  const res = state.pytorch.result;
  if (!res || !res.ok) return null;
  const overview = state.pytorch.view === "overview";
  // Walk tree; in overview mode, leaf-marked subtrees collapse to one node.
  const boxes = {}; const extra = {};
  const edges = {};
  let y = 0;
  const NODE_W = 200, NODE_H = 44, XG = 70, YG = 22;
  function walk(node, depth, pathPrefix) {
    const path = node.path;
    const isCollapsed = overview && (node.is_leaf || (node.children || []).length === 0);
    boxes[path || "__root__"] = {
      node_id: path || "__root__", x: depth * (NODE_W + XG), y: y * (NODE_H + YG),
      w: NODE_W, h: NODE_H, kind: "node", label: node.class_name,
    };
    extra[path || "__root__"] = { op: `${node.param_count} params` };
    y += 1;
    if (isCollapsed) return;
    for (const c of node.children || []) {
      edges[`e:${path}->${c.path}`] = { src: path || "__root__", dst: c.path };
      walk(c, depth + 1);
    }
  }
  walk(res.module_tree, 0);
  // compute edge points
  const edge_points = {};
  for (const [id, e] of Object.entries(edges)) {
    const sb = boxes[e.src], db = boxes[e.dst];
    if (!sb || !db) continue;
    edge_points[id] = { src: [sb.x + sb.w, sb.y + sb.h / 2], dst: [db.x, db.y + db.h / 2] };
  }
  const maxDepth = Math.max(0, ...Object.values(boxes).map((b) => b.x));
  return { boxes, edge_points, width: maxDepth + NODE_W + 40, height: y * (NODE_H + YG) };
}

function renderPtGraph() {
  const layout = ptLayout();
  if (!layout) return;
  const extra = {};
  for (const id in layout.boxes) extra[id] = { op: layout.boxes[id].label };
  ptGraph.setData(layout, extra);
}

function togglePtView() {
  if (!state.pytorch.result) { toast("请先加载 PyTorch 项目"); return; }
  state.pytorch.view = state.pytorch.view === "overview" ? "detail" : "overview";
  $("pt-view-mode").textContent = state.pytorch.view === "overview"
    ? "概览（模块树）" : "详细（展开子模块）";
  renderPtGraph();
}

function showPtDetail(path) {
  const panel = $("pt-detail");
  const res = state.pytorch.result;
  if (!path || !res || !res.ok) {
    panel.innerHTML = '<div class="panel"><h3>模块详情</h3><div class="meta">点击模块查看</div></div>';
    return;
  }
  const flat = flattenClient(res.module_tree, 0);
  const n = flat.find((x) => x.path === path);
  if (!n) { showPtDetail(null); return; }
  panel.innerHTML = `<div class="panel"><h3>模块详情</h3>
    ${kvTable([
      ["类名", n.class_name], ["路径", n.path || "(root)"],
      ["参数量", n.param_count], ["leaf", n.is_leaf ? "是" : "否"],
    ])}
    ${res.fx_graph ? "<p class='muted'>FX 算子图已捕获，可在详细视图下查看对应 call_module 节点。</p>" : ""}
    ${res.fx_error ? `<p style="color:var(--ambiguous)">FX 诊断: <code>${esc(res.fx_error)}</code></p>` : ""}
  </div>`;
}

// ===========================================================================
// init
// ===========================================================================
(async function init() {
  setMode("browse");
  try {
    const h = await apiGet("/api/health");
    $("sb-right").textContent = h.torch_available
      ? "服务正常 · torch 可用"
      : "服务正常 · torch 不可用（PyTorch 层级功能受限）";
  } catch (e) {
    $("sb-right").textContent = "无法连接后端服务";
  }
  // auto-load the default fixture for instant feedback
  loadBrowseModel();
})();
