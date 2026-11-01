// ModelToolbox frontend — the vendored official Netron viewer is the only
// graph renderer (browse = one instance, compare = two side-by-side
// instances). PyTorch module hierarchies render as a native HTML tree.
// Canvas-first interaction: welcome card, ⌘K palette, context menus,
// floating layer panel, status bar.
import { apiPost, apiGet, apiDownload } from "./api.js";
import { initBridge, netronApplyHeat, netronApplyDiffTint, netronFocusId,
         netronFocusByName, netronHideChrome, netronWireFollow,
         netronSetConsistentOpacity } from "./mtbx-bridge.js";

const $ = (id) => document.getElementById(id);

const state = {
  mode: "browse", // browse | compare | pytorch
  browse: { modelId: null, ir: null, meta: null, profileId: null, mapped: null },
  compare: {
    diffId: null, diff: null, match: null, folding: null,
    aModelId: null, bModelId: null, nodesA: {}, nodesB: {},
    navIdx: -1, mappedA: null, mappedB: null, nameA: "", nameB: "",
    useNetronCompare: true, diffMode: true,
    consistentOpacity: 0.25,
  },
  pytorch: { result: null, view: "overview", selected: null },
  pendingCompare: { a: null, b: null },
  layerPanelClosed: false,
  navList: [],
};

// ---------- init the Netron bridge with app dependencies ----------
const DIFF_TINT = {
  added: "#1f6f43", removed: "#8f2f2f", op_changed: "#8f4a2f",
  attr_changed: "#7d5a2f", shape_changed: "#7d2f5a", dtype_changed: "#5a2f7d",
  param_changed: "#2f5a7d", connection_changed: "#4a4a2f", ambiguous: "#8a6d1f",
};

initBridge({
  $,
  state,
  getLayer,
  DIFF_TINT,
  toast,
  setStatusText,
  findPartner,
});

// ---------- tiny helpers ----------
function toast(msg, ms = 2600) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), ms);
}
function setStatusText(msg, cls = "") {
  $("sb-left").textContent = msg;
  $("sb-left").style.color = cls === "err" ? "var(--removed)" : cls === "ok" ? "var(--added)" : "";
}
function setStatus(el, msg, cls = "") {
  el.textContent = msg || "";
  el.className = "status " + cls;
}
function esc(v) {
  return String(v == null ? "" : v)
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
function openProps() { $("props").classList.remove("hidden"); }

const DIFF_BADGE = {
  added: "A", removed: "D", op_changed: "M", attr_changed: "M",
  shape_changed: "S", dtype_changed: "T", param_changed: "P",
  connection_changed: "C", ambiguous: "?",
};
const HEAT_COLORS = ["#31567e", "#2f7fb5", "#2f9e6e", "#c9a227", "#c94f43"];
function heatColor(v, min, max) {
  if (v == null || max == null || max <= min) return HEAT_COLORS[2];
  const t = Math.min(Math.max((v - min) / (max - min || 1), 0), 1);
  return HEAT_COLORS[Math.min(Math.floor(t * 5), 4)];
}
function fmtShape(vi) {
  if (!vi) return "";
  const shp = vi.shape === null ? "?" : JSON.stringify(vi.shape).replace(/null/g, "?");
  return `${vi.dtype || "?"} ${shp}`;
}

// ---------- mode & welcome ----------
function hasContent() {
  return state.mode === "compare" ? !!state.compare.diffId
    : state.mode === "pytorch" ? !!(state.pytorch.result && state.pytorch.result.ok)
    : !!state.browse.modelId;
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
  $("sb-right").textContent =
    mode === "compare" ? `对比 · A ${state.compare.nameA} ⇄ B ${state.compare.nameB}`
    : mode === "pytorch" ? "PyTorch 层级"
    : state.browse.meta ? `${state.browse.meta.name} · ${state.browse.meta.node_count} 节点`
    : "";
  refreshWelcome();
  refreshLayerPanel();
  refreshLayers();
}

// ---------- diff helpers ----------
function diffByNode(side) {
  const m = {};
  if (!state.compare.diff) return m;
  for (const nd of state.compare.diff.node_diffs) {
    const id = side === "a" ? nd.a_node_id : nd.b_node_id;
    if (id) m[id] = nd;
  }
  return m;
}
function findPartner(side, nodeId) {
  const match = state.compare.match;
  if (!match) return null;
  for (const pair of match.matches) {
    if (side === "a" && pair.a_node_id === nodeId) return pair.b_node_id;
    if (side === "b" && pair.b_node_id === nodeId) return pair.a_node_id;
  }
  return null;
}

// ---------- model loading (browse = official Netron viewer) ----------
async function openModelFile(file) {
  if (!file) return;
  setStatusText(`加载 ${file.name}…`);
  try {
    const b64 = await fileToBase64(file);
    const t0 = performance.now();
    const res = await apiPost("/api/model/upload", { name: file.name, data_base64: b64 });
    const ms = Math.round(performance.now() - t0);
    state.browse = { modelId: res.model_id, ir: res.ir, meta: res.meta, profileId: null, mapped: null };
    setMode("browse");
    setStatusText(`${res.meta.name} — ${res.meta.node_count} 节点 / ${res.meta.edge_count} 边 (${ms}ms)`);
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
$("w-compare").addEventListener("click", openCompareDialog);

async function runCompare() {
  const fa = state.pendingCompare.a;
  const fb = state.pendingCompare.b;
  if (!fa || !fb) { setStatus($("cmp-status"), "请先选择两个模型文件", "err"); return; }
  setStatus($("cmp-status"), "解析与匹配中…");
  try {
    const t0 = performance.now();
    const b64a = await fileToBase64(fa);
    const b64b = await fileToBase64(fb);
    const res = await apiPost("/api/diff", {
      name_a: fa.name, data_a: b64a, name_b: fb.name, data_b: b64b,
      min_group_size: parseInt($("cmp-min-size").value || "3", 10),
      min_confidence: parseFloat($("cmp-min-conf").value || "0.9"),
    });
    const ms = Math.round(performance.now() - t0);
    $("dlg-compare").classList.add("hidden");
    applyDiffResult(res, ms, fa.name, fb.name);
  } catch (e) {
    setStatus($("cmp-status"), `对比失败: ${e.message}`, "err");
  }
}

function applyDiffResult(res, ms, nameA, nameB) {
  Object.assign(state.compare, {
    diffId: res.diff_id, diff: res.diff, match: res.match,
    folding: res.folding,
    aModelId: res.a_model_id, bModelId: res.b_model_id,
    nodesA: res.nodes_a || {}, nodesB: res.nodes_b || {},
    mappedA: null, mappedB: null, nameA, nameB,
    navIdx: -1, useNetronCompare: true, diffMode: true,
  });
  state.detailCache = { a: {}, b: {} };
  $("cmp-head-a").innerHTML = `A: <span>${esc(nameA)}</span>`;
  $("cmp-head-b").innerHTML = `B: <span>${esc(nameB)}</span>`;
  setMode("compare");
  applyCompareNetronViews();
  renderDiffSummary();
  const sum = res.match.summary;
  setStatusText(`匹配 ${sum.matched_count} · 歧义 ${sum.ambiguous_count} · A独有 ${sum.unmatched_a_count} · B独有 ${sum.unmatched_b_count} (${ms}ms)`);
}

// ---------- compare Netron views ----------
function applyCompareNetronViews() {
  const cmp = state.compare;
  if (!cmp.aModelId || !cmp.bModelId) return;
  const fa = $("netron-frame-a"), fb = $("netron-frame-b");
  const urlA = "/netron/?model=" + cmp.aModelId;
  const urlB = "/netron/?model=" + cmp.bModelId;
  if (!fa.src.endsWith(urlA)) fa.src = urlA;
  if (!fb.src.endsWith(urlB)) fb.src = urlB;
  fa.classList.remove("hidden"); fb.classList.remove("hidden");
  netronHideChrome("netron-frame-a");
  netronHideChrome("netron-frame-b");
  netronWireFollow("netron-frame-a", "a");
  netronWireFollow("netron-frame-b", "b");
  netronApplyDiffTint();
  netronSetConsistentOpacity(cmp.consistentOpacity);
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
    const b64a = await fileToBase64(fa);
    const b64b = await fileToBase64(fb);
    const res = await apiPost("/api/diff", {
      name_a: fa.name, data_a: b64a, name_b: fb.name, data_b: b64b,
      min_group_size: parseInt($("cmp-min-size").value || "3", 10),
      min_confidence: parseFloat($("cmp-min-conf").value || "0.9"),
    });
    const ms = Math.round(performance.now() - t0);
    $("dlg-compare").classList.add("hidden");
    applyDiffResult(res, ms, fa.name, fb.name);
  } catch (e) {
    setStatus($("cmp-status"), `对比失败: ${e.message}`, "err");
  }
}

// ---------- diff UI ----------
function renderDiffSummary() {
  if (!state.compare.diffMode) { $("props").classList.add("hidden"); return; }
  const d = state.compare.diff;
  const sum = d.summary;
  const rows = [["相同", sum.unchanged], ["新增(B)", sum.added], ["删除(A)", sum.removed],
    ["算子", sum.op_changed], ["属性", sum.attr_changed], ["shape", sum.shape_changed],
    ["dtype", sum.dtype_changed], ["参数", sum.param_changed], ["连接", sum.connection_changed],
    ["歧义", sum.ambiguous]].filter((r) => r[1]);
  const interesting = d.node_diffs.filter((nd) => nd.kind !== "unchanged");
  state.navList = interesting;
  openProps();
  $("props-title").textContent = "差异摘要";
  $("props-body").innerHTML =
    kvTable(rows) + `<p>${d.topology_identical ? "拓扑指纹一致" : "拓扑指纹不一致"}</p>` +
    (interesting.length
      ? "<h4>差异列表</h4><table>" + interesting.map((nd, i) =>
        `<tr style="cursor:pointer" data-i="${i}"><td>${DIFF_BADGE[nd.kind] || "M"} ${esc(nd.kind)}</td><td>${esc(nd.a_display || nd.b_display)}</td></tr>`).join("") + "</table>"
      : "<p>无结构差异 🎉</p>");
  $("props-body").querySelectorAll("tr[data-i]").forEach((el) =>
    el.addEventListener("click", () => gotoDiff(parseInt(el.dataset.i, 10))));
  $("diff-nav-pos").textContent = `0/${interesting.length}`;
}

function gotoDiff(i) {
  const list = state.navList || [];
  if (!list.length) return;
  state.compare.navIdx = ((i % list.length) + list.length) % list.length;
  const nd = list[state.compare.navIdx];
  $("diff-nav-pos").textContent = `${state.compare.navIdx + 1}/${list.length}`;
  if (nd.a_node_id) netronFocusId("netron-frame-a", nd.a_node_id, true);
  if (nd.b_node_id) netronFocusId("netron-frame-b", nd.b_node_id, true);
}
$("diff-prev").addEventListener("click", () => gotoDiff((state.compare.navIdx === -1 ? 0 : state.compare.navIdx) - 1));
$("diff-next").addEventListener("click", () => gotoDiff(state.compare.navIdx + 1));

// ---------- layers & heat ----------
function getLayer(id) {
  const el = $(id);
  return el ? el.checked : false;
}
function setLayer(id, checked) {
  const el = $(id);
  if (el) el.checked = checked;
}

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
  const heatMode = getLayer("heat-mode") || "duration";
  if (state.mode === "browse") {
    if (!state.browse.ir || !state.browse.modelId) { netronApplyHeat({}); return; }
    const heat = getLayer("layer-heat") ? heatStyles(state.browse.mapped, heatMode) : {};
    netronApplyHeat(heat);
  } else if (state.mode === "compare") {
    netronApplyDiffTint();
  }
}

// ---------- floating layer panel ----------
function refreshLayerPanel() {
  const panel = $("layer-panel");
  if (hasContent() && !state.layerPanelClosed) {
    const mode = state.mode;
    const rows = [];
    if (mode === "browse" && state.browse.modelId) {
      rows.push(["layer-heat", "性能热力图", getLayer("layer-heat")]);
    }
    if (mode === "compare") {
      rows.push(["layer-diff-mode", "Diff 模式", state.compare.diffMode]);
      if (state.compare.diffMode) {
        rows.push(["layer-diff", "Diff 染色", getLayer("layer-diff")]);
        rows.push(["slider-consistent", "一致部分透明度", state.compare.consistentOpacity]);
        rows.push(["sync-view", "点击跟随对侧", getLayer("sync-view") !== false]);
      }
    }
    const heatOn = getLayer("layer-heat");
    $("lp-toggles").innerHTML = rows.map(([id, label, checked]) => {
      if (id === "slider-consistent") {
        return `<label class="lp-toggle">一致部分透明度
          <input type="range" id="${id}" min="5" max="100" step="5"
                 value="${Math.round((checked || 0.25) * 100)}" style="width:110px;margin-left:auto">
          <span id="slider-consistent-val">${Math.round((checked || 0.25) * 100)}%</span></label>`;
      }
      return `<label class="lp-toggle"><input type="checkbox" id="${id}" ${checked ? "checked" : ""}> ${esc(label)}</label>`;
    }).join("") +
      (heatOn ? `<label class="lp-toggle" style="color:var(--muted)">着色依据
        <select id="heat-mode" class="menu-select" style="margin-left:auto">
          <option value="duration" selected>耗时</option>
          <option value="share">占比</option>
        </select></label>` : "");
    for (const [id] of rows) {
      const el = $(id);
      if (!el) continue;
      el.addEventListener("change", () => {
        if (id === "layer-diff-mode") {
          state.compare.diffMode = el.checked;
          refreshLayers();
          refreshLayerPanel();
          if (el.checked) { renderDiffSummary(); }
          else { $("props").classList.add("hidden"); }
          return;
        }
        if (id === "layer-heat" || id === "layer-diff") refreshLayers();
      });
      el.addEventListener("input", () => {
        if (id === "slider-consistent") {
          const v = parseInt(el.value, 10) / 100;
          const val = $("slider-consistent-val");
          if (val) val.textContent = el.value + "%";
          netronSetConsistentOpacity(v);
        }
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
$("w-open").addEventListener("click", () => $("file-open").click());
$("w-pytorch").addEventListener("click", () => setMode("pytorch"));

// ---------- context menu ----------
function closeCtxMenu() { $("ctx-menu").classList.add("hidden"); }
function openCtxMenu(x, y, items) {
  const menu = $("ctx-menu");
  menu.innerHTML = items.map((it) => {
    if (it === "-") return `<div class="ctx-sep"></div>`;
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
    el.addEventListener("click", () => {
      const act = el.dataset.action;
      closeCtxMenu();
      runAction(act);
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
  openCtxMenu(e.clientX, e.clientY, canvasMenuItems());
});

function canvasMenuItems() {
  const items = [
    { action: "open", icon: "📄", label: "打开模型…", hint: "O" },
    { action: "compare", icon: "⇆", label: "对比两个模型…", hint: "D" },
    { action: "pytorch", icon: "🐍", label: "PyTorch 项目…" },
    "-",
    { action: "props-model", icon: "ⓘ", label: "模型属性", disabled: !state.browse.meta },
    { action: "load-profile", icon: "⏱", label: "加载 Profile…", disabled: !state.browse.modelId },
    "-",
    { action: "panel-layers", icon: "▤", label: "图层面板" },
    { action: "palette", icon: "⌘", label: "命令面板", hint: "⌘K" },
  ];
  if (state.mode === "compare" && state.compare.diffId) {
    items.push("-");
    for (const f of ["json", "csv", "md", "html"]) {
      items.push({ action: `export:${f}`, icon: "↧", label: `导出 Diff ${f.toUpperCase()}` });
    }
  }
  if (state.mode === "browse" && state.browse.profileId) {
    items.push({ action: "export:pjson", icon: "↧", label: "导出 Profile JSON" });
    items.push({ action: "export:pcsv", icon: "↧", label: "导出 Profile CSV" });
  }
  return items;
}

function pytorchMenuItems() {
  return [
    { action: "pt-toggle", icon: "▦", label: "概览 / 详细切换" },
    { action: "pt-back", icon: "←", label: "返回模型画布" },
  ];
}

function runAction(action) {
  switch (action) {
    case "open": $("file-open").click(); break;
    case "compare": openCompareDialog(); break;
    case "pytorch": setMode("pytorch"); break;
    case "props-model": showModelProps(); break;
    case "load-profile": $("file-profile").click(); break;
    case "panel-layers":
      state.layerPanelClosed = false;
      refreshLayerPanel();
      break;
    case "palette": openPalette(); break;
    case "pt-toggle": {
      if (!state.pytorch.result) { toast("请先加载 PyTorch 项目"); break; }
      state.pytorch.view = state.pytorch.view === "overview" ? "detail" : "overview";
      renderPtMain();
      break;
    }
    case "pt-back": setMode("browse"); break;
    default:
      if (action.startsWith("export:")) exportAs(action.slice(7));
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
        netronFocusByName(n.display_name || n.id);
      }});
    }
  } else if (state.mode === "compare") {
    for (const [id, n] of Object.entries(state.compare.nodesA)) {
      if (`${id} ${n.display} ${n.op}`.toLowerCase().includes(q))
        out.push({ icon: "🅐", label: `${n.display || id}`, sub: `${n.op} · A 侧`, run: () => {
          netronFocusId("netron-frame-a", id);
        }});
    }
    for (const [id, n] of Object.entries(state.compare.nodesB)) {
      if (`${id} ${n.display} ${n.op}`.toLowerCase().includes(q))
        out.push({ icon: "🅑", label: `${n.display || id}`, sub: `${n.op} · B 侧`, run: () => {
          netronFocusId("netron-frame-b", id);
        }});
    }
  } else if (state.mode === "pytorch" && state.pytorch.result && state.pytorch.result.ok) {
    for (const n of flattenClient(state.pytorch.result.module_tree, 0)) {
      if (`${n.path} ${n.class_name}`.toLowerCase().includes(q))
        out.push({ icon: "▪", label: `${n.class_name}`, sub: n.path || "(root)", run: () => {
          showPtDetail(n.path);
        }});
    }
  }
  return out.slice(0, 40);
}

function openPalette() {
  $("palette").classList.remove("hidden");
  $("palette-input").value = "";
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
    renderPtMain();
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

function renderPtMain() {
  const res = state.pytorch.result;
  if (!res || !res.ok) { $("pt-main").innerHTML = ""; return; }
  const overview = state.pytorch.view === "overview";
  const html = [];
  const walk = (node, depth) => {
    const collapsed = overview && (node.is_leaf || !(node.children || []).length);
    html.push(`<div class="pt-row" data-path="${esc(node.path)}" style="padding-left:${depth * 18}px">
      <span><b>${esc(node.class_name)}</b> <span style="color:var(--muted)">${esc(node.path || "(root)")}</span>
      ${node.is_leaf ? '<span class="pt-leaf">leaf</span>' : ""}
      ${node.param_count ? `<span style="color:var(--muted)">${node.param_count.toLocaleString()} params</span>` : ""}</span>
    </div>`);
    if (collapsed) return;
    for (const c of node.children || []) walk(c, depth + 1);
  };
  walk(res.module_tree, 0);
  $("pt-main").innerHTML = html.join("");
  $("pt-main").querySelectorAll(".pt-row").forEach((el) =>
    el.addEventListener("click", () => {
      $("pt-main").querySelectorAll(".pt-row").forEach((r) => r.classList.remove("sel"));
      el.classList.add("sel");
      state.pytorch.selected = el.dataset.path;
      showPtDetail(el.dataset.path);
    }));
}

function showPtDetail(path) {
  const res = state.pytorch.result;
  if (!path || !res || !res.ok) { $("props").classList.add("hidden"); return; }
  const flat = flattenClient(res.module_tree, 0);
  const n = flat.find((x) => x.path === path);
  if (!n) return;
  openProps();
  $("props-title").textContent = "模块属性";
  $("props-body").innerHTML = kvTable([
    ["类名", n.class_name], ["路径", n.path || "(root)"],
    ["参数量", n.param_count], ["参数字节", n.param_bytes || 0],
    ["leaf", n.is_leaf ? "是" : "否"],
  ]) + (res.fx_error ? `<p style="color:var(--ambiguous)">FX 诊断: <code>${esc(res.fx_error)}</code></p>` : "");
}

// ---------- keyboard ----------
window.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    openPalette();
    return;
  }
  if (e.key === "Escape") {
    closeCtxMenu();
    $("dlg-compare").classList.add("hidden");
    closePalette();
    return;
  }
  const inField = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
  if (inField || e.metaKey || e.ctrlKey) return;
  if (e.key === "o" || e.key === "O") { e.preventDefault(); $("file-open").click(); }
  else if (e.key === "d" || e.key === "D") { e.preventDefault(); openCompareDialog(); }
});

// ---------- debug/test hook ----------
window.__mtbx = { state, applyCompareNetronViews, netronApplyDiffTint,
  netronFocusId, refreshLayers, refreshLayerPanel };

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
