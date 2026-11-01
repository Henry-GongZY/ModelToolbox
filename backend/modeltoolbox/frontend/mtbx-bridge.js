// Bridge between ModelToolbox and the vendored official Netron viewer.
// All functions talk to a same-origin Netron iframe through its internal
// `window.__view__` API. Init with initBridge({ $, state, getLayer,
// DIFF_TINT, toast, setStatusText, findPartner }).

let deps = {
    $: (id) => document.getElementById(id),
    state: null,
    getLayer: () => false,
    DIFF_TINT: {},
    toast: () => {},
    setStatusText: () => {},
    findPartner: () => null,
};

export function initBridge(d) {
    deps = Object.assign(deps, d);
}

const READY_POLL_MS = 400;
const READY_TRIES = 240; // dual viewers parse slowly; give them a long window

function frameReady(frameId, cb) {
    const f = deps.$(frameId);
    const ready = () => {
        let w = null;
        try {
            w = f.contentWindow;
            if (w && w.__view__ && w.__view__._target &&
                w.__view__._target._table && w.__view__._target._table.size > 0) {
                cb(w);
                return true;
            }
        } catch (e) { /* frame still loading */ }
        return false;
    };
    if (ready()) return;
    let tries = 0;
    const t = setInterval(() => {
        tries += 1;
        if (ready() || tries > READY_TRIES) clearInterval(t);
    }, READY_POLL_MS);
}

function frameOrWindowError(frameId) {
    try {
        return deps.$(frameId).contentWindow.__mtbxBridgeErr || null;
    } catch (e) { return null; }
}

export function netronHideChrome(frameId) {
    frameReady(frameId, (w) => {
        const doc = deps.$(frameId).contentDocument;
        let st = doc.getElementById("mtbx-style");
        if (!st) {
            st = doc.createElement("style");
            st.id = "mtbx-style";
            doc.head.appendChild(st);
        }
        st.textContent =
            ".toolbar, #menu-button, #titlebar { display: none !important; }\n" +
            ":root { --mtbx-consistent-opacity: " + deps.state.compare.consistentOpacity + "; }\n" +
            "g.mtbx-consistent { opacity: var(--mtbx-consistent-opacity); }\n" +
            "path.mtbx-consistent-edge { opacity: var(--mtbx-consistent-opacity); }\n" +
            "g.node { transition: opacity .25s; }\n" +
            "path { transition: opacity .25s; }";
    });
}

export function netronSetConsistentOpacity(v) {
    deps.state.compare.consistentOpacity = v;
    for (const frameId of ["netron-frame-a", "netron-frame-b", "netron-frame"]) {
        let f = null;
        try { f = deps.$(frameId); } catch (e) { continue; }
        if (!f) continue;
        try {
            const de = f.contentDocument && f.contentDocument.documentElement;
            if (de) de.style.setProperty("--mtbx-consistent-opacity", String(v));
        } catch (e) { /* frame not ready */ }
    }
}

// ---- browse-mode heatmap ------------------------------------------------

export function netronApplyHeat(heatMap, attempt) {
    frameReady("netron-frame", (w) => {
        const doc = deps.$("netron-frame").contentDocument;
        doc.querySelectorAll("path.mtbx-heat").forEach((p) => {
            p.classList.remove("mtbx-heat");
            p.style.fill = "";
        });
        let hit = 0;
        let built = 0;
        const target = w.__view__._target;
        // the table is keyed by the ONNX node OBJECT; match by its .name and
        // tint every row of the node (Netron rows are g.node-item > path)
        for (const [modelNode, gnode] of target._table.entries()) {
            const name = modelNode && modelNode.name;
            if (!name) continue;
            if (gnode && gnode.element) built += 1;
            const color = heatMap[name];
            if (color && gnode && gnode.element) {
                const paths = gnode.element.querySelectorAll("g.node-item path");
                if (paths.length) {
                    for (const p of paths) { p.style.fill = color; p.classList.add("mtbx-heat"); }
                    hit += 1;
                }
            }
        }
        // Netron builds node DOM after the model table exists — retry while
        // no node element has been built yet.
        if (built === 0 && (attempt || 0) < 40) {
            setTimeout(() => netronApplyHeat(heatMap, (attempt || 0) + 1), 400);
            return;
        }
        if (Object.keys(heatMap).length) {
            deps.setStatusText(
                hit ? `热力图已应用：${hit}/${Object.keys(heatMap).length} 节点着色（Netron 渲染）`
                    : "热力图：Netron 中未匹配到节点",
                hit ? "ok" : "err");
        }
    });
}

// ---- compare-mode diff view ----------------------------------------------

// view: { consistentNodes:Set, nodeKinds:{name:kind}, edgeKinds:{tensor:{kind}} }
export function netronApplyDiffView(frameId, side, view, attempt) {
    frameReady(frameId, (w) => {
        const doc = deps.$(frameId).contentDocument;
        doc.querySelectorAll("path.mtbx-heat").forEach((p) => {
            p.classList.remove("mtbx-heat");
            p.style.fill = "";
        });
        doc.querySelectorAll("g.mtbx-consistent").forEach((g) => g.classList.remove("mtbx-consistent"));
        doc.querySelectorAll("path.mtbx-consistent-edge").forEach((p) => {
            p.classList.remove("mtbx-consistent-edge");
            p.style.stroke = "";
        });
        netronHideChrome(frameId);

        const target = w.__view__._target;
        let built = 0;
        for (const [modelNode, gnode] of target._table.entries()) {
            const name = modelNode && modelNode.name;
            if (!name) continue;
            if (gnode && gnode.element) {
                const paths = gnode.element.querySelectorAll("g.node-item path");
                if (paths.length) built += 1;
                if (view.consistentNodes.has(name)) {
                    gnode.element.classList.add("mtbx-consistent");
                } else {
                    const kind = view.nodeKinds[name];
                    if (kind && deps.DIFF_TINT[kind]) {
                        for (const p of paths) {
                            p.style.fill = deps.DIFF_TINT[kind];
                            p.classList.add("mtbx-heat");
                        }
                    }
                }
            }
        }
        // edges: Netron stamps each edge path with id "edge-<tensor name>"
        for (const [tensor, info] of Object.entries(view.edgeKinds)) {
            const el = doc.getElementById("edge-" + tensor);
            if (!el) continue;
            if (info.kind === "unchanged") {
                el.classList.add("mtbx-consistent-edge");
            } else if (deps.DIFF_TINT[info.kind]) {
                el.style.stroke = deps.DIFF_TINT[info.kind];
            }
        }
        // Netron builds node DOM after the model table exists — retry until
        // rendered, otherwise the view silently lands nowhere.
        if (built === 0 && (attempt || 0) < 40) {
            setTimeout(() => netronApplyDiffView(frameId, side, view, (attempt || 0) + 1), 400);
            return;
        }
    });
}

export function netronFocusId(frameId, nodeName, silent) {
    frameReady(frameId, (w) => {
        const t = w.__view__._target;
        for (const [modelNode, gnode] of t._table.entries()) {
            if (modelNode && modelNode.name === nodeName && gnode) {
                t.scrollTo(t.select([modelNode], null));
                return;
            }
        }
        if (!silent) deps.toast("该节点在另一侧无对应（或为歧义）");
    });
}

let followLock = false;

export function netronWireFollow(frameId, side) {
    const f = deps.$(frameId);
    frameReady(frameId, () => {
        if (f.__followWired) return;
        f.__followWired = true;
        const doc = f.contentDocument;
        doc.addEventListener("click", (ev) => {
            const sync = deps.$("sync-view");
            if (!sync || !sync.checked) return;
            if (deps.state.mode !== "compare" || !deps.state.compare.useNetronCompare) return;
            if (followLock) return;
            const g = ev.target && ev.target.closest && ev.target.closest("g.node");
            if (!g) return;
            const name = g.getAttribute("id");
            if (!name) return;
            const partner = deps.findPartner(side, name);
            if (!partner) return;
            followLock = true;
            const other = side === "a" ? "netron-frame-b" : "netron-frame-a";
            netronFocusId(other, partner, true);
            setTimeout(() => { followLock = false; }, 400);
        }, true);
    });
}

// Compare-mode tint entry: builds node/edge views from the DiffIR stored in
// deps.state.compare.diff and applies them to both viewer frames.
export function netronApplyDiffTint() {
    const cmp = deps.state.compare;
    if (!cmp.useNetronCompare || !cmp.diff) return;
    const tintOn = deps.getLayer("layer-diff");
    const consistentA = new Set(), consistentB = new Set();
    const nodeKindsA = {}, nodeKindsB = {};
    const edgeKindsA = {}, edgeKindsB = {};
    for (const nd of cmp.diff.node_diffs) {
        if (nd.kind === "unchanged") {
            if (nd.a_node_id) consistentA.add(nd.a_node_id);
            if (nd.b_node_id) consistentB.add(nd.b_node_id);
            continue;
        }
        if (tintOn) {
            if (nd.a_node_id) nodeKindsA[nd.a_node_id] = nd.kind;
            if (nd.b_node_id) nodeKindsB[nd.b_node_id] = nd.kind;
        }
    }
    for (const ed of cmp.diff.edge_diffs) {
        const aE = ed.a_edge, bE = ed.b_edge;
        if (ed.kind === "unchanged") {
            if (bE && bE.tensor) edgeKindsB[bE.tensor] = { kind: "unchanged" };
            if (aE && aE.tensor) edgeKindsA[aE.tensor] = { kind: "unchanged" };
            continue;
        }
        if (tintOn) {
            if (bE && bE.tensor) edgeKindsB[bE.tensor] = { kind: ed.kind };
            if (aE && aE.tensor) edgeKindsA[aE.tensor] = { kind: ed.kind };
        }
    }
    netronApplyDiffView("netron-frame-a", "a",
        { consistentNodes: consistentA, nodeKinds: nodeKindsA, edgeKinds: edgeKindsA });
    netronApplyDiffView("netron-frame-b", "b",
        { consistentNodes: consistentB, nodeKinds: nodeKindsB, edgeKinds: edgeKindsB });
}

export function netronFocusByName(nodeName) {
    frameReady("netron-frame", (w) => {
        const t = w.__view__._target;
        for (const [modelNode, gnode] of t._table.entries()) {
            if (modelNode && modelNode.name === nodeName && gnode) {
                t.scrollTo(t.select([modelNode], 'sidebar'));
                return;
            }
        }
        deps.toast("该节点在 Netron 中无名称锚点（未命名节点）");
    });
}

export function bridgeError(frameId) {
    return frameOrWindowError(frameId);
}
