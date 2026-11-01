"""Layered graph layout (Sugiyama-lite) + aligned dual-model layout.

Deterministic: identical input -> identical output. Pure stdlib.
Node ids in LayoutResult include pseudo io nodes "__in__:<tensor>" /
"__out__:<tensor>" so graph IO is visible in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from modeltoolbox.ir import GraphIR

NODE_W = 176.0
NODE_H = 48.0
X_GAP = 96.0
Y_GAP = 32.0


@dataclass
class NodeBox:
    node_id: str
    x: float
    y: float
    w: float
    h: float
    kind: str  # "node" | "input" | "output"
    label: str


@dataclass
class LayoutResult:
    boxes: Dict[str, NodeBox] = field(default_factory=dict)
    width: float = 0.0
    height: float = 0.0
    ranks: Dict[str, int] = field(default_factory=dict)
    edge_points: Dict[str, dict] = field(default_factory=dict)
    orientation: str = "horizontal"  # "horizontal" (rank→x) | "vertical" (rank→y)
    x_origin: float = 0.0  # left channel reserved for skip-edge arcs (vertical)


def pseudo_in(tensor: str) -> str:
    return f"__in__:{tensor}"


def pseudo_out(tensor: str) -> str:
    return f"__out__:{tensor}"


def _ranks(ir: GraphIR) -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, List[str]]]:
    """Longest-path ranking over real nodes (cycles capped at V iterations)."""
    preds: Dict[str, List[str]] = {n.id: [] for n in ir.nodes}
    succs: Dict[str, List[str]] = {n.id: [] for n in ir.nodes}
    for e in ir.edges:
        if e.src and e.dst:
            succs[e.src].append(e.dst)
            preds[e.dst].append(e.src)

    ranks: Dict[str, int] = {}
    order = [n.id for n in ir.nodes]
    for _ in range(len(order) + 1):
        changed = False
        for nid in order:
            r = 0 if not preds[nid] else 1 + max(ranks.get(p, -1) for p in preds[nid])
            # During iteration a pred may be unranked (cycle); treat as -1.
            if preds[nid] and any(p not in ranks for p in preds[nid]):
                # not ready this pass unless all preds ranked
                if all(p in ranks for p in preds[nid]):
                    r = 1 + max(ranks[p] for p in preds[nid])
                else:
                    r = ranks.get(nid, 0)
            if ranks.get(nid) != r:
                ranks[nid] = r
                changed = True
        if not changed:
            break
    return ranks, preds, succs


def _sink_feeders(ir: GraphIR, ranks: Dict[str, int], preds, succs) -> None:
    """Sink pure-feeder nodes toward their consumers (minimize edge span).

    Longest-path ranking places every node as EARLY as possible, which
    strands short side-chains (e.g. the Constant→Concat that builds a
    Resize's scales) dozens of ranks away from the deep node that consumes
    them — the whole band then renders as one meaningless row. Sinking each
    node to ``min(successor ranks) - 1`` is feasibility-preserving (edges
    stay >= 1) and only moves nodes whose consumers are deeper; spine nodes
    have ceiling == own rank and never move. Iterated to a fixed point.
    """
    order = [n.id for n in ir.nodes]
    changed = True
    while changed:
        changed = False
        for nid in order:
            ss = succs.get(nid)
            if not ss:
                continue
            ceiling = min(ranks[s] for s in ss if s in ranks) - 1
            floor = 0
            if preds.get(nid):
                ranked = [ranks[p] for p in preds[nid] if p in ranks]
                if ranked:
                    floor = max(ranked) + 1
            if ceiling > ranks.get(nid, 0) and ceiling >= floor:
                ranks[nid] = ceiling
                changed = True


def layout_single(ir: GraphIR) -> LayoutResult:
    res = LayoutResult()
    if not ir.nodes:
        return res

    ranks, preds, succs = _ranks(ir)
    _sink_feeders(ir, ranks, preds, succs)

    graph_in_tensors = [v.name for v in ir.inputs]
    graph_out_tensors = [v.name for v in ir.outputs]

    # Pseudo io nodes participate in ordering:
    #  - an input pseudo node sits just left of its (single) consumer's rank
    #  - an output pseudo node sits just right of its producer's rank
    consumers: Dict[str, List[str]] = {}
    producers: Dict[str, List[str]] = {}
    for e in ir.edges:
        if e.src == "" and e.dst:
            consumers.setdefault(e.tensor, []).append(e.dst)
        if e.dst == "" and e.src:
            producers.setdefault(e.tensor, []).append(e.src)

    pseudo_rank: Dict[str, int] = {}
    for t in graph_in_tensors:
        cs = consumers.get(t, [])
        if cs and all(c in ranks for c in cs):
            pseudo_rank[pseudo_in(t)] = min(ranks[c] for c in cs) - 1
        else:
            pseudo_rank[pseudo_in(t)] = -1
    for t in graph_out_tensors:
        ps = producers.get(t, [])
        if ps and all(p in ranks for p in ps):
            pseudo_rank[pseudo_out(t)] = max(ranks[p] for p in ps) + 1
        else:
            pseudo_rank[pseudo_out(t)] = (max(ranks.values()) + 1) if ranks else 0

    # Layers: rank -> ordered ids
    layers: Dict[int, List[str]] = {}
    for nid, r in ranks.items():
        layers.setdefault(r, []).append(nid)
    for pid, r in pseudo_rank.items():
        layers.setdefault(r, []).append(pid)

    node_boxes = {n.id: n for n in ir.nodes}

    def label_of(nid: str) -> str:
        if nid.startswith("__in__:"):
            return nid[len("__in__:"):]
        if nid.startswith("__out__:"):
            return nid[len("__out__:"):]
        n = node_boxes.get(nid)
        return n.display_name if n else nid

    def kind_of(nid: str) -> str:
        if nid.startswith("__in__:"):
            return "input"
        if nid.startswith("__out__:"):
            return "output"
        return "node"

    # Barycenter ordering: iterate passes using neighbor order.
    def build_neighbor_key(pos_of):
        def neighbor_key(nid: str) -> Tuple[float, float, str]:
            neigh = preds.get(nid, []) + succs.get(nid, [])
            # graph-input/output pseudo neighbours
            for t in graph_in_tensors:
                if nid in consumers.get(t, []):
                    neigh.append((pseudo_in(t), t))
            for t in graph_out_tensors:
                if nid in producers.get(t, []):
                    neigh.append((pseudo_out(t), t))
            vals = [pos_of.get(x[0] if isinstance(x, tuple) else x, 0) for x in neigh]
            bary = sum(vals) / len(vals) if vals else 0.0
            return (bary, 0.0, nid)
        return neighbor_key

    sorted_ranks = sorted(layers)
    # initial order: by rank, then id
    positions_order: List[str] = []
    for r in sorted_ranks:
        positions_order.extend(sorted(layers[r]))
    layer_sets = {r: set(layers[r]) for r in sorted_ranks}

    for _ in range(4):
        pos_of = {nid: i for i, nid in enumerate(positions_order)}
        key_fn = build_neighbor_key(pos_of)
        new_order: List[str] = []
        for r in sorted_ranks:
            layer = [x for x in positions_order if x in layer_sets[r]]
            layer.sort(key=key_fn)
            new_order.extend(layer)
        positions_order = new_order

    # Assign coordinates
    index_in_rank: Dict[str, int] = {}
    counters: Dict[int, int] = {}
    for nid in positions_order:
        r = ranks.get(nid, pseudo_rank.get(nid, 0))
        i = counters.get(r, 0)
        index_in_rank[nid] = i
        counters[r] = i + 1

    max_rank = max(sorted_ranks) if sorted_ranks else 0

    # ---- orientation auto-pick -------------------------------------------
    # A deep, narrow graph (chain-like, e.g. UNet) reads top-to-bottom like
    # Netron; a wide, branchy graph reads left-to-right. Compare the space
    # each orientation would need; prefer vertical when horizontal is much
    # wider than tall.
    max_layer_size = max(counters.values()) if counters else 1
    horiz_span = (max_rank + 1) * (NODE_W + X_GAP)
    vert_span = max_layer_size * (NODE_W + X_GAP)
    horiz_h = max_layer_size * (NODE_H + Y_GAP)
    vertical = horiz_span > vert_span * 1.35 or (horiz_span > 5000 and horiz_h < 1000)

    if vertical:
        # Spine-centering: pick the through-path (one node per rank, chained)
        # and re-map its layer-internal slot to the horizontal middle, with
        # remaining nodes filling the freed slots in their existing relative
        # order. This yields Netron's "left skip channel / center spine /
        # right side-chains" structure instead of a left-hugging column.
        succ_sets: Dict[str, set] = {}
        for nid in preds:
            for p in preds[nid]:
                succ_sets.setdefault(p, set()).add(nid)
        spine_ids: set = set()
        spine_prev = None
        for r in sorted_ranks:
            layer = [n for n in layers[r] if kind_of(n) == "node"]
            if not layer:
                continue
            pick = None
            if spine_prev is not None:
                cands = [n for n in layer if n in succ_sets.get(spine_prev, set())]
                if cands:
                    pick = cands[0]
            if pick is None:
                pick = layer[0]
            spine_ids.add(pick)
            spine_prev = pick

        new_index: Dict[str, int] = {}
        # Give each connected non-spine side chain a stable global column.
        # The old per-rank slots put a side-chain back on x=0 whenever its
        # rank contained only that one node, which made deep residual graphs
        # look like a single column.  Components are computed after removing
        # the spine, so branches sharing a join still get separate channels.
        non_spine = {n for n in ranks if n not in spine_ids}
        side_adj = {n: set() for n in non_spine}
        for nid in non_spine:
            for other in preds.get(nid, []) + succs.get(nid, []):
                if other in non_spine:
                    side_adj[nid].add(other)
        components = []
        remaining = set(non_spine)
        while remaining:
            seed = min(remaining, key=lambda n: (ranks.get(n, 0), n))
            stack, comp = [seed], set()
            while stack:
                cur = stack.pop()
                if cur not in remaining:
                    continue
                remaining.remove(cur); comp.add(cur)
                stack.extend(side_adj[cur] & remaining)
            components.append(comp)
        # Reuse a side lane for components that occupy different vertical
        # ranges. Giving every component a permanent column makes deep models
        # unnecessarily wide even when their branches live far apart.
        components.sort(key=lambda c: (
            min(ranks.get(n, 0) for n in c),
            min((index_in_rank.get(n, 0), n) for n in c),
        ))
        lane_ends: List[int] = []
        side_col: Dict[str, int] = {}
        for comp in components:
            first_rank = min(ranks.get(n, 0) for n in comp)
            last_rank = max(ranks.get(n, 0) for n in comp)
            lane = next((i for i, end in enumerate(lane_ends) if end < first_rank), None)
            if lane is None:
                lane = len(lane_ends)
                lane_ends.append(last_rank)
            else:
                lane_ends[lane] = last_rank
            for nid in comp:
                side_col[nid] = 1 + lane
        for r in sorted_ranks:
            layer = [n for n in layers[r] if kind_of(n) == "node"]
            same_col = {}
            for nid in layer:
                if nid in spine_ids:
                    new_index[nid] = 0
                else:
                    base = side_col.get(nid, 1)
                    offset = same_col.get(base, 0)
                    new_index[nid] = base + offset
                    same_col[base] = offset + 1
            # pseudo io nodes keep their own placement (they follow anchors)
        index_in_rank = {**new_index,
                         **{n: index_in_rank[n] for n in index_in_rank
                            if kind_of(n) != "node"}}

        # pseudo io nodes follow their anchor node's column
        anchor_of: Dict[str, str] = {}
        for e in ir.edges:
            if e.src == "" and e.dst:
                anchor_of.setdefault(pseudo_in(e.tensor), e.dst)
            if e.dst == "" and e.src:
                anchor_of.setdefault(pseudo_out(e.tensor), e.src)

        X_ORIGIN = 320.0  # left channel: long skip edges arc through here
        res.x_origin = X_ORIGIN
        for nid in positions_order:
            r = ranks.get(nid, pseudo_rank.get(nid, 0))
            if kind_of(nid) in ("input", "output"):
                continue
            y = r * (NODE_H + Y_GAP) + (NODE_H + Y_GAP)  # shift down for input pseudos at rank -1
            x = X_ORIGIN + index_in_rank[nid] * (NODE_W + X_GAP)
            res.boxes[nid] = NodeBox(node_id=nid, x=x, y=y, w=NODE_W, h=NODE_H,
                                     kind=kind_of(nid), label=label_of(nid))
            if nid in ranks:
                res.ranks[nid] = r
        for nid in positions_order:
            if kind_of(nid) not in ("input", "output"):
                continue
            r = pseudo_rank.get(nid, 0)
            a = anchor_of.get(nid)
            x = res.boxes[a].x if a and a in res.boxes else X_ORIGIN
            y = r * (NODE_H + Y_GAP) + (NODE_H + Y_GAP)
            res.boxes[nid] = NodeBox(node_id=nid, x=x, y=y, w=NODE_W, h=NODE_H,
                                     kind=kind_of(nid), label=label_of(nid))
        res.orientation = "vertical"
    else:
        for nid in positions_order:
            r = ranks.get(nid, pseudo_rank.get(nid, 0))
            x = r * (NODE_W + X_GAP)
            y = index_in_rank[nid] * (NODE_H + Y_GAP)
            res.boxes[nid] = NodeBox(node_id=nid, x=x, y=y, w=NODE_W, h=NODE_H,
                                     kind=kind_of(nid), label=label_of(nid))
            if nid in ranks:
                res.ranks[nid] = r
        res.orientation = "horizontal"

    # Width/height from actual box extents (pseudo io nodes may extend both
    # ends beyond node ranks).
    if res.boxes:
        res.width = max(b.x + b.w for b in res.boxes.values()) - min(b.x for b in res.boxes.values())
        res.height = max(b.y + b.h for b in res.boxes.values()) - min(b.y for b in res.boxes.values())
    else:
        res.width = 0
        res.height = 0

    # Edge anchor points: horizontal → right/left edge midpoints;
    # vertical → bottom/top edge midpoints.
    for e in ir.edges:
        src_id = e.src if e.src else (
            pseudo_in(e.tensor) if pseudo_in(e.tensor) in res.boxes else None)
        dst_id = e.dst if e.dst else (
            pseudo_out(e.tensor) if pseudo_out(e.tensor) in res.boxes else None)
        if src_id is None:
            src_id = e.dst  # dangling: draw a stub from the consumer box
        if dst_id is None:
            dst_id = e.src
        sb = res.boxes.get(src_id)
        db = res.boxes.get(dst_id)
        if not sb or not db:
            continue
        if res.orientation == "vertical":
            res.edge_points[e.id] = {
                "src": [sb.x + sb.w / 2, sb.y + sb.h],
                "dst": [db.x + db.w / 2, db.y],
                "tensor": e.tensor,
            }
        else:
            res.edge_points[e.id] = {
                "src": [sb.x + sb.w, sb.y + sb.h / 2],
                "dst": [db.x, db.y + db.h / 2],
                "tensor": e.tensor,
            }
    return res


# ---- aligned dual layout ----------------------------------------------------


def layout_diff_pair(a: GraphIR, b: GraphIR, matches: List[Any]) -> Tuple[LayoutResult, LayoutResult]:
    """Two layouts sharing y positions for matched pairs.

    ``matches`` items need .match_id/.a_node_id/.b_node_id/.status and are
    filtered to status == "matched" (duck-typed, so tests can pass simple
    stand-ins).
    """
    la = layout_single(a)
    lb = layout_single(b)

    pairs = [(m.match_id, m.a_node_id, m.b_node_id) for m in matches
             if getattr(m, "status", "matched") == "matched"]

    # union ranks: matched pairs share max(rank_a, rank_b)
    union_rank: Dict[str, Dict[str, int]] = {"a": {}, "b": {}}
    for nid, r in la.ranks.items():
        union_rank["a"][nid] = r
    for nid, r in lb.ranks.items():
        union_rank["b"][nid] = r
    for mid, uid, vid in pairs:
        ra = la.ranks.get(uid, 0)
        rb = lb.ranks.get(vid, 0)
        r = max(ra, rb)
        union_rank["a"][uid] = r
        union_rank["b"][vid] = r

    def rebuild(lr: LayoutResult, side: str, ir: GraphIR) -> LayoutResult:
        if not lr.boxes:
            return lr
        out = LayoutResult()
        pair_by_node = {}
        for mid, uid, vid in pairs:
            pair_by_node[uid if side == "a" else vid] = (mid, uid, vid)
        # group ids per union rank, preserving within-rank order of lr
        by_rank: Dict[int, List[str]] = {}
        for nid in lr.boxes:
            r = union_rank[side].get(nid)
            if r is None:
                r = lr.ranks.get(nid, 0)
            by_rank.setdefault(r, []).append(nid)
        # matched ids get stable slots: order matched pairs by (rank, original y)
        slot_counters: Dict[int, int] = {}
        pair_slots: Dict[Tuple[int, str], int] = {}
        for nid in sorted(lr.boxes, key=lambda x: (union_rank[side].get(x, 0),
                                                   lr.boxes[x].y)):
            p = pair_by_node.get(nid)
            if not p:
                continue
            r = union_rank[side].get(nid, 0)
            slot = slot_counters.get(r, 0)
            pair_slots[(r, p[0])] = slot
            slot_counters[r] = slot + 1

        # unmatched nodes fill slots after matched ones in the same rank
        unmatched_slots: Dict[str, int] = {}
        matched_here_count: Dict[int, int] = {}
        for (r, _m), s in pair_slots.items():
            matched_here_count[r] = matched_here_count.get(r, 0) + 1
        for r in by_rank:
            base = matched_here_count.get(r, 0)
            i = 0
            for nid in by_rank[r]:
                if nid not in pair_by_node:
                    unmatched_slots[nid] = base + i
                    i += 1

        max_rank = max(by_rank) if by_rank else 0
        vertical = lr.orientation == "vertical"
        out.orientation = lr.orientation
        # Keep the same left skip channel as layout_single().  Without this,
        # a vertically oriented compare view silently moves its boxes back to
        # x=40 and long residual edges cross the graph instead of using the
        # Netron-style channel on the left.
        out.x_origin = lr.x_origin if vertical else 0.0
        STEP_W = NODE_H + Y_GAP if vertical else NODE_W + X_GAP
        STEP_S = NODE_W + X_GAP if vertical else NODE_H + Y_GAP
        slot_origin = out.x_origin if vertical else 0.0
        for nid, box in lr.boxes.items():
            if box.kind in ("input", "output"):
                continue  # placed after real nodes, following their neighbour
            p = pair_by_node.get(nid)
            r = union_rank[side].get(nid, lr.ranks.get(nid, 0))
            if p:
                slot = pair_slots[(r, p[0])]
            else:
                slot = unmatched_slots.get(nid, 0)
            if vertical:
                # rank runs downward, slots run rightward
                out.boxes[nid] = NodeBox(
                    node_id=nid, x=slot * STEP_S + slot_origin, y=(r + 1) * STEP_W,
                    w=NODE_W, h=NODE_H, kind=box.kind, label=box.label)
            else:
                out.boxes[nid] = NodeBox(
                    node_id=nid, x=r * STEP_W, y=slot * STEP_S,
                    w=NODE_W, h=NODE_H, kind=box.kind, label=box.label)
            if box.kind == "node":
                out.ranks[nid] = r

        # Pseudo io nodes follow the new y of the node they connect to, so
        # they stay visually attached on both sides.
        node_y = {nid: b.y for nid, b in out.boxes.items()}
        node_x = {nid: b.x for nid, b in out.boxes.items()}
        for nid, box in lr.boxes.items():
            if box.kind not in ("input", "output"):
                continue
            tensor = nid.split(":", 1)[-1]
            if box.kind == "input":
                anchor = next((e.dst for e in ir.edges
                               if e.src == "" and e.tensor == tensor and e.dst), None)
            else:
                anchor = next((e.src for e in ir.edges
                               if e.dst == "" and e.tensor == tensor and e.src), None)
            r = union_rank[side].get(nid, lr.ranks.get(nid, 0))
            if vertical:
                new_y = (r + 1) * STEP_W
                new_x = node_x.get(anchor, 40)
                if new_x is None:
                    new_x = box.x
                out.boxes[nid] = NodeBox(
                    node_id=nid, x=new_x, y=new_y,
                    w=NODE_W, h=NODE_H, kind=box.kind, label=box.label)
            else:
                new_x = r * STEP_W
                new_y = node_y.get(anchor)
                if new_y is None:
                    new_y = box.y
                out.boxes[nid] = NodeBox(
                    node_id=nid, x=new_x, y=new_y,
                    w=NODE_W, h=NODE_H, kind=box.kind, label=box.label)
        if out.boxes:
            out.width = max(b.x + b.w for b in out.boxes.values()) - min(b.x for b in out.boxes.values())
            out.height = max(b.y + b.h for b in out.boxes.values()) - min(b.y for b in out.boxes.values())
        # recompute edge points on new geometry
        for e in ir.edges:
            src_id = e.src if e.src else pseudo_in(e.tensor)
            dst_id = e.dst if e.dst else pseudo_out(e.tensor)
            if src_id not in out.boxes:
                src_id = e.src or e.dst
            if dst_id not in out.boxes:
                dst_id = e.dst or e.src
            sb = out.boxes.get(src_id)
            db = out.boxes.get(dst_id)
            if sb and db:
                if out.orientation == "vertical":
                    out.edge_points[e.id] = {
                        "src": [sb.x + sb.w / 2, sb.y + sb.h],
                        "dst": [db.x + db.w / 2, db.y],
                        "tensor": e.tensor,
                    }
                else:
                    out.edge_points[e.id] = {
                        "src": [sb.x + sb.w, sb.y + sb.h / 2],
                        "dst": [db.x, db.y + db.h / 2],
                        "tensor": e.tensor,
                    }
        return out

    la2 = rebuild(la, "a", a)
    lb2 = rebuild(lb, "b", b)

    # Hard guarantee: matched pairs share the same row (horizontal) or the
    # same column (vertical).
    for mid, uid, vid in pairs:
        ba = la2.boxes.get(uid)
        bb = lb2.boxes.get(vid)
        if ba and bb:
            if la2.orientation == "vertical":
                bb.x = ba.x
            else:
                bb.y = ba.y
    return la2, lb2
