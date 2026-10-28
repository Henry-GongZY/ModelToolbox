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


def layout_single(ir: GraphIR) -> LayoutResult:
    res = LayoutResult()
    if not ir.nodes:
        return res

    ranks, preds, succs = _ranks(ir)

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
    x_by_rank = {r: r * (NODE_W + X_GAP) for r in sorted_ranks}
    index_in_rank: Dict[str, int] = {}
    counters: Dict[int, int] = {}
    for nid in positions_order:
        r = ranks.get(nid, pseudo_rank.get(nid, 0))
        i = counters.get(r, 0)
        index_in_rank[nid] = i
        counters[r] = i + 1

    max_rank = max(sorted_ranks) if sorted_ranks else 0
    width = (max_rank + 1) * (NODE_W + X_GAP) - X_GAP
    for r in sorted_ranks:
        layer_h = counters.get(r, 0) * (NODE_H + Y_GAP) - Y_GAP
        width = max(width, (NODE_W + X_GAP) * 3)
        # per-rank height handled below via global height
    height = max(counters.values()) * (NODE_H + Y_GAP) - Y_GAP

    for nid in positions_order:
        r = ranks.get(nid, pseudo_rank.get(nid, 0))
        x = x_by_rank[r]
        y = index_in_rank[nid] * (NODE_H + Y_GAP)
        res.boxes[nid] = NodeBox(node_id=nid, x=x, y=y, w=NODE_W, h=NODE_H,
                                 kind=kind_of(nid), label=label_of(nid))
        if nid in ranks:
            res.ranks[nid] = r

    # Width/height from actual box extents (pseudo io nodes may extend both
    # ends beyond node ranks).
    if res.boxes:
        res.width = max(b.x + b.w for b in res.boxes.values()) - min(b.x for b in res.boxes.values())
        res.height = max(b.y + b.h for b in res.boxes.values()) - min(b.y for b in res.boxes.values())
    else:
        res.width = 0
        res.height = 0

    # Edge anchor points
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
        for nid, box in lr.boxes.items():
            if box.kind in ("input", "output"):
                continue  # placed after real nodes, following their neighbour
            p = pair_by_node.get(nid)
            r = union_rank[side].get(nid, lr.ranks.get(nid, 0))
            if p:
                slot = pair_slots[(r, p[0])]
            else:
                slot = unmatched_slots.get(nid, 0)
            out.boxes[nid] = NodeBox(
                node_id=nid, x=r * (NODE_W + X_GAP), y=slot * (NODE_H + Y_GAP),
                w=NODE_W, h=NODE_H, kind=box.kind, label=box.label)
            if box.kind == "node":
                out.ranks[nid] = r

        # Pseudo io nodes follow the new y of the node they connect to, so
        # they stay visually attached on both sides.
        node_y = {nid: b.y for nid, b in out.boxes.items()}
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
            new_y = node_y.get(anchor)
            r = union_rank[side].get(nid, lr.ranks.get(nid, 0))
            if new_y is None:
                new_y = box.y
            out.boxes[nid] = NodeBox(
                node_id=nid, x=r * (NODE_W + X_GAP), y=new_y,
                w=NODE_W, h=NODE_H, kind=box.kind, label=box.label)
        out.width = (max_rank + 1) * (NODE_W + X_GAP)
        out.height = max([b.y for b in out.boxes.values()] + [0]) + NODE_H
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
                out.edge_points[e.id] = {
                    "src": [sb.x + sb.w, sb.y + sb.h / 2],
                    "dst": [db.x, db.y + db.h / 2],
                    "tensor": e.tensor,
                }
        return out

    la2 = rebuild(la, "a", a)
    lb2 = rebuild(lb, "b", b)

    # Hard guarantee: matched pairs share the same y.
    for mid, uid, vid in pairs:
        ba = la2.boxes.get(uid)
        bb = lb2.boxes.get(vid)
        if ba and bb:
            bb.y = ba.y
    return la2, lb2
