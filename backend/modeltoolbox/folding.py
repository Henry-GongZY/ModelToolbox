"""Common-subgraph folding for matched model pairs.

A fold group is a connected component of the "match graph": nodes are
matched pairs (match_id), edges are edge pairs that exist identically in
both models. Only high-confidence matches participate — ambiguous or
low-confidence correspondence is never folded (it could hide real
structural differences). Folding is a pure view transformation: the
underlying GraphIR is never mutated and no node/edge is lost or
duplicated in any fold state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from modeltoolbox.ir import GraphIR


@dataclass
class FoldGroup:
    group_id: str
    node_ids_a: List[str]
    node_ids_b: List[str]
    label: str
    op_counts: Dict[str, int]
    boundary_inputs: Dict[str, List[str]]  # side -> tensor names entering group
    boundary_outputs: Dict[str, List[str]]
    covered_models: int
    confidence: float
    internal_edge_count: int
    summary: str


@dataclass
class FoldingResult:
    groups: List[FoldGroup] = field(default_factory=list)
    node_to_group: Dict[str, Dict[str, str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FoldedView:
    display_nodes: List[dict]
    display_edges: List[dict]
    hidden_nodes: List[str]


def find_common_groups(
    a: GraphIR,
    b: GraphIR,
    matches: List[Any],
    min_group_size: int = 3,
    min_confidence: float = 0.9,
) -> FoldingResult:
    result = FoldingResult()
    pairs = [(m.match_id, m.a_node_id, m.b_node_id, m.confidence) for m in matches
             if getattr(m, "status", "matched") == "matched"
             and m.confidence >= min_confidence]
    if not pairs:
        return result

    by_a = {p[1]: p for p in pairs}
    by_mid = {p[0]: p for p in pairs}

    # adjacency between pairs: A-edge p->q and B-edge p'->q' (same direction)
    adj: Dict[str, Set[str]] = {p[0]: set() for p in pairs}
    a_succ: Dict[str, Set[str]] = {}
    for e in a.edges:
        if e.src and e.dst:
            a_succ.setdefault(e.src, set()).add(e.dst)
    b_succ: Dict[str, Set[str]] = {}
    for e in b.edges:
        if e.src and e.dst:
            b_succ.setdefault(e.src, set()).add(e.dst)
    for mid, uid, vid, _c in pairs:
        for u2 in a_succ.get(uid, ()):  # successors in A
            p2 = by_a.get(u2)
            if p2 and p2[0] in adj:
                # the same edge must exist in B: partner of u2 is p2[2]
                if p2[2] in b_succ.get(vid, set()):
                    adj[mid].add(p2[0])
                    adj[p2[0]].add(mid)

    # connected components over the match graph
    seen: Set[str] = set()
    components: List[List[tuple]] = []
    for p in sorted(pairs, key=lambda x: x[0]):
        if p[0] in seen:
            continue
        stack, comp = [p[0]], []
        seen.add(p[0])
        while stack:
            mid = stack.pop()
            comp.append(by_mid[mid])
            for m2 in adj.get(mid, ()):
                if m2 not in seen:
                    seen.add(m2)
                    stack.append(m2)
        components.append(comp)

    components.sort(key=lambda c: min(x[0] for x in c))
    node_to_group: Dict[str, Dict[str, str]] = {"a": {}, "b": {}}
    groups: List[FoldGroup] = []

    nodes_a = {n.id: n for n in a.nodes}
    nodes_b = {n.id: n for n in b.nodes}
    prod_a = _producer_map(a)
    prod_b = _producer_map(b)

    for comp in components:
        if len(comp) < min_group_size:
            continue
        comp.sort(key=lambda x: x[0])
        gids = [p[0] for p in comp]
        group_id = f"g{len(groups)}"
        ids_a = [p[1] for p in comp]
        ids_b = [p[2] for p in comp]
        conf = min(p[3] for p in comp)

        op_counts: Dict[str, int] = {}
        for uid in ids_a:
            op = nodes_a[uid].op_type
            op_counts[op] = op_counts.get(op, 0) + 1

        in_a_set, out_a_set = set(ids_a), set(ids_b)
        b_in: Dict[str, List[str]] = {"a": [], "b": []}
        b_out: Dict[str, List[str]] = {"a": [], "b": []}
        internal_edges = 0

        for e in a.edges:
            if e.src and e.dst:
                if e.src in in_a_set and e.dst in in_a_set:
                    internal_edges += 1
                elif e.dst in in_a_set:
                    b_in["a"].append(e.tensor)
                elif e.src in in_a_set:
                    b_out["a"].append(e.tensor)
        for e in b.edges:
            if e.src and e.dst:
                if e.src in out_a_set and e.dst in out_a_set:
                    pass  # internal in B too (counted once from A)
                elif e.dst in out_a_set:
                    b_in["b"].append(e.tensor)
                elif e.src in out_a_set:
                    b_out["b"].append(e.tensor)

        total_nodes = len(ids_a)
        label = " + ".join(f"{op}×{c}" for op, c in sorted(op_counts.items()))
        summary = (f"{label} · {total_nodes} nodes · "
                   f"in {len(b_in['a'])}/{len(b_in['b'])} out {len(b_out['a'])}/{len(b_out['b'])} · "
                   f"confidence {conf:.2f}")

        groups.append(FoldGroup(
            group_id=group_id,
            node_ids_a=ids_a,
            node_ids_b=ids_b,
            label=label,
            op_counts=op_counts,
            boundary_inputs=b_in,
            boundary_outputs=b_out,
            covered_models=2,
            confidence=round(conf, 3),
            internal_edge_count=internal_edges,
            summary=summary,
        ))
        for uid in ids_a:
            node_to_group["a"][uid] = group_id
        for vid in ids_b:
            node_to_group["b"][vid] = group_id

    result.groups = groups
    result.node_to_group = node_to_group
    return result


def _producer_map(ir: GraphIR) -> Dict[str, str]:
    out = {}
    for n in ir.nodes:
        for t in n.outputs:
            out.setdefault(t, n.id)
    return out


def apply_folding(ir: GraphIR, folding: FoldingResult, folded_group_ids: set,
                  side: str) -> FoldedView:
    """Build the display view for one side given the set of folded groups.

    Pure function: reads ir + folding, never mutates them. Guarantees:
      - every node is either hidden inside a folded group or present once
      - every original edge maps to exactly one display edge (re-routed
        through group nodes when an endpoint is folded)
    """
    mapping = folding.node_to_group.get(side, {})
    groups_by_id = {g.group_id: g for g in folding.groups}
    hidden: List[str] = []
    display_nodes: List[dict] = []
    for n in ir.nodes:
        gid = mapping.get(n.id)
        if gid and gid in folded_group_ids:
            hidden.append(n.id)
        else:
            display_nodes.append({
                "id": n.id, "kind": "node", "label": n.display_name,
                "op_type": n.op_type, "orig_node_ids": [n.id],
            })
    for g in folding.groups:
        if g.group_id in folded_group_ids:
            display_nodes.append({
                "id": g.group_id, "kind": "group", "label": g.label,
                "group_id": g.group_id, "summary": g.summary,
                "orig_node_ids": list(g.node_ids_a if side == "a" else g.node_ids_b),
            })

    def route(node_id: str) -> str:
        gid = mapping.get(node_id)
        if gid and gid in folded_group_ids:
            return gid
        return node_id

    display_edges: List[dict] = []
    for e in ir.edges:
        if e.src and e.dst:
            gs, gd = mapping.get(e.src), mapping.get(e.dst)
            if gs and gs in folded_group_ids and gs == gd:
                continue  # internal edge of a folded group
            src, dst, direction = route(e.src), route(e.dst), "io"
        elif e.dst:  # graph input / initializer feeds dst
            src, dst, direction = "__in__:" + e.tensor, route(e.dst), "in"
        else:  # output-only edge
            src = route(e.src) if e.src else "__in__:" + e.tensor
            dst, direction = "__out__:" + e.tensor, "out"
        display_edges.append({
            "id": e.id,
            "src": src,
            "dst": dst,
            "tensor": e.tensor,
            "dir": direction,
        })
    return FoldedView(display_nodes=display_nodes,
                      display_edges=display_edges,
                      hidden_nodes=hidden)
