"""Cross-model node matching: multi-stage, explainable, ambiguity-aware.

Names are ONE piece of evidence, never the sole basis (see PRD P0-4).
Pipeline:
  1. Anchors  — io correspondence, unique op signatures, unique equal names,
                unique tensor names.
  2. Propagation — iterated mutual-best pairing scored by op/domain equality,
                attribute similarity, matched-neighbor consistency (topology),
                io-degree and shape/dtype compatibility.
  3. Ambiguity — score ties (symmetric branches, repeated operators) are NOT
                force-resolved; they are reported as ambiguous with candidates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from modeltoolbox.ir import GraphIR, NodeIR

MATCH_SCORE_THRESHOLD = 1.0
MATCH_MARGIN = 0.5
TIE_EPS = 1e-9


@dataclass
class NodeMatch:
    match_id: str
    a_node_id: str
    b_node_id: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    status: str = "matched"  # "matched" | "ambiguous" (only real pairs here)
    candidates: List[str] = field(default_factory=list)


@dataclass
class MatchResult:
    matches: List[NodeMatch] = field(default_factory=list)
    unmatched_a: List[str] = field(default_factory=list)
    unmatched_b: List[str] = field(default_factory=list)
    ambiguous_pairs: List[dict] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matches": [vars(m) | {"evidence": list(m.evidence)} for m in self.matches],
            "unmatched_a": list(self.unmatched_a),
            "unmatched_b": list(self.unmatched_b),
            "ambiguous_pairs": [dict(p) for p in self.ambiguous_pairs],
            "summary": dict(self.summary),
        }

    def match_id_of_a(self, node_id: str) -> Optional[str]:
        for m in self.matches:
            if m.a_node_id == node_id:
                return m.match_id
        return None

    def match_id_of_b(self, node_id: str) -> Optional[str]:
        for m in self.matches:
            if m.b_node_id == node_id:
                return m.match_id
        return None

    def partner_a(self, node_id: str) -> Optional[str]:
        for m in self.matches:
            if m.a_node_id == node_id:
                return m.b_node_id
        return None

    def partner_b(self, node_id: str) -> Optional[str]:
        for m in self.matches:
            if m.b_node_id == node_id:
                return m.a_node_id
        return None


# ---- graph context helpers -------------------------------------------------


class _Side:
    """Precomputed view of one graph for matching."""

    def __init__(self, ir: GraphIR, side: str):
        self.ir = ir
        self.side = side
        self.nodes: Dict[str, NodeIR] = {n.id: n for n in ir.nodes}
        self.preds: Dict[str, List[Tuple[str, str]]] = {n.id: [] for n in ir.nodes}
        self.succs: Dict[str, List[Tuple[str, str]]] = {n.id: [] for n in ir.nodes}
        for e in ir.edges:
            if e.src and e.dst:
                self.succs[e.src].append((e.dst, e.tensor))
                self.preds[e.dst].append((e.src, e.tensor))
        self.graph_input_tensors = {v.name for v in ir.inputs}
        # Tensors each node consumes that come straight from graph inputs.
        self.io_input_tensors: Dict[str, set] = {n.id: set() for n in ir.nodes}
        for e in ir.edges:
            if e.src == "" and e.dst and e.tensor in self.graph_input_tensors:
                self.io_input_tensors[e.dst].add(e.tensor)
        # Signature: op + domain + canonical attrs + param shapes.
        self.signature: Dict[str, tuple] = {}
        for n in ir.nodes:
            self.signature[n.id] = _signature(n)
        self.sig_counts: Dict[tuple, int] = {}
        for s in self.signature.values():
            self.sig_counts[s] = self.sig_counts.get(s, 0, ) + 1
        self.name_counts: Dict[str, int] = {}
        for n in ir.nodes:
            if n.name:
                self.name_counts[n.name] = self.name_counts.get(n.name, 0) + 1
        self.topology_fp = ir.fingerprints.get("topology", "")


def _signature(n: NodeIR) -> tuple:
    attrs = json.dumps(n.attributes, sort_keys=True, ensure_ascii=False, default=str)
    # Param NAMES are rename-tolerant identifiers (like node names) — the
    # structural signature uses only dtype+shape.
    params = tuple(sorted((p.dtype, tuple(p.shape or [])) for p in n.params))
    return (n.op_type, n.domain, attrs, params)


def _attr_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ka, kb = set(a), set(b)
    if not ka and not kb:
        return 1.0
    inter = ka & kb
    union = ka | kb
    same = sum(1 for k in inter if json.dumps(a[k], sort_keys=True, default=str)
               == json.dumps(b[k], sort_keys=True, default=str))
    return same / len(union) if union else 1.0


def _shape_eq(s1, s2) -> bool:
    if s1 is None and s2 is None:
        return True  # both unknown: not evidence of difference
    if s1 is None or s2 is None:
        return True  # cannot judge
    return list(s1) == list(s2)


# ---- matcher ---------------------------------------------------------------


def match_graphs(a: GraphIR, b: GraphIR) -> MatchResult:
    sa, sb = _Side(a, "a"), _Side(b, "b")
    pair_of_a: Dict[str, str] = {}   # a_id -> b_id
    pair_of_b: Dict[str, str] = {}
    conf: Dict[Tuple[str, str], float] = {}
    evid: Dict[Tuple[str, str], List[str]] = {}
    ambiguous: set = set()           # node ids marked ambiguous
    amb_candidates: Dict[str, List[str]] = {}
    anchors_used = 0

    def pair(uid: str, vid: str, c: float, ev: List[str]) -> None:
        pair_of_a[uid] = vid
        pair_of_b[vid] = uid
        conf[(uid, vid)] = c
        evid[(uid, vid)] = ev

    # ---- stage 1a: unique op-signature anchors ----------------------------
    for uid in sa.nodes:
        if sa.sig_counts[sa.signature[uid]] != 1:
            continue
        s = sa.signature[uid]
        vids = [vid for vid in sb.nodes if sb.signature.get(vid) == s]
        if len(vids) == 1:
            pair(uid, vids[0], 0.95, [f"anchor: unique op signature {s[0]}({s[1] or 'onnx'})"])
            anchors_used += 1

    # ---- stage 1b: unique equal names --------------------------------------
    for uid, n in sa.nodes.items():
        if uid in pair_of_a or not n.name:
            continue
        if sa.name_counts.get(n.name, 0) != 1:
            continue
        vids = [vid for vid, m in sb.nodes.items()
                if m.name == n.name and sb.name_counts.get(m.name, 0) == 1]
        if len(vids) == 1:
            pair(uid, vids[0], 0.8, [f"anchor: unique node name '{n.name}'"])
            anchors_used += 1

    # ---- stage 1c: graph io correspondence ---------------------------------
    if ({v.name for v in a.inputs} == {v.name for v in b.inputs}
            and len(a.inputs) == len(b.inputs)
            and {v.name for v in a.outputs} == {v.name for v in b.outputs}
            and len(a.outputs) == len(b.outputs)):
        # Nodes fed directly by a uniquely-named graph input / producing a
        # uniquely-named graph output correspond 1:1 when counts match.
        for t in {v.name for v in a.inputs}:
            ua = [e.dst for e in a.edges if e.src == "" and e.tensor == t and e.dst]
            ub = [e.dst for e in b.edges if e.src == "" and e.tensor == t and e.dst]
            if (len(ua) == 1 and len(ub) == 1 and ua[0] not in pair_of_a
                    and ub[0] not in pair_of_b
                    and sa.nodes[ua[0]].op_type == sb.nodes[ub[0]].op_type):
                pair(ua[0], ub[0], 1.0, [f"anchor: sole consumer of graph input '{t}'"])
                anchors_used += 1
        for t in {v.name for v in a.outputs}:
            ua = [e.src for e in a.edges if e.dst == "" and e.tensor == t and e.src]
            ub = [e.src for e in b.edges if e.dst == "" and e.tensor == t and e.src]
            if (len(ua) == 1 and len(ub) == 1 and ua[0] not in pair_of_a
                    and ub[0] not in pair_of_b
                    and sa.nodes[ua[0]].op_type == sb.nodes[ub[0]].op_type):
                pair(ua[0], ub[0], 1.0, [f"anchor: sole producer of graph output '{t}'"])
                anchors_used += 1

    # ---- stage 2: propagation ----------------------------------------------
    # Candidates are restricted to the neighbourhood of already-matched nodes
    # (frontier expansion) -> O(N * deg) instead of O(N^2) all-pairs scoring.
    # Score ties are NOT resolved and NOT yet final: they stay pending until
    # propagation converges, then are reported as ambiguous.
    topology_bonus_applied = bool(sa.topology_fp and sa.topology_fp == sb.topology_fp)

    def candidates_for(side: str, nid: str) -> List[str]:
        """Unmatched opposite-side nodes adjacent to partners of matched
        neighbours of `nid`; falls back to same-named graph-input consumers
        when the node has no matched neighbour yet."""
        s_this, s_other = (sa, sb) if side == "a" else (sb, sa)
        po_this = pair_of_a if side == "a" else pair_of_b
        po_other = pair_of_b if side == "a" else pair_of_a
        neigh = [p for p, _ in s_this.preds[nid]] + [c for c, _ in s_this.succs[nid]]
        # graph-input consumers: treat the io pseudo-anchor as a neighbour
        cands: Dict[str, None] = {}
        for m in neigh:
            partner = po_this.get(m)
            if partner is None:
                continue
            for other, _t in s_other.preds[partner]:
                if other not in po_other and other in s_other.nodes:
                    cands[other] = None
            for other, _t in s_other.succs[partner]:
                if other not in po_other and other in s_other.nodes:
                    cands[other] = None
        if cands:
            return list(cands)
        # fallback: nodes fed by the same-named graph input
        for t in s_this.io_input_tensors[nid]:
            for e in s_other.ir.edges:
                if e.src == "" and e.tensor == t and e.dst and e.dst not in po_other:
                    cands[e.dst] = None
        return list(cands)

    pending_ties: Dict[str, List[str]] = {}   # a_id -> tied b candidates
    frontier: List[str] = []
    # seed: unmatched nodes adjacent to any anchor (or fed by a graph input)
    for uid in sa.nodes:
        if uid not in pair_of_a:
            neigh = [p for p, _ in sa.preds[uid]] + [c for c, _ in sa.succs[uid]]
            if any(m in pair_of_a for m in neigh) or sa.io_input_tensors[uid]:
                frontier.append(uid)
    frontier.sort()

    while frontier:
        next_frontier: Dict[str, None] = {}
        newly_paired: List[Tuple[str, str, float, List[str]]] = []
        for uid in frontier:
            if uid in pair_of_a or uid in ambiguous:
                continue
            cands = candidates_for("a", uid)
            if not cands:
                continue
            scored = []
            for vid in cands:
                if vid in pair_of_b or vid in ambiguous:
                    continue
                s = _pair_score(sa, sb, uid, vid, pair_of_a, pair_of_b)
                if s > 0:
                    scored.append((s, vid))
            if not scored:
                continue
            scored.sort(key=lambda t: (-t[0], t[1]))
            best_s, best_v = scored[0]
            if best_s < MATCH_SCORE_THRESHOLD:
                continue
            second_s = scored[1][0] if len(scored) > 1 else float("-inf")
            if second_s >= best_s - TIE_EPS:
                pending_ties[uid] = [v for s, v in scored if s >= best_s - TIE_EPS]
                continue
            # mutual-best check from the B side
            rev = []
            for u2 in candidates_for("b", best_v):
                if u2 in pair_of_a or u2 in ambiguous:
                    continue
                s2 = _pair_score(sa, sb, u2, best_v, pair_of_a, pair_of_b)
                if s2 > 0:
                    rev.append((s2, u2))
            rev.sort(key=lambda t: (-t[0], t[1]))
            rev_second = rev[1][0] if len(rev) > 1 else float("-inf")
            if not rev or rev[0][1] != uid:
                continue
            if best_s - rev_second < MATCH_MARGIN:
                pending_ties[uid] = [v for s, v in scored if s >= best_s - TIE_EPS]
                continue
            ev = [f"propagated: score {best_s:.2f}"]
            if topology_bonus_applied:
                ev.append("topology fingerprints equal")
            newly_paired.append((uid, best_v, best_s, ev))

        for uid, vid, s, ev in newly_paired:
            if uid in pair_of_a or vid in pair_of_b:
                continue
            pair(uid, vid,
                 _confidence(sa, sb, uid, vid, s, pair_of_a, pair_of_b,
                             topology_bonus_applied),
                 ev)
            changed = True
            # expand frontier: unmatched neighbours of both endpoints
            for n2, _ in sa.succs[uid]:
                if n2 not in pair_of_a and n2 not in ambiguous:
                    next_frontier[n2] = None
            for n2, _ in sa.preds[uid]:
                if n2 not in pair_of_a and n2 not in ambiguous:
                    next_frontier[n2] = None
            for n2, _ in sb.succs[vid]:
                q = pair_of_b.get(n2)
                if q is None and n2 not in ambiguous:
                    # B-only node: its A-side partners come via other matches
                    pass
        if not newly_paired:
            break
        frontier = sorted(next_frontier)

    # Converged: remaining pending ties are genuinely ambiguous.
    for uid, cands in sorted(pending_ties.items()):
        if uid in pair_of_a or uid in ambiguous:
            continue
        cands = [c for c in cands if c not in pair_of_b]
        if not cands:
            continue
        ambiguous.add(uid)
        amb_candidates[uid] = cands
        for c in cands:
            ambiguous.add(c)
            amb_candidates.setdefault(c, [])

    # Symmetric leftover groups: same op/degree on both sides, no evidence to
    # tell them apart -> report as ambiguous rather than silently unmatched.
    def leftover_sig(side: str, nid: str) -> tuple:
        s = sa if side == "a" else sb
        n = s.nodes[nid]
        return (n.op_type, len([t for t in n.inputs if t]), len([t for t in n.outputs if t]))

    rem_a = [u for u in sa.nodes if u not in pair_of_a and u not in ambiguous]
    rem_b = [v for v in sb.nodes if v not in pair_of_b and v not in ambiguous]
    from collections import Counter
    sig_a = Counter(leftover_sig("a", u) for u in rem_a)
    sig_b = Counter(leftover_sig("b", v) for v in rem_b)
    for sig, cnt_a in sig_a.items():
        cnt_b = sig_b.get(sig, 0)
        if cnt_a >= 1 and cnt_b >= 1 and (cnt_a > 1 or cnt_b > 1):
            group_a = [u for u in rem_a if leftover_sig("a", u) == sig]
            group_b = [v for v in rem_b if leftover_sig("b", v) == sig]
            for u in group_a:
                ambiguous.add(u)
                amb_candidates[u] = group_b
            for v in group_b:
                ambiguous.add(v)
                amb_candidates.setdefault(v, [])

    # ---- assemble result ----------------------------------------------------
    matches: List[NodeMatch] = []
    for i, ((uid, vid), c) in enumerate(sorted(conf.items(), key=lambda kv: (kv[0][0], kv[0][1]))):
        matches.append(NodeMatch(
            match_id=f"m{i}",
            a_node_id=uid,
            b_node_id=vid,
            confidence=round(c, 3),
            evidence=evid.get((uid, vid), []),
        ))

    unmatched_a = [nid for nid in (n.id for n in a.nodes) if nid not in pair_of_a]
    unmatched_b = [nid for nid in (n.id for n in b.nodes) if nid not in pair_of_b]
    ambiguous_pairs: List[dict] = []
    seen_amb: set = set()
    for uid in sorted(ambiguous):
        if uid in sa.nodes:
            key = ("a", uid)
            if key in seen_amb:
                continue
            seen_amb.add(key)
            ambiguous_pairs.append({
                "a": uid,
                "b": None,
                "candidates": amb_candidates.get(uid, []),
                "reason": "multiple equivalent candidates (symmetric structure / repeated ops)",
            })
    for vid in sorted(ambiguous):
        if vid in sb.nodes and vid not in pair_of_b:
            key = ("b", vid)
            if key in seen_amb:
                continue
            seen_amb.add(key)
            ambiguous_pairs.append({
                "a": None,
                "b": vid,
                "candidates": amb_candidates.get(vid, []),
                "reason": "multiple equivalent candidates (symmetric structure / repeated ops)",
            })

    result = MatchResult(
        matches=matches,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        ambiguous_pairs=ambiguous_pairs,
        summary={
            "matched_count": len(matches),
            "ambiguous_count": len(ambiguous_pairs),
            "unmatched_a_count": len(unmatched_a),
            "unmatched_b_count": len(unmatched_b),
            "anchors_used": anchors_used,
        },
    )
    return result


def _pair_score(sa: _Side, sb: _Side, uid: str, vid: str,
                pair_of_a: Dict[str, str], pair_of_b: Dict[str, str]) -> float:
    nu, nv = sa.nodes[uid], sb.nodes[vid]
    if nu.op_type == nv.op_type:
        s = 2.0
        if nu.domain == nv.domain:
            s += 0.2
    else:
        s = 0.0  # different ops need strong topology evidence via neighbors
    s += 1.0 * _attr_similarity(nu.attributes, nv.attributes)

    # matched-neighbor consistency (both directions)
    for side_a, side_b, po_a, po_b, w_pos, w_neg in (
        (sa, sb, pair_of_a, pair_of_b, 1.0, -1.0),
    ):
        matched_preds = 0
        for p, _t in side_a.preds[uid]:
            q = po_a.get(p)
            if q is None:
                continue
            matched_preds += 1
            if any(d == q for d, _ in side_b.preds[vid]):
                s += w_pos
            else:
                s += w_neg
        for c, _t in side_a.succs[uid]:
            q = po_a.get(c)
            if q is None:
                continue
            matched_preds += 1
            if any(d == q for d, _ in side_b.succs[vid]):
                s += 0.8
            else:
                s += -0.8
        # unmatched graph-input consumers on both sides
        if (side_a.io_input_tensors[uid] and side_b.io_input_tensors[vid]
                and not side_a.preds[uid] and not side_b.preds[vid]):
            s += 0.5
        if not matched_preds and nu.op_type != nv.op_type:
            return 0.0  # no topology support for an op mismatch pair

    # io degree similarity
    in_sim = min(len(nu.inputs), len(nv.inputs)) / max(len(nu.inputs), len(nv.inputs), 1)
    out_sim = min(len(nu.outputs), len(nv.outputs)) / max(len(nu.outputs), len(nv.outputs), 1)
    s += 0.25 * (in_sim + out_sim)

    # shape/dtype compat on first outputs
    for t in nu.outputs:
        vi_u = nu.output_info.get(t)
        break
    else:
        vi_u = None
    for t in nv.outputs:
        vi_v = nv.output_info.get(t)
        break
    else:
        vi_v = None
    if vi_u and vi_v and _shape_eq(vi_u.shape, vi_v.shape):
        s += 0.25
    if vi_u and vi_v and vi_u.dtype == vi_v.dtype:
        s += 0.1
    if nu.name and nu.name == nv.name:
        s += 0.3
    return s


def _confidence(sa: _Side, sb: _Side, uid: str, vid: str, score: float,
                pair_of_a: Dict[str, str], pair_of_b: Dict[str, str],
                topology_bonus: bool) -> float:
    base = 0.45
    neighbor_support = 0
    for p, _t in sa.preds[uid]:
        q = pair_of_a.get(p)
        if q and any(d == q for d, _ in sb.preds[vid]):
            neighbor_support += 1
    for c, _t in sa.succs[uid]:
        q = pair_of_a.get(c)
        if q and any(d == q for d, _ in sb.succs[vid]):
            neighbor_support += 1
    conf = base + 0.15 * neighbor_support
    conf += 0.1 * _attr_similarity(sa.nodes[uid].attributes, sb.nodes[vid].attributes)
    if sa.nodes[uid].name and sa.nodes[uid].name == sb.nodes[vid].name:
        conf += 0.1
    if topology_bonus:
        conf += 0.05
    return max(0.3, min(0.95, conf))
