"""Diff engine: builds a versioned Diff IR from two GraphIRs and a MatchResult."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from modeltoolbox.ir import GraphIR
from modeltoolbox.matcher import MatchResult, match_graphs

DIFF_SCHEMA_VERSION = 1

DIFF_KINDS = [
    "unchanged", "added", "removed", "op_changed", "attr_changed",
    "shape_changed", "dtype_changed", "param_changed", "connection_changed",
    "ambiguous",
]

_KIND_PRIORITY = [
    "ambiguous", "op_changed", "connection_changed", "shape_changed",
    "dtype_changed", "param_changed", "attr_changed",
]


@dataclass
class FieldChange:
    field: str
    before: Any
    after: Any
    kind: str  # "changed" | "added" | "removed"


@dataclass
class NodeDiff:
    match_id: Optional[str]
    kind: str
    a_node_id: Optional[str]
    b_node_id: Optional[str]
    a_display: str
    b_display: str
    changes: List[FieldChange] = field(default_factory=list)
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class EdgeDiff:
    kind: str  # unchanged | added | removed | changed
    a_edge: Optional[dict]
    b_edge: Optional[dict]
    a_node_id: Optional[str] = None
    b_node_id: Optional[str] = None


@dataclass
class DiffIR:
    diff_schema_version: int = DIFF_SCHEMA_VERSION
    a_model_id: str = ""
    b_model_id: str = ""
    a_name: str = ""
    b_name: str = ""
    node_diffs: List[NodeDiff] = field(default_factory=list)
    edge_diffs: List[EdgeDiff] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    a_topology_fingerprint: str = ""
    b_topology_fingerprint: str = ""
    topology_identical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DiffIR":
        node_diffs = [
            NodeDiff(
                match_id=nd.get("match_id"),
                kind=nd["kind"],
                a_node_id=nd.get("a_node_id"),
                b_node_id=nd.get("b_node_id"),
                a_display=nd.get("a_display", ""),
                b_display=nd.get("b_display", ""),
                changes=[FieldChange(**c) for c in nd.get("changes", [])],
                confidence=nd.get("confidence", 1.0),
                evidence=list(nd.get("evidence", [])),
                explanation=nd.get("explanation", ""),
            )
            for nd in d.get("node_diffs", [])
        ]
        edge_diffs = [
            EdgeDiff(
                kind=ed["kind"],
                a_edge=ed.get("a_edge"),
                b_edge=ed.get("b_edge"),
                a_node_id=ed.get("a_node_id"),
                b_node_id=ed.get("b_node_id"),
            )
            for ed in d.get("edge_diffs", [])
        ]
        return DiffIR(
            diff_schema_version=d.get("diff_schema_version", DIFF_SCHEMA_VERSION),
            a_model_id=d.get("a_model_id", ""),
            b_model_id=d.get("b_model_id", ""),
            a_name=d.get("a_name", ""),
            b_name=d.get("b_name", ""),
            node_diffs=node_diffs,
            edge_diffs=edge_diffs,
            summary=d.get("summary", {}),
            warnings=list(d.get("warnings", [])),
            a_topology_fingerprint=d.get("a_topology_fingerprint", ""),
            b_topology_fingerprint=d.get("b_topology_fingerprint", ""),
            topology_identical=d.get("topology_identical", False),
        )

    @staticmethod
    def from_json(s: str) -> "DiffIR":
        return DiffIR.from_dict(json.loads(s))

    def node_diff_of_a(self, node_id: str) -> Optional[NodeDiff]:
        for nd in self.node_diffs:
            if nd.a_node_id == node_id:
                return nd
        return None

    def node_diff_of_b(self, node_id: str) -> Optional[NodeDiff]:
        for nd in self.node_diffs:
            if nd.b_node_id == node_id:
                return nd
        return None


# ---- helpers ---------------------------------------------------------------


def _vi_summary(vi) -> Optional[dict]:
    if vi is None:
        return None
    return {"dtype": vi.dtype, "shape": list(vi.shape) if vi.shape is not None else None}


def _param_map(n) -> Dict[str, tuple]:
    return {p.name: (p.dtype, tuple(p.shape or [])) for p in n.params}


def _producer_map(ir: GraphIR) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for e in ir.edges:
        if e.src and e.dst:
            out[e.tensor] = e.src
    return out


# ---- main construction -----------------------------------------------------


def diff_models(a: GraphIR, b: GraphIR) -> DiffIR:
    return build_diff_ir(a, b, match_graphs(a, b))


def build_diff_ir(a: GraphIR, b: GraphIR, match: MatchResult) -> DiffIR:
    ir = DiffIR(
        a_model_id=a.model_id,
        b_model_id=b.model_id,
        a_name=a.name,
        b_name=b.name,
        a_topology_fingerprint=a.fingerprints.get("topology", ""),
        b_topology_fingerprint=b.fingerprints.get("topology", ""),
    )
    ir.topology_identical = (
        bool(ir.a_topology_fingerprint)
        and ir.a_topology_fingerprint == ir.b_topology_fingerprint
    )

    prod_a = _producer_map(a)
    prod_b = _producer_map(b)
    nodes_a = {n.id: n for n in a.nodes}
    nodes_b = {n.id: n for n in b.nodes}

    matched_a = {m.a_node_id: m for m in match.matches}
    matched_b = {m.b_node_id: m for m in match.matches}
    amb_a = {p["a"] for p in match.ambiguous_pairs if p.get("a")}
    amb_b = {p["b"] for p in match.ambiguous_pairs if p.get("b")}

    # ---- paired nodes ----
    for m in match.matches:
        nu = nodes_a[m.a_node_id]
        nv = nodes_b[m.b_node_id]
        changes: List[FieldChange] = []
        kinds = set()

        if nu.op_type != nv.op_type or nu.domain != nv.domain:
            kinds.add("op_changed")
            if nu.op_type != nv.op_type:
                changes.append(FieldChange("op_type", nu.op_type, nv.op_type, "changed"))
            if nu.domain != nv.domain:
                changes.append(FieldChange("domain", nu.domain, nv.domain, "changed"))

        # attributes
        keys = set(nu.attributes) | set(nv.attributes)
        for k in sorted(keys):
            va, vb = nu.attributes.get(k), nv.attributes.get(k)
            if json.dumps(va, sort_keys=True, default=str) == json.dumps(vb, sort_keys=True, default=str):
                continue
            kinds.add("attr_changed")
            if k not in nu.attributes:
                changes.append(FieldChange(f"attr:{k}", None, vb, "added"))
            elif k not in nv.attributes:
                changes.append(FieldChange(f"attr:{k}", va, None, "removed"))
            else:
                changes.append(FieldChange(f"attr:{k}", va, vb, "changed"))

        # input/output tensor shape & dtype (by tensor position)
        for tag, la, lb in (("output", nu.outputs, nv.outputs), ("input", nu.inputs, nv.inputs)):
            for i, (ta, tb) in enumerate(zip(la, lb)):
                if not ta or not tb:
                    continue
                ia = nu.output_info.get(ta) if tag == "output" else nu.input_info.get(ta)
                ib = nv.output_info.get(tb) if tag == "output" else nv.input_info.get(tb)
                sa_, sb_ = _vi_summary(ia), _vi_summary(ib)
                if sa_ and sb_:
                    if sa_["shape"] != sb_["shape"] and not (
                        sa_["shape"] is None and sb_["shape"] is None
                    ):
                        kinds.add("shape_changed")
                        changes.append(FieldChange(
                            f"{tag}_shape:{ta or tb}", sa_["shape"], sb_["shape"], "changed"))
                    if sa_["dtype"] != sb_["dtype"]:
                        kinds.add("dtype_changed")
                        changes.append(FieldChange(
                            f"dtype:{ta or tb}", sa_["dtype"], sb_["dtype"], "changed"))

        # params: compare positionally (param names are rename-tolerant ids)
        pa, pb = _param_map(nu), _param_map(nv)
        pos_a = [(t, pa[t]) for t in nu.inputs if t in pa]
        pos_b = [(t, pb[t]) for t in nv.inputs if t in pb]
        if len(pos_a) != len(pos_b):
            kinds.add("param_changed")
            changes.append(FieldChange("param_count", len(pos_a), len(pos_b), "changed"))
        else:
            for i, ((ta, va), (tb, vb)) in enumerate(zip(pos_a, pos_b)):
                if va[1] != vb[1] or va[0] != vb[0]:
                    kinds.add("param_changed")
                    changes.append(FieldChange(
                        f"param[{i}]", [ta, list(va)], [tb, list(vb)], "changed"))
            # same-shape different-values are NOT a structural change (by design)

        # connection change: rename-tolerant upstream match-context of inputs
        norm_a = [(c, i) for i, c in enumerate(_ctx_upstream(nu, prod_a, matched_a))]
        norm_b = [(c, i) for i, c in enumerate(_ctx_upstream(nv, prod_b, matched_b))]
        if norm_a != norm_b:
            kinds.add("connection_changed")
            changes.append(FieldChange("inputs", norm_a, norm_b, "changed"))

        if kinds:
            kind = next(k for k in _KIND_PRIORITY if k in kinds)
            expl = _explain(kind, changes)
        else:
            kind = "unchanged"
            expl = "structures identical"
        ir.node_diffs.append(NodeDiff(
            match_id=m.match_id,
            kind=kind,
            a_node_id=nu.id,
            b_node_id=nv.id,
            a_display=nu.display_name,
            b_display=nv.display_name,
            changes=changes,
            confidence=m.confidence,
            evidence=list(m.evidence),
            explanation=expl,
        ))

    # ---- unmatched: removed / added ----
    for nid in match.unmatched_a:
        if nid in amb_a:
            continue
        n = nodes_a[nid]
        ir.node_diffs.append(NodeDiff(
            match_id=None, kind="removed", a_node_id=n.id, b_node_id=None,
            a_display=n.display_name, b_display="",
            confidence=1.0, explanation=f"node '{n.display_name}' exists only in model A",
        ))
    for nid in match.unmatched_b:
        if nid in amb_b:
            continue
        n = nodes_b[nid]
        ir.node_diffs.append(NodeDiff(
            match_id=None, kind="added", a_node_id=None, b_node_id=n.id,
            a_display="", b_display=n.display_name,
            confidence=1.0, explanation=f"node '{n.display_name}' exists only in model B",
        ))
    # ambiguous
    for p in match.ambiguous_pairs:
        na = nodes_a.get(p["a"]) if p.get("a") else None
        nb = nodes_b.get(p["b"]) if p.get("b") else None
        ir.node_diffs.append(NodeDiff(
            match_id=None, kind="ambiguous",
            a_node_id=na.id if na else None,
            b_node_id=nb.id if nb else None,
            a_display=na.display_name if na else "",
            b_display=nb.display_name if nb else "",
            confidence=0.0,
            explanation="multiple equivalent candidate matches; correspondence is ambiguous",
        ))

    # ---- edges ----
    def edge_key(ir_graph: GraphIR, e, side: str) -> tuple:
        matched = matched_a if side == "a" else matched_b
        if e.src:
            src_mid = matched[e.src].match_id if e.src in matched else ("UNMATCHED", e.src)
        else:
            src_mid = "__graph_src__"
        if e.dst:
            dst_mid = matched[e.dst].match_id if e.dst in matched else ("UNMATCHED", e.dst)
        else:
            dst_mid = "__graph_dst__"
        role = "g" if e.tensor in {v.name for v in ir_graph.inputs} else "n"
        return (src_mid, dst_mid, e.src_port, e.dst_port, role)

    ea_by_key: Dict[tuple, list] = {}
    eb_by_key: Dict[tuple, list] = {}
    for e in a.edges:
        k = edge_key(a, e, "a")
        if k:
            ea_by_key.setdefault(k, []).append(e)
    for e in b.edges:
        k = edge_key(b, e, "b")
        if k:
            eb_by_key.setdefault(k, []).append(e)
    for k in sorted(set(ea_by_key) | set(eb_by_key), key=str):
        la, lb = ea_by_key.get(k, []), eb_by_key.get(k, [])
        if la and lb:
            ea, eb = la[0], lb[0]
            kind = "unchanged" if ea.tensor == eb.tensor else "changed"
        elif la:
            ea, eb, kind = la[0], None, "removed"
        else:
            ea, eb, kind = None, lb[0], "added"
        ir.edge_diffs.append(EdgeDiff(
            kind=kind,
            a_edge=asdict(ea) if ea else None,
            b_edge=asdict(eb) if eb else None,
            a_node_id=ea.src or ea.dst if ea else None,
            b_node_id=eb.src or eb.dst if eb else None,
        ))

    # ---- summary ----
    summary = {k: 0 for k in DIFF_KINDS}
    for nd in ir.node_diffs:
        summary[nd.kind] = summary.get(nd.kind, 0) + 1
    for ed in ir.edge_diffs:
        summary["edge_" + ed.kind] = summary.get("edge_" + ed.kind, 0) + 1
    ir.summary = summary
    if not match.matches and (a.nodes or b.nodes):
        ir.warnings.append("no confident matches found; graphs may be unrelated")
    return ir


def _ctx_upstream(node, prod: Dict[str, str], matched: Dict[str, Any]) -> List[str]:
    """Rename-tolerant upstream context: list of 'match_id|graph_input|unmatched'."""
    out = []
    for t in node.inputs:
        if not t:
            continue
        p = prod.get(t)
        if p is None:
            out.append("graph_input_or_param")
        else:
            m = matched.get(p)
            out.append(m.match_id if m else "unmatched:" + p)
    return out


def _explain(kind: str, changes: List[FieldChange]) -> str:
    if kind == "op_changed":
        c = next((c for c in changes if c.field == "op_type"), None)
        return f"operator changed {c.before} -> {c.after}" if c else "operator/domain changed"
    if kind == "attr_changed":
        names = ", ".join(c.field.split(":", 1)[-1] for c in changes if c.field.startswith("attr:"))
        return f"attribute(s) changed: {names}"
    if kind == "shape_changed":
        return "input/output shape changed"
    if kind == "dtype_changed":
        return "data type changed"
    if kind == "param_changed":
        return "parameter shape/metadata changed"
    if kind == "connection_changed":
        return "connections to upstream nodes changed"
    return kind
