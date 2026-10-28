"""Graph IR: versioned, serializable internal graph representation.

This module is frontend-agnostic and has no ONNX/PyTorch dependency.
All identifiers are strictly separated (see docs/architecture.md section 4):

- ``model_id``  identity of one loaded model instance (assigned by session)
- ``node_id``   stable, unique inside one model
- ``edge_id``   unique inside one model
- ``match_id``  cross-model correspondence, only produced by the matcher
- ``group_id``  folding group, only produced by the folding layer
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


@dataclass
class ValueInfo:
    """A named tensor value: shape + dtype. Unknown dims are None."""

    name: str
    dtype: Optional[str] = None  # e.g. "FLOAT"; None = unknown
    shape: Optional[List[Optional[int]]] = None  # None = shape unknown entirely
    # shape is None (unknown) vs [] (scalar) distinction is preserved.


@dataclass
class ParamInfo:
    """Metadata for an initializer/parameter tensor (no payload data)."""

    name: str
    dtype: Optional[str] = None
    shape: Optional[List[Optional[int]]] = None
    num_elements: Optional[int] = None
    data_size_bytes: Optional[int] = None
    is_constant: bool = True


@dataclass
class SubgraphInfo:
    """Nested graph (e.g. ONNX If/Loop branches). Never silently dropped."""

    attribute: str  # which attribute holds it, e.g. "then_branch"
    node_count: int
    edge_count: int
    nodes: List["NodeIR"] = field(default_factory=list)
    edges: List["EdgeIR"] = field(default_factory=list)
    inputs: List[ValueInfo] = field(default_factory=list)
    outputs: List[ValueInfo] = field(default_factory=list)


@dataclass
class NodeIR:
    id: str
    name: str  # original name if any; may be ""
    display_name: str  # non-empty, for UI
    op_type: str
    domain: str = ""  # "" = ai.onnx default domain
    attributes: Dict[str, Any] = field(default_factory=dict)  # JSON-serializable
    inputs: List[str] = field(default_factory=list)  # tensor names, "" = optional empty slot
    outputs: List[str] = field(default_factory=list)
    # Per-port shape/dtype resolved from value_info; key = tensor name.
    input_info: Dict[str, ValueInfo] = field(default_factory=dict)
    output_info: Dict[str, ValueInfo] = field(default_factory=dict)
    params: List[ParamInfo] = field(default_factory=list)  # initializer inputs
    subgraphs: List[SubgraphInfo] = field(default_factory=list)
    level: int = 0  # topological level (0 = root graph); nested subgraph nodes > 0


@dataclass
class EdgeIR:
    id: str
    src: str  # node_id of producer; "" = graph input feeds this tensor
    dst: str  # node_id of consumer; "" = tensor is a graph output only
    tensor: str
    src_port: int = 0
    dst_port: int = 0


@dataclass
class GraphIR:
    model_id: str
    name: str
    schema_version: int = SCHEMA_VERSION
    producer: str = ""
    ir_version: Optional[int] = None
    opsets: Dict[str, int] = field(default_factory=dict)  # domain -> version
    metadata_props: Dict[str, str] = field(default_factory=dict)
    inputs: List[ValueInfo] = field(default_factory=list)
    outputs: List[ValueInfo] = field(default_factory=list)
    nodes: List[NodeIR] = field(default_factory=list)
    edges: List[EdgeIR] = field(default_factory=list)
    initializers: List[ParamInfo] = field(default_factory=list)
    fingerprints: Dict[str, str] = field(default_factory=dict)
    source_path: str = ""
    file_size_bytes: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    # ---- convenience -----------------------------------------------------
    def node(self, node_id: str) -> Optional[NodeIR]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GraphIR":
        """Rebuild a GraphIR from its dict form (as produced by to_dict)."""
        inputs = [ValueInfo(**v) for v in d.get("inputs", [])]
        outputs = [ValueInfo(**v) for v in d.get("outputs", [])]
        initializers = [ParamInfo(**p) for p in d.get("initializers", [])]

        def _vi_dict(raw: Dict[str, Any]) -> Dict[str, ValueInfo]:
            return {k: ValueInfo(**v) for k, v in raw.items()}

        def _param(raw: Dict[str, Any]) -> ParamInfo:
            return ParamInfo(**raw)

        def _subgraph(raw: Dict[str, Any]) -> SubgraphInfo:
            return SubgraphInfo(
                attribute=raw["attribute"],
                node_count=raw["node_count"],
                edge_count=raw["edge_count"],
                nodes=[NodeIR(**_node_fix(n)) for n in raw.get("nodes", [])],
                edges=[EdgeIR(**e) for e in raw.get("edges", [])],
                inputs=[ValueInfo(**v) for v in raw.get("inputs", [])],
                outputs=[ValueInfo(**v) for v in raw.get("outputs", [])],
            )

        def _node_fix(raw: Dict[str, Any]) -> Dict[str, Any]:
            # NodeIR has nested dataclasses that need explicit rebuilding.
            raw = dict(raw)
            raw["input_info"] = _vi_dict(raw.get("input_info", {}))
            raw["output_info"] = _vi_dict(raw.get("output_info", {}))
            raw["params"] = [_param(p) for p in raw.get("params", [])]
            raw["subgraphs"] = [_subgraph(s) for s in raw.get("subgraphs", [])]
            return raw

        nodes = [NodeIR(**_node_fix(n)) for n in d.get("nodes", [])]
        edges = [EdgeIR(**e) for e in d.get("edges", [])]
        return GraphIR(
            model_id=d["model_id"],
            name=d.get("name", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            producer=d.get("producer", ""),
            ir_version=d.get("ir_version"),
            opsets=d.get("opsets", {}),
            metadata_props=d.get("metadata_props", {}),
            inputs=inputs,
            outputs=outputs,
            nodes=nodes,
            edges=edges,
            initializers=initializers,
            fingerprints=d.get("fingerprints", {}),
            source_path=d.get("source_path", ""),
            file_size_bytes=d.get("file_size_bytes"),
            warnings=d.get("warnings", []),
        )

    @staticmethod
    def from_json(s: str) -> "GraphIR":
        return GraphIR.from_dict(json.loads(s))


# ---- fingerprints ---------------------------------------------------------


def structure_fingerprint(ir: GraphIR) -> str:
    """Hash over names-included structure (nodes+attrs+edges+io).

    Two models with identical structure AND identical names share this hash.
    """
    payload = {
        "inputs": [asdict(v) for v in ir.inputs],
        "outputs": [asdict(v) for v in ir.outputs],
        "nodes": [
            {
                "name": n.name,
                "op": n.op_type,
                "domain": n.domain,
                "attrs": n.attributes,
                "inputs": n.inputs,
                "outputs": n.outputs,
                "params": [(p.name, p.dtype, p.shape) for p in n.params],
            }
            for n in ir.nodes
        ],
        "edges": [(e.src, e.dst, e.tensor, e.src_port, e.dst_port) for e in ir.edges],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def topology_fingerprint(ir: GraphIR) -> str:
    """Hash over names-stripped topology + op types.

    Models that differ only in node/tensor naming share this hash.
    """
    # Map original node ids -> positional ids to strip naming.
    id_map = {n.id: f"N{i}" for i, n in enumerate(ir.nodes)}
    nodes = [
        {"op": n.op_type, "domain": n.domain, "attrs": _normalize_attr_for_topology(n.attributes)}
        for n in ir.nodes
    ]
    edges = sorted(
        (id_map.get(e.src, ""), id_map.get(e.dst, ""), e.src_port, e.dst_port) for e in ir.edges
    )
    io = [(v.dtype, v.shape) for v in ir.inputs], [(v.dtype, v.shape) for v in ir.outputs]
    payload = {"nodes": nodes, "edges": edges, "io": io}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalize_attr_for_topology(attrs: Dict[str, Any]) -> Any:
    """Attributes influence topology/semantics; keep them but sort keys."""
    return json.loads(json.dumps(attrs, sort_keys=True, ensure_ascii=False, default=str))


def assign_fingerprints(ir: GraphIR, content_sha256: Optional[str] = None) -> None:
    ir.fingerprints["structure"] = structure_fingerprint(ir)
    ir.fingerprints["topology"] = topology_fingerprint(ir)
    if content_sha256 is not None:
        ir.fingerprints["content"] = content_sha256


def build_edges(ir: GraphIR) -> None:
    """Derive EdgeIR list from node input/output tensor connections.

    Producer resolution: first producing node of a tensor wins; duplicate
    producers generate a warning (invalid but must not crash). Tensors that
    are graph inputs feed consumers with src="". Tensors that are graph
    outputs (or unconsumed) yield edges with dst="".
    """
    producers: Dict[str, tuple] = {}
    warnings: List[str] = []
    for n in ir.nodes:
        for port, t in enumerate(n.outputs):
            if not t:
                continue
            if t in producers:
                warnings.append(
                    f"tensor '{t}' produced by both '{producers[t][0]}' and '{n.id}'; "
                    "first producer wins"
                )
            else:
                producers[t] = (n.id, port)

    graph_inputs = {v.name for v in ir.inputs}
    input_like = graph_inputs | {p.name for p in ir.initializers}
    consumed = set()
    edges: List[EdgeIR] = []

    def add_edge(src: str, src_port: int, dst: str, dst_port: int, tensor: str) -> None:
        eid = f"e:{src or 'IN'}->{dst or 'OUT'}:{tensor}"
        edges.append(EdgeIR(id=eid, src=src, dst=dst, tensor=tensor, src_port=src_port, dst_port=dst_port))

    for n in ir.nodes:
        for port, t in enumerate(n.inputs):
            if not t:
                continue  # optional empty input slot
            consumed.add(t)
            if t in producers:
                src, src_port = producers[t]
            elif t in input_like:
                src, src_port = "", 0
            else:
                # Unknown producer (dangling tensor): treat like a graph input
                # but flag it so downstream knows the model is odd.
                warnings.append(f"tensor '{t}' consumed by '{n.id}' has no producer and is not a graph input")
                src, src_port = "", 0
            add_edge(src, src_port, n.id, port, t)

    # Graph outputs that no node consumes (or produced directly by inputs).
    for v in ir.outputs:
        if v.name in consumed:
            continue
        if v.name in producers:
            src, src_port = producers[v.name]
        elif v.name in input_like:
            src, src_port = "", 0
        else:
            warnings.append(f"graph output '{v.name}' has no producer")
            continue
        add_edge(src, src_port, "", 0, v.name)

    ir.edges = edges
    ir.warnings.extend(warnings)


def validate_ir(ir: GraphIR) -> List[str]:
    """Structural sanity checks; returns list of problems (does not raise)."""
    problems: List[str] = []
    ids = [n.id for n in ir.nodes]
    if len(ids) != len(set(ids)):
        problems.append("duplicate node ids")
    for n in ir.nodes:
        if not n.op_type:
            problems.append(f"node {n.id} has empty op_type")
    for e in ir.edges:
        if e.src and ir.node(e.src) is None:
            problems.append(f"edge {e.id} references missing src node {e.src}")
        if e.dst and ir.node(e.dst) is None:
            problems.append(f"edge {e.id} references missing dst node {e.dst}")
    return problems
