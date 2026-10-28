"""ONNX importer: ModelProto -> GraphIR.

Tolerant of partial shape-inference failures and odd but parseable models;
raises ModelLoadError with a readable message for unparseable input.
Nested graphs (If/Loop branches) are preserved, never silently dropped.
Initializer payloads are never serialized — metadata only.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

import onnx
from onnx import TensorProto, shape_inference

from modeltoolbox.ir import (
    GraphIR,
    NodeIR,
    ParamInfo,
    SubgraphInfo,
    ValueInfo,
    assign_fingerprints,
    build_edges,
    validate_ir,
)


class ModelLoadError(Exception):
    """Raised when a file cannot be parsed as an ONNX model at all."""


def load_onnx(path: str, model_id: str) -> GraphIR:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise ModelLoadError(f"cannot read model file '{path}': {e}") from e
    return load_onnx_bytes(data, model_id, source_path=path)


def load_onnx_bytes(data: bytes, model_id: str, source_path: str = "") -> GraphIR:
    if len(data) == 0:
        raise ModelLoadError("file is empty (0 bytes); not a valid ONNX model")

    try:
        model = onnx.ModelProto.FromString(data)
    except Exception as e:  # protobuf DecodeError and friends
        raise ModelLoadError(
            f"not a parseable ONNX model (protobuf decode failed: {e}); "
            "file may be corrupted or not ONNX"
        ) from e

    if not model.HasField("graph"):
        raise ModelLoadError("ONNX model has no graph; nothing to display")

    warnings: List[str] = []
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        warnings.append(f"onnx checker warning: {str(e)[:300]}")

    ir = GraphIR(
        model_id=model_id,
        name=model.graph.name or os.path.basename(source_path) or "model",
        producer=model.producer_name or "",
        ir_version=model.ir_version,
        source_path=source_path,
        file_size_bytes=len(data),
    )
    for opset in model.opset_import:
        ir.opsets[opset.domain or ""] = opset.version
    for prop in model.metadata_props:
        ir.metadata_props[prop.key] = prop.value

    _load_initializers(model.graph, ir)
    _load_inputs_outputs(model.graph, ir, warnings)
    _load_nodes(model.graph, ir, warnings, level=0, id_prefix="")
    _load_value_info(model, ir, warnings)
    _attach_params(ir)

    build_edges(ir)
    warnings.extend(validate_ir(ir))
    ir.warnings = warnings
    assign_fingerprints(ir, content_sha256=hashlib.sha256(data).hexdigest())
    return ir


# ---- initializers & graph IO ---------------------------------------------


def _dtype_name(elem_type: int) -> Optional[str]:
    try:
        return TensorProto.DataType.Name(elem_type)
    except (ValueError, KeyError):
        return None


def _tensor_meta(t: TensorProto) -> Tuple[Optional[str], Optional[List[Optional[int]]], Optional[int]]:
    dtype = _dtype_name(t.data_type)
    shape: List[Optional[int]] = [int(d) for d in t.dims]
    return dtype, shape, len(t.raw_data) if t.raw_data else None


def _load_initializers(graph: onnx.GraphProto, ir: GraphIR) -> None:
    for t in graph.initializer:
        dtype, shape, nbytes = _tensor_meta(t)
        num_elements: Optional[int] = 1
        for d in shape:
            num_elements *= d
        ir.initializers.append(
            ParamInfo(
                name=t.name,
                dtype=dtype,
                shape=shape,
                num_elements=num_elements,
                data_size_bytes=nbytes,
            )
        )


def _value_info_from_proto(vi: onnx.ValueInfoProto) -> ValueInfo:
    tt = vi.type.tensor_type
    dtype = _dtype_name(tt.elem_type)
    shape: Optional[List[Optional[int]]] = None
    if tt.HasField("shape"):
        shape = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                shape.append(int(d.dim_value) if d.dim_value >= 0 else None)
            else:
                shape.append(None)  # symbolic / missing dim -> unknown, never guessed
    return ValueInfo(name=vi.name, dtype=dtype, shape=shape)


def _load_inputs_outputs(graph: onnx.GraphProto, ir: GraphIR, warnings: List[str]) -> None:
    init_names = {t.name for t in graph.initializer}
    for vi in graph.input:
        if vi.name in init_names:
            # Common export pattern: parameters listed as graph inputs.
            warnings.append(
                f"'{vi.name}' is both graph input and initializer; treated as parameter"
            )
            continue
        ir.inputs.append(_value_info_from_proto(vi))
    for vi in graph.output:
        ir.outputs.append(_value_info_from_proto(vi))


def _load_value_info(model: onnx.ModelProto, ir: GraphIR, warnings: List[str]) -> None:
    vi_map: Dict[str, ValueInfo] = {}

    def collect(g: onnx.GraphProto) -> None:
        for vi in list(g.value_info) + list(g.input) + list(g.output):
            if vi.name not in vi_map:
                vi_map[vi.name] = _value_info_from_proto(vi)

    collect(model.graph)
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False)
        collect(inferred.graph)
    except Exception as e:
        warnings.append(f"shape inference failed (shapes may be unknown): {str(e)[:200]}")

    def attach(nodes: List[NodeIR]) -> None:
        for n in nodes:
            for t in n.inputs:
                if t and t in vi_map:
                    n.input_info[t] = vi_map[t]
            for t in n.outputs:
                if t and t in vi_map:
                    n.output_info[t] = vi_map[t]
            for sg in n.subgraphs:
                attach(sg.nodes)

    attach(ir.nodes)


# ---- attributes & subgraphs ------------------------------------------------


def _normalize_attr(
    attr: onnx.AttributeProto,
    ir: GraphIR,
    warnings: List[str],
    level: int,
    id_prefix: str,
    subgraph_out: List[SubgraphInfo],
) -> Any:
    t = attr.type
    if t == onnx.AttributeProto.INT:
        return int(attr.i)
    if t == onnx.AttributeProto.FLOAT:
        return float(attr.f)
    if t == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="replace")
    if t == onnx.AttributeProto.INTS:
        return [int(x) for x in attr.ints]
    if t == onnx.AttributeProto.FLOATS:
        return [float(x) for x in attr.floats]
    if t == onnx.AttributeProto.STRINGS:
        return [s.decode("utf-8", errors="replace") for s in attr.strings]
    if t == onnx.AttributeProto.TENSOR:
        return {
            "__tensor__": attr.t.name,
            "dtype": _dtype_name(attr.t.data_type),
            "shape": [int(d) for d in attr.t.dims],
        }
    if t == onnx.AttributeProto.SPARSE_TENSOR:
        return {"__sparse_tensor__": attr.sparse_tensor.values.name}
    if t == onnx.AttributeProto.TENSORS:
        return [
            {
                "__tensor__": t2.name,
                "dtype": _dtype_name(t2.data_type),
                "shape": [int(d) for d in t2.dims],
            }
            for t2 in attr.tensors
        ]
    if t == onnx.AttributeProto.GRAPH:
        sg = _parse_subgraph(attr.g, ir, warnings, level + 1, f"{id_prefix}{attr.name}/")
        subgraph_out.append(sg)
        return {"__subgraph__": attr.name, "nodes": sg.node_count}
    if t == onnx.AttributeProto.GRAPHS:
        infos = []
        for gi, g in enumerate(attr.graphs):
            sg = _parse_subgraph(g, ir, warnings, level + 1, f"{id_prefix}{attr.name}{gi}/")
            subgraph_out.append(sg)
            infos.append({"nodes": sg.node_count})
        return {"__subgraphs__": attr.name, "branches": infos}
    return f"<unhandled attr type {t}: {str(attr)[:80]}>"


def _parse_subgraph(
    g: onnx.GraphProto, ir: GraphIR, warnings: List[str], level: int, id_prefix: str
) -> SubgraphInfo:
    sg = SubgraphInfo(attribute=id_prefix.rstrip("/"), node_count=0, edge_count=0)
    sub_nodes: List[NodeIR] = []
    subgraph_out_holder: List[SubgraphInfo] = []
    used: set = set()
    for idx, n in enumerate(g.node):
        if n.name:
            nid = n.name
            if nid in used:
                k = 2
                while f"{n.name}#{k}" in used:
                    k += 1
                nid = f"{n.name}#{k}"
        else:
            nid = f"n{idx}"
        while nid in used:
            nid = f"{nid}#2"
        used.add(nid)
        node = NodeIR(
            id=f"{id_prefix}{nid}",
            name=n.name or "",
            display_name=n.name or f"{n.op_type}_{idx}",
            op_type=n.op_type,
            domain=n.domain or "",
            level=level,
            inputs=list(n.input),
            outputs=list(n.output),
        )
        for a in n.attribute:
            node.attributes[a.name] = _normalize_attr(
                a, ir, warnings, level, f"{id_prefix}{nid}/", subgraph_out_holder
            )
        node.subgraphs = subgraph_out_holder[:]
        sub_nodes.append(node)
    sg.nodes = sub_nodes
    sg.node_count = len(sub_nodes)

    shell = GraphIR(model_id=ir.model_id, name="subgraph")
    shell.inputs = [_value_info_from_proto(vi) for vi in g.input]
    shell.outputs = [_value_info_from_proto(vi) for vi in g.output]
    shell.nodes = sub_nodes
    build_edges(shell)
    sg.edges = shell.edges
    sg.edge_count = len(shell.edges)
    sg.inputs = shell.inputs
    sg.outputs = shell.outputs
    return sg


# ---- nodes ----------------------------------------------------------------


def _load_nodes(
    graph: onnx.GraphProto, ir: GraphIR, warnings: List[str], level: int, id_prefix: str
) -> None:
    used: set = set()
    for idx, n in enumerate(graph.node):
        if n.name:
            nid = n.name
            if nid in used:
                k = 2
                while f"{n.name}#{k}" in used:
                    k += 1
                nid = f"{n.name}#{k}"
        else:
            nid = f"n{idx}"
        while nid in used:
            nid = f"{nid}#2"
        used.add(nid)
        node = NodeIR(
            id=f"{id_prefix}{nid}",
            name=n.name or "",
            display_name=n.name or f"{n.op_type}_{idx}",
            op_type=n.op_type,
            domain=n.domain or "",
            level=level,
            inputs=list(n.input),
            outputs=list(n.output),
        )
        subs: List[SubgraphInfo] = []
        for a in n.attribute:
            node.attributes[a.name] = _normalize_attr(a, ir, warnings, level, f"{id_prefix}{nid}/", subs)
        node.subgraphs = subs
        ir.nodes.append(node)


def _attach_params(ir: GraphIR) -> None:
    init_by_name = {p.name: p for p in ir.initializers}
    for n in ir.nodes:
        seen: set = set()
        for t in n.inputs:
            p = init_by_name.get(t)
            if p is not None and t not in seen:
                n.params.append(p)
                seen.add(t)
