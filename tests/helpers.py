"""Shared helpers for building small GraphIR fixtures directly in tests.

Synthetic test data only — no real model files involved.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from modeltoolbox.ir import (
    EdgeIR,
    GraphIR,
    NodeIR,
    ParamInfo,
    ValueInfo,
    assign_fingerprints,
    build_edges,
)


def make_ir(
    nodes: Iterable[Any],
    inputs: Sequence[Tuple[str, Optional[str], Optional[List]]] = (),
    outputs: Sequence[Tuple[str, Optional[str], Optional[List]]] = (),
    model_id: str = "m",
    name: str = "test_model",
    initializers: Sequence[Tuple[str, str, Optional[List]]] = (),
    with_fingerprints: bool = True,
) -> GraphIR:
    """Build a GraphIR.

    nodes: each entry is either
      - a tuple (node_id, op_type, input_tensors, output_tensors)
      - a tuple (node_id, op_type, input_tensors, output_tensors, attributes_dict)
      - a dict of NodeIR kwargs
    inputs/outputs: (name, dtype, shape) triples
    initializers: (name, dtype, shape) triples
    Edges are derived automatically via build_edges().
    """
    ir = GraphIR(model_id=model_id, name=name)
    for spec in nodes:
        if isinstance(spec, dict):
            ir.nodes.append(NodeIR(**spec))
            continue
        nid, op, ins, outs = spec[0], spec[1], list(spec[2]), list(spec[3])
        attrs = spec[4] if len(spec) > 4 else {}
        disp = nid if nid else f"{op}_{len(ir.nodes)}"
        ir.nodes.append(
            NodeIR(id=nid, name=nid, display_name=disp, op_type=op, attributes=attrs, inputs=ins, outputs=outs)
        )
    ir.inputs = [ValueInfo(name=n, dtype=d, shape=s) for n, d, s in inputs]
    ir.outputs = [ValueInfo(name=n, dtype=d, shape=s) for n, d, s in outputs]
    ir.initializers = [
        ParamInfo(name=n, dtype=d, shape=s, num_elements=(s and _prod(s)))
        for n, d, s in initializers
    ]
    # Attach initializer metadata to consuming nodes.
    init_by_name = {p.name: p for p in ir.initializers}
    for n in ir.nodes:
        for t in n.inputs:
            if t in init_by_name and t not in {p.name for p in n.params}:
                n.params.append(init_by_name[t])
    # Attach value info (graph inputs/outputs) to nodes, mirroring the importer.
    vi_by_name = {v.name: v for v in ir.inputs}
    vi_by_name.update({v.name: v for v in ir.outputs})
    for n in ir.nodes:
        for t in n.inputs:
            if t in vi_by_name:
                n.input_info[t] = vi_by_name[t]
        for t in n.outputs:
            if t in vi_by_name:
                n.output_info[t] = vi_by_name[t]
    build_edges(ir)
    if with_fingerprints:
        assign_fingerprints(ir)
    return ir


def _prod(shape: Optional[List]) -> Optional[int]:
    if shape is None:
        return None
    total = 1
    for d in shape:
        if d is None:
            return None
        total *= int(d)
    return total
