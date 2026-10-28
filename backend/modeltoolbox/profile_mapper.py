"""Map ProfileRecords onto GraphIR nodes with explicit reliability tiers.

Reliability (high -> low):
  1. model fingerprint match (warning if mismatched — never silently applied)
  2. exact node_id
  3. exact node_name (unique; duplicates -> ambiguous)
  4. tensor name -> producer node
  5. unique op_type fallback (multiple instances -> ambiguous)
Order-of-appearance assignment is NEVER used. Every record ends up
mapped / unmapped / ambiguous / group-level — nothing is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from modeltoolbox.ir import GraphIR
from modeltoolbox.profile_adapter import ProfileData, ProfileRecord


@dataclass
class NodePerf:
    node_id: str
    duration_us: float
    calls: int
    share: float
    status: str = "mapped"
    record_indices: List[int] = field(default_factory=list)


@dataclass
class UnmappedRecord:
    record_index: int
    node_ref: Optional[str]
    ref_kind: str
    reason: str
    candidates: List[str] = field(default_factory=list)


@dataclass
class MappedProfile:
    per_node: Dict[str, NodePerf] = field(default_factory=dict)
    unmapped: List[UnmappedRecord] = field(default_factory=list)
    ambiguous: List[UnmappedRecord] = field(default_factory=list)
    coverage: float = 0.0
    total_mapped_us: float = 0.0
    end_to_end_us: Optional[float] = None
    model_fingerprint_matched: Optional[bool] = None
    warnings: List[str] = field(default_factory=list)


def map_profile(profile: ProfileData, ir: GraphIR) -> MappedProfile:
    out = MappedProfile(end_to_end_us=profile.end_to_end_us)
    out.warnings.extend(profile.warnings)

    # ---- fingerprint gate ----
    fp = profile.model_fingerprint
    if fp:
        known = {ir.fingerprints.get("content"), ir.fingerprints.get("structure")}
        out.model_fingerprint_matched = fp in known
        if not out.model_fingerprint_matched:
            out.warnings.append(
                "profile model_fingerprint does not match this model; "
                "mappings may be wrong — verify before trusting"
            )

    nodes_by_id = {n.id: n for n in ir.nodes}
    by_name: Dict[str, List[str]] = {}
    for n in ir.nodes:
        for key in {n.name, n.display_name}:
            if key:
                by_name.setdefault(key, []).append(n.id)
    tensor_to_producer: Dict[str, str] = {}
    for n in ir.nodes:
        for t in n.outputs:
            tensor_to_producer.setdefault(t, n.id)
    by_op: Dict[str, List[str]] = {}
    for n in ir.nodes:
        by_op.setdefault(n.op_type, []).append(n.id)

    def record_unmapped(rec: ProfileRecord, reason: str, candidates: Optional[List[str]] = None):
        u = UnmappedRecord(rec.index, rec.node_ref, rec.ref_kind, reason, candidates or [])
        out.unmapped.append(u)
        if reason.startswith("ambiguous"):
            out.ambiguous.append(u)

    total_mapped = 0.0
    total_all = 0.0

    for rec in profile.records:
        total_all += rec.duration_us

        if rec.fused or rec.ref_kind == "kernel":
            # Kernel records may cover several nodes; we do not pretend they
            # are a single node's precise cost.
            record_unmapped(rec, "fused_kernel")
            continue

        target: Optional[str] = None
        if rec.ref_kind == "node_id" and rec.node_ref in nodes_by_id:
            target = rec.node_ref
        elif rec.ref_kind == "node_name" and rec.node_ref:
            hits = by_name.get(rec.node_ref, [])
            if len(hits) == 1:
                target = hits[0]
            elif len(hits) > 1:
                record_unmapped(rec, f"ambiguous:{len(hits)} nodes named '{rec.node_ref}'", hits)
                continue
        elif rec.ref_kind == "tensor" and rec.node_ref:
            target = tensor_to_producer.get(rec.node_ref)
            if target is None and any(v.name == rec.node_ref for v in ir.inputs):
                record_unmapped(rec, "tensor is a graph input; no producer node")
                continue

        if target is None and rec.op_type:
            cands = by_op.get(rec.op_type, [])
            if len(cands) == 1:
                target = cands[0]
                out.warnings.append(
                    f"record {rec.index} mapped by unique op_type '{rec.op_type}'"
                )
            elif len(cands) > 1:
                record_unmapped(rec, f"ambiguous:{len(cands)} '{rec.op_type}' nodes", cands)
                continue

        if target is None:
            record_unmapped(rec, "no node matched measurement reference")
            continue

        total_mapped += rec.duration_us
        perf = out.per_node.get(target)
        if perf is None:
            out.per_node[target] = NodePerf(
                node_id=target, duration_us=rec.duration_us, calls=rec.calls,
                share=0.0, record_indices=[rec.index])
        else:
            perf.duration_us += rec.duration_us
            perf.calls += rec.calls
            perf.record_indices.append(rec.index)

    if total_mapped > 0:
        for perf in out.per_node.values():
            perf.share = round(perf.duration_us / total_mapped, 6)
    elif profile.records:
        out.warnings.append("no records could be mapped; shares are all 0")

    out.total_mapped_us = total_mapped
    out.coverage = round(total_mapped / total_all, 6) if total_all > 0 else 0.0
    return out
