"""Exporters: diff/profile -> JSON / CSV / Markdown / HTML.

All exports are re-readable (JSON round-trips; CSV has a documented header;
HTML/MD embed the same summary numbers).
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List

from modeltoolbox.diff_engine import DiffIR

CSV_DIFF_HEADER = [
    "kind", "match_id", "a_node_id", "b_node_id", "a_display", "b_display",
    "confidence", "changes", "explanation",
]

CSV_PROFILE_HEADER = [
    "node_id", "op_type", "duration_us", "calls", "share", "status",
    "record_indices",
]


# ---- diff exports -----------------------------------------------------------


def diff_summary(diff: DiffIR) -> Dict[str, Any]:
    return {
        "a_name": diff.a_name,
        "b_name": diff.b_name,
        "topology_identical": diff.topology_identical,
        "node_counts": {k: v for k, v in diff.summary.items()
                        if not k.startswith("edge_")},
        "edge_counts": {k[len("edge_"):]: v for k, v in diff.summary.items()
                        if k.startswith("edge_")},
        "warnings": list(diff.warnings),
    }


def export_diff_json(diff: DiffIR) -> bytes:
    payload = {
        "diff_schema_version": diff.diff_schema_version,
        "summary": diff_summary(diff),
        "a_topology_fingerprint": diff.a_topology_fingerprint,
        "b_topology_fingerprint": diff.b_topology_fingerprint,
        "nodes": [asdict_lite(nd) for nd in diff.node_diffs],
        "edges": [
            {
                "kind": ed.kind,
                "a_edge": ed.a_edge,
                "b_edge": ed.b_edge,
                "a_node_id": ed.a_node_id,
                "b_node_id": ed.b_node_id,
            }
            for ed in diff.edge_diffs if ed.kind != "unchanged"
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def asdict_lite(nd) -> dict:
    return {
        "match_id": nd.match_id,
        "kind": nd.kind,
        "a_node_id": nd.a_node_id,
        "b_node_id": nd.b_node_id,
        "a_display": nd.a_display,
        "b_display": nd.b_display,
        "confidence": nd.confidence,
        "changes": [dict(field=c.field, before=c.before, after=c.after, kind=c.kind)
                    for c in nd.changes],
        "evidence": list(nd.evidence),
        "explanation": nd.explanation,
    }


def export_diff_csv(diff: DiffIR) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_DIFF_HEADER)
    for nd in diff.node_diffs:
        changes = "; ".join(
            f"{c.field}: {json.dumps(c.before, default=str)} -> "
            f"{json.dumps(c.after, default=str)}" for c in nd.changes)
        w.writerow([nd.kind, nd.match_id or "", nd.a_node_id or "",
                    nd.b_node_id or "", nd.a_display, nd.b_display,
                    nd.confidence, changes, nd.explanation])
    return buf.getvalue().encode("utf-8")


def export_diff_markdown(diff: DiffIR) -> bytes:
    s = diff_summary(diff)
    lines: List[str] = [
        "# Model Diff Report (synthetic-toolbox generated)",
        "",
        f"- Model A: `{s['a_name']}`",
        f"- Model B: `{s['b_name']}`",
        f"- Topology identical: **{s['topology_identical']}**",
        "",
        "## Node differences",
        "",
        "| kind | count |",
        "|---|---|",
    ]
    for k, v in sorted(s["node_counts"].items()):
        if v:
            lines.append(f"| {k} | {v} |")
    lines += ["", "## Edge differences", ""]
    any_edge = False
    for k, v in sorted(s["edge_counts"].items()):
        if v:
            any_edge = True
            lines.append(f"- {k}: {v}")
    if not any_edge:
        lines.append("- none")
    lines += ["", "## Changed / new / removed nodes", ""]
    interesting = [nd for nd in diff.node_diffs if nd.kind != "unchanged"]
    if interesting:
        lines.append("| kind | A | B | confidence | explanation |")
        lines.append("|---|---|---|---|---|")
        for nd in interesting:
            lines.append(
                f"| {nd.kind} | {nd.a_display or '—'} | {nd.b_display or '—'} "
                f"| {nd.confidence:.2f} | {nd.explanation} |")
    else:
        lines.append("_No structural differences._")
    if diff.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in diff.warnings]
    return ("\n".join(lines) + "\n").encode("utf-8")


def export_diff_html(diff: DiffIR) -> bytes:
    s = diff_summary(diff)
    rows = []
    for nd in diff.node_diffs:
        if nd.kind == "unchanged":
            continue
        rows.append(
            f"<tr class='{nd.kind}'><td>{nd.kind}</td><td>{_esc(nd.a_display)}</td>"
            f"<td>{_esc(nd.b_display)}</td><td>{nd.confidence:.2f}</td>"
            f"<td>{_esc(nd.explanation)}</td></tr>")
    counts = ", ".join(f"{k}: {v}" for k, v in sorted(s["node_counts"].items()) if v)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Diff report: {_esc(s['a_name'])} vs {_esc(s['b_name'])}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #222 }}
table {{ border-collapse: collapse; width: 100% }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 13px }}
tr.added td:first-child {{ color: #0a7d33 }}
tr.removed td:first-child {{ color: #b3261e }}
tr.op_changed, tr.attr_changed, tr.shape_changed, tr.dtype_changed,
tr.param_changed, tr.connection_changed {{ background: #fdecea }}
tr.ambiguous {{ background: #fff3cd }}
</style></head><body>
<h1>Model Diff Report</h1>
<p><b>A:</b> {_esc(s['a_name'])} &nbsp; <b>B:</b> {_esc(s['b_name'])}<br>
Topology identical: <b>{s['topology_identical']}</b></p>
<h2>Summary</h2><p>{_esc(counts) or 'no node differences'}</p>
<h2>Differences</h2>
<table><tr><th>kind</th><th>model A</th><th>model B</th><th>conf</th><th>explanation</th></tr>
{''.join(rows) or '<tr><td colspan=5>no structural differences</td></tr>'}
</table>
<h2>Warnings</h2><ul>{''.join('<li>' + _esc(w) + '</li>' for w in diff.warnings) or '<li>none</li>'}</ul>
</body></html>"""
    return html.encode("utf-8")


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---- profile exports --------------------------------------------------------


def export_profile_json(mapped) -> bytes:
    payload = {
        "profile_schema_version": 1,
        "total_mapped_us": mapped.total_mapped_us,
        "end_to_end_us": mapped.end_to_end_us,
        "coverage": mapped.coverage,
        "model_fingerprint_matched": mapped.model_fingerprint_matched,
        "per_node": [
            {
                "node_id": p.node_id,
                "duration_us": p.duration_us,
                "calls": p.calls,
                "share": p.share,
                "status": p.status,
                "record_indices": p.record_indices,
            }
            for p in mapped.per_node.values()
        ],
        "unmapped": [
            {
                "record_index": u.record_index,
                "node_ref": u.node_ref,
                "ref_kind": u.ref_kind,
                "reason": u.reason,
                "candidates": u.candidates,
            }
            for u in mapped.unmapped
        ],
        "warnings": list(mapped.warnings),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def export_profile_csv(mapped, op_by_node: Dict[str, str] = None) -> bytes:
    op_by_node = op_by_node or {}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_PROFILE_HEADER)
    for p in mapped.per_node.values():
        w.writerow([p.node_id, op_by_node.get(p.node_id, ""), p.duration_us,
                    p.calls, p.share, p.status,
                    " ".join(str(i) for i in p.record_indices)])
    for u in mapped.unmapped:
        w.writerow([u.node_ref or "", "", "", "", "", f"unmapped:{u.reason}",
                    u.record_index])
    return buf.getvalue().encode("utf-8")


def export_profile_markdown(mapped, op_by_node: Dict[str, str] = None) -> bytes:
    op_by_node = op_by_node or {}
    lines = [
        "# Profile Report",
        "",
        f"- Total mapped: {mapped.total_mapped_us:.1f} us",
        f"- End-to-end (if provided): "
        f"{mapped.end_to_end_us if mapped.end_to_end_us is not None else 'n/a'} us",
        f"- Coverage: {mapped.coverage:.1%}",
        f"- Fingerprint matched: {mapped.model_fingerprint_matched}",
        "",
        "| node | op | duration_us | calls | share |",
        "|---|---|---|---|---|",
    ]
    for p in sorted(mapped.per_node.values(), key=lambda x: -x.duration_us):
        lines.append(f"| {p.node_id} | {op_by_node.get(p.node_id, '')} "
                     f"| {p.duration_us:.1f} | {p.calls} | {p.share:.2%} |")
    if mapped.unmapped:
        lines += ["", "## Unmapped records", "",
                  "| record | ref | kind | reason |", "|---|---|---|---|"]
        for u in mapped.unmapped:
            lines.append(f"| {u.record_index} | {u.node_ref or '—'} "
                         f"| {u.ref_kind} | {u.reason} |")
    if mapped.warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in mapped.warnings]
    return ("\n".join(lines) + "\n").encode("utf-8")
