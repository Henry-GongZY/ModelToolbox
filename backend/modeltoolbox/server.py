"""Local HTTP server: static frontend + JSON API.

Run: .venv/bin/python -m modeltoolbox.server [--port 8765]
Single-user local tool. Binds 127.0.0.1 by default. No telemetry; model and
profile files are read locally and never sent anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from modeltoolbox.diff_engine import DiffIR, diff_models
from modeltoolbox.folding import apply_folding, find_common_groups
from modeltoolbox.layout import layout_diff_pair, layout_single
from modeltoolbox.matcher import match_graphs
from modeltoolbox.onnx_importer import ModelLoadError, load_onnx, load_onnx_bytes
from modeltoolbox.profile_adapter import ProfileFormatError, load_profile, load_profile_text
from modeltoolbox.profile_mapper import map_profile
from modeltoolbox.pytorch_importer import (
    PyTorchImportSpec,
    is_torch_available,
    load_pytorch_module,
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "frontend")

MAX_BODY = 64 * 1024 * 1024  # 64MB request cap


class _Store:
    """In-memory session store (single-user local tool)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.models: Dict[str, dict] = {}        # model_id -> {"ir": GraphIR, "ts": float}
        self.diffs: Dict[str, dict] = {}         # diff_id -> {...}
        self.profiles: Dict[str, dict] = {}      # profile_id -> {...}
        self.pytorch: Dict[str, dict] = {}       # pt_id -> result dict

    def gc(self, max_age_s: float = 3600 * 6):
        now = time.time()
        for bucket in (self.models, self.diffs, self.profiles, self.pytorch):
            stale = [k for k, v in bucket.items() if now - v.get("ts", now) > max_age_s]
            for k in stale:
                del bucket[k]


STORE = _Store()


class Handler(BaseHTTPRequestHandler):
    server_version = "ModelToolbox/0.1"

    # ---- plumbing ----
    def log_message(self, fmt, *args):  # quiet by default
        pass

    def _json(self, obj: Any, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400, kind: str = "bad_request"):
        self._json({"error": {"kind": kind, "message": message}}, status)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError(f"request body too large ({length} bytes)")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            obj = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON body: {e}")
        if not isinstance(obj, dict):
            raise ValueError("JSON body must be an object")
        return obj

    # ---- static ----
    def _serve_static(self, path: str):
        if path == "/":
            path = "/index.html"
        fp = os.path.normpath(os.path.join(FRONTEND_DIR, path.lstrip("/")))
        if not fp.startswith(FRONTEND_DIR) or not os.path.isfile(fp):
            self._error(f"not found: {path}", 404, "not_found")
            return
        ctype = {
            ".html": "text/html; charset=utf-8", ".js": "text/javascript",
            ".css": "text/css", ".svg": "image/svg+xml", ".json": "application/json",
        }.get(os.path.splitext(fp)[1], "application/octet-stream")
        with open(fp, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---- routing ----
    def do_GET(self):
        try:
            p = urlparse(self.path).path
            if p == "/api/health":
                self._json({"ok": True, "service": "modeltoolbox",
                            "torch_available": is_torch_available(),
                            "sessions": {k: len(v) for k, v in (
                                ("models", STORE.models), ("diffs", STORE.diffs),
                                ("profiles", STORE.profiles))}})
            elif p.startswith("/api/"):
                self._error("unknown API endpoint", 404, "not_found")
            else:
                self._serve_static(p)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self._error(f"internal error: {e}", 500, "internal")

    def do_DELETE(self):
        try:
            p = urlparse(self.path).path
            parts = p.strip("/").split("/")
            if len(parts) == 3 and parts[1] in ("model", "diff", "profile"):
                bucket = {"model": STORE.models, "diff": STORE.diffs,
                          "profile": STORE.profiles}[parts[1]]
                with STORE.lock:
                    removed = bucket.pop(parts[2], None)
                if removed is None:
                    self._error("no such session id", 404, "not_found")
                else:
                    self._json({"ok": True, "deleted": parts[2]})
            else:
                self._error("unknown endpoint", 404, "not_found")
        except Exception as e:
            self._error(f"internal error: {e}", 500, "internal")

    def do_POST(self):
        try:
            p = urlparse(self.path).path
            try:
                body = self._read_json_body()
            except ValueError as e:
                self._error(str(e), 400)
                return
            routes = {
                "/api/model/load": self._model_load,
                "/api/model/upload": self._model_upload,
                "/api/model/nodes": self._model_nodes,
                "/api/model/layout": self._model_layout,
                "/api/diff": self._diff,
                "/api/fold": self._fold,
                "/api/profile/load": self._profile_load,
                "/api/profile/upload": self._profile_upload,
                "/api/pytorch/load": self._pytorch_load,
                "/api/export/diff": self._export_diff,
                "/api/export/profile": self._export_profile,
            }
            handler = routes.get(p)
            if handler is None:
                self._error("unknown API endpoint", 404, "not_found")
                return
            handler(body)
        except ModelLoadError as e:
            self._error(str(e), 422, "model_load_error")
        except ProfileFormatError as e:
            self._error(str(e), 422, "profile_format_error")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._error(f"internal error: {type(e).__name__}: {e}", 500, "internal")

    # ---- handlers ----
    def _model_upload(self, body):
        """Load a model from uploaded bytes (Netron-style file picker)."""
        import base64
        name = body.get("name") or "uploaded.onnx"
        data_b64 = body.get("data_base64") or ""
        if not data_b64:
            self._error("'data_base64' is required"); return
        try:
            data = base64.b64decode(data_b64, validate=True)
        except Exception as e:
            self._error(f"invalid base64 payload: {e}"); return
        model_id = uuid.uuid4().hex[:12]
        ir = load_onnx_bytes(data, model_id, source_path=name)
        layout = layout_single(ir)
        with STORE.lock:
            STORE.models[model_id] = {"ir": ir, "ts": time.time()}
            STORE.gc()
        self._json(self._model_payload(ir, layout))

    def _model_nodes(self, body):
        """On-demand node details (used by expandable graph nodes)."""
        model_id = body.get("model_id")
        ids = body.get("ids") or []
        with STORE.lock:
            rec = STORE.models.get(model_id) if model_id else None
        if rec is None:
            self._error(f"unknown or missing model_id: {model_id!r}"); return
        ir = rec["ir"]
        want = set(ids)
        nodes = [n.to_dict() if hasattr(n, "to_dict") else _node_dict(n)
                 for n in ir.nodes if n.id in want]
        self._json({"model_id": model_id, "nodes": nodes})

    def _profile_upload(self, body):
        """Load a profile from uploaded text (Netron-style file picker)."""
        import base64
        name = body.get("name") or "uploaded profile"
        data_b64 = body.get("data_base64") or ""
        if not data_b64:
            self._error("'data_base64' is required"); return
        try:
            text = base64.b64decode(data_b64, validate=True).decode("utf-8-sig")
        except Exception as e:
            self._error(f"cannot decode uploaded profile: {e}"); return
        model_id = body.get("model_id")
        ir = None
        if model_id:
            with STORE.lock:
                rec = STORE.models.get(model_id)
                ir = rec["ir"] if rec else None
            if ir is None:
                self._error(f"unknown model_id: {model_id!r}"); return
        profile = load_profile_text(text, source_path=name)
        mapped = map_profile(profile, ir) if ir is not None else None
        profile_id = uuid.uuid4().hex[:12]
        with STORE.lock:
            STORE.profiles[profile_id] = {"profile": profile, "mapped": mapped,
                                          "model_id": model_id, "ts": time.time()}
            STORE.gc()
        self._json(self._profile_payload(profile_id, profile, mapped, ir))

    @staticmethod
    def _model_payload(ir, layout):
        return {
            "model_id": ir.model_id,
            "meta": {
                "name": ir.name, "producer": ir.producer,
                "source_path": ir.source_path,
                "file_size_bytes": ir.file_size_bytes,
                "opsets": ir.opsets, "ir_version": ir.ir_version,
                "node_count": len(ir.nodes), "edge_count": len(ir.edges),
                "param_count": sum(p.num_elements or 0 for p in ir.initializers),
                "fingerprints": ir.fingerprints,
                "inputs": [vars(v) for v in ir.inputs],
                "outputs": [vars(v) for v in ir.outputs],
                "warnings": ir.warnings,
            },
            "ir": ir.to_dict(),
            "layout": _layout_dict(layout),
        }

    @staticmethod
    def _profile_payload(profile_id, profile, mapped, ir):
        resp = {
            "profile_id": profile_id,
            "summary": {
                "format": profile.format, "unit": profile.unit,
                "model_name": profile.model_name, "device": profile.device,
                "record_count": len(profile.records),
                "end_to_end_us": profile.end_to_end_us,
                "model_fingerprint": profile.model_fingerprint,
                "warnings": profile.warnings,
            },
            "records": [
                {"index": r.index, "node_ref": r.node_ref, "ref_kind": r.ref_kind,
                 "op_type": r.op_type, "duration_us": r.duration_us,
                 "calls": r.calls, "fused": r.fused, "kernel_name": r.kernel_name}
                for r in profile.records
            ],
        }
        if mapped is not None:
            nodes_by_id = {n.id: n for n in ir.nodes}
            resp["mapped"] = {
                "per_node": {nid: {"node_id": p.node_id,
                                   "duration_us": p.duration_us,
                                   "calls": p.calls, "share": p.share,
                                   "record_indices": p.record_indices}
                             for nid, p in mapped.per_node.items()},
                "unmapped": [vars(u) for u in mapped.unmapped],
                "coverage": mapped.coverage,
                "total_mapped_us": mapped.total_mapped_us,
                "end_to_end_us": mapped.end_to_end_us,
                "model_fingerprint_matched": mapped.model_fingerprint_matched,
                "warnings": mapped.warnings,
            }
            resp["op_by_node"] = {nid: n.op_type for nid, n in nodes_by_id.items()}
        return resp

    def _model_load(self, body):
        path = body.get("path")
        if not path or not isinstance(path, str):
            self._error("'path' is required"); return
        model_id = uuid.uuid4().hex[:12]
        ir = load_onnx(path, model_id)
        layout = layout_single(ir)
        with STORE.lock:
            STORE.models[model_id] = {"ir": ir, "ts": time.time()}
            STORE.gc()
        self._json(self._model_payload(ir, layout))

    def _model_layout(self, body):
        ir = self._get_model(body)
        layout = layout_single(ir)
        self._json({"model_id": ir.model_id, "layout": _layout_dict(layout)})

    def _get_model(self, body):
        model_id = body.get("model_id")
        with STORE.lock:
            rec = STORE.models.get(model_id) if model_id else None
        if rec is None:
            raise KeyError(f"unknown or missing model_id: {model_id!r}")
        return rec["ir"]

    def _diff(self, body):
        import base64 as _b64
        path_a, path_b = body.get("path_a"), body.get("path_b")
        data_a, data_b = body.get("data_a"), body.get("data_b")
        if data_a and data_b:
            # uploaded bytes (Netron-style picker); optional display names
            name_a = body.get("name_a") or "model_a.onnx"
            name_b = body.get("name_b") or "model_b.onnx"
            try:
                a = load_onnx_bytes(_b64.b64decode(data_a, validate=True),
                                    uuid.uuid4().hex[:12], source_path=name_a)
                b = load_onnx_bytes(_b64.b64decode(data_b, validate=True),
                                    uuid.uuid4().hex[:12], source_path=name_b)
            except Exception as e:
                if isinstance(e, ModelLoadError):
                    raise
                raise ModelLoadError(f"invalid uploaded model payload: {e}")
        elif path_a and path_b:
            a = load_onnx(path_a, uuid.uuid4().hex[:12])
            b = load_onnx(path_b, uuid.uuid4().hex[:12])
        else:
            self._error("'data_a/data_b' (uploads) or 'path_a/path_b' are required")
            return
        match = match_graphs(a, b)
        diff = diff_models(a, b)
        la, lb = layout_diff_pair(a, b, match.matches)
        folding = find_common_groups(
            a, b, match.matches,
            min_group_size=int(body.get("min_group_size", 3)),
            min_confidence=float(body.get("min_confidence", 0.9)))
        diff_id = uuid.uuid4().hex[:12]
        with STORE.lock:
            STORE.diffs[diff_id] = {"a": a, "b": b, "match": match,
                                    "diff": diff, "folding": folding,
                                    "ts": time.time()}
            # expose the pair as loadable models so profiles can map to them
            STORE.models[a.model_id] = {"ir": a, "ts": time.time()}
            STORE.models[b.model_id] = {"ir": b, "ts": time.time()}
            STORE.gc()
        self._json({
            "diff_id": diff_id,
            "a_model_id": a.model_id, "b_model_id": b.model_id,
            "match": match.to_dict(),
            "diff": diff.to_dict(),
            "folding": _folding_dict(folding),
            "layouts": {"a": _layout_dict(la), "b": _layout_dict(lb)},
            # light node summaries so the UI can render Netron-style nodes
            "nodes_a": {n.id: {"op": n.op_type, "display": n.display_name,
                                "outputs": [t for t in n.outputs if t]}
                        for n in a.nodes},
            "nodes_b": {n.id: {"op": n.op_type, "display": n.display_name,
                                "outputs": [t for t in n.outputs if t]}
                        for n in b.nodes},
        })

    def _fold(self, body):
        diff_id = body.get("diff_id")
        with STORE.lock:
            rec = STORE.diffs.get(diff_id) if diff_id else None
        if rec is None:
            self._error(f"unknown or missing diff_id: {diff_id!r}"); return
        folded = set(body.get("folded_group_ids") or [])
        folding = rec["folding"]
        va = apply_folding(rec["a"], folding, folded, "a")
        vb = apply_folding(rec["b"], folding, folded, "b")
        self._json({
            "diff_id": diff_id,
            "folded": sorted(folded),
            "views": {
                "a": {"display_nodes": va.display_nodes,
                      "display_edges": va.display_edges,
                      "hidden_count": len(va.hidden_nodes)},
                "b": {"display_nodes": vb.display_nodes,
                      "display_edges": vb.display_edges,
                      "hidden_count": len(vb.hidden_nodes)},
            },
        })

    def _profile_load(self, body):
        path = body.get("path")
        model_id = body.get("model_id")
        if not path:
            self._error("'path' is required"); return
        ir = None
        if model_id:
            with STORE.lock:
                rec = STORE.models.get(model_id)
                ir = rec["ir"] if rec else None
            if ir is None:
                self._error(f"unknown model_id: {model_id!r}"); return
        profile = load_profile(path)
        mapped = map_profile(profile, ir) if ir is not None else None
        profile_id = uuid.uuid4().hex[:12]
        with STORE.lock:
            STORE.profiles[profile_id] = {"profile": profile, "mapped": mapped,
                                          "model_id": model_id, "ts": time.time()}
            STORE.gc()
        self._json(self._profile_payload(profile_id, profile, mapped, ir))

    def _pytorch_load(self, body):
        try:
            spec = PyTorchImportSpec(
                project_path=body.get("project_path") or "",
                import_path=body.get("import_path") or "",
                class_name=body.get("class_name") or "",
                init_kwargs=body.get("init_kwargs") or {},
                input_shapes=body.get("input_shapes") or {},
                leaf_rules=body.get("leaf_rules") or [],
                trace=bool(body.get("trace", True)),
                timeout_s=float(body.get("timeout_s", 60)),
            )
        except (TypeError, ValueError) as e:
            self._error(f"invalid pytorch spec: {e}"); return
        result = load_pytorch_module(spec)
        pt_id = uuid.uuid4().hex[:12]
        with STORE.lock:
            STORE.pytorch[pt_id] = {"result": result, "ts": time.time()}
        self._json({
            "pt_id": pt_id, "ok": result.ok,
            "module_tree": result.module_tree,
            "fx_graph": result.fx_graph, "fx_error": result.fx_error,
            "degraded": result.degraded,
            "error": result.error,
            "warnings": result.warnings,
        })

    def _export_diff(self, body):
        diff_id = body.get("diff_id")
        fmt = body.get("format", "json")
        with STORE.lock:
            rec = STORE.diffs.get(diff_id) if diff_id else None
        if rec is None:
            self._error(f"unknown or missing diff_id: {diff_id!r}"); return
        diff: DiffIR = rec["diff"]
        payload, ctype, ext = _diff_export(diff, fmt)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Disposition",
                         f"attachment; filename=diff_report.{ext}")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _export_profile(self, body):
        profile_id = body.get("profile_id")
        fmt = body.get("format", "json")
        with STORE.lock:
            rec = STORE.profiles.get(profile_id) if profile_id else None
        if rec is None or rec["mapped"] is None:
            self._error("unknown profile_id (or profile was loaded without a "
                        "model so nothing was mapped)"); return
        mapped = rec["mapped"]
        ir = STORE.models.get(rec["model_id"], {}).get("ir") if rec.get("model_id") else None
        op_by_node = {n.id: n.op_type for n in ir.nodes} if ir else {}
        from modeltoolbox import export as ex
        if fmt == "csv":
            payload, ctype, ext = ex.export_profile_csv(mapped, op_by_node), "text/csv", "csv"
        elif fmt == "md":
            payload, ctype, ext = ex.export_profile_markdown(mapped, op_by_node), "text/markdown", "md"
        elif fmt == "html":
            md = ex.export_profile_markdown(mapped, op_by_node).decode("utf-8")
            payload, ctype, ext = ("<pre>" + md + "</pre>").encode(), "text/html", "html"
        else:
            payload, ctype, ext = ex.export_profile_json(mapped), "application/json", "json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Disposition",
                         f"attachment; filename=profile_report.{ext}")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _diff_export(diff: DiffIR, fmt: str):
    from modeltoolbox import export as ex
    if fmt == "csv":
        return ex.export_diff_csv(diff), "text/csv", "csv"
    if fmt == "md":
        return ex.export_diff_markdown(diff), "text/markdown", "md"
    if fmt == "html":
        return ex.export_diff_html(diff), "text/html", "html"
    if fmt == "json":
        return ex.export_diff_json(diff), "application/json", "json"
    raise ValueError(f"unsupported export format: {fmt!r}")


def _node_dict(n):
    from dataclasses import asdict
    return asdict(n)


def _layout_dict(lr) -> dict:
    return {
        "boxes": {nid: vars(b) for nid, b in lr.boxes.items()},
        "width": lr.width, "height": lr.height,
        "ranks": lr.ranks,
        "edge_points": lr.edge_points,
    }


def _folding_dict(folding) -> dict:
    from dataclasses import asdict
    return {
        "groups": [asdict(g) for g in folding.groups],
        "node_to_group": folding.node_to_group,
        "warnings": folding.warnings,
    }


def main():
    parser = argparse.ArgumentParser(description="ModelToolbox local server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ModelToolbox serving on http://{args.host}:{args.port} (Ctrl-C to stop)",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
