"""PyTorch worker: runs INSIDE an isolated subprocess (never imported by the
parent process). Reads one JSON request on stdin, writes one JSON response on
stdout, always exits 0 on protocol completion.

Security note: this worker executes user-provided Python code (importing the
target project). Subprocess isolation with a hard timeout is process
separation, NOT a sandbox — only point it at local projects you trust.
"""

from __future__ import annotations

import fnmatch
import json
import sys
import traceback
from typing import Any, Dict, List, Optional


def _module_tree(module, leaf_rules: List[Dict[str, str]]) -> Dict[str, Any]:
    """Serialize nn.Module hierarchy. leaf_rules mark whole-subtree leaves."""

    def is_leaf(path: str, cls_name: str) -> bool:
        for rule in leaf_rules:
            if not isinstance(rule, dict):
                continue
            cn = rule.get("class_name")
            if cn and cls_name == cn:
                return True
            pat = rule.get("path_pattern")
            if pat and fnmatch.fnmatch(path, pat):
                return True
        return False

    def params_meta(mod):
        count = 0
        total = 0
        for p in mod.parameters(recurse=False):
            count += 1
            try:
                total += p.numel() * p.element_size()
            except Exception:
                pass
        return count

    def bytes_meta(mod):
        total = 0
        for p in mod.parameters(recurse=False):
            try:
                total += p.numel() * p.element_size()
            except Exception:
                pass
        for b in mod.buffers(recurse=False):
            try:
                total += b.numel() * b.element_size()
            except Exception:
                pass
        return total

    def walk(mod, path: str) -> Dict[str, Any]:
        cls_name = type(mod).__name__
        node: Dict[str, Any] = {
            "path": path,
            "class_name": cls_name,
            "module_type": cls_name,
            "param_count": params_meta(mod),
            "param_bytes": bytes_meta(mod),
            "buffer_count": len(list(mod.buffers(recurse=False))),
            "is_leaf": False,
            "children": [],
        }
        if path and is_leaf(path, cls_name):
            node["is_leaf"] = True
            return node
        children = []
        for name, child in mod.named_children():
            child_path = f"{path}.{name}" if path else name
            children.append(walk(child, child_path))
        node["children"] = children
        return node

    return walk(module, "")


def _fx_graph(module) -> Optional[Dict[str, Any]]:
    import torch  # noqa: F401  (worker-local)
    from torch.fx import symbolic_trace

    gm = symbolic_trace(module)
    nodes = []
    for n in gm.graph.nodes:
        def fmt(a):
            if isinstance(a, torch.fx.Node):
                return a.name
            if isinstance(a, (list, tuple)):
                return [fmt(x) for x in a]
            if isinstance(a, dict):
                return {k: fmt(v) for k, v in a.items()}
            if isinstance(a, torch.Tensor):
                return f"<tensor {tuple(a.shape)}>"
            if isinstance(a, (int, float, bool, str)) or a is None:
                return a
            return str(a)

        nodes.append({
            "name": n.name,
            "op": n.op,
            "target": str(n.target),
            "args": [fmt(a) for a in n.args],
            "kwargs": {k: fmt(v) for k, v in n.kwargs.items()},
        })
    return {"nodes": nodes}


def handle(request: Dict[str, Any]) -> Dict[str, Any]:
    project_path = request.get("project_path") or ""
    import_path = request.get("import_path") or ""
    class_name = request.get("class_name") or ""

    sys.path.insert(0, project_path)
    try:
        import importlib

        mod_ref = importlib.import_module(import_path)
        cls = getattr(mod_ref, class_name, None)
        if cls is None:
            raise AttributeError(
                f"module '{import_path}' has no attribute '{class_name}'")
        init_kwargs = request.get("init_kwargs") or {}
        module = cls(**init_kwargs)
        module.eval()

        tree = _module_tree(module, request.get("leaf_rules") or [])

        fx_graph = None
        fx_error = None
        if request.get("trace", True):
            try:
                fx_graph = _fx_graph(module)
            except Exception as e:  # dynamic control flow etc. -> degrade
                fx_error = f"{type(e).__name__}: {e}"

        return {
            "ok": True, "module_tree": tree, "fx_graph": fx_graph,
            "fx_error": fx_error, "error": None, "stderr_tail": "",
        }
    except Exception as e:
        return {
            "ok": False, "module_tree": None, "fx_graph": None,
            "fx_error": None,
            "error": f"{type(e).__name__}: {e}",
            "stderr_tail": traceback.format_exc()[-4096:],
        }


def main() -> int:
    try:
        line = sys.stdin.readline()
        request = json.loads(line)
        response = handle(request)
    except Exception as e:  # never die without answering
        response = {
            "ok": False, "module_tree": None, "fx_graph": None,
            "fx_error": None,
            "error": f"worker protocol failure: {type(e).__name__}: {e}",
            "stderr_tail": traceback.format_exc()[-4096:],
        }
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
