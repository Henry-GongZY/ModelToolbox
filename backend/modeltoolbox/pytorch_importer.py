"""Parent-side PyTorch importer: manages the isolated worker subprocess.

This module NEVER imports torch. It spawns
``python -m modeltoolbox.pytorch_worker`` with a JSON request, enforces a
hard timeout (kills the whole process group), captures stderr tails, and
degrades gracefully (module tree without FX graph) when tracing fails.

Security: subprocess isolation + timeout is process separation, not a
sandbox. Only import local PyTorch projects the user explicitly trusts.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PyTorchImportSpec:
    project_path: str
    import_path: str
    class_name: str
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    input_shapes: Dict[str, List] = field(default_factory=dict)
    leaf_rules: List[Dict[str, str]] = field(default_factory=list)
    trace: bool = True
    timeout_s: float = 60.0


@dataclass
class PyTorchResult:
    ok: bool
    module_tree: Optional[dict]
    fx_graph: Optional[dict]
    fx_error: Optional[str]
    error: Optional[str]
    stderr_tail: str
    degraded: bool
    warnings: List[str]


def is_torch_available() -> bool:
    """Probe torch availability in a throwaway worker subprocess."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, timeout=120,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _validate(spec: PyTorchImportSpec) -> Optional[str]:
    import os
    if not spec.project_path or not os.path.isdir(spec.project_path):
        return f"project_path is not a directory: '{spec.project_path}'"
    if not spec.import_path or not isinstance(spec.import_path, str):
        return "import_path must be a non-empty string (module or 'pkg.module')"
    if not spec.class_name or not isinstance(spec.class_name, str):
        return "class_name must be a non-empty string"
    return None


def load_pytorch_module(spec: PyTorchImportSpec) -> PyTorchResult:
    err = _validate(spec)
    if err:
        return PyTorchResult(ok=False, module_tree=None, fx_graph=None,
                             fx_error=None, error=err, stderr_tail="",
                             degraded=False, warnings=[])

    request = {
        "op": "trace",
        "project_path": spec.project_path,
        "import_path": spec.import_path,
        "class_name": spec.class_name,
        "init_kwargs": spec.init_kwargs,
        "input_shapes": spec.input_shapes,
        "leaf_rules": spec.leaf_rules,
        "trace": spec.trace,
    }
    warnings: List[str] = []
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "modeltoolbox.pytorch_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            start_new_session=True,  # own process group for clean kill
        )
    except OSError as e:
        return PyTorchResult(ok=False, module_tree=None, fx_graph=None,
                             fx_error=None, error=f"cannot spawn worker: {e}",
                             stderr_tail="", degraded=False, warnings=[])

    try:
        out, errtext = proc.communicate(json.dumps(request), timeout=spec.timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        return PyTorchResult(
            ok=False, module_tree=None, fx_graph=None, fx_error=None,
            error=f"timed out after {spec.timeout_s:.0f}s while importing "
                  f"'{spec.import_path}.{spec.class_name}'; worker killed",
            stderr_tail="(worker killed after timeout)", degraded=False,
            warnings=warnings)

    stderr_tail = (errtext or "")[-4096:]
    if stderr_tail.strip():
        # torch import warnings etc. — surfaced, never swallowed silently
        warnings.append("worker stderr output captured (see stderr_tail)")

    if not out.strip():
        return PyTorchResult(
            ok=False, module_tree=None, fx_graph=None, fx_error=None,
            error=f"worker produced no response (exit code {returncode})",
            stderr_tail=stderr_tail, degraded=False, warnings=warnings)

    try:
        resp = json.loads(out.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return PyTorchResult(
            ok=False, module_tree=None, fx_graph=None, fx_error=None,
            error="worker response was not valid JSON", stderr_tail=stderr_tail,
            degraded=False, warnings=warnings)

    ok = bool(resp.get("ok"))
    degraded = bool(ok and resp.get("module_tree") and not resp.get("fx_graph")
                    and resp.get("fx_error"))
    if resp.get("fx_error"):
        warnings.append("FX trace failed; showing module tree only (degraded view)")
    return PyTorchResult(
        ok=ok,
        module_tree=resp.get("module_tree"),
        fx_graph=resp.get("fx_graph"),
        fx_error=resp.get("fx_error"),
        error=resp.get("error"),
        stderr_tail=resp.get("stderr_tail") or stderr_tail,
        degraded=degraded,
        warnings=warnings,
    )


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        import os
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def flatten_tree(tree: dict) -> List[dict]:
    """Pre-order flatten; each item keeps 'path' and 'depth'."""
    out: List[dict] = []

    def walk(node: dict, depth: int) -> None:
        item = dict(node)
        item["depth"] = depth
        item.pop("children", None)
        out.append(item)
        for c in node.get("children", []) or []:
            walk(c, depth + 1)

    walk(tree, 0)
    return out
