import os
import sys
import textwrap
import time

import pytest

from modeltoolbox.pytorch_importer import (
    PyTorchImportSpec,
    flatten_tree,
    load_pytorch_module,
)

torch_ready = pytest.importorskip  # placeholder to keep flake quiet


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))


@pytest.fixture(scope="module")
def good_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("pt_good")
    _write(os.path.join(root, "mymodels.py"), '''
        import torch
        import torch.nn as nn

        class CustomBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.lin = nn.Linear(dim, dim)

            def forward(self, x):
                return self.lin(x)

        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc = nn.Linear(dim, dim)
                self.act = nn.ReLU()

            def forward(self, x):
                return self.act(self.fc(x))

        class Net(nn.Module):
            def __init__(self, dim=8):
                super().__init__()
                self.blocks = nn.ModuleList([Block(dim), Block(dim)])
                self.custom = CustomBlock(dim)
                self.head = nn.Linear(dim, 2)

            def forward(self, x):
                for b in self.blocks:
                    x = b(x)
                return self.head(self.custom(x))

        class DynamicNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                if x.sum() > 0:        # data-dependent control flow:
                    x = x + 1          # symbolic_trace cannot handle this
                return self.fc(x)
    ''')
    return str(root)


def _spec(project, **kw):
    return PyTorchImportSpec(project_path=project, import_path="mymodels",
                             class_name="Net", timeout_s=90, **kw)


def test_normal_import_tree_structure(good_project):
    res = load_pytorch_module(_spec(good_project))
    assert res.ok, res.error
    tree = res.module_tree
    assert tree["class_name"] == "Net" and tree["path"] == ""
    paths = {n["path"] for n in flatten_tree(tree)}
    # repeated module instances appear as distinct paths
    assert "blocks.0" in paths and "blocks.1" in paths
    assert "custom" in paths and "custom.lin" in paths
    assert "head" in paths
    # parameter accounting
    flat = {n["path"]: n for n in flatten_tree(tree)}
    assert flat["head"]["param_count"] > 0
    assert flat[""]["param_count"] == 0  # root holds no direct params
    assert flat["custom.lin"]["class_name"] == "Linear"
    assert res.fx_graph is not None, res.fx_error
    assert res.degraded is False


def test_leaf_rule_by_class_name(good_project):
    res = load_pytorch_module(_spec(good_project, leaf_rules=[{"class_name": "CustomBlock"}]))
    assert res.ok
    flat = {n["path"]: n for n in flatten_tree(res.module_tree)}
    assert flat["custom"]["is_leaf"] is True
    # flatten_tree prunes children; a leaf must not carry any
    assert "children" not in flat["custom"]
    assert flat["blocks.0"]["is_leaf"] is False


def test_leaf_rule_by_path_pattern(good_project):
    res = load_pytorch_module(_spec(good_project, leaf_rules=[{"path_pattern": "blocks.0"}]))
    assert res.ok
    flat = {n["path"]: n for n in flatten_tree(res.module_tree)}
    assert flat["blocks.0"]["is_leaf"] is True
    assert flat["blocks.1"]["is_leaf"] is False


def test_fx_success_path(good_project):
    res = load_pytorch_module(_spec(good_project))
    fx = res.fx_graph
    assert fx is not None
    names = [n["name"] for n in fx["nodes"]]
    assert "x" in names  # placeholder
    ops = {n["op"] for n in fx["nodes"]}
    assert "call_module" in ops or "call_function" in ops
    assert "output" in ops


def test_fx_failure_degrades(good_project):
    # DynamicNet uses `if self.training` — symbolic_trace raises.
    res = load_pytorch_module(PyTorchImportSpec(
        project_path=good_project, import_path="mymodels",
        class_name="DynamicNet", timeout_s=90))
    assert res.ok is True
    assert res.module_tree is not None
    assert res.fx_graph is None
    assert res.fx_error, "expected a readable fx diagnostic"
    assert res.degraded is True
    assert any("degraded" in w or "FX" in w for w in res.warnings)


def test_broken_project_readable_error(tmp_path):
    _write(os.path.join(tmp_path, "broken.py"), "def broken(:\n    pass\n")
    res = load_pytorch_module(PyTorchImportSpec(
        project_path=str(tmp_path), import_path="broken", class_name="X",
        timeout_s=60))
    assert res.ok is False
    assert res.error and "SyntaxError" in res.error
    assert res.module_tree is None


def test_missing_class_readable_error(good_project):
    res = load_pytorch_module(PyTorchImportSpec(
        project_path=good_project, import_path="mymodels",
        class_name="NoSuchClass", timeout_s=60))
    assert res.ok is False
    assert "NoSuchClass" in res.error


def test_param_validation_no_subprocess(tmp_path):
    res = load_pytorch_module(PyTorchImportSpec(
        project_path=str(tmp_path / "nope"), import_path="m", class_name="C"))
    assert res.ok is False
    assert "project_path" in res.error


def test_timeout_kills_worker(tmp_path):
    _write(os.path.join(tmp_path, "sleeper.py"), '''
        import time

        class Sleeper:
            def __init__(self):
                time.sleep(30)
    ''')
    t0 = time.time()
    res = load_pytorch_module(PyTorchImportSpec(
        project_path=str(tmp_path), import_path="sleeper", class_name="Sleeper",
        timeout_s=2))
    dt = time.time() - t0
    assert res.ok is False
    assert "timed out" in res.error
    assert dt < 15, f"timeout took {dt:.1f}s"


def test_flatten_tree_unique_paths(good_project):
    res = load_pytorch_module(_spec(good_project))
    flat = flatten_tree(res.module_tree)
    paths = [n["path"] for n in flat]
    assert len(paths) == len(set(paths))
    depths = [n["depth"] for n in flat]
    assert depths[0] == 0 and all(d >= 0 for d in depths)
