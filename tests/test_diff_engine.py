import os
import subprocess
import sys

import pytest

from helpers import make_ir
from modeltoolbox.diff_engine import DIFF_KINDS, diff_models
from modeltoolbox.onnx_importer import load_onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "fixtures")

LIN_A = dict(
    nodes=[
        ("conv_1", "Conv", ["x", "w1"], ["h1"], {"strides": [1, 1]}),
        ("relu_1", "Relu", ["h1"], ["h2"]),
        ("conv_2", "Conv", ["h2", "w2"], ["y"]),
    ],
    inputs=[("x", "FLOAT", [1, 1, 8, 8])],
    outputs=[("y", "FLOAT", [1, 1, 8, 8])],
    initializers=[("w1", "FLOAT", [2, 1, 3, 3]), ("w2", "FLOAT", [1, 2, 3, 3])],
)


def _kinds(d):
    return {nd.kind for nd in d.node_diffs}


def test_identical_models_all_unchanged():
    a = make_ir(**LIN_A, model_id="a")
    b = make_ir(**LIN_A, model_id="b")
    d = diff_models(a, b)
    assert _kinds(d) == {"unchanged"}
    assert d.summary["unchanged"] == 3
    assert d.topology_identical is True
    assert all(nd.changes == [] for nd in d.node_diffs)


def test_renamed_structure_unchanged():
    a = make_ir(**LIN_A, model_id="a")
    b = make_ir(
        nodes=[("c1r", "Conv", ["xr", "w1r"], ["h1r"], {"strides": [1, 1]}),
               ("r1r", "Relu", ["h1r"], ["h2r"]),
               ("c2r", "Conv", ["h2r", "w2r"], ["yr"])],
        inputs=[("xr", "FLOAT", [1, 1, 8, 8])],
        outputs=[("yr", "FLOAT", [1, 1, 8, 8])],
        initializers=[("w1r", "FLOAT", [2, 1, 3, 3]), ("w2r", "FLOAT", [1, 2, 3, 3])],
        model_id="b",
    )
    d = diff_models(a, b)
    assert _kinds(d) == {"unchanged"}
    assert d.topology_identical is True
    # names differ but no field-level changes may be reported
    assert all(nd.changes == [] for nd in d.node_diffs)


def test_added_removed_op_replaced():
    a = make_ir(**LIN_A, model_id="a")
    b = make_ir(
        nodes=LIN_A["nodes"] + [("scale_1", "Mul", ["y"], ["y2"])],
        inputs=LIN_A["inputs"], outputs=[("y2", "FLOAT", [1, 1, 8, 8])],
        initializers=LIN_A["initializers"], model_id="b",
    )
    d = diff_models(a, b)
    kinds = _kinds(d)
    assert "added" in kinds and "unchanged" in kinds
    added = [nd for nd in d.node_diffs if nd.kind == "added"]
    assert len(added) == 1 and added[0].b_display == "scale_1"
    assert added[0].a_node_id is None and added[0].match_id is None
    assert d.summary["added"] == 1
    # edges around the added node: old output edge gone, new edges appear
    assert d.summary.get("edge_added", 0) >= 1
    assert d.summary.get("edge_removed", 0) >= 1

    d2 = diff_models(b, a)
    removed = [nd for nd in d2.node_diffs if nd.kind == "removed"]
    assert len(removed) == 1 and removed[0].a_display == "scale_1"

    # op replaced
    b_op = make_ir(
        nodes=[("conv_1", "Conv", ["x", "w1"], ["h1"], {"strides": [1, 1]}),
               ("lrelu", "LeakyRelu", ["h1"], ["h2"]),
               ("conv_2", "Conv", ["h2", "w2"], ["y"])],
        inputs=LIN_A["inputs"], outputs=LIN_A["outputs"],
        initializers=LIN_A["initializers"], model_id="b",
    )
    d3 = diff_models(a, b_op)
    op_changed = [nd for nd in d3.node_diffs if nd.kind == "op_changed"]
    assert len(op_changed) == 1
    assert op_changed[0].changes[0].field == "op_type"
    assert op_changed[0].changes[0].before == "Relu"
    assert op_changed[0].changes[0].after == "LeakyRelu"
    assert "Relu -> LeakyRelu" in op_changed[0].explanation


def test_attr_shape_dtype_param_changes_isolated():
    base = LIN_A
    # attr only
    a = make_ir(**base, model_id="a")
    b = make_ir(
        nodes=[("conv_1", "Conv", ["x", "w1"], ["h1"], {"strides": [2, 2]}),
               ("relu_1", "Relu", ["h1"], ["h2"]),
               ("conv_2", "Conv", ["h2", "w2"], ["y"])],
        inputs=base["inputs"], outputs=base["outputs"],
        initializers=base["initializers"], model_id="b",
    )
    d = diff_models(a, b)
    assert _kinds(d) == {"attr_changed", "unchanged"}
    ac = [nd for nd in d.node_diffs if nd.kind == "attr_changed"][0]
    assert ac.changes[0].field == "attr:strides"
    assert ac.changes[0].before == [1, 1] and ac.changes[0].after == [2, 2]

    # dtype only
    b2 = make_ir(
        nodes=[("conv_1", "Conv", ["x", "w1"], ["h1"], {"strides": [1, 1]}),
               ("relu_1", "Relu", ["h1"], ["h2"]),
               ("conv_2", "Conv", ["h2", "w2"], ["y"])],
        inputs=[("x", "DOUBLE", [1, 1, 8, 8])],
        outputs=[("y", "DOUBLE", [1, 1, 8, 8])],
        initializers=[("w1", "DOUBLE", [2, 1, 3, 3]), ("w2", "DOUBLE", [1, 2, 3, 3])],
        model_id="b",
    )
    d2 = diff_models(a, b2)
    kinds2 = _kinds(d2)
    assert "dtype_changed" in kinds2
    dc = [nd for nd in d2.node_diffs if nd.kind == "dtype_changed"]
    assert all(any(c.field.startswith("dtype:") for c in nd.changes) for nd in dc)

    # shape only
    b3 = make_ir(
        nodes=base["nodes"],
        inputs=[("x", "FLOAT", [1, 1, 16, 16])],
        outputs=[("y", "FLOAT", [1, 1, 16, 16])],
        initializers=base["initializers"], model_id="b",
    )
    d3 = diff_models(a, b3)
    kinds3 = _kinds(d3)
    assert "shape_changed" in kinds3
    assert "dtype_changed" not in kinds3

    # param shape only
    b4 = make_ir(
        nodes=[("conv_1", "Conv", ["x", "w1"], ["h1"], {"strides": [1, 1]}),
               ("relu_1", "Relu", ["h1"], ["h2"]),
               ("conv_2", "Conv", ["h2", "w2"], ["y"])],
        inputs=base["inputs"], outputs=base["outputs"],
        initializers=[("w1", "FLOAT", [4, 1, 3, 3]), ("w2", "FLOAT", [1, 2, 3, 3])],
        model_id="b",
    )
    d4 = diff_models(a, b4)
    kinds4 = _kinds(d4)
    assert "param_changed" in kinds4
    pc = [nd for nd in d4.node_diffs if nd.kind == "param_changed"][0]
    assert pc.changes[0].field == "param[0]"
    assert pc.changes[0].before[1] == ["FLOAT", (2, 1, 3, 3)]
    assert pc.changes[0].after[1] == ["FLOAT", (4, 1, 3, 3)]


def test_connection_changed():
    a = make_ir(
        nodes=[("c1", "Conv", ["x", "w1"], ["m1"], {"strides": [1, 1]}),
               ("c2", "Conv", ["x", "w2"], ["m2"]),
               ("add", "Add", ["m1", "m2"], ["y"])],
        inputs=[("x", "FLOAT", [1, 1, 8, 8])], outputs=[("y", "FLOAT", [1, 1, 8, 8])],
        initializers=[("w1", "FLOAT", [1, 1, 3, 3]), ("w2", "FLOAT", [1, 1, 3, 3])],
        model_id="a",
    )
    b = make_ir(
        nodes=[("c1", "Conv", ["x", "w1"], ["m1"], {"strides": [1, 1]}),
               ("c2", "Conv", ["x", "w2"], ["m2"]),
               ("add", "Add", ["m2", "m1"], ["y"])],
        inputs=[("x", "FLOAT", [1, 1, 8, 8])], outputs=[("y", "FLOAT", [1, 1, 8, 8])],
        initializers=[("w1", "FLOAT", [1, 1, 3, 3]), ("w2", "FLOAT", [1, 1, 3, 3])],
        model_id="b",
    )
    d = diff_models(a, b)
    # swapping Add's input order changes connection context
    conn = [nd for nd in d.node_diffs if nd.kind == "connection_changed"]
    assert len(conn) == 1
    assert conn[0].a_display == "add"
    assert any(c.field == "inputs" for c in conn[0].changes)


def test_unknown_shapes_not_reported_as_changed():
    a = make_ir(nodes=[("n1", "Relu", ["x"], ["y"])],
                inputs=[("x", "FLOAT", None)], outputs=[("y", "FLOAT", None)],
                model_id="a", with_fingerprints=False)
    b = make_ir(nodes=[("n1", "Relu", ["x"], ["y"])],
                inputs=[("x", "FLOAT", None)], outputs=[("y", "FLOAT", None)],
                model_id="b", with_fingerprints=False)
    d = diff_models(a, b)
    assert _kinds(d) == {"unchanged"}


def test_end_to_end_real_fixtures():
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_fixtures.py")],
                   check=True, capture_output=True, cwd=ROOT)
    a = load_onnx(os.path.join(FIXTURES, "linear.onnx"), "fa")
    b = load_onnx(os.path.join(FIXTURES, "renamed_linear.onnx"), "fb")
    d = diff_models(a, b)
    assert _kinds(d) == {"unchanged"}, d.summary
    assert d.topology_identical is True

    b2 = load_onnx(os.path.join(FIXTURES, "op_replaced.onnx"), "fc")
    d2 = diff_models(a, b2)
    kinds = _kinds(d2)
    assert "op_changed" in kinds and "unchanged" in kinds

    b3 = load_onnx(os.path.join(FIXTURES, "attr_changed.onnx"), "fd")
    d3 = diff_models(a, b3)
    # strides [2,2] changes both the attribute AND (legitimately) the output
    # shape; primary kind follows priority (shape_changed), and both changes
    # must be listed at field level.
    attr_nd = next(nd for nd in d3.node_diffs
                   if any(c.field == "attr:strides" for c in nd.changes))
    fields = {c.field for c in attr_nd.changes}
    assert "attr:strides" in fields and "shape_changed" in _kinds(d3)
    strides = next(c for c in attr_nd.changes if c.field == "attr:strides")
    assert strides.after == [2, 2]

    b4 = load_onnx(os.path.join(FIXTURES, "node_added.onnx"), "fe")
    d4 = diff_models(a, b4)
    assert d4.summary["added"] == 1

    b5 = load_onnx(os.path.join(FIXTURES, "shape_changed.onnx"), "ff")
    d5 = diff_models(a, b5)
    assert "shape_changed" in _kinds(d5)

    b6 = load_onnx(os.path.join(FIXTURES, "weights_differ.onnx"), "fg")
    d6 = diff_models(a, b6)
    assert _kinds(d6) == {"unchanged"}  # same structure, weights not compared


def test_roundtrip_json_and_determinism():
    a = make_ir(**LIN_A, model_id="a")
    b = make_ir(
        nodes=LIN_A["nodes"] + [("extra", "Mul", ["y"], ["y2"])],
        inputs=LIN_A["inputs"], outputs=[("y2", "FLOAT", [1, 1, 8, 8])],
        initializers=LIN_A["initializers"], model_id="b",
    )
    d1 = diff_models(a, b)
    d2 = diff_models(a, b)
    assert d1.to_json() == d2.to_json()
    d3 = type(d1).from_json(d1.to_json())
    assert d3.to_json() == d1.to_json()
    assert set(DIFF_KINDS) >= set(d1.summary.keys()) - {"edge_added", "edge_removed",
                                                        "edge_changed", "edge_unchanged"}
