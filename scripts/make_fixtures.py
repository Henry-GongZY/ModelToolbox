#!/usr/bin/env python
"""Generate deterministic synthetic ONNX fixtures + sample profiles into fixtures/.

All data is synthetic (generated with fixed seeds / fixed values) — no real
models are used. Run: .venv/bin/python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures")


def _conv_w(name: str, shape, seed: int) -> onnx.TensorProto:
    rng = np.random.RandomState(seed)
    return numpy_helper.from_array(rng.randn(*shape).astype(np.float32) * 0.1, name)


def make_linear(name_suffix: str = "", rename: bool = False, seed: int = 1):
    """x[1,1,8,8] -> Conv -> Relu -> Conv -> y[1,1,8,8]."""
    if rename:
        nodes = [
            helper.make_node("Conv", ["feat_in", "stem_block/conv_fwd.weight"], ["stem_out"], "stem_block/conv_fwd"),
            helper.make_node("Relu", ["stem_out"], ["stem_act"], "stem_block/act_fwd"),
            helper.make_node("Conv", ["stem_act", "head_block/conv_fwd.weight"], ["head_out"], "head_block/conv_fwd"),
        ]
        initializer = [_conv_w("stem_block/conv_fwd.weight", [2, 1, 3, 3], seed),
                       _conv_w("head_block/conv_fwd.weight", [1, 2, 3, 3], seed + 1)]
        graph_in, graph_out = "feat_in", "head_out"
    else:
        nodes = [
            helper.make_node("Conv", ["x", "conv_1.weight"], ["h1"], "conv_1"),
            helper.make_node("Relu", ["h1"], ["h2"], "relu_1"),
            helper.make_node("Conv", ["h2", "conv_2.weight"], ["y"], "conv_2"),
        ]
        initializer = [_conv_w("conv_1.weight", [2, 1, 3, 3], seed),
                       _conv_w("conv_2.weight", [1, 2, 3, 3], seed + 1)]
        graph_in, graph_out = "x", "y"

    graph = helper.make_graph(
        nodes, f"linear_{name_suffix or 'model'}",
        [helper.make_tensor_value_info(graph_in, TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info(graph_out, TensorProto.FLOAT, [1, 1, 8, 8])],
        initializer=initializer,
    )
    return helper.make_model(graph, producer_name="modeltoolbox-fixture-gen")


def build_all() -> dict:
    out = {}

    linear = make_linear()
    out["linear.onnx"] = linear

    # branch/join
    bj = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["x", "conv_1.weight"], ["c"], "conv_1"),
                helper.make_node("Relu", ["c"], ["r"], "relu_1"),
                helper.make_node("Sigmoid", ["c"], ["s"], "sig_1"),
                helper.make_node("Add", ["r", "s"], ["y"], "add_1"),
            ],
            "branch_join",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 8, 8])],
            initializer=[_conv_w("conv_1.weight", [1, 1, 3, 3], 7)],
        ),
        producer_name="modeltoolbox-fixture-gen",
    )
    out["branch_join.onnx"] = bj

    out["renamed_linear.onnx"] = make_linear("renamed", rename=True)

    # attr changed: conv_1 strides [2,2]
    ac = make_linear()
    ac.graph.node[0].ClearField("attribute")
    ac.graph.node[0].attribute.extend([helper.make_attribute("strides", [2, 2])])
    # rebuild with pads default absent; keep it minimal
    out["attr_changed.onnx"] = ac

    # op replaced: Relu -> LeakyRelu
    oprep = make_linear()
    oprep.graph.node[1].op_type = "LeakyRelu"
    out["op_replaced.onnx"] = oprep

    # node added: extra Relu before conv_2
    na = make_linear()
    na.graph.node.insert(2, helper.make_node("Identity", ["h2"], ["h2i"], "ident_extra"))
    na.graph.node[3].input[0] = "h2i"
    out["node_added.onnx"] = na

    # node removed: drop relu_1 (conv_2 consumes h1)
    nr = make_linear()
    del nr.graph.node[1]
    nr.graph.node[1].input[0] = "h1"
    out["node_removed.onnx"] = nr

    # shape changed
    sc = make_linear()
    sc.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 16
    sc.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 16
    out["shape_changed.onnx"] = sc

    # dtype changed: DOUBLE everywhere
    dc = make_linear()
    dc.graph.input[0].type.tensor_type.elem_type = TensorProto.DOUBLE
    dc.graph.output[0].type.tensor_type.elem_type = TensorProto.DOUBLE
    for n in dc.graph.node:
        if n.op_type == "Conv":
            for a in n.attribute:
                pass
    # conv weights must be double too
    new_inits = []
    for t in dc.graph.initializer:
        arr = numpy_helper.to_array(t).astype(np.float64)
        new_inits.append(numpy_helper.from_array(arr, t.name))
    dc.graph.ClearField("initializer")
    dc.graph.initializer.extend(new_inits)
    out["dtype_changed.onnx"] = dc

    # weights differ (same structure, different values)
    wd = make_linear(seed=99)
    out["weights_differ.onnx"] = wd

    # ambiguous diamond: two identical Relu branches with EMPTY names (real
    # exports often drop node names) — no anchor can tell the branches apart
    ad = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["x", "conv_1.weight"], ["c"], "conv_1"),
                helper.make_node("Relu", ["c"], ["r1"], ""),
                helper.make_node("Relu", ["c"], ["r2"], ""),
                helper.make_node("Add", ["r1", "r2"], ["y"], "add_1"),
            ],
            "ambiguous_diamond",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 8, 8])],
            initializer=[_conv_w("conv_1.weight", [1, 1, 3, 3], 7)],
        ),
        producer_name="modeltoolbox-fixture-gen",
    )
    out["ambiguous_diamond.onnx"] = ad

    # nested If: y = x if cond else -x (branches contain real nodes)
    then_g = helper.make_graph(
        [
            helper.make_node("Identity", ["x_if"], ["t1"], "then_ident"),
            helper.make_node("Relu", ["t1"], ["t2"], "then_relu"),
        ],
        "then_branch", [], [],
    )
    else_g = helper.make_graph(
        [
            helper.make_node("Neg", ["x_if"], ["e1"], "else_neg"),
            helper.make_node("Relu", ["e1"], ["e2"], "else_relu"),
        ],
        "else_branch", [], [],
    )
    # Give branches proper io typing
    for g in (then_g, else_g):
        g.input.extend([helper.make_tensor_value_info("x_if", TensorProto.FLOAT, [1, 1, 8, 8])])
        g.output.extend([helper.make_tensor_value_info("t2" if g is then_g else "e2", TensorProto.FLOAT, [1, 1, 8, 8])])
    if_node = helper.make_node(
        "If", ["cond"], ["y"],
        then_branch=then_g, else_branch=else_g, name="if_1",
    )
    ni = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Less", ["x", "zero"], ["cond"], "less_1"),
                if_node,
            ],
            "nested_if",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8, 8]),
                helper.make_tensor_value_info("zero", TensorProto.FLOAT, [1, 1, 8, 8]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 8, 8])],
            initializer=[numpy_helper.from_array(np.zeros((1, 1, 8, 8), np.float32), "zero")],
        ),
        producer_name="modeltoolbox-fixture-gen",
    )
    out["nested_if.onnx"] = ni

    # bad files
    out["bad_not_onnx.bin"] = None  # random bytes, handled below
    out["empty.onnx"] = b""

    truncated = linear.SerializeToString()[: len(linear.SerializeToString()) // 2]
    out["truncated.onnx"] = truncated

    # profile samples for linear.onnx
    profile = {
        "schema_version": 1,
        "model": "linear.onnx",
        "device": "synthetic-board",
        "unit": "us",
        "end_to_end_us": 1500.0,
        "measurements": [
            {"node_name": "conv_1", "op_type": "Conv", "duration": 620.5, "calls": 1},
            {"node_name": "relu_1", "op_type": "Relu", "duration": 35.2, "calls": 1},
            {"node_name": "conv_2", "op_type": "Conv", "duration": 844.3, "calls": 1},
        ],
    }
    out["profile_linear.json"] = json.dumps(profile, indent=2)
    csv_lines = ["node_name,op_type,duration_us,calls"]
    for m in profile["measurements"]:
        csv_lines.append(f"{m['node_name']},{m['op_type']},{m['duration']},{m['calls']}")
    out["profile_linear.csv"] = "\n".join(csv_lines) + "\n"
    return out


def main() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    rng = np.random.RandomState(42)
    for name, obj in build_all().items():
        path = os.path.join(FIXTURE_DIR, name)
        if name == "bad_not_onnx.bin":
            with open(path, "wb") as f:
                f.write(rng.bytes(256))
            continue
        if name == "empty.onnx":
            with open(path, "wb") as f:
                pass
            continue
        if isinstance(obj, bytes):
            with open(path, "wb") as f:
                f.write(obj)
        elif isinstance(obj, str):
            with open(path, "w") as f:
                f.write(obj)
        else:
            onnx.checker.check_model(obj)
            with open(path, "wb") as f:
                f.write(obj.SerializeToString())
        print("wrote", name)

    # Self-check: renamed_linear has different structure fp but same topology fp.
    from modeltoolbox.onnx_importer import load_onnx
    from modeltoolbox.ir import assign_fingerprints

    a = load_onnx(os.path.join(FIXTURE_DIR, "linear.onnx"), "fa")
    b = load_onnx(os.path.join(FIXTURE_DIR, "renamed_linear.onnx"), "fb")
    assert a.fingerprints["structure"] != b.fingerprints["structure"], "structure fp should differ"
    assert a.fingerprints["topology"] == b.fingerprints["topology"], "topology fp should match"
    print("self-check OK: renamed_linear topology matches linear")


if __name__ == "__main__":
    main()
