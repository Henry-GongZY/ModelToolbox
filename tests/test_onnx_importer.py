import json
import os
import subprocess
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from modeltoolbox.ir import GraphIR
from modeltoolbox.onnx_importer import ModelLoadError, load_onnx, load_onnx_bytes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "fixtures")
SCRIPT = os.path.join(ROOT, "scripts", "make_fixtures.py")


@pytest.fixture(scope="module", autouse=True)
def fixtures_ready():
    subprocess.run(
        [sys.executable, SCRIPT], check=True, capture_output=True, cwd=ROOT
    )
    return FIXTURES


def fx(name):
    return os.path.join(FIXTURES, name)


# ---- inline-constructed small model ----------------------------------------


def _tiny_model(with_names=True, sym_dim=False):
    nodes = [
        helper.make_node("Neg", ["a"], ["b"], "neg1" if with_names else ""),
        helper.make_node("Abs", ["b"], ["c"], "abs1" if with_names else ""),
    ]
    shape = [2, "N"] if sym_dim else [2, 3]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, shape)],
        initializer=[
            numpy_helper.from_array(np.array([1.0, 2.0, 3.0], np.float32), "bias")
        ],
    )
    return helper.make_model(graph, producer_name="test")


def test_basic_nodes_edges_ids():
    ir = load_onnx_bytes(_tiny_model().SerializeToString(), "t1")
    assert [n.op_type for n in ir.nodes] == ["Neg", "Abs"]
    assert [n.id for n in ir.nodes] == ["neg1", "abs1"]
    # b: neg1->abs1 ; a: graph input -> neg1 ; c: abs1 -> graph output
    assert len(ir.edges) == 3
    by_id = {e.id: e for e in ir.edges}
    assert any(e.src == "neg1" and e.dst == "abs1" and e.tensor == "b" for e in ir.edges)
    assert any(e.src == "" and e.dst == "neg1" and e.tensor == "a" for e in ir.edges)
    assert any(e.src == "abs1" and e.dst == "" and e.tensor == "c" for e in ir.edges)
    assert ir.producer == "test"
    assert ir.warnings == []


def test_shape_inference_and_symbolic_dim():
    ir = load_onnx_bytes(_tiny_model().SerializeToString(), "t2")
    assert ir.inputs[0].shape == [2, 3]
    assert ir.inputs[0].dtype == "FLOAT"
    assert ir.outputs[0].shape == [2, 3]
    assert ir.nodes[0].output_info["b"].shape == [2, 3]

    ir2 = load_onnx_bytes(_tiny_model(sym_dim=True).SerializeToString(), "t3")
    assert ir2.inputs[0].shape == [2, None]  # symbolic dim -> unknown, not guessed


def test_empty_and_duplicate_node_names():
    # Two nodes with empty names -> synthesized unique ids; display falls back.
    ir = load_onnx_bytes(_tiny_model(with_names=False).SerializeToString(), "t4")
    ids = [n.id for n in ir.nodes]
    assert len(set(ids)) == 2
    assert ids[0] == "n0" and ids[1] == "n1"
    assert ir.nodes[0].name == "" and ir.nodes[0].display_name

    # Duplicate explicit names -> suffix, original name preserved.
    model = _tiny_model()
    model.graph.node[1].name = "neg1"
    ir2 = load_onnx_bytes(model.SerializeToString(), "t5")
    ids2 = [n.id for n in ir2.nodes]
    assert ids2 == ["neg1", "neg1#2"]
    assert ir2.nodes[1].name == "neg1"


def test_initializer_metadata_and_params():
    ir = load_onnx(fx("linear.onnx"), "m1")
    assert len(ir.initializers) == 2
    w1 = [p for p in ir.initializers if p.name == "conv_1.weight"][0]
    assert w1.dtype == "FLOAT"
    assert w1.shape == [2, 1, 3, 3]
    assert w1.num_elements == 18
    assert w1.data_size_bytes == 18 * 4
    conv1 = ir.node("conv_1")
    assert conv1 is not None
    assert [p.name for p in conv1.params] == ["conv_1.weight"]
    # No payload anywhere in JSON.
    js = ir.to_json()
    assert "raw_data" not in js
    # Initializer-as-graph-input pattern is not present here, but params are
    # excluded from ir.inputs:
    assert [v.name for v in ir.inputs] == ["x"]
    assert [v.name for v in ir.outputs] == ["y"]


def test_graph_input_initializer_pattern():
    # Model where W is declared as graph input AND initializer (common export).
    model = _tiny_model()
    model.graph.input.append(
        helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3])
    )
    ir = load_onnx_bytes(model.SerializeToString(), "t6")
    assert "bias" not in [v.name for v in ir.inputs]
    assert "bias" in [p.name for p in ir.initializers]
    assert any("'bias'" in w for w in ir.warnings)


def test_nested_if_subgraphs_preserved():
    ir = load_onnx(fx("nested_if.onnx"), "m2")
    if_node = ir.node("if_1")
    assert if_node is not None
    assert len(if_node.subgraphs) == 2
    branch_names = {sg.attribute for sg in if_node.subgraphs}
    assert any("then_branch" in b for b in branch_names)
    total = sum(sg.node_count for sg in if_node.subgraphs)
    assert total == 4  # 2 nodes per branch
    # Subgraph node ids are prefixed to avoid collision.
    all_sg_ids = [n.id for sg in if_node.subgraphs for n in sg.nodes]
    assert all(n.id and n.id.startswith("if_1/") for n in sum([sg.nodes for sg in if_node.subgraphs], []))
    # Also surfaced via attribute marker.
    assert "__subgraph__" in if_node.attributes["then_branch"]


def test_fingerprints_and_paths():
    ir = load_onnx(fx("linear.onnx"), "m3")
    assert set(ir.fingerprints) >= {"structure", "topology", "content"}
    assert ir.file_size_bytes == os.path.getsize(fx("linear.onnx"))
    assert ir.source_path.endswith("linear.onnx")
    import hashlib
    with open(fx("linear.onnx"), "rb") as f:
        assert ir.fingerprints["content"] == hashlib.sha256(f.read()).hexdigest()


def test_roundtrip_json():
    ir = load_onnx(fx("linear.onnx"), "m4")
    js = ir.to_json()
    ir2 = GraphIR.from_json(js)
    assert ir2.to_json() == js


def test_bad_files_raise_readable():
    for name in ("bad_not_onnx.bin", "empty.onnx", "truncated.onnx"):
        with pytest.raises(ModelLoadError) as exc:
            load_onnx(fx(name), "bad")
        msg = str(exc.value).lower()
        assert any(k in msg for k in ("empty", "parse", "decode", "corrupt")), name
        assert "traceback" not in msg


def test_fixture_diff_family_loads():
    """All valid diff-family fixtures parse and differ from linear as expected."""
    from modeltoolbox.ir import assign_fingerprints

    base = load_onnx(fx("linear.onnx"), "base")
    expected_nodes = {"node_removed.onnx": 2}
    for name in ("attr_changed.onnx", "op_replaced.onnx", "node_added.onnx",
                 "node_removed.onnx", "shape_changed.onnx", "dtype_changed.onnx",
                 "weights_differ.onnx", "renamed_linear.onnx", "branch_join.onnx",
                 "ambiguous_diamond.onnx"):
        ir = load_onnx(fx(name), name)
        assert len(ir.nodes) >= expected_nodes.get(name, 3), name
        assert ir.fingerprints["content"] != base.fingerprints["content"], name
    wd = load_onnx(fx("weights_differ.onnx"), "wd")
    assert wd.fingerprints["structure"] == base.fingerprints["structure"]
    assert wd.fingerprints["topology"] == base.fingerprints["topology"]


def test_profile_fixtures_exist_and_valid_json():
    with open(fx("profile_linear.json")) as f:
        prof = json.load(f)
    assert prof["schema_version"] == 1
    assert prof["unit"] == "us"
    assert len(prof["measurements"]) == 3
    with open(fx("profile_linear.csv")) as f:
        lines = f.read().strip().splitlines()
    assert lines[0] == "node_name,op_type,duration_us,calls"
    assert len(lines) == 4


def test_load_missing_file():
    with pytest.raises(ModelLoadError) as exc:
        load_onnx("/nonexistent/model.onnx", "x")
    assert "cannot read" in str(exc.value)
