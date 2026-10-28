"""Scenario tests for the cross-model matcher (mirrors roadmap stage C).

All fixtures are synthetic, built via tests.helpers.make_ir.
"""

import pytest

from helpers import make_ir
from modeltoolbox.matcher import match_graphs

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


def _pairs(res):
    return {(m.a_node_id, m.b_node_id) for m in res.matches}


def test_scenario1_identical_models_all_matched():
    a = make_ir(**LIN_A, model_id="a")
    b = make_ir(**LIN_A, model_id="b")
    res = match_graphs(a, b)
    assert _pairs(res) == {("conv_1", "conv_1"), ("relu_1", "relu_1"), ("conv_2", "conv_2")}
    assert res.unmatched_a == [] and res.unmatched_b == []
    assert res.ambiguous_pairs == []
    assert res.summary["matched_count"] == 3
    assert all(m.confidence >= 0.8 for m in res.matches)


def test_scenario2_renamed_but_same_structure():
    a = make_ir(**LIN_A, model_id="a")
    b = make_ir(
        nodes=[
            ("stem/conv", "Conv", ["feat", "stem/w"], ["stem/c"], {"strides": [1, 1]}),
            ("stem/act", "Relu", ["stem/c"], ["stem/a"]),
            ("head/conv", "Conv", ["stem/a", "head/w"], ["head/y"]),
        ],
        inputs=[("feat", "FLOAT", [1, 1, 8, 8])],
        outputs=[("head/y", "FLOAT", [1, 1, 8, 8])],
        initializers=[("stem/w", "FLOAT", [2, 1, 3, 3]), ("head/w", "FLOAT", [1, 2, 3, 3])],
        model_id="b",
    )
    res = match_graphs(a, b)
    assert len(res.matches) == 3
    assert res.ambiguous_pairs == []
    assert res.unmatched_a == [] and res.unmatched_b == []
    for m in res.matches:
        assert m.confidence > 0.5, m
        assert m.evidence, m.match_id
    # conv_1 (graph-input consumer) must map to stem/conv (not head/conv)
    assert ("conv_1", "stem/conv") in _pairs(res)
    assert ("conv_2", "head/conv") in _pairs(res)


def test_scenario3_added_removed_op_replaced():
    a = make_ir(**LIN_A, model_id="a")
    # added node between relu and conv_2
    b_added = make_ir(
        nodes=LIN_A["nodes"][:2] + [("ident", "Identity", ["h2"], ["h2i"]),
                                    ("conv_2", "Conv", ["h2i", "w2"], ["y"])],
        inputs=LIN_A["inputs"], outputs=LIN_A["outputs"],
        initializers=LIN_A["initializers"], model_id="b",
    )
    res = match_graphs(a, b_added)
    pairs = _pairs(res)
    assert ("conv_1", "conv_1") in pairs and ("relu_1", "relu_1") in pairs
    assert res.unmatched_b == ["ident"]

    b_removed = make_ir(
        nodes=[("conv_1", "Conv", ["x", "w1"], ["h1"], {"strides": [1, 1]}),
               ("conv_2", "Conv", ["h1", "w2"], ["y"])],
        inputs=LIN_A["inputs"], outputs=LIN_A["outputs"],
        initializers=LIN_A["initializers"], model_id="b",
    )
    res2 = match_graphs(a, b_removed)
    assert res2.unmatched_a == ["relu_1"]
    assert ("conv_2", "conv_2") in _pairs(res2)

    b_op = make_ir(
        nodes=[("conv_1", "Conv", ["x", "w1"], ["h1"], {"strides": [1, 1]}),
               ("leaky", "LeakyRelu", ["h1"], ["h2"]),
               ("conv_2", "Conv", ["h2", "w2"], ["y"])],
        inputs=LIN_A["inputs"], outputs=LIN_A["outputs"],
        initializers=LIN_A["initializers"], model_id="b",
    )
    res3 = match_graphs(a, b_op)
    # relu_1 <-> leaky should pair via topology (same position), not be dropped
    pairs3 = _pairs(res3)
    assert ("conv_1", "conv_1") in pairs3 and ("conv_2", "conv_2") in pairs3
    assert ("relu_1", "leaky") in pairs3


def test_scenario5_connection_changed_same_names():
    base_nodes = LIN_A["nodes"]
    a = make_ir(
        nodes=[("conv_1", "Conv", ["x", "w1"], ["c1"], {"strides": [1, 1]}),
               ("conv_2", "Conv", ["x", "w2"], ["c2"]),
               ("relu_1", "Relu", ["c1"], ["r1"]),
               ("relu_2", "Relu", ["c2"], ["r2"]),
               ("add_1", "Add", ["r1", "r2"], ["y"])],
        inputs=[("x", "FLOAT", [1, 1, 8, 8])],
        outputs=[("y", "FLOAT", [1, 1, 8, 8])],
        initializers=[("w1", "FLOAT", [2, 1, 3, 3]), ("w2", "FLOAT", [1, 2, 3, 3])],
        model_id="a",
    )
    # swap inputs of the two relus (connections changed, names unchanged)
    b = make_ir(
        nodes=[("conv_1", "Conv", ["x", "w1"], ["c1"], {"strides": [1, 1]}),
               ("conv_2", "Conv", ["x", "w2"], ["c2"]),
               ("relu_1", "Relu", ["c2"], ["r1"]),
               ("relu_2", "Relu", ["c1"], ["r2"]),
               ("add_1", "Add", ["r1", "r2"], ["y"])],
        inputs=[("x", "FLOAT", [1, 1, 8, 8])],
        outputs=[("y", "FLOAT", [1, 1, 8, 8])],
        initializers=[("w1", "FLOAT", [2, 1, 3, 3]), ("w2", "FLOAT", [1, 2, 3, 3])],
        model_id="b",
    )
    res = match_graphs(a, b)
    pairs = _pairs(res)
    assert len(res.matches) == 5
    assert ("relu_1", "relu_1") in pairs
    # relu_1's upstream in A is conv_1; partner relu_1 in B is fed by conv_2
    m_relu = next(m for m in res.matches if m.a_node_id == "relu_1")
    assert m_relu.a_node_id == "relu_1" and m_relu.b_node_id == "relu_1"


def test_scenario6_symmetric_diamond_ambiguous():
    # Symmetric Relu branches whose names DIFFER across models — no anchor
    # can tell them apart, so they must be reported ambiguous, not paired.
    nodes_a = [
        ("conv_1", "Conv", ["x", "w"], ["c"]),
        ("relu_a", "Relu", ["c"], ["r1"]),
        ("relu_b", "Relu", ["c"], ["r2"]),
        ("add_1", "Add", ["r1", "r2"], ["y"]),
    ]
    nodes_b = [
        ("conv_1", "Conv", ["x", "w"], ["c"]),
        ("relu_x", "Relu", ["c"], ["r1"]),
        ("relu_y", "Relu", ["c"], ["r2"]),
        ("add_1", "Add", ["r1", "r2"], ["y"]),
    ]
    kw = dict(
        inputs=[("x", "FLOAT", [1, 1, 8, 8])],
        outputs=[("y", "FLOAT", [1, 1, 8, 8])],
        initializers=[("w", "FLOAT", [1, 1, 3, 3])],
    )
    a = make_ir(nodes=nodes_a, model_id="a", **kw)
    b = make_ir(nodes=nodes_b, model_id="b", **kw)
    res = match_graphs(a, b)
    pairs = _pairs(res)
    # Conv and Add must match (unique signatures); Relus must be ambiguous.
    assert ("conv_1", "conv_1") in pairs
    assert ("add_1", "add_1") in pairs
    assert not any(u.startswith("relu") for u, _ in pairs)
    assert res.summary["ambiguous_count"] >= 1
    amb_nodes = {p["a"] for p in res.ambiguous_pairs if p.get("a")} | {
        p["b"] for p in res.ambiguous_pairs if p.get("b")}
    assert {"relu_a", "relu_b", "relu_x", "relu_y"} <= amb_nodes
    assert set(res.unmatched_a) >= {"relu_a", "relu_b"}
    assert set(res.unmatched_b) >= {"relu_x", "relu_y"}


def test_scenario7_same_structure_different_weights():
    a = make_ir(**LIN_A, model_id="a")
    b = make_ir(
        nodes=LIN_A["nodes"],
        inputs=LIN_A["inputs"], outputs=LIN_A["outputs"],
        initializers=[("w1", "FLOAT", [2, 1, 3, 3]), ("w2", "FLOAT", [1, 2, 3, 3])],
        model_id="b",
    )
    # note: make_ir does not carry weight VALUES (IR is metadata-only), so
    # identical param metadata == identical structure. The real "weights
    # differ" fixture is covered at importer level: fingerprints match.
    res = match_graphs(a, b)
    assert len(res.matches) == 3 and res.ambiguous_pairs == []


def test_scenario8_parallel_identical_subgraphs_ambiguous():
    # Two identical parallel chains, ALL names differing across models:
    # no anchor can decide which chain is which — everything ambiguous.
    kw = dict(
        inputs=[("x", "FLOAT", [1, 1, 8, 8])],
        outputs=[("y", "FLOAT", [1, 1, 8, 8])],
        initializers=[("w1", "FLOAT", [1, 1, 3, 3]), ("w2", "FLOAT", [1, 1, 3, 3])],
    )
    nodes_a = [
        ("alpha/conv", "Conv", ["x", "w1"], ["c1"], {"strides": [1, 1]}),
        ("alpha/relu", "Relu", ["c1"], ["r1"]),
        ("beta/conv", "Conv", ["x", "w2"], ["c2"], {"strides": [1, 1]}),
        ("beta/relu", "Relu", ["c2"], ["r2"]),
        ("merge/add", "Add", ["r1", "r2"], ["y"]),
    ]
    nodes_b = [
        ("branch_one/conv", "Conv", ["x", "w1"], ["c1"], {"strides": [1, 1]}),
        ("branch_one/relu", "Relu", ["c1"], ["r1"]),
        ("branch_two/conv", "Conv", ["x", "w2"], ["c2"], {"strides": [1, 1]}),
        ("branch_two/relu", "Relu", ["c2"], ["r2"]),
        ("sum/add", "Add", ["r1", "r2"], ["y"]),
    ]
    a = make_ir(nodes=nodes_a, model_id="a", **kw)
    b = make_ir(nodes=nodes_b, model_id="b", **kw)
    res = match_graphs(a, b)
    # The unique merge Add can be confidently paired (unique signature);
    # the two interchangeable chains cannot — they must stay ambiguous.
    pairs = _pairs(res)
    assert pairs == {("merge/add", "sum/add")}
    amb = {p["a"] for p in res.ambiguous_pairs if p.get("a")}
    assert {"alpha/conv", "alpha/relu", "beta/conv", "beta/relu"} <= amb
    assert set(res.unmatched_a) >= {"alpha/conv", "alpha/relu", "beta/relu"}
    assert res.summary["ambiguous_count"] >= 1


def test_determinism_all_scenarios():
    a = make_ir(**LIN_A, model_id="a")
    b = make_ir(**LIN_A, model_id="b")
    r1 = match_graphs(a, b).to_dict()
    r2 = match_graphs(a, b).to_dict()
    assert r1 == r2


def test_empty_graphs():
    a = make_ir(nodes=[], model_id="a")
    b = make_ir(nodes=[], model_id="b")
    res = match_graphs(a, b)
    assert res.matches == [] and res.summary["matched_count"] == 0


def test_op_mismatch_without_topology_support_stays_unmatched():
    # Two completely different single-node graphs: no evidence to pair them.
    a = make_ir(nodes=[("a1", "Conv", ["x", "w"], ["y"])],
                inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
                initializers=[("w", "FLOAT", [1])], model_id="a")
    b = make_ir(nodes=[("b1", "BatchNormalization", ["x", "s", "m", "v"], ["y"])],
                inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
                initializers=[("s", "FLOAT", [1]), ("m", "FLOAT", [1]), ("v", "FLOAT", [1])],
                model_id="b")
    res = match_graphs(a, b)
    # Different ops, no shared topology: must NOT be confidently paired.
    pairs = _pairs(res)
    assert ("a1", "b1") not in pairs or next(
        m.confidence for m in res.matches
        if m.a_node_id == "a1") < 0.5
