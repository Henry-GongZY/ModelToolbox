import json

from helpers import make_ir
from modeltoolbox.folding import apply_folding, find_common_groups


class _M:
    def __init__(self, mid, a, b, conf=1.0, status="matched"):
        self.match_id, self.a_node_id, self.b_node_id = mid, a, b
        self.confidence, self.status = conf, status


def _pair_graphs():
    kw = dict(
        inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
        initializers=[("w", "FLOAT", [1])],
    )
    nodes = [
        ("pre", "Relu", ["x"], ["p1"]),
        ("c1", "Conv", ["p1", "w"], ["m1"]),
        ("r1", "Relu", ["m1"], ["m2"]),
        ("c2", "Conv", ["m2", "w"], ["m3"]),
        ("r2", "Relu", ["m3"], ["m4"]),
        ("post", "Sigmoid", ["m4"], ["y"]),
    ]
    a = make_ir(nodes=nodes, model_id="a", **kw)
    b = make_ir(nodes=nodes, model_id="b", **kw)
    # pre/post are context nodes WITHOUT matches — the matched region is
    # exactly c1..r2, so the fold candidate cannot swallow the whole graph.
    matches = [_M("m1", "c1", "c1", 0.95), _M("m2", "r1", "r1", 0.95),
               _M("m3", "c2", "c2", 0.95), _M("m4", "r2", "r2", 0.95)]
    return a, b, matches


def test_linear_all_matched_one_group():
    a, b, matches = _pair_graphs()
    fr = find_common_groups(a, b, matches)
    assert len(fr.groups) == 1
    g = fr.groups[0]
    assert set(g.node_ids_a) == {"c1", "r1", "c2", "r2"}
    assert set(g.node_ids_b) == {"c1", "r1", "c2", "r2"}
    assert g.label == "Conv×2 + Relu×2"
    assert g.confidence == 0.95
    assert g.boundary_inputs["a"] == ["p1"]
    assert g.boundary_outputs["a"] == ["m4"]
    assert g.internal_edge_count == 3


def test_min_group_size_respected():
    a, b, matches = _pair_graphs()
    fr = find_common_groups(a, b, matches, min_group_size=5)
    assert fr.groups == []  # component of 4 < 5


def test_min_confidence_respected():
    a, b, matches = _pair_graphs()
    matches[0].confidence = 0.5  # c1 below threshold breaks the chain
    fr = find_common_groups(a, b, matches, min_confidence=0.9)
    # c1 drops out; the remaining matched chain r1->c2->r2 still folds.
    assert len(fr.groups) == 1
    assert set(fr.groups[0].node_ids_a) == {"r1", "c2", "r2"}
    # boundary now starts at c1's output m1
    assert fr.groups[0].boundary_inputs["a"] == ["m1"]


def test_ambiguous_never_folded():
    a, b, matches = _pair_graphs()
    matches[2].status = "ambiguous"  # r1 ambiguous -> chain broken at r1
    fr = find_common_groups(a, b, matches)
    assert fr.groups == []


def test_partial_match_group():
    kw = dict(inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
              initializers=[("w", "FLOAT", [1])])
    a = make_ir(nodes=[("c1", "Conv", ["x", "w"], ["m1"]),
                       ("r1", "Relu", ["m1"], ["m2"]),
                       ("c2", "Conv", ["m2", "w"], ["m3"]),
                       ("r2", "Relu", ["m3"], ["y"])], model_id="a", **kw)
    b = make_ir(nodes=[("c1", "Conv", ["x", "w"], ["m1"]),
                       ("r1", "Relu", ["m1"], ["m2"])], model_id="b", **kw)
    matches = [_M("m0", "c1", "c1", 0.95), _M("m1", "r1", "r1", 0.95)]
    fr = find_common_groups(a, b, matches, min_group_size=2)
    assert len(fr.groups) == 1
    assert fr.groups[0].node_ids_a == ["c1", "r1"]
    assert fr.groups[0].node_ids_b == ["c1", "r1"]
    assert fr.groups[0].boundary_outputs["a"] == ["m2"]
    # In B the region dead-ends (m2 has no consumer and is not a graph
    # output), so B has no outgoing boundary edge — a real asymmetry.
    assert fr.groups[0].boundary_outputs["b"] == []


def _check_no_loss(ir, folding, folded_ids, side):
    view = apply_folding(ir, folding, folded_ids, side)
    mapping = folding.node_to_group.get(side, {})
    all_nodes = {n.id for n in ir.nodes}
    node_displayed = set()
    for dn in view.display_nodes:
        if dn["kind"] == "node":
            node_displayed.update(dn["orig_node_ids"])
        else:  # group summary covers only hidden nodes
            assert set(dn["orig_node_ids"]) <= set(view.hidden_nodes)
            assert dn["orig_node_ids"], "group summary must not be empty"
    assert node_displayed == all_nodes - set(view.hidden_nodes)
    assert set(view.hidden_nodes) == {
        nid for nid in all_nodes
        if mapping.get(nid) in folded_ids}
    # every non-internal edge appears exactly once
    internal = set()
    for g in folding.groups:
        if g.group_id in folded_ids:
            internal.update(_internal_edges(ir, g, side))
    view_edge_ids = [e["id"] for e in view.display_edges]
    assert len(view_edge_ids) == len(set(view_edge_ids))
    expected = {e.id for e in ir.edges} - internal
    assert set(view_edge_ids) == expected


def _internal_edges(ir, g, side):
    ids = set(g.node_ids_a if side == "a" else g.node_ids_b)
    return {e.id for e in ir.edges
            if e.src in ids and e.dst in ids}


def test_apply_folding_all_states_no_loss():
    a, b, matches = _pair_graphs()
    fr = find_common_groups(a, b, matches)
    gid = fr.groups[0].group_id
    for state in (set(), {gid}, {"nonexistent"}):
        _check_no_loss(a, fr, state, "a")
        _check_no_loss(b, fr, state, "b")


def test_apply_folding_summary_node_and_rerouting():
    a, b, matches = _pair_graphs()
    fr = find_common_groups(a, b, matches)
    gid = fr.groups[0].group_id
    view = apply_folding(a, fr, {gid}, "a")
    kinds = {dn["kind"] for dn in view.display_nodes}
    assert kinds == {"group", "node"}
    group_node = next(dn for dn in view.display_nodes if dn["kind"] == "group")
    assert group_node["id"] == gid and "Conv×2" in group_node["label"]
    # pre -> group ; group -> post  (rerouted through group node)
    edge_pairs = {(e["src"], e["dst"]) for e in view.display_edges}
    assert ("pre", gid) in edge_pairs
    assert (gid, "post") in edge_pairs
    # io pseudo edges preserved
    assert any(e["src"] == "__in__:x" and e["dst"] == "pre" for e in view.display_edges)
    assert any(e["dst"] == "__out__:y" and e["src"] == "post" for e in view.display_edges)
    # summary text carries counts and confidence
    assert "confidence" in group_node["summary"]


def test_apply_folding_expanded_is_identity():
    a, b, matches = _pair_graphs()
    fr = find_common_groups(a, b, matches)
    view = apply_folding(a, fr, set(), "a")
    assert view.hidden_nodes == []
    assert {dn["id"] for dn in view.display_nodes} == {n.id for n in a.nodes}
    assert len(view.display_edges) == len(a.edges)


def test_determinism():
    a, b, matches = _pair_graphs()
    r1 = find_common_groups(a, b, matches)
    r2 = find_common_groups(a, b, matches)
    assert [g.group_id for g in r1.groups] == [g.group_id for g in r2.groups]
    assert [g.summary for g in r1.groups] == [g.summary for g in r2.groups]


def test_no_matches_no_groups():
    a = make_ir(nodes=[("n1", "Relu", ["x"], ["y"])],
                inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
                model_id="a")
    fr = find_common_groups(a, a, [])
    assert fr.groups == []
    view = apply_folding(a, fr, set(), "a")
    assert len(view.display_nodes) == 1
