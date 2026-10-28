import time

from helpers import make_ir
from modeltoolbox.layout import NODE_H, NODE_W, X_GAP, Y_GAP, layout_diff_pair, layout_single


def _linear():
    return make_ir(
        nodes=[("n1", "Conv", ["x", "w"], ["h1"]), ("n2", "Relu", ["h1"], ["h2"]),
               ("n3", "Conv", ["h2", "w2"], ["y"])],
        inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
        initializers=[("w", "FLOAT", [1]), ("w2", "FLOAT", [1])], model_id="m",
    )


def test_linear_ranks_and_positions():
    ir = _linear()
    lr = layout_single(ir)
    assert lr.ranks == {"n1": 0, "n2": 1, "n3": 2}
    assert lr.boxes["n2"].x == lr.boxes["n1"].x + NODE_W + X_GAP
    assert lr.boxes["n3"].x == lr.boxes["n2"].x + NODE_W + X_GAP
    # distinct y within the whole layout (each on own rank so all can share y)
    assert lr.boxes["n1"].y == 0
    assert lr.width > NODE_W and lr.height >= NODE_H


def test_branch_join_same_rank_no_overlap():
    ir = make_ir(
        nodes=[("c", "Conv", ["x", "w"], ["c1"]), ("r", "Relu", ["c1"], ["r1"]),
               ("s", "Sigmoid", ["c1"], ["s1"]), ("a", "Add", ["r1", "s1"], ["y"])],
        inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
        initializers=[("w", "FLOAT", [1])], model_id="m",
    )
    lr = layout_single(ir)
    assert lr.ranks["r"] == lr.ranks["s"] == 1
    assert lr.ranks["a"] == 2
    assert lr.boxes["r"].y != lr.boxes["s"].y
    assert abs(lr.boxes["r"].y - lr.boxes["s"].y) >= NODE_H + Y_GAP


def test_pseudo_io_nodes_present():
    ir = _linear()
    lr = layout_single(ir)
    assert "__in__:x" in lr.boxes and lr.boxes["__in__:x"].kind == "input"
    assert "__out__:y" in lr.boxes and lr.boxes["__out__:y"].kind == "output"
    # input pseudo sits left of its consumer's rank
    assert lr.boxes["__in__:x"].x < lr.boxes["n1"].x
    assert lr.boxes["__out__:y"].x > lr.boxes["n3"].x


def test_edge_points_on_box_edges():
    ir = _linear()
    lr = layout_single(ir)
    for eid, pts in lr.edge_points.items():
        for side in ("src", "dst"):
            assert len(pts[side]) == 2
    e = next(eid for eid, e in ((e.id, e) for e in ir.edges)
             if e.src == "n1" and e.dst == "n2")
    pts = lr.edge_points[e]
    n1, n2 = lr.boxes["n1"], lr.boxes["n2"]
    assert pts["src"] == [n1.x + n1.w, n1.y + n1.h / 2]
    assert pts["dst"] == [n2.x, n2.y + n2.h / 2]


def test_deterministic():
    ir = _linear()
    import dataclasses
    r1 = {k: dataclasses.asdict(v) for k, v in layout_single(ir).boxes.items()}
    r2 = {k: dataclasses.asdict(v) for k, v in layout_single(ir).boxes.items()}
    assert r1 == r2


def test_empty_and_single():
    lr = layout_single(make_ir(nodes=[], model_id="e"))
    assert lr.boxes == {}
    lr2 = layout_single(make_ir(nodes=[("only", "Relu", ["x"], ["y"])],
                                inputs=[("x", "FLOAT", [1])],
                                outputs=[("y", "FLOAT", [1])], model_id="s"))
    assert set(lr2.boxes) == {"only", "__in__:x", "__out__:y"}


class _M:
    def __init__(self, mid, a, b):
        self.match_id, self.a_node_id, self.b_node_id = mid, a, b
        self.status = "matched"


def test_diff_pair_alignment():
    a = make_ir(
        nodes=[("c1", "Conv", ["x", "w"], ["h1"]), ("r1", "Relu", ["h1"], ["h2"]),
               ("c2", "Conv", ["h2", "w2"], ["y"])],
        inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
        initializers=[("w", "FLOAT", [1]), ("w2", "FLOAT", [1])], model_id="a",
    )
    b = make_ir(
        nodes=[("cc1", "Conv", ["x", "w"], ["h1"]),
               ("cc2", "Conv", ["h1", "w2"], ["y"])],
        inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
        initializers=[("w", "FLOAT", [1]), ("w2", "FLOAT", [1])], model_id="b",
    )
    matches = [_M("m0", "c1", "cc1"), _M("m1", "c2", "cc2")]
    la, lb = layout_diff_pair(a, b, matches)
    # matched pairs share y even though ranks differ (c2: rank2 in A, rank1 in B)
    assert la.boxes["c1"].y == lb.boxes["cc1"].y
    assert la.boxes["c2"].y == lb.boxes["cc2"].y
    # removed node only on side A
    assert "r1" in la.boxes and "r1" not in lb.boxes


def test_1k_chain_fast():
    nodes = [("n0", "Relu", ["x"], ["t0"])]
    for i in range(1, 1000):
        nodes.append((f"n{i}", "Relu", [f"t{i-1}"], [f"t{i}"]))
    ir = make_ir(nodes=nodes, inputs=[("x", "FLOAT", [1])],
                 outputs=[("t999", "FLOAT", [1])], model_id="big")
    t0 = time.time()
    lr = layout_single(ir)
    dt = time.time() - t0
    assert len(lr.boxes) == 1002  # 1000 nodes + input + output pseudo
    assert dt < 5.0, f"layout took {dt:.2f}s"
