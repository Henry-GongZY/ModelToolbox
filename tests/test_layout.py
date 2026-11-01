import time

from helpers import make_ir
from modeltoolbox.layout import NODE_H, NODE_W, X_GAP, Y_GAP, layout_diff_pair, layout_single


def test_side_chain_sinks_to_consumer():
    """Regression (Netron-structure parity): a Constant→Concat side chain
    feeding a DEEP Resize must sink next to its consumer instead of sitting
    in a top band far away. Longest-path ranking puts it early; the sink
    pass must pull it down to just above the consumer."""
    deep = make_ir(
        nodes=[("r0", "Relu", ["x"], ["t0"])] +
              [(f"n{i}", "Relu", [f"t{i-1}"], [f"t{i}"]) for i in range(1, 12)] +
              [("const", "Constant", [], ["c_out"]),
               ("const2", "Constant", [], ["c2_out"]),
               ("concat", "Concat", ["c_out", "c2_out"], ["cat_out"]),
               ("resize", "Resize", ["t11", "cat_out"], ["y"])],
        inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
        model_id="sink",
    )
    lr = layout_single(deep)
    assert lr.ranks["resize"] == 12
    # concat must sit immediately above its Resize, not in a top band
    assert lr.ranks["concat"] == lr.ranks["resize"] - 1
    assert lr.ranks["const"] == lr.ranks["concat"] - 1
    # and the main spine is untouched
    assert lr.ranks["n11"] == 11


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
    # a deep narrow chain auto-picks the vertical orientation (Netron-style)
    assert lr.orientation == "vertical"
    assert lr.ranks == {"n1": 0, "n2": 1, "n3": 2}
    assert lr.boxes["n2"].y == lr.boxes["n1"].y + NODE_H + Y_GAP
    assert lr.boxes["n3"].y == lr.boxes["n2"].y + NODE_H + Y_GAP
    # row 0 is reserved for the graph-input pseudo node
    assert lr.boxes["n1"].y == NODE_H + Y_GAP
    assert lr.boxes["__in__:x"].y == 0
    assert lr.width >= NODE_W and lr.height >= NODE_H


def test_wide_branchy_graph_stays_horizontal():
    # a graph with many parallel nodes per rank must stay horizontal
    nodes = [(f"n{i}", "Relu", ["x"], [f"t{i}"]) for i in range(12)]
    nodes.append(("a", "Add", [f"t{i}" for i in range(12)], ["y"]))
    ir = make_ir(nodes=nodes, inputs=[("x", "FLOAT", [1])],
                 outputs=[("y", "FLOAT", [1])], model_id="wide")
    lr = layout_single(ir)
    assert lr.orientation == "horizontal"
    assert lr.boxes["a"].x > lr.boxes["n0"].x


def test_vertical_side_chain_gets_separate_channel():
    """Deep vertical graphs keep feeder branches beside the spine."""
    ir = make_ir(
        nodes=[("n0", "Conv", ["x"], ["t0"])] +
              [(f"n{i}", "Relu", [f"t{i-1}"], [f"t{i}"]) for i in range(1, 7)] +
              [("aux0", "Shape", ["t1"], ["shape"]),
               ("aux1", "Constant", [], ["axes"]),
               ("aux2", "Reshape", ["shape", "axes"], ["side"]),
               ("join", "Add", ["t6", "side"], ["y"])],
        inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
        model_id="side-channel",
    )
    lr = layout_single(ir)
    assert lr.orientation == "vertical"
    spine_x = lr.boxes["n3"].x
    assert spine_x == lr.boxes["n6"].x
    assert lr.boxes["aux0"].x > spine_x
    assert lr.boxes["aux2"].x > spine_x
    assert lr.x_origin > 0


def test_vertical_diff_preserves_skip_channel():
    a = _linear()
    b = _linear()
    la, lb = layout_diff_pair(a, b, [])
    assert la.orientation == lb.orientation == "vertical"
    assert la.x_origin == lb.x_origin > 0
    assert min(box.x for box in la.boxes.values()) >= la.x_origin


def test_branch_join_same_rank_no_overlap():
    ir = make_ir(
        nodes=[("c", "Conv", ["x", "w"], ["c1"]), ("r", "Relu", ["c1"], ["r1"]),
               ("s", "Sigmoid", ["c1"], ["s1"]), ("a", "Add", ["r1", "s1"], ["y"])],
        inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])],
        initializers=[("w", "FLOAT", [1])], model_id="m",
    )
    lr = layout_single(ir)
    if lr.orientation == "vertical":
        # same rank → same y; siblings occupy distinct x slots
        assert lr.boxes["r"].y == lr.boxes["s"].y
        assert abs(lr.boxes["r"].x - lr.boxes["s"].x) >= NODE_W + X_GAP
        assert lr.boxes["a"].y > lr.boxes["r"].y
    else:
        assert lr.boxes["r"].y != lr.boxes["s"].y
        assert abs(lr.boxes["r"].y - lr.boxes["s"].y) >= NODE_H + Y_GAP


def test_pseudo_io_nodes_present():
    ir = _linear()
    lr = layout_single(ir)
    assert "__in__:x" in lr.boxes and lr.boxes["__in__:x"].kind == "input"
    assert "__out__:y" in lr.boxes and lr.boxes["__out__:y"].kind == "output"
    # input pseudo sits above its consumer's rank in vertical orientation
    assert lr.boxes["__in__:x"].y < lr.boxes["n1"].y
    assert lr.boxes["__out__:y"].y > lr.boxes["n3"].y


def test_edge_points_on_box_edges():
    ir = _linear()
    lr = layout_single(ir)
    assert lr.orientation == "vertical"
    n1, n2 = lr.boxes["n1"], lr.boxes["n2"]
    e = next(e.id for e in ir.edges if e.src == "n1" and e.dst == "n2")
    pts = lr.edge_points[e]
    # vertical orientation: source exits the bottom edge, enters the top edge
    assert pts["src"] == [n1.x + n1.w / 2, n1.y + n1.h]
    assert pts["dst"] == [n2.x + n2.w / 2, n2.y]


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
