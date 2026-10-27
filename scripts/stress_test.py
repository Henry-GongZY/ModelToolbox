#!/usr/bin/env python
"""Stress test: 1k / 5k-node synthetic DAGs through the full pipeline.

Measures REAL timings on THIS machine (no invented numbers):
  ONNX build -> import(GraphIR) -> JSON serialize -> match -> diff ->
  single layout -> aligned pair layout -> HTTP round-trip (load + diff).

Run: .venv/bin/python scripts/stress_test.py [--sizes 1000 5000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from modeltoolbox.diff_engine import build_diff_ir  # noqa: E402
from modeltoolbox.layout import layout_diff_pair, layout_single  # noqa: E402
from modeltoolbox.matcher import match_graphs  # noqa: E402
from modeltoolbox.onnx_importer import load_onnx  # noqa: E402


def build_chain_dag(n: int, seed: int = 7, branch_every: int = 25) -> onnx.ModelProto:
    """n-node DAG: long chain with periodic parallel branches that rejoin.

    Pure chains are the trivial case; periodic diamonds exercise the matcher
    and layout with real branching.
    """
    rng = np.random.RandomState(seed)
    nodes = []
    inits = []
    prev = "x"
    for i in range(n):
        outs = [f"t{i}"]
        if i % branch_every == branch_every - 1 and i + 1 < n:
            # fork: two ops from prev, joined by Add next step
            nodes.append(helper.make_node("Relu", [prev], [f"a{i}"], f"reluA_{i}"))
            nodes.append(helper.make_node("Sigmoid", [prev], [f"b{i}"], f"sigB_{i}"))
            nodes.append(helper.make_node("Add", [f"a{i}", f"b{i}"], [f"t{i}"], f"add_{i}"))
            prev = f"t{i}"
            continue
        w = numpy_helper.from_array(
            rng.randn(4, 4).astype(np.float32) * 0.05, f"w{i}")
        inits.append(w)
        nodes.append(helper.make_node("Conv", [prev, f"w{i}"], outs, f"conv_{i}"))
        prev = f"t{i}"
    graph = helper.make_graph(
        nodes, f"stress_{n}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4, 4])],
        [helper.make_tensor_value_info(f"t{n-1}", TensorProto.FLOAT, [1, 4, 4, 4])],
        initializer=inits,
    )
    return helper.make_model(graph, producer_name="stress-gen")


def rename_all(model: onnx.ModelProto) -> onnx.ModelProto:
    """Copy with every node/tensor name rewritten, topology intact."""
    m2 = onnx.ModelProto()
    m2.CopyFrom(model)
    # 1) rename every intermediate/output tensor, recording the mapping
    tensor_map = {}
    for i, n in enumerate(m2.graph.node):
        for j, t in enumerate(n.output):
            if t:
                nt = f"renamed/out_{i}_{j}"
                tensor_map[t] = nt
                n.output[j] = nt
    # 2) rename initializers
    init_map = {}
    for t in m2.graph.initializer:
        nt = f"renamed/{t.name}"
        init_map[t.name] = nt
        t.name = nt
    # 3) rewrite node inputs via tensor mapping (graph inputs keep names)
    for i, n in enumerate(m2.graph.node):
        n.name = f"renamed/op_{i}"
        for j, t in enumerate(n.input):
            if not t:
                continue
            n.input[j] = tensor_map.get(t, init_map.get(t, t))
    # 4) graph outputs
    for go in m2.graph.output:
        go.name = tensor_map.get(go.name, go.name)
    return m2


def timeit(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return time.perf_counter() - t0, r


def run_size(n: int, results: dict) -> None:
    print(f"\n=== {n} nodes ===")
    row = {"nodes": n}

    t, model = timeit(build_chain_dag, n)
    row["build_onnx_s"] = round(t, 3)
    m2 = rename_all(model)

    p1 = os.path.join(ROOT, "fixtures", f"stress_{n}_a.onnx")
    p2 = os.path.join(ROOT, "fixtures", f"stress_{n}_b.onnx")
    with open(p1, "wb") as f:
        f.write(model.SerializeToString())
    with open(p2, "wb") as f:
        f.write(m2.SerializeToString())

    t, ir_a = timeit(load_onnx, p1, "sa")
    row["import_ir_s"] = round(t, 3)
    ir_b = load_onnx(p2, "sb")

    t, js = timeit(lambda: ir_a.to_json())
    row["serialize_json_s"] = round(t, 3)
    row["json_mb"] = round(len(js) / 1e6, 2)

    t, match = timeit(match_graphs, ir_a, ir_b)
    row["match_s"] = round(t, 3)
    row["matched"] = match.summary["matched_count"]
    row["ambiguous"] = match.summary["ambiguous_count"]

    t, diff = timeit(build_diff_ir, ir_a, ir_b, match)
    row["diff_build_s"] = round(t, 3)
    row["diff_unchanged"] = diff.summary.get("unchanged", 0)

    t, la = timeit(layout_single, ir_a)
    row["layout_single_s"] = round(t, 3)

    t, (l1, l2) = timeit(layout_diff_pair, ir_a, ir_b, match.matches)
    row["layout_pair_s"] = round(t, 3)

    # HTTP round-trip through the real server API
    base = "http://127.0.0.1:8765"
    try:
        req = urllib.request.Request(
            base + "/api/diff",
            data=json.dumps({"path_a": p1, "path_b": p2}).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read()
        row["http_diff_s"] = round(time.perf_counter() - t0, 3)
        row["http_resp_mb"] = round(len(body) / 1e6, 2)
    except Exception as e:
        row["http_diff_s"] = None
        row["http_error"] = str(e)[:120]

    # frontend render proxy: number of SVG elements the browser would draw
    row["visible_nodes_note"] = "frontend culls to viewport; full DOM render not required"

    results.append(row)
    print(json.dumps(row, indent=2))
    # clean up big fixtures to keep the workspace lean
    os.remove(p1)
    os.remove(p2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000])
    args = ap.parse_args()
    results = []
    for n in args.sizes:
        run_size(n, results)
    out = os.path.join(ROOT, "docs", "stress_results.json")
    with open(out, "w") as f:
        json.dump({
            "environment": {
                "python": sys.version.split()[0],
                "platform": sys.platform,
                "note": "single machine, localhost HTTP, CPU only",
            },
            "results": results,
        }, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
