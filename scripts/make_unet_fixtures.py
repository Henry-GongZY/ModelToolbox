#!/usr/bin/env python
"""Generate the UNet fixture family: PyTorch sources (already on disk under
fixtures/pytorch/) and their ONNX exports (512x512 input).

Run: .venv/bin/python scripts/make_unet_fixtures.py   (requires torch)

Outputs:
  fixtures/unet5_512.onnx          - 5-level UNet
  fixtures/unet5_512_stem.onnx     - same UNet + stem convs + global residual
"""

from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fixtures", "pytorch"))

from unet_models import UNet5, UNet5Stem  # noqa: E402

FIXTURES = os.path.join(ROOT, "fixtures")


def export_onnx(model: torch.nn.Module, path: str) -> None:
    model = model.eval()
    dummy = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            path,
            opset_version=17,
            do_constant_folding=False,  # keep the graph structure explicit
            input_names=["x"],
            output_names=["y"],
        )
    print(f"wrote {os.path.basename(path)} "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")


def main() -> None:
    torch.manual_seed(7)
    export_onnx(UNet5(), os.path.join(FIXTURES, "unet5_512.onnx"))
    export_onnx(UNet5Stem(), os.path.join(FIXTURES, "unet5_512_stem.onnx"))

    # deterministic per-node profile sample (synthetic durations)
    import random
    random.seed(11)
    from modeltoolbox.onnx_importer import load_onnx as _lo
    _ir = _lo(os.path.join(FIXTURES, "unet5_512.onnx"), "gen")
    base_us = {"Conv": 180, "Relu": 12, "MaxPool": 40, "Resize": 55, "Concat": 18,
               "Add": 15, "Constant": 0.5, "Sigmoid": 12, "Mul": 9}
    measurements = []
    for _n in _ir.nodes:
        if not _n.name:
            continue
        _base = base_us.get(_n.op_type, 20)
        measurements.append({"node_name": _n.name, "op_type": _n.op_type,
                             "duration": round(_base * (0.7 + random.random() * 0.6), 1),
                             "calls": 1})
    _profile = {"schema_version": 1, "model": "unet5_512.onnx",
                "device": "synthetic-board", "unit": "us",
                "end_to_end_us": round(sum(m["duration"] for m in measurements), 1),
                "measurements": measurements}
    with open(os.path.join(FIXTURES, "profile_unet5.json"), "w") as f:
        json.dump(_profile, f, indent=1)
    print(f"wrote profile_unet5.json ({len(measurements)} records)")

    # quick structural self-check through our own importer
    sys.path.insert(0, os.path.join(ROOT, "backend"))
    from modeltoolbox.onnx_importer import load_onnx

    a = load_onnx(os.path.join(FIXTURES, "unet5_512.onnx"), "ua")
    b = load_onnx(os.path.join(FIXTURES, "unet5_512_stem.onnx"), "ub")
    print(f"UNet5:        {len(a.nodes)} nodes, {len(a.edges)} edges, "
          f"input {a.inputs[0].shape}")
    print(f"UNet5Stem:    {len(b.nodes)} nodes, {len(b.edges)} edges")
    assert a.inputs[0].shape == [1, 1, 512, 512]
    assert len(b.nodes) > len(a.nodes), "stem variant must add nodes"

    from modeltoolbox.matcher import match_graphs

    res = match_graphs(a, b)
    s = res.summary
    print(f"match: {s['matched_count']} matched, {s['ambiguous_count']} ambiguous, "
          f"A-only {s['unmatched_a_count']}, B-only {s['unmatched_b_count']}")
    coverage = s["matched_count"] / max(1, len(a.nodes))
    print(f"UNet-core match coverage: {coverage:.1%}")
    assert coverage > 0.9, "the shared UNet core must match almost entirely"
    print("self-check OK")


if __name__ == "__main__":
    main()
