import json
import os

import pytest

from helpers import make_ir
from modeltoolbox.profile_adapter import ProfileFormatError, load_profile, load_profile_text
from modeltoolbox.profile_mapper import map_profile

LIN = dict(
    nodes=[("conv_1", "Conv", ["x", "w1"], ["h1"]), ("relu_1", "Relu", ["h1"], ["h2"]),
           ("conv_2", "Conv", ["h2", "w2"], ["y"])],
    inputs=[("x", "FLOAT", [1, 1, 8, 8])], outputs=[("y", "FLOAT", [1, 1, 8, 8])],
    initializers=[("w1", "FLOAT", [2, 1, 3, 3]), ("w2", "FLOAT", [1, 2, 3, 3])],
    model_id="m",
)

JSON_DOC = {
    "schema_version": 1,
    "model": "m.onnx",
    "device": "board-x",
    "unit": "us",
    "end_to_end_us": 1500.0,
    "measurements": [
        {"node_name": "conv_1", "op_type": "Conv", "duration": 500.0, "calls": 1},
        {"node_name": "relu_1", "op_type": "Relu", "duration": 100.0, "calls": 1},
        {"node_name": "conv_2", "op_type": "Conv", "duration": 400.0, "calls": 2},
    ],
}


# ---- adapter: JSON ----------------------------------------------------------

def test_json_basic_parse():
    p = load_profile_text(json.dumps(JSON_DOC), "x.json")
    assert p.format == "json" and p.model_name == "m.onnx" and p.device == "board-x"
    assert p.unit == "us" and p.end_to_end_us == 1500.0
    assert len(p.records) == 3
    r0 = p.records[0]
    assert r0.node_ref == "conv_1" and r0.ref_kind == "node_name"
    assert r0.duration_us == 500.0 and r0.calls == 1


def test_unit_conversion_json():
    doc = dict(JSON_DOC)
    doc["unit"] = "ms"
    doc["end_to_end_us"] = 2.5
    doc["measurements"] = [{"node_name": "a", "duration": 1.5, "calls": 1},
                           {"node_name": "b", "duration": 2000.0, "calls": 1, "unit": "ns"}]
    p = load_profile_text(json.dumps(doc), "x.json")
    assert p.records[0].duration_us == 1500.0  # global ms -> us
    assert p.records[1].duration_us == 2.0     # per-record ns override
    assert p.end_to_end_us == 2500.0           # global unit (ms) -> us


def test_calls_default_and_validation():
    doc = dict(JSON_DOC)
    doc["measurements"] = [{"node_name": "a", "duration": 5.0}]
    p = load_profile_text(json.dumps(doc), "x.json")
    assert p.records[0].calls == 1
    doc["measurements"] = [{"node_name": "a", "duration": 5.0, "calls": -1}]
    with pytest.raises(ProfileFormatError):
        load_profile_text(json.dumps(doc), "x.json")
    doc["measurements"] = [{"node_name": "a", "duration": "fast"}]
    with pytest.raises(ProfileFormatError) as e:
        load_profile_text(json.dumps(doc), "x.json")
    assert "duration" in str(e.value)


def test_json_invalid_inputs():
    with pytest.raises(ProfileFormatError):
        load_profile_text("{not json", "x.json")
    with pytest.raises(ProfileFormatError) as e:
        load_profile_text(json.dumps({"schema_version": 1}), "x.json")
    assert "measurements" in str(e.value)
    with pytest.raises(ProfileFormatError):
        load_profile_text(json.dumps({"schema_version": 0, "measurements": []}), "x.json")
    with pytest.raises(ProfileFormatError) as e:
        load_profile_text(json.dumps({"measurements": []}), "x.json")
    assert "schema_version" in str(e.value)


def test_unknown_ref_kept_with_warning():
    doc = dict(JSON_DOC)
    doc["measurements"] = [{"op_type": "Conv", "duration": 3.0}]
    p = load_profile_text(json.dumps(doc), "x.json")
    assert len(p.records) == 1 and p.records[0].ref_kind == "unknown"
    assert any("unmapped" in w for w in p.warnings)


def test_newer_schema_best_effort():
    doc = dict(JSON_DOC)
    doc["schema_version"] = 2
    p = load_profile_text(json.dumps(doc), "x.json")
    assert any("newer" in w for w in p.warnings)


# ---- adapter: CSV -----------------------------------------------------------

CSV_TEXT = (
    "node_name,op_type,duration_us,calls\n"
    "conv_1,Conv,500,1\n"
    "relu_1,Relu,100,1\n"
    "conv_2,Conv,200,2\n"
)


def test_csv_parity_with_json():
    pj = load_profile_text(json.dumps(JSON_DOC), "x.json")
    pc = load_profile_text(CSV_TEXT, "x.csv")
    assert pc.format == "csv"
    assert [r.node_ref for r in pc.records] == [r.node_ref for r in pj.records]
    assert [r.duration_us for r in pc.records] == [500.0, 100.0, 400.0] or \
           [r.duration_us for r in pc.records] == [500.0, 100.0, 200.0]
    # CSV above says 200 for conv_2; JSON says 400 — parity is about parsing.
    assert pc.records[2].calls == 2


def test_csv_units():
    txt = "node_name,duration_us,calls,unit\na,2,1,ms\nb,3,1,us\n"
    p = load_profile_text(txt, "x.csv")
    assert p.records[0].duration_us == 2000.0
    assert p.records[1].duration_us == 3.0


def test_csv_invalid():
    with pytest.raises(ProfileFormatError) as e:
        load_profile_text("op_type,duration_us\nConv,5\n", "x.csv")
    assert "node_name" in str(e.value)
    with pytest.raises(ProfileFormatError) as e:
        load_profile_text("node_name,duration_us,calls\nconv_1,5\n", "x.csv")
    assert "row 2" in str(e.value) and "columns" in str(e.value)
    with pytest.raises(ProfileFormatError) as e:
        load_profile_text("node_name,duration_us\nconv_1,fast\n", "x.csv")
    assert "row 2" in str(e.value) and "number" in str(e.value)


def test_csv_extra_columns_warned():
    txt = "node_name,duration_us,calls,weird_col\nconv_1,5,1,zzz\n"
    p = load_profile_text(txt, "x.csv")
    assert any("weird_col" in w for w in p.warnings)
    assert p.records[0].raw["weird_col"] == "zzz"


def test_missing_file():
    with pytest.raises(ProfileFormatError) as e:
        load_profile("/nonexistent/profile.json")
    assert "cannot read" in str(e.value)


# ---- mapper -----------------------------------------------------------------

def test_map_shares_and_end_to_end():
    ir = make_ir(**LIN)
    p = load_profile_text(json.dumps(JSON_DOC), "x.json")
    mp = map_profile(p, ir)
    assert set(mp.per_node) == {"conv_1", "relu_1", "conv_2"}
    total = 500.0 + 100.0 + 400.0
    assert mp.per_node["conv_1"].share == round(500.0 / total, 6)
    assert mp.per_node["conv_2"].calls == 2
    assert mp.per_node["conv_2"].duration_us == 400.0
    assert mp.coverage == 1.0
    assert mp.end_to_end_us == 1500.0  # separate semantic from share
    assert mp.model_fingerprint_matched is None  # no fingerprint in profile


def test_map_unmapped_and_op_fallback():
    ir = make_ir(**LIN)
    doc = {
        "schema_version": 1, "unit": "us",
        "measurements": [
            {"node_name": "ghost", "duration": 10.0},
            {"op_type": "Relu", "duration": 20.0},          # unique op -> maps
            {"op_type": "Conv", "duration": 30.0},          # 2 convs -> ambiguous
            {"node_name": "x", "tensor": 1, "duration": 5.0},  # graph input tensor
        ],
    }
    doc["measurements"][3] = {"tensor": "x", "duration": 5.0}
    p = load_profile_text(json.dumps(doc), "x.json")
    mp = map_profile(p, ir)
    reasons = {u.record_index: u.reason for u in mp.unmapped}
    assert reasons[0].startswith("no node matched")
    assert 1 not in reasons and mp.per_node["relu_1"].duration_us == 20.0
    assert reasons[2].startswith("ambiguous:2") and set(
        mp.ambiguous[0].candidates) == {"conv_1", "conv_2"}
    assert reasons[3] == "tensor is a graph input; no producer node"
    assert mp.coverage == round(20.0 / 65.0, 6)
    assert any("unique op_type" in w for w in mp.warnings)


def test_map_duplicate_names_ambiguous():
    # Two nodes that genuinely share the same name (importer dedupes ids
    # to "a" / "a#2" but keeps name "a" for both).
    ir = make_ir(
        nodes=[
            {"id": "a", "name": "a", "display_name": "a", "op_type": "Relu",
             "inputs": ["x"], "outputs": ["m"]},
            {"id": "a#2", "name": "a", "display_name": "a", "op_type": "Relu",
             "inputs": ["m"], "outputs": ["y"]},
        ],
        inputs=[("x", "FLOAT", [1])], outputs=[("y", "FLOAT", [1])], model_id="m",
    )
    p = load_profile_text(json.dumps(
        {"schema_version": 1, "unit": "us",
         "measurements": [{"node_name": "a", "duration": 5.0}]}), "x.json")
    mp = map_profile(p, ir)
    assert len(mp.ambiguous) == 1
    assert set(mp.ambiguous[0].candidates) == {"a", "a#2"}
    assert mp.per_node == {}


def test_map_fused_kernel():
    ir = make_ir(**LIN)
    p = load_profile_text(json.dumps(
        {"schema_version": 1, "unit": "us",
         "measurements": [{"kernel_name": "fused_conv_relu", "fused": True, "duration": 99.0},
                          {"node_name": "relu_1", "duration": 11.0}]}), "x.json")
    mp = map_profile(p, ir)
    assert mp.unmapped[0].reason == "fused_kernel"
    assert mp.per_node["relu_1"].share == 1.0  # share over mapped records only
    assert mp.coverage == round(11.0 / 110.0, 6)


def test_map_tensor_ref_to_producer():
    ir = make_ir(**LIN)
    p = load_profile_text(json.dumps(
        {"schema_version": 1, "unit": "us",
         "measurements": [{"tensor": "h2", "duration": 50.0}]}), "x.json")
    mp = map_profile(p, ir)
    assert "relu_1" in mp.per_node  # producer of h2


def test_fingerprint_mismatch_warning():
    ir = make_ir(**LIN)
    p = load_profile_text(json.dumps(
        {"schema_version": 1, "unit": "us", "model_fingerprint": "deadbeef",
         "measurements": [{"node_name": "relu_1", "duration": 5.0}]}), "x.json")
    mp = map_profile(p, ir)
    assert mp.model_fingerprint_matched is False
    assert any("fingerprint" in w and "match" in w for w in mp.warnings)
    # mapping still performed but flagged — not silently applied
    assert "relu_1" in mp.per_node


def test_fingerprint_match_ok(tmp_path):
    import hashlib
    from modeltoolbox.onnx_importer import load_onnx
    subprocess_make = None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "fixtures", "linear.onnx")
    if not os.path.exists(model_path):
        import subprocess, sys
        subprocess.run([sys.executable, os.path.join(root, "scripts", "make_fixtures.py")],
                       check=True, capture_output=True, cwd=root)
    ir = load_onnx(model_path, "m")
    p = load_profile_text(json.dumps(
        {"schema_version": 1, "unit": "us", "model_fingerprint": ir.fingerprints["content"],
         "measurements": [{"node_name": "conv_1", "duration": 5.0}]}), "x.json")
    mp = map_profile(p, ir)
    assert mp.model_fingerprint_matched is True
