"""Integration tests for the HTTP API: real server on an ephemeral port."""

import csv
import io
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from modeltoolbox.server import Handler  # noqa: E402

FIXTURES = os.path.join(ROOT, "fixtures")


@pytest.fixture(scope="module", autouse=True)
def fixtures_ready():
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_fixtures.py")],
                   check=True, capture_output=True, cwd=ROOT)
    return FIXTURES


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, json.loads(r.read())


def _post(url, payload, expect=200):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"raw": body.decode(errors="replace")}


def _download(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.headers.get("Content-Type"), r.read()


def fx(name):
    return os.path.join(FIXTURES, name)


def test_health(server):
    status, body = _get(server + "/api/health")
    assert status == 200 and body["ok"] is True
    assert "torch_available" in body


def test_static_index(server):
    with urllib.request.urlopen(server + "/", timeout=30) as r:
        assert r.status == 200
        assert b"html" in r.read().lower()
    try:
        _get(server + "/api/nope")
        assert False, "expected 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_model_load_good(server):
    status, body = _post(server + "/api/model/load", {"path": fx("linear.onnx")})
    assert status == 200
    assert body["meta"]["node_count"] == 3
    assert body["meta"]["fingerprints"]["content"]
    assert len(body["ir"]["nodes"]) == 3
    assert body["layout"]["boxes"]["conv_1"]["x"] >= 0


def test_model_load_bad_files(server):
    for name in ("bad_not_onnx.bin", "empty.onnx", "truncated.onnx"):
        status, body = _post(server + "/api/model/load", {"path": fx(name)})
        assert status == 422, name
        assert body["error"]["kind"] == "model_load_error"
        assert body["error"]["message"]
    # nonexistent path
    status, body = _post(server + "/api/model/load", {"path": "/no/such.onnx"})
    assert status == 422


def test_model_load_missing_path_field(server):
    status, body = _post(server + "/api/model/load", {})
    assert status == 400


def test_diff_end_to_end(server):
    status, body = _post(server + "/api/diff", {
        "path_a": fx("linear.onnx"), "path_b": fx("renamed_linear.onnx")})
    assert status == 200
    assert body["diff"]["summary"]["unchanged"] == 3
    assert body["match"]["summary"]["matched_count"] == 3
    assert body["layouts"]["a"]["boxes"] and body["layouts"]["b"]["boxes"]
    # matched pairs share y
    a_boxes = body["layouts"]["a"]["boxes"]
    b_boxes = body["layouts"]["b"]["boxes"]
    for m in body["match"]["matches"]:
        assert a_boxes[m["a_node_id"]]["y"] == b_boxes[m["b_node_id"]]["y"]


def test_diff_missing_args(server):
    status, body = _post(server + "/api/diff", {"path_a": fx("linear.onnx")})
    assert status == 400


def test_fold_roundtrip(server):
    status, body = _post(server + "/api/diff", {
        "path_a": fx("linear.onnx"), "path_b": fx("linear.onnx")})
    assert status == 200
    diff_id = body["diff_id"]
    # identical 3-node chain: is there a fold group? min size 3 -> yes
    groups = body["folding"]["groups"]
    assert groups, "identical 3-node chain should fold"
    gid = groups[0]["group_id"]
    status, folded = _post(server + "/api/fold", {
        "diff_id": diff_id, "folded_group_ids": [gid]})
    assert status == 200
    assert folded["views"]["a"]["hidden_count"] == 3
    kinds = {n["kind"] for n in folded["views"]["a"]["display_nodes"]}
    assert "group" in kinds
    # expand again
    status, expanded = _post(server + "/api/fold", {
        "diff_id": diff_id, "folded_group_ids": []})
    assert expanded["views"]["a"]["hidden_count"] == 0


def test_fold_bad_diff_id(server):
    status, body = _post(server + "/api/fold", {"diff_id": "nope"})
    assert status == 400


def test_profile_load_with_model(server):
    status, m = _post(server + "/api/model/load", {"path": fx("linear.onnx")})
    model_id = m["model_id"]
    status, body = _post(server + "/api/profile/load", {
        "path": fx("profile_linear.json"), "model_id": model_id})
    assert status == 200
    assert body["mapped"]["per_node"]["conv_1"]["duration_us"] == 620.5
    assert body["mapped"]["coverage"] == 1.0
    assert body["op_by_node"]["conv_1"] == "Conv"


def test_profile_load_without_model(server):
    status, body = _post(server + "/api/profile/load", {
        "path": fx("profile_linear.json")})
    assert status == 200
    assert "mapped" not in body  # no model to map against — honest response


def test_profile_load_bad_file(server):
    status, m = _post(server + "/api/model/load", {"path": fx("linear.onnx")})
    status, body = _post(server + "/api/profile/load", {
        "path": fx("bad_not_onnx.bin"), "model_id": m["model_id"]})
    assert status == 422
    assert body["error"]["kind"] == "profile_format_error"


def test_export_diff_formats(server):
    status, body = _post(server + "/api/diff", {
        "path_a": fx("linear.onnx"), "path_b": fx("op_replaced.onnx")})
    diff_id = body["diff_id"]
    for fmt, marker in (("json", b'"summary"'), ("csv", b"kind,match_id"),
                        ("md", b"# Model Diff Report"), ("html", b"<!doctype html>")):
        ctype, payload = _download(server + "/api/export/diff",
                                   {"diff_id": diff_id, "format": fmt})
        assert marker in payload, fmt
        if fmt == "json":
            parsed = json.loads(payload)
            assert parsed["summary"]["node_counts"]["op_changed"] == 1
        if fmt == "csv":
            rows = list(csv.reader(io.StringIO(payload.decode())))
            assert rows[0][0] == "kind"
            assert any(r[0] == "op_changed" for r in rows[1:])


def test_export_profile_formats(server):
    status, m = _post(server + "/api/model/load", {"path": fx("linear.onnx")})
    status, p = _post(server + "/api/profile/load", {
        "path": fx("profile_linear.json"), "model_id": m["model_id"]})
    pid = p["profile_id"]
    ctype, payload = _download(server + "/api/export/profile",
                               {"profile_id": pid, "format": "json"})
    parsed = json.loads(payload)
    assert parsed["coverage"] == 1.0
    assert any(n["node_id"] == "conv_1" for n in parsed["per_node"])
    ctype, payload = _download(server + "/api/export/profile",
                               {"profile_id": pid, "format": "csv"})
    rows = list(csv.reader(io.StringIO(payload.decode())))
    assert rows[0][:2] == ["node_id", "op_type"]


def test_delete_session(server):
    status, m = _post(server + "/api/model/load", {"path": fx("linear.onnx")})
    mid = m["model_id"]
    req = urllib.request.Request(server + f"/api/model/{mid}", method="DELETE")
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.status == 200
    status, body = _post(server + "/api/model/layout", {"model_id": mid})
    assert status == 500 or status == 400
    # unknown id delete -> 404
    req = urllib.request.Request(server + "/api/model/zzz", method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=30)
        assert False
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_invalid_json_body(server):
    req = urllib.request.Request(server + "/api/model/load", data=b"{bad json",
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30)
        assert False
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_model_upload_and_nodes(server):
    import base64
    with open(fx("linear.onnx"), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    status, body = _post(server + "/api/model/upload",
                         {"name": "linear.onnx", "data_base64": b64})
    assert status == 200
    assert body["meta"]["node_count"] == 3
    assert body["meta"]["source_path"] == "linear.onnx"
    # bad payload
    status, err = _post(server + "/api/model/upload", {"data_base64": "!!not-b64!!"})
    assert status == 400
    status, err = _post(server + "/api/model/upload",
                        {"data_base64": base64.b64encode(b"junk").decode()})
    assert status == 422
    # on-demand node details for expandable nodes
    status, nodes = _post(server + "/api/model/nodes", {
        "model_id": body["model_id"], "ids": ["conv_1"]})
    assert status == 200
    assert nodes["nodes"][0]["id"] == "conv_1"
    assert nodes["nodes"][0]["attributes"] == {} or isinstance(nodes["nodes"][0]["attributes"], dict)
    assert any(p["name"] == "conv_1.weight" for p in nodes["nodes"][0]["params"])


def test_profile_upload(server):
    import base64
    status, m = _post(server + "/api/model/load", {"path": fx("linear.onnx")})
    with open(fx("profile_linear.json"), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    status, body = _post(server + "/api/profile/upload", {
        "name": "profile_linear.json", "data_base64": b64,
        "model_id": m["model_id"]})
    assert status == 200
    assert body["mapped"]["coverage"] == 1.0
