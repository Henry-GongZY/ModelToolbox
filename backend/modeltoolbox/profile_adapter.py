"""Profile adapters for the standard JSON and CSV formats.

Only documented formats are accepted — no vendor-format guessing.
Every measurement record is preserved (never silently dropped).

Standard JSON (v1):
{
  "schema_version": 1,
  "model": "model.onnx",
  "device": "target-board-name",
  "unit": "us",
  "end_to_end_us": 1500.0,            # optional
  "model_fingerprint": "<sha256>",    # optional extension
  "measurements": [
    {"node_name": "conv_1", "op_type": "Conv", "duration": 123.4, "calls": 1,
     "node_id": "...", "tensor": "...", "kernel_name": "...", "fused": false,
     "unit": "ms", "min": .., "avg": .., "max": ..}
  ]
}

CSV columns: node_name,op_type,duration_us,calls[,unit,node_id,tensor,kernel_name,fused]
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

UNIT_TO_US = {"us": 1.0, "μs": 1.0, "µs": 1.0, "ns": 0.001, "ms": 1000.0, "s": 1e6}

CSV_REQUIRED = ["node_name", "duration_us"]
CSV_KNOWN = ["node_name", "op_type", "duration_us", "calls", "unit", "node_id",
             "tensor", "kernel_name", "fused"]


class ProfileFormatError(Exception):
    """Raised for malformed profile files; message is human-readable."""


@dataclass
class ProfileRecord:
    index: int
    raw: Dict[str, Any]
    node_ref: Optional[str]
    ref_kind: str  # node_name | node_id | tensor | kernel | unknown
    op_type: Optional[str]
    duration_us: float
    calls: int
    fused: bool = False
    kernel_name: Optional[str] = None


@dataclass
class ProfileData:
    schema_version: int
    source_path: str
    model_name: Optional[str]
    device: Optional[str]
    unit: str
    model_fingerprint: Optional[str]
    end_to_end_us: Optional[float]
    records: List[ProfileRecord]
    format: str  # json | csv
    warnings: List[str] = field(default_factory=list)


def _num(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ProfileFormatError(f"{where}: duration must be a number, got {v!r}")
    if v < 0:
        raise ProfileFormatError(f"{where}: duration must be >= 0, got {v}")
    return float(v)


def _calls(v: Any, where: str) -> int:
    if v is None:
        return 1
    if isinstance(v, bool) or not isinstance(v, int):
        if isinstance(v, float) and v.is_integer():
            return int(v)
        raise ProfileFormatError(f"{where}: calls must be an integer, got {v!r}")
    if v < 0:
        raise ProfileFormatError(f"{where}: calls must be >= 0, got {v}")
    return int(v)


def _unit_factor(u: Optional[str], where: str) -> float:
    if u is None:
        return 1.0
    key = str(u).strip().lower()
    # normalize unicode micro sign variants
    if key in ("μs", "µs"):
        key = "us"
    if key not in UNIT_TO_US:
        raise ProfileFormatError(
            f"{where}: unsupported unit '{u}' (supported: us, ms, s, ns)")
    return UNIT_TO_US[key]


def load_profile(path: str) -> ProfileData:
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8-sig")
    except OSError as e:
        raise ProfileFormatError(f"cannot read profile file '{path}': {e}") from e
    except UnicodeDecodeError as e:
        raise ProfileFormatError(
            f"profile file '{path}' is not valid UTF-8 text: {e}") from e
    if not text.strip():
        raise ProfileFormatError(f"profile file '{path}' is empty")
    lower = path.lower()
    if lower.endswith(".json") or (not lower.endswith(".csv") and text.lstrip().startswith("{")):
        return _load_json(text, path)
    return _load_csv(text, path)


def load_profile_text(text: str, source_path: str = "") -> ProfileData:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return _load_json(text, source_path)
    return _load_csv(text, source_path)


def _load_json(text: str, path: str) -> ProfileData:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProfileFormatError(f"invalid JSON in '{path}': {e}") from e
    if not isinstance(doc, dict):
        raise ProfileFormatError("profile JSON root must be an object")
    schema = doc.get("schema_version")
    warnings: List[str] = []
    if schema is None:
        raise ProfileFormatError("profile JSON missing 'schema_version'")
    if not isinstance(schema, int):
        raise ProfileFormatError(f"'schema_version' must be an integer, got {schema!r}")
    if schema > 1:
        warnings.append(f"schema_version {schema} is newer than supported v1; parsing best-effort")
    if schema < 1:
        raise ProfileFormatError(f"unsupported schema_version {schema}")
    measurements = doc.get("measurements")
    if measurements is None:
        raise ProfileFormatError("profile JSON missing 'measurements'")
    if not isinstance(measurements, list):
        raise ProfileFormatError("'measurements' must be an array")

    unit = doc.get("unit", "us")
    base_factor = _unit_factor(unit, "global unit")
    e2e = doc.get("end_to_end_us")
    if e2e is not None:
        e2e = _num(e2e, "'end_to_end_us'") * base_factor

    records: List[ProfileRecord] = []
    for i, m in enumerate(measurements):
        where = f"measurements[{i}]"
        if not isinstance(m, dict):
            raise ProfileFormatError(f"{where}: must be an object")
        raw = dict(m)
        rec_unit = m.get("unit", unit)
        factor = _unit_factor(rec_unit, f"{where}.unit") if rec_unit != unit else base_factor
        if "duration" not in m:
            raise ProfileFormatError(f"{where}: missing 'duration'")
        duration_us = _num(m.get("duration"), f"{where}.duration") * factor
        calls = _calls(m.get("calls"), f"{where}.calls")

        node_id = m.get("node_id")
        node_name = m.get("node_name")
        tensor = m.get("tensor")
        kernel_name = m.get("kernel_name")
        fused = bool(m.get("fused", False))
        if node_id:
            ref, kind = str(node_id), "node_id"
        elif node_name:
            ref, kind = str(node_name), "node_name"
        elif tensor:
            ref, kind = str(tensor), "tensor"
        elif kernel_name:
            ref, kind = str(kernel_name), "kernel"
        else:
            ref, kind = None, "unknown"
            warnings.append(f"{where}: no node_name/node_id/tensor/kernel_name; record kept as unmapped")

        records.append(ProfileRecord(
            index=i, raw=raw, node_ref=ref, ref_kind=kind,
            op_type=m.get("op_type"), duration_us=duration_us, calls=calls,
            fused=fused, kernel_name=kernel_name,
        ))

    return ProfileData(
        schema_version=schema,
        source_path=path,
        model_name=doc.get("model"),
        device=doc.get("device"),
        unit=str(unit),
        model_fingerprint=doc.get("model_fingerprint") or doc.get("model_sha256"),
        end_to_end_us=e2e,
        records=records,
        format="json",
        warnings=warnings,
    )


def _load_csv(text: str, path: str) -> ProfileData:
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not rows:
        raise ProfileFormatError(f"profile CSV '{path}' has no rows")
    header = [h.strip() for h in rows[0]]
    missing = [c for c in CSV_REQUIRED if c not in header]
    if missing:
        raise ProfileFormatError(
            f"profile CSV '{path}' missing required column(s): {', '.join(missing)}")
    unknown = [c for c in header if c not in CSV_KNOWN]
    warnings = [f"unknown column(s) ignored: {', '.join(unknown)}"] if unknown else []

    unit_col = "unit" in header
    global_unit = "us"
    idx = {c: header.index(c) for c in header}

    # pre-scan unit column: if all rows share one unit, treat as global
    units = set()
    for rno, row in enumerate(rows[1:], start=2):
        if unit_col:
            if len(row) <= idx["unit"]:
                raise ProfileFormatError(f"row {rno}: expected {len(header)} columns, got {len(row)}")
            units.add(row[idx["unit"]].strip() or "us")
        if len(row) != len(header):
            raise ProfileFormatError(
                f"row {rno}: expected {len(header)} columns, got {len(row)}")
    if unit_col and len(units) == 1:
        global_unit = units.pop()

    factor = _unit_factor(global_unit, "unit column")
    records: List[ProfileRecord] = []
    for i, row in enumerate(rows[1:]):
        where = f"row {i + 2}"
        rec_unit = row[idx["unit"]].strip() if unit_col else global_unit
        f = _unit_factor(rec_unit, f"{where}.unit") if rec_unit != global_unit else factor
        try:
            duration = float(row[idx["duration_us"]])
        except ValueError:
            raise ProfileFormatError(
                f"{where}: duration_us must be a number, got {row[idx['duration_us']]!r}")
        duration_us = _num(duration, f"{where}.duration_us") * f
        calls = _calls(int(row[idx["calls"]]) if idx.get("calls") is not None and row[idx["calls"]].strip() else None, f"{where}.calls") \
            if "calls" in header else 1

        def col(name):
            j = idx.get(name)
            return row[j].strip() if j is not None and row[j].strip() else None

        node_id, node_name, tensor, kernel_name = (
            col("node_id"), col("node_name"), col("tensor"), col("kernel_name"))
        fused = (col("fused") or "").lower() in ("1", "true", "yes")
        if node_id:
            ref, kind = node_id, "node_id"
        elif node_name:
            ref, kind = node_name, "node_name"
        elif tensor:
            ref, kind = tensor, "tensor"
        elif kernel_name:
            ref, kind = kernel_name, "kernel"
        else:
            ref, kind = None, "unknown"
            warnings.append(f"{where}: no node_name/node_id/tensor; record kept as unmapped")
        raw = {header[j]: row[j] for j in range(min(len(header), len(row)))}
        records.append(ProfileRecord(
            index=i, raw=raw, node_ref=ref, ref_kind=kind,
            op_type=col("op_type"), duration_us=duration_us, calls=calls,
            fused=fused, kernel_name=kernel_name,
        ))

    return ProfileData(
        schema_version=1,
        source_path=path,
        model_name=None,
        device=None,
        unit=global_unit,
        model_fingerprint=None,
        end_to_end_us=None,
        records=records,
        format="csv",
        warnings=warnings,
    )
