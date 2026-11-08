"""Netron viewer assets, loaded from the installed `netron` pip package.

The viewer frontend is not vendored in this repository. It ships with the
`netron` package (MIT License, Copyright (c) Lutz Roeder,
https://github.com/lutzroeder/netron) and is served at runtime from
site-packages. The telemetry-removal patch (frontend/netron.patch) is
applied in memory, keeping ModelToolbox strictly local-first: nothing is
ever sent to an external service.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re

NETRON_PIN = ">=9.3.5,<10"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.join(_HERE, "frontend", "netron.patch")

# Files the local patch touches; everything else is served as shipped.
_PATCHED_FILES = ("base.js", "browser.js")

_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

_cache: dict = {}


class NetronAssetsError(RuntimeError):
    """The viewer assets are missing or the local patch no longer applies."""


def netron_version() -> str:
    """Installed netron version, or 'unknown' if not importable metadata."""
    try:
        return importlib.metadata.version("netron")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def netron_dir() -> str:
    """Directory of the installed netron package (located, not executed)."""
    spec = importlib.util.find_spec("netron")
    if spec is None or not spec.submodule_search_locations:
        raise NetronAssetsError(
            "the 'netron' package is not installed; it provides the model "
            "viewer frontend. Install it with: pip install 'netron"
            + NETRON_PIN + "'"
        )
    return list(spec.submodule_search_locations)[0]


def _patch_lines() -> dict:
    """Parse frontend/netron.patch into {filename: [(old_start, old_count, lines), ...]}."""
    with open(_PATCH_PATH, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    hunks: dict = {}
    target = None
    current = None
    for line in lines:
        if line.startswith("--- "):
            continue
        if line.startswith("+++ b/"):
            target = line[len("+++ b/"):]
            hunks.setdefault(target, [])
            current = None
        else:
            m = _HUNK_RE.match(line)
            if m:
                if target is None:
                    raise NetronAssetsError(
                        f"netron.patch is malformed: hunk before any '+++ b/' "
                        f"header (near {line!r})")
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                current = (old_start, old_count, new_count, [])
                hunks[target].append(current)
            elif current is not None:
                old_start, old_count, new_count, changes = current
                done = sum(1 for t, _ in changes if t in (" ", "-"))
                done_new = sum(1 for t, _ in changes if t in (" ", "+"))
                if done >= old_count and done_new >= new_count:
                    current = None  # hunk complete; ignore trailing blank lines
                elif line.startswith((" ", "-", "+")):
                    changes.append((line[0], line[1:]))
                elif line.startswith("\\"):  # "\ No newline at end of file"
                    continue
                elif not line:
                    changes.append((" ", ""))
    return hunks


def _apply_hunks(source: str, hunks) -> str:
    """Apply unified-diff hunks to `source`; fail loudly on any mismatch."""
    src = source.splitlines()
    trailing_newline = source.endswith("\n")
    out = []
    pos = 0
    for old_start, old_count, _new_count, changes in hunks:
        # a count of 0 means "insert after line old_start"
        start = old_start if old_count == 0 else old_start - 1
        if start < pos or start > len(src):
            raise NetronAssetsError(
                "netron.patch does not match the installed netron version "
                f"({netron_version()}): hunk starts at line {old_start}, "
                "which is out of order. Re-generate the patch against the "
                "installed version."
            )
        out.extend(src[pos:start])
        pos = start
        for tag, text in changes:
            if tag in (" ", "-"):
                if pos >= len(src) or src[pos] != text:
                    raise NetronAssetsError(
                        "netron.patch does not match the installed netron "
                        f"version ({netron_version()}): expected {text!r} at "
                        f"line {pos + 1}. Re-generate the patch against the "
                        "installed version."
                    )
                if tag == " ":
                    out.append(text)
                pos += 1
            else:  # '+'
                out.append(text)
    out.extend(src[pos:])
    result = "\n".join(out)
    if trailing_newline:
        result += "\n"
    return result


def _check_name(name: str):
    parts = name.replace("\\", "/").split("/")
    if (not name or os.path.isabs(name) or ".." in parts
            or parts[0] in ("", ".") or parts[-1] in ("", ".")):
        raise NetronAssetsError(f"netron: invalid asset path {name!r}")


def read(name: str) -> bytes:
    """Return file content from the netron package, patched where applicable."""
    _check_name(name)
    if name in _cache:
        return _cache[name]
    path = os.path.join(netron_dir(), *name.split("/"))
    try:
        with open(path, "rb") as f:
            text = f.read().decode("utf-8")
    except OSError as exc:
        raise NetronAssetsError(f"netron: cannot read {name}: {exc}") from exc
    if name in _PATCHED_FILES:
        patched = _patch_lines().get(name, [])
        text = _apply_hunks(text, patched) if patched else text
    data = text.encode("utf-8")
    _cache[name] = data
    return data


def warmup():
    """Pre-load the patched viewer files; raises NetronAssetsError on failure."""
    read("index.html")
    for name in _PATCHED_FILES:
        read(name)
