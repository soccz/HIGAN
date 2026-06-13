"""Experiment IO helpers for reproducible paper runs.

These helpers keep protocol/config metadata out of ad hoc result code.  The
goal is not to make a successful-looking table, but to make every generated
metrics.json traceable to a fixed protocol file, command line, and code state.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[2]


def resolve_paper_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else PAPER / p


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fp:
            for block in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except FileNotFoundError:
        return None


def load_json(path: str | Path) -> dict[str, Any]:
    p = resolve_paper_path(path)
    if p is None:
        raise ValueError("path must not be None")
    return json.loads(p.read_text())


def protocol_metadata(protocol: str | Path | None,
                      protocol_key: str | None = None) -> dict[str, Any]:
    p = resolve_paper_path(protocol)
    if p is None:
        return {"protocol_path": None, "protocol_key": protocol_key}
    data = json.loads(p.read_text()) if p.exists() else None
    return {
        "protocol_path": str(p),
        "protocol_key": protocol_key,
        "protocol_sha256": sha256_file(p),
        "protocol_version": data.get("version") if isinstance(data, dict) else None,
    }


def execution_metadata(*, protocol: str | Path | None = None,
                       protocol_key: str | None = None,
                       extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta.update(protocol_metadata(protocol, protocol_key))
    if extra:
        meta.update(extra)
    return meta


def write_json_atomic(path: str | Path, payload: Any) -> None:
    p = resolve_paper_path(path)
    if p is None:
        raise ValueError("path must not be None")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(p)
