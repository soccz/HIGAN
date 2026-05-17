"""Deterministic-execution helper for paper-grade reproducibility.

Usage at the top of every experiment script:

    from lib.reproducibility import set_deterministic, run_metadata
    set_deterministic(seed=2027)
    ...
    payload["_meta"] = run_metadata(seed=2027)
    json.dump(payload, fp, indent=2)

Calling set_deterministic() pins:
- torch / cuda / cuda_all manual_seed
- numpy + random module seed
- torch.backends.cudnn.deterministic = True, .benchmark = False
- CUBLAS_WORKSPACE_CONFIG for the deterministic cuBLAS workspace
- torch.use_deterministic_algorithms(True, warn_only=True)
  (warn_only because some ops — e.g. scatter_add for atomic — have no
  deterministic CUDA kernel; we accept those warnings rather than failing.)

run_metadata() returns a dict capturing the run environment so it can
be embedded in each metrics.json:
- seed actually used
- torch / numpy / python versions
- git commit (if inside a repo)
- GPU name + CUDA driver
- timestamp (UTC, ISO8601)
"""
from __future__ import annotations
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


_DETERMINISTIC_SET = False
_CURRENT_SEED: int | None = None


def set_deterministic(seed: int = 2027, *, strict: bool = False) -> None:
    """Pin all random sources and disable non-deterministic CUDA kernels.

    Args:
        seed: integer seed applied to all RNGs.
        strict: if True, use_deterministic_algorithms(warn_only=False) —
            any op without a deterministic kernel will raise. Default
            False (warn-only) because torch.func.jvp through some
            attention ops emits warnings under strict mode.
    """
    global _DETERMINISTIC_SET, _CURRENT_SEED
    import torch
    import numpy as np

    # Cublas workspace required for deterministic matmul on CUDA 10.2+
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN: disable autotuner + force deterministic conv kernels
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch-wide deterministic mode
    try:
        torch.use_deterministic_algorithms(True, warn_only=not strict)
    except RuntimeError:
        # older torch versions
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    _DETERMINISTIC_SET = True
    _CURRENT_SEED = seed


def _git_commit() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        ).strip()
        return sha
    except Exception:
        return None


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        )
        return bool(out.strip())
    except Exception:
        return False


def run_metadata(*, seed: int | None = None,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a JSON-serialisable metadata dict to embed in metrics.json."""
    import torch
    import numpy as np

    meta: dict[str, Any] = {
        "seed": seed if seed is not None else _CURRENT_SEED,
        "deterministic_set": _DETERMINISTIC_SET,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": (torch.version.cuda
                 if torch.version.cuda else "cpu-only"),
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }
    if extra:
        meta.update(extra)
    return meta
