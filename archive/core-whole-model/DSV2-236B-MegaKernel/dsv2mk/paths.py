"""Resolution of the DeepSeek-V2 weight snapshot and the sharded-weight cache."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-V2"
_HUB_DIR_NAME = "models--" + MODEL_ID.replace("/", "--")


def model_snapshot() -> str:
    """Directory holding config.json, the tokenizer and the safetensors shards.

    Set DSV2MK_SNAPSHOT to point at an explicit directory; otherwise the newest
    snapshot of MODEL_ID in the HuggingFace hub cache is used.
    """
    env = os.environ.get("DSV2MK_SNAPSHOT")
    if env:
        path = Path(env)
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"DSV2MK_SNAPSHOT={env} has no config.json")
        return str(path) + os.sep

    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snapshots = hub / _HUB_DIR_NAME / "snapshots"
    candidates = sorted(
        (d for d in snapshots.glob("*") if (d / "config.json").exists()),
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {MODEL_ID} snapshot under {snapshots}. Download it first, or set "
            f"DSV2MK_SNAPSHOT to a directory containing config.json."
        )
    return str(candidates[-1]) + os.sep


def weight_cache_dir() -> Path:
    """Where the per-rank TP/EP-sharded (and FP8-quantised) weights are cached."""
    env = os.environ.get("DSV2MK_WEIGHT_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dsv2mk" / "weights"
