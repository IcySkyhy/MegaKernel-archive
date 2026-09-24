"""On-disk cache of the per-rank sharded (and FP8-quantised) weights.

Sharding + quantising 236B parameters costs minutes per rank, so the result is
keyed by a signature over the snapshot, the shard geometry and the hashes of
the source files that produce it. Any change to those invalidates the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from dsv2mk.paths import weight_cache_dir


_PKG_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_FILES_FOR_SIG = (
    _PKG_ROOT / "runtime" / "weight_loader.py",
    _PKG_ROOT / "quant" / "__init__.py",
    _PKG_ROOT / "quant" / "fp8_block_quant.py",
)

_META_SIG = "dsv2mk_signature"
_META_VERSION = "dsv2mk_cache_version"
_CACHE_VERSION = "1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_signature(
    snapshot_dir: str | Path,
    layer_range: tuple[int, int],
    dtype: torch.dtype,
    fp8_stages: Optional[Iterable[str]],
    ep_size: int,
    ep_rank: int,
    tp_size: int,
    tp_rank: int,
    include_layer0: bool,
) -> str:
    snapshot_dir = Path(snapshot_dir)
    fp8_list = sorted(list(fp8_stages)) if fp8_stages else []
    src_hashes = {p.name: _sha256_file(p) for p in _SOURCE_FILES_FOR_SIG if p.exists()}
    payload = {
        "snapshot": snapshot_dir.name,
        "layer_range": list(layer_range),
        "dtype": str(dtype),
        "fp8_stages": fp8_list,
        "ep_size": int(ep_size),
        "ep_rank": int(ep_rank),
        "tp_size": int(tp_size),
        "tp_rank": int(tp_rank),
        "include_layer0": bool(include_layer0),
        "cache_version": _CACHE_VERSION,
        "src_hashes": src_hashes,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(cache_dir: Path, signature: str, tp_rank: int, ep_rank: int) -> Path:
    return Path(cache_dir) / f"rank{tp_rank}-{ep_rank}_{signature[:16]}.safetensors"


def _purge_stale_for_rank(cache_dir: Path, tp_rank: int, ep_rank: int,
                          keep_signature: str | None = None) -> int:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    prefix = f"rank{tp_rank}-{ep_rank}_"
    keep_stem = f"rank{tp_rank}-{ep_rank}_{keep_signature[:16]}" if keep_signature else None
    bytes_freed = 0
    for p in cache_dir.glob(f"{prefix}*.safetensors"):
        if keep_stem is not None and p.stem == keep_stem:
            continue
        try:
            bytes_freed += p.stat().st_size
            p.unlink()
        except OSError:
            pass
    for p in cache_dir.glob(f"{prefix}*.safetensors.tmp"):
        try:
            bytes_freed += p.stat().st_size
            p.unlink()
        except OSError:
            pass
    return bytes_freed


def try_load_cached(
    cache_dir: str | Path,
    signature: str,
    device: str,
    tp_rank: int,
    ep_rank: int,
) -> Optional[dict]:
    path = _cache_path(Path(cache_dir), signature, tp_rank, ep_rank)
    if not path.exists():
        return None

    out: dict = {}
    try:
        with safe_open(str(path), framework="pt", device=device) as f:
            meta = f.metadata() or {}
            if meta.get(_META_SIG) and meta[_META_SIG] != signature:
                return None
            for k in f.keys():
                out[k] = f.get_tensor(k)
    except Exception:
        out = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if device != "cpu":
                    t = t.to(device)
                out[k] = t.contiguous() if not t.is_contiguous() else t

    if "_layer_range" in out:
        lr = out.pop("_layer_range").to("cpu").tolist()
        out["layer_range"] = (int(lr[0]), int(lr[1]))
    return out


def _estimate_bytes(weights: dict) -> int:
    total = 0
    for v in weights.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
    return total


def save_cache(
    cache_dir: str | Path,
    signature: str,
    weights: dict,
    tp_rank: int,
    ep_rank: int,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    to_write: dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        if k == "config":
            continue
        if k == "layer_range":
            if isinstance(v, range):
                lr_start, lr_stop = int(v.start), int(v.stop)
            else:
                lr_start, lr_stop = int(v[0]), int(v[1])
            to_write["_layer_range"] = torch.tensor(
                [lr_start, lr_stop], dtype=torch.int64
            )
            continue
        if not isinstance(v, torch.Tensor):
            continue
        t = v.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        to_write[k] = t

    freed = _purge_stale_for_rank(cache_dir, tp_rank, ep_rank,
                                  keep_signature=signature)
    if freed > 0:
        print(f"[weight_cache] purged {freed / 1024**3:.1f} GiB of "
              f"stale rank{tp_rank}-{ep_rank} cache before write")

    safety_gib = float(os.environ.get("DSV2MK_CACHE_SAFETY_GIB", "50"))
    safety_bytes = int(safety_gib * 1024**3)
    needed = _estimate_bytes(to_write) + (16 << 20)
    free = shutil.disk_usage(cache_dir).free
    if free < needed + safety_bytes:
        raise RuntimeError(
            f"insufficient disk space for weight cache at {cache_dir}: "
            f"need {needed / 1024**3:.1f} GiB + {safety_gib:.0f} GiB safety = "
            f"{(needed + safety_bytes) / 1024**3:.1f} GiB, have "
            f"{free / 1024**3:.1f} GiB free. "
            f"Lower the safety margin via DSV2MK_CACHE_SAFETY_GIB=<n>, "
            f"set DSV2MK_WEIGHT_CACHE to a larger filesystem, "
            f"or disable caching with DSV2MK_WEIGHT_CACHE_DISABLE=1."
        )

    path = _cache_path(cache_dir, signature, tp_rank, ep_rank)
    tmp = path.with_suffix(".safetensors.tmp")
    save_file(
        to_write,
        str(tmp),
        metadata={
            _META_SIG: signature,
            _META_VERSION: _CACHE_VERSION,
            "tp_rank": str(tp_rank),
            "ep_rank": str(ep_rank),
        },
    )
    os.replace(tmp, path)


def purge_all(cache_dir: str | Path) -> int:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    bytes_freed = 0
    for p in cache_dir.glob("rank*.safetensors"):
        try:
            bytes_freed += p.stat().st_size
            p.unlink()
        except OSError:
            pass
    for p in cache_dir.glob("rank*.safetensors.tmp"):
        try:
            bytes_freed += p.stat().st_size
            p.unlink()
        except OSError:
            pass
    return bytes_freed


def cache_enabled() -> bool:
    return os.environ.get("DSV2MK_WEIGHT_CACHE_DISABLE", "0") not in ("1", "true", "True")


def default_cache_dir() -> Path:
    return weight_cache_dir()

