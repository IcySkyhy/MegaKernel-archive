#!/usr/bin/env python3
"""Dump the official single-layer reference activations.

    /workspace/venvs/baseline/bin/python3 run_reference.py \
        --layer 0 --m 1 --tag layer0_decode_m1

Outputs `reference/<tag>/manifest.json` + one `.npy` per dumped segment.

Layer-boundary input modes (`--input`):

  embed       hidden = embed_tokens(input_ids).repeat(1, hc_count)
              This is EXACTLY what the model does at nvidia/model.py:506, so for
              layer 0 (no PLE upstream) it is the true layer input.
  synthetic   deterministic normal(0, --sigma) bf16 with the RNG seeded by
              (--seed, layer, m). Use when no real activations are available.
  chain:N     run layers 0..N-1 with the real weights (PLE skipped, see README)
              and use the resulting state as layer N's input. The result is the
              true layer-N input *of the model with PLE disabled*.

`--pending` selects the layer-boundary pending state:
  none        prev_block_output = prev_injection = None (real layer-0 state)
  synth       deterministic pending block/injection (exercises the fused
              combine_and_mix branch of the *first* mixer too)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Optional CPU bound for a shared box: `M39_THREADS=4 python3 run_reference.py ...`
# Unset -> torch's default intra-op thread count (which is what the archived
# evidence was produced with; changing it is not expected to change any bit,
# but pin it if you need to reproduce on a loaded machine).
if os.environ.get("M39_THREADS"):
    torch.set_num_threads(int(os.environ["M39_THREADS"]))

from ref.ckpt import LayerWeights  # noqa: E402
from ref.layer import DecoderLayer, final_mixer  # noqa: E402

DTYPE_ALIASES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
}


def to_numpy(t: torch.Tensor):
    """-> (ndarray, dtype-name). bf16 is stored as uint16 bits."""
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy(), "bfloat16"
    if t.dtype == torch.float32:
        return t.numpy(), "float32"
    if t.dtype == torch.int32:
        return t.numpy(), "int32"
    if t.dtype == torch.int64:
        return t.numpy(), "int64"
    if t.dtype == torch.uint8:
        return t.numpy(), "uint8"
    raise TypeError(f"unsupported dump dtype {t.dtype}")


def save_dump(out_dir: Path, taps: dict, extra: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"segments": {}, "extra": extra or {}}
    for name, val in sorted(taps.items()):
        if val is None:
            continue
        if not isinstance(val, torch.Tensor):
            continue
        arr, dt = to_numpy(val)
        fname = name.replace("/", "_") + ".npy"
        np.save(out_dir / fname, arr)
        manifest["segments"][name] = {
            "file": fname,
            "dtype": dt,
            "shape": list(arr.shape),
            "sha256_stored_bytes": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------------------
def make_hidden(cfg, mode: str, layer_idx: int, m: int, seed: int, sigma: float,
                ids: torch.Tensor | None = None):
    hc = cfg["hc_count"]
    hidden_size = cfg["hidden_size"]
    if mode == "embed":
        emb = LayerWeights.embed_tokens().float()  # [vocab, 2560]
        if ids is None:
            ids = torch.arange(1000, 1000 + m, dtype=torch.long) % emb.shape[0]
        e = emb[ids]  # [m, 2560]
        return e.repeat(1, hc).to(torch.bfloat16)
    if mode == "synthetic":
        g = torch.Generator().manual_seed(seed * 100003 + layer_idx)
        e = torch.randn(m, hidden_size, generator=g) * sigma
        return e.repeat(1, hc).to(torch.bfloat16)
    raise ValueError(f"unknown input mode {mode!r}")


def make_pending(cfg, mode: str, layer_idx: int, m: int, seed: int, sigma: float):
    if mode == "none":
        return None, None
    if mode == "synth":
        g = torch.Generator().manual_seed(seed * 7919 + layer_idx + 17)
        block = (torch.randn(m, cfg["hidden_size"], generator=g) * sigma).to(torch.bfloat16)
        inj = (torch.randn(m, cfg["hc_count"], generator=g) * 0.5).to(torch.bfloat16)
        return block, inj
    raise ValueError(f"unknown pending mode {mode!r}")


def run_layer(
    layer_idx: int,
    m: int,
    input_mode: str,
    pending: str,
    seed: int,
    sigma: float,
    precision: str,
    start_pos: int,
    warmup: int = 0,
):
    """Run one layer once; returns (taps, meta).

    `warmup > 0` first runs `warmup` tokens through the layer WITHOUT dumping
    them, so the mamba ssm/conv states (GDN) and the paged-equivalent caches
    (QSA) are non-degenerate when the dumped step runs. This matters: a bare
    first decode step at position 0 has `ssm_state == 0` and
    `visible_blocks == 0`, which makes the dump a poor test target.
    """
    t0 = time.time()
    weights = LayerWeights(layer_idx)
    cfg = weights.text_config()
    layer = DecoderLayer(layer_idx, weights, cfg, precision=precision)
    build_s = time.time() - t0

    positions = torch.arange(start_pos + warmup, start_pos + warmup + m, dtype=torch.long)
    taps: dict[str, torch.Tensor] = {}
    meta = {
        "layer_idx": layer_idx,
        "layer_type": layer.layer_type,
        "m": m,
        "dump_positions": [int(positions[0]), int(positions[-1])],
        "input_mode": input_mode,
        "pending": pending,
        "seed": seed,
        "sigma": sigma,
        "moe_precision": precision,
        "start_pos": start_pos,
        # NOTE: no wall-clock values here. They used to live in `extra` and made
        # the manifest sha256 (printed in the comparison reports) change on every
        # rebuild, i.e. the archived evidence was not byte-reproducible.
        # Timing is recorded in stdout -> evidence/reference_build.log instead.
    }

    state = layer.init_state()

    def _make_hidden_for(off: int, count: int):
        if input_mode.startswith("chain:"):
            return None  # handled below
        if input_mode == "embed":
            ids = torch.arange(1000 + off, 1000 + off + count, dtype=torch.long)
            return make_hidden(cfg, "embed", layer_idx, count, seed, sigma, ids=ids)
        return make_hidden(cfg, input_mode, layer_idx, count, seed + off, sigma)

    if input_mode.startswith("chain:"):
        nprior = int(input_mode.split(":")[1])
        hidden = make_hidden(cfg, "embed", 0, m, seed, sigma)
        block_out = None
        inj = None
        prior = []
        for li in range(nprior):
            lw = LayerWeights(li)
            lc = lw.text_config()
            lay = DecoderLayer(li, lw, lc, precision=precision)
            st = lay.init_state()
            prior.append(li)
            hidden, block_out, inj, st, _ = lay.forward(
                hidden, block_out, inj, positions, state=st
            )
        meta["chain"] = prior
    else:
        if warmup:
            # prime the mamba/attention state; this pass is NOT dumped
            wpos = torch.arange(start_pos, start_pos + warmup, dtype=torch.long)
            wh = _make_hidden_for(start_pos, warmup)
            _, _, _, state, _ = layer.forward(wh, None, None, wpos, state=state)
            meta["warmup"] = warmup
        hidden = _make_hidden_for(start_pos + warmup, m)

    prev_block, prev_inj = make_pending(cfg, pending, layer_idx, m, seed, sigma)
    taps["layer.input.hidden"] = hidden
    if prev_block is not None:
        taps["layer.input.prev_block_output"] = prev_block
        taps["layer.input.prev_injection"] = prev_inj

    hidden_out, mlp_out, inj_out, state_out, ltaps = layer.forward(
        hidden, prev_block, prev_inj, positions, state=state
    )
    taps.update(ltaps)
    taps["layer.out.hidden"] = hidden_out
    taps["layer.out.block_output"] = mlp_out
    taps["layer.out.injection"] = inj_out
    if state_out is not None:
        conv_state, ssm_state = state_out
        taps["layer.state.conv_state"] = conv_state
        taps["layer.state.ssm_state"] = ssm_state

    meta["_timing"] = {"build_seconds": round(build_s, 2),
                       "run_seconds": round(time.time() - t0 - build_s, 2)}
    return taps, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, required=True, help="0-based layer index (0 = GDN, 3 = QSA)")
    ap.add_argument("--m", type=int, default=1, help="tokens in this call (1 = decode)")
    ap.add_argument("--tag", required=True, help="output subdirectory under reference/")
    ap.add_argument("--input", default=None, help="embed | synthetic | chain:N (default: embed for layer 0, synthetic otherwise)")
    ap.add_argument("--pending", default="none", choices=["none", "synth"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=0.05, help="synthetic input stddev (embed activations are ~0.05)")
    ap.add_argument("--start-pos", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=0,
                    help="prime the GDN/QSA state with this many tokens first (not dumped)")
    ap.add_argument("--moe-precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--out", default=str(HERE / "reference"))
    ap.add_argument("--final-mixer", action="store_true", help="also run the global final mixer on the layer output")
    args = ap.parse_args()

    input_mode = args.input or ("embed" if args.layer == 0 else "synthetic")
    taps, meta = run_layer(
        args.layer, args.m, input_mode, args.pending, args.seed, args.sigma,
        args.moe_precision, args.start_pos, warmup=args.warmup,
    )

    if args.final_mixer:
        weights = LayerWeights(args.layer)
        cfg = weights.text_config()
        mw = LayerWeights.global_mixer()
        _, multi_hidden, sample_hidden = final_mixer(
            taps["layer.out.hidden"], taps["layer.out.block_output"],
            taps["layer.out.injection"], mw, cfg["hc_count"],
        )
        taps["final.multi_hidden"] = multi_hidden
        taps["final.sample_hidden"] = sample_hidden

    out_dir = Path(args.out) / args.tag
    # wall time is printed, not stored in the manifest (see meta["_timing"] above)
    timing = meta.pop("_timing", {})
    manifest = save_dump(out_dir, taps, extra=meta)

    print(f"[run_reference] tag={args.tag} layer={args.layer} type={meta['layer_type']} m={args.m}")
    print(f"  input={input_mode} pending={args.pending} moe={args.moe_precision} "
          f"build={timing.get('build_seconds')}s run={timing.get('run_seconds')}s "
          f"[wall clock, shared box -> qualitative only, NOT a manifest field "
          f"(this stdout line IS tee'd into evidence/reference_build.log)]")
    try:
        shown = out_dir.relative_to(HERE)
    except ValueError:
        shown = out_dir.name          # never print an absolute checkout path:
    print(f"  {len(manifest['segments'])} segments -> {shown}")  # it would make
    # evidence/reference_build.log differ when the package is reproduced elsewhere
    for name in sorted(manifest["segments"]):
        s = manifest["segments"][name]
        print(f"    {name:34s} {s['dtype']:8s} {str(s['shape']):18s} {s['sha256_stored_bytes'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
