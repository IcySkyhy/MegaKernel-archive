#!/usr/bin/env python3
"""Interleaved full-model gate for the QSA MTP completed-group amendment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.hybrid_speculative import HybridStats, self_mtp_generate_step
import mlx_lm.models.qwen4_exp as qwen4


LAB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB_ROOT))
from bench_qwen38_flash_next_server_contexts import make_prompt  # noqa: E402


DEFAULT_MODEL = (
    "/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
)


def _digest(tokens: list[int]) -> str:
    body = ",".join(map(str, tokens)).encode("ascii")
    return hashlib.sha256(body).hexdigest()


def _thermal() -> dict[str, str]:
    result = subprocess.run(
        ["pmset", "-g", "therm"], check=False, capture_output=True, text=True
    )
    rows = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key.strip()] = value.strip()
    return rows


def _run(model, prompt, *, label: str, share: bool, amend: bool, depth: int,
         max_tokens: int) -> dict:
    qwen4._QSA_MTP_AMEND_COMPLETE_GROUPS = amend
    qwen4.qsa_mtp_amendment_status(reset=True)
    stats = HybridStats()
    tokens: list[int] = []
    thermal_before = _thermal()
    started = time.perf_counter()
    first = None
    for token, _logprobs, _from_draft in self_mtp_generate_step(
        prompt,
        model,
        num_draft=depth,
        max_tokens=max_tokens,
        persistent_mtp=True,
        mtp_share_qsa_indices=share,
        stats=stats,
    ):
        tokens.append(int(token))
        first = first or time.perf_counter()
    ended = time.perf_counter()
    if first is None:
        raise RuntimeError("MTP generation returned no tokens")
    receipt = qwen4.qsa_mtp_amendment_status()
    row = {
        "label": label,
        "share_qsa_indices": share,
        "amend_complete_groups": amend,
        "prompt_tokens": int(prompt.size),
        "completion_tokens": len(tokens),
        "ttft_s": first - started,
        "decode_tps_after_first": (len(tokens) - 1) / max(ended - first, 1e-9),
        "acceptance_rate": stats.draft_accepted / max(stats.draft_proposed, 1),
        "draft_proposed": stats.draft_proposed,
        "draft_accepted": stats.draft_accepted,
        "token_sha256": _digest(tokens),
        "token_ids": tokens,
        "qsa_amendment": receipt,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "thermal_before": thermal_before,
        "thermal_after": _thermal(),
    }
    mx.clear_cache()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--order", default="recompute,frozen,amended")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    text, _size, _filler = make_prompt(
        tokenizer, max(args.context - 3, 256), f"qsa-amend-{args.context}"
    )
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=True,
    )
    filler = tokenizer.encode(" x", add_special_tokens=False)
    if not filler:
        raise RuntimeError("tokenizer produced no alignment filler token")
    while len(prompt_ids) % 4 != 3:
        prompt_ids.append(int(filler[0]))
    prompt = mx.array(prompt_ids, dtype=mx.uint32)

    arms = {
        "recompute": (False, False),
        "frozen": (True, False),
        "amended": (True, True),
    }
    order = [part.strip() for part in args.order.split(",") if part.strip()]
    if not order or any(label not in arms for label in order):
        raise ValueError(f"order must use {sorted(arms)}")

    rows = []
    for index, label in enumerate(order):
        share, amend = arms[label]
        mx.reset_peak_memory()
        row = _run(
            model,
            prompt,
            label=label,
            share=share,
            amend=amend,
            depth=args.depth,
            max_tokens=args.max_tokens,
        )
        row["run_index"] = index
        rows.append(row)
        print(
            f"{index} {label}: {row['decode_tps_after_first']:.2f} t/s "
            f"accept={row['acceptance_rate']:.1%} qsa={row['qsa_amendment']}",
            flush=True,
        )

    if len({row["token_sha256"] for row in rows}) != 1:
        raise RuntimeError("greedy target output differs across QSA arms")
    amended_rows = [row for row in rows if row["label"] == "amended"]
    if not amended_rows or not all(
        row["qsa_amendment"]["amendments"] > 0
        and row["qsa_amendment"]["blocks_appended"] > 0
        for row in amended_rows
    ):
        raise RuntimeError("amended arm did not engage the completed-group mechanism")

    payload = {
        "schema": "mlx-uag.qsa-mtp-amendment-gate/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "mlx_version": importlib.metadata.version("mlx"),
        "context_requested": args.context,
        "prompt_tokens": int(prompt.size),
        "prompt_mod_compress_ratio": int(prompt.size) % 4,
        "mtp_num_draft": args.depth,
        "max_tokens": args.max_tokens,
        "persistent_mtp": True,
        "sampling": "greedy",
        "order": order,
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
