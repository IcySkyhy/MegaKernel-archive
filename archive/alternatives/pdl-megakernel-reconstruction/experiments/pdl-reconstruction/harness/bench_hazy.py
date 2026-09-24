#!/usr/bin/env python3
"""Mechanism-oriented Hazy Megakernels benchmark with synthetic weights.

This intentionally avoids tokenizer/model downloads.  It instantiates the exact
Llama-3.2-1B geometry expected by the compiled H100 demo and emits one JSON row
per measured trial.  Headline comparisons should use multiple process-level
trials so model initialization and thermal order can be randomized externally.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch
from transformers import LlamaConfig

from megakernels.dispatch import make_mk_interpreter, make_schedule_builder
from megakernels.generators import MK_Generator
from megakernels.llama import LlamaForCausalLM
from megakernels.model_types import BatchState, ExtraModelConfig
from megakernels.scheduler import assign_to_sms, tensorize_instructions

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--modes", default="torch,torch_graph,mk,mk_graph,mk_kernel,mk_noop,rest_only")
    parser.add_argument("--contexts", default="1,128,512,2048")
    parser.add_argument("--layers", default="1,4,8,16")
    parser.add_argument("--schedulers", default="rr,zz,wave,dag")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Bracket measured iterations with cudaProfilerStart/Stop.",
    )
    return parser.parse_args()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def model_config(max_context: int) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=max(8192, max_context + 2),
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        tie_word_embeddings=True,
        torch_dtype=torch.bfloat16,
    )


def make_model(device: str, max_context: int, seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    extra = ExtraModelConfig(
        interleave_rope=True,
        max_len_override=max(8192, max_context + 2),
        max_batch_size=1,
    )
    model = LlamaForCausalLM(model_config(max_context), extra)
    model.device = torch.device(device)
    model.dtype = torch.bfloat16
    model.requires_grad_(False)
    model.model.interleave_rope()
    model.stack_params()
    model.setup_caches()
    return model


def time_samples(
    fn: Callable[[], object],
    warmup: int,
    iterations: int,
    cuda_profiler_range: bool = False,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStop()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends, strict=True)]


def capture_graph(fn: Callable[[], object]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def build_schedule(model: LlamaForCausalLM, scheduler: str, layers: int | None = None):
    builder = make_schedule_builder("latency")
    schedule = builder.build(model, layer_limit=layers)
    queues = assign_to_sms(scheduler, schedule=schedule)
    tensorize_instructions(schedule.globs, queues)
    return schedule


def reset_caches(model: LlamaForCausalLM) -> None:
    model.stacked_kv_cache[0].zero_()
    model.stacked_kv_cache[1].zero_()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def emit_rows(
    output,
    *,
    common: dict,
    variant: str,
    context: int,
    samples: list[float],
    correct: bool | None,
    trial: int,
    notes: str = "",
) -> None:
    summary = {
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "p10_us": percentile(samples, 0.10),
        "p90_us": percentile(samples, 0.90),
        "stdev_us": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_us": min(samples),
        "max_us": max(samples),
    }
    row = {
        **common,
        "variant": variant,
        "context": context,
        "trial": trial,
        "correct": correct,
        "notes": notes,
        "samples_us": samples,
        **summary,
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    if output is not None:
        output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    contexts = [int(item) for item in args.contexts.split(",") if item.strip()]
    layer_counts = [int(item) for item in args.layers.split(",") if item.strip()]
    schedulers = [item.strip() for item in args.schedulers.split(",") if item.strip()]
    if not contexts:
        raise ValueError("At least one context is required")

    torch.cuda.set_device(args.device)
    model = make_model(args.device, max(contexts), args.seed)
    interpreter = make_mk_interpreter(
        "latency", args.repo / "demos" / "low-latency-llama"
    )
    token = torch.tensor([[17]], device=args.device, dtype=torch.long)
    repo = args.repo
    common = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "framework": "hazy",
        "workload": "llama-3.2-1b-synthetic-decode",
        "gpu": torch.cuda.get_device_name(args.device),
        "device_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "repo_commit": git_commit(repo),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "pid": os.getpid(),
    }

    output = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("a", encoding="utf-8")

    schedule_cache: dict[tuple[str, int | None], object] = {}

    def schedule_for(scheduler: str, layers: int | None = None):
        key = (scheduler, layers)
        if key not in schedule_cache:
            schedule_cache[key] = build_schedule(model, scheduler, layers)
        return schedule_cache[key]

    correctness = True
    if not args.skip_correctness:
        reset_caches(model)
        torch_state = BatchState(
            input_ids=token,
            position_ids=torch.tensor([[0]], device=args.device, dtype=torch.long),
            seq_len=1,
        )
        torch_token = model(torch_state).output_ids
        reset_caches(model)
        schedule = schedule_for("rr")
        mk_gen = MK_Generator(model, interpreter, schedule)
        mk_token = mk_gen.run(token, pos_id=0)
        correctness = bool(
            torch.equal(torch_token.reshape(-1), mk_token.reshape(-1))
        )
        print(
            json.dumps(
                {
                    **common,
                    "record_type": "correctness",
                    "torch_token": torch_token.tolist(),
                    "mk_token": mk_token.tolist(),
                    "correct": correctness,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for context in contexts:
        pos_id = context - 1
        position_ids = torch.tensor([[pos_id]], device=args.device, dtype=torch.long)
        state = BatchState(input_ids=token, position_ids=position_ids, seq_len=context)

        context_correct = correctness
        context_reference_token = None
        if not args.skip_correctness:
            reset_caches(model)
            context_reference_token = model(state).output_ids
            reset_caches(model)
            context_schedule = schedule_for("rr")
            context_mk_token = MK_Generator(
                model, interpreter, context_schedule
            ).run(token, pos_id=pos_id)
            context_correct = bool(
                torch.equal(
                    context_reference_token.reshape(-1),
                    context_mk_token.reshape(-1),
                )
            )
            print(
                json.dumps(
                    {
                        **common,
                        "record_type": "correctness",
                        "context": context,
                        "torch_token": context_reference_token.tolist(),
                        "mk_token": context_mk_token.tolist(),
                        "correct": context_correct,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        for mode in modes:
            schedule = schedule_for("rr")
            notes = ""
            if mode == "torch":
                fn = lambda: model(state)
            elif mode == "torch_graph":
                graph = capture_graph(lambda: model(state))
                fn = graph.replay
            elif mode in {"mk", "mk_graph"}:
                gen = MK_Generator(model, interpreter, schedule)
                fn = lambda: gen.run(token, pos_id=pos_id)
                if mode == "mk_graph":
                    graph = capture_graph(fn)
                    fn = graph.replay
            elif mode == "mk_kernel":
                schedule.globs.pos_id = pos_id
                fn = lambda: (schedule.globs.barriers.zero_(), interpreter.interpret(schedule.globs))
            elif mode == "mk_noop":
                noop_schedule = build_schedule(model, "rr")
                noop_gen = MK_Generator(
                    model, interpreter, noop_schedule, skip_rest=True
                )
                noop_gen.replace_with_noops()
                fn = lambda: noop_gen.run(token, pos_id=pos_id)
            elif mode == "rest_only":
                gen = MK_Generator(model, interpreter, schedule, skip_mk=True)
                fn = lambda: gen.run(token, pos_id=pos_id)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            for trial in range(args.trials):
                samples = time_samples(
                    fn,
                    args.warmup,
                    args.iterations,
                    args.cuda_profiler_range,
                )
                mode_correct = (
                    context_correct
                    if mode
                    in {"torch", "torch_graph", "mk", "mk_graph", "mk_kernel"}
                    else None
                )
                emit_rows(
                    output,
                    common=common,
                    variant=mode,
                    context=context,
                    samples=samples,
                    correct=mode_correct,
                    trial=trial,
                    notes=notes,
                )

        for scheduler in schedulers:
            schedule = schedule_for(scheduler)
            schedule.globs.pos_id = pos_id
            scheduler_correct = None
            if context_reference_token is not None:
                reset_caches(model)
                scheduler_token = MK_Generator(
                    model, interpreter, schedule
                ).run(token, pos_id=pos_id)
                scheduler_correct = bool(
                    torch.equal(
                        context_reference_token.reshape(-1),
                        scheduler_token.reshape(-1),
                    )
                )
            fn = lambda s=schedule: (s.globs.barriers.zero_(), interpreter.interpret(s.globs))
            samples = time_samples(
                fn,
                args.warmup,
                args.iterations,
                args.cuda_profiler_range,
            )
            emit_rows(
                output,
                common=common,
                variant=f"mk_sched_{scheduler}",
                context=context,
                samples=samples,
                correct=scheduler_correct,
                trial=0,
            )

        for layers in layer_counts:
            if layers > model.config.num_hidden_layers:
                continue
            schedule = schedule_for("rr", layers)
            schedule.globs.pos_id = pos_id
            fn = lambda s=schedule: (s.globs.barriers.zero_(), interpreter.interpret(s.globs))
            samples = time_samples(
                fn,
                args.warmup,
                args.iterations,
                args.cuda_profiler_range,
            )
            emit_rows(
                output,
                common=common,
                variant=f"mk_layers_{layers}",
                context=context,
                samples=samples,
                correct=None,
                trial=0,
                notes="kernel-only partial schedule; no lm_head unless layers=16",
            )

    if output is not None:
        output.close()


if __name__ == "__main__":
    main()
