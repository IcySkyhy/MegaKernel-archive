#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

"""Run correctness-gated, sequential Qwen3.8 worker A/B benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time
from typing import TextIO


TAG = "QWEN_CHAT "
READY = TAG + "READY BF16_EXACT_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_worker(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("workers must use LABEL=/path/to/worker")
    label, path_raw = raw.split("=", 1)
    path = Path(path_raw).expanduser().resolve()
    if not label or not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(f"invalid worker: {raw}")
    return label, path


def parse_file(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {raw}")
    return path


def parse_executable(raw: str) -> Path:
    path = parse_file(raw)
    if not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(f"not executable: {raw}")
    return path


class LineReader:
    """Move blocking pipe reads to one daemon so protocol waits retain hard deadlines."""

    def __init__(self, stream: TextIO) -> None:
        self.lines: queue.Queue[str | None] = queue.Queue()

        def read() -> None:
            for line in stream:
                self.lines.put(line.rstrip("\r\n"))
            self.lines.put(None)

        threading.Thread(target=read, daemon=True).start()

    def read(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("worker response timed out")
        try:
            line = self.lines.get(timeout=remaining)
        except queue.Empty as error:
            raise TimeoutError("worker response timed out") from error
        if line is None:
            raise RuntimeError("worker exited before completing the protocol")
        return line


def read_line(reader: LineReader, deadline: float) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("worker response timed out")
    return reader.read(deadline)


def run_worker(
    label: str,
    worker: Path,
    model_dir: Path,
    prompt: str,
    warmup: int,
    repeats: int,
    tokens: int,
    timeout: float,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment["QWEN_MODEL_DIR"] = str(model_dir)
    process = subprocess.Popen(
        [str(worker)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    reader = LineReader(process.stdout)
    started = time.monotonic()
    deadline = started + timeout
    load_log: list[str] = []
    summaries: list[dict[str, object]] = []
    try:
        while True:
            line = read_line(reader, deadline)
            if line == READY:
                break
            if line.startswith(TAG + "READY"):
                raise RuntimeError(
                    f"{label}: worker lacks the exact BF16 autoregressive correctness gate"
                )
            if line.startswith(TAG + "ERROR "):
                raise RuntimeError(line[len(TAG + "ERROR ") :])
            load_log.append(line)

        process.stdin.write(f"BENCHMARK {warmup} {repeats} {tokens} {prompt}\n")
        process.stdin.flush()
        while True:
            line = read_line(reader, deadline)
            if line.startswith(TAG + "SUMMARY "):
                summaries.append(json.loads(line[len(TAG + "SUMMARY ") :]))
            elif line == TAG + f"DONE {repeats}":
                break
            elif line.startswith(TAG + "ERROR "):
                raise RuntimeError(line[len(TAG + "ERROR ") :])

        process.stdin.write("QUIT\n")
        process.stdin.flush()
        while read_line(reader, deadline) != TAG + "BYE":
            pass
        process.wait(timeout=max(1.0, deadline - time.monotonic()))
        if process.returncode != 0:
            raise RuntimeError(f"worker exited with status {process.returncode}")
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdin.close()
        process.stdout.close()
        raise
    process.stdin.close()
    process.stdout.close()
    if len(summaries) != 2:
        raise RuntimeError(f"{label}: expected two summaries, received {len(summaries)}")
    if not all(summary.get("correctnessOk") is True for summary in summaries):
        raise RuntimeError(f"{label}: benchmark correctness gate failed")
    cases = {str(summary.get("caseId")): summary for summary in summaries}
    if "qwen38_27b_decode" not in cases or "qwen38_27b_ttft" not in cases:
        raise RuntimeError(f"{label}: worker returned unexpected benchmark cases")
    return {
        "label": label,
        "protocol": "BF16_EXACT_V1",
        "worker": str(worker),
        "workerSha256": sha256(worker),
        "elapsedSeconds": time.monotonic() - started,
        "loadLog": load_log,
        "summaries": summaries,
        "decodeTokensPerSecondP50": cases["qwen38_27b_decode"].get(
            "throughputItemsPerSecondP50"
        ),
    }


def run_llama_baseline(
    llama_bench: Path,
    gguf: Path,
    warmup: int,
    repeats: int,
    tokens: int,
    timeout: float,
) -> dict[str, object]:
    command = [
        str(llama_bench),
        "-m",
        str(gguf),
        "-p",
        "0",
        "-n",
        str(tokens),
        "-ngl",
        "99",
        "-fa",
        "1",
        "-r",
        str(repeats),
        "-o",
        "json",
    ]
    if warmup == 0:
        command.append("--no-warmup")
    started = time.monotonic()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"llama-bench exited with status {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("llama-bench did not return JSON") from error
    if not isinstance(payload, list):
        raise RuntimeError("llama-bench JSON must be an array")
    generation = [
        row
        for row in payload
        if isinstance(row, dict)
        and row.get("n_prompt") == 0
        and row.get("n_gen") == tokens
    ]
    if len(generation) != 1:
        raise RuntimeError(
            f"expected one llama-bench generation row, received {len(generation)}"
        )
    row = generation[0]
    samples_raw = row.get("samples_ts")
    if not isinstance(samples_raw, list) or len(samples_raw) != repeats:
        raise RuntimeError(
            f"expected {repeats} llama-bench throughput samples, got {samples_raw!r}"
        )
    try:
        samples = [float(value) for value in samples_raw]
    except (TypeError, ValueError) as error:
        raise RuntimeError("llama-bench samples_ts contains a non-number") from error
    if any(value <= 0.0 for value in samples):
        raise RuntimeError("llama-bench samples_ts must be positive")
    return {
        "executable": str(llama_bench),
        "executableSha256": sha256(llama_bench),
        "gguf": str(gguf),
        "ggufBytes": gguf.stat().st_size,
        "command": command,
        "elapsedSeconds": time.monotonic() - started,
        "stderr": completed.stderr.splitlines(),
        "result": row,
        "decodeTokensPerSecondP50": statistics.median(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="append", type=parse_worker, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--prompt", default="1234", help="comma-separated token IDs")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--llama-bench", type=parse_executable)
    parser.add_argument("--gguf", type=parse_file)
    parser.add_argument("--require-beats-llama", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.tokens < 2 or args.timeout <= 0:
        parser.error("warmup must be nonnegative; repeats, tokens, and timeout must be positive")
    model_dir = args.model_dir.expanduser().resolve()
    if not (model_dir / "model.safetensors.index.json").is_file():
        parser.error(f"not a sharded Qwen checkpoint: {model_dir}")
    if (args.llama_bench is None) != (args.gguf is None):
        parser.error("--llama-bench and --gguf must be supplied together")
    if args.require_beats_llama and args.llama_bench is None:
        parser.error("--require-beats-llama requires a llama.cpp baseline")
    if args.llama_bench is not None and args.warmup not in (0, 1):
        parser.error("llama-bench supports either zero or one warmup run")

    llama_baseline = None
    if args.llama_bench is not None:
        print(f"benchmarking llama.cpp: {args.llama_bench}", flush=True)
        llama_baseline = run_llama_baseline(
            args.llama_bench,
            args.gguf,
            args.warmup,
            args.repeats,
            args.tokens,
            args.timeout,
        )
    results: list[dict[str, object]] = []
    for label, worker in args.worker:
        print(f"benchmarking {label}: {worker}", flush=True)
        results.append(
            run_worker(
                label,
                worker,
                model_dir,
                args.prompt,
                args.warmup,
                args.repeats,
                args.tokens,
                args.timeout,
            )
        )
    if llama_baseline is not None:
        baseline_p50 = float(llama_baseline["decodeTokensPerSecondP50"])
        for result in results:
            worker_p50 = float(result["decodeTokensPerSecondP50"])
            result["vsLlamaCpp"] = {
                "baselineDecodeTokensPerSecondP50": baseline_p50,
                "speedup": worker_p50 / baseline_p50,
                "beatsBaseline": worker_p50 > baseline_p50,
            }
    artifact = {
        "schemaVersion": 3,
        "correctnessGate": {
            "reference": "untimed_bf16_autoregressive",
            "candidate": "timed_mxfp8_autoregressive",
            "requirement": "exact_token_sequence_each_repeat",
            "referenceIncludedInTiming": False,
        },
        "modelDirectory": str(model_dir),
        "promptTokenIds": args.prompt,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "maxNewTokens": args.tokens,
        "llamaCppBaseline": llama_baseline,
        "workers": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    winner = max(results, key=lambda row: float(row["decodeTokensPerSecondP50"]))
    summary = f"winner={winner['label']} decode_p50={winner['decodeTokensPerSecondP50']}"
    if llama_baseline is not None:
        comparison = winner["vsLlamaCpp"]
        summary += (
            f" llama_p50={comparison['baselineDecodeTokensPerSecondP50']}"
            f" speedup={comparison['speedup']} beats_llama={comparison['beatsBaseline']}"
        )
    print(f"{summary} artifact={args.output}", flush=True)
    if args.require_beats_llama and not winner["vsLlamaCpp"]["beatsBaseline"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
