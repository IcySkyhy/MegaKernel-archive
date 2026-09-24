#!/usr/bin/env python3
"""Score mini-sgl samples against multiple single-node vLLM servers.

The AFD large-node alignment run produces one sample.json.  For the vLLM
baseline, scoring is embarrassingly parallel: each prompt+generation can be
sent to any vLLM server with the same model.  This helper starts one vLLM
server per Ray node, shards the samples across those servers, and merges the
alignment report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import ray

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT = SCRIPT_DIR.parents[1]
VALIDATE_DIR = SCRIPT_ROOT / "validate"
for path in (SCRIPT_DIR, VALIDATE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_minisgl_vllm import (  # noqa: E402
    MiniSGLSample,
    compare_prompt,
    get_first_served_model,
    load_tokenizer,
    print_report,
    sample_bundle_from_json,
    score_with_vllm,
    summarize,
)


def _node_resource_key(node_ip: str) -> str:
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        if node.get("NodeManagerAddress") != node_ip:
            continue
        for resource_name in node.get("Resources", {}):
            if str(resource_name).startswith("node:"):
                return str(resource_name)
    raise RuntimeError(f"Could not find Ray node resource for {node_ip}")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_name(value: str) -> str:
    return value.replace(".", "_").replace(":", "_").replace("-", "_")


def _http_ready(url: str, timeout: float = 2.0) -> bool:
    try:
        with urlopen(f"{url.rstrip('/')}/v1/models", timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, URLError):
        return False


def _requires_vllm_deepep(extra_args: str) -> bool:
    try:
        argv = shlex.split(extra_args)
    except ValueError as exc:
        raise RuntimeError(f"Invalid --extra-args: {extra_args!r}") from exc
    for index, arg in enumerate(argv):
        if arg == "--all2all-backend" and index + 1 < len(argv):
            return argv[index + 1].startswith("deepep_")
        if arg.startswith("--all2all-backend="):
            return arg.split("=", 1)[1].startswith("deepep_")
    return False


def _check_vllm_deepep_available(extra_args: str) -> None:
    if not _requires_vllm_deepep(extra_args):
        return
    if importlib.util.find_spec("deep_ep") is not None:
        return
    raise RuntimeError(
        "vLLM DeepEP all2all was requested in --extra-args, but Python package "
        "`deep_ep` is not importable in the active environment. Install vLLM's "
        "EP kernels first (see "
        "https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md) "
        "or use a non-DeepEP vLLM backend for scoring."
    )


class _VllmServerActor:
    def __init__(
        self,
        *,
        calib_dir: str,
        conda_sh: str,
        env_name: str,
        cuda_home: str,
        model: str,
        port: int,
        host: str,
        tp_size: int,
        extra_args: str,
        log_path: str,
    ) -> None:
        self.calib_dir = calib_dir
        self.conda_sh = conda_sh
        self.env_name = env_name
        self.cuda_home = cuda_home
        self.model = model
        self.port = int(port)
        self.host = host
        self.tp_size = int(tp_size)
        self.extra_args = extra_args
        self.log_path = log_path
        self.proc: subprocess.Popen[str] | None = None
        self.log_file: Any = None

    def start(self) -> dict[str, Any]:
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        argv = [
            "vllm",
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tp_size),
        ] + shlex.split(self.extra_args)
        quoted = " ".join(shlex.quote(arg) for arg in argv)
        script = f"""
set -eo pipefail
source {shlex.quote(self.conda_sh)}
conda activate {shlex.quote(self.env_name)}
cd {shlex.quote(self.calib_dir)}

HOSTNAME_SAFE="$(hostname | tr '.-' '__')"
DEEPEP_BUILD_LIB_DIR="${{DEEPEP_BUILD_LIB_DIR:-{self.calib_dir}/cache/vllm_deepep_epv2_sm100/${{HOSTNAME_SAFE}}/build/lib.linux-aarch64-cpython-312}}"
DEEPGEMM_LIB_DIR="${{DEEPGEMM_LIB_DIR:-{self.calib_dir}/cache/vllm_deepgemm_build/${{HOSTNAME_SAFE}}/build/lib.linux-aarch64-cpython-312}}"
export PYTHONPATH="${{DEEPEP_BUILD_LIB_DIR}}:${{DEEPGEMM_LIB_DIR}}:${{PYTHONPATH:-}}"

export CUDA_HOME="${{CUDA_HOME:-{self.cuda_home}}}"
export CUDA_PATH="${{CUDA_PATH:-$CUDA_HOME}}"
export CUDA_NVCC_EXECUTABLE="${{CUDA_NVCC_EXECUTABLE:-$CUDA_HOME/bin/nvcc}}"
export TRITON_PTXAS_BLACKWELL_PATH="${{TRITON_PTXAS_BLACKWELL_PATH:-$CUDA_HOME/bin/ptxas}}"

export DG_JIT_CACHE_DIR="${{DG_JIT_CACHE_DIR:-{self.calib_dir}/cache/vllm_deepgemm/${{HOSTNAME_SAFE}}}}"
export EP_JIT_CACHE_DIR="${{EP_JIT_CACHE_DIR:-{self.calib_dir}/cache/vllm_deepep_epv2_jit/${{HOSTNAME_SAFE}}}}"
export VLLM_CACHE_ROOT="${{VLLM_CACHE_ROOT:-{self.calib_dir}/cache/vllm/${{HOSTNAME_SAFE}}}}"
export FLASHINFER_WORKSPACE_BASE="${{FLASHINFER_WORKSPACE_BASE:-{self.calib_dir}/cache/flashinfer/${{HOSTNAME_SAFE}}}}"
mkdir -p "$DG_JIT_CACHE_DIR" "$EP_JIT_CACHE_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"

export EP_SUPPRESS_NCCL_CHECK="${{EP_SUPPRESS_NCCL_CHECK:-1}}"
export VLLM_USE_DEEP_GEMM="${{VLLM_USE_DEEP_GEMM:-1}}"
export VLLM_MOE_USE_DEEP_GEMM="${{VLLM_MOE_USE_DEEP_GEMM:-1}}"
export VLLM_USE_DEEP_GEMM_E8M0="${{VLLM_USE_DEEP_GEMM_E8M0:-1}}"
export VLLM_DEEP_GEMM_WARMUP="${{VLLM_DEEP_GEMM_WARMUP:-skip}}"
export VLLM_DEEPEPLL_FP8_DISPATCH="${{VLLM_DEEPEPLL_FP8_DISPATCH:-1}}"
export VLLM_DEEPEPLL_UE8M0_DISPATCH="${{VLLM_DEEPEPLL_UE8M0_DISPATCH:-0}}"
export VLLM_USE_FLASHINFER_MOE_FP8="${{VLLM_USE_FLASHINFER_MOE_FP8:-0}}"
export VLLM_WORKER_MULTIPROC_METHOD="${{VLLM_WORKER_MULTIPROC_METHOD:-spawn}}"
export VLLM_RPC_TIMEOUT="${{VLLM_RPC_TIMEOUT:-1800000}}"
export NVSHMEM_QP_DEPTH="${{NVSHMEM_QP_DEPTH:-4096}}"
export PYTHONUNBUFFERED=1

exec {quoted}
"""
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            ["bash", "-lc", script],
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return {
            "pid": self.proc.pid,
            "hostname": socket.gethostname(),
            "log_path": self.log_path,
        }

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def poll(self) -> int | None:
        if self.proc is None:
            return None
        return self.proc.poll()

    def stop(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
            except ProcessLookupError:
                pass
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None


def _start_servers(args: argparse.Namespace, nodes: list[str]) -> tuple[list[Any], list[str]]:
    actors: list[Any] = []
    urls: list[str] = []
    RemoteServer = ray.remote(_VllmServerActor)
    for index, node_ip in enumerate(nodes):
        log_path = str(Path(args.log_dir) / f"vllm_shard_{index:02d}_{_safe_name(node_ip)}.log")
        actor = RemoteServer.options(
            num_cpus=1,
            num_gpus=int(args.gpus_per_server),
            resources={_node_resource_key(node_ip): 0.01},
        ).remote(
            calib_dir=str(args.calib_dir),
            conda_sh=str(args.conda_sh),
            env_name=str(args.env_name),
            cuda_home=str(args.cuda_home),
            model=str(args.model),
            port=int(args.port),
            host=str(args.server_bind_host),
            tp_size=int(args.tp_size),
            extra_args=str(args.extra_args),
            log_path=log_path,
        )
        actors.append(actor)
        urls.append(f"http://{node_ip}:{args.port}")

    starts = ray.get([actor.start.remote() for actor in actors])
    for node_ip, info in zip(nodes, starts, strict=True):
        print(
            f"[vllm-shard] start node={node_ip} host={info['hostname']} "
            f"pid={info['pid']} log={info['log_path']}",
            flush=True,
        )
    return actors, urls


def _wait_servers(actors: list[Any], urls: list[str], timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    ready = [False] * len(urls)
    while not all(ready):
        for index, url in enumerate(urls):
            if ready[index]:
                continue
            if _http_ready(url):
                ready[index] = True
                print(f"[vllm-shard] ready {url}", flush=True)
                continue
            if ray.get(actors[index].poll.remote()) is not None:
                raise RuntimeError(f"vLLM shard exited before ready: {url}")
        if time.monotonic() >= deadline:
            missing = [url for url, is_ready in zip(urls, ready, strict=True) if not is_ready]
            raise TimeoutError(f"vLLM shards not ready after {timeout_s}s: {missing}")
        time.sleep(2)


def _score_shard(
    *,
    url: str,
    served_model: str,
    tokenizer: Any,
    pairs: list[tuple[int, dict[str, Any]]],
    prompt_logprobs: int,
    timeout: float,
    concurrency: int,
) -> list[tuple[int, dict[str, Any]]]:
    def score_one(index: int, raw_sample: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        sample = MiniSGLSample(
            prompt=raw_sample["prompt"],
            generated_text=raw_sample["generated_text"],
            generated_token_ids=[int(token_id) for token_id in raw_sample["generated_token_ids"]],
        )
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=True)
        scored_choice = score_with_vllm(
            base_url=url,
            served_model=served_model,
            prompt_token_ids=prompt_ids + sample.generated_token_ids,
            prompt_logprobs=prompt_logprobs,
            timeout=timeout,
        )
        return index, compare_prompt(tokenizer, sample, scored_choice, prompt_logprobs)

    results: list[tuple[int, dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_to_index = {
            executor.submit(score_one, index, raw_sample): index
            for index, raw_sample in pairs
        }
        for done, future in enumerate(concurrent.futures.as_completed(future_to_index), start=1):
            index = future_to_index[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"vLLM shard {url} failed sample index {index}") from exc
            if done % 25 == 0 or done == len(pairs):
                print(f"[vllm-shard] scored {url} {done}/{len(pairs)}", flush=True)
    return results


def _score_all(args: argparse.Namespace, urls: list[str]) -> dict[str, Any]:
    bundle = sample_bundle_from_json(args.sample_json)
    tokenizer_model = args.model or bundle.get("tokenizer_model")
    if not tokenizer_model:
        raise RuntimeError("Tokenizer model must be supplied via --model or stored in sample-json.")
    tokenizer = load_tokenizer(tokenizer_model)
    samples = list(bundle["samples"])
    shards: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in urls]
    for index, sample in enumerate(samples):
        shards[index % len(urls)].append((index, sample))

    served_models = [get_first_served_model(url, args.timeout) for url in urls]
    merged: list[dict[str, Any] | None] = [None] * len(samples)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as executor:
        future_to_url = {
            executor.submit(
                _score_shard,
                url=url,
                served_model=served_model,
                tokenizer=tokenizer,
                pairs=pairs,
                prompt_logprobs=int(args.prompt_logprobs),
                timeout=float(args.timeout),
                concurrency=int(args.score_concurrency),
            ): url
            for url, served_model, pairs in zip(urls, served_models, shards, strict=True)
            if pairs
        }
        for future in concurrent.futures.as_completed(future_to_url):
            for index, result in future.result():
                merged[index] = result

    finalized = [result for result in merged if result is not None]
    summary = summarize(finalized)
    return {
        "tokenizer_model": tokenizer_model,
        "prompt_logprobs": int(args.prompt_logprobs),
        "shard_urls": urls,
        "summary": summary,
        "results": finalized,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-json", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--node-list", required=True, help="Comma-separated Ray node IPs.")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--score-concurrency", type=int, default=16)
    parser.add_argument("--prompt-logprobs", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument("--gpus-per-server", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--server-bind-host", default="0.0.0.0")
    def default_conda_sh() -> Path:
        if os.environ.get("CONDA_SH"):
            return Path(os.environ["CONDA_SH"])
        if os.environ.get("CONDA_EXE"):
            return Path(os.environ["CONDA_EXE"]).resolve().parents[1] / "etc/profile.d/conda.sh"
        for root in ("miniforge3", "miniconda3"):
            candidate = Path.home() / root / "etc/profile.d/conda.sh"
            if candidate.exists():
                return candidate
        return Path.home() / "miniconda3/etc/profile.d/conda.sh"

    def default_cuda_home() -> Path:
        if os.environ.get("CUDA_13_HOME"):
            return Path(os.environ["CUDA_13_HOME"])
        if os.environ.get("CUDA_HOME"):
            return Path(os.environ["CUDA_HOME"])
        nvcc = shutil.which("nvcc")
        if nvcc:
            return Path(nvcc).resolve().parents[1]
        return Path.home() / "cuda/cuda-13.0"

    default_conda_sh = default_conda_sh()
    default_cuda_home = default_cuda_home()
    default_env_name = os.environ.get("ENV_NAME", os.environ.get("CONDA_DEFAULT_ENV", "minisgl-cuda130"))
    parser.add_argument("--calib-dir", type=Path, default=SCRIPT_DIR.parents[2])
    parser.add_argument("--conda-sh", type=Path, default=default_conda_sh)
    parser.add_argument("--env-name", default=default_env_name)
    parser.add_argument("--cuda-home", type=Path, default=default_cuda_home)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.sample_json = args.sample_json.resolve()
    args.output_json = args.output_json.resolve()
    args.log_dir = args.log_dir.resolve()
    args.calib_dir = args.calib_dir.resolve()
    args.conda_sh = args.conda_sh.resolve()
    args.cuda_home = args.cuda_home.resolve()
    nodes = _csv(args.node_list)
    if not nodes:
        raise RuntimeError("--node-list is empty")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    _check_vllm_deepep_available(str(args.extra_args))

    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True)
    actors: list[Any] = []
    try:
        actors, urls = _start_servers(args, nodes)
        _wait_servers(actors, urls, int(args.ready_timeout))
        report = _score_all(args, urls)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print_report(report["summary"], report["results"])
    finally:
        if actors:
            ray.get([actor.stop.remote() for actor in actors], timeout=120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
