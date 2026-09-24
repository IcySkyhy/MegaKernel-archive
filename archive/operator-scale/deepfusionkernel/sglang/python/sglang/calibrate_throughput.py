"""
Use "ENABLE_MY_TRITON=1" for custom triton kernels

For a specific GPU type and model config (mat sizes, TP size), profile kernels' performance curves across batch sizes,
added with latency model of inter-device communication (which depends on comm times and transferred data size),
to determine whether to invoke custom Triton fused kernel(s).

We assume greedy scheduling: if a kernel is better in kernel calibration, it would be the optimal choice in whole-model running
"""

import argparse
import dataclasses
import datetime
import functools
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import re
import string
import time
from typing import Iterable, List, Tuple, Dict

import numpy as np
import scipy
import scipy.optimize
import torch
import torch.distributed as dist
import triton

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.srt.layers.fused_gmlp import *


KERNELS = {
    "m_tkn": gatedmlp_m_tkn_kernel,
    "m_tkn_vec": gatedmlp_m_tkn_vec_kernel,
    "mt_kn": gatedmlp_mt_kn_kernel,
    "mt_kn_vec": gatedmlp_mt_kn_vec_kernel,
    "mt_kn_keepA2": gatedmlp_mt_kn_keepA2_kernel,
    "mt_kn_keepA2_vec": gatedmlp_mt_kn_keepA2_vec_kernel,
    "t_mkn": gatedmlp_t_mkn_kernel,
    "t_mkn_vec": gatedmlp_t_mkn_vec_kernel,
    "sep_knls": (gatedmlp_separated_a2_kernel, gatedmlp_separated_y_kernel),
    "sep_knls_vec": (gatedmlp_separated_a2_vec_kernel, gatedmlp_separated_y_vec_kernel),
}


def flatten(S):
    """
    Flatten a recursive list or tuple
    """
    if isinstance(S, (list, tuple)) and len(S) == 0:
        return list(S)
    if isinstance(S[0], (list, tuple)):
        return flatten(S[0]) + flatten(S[1:])
    return list(S[:1]) + flatten(S[1:])


KERNEL_LIST = flatten(list(KERNELS.values()))


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # disable_my_triton: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_filename_prefix: str = "profile"
    best_config_pkl_path: str | Tuple[str] = ("",)
    # This is only used for throughput calibration
    calibration_batch_size: Tuple[int] = (0,)
    cali_res_load_path: Tuple[str] = ("",)
    cali_save_path_prefix: str = "calibration-result"
    interp_method: str = "linear"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument(
            "--profile", action="store_true", help="Use Torch Profiler."
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        parser.add_argument(
            "--best-config-pkl-path",
            type=str,
            nargs="+",
            default=BenchArgs.best_config_pkl_path,
            help="Path(s) to the Triton Autotuner cache pickle file(s)",
        )
        parser.add_argument(
            "--calibration-batch-size",
            type=int,
            nargs="+",
            default=BenchArgs.calibration_batch_size,
            help="Batch sizes for calibration of SGLang and custom Triton kernel performance",
        )
        parser.add_argument(
            "--cali-res-load-path",
            type=str,
            nargs="+",
            default=BenchArgs.cali_res_load_path,
            help="Path(s) to the calibration record JSON file(s)",
        )
        parser.add_argument(
            "--cali-save-path-prefix",
            type=str,
            default=BenchArgs.cali_save_path_prefix,
            help="Path for saving calibration results",
        )
        parser.add_argument(
            "--interp-method",
            type=str,
            default=BenchArgs.interp_method,
            help=f"Interpolation method for calibration, available options are {INTERPOLATION_STRATEGIES}",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def apply_autotuner_cache_records(
    best_config_pkl_path, autotuners: Iterable[triton.runtime.Autotuner]
):
    with open(best_config_pkl_path, "rb") as f:
        records: Dict[str, dict] = pickle.load(f)

    logging.info(f"Loading caches from {best_config_pkl_path}")
    for fn_name in records:
        if fn_name == "args":
            continue
        cache = records[fn_name]["cache"]
        keys = records[fn_name]["keys"]
        if cache == {}:
            logging.warning(
                f"Empty cache for {fn_name}, possibly because when running autotuning there is only one config."
            )
            continue
        for autotuner in autotuners:
            assert isinstance(autotuner, triton.runtime.Autotuner)
            if autotuner.base_fn.__name__ != fn_name:
                continue
            if autotuner.keys != keys:
                logging.warning(
                    f"Not loading autotuner of {fn_name} due to unmatched keys"
                )
                continue
            autotuner.cache.update(cache)
            logging.info(f"Loaded autotuner cache for {fn_name}")


def load_model(server_args, port_args, tp_rank, enable_my_triton=None):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    model_config = ModelConfig.from_server_args(
        server_args, enable_my_triton=enable_my_triton
    )
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer):
    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
    ]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
    return reqs


def prepare_synthetic_inputs_for_latency_test(batch_size, input_len):
    input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return reqs


@torch.no_grad
def extend(reqs, model_runner, disable_cuda_graph=None):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    _maybe_prepare_dp_attn_batch(batch, model_runner, disable_cuda_graph)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner, disable_cuda_graph=None):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_dp_attn_batch(batch, model_runner, disable_cuda_graph)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_dp_attn_batch(
    batch: ScheduleBatch, model_runner, disable_cuda_graph=None
):
    if model_runner.server_args.enable_dp_attention:
        Scheduler.prepare_dp_attn_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_cpu_group=model_runner.tp_group.cpu_group,
            get_idle_batch=None,
            disable_cuda_graph=(
                disable_cuda_graph
                if disable_cuda_graph is not None
                else model_runner.server_args.disable_cuda_graph
            ),
            spec_algorithm=SpeculativeAlgorithm.NONE,
            speculative_num_draft_tokens=None,
        )


def correctness_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs
    input_ids, reqs = prepare_inputs_for_correctness_test(bench_args, tokenizer)
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # Prepare extend inputs
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def synchronize(device):
    torch.get_device_module(device).synchronize()


def update_module_mlp_forward(module: torch.nn.Module, kernel_ver: str):
    # print(f"Setting decode forward to {kernel_ver}")
    for n, m in module.named_modules():
        if hasattr(m, "update_forward_method"):
            m.update_forward_method(kernel_ver)


def clean_names_in_json(input_filename, output_filename):
    """
    Cleans the "name" fields in a JSON file by replacing non-ASCII characters with 'x'
    and removing internal quotation marks.

    Example of problematic input:
        {
            "name": "@"�sP(0): flat_tensor"
        }
    """
    with open(input_filename, "r", encoding="utf-8", errors="replace") as file:
        content = file.read()

        # # Decode Unicode escape sequences
        # content = content.encode().decode('unicode_escape')

        # Regex to find "name": "<value>"
        def replace_non_ascii_and_quotes(match):
            name = match.group(1)
            visible_printable = "".join(
                c for c in string.printable if c not in "\t\n\r\x0b\x0c}{"
            )
            cleaned_name = "".join(c if c in visible_printable else "x" for c in name)
            cleaned_name = cleaned_name.replace('"', "y")  # Replace internal quotes
            return f'"name": "{cleaned_name}"'

        # Apply regex to clean names
        cleaned_content = re.sub(
            r'"name": "([\s\S]*?)"(?=, |\}|\s*\})',
            replace_non_ascii_and_quotes,
            content,
            flags=re.DOTALL,
        )

    # Write the cleaned JSON data to a new file
    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write(cleaned_content)


def calibrate_kernel_once(
    model_runner,
    rank_print,
    reqs,
    batch_size,
    device,
    now,
    kernel_ver,
    profile=True,
    tp_rank=0,
):
    # Assuming all layers use same MLP config, so the result goes to `default`
    # Return latencies of kernels and comm separately
    # TODO calibrate each layer separately

    max_batch_size = model_runner.max_total_num_tokens
    if batch_size > max_batch_size:
        rank_print(f"skipping bs={batch_size} calibration due to max batch size limit")
        return

    # Clear the pools.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    measurement_results = {
        "run_name": "calibration",
        "batch_size": batch_size,
    }

    tot_latency = 0

    profiler = None
    if profile:
        if tp_rank == 0:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                # with_stack=True,
            )
            profiler.start()

    # Prefill
    synchronize(device)
    for n, m in model_runner.model.named_modules():
        if hasattr(m, "prefill_mode"):
            m.prefill_mode()
    tic = time.time()
    next_token_ids, _, batch = extend(reqs, model_runner, disable_cuda_graph=True)
    synchronize(device)
    prefill_latency = time.time() - tic
    tot_latency += prefill_latency
    throughput = 1 * batch_size / prefill_latency

    # Decode
    decode_latencies = []
    for n, m in model_runner.model.named_modules():
        if hasattr(m, "decode_mode"):
            m.decode_mode()
    for i in range(5 - 1):
        synchronize(device)
        tic = time.time()
        next_token_ids, _ = decode(
            next_token_ids, batch, model_runner, disable_cuda_graph=True
        )
        synchronize(device)
        latency = time.time() - tic
        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)

    # Save torch profiling results
    if profile:
        if tp_rank == 0:
            profiler.stop()
        profile_filename = f"data/calibration/calibration_{now}_{kernel_ver}_batch{batch_size}.trace.json"
        profile_filename_t = f"data/calibration/calibration-tmp_{now}_{kernel_ver}_batch{batch_size}.trace.json"
        if tp_rank == 0:
            parent_dir = os.path.dirname(os.path.abspath(profile_filename))
            os.makedirs(parent_dir, exist_ok=True)
            profiler.export_chrome_trace(profile_filename_t)
            # Clean up JSON
            clean_names_in_json(profile_filename_t, profile_filename)
            os.remove(profile_filename_t)
            rank_print(
                f"Calibration: torch profiler chrome trace saved to {profile_filename}"
            )
        else:
            t = time.time()
            while not os.path.exists(profile_filename):
                if time.time() - t > 30:
                    logging.error("Timeout waiting torch profiler export chrome trace")
                    break
            time.sleep(1)

        # Extract MLP kernels from exported chrome trace
        with open(profile_filename, "r", encoding="utf-8", errors="replace") as f:
            profile_data = json.load(f)
        profile_data = profile_data["traceEvents"]
        # Find the last Calibrate_MLP
        last_start_ts = 0
        last_end_ts = 0
        for event in profile_data:
            if (
                event["name"] == "Calibrate_MLP"
                and event["cat"] == "gpu_user_annotation"
            ):
                start_ts = event["ts"]
                dur = event["dur"]
                end_ts = start_ts + dur
                if start_ts > last_start_ts:
                    last_start_ts = start_ts
                    last_end_ts = end_ts
        if last_start_ts == last_end_ts == 0:
            raise RuntimeError("Failed to find `Calibrate_MLP` in profiling trace")
        # Find the add-and-norm after MLP
        next_add_norm_ts = 0
        for event in profile_data:
            if "FusedAddRMSNormKernel" in event["name"] and event["cat"] == "kernel":
                start_ts = event["ts"]
                if start_ts >= last_end_ts:
                    next_add_norm_ts = start_ts
        if next_add_norm_ts == 0:
            raise RuntimeError(
                "Failed to find `FusedAddRMSNormKernel` after the last `Calibrate_MLP` in profiling trace"
            )
        # Get kernels invoked within the last Calibrate_MLP
        compute_dur = 0.0
        comm_dur = 0.0
        out_comm_dur = 0.0
        for event in profile_data:
            if (
                "cat" in event
                and event["cat"] == "kernel"
                and last_start_ts <= event["ts"] < last_end_ts
            ):
                if "reduce" in event["name"].lower():
                    comm_dur += event["dur"]
                elif "memset" in event["name"].lower():
                    pass
                else:
                    compute_dur += event["dur"]
            elif (
                "cat" in event
                and event["cat"] == "kernel"
                and "reduce" in event["name"].lower()
                and next_add_norm_ts >= event["ts"] >= last_end_ts
            ):
                out_comm_dur = event["dur"]

        return {"roofline": compute_dur, "comm": comm_dur + out_comm_dur}

    return


def calibrate_kernel(
    model_runner,
    server_args,
    port_args,
    bench_args,
    tp_rank,
    now,
):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Run the sweep
    batch_sizes = sorted(set(bench_args.calibration_batch_size))
    bs = batch_sizes[0]
    sglang_res = []
    triton_res = []
    seq_len = 1

    model_runner.calibration_disable_cuda_graph = True
    for n, m in model_runner.model.named_modules():
        if hasattr(m, "enable_calibration_profile"):
            m.enable_calibration_profile = True

    rank_print("Calibration warmup ...")
    reqs = prepare_synthetic_inputs_for_latency_test(bs, seq_len)
    update_module_mlp_forward(model_runner.model, "sglang")
    sglang_ret = calibrate_kernel_once(
        model_runner,
        rank_print,
        reqs,
        bs,
        server_args.device,
        now,
        kernel_ver="sglang",
        profile=False,
        tp_rank=tp_rank,
    )
    update_module_mlp_forward(model_runner.model, "triton")
    triton_ret = calibrate_kernel_once(
        model_runner,
        rank_print,
        reqs,
        bs,
        server_args.device,
        now,
        kernel_ver="triton",
        profile=False,
        tp_rank=tp_rank,
    )

    rank_print("Calibrate throughputs ...")
    for bs in batch_sizes:
        reqs = prepare_synthetic_inputs_for_latency_test(bs, seq_len)
        update_module_mlp_forward(model_runner.model, "sglang")
        sglang_ret = calibrate_kernel_once(
            model_runner,
            rank_print,
            reqs,
            bs,
            server_args.device,
            now,
            kernel_ver="sglang",
            tp_rank=tp_rank,
        )
        if sglang_ret is not None:
            sglang_res.append(sglang_ret)
        update_module_mlp_forward(model_runner.model, "triton")
        triton_ret = calibrate_kernel_once(
            model_runner,
            rank_print,
            reqs,
            bs,
            server_args.device,
            now,
            kernel_ver="triton",
            tp_rank=tp_rank,
        )
        if triton_ret is not None:
            triton_res.append(triton_ret)

    model_runner.calibration_disable_cuda_graph = False
    for n, m in model_runner.model.named_modules():
        if hasattr(m, "enable_calibration_profile"):
            m.enable_calibration_profile = False

    # sglang_res = {b: r for b, r in zip(batch_sizes, sglang_res)}
    # triton_res = {b: r for b, r in zip(batch_sizes, triton_res)}
    # return {"sglang": sglang_res, "triton": triton_res}

    hidden, interm = (
        model_runner.model_config.hf_config.hidden_size,
        model_runner.model_config.hf_config.intermediate_size // server_args.tp_size,
    )
    gpu_type = torch.cuda.get_device_name()
    sglang_data = {
        "roofline": {
            "latency": [r["roofline"] for r in sglang_res],
            "batch_size": batch_sizes,
            "weight_size": [(hidden, interm) for _ in range(len(batch_sizes))],
            "device_cnt": [server_args.tp_size for _ in range(len(batch_sizes))],
            "node_cnt": [server_args.nnodes for _ in range(len(batch_sizes))],
            "kernel_ver": "sglang",
            "gpu_type": gpu_type,
        },
        "comm": {
            "latency": [r["comm"] for r in sglang_res],
            "batch_size": batch_sizes,
            "weight_size": [(hidden, interm) for _ in range(len(batch_sizes))],
            "device_cnt": [server_args.tp_size for _ in range(len(batch_sizes))],
            "node_cnt": [server_args.nnodes for _ in range(len(batch_sizes))],
            "kernel_ver": "sglang",
            "gpu_type": gpu_type,
        },
    }
    triton_data = {
        "roofline": {
            "latency": [r["roofline"] for r in triton_res],
            "batch_size": batch_sizes,
            "weight_size": [(hidden, interm) for _ in range(len(batch_sizes))],
            "device_cnt": [server_args.tp_size for _ in range(len(batch_sizes))],
            "node_cnt": [server_args.nnodes for _ in range(len(batch_sizes))],
            "kernel_ver": "triton",
            "gpu_type": gpu_type,
        },
        "comm": {
            "latency": [r["comm"] for r in triton_res],
            "batch_size": batch_sizes,
            "weight_size": [(hidden, interm) for _ in range(len(batch_sizes))],
            "device_cnt": [server_args.tp_size for _ in range(len(batch_sizes))],
            "node_cnt": [server_args.nnodes for _ in range(len(batch_sizes))],
            "kernel_ver": "triton",
            "gpu_type": gpu_type,
        },
    }
    return {gpu_type: {"default": {"sglang": sglang_data, "triton": triton_data}}}


def fit_roofline(
    latency: List[float],
    batch_size: List[int],
    weight_size: List[Tuple[int]],
    kernel_ver: str,
    gpu_type: str,
):
    assert kernel_ver in ["sglang", "triton"]
    hidden_size = [s[0] for s in weight_size]
    interm_size = [s[1] for s in weight_size]

    def single_gpu_roofline_fn(x, bw, speed, c):
        bs, hidden, interm = x
        flop = 6 * bs * hidden * interm + 6 * bs * interm
        byte = (
            3 * bs * hidden + 8 * bs * interm + 3 * hidden * interm
            if kernel_ver == "sglang"
            else 2 * bs * hidden + 2 * bs * interm + 3 * hidden * interm
        )
        return 2 * byte / bw + flop / speed + c

    if "A100" in gpu_type:
        if "40GB" in gpu_type:
            init_guess = (1555 * 2**30, 312 * 10**12, 0)
        elif "PCIe" in gpu_type:
            init_guess = (1935 * 2**30, 312 * 10**12, 0)
        else:
            init_guess = (2039 * 2**30, 312 * 10**12, 0)
    elif "A6000" in gpu_type:
        init_guess = (768 * 2**30, 150 * 10**12, 0)
    else:
        init_guess = (2**40, 300 * 10**12, 0)
    popt, pcov = scipy.optimize.curve_fit(
        single_gpu_roofline_fn,
        (batch_size, hidden_size, interm_size),
        latency,
        init_guess,
        bounds=([1, 1, 0], [np.inf, np.inf, np.inf]),
    )
    bw_opt, speed_opt, const = popt

    def get_roofline(pred_data):
        for bs, ws, t in zip(batch_size, weight_size, latency):
            if bs == pred_data[0] and ws[0] == pred_data[1] and ws[1] == pred_data[2]:
                return t
        return single_gpu_roofline_fn(pred_data, bw=bw_opt, speed=speed_opt, c=const)

    return get_roofline


def fit_comm(
    latency: List[float],
    batch_size: List[int],
    weight_size: List[Tuple[int]],
    device_cnt: List[int],
    node_cnt: List[int],
    kernel_ver: str,
    gpu_type: str,
):
    assert kernel_ver in ["sglang", "triton"]
    hidden_size = [s[0] for s in weight_size]
    interm_size = [s[1] for s in weight_size]

    def comm_fn(x, d0, d1, d2, n0, n1, n2):
        bs, hidden, interm, device_cnt, node_cnt = x
        byte = (
            3 * bs * hidden + 8 * bs * interm + 3 * hidden * interm
            if kernel_ver == "sglang"
            else 2 * bs * hidden + 2 * bs * interm + 3 * hidden * interm
        )
        inter_device = d0 + d1 * device_cnt + d2 * device_cnt**2
        inter_node = n0 + n1 * node_cnt + n2 * node_cnt**2
        return byte * inter_device + byte * inter_node

    if "A100" in gpu_type:
        device_bw = 600 * 2**30
        node_bw = 64 * 2**30
    elif "A6000" in gpu_type:
        device_bw = 112.5 * 2 * 2**30
        node_bw = 64 * 2**30
    else:
        device_bw = 400 * 2**30
        node_bw = 64 * 2**30
    init_guess = (0, 1 / device_bw, 1 / device_bw, 0, 1 / node_bw, 1 / node_bw)
    popt, pcov = scipy.optimize.curve_fit(
        comm_fn,
        (batch_size, hidden_size, interm_size, device_cnt, node_cnt),
        latency,
        init_guess,
        bounds=(0, np.inf),
    )
    d0, d1, d2, n0, n1, n2 = popt

    def get_comm(pred_data):
        for bs, ws, dc, nc, t in zip(
            batch_size, weight_size, device_cnt, node_cnt, latency
        ):
            if (
                bs == pred_data[0]
                and ws[0] == pred_data[1]
                and ws[1] == pred_data[2]
                and dc == pred_data[3]
                and nc == pred_data[4]
            ):
                return t
        return comm_fn(pred_data, d0=d0, d1=d1, d2=d2, n0=n0, n1=n1, n2=n2)

    return get_comm


def latency_model(data: Dict[str, Dict]):
    """
    Suppose latency depends on batch size, weight size, device cnt, node cnt, interconnection property.
    Assume FP16 data and FP32/TF32 compute, TP only within a node

    latency = Constant_overhead + total_time(bs, weight_size) + comm_overhead(bs, weight_size, device_cnt, node_cnt)
      total_time(bs, weight_size) = 2 * byte / BW + flop / Speed
        flop = 6 * bs * hidden * interm + 6 * bs * interm
        byte = sglang_kernel ? 3 * bs * hidden + 8 * bs * interm + 3 * hidden * interm : 2 * bs * hidden + 2 * bs * interm + 3 * hidden * interm
      comm_overhead = byte * inter_device + byte * inter_node
        inter_device = Quad1(device_cnt)
        inter_node = Quad2(node_cnt)
    GPU type-specific params: Constant_overhead, BW, Speed, Quad1, Quad2

    `data` format: {
      "roofline": {
        "latency": List[second],
        "batch_size": List[int],
        "weight_size": List[(int, int)],
        "device_cnt": List[int],
        "node_cnt": List[int],
        "kernel_ver": str,
        "gpu_type": str,
      },
      "comm": ...,
    }
    """
    roofline_data = data["roofline"]
    comm_data = data["comm"]

    roofline_fn = fit_roofline(
        roofline_data["latency"],
        roofline_data["batch_size"],
        roofline_data["weight_size"],
        roofline_data["kernel_ver"],
        roofline_data["gpu_type"],
    )
    comm_fn = fit_comm(
        comm_data["latency"],
        comm_data["batch_size"],
        comm_data["weight_size"],
        comm_data["device_cnt"],
        comm_data["node_cnt"],
        comm_data["kernel_ver"],
        comm_data["gpu_type"],
    )
    return lambda pred: (
        roofline_fn(
            [pred["batch_size"], pred["weight_size"][0], pred["weight_size"][1]]
        )
        + comm_fn(
            [
                pred["batch_size"],
                pred["weight_size"][0],
                pred["weight_size"][1],
                pred["device_cnt"],
                pred["node_cnt"],
            ]
        )
    )


INTERPOLATION_STRATEGIES = [
    "linear",
    "reciprocal",  # y = a + b / (x + c)  with a = compute bound and y=0 when x=0
    "log-linear",  # i.e. power, since y = c * x^a  <=>  log y = log c + a * log x
]


def _interp_linear(xs, xp, yp):
    idx = np.argsort(xp)
    xp = np.asarray(xp)[idx]
    yp = np.asarray(xp)[yp]
    ys = np.where(
        xs < min(xp),
        np.interp(xs, (0, xp[idx[0]]), (0, yp[idx[0]])),
        np.where(xs > max(xp), np.full_like(xs, yp[idx[-1]]), np.interp(xs, xp, yp)),
    )
    return ys


def _interp_log_linear(xs, xp, yp):
    xs, xp, yp = np.log2((xs, xp, yp))
    ys = _interp_linear(xs, xp, yp)
    return 2**ys


def _interp_reciprocal(xs, xp, yp):
    def _rcp_fn(x, a, b, c):
        return a + b / (x + c)

    popt, pcov = scipy.optimize.curve_fit(
        _rcp_fn, xp, yp, [1, -1, 1], bounds=(0, [np.inf, -np.inf, np.inf])
    )
    a, b, c = popt
    ys = _rcp_fn(xs, a, b, c)
    return ys


def _choose_kernel(
    batch_size: int | List[int],
    calibration_result: Dict[str, Dict[int, float]],
    interpolation_strategy="linear",
) -> str | List[str]:
    # Should have one calibration_results per MLP block
    assert (
        interpolation_strategy in INTERPOLATION_STRATEGIES
    ), f"Unsupported throughput interpolation strategy, must be one of {INTERPOLATION_STRATEGIES}"
    match interpolation_strategy:
        case "linear":
            interpolation_fn = _interp_linear
        case "reciprocal":
            interpolation_fn = _interp_reciprocal
        case "log-linear":
            interpolation_fn = _interp_log_linear
        case _:
            raise ValueError("Invalid interpolation strategy")

    def get_optimal_kernel(bs: int) -> str:
        pred: Dict[str, float] = {}
        for name in calibration_result:
            datapoints = calibration_result[name]
            if bs in datapoints:
                pred[name] = datapoints[bs]
            else:
                pred[name] = interpolation_fn(
                    bs, list(datapoints.keys()), list(datapoints.values())
                )
        opt_name = max(pred, key=pred.get)
        return opt_name

    if isinstance(batch_size, int):
        return get_optimal_kernel(batch_size)
    else:
        optimal = []
        for bs in batch_size:
            optimal.append(get_optimal_kernel(bs))
        return optimal


def recursive_update_dict(a, b):
    assert type(a) == type(b)
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    if not isinstance(a, dict) and not isinstance(b, dict):
        return b
    for key in b:
        if key in a:
            a[key] = recursive_update_dict(a[key], b[key])
        else:
            a[key] = b[key]
    return a


def apply_calibration(
    model: torch.nn.Module,
    calibration_data: (
        Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List | str]]]]]
        | List[Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List | str]]]]]]
    ),
    pred_data: Dict[str, int | Tuple[int]],
    current_gpu_type: str,
    pred_fn_cache=None,
):
    """
    Apply calibration results given calibrated datapoints and requested batch size

    calibration_data format:
    {
        gpu_type: {
            "default"|module_name: {
                kernel_ver : {
                    "roofline": {
                        "latency": List[second],
                        "batch_size": List[int],
                        "weight_size": List[(int, int)],
                        "device_cnt": List[int],
                        "node_cnt": List[int],
                        "kernel_ver": str,
                        "gpu_type": str,
                    },
                    "comm": ...,
                }
            }
        }
    }
    pred_data format:
    {
        "batch_size": int,
        "weight_size": (int, int),
        "device_cnt": int,
        "node_cnt": int,
    }
    """
    if isinstance(calibration_data, list):
        calibrate_data = {}
        for r in calibration_data:
            calibrate_data = recursive_update_dict(calibrate_data, r)
    else:
        calibrate_data = calibration_data

    calibrate_data = calibrate_data[current_gpu_type]
    if pred_fn_cache is None:
        pred_fn_cache = {}
        default_data = calibrate_data.pop("default")
        default_sglang_latency_fn = latency_model(default_data["sglang"])
        default_triton_latency_fn = latency_model(default_data["triton"])
        pred_fn_cache["default"] = {
            "sglang": default_sglang_latency_fn,
            "triton": default_triton_latency_fn,
        }
    else:
        logging.info("Found latency model cache, so skipping latency model fitting")
        default_sglang_latency_fn = pred_fn_cache["default"]["sglang"]
        default_triton_latency_fn = pred_fn_cache["default"]["triton"]
    for name, module in model.named_modules():
        if hasattr(module, "enable_calibration") and module.enable_calibration:
            if pred_fn_cache is not None:
                if name in pred_fn_cache:
                    sglang_latency_fn = pred_fn_cache[name]["sglang"]
                    triton_latency_fn = pred_fn_cache[name]["triton"]
                else:
                    sglang_latency_fn = default_sglang_latency_fn
                    triton_latency_fn = default_triton_latency_fn
            else:
                if name in calibrate_data:
                    sglang_latency_fn = latency_model(calibrate_data[name]["sglang"])
                    triton_latency_fn = latency_model(calibrate_data[name]["triton"])
                    pred_fn_cache[name] = {
                        "sglang": sglang_latency_fn,
                        "triton": triton_latency_fn,
                    }
                else:
                    sglang_latency_fn = default_sglang_latency_fn
                    triton_latency_fn = default_triton_latency_fn

            sglang_latency = sglang_latency_fn(pred_data)
            triton_latency = triton_latency_fn(pred_data)
            if triton_latency < sglang_latency:
                module.update_forward_method("triton")
            else:
                module.update_forward_method("sglang")

    return pred_fn_cache


def load_calibration_result(filename):
    with open(filename, "r") as f:
        res = json.load(f)
    return res


def save_calibration_result(filename, res):
    if os.path.exists(filename):
        raise FileExistsError("Calibration result saving filename already exists")
    with open(filename, "w+") as f:
        res = json.dump(res, f)


def calibration_pipeline(
    model_runner: ModelRunner,
    server_args: ServerArgs,
    port_args,
    bench_args: BenchArgs,
    tp_rank,
):
    # assert isinstance(model_runner.model, (MyLlamaForCausalLM, MyQwen2ForCausalLM)), f"Unsupported model class `{type(model_runner.model)}` for calibration"
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load calibration records
    calibration_data = []
    if bench_args.cali_res_load_path != ("",):
        for p in bench_args.cali_res_load_path:
            data = load_calibration_result(p)
            calibration_data.append(data)

    # Run calibration for every MLP size; save calibration results
    if bench_args.calibration_batch_size != (0,):
        logging.info(
            f"Running calibration on batch sizes {bench_args.calibration_batch_size}"
        )
        new_cali_data = calibrate_kernel(
            model_runner, server_args, port_args, bench_args, tp_rank, now
        )
        if tp_rank == 0:
            save_path = f"{bench_args.cali_save_path_prefix}_{now}.json"
            save_calibration_result(save_path, new_cali_data)
            rank_print(f"Saved calibration result to {save_path}")
        calibration_data.append(new_cali_data)
    else:
        logging.info(f"Skipping calibration")

    # # Fit latency model and apply result to LLM
    # current_gpu_type = torch.cuda.get_device_name()
    # pred_fn_cache = None
    # for bs in bench_args.batch_size:
    #     pred_data = {
    #         "batch_size": bs,
    #         "weight_size": (model_runner.model_config.hf_config.hidden_size, model_runner.model_config.hf_config.intermediate_size),
    #         "device_cnt": server_args.tp_size,
    #         "node_cnt": server_args.nnodes,
    #     }
    #     pred_fn_cache = apply_calibration(
    #         model_runner.model,
    #         calibration_data,
    #         pred_data,
    #         current_gpu_type,
    #         pred_fn_cache=pred_fn_cache,
    #     )
    return calibration_data


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    device,
    log_decode_step,
    profile,
    profile_filename_prefix,
):
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    # Clear the pools.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    tot_latency = 0

    profiler = None
    if profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        )
        profiler.start()

    # Prefill
    synchronize(device)
    for n, m in model_runner.model.named_modules():
        if hasattr(m, "prefill_mode"):
            m.prefill_mode()
    tic = time.time()
    next_token_ids, _, batch = extend(reqs, model_runner)
    synchronize(device)
    prefill_latency = time.time() - tic
    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    # Decode
    decode_latencies = []
    for n, m in model_runner.model.named_modules():
        if hasattr(m, "decode_mode"):
            m.decode_mode()
    for i in range(output_len - 1):
        synchronize(device)
        tic = time.time()
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        synchronize(device)
        latency = time.time() - tic
        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )

    if profile:
        profiler.stop()
        profile_filename = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}.trace.json.gz"
        parent_dir = os.path.dirname(os.path.abspath(profile_filename))
        os.makedirs(parent_dir, exist_ok=True)
        profiler.export_chrome_trace(profile_filename)
        rank_print(f"torch profiler chrome trace saved to {profile_filename}")

    # Record decode timing from 2nd output
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load pickled best configs
    if bench_args.best_config_pkl_path != ("",):
        if isinstance(bench_args.best_config_pkl_path, str):
            bench_args.best_config_pkl_path = [bench_args.best_config_pkl_path]
        for path in bench_args.best_config_pkl_path:
            # best_config_pkl_path = f"{path}.pkl"
            autotuners = [
                k for k in KERNEL_LIST if isinstance(k, triton.runtime.Autotuner)
            ]
            apply_autotuner_cache_records(
                best_config_pkl_path=path,
                autotuners=autotuners,
            )

    # Load the model
    model_runner, tokenizer = load_model(
        server_args, port_args, tp_rank, enable_my_triton=True
    )

    # Calibration
    calibration_data = calibration_pipeline(
        model_runner, server_args, port_args, bench_args, tp_rank
    )

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0]
    )

    # Run the sweep
    result_list = []
    pred_fn_cache = None
    for bs in bench_args.batch_size:
        # Fit latency model and apply result to LLM
        current_gpu_type = torch.cuda.get_device_name()
        pred_data = {
            "batch_size": bs,
            "weight_size": (
                model_runner.model_config.hf_config.hidden_size,
                model_runner.model_config.hf_config.intermediate_size
                // server_args.tp_size,
            ),
            "device_cnt": server_args.tp_size,
            "node_cnt": server_args.nnodes,
        }
        pred_fn_cache = apply_calibration(
            model_runner.model,
            calibration_data,
            pred_data,
            current_gpu_type,
            pred_fn_cache=pred_fn_cache,
        )
        # Re-capture CUDA graph
        model_runner.init_cuda_graphs()
        # Warm up
        rank_print("Warmup ...")
        latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            min(
                32, bench_args.output_len[0]
            ),  # shorter decoding to speed up the warmup
            server_args.device,
            log_decode_step=0,
            profile=False,
            profile_filename_prefix="",  # not used
        )

        # Benchmark
        rank_print("Benchmark ...")
        for il, ol in itertools.product(bench_args.input_len, bench_args.output_len):
            reqs = prepare_synthetic_inputs_for_latency_test(bs, il)
            ret = latency_test_run_once(
                bench_args.run_name,
                model_runner,
                rank_print,
                reqs,
                bs,
                il,
                ol,
                server_args.device,
                bench_args.log_decode_step,
                bench_args.profile if tp_rank == 0 else None,
                bench_args.profile_filename_prefix,
            )
            if ret is not None:
                result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def main(server_args, bench_args):
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
