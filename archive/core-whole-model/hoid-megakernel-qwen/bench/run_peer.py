"""One measurement of a peer engine in one fresh process: ``--engine sglang`` or ``--engine trtllm``
(the TensorRT-LLM LLM API), ``--batch`` concurrent requests at ``--context``.

The workload, pass sequence and report match ``run_one.py``: engine startup, one untimed pass of
the exact workload, the timed pass, then the correctness-gate pass. Every row is submitted in one
go and all prefills land in one step before the first decode step (each engine's check enforces it;
TensorRT-LLM needs ``admit_rows_together`` for it).

Timers, per engine, over the same window as ``run_one.py`` (last row's prefill token -> final token):
- sglang: the frontend stamps each scheduler step's output batch on arrival
  (``workload.decode_window``, which also checks every step carries one token per row).
- trtllm: each streamed token is stamped on arrival (``workload.stream_window``).

``--set key=value`` (JSON value) overrides an engine option; ``peer_configs.json`` holds the
tuned options ``run_all.py`` passes for each engine and batch size.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import statistics
import sys
import threading
import time

from workload import (BATCHES, GATED, GENERATED, MAX_MODEL_LEN, PROMPT_TOKENS, decode_window, gate_prompt,
                      prompt, stream_window)

VERSIONS = {'sglang': '0.5.21', 'trtllm': '1.2.1'}
ADMISSION_SECONDS = 2.0


def admit_rows_together():
    """Make an idle TensorRT-LLM executor wait for the whole batch before it schedules.

    Requests reach the executor one by one and an idle executor starts on the first, so row 0
    prefills alone and the other rows prefill a step later, next to row 0's first decode token.
    vLLM, Hoid and SGLang see every row before their first step; this gives TensorRT-LLM the same
    start. Only the idle wake-up waits (until ``max_batch_size`` requests are in, at most
    ``ADMISSION_SECONDS``); fetches with requests in flight are untouched, so decode steps are not.
    The patch reaches the executor because its MPI workers re-import this module.
    """
    try:
        from tensorrt_llm._torch.pyexecutor.executor_request_queue import ExecutorRequestQueue
    except ImportError:
        return
    fetch = ExecutorRequestQueue._get_from_request_queue

    def fetch_whole_batch(self, timeout):
        items = fetch(self, timeout)
        idle = timeout is None or timeout.total_seconds() > 0
        deadline = time.monotonic() + ADMISSION_SECONDS
        while idle and 0 < len(items) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                items.append(self.request_queue.get(timeout=remaining))
            except queue.Empty:
                break
        return items

    ExecutorRequestQueue._get_from_request_queue = fetch_whole_batch
    print(f'run_peer: whole-batch admission installed in pid {os.getpid()}', file=sys.stderr, flush=True)


admit_rows_together()


def prefill_budget(batch, prompt_tokens):
    """A token budget that prefills every row in one step, as for stock vLLM."""
    return max(MAX_MODEL_LEN, -(-batch * prompt_tokens // 1024) * 1024)


class SGLang:
    def __init__(self, weights, batch, prompt_tokens, gpu_memory_utilization, overrides):
        import sglang
        self.version = sglang.__version__
        budget = prefill_budget(batch, prompt_tokens)
        self.options = dict(
            model_path=str(weights), dtype='bfloat16', kv_cache_dtype='auto', tp_size=1, random_seed=0,
            context_length=MAX_MODEL_LEN, max_running_requests=batch, disable_radix_cache=True,
            mem_fraction_static=gpu_memory_utilization, skip_tokenizer_init=True,
            chunked_prefill_size=budget, max_prefill_tokens=budget, cuda_graph_max_bs_decode=batch,
            stream_interval=1, log_level='warning')
        self.options.update(overrides)
        self.engine = sglang.Engine(**self.options)
        manager = self.engine.tokenizer_manager
        original = manager._handle_batch_output
        self.events = None

        async def record(recv_obj):
            if self.events is not None and getattr(recv_obj, 'output_ids', None) is not None:
                self.events.append(dict(timestamp=time.monotonic(),
                                        tokens={rid: len(ids) for rid, ids in zip(recv_obj.rids, recv_obj.output_ids)}))
            return await original(recv_obj)
        manager._handle_batch_output = record
        self.passes = 0

    def drive(self, prompts, max_tokens):
        """Returns (timing, tokens per row)."""
        self.passes += 1
        names = list(prompts)
        rids = {f'{name}-p{self.passes}': name for name in names}
        params = dict(temperature=0, max_new_tokens=max_tokens, ignore_eos=True)
        tokens = {name: [] for name in names}
        self.events = events = []

        async def run():
            stream = await self.engine.async_generate(
                input_ids=[prompts[n] for n in names], sampling_params=[params] * len(names),
                rid=list(rids), stream=True)
            async for chunk in stream:
                tokens[names[chunk['index']]] = list(chunk['output_ids'])
        self.engine.loop.run_until_complete(run())
        self.events = None
        named = [dict(timestamp=e['timestamp'], tokens={rids[r]: n for r, n in e['tokens'].items() if r in rids})
                 for e in events]
        return decode_window(named, names, max_tokens - 1), tokens

    def close(self):
        self.engine.shutdown()


class TensorRTLLM:
    """``backend=pytorch`` (the default LLM API) or ``backend=tensorrt`` (a compiled TensorRT engine,
    built at startup and cached across processes)."""

    def __init__(self, weights, batch, prompt_tokens, gpu_memory_utilization, overrides):
        import tensorrt_llm
        from tensorrt_llm.llmapi import KvCacheConfig
        self.version = tensorrt_llm.__version__
        overrides = dict(overrides)
        backend = overrides.pop('backend', 'pytorch')
        kv = dict(enable_block_reuse=False, free_gpu_memory_fraction=gpu_memory_utilization,
                  **overrides.pop('kv_cache_config', {}))
        common = dict(model=str(weights), dtype='bfloat16', tensor_parallel_size=1, enable_chunked_prefill=False,
                      skip_tokenizer_init=True)
        budget = prefill_budget(batch, prompt_tokens)
        if backend == 'pytorch':
            from tensorrt_llm import LLM
            from tensorrt_llm.llmapi import CudaGraphConfig, TorchCompileConfig
            graphs = overrides.pop('cuda_graph_config', {'batch_sizes': [batch]})
            compile_config = overrides.pop('torch_compile_config', None)
            self.options = dict(common, backend=backend, max_batch_size=batch, max_seq_len=MAX_MODEL_LEN,
                                max_num_tokens=budget, kv_cache_config=kv, cuda_graph_config=graphs,
                                torch_compile_config=compile_config, **overrides)
            built = dict(self.options, kv_cache_config=KvCacheConfig(**kv),
                         cuda_graph_config=CudaGraphConfig(**graphs) if graphs is not None else None,
                         torch_compile_config=TorchCompileConfig(**compile_config) if compile_config else None)
        elif backend == 'tensorrt':
            from tensorrt_llm._tensorrt_engine import LLM
            from tensorrt_llm.builder import BuildConfig
            from tensorrt_llm.llmapi import ExtendedRuntimePerfKnobConfig
            build = dict(max_batch_size=batch, max_seq_len=MAX_MODEL_LEN, max_input_len=MAX_MODEL_LEN,
                         max_num_tokens=budget, **overrides.pop('build_config', {}))
            knobs = dict(cuda_graph_mode=True, multi_block_mode=True,
                         **overrides.pop('extended_runtime_perf_knob_config', {}))
            self.options = dict(common, backend=backend, kv_cache_config=kv, build_config=build,
                                extended_runtime_perf_knob_config=knobs, enable_build_cache=True, **overrides)
            built = dict(self.options, kv_cache_config=KvCacheConfig(**kv), build_config=BuildConfig(**build),
                         extended_runtime_perf_knob_config=ExtendedRuntimePerfKnobConfig(**knobs))
        else:
            raise ValueError(f'unknown TensorRT-LLM backend {backend!r}')
        del built['backend']
        self.llm = LLM(**built)
        # The tokenizer is skipped; EOS is ignored anyway, but the sampler needs an end id.
        eos = json.loads((weights / 'generation_config.json').read_text())['eos_token_id']
        self.end_id = eos[0] if isinstance(eos, list) else eos

    def drive(self, prompts, max_tokens):
        import asyncio
        from tensorrt_llm import SamplingParams
        names = list(prompts)
        params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True, detokenize=False,
                                end_id=self.end_id, pad_id=self.end_id, return_perf_metrics=True)
        arrivals = {name: [] for name in names}
        tokens = {name: [] for name in names}
        timing = {}

        async def row(name, request):
            async for output in request:
                now = time.monotonic()
                ids = output.outputs[0].token_ids
                arrivals[name].extend([now] * (len(ids) - len(tokens[name])))
                tokens[name] = list(ids)
            timing[name] = output.outputs[0].request_perf_metrics.timing_metrics

        async def run():
            requests = [self.llm.generate_async(prompts[n], params, streaming=True) for n in names]
            await asyncio.gather(*(row(n, r) for n, r in zip(names, requests)))
        asyncio.run(run())
        # Client arrival order cannot prove one prefill step (a late row's prefill token arrives with an
        # early row's first decode token), so compare first tokens on the executor's clock instead.
        firsts = [timing[name].first_token_time.total_seconds() for name in names]
        step = statistics.median(b - a for name in names for a, b in zip(arrivals[name], arrivals[name][1:]))
        if max(firsts) - min(firsts) >= step:
            raise ValueError(f'rows were not prefilled in one step: first tokens '
                             f'{(max(firsts) - min(firsts)) * 1e3:.3f} ms apart, median step {step * 1e3:.3f} ms')
        for name in names:
            # A streamed result that carried several tokens means the window cannot be resolved per step.
            if len(set(arrivals[name])) != len(arrivals[name]):
                raise RuntimeError(f'{name}: tokens arrived coalesced; per-step stamps are unavailable')
        return stream_window(arrivals, max_tokens - 1), tokens

    def close(self):
        self.llm.shutdown()


ENGINES = {'sglang': SGLang, 'trtllm': TensorRTLLM}


def watchdog(seconds):
    """Kill the whole process group (engine workers included) if the run hangs."""
    def fire():
        print(f'run_peer watchdog: no completion after {seconds} s', file=sys.stderr, flush=True)
        os.killpg(os.getpgrp(), signal.SIGKILL)
    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--engine', choices=tuple(ENGINES), required=True)
    parser.add_argument('--batch', type=int, choices=BATCHES, required=True)
    parser.add_argument('--context', choices=tuple(PROMPT_TOKENS), required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.5)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=JSON',
                        help='engine option override, e.g. --set attention_backend=\'"fa3"\'')
    parser.add_argument('--watchdog', type=float, default=1800)
    args = parser.parse_args()
    if args.report.exists():
        sys.exit(f'{args.report} exists; reports are never overwritten')
    overrides = {}
    for item in args.set:
        key, _, value = item.partition('=')
        overrides[key] = json.loads(value)
    watchdog(args.watchdog)
    weights = args.weights.resolve()
    prompt_tokens = PROMPT_TOKENS[args.context]
    prompts = {f'row{r}': prompt(r, prompt_tokens) for r in range(args.batch)}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(weights)
    gate_prompts = {f'gate{r}': gate_prompt(tokenizer, r, prompt_tokens) for r in range(args.batch)}

    import torch
    started = time.monotonic()
    engine = ENGINES[args.engine](weights, args.batch, prompt_tokens, args.gpu_memory_utilization, overrides)
    startup = time.monotonic() - started
    if engine.version != VERSIONS[args.engine]:
        sys.exit(f'{args.engine} {VERSIONS[args.engine]} required, found {engine.version}')
    rows = list(prompts)
    try:
        warm, warm_tokens = engine.drive(prompts, GENERATED)
        timing, tokens = engine.drive(prompts, GENERATED)
        # The gate rows decode together as one batch, like the timed rows (the window check proves it).
        _, gate_tokens = engine.drive(gate_prompts, GATED)
    finally:
        engine.close()
    for name, generated in {**tokens, **warm_tokens}.items():
        if len(generated) != GENERATED:
            raise RuntimeError(f'{name} generated {len(generated)} tokens, expected {GENERATED}')
    digest = lambda t: hashlib.sha256(json.dumps([t[r] for r in rows]).encode()).hexdigest()
    report = dict(
        schema='qwen3-4b-vllm-hoid.run.v1', engine=args.engine, batch=args.batch, context=args.context,
        prompt_tokens=prompt_tokens, generated_tokens_per_row=GENERATED, **timing,
        startup_seconds=startup, warmup_tokens_per_second=warm['tokens_per_second'],
        tokens=tokens, tokens_sha256=digest(tokens), warmup_tokens_sha256=digest(warm_tokens),
        gate_tokens=gate_tokens, engine_options=engine.options, engine_version=engine.version,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        timer=('frontend arrival stamps of each scheduler step: last prefill token ready -> final token ready'
               if args.engine == 'sglang' else
               'per-token stream arrival stamps: last prefill token ready -> final token ready'))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as output:
        output.write(json.dumps(report, indent=1, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ('engine', 'batch', 'context', 'tokens_per_second',
                                             'tpot_ms_p50')}), flush=True)


if __name__ == '__main__':
    main()
