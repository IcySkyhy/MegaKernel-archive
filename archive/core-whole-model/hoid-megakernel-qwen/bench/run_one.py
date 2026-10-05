"""One measurement in one fresh process: ``--engine hoid`` (vLLM + megakernel plugin) or
``--engine stock`` (plain vLLM 0.29.0), ``--batch`` concurrent requests at ``--context``.

Sequence: engine startup (profiling, CUDA-graph capture, torch.compile for stock), one untimed
pass of the exact workload, the timed pass, then the correctness-gate pass: ``--batch``
natural-language prompts of the same length, decoded together for ``GATED`` tokens. All requests are submitted while the scheduler
is paused and released together, so every row prefills before the first decode step.

Timer: the engine core stamps each step's outputs when its sampled tokens are ready. The decode
window runs from the step that delivers the last row's prefill token to the step that delivers
the final token (``workload.decode_window``); startup, warmup and prefill lie outside it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

from workload import BATCHES, GATED, GENERATED, MAX_MODEL_LEN, PROMPT_TOKENS, decode_window, gate_prompt, prompt

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ENV = {'VLLM_PLUGINS': 'qwen3_megakernel', 'VLLM_USE_V2_MODEL_RUNNER': '1'}


def engine_options(engine, weights, batch, prompt_tokens, bundle, gpu_memory_utilization):
    common = dict(model=str(weights), dtype='bfloat16', tensor_parallel_size=1, seed=0,
                  max_model_len=MAX_MODEL_LEN, max_num_seqs=batch, enable_prefix_caching=False,
                  gpu_memory_utilization=gpu_memory_utilization, skip_tokenizer_init=True,
                  generation_config='vllm')
    if engine == 'stock':
        # vLLM defaults otherwise: FA3, FULL_AND_PIECEWISE CUDA graphs, torch.compile, async
        # scheduling, chunked prefill. The batched-token budget lets every prompt prefill in one step.
        # Programmatic Dependent Launch is off by default and vLLM has no switch for it on a dense
        # BF16 model on Hopper; Inductor's switch turns it on for the compiled Triton kernels (norms,
        # RoPE, SiLU-mul, residual adds). cuBLAS GEMMs and FA3 attention have no such switch.
        prompts = batch * prompt_tokens
        return dict(common, max_num_batched_tokens=max(MAX_MODEL_LEN, -(-prompts // 1024) * 1024),
                    compilation_config={'inductor_compile_config': {'triton.enable_pdl': True}})
    return dict(common, kv_cache_dtype='bfloat16', block_size=32, load_format='safetensors',
                max_num_batched_tokens=MAX_MODEL_LEN, enable_chunked_prefill=False,
                attention_config={'backend': 'FLASH_ATTN', 'flash_attn_version': 3},
                compilation_config={'cudagraph_mode': 'FULL_DECODE_ONLY',
                                    'cudagraph_capture_sizes': [1, 2, 4, 8] if batch == 8 else [1, 2, 4]},
                scheduler_cls='vllm_qwen3_megakernel.scheduler.MegakernelScheduler',
                worker_cls='vllm_qwen3_megakernel.worker.StandaloneWorker',
                hf_overrides={'architectures': ['StandaloneQwen3ForCausalLM']},
                additional_config={'megakernel': {
                    'bundle': str(bundle), 'raw_config': json.loads((weights / 'config.json').read_text())}})


class OutputRecorder:
    """Wraps the frontend output processor to keep the engine core's own step timestamps."""

    def __init__(self, engine):
        self.events = None
        self.original = engine.output_processor.process_outputs
        engine.output_processor.process_outputs = self

    def __call__(self, outputs, engine_core_timestamp=None, iteration_stats=None):
        if self.events is not None and outputs:
            self.events.append(dict(timestamp=float(engine_core_timestamp),
                                    tokens={o.request_id: len(o.new_token_ids) for o in outputs}))
        return self.original(outputs, engine_core_timestamp=engine_core_timestamp,
                             iteration_stats=iteration_stats)


def drive_pass(llm, recorder, prompts, max_tokens=GENERATED):
    """Submit every row while the scheduler is paused, release them together and drain.
    Returns output events keyed by row name and each row's generated tokens."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    engine = llm.llm_engine
    core = engine.engine_core
    params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True, detokenize=False,
                            output_kind=RequestOutputKind.DELTA)
    core.call_utility('pause_scheduler', 'keep', False)
    for name, ids in prompts.items():
        engine.add_request(name, {'prompt_token_ids': ids}, params)
    internal = {engine.output_processor.external_req_ids[name][0]: name for name in prompts}
    recorder.events = events = []
    tokens = {name: [] for name in prompts}
    core.call_utility('resume_scheduler')
    while engine.has_unfinished_requests():
        for output in engine.step():
            tokens[output.request_id].extend(output.outputs[0].token_ids)
    recorder.events = None
    for name, generated in tokens.items():
        if len(generated) != max_tokens:
            raise RuntimeError(f'{name} generated {len(generated)} tokens, expected {max_tokens}')
    named = [dict(timestamp=e['timestamp'],
                  tokens={internal[r]: n for r, n in e['tokens'].items() if r in internal}) for e in events]
    return named, tokens


def watchdog(seconds):
    """Kill the whole process group (engine core included) if the run hangs."""
    def fire():
        print(f'run_one watchdog: no completion after {seconds} s', file=sys.stderr, flush=True)
        os.killpg(os.getpgrp(), signal.SIGKILL)
    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--engine', choices=('hoid', 'stock'), required=True)
    parser.add_argument('--batch', type=int, choices=BATCHES, required=True)
    parser.add_argument('--context', choices=tuple(PROMPT_TOKENS), required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'src' / 'vllm_qwen3_megakernel' / 'cubins')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.5)
    parser.add_argument('--watchdog', type=float, default=1800)
    args = parser.parse_args()
    if args.report.exists():
        sys.exit(f'{args.report} exists; reports are never overwritten')
    for key, value in PLUGIN_ENV.items():
        if (os.environ.get(key) == value) != (args.engine == 'hoid'):
            sys.exit(f'{key}={value} must be set for the hoid engine and unset for stock')
    watchdog(args.watchdog)
    weights, bundle = args.weights.resolve(), args.bundle.resolve()
    prompt_tokens = PROMPT_TOKENS[args.context]
    options = engine_options(args.engine, weights, args.batch, prompt_tokens, bundle,
                             args.gpu_memory_utilization)
    prompts = {f'row{r}': prompt(r, prompt_tokens) for r in range(args.batch)}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(weights)
    gate_prompts = {f'gate{r}': gate_prompt(tokenizer, r, prompt_tokens) for r in range(args.batch)}

    import torch
    import vllm
    from vllm import LLM
    if vllm.__version__ != '0.29.0':
        sys.exit(f'vLLM 0.29.0 required, found {vllm.__version__}')
    started = time.monotonic()
    llm = LLM(**options)
    startup = time.monotonic() - started
    recorder = OutputRecorder(llm.llm_engine)
    rows = list(prompts)
    try:
        warm_events, warm_tokens = drive_pass(llm, recorder, prompts)
        warm = decode_window(warm_events, rows, GENERATED - 1)
        events, tokens = drive_pass(llm, recorder, prompts)
        timing = decode_window(events, rows, GENERATED - 1)
        gate_events, gate_tokens = drive_pass(llm, recorder, gate_prompts, GATED)
        # Proves the gate rows decoded together as one batch, like the timed rows.
        decode_window(gate_events, list(gate_prompts), GATED - 1)
    finally:
        llm.llm_engine.engine_core.shutdown()
    digest = lambda t: hashlib.sha256(json.dumps([t[r] for r in rows]).encode()).hexdigest()
    report = dict(
        schema='qwen3-4b-vllm-hoid.run.v1', engine=args.engine, batch=args.batch, context=args.context,
        prompt_tokens=prompt_tokens, generated_tokens_per_row=GENERATED, **timing,
        startup_seconds=startup, warmup_tokens_per_second=warm['tokens_per_second'],
        tokens=tokens, tokens_sha256=digest(tokens), warmup_tokens_sha256=digest(warm_tokens),
        gate_tokens=gate_tokens,
        engine_options=options, vllm=vllm.__version__, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        bundle_index_sha256=(hashlib.sha256((bundle / 'index.json').read_bytes()).hexdigest()
                             if args.engine == 'hoid' else None),
        timer='engine-core step stamps: last prefill token ready -> final token ready')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as output:
        output.write(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('engine', 'batch', 'context', 'tokens_per_second',
                                             'tpot_ms_p50')}), flush=True)


if __name__ == '__main__':
    main()
