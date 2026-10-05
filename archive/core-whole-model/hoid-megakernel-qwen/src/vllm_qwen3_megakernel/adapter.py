"""Pinned semantic phases, physical-cache admission and stock weight borrowing."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .config import (BUCKETS, EngineConfig, HF_CONFIG, KVCapacity, MAX_MODEL_LEN, PAGE,
                     PageDescriptor, RequestConfig, SUPPORTED_MODEL, TABLE_ENTRIES,
                     UnsupportedConfig, VLLM_VERSION,
                     page_cache_layout, validate_engine_config, validate_max_num_seqs,
                     validate_model_config, validate_request)
from enum import Enum, auto

ARCHITECTURE = 'StandaloneQwen3ForCausalLM'
FAULT_TOKEN = -2147483000

@dataclass(frozen=True)
class AdapterConfig:
    bundle: str
    raw_config: dict

    @classmethod
    def from_vllm(cls, config):
        from importlib.metadata import version
        if version('vllm') != VLLM_VERSION:
            raise UnsupportedConfig('requires pinned vLLM 0.29.0')
        extra = config.additional_config
        if not isinstance(extra, dict) or set(extra) != {'megakernel'}:
            raise UnsupportedConfig('explicit megakernel additional_config required')
        value = extra['megakernel']
        if not isinstance(value, dict) or set(value) != {'bundle', 'raw_config'}:
            raise UnsupportedConfig('bundle and raw_config required')
        result = cls(**value)
        if not Path(result.bundle).is_absolute():
            raise UnsupportedConfig('bundle path must be absolute')
        validate_model_config(result.raw_config, repository=SUPPORTED_MODEL.repository,
                              revision=SUPPORTED_MODEL.revision)
        validate_max_num_seqs(config.scheduler_config.max_num_seqs)
        from .scheduler import resolved_scheduler_is_active
        if not resolved_scheduler_is_active(config):
            raise UnsupportedConfig('--scheduler-cls vllm_qwen3_megakernel.scheduler.MegakernelScheduler '
                                    'with async scheduling is required')
        return result

class Phase(Enum):
    IDLE = auto()
    PREFILL = auto()
    DECODE = auto()
    PROFILE = auto()
    CAPTURE = auto()


class Lifecycle:
    """CPU-testable state machine; every bucket's validator derives row liveness on device."""
    def __init__(self, buckets=(1,)):
        self.phase = Phase.IDLE
        self.generation = 0
        self.profiling_cache = False
        self.graphs_live = False
        self.poisoned = False
        self.buckets = tuple(buckets)
        self.bucket = None

    def check(self):
        if self.poisoned:
            raise RuntimeError("standalone worker is poisoned; restart required")

    def fail(self, message):
        self.poisoned = True
        raise RuntimeError(message + "; standalone worker restart required")

    @contextmanager
    def scope(self, phase, bucket=None):
        self.check()
        old, old_bucket = self.phase, self.bucket
        self.phase = phase
        if bucket is not None:
            self.bucket = bucket
        try:
            yield
        finally:
            self.phase, self.bucket = old, old_bucket

    def cache_initialized(self, *, profiling):
        self.check()
        if self.graphs_live:
            self.fail("cannot replace KV allocations with live graphs")
        self.generation += 1
        self.profiling_cache = profiling
        self.bucket = None

    def prepared(self, *, has_prefill, num_reqs, num_tokens, dummy=False, all_prefill=True,
                 graph_mode=None, graph_reqs=None):
        self.check()
        self.bucket = None
        if dummy:
            self.phase = Phase.PROFILE
        elif has_prefill:
            if not all_prefill:
                self.fail("mixed prefill/decode step; the MegakernelScheduler must be active")
            self.phase = Phase.PREFILL
        else:
            if self.profiling_cache or not self.generation or num_reqs != num_tokens:
                self.fail("decode outside the real-cache one-token-per-row envelope")
            if graph_mode != "FULL" or graph_reqs not in self.buckets or not 1 <= num_reqs <= graph_reqs:
                self.fail(f"decode of {num_reqs} rows has no captured FULL bucket "
                          f"(descriptor {graph_mode}/{graph_reqs}, buckets {self.buckets})")
            self.phase = Phase.DECODE
            self.bucket = graph_reqs

    def completed_output(self, output, *, raw_tokens=None):
        self.check()
        # The stock completion method truncates rows using num_sampled. Inspect
        # the already-copied raw rows too, so cancellation cannot hide a fault.
        rows = output.sampled_token_ids if raw_tokens is None else raw_tokens
        if any(FAULT_TOKEN in row for row in rows):
            self.fail("native decode reported an asynchronous device fault")
        return output



def validate_resolved_model(config):
    """Check actual normalized model semantics, not just the raw input document."""
    model = config.model_config
    hf = model.hf_config
    expected = HF_CONFIG
    architecture = ARCHITECTURE
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "rms_norm_eps", "hidden_act",
                "attention_bias", "tie_word_embeddings", "vocab_size", "sliding_window"):
        actual = getattr(hf, key, None)
        if actual != expected[key] or (type(expected[key]) is int and type(actual) is not int):
            raise UnsupportedConfig(f"resolved model differs: {key}")
    rope = hf.rope_parameters
    if rope.get("rope_type") != "default" or rope.get("rope_theta") != 5000000:
        raise UnsupportedConfig("only pinned default NeoX RoPE is supported")
    if (getattr(hf, "is_causal", True) is not True
            or getattr(hf, "dual_chunk_attention_config", None) is not None
            or getattr(hf, "use_sliding_window", False)):
        raise UnsupportedConfig("unsupported attention semantics")
    if hf.architectures != [architecture]:
        raise UnsupportedConfig("distinct standalone architecture required")
    if model.hf_overrides != {"architectures": [architecture]}:
        raise UnsupportedConfig("only the distinct architecture override is permitted")



def validate_cache_layouts(caches, *, resolved_layout, blocks, profile=None):
    """Qualify the actual per-layer views, not just the logical shape/config."""
    layer_count = 36
    layouts = []
    for t in caches:
        layout = page_cache_layout(t.shape, t.stride())
        if (str(t.dtype) != "torch.bfloat16" or str(t.layout) != "torch.strided"
                or t.is_conj() or t.is_neg() or t.shape[0] != blocks):
            raise UnsupportedConfig("unexpected actual FA3 shared KV layout/capacity")
        layouts.append(layout)
    if len(layouts) != layer_count or len(set(layouts)) != 1:
        raise UnsupportedConfig(f"all {layer_count} cache layers must have uniform capacity and layout")
    if layouts[0] != resolved_layout:
        raise UnsupportedConfig("actual cache strides disagree with resolved physical layout")
    return layouts[0]


def validate_runner(runner, *, buckets, profiling=False):
    """Every exclusion is supplied from resolved config; none use admission defaults."""
    from .scheduler import resolved_scheduler_is_active
    c = runner.vllm_config
    m, p, s, kv, cc = (c.model_config, c.parallel_config, c.scheduler_config,
                       c.cache_config, c.compilation_config)
    profile = runner.model.megakernel_config
    validate_resolved_model(c)
    layers = [layer.self_attn.attn for layer in runner.model.model.layers]
    if len(layers) != 36 or len(runner.kv_cache_config.kv_cache_groups) != 1:
        raise UnsupportedConfig(f"exactly {36} Attention modules and one cache group required")
    names = [f"model.layers.{i}.self_attn.attn" for i in range(36)]
    if ([a.layer_name for a in layers] != names
            or set(runner.kv_cache_config.kv_cache_groups[0].layer_names) != set(names)):
        raise UnsupportedConfig("actual Attention/cache provenance differs from model layer names")
    if any(a.attn_backend.get_name() != "FLASH_ATTN" or a.impl.vllm_flash_attn_version != 3 for a in layers):
        raise UnsupportedConfig("actual backend must be FlashAttention 3")
    if runner.kernel_block_sizes != [32] or kv.block_size != 32:
        raise UnsupportedConfig("allocator and attention kernel pages must both be 32 tokens")
    if kv.kv_cache_layout != "LBNHC":
        raise UnsupportedConfig(f"VLLM_KV_CACHE_LAYOUT must resolve to LBNHC, got {kv.kv_cache_layout}")
    if not resolved_scheduler_is_active(c):
        raise UnsupportedConfig("the split-phase MegakernelScheduler is not the active scheduler")
    rows = s.max_num_seqs
    if served_buckets(buckets, rows) != tuple(buckets):
        raise UnsupportedConfig(f"prepared buckets {tuple(buckets)} differ from the ones "
                                f"max_num_seqs {rows} serves")
    table = runner.block_tables.input_block_tables
    if (len(table) != 1 or table[0].shape[0] < rows or table[0].shape[1] != TABLE_ENTRIES
            or table[0].stride() != (TABLE_ENTRIES, 1) or str(table[0].dtype) != "torch.int32"):
        raise UnsupportedConfig(f"block table rows must be contiguous I32 with {TABLE_ENTRIES} entries")
    layout = validate_cache_layouts([a.kv_cache for a in layers],
                                   resolved_layout=kv.kv_cache_layout,
                                   blocks=runner.kv_cache_config.num_blocks, profile=profile)
    mode = runner.cudagraph_manager.cudagraph_mode.name
    off = c.offload_config
    # No other requests reserve prefill pages; the extra page covers async lookahead.
    blocks = runner.kv_cache_config.num_blocks
    engine = EngineConfig(
        vllm_version=VLLM_VERSION,
        model_runner="V2" if c.use_v2_model_runner else "V1",
        max_model_len=m.max_model_len, max_num_batched_tokens=s.max_num_batched_tokens,
        pages=PageDescriptor(backend="FLASH_ATTN", backend_version="3", layout=layout,
                             allocator_block_size_tokens=kv.block_size, kernel_block_size_tokens=32),
        dtype=str(m.dtype).removeprefix("torch."), kv_cache_dtype=kv.cache_dtype,
        tensor_parallel_size=p.tensor_parallel_size, pipeline_parallel_size=p.pipeline_parallel_size,
        data_parallel_size=p.data_parallel_size, prefill_context_parallel_size=p.prefill_context_parallel_size,
        decode_context_parallel_size=p.decode_context_parallel_size, max_num_seqs=s.max_num_seqs,
        kv_cache_groups=len(runner.kv_cache_config.kv_cache_groups),
        enable_chunked_prefill=s.enable_chunked_prefill, enable_prefix_caching=kv.enable_prefix_caching,
        speculative_config=c.speculative_config is not None, enable_lora=c.lora_config is not None,
        cpu_offload_gb=off.uva.cpu_offload_gb, offload_group_size=off.prefetch.offload_group_size,
        kv_offloading_size=kv.kv_offloading_size, kv_transfer_config=c.kv_transfer_config is not None,
        enable_dbo=p.enable_dbo, quantization=m.quantization,
        hf_overrides=False, rope_overrides=False,
        num_lookahead_tokens=c.num_lookahead_tokens, cudagraph_mode=mode,
        cudagraph_capture_sizes=tuple(cc.cudagraph_capture_sizes))
    capacity = KVCapacity(total_allocator_blocks=blocks, unavailable_blocks=0,
                          watermark_blocks=int(s.watermark * blocks),
                          runner_reserved_blocks=1, runner_reserved_tokens=0)
    if profiling:
        capacity = KVCapacity(total_allocator_blocks=s.max_num_seqs * (MAX_MODEL_LEN // PAGE) + 2,
                              unavailable_blocks=0, watermark_blocks=0,
                              runner_reserved_blocks=1, runner_reserved_tokens=0)
    validate_engine_config(engine, capacity)
    return {f"kv.{i}": a.kv_cache for i, a in enumerate(layers)}



def borrow_weights(model, *, profile=None):
    import torch
    layer_count = 36
    if len(model.model.layers) != layer_count:
        raise UnsupportedConfig(f"borrowed model must have exactly {layer_count} layers")
    tensors = {}
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}."
        attn, mlp = layer.self_attn, layer.mlp
        qkv, gu = attn.qkv_proj.weight, mlp.gate_up_proj.weight
        if qkv.shape != (6144, 2560) or gu.shape != (19456, 2560):
            raise UnsupportedConfig("expected TP1 packed row-major projections")
        local = {
            "self_attn.q_proj.weight": qkv[:4096],
            "self_attn.k_proj.weight": qkv[4096:5120],
            "self_attn.v_proj.weight": qkv[5120:],
            "self_attn.o_proj.weight": attn.o_proj.weight,
            "self_attn.q_norm.weight": attn.q_norm.weight,
            "self_attn.k_norm.weight": attn.k_norm.weight,
            "mlp.gate_proj.weight": gu[:9728], "mlp.up_proj.weight": gu[9728:],
            "mlp.down_proj.weight": mlp.down_proj.weight,
            "input_layernorm.weight": layer.input_layernorm.weight,
            "post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
        }
        for name, tensor in local.items():
            if not tensor.is_contiguous() or tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda":
                raise UnsupportedConfig(f"cannot borrow weight {prefix + name}")
            tensors[prefix + name] = tensor.detach()
    cache = model.model.layers[0].self_attn.rotary_emb.cos_sin_cache
    for layer in model.model.layers:
        rope = layer.self_attn.rotary_emb
        if (rope.cos_sin_cache is not cache or rope.head_size != 128 or rope.rotary_dim != 128
                or rope.base != 5000000 or not rope.is_neox_style):
            raise UnsupportedConfig("all layers must share the pinned stock NeoX RoPE cache")
    if cache.dtype != torch.bfloat16 or cache.shape[0] < MAX_MODEL_LEN or cache.shape[1] != 128:
        raise UnsupportedConfig("actual stock RoPE must be BF16 [positions,128]")
    # Preserve stock BF16 rounding instead of regenerating RoPE with host trigonometry.
    tensors["cos"] = cache[:MAX_MODEL_LEN, :64].to(dtype=torch.float32)
    tensors["sin"] = cache[:MAX_MODEL_LEN, 64:].to(dtype=torch.float32)
    return tensors


def guard_request_data(data, max_model_len):
    params = data.sampling_params
    if params is None or data.prompt_token_ids is None:
        raise UnsupportedConfig("only tokenized text generation requests are supported")
    validate_request(RequestConfig(prompt_tokens=len(data.prompt_token_ids), max_tokens=params.max_tokens,
                                   n=params.n), max_model_len=max_model_len)
    if (data.lora_request is not None or data.mm_features or data.prompt_embeds is not None
            or params.prompt_logprobs is not None or params.logprobs is not None
            or params.structured_outputs is not None):
        raise UnsupportedConfig("LoRA, multimodal, logprobs and structured output requests are unsupported")



def install_request_guard():
    """Run after stock tokenization but before EngineCore/KV allocation.

    InputProcessor.__init__ constructs process_inputs_async from this method,
    so installing before run_server covers both synchronous and async rendering.
    """
    from functools import wraps
    from vllm.v1.engine.input_processor import InputProcessor
    original = InputProcessor.process_inputs
    if getattr(original, "_megakernel_guard", False):
        return

    @wraps(original)
    def checked(self, *args, **kwargs):
        request = original(self, *args, **kwargs)
        if "megakernel" in self.vllm_config.additional_config:
            guard_request_data(request, self.model_config.max_model_len)
            if request.resumable or request.session_id is not None:
                raise UnsupportedConfig("resumable/session requests are unsupported")
        return request

    checked._megakernel_guard = True
    InputProcessor.process_inputs = checked


async def request_envelope(request, call_next):
    """Documented vLLM --middleware hook; reject beam expansion at the HTTP edge."""
    from starlette.responses import JSONResponse
    if request.method == "POST":
        if request.url.path not in ("/v1/completions", "/v1/chat/completions"):
            return JSONResponse({"error": "only text completions/chat are supported"}, status_code=400)
        try:
            data = await request.json()
            if not isinstance(data, dict):
                raise UnsupportedConfig("expected a JSON request object")
            for key in ("n", "best_of", "beam_width"):
                if key in data and data[key] is not None and (type(data[key]) is not int or data[key] != 1):
                    raise UnsupportedConfig(f"{key} must be 1")
            for key in ("use_beam_search", "logprobs", "prompt_logprobs", "top_logprobs",
                        "structured_outputs", "response_format", "tools", "tool_choice"):
                if data.get(key) not in (None, False):
                    raise UnsupportedConfig(f"{key} is unsupported")
        except (ValueError, TypeError) as error:
            return JSONResponse({"error": str(error)}, status_code=400)
    return await call_next(request)


def served_buckets(bundle_buckets, max_num_seqs):
    """The bundle buckets a max_num_seqs engine serves: every bucket up to it must be present
    (a live batch pads to the smallest bucket that holds it)."""
    validate_max_num_seqs(max_num_seqs)
    missing = [b for b in BUCKETS if b <= max_num_seqs and b not in bundle_buckets]
    if missing:
        raise UnsupportedConfig(f"max_num_seqs {max_num_seqs} needs bundle buckets {missing}")
    return tuple(b for b in sorted(bundle_buckets) if b <= max_num_seqs)


class NativeAdapter:
    """Borrow checkpoint/cache once per generation; never copy model or KV data."""
    def __init__(self, model, config, device, execution_stream, *, max_num_seqs):
        import torch
        from vllm.logger import init_logger
        from .bundle import load_bundle
        from .decoder import BindingToken
        from .generation import Generation
        validate_max_num_seqs(max_num_seqs)
        self.model, self.config = model, config
        self.device, self.execution_stream = device, execution_stream
        self.bundle = load_bundle(config.bundle)
        # vLLM sizes its row buffers (block tables, seq_lens) to max_num_seqs, and never
        # schedules a larger decode batch, so only the buckets up to it are prepared.
        self.buckets = served_buckets(self.bundle.buckets, max_num_seqs)
        init_logger(__name__).info(
            "Qwen3 megakernel serving buckets %s (max_num_seqs %d; bundle buckets %s)",
            list(self.buckets), max_num_seqs, list(self.bundle.buckets))
        self.embedded = {b: torch.empty((b, 2560), dtype=torch.bfloat16, device=device)
                         for b in self.buckets}
        self.outputs = {b: torch.empty((b, 2560), dtype=torch.bfloat16, device=device)
                        for b in self.buckets}
        self.state = Lifecycle(self.buckets)
        self.token = BindingToken()
        self.weights = borrow_weights(model)
        self.bound = False
        for bucket in self.buckets:
            model.prepare_final_norm(self.outputs[bucket])
        caches = {}
        for i in range(36):
            cache = torch.empty((2, 32, 8, 256), dtype=torch.bfloat16, device=device)
            caches[f'kv.{i}'] = cache.permute(0, 2, 1, 3)
        rows = max(self.buckets)
        metadata = dict(positions=torch.zeros(rows, dtype=torch.int64, device=device),
                        seq_lens=torch.zeros(rows, dtype=torch.int32, device=device),
                        slots=torch.zeros(rows, dtype=torch.int64, device=device),
                        table=torch.zeros((rows, TABLE_ENTRIES), dtype=torch.int32, device=device))
        self.generation = Generation(self._prepare(caches, metadata), 1, profiling=True)

    @property
    def native(self):
        return self.generation.decoder

    def workspace(self, bucket):
        return self.native.bucket(bucket)

    def release_owned_references(self):
        model = self.model
        if getattr(model, 'megakernel_adapter', None) is self:
            model.megakernel_adapter = None
        self.state.poisoned = True
        self.weights.clear()
        for name in ('model', 'weights', 'device', 'execution_stream', 'embedded',
                     'outputs', 'generation', 'bundle'):
            setattr(self, name, None)
        self.bound = False

    def _prepare(self, caches, metadata):
        from .decoder import BatchedDecoder, DecoderSet
        decoders = {}
        for bucket in self.buckets:
            program = self.bundle.programs[bucket]
            bindings = dict(self.weights)
            bindings[program.embedded] = self.embedded[bucket]
            bindings.update({name: caches[name] for name in (f'kv.{i}' for i in range(36))})
            names = {b.name for b in program.bindings} - {program.rows_ctl}
            decoders[bucket] = BatchedDecoder(
                program, stream=self.execution_stream,
                bindings={name: bindings[name] for name in names},
                output=self.outputs[bucket], positions=metadata['positions'][:bucket],
                seq_lens=metadata['seq_lens'][:bucket], slots=metadata['slots'][:bucket],
                table=metadata['table'][:bucket], token=self.token)
        return DecoderSet(decoders, self.execution_stream)

    def bind_runner(self, runner, *, profiling):
        caches = validate_runner(runner, profiling=profiling, buckets=self.buckets)
        buffers, tables = runner.input_buffers, runner.block_tables
        metadata = dict(positions=buffers.positions, seq_lens=buffers.seq_lens,
                        slots=tables.slot_mappings[0], table=tables.input_block_tables[0])

        def rebind():
            # Runs after the previous generation is retired: from here on, any
            # decoder bound to the previous allocations fails its step check.
            self.token.advance()
            return self._prepare(caches, metadata)

        self.generation = self.generation.replace_profiling(rebind, profiling=profiling)
        self.state.cache_initialized(profiling=profiling)
        self.bound = True

    def decode(self, hidden):
        import torch
        self.state.check()
        if not self.bound or self.state.phase not in (Phase.DECODE, Phase.CAPTURE):
            self.state.fail('decode outside admitted phase/cache generation')
        rows = hidden.shape[0]
        if (hidden.ndim != 2 or hidden.shape[1] != 2560 or hidden.dtype != torch.bfloat16
                or rows != self.state.bucket or rows not in self.buckets):
            self.state.fail(f'decode requires BF16 [bucket,2560] stock embedding for bucket '
                            f'{self.state.bucket}, got {tuple(hidden.shape)}')
        self.embedded[rows].copy_(hidden)
        self.workspace(rows).enqueue(torch.cuda.current_stream(self.device))
        return self.outputs[rows]
