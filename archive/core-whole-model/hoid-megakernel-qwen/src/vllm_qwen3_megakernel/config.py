"""Strict pinned model, engine and request admission."""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

VLLM_VERSION = "0.29.0"
MAX_MODEL_LEN = 33792
PAGE = 32
TABLE_ENTRIES = MAX_MODEL_LEN // PAGE
BUCKETS = (1, 2, 4, 8)
MAX_NUM_SEQS = 8
# Explicit FULL_DECODE_ONLY capture sizes: every bucket, or (up to 4 rows) the pre-8 ladder.
CAPTURE_SIZES = (BUCKETS, (1, 2, 4))


class UnsupportedConfig(ValueError):
    """The supplied data is outside the proposed integration envelope."""


@dataclass(frozen=True)
class SupportedModel:
    repository: str = "Qwen/Qwen3-4B-Instruct-2507"
    revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    config_sha256: str = "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba"
    # qwen3.py applies per-head Q/K RMSNorm; get_rope defaults to NeoX style.
    qk_normalization: str = "per_head_rmsnorm"
    rope_style: str = "neox"
    runtime_qualified: bool = field(default=False, init=False)

    @property
    def config_url(self) -> str:
        return (
            f"https://huggingface.co/{self.repository}/resolve/"
            f"{self.revision}/config.json"
        )


SUPPORTED_MODEL = SupportedModel()

# Official raw HF config at config_url, fetched and SHA256 checked independently
# of the existing standalone example. Immutable values (including architectures).
HF_CONFIG = MappingProxyType({
    "architectures": ("Qwen3ForCausalLM",),
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9728,
    "max_position_embeddings": 262144,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 5000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
})


def _equal(name: str, actual: Any, expected: Any) -> None:
    # bool is an int subclass: False must not satisfy a count or vice versa.
    if isinstance(actual, bool) != isinstance(expected, bool) or actual != expected:
        raise UnsupportedConfig(f"{name}: expected {expected!r}, got {actual!r}")


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise UnsupportedConfig(f"{name}: expected an integer >= {minimum}, got {value!r}")


def validate_model_config(
    config: Mapping[str, Any], *, repository: str, revision: str
) -> SupportedModel:
    """Check the pinned *raw* HF config, before Transformers normalization.

    Unknown keys fail closed: a normalized/overridden config must not silently
    change Q/K normalization, biases, RoPE, or attention behavior. This validates
    config semantics, not model/tokenizer artifact hashes or loaded tensors.
    """
    _equal("repository", repository, SUPPORTED_MODEL.repository)
    _equal("revision", revision, SUPPORTED_MODEL.revision)
    unknown = config.keys() - HF_CONFIG.keys()
    if unknown:
        raise UnsupportedConfig(f"unrecognized raw model config fields: {sorted(unknown)}")
    for name, expected in HF_CONFIG.items():
        if name not in config:
            raise UnsupportedConfig(f"missing model config field: {name}")
        actual = config[name]
        if name == "architectures" and isinstance(actual, list):
            actual = tuple(actual)
        if type(expected) is int:
            _integer(name, actual)
        _equal(name, actual, expected)
    return SUPPORTED_MODEL


def page_cache_layout(shape, strides):
    """Exact dense per-layer permutations; names match the pinned allocator enum.

    Logical [B,H,N,C] alone says nothing about physical order. The full span of
    either admitted permutation is B*65536 BF16 elements, without holes/repacking.
    """
    if len(shape) != 4 or shape[0] <= 0 or tuple(shape[1:]) != (8, 32, 256):
        raise UnsupportedConfig("native KV requires logical [blocks,8,32,256]")
    layouts = {(65536, 8192, 256, 1): "LBHNC", (65536, 256, 2048, 1): "LBNHC"}
    try:
        return layouts[tuple(strides)]
    except KeyError:
        raise UnsupportedConfig("native KV requires exact dense LBHNC or LBNHC strides") from None


@dataclass(frozen=True, kw_only=True)
class PageDescriptor:
    """Observed allocation/table units, NOT a supported paged-cache ABI.

    Describe the actual backend version and layout; do not assume a page size or
    equate allocator blocks with attention-facing kernel blocks. For the pinned
    FlashAttention source, logical shape is [blocks, heads, page, 2*dim]; layout
    names identify PHYSICAL order (LBNHC token-major or LBHNC head-major), not
    logical shape. Actual strides, allocations and size need runtime qualification.
    """

    backend: str
    backend_version: str
    layout: str
    allocator_block_size_tokens: int
    kernel_block_size_tokens: int


@dataclass(frozen=True, kw_only=True)
class EngineConfig:
    """Proposed settings; construct from resolved worker/engine configuration.

    Exclusion defaults describe the proposal, not proof that the engine resolved
    them. The future adapter must read and supply every relevant setting. Boolean
    *_config fields report presence (non-None), including an empty config object.
    """

    vllm_version: str
    model_runner: str
    max_model_len: int
    max_num_batched_tokens: int
    pages: PageDescriptor
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    max_num_seqs: int = MAX_NUM_SEQS
    kv_cache_groups: int = 1
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    speculative_config: bool = False
    enable_lora: bool = False
    cpu_offload_gb: float = 0.0
    offload_group_size: int = 0
    kv_offloading_size: float | None = None
    kv_transfer_config: bool = False
    enable_dbo: bool = False
    quantization: str | None = None
    hf_overrides: bool = False
    rope_overrides: bool = False
    num_lookahead_tokens: int = 0
    cudagraph_mode: str = "FULL_DECODE_ONLY"
    cudagraph_capture_sizes: tuple[int, ...] = BUCKETS


@dataclass(frozen=True, kw_only=True)
class KVCapacity:
    """Resolved upper bounds in *allocator-block* units, for one KV group.

    All counts are mandatory, including zeros: unknown reservations cannot pass.
    ``total_allocator_blocks`` includes the pinned BlockPool's one null block.
    ``unavailable_blocks`` excludes that null block and counts all other occupied
    or withheld blocks. Watermark and runner block reservations are additional,
    disjoint headroom; runner token reservations are additional live token slots.
    Supply measured/resolved worst-case counts, not free GPU bytes or estimates.
    Do not multiply block counts by the model's layer count.

    The caller must ensure these bounds stay true until request retirement and
    serialize workspace/cache use. Arithmetic cannot enforce this at runtime.
    """

    total_allocator_blocks: int
    unavailable_blocks: int
    watermark_blocks: int
    runner_reserved_blocks: int
    runner_reserved_tokens: int


@dataclass(frozen=True)
class Stage0Admission:
    required_request_blocks: int
    available_request_blocks: int
    pages: PageDescriptor
    runtime_qualified: bool = field(default=False, init=False)
    pending_qualification: tuple[str, ...] = (
        "H200 device, dependencies, model/tokenizer artifacts and loaded weights",
        "attention backend/version, page sizes, layout, strides and table units",
        "RoPE table precision and cross-engine hidden/logit/KV numerical policy",
        "V2 phase/lifecycle seam, compilation, graph capture/replay and async scheduling",
        "observed reservation bounds, exclusive workspace use and preemption failure handling",
    )


def validate_max_num_seqs(value) -> None:
    _integer("max_num_seqs", value, 1)
    if value not in BUCKETS:
        raise UnsupportedConfig(f"max_num_seqs must be one of {BUCKETS}, got {value}")


def validate_capture_sizes(sizes, max_num_seqs) -> None:
    """vLLM captures only the sizes up to max_num_seqs; larger listed sizes are inert."""
    if type(sizes) is not tuple:
        raise UnsupportedConfig("cudagraph_capture_sizes must be an immutable tuple")
    for size in sizes:
        _integer("cudagraph_capture_sizes", size, 1)
    if sizes not in CAPTURE_SIZES or max(sizes) < max_num_seqs:
        raise UnsupportedConfig(
            f"cudagraph_capture_sizes: expected {BUCKETS} (or (1, 2, 4) up to 4 rows) covering "
            f"max_num_seqs {max_num_seqs}, got {sizes!r}")


def validate_engine_config(engine: EngineConfig, capacity: KVCapacity) -> Stage0Admission:
    """Check a candidate and its conservative full-envelope capacity bound.

    At the selected pin, VllmConfig.num_lookahead_tokens is zero without
    speculation. BlockPool reserves one null block. KVCacheManager allocates
    ceil(tokens / allocator_block_size) full-attention blocks; scheduler admission
    also retains watermark/reserved blocks. This bound reserves the whole
    MAX_MODEL_LEN envelope, not just the current prompt. Extra runner
    token slots are conservatively added *before* rounding, even if the engine
    caps a particular allocation at max_model_len. No async reservation is
    guessed to be zero. This is a static necessary guard, not a preemption hook.
    """
    for name, expected in (
        ("vllm_version", VLLM_VERSION), ("model_runner", "V2"),
        ("dtype", "bfloat16"),
        ("kv_cache_dtype", "bfloat16"), ("cudagraph_mode", "FULL_DECODE_ONLY"),
        ("quantization", None),
    ):
        _equal(name, getattr(engine, name), expected)
    for name in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
        "prefill_context_parallel_size", "decode_context_parallel_size",
        "kv_cache_groups",
    ):
        _integer(name, getattr(engine, name), 1)
        _equal(name, getattr(engine, name), 1)
    validate_max_num_seqs(engine.max_num_seqs)
    validate_capture_sizes(engine.cudagraph_capture_sizes, engine.max_num_seqs)
    for name in (
        "enable_chunked_prefill", "enable_prefix_caching", "speculative_config",
        "enable_lora", "kv_transfer_config", "enable_dbo", "hf_overrides", "rope_overrides",
    ):
        _equal(name, getattr(engine, name), False)
    _equal("cpu_offload_gb", engine.cpu_offload_gb, 0.0)
    _integer("offload_group_size", engine.offload_group_size)
    _equal("offload_group_size", engine.offload_group_size, 0)
    # At this pin even a numeric zero selects a KV offload connector.
    _equal("kv_offloading_size", engine.kv_offloading_size, None)
    _integer("num_lookahead_tokens", engine.num_lookahead_tokens)
    _equal("num_lookahead_tokens", engine.num_lookahead_tokens, 0)
    _integer("max_model_len", engine.max_model_len, 1)
    # vLLM sizes block-table rows from max_model_len; the programs read TABLE_ENTRIES per row.
    _equal("max_model_len", engine.max_model_len, MAX_MODEL_LEN)
    _integer("max_num_batched_tokens", engine.max_num_batched_tokens, engine.max_model_len)
    pages = engine.pages
    for name in ("backend", "backend_version", "layout"):
        value = getattr(pages, name)
        if not isinstance(value, str) or not value.strip():
            raise UnsupportedConfig(f"pages.{name}: an explicit observed descriptor is required")
    for name in ("allocator_block_size_tokens", "kernel_block_size_tokens"):
        _integer(f"pages.{name}", getattr(pages, name), 1)
    _equal("pages.layout", pages.layout, "LBNHC")
    for name in (
        "total_allocator_blocks", "unavailable_blocks", "watermark_blocks",
        "runner_reserved_blocks", "runner_reserved_tokens",
    ):
        _integer(name, getattr(capacity, name))
    reserved = (
        1 + capacity.unavailable_blocks + capacity.watermark_blocks
        + capacity.runner_reserved_blocks
    )
    available = capacity.total_allocator_blocks - reserved
    tokens = engine.max_model_len + engine.num_lookahead_tokens + capacity.runner_reserved_tokens
    block_size = pages.allocator_block_size_tokens
    # Every admitted row may grow to the full envelope at once.
    required = engine.max_num_seqs * ((tokens + block_size - 1) // block_size)
    if available < required:
        raise UnsupportedConfig(
            f"insufficient KV capacity: need {required} allocator blocks for "
            f"{engine.max_num_seqs} x {tokens} token slots, have {available} after "
            f"{reserved} reserved/unavailable blocks"
        )
    return Stage0Admission(required, available, pages)


@dataclass(frozen=True, kw_only=True)
class RequestConfig:
    """Token counts are after templating/BOS insertion and before allocation."""

    prompt_tokens: int
    max_tokens: int
    n: int = 1
    best_of: int = 1
    use_beam_search: bool = False
    beam_width: int = 1


def validate_request(request: RequestConfig, *, max_model_len: int) -> None:
    """Reject multiplicity/overlength; queued HTTP requests remain permitted.

    Empty prompts must first follow the engine's rejection/BOS policy. The last
    sampled token is not normally fed back, but reserve prompt + max_tokens as
    the conservative admission length. A one-token prompt is still prefill.
    """
    _integer("max_model_len", max_model_len, 1)
    if max_model_len > MAX_MODEL_LEN:
        raise UnsupportedConfig(f"max_model_len exceeds standalone envelope {MAX_MODEL_LEN}")
    for name in ("n", "best_of", "beam_width"):
        _integer(name, getattr(request, name), 1)
        _equal(name, getattr(request, name), 1)
    _equal("use_beam_search", request.use_beam_search, False)
    _integer("prompt_tokens", request.prompt_tokens, 1)
    _integer("max_tokens", request.max_tokens, 1)
    if request.prompt_tokens + request.max_tokens > max_model_len:
        raise UnsupportedConfig("prompt_tokens + max_tokens exceeds max_model_len")
