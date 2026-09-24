"""Kernel-source variants for decode experimentation on RTX 3090."""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_KERNEL_PATH = REPO_ROOT / "csrc" / "megakernel" / "fused_decode_ldg.cu"
QWEN_PERSISTENT_KERNEL_PATH = REPO_ROOT / "experimental" / "qwen_persistent_kernel.cu"


def _replace_pattern_once(source: str, pattern: str, replacement: str, label: str) -> str:
    """Replace a regex pattern once, failing loudly if not found."""
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"failed to apply variant transform: {label}")
    return updated


def _replace_text_once(source: str, old: str, new: str, label: str) -> str:
    """Replace an exact text block once, failing loudly if not found."""
    count = source.count(old)
    if count != 1:
        raise ValueError(f"failed to apply variant transform: {label}")
    return source.replace(old, new, 1)


def _replace_text_all(source: str, old: str, new: str, label: str) -> str:
    """Replace all exact text matches, failing if none are found."""
    count = source.count(old)
    if count < 1:
        raise ValueError(f"failed to apply variant transform: {label}")
    return source.replace(old, new)


def _apply_fast_math(source: str) -> str:
    """Switch exp/silu hot paths to fast approximate PTX ops."""
    anchor = """__device__ __forceinline__ float ldg_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

"""

    fast_math_helpers = """constexpr float LOG2E = 1.44269504088896340736f;

__device__ __forceinline__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float ptx_rcp(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float fast_exp(float x) {
    return ptx_exp2(x * LOG2E);
}

"""

    if "ptx_exp2" not in source:
        source = source.replace(anchor, anchor + fast_math_helpers)

    source = source.replace(
        "return x / (1.0f + expf(-x));",
        "return x * ptx_rcp(1.0f + fast_exp(-x));",
    )
    source = source.replace("expf(", "fast_exp(")
    return source


def _apply_num_blocks(source: str, num_blocks: int) -> str:
    """Retune decode cooperative grid size."""
    return _replace_pattern_once(
        source,
        r"^constexpr int LDG_NUM_BLOCKS = \d+;$",
        f"constexpr int LDG_NUM_BLOCKS = {num_blocks};",
        label="LDG_NUM_BLOCKS",
    )


def _apply_qwen_macro_tuning(source: str, num_blocks: int, block_size: int) -> str:
    """Retune qwen persistent-kernel launch macros for Ampere experiments."""
    source = _replace_pattern_once(
        source,
        r"^#define LDG_NUM_BLOCKS \d+$",
        f"#define LDG_NUM_BLOCKS {num_blocks}",
        label="qwen_LDG_NUM_BLOCKS",
    )
    source = _replace_pattern_once(
        source,
        r"^#define LDG_BLOCK_SIZE \d+$",
        f"#define LDG_BLOCK_SIZE {block_size}",
        label="qwen_LDG_BLOCK_SIZE",
    )
    return source


def _apply_attention_prefetch(source: str) -> str:
    """Use idle decode blocks to prefetch O/Gate weights while attention runs."""
    has_prefetch_signature = (
        "const __nv_bfloat16* __restrict__ o_weight" in source
        and "const __nv_bfloat16* __restrict__ gate_weight" in source
    )
    if has_prefetch_signature:
        if "LDG_PREFETCH_BYTES_PER_IDLE_BLOCK" not in source:
            # Newer baselines already have prefetch signatures but use different constants.
            pattern = (
                r"(__device__ void ldg_attention\([\s\S]*?\)\s*\{\n"
                r"\s*int block_id = blockIdx\.x;\n"
                r"\s*int num_blocks = gridDim\.x;\n"
                r"\s*int warp_id = threadIdx\.x / WARP_SIZE;\n"
                r"\s*int lane_id = threadIdx\.x % WARP_SIZE;\n)"
            )
            replacement = r"\1\n    constexpr int LDG_PREFETCH_BYTES_PER_IDLE_BLOCK = 32768;\n"
            source, count = re.subn(pattern, replacement, source, count=1)
            if count != 1:
                raise ValueError("failed to apply variant transform: attn_prefetch_constant")
        return source

    old_sig = """__device__ void ldg_attention(
    cg::grid_group& grid,
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale
) {"""
    new_sig = """__device__ void ldg_attention(
    cg::grid_group& grid,
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ gate_weight
) {"""
    source = _replace_text_once(source, old_sig, new_sig, "attn_prefetch_signature")

    old_head_setup = """    int heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int head_start = block_id * heads_per_block;
    int head_end = min(head_start + heads_per_block, NUM_Q_HEADS);
"""
    new_head_setup = """    int heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int head_start = block_id * heads_per_block;
    int head_end = min(head_start + heads_per_block, NUM_Q_HEADS);

    // On GA102, idle blocks overlap attention with a small L2 prefetch window.
    const int active_attn_blocks = NUM_Q_HEADS;
    if (block_id >= active_attn_blocks) {
        constexpr int LDG_PREFETCH_BYTES_PER_IDLE_BLOCK = 32768;
        constexpr int LDG_PREFETCH_ELEMS_PER_IDLE_BLOCK =
            LDG_PREFETCH_BYTES_PER_IDLE_BLOCK / sizeof(__nv_bfloat16);

        int idle_id = block_id - active_attn_blocks;
        int idle_blocks = num_blocks - active_attn_blocks;
        if (idle_blocks < 1) {
            idle_blocks = 1;
        }

        int o_total = Q_SIZE * HIDDEN_SIZE;
        int gate_total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
        int total = o_total + gate_total;

        int start = (idle_id * LDG_PREFETCH_ELEMS_PER_IDLE_BLOCK) % total;
        int end = start + LDG_PREFETCH_ELEMS_PER_IDLE_BLOCK;
        if (end > total) {
            end = total;
        }

        float prefetch_sink = 0.0f;
        for (int idx = start + threadIdx.x; idx < end; idx += LDG_BLOCK_SIZE) {
            const __nv_bfloat16* ptr = (idx < o_total)
                ? (o_weight + idx)
                : (gate_weight + (idx - o_total));
            prefetch_sink += __bfloat162float(__ldg(ptr));
        }
        asm volatile(\"\" : : \"f\"(prefetch_sink));
    }
"""
    source = _replace_text_once(source, old_head_setup, new_head_setup, "attn_prefetch_body")

    old_call = """        ldg_attention(
            grid, g_q, layer_k_cache, layer_v_cache, g_attn_out,
            cache_len, max_seq_len, attn_scale
        );"""
    new_call = """        ldg_attention(
            grid, g_q, layer_k_cache, layer_v_cache, g_attn_out,
            cache_len, max_seq_len, attn_scale,
            w.o_proj_weight, w.gate_proj_weight
        );"""
    source = _replace_text_once(source, old_call, new_call, "attn_prefetch_call")
    return source


def _apply_uint4_weights(source: str) -> str:
    """Switch key weight matvec paths from uint2 loads to uint4 loads."""
    source = _replace_text_once(
        source,
        """            // Use vec4 loads with __ldg through uint2
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * g_normalized[k] +
                       __bfloat162float(w_ptr[1]) * g_normalized[k+1] +
                       __bfloat162float(w_ptr[2]) * g_normalized[k+2] +
                       __bfloat162float(w_ptr[3]) * g_normalized[k+3];
            }
""",
        """            // Use vec8 loads with __ldg through uint4
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

                sum += __bfloat162float(w_ptr[0]) * g_normalized[k] +
                       __bfloat162float(w_ptr[1]) * g_normalized[k+1] +
                       __bfloat162float(w_ptr[2]) * g_normalized[k+2] +
                       __bfloat162float(w_ptr[3]) * g_normalized[k+3] +
                       __bfloat162float(w_ptr[4]) * g_normalized[k+4] +
                       __bfloat162float(w_ptr[5]) * g_normalized[k+5] +
                       __bfloat162float(w_ptr[6]) * g_normalized[k+6] +
                       __bfloat162float(w_ptr[7]) * g_normalized[k+7];
            }
""",
        "uint4_matvec_qkv",
    )

    source = _replace_text_once(
        source,
        """            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < Q_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(o_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * attn_out[k] +
                       __bfloat162float(w_ptr[1]) * attn_out[k+1] +
                       __bfloat162float(w_ptr[2]) * attn_out[k+2] +
                       __bfloat162float(w_ptr[3]) * attn_out[k+3];
            }
""",
        """            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < Q_SIZE; k += WARP_SIZE * 8) {
                uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(o_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

                sum += __bfloat162float(w_ptr[0]) * attn_out[k] +
                       __bfloat162float(w_ptr[1]) * attn_out[k+1] +
                       __bfloat162float(w_ptr[2]) * attn_out[k+2] +
                       __bfloat162float(w_ptr[3]) * attn_out[k+3] +
                       __bfloat162float(w_ptr[4]) * attn_out[k+4] +
                       __bfloat162float(w_ptr[5]) * attn_out[k+5] +
                       __bfloat162float(w_ptr[6]) * attn_out[k+6] +
                       __bfloat162float(w_ptr[7]) * attn_out[k+7];
            }
""",
        "uint4_o_proj",
    )

    source = _replace_text_once(
        source,
        """            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 g_u2 = __ldg(reinterpret_cast<const uint2*>(gate_row + k));
                uint2 u_u2 = __ldg(reinterpret_cast<const uint2*>(up_row + k));
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u2);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u2);

                gate_sum += __bfloat162float(g_ptr[0]) * g_activations[k] +
                            __bfloat162float(g_ptr[1]) * g_activations[k+1] +
                            __bfloat162float(g_ptr[2]) * g_activations[k+2] +
                            __bfloat162float(g_ptr[3]) * g_activations[k+3];

                up_sum += __bfloat162float(u_ptr[0]) * g_activations[k] +
                          __bfloat162float(u_ptr[1]) * g_activations[k+1] +
                          __bfloat162float(u_ptr[2]) * g_activations[k+2] +
                          __bfloat162float(u_ptr[3]) * g_activations[k+3];
            }
""",
        """            #pragma unroll 4
            for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                uint4 g_u4 = __ldg(reinterpret_cast<const uint4*>(gate_row + k));
                uint4 u_u4 = __ldg(reinterpret_cast<const uint4*>(up_row + k));
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u4);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u4);

                gate_sum += __bfloat162float(g_ptr[0]) * g_activations[k] +
                            __bfloat162float(g_ptr[1]) * g_activations[k+1] +
                            __bfloat162float(g_ptr[2]) * g_activations[k+2] +
                            __bfloat162float(g_ptr[3]) * g_activations[k+3] +
                            __bfloat162float(g_ptr[4]) * g_activations[k+4] +
                            __bfloat162float(g_ptr[5]) * g_activations[k+5] +
                            __bfloat162float(g_ptr[6]) * g_activations[k+6] +
                            __bfloat162float(g_ptr[7]) * g_activations[k+7];

                up_sum += __bfloat162float(u_ptr[0]) * g_activations[k] +
                          __bfloat162float(u_ptr[1]) * g_activations[k+1] +
                          __bfloat162float(u_ptr[2]) * g_activations[k+2] +
                          __bfloat162float(u_ptr[3]) * g_activations[k+3] +
                          __bfloat162float(u_ptr[4]) * g_activations[k+4] +
                          __bfloat162float(u_ptr[5]) * g_activations[k+5] +
                          __bfloat162float(u_ptr[6]) * g_activations[k+6] +
                          __bfloat162float(u_ptr[7]) * g_activations[k+7];
            }
""",
        "uint4_gate_up",
    )

    source = _replace_text_once(
        source,
        """            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 4) {
                uint2 d_u2 = __ldg(reinterpret_cast<const uint2*>(down_row + k));
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u2);

                sum += __bfloat162float(d_ptr[0]) * g_mlp_intermediate[k] +
                       __bfloat162float(d_ptr[1]) * g_mlp_intermediate[k+1] +
                       __bfloat162float(d_ptr[2]) * g_mlp_intermediate[k+2] +
                       __bfloat162float(d_ptr[3]) * g_mlp_intermediate[k+3];
            }
""",
        """            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {
                uint4 d_u4 = __ldg(reinterpret_cast<const uint4*>(down_row + k));
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u4);

                sum += __bfloat162float(d_ptr[0]) * g_mlp_intermediate[k] +
                       __bfloat162float(d_ptr[1]) * g_mlp_intermediate[k+1] +
                       __bfloat162float(d_ptr[2]) * g_mlp_intermediate[k+2] +
                       __bfloat162float(d_ptr[3]) * g_mlp_intermediate[k+3] +
                       __bfloat162float(d_ptr[4]) * g_mlp_intermediate[k+4] +
                       __bfloat162float(d_ptr[5]) * g_mlp_intermediate[k+5] +
                       __bfloat162float(d_ptr[6]) * g_mlp_intermediate[k+6] +
                       __bfloat162float(d_ptr[7]) * g_mlp_intermediate[k+7];
            }
""",
        "uint4_down_proj",
    )

    source = _replace_text_all(
        source,
        """        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k+1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k+2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k+3];
        }
""",
        """        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k+1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k+2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k+3] +
                   __bfloat162float(w_ptr[4]) * s_hidden[k+4] +
                   __bfloat162float(w_ptr[5]) * s_hidden[k+5] +
                   __bfloat162float(w_ptr[6]) * s_hidden[k+6] +
                   __bfloat162float(w_ptr[7]) * s_hidden[k+7];
        }
""",
        "uint4_lm_head",
    )
    return source


def load_base_source() -> str:
    """Load the baseline decode kernel source."""
    return BASE_KERNEL_PATH.read_text()


def load_qwen_persistent_source() -> str:
    """Load qwen-style persistent decode kernel source."""
    if not QWEN_PERSISTENT_KERNEL_PATH.exists():
        raise FileNotFoundError(
            f"missing experimental kernel source: {QWEN_PERSISTENT_KERNEL_PATH}"
        )
    return QWEN_PERSISTENT_KERNEL_PATH.read_text()


def list_variants() -> tuple[str, ...]:
    """Return supported variant names."""
    return (
        "baseline",
        "fastmath",
        "prefetch",
        "uint4",
        "uint4_prefetch",
        "blocks128",
        "blocks128_fastmath",
        "blocks96",
        "blocks112",
        "qwen_persistent",
        "qwen_persistent_tuned",
    )


def build_variant_source(variant: str) -> str:
    """Build a kernel source string for a named variant."""
    src = load_base_source()

    if variant == "baseline":
        return src
    if variant == "fastmath":
        return _apply_fast_math(src)
    if variant == "prefetch":
        return _apply_attention_prefetch(src)
    if variant == "uint4":
        return _apply_uint4_weights(src)
    if variant == "uint4_prefetch":
        return _apply_attention_prefetch(_apply_uint4_weights(src))
    if variant == "blocks128":
        return _apply_num_blocks(src, 128)
    if variant == "blocks128_fastmath":
        return _apply_fast_math(_apply_num_blocks(src, 128))
    if variant == "blocks96":
        return _apply_num_blocks(src, 96)
    if variant == "blocks112":
        return _apply_num_blocks(src, 112)
    if variant == "qwen_persistent":
        return load_qwen_persistent_source()
    if variant == "qwen_persistent_tuned":
        return _apply_qwen_macro_tuning(load_qwen_persistent_source(), 112, 256)

    options = ", ".join(list_variants())
    raise ValueError(f"unknown variant '{variant}', expected one of: {options}")
