#include "llama.cuh"

#include "rms_matvec_rope_append.cu"
#include "attention_partial.cu"
#include "attention_partial_direct.cu"
#include "attention_reduction.cu"
#include "matvec_adds.cu"
#include "upgate.cu"
#include "upgate_direct_rms.cu"
#include "upgate_direct_ldg.cu"
#include "upgate_direct_store.cu"
#include "upgate_overlap_up_reduce.cu"
#include "upgate_overlap_up_fixed4.cu"
#include "rms_lm_head.cu"
#include "opcode5_direct.cuh"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace megakernel;

using rms_qkv_rope_append_op =
    rms_qkv_rope_append<default_config, llama_1b_globals>;
using attention_partial_op =
    attention_partial<default_config, llama_1b_globals>;
using attention_reduction_op =
    attention_reduction<default_config, llama_1b_globals>;
using o_proj_op = o_proj<default_config, llama_1b_globals>;
using rms_upgate_silu_op = rms_upgate_silu<default_config, llama_1b_globals>;
using downproj_op = downproj<default_config, llama_1b_globals>;
using rms_lm_head_op = rms_lm_head<default_config, llama_1b_globals>;

struct opcode5_direct_config : default_config {
    static constexpr bool SINGLE_OP_DIRECT = true;
    static constexpr bool DIRECT_INDEX_BY_BLOCK = false;
    static constexpr int DIRECT_ACTIVE_BLOCKS = 0;
};
struct opcode5_timed_config : opcode5_direct_config {
    static constexpr bool TIMING_RECORD_ENABLED = true;
};

// Intentionally incorrect direct-TK mechanism ablations.  Four physical
// pages are cyclically aliased by the matvec stages.  The r96 control removes
// the SMEM admission blocker but retains the original register footprint;
// r48 also asks ptxas for two resident CTAs and gives the consumer/nonconsumer
// warpgroups 48/32 registers per thread.
struct direct_pdl_4page_r96_config : opcode5_direct_config {
    static constexpr int NUM_PAGES = 4;
    static constexpr int DYNAMIC_SHARED_MEMORY = NUM_PAGES * PAGE_SIZE + 1024;
    static constexpr bool DIRECT_PDL_WAIT = true;
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_LOAD = true;
    // Historical four-page arrival control.  Keep this explicit now that the
    // base configuration truthfully defaults all direct-PDL controls off.
    static constexpr bool DIRECT_PDL_TRIGGER_WAIT_FOR_LOAD_ARRIVAL = true;
    static constexpr bool DIRECT_SKIP_GLOBAL_COUNTERS = true;
    static constexpr bool DIRECT_ALIAS_FOUR_PAGES = true;
};
struct direct_pdl_4page_r48_config : direct_pdl_4page_r96_config {
    static constexpr int DIRECT_MIN_BLOCKS_PER_SM = 2;
    static constexpr int CONSUMER_REGISTERS = 48;
    static constexpr int NON_CONSUMER_REGISTERS = 32;
};
struct direct_pdl_4page_issue_r96_config : direct_pdl_4page_r96_config {
    static constexpr bool DIRECT_PDL_TRIGGER_WAIT_FOR_LOAD_ARRIVAL = false;
};
struct direct_pdl_4page_issue_r48_config : direct_pdl_4page_r48_config {
    static constexpr bool DIRECT_PDL_TRIGGER_WAIT_FOR_LOAD_ARRIVAL = false;
};
struct direct_pdl_4page_issue_alias_config
    : direct_pdl_4page_issue_r96_config {
    static constexpr bool MATVEC_ALIAS_OPERANDS = true;
};
struct direct_pdl_4page_issue_chunk16_config
    : direct_pdl_4page_issue_r96_config {
    static constexpr int MATVEC_CHUNK_COLS = 16;
};
struct direct_pdl_4page_issue_4cw_config
    : direct_pdl_4page_issue_r96_config {
    static constexpr int NUM_CONSUMER_WARPS = 4;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int DIRECT_MIN_BLOCKS_PER_SM = 2;
};
struct direct_pdl_4page_issue_4cw_alias_natural_config
    : direct_pdl_4page_issue_4cw_config {
    static constexpr bool MATVEC_ALIAS_OPERANDS = true;
    // Do not ask launch_bounds for two blocks. Measure the natural compiler
    // allocation after shortening the operand-tile live range.
    static constexpr int DIRECT_MIN_BLOCKS_PER_SM = 1;
};

// Intentionally incorrect register/admission ablation. The loader still
// issues all four 16x512 BF16 TMA chunks for every iteration, but only the
// first ActiveConsumerWarps of the original logical 16 K partitions are
// consumed. Unlike issue_4cw, work is not redistributed to the remaining
// consumers. Do not use launch_bounds to force a two-CTA register ceiling.
template <int ActiveConsumerWarps>
struct direct_pdl_4page_issue_partial_config
    : direct_pdl_4page_issue_r96_config {
    static_assert(ActiveConsumerWarps == 4 || ActiveConsumerWarps == 8 ||
                  ActiveConsumerWarps == 12);
    static constexpr int NUM_CONSUMER_WARPS = ActiveConsumerWarps;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int DIRECT_LOGICAL_CONSUMER_WARPS = 16;
    static constexpr int DIRECT_MIN_BLOCKS_PER_SM = 1;
};
using direct_pdl_4page_issue_partial12cw_config =
    direct_pdl_4page_issue_partial_config<12>;
using direct_pdl_4page_issue_partial8cw_config =
    direct_pdl_4page_issue_partial_config<8>;
using direct_pdl_4page_issue_partial4cw_config =
    direct_pdl_4page_issue_partial_config<4>;
struct direct_pdl_4page_issue_partial4cw_regpad_config
    : direct_pdl_4page_issue_partial4cw_config {
    static constexpr int DIRECT_REGISTER_PADDING = 64;
};

// Correct full-resource control: preserve the original three-stage, 13-page
// TK pipeline and change only the cross-kernel dependency protocol.  Its
// SMEM/register footprint intentionally prevents same-SM CTA co-residency.
struct direct_pdl_13page_issue_r96_config : opcode5_direct_config {
    static constexpr bool DIRECT_PDL_WAIT = true;
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_LOAD = true;
    static constexpr bool DIRECT_PDL_TRIGGER_WAIT_FOR_LOAD_ARRIVAL = false;
    static constexpr bool DIRECT_SKIP_GLOBAL_COUNTERS = true;
};
// Matched trigger-timing control: identical 13-page/three-stage resources,
// but delay the PDL trigger until the final weight TMA has arrived rather
// than merely been issued.
struct direct_pdl_13page_arrival_r96_config
    : direct_pdl_13page_issue_r96_config {
    static constexpr bool DIRECT_PDL_TRIGGER_WAIT_FOR_LOAD_ARRIVAL = true;
};
// Fine-grained virtual-TP opcode5->opcode6 edge. PDL still admits opcode6
// early, while opcode5 publishes the original four SiLU shard counters and
// opcode6 waits only for its reduction shard instead of the whole grid.
struct direct_pdl_13page_issue_r96_vtp_signal_config
    : direct_pdl_13page_issue_r96_config {
    static constexpr bool DIRECT_SKIP_GLOBAL_COUNTERS = false;
};
struct direct_pdl_13page_issue_r96_vtp_wait_config
    : direct_pdl_13page_issue_r96_config {
    static constexpr bool DIRECT_PDL_WAIT = false;
};
struct direct_pdl_13page_issue_r96_trace_config
    : direct_pdl_13page_issue_r96_config {
    static constexpr bool DIRECT_GLOBAL_TRACE_ENABLED = true;
};
// Compact-grid opcode 5 has exactly 128 nonempty instructions under the
// accepted one-wave schedule.  Index by block rather than SM because CUDA may
// place the 128 CTAs on any subset of the 132 H100 SMs.
struct direct_pdl_13page_issue_r96_compact_op5_config
    : direct_pdl_13page_issue_r96_config {
    static constexpr bool DIRECT_INDEX_BY_BLOCK = true;
    static constexpr int DIRECT_ACTIVE_BLOCKS =
        LLAMA_1B_INTERMEDIATE_DIM / LLAMA_1B_HEAD_DIM;
};

// Correct one-stage variants: page 0 holds activation/RMS data and pages 1-4
// hold one complete 16x2048 BF16 weight tile.  Unlike the four-page ceiling
// experiment these entries do not alias any data pages.
struct direct_pdl_s1_issue_r96_config : opcode5_direct_config {
    static constexpr int MATVEC_INPUT_PIPELINE_STAGES = 1;
    static constexpr int NUM_PAGES = 5;
    static constexpr int DYNAMIC_SHARED_MEMORY = NUM_PAGES * PAGE_SIZE + 1024;
    static constexpr bool DIRECT_PDL_WAIT = true;
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_LOAD = true;
    static constexpr bool DIRECT_PDL_TRIGGER_WAIT_FOR_LOAD_ARRIVAL = false;
    static constexpr bool DIRECT_SKIP_GLOBAL_COUNTERS = true;
};
struct direct_pdl_s1_issue_r96_trace_config
    : direct_pdl_s1_issue_r96_config {
    static constexpr bool DIRECT_GLOBAL_TRACE_ENABLED = true;
};
struct direct_pdl_s1_issue_r48_config : direct_pdl_s1_issue_r96_config {
    static constexpr int DIRECT_MIN_BLOCKS_PER_SM = 2;
    static constexpr int CONSUMER_REGISTERS = 48;
    static constexpr int NON_CONSUMER_REGISTERS = 32;
};
struct direct_pdl_s1_issue_4cw_config : direct_pdl_s1_issue_r96_config {
    static constexpr int NUM_CONSUMER_WARPS = 4;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int DIRECT_MIN_BLOCKS_PER_SM = 2;
};
template <int MicrotileCols, int ConsumerWarps>
struct direct_pdl_s1_issue_micro_config
    : direct_pdl_s1_issue_r96_config {
    static_assert(MicrotileCols == 16 || MicrotileCols == 32 ||
                  MicrotileCols == 64 || MicrotileCols == 128);
    static_assert(ConsumerWarps == 4 || ConsumerWarps == 8 ||
                  ConsumerWarps == 16);
    static constexpr int NUM_CONSUMER_WARPS = ConsumerWarps;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS =
        NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int MATVEC_SMEM_MICROTILE_COLS = MicrotileCols;
    // First inspect the natural allocation. A separate candidate can ask
    // ptxas for two blocks only if the natural result misses the threshold.
    static constexpr int DIRECT_MIN_BLOCKS_PER_SM = 1;
};
using direct_pdl_s1_issue_micro16_16cw_config =
    direct_pdl_s1_issue_micro_config<16, 16>;
using direct_pdl_s1_issue_micro16_8cw_config =
    direct_pdl_s1_issue_micro_config<16, 8>;
using direct_pdl_s1_issue_micro16_4cw_config =
    direct_pdl_s1_issue_micro_config<16, 4>;
using direct_pdl_s1_issue_micro32_16cw_config =
    direct_pdl_s1_issue_micro_config<32, 16>;
using direct_pdl_s1_issue_micro32_8cw_config =
    direct_pdl_s1_issue_micro_config<32, 8>;
using direct_pdl_s1_issue_micro64_16cw_config =
    direct_pdl_s1_issue_micro_config<64, 16>;
using direct_pdl_s1_issue_micro64_8cw_config =
    direct_pdl_s1_issue_micro_config<64, 8>;
using direct_pdl_s1_issue_micro128_16cw_config =
    direct_pdl_s1_issue_micro_config<128, 16>;
using direct_pdl_s1_issue_micro128_8cw_config =
    direct_pdl_s1_issue_micro_config<128, 8>;
struct direct_pdl_s1_issue_micro16_16cw_trace_config
    : direct_pdl_s1_issue_micro16_16cw_config {
    static constexpr bool DIRECT_GLOBAL_TRACE_ENABLED = true;
};
// Full-path resource candidate. Matvec opcodes retain all 16 logical
// reduction warps and use the correct micro16 compute path. Attention keeps
// its original math, but needs only four consumer warps because only consumer
// warp zero executes the GQA body.
struct direct_pdl_s1_issue_micro16_all_matvec_config
    : direct_pdl_s1_issue_micro16_16cw_config {
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_LOAD = false;
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_STORE = true;
};
struct direct_pdl_s1_issue_micro16_all_attention_config
    : direct_pdl_s1_issue_micro16_4cw_config {
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_LOAD = false;
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_STORE = true;
};
template <typename Base> struct opcode2_direct_pdl_config : Base {
    static constexpr bool DIRECT_INDEX_BY_BLOCK = true;
    static constexpr int DIRECT_ACTIVE_BLOCKS = LLAMA_1B_NUM_KV_HEADS;
};
template <typename Base> struct opcode4_direct_pdl_config : Base {
    static constexpr bool DIRECT_INDEX_BY_BLOCK = true;
    static constexpr int DIRECT_ACTIVE_BLOCKS =
        LLAMA_1B_HIDDEN_DIM / LLAMA_1B_MATVEC_BLOCK_SIZE;
};
template <bool RegisterBufferedStage>
struct opcode5_s1_config : opcode5_direct_config {
    static constexpr int MATVEC_INPUT_PIPELINE_STAGES = 1;
    static constexpr bool MATVEC_REGISTER_BUFFERED_STAGE =
        RegisterBufferedStage;
    static constexpr int NUM_PAGES = 1 + 4 * MATVEC_INPUT_PIPELINE_STAGES;
    // The direct wrapper aligns the dynamic allocation upward to 1 KiB.
    static constexpr int DYNAMIC_SHARED_MEMORY =
        NUM_PAGES * PAGE_SIZE + 1024;
};
using opcode5_s1_late_config = opcode5_s1_config<false>;
using opcode5_s1_reg_config = opcode5_s1_config<true>;
struct opcode2_direct_config : opcode5_direct_config {
    static constexpr bool DIRECT_INDEX_BY_BLOCK = true;
    static constexpr int DIRECT_ACTIVE_BLOCKS = LLAMA_1B_NUM_KV_HEADS;
};
struct opcode2_timed_config : opcode2_direct_config {
    static constexpr bool TIMING_RECORD_ENABLED = true;
};
struct opcode3_direct_config : opcode5_direct_config {
    static constexpr bool DIRECT_INDEX_BY_BLOCK = true;
    static constexpr int DIRECT_ACTIVE_BLOCKS =
        LLAMA_1B_NUM_ATTENTION_HEADS / 4;
};
struct opcode3_timed_config : opcode3_direct_config {
    static constexpr bool TIMING_RECORD_ENABLED = true;
};
struct opcode3_vm_timed_config : default_config {
    static constexpr bool TIMING_RECORD_ENABLED = true;
};
struct opcode4_direct_config : opcode5_direct_config {
    static constexpr bool DIRECT_INDEX_BY_BLOCK = true;
    static constexpr int DIRECT_ACTIVE_BLOCKS =
        LLAMA_1B_HIDDEN_DIM / LLAMA_1B_MATVEC_BLOCK_SIZE;
};
struct opcode4_timed_config : opcode4_direct_config {
    static constexpr bool TIMING_RECORD_ENABLED = true;
};
template <bool RegisterBufferedStage>
struct opcode4_s1_config : opcode5_s1_config<RegisterBufferedStage> {
    static constexpr bool DIRECT_INDEX_BY_BLOCK = true;
    static constexpr int DIRECT_ACTIVE_BLOCKS =
        LLAMA_1B_HIDDEN_DIM / LLAMA_1B_MATVEC_BLOCK_SIZE;
};
using opcode4_s1_late_config = opcode4_s1_config<false>;
using opcode4_s1_reg_config = opcode4_s1_config<true>;
struct opcode7_direct_config : opcode5_direct_config {
    static constexpr bool DIRECT_INDEX_BY_BLOCK = true;
    static constexpr int DIRECT_ACTIVE_BLOCKS = H100_SM_COUNT;
};
struct opcode7_timed_config : opcode7_direct_config {
    static constexpr bool TIMING_RECORD_ENABLED = true;
};
struct opcode7_direct_pdl_wait_config : opcode7_direct_config {
    static constexpr bool DIRECT_PDL_WAIT = true;
};
struct opcode7_direct_pdl_wait_trace_config
    : opcode7_direct_pdl_wait_config {
    static constexpr bool DIRECT_GLOBAL_TRACE_ENABLED = true;
};
struct opcode7_direct_pdl_s1_micro16_all_config
    : direct_pdl_s1_issue_micro16_all_matvec_config {
    static constexpr bool DIRECT_INDEX_BY_BLOCK = true;
    static constexpr int DIRECT_ACTIVE_BLOCKS = H100_SM_COUNT;
    // Opcode 7 consumes the opcode6 PDL edge, but argmax remains an ordinary
    // completion successor and does not need a programmatic trigger.
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_LOAD = false;
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_STORE = false;
};
using rms_upgate_silu_direct_op =
    rms_upgate_silu<opcode5_direct_config, llama_1b_globals>;
using rms_upgate_silu_timed_op =
    rms_upgate_silu<opcode5_timed_config, llama_1b_globals>;
using rms_upgate_silu_s1_late_op =
    rms_upgate_silu<opcode5_s1_late_config, llama_1b_globals>;
using rms_upgate_silu_s1_reg_op =
    rms_upgate_silu<opcode5_s1_reg_config, llama_1b_globals>;
using rms_upgate_silu_direct_ldg_op =
    rms_upgate_silu_direct_ldg<opcode5_direct_config, llama_1b_globals>;
using rms_upgate_silu_direct_ldg_timed_op =
    rms_upgate_silu_direct_ldg<opcode5_timed_config, llama_1b_globals>;
using rms_upgate_silu_direct_rms_op =
    rms_upgate_silu_direct_rms<opcode5_direct_config, llama_1b_globals>;
using rms_upgate_silu_direct_rms_timed_op =
    rms_upgate_silu_direct_rms<opcode5_timed_config, llama_1b_globals>;
using rms_upgate_silu_direct_store_op =
    rms_upgate_silu_direct_store<opcode5_direct_config, llama_1b_globals>;
using rms_upgate_silu_direct_store_timed_op =
    rms_upgate_silu_direct_store<opcode5_timed_config, llama_1b_globals>;
using rms_upgate_silu_overlap_up_reduce_op =
    rms_upgate_silu_overlap_up_reduce<opcode5_direct_config,
                                      llama_1b_globals>;
using rms_upgate_silu_overlap_up_reduce_timed_op =
    rms_upgate_silu_overlap_up_reduce<opcode5_timed_config,
                                      llama_1b_globals>;
using rms_upgate_silu_overlap_up_no_atomic_op =
    rms_upgate_silu_overlap_up_reduce<opcode5_direct_config,
                                      llama_1b_globals, false>;
using rms_upgate_silu_overlap_up_no_atomic_timed_op =
    rms_upgate_silu_overlap_up_reduce<opcode5_timed_config,
                                      llama_1b_globals, false>;
using rms_upgate_silu_overlap_up_fixed4_op =
    rms_upgate_silu_overlap_up_fixed4<opcode5_direct_config,
                                      llama_1b_globals>;
using rms_upgate_silu_overlap_up_fixed4_timed_op =
    rms_upgate_silu_overlap_up_fixed4<opcode5_timed_config,
                                      llama_1b_globals>;
using rms_upgate_silu_overlap_up_fixed4_no_atomic_op =
    rms_upgate_silu_overlap_up_fixed4<opcode5_direct_config,
                                      llama_1b_globals, false>;
using rms_upgate_silu_overlap_up_fixed4_no_atomic_timed_op =
    rms_upgate_silu_overlap_up_fixed4<opcode5_timed_config,
                                      llama_1b_globals, false>;
using rms_upgate_silu_overlap_up_fixed4_tc_no_atomic_op =
    rms_upgate_silu_overlap_up_fixed4<opcode5_direct_config,
                                      llama_1b_globals, false, true>;
using rms_upgate_silu_overlap_up_fixed4_tc_no_atomic_timed_op =
    rms_upgate_silu_overlap_up_fixed4<opcode5_timed_config,
                                      llama_1b_globals, false, true>;
using downproj_direct_op =
    downproj<opcode5_direct_config, llama_1b_globals>;
using downproj_direct_timed_op =
    downproj<opcode5_timed_config, llama_1b_globals>;
using downproj_s1_late_op =
    downproj<opcode5_s1_late_config, llama_1b_globals>;
using downproj_s1_reg_op =
    downproj<opcode5_s1_reg_config, llama_1b_globals>;
using qkv_direct_op =
    rms_qkv_rope_append<opcode5_direct_config, llama_1b_globals>;
using qkv_direct_timed_op =
    rms_qkv_rope_append<opcode5_timed_config, llama_1b_globals>;
using attention_partial_direct_op =
    attention_partial_direct<opcode2_direct_config, llama_1b_globals>;
using attention_partial_direct_timed_op =
    attention_partial_direct<opcode2_timed_config, llama_1b_globals>;
using attention_reduction_direct_op =
    attention_reduction<opcode3_direct_config, llama_1b_globals>;
using attention_reduction_direct_timed_op =
    attention_reduction<opcode3_timed_config, llama_1b_globals>;
using attention_reduction_vm_timed_op =
    attention_reduction<opcode3_vm_timed_config, llama_1b_globals>;
using o_proj_direct_op =
    o_proj<opcode4_direct_config, llama_1b_globals>;
using o_proj_direct_timed_op =
    o_proj<opcode4_timed_config, llama_1b_globals>;
using o_proj_s1_late_op =
    o_proj<opcode4_s1_late_config, llama_1b_globals>;
using o_proj_s1_reg_op =
    o_proj<opcode4_s1_reg_config, llama_1b_globals>;
using lm_head_direct_op =
    rms_lm_head<opcode7_direct_config, llama_1b_globals>;
using lm_head_direct_pdl_wait_op =
    rms_lm_head<opcode7_direct_pdl_wait_config, llama_1b_globals>;
using lm_head_direct_pdl_wait_trace_op =
    rms_lm_head<opcode7_direct_pdl_wait_trace_config, llama_1b_globals>;
using lm_head_direct_timed_op =
    rms_lm_head<opcode7_timed_config, llama_1b_globals>;

#define DEFINE_DIRECT_PDL_OPS(suffix, base_config)                            \
    using opcode1_direct_pdl_##suffix##_op =                                  \
        rms_qkv_rope_append<base_config, llama_1b_globals>;                   \
    using opcode2_direct_pdl_##suffix##_config =                              \
        opcode2_direct_pdl_config<base_config>;                               \
    using opcode2_direct_pdl_##suffix##_op =                                  \
        attention_partial_direct<opcode2_direct_pdl_##suffix##_config,        \
                                 llama_1b_globals>;                           \
    using opcode4_direct_pdl_##suffix##_config =                              \
        opcode4_direct_pdl_config<base_config>;                               \
    using opcode4_direct_pdl_##suffix##_op =                                  \
        o_proj<opcode4_direct_pdl_##suffix##_config, llama_1b_globals>;        \
    using opcode5_direct_pdl_##suffix##_op =                                  \
        rms_upgate_silu<base_config, llama_1b_globals>;                       \
    using opcode6_direct_pdl_##suffix##_op =                                  \
        downproj<base_config, llama_1b_globals>

DEFINE_DIRECT_PDL_OPS(r96, direct_pdl_4page_r96_config);
DEFINE_DIRECT_PDL_OPS(r48, direct_pdl_4page_r48_config);
DEFINE_DIRECT_PDL_OPS(issue_r96, direct_pdl_4page_issue_r96_config);
DEFINE_DIRECT_PDL_OPS(issue_r48, direct_pdl_4page_issue_r48_config);
DEFINE_DIRECT_PDL_OPS(issue_alias, direct_pdl_4page_issue_alias_config);
DEFINE_DIRECT_PDL_OPS(issue_chunk16,
                      direct_pdl_4page_issue_chunk16_config);
DEFINE_DIRECT_PDL_OPS(issue_4cw, direct_pdl_4page_issue_4cw_config);
DEFINE_DIRECT_PDL_OPS(issue_4cw_alias_natural,
                      direct_pdl_4page_issue_4cw_alias_natural_config);
DEFINE_DIRECT_PDL_OPS(issue_partial12cw,
                      direct_pdl_4page_issue_partial12cw_config);
DEFINE_DIRECT_PDL_OPS(issue_partial8cw,
                      direct_pdl_4page_issue_partial8cw_config);
DEFINE_DIRECT_PDL_OPS(issue_partial4cw,
                      direct_pdl_4page_issue_partial4cw_config);
DEFINE_DIRECT_PDL_OPS(issue_partial4cw_regpad,
                      direct_pdl_4page_issue_partial4cw_regpad_config);
DEFINE_DIRECT_PDL_OPS(13page_issue_r96,
                      direct_pdl_13page_issue_r96_config);
DEFINE_DIRECT_PDL_OPS(13page_arrival_r96,
                      direct_pdl_13page_arrival_r96_config);
DEFINE_DIRECT_PDL_OPS(13page_issue_r96_trace,
                      direct_pdl_13page_issue_r96_trace_config);
using opcode5_direct_pdl_13page_issue_r96_vtp_signal_op =
    rms_upgate_silu<direct_pdl_13page_issue_r96_vtp_signal_config,
                    llama_1b_globals>;
using opcode6_direct_pdl_13page_issue_r96_vtp_wait_op =
    downproj<direct_pdl_13page_issue_r96_vtp_wait_config,
             llama_1b_globals>;
using opcode5_direct_pdl_13page_issue_r96_compact_op =
    rms_upgate_silu<direct_pdl_13page_issue_r96_compact_op5_config,
                    llama_1b_globals>;
DEFINE_DIRECT_PDL_OPS(s1_issue_r96, direct_pdl_s1_issue_r96_config);
DEFINE_DIRECT_PDL_OPS(s1_issue_r96_trace,
                      direct_pdl_s1_issue_r96_trace_config);
DEFINE_DIRECT_PDL_OPS(s1_issue_r48, direct_pdl_s1_issue_r48_config);
DEFINE_DIRECT_PDL_OPS(s1_issue_4cw, direct_pdl_s1_issue_4cw_config);

// Single-op feasibility screen: instantiate the register microtile only for
// opcode 4. Do not change QKV, attention, up/gate, or down projection until
// this simplest GEMV demonstrates a single-kernel win.
#define DEFINE_DIRECT_PDL_MICRO_OP4(suffix, base_config)                      \
    using opcode4_direct_pdl_##suffix##_config =                              \
        opcode4_direct_pdl_config<base_config>;                               \
    using opcode4_direct_pdl_##suffix##_op =                                  \
        o_proj<opcode4_direct_pdl_##suffix##_config, llama_1b_globals>

DEFINE_DIRECT_PDL_MICRO_OP4(
    s1_micro16_16cw, direct_pdl_s1_issue_micro16_16cw_config);
DEFINE_DIRECT_PDL_MICRO_OP4(
    s1_micro16_16cw_trace,
    direct_pdl_s1_issue_micro16_16cw_trace_config);
DEFINE_DIRECT_PDL_MICRO_OP4(
    s1_micro16_8cw, direct_pdl_s1_issue_micro16_8cw_config);
DEFINE_DIRECT_PDL_MICRO_OP4(
    s1_micro16_4cw, direct_pdl_s1_issue_micro16_4cw_config);
DEFINE_DIRECT_PDL_MICRO_OP4(
    s1_micro32_16cw, direct_pdl_s1_issue_micro32_16cw_config);
DEFINE_DIRECT_PDL_MICRO_OP4(
    s1_micro32_8cw, direct_pdl_s1_issue_micro32_8cw_config);
DEFINE_DIRECT_PDL_MICRO_OP4(
    s1_micro64_16cw, direct_pdl_s1_issue_micro64_16cw_config);
DEFINE_DIRECT_PDL_MICRO_OP4(
    s1_micro64_8cw, direct_pdl_s1_issue_micro64_8cw_config);
DEFINE_DIRECT_PDL_MICRO_OP4(
    s1_micro128_16cw, direct_pdl_s1_issue_micro128_16cw_config);
DEFINE_DIRECT_PDL_MICRO_OP4(
    s1_micro128_8cw, direct_pdl_s1_issue_micro128_8cw_config);
#undef DEFINE_DIRECT_PDL_MICRO_OP4

// Complete-decode micro candidate. Unlike the preceding opcode-4 feasibility
// entries, this set covers every opcode active in P32/D128. Opcode 3 is absent
// from that schedule.
using opcode1_direct_pdl_s1_micro16_all_op =
    rms_qkv_rope_append<
        direct_pdl_s1_issue_micro16_all_matvec_config, llama_1b_globals>;
struct opcode2_direct_pdl_s1_micro16_all_config
    : opcode2_direct_pdl_config<
          direct_pdl_s1_issue_micro16_all_attention_config> {};
using opcode2_direct_pdl_s1_micro16_all_op =
    attention_partial_direct<
        opcode2_direct_pdl_s1_micro16_all_config, llama_1b_globals>;
struct opcode4_direct_pdl_s1_micro16_all_config
    : opcode4_direct_pdl_config<
          direct_pdl_s1_issue_micro16_all_matvec_config> {};
using opcode4_direct_pdl_s1_micro16_all_op =
    o_proj<opcode4_direct_pdl_s1_micro16_all_config, llama_1b_globals>;
using opcode5_direct_pdl_s1_micro16_all_op =
    rms_upgate_silu<
        direct_pdl_s1_issue_micro16_all_matvec_config, llama_1b_globals>;
using opcode6_direct_pdl_s1_micro16_all_op =
    downproj<
        direct_pdl_s1_issue_micro16_all_matvec_config, llama_1b_globals>;
using opcode7_direct_pdl_s1_micro16_all_op =
    rms_lm_head<
        opcode7_direct_pdl_s1_micro16_all_config, llama_1b_globals>;

#undef DEFINE_DIRECT_PDL_OPS

#define LLAMA_GLOBAL_MEMBER_POINTERS                                          \
        &llama_1b_globals::Bar, &llama_1b_globals::instructions,              \
        &llama_1b_globals::timings, &llama_1b_globals::qkv_weights,           \
        &llama_1b_globals::attn_norm_weights, &llama_1b_globals::o_weights,   \
        &llama_1b_globals::mlp_norm_weights, &llama_1b_globals::up_weights,   \
        &llama_1b_globals::gate_weights, &llama_1b_globals::down_weights,     \
        &llama_1b_globals::lm_head_norm_weights,                              \
        &llama_1b_globals::lm_head_weights, &llama_1b_globals::k_cache,       \
        &llama_1b_globals::v_cache, &llama_1b_globals::rope_cos,              \
        &llama_1b_globals::rope_sin, &llama_1b_globals::hidden_states,        \
        &llama_1b_globals::q_post_rope, &llama_1b_globals::attn_out,          \
        &llama_1b_globals::attn_lse_intermediates,                            \
        &llama_1b_globals::attn_out_intermediates,                            \
        &llama_1b_globals::silu_out, &llama_1b_globals::logits,               \
        &llama_1b_globals::pos_id, &llama_1b_globals::attn_scale,             \
        &llama_1b_globals::rms_norm_eps,                                      \
        &llama_1b_globals::skip_attn_reduction

#define BIND_LLAMA_GLOBALS(kernel, name)                                      \
    kittens::py::bind_kernel<kernel>(m, name, LLAMA_GLOBAL_MEMBER_POINTERS)

// Keep the dynamic-SMEM launcher local to this demo.  The experiment trees
// originally patched ThunderKittens' generic pyutils header, which made the
// superproject depend on an unpublished submodule working-tree change.
template <auto Kernel, int DynamicSharedMemory, int GridBlocks = 0,
          int Threads = 0, typename TGlobal>
static void bind_kernel_dynamic_smem(
    auto m, auto name, auto TGlobal::*...member_ptrs) {
    m.def(name,
          [](kittens::py::object<decltype(member_ptrs)>...args,
             pybind11::kwargs kwargs) {
              TGlobal g{
                  kittens::py::from_object<typename kittens::py::trait<
                      decltype(member_ptrs)>::member_type>::make(args)...};
              cudaStream_t stream = nullptr;
              if (kwargs.contains("stream")) {
                  uintptr_t stream_ptr =
                      kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
                  stream = reinterpret_cast<cudaStream_t>(stream_ptr);
              }
              cudaFuncSetAttribute(Kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   DynamicSharedMemory);
              const dim3 grid = GridBlocks == 0 ? g.grid() : dim3(GridBlocks);
              const dim3 block = Threads == 0 ? g.block() : dim3(Threads);
              Kernel<<<grid, block, DynamicSharedMemory, stream>>>(g);
          });
}

#define BIND_LLAMA_GLOBALS_DYNAMIC(kernel, name, dynamic_smem)                \
    bind_kernel_dynamic_smem<kernel, dynamic_smem>(                           \
        m, name, LLAMA_GLOBAL_MEMBER_POINTERS)

#define BIND_LLAMA_GLOBALS_DYNAMIC_GRID(kernel, name, dynamic_smem, blocks)   \
    bind_kernel_dynamic_smem<kernel, dynamic_smem, blocks>(                   \
        m, name, LLAMA_GLOBAL_MEMBER_POINTERS)

template <typename Config>
static pybind11::dict direct_pdl_resource_metadata() {
    pybind11::dict metadata;
    metadata["num_pages"] = Config::NUM_PAGES;
    metadata["matvec_input_pipeline_stages"] =
        Config::MATVEC_INPUT_PIPELINE_STAGES;
    metadata["alias_four_pages"] = Config::DIRECT_ALIAS_FOUR_PAGES;
    metadata["dynamic_shared_memory_bytes"] = Config::DYNAMIC_SHARED_MEMORY;
    metadata["num_threads"] = Config::NUM_THREADS;
    metadata["trigger_wait_for_load_arrival"] =
        Config::DIRECT_PDL_TRIGGER_WAIT_FOR_LOAD_ARRIVAL;
    return metadata;
}

// globals_t::block() is fixed to the production 20-warp VM configuration.
// The four-consumer-warp admission control needs an explicit 8-warp launch;
// launching its __launch_bounds__(256) kernels through the ordinary binder
// would incorrectly request 640 threads and poison graph capture.
#define BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(kernel, name, dynamic_smem, threads) \
    bind_kernel_dynamic_smem<kernel, dynamic_smem, 0, threads>(               \
        m, name, LLAMA_GLOBAL_MEMBER_POINTERS)

PYBIND11_MODULE(mk_llama, m) {
    m.doc() = "";
    m.attr("cuda_compiler_version") =
        __CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 +
        __CUDACC_VER_BUILD__;
    m.attr("cuda_compiler_major") = __CUDACC_VER_MAJOR__;
    m.attr("cuda_compiler_minor") = __CUDACC_VER_MINOR__;
    m.attr("cuda_compiler_build") = __CUDACC_VER_BUILD__;
    m.attr("cuda_runtime_header_version") = CUDART_VERSION;
#ifdef ENABLE_PERSISTENT_VM
    // CUDA 13.1's H100 ptxas rejects the VM kernel's setmaxnreg instructions.
    // Matched persistent-vs-direct controls build this optional entry with the
    // accepted CUDA 12.8 toolchain; ordinary CUDA 13.1 ablation builds leave
    // it out.
    BIND_LLAMA_GLOBALS(
        (mk<default_config, llama_1b_globals, attention_partial_op,
            attention_reduction_op, rms_qkv_rope_append_op, downproj_op,
            o_proj_op, rms_upgate_silu_op, rms_lm_head_op>),
        "mk_llama");
#endif
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        rms_upgate_silu_direct_op>),
        "opcode5_direct");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_timed_config, llama_1b_globals,
                        rms_upgate_silu_timed_op>),
        "opcode5_direct_timed");
    BIND_LLAMA_GLOBALS_DYNAMIC(
        (opcode5_direct<opcode5_s1_late_config, llama_1b_globals,
                        rms_upgate_silu_s1_late_op>),
        "opcode5_direct_s1_late",
        opcode5_s1_late_config::DYNAMIC_SHARED_MEMORY);
    BIND_LLAMA_GLOBALS_DYNAMIC(
        (opcode5_direct<opcode5_s1_reg_config, llama_1b_globals,
                        rms_upgate_silu_s1_reg_op>),
        "opcode5_direct_s1_reg",
        opcode5_s1_reg_config::DYNAMIC_SHARED_MEMORY);
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        rms_upgate_silu_direct_ldg_op>),
        "opcode5_direct_ldg");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_timed_config, llama_1b_globals,
                        rms_upgate_silu_direct_ldg_timed_op>),
        "opcode5_direct_ldg_timed");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        rms_upgate_silu_direct_rms_op>),
        "opcode5_direct_rms");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_timed_config, llama_1b_globals,
                        rms_upgate_silu_direct_rms_timed_op>),
        "opcode5_direct_rms_timed");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        rms_upgate_silu_direct_store_op>),
        "opcode5_direct_store");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_timed_config, llama_1b_globals,
                        rms_upgate_silu_direct_store_timed_op>),
        "opcode5_direct_store_timed");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        rms_upgate_silu_overlap_up_reduce_op>),
        "opcode5_direct_overlap_up");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_timed_config, llama_1b_globals,
                        rms_upgate_silu_overlap_up_reduce_timed_op>),
        "opcode5_direct_overlap_up_timed");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        rms_upgate_silu_overlap_up_no_atomic_op>),
        "opcode5_direct_overlap_up_no_atomic");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_timed_config, llama_1b_globals,
                        rms_upgate_silu_overlap_up_no_atomic_timed_op>),
        "opcode5_direct_overlap_up_no_atomic_timed");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        rms_upgate_silu_overlap_up_fixed4_op>),
        "opcode5_direct_overlap_up_fixed4");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_timed_config, llama_1b_globals,
                        rms_upgate_silu_overlap_up_fixed4_timed_op>),
        "opcode5_direct_overlap_up_fixed4_timed");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        rms_upgate_silu_overlap_up_fixed4_no_atomic_op>),
        "opcode5_direct_overlap_up_fixed4_no_atomic");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_timed_config, llama_1b_globals,
                        rms_upgate_silu_overlap_up_fixed4_no_atomic_timed_op>),
        "opcode5_direct_overlap_up_fixed4_no_atomic_timed");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        rms_upgate_silu_overlap_up_fixed4_tc_no_atomic_op>),
        "opcode5_direct_overlap_up_fixed4_tc_no_atomic");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<
            opcode5_timed_config, llama_1b_globals,
            rms_upgate_silu_overlap_up_fixed4_tc_no_atomic_timed_op>),
        "opcode5_direct_overlap_up_fixed4_tc_no_atomic_timed");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        downproj_direct_op>),
        "opcode6_direct");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_timed_config, llama_1b_globals,
                        downproj_direct_timed_op>),
        "opcode6_direct_timed");
    BIND_LLAMA_GLOBALS_DYNAMIC(
        (opcode5_direct<opcode5_s1_late_config, llama_1b_globals,
                        downproj_s1_late_op>),
        "opcode6_direct_s1_late",
        opcode5_s1_late_config::DYNAMIC_SHARED_MEMORY);
    BIND_LLAMA_GLOBALS_DYNAMIC(
        (opcode5_direct<opcode5_s1_reg_config, llama_1b_globals,
                        downproj_s1_reg_op>),
        "opcode6_direct_s1_reg",
        opcode5_s1_reg_config::DYNAMIC_SHARED_MEMORY);
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_direct_config, llama_1b_globals,
                        qkv_direct_op>),
        "opcode1_direct");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode5_timed_config, llama_1b_globals,
                        qkv_direct_timed_op>),
        "opcode1_direct_timed");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode2_direct_config, llama_1b_globals,
                        attention_partial_direct_op>),
        "opcode2_direct");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode2_timed_config, llama_1b_globals,
                        attention_partial_direct_timed_op>),
        "opcode2_direct_timed");
#if 0
    BIND_LLAMA_GLOBALS(
        (mk<opcode3_vm_timed_config, llama_1b_globals,
            attention_reduction_vm_timed_op>),
        "opcode3_vm_timed");
#endif
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode3_direct_config, llama_1b_globals,
                        attention_reduction_direct_op>),
        "opcode3_direct");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode3_timed_config, llama_1b_globals,
                        attention_reduction_direct_timed_op>),
        "opcode3_direct_timed");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode4_direct_config, llama_1b_globals,
                        o_proj_direct_op>),
        "opcode4_direct");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode4_timed_config, llama_1b_globals,
                        o_proj_direct_timed_op>),
        "opcode4_direct_timed");
    BIND_LLAMA_GLOBALS_DYNAMIC(
        (opcode5_direct<opcode4_s1_late_config, llama_1b_globals,
                        o_proj_s1_late_op>),
        "opcode4_direct_s1_late",
        opcode4_s1_late_config::DYNAMIC_SHARED_MEMORY);
    BIND_LLAMA_GLOBALS_DYNAMIC(
        (opcode5_direct<opcode4_s1_reg_config, llama_1b_globals,
                        o_proj_s1_reg_op>),
        "opcode4_direct_s1_reg",
        opcode4_s1_reg_config::DYNAMIC_SHARED_MEMORY);
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode7_direct_config, llama_1b_globals,
                        lm_head_direct_op>),
        "opcode7_direct");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode7_direct_pdl_wait_config, llama_1b_globals,
                        lm_head_direct_pdl_wait_op>),
        "opcode7_direct_pdl_wait");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode7_direct_pdl_wait_trace_config,
                        llama_1b_globals,
                        lm_head_direct_pdl_wait_trace_op>),
        "opcode7_direct_pdl_wait_trace");
    BIND_LLAMA_GLOBALS(
        (opcode5_direct<opcode7_timed_config, llama_1b_globals,
                        lm_head_direct_timed_op>),
        "opcode7_direct_timed");

#define BIND_DIRECT_PDL_OPS(export_prefix, suffix, base_config)               \
    BIND_LLAMA_GLOBALS_DYNAMIC(                                               \
        (opcode5_direct<base_config, llama_1b_globals,                        \
                        opcode1_direct_pdl_##suffix##_op>),                   \
        "opcode1_direct_pdl_" export_prefix #suffix,                         \
        base_config::DYNAMIC_SHARED_MEMORY);                                  \
    BIND_LLAMA_GLOBALS_DYNAMIC(                                               \
        (opcode5_direct<opcode2_direct_pdl_##suffix##_config,                 \
                        llama_1b_globals,                                     \
                        opcode2_direct_pdl_##suffix##_op>),                   \
        "opcode2_direct_pdl_" export_prefix #suffix,                         \
        opcode2_direct_pdl_##suffix##_config::DYNAMIC_SHARED_MEMORY);         \
    BIND_LLAMA_GLOBALS_DYNAMIC(                                               \
        (opcode5_direct<opcode4_direct_pdl_##suffix##_config,                 \
                        llama_1b_globals,                                     \
                        opcode4_direct_pdl_##suffix##_op>),                   \
        "opcode4_direct_pdl_" export_prefix #suffix,                         \
        opcode4_direct_pdl_##suffix##_config::DYNAMIC_SHARED_MEMORY);         \
    BIND_LLAMA_GLOBALS_DYNAMIC(                                               \
        (opcode5_direct<base_config, llama_1b_globals,                        \
                        opcode5_direct_pdl_##suffix##_op>),                   \
        "opcode5_direct_pdl_" export_prefix #suffix,                         \
        base_config::DYNAMIC_SHARED_MEMORY);                                  \
    BIND_LLAMA_GLOBALS_DYNAMIC(                                               \
        (opcode5_direct<base_config, llama_1b_globals,                        \
                        opcode6_direct_pdl_##suffix##_op>),                   \
        "opcode6_direct_pdl_" export_prefix #suffix,                         \
        base_config::DYNAMIC_SHARED_MEMORY)

#define BIND_DIRECT_PDL_OPS_THREADS(export_prefix, suffix, base_config)       \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<base_config, llama_1b_globals,                        \
                        opcode1_direct_pdl_##suffix##_op>),                   \
        "opcode1_direct_pdl_" export_prefix #suffix,                         \
        base_config::DYNAMIC_SHARED_MEMORY, base_config::NUM_THREADS);        \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<opcode2_direct_pdl_##suffix##_config,                 \
                        llama_1b_globals,                                     \
                        opcode2_direct_pdl_##suffix##_op>),                   \
        "opcode2_direct_pdl_" export_prefix #suffix,                         \
        opcode2_direct_pdl_##suffix##_config::DYNAMIC_SHARED_MEMORY,          \
        base_config::NUM_THREADS);                                            \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<opcode4_direct_pdl_##suffix##_config,                 \
                        llama_1b_globals,                                     \
                        opcode4_direct_pdl_##suffix##_op>),                   \
        "opcode4_direct_pdl_" export_prefix #suffix,                         \
        opcode4_direct_pdl_##suffix##_config::DYNAMIC_SHARED_MEMORY,          \
        base_config::NUM_THREADS);                                            \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<base_config, llama_1b_globals,                        \
                        opcode5_direct_pdl_##suffix##_op>),                   \
        "opcode5_direct_pdl_" export_prefix #suffix,                         \
        base_config::DYNAMIC_SHARED_MEMORY, base_config::NUM_THREADS);        \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<base_config, llama_1b_globals,                        \
                        opcode6_direct_pdl_##suffix##_op>),                   \
        "opcode6_direct_pdl_" export_prefix #suffix,                         \
        base_config::DYNAMIC_SHARED_MEMORY, base_config::NUM_THREADS)

// Single-op feasibility bindings for opcode 4 only.
#define BIND_DIRECT_PDL_MICRO_OP4(suffix, base_config)                        \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<opcode4_direct_pdl_##suffix##_config,                 \
                        llama_1b_globals,                                     \
                        opcode4_direct_pdl_##suffix##_op>),                   \
        "opcode4_direct_pdl_4page_" #suffix,                                 \
        opcode4_direct_pdl_##suffix##_config::DYNAMIC_SHARED_MEMORY,          \
        base_config::NUM_THREADS)

// Both calls below bind the same device functions. The binding name and
// requested dynamic-SMEM size are the only differences, making the 128-KiB
// version an admission-only control.
#define BIND_DIRECT_PDL_MICRO_ALL(binding_name, dynamic_smem)                 \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<                                                      \
            direct_pdl_s1_issue_micro16_all_matvec_config,                   \
            llama_1b_globals, opcode1_direct_pdl_s1_micro16_all_op>),        \
        "opcode1_direct_pdl_4page_" binding_name, dynamic_smem,              \
        direct_pdl_s1_issue_micro16_all_matvec_config::NUM_THREADS);         \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<                                                      \
            opcode2_direct_pdl_s1_micro16_all_config, llama_1b_globals,      \
            opcode2_direct_pdl_s1_micro16_all_op>),                          \
        "opcode2_direct_pdl_4page_" binding_name, dynamic_smem,              \
        direct_pdl_s1_issue_micro16_all_attention_config::NUM_THREADS);      \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<                                                      \
            opcode4_direct_pdl_s1_micro16_all_config, llama_1b_globals,      \
            opcode4_direct_pdl_s1_micro16_all_op>),                          \
        "opcode4_direct_pdl_4page_" binding_name, dynamic_smem,              \
        direct_pdl_s1_issue_micro16_all_matvec_config::NUM_THREADS);         \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<                                                      \
            direct_pdl_s1_issue_micro16_all_matvec_config,                   \
            llama_1b_globals, opcode5_direct_pdl_s1_micro16_all_op>),        \
        "opcode5_direct_pdl_4page_" binding_name, dynamic_smem,              \
        direct_pdl_s1_issue_micro16_all_matvec_config::NUM_THREADS);         \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<                                                      \
            direct_pdl_s1_issue_micro16_all_matvec_config,                   \
            llama_1b_globals, opcode6_direct_pdl_s1_micro16_all_op>),        \
        "opcode6_direct_pdl_4page_" binding_name, dynamic_smem,              \
        direct_pdl_s1_issue_micro16_all_matvec_config::NUM_THREADS);         \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<                                                      \
            opcode7_direct_pdl_s1_micro16_all_config, llama_1b_globals,      \
            opcode7_direct_pdl_s1_micro16_all_op>),                          \
        "opcode7_direct_pdl_" binding_name, dynamic_smem,                    \
        opcode7_direct_pdl_s1_micro16_all_config::NUM_THREADS)

// Admission-only control for the partial-4-consumer ablation.  These Python
// entries launch the exact same device functions as issue_partial4cw, but
// reserve enough otherwise-unused dynamic SMEM to prevent a second CTA from
// residing on the same H100 SM.  In a one-wave, mask0 launch this changes no
// executed instruction; across PDL-linked grids it changes only admission.
#define BIND_DIRECT_PDL_PARTIAL4CW_SMEMBLOCK(dynamic_smem)                    \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<direct_pdl_4page_issue_partial4cw_config,             \
                        llama_1b_globals,                                     \
                        opcode1_direct_pdl_issue_partial4cw_op>),             \
        "opcode1_direct_pdl_4page_issue_partial4cw_smemblock",               \
        dynamic_smem,                                                         \
        direct_pdl_4page_issue_partial4cw_config::NUM_THREADS);               \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<opcode2_direct_pdl_issue_partial4cw_config,           \
                        llama_1b_globals,                                     \
                        opcode2_direct_pdl_issue_partial4cw_op>),             \
        "opcode2_direct_pdl_4page_issue_partial4cw_smemblock",               \
        dynamic_smem,                                                         \
        direct_pdl_4page_issue_partial4cw_config::NUM_THREADS);               \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<opcode4_direct_pdl_issue_partial4cw_config,           \
                        llama_1b_globals,                                     \
                        opcode4_direct_pdl_issue_partial4cw_op>),             \
        "opcode4_direct_pdl_4page_issue_partial4cw_smemblock",               \
        dynamic_smem,                                                         \
        direct_pdl_4page_issue_partial4cw_config::NUM_THREADS);               \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<direct_pdl_4page_issue_partial4cw_config,             \
                        llama_1b_globals,                                     \
                        opcode5_direct_pdl_issue_partial4cw_op>),             \
        "opcode5_direct_pdl_4page_issue_partial4cw_smemblock",               \
        dynamic_smem,                                                         \
        direct_pdl_4page_issue_partial4cw_config::NUM_THREADS);               \
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(                                       \
        (opcode5_direct<direct_pdl_4page_issue_partial4cw_config,             \
                        llama_1b_globals,                                     \
                        opcode6_direct_pdl_issue_partial4cw_op>),             \
        "opcode6_direct_pdl_4page_issue_partial4cw_smemblock",               \
        dynamic_smem,                                                         \
        direct_pdl_4page_issue_partial4cw_config::NUM_THREADS)

    BIND_DIRECT_PDL_OPS("4page_", r96, direct_pdl_4page_r96_config);
    BIND_DIRECT_PDL_OPS("4page_", r48, direct_pdl_4page_r48_config);
    BIND_DIRECT_PDL_OPS("4page_", issue_r96,
                        direct_pdl_4page_issue_r96_config);
    BIND_DIRECT_PDL_OPS("4page_", issue_r48,
                        direct_pdl_4page_issue_r48_config);
    BIND_DIRECT_PDL_OPS("4page_", issue_alias,
                        direct_pdl_4page_issue_alias_config);
    BIND_DIRECT_PDL_OPS("4page_", issue_chunk16,
                        direct_pdl_4page_issue_chunk16_config);
    BIND_DIRECT_PDL_OPS_THREADS("4page_", issue_4cw,
                                direct_pdl_4page_issue_4cw_config);
    BIND_DIRECT_PDL_OPS_THREADS(
        "4page_",
        issue_4cw_alias_natural,
        direct_pdl_4page_issue_4cw_alias_natural_config);
    BIND_DIRECT_PDL_OPS_THREADS(
        "4page_",
        issue_partial12cw,
        direct_pdl_4page_issue_partial12cw_config);
    BIND_DIRECT_PDL_OPS_THREADS(
        "4page_",
        issue_partial8cw,
        direct_pdl_4page_issue_partial8cw_config);
    BIND_DIRECT_PDL_OPS_THREADS(
        "4page_",
        issue_partial4cw,
        direct_pdl_4page_issue_partial4cw_config);
    BIND_DIRECT_PDL_OPS_THREADS(
        "4page_",
        issue_partial4cw_regpad,
        direct_pdl_4page_issue_partial4cw_regpad_config);
    BIND_DIRECT_PDL_PARTIAL4CW_SMEMBLOCK(128 * 1024);
    BIND_DIRECT_PDL_OPS("", 13page_issue_r96,
                        direct_pdl_13page_issue_r96_config);
    BIND_DIRECT_PDL_OPS("", 13page_arrival_r96,
                        direct_pdl_13page_arrival_r96_config);
    BIND_DIRECT_PDL_OPS("", 13page_issue_r96_trace,
                        direct_pdl_13page_issue_r96_trace_config);
    BIND_LLAMA_GLOBALS_DYNAMIC(
        (opcode5_direct<
            direct_pdl_13page_issue_r96_vtp_signal_config,
            llama_1b_globals,
            opcode5_direct_pdl_13page_issue_r96_vtp_signal_op>),
        "opcode5_direct_pdl_4page_13page_issue_r96_vtp_signal",
        direct_pdl_13page_issue_r96_vtp_signal_config::DYNAMIC_SHARED_MEMORY);
    BIND_LLAMA_GLOBALS_DYNAMIC(
        (opcode5_direct<
            direct_pdl_13page_issue_r96_vtp_wait_config,
            llama_1b_globals,
            opcode6_direct_pdl_13page_issue_r96_vtp_wait_op>),
        "opcode6_direct_pdl_4page_13page_issue_r96_vtp_wait",
        direct_pdl_13page_issue_r96_vtp_wait_config::DYNAMIC_SHARED_MEMORY);
    // Compact-grid control.  Opcode 1 and 6 genuinely use all 132 CTAs.
    // Opcode 2 has eight KV-head instructions, opcode 4 has 128 output
    // blocks, and opcode 5 has 128 nonempty one-wave instructions.
    BIND_LLAMA_GLOBALS_DYNAMIC_GRID(
        (opcode5_direct<direct_pdl_13page_issue_r96_config, llama_1b_globals,
                        opcode1_direct_pdl_13page_issue_r96_op>),
        "opcode1_direct_pdl_13page_issue_r96_compact",
        direct_pdl_13page_issue_r96_config::DYNAMIC_SHARED_MEMORY,
        H100_SM_COUNT);
    BIND_LLAMA_GLOBALS_DYNAMIC_GRID(
        (opcode5_direct<opcode2_direct_pdl_13page_issue_r96_config,
                        llama_1b_globals,
                        opcode2_direct_pdl_13page_issue_r96_op>),
        "opcode2_direct_pdl_13page_issue_r96_compact",
        opcode2_direct_pdl_13page_issue_r96_config::DYNAMIC_SHARED_MEMORY,
        LLAMA_1B_NUM_KV_HEADS);
    BIND_LLAMA_GLOBALS_DYNAMIC_GRID(
        (opcode5_direct<opcode4_direct_pdl_13page_issue_r96_config,
                        llama_1b_globals,
                        opcode4_direct_pdl_13page_issue_r96_op>),
        "opcode4_direct_pdl_13page_issue_r96_compact",
        opcode4_direct_pdl_13page_issue_r96_config::DYNAMIC_SHARED_MEMORY,
        LLAMA_1B_HIDDEN_DIM / LLAMA_1B_MATVEC_BLOCK_SIZE);
    BIND_LLAMA_GLOBALS_DYNAMIC_GRID(
        (opcode5_direct<
            direct_pdl_13page_issue_r96_compact_op5_config, llama_1b_globals,
            opcode5_direct_pdl_13page_issue_r96_compact_op>),
        "opcode5_direct_pdl_13page_issue_r96_compact",
        direct_pdl_13page_issue_r96_compact_op5_config::DYNAMIC_SHARED_MEMORY,
        LLAMA_1B_INTERMEDIATE_DIM / LLAMA_1B_HEAD_DIM);
    BIND_LLAMA_GLOBALS_DYNAMIC_GRID(
        (opcode5_direct<direct_pdl_13page_issue_r96_config, llama_1b_globals,
                        opcode6_direct_pdl_13page_issue_r96_op>),
        "opcode6_direct_pdl_13page_issue_r96_compact",
        direct_pdl_13page_issue_r96_config::DYNAMIC_SHARED_MEMORY,
        H100_SM_COUNT);
    BIND_DIRECT_PDL_OPS("", s1_issue_r96,
                        direct_pdl_s1_issue_r96_config);
    // Opcode 4 has exactly 128 output blocks. This entry launches the same
    // r96 device function with no four empty CTAs, isolating host/grid
    // admission overhead from all compute and memory-path changes.
    BIND_LLAMA_GLOBALS_DYNAMIC_GRID(
        (opcode5_direct<opcode4_direct_pdl_s1_issue_r96_config,
                        llama_1b_globals,
                        opcode4_direct_pdl_s1_issue_r96_op>),
        "opcode4_direct_pdl_4page_s1_issue_r96_compact",
        opcode4_direct_pdl_s1_issue_r96_config::DYNAMIC_SHARED_MEMORY,
        LLAMA_1B_HIDDEN_DIM / LLAMA_1B_MATVEC_BLOCK_SIZE);
    BIND_DIRECT_PDL_OPS("", s1_issue_r96_trace,
                        direct_pdl_s1_issue_r96_trace_config);
    BIND_DIRECT_PDL_OPS("", s1_issue_r48,
                        direct_pdl_s1_issue_r48_config);
    BIND_DIRECT_PDL_OPS_THREADS("", s1_issue_4cw,
                                direct_pdl_s1_issue_4cw_config);
    BIND_DIRECT_PDL_MICRO_OP4(
        s1_micro16_16cw,
        direct_pdl_s1_issue_micro16_16cw_config);
    BIND_DIRECT_PDL_MICRO_OP4(
        s1_micro16_16cw_trace,
        direct_pdl_s1_issue_micro16_16cw_trace_config);
    // Trace-enabled copy of the same micro16 device code with admission
    // blocked by otherwise-unused dynamic SMEM.
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(
        (opcode5_direct<opcode4_direct_pdl_s1_micro16_16cw_trace_config,
                        llama_1b_globals,
                        opcode4_direct_pdl_s1_micro16_16cw_trace_op>),
        "opcode4_direct_pdl_4page_s1_micro16_16cw_trace_smemblock",
        128 * 1024,
        direct_pdl_s1_issue_micro16_16cw_trace_config::NUM_THREADS);
    // Identical device function and thread geometry; only the requested
    // dynamic-SMEM launch size changes.  128 KiB plus the kernel's static
    // SMEM prevents a second CTA from residing on the same H100 SM and gives
    // the micro16 PDL experiment a clean admission-only control.
    BIND_LLAMA_GLOBALS_DYNAMIC_THREADS(
        (opcode5_direct<opcode4_direct_pdl_s1_micro16_16cw_config,
                        llama_1b_globals,
                        opcode4_direct_pdl_s1_micro16_16cw_op>),
        "opcode4_direct_pdl_4page_s1_micro16_16cw_smemblock",
        128 * 1024,
        direct_pdl_s1_issue_micro16_16cw_config::NUM_THREADS);
    BIND_DIRECT_PDL_MICRO_OP4(
        s1_micro16_8cw,
        direct_pdl_s1_issue_micro16_8cw_config);
    BIND_DIRECT_PDL_MICRO_OP4(
        s1_micro16_4cw,
        direct_pdl_s1_issue_micro16_4cw_config);
    BIND_DIRECT_PDL_MICRO_OP4(
        s1_micro32_16cw,
        direct_pdl_s1_issue_micro32_16cw_config);
    BIND_DIRECT_PDL_MICRO_OP4(
        s1_micro32_8cw,
        direct_pdl_s1_issue_micro32_8cw_config);
    BIND_DIRECT_PDL_MICRO_OP4(
        s1_micro64_16cw,
        direct_pdl_s1_issue_micro64_16cw_config);
    BIND_DIRECT_PDL_MICRO_OP4(
        s1_micro64_8cw,
        direct_pdl_s1_issue_micro64_8cw_config);
    BIND_DIRECT_PDL_MICRO_OP4(
        s1_micro128_16cw,
        direct_pdl_s1_issue_micro128_16cw_config);
    BIND_DIRECT_PDL_MICRO_OP4(
        s1_micro128_8cw,
        direct_pdl_s1_issue_micro128_8cw_config);
    BIND_DIRECT_PDL_MICRO_ALL(
        "s1_micro16_all_store",
        direct_pdl_s1_issue_micro16_all_matvec_config::DYNAMIC_SHARED_MEMORY);
    BIND_DIRECT_PDL_MICRO_ALL(
        "s1_micro16_all_store_smemblock", 128 * 1024);

    m.attr("direct_pdl_13page_issue_r96_metadata") =
        direct_pdl_resource_metadata<direct_pdl_13page_issue_r96_config>();
    m.attr("direct_pdl_13page_arrival_r96_metadata") =
        direct_pdl_resource_metadata<direct_pdl_13page_arrival_r96_config>();
#undef BIND_DIRECT_PDL_MICRO_ALL
#undef BIND_DIRECT_PDL_MICRO_OP4
#undef BIND_DIRECT_PDL_OPS_THREADS
#undef BIND_DIRECT_PDL_PARTIAL4CW_SMEMBLOCK
#undef BIND_DIRECT_PDL_OPS
}

#undef BIND_LLAMA_GLOBALS
#undef BIND_LLAMA_GLOBALS_DYNAMIC
#undef BIND_LLAMA_GLOBALS_DYNAMIC_GRID
#undef BIND_LLAMA_GLOBALS_DYNAMIC_THREADS
#undef LLAMA_GLOBAL_MEMBER_POINTERS
