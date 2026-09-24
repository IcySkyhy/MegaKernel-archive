#pragma once

#include "kittens.cuh"

namespace megakernel {

struct default_config {
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // One controller warp, one load warp, one store warp, and one mma warp.
    static constexpr int NUM_CONSUMER_WARPS = 16;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = ::kittens::MAX_SHARED_MEMORY;

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 4096;
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
                  (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 +
                   DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY =
        ::kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 13, "NUM_PAGES must be 13");

    // The original matvec pipeline keeps three complete 16x2048 BF16 weight
    // tiles in flight. Standalone experiments override these independently of
    // the VM instruction-ring depth.
    static constexpr int MATVEC_INPUT_PIPELINE_STAGES = 3;
    static constexpr bool MATVEC_REGISTER_BUFFERED_STAGE = false;
    // Mechanism-only ablation: end the activation register-tile lifetime
    // before loading the weight register tile. The activation broadcast is
    // kept by an empty inline-PTX compiler barrier, and the arithmetic uses
    // the weight tile twice. This preserves instruction/load volume but
    // intentionally produces incorrect output.
    static constexpr bool MATVEC_ALIAS_OPERANDS = false;
    // Zero keeps the original full-width register tile. Hopper experiments
    // may stream the K dimension through a smaller register tile while
    // retaining the complete SMEM load and arithmetic work.
    static constexpr int MATVEC_CHUNK_COLS = 0;
    // Correct low-register path: spill the already-computed FP32 activation
    // vector into the unused second half of the activation page, then stream
    // activation and weight microtiles from SMEM. Unlike MATVEC_CHUNK_COLS,
    // this avoids constructing the full aligned activation register vector.
    static constexpr int MATVEC_SMEM_MICROTILE_COLS = 0;

    static constexpr bool TIMING_RECORD_ENABLED = false;

    // Standalone candidates can prove that no later VM instruction will
    // reuse a page, so page-lifetime waits/arrivals are unnecessary.
    static constexpr bool SINGLE_OP_DIRECT = false;

    // Direct-TK PDL ablation controls.  They default off so the extracted
    // standalone reference entries retain their original behavior.
    static constexpr bool DIRECT_PDL_WAIT = false;
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_LOAD = false;
    static constexpr bool DIRECT_PDL_TRIGGER_AFTER_STORE = false;
    static constexpr bool DIRECT_PDL_TRIGGER_GATE_ON_CONSUMER_WAIT = false;
    static constexpr bool DIRECT_PDL_TRIGGER_WAIT_FOR_LOAD_ARRIVAL = false;
    static constexpr bool DIRECT_SKIP_GLOBAL_COUNTERS = false;
    static constexpr bool DIRECT_ALIAS_FOUR_PAGES = false;
    // Occupancy-control ablation. Empty inline asm keeps this many FP32
    // registers live across the direct kernel without emitting instructions.
    static constexpr int DIRECT_REGISTER_PADDING = 0;
    // Zero means that the matvec K partition follows NUM_CONSUMER_WARPS.
    // Mechanism-only candidates can retain the original logical partition
    // while launching fewer consumers, leaving some loaded K slices unused.
    static constexpr int DIRECT_LOGICAL_CONSUMER_WARPS = 0;

    // Cross-kernel trace mode writes absolute %globaltimer samples and %smid
    // metadata directly to each standalone node's unique timings tensor.
    // Keep this separate from TIMING_RECORD_ENABLED, whose clock64 deltas are
    // meaningful only within one CTA.
    static constexpr bool DIRECT_GLOBAL_TRACE_ENABLED = false;
    static constexpr int DIRECT_MIN_BLOCKS_PER_SM = 1;
    // The CUDA 13.1 H100 toolchain rejects setmaxnreg for sm_90a.  Hopper
    // direct kernels therefore use ptxas' static allocation only.
    static constexpr bool DIRECT_USE_SETMAXNREG = false;

    // The upstream declaration was `bool = 20`, which converted to true and
    // therefore made every measured __nanosleep call one nanosecond.  Preserve
    // that measured behavior while giving the duration an honest integer type.
    static constexpr int GMEM_SPIN_LOOP_SLEEP_NANOS = 1;

    static constexpr int CONSUMER_REGISTERS = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};
template <typename config>
using instruction_layout = kittens::gl<int, 1, -1, -1, config::INSTRUCTION_WIDTH>;
template <typename config>
using timing_layout = kittens::gl<int, 1, -1, -1, config::TIMING_WIDTH>;

template <typename config> void print_config() {
    std::cout << "---------------- CONFIG INFO ----------------" << std::endl;
    std::cout << "INSTRUCTION_PIPELINE_STAGES: "
              << config::INSTRUCTION_PIPELINE_STAGES << std::endl;
    std::cout << "INSTRUCTION_WIDTH: " << config::INSTRUCTION_WIDTH
              << std::endl;
    std::cout << "TIMING_WIDTH: " << config::TIMING_WIDTH << std::endl;
    std::cout << "NUM_CONSUMER_WARPS: " << config::NUM_CONSUMER_WARPS
              << std::endl;
    std::cout << "NUM_WARPS: " << config::NUM_WARPS << std::endl;
    std::cout << "NUM_THREADS: " << config::NUM_THREADS << std::endl;
    std::cout << "NUM_BLOCKS: " << config::NUM_BLOCKS << std::endl;
    std::cout << "CLUSTER_BLOCKS: " << config::CLUSTER_BLOCKS << std::endl;
    std::cout << "MAX_SHARED_MEMORY: " << config::MAX_SHARED_MEMORY
              << std::endl;
    std::cout << "STATIC_SHARED_MEMORY: " << config::STATIC_SHARED_MEMORY
              << std::endl;
    std::cout << "PAGE_SIZE: " << config::PAGE_SIZE << std::endl;
    std::cout << "NUM_PAGES: " << config::NUM_PAGES << std::endl;
    std::cout << "SCRATCH_BYTES: " << config::SCRATCH_BYTES << std::endl;
    std::cout << "DYNAMIC_SEMAPHORES: " << config::DYNAMIC_SEMAPHORES
              << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
}

} // namespace megakernel
