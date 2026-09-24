#pragma once

#include "llama.cuh"

// A single-op launch wrapper for the existing Hazy opcode-5 TK roles.  It
// preserves the original loader/consumer/storer pipeline and shared-memory
// page layout, while removing the instruction-ring controller, runtime opcode
// dispatch, and worker await/finish protocol.
namespace megakernel {

// Direct-only markers.  Keep these outside the generic role-event range so a
// timed standalone launch can report both the exact trace-compatible
// dependency-ready -> role-end interval and the full direct CTA body.
constexpr int TEVENT_DIRECT_ROLES_START = 126;
constexpr int TEVENT_DIRECT_CTA_END = 127;

template <int Count> struct direct_register_padding {
    uint32_t slots[Count];

    __device__ inline void begin() {
        if (threadIdx.x != 0) {
            return;
        }
#pragma unroll
        for (int i = 0; i < Count; i++) {
            asm volatile("mov.u32 %0, %%clock;\n"
                         : "=r"(slots[i])
                         :
                         : "memory");
        }
    }

    __device__ inline void end() {
        if (threadIdx.x != 0) {
            return;
        }
#pragma unroll
        for (int i = 0; i < Count; i++) {
            asm volatile("mov.u32 %0, %0;\n"
                         : "+r"(slots[i])
                         :
                         : "memory");
        }
    }
};

template <> struct direct_register_padding<0> {
    __device__ inline void begin() {}
    __device__ inline void end() {}
};

template <typename config, typename globals>
__device__ inline void direct_store_timings(int *timings, int worker_id,
                                            const globals &g) {
    constexpr int bytes = config::TIMING_WIDTH * sizeof(int);
    uint32_t src_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(timings));
    uint64_t dst_ptr = reinterpret_cast<uint64_t>(
        &g.timings[kittens::coord<>{worker_id, 0, 0}]);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                 :
                 : "l"(dst_ptr), "r"(src_ptr), "n"(bytes)
                 : "memory");
    kittens::tma::store_commit_group();
}

template <typename config, typename globals, typename op>
__device__ inline void opcode5_direct_internal(const globals &g) {
    const int direct_worker_id = config::DIRECT_INDEX_BY_BLOCK
                                     ? static_cast<int>(blockIdx.x)
                                     : static_cast<int>(get_worker_id());
    direct_trace_init<config>(g, op::opcode, direct_worker_id);
    if (threadIdx.x == 0) {
        direct_trace_record<config>(g, DTRACE_CTA_ENTRY);
    }
    if constexpr (config::DIRECT_ACTIVE_BLOCKS > 0) {
        if (static_cast<int>(blockIdx.x) >= config::DIRECT_ACTIVE_BLOCKS) {
            if (threadIdx.x == 0) {
                direct_trace_record<config>(g, DTRACE_CTA_END);
            }
            return;
        }
    }
    direct_register_padding<config::DIRECT_REGISTER_PADDING> register_padding;
    register_padding.begin();
    uint64_t start_time = static_cast<uint64_t>(clock64());
    __shared__ alignas(128) instruction_state_t<config>
        instruction_state[config::INSTRUCTION_PIPELINE_STAGES];
    __shared__ kittens::semaphore
        page_finished[config::NUM_PAGES]
                     [config::INSTRUCTION_PIPELINE_STAGES_BITS],
        instruction_arrived[config::INSTRUCTION_PIPELINE_STAGES],
        instruction_finished[config::INSTRUCTION_PIPELINE_STAGES],
#ifdef KITTENS_BLACKWELL
        tensor_finished,
#endif
        semaphores_ready;

    extern __shared__ int dynamic_shm[];
    void *aligned_shm_addr = reinterpret_cast<void *>(
        (1023 + reinterpret_cast<uint64_t>(&dynamic_shm[0])) &
        ~static_cast<uint64_t>(1023));
    typename state<config>::page_array_t &pages =
        *reinterpret_cast<typename state<config>::page_array_t *>(
            aligned_shm_addr);
#ifdef KITTENS_BLACKWELL
    typename state<config>::tensor_allocator_t tensor_alloc{};
#endif

    state<config> mks{instruction_state,
                      instruction_arrived,
                      instruction_finished,
                      0,
                      0,
                      {/* unused register pid cache */},
                      pages,
                      page_finished,
#ifdef KITTENS_BLACKWELL
                      tensor_finished,
#endif
                      semaphores_ready,
                      start_time
#ifdef KITTENS_BLACKWELL
                      ,
                      tensor_alloc
#endif
    };

    // Full-grid direct launches retain Hazy's SM-indexed assignment.  Sparse
    // standalone launches use blockIdx so every requested instruction is
    // deterministic even though CUDA does not promise a particular SM ID.
    if (threadIdx.x < config::INSTRUCTION_WIDTH) {
        instruction_state[0].instructions[threadIdx.x] =
            g.instructions[kittens::coord<>{direct_worker_id, 0,
                                            static_cast<int>(threadIdx.x)}];
    }
    if constexpr (config::TIMING_RECORD_ENABLED) {
        if (threadIdx.x < config::TIMING_WIDTH) {
            instruction_state[0].timings[threadIdx.x] = 0;
        }
    }
    if (threadIdx.x < config::NUM_PAGES) {
        instruction_state[0].pid_order[threadIdx.x] = threadIdx.x;
        if constexpr (!config::SINGLE_OP_DIRECT) {
            for (int bit = 0; bit < config::INSTRUCTION_PIPELINE_STAGES_BITS;
                 ++bit) {
                int count = config::NUM_CONSUMER_WARPS * (1 << bit);
                init_semaphore(page_finished[threadIdx.x][bit], count);
                arrive(page_finished[threadIdx.x][bit], count);
            }
        }
    }
    if (threadIdx.x == 0) {
#ifdef KITTENS_BLACKWELL
        init_semaphore(tensor_finished, config::NUM_CONSUMER_WARPS);
        arrive(tensor_finished, config::NUM_CONSUMER_WARPS);
#endif
        op::controller::init_semaphores(g, mks);
        if constexpr (
            config::DIRECT_PDL_TRIGGER_GATE_ON_CONSUMER_WAIT) {
            *reinterpret_cast<int *>(
                reinterpret_cast<uint8_t *>(mks.scratch()) +
                config::SCRATCH_BYTES - sizeof(int)) = 0;
        }
    }

    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    mks.pid_order_shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
        &instruction_state[0].pid_order[0]));
    if (threadIdx.x == 0) {
        mks.record(TEVENT_DIRECT_ROLES_START);
        direct_trace_record<config>(g, DTRACE_ROLES_START);
    }

    if (kittens::warpid() < config::NUM_CONSUMER_WARPS) {
        if constexpr (config::DIRECT_USE_SETMAXNREG) {
            kittens::warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        }
        if (kittens::laneid() == 0) {
            mks.record(TEVENT_CONSUMER_START + 2 * kittens::warpid());
            if (kittens::warpid() == 0) {
                direct_trace_record<config>(g, DTRACE_CONSUMER_START);
            }
        }
        op::consumer::run(g, mks);
        if (kittens::laneid() == 0) {
            mks.record(TEVENT_CONSUMER_START + 2 * kittens::warpid() + 1);
            if (kittens::warpid() == 0) {
                direct_trace_record<config>(g, DTRACE_CONSUMER_END);
            }
        }
    } else {
        if constexpr (config::DIRECT_USE_SETMAXNREG) {
            kittens::warpgroup::decrease_registers<config::NON_CONSUMER_REGISTERS>();
        }
        switch (kittens::warpgroup::warpid()) {
        case 0:
            if (kittens::laneid() == 0) {
                mks.record(TEVENT_LOADER_START);
                direct_trace_record<config>(g, DTRACE_LOADER_START);
            }
            op::loader::run(g, mks);
            if (kittens::laneid() == 0) {
                mks.record(TEVENT_LOADER_START + 1);
                direct_trace_record<config>(g, DTRACE_LOADER_END);
            }
            break;
        case 1:
            if (kittens::laneid() == 0) {
                mks.record(TEVENT_STORER_START);
                direct_trace_record<config>(g, DTRACE_STORER_START);
            }
            op::storer::run(g, mks);
            if (kittens::laneid() == 0) {
                mks.record(TEVENT_STORER_START + 1);
                direct_trace_record<config>(g, DTRACE_STORER_END);
            }
            break;
        case 2:
            if (kittens::laneid() == 0) {
                mks.record(TEVENT_LAUNCHER_START);
                direct_trace_record<config>(g, DTRACE_LAUNCHER_START);
            }
            op::launcher::run(g, mks);
            if (kittens::laneid() == 0) {
                mks.record(TEVENT_LAUNCHER_START + 1);
                direct_trace_record<config>(g, DTRACE_LAUNCHER_END);
            }
            break;
        case 3:
            // The normal VM controller is intentionally removed.
            break;
        default:
            asm volatile("trap;");
        }
    }
    kittens::everyone::sync(15);
    if (threadIdx.x == 0) {
        direct_trace_record<config>(g, DTRACE_CTA_END);
    }
    if constexpr (config::TIMING_RECORD_ENABLED) {
        if (threadIdx.x == 0) {
            mks.record(TEVENT_DIRECT_CTA_END);
            direct_store_timings<config, globals>(
                &instruction_state[0].timings[0], direct_worker_id, g);
            kittens::tma::store_async_read_wait();
        }
    }
    register_padding.end();
}

template <typename config, typename globals, typename op>
__launch_bounds__(config::NUM_THREADS, config::DIRECT_MIN_BLOCKS_PER_SM)
__global__ void opcode5_direct(
    const __grid_constant__ globals g) {
    opcode5_direct_internal<config, globals, op>(g);
}

} // namespace megakernel
