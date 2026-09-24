#pragma once

#include "llama.cuh"

template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics>
struct matvec_pipeline {
    static constexpr int INPUT_PIPELINE_STAGES =
        Config::MATVEC_INPUT_PIPELINE_STAGES;
    static constexpr int OUTPUT_PIPELINE_STAGES = 3;
    static constexpr int STAGE_PAGES = 4;
    static constexpr int ACTIVATION_PAGE = 0;
    static constexpr int WEIGHTS_START_PAGE = 1;

    static constexpr int LOGICAL_CONSUMER_WARPS =
        Config::DIRECT_LOGICAL_CONSUMER_WARPS > 0
            ? Config::DIRECT_LOGICAL_CONSUMER_WARPS
            : Config::NUM_CONSUMER_WARPS;
    static_assert(LOGICAL_CONSUMER_WARPS % STAGE_PAGES == 0);
    static_assert(Config::NUM_CONSUMER_WARPS <= LOGICAL_CONSUMER_WARPS);

    static constexpr int REDUCTION_DIM_PER_WARP =
        Globals::hidden_dim / LOGICAL_CONSUMER_WARPS;

    static constexpr int SEM_COUNT =
        1 + (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2;

    static constexpr int SCRATCH_BYTES_PER_WARP = 16 * sizeof(float);
    static constexpr int SCRATCH_BYTES_PER_STAGE =
        SCRATCH_BYTES_PER_WARP * Config::NUM_CONSUMER_WARPS;
    static constexpr int USED_SCRATCH_BYTES =
        OUTPUT_PIPELINE_STAGES * SCRATCH_BYTES_PER_STAGE;
    static_assert(USED_SCRATCH_BYTES <= Config::SCRATCH_BYTES,
                  "USED_SCRATCH_BYTES must be less than SCRATCH_BYTES");

    // Pages (very naive for now, no fine-grained usage)
    __device__ static inline int get_activation_page(megakernel::state<Config> &s) {
        return s.pid(ACTIVATION_PAGE);
    }

    __device__ static inline int get_weight_page(megakernel::state<Config> &s, int stage,
                                                 int offset) {
        if constexpr (Config::DIRECT_ALIAS_FOUR_PAGES) {
            // Mechanism-only ablation: four logical weight chunks alias the
            // four physical pages, and every pipeline stage reuses them.
            // Offset zero also aliases the activation page.  Results are
            // intentionally invalid; this isolates CTA admission/PDL overlap.
            return offset;
        }
        return s.pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + offset);
    }

    __device__ static inline kittens::semaphore &activations_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::semaphore &weights_arrived(megakernel::state<Config> &s,
                                                        int stage) {
        return s.semaphores()[1 + stage];
    }
    __device__ static inline kittens::semaphore &weights_finished(megakernel::state<Config> &s,
                                                         int stage) {
        return s.semaphores()[1 + INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &outputs_arrived(megakernel::state<Config> &s,
                                                        int stage) {
        return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &outputs_finished(megakernel::state<Config> &s,
                                                         int stage) {
        return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES +
                              OUTPUT_PIPELINE_STAGES + stage];
    }

    __device__ static inline kittens::sv_bf<Globals::hidden_dim> &
    get_activations(megakernel::state<Config> &s) {
        return *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
            s.pages[get_activation_page(s)].ptr());
    }

    __device__ static inline kittens::sv_fl<Globals::hidden_dim> &
    get_normalized_activations(megakernel::state<Config> &s) {
        static_assert(sizeof(kittens::sv_bf<Globals::hidden_dim>) +
                          sizeof(kittens::sv_bf<Globals::hidden_dim>) +
                          sizeof(kittens::sv_fl<Globals::hidden_dim>) <=
                      Config::PAGE_SIZE);
        return *reinterpret_cast<kittens::sv_fl<Globals::hidden_dim> *>(
            s.pages[get_activation_page(s)].ptr(
                2 * sizeof(kittens::sv_bf<Globals::hidden_dim>)));
    }

    __device__ static inline uint8_t *get_output_start(megakernel::state<Config> &s,
                                                       int stage) {
        return (uint8_t *)s.scratch() + (stage * SCRATCH_BYTES_PER_STAGE);
    }

    __device__ static inline int
    release_lid(const Globals &g, typename Config::instruction_t &instruction,
                int &query) {
        // NOTE: assumes a three stage pipeline

        if constexpr (Config::SINGLE_OP_DIRECT) {
            return query;
        }

        parsed_instruction inst{instruction};
        // unused pages, then activation, then weights

        static_assert(INPUT_PIPELINE_STAGES == 3 || Config::SINGLE_OP_DIRECT,
                      "INPUT_PIPELINE_STAGES must be 3");

        auto iters = inst.iters;
        auto remainder = iters % INPUT_PIPELINE_STAGES;

        // special handling for 1 and 2 because only then do
        // we free pages before the activation/rms scale (page 0)
        if (iters == 1) {
            int ret_order[13] = {5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4};
            return ret_order[query];
        } else if (iters == 2) {
            int ret_order[13] = {9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8};
            return ret_order[query];
        } else if (remainder == 1) {
            int ret_order[13] = {0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4};
            return ret_order[query];
        } else if (remainder == 2) {
            int ret_order[13] = {0, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8};
            return ret_order[query];
        } else {
            int ret_order[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            return ret_order[query];
        }
    }

    __device__ static inline int init_semaphores(megakernel::state<Config> &s) {
        init_semaphore(activations_arrived(s), 1);
        for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
            init_semaphore(weights_arrived(s, i), 1);
            init_semaphore(weights_finished(s, i), Config::NUM_CONSUMER_WARPS);
        }
        for (int i = 0; i < OUTPUT_PIPELINE_STAGES; i++) {
            init_semaphore(outputs_arrived(s, i), Config::NUM_CONSUMER_WARPS);
            init_semaphore(outputs_finished(s, i), 1);
        }
        return SEM_COUNT;
    }

    __device__ static inline void loader_loop(megakernel::state<Config> &s,
                                              const Globals &g) {
        parsed_instruction inst{s};

        auto needed_pages =
            1 + min(inst.iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;

        if (kittens::laneid() == 0) {

            int input_stage = 0;
            for (int iter = 0; iter < inst.iters; iter++) {
                kittens::wait(weights_finished(s, input_stage),
                     (iter % (2 * INPUT_PIPELINE_STAGES)) <
                         INPUT_PIPELINE_STAGES);

                auto &sem = weights_arrived(s, input_stage);
                kittens::tma::expect_bytes(sem, sizeof(kittens::bf16) * 2048 * 16);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    int weight_page = get_weight_page(s, input_stage, i);
                    if (iter < INPUT_PIPELINE_STAGES) {
                        s.wait_page_ready(weight_page);
                    }
                    auto &weight_chunk = reinterpret_cast<kittens::st_bf<16, 512> &>(
                        s.pages[weight_page]);

                    if (iter == 0 && i == 0) {
                        s.record(megakernel::TEVENT_FIRST_LOAD);
                        megakernel::direct_trace_record<Config>(
                            g, megakernel::DTRACE_WEIGHT_FIRST_ISSUE);
                    } else if (iter == inst.iters - 1 && i == 3) {
                        s.record(megakernel::TEVENT_LAST_LOAD);
                    }
                    if (iter == inst.iters - 1 && i == 3) {
                        megakernel::direct_trace_record<Config>(
                            g, megakernel::DTRACE_WEIGHT_LAST_ISSUE);
                    }

                    pipeline_specifics::load_iter(s, g, inst, iter, i,
                                                  weight_chunk, sem);
                }

                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }

            if constexpr (Config::DIRECT_PDL_TRIGGER_AFTER_LOAD) {
                if constexpr (
                    Config::DIRECT_PDL_TRIGGER_GATE_ON_CONSUMER_WAIT) {
                    // The consumer performs the actual dependency wait at
                    // the original activation-ready point. Gate the trigger
                    // on that CTA-local handoff so a blocked kernel cannot
                    // cascade admission through multiple future grids.
                    auto *dependency_ready =
                        reinterpret_cast<volatile int *>(
                            reinterpret_cast<uint8_t *>(s.scratch()) +
                            Config::SCRATCH_BYTES - sizeof(int));
                    while (*dependency_ready == 0) {
                        __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                    }
                }
                if constexpr (
                    Config::DIRECT_PDL_TRIGGER_WAIT_FOR_LOAD_ARRIVAL) {
                    // Arrival-trigger control: wait until the final staged
                    // weight transaction has actually reached SMEM.
                    if (inst.iters > 0) {
                        const int final_iter = inst.iters - 1;
                        const int final_stage =
                            final_iter % INPUT_PIPELINE_STAGES;
                        const int final_phase =
                            (final_iter % (2 * INPUT_PIPELINE_STAGES)) >=
                            INPUT_PIPELINE_STAGES;
                        kittens::wait(weights_arrived(s, final_stage),
                                      final_phase);
                    }
                }
                // Issue-trigger control reaches here immediately after the
                // loader has issued every weight TMA.  The successor still
                // waits before reading the producer's activation output.
                megakernel::direct_trace_record<Config>(
                    g, megakernel::DTRACE_TRIGGER_BEGIN);
                cudaTriggerProgrammaticLaunchCompletion();
                megakernel::direct_trace_record<Config>(
                    g, megakernel::DTRACE_TRIGGER_END);
            }
        } else if (kittens::laneid() >= needed_pages && kittens::laneid() < Config::NUM_PAGES) {
            auto pid = s.pid(kittens::laneid());
            s.wait_page_ready(pid);
            s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
        }
    }

    template <typename rv_t>
    __device__ static inline void
    consumer_loop(megakernel::state<Config> &s, const Globals &g, rv_t &activations_vec) {
        // Setup
        parsed_instruction inst{s};

        constexpr int WARPS_PER_PAGE =
            LOGICAL_CONSUMER_WARPS / STAGE_PAGES;

        using normalized_sv_t =
            kittens::sv_fl<REDUCTION_DIM_PER_WARP>;
        normalized_sv_t *normalized_activations_smem = nullptr;
        if constexpr (Config::MATVEC_SMEM_MICROTILE_COLS > 0) {
            normalized_activations_smem =
                &reinterpret_cast<normalized_sv_t *>(
                    &get_normalized_activations(s))[kittens::warpid()];
            kittens::warp::store(*normalized_activations_smem,
                                 activations_vec);
            kittens::warp::sync();
        }

        int page_index = kittens::warpid() / WARPS_PER_PAGE;

        int input_stage = 0, output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            int weight_page = get_weight_page(s, input_stage, page_index);
            kittens::wait(weights_arrived(s, input_stage),
                 (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);
            kittens::wait(outputs_finished(s, output_stage),
                 (i % (2 * OUTPUT_PIPELINE_STAGES)) < OUTPUT_PIPELINE_STAGES);
            kittens::st_bf<16, REDUCTION_DIM_PER_WARP> &weights =
                reinterpret_cast<kittens::st_bf<16, REDUCTION_DIM_PER_WARP> *>(
                    s.pages[weight_page].ptr())[kittens::warpid() % WARPS_PER_PAGE];

            kittens::sv_fl<16> &out_smem = *reinterpret_cast<kittens::sv_fl<16> *>(
                get_output_start(s, output_stage) +
                (kittens::warpid() * SCRATCH_BYTES_PER_WARP));

            if (i == 0) {
                s.record(megakernel::TEVENT_FIRST_USE);
                if (kittens::warpid() == 0 &&
                    kittens::laneid() == 0) {
                    megakernel::direct_trace_record<Config>(
                        g, megakernel::DTRACE_WEIGHT_FIRST_READY);
                }
            } else if (i == inst.iters - 1) {
                s.record(megakernel::TEVENT_LAST_USE);
            }
            if (i == inst.iters - 1) {
                if (kittens::warpid() == 0 &&
                    kittens::laneid() == 0) {
                    megakernel::direct_trace_record<Config>(
                        g, megakernel::DTRACE_WEIGHT_LAST_READY);
                }
            }

            if constexpr (Config::MATVEC_SMEM_MICROTILE_COLS > 0) {
                matvec_smem_microtile<
                    Config::MATVEC_SMEM_MICROTILE_COLS>(
                    out_smem, weights, *normalized_activations_smem);
            } else {
                matvec<Config::MATVEC_REGISTER_BUFFERED_STAGE,
                       Config::MATVEC_ALIAS_OPERANDS,
                       Config::MATVEC_CHUNK_COLS>(
                    out_smem, weights, activations_vec,
                    &weights_finished(s, input_stage));
            }

            kittens::warp::sync();
            kittens::warp::arrive(outputs_arrived(s, output_stage));
            if constexpr (!Config::MATVEC_REGISTER_BUFFERED_STAGE ||
                          Config::MATVEC_SMEM_MICROTILE_COLS > 0) {
                kittens::warp::arrive(weights_finished(s, input_stage));
            }

            if (i >= inst.iters - INPUT_PIPELINE_STAGES) {
// Release pages.
#pragma unroll
                for (int j = 0; j < STAGE_PAGES; j++) {
                    s.warp_finish_page(get_weight_page(s, input_stage, j), 1);
                }
            }

            input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
    }

    template <int iter_scale = 1>
    __device__ static inline void storer_loop(megakernel::state<Config> &s,
                                              const Globals &g) {
        parsed_instruction inst{s};

        int output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            auto &sem = outputs_arrived(s, output_stage);
            auto bit =
                (i % (2 * OUTPUT_PIPELINE_STAGES)) >= OUTPUT_PIPELINE_STAGES;

            kittens::wait(sem, bit);

            if (i == 0) {
                s.record(megakernel::TEVENT_FIRST_STORE);
                megakernel::direct_trace_record<Config>(
                    g, megakernel::DTRACE_FIRST_STORE);
            } else if (i == inst.iters - 1) {
                s.record(megakernel::TEVENT_LAST_STORE);
            }
            if (i == inst.iters - 1) {
                megakernel::direct_trace_record<Config>(
                    g, megakernel::DTRACE_LAST_STORE);
            }

            pipeline_specifics::store(s, g, inst, i, output_stage);

            if ((i + 1) % iter_scale == 0) {
                for (int j = 0; j < iter_scale; j++) {
                    auto stage_to_arrive = (i - j) % OUTPUT_PIPELINE_STAGES;
                    kittens::warp::arrive(outputs_finished(s, stage_to_arrive));
                }
            }
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
    }
};

template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics, auto ActPtr, auto RmsPtr>
struct rms_matvec_pipeline
    : public matvec_pipeline<Config, Globals, parsed_instruction,
                             pipeline_specifics> {
    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics>;

    static constexpr int REDUCTION_DIM_PER_WARP =
        pipeline::REDUCTION_DIM_PER_WARP;

    static constexpr int SEM_COUNT = 1 + pipeline::SEM_COUNT;

    __device__ static inline kittens::semaphore &rms_scale_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }

    __device__ static inline kittens::sv_bf<Globals::hidden_dim> &
    get_rms_scale(megakernel::state<Config> &s) {
        return *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
            s.pages[get_activation_page(s)].ptr(
                sizeof(kittens::sv_bf<Globals::hidden_dim>)));
    }

    __device__ static inline int init_semaphores(megakernel::state<Config> &s) {
        pipeline::init_semaphores(s);
        init_semaphore(rms_scale_arrived(s), 1);
        return SEM_COUNT;
    }

    __device__ static inline void loader_loop(megakernel::state<Config> &s,
                                              const Globals &g, int layer_idx) {
        if (kittens::laneid() == 0) {
            int activation_page = get_activation_page(s);
            s.wait_page_ready(activation_page);

            auto &rms_scale = get_rms_scale(s);
            auto &sem = rms_scale_arrived(s);

            kittens::tma::expect(sem, rms_scale);
            kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(rms_scale, g.*RmsPtr,
                                                      {layer_idx, 0}, sem);
        }

        pipeline::loader_loop(s, g);
    }

    __device__ static inline void launcher_loop(megakernel::state<Config> &s,
                                                const Globals &g) {
        if (kittens::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
            s.wait_tensor_ready();
            arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
        }
    }

    __device__ static inline void consumer_loop(megakernel::state<Config> &s,
                                                const Globals &g) {

        using sv_t = kittens::sv_bf<REDUCTION_DIM_PER_WARP>;
        auto &rms_scale_smem =
            reinterpret_cast<sv_t *>(&get_rms_scale(s))[kittens::warpid()];
        auto &activations_smem =
            reinterpret_cast<sv_t *>(&get_activations(s))[kittens::warpid()];

        if (kittens::laneid() == 0 && kittens::warpid() == 0) {
            parsed_instruction inst{s};

            int activation_page = get_activation_page(s);

            s.wait_page_ready(activation_page);
            auto &activations = get_activations(s);

            auto &sem = activations_arrived(s);

            // kittens::tma::expect(sem, activations);

            // Activation
            s.record(megakernel::TEVENT_AT_GMEM_WAIT);
            megakernel::direct_trace_record<Config>(
                g, megakernel::DTRACE_ACTIVATION_WAIT_BEGIN);
            if constexpr (Config::DIRECT_PDL_WAIT) {
                cudaGridDependencySynchronize();
                if constexpr (
                    Config::DIRECT_PDL_TRIGGER_GATE_ON_CONSUMER_WAIT) {
                    auto *dependency_ready =
                        reinterpret_cast<int *>(
                            reinterpret_cast<uint8_t *>(s.scratch()) +
                            Config::SCRATCH_BYTES - sizeof(int));
                    atomicExch(dependency_ready, 1);
                }
            } else {
                pipeline_specifics::gmem_wait(g, s);
            }
            s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            megakernel::direct_trace_record<Config>(
                g, megakernel::DTRACE_ACTIVATION_WAIT_END);

            // kittens::tma::load_async<cache_policy::EVICT_LAST>(activations, g.*ActPtr,
            // {}, sem);
        }
        kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

        kittens::warp::load(activations_smem, g.*ActPtr, {kittens::warpid()});
        if (kittens::warpid() == 0 && kittens::laneid() == 0) {
            megakernel::direct_trace_record<Config>(
                g, megakernel::DTRACE_ACTIVATION_LOAD_DONE);
        }

        auto activation_page = get_activation_page(s);

        kittens::wait(rms_scale_arrived(s), 0);

        auto activations_vec = rms_norm<Config>(
            rms_scale_smem, activations_smem, g.rms_norm_eps,
            pipeline::get_output_start(s, pipeline::OUTPUT_PIPELINE_STAGES));

        kittens::warp::sync();
        if constexpr (Config::MATVEC_SMEM_MICROTILE_COLS == 0) {
            s.warp_finish_page(activation_page, 1);
        }

        pipeline::consumer_loop(s, g, activations_vec);
    }
};
