#pragma once

// Shape-specialized storer for the active128 mapping: every active CTA owns
// exactly four output blocks (eight up/gate iterations); four inactive CTAs
// carry an empty instruction and exit immediately.
template <typename Config, typename Globals, bool SignalVmBarrier = true,
          bool UseTensorCore = false>
struct rms_upgate_silu_overlap_up_fixed4 {
    using overlap =
        rms_upgate_silu_overlap_up_reduce<Config, Globals, SignalVmBarrier>;
    using base = rms_upgate_silu<Config, Globals>;
    using parsed_instruction = typename base::parsed_instruction;
    using pipeline = typename base::pipeline;
    using weight_pipeline = typename pipeline::pipeline;

    static constexpr int opcode = base::opcode;
    using controller = typename overlap::controller;

    template <kittens::ducks::st::all St>
    static __device__ inline void
    matvec_candidate(kittens::sv_fl<St::rows> &out_smem, St &weights_smem,
                     kittens::rv_fl<St::cols> &activations) {
        if constexpr (!UseTensorCore) {
            matvec(out_smem, weights_smem, activations);
        } else {
            // Reuse TK's Hopper BF16 MMA path as an explicit ablation.  N=16
            // duplicates the single activation column, so row_max selects one
            // of sixteen identical FP32 accumulators after the AB^T MMA.
            using weights_rt = kittens::rt_bf<St::rows, St::cols>;
            using activation_row = typename weights_rt::row_vec;
            using output_col =
                typename kittens::rt_fl<16, 16>::col_vec;
            using output_vec = kittens::rv_fl<St::rows>;

            activation_row activation_bf16;
            kittens::warp::copy(activation_bf16, activations);

            weights_rt broadcast_activations, weights;
            kittens::warp::broadcast_col(broadcast_activations,
                                          activation_bf16);
            kittens::warp::load(weights, weights_smem);

            kittens::rt_fl<16, 16> duplicated_output;
            kittens::warp::zero(duplicated_output);
            kittens::warp::mma_ABt(duplicated_output, weights,
                                   broadcast_activations,
                                   duplicated_output);

            output_col one_output_col;
            kittens::warp::row_max(one_output_col, duplicated_output);
            output_vec output;
            kittens::warp::copy(output, one_output_col);
            if (kittens::laneid() < 16) {
                out_smem[kittens::laneid()] = output[0][0];
            }
            kittens::warp::sync();
        }
    }

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (inst.iters == 0 || kittens::laneid() != 0) {
                return;
            }

            auto &rms_scale = pipeline::get_rms_scale(s);
            auto &rms_sem = pipeline::rms_scale_arrived(s);
            kittens::tma::expect(rms_sem, rms_scale);
            kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                rms_scale, g.mlp_norm_weights, {inst.layer_idx, 0}, rms_sem);

#pragma unroll
            for (int iter = 0; iter < 8; ++iter) {
                int input_stage = iter % weight_pipeline::INPUT_PIPELINE_STAGES;
                bool finished_phase =
                    (iter % (2 * weight_pipeline::INPUT_PIPELINE_STAGES)) <
                    weight_pipeline::INPUT_PIPELINE_STAGES;
                kittens::wait(
                    weight_pipeline::weights_finished(s, input_stage),
                    finished_phase);

                auto &sem =
                    weight_pipeline::weights_arrived(s, input_stage);
                kittens::tma::expect_bytes(
                    sem, sizeof(kittens::bf16) * Globals::hidden_dim * 16);
#pragma unroll
                for (int chunk = 0; chunk < 4; ++chunk) {
                    int page = weight_pipeline::get_weight_page(
                        s, input_stage, chunk);
                    auto &weight_chunk =
                        reinterpret_cast<kittens::st_bf<16, 512> &>(
                            s.pages[page]);
                    if (iter == 0 && chunk == 0) {
                        s.record(megakernel::TEVENT_FIRST_LOAD);
                    } else if (iter == 7 && chunk == 3) {
                        s.record(megakernel::TEVENT_LAST_LOAD);
                    }
                    base::pipeline_specifics::load_iter(
                        s, g, inst, iter, chunk, weight_chunk, sem);
                }
            }
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (inst.iters == 0) {
                return;
            }

            constexpr int reduction_dim_per_warp =
                Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;
            constexpr int warps_per_page =
                Config::NUM_CONSUMER_WARPS / weight_pipeline::STAGE_PAGES;
            using activation_svec =
                kittens::sv_bf<reduction_dim_per_warp>;

            auto &rms_scale_smem =
                reinterpret_cast<activation_svec *>(
                    &pipeline::get_rms_scale(s))[kittens::warpid()];
            auto &activations_smem =
                reinterpret_cast<activation_svec *>(
                    &pipeline::get_activations(s))[kittens::warpid()];

            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                base::pipeline_specifics::gmem_wait(g, s);
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

            kittens::warp::load(activations_smem, g.hidden_states,
                                {kittens::warpid()});
            kittens::wait(pipeline::rms_scale_arrived(s), 0);
            auto activations = rms_norm<Config>(
                rms_scale_smem, activations_smem, g.rms_norm_eps,
                weight_pipeline::get_output_start(
                    s, weight_pipeline::OUTPUT_PIPELINE_STAGES));

            int page_index = kittens::warpid() / warps_per_page;
#pragma unroll
            for (int iter = 0; iter < 8; ++iter) {
                int input_stage = iter % weight_pipeline::INPUT_PIPELINE_STAGES;
                int output_stage =
                    iter % weight_pipeline::OUTPUT_PIPELINE_STAGES;
                bool weight_phase =
                    (iter % (2 * weight_pipeline::INPUT_PIPELINE_STAGES)) >=
                    weight_pipeline::INPUT_PIPELINE_STAGES;
                bool output_phase =
                    (iter % (2 * weight_pipeline::OUTPUT_PIPELINE_STAGES)) <
                    weight_pipeline::OUTPUT_PIPELINE_STAGES;
                kittens::wait(
                    weight_pipeline::weights_arrived(s, input_stage),
                    weight_phase);
                kittens::wait(
                    weight_pipeline::outputs_finished(s, output_stage),
                    output_phase);

                int page = weight_pipeline::get_weight_page(
                    s, input_stage, page_index);
                auto &weights =
                    reinterpret_cast<
                        kittens::st_bf<16, reduction_dim_per_warp> *>(
                        s.pages[page].ptr())[kittens::warpid() %
                                             warps_per_page];
                auto &out_smem = *reinterpret_cast<kittens::sv_fl<16> *>(
                    weight_pipeline::get_output_start(s, output_stage) +
                    kittens::warpid() *
                        weight_pipeline::SCRATCH_BYTES_PER_WARP);

                if (iter == 0) {
                    s.record(megakernel::TEVENT_FIRST_USE);
                } else if (iter == 7) {
                    s.record(megakernel::TEVENT_LAST_USE);
                }
                matvec_candidate(out_smem, weights, activations);
                kittens::warp::sync();
                kittens::warp::arrive(
                    weight_pipeline::outputs_arrived(s, output_stage));
                kittens::warp::arrive(
                    weight_pipeline::weights_finished(s, input_stage));
            }
        }
    };

    struct storer {
        template <int FirstBlock, int BlockStride>
        static __device__ void run_blocks(const Globals &g,
                                          megakernel::state<Config> &s) {
            static_assert(FirstBlock >= 0 && FirstBlock < 4);
            static_assert(BlockStride > 0);

            parsed_instruction inst{s};
            if (inst.iters == 0) {
                return;
            }

#pragma unroll
            for (int block = FirstBlock; block < 4; block += BlockStride) {
                int up_idx = 2 * block;
                int gate_idx = up_idx + 1;
                int up_stage = up_idx % pipeline::OUTPUT_PIPELINE_STAGES;
                int gate_stage = gate_idx % pipeline::OUTPUT_PIPELINE_STAGES;
                bool up_phase =
                    (up_idx % (2 * pipeline::OUTPUT_PIPELINE_STAGES)) >=
                    pipeline::OUTPUT_PIPELINE_STAGES;
                bool gate_phase =
                    (gate_idx % (2 * pipeline::OUTPUT_PIPELINE_STAGES)) >=
                    pipeline::OUTPUT_PIPELINE_STAGES;

                kittens::wait(pipeline::outputs_arrived(s, up_stage), up_phase);
                if (block == 0) {
                    s.record(megakernel::TEVENT_FIRST_STORE);
                }

                kittens::rv_fl<16> up, gate;
                auto *up_scratch = pipeline::get_output_start(s, up_stage);
                matvec_reduce<Config, kittens::sv_fl<16>,
                              kittens::rv_fl<16>,
                              pipeline::SCRATCH_BYTES_PER_WARP>(up_scratch,
                                                                 up);
                if constexpr (!SignalVmBarrier) {
                    // The up scratch has been consumed into registers; it no
                    // longer needs to wait for gate epilogue/store.
                    kittens::warp::arrive(
                        pipeline::outputs_finished(s, up_stage));
                }

                kittens::wait(pipeline::outputs_arrived(s, gate_stage),
                              gate_phase);
                if (block == 3) {
                    s.record(megakernel::TEVENT_LAST_STORE);
                }
                auto *gate_scratch = pipeline::get_output_start(s, gate_stage);
                matvec_reduce<Config, kittens::sv_fl<16>,
                              kittens::rv_fl<16>,
                              pipeline::SCRATCH_BYTES_PER_WARP>(gate_scratch,
                                                                 gate);

                if constexpr (SignalVmBarrier) {
                    // Keep the original TK epilogue for the VM-compatible
                    // control candidate.
                    kittens::rv_fl<16> exp_scratch;
                    kittens::warp::mul(exp_scratch, gate, -1.f);
                    kittens::warp::exp(exp_scratch, exp_scratch);
                    kittens::warp::add(exp_scratch, exp_scratch, 1.f);
                    kittens::warp::div(gate, gate, exp_scratch);
                    kittens::warp::mul(gate, up, gate);
                } else {
                    // rv_fl<16> holds one useful scalar per lane.  The
                    // standalone edge does not need the generic vector-map
                    // temporaries, and BF16 output makes fast float division
                    // more than accurate enough for the original tolerance.
                    float gate_value = gate[0][0];
                    gate[0][0] = up[0][0] *
                                  __fdividef(
                                      gate_value,
                                      1.f + __expf(-gate_value));
                }

                int block_idx = inst.block_idxs[block];
                if constexpr (SignalVmBarrier) {
                    auto &out_smem = *reinterpret_cast<kittens::sv_bf<16> *>(
                        gate_scratch);
                    kittens::warp::sync();
                    kittens::warp::store(out_smem, gate);
                    kittens::warp::sync();

                    if (kittens::laneid() == 0) {
                        kittens::tma::store_async<
                            kittens::cache_policy::EVICT_LAST>(
                            g.silu_out, out_smem, {block_idx});
                        kittens::tma::store_async_wait();
                        s.record(megakernel::TEVENT_AT_GMEM_STORE);
                        atomicAdd(
                            &g.Bar[{inst.layer_idx, opcode - 1,
                                    block_idx * Globals::matvec_block_size /
                                        Globals::hidden_dim}],
                            1);
                        s.record(megakernel::TEVENT_DONE_GMEM_STORE);
                    }
                    kittens::warp::sync();
                } else {
                    // A 16-element BF16 result is only 32 bytes.  For the
                    // standalone CUDA-Graph edge, write it directly from the
                    // reduction registers and let kernel completion provide
                    // publication to the next node.
                    auto &output = const_cast<
                        typename Globals::activations_big_indim_t &>(
                        g.silu_out);
                    kittens::warp::store(output, gate, {block_idx});
                    kittens::warp::sync();
                }

                if constexpr (SignalVmBarrier) {
                    kittens::warp::arrive(
                        pipeline::outputs_finished(s, up_stage));
                    kittens::warp::arrive(
                        pipeline::outputs_finished(s, gate_stage));
                } else {
                    kittens::warp::arrive(
                        pipeline::outputs_finished(s, gate_stage));
                }
            }
        }

        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if constexpr (SignalVmBarrier) {
                run_blocks<0, 1>(g, s);
            } else {
                run_blocks<0, 2>(g, s);
            }
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if constexpr (SignalVmBarrier) {
                overlap::launcher::run(g, s);
            } else {
                // The Hopper launcher role is otherwise idle.  Split the
                // fixed four-block epilogue across two non-consumer warps.
                storer::template run_blocks<1, 2>(g, s);
            }
        }
    };
};
