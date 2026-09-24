#pragma once

// Standalone opcode-5 variant: preserve the original TMA and matvec pipeline,
// but consume/reduce each up partial as soon as it becomes ready.  The base
// storer waits for gate and then reduces up+gate serially, leaving this work
// exposed on the tail.
template <typename Config, typename Globals, bool SignalVmBarrier = true>
struct rms_upgate_silu_overlap_up_reduce
    : rms_upgate_silu<Config, Globals> {
    using base = rms_upgate_silu<Config, Globals>;
    using parsed_instruction = typename base::parsed_instruction;
    using pipeline = typename base::pipeline;

    static constexpr int opcode = base::opcode;

    struct controller : base::controller {};
    struct loader : base::loader {};
    struct launcher : base::launcher {};
    struct consumer : base::consumer {};

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};

            for (int up_idx = 0; up_idx < inst.iters; up_idx += 2) {
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
                if (up_idx == 0) {
                    s.record(megakernel::TEVENT_FIRST_STORE);
                }

                kittens::rv_fl<16> up, gate, exp_scratch;
                auto *up_scratch = pipeline::get_output_start(s, up_stage);
                matvec_reduce<Config, kittens::sv_fl<16>,
                              kittens::rv_fl<16>,
                              pipeline::SCRATCH_BYTES_PER_WARP>(up_scratch,
                                                                 up);

                // Gate compute and output publication proceed concurrently
                // with the up reduction above.
                kittens::wait(pipeline::outputs_arrived(s, gate_stage),
                              gate_phase);
                if (gate_idx == inst.iters - 1) {
                    s.record(megakernel::TEVENT_LAST_STORE);
                }
                auto *gate_scratch = pipeline::get_output_start(s, gate_stage);
                matvec_reduce<Config, kittens::sv_fl<16>,
                              kittens::rv_fl<16>,
                              pipeline::SCRATCH_BYTES_PER_WARP>(gate_scratch,
                                                                 gate);

                kittens::warp::mul(exp_scratch, gate, -1.f);
                kittens::warp::exp(exp_scratch, exp_scratch);
                kittens::warp::add(exp_scratch, exp_scratch, 1.f);
                kittens::warp::div(gate, gate, exp_scratch);
                kittens::warp::mul(gate, up, gate);

                auto &out_smem = *reinterpret_cast<kittens::sv_bf<16> *>(
                    gate_scratch);
                kittens::warp::sync();
                kittens::warp::store(out_smem, gate);
                kittens::warp::sync();

                if (kittens::laneid() == 0) {
                    int block_idx = inst.block_idxs[up_idx / 2];
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.silu_out, out_smem, {block_idx});
                    kittens::tma::store_async_wait();
                    s.record(megakernel::TEVENT_AT_GMEM_STORE);
                    if constexpr (SignalVmBarrier) {
                        atomicAdd(
                            &g.Bar[{inst.layer_idx, opcode - 1,
                                    block_idx * Globals::matvec_block_size /
                                        Globals::hidden_dim}],
                            1);
                    }
                    s.record(megakernel::TEVENT_DONE_GMEM_STORE);
                }
                kittens::warp::sync();

                kittens::warp::arrive(
                    pipeline::outputs_finished(s, up_stage));
                kittens::warp::arrive(
                    pipeline::outputs_finished(s, gate_stage));
            }
        }
    };
};
