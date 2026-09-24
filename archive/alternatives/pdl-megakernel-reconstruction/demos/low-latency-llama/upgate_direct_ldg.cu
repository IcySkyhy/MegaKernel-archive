#pragma once

// Standalone opcode-5 variant: keep Hazy's CUDA-core matvec/reduction and
// output pipeline, but let each consumer warp load its one-use BF16 weight
// slice directly from global memory into registers.  No weight is staged by
// TMA or reread from shared memory.
template <typename Config, typename Globals>
struct rms_upgate_silu_direct_ldg : rms_upgate_silu<Config, Globals> {
    using base = rms_upgate_silu<Config, Globals>;
    using parsed_instruction = typename base::parsed_instruction;
    using pipeline = typename base::pipeline;

    static constexpr int opcode = base::opcode;

    struct controller : base::controller {};
    struct launcher : base::launcher {};
    struct storer : base::storer {};

    // Consumers issue all weight loads in this variant.
    struct loader {
        static __device__ void run(const Globals &, megakernel::state<Config> &) {}
    };

    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            constexpr int reduction_dim_per_warp =
                Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;
            using activation_vec = kittens::rv_fl<reduction_dim_per_warp>;
            using weight_tile = kittens::rt_fl<16, reduction_dim_per_warp>;
            using weight_row_vec = typename weight_tile::row_vec;
            using weight_col_vec = typename weight_tile::col_vec;
            using output_vec = kittens::rv_fl<16>;

            parsed_instruction inst{s};

            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                base::pipeline_specifics::gmem_wait(g, s);
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

            // Direct global-to-register activation and RMS-scale loads avoid
            // an otherwise single-use shared-memory round trip.
            activation_vec activations, squared, rms_scale;
            kittens::warp::load(activations, g.hidden_states,
                                {kittens::warpid()});
            kittens::warp::copy(squared, activations);
            kittens::warp::mul(squared, squared, squared);
            float partial_sum = kittens::warp::sum(squared);

            float *rms_partial_sums = reinterpret_cast<float *>(
                pipeline::get_output_start(s,
                                           pipeline::OUTPUT_PIPELINE_STAGES));
            if (kittens::laneid() == 0) {
                rms_partial_sums[kittens::warpid()] = partial_sum;
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(0);

            float full_sum = 0.f;
#pragma unroll
            for (int warp = 0; warp < Config::NUM_CONSUMER_WARPS; ++warp) {
                full_sum += rms_partial_sums[warp];
            }
            float rms = rsqrtf(full_sum / float(Globals::hidden_dim) +
                               g.rms_norm_eps);
            kittens::warp::mul(activations, activations, rms);
            kittens::warp::load(rms_scale, g.mlp_norm_weights,
                                {inst.layer_idx, kittens::warpid()});
            kittens::warp::mul(activations, activations, rms_scale);

            int output_stage = 0;
            for (int iter = 0; iter < inst.iters; ++iter) {
                auto output_phase =
                    (iter % (2 * pipeline::OUTPUT_PIPELINE_STAGES)) <
                    pipeline::OUTPUT_PIPELINE_STAGES;
                kittens::wait(pipeline::outputs_finished(s, output_stage),
                              output_phase);

                int block_idx = inst.block_idxs[iter / 2];
                weight_tile weights, broadcast_activations;
                if (iter == 0) {
                    s.record(megakernel::TEVENT_FIRST_LOAD);
                } else if (iter == inst.iters - 1) {
                    s.record(megakernel::TEVENT_LAST_LOAD);
                }
                if ((iter & 1) == 0) {
                    kittens::warp::load(
                        weights, g.up_weights,
                        {inst.layer_idx, block_idx, kittens::warpid()});
                } else {
                    kittens::warp::load(
                        weights, g.gate_weights,
                        {inst.layer_idx, block_idx, kittens::warpid()});
                }

                if (iter == 0) {
                    s.record(megakernel::TEVENT_FIRST_USE);
                } else if (iter == inst.iters - 1) {
                    s.record(megakernel::TEVENT_LAST_USE);
                }

                weight_row_vec row_activations;
                kittens::warp::copy(row_activations, activations);
                kittens::warp::broadcast_col(broadcast_activations,
                                             row_activations);
                kittens::warp::mul(broadcast_activations,
                                   broadcast_activations, weights);
                weight_col_vec partial_output;
                kittens::warp::row_sum(partial_output,
                                       broadcast_activations);
                output_vec output;
                kittens::warp::copy(output, partial_output);

                auto *output_scratch = pipeline::get_output_start(s, output_stage);
                auto &out_smem = *reinterpret_cast<kittens::sv_fl<16> *>(
                    output_scratch + kittens::warpid() *
                                         pipeline::SCRATCH_BYTES_PER_WARP);
                if (kittens::laneid() < 16) {
                    out_smem[kittens::laneid()] = output[0][0];
                }
                kittens::warp::sync();
                kittens::warp::arrive(
                    pipeline::outputs_arrived(s, output_stage));

                output_stage =
                    (output_stage + 1) % pipeline::OUTPUT_PIPELINE_STAGES;
            }
        }
    };
};
