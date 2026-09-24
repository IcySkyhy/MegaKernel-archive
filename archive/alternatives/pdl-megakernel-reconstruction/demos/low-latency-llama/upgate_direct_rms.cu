#pragma once

// Standalone opcode-5 variant: preserve the original triple-buffered TMA
// weight pipeline, but load the 2K activation and RMS scale directly into the
// consumer registers instead of staging each through a one-use shared page.
template <typename Config, typename Globals>
struct rms_upgate_silu_direct_rms : rms_upgate_silu<Config, Globals> {
    using base = rms_upgate_silu<Config, Globals>;
    using parsed_instruction = typename base::parsed_instruction;
    using pipeline = typename base::pipeline;
    using weight_pipeline = typename pipeline::pipeline;

    static constexpr int opcode = base::opcode;

    struct controller : base::controller {};
    struct launcher : base::launcher {};
    struct storer : base::storer {};

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            // Skip rms_matvec_pipeline::loader_loop's RMS-scale TMA and issue
            // only the original triple-buffered weight loads.
            weight_pipeline::loader_loop(s, g);
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            constexpr int reduction_dim_per_warp =
                Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;
            using activation_vec = kittens::rv_fl<reduction_dim_per_warp>;

            parsed_instruction inst{s};
            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                base::pipeline_specifics::gmem_wait(g, s);
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

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

            weight_pipeline::consumer_loop(s, g, activations);
        }
    };
};
