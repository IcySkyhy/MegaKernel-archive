#pragma once

// Standalone opcode-5 variant: retain the original RMS and triple-buffered
// TMA weight pipeline, but replace each 32-byte TMA output store and VM
// dependency atomic with a direct register-to-global warp store.  In the
// standalone CUDA Graph, the next kernel node is the release/acquire edge.
template <typename Config, typename Globals>
struct rms_upgate_silu_direct_store {
    static constexpr int opcode = OPCODE_RMS_DoubleMatVecSiLU;
    static constexpr int prev_opcode = OPCODE_O_ProjResidual;
    static constexpr int expected_arrivals =
        Globals::hidden_dim / Globals::matvec_block_size;

    using base = rms_upgate_silu<Config, Globals>;
    using parsed_instruction = typename base::parsed_instruction;

    struct pipeline_specifics : base::pipeline_specifics {
        static __device__ inline void
        store(megakernel::state<Config> &s, const Globals &g,
              parsed_instruction &inst, int output_idx, int output_stage) {
            if ((output_idx & 1) == 0) {
                return;
            }

            int previous_stage = (output_idx - 1) % 3;
            auto *gate_scratch =
                pipeline::get_output_start(s, output_stage);
            auto *up_scratch =
                pipeline::get_output_start(s, previous_stage);

            kittens::rv_fl<16> up, gate, exp_scratch;
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(up_scratch, up);
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(gate_scratch, gate);

            kittens::warp::mul(exp_scratch, gate, -1.f);
            kittens::warp::exp(exp_scratch, exp_scratch);
            kittens::warp::add(exp_scratch, exp_scratch, 1.f);
            kittens::warp::div(gate, gate, exp_scratch);
            kittens::warp::mul(gate, up, gate);

            int block_idx = inst.block_idxs[output_idx / 2];
            auto &output = const_cast<typename Globals::activations_big_indim_t &>(
                g.silu_out);
            kittens::warp::store(output, gate, {block_idx});
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };

    using pipeline =
        rms_matvec_pipeline<Config, Globals, parsed_instruction,
                            pipeline_specifics, &Globals::hidden_states,
                            &Globals::mlp_norm_weights>;
    static_assert(pipeline::OUTPUT_PIPELINE_STAGES == 3);

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &,
                                              megakernel::state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            pipeline::loader_loop(s, g, inst.layer_idx);
        }
    };
    struct launcher {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };
    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::consumer_loop(s, g);
        }
    };
    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::template storer_loop<2>(s, g);
        }
    };
};
