#pragma once

// The context-1 latency schedule contains one partial-attention instruction
// for each of eight KV heads.  The direct kernel still launches one resident
// CTA per H100 SM, so negative kv_head_idx instructions mark inactive CTAs.
// Active CTAs run the original TK roles unchanged.
template <typename Config, typename Globals>
struct attention_partial_direct : attention_partial<Config, Globals> {
    using base = attention_partial<Config, Globals>;
    using parsed_instruction = typename base::parsed_instruction;
    using controller = typename base::controller;

    static __device__ inline bool active(megakernel::state<Config> &s) {
        return parsed_instruction{s}.kv_head_idx >= 0;
    }

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (active(s)) {
                base::loader::run(g, s);
            }
        }
    };
    struct launcher {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (active(s)) {
                base::launcher::run(g, s);
            }
        }
    };
    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (active(s)) {
                base::consumer::run(g, s);
            }
        }
    };
    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (active(s)) {
                base::storer::run(g, s);
            }
        }
    };
};
