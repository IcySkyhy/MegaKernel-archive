# MegaMoE execution boundary

Compare a one-kernel role-specialized design with a streaming multi-kernel pipeline on the critical path. TeraMoE and DeepGEMM bring dispatch, expert compute, and combine into one persistent substrate. StreamEP keeps optimized kernels separate and uses readiness signals across two streams.

Prefer the one-kernel path when its role partition removes stage barriers without starving tensor-core work. Prefer the streaming path when modular kernels, independent scheduling, or portability outweigh the remaining launch/stream overhead.

Evidence: `ev-teramoe-five-roles`, `ev-deepgemm-megamoe-boundary`, `ev-streamep-alternative`, `ev-flashinfer-mega-split`, `ev-deepep-communication-boundary`.
