# Static tile instruction streams

Flatten operator tiles into a persistent replay program when shapes and dependencies are predictable. Machete compiles tile-coordinate barrier formulas beside a compact stream; Hazy combines per-SM instruction sequences with controller/loader/consumer/storer roles and shared-memory pages.

The central tradeoff is compile-time schedule quality versus runtime flexibility. A broad producer wave avoids waits but may underfill the grid; a readiness-aware host scheduler can interleave tiles while retaining a static device stream.

Evidence: `ev-machete-formula-barriers`, `ev-machete-readiness-scheduler`, `ev-hazy-interpreter`, `ev-hazy-paged-smem`.
