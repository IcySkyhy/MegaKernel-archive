# From the hand-written kernels to MLIR passes

The executable target is currently **two timesteps per launch**. The next
compiler experiment should reproduce that transformation and its numerical
contract before pursuing a persistent, whole-simulation kernel. No MLIR
passes are implemented in this repository.

| Compiler responsibility | Hand-written reference | Required reasoning |
| --- | --- | --- |
| Spatial mapping | B0 → C1/C2/C3 → S2-C/D | Separate CUDA workers from a logical tile; cover tails exactly once |
| Producer-region derivation | V1/V2 first loop | Invert each directional scatter: producer = consumer − direction |
| Halo computation | Full producer box | Replicate neighboring first-step work without cross-block dependencies |
| Intermediate promotion | V1 → V2 | Store post-collision producer state or streamed core state in workgroup memory |
| Synchronization | One unconditional block barrier | Prove every required shared value is written before it is read |
| Final output ownership | Second collision and global scatter | Fixed-direction scatter is injective; preserve padding writes |
| Resource selection | 128/256 workers, three shapes | Cost shared capacity, registers, redundant work and transactions together |
| Time-loop rewrite | Host buffer alternation | Advance by two; select the final buffer; account for observable intermediate states |

A useful pipeline would recognize the stencil and immutable flags, construct
consumer tiles, backward-slice their producer requirements, and promote the
intermediate. Only after proving ownership and synchronization should it
lower to the GPU dialect's launch/thread/block operations and workgroup
memory. [GPU dialect documentation](https://mlir.llvm.org/docs/Dialects/GPU/)
describes the target operations and memory model.

Important legality conditions in this benchmark:

- The two buffers are distinct. Flags and never-produced population slots
  have matching initial values and remain invariant.
- The source uses flattened scatter semantics. Clipping a geometrically
  out-of-range Y predecessor changes the benchmark.
- Collision expression order, FP32 persistent state and `-ffast-math`
  settings remain fixed. The source gate and bitwise tests form the oracle.
- Odd-step states are not externally consumed between fused steps. After
  fusion, the inactive buffer is state N−2, whereas one-step execution leaves
  N−1. A compiler must prove that omitted state dead, preserve it if live,
  or explicitly change the interface contract. Final-state agreement alone
  cannot justify dropping a live intermediate.
- This driver accepts positive even horizons. A general pass must preserve
  odd horizons, for example with a final one-step remainder.

Use matched S1 and temporal mappings to separate mapping overhead from the
complete fusion gain, and compare worker counts separately. V1/V2 change
both the shared representation and generated addressing, so their timing
difference is not a pure measurement of shared capacity or occupancy.

For a whole-simulation launch, investigate dependency-closed larger temporal
tiles or a supported global synchronization strategy. A plain loop around
these kernels with `gpu.barrier` does not synchronize blocks. Larger temporal
depth also expands producer regions and live storage. The measurements here
establish that hand fusion can help; they neither implement that full-horizon
algorithm nor establish an upper bound on attainable performance.
