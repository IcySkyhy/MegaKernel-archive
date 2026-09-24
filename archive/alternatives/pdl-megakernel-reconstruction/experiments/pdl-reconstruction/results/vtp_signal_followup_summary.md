# Opcode5-to-opcode6 virtual-TP signal follow-up

Date: 2026-07-23
GPU: NVIDIA H100 80 GB HBM3
Workload: Llama-3.2-1B, batch 1, BF16, P32/D128
Timing mode: one CUDA Graph per decode step; timeline, profiler, and in-kernel
timing records disabled.

## Wait implementation

The opcode6 activation dependency follows the original megakernel pattern:

- only `warpid() == 0 && laneid() == 0` polls the global shard counter;
- ThunderKittens defines these as `threadIdx.x >> 5` and
  `threadIdx.x & 31`, so exactly `threadIdx.x == 0` waits per CTA;
- the remaining consumer warps wait at the existing consumer-group barrier;
- opcode5 publishes the original four counters only after its asynchronous
  output store is globally complete.

The 20/64/128 ns variants change only the single waiter's `__nanosleep`
immediate. SASS confirmed `NANOSLEEP 0x14` for the 20 ns entry. The opcode6
variants use 89 registers and have no local-memory spills. The compact
opcode5 producer uses 92 registers, 16 barriers, and has no spills.

## P32/D128 result

One formal W3/I10 trajectory trial:

| Candidate | Body (us) | Body + LM (us) | Full (us) | Body delta vs base |
|---|---:|---:|---:|---:|
| Whole-grid PDL base | 810.414 | 977.227 | 1010.251 | 0.000 |
| Interleaved signal wait, 1 ns | 821.079 | 988.654 | 1021.571 | +10.665 |
| Interleaved signal wait, 64 ns | 821.041 | 988.557 | 1021.587 | +10.627 |
| Shard-local signal wait, 64 ns | 816.501 | 983.890 | 1016.895 | +6.087 |

The 64 ns backoff is indistinguishable from 1 ns at D128. Shard-local
producer placement recovers 4.578 us of the interleaved VTP regression, but
the comparison changes producer placement and therefore is not sufficient to
claim a signal-wait benefit.

## Matched producer controls

Short P32/D2 W5/I30 controls use identical opcode5 producer placement for the
whole-grid and signal-wait variants.

### Contiguous shard-local, 128 CTA

| Candidate | Body (us) | Full (us) |
|---|---:|---:|
| Whole-grid base, original producer | 831.958 | 1013.691 |
| Whole-grid wait, compact contiguous shard-local producer | 835.895 | 1018.321 |
| Signal wait, same compact contiguous producer | 838.701 | 1019.875 |

With identical producers, signal wait is 2.805 us slower in the body.

### Striped shard-local, 128 CTA

The striped schedule keeps each CTA inside one 2048-element shard, while the
four load rounds across the 32 CTAs cover contiguous block ranges. This
removes the producer-layout penalty without giving up 32-SM shard waves.

| Candidate | Body (us) | Full (us) |
|---|---:|---:|
| Whole-grid base, original producer | 833.498 | 1014.454 |
| Whole-grid wait, compact striped shard-local producer | 833.665 | 1017.783 |
| Signal wait, same compact striped producer | 836.741 | 1019.622 |

The matched producer body is within 0.167 us of the original base, while
signal wait is still 3.076 us slower than its exact whole-grid control.

### Formal striped P32/D128 control

The final control repeats the two striped 128-CTA variants in six fresh
processes (three per variant), with three warmups and ten timed generations
per process. Runs are interleaved by variant. Timeline collection, profiler
collection, and device-side timing records are disabled.

| Three-process mean | Body (us) | Native body (us) | Body gap (us) | Full step (us) |
|---|---:|---:|---:|---:|
| Whole-grid wait | 810.059 +/- 0.185 | 778.017 +/- 0.267 | **32.042 +/- 0.090** | 1010.177 +/- 0.268 |
| Per-shard signal wait | 814.945 +/- 0.022 | 777.869 +/- 0.151 | **37.077 +/- 0.172** | 1015.225 +/- 0.056 |

Values after `+/-` are sample standard deviations across fresh processes.
With an identical producer, the raw split body rises by 4.886 us. Normalizing
each process to its matched persistent body, the signal substitution increases
the residual by 5.035 us. The corresponding full-step gaps are 27.346 +/- 0.163
us and 32.485 +/- 0.048 us.

All short-run candidates matched native output tokens and were repeat
deterministic. Free-running D128 token hashes remain nondeterministic for both
the extracted TK and native atomic-reduction paths; no arithmetic or precision
was changed.

## Decision

Reject the global-counter replacement as a performance candidate.

- Signal publication itself is cheap: the earlier signal-only control was
  within about 0.7 us of the whole-grid base at D128.
- Wait backoff is not the missing factor.
- Aligning shard readiness with 32-SM producer release is useful and explains
  about 4.6 us of the original regression.
- After removing empty CTAs and the producer-layout penalty, the per-shard
  signal wait remains slower than `cudaGridDependencySynchronize` with the
  exact same producer: about 3 us in the short control and 4.9 us in the
  formal P32/D128 body.

The remaining distinction is the persistent megakernel's resident
loader/consumer roles and global task scheduler. A separate-grid PDL chain can
admit the successor loader, but a global spin signal does not reproduce the
persistent scheduler's low-cost task handoff.

## Artifacts

The build log and raw directories are in the
[blog-v1.0 evidence release](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/releases/tag/blog-v1.0)
under `direct-tk-pdl/`.

- Source SHA256:
  `b59fa016423497a4646c8ce64e1cb04df2470d07062e6d6918ce32eec50866f1`
- Harness SHA256:
  `c260850fd547abd994869e2a6cd96a4e03ac1ad37c8e3a5bda95c89b12172b07`
- Remote module SHA256:
  `24669fbe70b3f82f75bb4173aa9a431e9a4a554cadec43b78f1f20e5c2c7324e`
- Build log: `build_logs/vtp_compact_signal_build_20260723.log`
- Raw results:
  `wait_backoff_short/`, `wait_backoff_formal_d128/`,
  `shardlocal_short/`, `shardlocal_formal_d128/`,
  `compact_shardlocal_short/`, `compact_signal_control_short/`, and
  `striped_compact_short/`, and `striped_matched_formal_d128/`.
