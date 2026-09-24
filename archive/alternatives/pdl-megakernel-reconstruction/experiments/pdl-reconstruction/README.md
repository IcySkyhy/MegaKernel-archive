# PDL reconstruction experiments

## Measurement contract

The engineering note uses Llama-3.2-1B at batch one with a 32-token prompt
and 128 generated tokens. The active transformer body contains 16 repetitions
of opcodes `1 -> 2 -> 4 -> 5 -> 6`, followed by the LM head: 81 kernels and
80 graph edges in the complete direct path.

Body experiments use synthetic deterministic weights and validate
graph-versus-eager tensors at representative positions. Complete-step
experiments use verified real weights and separately time the transformer
body, body plus LM head, and embedding-to-token-publication scope.

## Implementations

| Harness name | Meaning |
|---|---|
| `direct_baseline` | extracted TK kernels with ordinary completion edges |
| `pdl_13page_issue_r96` | accepted 13-page/three-stage PDL chain |
| `pdl_13page_arrival_r96` | matched trigger-after-arrival control |
| `pdl_13page_issue_r96_vtp56` | opcode-5/6 shard-signal control |
| `pdl_s1_issue_r96` | correct one-stage resource control |

Four-page configurations are intentionally incorrect mechanism controls: they
alias shared-memory pages and must not be reported as model results.

## Build and smoke

Follow the root README to build the direct/PDL extensions, then run a
one-position smoke:

```bash
POSITION_END=32 WARMUP=1 ITERATIONS=2 \
  ./experiments/pdl-reconstruction/run_body.sh
```

For the matched trigger control:

```bash
IMPLEMENTATION=pdl_13page_arrival_r96 \
WARMUP=1 ITERATIONS=2 \
  ./experiments/pdl-reconstruction/run_body.sh
```

`run_body.sh` is the portable entry point. The exact machine-specific formal
launchers remain preserved in the dated rescue branches; they are omitted
from `main` so public users do not inherit private `/workspace` layouts.

The formal synthetic-weight body sweep uses the runner defaults: positions
32--158, five warmups, and 30 samples at each position. `POSITION_START`,
`POSITION_END`, `WARMUP`, and `ITERATIONS` bound a smoke run. For the ordinary
completion control, use `IMPLEMENTATION=direct_baseline`; the runner then
selects completion edges and reports no PDL edge mask.

## Real-weight end-to-end reproduction

The headline 983.127-us persistent control and 1009.711-us 81-kernel PDL
result use `unsloth/Llama-3.2-1B-Instruct` at immutable revision
`5a8abab4a5d6f164389b1079fb721cfab8d7126c`. The required
`model.safetensors` SHA-256 is
`1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`;
`run_e2e.sh` first validates the compiled bindings and then verifies the model
digest before loading the checkpoint.

First build the combined module with CUDA 12.8 and `PERSISTENT_VM=1` as shown
in the root README. Then run the formal P32/D128 contract:

```bash
MODEL_PATH=/path/to/verified/Llama-3.2-1B-Instruct \
  ./experiments/pdl-reconstruction/run_e2e.sh
```

The defaults are position 32, a 32-token prompt, 128 total output tokens,
three warmups, ten measured iterations, the accepted `13page_issue_r96`
body, programmatic graph edges, and mask `0x1f`. The runner first launches an
independent fixed-position process without `--allow-correctness-mismatch`.
At position 32 the v2 release contract embeds token 1000; this fixed-position
input is the token ID fed into that one decode step, not the token the model
is expected to generate. It is separate from the trajectory prompt token
base, which also happens to be 1000. The generic harness keeps its historical
default token 17, but its archived results belong to a different input
contract. Candidate graph
logits/hidden/K/V must pass the direct eager-vs-graph envelope and preserve
the eager token. Comparisons with the independently scheduled persistent VM,
including its token and tensor envelopes, remain recorded diagnostics. Only
after the direct dependency checks and all 80 graph-edge
rewrites pass does the P32/D128 latency process start. The input token,
numerical limits, and required checks are versioned in
`correctness_contract.json`; the correctness and binding contracts and result
are identified by SHA-256. Bit-exact replay and ordinary elementwise
`allclose` remain diagnostics because parallel BF16 atomic-add reduction
order can differ between launches. The runner outputs
`artifacts/local/e2e-p32-d128.correctness.preflight.json`,
`artifacts/local/e2e-p32-d128.correctness.json`, and
`artifacts/local/e2e-p32-d128.json`. Both result files record the preflight
sidecar's SHA-256, and the trajectory result also records the fixed-position
result's SHA-256.
A short plumbing smoke is:

```bash
MODEL_PATH=/path/to/verified/Llama-3.2-1B-Instruct \
OUTPUT_LENGTH=3 WARMUP=1 ITERATIONS=1 \
  ./experiments/pdl-reconstruction/run_e2e.sh
```

If the candidate module was built without the persistent entry, set
`NATIVE_REFERENCE_MK=/path/to/mk_llama...so` to a persistent module built from
this source tree with the same Python ABI and CUDA 12.8 toolchain. The external
persistent module's nvcc major/minor/build and CUDA-header version must match
the candidate modules exactly. The persisted preflight and both result JSON
files record the resolved module paths and compile provenance; each harness
process verifies that the loaded module hashes still match the preflight.

The formal command deliberately records rather than aborts on a free-running
trajectory token mismatch. Parallel atomic reduction order makes both arms
nondeterministic near low-margin argmax decisions, so full-trajectory token
equality is not the correctness oracle. Fixed-position tensor comparisons and
one-step checks are the correctness contract; the engineering note reports
this limitation explicitly. Exact graph-edge counts are also mandatory: the
published mask rewrites 80 full/body+LM edges and 79 body-only edges in every
decode graph. A short rewrite fails the run. The trajectory JSON records its
same-run matched persistent threshold, all edge counts, module provenance, and
the linked correctness artifact.

The release contract is opt-in at the harness level through
`--enforce-correctness-contract`; `run_e2e.sh` always enables it. Direct
harness invocations without that flag retain the generic synthetic,
non-P32, and capture-mode checks instead of being forced onto the blog's
checkpoint/position/token contract.

## Evidence map

| Engineering-note claim | Curated evidence |
|---|---|
| End-to-end 97.4% result | `results/end_to_end_reconstruction.csv` |
| 34.032-us issue-vs-arrival saving | `results/trigger-timing/` |
| Per-edge and matched-scope decomposition | `results/direct_tk_pdl_summary.md` |
| VTP regression | `results/vtp_signal_followup_summary.md` |
| Resource/microtile controls | `results/all_micro_summary.md` |

Raw event traces, build/SASS records, and repeated per-position JSON files are
not tracked in Git. They are published with provenance and SHA-256 manifests
in the [blog-v1.0 evidence release](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/releases/tag/blog-v1.0).

## Correctness note

The attention wait belongs to the partial that owns the newest KV block, not
necessarily the numerically last partial. The canonical source includes that
guard fix. P32/D128 has one attention partial, so the fix does not change any
reported measurement, but it is required for longer contexts.
