# Lean CUDA Qwen

Lean 4 CUDA model components, persistent kernels, checkpoint loading, LoRA training, reference fixtures, and runnable applications for Qwen3.6 and Qwen3.8. The repository uses a staged CUDA-enabled Lean compiler, CUDA 13, and NVIDIA GPUs for device execution.

## Library

`lib/LeanCudaQwen` contains the application-level components used across examples. The library
builds with Lake against the repository's pinned CUDA-enabled Lean compiler.

| Module | Components |
| --- | --- |
| `LeanCudaQwen.Qwen36` | Qwen3.6/3.8 configuration, numerical model, CUDA primitives, LoRA training, checkpoint loading, and persistent execution. |

## Reproducible setup

Install the pinned public [Lean CUDA nightly binary](https://github.com/ranvier-labs/lean4-cuda-nightly)
on Linux x86-64 or AArch64. No compiler source checkout is needed:

```bash
bash scripts/install-nightly.sh
./scripts/build.sh
QWEN_CUDA_ARCH=sm_90a examples/qwen38_chat/build_worker.sh
```

The installer verifies the release SHA-256 checksum and installs under
`.lake/toolchains/nightly-2026-09-08`. The release tag and version are pinned in
`toolchain/lean-cuda.env`; `lean-toolchain` points to this local installation.
The binary is selected automatically, or explicitly with `LEAN_CUDA_TOOLCHAIN_ROOT`.

Prerequisites: Git, curl, zstd, a C/C++ toolchain, LLVM's `lld` linker, GMP/OpenSSL/libuv development libraries,
and CUDA 13 for native/device compilation. Building does not require a GPU or model weights.
For execution, choose `QWEN_CUDA_ARCH` for your NVIDIA GPU (the default is `sm_121a`)
and provide a compatible checkpoint.

CI builds the complete Lean library and compiles and links the CUDA chat worker on hosted
x86-64 and AArch64 runners using only the published binary and CUDA toolkit. It also runs
Python contract tests. These are build gates, not GPU numerical-parity or training gates.
The other regression/training harnesses retain their source-backend build recipes.

## Backend checkout

For compiler development and the source-based regression/training harnesses, the original
backend setup remains available to users with access to the companion compiler repository:

```bash
./scripts/setup-backend.sh
```

To reuse an existing checkout of the pinned source revision, set `LEAN_CUDA_ROOT`
(and unset `LEAN_CUDA_TOOLCHAIN_ROOT` if previously exported):

```bash
export LEAN_CUDA_ROOT=/path/to/lean4-cuda-backend
./scripts/build.sh
```

The harness rejects a different Git revision, a missing staged `lean`/`leanc`/`lake`, or a compiler
whose `lean --features` does not report `[CUDA]`. `LEAN_CUDA_ALLOW_UNPINNED=1` is an explicit
development-only escape hatch. Sibling checkout discovery remains as a convenience but is subject
to the same validation.

## Applications

| Example | What it demonstrates |
| --- | --- |
| `qwen36_megakernel/` | Exact Qwen3.6-27B recurrent Gated DeltaNet decode geometry in a cooperative-grid persistent launch, checked against separate launches and a host Lean oracle. |
| `qwen38_chat/` | Persistent real-checkpoint Qwen3.8 chat worker, HTTP bridge, browser client, command-line client, and correctness-gated decode benchmark. |
| `qwen38_train/` | Projection-wide Qwen3.8-27B LoRA SFT, DPO, and outcome-GRPO over frozen packed BF16 checkpoint buffers, with separate full SFT/GRPO megakernels, a DPO CUDA graph, bounded gradient diagnostics, and exact trainer-state resume. |


## Running

Build the model library and run the pure elaboration configuration gate:

```bash
./scripts/build.sh
tests/workload_regressions/run.sh cuda_qwen36_config
```

Run the complete migrated regression suite:

```bash
tests/workload_regressions/run.sh
```

Run the compact megakernel example or the real-checkpoint chat application:

```bash
examples/qwen36_megakernel/run.sh
examples/qwen38_chat/run.sh
```

CUDA runners compile without device execution when the required GPU is absent. Generated C, CUDA, binaries, measurements, and local model data are ignored.

## Status

This repository tracks the Qwen application layer. Backend attributes, the launch ABI, and the build
recipe follow the companion CUDA-enabled Lean compiler.

## License

Repository code is licensed under [Apache-2.0](LICENSE). Third-party source notices are
preserved. Model checkpoints are not distributed here and retain their respective licenses.
