# Contributing

This project targets an experimental, pinned Lean CUDA compiler. Please keep changes reproducible
against the revisions recorded in `toolchain/lean-cuda.env` and `lake-manifest.json`.

## Setup

```bash
bash scripts/install-nightly.sh
./scripts/build.sh
```

The default setup uses the published binary, not compiler source. CI builds the library and\nlinks the chat worker on Linux x86-64 and AArch64. CUDA 13 is required. Full device tests need a compatible NVIDIA GPU; compilation-only success is
not sufficient evidence for numerical or training changes.

## Validation

Run the Python data-contract tests and shell syntax checks for every change:

```bash
python3 -m unittest discover -s examples/qwen38_train -p 'test_*.py' -v
python3 -m unittest discover -s examples/qwen38_chat -p 'test_*.py' -v
git ls-files '*.sh' -z | xargs -0 -n1 bash -n
git diff --check
```

Run `tests/workload_regressions/run.sh` for CUDA/model changes. Real-checkpoint parity and training
gates require `QWEN_MODEL_DIR`; report skipped hardware/model gates explicitly.

## Changes

- Keep generated binaries, CUDA translations, model weights, and local datasets out of Git.
- Update binary format magic/version, readers, writers, tests, and documentation together.
- Never silently substitute a current-policy score for a recorded behavior-policy score.
- Reject nonfinite training inputs/results before publishing a checkpoint.
- Use commit subjects in `type(scope): subject` form.

By contributing, you agree that your contribution is licensed under Apache-2.0.
