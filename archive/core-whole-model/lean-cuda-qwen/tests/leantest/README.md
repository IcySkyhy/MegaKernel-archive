# Qwen3.6 CUDA LeanTest suite

This package uses the pinned [LeanTest](https://github.com/cpehle/lean_test) framework to run every
plain-Lean CUDA Qwen3.6 boundary as a registered `@[test]`. Tests are sequential by default because
they share one GPU and example-local generated artifacts.

First run `../../scripts/setup-dependencies.sh`. Then run the complete suite from this directory
through the pinned CUDA-toolchain wrapper:

```bash
./run.sh --fail-fast
```

Run one case by declaration-name substring:

```bash
./run.sh --filter projectionMegakernel --fail-fast
```

The wrapper and every child runner honor `LEAN_CUDA_ROOT`, validate it against the repository's
immutable backend revision, and otherwise use the repo-local backend installed by
`scripts/setup-backend.sh`. Override the example repository location with `LEAN_CUDA_QWEN_ROOT`
when the suite is invoked outside this worktree.
