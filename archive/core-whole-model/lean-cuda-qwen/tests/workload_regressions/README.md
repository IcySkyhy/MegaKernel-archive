# CUDA workload regressions

These tests cover the Qwen configuration provider, primitives, attention, DeltaNet, MLP, model, LoRA, training, safetensors, and persistent megakernel components in `lib/LeanCudaQwen`. They preserve elaboration, CUDA code-generation, runtime, and reference-parity gates.

Set `LEAN_CUDA_ROOT` to a staged CUDA-enabled Lean checkout. Run the complete suite with:

```bash
tests/workload_regressions/run.sh
```

Pass directory names to run a subset:

```bash
tests/workload_regressions/run.sh cuda_qwen36_config cuda_qwen36_weights
```

Tests skip CUDA compilation or execution when the required toolkit or GPU architecture is absent.
