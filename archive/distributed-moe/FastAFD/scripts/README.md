# FastAFD Scripts

This directory is organized by workflow.

## Public Entry Points

Quickstart:

```bash
./scripts/quickstart/qwen3_30b_a3b_sample.sh
```

Correctness alignment:

```bash
./scripts/validate/qwen3_30b_a3b_alignment.sh
./scripts/validate/qwen3_30b_a3b_fastafd_alignment.sh
```

Large-scale AFD preset:

```bash
./scripts/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh
```

Published Qwen3-235B and MiniMax M2.5 result presets live under
`scripts/experiments/afd/qwen3_235b/` and `scripts/experiments/afd/minimax_m25/`;
they dispatch through `scripts/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh`.

## Layout

- `quickstart/`: mini-sgl sampling workflow.
- `serve/`: mini-sgl, FastAFD, and vLLM server launchers.
- `validate/`: mini-sgl/vLLM and FastAFD/vLLM alignment pipelines.
- `experiments/afd/qwen3_30b/`: the FastAFD launcher the presets dispatch to.
- `experiments/afd/qwen3_235b/`: Qwen3-235B published-result presets.
- `experiments/afd/minimax_m25/`: MiniMax M2.5 published-result presets.
- `experiments/vllm/`: the sharded vLLM alignment scorer.
- `data_gen/`: prompt generation helper.
- `lib/`: shared shell helpers.
