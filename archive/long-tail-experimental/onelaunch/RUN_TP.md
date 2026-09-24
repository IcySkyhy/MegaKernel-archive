# Running the tensor-parallel benchmark (Part 3)

Part 3 needs **2 GPUs, ideally NVLink-connected** (e.g. a RunPod `2x H100 SXM`,
`2x A100 SXM`, or `2x RTX 4090` pod — though 4090s have no NVLink, so comm goes
over PCIe and the number will be worse). The sharding math is already verified on
a single GPU; this box only produces the multi-GPU latency number.

## Setup on the pod

```bash
cd onelaunch
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match the pod's CUDA
pip install triton transformers accelerate safetensors
pip install -e .   # or: export PYTHONPATH=src
```

## Verify sharding correctness (single GPU, cheap)

```bash
python -m onelaunch.model_tp            # expect: argmax 8/8 vs single-GPU oracle
```

## Run the 2-GPU benchmark

```bash
cd src
torchrun --nproc_per_node=2 -m onelaunch.model_tp_dist --bench
```

Rank 0 prints `decode X.XXX ms/token`. Compare against:

| point | ms/token | note |
|---|---|---|
| single-GPU weight-bandwidth floor | 3.063 | one GPU streams ~3.0 GB |
| single-GPU fused megakernel (measured) | 3.716 | Part 1 result |
| vLLM 0.11 single-GPU | 4.414 | SOTA reference |
| **ideal TP=2 floor** | **1.531** | each GPU streams ~1.5 GB |
| **TP=2 measured** | *(fill in)* | floor + un-overlapped NVLink all-reduce |

Record the number in `results/baselines_tp.json` under `pending_measurements`.

## What to look for (the honest story)

- **Does TP=2 beat the single-GPU 3.063 ms floor?** It should, because batch-1
  decode is weight-bandwidth-bound and each GPU now reads half the weights.
- **How far above the 1.531 ms ideal floor?** The gap is the NVLink all-reduce
  latency (2 per layer, tiny payloads) that can't be hidden at batch 1 — that's
  the honest cost of tensor parallelism for single-stream decode, and the number
  worth reporting plainly. On PCIe (no NVLink) expect a much larger gap.
- Optionally sweep `--nproc_per_node` 1/2/4 and plot ms/token vs GPU count against
  the per-GPU bandwidth floor to show where comm latency starts to dominate.
