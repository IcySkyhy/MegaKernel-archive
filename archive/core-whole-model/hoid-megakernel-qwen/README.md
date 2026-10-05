# Hoid megakernel for Qwen3 4b

## TL;DR 
We've built a megakernel for qwen3 4b model that beats SOTA inference engines on LLM decode. This repo holds the code for 1. reproducing the results 2. running a vLLM server instance with hoid's megakernel as a plugin. For fair comparison, all measurements have been done with hoid's megakernel integrated into vLLM, to normalize for scheduling overhead. 

![Grouped bar chart of decode interactivity in tokens per second per user at batch sizes 1, 2, 4 and 8. Hoid megakernel: 317, 283, 237, 176. vLLM 0.29.0: 255, 236, 208, 166. SGLang 0.5.21: 270, 241, 209, 169. TensorRT-LLM 1.2.1: 219, 198, 178, 143. Hoid is fastest at every batch size.](results/h200-8k-interactivity.svg)


## Requirements

- One NVIDIA H200 (141 GB, 132 SMs). The megakernel is built for exactly this GPU:
  `hoid-serve` and `hoid-bench` check the first visible GPU and refuse to run on any other.
- An NVIDIA driver for CUDA 13 (580 or newer), Linux x86-64.
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Run the server

```sh
uv sync
uv run hoid-serve
```

The first start downloads the checkpoint (8 GB) at its pinned revision. The server
listens on port 8000 and serves the model as `Qwen/Qwen3-4B-Instruct-2507`:

```sh
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}]
}'
```

Any extra arguments go to `vllm serve`, for example `uv run hoid-serve --port 9000`.

## Reproduce the benchmark

```sh
uv run hoid-bench
```

This reproduces the chart above. It builds the stock vLLM, SGLang and TensorRT-LLM
environments from the lock files in `bench/envs/`, computes the FP32 reference, and runs
64 measurements. Each is a fresh process, so the run takes about 1.5 hours. The results
land in `results/latest/RESULTS.md`; the published runs are in `results/`.

- **Subsets:** `--engines hoid,stock`, `--batches 1,8` and `--repeats 2` run part of the matrix.
- **Resuming:** rerunning resumes in the same `--out` directory, keeping finished reports.
- **Shared GPU:** `--lock FILE` holds an exclusive `flock` around every GPU process.



## License

See [LICENSE](LICENSE).
