# Deep Kernel Fusion for Transformers

___Deep Kernel Fusion for Transformers___ provides a series of deeply fused GPU kernels for SwiGLU MLP blocks in the Transformer architecture. It aims to accelerate autoregressive decoding speed, particularly in long-context agentic workloads, by reducing redundant memory accesses and improving cache locality.

This project contains scripts of _DeepFusionKernel_, a family of kernel fusion strategies, integrated into SGLang [[1]](#1) together with an automatic kernel scheduler that profiles and selects the optimal kernel at loading time based on workload and hardware characteristics.


## Getting started

The basic _DeepFusionKernel_ is implemented using Triton in `fused_transformer/fused_mlp/fused_gmlp.py`.

### Software pre-requisites

You will need to install __NVIDIA Nsight Compute__ for kernel performance profiling, you can find the installation guide here:

https://docs.nvidia.com/nsight-compute/index.html

Additionally, you'll need the dependencies listed in `requirements.txt`, which can be installed with pip:

> pip install -r requirements.txt

For _DeepFusionKernel_ integrated with SGLang, you'll need to install SGLang from the source code with pip:

```
cd sglang
pip install -e "python[all]"
```

### Running kernel profiling

Here we use the config with Llama 3.1-70B with TP=4 as an example. First, run Triton autotuning before profiling the performance:

```bash
python -m fused_transformer.fused_mlp.run_autotune --config-name config-probe-70B-sep_knls wandb.type=autotune best_config_path=BEST_AUTOTUNE_CONFIG_PATH
```

After autotuning, load the cached best configurations and profile the kernels with NVIDIA Nsight Compute (ncu):

```bash
ncu --export NCU_REPORT_FILENAME --force-overwrite --target-processes application-only --set roofline /PATH/TO/PYTHON -m fused_transformer.fused_mlp.profiling --config-name config-probe-70B-sep_knls best_config_path=BEST_AUTOTUNE_CONFIG_PATH wandb.type=profile
```

The profiling report can be checked with NVIDIA Nsight Compute.


### Running full-model evaluations

First, run Triton autotuning. 
Note that if tensor parallelism (TP) is used, the MLP weight matrices will be split across the intermediate dimension. We want to autotune for matrix of the size on a single GPU. Take Llama 3.1 70B running on 4 GPUs as an example:

```bash
python -m fused_transformer.fused_mlp.run_autotune --config-name config-probe-70B-sep_knls wandb.type=autotune best_config_path=BEST_AUTOTUNE_CONFIG_PATH intermediate_size=7168  # as TP=4
```

To evaluate _DeepFusionKernel_ via SGLang:

```bash
# run SGLang baseline
ENABLE_MY_TRITON=0 python -m sglang.bench_one_batch \
    --model-path meta-llama/Llama-3.1-70B \
    --batch 1 16 64 \
    --input-len 1 \
    --output-len 1024 \
    --attention-backend flashinfer \
    --dtype float16 \
    --run-name Llama3.1-70B-baseline
# run SGLang with DeepFusionKernel
ENABLE_MY_TRITON=1 python -m sglang.bench_one_batch \
    --model-path meta-llama/Llama-3.1-70B \
    --batch 1 16 64 \
    --input-len 1 \
    --output-len 1024 \
    --run-name Llama3.1-70B-deepfusionkernel \
    --attention-backend flashinfer \
    --dtype float16 \
    --best-config-pkl-path /data/BEST_AUTOTUNE_CONFIG_PATH
```

The kernel scheduler will predict performance of configurations not been profiled. To run with the kernel scheduler via SGLang, profiling on batch size e.g. 1, 8, 64:
```bash
ENABLE_MY_TRITON=1 python -m sglang.calibrate_throughput \
    --model-path meta-llama/Llama-3.1-70B \
    --batch 1 16 64 \
    --input-len 1 \
    --output-len 1024 \
    --run-name Llama3.1-70B-deepfusionkernel \
    --attention-backend flashinfer \
    --dtype float16 \
    --best-config-pkl-path /data/BEST_AUTOTUNE_CONFIG_PATH \ 
    --calibration-batch-size 1 8 64 \
```


## Open-source assets used

| Design | License |
|--|--|
| <a id="1">[1]</a> SGLang [GitHub](https://github.com/sgl-project/sglang/tree/v0.4.6.post4) | All code under `sglang/` is released under the [Apache-2.0 License](http://www.apache.org/licenses/).|
