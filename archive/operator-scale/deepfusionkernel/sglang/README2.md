# SGLang with _DeepFusionKernel_

DeepFusionKernel leverage the optimizations of SGLang and achieves further decoding accelerations. Furthermore, a lightweight profiling-based kernel scheduler enables automatic optimal kernel selection at pre-inference time based on the incoming workload.


## Script Changes to SGLang

For learnability, we list our updates on the SGLang files. We only updated scripts in `./python/sglang/`.

### _DeepFusionKernel_ Implementation

_DeepFusionKernel_ adapted to SGLang's model implementation style is in `sglang/python/sglang/srt/layers/fused_gmlp.py`

### Models Adapted to _DeepFusionKernel_

- Llama model with _DeepFusionKernel_: `sglang/python/sglang/srt/models/llama_my_triton.py`
- Qwen2 model with _DeepFusionKernel_: `sglang/python/sglang/srt/models/qwen2_my_triton.py

### Benchmarking Scripts

- Vanilla SGLang and SGLang with _DeepFusionKernel_ but without the kernel scheduler can be evaluated with `sglang/python/sglang/bench_one_batch.py`.
- SGLang with _DeepFusionKernel_ and the kernel scheduler can be evaluated with `sglang/python/sglang/calibrate_throughput.py`.

### Other Changes

We also updates the following files:

- `sglang/python/sglang/srt/configs/model_config.py`: Added _DeepFusionKernel_ on/off to model config.
- `sglang/python/sglang/srt/hf_transformers_utils.py`: _DeepFusionKernel_-version model loading. 
- `sglang/python/sglang/srt/model_executor/cuda_graph_runner.py`: Updated CUDA Graphs capturing process at the warmup stage.
- `sglang/python/sglang/srt/model_executor/model_runner.py`: Disabled CUDA Graphs at scheduler profiling stage.
- `sglang/python/sglang/srt/server_args.py`: Automatic TP and PP size selection.


