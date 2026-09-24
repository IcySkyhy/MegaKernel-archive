# Hazy Research Megakernels (H100)

Stanford's Hazy Research group developed two megakernels using an instruction-interpreter model.

## Low-Latency Megakernel (Llama-1B)

Single GPU, batch size 1. The entire forward pass runs in one kernel.

Key ideas:
- Instructions: fine-grained units of work (one thread block's worth)
- Interpreter: on-SM loop that executes instructions from a pre-built queue
- Shared memory paging: 13 pages of 16 KB, explicitly requested and released
- Counter-based synchronization: array of integers in global memory, incrementing on completion

Results: 78% bandwidth utilization on H100. Sub-millisecond forward pass for a 1B+ parameter model.

Blog: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles

## High-Throughput Megakernel (Llama-70B)

8 GPUs, tensor-parallel, large batch sizes. Integrated into the Tokasaurus inference engine.

Key ideas:
- Distributed transpose: replicate O projection to eliminate post-attention reduce-scatter
- Global work queue: dynamic instruction scheduling across SMs
- Interleaving: overlap network-bound and compute-bound instructions
- Cross-GPU communication via NVLink stores from dedicated storer threads

Results: 22% faster than SGLang on 65,536 ShareGPT prompts.

Blog (intro): https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro
Blog (main): https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main

## Authors

Benjamin Spector, Jordan Juravsky, Stuart Sul, Dylan Lim, Owen Dugan, Simran Arora, Chris Re
