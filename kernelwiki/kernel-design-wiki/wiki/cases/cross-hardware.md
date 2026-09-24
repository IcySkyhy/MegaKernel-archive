# Cross-hardware device-resident substrates

CUDA cooperative grids are not the only qualifying substrate. AWS NKI Transformer TKG, TPU Pallas Fused EP MoE, and Ascend mixed AIC/AIV pipelines keep multiple model or communication stages inside one device invocation.

Transfer the semantic axes—boundary, task unit, dependency, role, memory, topology—not CUDA-specific implementation details.

Evidence: `ev-nki-transformer-tkg`, `ev-pallas-fused-ep-moe`, `ev-pto-aic-aiv-pipeline`.
