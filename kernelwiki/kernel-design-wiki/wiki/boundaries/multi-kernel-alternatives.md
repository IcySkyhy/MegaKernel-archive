# Multi-kernel alternatives

PDL, CUDA Graph, and multiple persistent kernels on coordinated streams can recover much of the launch/overlap benefit while preserving modular kernels. They remain distinct launch semantics and should be modeled as alternatives rather than relabeled MegaKernels.

Evidence: `ev-pdl-alternative`, `ev-streamep-alternative`, `ev-mak-phase-control`.
