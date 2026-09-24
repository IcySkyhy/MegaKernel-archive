# Agent-facing schedule IR and feedback loop

Give a design agent structured levers—task graph, dependency, worker roles, placement, memory lifetime, and topology—instead of asking it to rewrite a monolithic kernel on every iteration. AutoMegaKernel, MPK, Machete, and MegaTritonKernel provide complementary representations.

Use a KDA-style task contract and candidate ledger. Correctness and performance must be measured at the consumer boundary; successful and well-explained rejected episodes feed back into the knowledge base.

Evidence: `ev-amk-counter-dag`, `ev-mpk-runtime-contract`, `ev-machete-formula-barriers`, `ev-megatriton-task-graph`.
