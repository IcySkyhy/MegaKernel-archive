# Role-specialized MegaMoE

MoE supplies natural device roles: dispatch/receive, schedule, expert compute, gather, and combine. Token or tile readiness lets a later role start before the whole prior stage completes. The design win comes from removing critical-path stage barriers and overlapping fabric traffic with compute, not only from saving launches.

DeepGEMM, Mixture-of-Kittens, TeraMoE, Ascend PTO, and TPU Pallas show different hardware realizations. Their stage boundaries, training/serving scope, memory model, and topology are not interchangeable.

Evidence: `ev-deepgemm-megamoe-boundary`, `ev-mok-forward-backward`, `ev-teramoe-five-roles`, `ev-pto-aic-aiv-pipeline`, `ev-pallas-fused-ep-moe`.
