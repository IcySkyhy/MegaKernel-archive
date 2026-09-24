# DeepGEMM MegaMoE lineage

The original DeepGEMM MegaMoE design, the TIRx implementation-stack port, and FlashInfer's integrated backend should not be treated as three independent discoveries in training/evaluation splits. Preserve separate project and backend cards, but group the design lineage.

Evidence: `ev-deepgemm-megamoe-boundary`, `ev-flashinfer-mega-split`; relations: `rel-tirx-deepgemm`, `rel-flashinfer-deepgemm`.
