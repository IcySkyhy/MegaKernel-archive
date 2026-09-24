# Whole-generation boundary contrast

`model-as-a-kernel` includes prompt consumption, all forward phases, LM head, greedy argmax, and token feedback in `mega_kernel`. `qwen_megakernel` includes embedding, layers, and final norm in its main kernel, then launches LM head/argmax separately.

This pair is a useful boundary task because both are genuine model-scale kernels, yet only the former establishes a one-launch generation loop.

Evidence: `ev-mak-whole-generation`, `ev-qwen-boundary`, `ev-mak-phase-control`.
