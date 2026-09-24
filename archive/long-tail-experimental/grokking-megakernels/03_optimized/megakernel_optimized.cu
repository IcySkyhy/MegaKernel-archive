/*
 * This is the same kernel as 02_megakernel/megakernel.cu.
 *
 * The optimizations described in Chapters 5-6 (redundant RMSNorm,
 * block divergence with L2 prefetch, 128-bit vectorized loads)
 * are already present in the production kernel.
 *
 * See the book for a walkthrough of each optimization and its impact:
 *   - Redundant RMSNorm: +42% (eliminates 56 grid.sync calls)
 *   - Block divergence + L2 prefetch: +2x (uses idle blocks during attention)
 *   - 128-bit vectorized loads: +3.5%
 *
 * The original unoptimized kernel is not included in this repo because
 * it was developed iteratively. The DEVLOG at the MegaQwen repo documents
 * the full optimization journey:
 *   https://github.com/Infatoshi/megaqwen/blob/main/DEVLOG.md
 *
 * To see what the 5090 (Blackwell) version looks like with persistent
 * kernels and atomic barriers instead of cooperative groups, see:
 *   https://github.com/AlpinDale/qwen_megakernel
 */
