# Breaking Through on Blackwell (RTX 5090)

AlpinDale ported MegaQwen to the RTX 5090 and pushed it from 494 to 1000 tok/s.

## Key Changes from the RTX 3090 Version

1. Replaced cooperative groups with a persistent kernel using custom atomic barriers
2. Replaced two full-grid barriers with lightweight flag-based partial barriers
3. Added productive spin: 112 idle blocks prefetch 23 MB of MLP weights during attention using `prefetch.global.L2` PTX instructions

## Results

| Stage | tok/s | BW utilization |
|---|---|---|
| Cooperative port | 494 | 35.2% |
| + 5090 tuning | 813 | 57.9% |
| + Persistent atomic barriers | 890 | 63.4% |
| + Flag-based partial barriers | 905 | 64.4% |
| + Productive spin | 1000 | 71.2% |

## Hardware Differences

- 170 SMs (vs 82 on RTX 3090)
- 96 MB L2 cache (vs 6 MB)
- GDDR7 at 1674 GB/s achievable (vs 936 GB/s GDDR6X)

## Links

- Blog post: https://blog.alpindale.net/posts/5090_decode_optimization/
- Code: https://github.com/AlpinDale/qwen_megakernel
- Author: AlpinDale
