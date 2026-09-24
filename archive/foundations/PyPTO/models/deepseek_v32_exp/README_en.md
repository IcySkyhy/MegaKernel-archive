# DeepSeek V3.2 Examples

This directory contains a series of PyPTO DeepSeek V3.2 EXP development sample code. We have decomposed DeepSeek-V3.2-Exp and delivered five operators: mla prolog, lightning indexer prolog, sparse flash attention, mla_indexer_prolog, and lightning indexer.

## Parameter Description/Constraints
- Shape format field meaning description
    | Field Name    | English Full Name/Meaning                  | Value Rules and Description                                                                 |
    |---------------|---------------------------------------------|----------------------------------------------------------------------------------------------|
    | b             | Batch (input sample batch size)             | Range: decode scenario 1 to 128, prefill scenario fixed to 1                                                          |
    | s1            | Query Seq-Length                            | Range: decode scenario 1 to 4, prefill scenario 1 to 1K                                                            |
    | s2            | Key Seq-Length                              | Range: 1 to 128K                                                             |
    | h             | Head-Size (hidden layer size)               | Fixed value: 7168                                                            |
    | n_q(n1)       | Query Head-Num (multi-head count)           | Range: 128                                       |
    | n_kv(n2)      | KV Head-Num                                 | Range: 1                                      |
    | kv_lora_rank  | KV low-rank matrix dimension                | Range: 512                                      |
    | rope_dim      | QK position encoding dimension              | Range: 64                                      |
    | v_head_dim    | Value head dimension                        | Range: 128                                      |
    | q_head_dim    | Query head dimension                        | Range: 192                                      |
    | q_lora_rank   | Query low-rank matrix dimension             | Range: 1536                                      |
    | idx_n_heads   | Indexer query head num                      | Fixed value: 64                                                             |
    | idx_head_dim  | Indexer query head dimension                | Fixed value: 128                                                             |
    | selected_count| Topk selected count                         | Fixed value: 2048                                                             |
    | block_num     | Number of per-tile blocks in PagedAttention scenario | Computed as `ceil(B*Skv/BlockSize)` (Skv represents the KV sequence length, allows 0) |
    | block_size    | Block size in PagedAttention scenario       | Range: 128                                                           |
    | t             | Size after BS axis merge                    | Range: b * s1|

# mla_prolog_quant

## Function Description

The MLA Prolog module transforms the hidden state $\bold{X}$ into the query projection $\bold{q}$, key projection $\bold{k}$, and value projection $\bold{v}$, with a structure consistent with DeepSeek V3 architecture. In the decode phase, it uses weight absorption technology.

## Computation Formulas

**RmsNorm Formula**
$$
\text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
$$
$$
\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
$$

**Path 1: Standard Query Computation**

Includes down-projection, RmsNorm, and two up-projections:
$$
c^Q = RmsNorm(x \cdot W^{DQ})
$$
$$
q^C = c^Q \cdot W^{UQ}
$$
$$
q^N = q^C \cdot W^{UK}
$$

**Path 2: Positional Encoding Query Computation**

Applies ROPE rotary position encoding to the Query:
$$
q^R = ROPE(c^Q \cdot W^{QR})
$$

**Path 3: Standard Key Computation**

Includes down-projection and RmsNorm, stores the result in cache:
$$
c^{KV} = RmsNorm(x \cdot W^{DKV})
$$
$$
k^C = Cache(c^{KV})
$$

**Path 4: Positional Encoding Key Computation**

Applies ROPE rotary position encoding to the Key and stores the result in cache:
$$
k^R = Cache(ROPE(x \cdot W^{KR}))
$$

## Function Prototype
```
def mla_prolog_quant_compute(token_x, w_dq, w_uq_qr, dequant_scale, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, cos,
        sin, cache_index, kv_cache, kr_cache, k_scale_cache, q_norm_out, q_norm_scale_out, query_nope_out,
        query_rope_out, kv_cache_out, kr_cache_out, k_scale_cache_out, epsilon_cq, epsilon_ckv, cache_mode,
        tile_config, rope_cfg):
```

## Parameter Description

-   **token_x** (`Tensor`): Input tensor used to compute Query and Key. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, h].
-   **w_dq** (`Tensor`): Down-projection weight matrix $W^{DQ}$ for Query computation. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `bfloat16`. Shape is [h, q_lora_rank].
-   **w_uq_qr** (`Tensor`): Up-projection weight matrix $W^{UQ}$ and positional encoding weight matrix $W^{QR}$ for Query computation. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `int8`. Shape is [q_lora_rank, n_q * q_head_dim].
-   **dequant_scale** (`Tensor`): Per-channel parameter for dequantizing w_uq_qr after MatmulQcQr matrix multiplication. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float`. Shape is [n_q*q_head_dim, 1].
-   **w_uk** (`Tensor`): Up-projection weight $W^{UK}$ for Key computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [n_q, qk_nope_head_dim, kv_lora_rank].
-   **w_dkv_kr** (`Tensor`): Down-projection weight matrix $W^{DKV}$ and positional encoding weight matrix $W^{KR}$ for Key computation. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `bfloat16`. Shape is [h, kv_lora_rank+rope_dim].
-   **gamma_cq** (`Tensor`): $\gamma$ parameter in the RmsNorm formula for computing $c^Q$. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [q_lora_rank].
-   **gamma_ckv** (`Tensor`): $\gamma$ parameter in the RmsNorm formula for computing $c^{KV}$. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [kv_lora_rank].
-   **cos** (`Tensor`): Cosine parameter matrix for rotary position encoding. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, rope_dim].
-   **sin** (`Tensor`): Sine parameter matrix for rotary position encoding. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, rope_dim].
-   **cache_index** (`Tensor`): Index for storing kv_cache and kr_cache. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int64`. Shape is [t].
-   **kv_cache** (`Tensor`): aclTensor for cache index, updated in place (corresponding to $k^C$ in the formula). Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. cache_mode is "PA_BSND". Shape is [block_num, block_size, n_kv, kv_lora_rank].
-   **kr_cache** (`Tensor`): Cache for key positional encoding, updated in place (corresponding to $k^R$ in the formula). Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `bfloat16`. Shape is [block_num, block_size, n_kv, rope_dim].
-   **k_scale_cache** (`Tensor`): Cache for key dequantization factors. Required parameter. Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `float`. Shape is [block_num, block_size, n_kv, 4].
-   **epsilon_cq** (`float`): $\epsilon$ parameter in the RmsNorm formula for computing $c^Q$. If the user does not specify, the recommended value is 1e-05. Supports double type only. Default value is 1e-05.
-   **epsilon_ckv** (`float`): $\epsilon$ parameter in the RmsNorm formula for computing $c^{KV}$. If the user does not specify, the recommended value is 1e-05. Supports double type only. Default value is 1e-05.
-   **cache_mode** (`str`): KV cache mode. Supports "PA_BSND".
-   **tile_config** (`class MlaTileConfig`): Tile split configuration.
-   **rope_cfg** (`class RopeTileShapeConfig`): Rope tile split configuration.

## Return Value Description
-   **q_norm_out** (`Tensor`): Output tensor after RmsNorm_cq on Query (corresponding to $q^C$). Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. Shape is [t, q_lora_rank].
-   **q_norm_scale_out** (`Tensor`): Dequantization parameter after RmsNorm_cq on Query. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float`. Shape is [t, 1].
-   **q_nope_out** (`Tensor`): Output tensor of Query in the formula (corresponding to $q^N$). Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, n_q, kv_lora_rank].
-   **q_rope_out** (`Tensor`): Output tensor of Query positional encoding in the formula (corresponding to $q^R$). Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, n_q, rope_dim].
-   **kv_cache_out** (`Tensor`): Tensor output of Key to `kv_cache` (corresponding to $k^C$). Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `int8`. Shape is [block_num, block_size, n_kv, kv_lora_rank].
-   **kr_cache_out** (`Tensor`): Tensor output of Key positional encoding to `kr_cache` (corresponding to $k^R$). Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `bfloat16`. Shape is [block_num, block_size, n_kv, qk_rope_dim].
-   **k_scale_cache_out** (`Tensor`): Dequantization parameter output after Key dequantization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float`. cache_mode is "PA_BSND". Shape is [block_num, block_size, n_kv, 4].

## Call Sample

- For details, refer to [deepseekv32_mla_prolog_quant.py](deepseekv32_mla_prolog_quant.py).

# lightning_indexer_prolog
## Function Description

Used in Deepseek IndexerAttention to compute the query, key, and weights required for Lightning Indexer.
The quantization strategy for Indexer Prolog is as follows: Q_b_proj uses W8A8 quantization, and all other Linear layers do not use quantization. Query uses A8 quantization. Key (cache) uses C8 quantization. Dequantization factors are stored as FP16. Weights are stored as FP16.

## Computation Formulas
**Query Computation Formula:**

The Q computation uses dynamic Per-Token-Head quantization, where the Hadamard transform is implemented by right-multiplying the matrix by hadamard_q. Both $\bold{q}$ and $\bold{w}_{qb}$ are of type Int8.

$$
\bold{q}, \bold{q}_{scale} = \text{DynamicQuant}(\text{Hadamard}(\text{RoPE}(\text{DeQuant}(\bold{q} \cdot \bold{w}_{qb}))))
$$

**Key(cache) Computation Formula:**

The Cache computation also uses dynamic Per-Token-Head quantization, where the Hadamard transform is implemented by right-multiplying the matrix by hadamard_k.

$$
\bold{k}, \bold{k}_{scale} = \text{DynamicQuant}(\text{Hadamard}(\text{RoPE}(\text{LayerNorm}(\bold{x} \cdot \bold{w}_k))))
$$

**Weights Computation Formula:**

The Weights computation does not use quantization and must be converted to the FP16 data type for subsequent Lightning Indexer computation.

$$
\bold{weight} = (\bold{x} \cdot \bold{w}_{proj}) * \text{scale}
$$

## Function Prototype

```
def lightning_indexer_prolog_quant_compute(x_in, q_norm_in, q_norm_scale_in, w_qb_in, w_qb_scale_in, wk_in, w_proj_in,
                ln_gamma_k_in, ln_beta_k_in, cos_idx_rope_in, sin_idx_rope_in, hadamard_q_in, hadamard_k_in, k_int8_in, k_scale_in,
                k_cache_index_in, q_int8_out, q_scale_out, k_int8_out, k_scale_out, weights_out, attrs, configs):
```

## Parameter Description

-   **x_in** (`Tensor`): Hidden state token. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, h].
-   **q_norm_in** (`Tensor`): Quantized query after rmsnorm. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. Shape is [t, q_lora_rank].
-   **q_norm_scale_in** (`Tensor`): Dequantization factor for query. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. Shape is [t, 1].
-   **wq_b_in** (`Tensor`): Query weight. Required parameter. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `int8`. Shape is [q_lora_rank, idx_n_heads*idx_head_dim].
-   **wq_qb_scale_in** (`Tensor`): Weight dequantization factor for query. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. Shape is [idx_n_heads*idx_head_dim, 1].
-   **wk_in** (`Tensor`): Key weight. Required parameter. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `bfloat16`. Shape is [h, idx_head_dim].
-   **w_proj_in** (`Tensor`): Weights weight. Required parameter. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `bfloat16`. Shape is [h, idx_n_heads].
-   **ln_gamma_k_in** (`Tensor`): Key layernorm scale. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [idx_head_dim].
-   **ln_beta_k_in** (`Tensor`): Key layernorm shift. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [idx_head_dim].
-   **cos_idx_rope_in** (`Tensor`): Cosine for RoPE. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, rope_dim].
-   **sin_idx_rope_in** (`Tensor`): Sine for RoPE. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, rope_dim].
-   **hadamard_q_in** (`Tensor`): Weight matrix for query Hadamard transform. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [idx_head_dim, idx_head_dim].
-   **hadamard_k_in** (`Tensor`): Weight matrix for key Hadamard transform. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [idx_head_dim, idx_head_dim].
-   **k_int8_in** (`Tensor`): Key cache (k_cache). Required parameter. Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `int8`. Shape is [block_num, block_size, n_kv, idx_head_dim].
-   **k_scale_in** (`Tensor`): Key dequantization factor cache. Required parameter. Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `float16`. Shape is [block_num, block_size, n_kv, 1].
-   **k_cache_index_in** (`Tensor`): Position for updating the key cache. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int64`. Shape is [t].
-   **attrs.layernorm_epsilon_k** (`float`): Key layernorm epsilon to prevent division by zero. Required parameter. Data type supports `float32`.
-   **attrs.layout_query** (`str`): Optional parameter identifying the data layout format of the input `query`. Default value is "TND". Currently supports only "TND".
-   **attrs.layout_key** (`str`): Optional parameter identifying the data layout format of the input `key`. Default value is "PA_BSND". Currently supports only "PA_BSND".
-   **configs** (`class IndexerPrologQuantConfigs`): Tile split configuration.

## Return Value Description

-   **q_int8_out** (`Tensor`): Output tensor of query in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. Shape is [t, idx_n_heads, idx_head_dim].
-   **q_scale_out** (`Tensor`): Output tensor of query dequantization factor in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float16`. Shape is [t, idx_n_heads, 1].
-   **k_int8_out** (`Tensor`): Output tensor of key cache (k_cache). Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `int8`. Shape is [block_num, block_size, n_kv, idx_head_dim].
-   **k_scale_out** (`Tensor`): Output tensor of key dequantization factor cache. Does not support non-contiguous tensors. cache_mode is "PA_BSND". Data format supports ND. Data type supports `float16`. Shape is [block_num, block_size, n_kv, 1].
-   **weights_out** (`Tensor`): Output tensor of weights in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float16`. Shape is [t, idx_n_heads].

## Call Sample
- For operator source code execution, refer to [deepseekv32_lightning_indexer_prolog_quant.py](deepseekv32_lightning_indexer_prolog_quant.py).

# sparse_flash_attention_quant

## Function Description

For each query token $\bold{x}_i$, the indexing module computes a relevance score $I_{i,j}$ for each key-value cache item (representing a key-value pair or MLA latent representation). It then computes the output $\bold{o}_i$ by applying the attention mechanism to the query token $\bold{x}_i$ and the top $k$ cache items with the highest scores:

## Computation Formula

$$
\bold{o}_i = \text{Attn}(\bold{x}_i, \{\bold{c}_j | j \in \text{Top-k}(\bold{I}_{i, :})\})
$$

## Function Prototype

```
def sparse_flash_attention_quant_compute(query_nope, query_rope, key_nope_2d, key_rope_2d, k_nope_scales,
        topk_indices, block_table, kv_act_seqs, attention_out, nq, n_kv, softmax_scale, topk, block_size,
        max_blocknum_perbatch, tile_config):
```

## Parameter Description

-   **query_nope** (`Tensor`): Required parameter. RoPE information of query in MLA structure. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t * n_q, kv_lora_rank].
-   **query_rope** (`Tensor`): Required parameter. Nope information of query in MLA structure. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t * n_q, rope_dim].
-   **key_nope_2d** (`Tensor`): Required parameter. RoPE information of key in MLA structure. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. Shape is [block_num * block_size, kv_lora_rank].
-   **key_rope_2d** (`Tensor`): Required parameter. Nope information of key in MLA structure. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [block_num * block_size, rope_dim].
-   **k_nope_scales** (`Tensor`): Required parameter. Dequantization scaling factor for k_nope. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float`. Shape is [block_num * block_size, 4].
-   **topk_indices** (`Tensor`): Required parameter. Topk index selected for each token. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int32`. Shape is [t, n_kv * selected_count].
-   **block_table** (`Tensor`): Required parameter. Block mapping table used for KV storage in PageAttention. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int32`. Shape is [b, s2_max/block_size], where the second dimension indicates a length not less than the number of blocks corresponding to the largest s2 among all batches (that is, ceiling of s2_max / block_size).
-   **kv_act_seqs** (`Tensor`): Required parameter. Data format supports ND. Indicates the number of valid tokens for `key` and `value` in different batches. Data type supports `int32`. Shape is [b].
-   **nq** (`int`): Required parameter. Scaling coefficient used as the scalar value for Muls after query and key matrix multiplication. Data type supports float.
-   **n_kv** (`int`): Required parameter. Scaling coefficient used as the scalar value for Muls after query and key matrix multiplication. Data type supports float.
-   **softmax_scale** (`float`): Required parameter. Scaling coefficient used as the scalar value for Muls after query and key matrix multiplication. Data type supports float.
-   **topk** (`int`): Required parameter. Number of selected tokens. Data type supports int.
-   **block_size** (`int`): Required parameter. Block size for the sparse phase. Data type supports int.
-   **max_blocknum_perbatch** (`int`): Required parameter. Maximum number of blocks per batch. Data type supports int.
-   **tile_config** (`class SaTileShapeConfig`): TileShapeConfig configuration structure. Represents tile split configuration. Configuration item data type supports int.

## Return Value Description

-   **attention_out** (`Tensor`): Output in the formula. Data format supports ND. Data type supports `bfloat16`. Output shape is [b, s1, n_q, kv_lora_rank].

## Call Sample

- For details, refer to [deepseekv32_sparse_flash_attention_quant.py](deepseekv32_sparse_flash_attention_quant.py).

# sparse_attention_antiquant

## Function Description

sa_antiquant is an optimization based on sfa_quant that stores in 8-bit and computes in 16-bit. In the sfa_quant scenario, key_nope_2d, key_rope_2d, and k_nope_scales are of types int8, bf16, and fp32 respectively. During subsequent attention computation, these three tensors are stored discretely, requiring three discrete memory access instructions to separately dequantize and concatenate. In contrast, sa_antiquant merges the nope, rope, and nope_scale of the same token along the trailing axis, requiring only one discrete memory access instruction. This saves b * s * topk discrete memory access instructions in total, reducing load instructions and improving load efficiency.

## Function Prototype

```
def sparse_attention_antiquant_compute(query_nope, query_rope, nope_cache, topk_indices, block_table,
        kv_act_seqs, attention_out, nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch,
        tile_config):
```

## Parameter Description

-   **query_nope** (`Tensor`): Required parameter. RoPE information of query in MLA structure. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t * n_q, kv_lora_rank].
-   **query_rope** (`Tensor`): Required parameter. Nope information of query in MLA structure. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t * n_q, rope_dim].
-   **nope_cache** (`Tensor`): Required parameter. Dequantization scaling factor for key in MLA structure. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. Shape is [block_num * block_size, kv_lora_rank + rope_dim * 2 + 4 * scale_size], where scale_size = 4.
-   **topk_indices** (`Tensor`): Required parameter. Topk index selected for each token. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int32`. Shape is [t, n_kv * selected_count].
-   **block_table** (`Tensor`): Required parameter. Block mapping table used for KV storage in PageAttention. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int32`. Shape is [b, s2_max/block_size], where the second dimension indicates a length not less than the number of blocks corresponding to the largest s2 among all batches (that is, ceiling of s2_max / block_size).
-   **kv_act_seqs** (`Tensor`): Required parameter. Data format supports ND. Indicates the number of valid tokens for `key` and `value` in different batches. Data type supports `int32`. Shape is [b].
-   **nq** (`int`): Required parameter. Scaling coefficient used as the scalar value for Muls after query and key matrix multiplication. Data type supports float.
-   **n_kv** (`int`): Required parameter. Scaling coefficient used as the scalar value for Muls after query and key matrix multiplication. Data type supports float.
-   **softmax_scale** (`float`): Required parameter. Scaling coefficient used as the scalar value for Muls after query and key matrix multiplication. Data type supports float.
-   **topk** (`int`): Required parameter. Number of selected tokens. Data type supports int.
-   **block_size** (`int`): Required parameter. Block size for the sparse phase. Data type supports int.
-   **max_blocknum_perbatch** (`int`): Required parameter. Maximum number of blocks per batch. Data type supports int.
-   **tile_config** (`class SaTileShapeConfig`): TileShapeConfig configuration structure. Represents tile split configuration. Configuration item data type supports int.

## Return Value Description

-   **attention_out** (`Tensor`): Output in the formula. Data format supports ND. Data type supports `bfloat16`. Output shape is [b * s1 * n_q, kv_lora_rank].

## Call Sample

- For details, refer to [deepseekv32_sparse_attention_antiquant.py](deepseekv32_sparse_attention_antiquant.py).

# mla_indexer_prolog_quant

## Function Description

The MLA Indexer Prolog module fuses the MLA Prolog and Lightning Indexer Prolog operators at a larger scope, implementing pipeline parallelism between operators and improving operator performance.

## Function Prototype
```
def mla_indexer_prolog_quant_compute(
    token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
    mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
    mla_k_scale_cache, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
    ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in, ip_hadamard_k_in,
    ip_k_cache, ip_k_cache_scale, mla_query_nope_out, mla_query_rope_out,
    mla_q_norm_out, mla_q_norm_scale_out, mla_kv_cache_out, mla_kr_cache_out,
    mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out, ip_k_int8_out,
    ip_k_scale_out, ip_weights_out, mla_epsilon_cq, mla_epsilon_ckv,
    mla_cache_mode, mla_tile_config,
    ip_attrs, ip_configs, rope_cfg
):
```

## Parameter Description

-   **token_x** (`Tensor`): Input tensor used to compute Query and Key. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, h].
-   **mla_w_dq** (`Tensor`): Down-projection weight matrix $W^{DQ}$ for Query computation. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `bfloat16`. Shape is [h, q_lora_rank].
-   **mla_w_uq_qr** (`Tensor`): Up-projection weight matrix $W^{UQ}$ and positional encoding weight matrix $W^{QR}$ for Query computation. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `int8`. Shape is [q_lora_rank, n_q*q_head_dim].
-   **mla_dequant_scale** (`Tensor`): Per-channel parameter for dequantizing w_uq_qr after MatmulQcQr matrix multiplication. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float`. Shape is [n_q*q_head_dim, 1].
-   **mla_w_uk** (`Tensor`): Up-projection weight $W^{UK}$ for Key computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [n_q, qk_nope_head_dim, kv_lora_rank].
-   **mla_w_dkv_kr** (`Tensor`): Down-projection weight matrix $W^{DKV}$ and positional encoding weight matrix $W^{KR}$ for Key computation. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `bfloat16`. Shape is [h, kv_lora_rank+rope_dim].
-   **mla_gamma_cq** (`Tensor`): $\gamma$ parameter in the RmsNorm formula for computing $c^Q$. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [q_lora_rank].
-   **mla_gamma_ckv** (`Tensor`): $\gamma$ parameter in the RmsNorm formula for computing $c^{KV}$. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [kv_lora_rank].
-   **cos** (`Tensor`): Cosine parameter matrix for rotary position encoding. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, rope_dim].
-   **sin** (`Tensor`): Sine parameter matrix for rotary position encoding. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, rope_dim].
-   **cache_index** (`Tensor`): Index for storing kv_cache and kr_cache. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int64`. Shape is [T].
-   **mla_kv_cache** (`Tensor`): aclTensor for cache index, updated in place (corresponding to $k^C$ in the formula). Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `int8`. Shape is [block_num, block_size, n_kv, kv_lora_rank].
-   **mla_kr_cache** (`Tensor`): Cache for key positional encoding, updated in place (corresponding to $k^R$ in the formula). Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `bfloat16`. Shape is [block_num, block_size, n_kv, rope_dim].
-   **mla_k_scale_cache** (`Tensor`): Cache for key dequantization factors. Required parameter. Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `float`. Shape is [block_num, block_size, n_kv, 4].
-   **ip_w_qb_in** (`Tensor`): Query weight. Required parameter. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `int8`. Shape is [q_lora_rank, idx_n_heads*idx_head_dim].
-   **ip_w_qb_scale_in** (`Tensor`): Weight dequantization factor for query. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. Shape is [idx_n_heads*idx_head_dim, 1].
-   **ip_wk_in** (`Tensor`): Key weight. Required parameter. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `bfloat16`. Shape is [h, idx_head_dim].
-   **ip_w_proj_in** (`Tensor`): Weights weight. Required parameter. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `bfloat16`. Shape is [h, idx_n_heads].
-   **ip_ln_gamma_k_in** (`Tensor`): Key layernorm scale. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [idx_head_dim].
-   **ip_ln_beta_k_in** (`Tensor`): Key layernorm shift. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [idx_head_dim].
-   **ip_hadamard_q_in** (`Tensor`): Weight matrix for query Hadamard transform. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [idx_head_dim, idx_head_dim].
-   **ip_hadamard_k_in** (`Tensor`): Weight matrix for key Hadamard transform. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [idx_head_dim, idx_head_dim].
-   **ip_k_cache** (`Tensor`): Key cache. Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. cache_mode is "PA_BSND". Shape is [block_num, block_size, n_kv, idx_head_dim].
-   **ip_k_cache_scale** (`Tensor`): Key dequantization factor cache. Required parameter. Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `float16`. Shape is [block_num, block_size, n_kv, 1].
-   **mla_epsilon_cq** (`float`): $\epsilon$ parameter in the RmsNorm formula for computing $c^Q$. If the user does not specify, the recommended value is 1e-05. Supports double type only. Default value is 1e-05.
-   **mla_epsilon_ckv** (`float`): $\epsilon$ parameter in the RmsNorm formula for computing $c^{KV}$. If the user does not specify, the recommended value is 1e-05. Supports double type only. Default value is 1e-05.
-   **mla_cache_mode** (`str`): KV cache mode. Supports "PA_BSND".
-   **mla_tile_config** (`class MlaTileConfig`): Tile split configuration for the mla subgraph.
-   **ip_attrs** (`class IndexerPrologQuantAttr`): Attribute values required for lightning indexer prolog subgraph computation, including layernorm_epsilon_k, layout\_query, layout\_key.
-   **ip.layernorm_epsilon_k** (`float`): Key layernorm epsilon to prevent division by zero. Required parameter. Data type supports `float32`.
-   **ip.layout_query** (`str`): Optional parameter identifying the data layout format of the input `query`. Default value is "TND". Currently supports only "TND".
-   **ip.layout_key** (`str`): Optional parameter identifying the data layout format of the input `key`. Default value is "PA_BSND". Currently supports only "PA_BSND".
-   **ip_config** (`class IndexerPrologQuantConfigs`): Tile split configuration and dynamic binning configuration for the ip subgraph.
-   **rope_cfg** (`class RopeTileShapeConfig`): Tile split configuration and dynamic binning configuration for the rope subgraph.

## Return Value Description
-   **mla_query_nope_out** (`Tensor`): Output tensor of Query in the formula (corresponding to $q^N$). Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, n_q, kv_lora_rank].
-   **mla_query_rope_out** (`Tensor`): Output tensor of Query positional encoding in the formula (corresponding to $q^R$). Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, n_q, rope_dim].
-   **mla_q_norm_out** (`Tensor`): Output after rmsnorm transformation and quantization of the Query positional encoding output tensor in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. Shape is [t, q_lora_rank].
-   **mla_q_norm_scale_out** (`Tensor`): Dequantization coefficient output after rmsnorm transformation and quantization of the Query positional encoding output tensor in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. Shape is [t, 1].
-   **mla_kv_cache_out** (`Tensor`): Tensor output of Key to `kv_cache` (corresponding to $k^C$). Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `int8`. Shape is [block_num, block_size, n_kv, kv_lora_rank].
-   **mla_kr_cache_out** (`Tensor`): Tensor output of Key positional encoding to `kr_cache` (corresponding to $k^R$). Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `bfloat16`. Shape is [block_num, block_size, n_kv, qk_rope_dim].
-   **mla_k_scale_cache_out** (`Tensor`): Dequantization parameter output after Key dequantization. Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `float`. Shape is [block_num, block_size, n_kv, 4].
-   **ip_q_int8_out** (`Tensor`): Output tensor of query in the formula. Data format supports ND. Data type supports `int8`. Shape is [t, idx_n_heads, idx_head_dim].
-   **ip_q_scale_out** (`Tensor`): Output tensor of query dequantization factor in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float16`. Shape is [t, idx_n_heads, 1].
-   **ip_k_int8_out** (`Tensor`): Output tensor of key cache (k_cache). Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `int8`. Shape is [block_num, block_size, n_kv, idx_head_dim].
-   **ip_k_scale_out** (`Tensor`): Output tensor of key dequantization factor cache. Does not support non-contiguous tensors. Data format supports ND. cache_mode is "PA_BSND". Data type supports `float16`. Shape is [block_num, block_size, n_kv, 1].
-   **ip_weights_out** (`Tensor`): Output tensor of weights in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float16`. Shape is [t, idx_n_heads].

## Call Sample

- For details, refer to [deepseekv32_mla_indexer_prolog_quant.py](deepseekv32_mla_indexer_prolog_quant.py).

# lightning_indexer

## Function Description

Lightning Indexer obtains the top-$k$ positions for each token through a series of operations. For an Index Query $Q_{index}\in\R^{g\times d}$ corresponding to a token, given the context Index Key $K_{index}\in\R^{S_{k}\times d}, W\in\R^{g\times 1}$, where $g$ is the group size for GQA, $d$ is the dimension of each head, and $S_{k}$ is the context length, the specific computation formula of Lightning Indexer is as follows:
$$
\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
$$

## Function Prototype

```
def lightning_indexer_decode_compute(
    idx_query, idx_query_scale, idx_key_cache, idx_key_scale, idx_weight, act_seq_key, block_table, topk_res,
    unroll_list, configs, selected_count):
```

## Parameter Description

-   **idx_query** (`Tensor`): Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. Shape is [t, n_q, idx_head_dim].
-   **idx_query_scale** (`Tensor`): Required parameter. Scaling factor for idx_query. Data format supports ND. Data type supports `float16`. Shape is [t, n_q, idx_head_dim].
-   **idx_key_cache** (`Tensor`): Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. Shape is [t, n_kv, idx_head_dim].
-   **idx_key_scale** (`Tensor`): Required parameter. Scaling factor for idx_key_cache. Data format supports ND. Data type supports `float16`. Shape is [t, n_kv, idx_head_dim].
-   **idx_weight** (`Tensor`): Required parameter. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float16`. Supports input shape [t, n_q].
-   **act_seq_key** (`Tensor`): Required parameter. Indicates the number of valid `key` tokens in different batches. Data type supports `int32`. Shape is [b].
-   **block_table** (`Tensor`): Required parameter. Block mapping table used for KV storage in PageAttention. Data format supports ND. Data type supports `int32`. Shape is [b, ceilDiv(max(s2), block_size)], where max(s2) is the maximum value in s2 and ceilDiv indicates ceiling division.
-   **unroll_list** (`List`): Optional parameter. Multi-gear split configuration.
-   **configs** (`class LightningIndexerConfigs`): Optional parameter. LightningIndexerConfigs configuration structure. Represents tile split configuration and optimization options.
-   **selected_count** (`int`): Required parameter. Topk selection count. Default is 2048.

## Return Value Description

-   **topk_res** (`Tensor`): Output in the formula. Data type supports `int32`. Data format supports ND. Output shape is [t, n_kv, selected_count].

## Call Sample

- For details, refer to [deepseekv32_lightning_indexer_quant.py](deepseekv32_lightning_indexer_quant.py).
