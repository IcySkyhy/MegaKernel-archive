# Compressed_Flash_Attention

## Function Description

- API Function: The `CompressedFlashAttention` operator aims to complete the Attention computation described by the following formula, supporting Compressed Attention.
- Computation Formula:

  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$

  Where $\tilde{K}=\tilde{V}$ is the $KV$ that actually participates in computation, controlled by input parameters such as kv_cache and kv_win.

## Function Prototype

```
torch.ops.pypto.compress_flash_attention(
    q,
    cmp_kv,
    sinks,
    cmp_block_table,
    seqused_kv,
    ori_kv,
    ori_block_table,
    cmp_ratio
) -> (Tensor)
```

## Parameter Description

- **q** (`Tensor`): Required parameter. Corresponds to $Q$ in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_query` is BSND, shape is [B*S1, N1, D], where N1 supports only 64.
- **cmp_kv** (`Tensor`): Required parameter. Part of $\tilde{K}$ and $\tilde{V}$ in the formula, representing compressed KV. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_kv` is PA_ND, shape is [block\_num, cmp\_block\_size, KV\_N, D], where block\_num2 is the total number of blocks for PageAttention, cmp\_block\_size is the number of tokens in a block (multiple of 16, maximum 1024).
- **sinks** (`Tensor`): Required parameter. Attention sink tensor. Data format supports ND. Data type supports `float32`. Shape is [N1].
- **cmp_block_table** (`Tensor`): Required parameter. Block mapping table used for cmpKvCache storage in PageAttention. Data format supports ND. Data type supports `int32`. Shape is 2-dimensional, where the first dimension length is B and the second dimension length is not less than the number of blocks corresponding to the largest S3 among all batches (that is, ceiling of S3\_max / block\_size).
- **seqused_kv** (`Tensor`): Required parameter. Indicates the number of tokens in `ori_kv` that actually participate in computation for different batches. Dimension is B. Data format supports ND. Data type supports `int32`. If not provided, all tokens participate in the computation.
- **ori_kv** (`Tensor`): Required parameter. Part of $\tilde{K}$ and $\tilde{V}$ in the formula, representing original uncompressed KV. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_kv` is PA_ND, shape is [block\_num1, ori\_block\_size, KV\_N, D], where block\_num1 is the total number of blocks for PageAttention, ori\_block\_size is the number of tokens in a block (multiple of 16, maximum 1024), and KV\_N supports only 1.
- **ori_block_table** (`Tensor`): Required parameter. Block mapping table used for oriKvCache storage in PageAttention. Data format supports ND. Data type supports `int32`. Shape is 2-dimensional, where the first dimension length is B and the second dimension length is not less than the number of blocks corresponding to the largest S2 among all batches (that is, ceiling of S2\_max / block\_size).
- **cmp_ratio** (`int`): Required parameter. Compression ratio for ori_kv. Data type supports `int`. Supports 128.

## Return Value Description

- **attention\_out** (`Tensor`): Output in the formula. Data format supports ND. Data type supports `bfloat16`. Shape is [B, S1, N1, D].

## Constraint Description

- This interface supports inference scenarios only.
- This interface supports aclgraph mode.
- The D value in parameter q, seqused_kv, and kv_cache must be equal and is 512.
- The data types of parameters seqused_kv and kv_cache must be consistent.
- This interface supports decode scenarios only, not prefill scenarios.
- block_size supports 128.

## Invocation Method

```
python3  models/deepseek_v4/test_compress_flash_attention.py
```

# Compressor
## Function Description

- API Function: The Compressor compresses the KV cache of every 4 or 128 tokens into one, and then performs DSA computation between each token and these compressed KV caches. In long sequence scenarios, the Compressor can effectively reduce computation overhead.
- The main computation process is:

  1. Perform Matmul on input $X$ and $W^{KV}$ to obtain $kv\_state$, perform Matmul on input $X$ and $W^{Gate}$ and then add with $Ape$ to obtain $score\_state$, and update $kv\_state$ and $score\_state$ based on the input start_pos.
  2. Rearrange $kv\_state$ and $score\_state$, perform softmax on $score\_state$, multiply the softmax result with $kv\_state$, and then perform ReduceSum.
  3. Perform RmsNorm and ROPE operations based on input data norm_weight, rope_sin, rope_cos, and optionally perform Hadamard Transform based on rotate, to obtain the $cmp\_kv$ result output.

## Function Prototype

```
torch.ops.pypto.compressor(
    x,
    kv_state,
    score_state,
    kv_block_table,
    state_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    hadamard,
    start_pos,
    ratio,
    rope_head_dim,
    rotate
) -> (Tensor)
```

## Parameter Description

- **x** (`Tensor`): Required parameter. Original uncompressed data, corresponding to $X$ in the formula. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `bfloat16`. Supports input shape [B, S, H].
- **kv\_state** (`Tensor`): Required parameter. Historical data of kv\_state, corresponding to $kv\_state$ in the formula. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `float32`. Supports input shape [block_num, block_size, coff*D].
- **score\_state** (`Tensor`): Required parameter. Historical data of score\_state, corresponding to $score\_state$ in the formula. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `float32`. Supports input shape [block_num, block_size, coff*D].
- **kv\_block\_table** (`Tensor`): Required parameter. Page table for historical data in kv\_state. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `int32`. Supports input shape [B, ceil(max_S/block_size)].
- **score\_block\_table** (`Tensor`): Required parameter. Page table for historical data in score\_state. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `int32`. Supports input shape [B, ceil(max_S/block_size)].
- **sin** (`Tensor`): Required parameter. Weight coefficient for Rope computation. Data type supports `bfloat16`. Supports input shape [min(T, T//ratio+B), rope_head_dim].
- **cos** (`Tensor`): Required parameter. Weight coefficient for Rope computation. Data type supports `bfloat16`. Supports input shape [min(T, T//ratio+B), rope_head_dim].
- **wkv** (`Tensor`): Required parameter. Weight parameter for KV and compression weight, corresponding to $W^{KV}$ in the formula. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `bfloat16`. Supports input shape [coff*D, H].
- **wgate** (`Tensor`): Required parameter. Weight parameter for KV and compression weight, corresponding to $W^{Gate}$ in the formula. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `bfloat16`. Supports input shape [coff*D, H].
- **ape** (`Tensor`): Required parameter. Input positional biases, corresponding to $Ape$ in the formula. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `float32`. Supports input shape [ratio, coff*D].
- **weight** (`Tensor`): Required parameter. Weight coefficient for RmsNorm computation. Data type supports `bfloat16`. Supports input shape [D,].
- **start\_pos** (`Tensor`): Optional parameter. Starting position for computation. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `int32`. Supports input shape [B,]. When the input is None, computation starts from 0.
- **hadamard** (`Tensor`): Optional parameter. Weight matrix for Hadamard Transform. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `bfloat16`. Supports input shape [D, D].
- **ratio** (`int`): Required parameter. Data compression ratio. Supports 4/128.
- **rope\_head\_dim** (`int`): Required parameter. Minimum hidden layer unit for rope_cos and rope_sin. Currently supports only 64.
- **rotate** (`bool`): Required parameter. Indicates whether an additional Hadamard Transform is needed.

## Return Value Description

- **out** (`Tensor`): Required output. Compressed data. Does not support non-contiguous tensors. Data format supports $ND$. Data type supports `bfloat16`. Supports output shape [min(T, T // ratio + B), D]. The output data value for uncompressed entries is zero.

## Constraint Description

- This interface supports B generalization.
- S supports 1/2/3/4.
- D supports 128/512.
- H supports 4096.
- block_size supports 128.

## Invocation Method

```
python3  models/deepseek_v4/test_compressor.py
```

# Quant_Lightning_Indexer_Prolog

## Function Description

- API Function: The `QuantLightningIndexerProlog` operator aims to complete the Prolog computation described by the following formula, mainly providing input q, weight, and q_scale for subsequent LightningIndexer computation.
- Computation Formula:

  The computation formula for q and q_scale is:

  $$
  q\_tmp = \text{qr}@{idx\_wq\_b} \cdot \text{qr\_scale} \cdot \text{idx\_wq\_b\_scale}
  $$

  $$
  q\_hadamard = \text{Cat}(\{q\_tmp[:, :nope\_dim], Rope(q\_tmp[:, nope\_dim:])\}, -1)@hadamard
  $$

  $$
  q, q\_scale = Quant(q\_hadamard)
  $$

  Where Rope represents rotary position encoding computation and Quant represents quantization computation.
  The computation formula for Weights is:

  $$
  weights = x@\text{weights\_proj} \cdot {\frac{1}{\sqrt{\text{idx\_nq} \cdot \text{head\_dim}}}}
  $$

## Function Prototype

```
torch.ops.pypto.quant_lightning_indexer_prolog(
    qr,
    idx_wq_b,
    x,
    weights_proj,
    cos,
    sin,
    hadamard,
    qr_scale,
    idx_wq_b_scale
) -> (Tensor, Tensor, Tensor)
```

## Parameter Description

- **qr** (`Tensor`): Required parameter. Left input for q matrix computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. When `layout_query` is TND, shape is [t, q_lora_rank].
- **idx_wq_b** (`Tensor`): Required parameter. Right input for q matrix computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. When `layout_query` is TND, shape is [q_lora_rank, idx_nq*head_dim].
- **x** (`Tensor`): Required parameter. Left input for weights matrix computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_query` is TND, shape is [t, h].
- **weights_proj** (`Tensor`): Required parameter. Right input for weights matrix computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_query` is TND, shape is [h, idx_nq].
- **cos** (`Tensor`): Required parameter. Used for q positional encoding computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_query` is TND, shape is [t, rope_dim].
- **sin** (`Tensor`): Required parameter. Used for q positional encoding computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_query` is TND, shape is [t, rope_dim].
- **hadamard** (`Tensor`): Required parameter. Right input for q Hadamard matrix computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_query` is TND, shape is [head_dim, head_dim].
- **qr_scale** (`Tensor`): Required parameter. Dequantization coefficient input after qr matrix computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. When `layout_query` is TND, shape is [t, 1].
- **idx_wq_b_scale** (`Tensor`): Required parameter. Multiplication input after qr matrix computation. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. When `layout_query` is TND, shape is [idx_nq * head_dim, 1].

## Return Value Description

- **q** (`Tensor`): Required output. Output q in the formula. Data format supports ND. Data type supports `int8`. When layout\_query is TND, shape is [t, idx_nq * head_dim].
- **weights** (`Tensor`): Required output. Output weights in the formula. Data format supports ND. Data type supports `float16`. When layout\_query is TND, shape is [t, idx_nq].
- **q_scale** (`Tensor`): Required output. Output q_scale in the formula. Data format supports ND. Data type supports `float16`. When layout\_query is TND, shape is [t, idx_nq].

## Constraint Description

- This interface supports inference scenarios only.
- This interface supports aclgraph mode.
- q_lora_rank, idx_nq, head_dim, h, rope_dim support default values only. t supports [1 to 64k].
- All input and output data layouts support TND only.
- All input and output data types support only the listed scenarios, not additional types.

## Invocation Method

```
python3  models/deepseek_v4/test_lightning_indexer_prolog_quant.py
```

# Mla_Prolog

## Function Description

The MLA Prolog module transforms the hidden states $x$ into $Query$ and ${Key-Value}$.

## Computation Formula

1. $Query(q)$ Computation
   The Query computation includes two down-sampling steps and RmsNorm (where the second RmsNorm weight is always 1), and finally performs inplace interleaved rope computation on the last rope\_dim dimensions along the -1 axis:

$$
c^Q = RmsNorm(x @ wq\_a)
$$

$$
q = RmsNorm(c^Q @ wq\_b)
$$

$$
q[..., -rope\_dim:] = ROPE(q[..., -rope\_dim:])
$$

2. $Key-Value(kv)$ Computation
   The kv computation includes one down-sampling step and RmsNorm, and finally performs inplace interleaved rope computation on the last rope\_dim dimensions along the -1 axis:

$$
kv = RmsNorm(x @ wkv)
$$

$$
kv[..., -rope\_dim:] = ROPE(kv[..., -rope\_dim:])
$$

## Function Prototype

```
torch.ops.pypto.mla_prolog_quant(
    token_x,
    wq_a,
    wq_b,
    wkv,
    rope_cos,
    rope_sin,
    gamma_cq,
    gamma_ckv,
    wq_b_scale
) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameter Description

- **token_x** (`Tensor`): Input tensor for computing Query and Key-Value in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, h].
- **wq_a** (`Tensor`): Down-sampling weight matrix $wq_a$ for Query computation in the formula. Data format supports NZ/ND. Data type supports `bfloat16`. Shape is [h, q_lora_rank].
- **wq_b** (`Tensor`): Up-sampling weight matrix $wq_b$ for Query computation in the formula. Data format supports NZ/ND. Data type supports `int8`. Shape is [q_lora_rank, num_heads*head_dim].
- **wkv** (`Tensor`): Down-sampling weight matrix $wkv$ for Key-Value computation in the formula. Data format supports NZ/ND. Data type supports `bfloat16`. Shape is [h, head_dim].
- **rope_cos** (`Tensor`): Cosine parameter matrix for rotary position encoding. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, rope_dim].
- **rope_sin** (`Tensor`): Sine parameter matrix for rotary position encoding. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, rope_dim].
- **gamma_cq** (`Tensor`): $\gamma$ parameter in the RmsNorm formula for computing $c^Q$. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [q_lora_rank].
- **gamma_ckv** (`Tensor`): $\gamma$ parameter in the RmsNorm formula for computing $c^{KV}$. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [head_dim].
- **wq_b_scale** (`Tensor`): Per-channel parameter for dequantizing wq_b after matrix multiplication. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float`. Shape is [num_heads*head_dim, 1].

## Return Value Description

- **q_out** (`Tensor`): Output tensor of Query in the formula (corresponding to $q$). Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, num_heads, head_dim].
- **kv_out** (`Tensor`): Output tensor of Key-Value in the formula (corresponding to $kv$). Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [t, head_dim].
- **qr_out** (`Tensor`): Output tensor after the first rmsnorm and quantization on Query (corresponding to $c^Q$). Does not support non-contiguous tensors. Data format supports ND. Data type supports `int8`. Shape is [t, q_lora_rank].
- **qr_scale_out** (`Tensor`): Output tensor after the first rmsnorm on Query (corresponding to $c^Q$). Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. Shape is [t, 1].

## Constraint Description

- This interface supports inference scenarios only.
- This interface supports aclgraph mode.
- head_dim supports 512, h supports 4096, q_lora_rank supports 1024, num_heads supports 64, rope_dim supports 64.
- t supports the range [1, 64k].
- A5 does not support the int8 quantized version yet.
- For the non-quantized implementation, refer to the sample.

## Invocation Method

```
Quantized:
python3  models/deepseek_v4/test_mla_prolog_quant_v4.py

Non-quantized:
python3  models/deepseek_v4/test_mla_prolog_v4.py
```

# Sliding_Window_Attention

## Function Description

- API Function: The `SlidingWindowAttention` operator aims to complete the Attention computation described by the following formula, supporting Sliding Window Attention.
- Computation Formula:

  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$

  Where $\tilde{K}=\tilde{V}$ is the $KV$ that actually participates in computation, controlled by input parameters such as kv_cache and kv_win.

## Function Prototype

```
torch.ops.pypto.sliding_window_attention(
    q,
    ori_block_table,
    ori_kv,
    seqused_kv,
    sinks,
    win_size,
    mask,
    cu_seqlens_q
) -> (Tensor)
```

## Parameter Description

- **q** (`Tensor`): Required parameter. Corresponds to $Q$ in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [T1, N1, D], where N1 supports only 64 and D supports only 512.
- **ori_block_table** (`Tensor`): Required parameter. Block mapping table used for oriKvCache storage in PageAttention. Data format supports ND. Data type supports `int32`. Shape is 2-dimensional, where the first dimension length is B and the second dimension length is not less than the number of blocks corresponding to the largest S2 among all batches (that is, ceiling of S2\_max / block\_size). block\_size supports only 128.
- **ori_kv** (`Tensor`): Required parameter. Original KV. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [block\_num1, block\_size, N2, D], where block\_num1 is the total number of blocks for PageAttention, block\_size is the number of tokens in a block (supports only 128), and N2 supports only 1.
- **seqused_kv** (`Tensor`): Required parameter. Indicates the input sample sequence length S2 of `ori_kv` for different batches. Dimension is B. Data format supports ND. Data type supports `int32`.
- **sinks** (`Tensor`): Required parameter. Attention sink tensor. Data format supports ND. Data type supports `float32`. Shape is [N1].
- **win_size** (`Int`): Required parameter. Window size. Data type supports `int32`. Supports only 128.
- **mask** (`Tensor`): Required parameter. Mask used during computation. Data type supports `bool`. The generation method is fixed; call the get_mask method. Shape is [4 * N1, 4 * block\_size], where N1 supports only 64 and block\_size supports only 128.
- **cu_seqlens_q** (`Tensor`): Required parameter. Indicates the number of valid tokens for `q` in different batches. Dimension is B+1. The value of each element in the parameter represents the sum of the token counts of the current batch and all previous batches (that is, prefix sum). Therefore, the value of a later element must be greater than or equal to the value of the previous element. Data type supports `int32`.

## Return Value Description

- **atten\_out** (`Tensor`): Attention computation result. Data format supports ND. Data type supports `bfloat16`. Shape is [T1, N1, D].

## Constraint Description

- This interface supports inference scenarios only.
- This interface supports aclgraph mode.
- The D value in parameter q and ori_kv must be equal and is 512.
- The data types of parameters q and ori_kv must be consistent.
- block_size supports 128.

## Invocation Method

```
python3  models/deepseek_v4/test_win_attention.py
```

# Sparse_Compress_Flash_Attention

## Function Description

- API Function: The `SparseCompressFlashAttention` operator aims to complete the Attention computation described by the following formula, supporting Sparse Compressed Attention.
- Computation Formula:

  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$

  Where $\tilde{K}=\tilde{V}$ is the $KV$ that actually participates in computation, controlled by input parameters such as ori_kv, cmp_kv, and cmp_kv.

## Function Prototype

```
torch.ops.pypto.sparse_compress_flash_attention(
    query,
    q_act_seqs,
    ori_kv,
    cmp_kv,
    ori_block_table,
    cmp_block_table,
    atten_sink,
    seqused_kv,
    cmp_sparse_indices,
    softmax_scale,
    win_size,
    cmp_ratio
) -> (Tensor)
```

## Parameter Description

- **query** (`Tensor`): Required parameter. Corresponds to $Q$ in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [T1*N1, D], where N1 supports only 64.
- **q_act_seqs** (`Tensor`): Required parameter. Takes effect when `layout_query` is TND. Indicates the number of valid tokens for `q` in different batches. Dimension is B+1. The value of each element in the parameter represents the sum of the token counts of the current batch and all previous batches (that is, prefix sum). Therefore, the value of a later element must be greater than or equal to the value of the previous element. Data type supports `int32`.
- **ori_kv** (`Tensor`): Required parameter. Part of $\tilde{K}$ and $\tilde{V}$ in the formula, representing original uncompressed KV. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_kv` is PA_ND, shape is [block\_num1* ori\_block\_size, KV\_N*D], where block\_num1 is the total number of blocks for PageAttention, ori\_block\_size is the number of tokens in a block (supports 128), and KV\_N supports only 1.
- **cmp_kv** (`Tensor`): Required parameter. Part of $\tilde{K}$ and $\tilde{V}$ in the formula, representing compressed KV. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_kv` is PA_ND, shape is [block\_num2* cmp\_block\_size, KV\_N*D], where block\_num2 is the total number of blocks for PageAttention, cmp\_block\_size is the number of tokens in a block (supports 128).
- **ori_block_table** (`Tensor`): Required parameter. Block mapping table used for oriKvCache storage in PageAttention. Data format supports ND. Data type supports `int32`. Shape is 2-dimensional, where the first dimension length is B and the second dimension length is not less than the number of blocks corresponding to the largest S2 among all batches (that is, ceiling of S2\_max / block\_size).
- **cmp_block_table** (`Tensor`): Required parameter. Block mapping table used for cmpKvCache storage in PageAttention. Data format supports ND. Data type supports `int32`. Shape is 2-dimensional, where the first dimension length is B and the second dimension length is not less than the number of blocks corresponding to the largest S3 among all batches (that is, ceiling of S3\_max / block\_size).
- **atten_sink** (`Tensor`): Required parameter. Attention sink tensor. Data format supports ND. Data type supports `float32`. Shape is [N1].
- **seqused_kv** (`Tensor`): Required parameter. Indicates the number of tokens in `ori_kv` that actually participate in computation for different batches. Dimension is B. Data format supports ND. Data type supports `int32`. If not provided, all tokens participate in the computation.
- **cmp_sparse_indices** (`Tensor`): Required parameter. Represents the index for discretely fetching cmpKvCache. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int32`. When `layout_query` is TND, shape must be [Q\_T * KV\_N, K2], where K2 is the number of tokens selected at once from `cmp_kv`. K2 supports only 512.
- **softmax_scale** (`double`): Required parameter. Scaling coefficient used as the scalar value for Muls after matrix multiplication of q with ori_kv and cmp_kv. Data type supports `float`.
- **win_size** (`int`): Required parameter. Window size. Data type supports int32. Supports only 128.
- **cmp_ratio** (`int`): Required parameter. Compression ratio for ori_kv. Data type supports `int`. Supports 4.

## Return Value Description

- **attention\_out** (`Tensor`): Output in the formula. Data format supports ND. Data type supports `bfloat16`. Shape is [T1*N1, D].

## Constraint Description

- This interface supports inference scenarios only.
- This interface supports aclgraph mode.
- The D value in parameter q, ori_kv, and cmp_kv must be equal and is 512.
- The data types of parameters q, ori_kv, and cmp_kv must be consistent.
- To improve operator performance, high-dimensional axis merging is applied to q, ori_kv, cmp_kv, and attention_out.
- Supports TND format only.
- block_size supports 128.

## Invocation Method

```
python3  models/deepseek_v4/test_sparse_compress_flash_attention.py
```

# hc_pre

## Function Description

- API Function: The hc_pre operator aims to complete the following computation process.
- Computation Process:

1. Compute the denominator of RMSNorm

$$
rsqrt = \sqrt{\frac{1}{\frac{1}{n}\sum_{i=1}^n x_i^2 + \epsilon}}
$$

2. Compute mixes

$$
mixes = (x @ hc\_fn) \odot rsqrt
$$

3. Sinkhorn-Knopp Algorithm

$$
pre, post, comb = sinkhorn(mixes, hc\_scale, hc\_base, hc\_mult, hc\_sinkhorn\_iters)
$$

Each iteration of the Sinkhorn-Knopp algorithm performs row-wise normalization followed by column-wise normalization. $hc\_sinkhorn\_iters$ controls the number of iterations.

4. Compute y using pre and x

$$
y = rowsum(pre \odot x)
$$

## Function Prototype

```
torch.ops.pypto.hc_pre(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    hc_mult: int=4,
    hc_split_sinkhorn_iters: int=20,
    hc_eps: float=1e-6
) -> (Tensor, Tensor, Tensor)
```

## Parameter Description

- **x** (`Tensor`): Required parameter. Corresponds to $x$ in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. When `layout_x` is TND, shape is [t, hc_mult, h].
- **hc_fn** (`Tensor`): Required parameter. Corresponds to $hc\_fn$ in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. When `layout_x` is TND, shape is [mix_hc, hc_mult*h], where mix_hc = (2+hc_mult)*hc_mult.
- **hc_scale** (`Tensor`): Required parameter. Corresponds to $hc\_scale$ in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. Shape is [3, ].
- **hc_base** (`Tensor`): Corresponds to $hc\_base$ in the formula. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. Shape is [mix_hc, ].
- **hc_mult** (`int`): Optional parameter. Expansion rate in mHC. Data type supports `int`. Default is `4`.
- **hc_split_sinkhorn_iters** (`int`): Optional parameter. Number of sinkhorn iterations. Data type supports `int`. Default is `20`.
- **hc_eps** (`float`): Optional parameter. Addition value for numerical stability in RMSNorm denominator computation and Sinkhorn-Knopp computation. Data type supports `float`. Default is `1e-6`.

## Return Value Description

- **y** (`Tensor`): Output in the formula. Data format is ND. Data type is `bfloat16`. When layout\_x is TND, shape is [t, h].
- **post** (`Tensor`): Output post from sinkhorn in the formula. Data format is ND. Data type is `float`. When layout\_x is TND, shape is [t, hc_mult].
- **comb** (`Tensor`): Output comb from sinkhorn in the formula. Data format is ND. Data type is `float`. When layout\_x is TND, shape is [t, hc_mult, hc_mult].

## Constraint Description

- This interface supports inference scenarios only.
- In input parameter x, shape [t, hc_mult, h] supports h as `4096` only.
- The shape, dtype, and so on of the input parameters must be consistent with the parameter description.
- The range of t is [1, 64k].

## Invocation Method

```
python3  models/deepseek_v4/test_hc_pre.py
```
