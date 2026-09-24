# attention_pre_quant

## Function Description

The `attention_pre_quant` operator corresponds to the pre-computation logic of the Attention module in the GLM-4.5 network, fusing the following key operations:

- Input Layer Normalization
- Input Quantization
- Quantized Matrix Multiplication (Quantized MatMul)
- Q/K Layer Normalization (Query/Key Layer Normalization)
- Q/K RoPE (Rotary Position Embedding)

By fusing multiple small operators, this operator significantly improves execution efficiency and memory bandwidth utilization on the NPU.

## Mathematical Formulas

$
\text{qkv} = \text{x} @ \text{weight}
$

$
\begin{aligned}
\text{q}, \text{k}, \text{v} &= \text{Split}(\text{qkv}, [\text{q\_head\_size}, \text{k\_head\_size}, \text{v\_head\_size}] )
\end{aligned}
$

$
\text{RMSNorm}(\text{q}) = \frac{\text{q}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} q_i^2} + \epsilon} \odot \text{q\_gamma}
$

$
\text{RMSNorm}(\text{k}) = \frac{\text{k}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} k_i^2} + \epsilon} \odot \text{k\_gamma}
$

$
\text{RoPE}(\text{q}, m) = \bigoplus_{i=1}^{d/2} \begin{bmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{bmatrix} \begin{bmatrix} q_{2i-1} \\ q_{2i} \end{bmatrix}
$

$
\text{RoPE}(\text{k}, m) = \bigoplus_{i=1}^{d/2} \begin{bmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{bmatrix} \begin{bmatrix} k_{2i-1} \\ k_{2i} \end{bmatrix}
$

## Function Prototype

```
def attention_pre_quant(
    hidden_states : torch.Tensor,
    residual : Optional[torch.Tensor],
    input_layernorm_weight : torch.Tensor,
    input_layernorm_bias : torch.Tensor,
    atten_qkv_input_scale_reciprocal : torch.Tensor,
    atten_qkv_input_offset : torch.Tensor,
    atten_qkv_weight : torch.Tensor,
    atten_qkv_quant_bias : torch.Tensor,
    atten_qkv_deq_scale : torch.Tensor,
    atten_q_norm_weight : torch.Tensor,
    atten_q_norm_bias : torch.Tensor,
    atten_k_norm_weight : torch.Tensor,
    atten_k_norm_bias : torch.Tensor,
    cos : torch.Tensor,
    sin : torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual_res: torch.Tensor
) -> None:
```

## Parameter Description

>**Explanation:**<br>
>
>- batch_size represents the input sample batch size (currently supports a range of 1 to 32), seq_len represents the input sample sequence length (currently supports 1), num_tokens represents the size after merging batch_size and seq_len axes, hidden_size represents the model hidden layer dimension (currently supports 5120), total_head_size represents the total output dimension after QKV projection (currently supports 1792), head_size represents the dimension size of each attention head (currently supports 128), half_rotary_dim represents the rotary position encoding dimension (currently supports 32), q_size represents the output dimension after query projection (currently supports 1536), and kv_size represents the output dimension after key/value projection (currently supports 128).

-   **hidden_states** (`Tensor`): Input tensor. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [num_tokens, hidden_size].

-   **residual** (`Tensor`): Residual connection input tensor. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [num_tokens, hidden_size].

-   **input_layernorm_weight** (`Tensor`): Weight parameter for input layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [hidden_size].

-   **input_layernorm_bias** (`Tensor`): Bias parameter for input layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [hidden_size].

-   **atten_qkv_input_scale_reciprocal** (`Tensor`): Reciprocal of the scaling coefficient for attention QKV input quantization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [hidden_size].

-   **atten_qkv_input_offset** (`Tensor`): Offset for attention QKV input quantization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [hidden_size].

-   **atten_qkv_weight** (`Tensor`): Weight matrix for attention QKV. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `int8`. Shape is [hidden_size, total_head_size].

-   **atten_qkv_quant_bias** (`Tensor`): Bias for attention QKV weight quantization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int32`. Shape is [total_head_size].

-   **atten_qkv_deq_scale** (`Tensor`): Dequantization scaling coefficient for attention QKV weight quantization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. Shape is [total_head_size].

-   **atten_q_norm_weight** (`Tensor`): Weight parameter for query layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [head_size].

-   **atten_q_norm_bias** (`Tensor`): Bias parameter for query layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [head_size].

-   **atten_k_norm_weight** (`Tensor`): Weight parameter for key layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [head_size].

-   **atten_k_norm_bias** (`Tensor`): Bias parameter for key layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [head_size].

-   **cos** (`Tensor`): Cosine values for rotary position encoding. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [num_tokens, 1, half_rotary_dim].

-   **sin** (`Tensor`): Sine values for rotary position encoding. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [num_tokens, 1, half_rotary_dim].

-   **query** (`Tensor`): Computed query vector. Does not support non-contiguous tensors. Data format supports ND. Data type is `bfloat16`. Shape is [num_tokens, q_size].

-   **key** (`Tensor`): Computed key vector. Does not support non-contiguous tensors. Data format supports ND. Data type is `bfloat16`. Shape is [num_tokens, kv_size].

-   **value** (`Tensor`): Computed value vector. Does not support non-contiguous tensors. Data format supports ND. Data type is `bfloat16`. Shape is [num_tokens, kv_size].

-   **residual_res** (`Tensor`): Updated residual output. Does not support non-contiguous tensors. Data format supports ND. Data type is `bfloat16`. Shape is [num_tokens, hidden_size].

## Call Sample

- For details, refer to [glm_attention_pre_quant](./glm_attention_pre_quant.py).

# attention

## Function Description

The `attention` operator is an attention mechanism optimization technology based on an advanced paging design, specifically designed for large model inference scenarios. It effectively addresses three core challenges that traditional attention mechanisms face when processing long sequences and dynamic batching:

- Memory Fragmentation: Frequent sequence growth/shrinkage causes discontinuous cache allocation.
- Low Memory Utilization: Fixed-size KV cache blocks result in a large amount of idle space.
- Limited Inference Throughput: Sequence alignment and redundant computation slow down overall processing speed.

By introducing a mechanism similar to operating system paging management, it achieves flexible management of non-contiguous cache blocks, significantly improving memory usage efficiency and inference throughput.

## Mathematical Formulas

$
\text{atten} = \text{Softmax}\left(\text{Zoom}(Q \cdot K^T)\right ) \cdot V
$

Where:
- Q, K, V represent Query, Key, and Value respectively, obtained through linear transformation of input variables.
- Zoom represents the scaling operation, intended to prevent the dot product from becoming too large and causing gradient vanishing after Softmax.
- Softmax normalizes the K weights corresponding to each Q and outputs attention weights.
- atten is the weighted sum of attention weights and V, fusing contextual information.

## Function Prototype

```
def attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor
) -> None:
```

## Parameter Description

>**Explanation:**<br>
>
>- batch_size represents the input sample batch size (currently supports a range of 1 to 32), seq_len represents the input sample sequence length (currently supports 1), num_tokens represents the size after merging batch_size and seq_len axes, num_head represents the multi-head count on the query side (currently supports 12), head_size represents the dimension of each attention head (currently supports 128), num_blocks represents the total number of available cache blocks, block_size represents the number of tokens each cache block can hold (currently supports 128), kv_head_num represents the multi-head count on the key/value side (currently supports 1), and max_num_blocks_per_query represents the maximum number of cache blocks a single request can occupy.

-   **query** (`Tensor`): Data format supports ND. Data type supports `bfloat16`. Shape is [num_tokens, num_head, head_size].

-   **key_cache** (`Tensor`): Data format supports ND. Data type supports `bfloat16`. Shape is [num_blocks, block_size, kv_head_num, head_size].

-   **value_cache** (`Tensor`): Data format supports ND. Data type supports `bfloat16`. Shape is [num_blocks, block_size, kv_head_num, head_size].

-   **block_tables** (`Tensor`): Data format supports ND. Data type supports `int32`. Shape is [batch_size, max_num_blocks_per_query].

-   **actual_seqs** (`Tensor`): Data format supports ND. Data type supports `int32`. Shape is [batch_size].

-   **attn_res** (`Tensor`): Data format supports ND. Data type supports `bfloat16`. Shape is [num_tokens, num_head, head_size].

## Call Sample

- For details, refer to [glm_attention.py](./glm_attention.py).

# gate

## Function Description

The `gate` operator corresponds to the matmul operation before expert selection in the GLM4.5 network, projecting the model main dimension from d_model to the router-specific dimension d_router.

## Mathematical Formulas
$
\text{gateOut} = \text{hiddenStates} @ \text{weight}
$

## Function Prototype

```
def gate(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    router_logits_out: torch.Tensor
):
```

## Parameter Description

>
>- batch_size represents the input sample batch size (currently supports a range of 1 to 32), seq_len represents the input sample sequence length (currently supports 1), num_tokens represents the size after merging batch_size and seq_len axes, and num_router_experts represents the number of routing experts (currently supports 160).

-   **gate_weight** (`Tensor`): Projection weight matrix that transforms general features into routing-specific features. Does not support non-contiguous tensors. Data format supports ND. Data type supports `FP32`. Shape is [num_router_experts, hidden_size].

-   **hidden_states** (`Tensor`): Current layer input feature matrix. Does not support non-contiguous tensors. Data format supports ND. Data type supports `FP32`. Shape is [num_tokens, hidden_size].

-   **router_logits_res** (`Tensor`): Routing feature matrix optimized through projection weight. Data format supports ND. Data type supports `FP32`. Shape is [num_tokens, num_router_experts].

## Call Sample

- For details, refer to [glm_gate](./glm_gate.py).

# select_experts

## Function Description

The `select_experts` operator corresponds to the select_experts (expert selector) in the GLM4.5 network, which is a core component of the MoE architecture. It is responsible for intelligently allocating input tokens to different expert networks for processing.

## Mathematical Formulas

$
\text{topk\_weights} = \text{topk\_weights} + \text{e\_score\_bias}
$

$
\text{topk\_weights} = \text{group\_top\_k}(\text{topk\_weights, num\_expert\_group, topk\_group})
$

$
\text{topk\_ids} = \text{topk}(\text{topk\_weights, top\_k})\text{topk\_weights}
$

$
\text{topk\_weights} = \text{renormalize}(\text{topk\_weights})
$

## Function Prototype

```
def select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    topk_group: int,
    num_expert_group: int,
    e_score_correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
```

## Parameter Description

>
>- batch_size represents the input sample batch size (currently supports a range of 1 to 32), seq_len represents the input sample sequence length (currently supports 1), num_tokens represents the size after merging batch_size and seq_len axes, num_router_experts represents the number of routing experts (currently supports 160), and num_experts_per_topk represents the number of experts allocated to each token (currently supports 8).

-   **router_logits** (`Tensor`): Routing scores representing the number of experts. Data format supports ND. Data type supports `float32`. Shape is [num_tokens, num_router_experts].

-   **top_k** (`int`): Number of experts. Data type supports `int8`.

-   **renormalize** (`bool`): Indicates whether to renormalize the routing weights. Data type supports `bool`.

-   **topk_group** (`int`): Number of selectable experts. Data type supports `int32`.

-   **num_expert_group** (`int`): Number of experts per group. Data type supports `int32`.

-   **e_score_correction_bias** (`Tensor`): Correction bias value for experts. Data format supports ND. Data type supports `bfloat16`. Shape is [num_router_experts].

-   **topk_weights** (`Tensor`): Responsibility weight for each expert. Data format supports ND. Data type supports `FP32`. Shape is [num_tokens, num_experts_per_topk].

-   **topk_ids** (`Tensor`): Expert IDs selected for each token. Data format supports ND. Data type supports `int32`. Shape is [num_tokens, num_experts_per_topk].

## Call Sample

- For details, refer to [glm_select_experts](./glm_select_experts.py).

# ffn_shared_expert_quant

## Function Description

The `ffn_shared_expert_quant` operator corresponds to the computation logic of the MoE shared expert in the GLM4.5 network. It includes `symmetric_quantization_per_token`, `matmul`, `dequant_dynamic`, and `swiglu`. It is used for quantized forward propagation computation of a single shared expert, learning general feature representations by reusing the same set of weight parameters across different tasks or data streams, while reducing the total number of model parameters.

## Mathematical Formulas

$
\text{hiddenStatesQuant}, \text{hiddenStatesScale} = \text{Quant}(\text{hiddenStates})
$

$
\text{swigluOut} = \text{Swiglu}((\text{hiddenStatesQuant} @ \text{w13}) \odot \text{hiddenStatesScale} \odot \text{w13Scale})
$

$
\text{downProjQuant}, \text{downProjScale} = \text{Quant}(\text{swigluOut})
$

$
\text{ffnRes} = (\text{downProjQuant} @ \text{w2}) \odot \text{downProjScale} \odot \text{w2Scale}
$

## Function Prototype

```
def ffn_shared_expert_quant(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    ffn_res: torch.Tensor
) -> None:
```

## Parameter Description
>**Explanation:**<br>
>
>- batch_size represents the input sample batch size (currently supports a range of 1 to 32), seq_len represents the input sample sequence length (currently supports 1), num_tokens represents the size after merging batch_size and seq_len axes, hidden_size represents the hidden layer size (currently supports 5120), and intermediate_size represents the dimension of the intermediate layer (currently supports 1536).

-   **hidden_states** (`Tensor`): Input feature vector for the current shared expert. Supports contiguous tensors only. Data format is ND. Data type supports `bfloat16`. Shape is [num_tokens, hidden_size].

-   **w13** (`Tensor`): Quantized weight for gate_proj and up_proj. Supports contiguous tensors only. Data format is NZ. Data type supports `int8`. Shape is [hidden_size, intermediate_size * 2].

-   **w13_scale** (`Tensor`): Scaling factor for w13 weights. Supports contiguous tensors only. Data format is ND. Data type supports `bfloat16`. Shape is [intermediate_size * 2].

-   **w2** (`Tensor`): Quantized weight for down_proj. Supports contiguous tensors only. Data format is NZ. Data type supports `int8`. Shape is [intermediate_size, hidden_size].

-   **w2_scale** (`Tensor`): Scaling factor for w2 weights. Supports contiguous tensors only. Data format is ND. Data type supports `bfloat16`. Shape is [hidden_size].

-   **ffn_res** (`Tensor`): Output tensor. Supports contiguous tensors only. Data format is ND. Data type supports `bfloat16`. Shape is [num_tokens, hidden_size].

## Call Sample

- For details, refer to [glm_ffn_shared_expert_quant](./glm_ffn_shared_expert_quant.py).

# attention_fusion

## Function Description

The `attention_fusion` operator is a deeply collaborative fusion version of `attention_pre_quant` and `attention`. It not only inherits all the advantages of the predecessor operators but also breaks through the data movement barriers between modules through end-to-end operator-level fusion, achieving full-link efficient execution from input to output.

## Mathematical Formulas

$
\text{qkv} = \text{x} @ \text{weight}
$

$
\begin{aligned}
\text{q}, \text{k}, \text{v} &= \text{Split}(\text{qkv}, [\text{q\_head\_size}, \text{k\_head\_size}, \text{v\_head\_size}] )
\end{aligned}
$

$
\text{RMSNorm}(\text{q}) = \frac{\text{q}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} q_i^2} + \epsilon} \odot \text{q\_gamma}
$

$
\text{RMSNorm}(\text{k}) = \frac{\text{k}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} k_i^2} + \epsilon} \odot \text{k\_gamma}
$

$
\text{RoPE}(\text{q}, m) = \bigoplus_{i=1}^{d/2} \begin{bmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{bmatrix} \begin{bmatrix} q_{2i-1} \\ q_{2i} \end{bmatrix}
$

$
\text{RoPE}(\text{k}, m) = \bigoplus_{i=1}^{d/2} \begin{bmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{bmatrix} \begin{bmatrix} k_{2i-1} \\ k_{2i} \end{bmatrix}
$

$
\text{atten} = \text{Softmax}\left(\text{Zoom}(Q \cdot K^T)\right ) \cdot V
$

## Function Prototype

```
def attention(
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        input_layernorm_weight: torch.Tensor,
        input_layernorm_bias: torch.Tensor,
        qkv_proj_scale: torch.Tensor,
        qkv_proj_offset: torch.Tensor,
        qkv_proj_weight: torch.Tensor,
        qkv_proj_quant_bias: torch.Tensor,
        qkv_proj_deq_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        q_norm_bias: torch.Tensor,
        k_norm_weight: torch.Tensor,
        k_norm_bias: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        actual_seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        enable_residual: bool,
        eps: float,
        num_decode_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
```

## Parameter Description

>**Explanation:**<br>
>
>- batch_size represents the input sample batch size (currently supports a range of 1 to 32), seq_len represents the input sample sequence length (currently supports 1), num_tokens represents the size after merging batch_size and seq_len axes, hidden_size represents the model hidden layer dimension (currently supports 5120), total_head_size represents the total output dimension after QKV projection (currently supports 1792), head_size represents the dimension size of each attention head (currently supports 128), half_rotary_dim represents the rotary position encoding dimension (currently supports 32), num_blocks represents the total number of available cache blocks, block_size represents the number of tokens each cache block can hold (currently supports 128), kv_head_num represents the multi-head count on the key/value side (currently supports 1), and max_num_blocks_per_query represents the maximum number of cache blocks a single request can occupy.

-   **hidden_states** (`Tensor`): Input tensor. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [num_tokens, hidden_size].

-   **residual** (`Tensor`): Residual connection input tensor. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [num_tokens, hidden_size].

-   **input_layernorm_weight** (`Tensor`): Weight parameter for input layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [hidden_size].

-   **input_layernorm_bias** (`Tensor`): Bias parameter for input layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [hidden_size].

-   **qkv_proj_scale** (`Tensor`): Reciprocal of the scaling coefficient for attention QKV input quantization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [hidden_size].

-   **qkv_proj_offset** (`Tensor`): Offset for attention QKV input quantization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [hidden_size].

-   **qkv_proj_weight** (`Tensor`): Weight matrix for attention QKV. Does not support non-contiguous tensors. Data format supports NZ. Data type supports `int8`. Shape is [hidden_size, total_head_size].

-   **qkv_proj_quant_bias** (`Tensor`): Bias for attention QKV weight quantization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `int32`. Shape is [total_head_size].

-   **qkv_proj_deq_scale** (`Tensor`): Dequantization scaling coefficient for attention QKV weight quantization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `float32`. Shape is [total_head_size].

-   **q_norm_weight** (`Tensor`): Weight parameter for query layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [head_size].

-   **q_norm_bias** (`Tensor`): Bias parameter for query layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [head_size].

-   **k_norm_weight** (`Tensor`): Weight parameter for key layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [head_size].

-   **k_norm_bias** (`Tensor`): Bias parameter for key layer normalization. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [head_size].

-   **cos** (`Tensor`): Cosine values for rotary position encoding. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [num_tokens, 1, half_rotary_dim].

-   **sin** (`Tensor`): Sine values for rotary position encoding. Does not support non-contiguous tensors. Data format supports ND. Data type supports `bfloat16`. Shape is [num_tokens, 1, half_rotary_dim].

-   **key_cache** (`Tensor`): Data format supports ND. Data type supports `bfloat16`. Shape is [num_blocks, block_size, kv_head_num, head_size].

-   **value_cache** (`Tensor`): Data format supports ND. Data type supports `bfloat16`. Shape is [num_blocks, block_size, kv_head_num, head_size].

-   **block_tables** (`Tensor`): Data format supports ND. Data type supports `int32`. Shape is [batch_size, max_num_blocks_per_query].

-   **actual_seq_lens** (`Tensor`): Data format supports ND. Data type supports `int32`. Shape is [batch_size].

-   **slot_mapping** (`Tensor`): Data format supports ND. Data type supports `int32`. Shape is [batch_size].

-   **enable_residual** (`bool`): Indicates whether to use residual connection. Data type supports `bool`.

-   **eps** (`float`): Precision value. Data type supports `float`.

-   **num_decode_tokens** (`int`): Data type supports `int`.

## Call Sample

- For details, refer to [glm_attention_fusion](./glm_attention_fusion.py).

# moe_fusion

## Function Description

The `moe_fusion` operator is a deeply collaborative fusion version of `gate`, `select_experts`, and `ffn_shared_expert_quant`, designed to achieve low-latency MoE inference.

## Mathematical Formulas

$
\text{topk\_weights} = \text{topk\_weights} + \text{e\_score\_bias}
$

$
\text{topk\_weights} = \text{group\_top\_k}(\text{topk\_weights, num\_expert\_group, topk\_group})
$

$
\text{topk\_ids} = \text{topk}(\text{topk\_weights, top\_k})\text{topk\_weights}
$

$
\text{topk\_weights} = \text{renormalize}(\text{topk\_weights})
$

$
\text{hidden\_states\_quant}, \text{hidden\_states\_scale} = \text{quant}(\text{hidden\_states})
$

$
\text{swiglu\_out} = \text{swiglu}((\text{hidden\_states\_quant} @ \text{w13}) \odot \text{hidden\_states\_scale} \odot \text{w13\_scale})
$

$
\text{down\_proj\_quant}, \text{down\_proj\_scale} = \text{quant}(\text{swiglu\_out})
$

$
\text{ffn\_res} = (\text{down\_proj\_quant} @ \text{w2}) \odot \text{down\_proj\_scale} \odot \text{w2\_scale}
$

$
\text{gate\_out} = \text{hidden\_states} @ \text{weight}
$

## Function Prototype

```
def moe_fusion(
        gate_weight: torch.Tensor,
        hidden_states: torch.Tensor,
        top_k: int,
        renormalize: bool,
        topk_group: int,
        num_expert_group: int,
        e_score_bias: torch.Tensor,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        ffn_res: torch.Tensor
):
```

## Parameter Description
>**Explanation:**<br>
>
>- batch_size represents the input sample batch size (currently supports a range of 1 to 32), seq_len represents the input sample sequence length (currently supports 1), num_tokens represents the size after merging batch_size and seq_len axes, num_router_experts represents the number of routing experts (currently supports 160), hidden_size represents the hidden layer size (currently supports 5120), intermediate_size represents the dimension of the intermediate layer (currently supports 1536), and num_experts_per_topk represents the number of experts allocated to each token (currently supports 8).

-   **gate_weight** (`Tensor`): Projection weight matrix that transforms general features into routing-specific features. Does not support non-contiguous tensors. Data format supports ND. Data type supports `FP32`. Shape is [num_router_experts, hidden_size].

-   **hidden_states** (`Tensor`): Input feature vector for the current shared expert. Supports contiguous tensors only. Data format is ND. Data type supports `bfloat16`. Shape is [num_tokens, hidden_size].

-   **top_k** (`int`): Number of experts. Data type supports `int8`.

-   **renormalize** (`bool`): Indicates whether to renormalize the routing weights. Data type supports `bool`.

-   **topk_group** (`int`): Number of selectable experts. Data type supports `int32`.

-   **num_expert_group** (`int`): Number of experts per group. Data type supports `int32`.

-   **e_score_bias** (`Tensor`): Correction bias value for experts. Data format supports ND. Data type supports `bfloat16`. Shape is [num_router_experts].

-   **w13** (`Tensor`): Quantized weight for gate_proj and up_proj. Supports contiguous tensors only. Data format is NZ. Data type supports `int8`. Shape is [hidden_size, intermediate_size * 2].

-   **w13_scale** (`Tensor`): Scaling factor for w13 weights. Supports contiguous tensors only. Data format is ND. Data type supports `bfloat16`. Shape is [intermediate_size * 2].

-   **w2** (`Tensor`): Quantized weight for down_proj. Supports contiguous tensors only. Data format is NZ. Data type supports `int8`. Shape is [intermediate_size, hidden_size].

-   **w2_scale** (`Tensor`): Scaling factor for w2 weights. Supports contiguous tensors only. Data format is ND. Data type supports `bfloat16`. Shape is [hidden_size].

-   **topk_weights** (`Tensor`): Responsibility weight for each expert. Data format supports ND. Data type supports `FP32`. Shape is [num_tokens, num_experts_per_topk].

-   **topk_ids** (`Tensor`): Expert IDs selected for each token. Data format supports ND. Data type supports `int32`. Shape is [num_tokens, num_experts_per_topk].

-   **ffn_res** (`Tensor`): Output tensor. Supports contiguous tensors only. Data format is ND. Data type supports `bfloat16`. Shape is [num_tokens, hidden_size].

## Call Sample

- For details, refer to [glm_moe_fusion](./glm_moe_fusion.py).
