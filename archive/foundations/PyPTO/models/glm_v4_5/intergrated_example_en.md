## GLM-4.5 Model PyPTO Operator Replacement Guide

This document uses the `gate` operator in the `glm_gate.py` file under the `models/glm_v4_5` directory as a typical sample (the replacement logic for other operators is identical). It focuses on introducing a convenient PyPTO operator replacement scheme based on the vllm project, aiming to provide practical support for operator adaptation after full-network fusion of the GLM-4.5 model.

### 1. Adjust the Directory Structure

1. Create a new `glm_pto_kernels` directory at the same level as the existing `vllm` and `vllm_ascend` directories.
2. Copy the `glm_gate.py` file from the `models/glm_v4_5` directory into the newly created `glm_pto_kernels` directory.
3. Create an empty `__init__.py` file in the `glm_pto_kernels` directory.

- The complete directory structure after adjustment is as follows:

```
glm-net/
├── glm_pto_kernels/
│   ├── __init__.py
│   └── glm_gate.py
├── vllm/
│   └── ......
└── vllm_ascend/
    └── ......
```

### 2. Configure the Operator Adaptation Layer Interface

In the newly created `__init__.py` file under the `glm_pto_kernels` directory, define the operator adaptation layer interface as the bridging layer between the native code and the PyPTO operator.

Sample: `gate` operator adaptation layer definition. Add the following function in `glm_pto_kernels/__init__.py`:

```
def gate(gate_layer, hidden_states):
    # Import the PyPTO-implemented gate operator
    from glm_pto_kernels.glm_gate import gate as gate_pto

    # Get the batch size (bs) of the input tensor and the dimension (ne) of the gate layer weight
    bs = hidden_states.shape[0]
    ne = gate_layer.weight.shape[0]

    # Initialize the output tensor: dimension is (bs, ne), data type/device is consistent with the gate layer weight
    router_logits_res = torch.empty(
        (bs, ne),
        dtype=gate_layer.weight.dtype,
        device=hidden_states.device
    )

    # Call the PyPTO-implemented gate operator to complete the core computation
    gate_pto(
        gate_layer.weight,
        hidden_states,
        router_logits_res
    )

    # Return the computation result
    return router_logits_res
```

### 3. Modify the Operator Invocation Logic

Open the target file `vllm/model_executor/models/glm4_moe.py`. Add the following code in the import area at the top of the file to import the PyPTO operator adaptation layer library:

```
import glm_pto_kernels
```

In the same file, locate the `Glm4MoE.forward()` method, find the native invocation code for the `gate` operator, and replace it with the PyPTO-implemented `gate` operator invocation code:

- Original invocation code

```
router_logits = self.gate(hidden_states.to(dtype=torch.float32))
```

- Code after replacement

```
router_logits = glm_pto_kernels.gate(self.gate, hidden_states.to(dtype=torch.float32))
```

After completing the above modifications and saving the file, you can switch the `gate` operator from the native implementation to the PyPTO operator implementation. When the model runs, it will use the PyPTO-implemented `gate` operator logic.

### 4. Adaptation Layer Interfaces for Other Operators
The following provides adaptation layer interface definitions for various operators under the `models/glm_v4_5` directory. Copy the corresponding functions directly into the `glm_pto_kernels/__init__.py` file to complete the adaptation layer configuration. Each function comment indicates the target replacement file, function, and specific replacement method. Follow the instructions to complete the operator switch.

- Adaptation layer function related to glm_attention.py
```
# Configure the operator switch to flexibly switch between PyPTO and the original scheme
USE_PTO_FA              = True

# Function definition and adaptation description
def paged_attention(query, key_cache, value_cache, block_tables, actual_seqs_cpu, output):
    '''
    [Adaptation Description]
    Target file: vllm_ascend/attention/attention_v1.py
    Target function: _forward_decode_only()
    Replacement method: Replace the torch_npu._npu_paged_attention() call with the current function. The replacement code is as follows, with indentation already adjusted to match the source code and ready for direct copy:
                 # <----Copy start---->
                 if glm_pto_kernels.USE_PTO_FA:
                     # PTO kernel implementation
                     glm_pto_kernels.paged_attention(
                         query, self.key_cache, self.value_cache,
                         attn_metadata.block_tables, attn_metadata.seq_lens, output
                     )
                 else:
                     # Original scheme
                     torch_npu._npu_paged_attention(
                         query=query,
                         key_cache=self.key_cache,
                         value_cache=self.value_cache,
                         num_kv_heads=self.num_kv_heads,
                         num_heads=self.num_heads,
                         scale_value=self.scale,
                         block_table=attn_metadata.block_tables,
                         context_lens=attn_metadata.seq_lens,
                         out=output)
                 # <----Copy end---->
    '''
    from glm_pto_kernels.glm_attention import attention
    attention(
        query, key_cache, value_cache, block_tables, actual_seqs_cpu, output
    )
    return output
```

- Adaptation layer function related to glm_attention_pre_quant.py
```
# Configure the operator switch to flexibly switch between PyPTO and the original scheme
USE_PTO_FA_PRE          = True

# Function definition and adaptation description
def attention_pre(hidden_states, residual, layer, attention, positions):
    '''
    [Adaptation Description]
    Target file: vllm/model_executor/models/glm4_moe.py
    Target function: Glm4MoeDecoderLayer.forward()
    Replacement method:
    1. Modify Glm4MoeAttention.forward() and replace it with the following code, with indentation already adjusted to match the source code and ready for direct copy:
    # <----Copy start---->
    def forward(
        self,
        layer,  # Add layer input
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,  # Add residual input
    ) -> torch.Tensor:
        if glm_pto_kernels.USE_PTO_FA_PRE:
            # PTO kernel implementation
            q, k, v, residual = glm_pto_kernels.attention_pre(hidden_states, residual, layer, self)
        else:
            # Original scheme
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            if self.use_qk_norm:
                q = self.q_norm(q.reshape(-1, self.num_heads,
                                        self.head_dim)).reshape(q.shape)
                k = self.k_norm(k.reshape(-1, self.num_kv_heads,
                                        self.head_dim)).reshape(k.shape)

            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output, residual  # Add residual output
    # <----Copy end---->

    2. Merge input_layernorm() and self_attn() and replace them with the following code, with indentation already adjusted to match the source code and ready for direct copy:
        # <----Copy start---->
        if glm_pto_kernels.USE_PTO_FA_PRE:
            # PTO kernel implementation
            hidden_states, residual = self.self_attn(self, positions=positions,
                                        hidden_states=hidden_states, residual=residual)
        else:
            # Original scheme
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
            hidden_states = self.self_attn(self, positions=positions,
                                        hidden_states=hidden_states, residual=residual)
        # <----Copy end---->
    '''
    from glm_pto_kernels.glm_attention_pre_quant import attention_pre_quant as attention_pre_quant_pto
    cos, sin = attention.rotary_emb.cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    cos = cos.unsqueeze(1).contiguous()
    sin = sin.unsqueeze(1).contiguous()
    bs = hidden_states.shape[0]
    q = torch.empty((bs, attention.q_size), dtype=hidden_states.dtype, device=f'{hidden_states.device}')
    k = torch.empty((bs, attention.kv_size), dtype=hidden_states.dtype, device=f'{hidden_states.device}')
    v = torch.empty((bs, attention.kv_size), dtype=hidden_states.dtype, device=f'{hidden_states.device}')
    residual_res = torch.empty((bs, hidden_states.shape[1]),
                               dtype=hidden_states.dtype, device=f'{hidden_states.device}')
    if residual is None:
        residual = torch.zeros((bs, hidden_states.shape[1]),
                               dtype=hidden_states.dtype, device=f'{hidden_states.device}')
    attention_pre_quant_pto(
        hidden_states, residual,
        layer.input_layernorm.weight.data,
        layer.input_layernorm.bias.data,
        attention.qkv_proj.aclnn_input_scale_reciprocal.data,
        attention.qkv_proj.aclnn_input_offset.data,
        attention.qkv_proj.weight.data,
        attention.qkv_proj.quant_bias.data,
        attention.qkv_proj.deq_scale.data,
        attention.q_norm.weight.data,
        attention.q_norm.bias.data,
        attention.k_norm.weight.data,
        attention.k_norm.bias.data,
        cos, sin, q, k, v, residual_res
    )

    return q, k, v, residual_res
```

- Adaptation layer function related to glm_ffn_shared_expert_quant.py
```
# Configure the operator switch to flexibly switch between PyPTO and the original scheme
USE_PTO_SHARE_EXEPERTS  = True

# Function definition and adaptation description
def ffn_share_expert_quant(layer, hidden_states):
    '''
    [Adaptation Description]
    Target file: glm_pto_kernels/glm_ffn_shared_expert_quant.py
    Target function: ffn_share_expert_quant()
    Replacement method: Replace the _native_select_experts() call with the current function. The replacement code is as follows, with indentation already adjusted to match the source code and ready for direct copy:
        # <----Copy start---->
        if glm_pto_kernels.USE_PTO_SHARE_EXEPERTS and self.is_down_proj_quant:
            return glm_pto_kernels.ffn_share_expert_quant(self, x)
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
        # <----Copy end---->
    '''
    from glm_pto_kernels.glm_ffn_shared_expert_quant import ffn_shared_expert_quant as ffn_shared_expert_quant_pto
    ffn_res = torch.empty_like(hidden_states, device=hidden_states.device)
    w13_int8 = layer.gate_up_proj.weight
    w13_scale = layer.gate_up_proj.weight_scale
    w2_int8 = layer.down_proj.weight
    w2_scale = layer.down_proj.weight_scale
    ffn_shared_expert_quant_pto(hidden_states, w13_int8, w13_scale, w2_int8, w2_scale, ffn_res)
    return ffn_res
```

- Adaptation layer function related to glm_select_experts.py
```
# Configure the operator switch to flexibly switch between PyPTO and the original scheme
USE_PTO_SELECT_EXEPERTS = True

# Function definition and adaptation description
def select_experts(router_logits, top_k, renormalize,
                   topk_group=None, num_expert_group=None, e_score_correction_bias=None):
    '''
    [Adaptation Description]
    Target file: vllm_ascend/ops/fused_moe/experts_selector.py
    Target function: select_experts()
    Replacement method: Replace the _native_select_experts() call with the current function. The replacement code is as follows, with indentation already adjusted to match the source code and ready for direct copy:
        # <----Copy start---->
        if glm_pto_kernels.USE_PTO_SELECT_EXEPERTS:
            # PTO kernel implementation
            topk_weights, topk_ids = glm_pto_kernels.select_experts(
                router_logits, top_k, renormalize, topk_group, num_expert_group, e_score_correction_bias
            )
        else:
            # Original Fallback
            topk_weights, topk_ids = _native_select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                global_num_experts=global_num_experts,
            )
        # <----Copy end---->
    '''
    from glm_pto_kernels.glm_select_experts import select_experts
    bs = router_logits.shape[0]
    device_info = router_logits.device
    topk_weights = torch.empty(
    (bs, top_k), dtype=router_logits.dtype, device=device_info)
    topk_ids = torch.empty((bs, top_k), dtype=torch.int32, device=device_info)
    select_experts(
        router_logits, top_k, renormalize, topk_group, num_expert_group, e_score_correction_bias, topk_weights, topk_ids
    )
    return topk_weights, topk_ids
```

When replacing, strictly match the target file and target function in the comments, directly copy the code snippet inside the comments (the indentation has been adapted to the source code). Use the switch variable to determine whether to call the PyPTO adaptation layer function. If not enabled, execute the original logic, ensuring a non-intrusive switch.
