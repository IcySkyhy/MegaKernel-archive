# Quantization-Aware Training (QAT) Operator Interface Documentation

This document contains the interface descriptions for Quantization-Aware Training (QAT) related operators, covering both symmetric and asymmetric quantization.

## Table of Contents

1. [ai_infra_qat_symmetric_per_tensor](#ai_infra_qat_symmetric_per_tensor) - Symmetric per-tensor quantization operator
2. [ai_infra_qat_symmetric_per_channel](#ai_infra_qat_symmetric_per_channel) - Symmetric per-channel quantization operator
3. [ai_infra_qat_asymmetric_per_group](#ai_infra_qat_asymmetric_per_group) - Asymmetric per-group quantization operator

---

# ai_infra_qat_symmetric_per_tensor

Symmetric Quantization-Aware Training (QAT) operator, including both forward and backward operators. This operator simulates symmetric quantization on weights, introducing quantization noise during training so that the model can adapt to the accuracy loss from quantization.

It is suitable for Embedding layer scenarios, where scale is a scalar (shape (1,1)) and all weight elements share the same scaling factor.

## Forward Operator: ai_infra_qat_symmetric_per_tensor

### Description

Symmetric Quantization-Aware Training (QAT) forward operator. This operator simulates symmetric quantization on weights, introducing quantization noise during training so that the model can adapt to the accuracy loss from quantization.

This operator is suitable for Embedding layer scenarios, where scale is a scalar (shape (1,1)) and all weight elements share the same scaling factor.

### Interface Definition

```python
def ai_infra_qat_symmetric_per_tensor(
    weight: Tensor,      # BF16, shape: (N, M)
    scale: Tensor,       # BF16, shape: (1, 1)
    eps: float,          # minimum scale threshold
    min_v: float,        # quantization lower bound
    max_v: float,        # quantization upper bound
) -> Tensor:             # BF16, shape: (N, M)
```

### Parameter Description

#### Input Parameters

| Parameter | Type | Shape | Description |
|-------|------|------|------|
| weight | Tensor(BF16) | (N, M) | Input weight tensor, N is the output feature dimension, M is the input feature dimension |
| scale | Tensor(BF16) | (1, 1) | Quantization scaling factor, scalar form, all weight elements share the same scaling factor |
| eps | float | - | Minimum threshold for scale, prevents division by zero, uses eps as replacement when scale is less than eps |
| min_v | float | - | Quantization lower bound, -128 for INT8 quantization |
| max_v | float | - | Quantization upper bound, 127 for INT8 quantization |

#### Output Parameters

| Parameter | Type | Shape | Description |
|-------|------|------|------|
| output | Tensor(BF16) | (N, M) | Pseudo-quantized weight tensor, same shape as input weight |

### Algorithm Principle

The core of symmetric quantization-aware training is the use of the Straight-Through Estimator (STE) to approximate the gradient of the quantization operation. The operator implements this through the following steps:

#### Step 1: Scale Anti-Zero Protection

$$
s' = \begin{cases} s, & \text{if } s > \varepsilon \\ \varepsilon, & \text{otherwise} \end{cases}
$$

Where $s$ is the input scale, and $\varepsilon$ is the eps parameter.

#### Step 2: Normalization

$$
W_{\text{norm}} = \frac{W}{s'}
$$

Divide the weight by the scaling factor to obtain the normalized weight.

#### Step 3: Pseudo-Quantization (STE-Approximated Rounding and Clamping)

$$
W_{\text{quant}} = \text{detach}\left(\text{round}(W_{\text{norm}}) - W_{\text{norm}}\right) + W_{\text{norm}}
$$

$$
W_{\text{clamp}} = \text{clamp}(W_{\text{quant}}, V_{\text{min}}, V_{\text{max}})
$$

Use detach() to block gradient propagation of the rounding operation, implementing the STE approximation. Then clamp the quantized values to the valid range.

#### Step 4: Dequantization

$$
W_q = W_{\text{clamp}} \times s'
$$

Multiply the clamped values by the scaling factor to restore the original numerical range.

### Constraints

1. `grad_output` and `weight` must have the same shape (N, M). M is in [128, 3072] and divisible by 128. Data type is BF16.
2. `scale` shape must be (1, 1). Data type is BF16.
3. `eps` must be in the range (0, 1). Data type is float.
4. `min_v` must be less than `max_v`, and both must be floating-point integers. Data type is float, with all-zero fractional part.

### Supported Specifications

- Data types: BF16 (input/output), FP32 (internal computation)
- Chip platforms: A2/A3

### Usage Sample

```python
import torch
import pypto

# Set device
torch.npu.set_device(0)

# Create input tensors
N, M = 1024, 2048
weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0", requires_grad=True)
scale = torch.tensor([[0.1]], dtype=torch.bfloat16, device="npu:0", requires_grad=True)

# Forward propagation
output = ai_infra_qat_symmetric_per_tensor(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0)

# Simulate upstream gradient
grad_output = torch.ones_like(output)

# Backward propagation
grad_weight, grad_scale = ai_infra_qat_symmetric_per_tensor_backward(
    grad_output, weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0
)

print(f"Weight gradient shape: {grad_weight.shape}")
print(f"Scale gradient shape: {grad_scale.shape}")
```

---

# ai_infra_qat_symmetric_per_channel

Symmetric Quantization-Aware Training (QAT) operator (N scale version), including both forward and backward operators. This operator simulates symmetric quantization on weights, supporting independent scaling factors per output channel.

It is suitable for Lm Head layer scenarios, where scale is a vector (shape (N,1)) and each output channel has its own independent scaling factor.

## Forward Operator: ai_infra_qat_symmetric_per_channel

### Description

Symmetric Quantization-Aware Training (QAT) forward operator (N scale version). This operator simulates symmetric quantization on weights, supporting independent scaling factors per output channel.

This operator is suitable for Lm Head layer scenarios, where scale is a vector (shape (N,1)) and each output channel has its own independent scaling factor.

### Interface Definition

```python
def ai_infra_qat_symmetric_per_channel(
    weight: Tensor,      # BF16, shape: (N, M)
    scale: Tensor,       # BF16, shape: (N, 1)
    eps: float = 1e-4,   # minimum scale threshold
    min_v: float = -128.0,  # quantization lower bound
    max_v: float = 127.0,   # quantization upper bound
) -> Tensor:             # BF16, shape: (N, M)
```

### Parameter Description

#### Input Parameters

| Parameter | Type | Shape | Default | Description |
|-------|------|------|--------|------|
| weight | Tensor(BF16) | (N, M) | Required | Input weight tensor, N is the output feature dimension, M is the input feature dimension |
| scale | Tensor(BF16) | (N, 1) | Required | Quantization scaling factor, each output channel has its own independent scaling factor |
| eps | float | - | 1e-4 | Minimum threshold for scale, prevents division by zero |
| min_v | float | - | -128.0 | Quantization lower bound, -128 for INT8 quantization |
| max_v | float | - | 127.0 | Quantization upper bound, 127 for INT8 quantization |

#### Output Parameters

| Parameter | Type | Shape | Description |
|-------|------|------|------|
| output | Tensor(BF16) | (N, M) | Pseudo-quantized weight tensor, same shape as input weight |

### Algorithm Principle

The algorithm principle is the same as symmetric_qat, with the main difference being the shape of scale. When scale is (N, 1), each output channel uses its own independent scaling factor.

#### Step 1: Scale Anti-Zero Protection

$$
s'_i = \begin{cases} s_i, & \text{if } s_i > \varepsilon \\ \varepsilon, & \text{otherwise} \end{cases}
$$

Where $s_i$ is the scale of the i-th output channel.

#### Step 2: Normalization (Broadcast Scaling)

$$
W_{\text{norm}}[i, j] = \frac{W[i, j]}{s'_i}
$$

Scale broadcasts along the M dimension, implementing per-channel normalization.

#### Step 3: Pseudo-Quantization (STE-Approximated Rounding and Clamping)

$$
W_{\text{quant}} = \text{detach}\left(\text{round}(W_{\text{norm}}) - W_{\text{norm}}\right) + W_{\text{norm}}
$$

$$
W_{\text{clamp}} = \text{clamp}(W_{\text{quant}}, V_{\text{min}}, V_{\text{max}})
$$

#### Step 4: Dequantization

$$
W_q[i, j] = W_{\text{clamp}}[i, j] \times s'_i
$$

### Constraints

1. `weight` must be a 2-dimensional tensor with shape (N, M). M is in [128, 3072] and divisible by 128. Data type is BF16.
2. `scale` must be a 2-dimensional tensor with shape (N, 1). Data type is BF16
3. `eps` must be in the range (0, 1). Data type is float
4. `min_v` must be less than `max_v`, and both must be floating-point integers. Data type is float, with all-zero fractional part.

### Supported Specifications

- Data types: BF16 (input/output), FP32 (internal computation)
- Chip platforms: A2/A3

### Usage Sample

```python
import torch
import pypto

# Set device
torch.npu.set_device(0)

# Create input tensors - Lm Head scenario
# For example: vocabulary size 153376, hidden dimension 2048
N, M = 153376, 2048
weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0")
scale = torch.abs(torch.randn(N, 1, dtype=torch.bfloat16, device="npu:0")) + 0.01

# Create and call operator
output = ai_infra_qat_symmetric_per_channel(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0)

print(f"Input weight shape: {weight.shape}")
print(f"Scale shape: {scale.shape}")
print(f"Output weight shape: {output.shape}")
```

---

## Backward Operator: ai_infra_qat_symmetric_per_channel_backward

### Description

Symmetric Quantization-Aware Training (QAT) backward operator (N scale version). This operator computes the gradients of the symmetric quantization operation, supporting gradient computation for independent scaling factors per output channel.

This operator is suitable for Lm Head layer scenarios, where scale is a vector (shape (N,1)), corresponding to the backpropagation of the forward operator ai_infra_qat_symmetric_per_channel.

### Interface Definition

```python
def ai_infra_qat_symmetric_per_channel_backward(
    grad_output: Tensor,  # BF16, shape: (N, M)
    weight: Tensor,       # BF16, shape: (N, M)
    scale: Tensor,        # BF16, shape: (N, 1)
    eps: float = 1e-4,    # minimum scale threshold
    min_v: float = -128.0,  # quantization lower bound
    max_v: float = 127.0,   # quantization upper bound
) -> Tuple[Tensor, Tensor]:  # (grad_weight, grad_scale)
```

### Parameter Description

#### Input Parameters

| Parameter | Type | Shape | Default | Description |
|-------|------|------|--------|------|
| grad_output | Tensor(BF16) | (N, M) | Required | Upstream gradient tensor |
| weight | Tensor(BF16) | (N, M) | Required | Original input weight tensor |
| scale | Tensor(BF16) | (N, 1) | Required | Quantization scaling factor, one per output channel |
| eps | float | - | 1e-4 | Minimum threshold for scale |
| min_v | float | - | -128.0 | Quantization lower bound |
| max_v | float | - | 127.0 | Quantization upper bound |

#### Output Parameters

| Parameter | Type | Shape | Description |
|-------|------|------|------|
| grad_weight | Tensor(BF16) | (N, M) | Gradient with respect to weight |
| grad_scale | Tensor(BF16) | (N, 1) | Gradient with respect to scaling factor, one per output channel |

### Algorithm Principle

The algorithm principle is similar to ai_infra_qat_symmetric_per_tensor_backward, but since scale has shape (N, 1), the gradient computation differs.

#### Key Differences

Since each output channel has its own independent scale, gradient computation does not require summation across channels, only summation along the M dimension:

**Gradient with respect to W**:
$$
\frac{\partial\text{Loss}}{\partial W}[i,j] = \frac{\partial\text{Loss}}{\partial W_q}[i,j] \cdot \mathbf{1}_{\text{mask}}[i,j] \cdot \frac{1}{s'_i}
$$

**Gradient with respect to s**:
$$
\frac{\partial\text{Loss}}{\partial s_i} = \left[ \sum_{j=1}^{M} \left( \frac{\partial\text{Loss}}{\partial W_q}[i,j] \cdot W_{\text{clamp}}[i,j] \right) - \frac{1}{s'_i} \sum_{j=1}^{M} \left( \frac{\partial\text{Loss}}{\partial W_q}[i,j] \cdot \mathbf{1}_{\text{mask}}[i,j] \cdot W[i,j] \right) \right] \cdot \mathbf{1}_{s_i > \varepsilon}
$$

#### Gradient Computation Steps

1. **Recompute forward intermediates**: Recompute normalized, rounded, and clamped values from weight and scale
2. **Compute clamping mask**: Determine whether elements are within the [min_v, max_v] range
3. **Compute scale mask**: Determine whether scale is greater than eps
4. **Compute grad_weight**: Upstream gradient multiplied by clamping mask
5. **Compute grad_scale**: Multiplication path plus division path, summed along the M dimension

### Constraints

1. `grad_output` and `weight` must have the same shape (N, M). M is in [128, 3072] and divisible by 128. Data type is BF16.
2. `scale` shape must be (N, 1). Data type is BF16.
3. `eps` must be in the range (0, 1). Data type is float.
4. `min_v` must be less than `max_v`, and both must be floating-point integers. Data type is float, with all-zero fractional part.

### Supported Specifications

- Data types: BF16 (input/output), FP32 (internal computation)
- Chip platforms: A2/A3

### Usage Sample

```python
import torch
import pypto

# Set device
torch.npu.set_device(0)

# Create input tensors - Lm Head scenario
N, M = 153376, 2048
weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0", requires_grad=True)
scale = torch.abs(torch.randn(N, 1, dtype=torch.bfloat16, device="npu:0")) + 0.01
scale.requires_grad_(True)

# Forward propagation
output = ai_infra_qat_symmetric_per_channel(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0)

# Simulate upstream gradient
grad_output = torch.ones_like(output)

# Backward propagation
grad_weight, grad_scale = ai_infra_qat_symmetric_per_channel_backward(
    grad_output, weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0
)

print(f"Weight gradient shape: {grad_weight.shape}")
print(f"Scale gradient shape: {grad_scale.shape}")
```

---

## Differences Between ai_infra_qat_symmetric_per_tensor and ai_infra_qat_symmetric_per_channel

| Feature | ai_infra_qat_symmetric_per_tensor | ai_infra_qat_symmetric_per_channel |
|-----|--------------------------------|-----------------------------------|
| scale shape | (1, 1) | (N, 1) |
| Scaling granularity | Global uniform scaling | Per-channel independent scaling |
| Applicable scenario | Embedding layer | Lm Head layer |
| Quantization precision | Lower | Higher |
| grad_scale shape | (1, 1) | (N, 1) |
| Gradient summation dimension | Sum along N and M dimensions | Sum only along M dimension |

---

# ai_infra_qat_asymmetric_per_group

Asymmetric Quantization-Aware Training (QAT) operator, including both forward and backward operators. This operator simulates asymmetric quantization on weights, supporting group quantization and a learnable offset.

This operator is suitable for Transformer Linear layer scenarios, achieving finer quantization granularity through group quantization and improving post-quantization model accuracy. Asymmetric quantization can better adapt to asymmetry in weight distributions compared to symmetric quantization.

## Forward Operator: ai_infra_qat_asymmetric_per_group

### Description

Asymmetric Quantization-Aware Training (QAT) forward operator. This operator simulates asymmetric quantization on weights, supporting group quantization and a learnable offset.

This operator is suitable for Transformer Linear layer scenarios, achieving finer quantization granularity through group quantization and improving post-quantization model accuracy. Asymmetric quantization can better adapt to asymmetry in weight distributions compared to symmetric quantization.

### Interface Definition

```python
def ai_infra_qat_asymmetric_per_group(
    weight: Tensor,      # BF16, shape: (N, M)
    scale: Tensor,       # BF16, shape: (N*M/group_size, 1)
    offset: Tensor,      # BF16, shape: (N*M/group_size, 1)
    group_size: int = 128,  # group size
    bit: int = 4,        # quantization bit width, supports 2, 3, 4
    eps: float = 1e-4,   # minimum scale threshold
    clip_val: float = 0.99,  # clamp value
) -> Tensor:             # BF16, shape: (N, M)
```

### Parameter Description

#### Input Parameters

| Parameter | Type | Shape | Default | Description |
|-------|------|------|--------|------|
| weight | Tensor(BF16) | (N, M) | Required | Input weight tensor, N is the output feature dimension, M is the input feature dimension |
| scale | Tensor(BF16) | (N*M/group_size, 1) | Required | Quantization scaling factor, one scaling factor per group |
| offset | Tensor(BF16) | (N*M/group_size, 1) | Required | Quantization offset, one offset per group |
| group_size | int | - | 128 | Group size, number of weight elements per group |
| bit | int | - | 4 | Quantization bit width, supports 2, 3, and 4-bit quantization |
| eps | float | - | 1e-4 | Minimum threshold for scale, prevents division by zero |
| clip_val | float | - | 0.99 | Clamp value, limits the quantization range |

#### Output Parameters

| Parameter | Type | Shape | Description |
|-------|------|------|------|
| output | Tensor(BF16) | (N, M) | Pseudo-quantized weight tensor, same shape as input weight |

### Algorithm Principle

Asymmetric quantization-aware training is based on the enhanced LSQ+ (Learned Step Size Quantization Plus) algorithm, achieving more flexible quantization through group quantization and learnable scale/offset.

#### Core Formulas

##### Step 1: Scale Anti-Zero Protection

$$
s' = \begin{cases} s, & \text{if } s > \varepsilon \\ \varepsilon, & \text{otherwise} \end{cases}
$$

##### Step 2: Reshape Weight into Group Form

$$
W_{\text{group}} = \text{reshape}(W, [G, \text{group\_size}])
$$

Where $G = \frac{N \times M}{\text{group\_size}}$ is the total number of groups.

##### Step 3: Compute Quantization Parameters

$$
\alpha = s' \times n_{\text{levels}}, \quad n_{\text{levels}} = 2^{(\text{bit}-1)}
$$

$$
\text{shift} = 0.5
$$

##### Step 4: Asymmetric Quantization

$$
W_{\text{shifted}} = W_{\text{group}} - \text{offset}
$$

$$
W_{\text{clipped}} = \text{clamp}\left(\frac{W_{\text{shifted}}}{\alpha}, -\text{clip\_val}, \text{clip\_val}\right) \times n_{\text{levels}} - \text{shift}
$$

##### Step 5: Pseudo-Quantization (STE)

$$
W_{\text{rounded}} = \text{detach}(\text{round}(W_{\text{clipped}}) - W_{\text{clipped}}) + W_{\text{clipped}}
$$

##### Step 6: Dequantization

$$
W_{\text{unshifted}} = W_{\text{rounded}} + \text{shift}
$$

$$
W_{\text{denorm}} = \frac{W_{\text{unshifted}}}{n_{\text{levels}}}
$$

$$
W_{\text{out}} = W_{\text{denorm}} \times \alpha + \text{offset}
$$

### Group Quantization Description

Group quantization divides the weight matrix into multiple small groups, each with its own independent scale and offset:

```
Weight matrix (N, M):
+-----+-----+-----+-----+
| Group 0 | Group 1 | ... | Group G-1 |
| (128 elements) | (128 elements) | ... | (128 elements) |
+-----+-----+-----+-----+
     |            |              |
     v            v              v
  scale[0]    scale[1]       scale[G-1]
  offset[0]   offset[1]      offset[G-1]
```

### Constraints

1. `group_size` takes values 64, 128, or 256. Data type is int.
2. `weight` must be a 2-dimensional tensor with shape (N, M). M is in [128, 3072] and divisible by `group_size`. Data type is BF16.
3. `scale` and `offset` shapes must be (N*M/group_size, 1). Data type is BF16.
4. `bit` can only be 2, 3, or 4. Data type is int.
5. `eps` must be in the range (0, 1). Data type is float.
6. `clip_val` must be in the range (0, 1). Data type is float.

### Supported Specifications

- Data types: BF16 (input/output), FP32 (internal computation)
- Chip platforms: A2/A3
- Supported bit widths: 2-bit, 3-bit, 4-bit

### Usage Sample

```python
import torch
import pypto

# Set device
torch.npu.set_device(0)

# Create input tensors - Transformer Linear layer scenario
# For example: FFN layer, input dimension 3072, output dimension 768
N, M = 768, 3072
group_size = 128
bit = 4

# Compute number of groups
num_groups = N * M // group_size

# Create input
weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0")
scale = torch.abs(torch.randn(num_groups, 1, dtype=torch.bfloat16, device="npu:0")) + 0.01
offset = torch.randn(num_groups, 1, dtype=torch.bfloat16, device="npu:0")

# Create and call operator
output = ai_infra_qat_asymmetric_per_group(weight, scale, offset, group_size=group_size, bit=bit, eps=1e-4, clip_val=0.99)

print(f"Input weight shape: {weight.shape}")
print(f"Scale shape: {scale.shape}")
print(f"Offset shape: {offset.shape}")
print(f"Output weight shape: {output.shape}")
```

---

## Backward Operator: ai_infra_qat_asymmetric_per_group_backward

### Description

Asymmetric Quantization-Aware Training (QAT) backward operator. This operator computes the gradients of the asymmetric quantization operation, supporting gradient computation for weights, scaling factors, and offsets.

This operator is suitable for Transformer Linear layer scenarios, corresponding to the backpropagation of the forward operator ai_infra_qat_asymmetric_per_group, achieving fine-grained gradient computation through group quantization.

### Interface Definition

```python
def ai_infra_qat_asymmetric_per_group_backward(
    grad_output: Tensor,  # BF16, shape: (N, M)
    weight: Tensor,       # BF16, shape: (N, M)
    scale: Tensor,        # BF16, shape: (N*M/group_size, 1)
    offset: Tensor,       # BF16, shape: (N*M/group_size, 1)
    group_size: int = 128,  # group size
    bit: int = 4,        # quantization bit width
    eps: float = 1e-4,   # minimum scale threshold
    clip_val: float = 0.99,  # clamp value
) -> Tuple[Tensor, Tensor, Tensor]:  # (grad_weight, grad_scale, grad_offset)
```

### Parameter Description

#### Input Parameters

| Parameter | Type | Shape | Default | Description |
|-------|------|------|--------|------|
| grad_output | Tensor(BF16) | (N, M) | Required | Upstream gradient tensor |
| weight | Tensor(BF16) | (N, M) | Required | Original input weight tensor |
| scale | Tensor(BF16) | (N*M/group_size, 1) | Required | Quantization scaling factor |
| offset | Tensor(BF16) | (N*M/group_size, 1) | Required | Quantization offset |
| group_size | int | - | 128 | Group size |
| bit | int | - | 4 | Quantization bit width, supports 2, 3, 4 |
| eps | float | - | 1e-4 | Minimum threshold for scale |
| clip_val | float | - | 0.99 | Clamp value |

#### Output Parameters

| Parameter | Type | Shape | Description |
|-------|------|------|------|
| grad_weight | Tensor(BF16) | (N, M) | Gradient with respect to weight |
| grad_scale | Tensor(BF16) | (N*M/group_size, 1) | Gradient with respect to scaling factor |
| grad_offset | Tensor(BF16) | (N*M/group_size, 1) | Gradient with respect to offset |

### Algorithm Principle

Backpropagation is based on the gradient formulas of the LSQ+ algorithm, requiring recomputation of forward intermediate variables and correct handling of gradients in clamped and unclamped regions.

#### Forward State Recompute

First recompute the following intermediate variables:
- `protected_scale`: Scale after anti-zero protection
- `alpha`: = protected_scale multiplied by n_levels
- `weight_shifted`: = weight minus offset
- `weight_scaled`: = weight_shifted / alpha (unclamped)
- `weight_clipped`: Clamped values
- `weight_denorm`: Dequantized values

#### Gradient Computation

##### Clamping Mask (STE Activation Region)

$$
\text{mask}[i,j] = \begin{cases} 1, & -\text{clip\_val} \leq W_{\text{scaled}}[i,j] \leq \text{clip\_val} \\ 0, & \text{otherwise} \end{cases}
$$

##### grad_weight Computation

Only elements within the clamping range propagate gradients:

$$
\frac{\partial\text{Loss}}{\partial W} = \frac{\partial\text{Loss}}{\partial W_{\text{out}}} \odot \text{mask}
$$

##### grad_offset Computation

Gradients outside the clamping region accumulate into offset:

$$
\frac{\partial\text{Loss}}{\partial \text{offset}} = \sum_{j \in \text{group}} \left( \frac{\partial\text{Loss}}{\partial W_{\text{out}}} \odot (1 - \text{mask}) \right)
$$

Sum along the group dimension.

##### grad_scale Computation

$$
\frac{\partial\text{Loss}}{\partial s} = \frac{\partial\text{Loss}}{\partial \alpha} \times n_{\text{levels}} \times \mathbf{1}_{s > \varepsilon}
$$

Where:

$$
\frac{\partial\text{Loss}}{\partial \alpha} = \sum_{j \in \text{group}} \left( \frac{\partial\text{Loss}}{\partial W_{\text{out}}} \odot (W_{\text{denorm}} - W_{\text{scaled}} \odot \text{mask}) \right)
$$

### Key Implementation Details

#### Cache-Free Design

This operator uses a cache-free design, recomputing forward intermediate variables during backpropagation, avoiding the memory overhead of storing large amounts of intermediate state.

#### Mask Generation

Use numerical methods to generate the mask, avoiding direct conditional statements:

```python
# Determine whether within clamping range
diff = weight_norm - weight_clipped
abs_diff = abs(diff)
is_out = clip(abs_diff * big_number, 0.0, 1.0)
mask = 1.0 - is_out
```

#### Gradient Accumulation

For scale and offset gradients, sum along the group dimension:

```python
grad_offset = (grad_output * (1 - mask)).sum(dim=1, keepdim=True)
grad_alpha = (grad_output * (weight_denorm - weight_scaled * mask)).sum(dim=1, keepdim=True)
```

### Constraints

1. `group_size` takes values 64, 128, or 256. Data type is int.
2. `grad_output` and `weight` must have the same shape (N, M). M is in [128, 3072] and divisible by `group_size`. Data type is BF16.
3. `scale` and `offset` shapes must be (N*M/group_size, 1). Data type is BF16.
4. `bit` can only be 2, 3, or 4. Data type is int.
5. `eps` must be in the range (0, 1). Data type is float.
6. `clip_val` must be in the range (0, 1). Data type is float.

### Supported Specifications

- Data types: BF16 (input/output), FP32 (internal computation)
- Chip platforms: A2/A3
- Supported bit widths: 2-bit, 3-bit, 4-bit

### Usage Sample

```python
import torch
import pypto

# Set device
torch.npu.set_device(0)

# Create input tensors
N, M = 1024, 2048
group_size = 128
bit = 4
num_groups = N * M // group_size

weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0", requires_grad=True)
scale = torch.abs(torch.randn(num_groups, 1, dtype=torch.bfloat16, device="npu:0")) + 0.01
scale.requires_grad_(True)
offset = torch.randn(num_groups, 1, dtype=torch.bfloat16, device="npu:0", requires_grad=True)

# Forward propagation
output = ai_infra_qat_asymmetric_per_group(weight, scale, offset, group_size=group_size, bit=bit, eps=1e-4, clip_val=0.99)

# Simulate upstream gradient
grad_output = torch.ones_like(output)

# Backward propagation
grad_weight, grad_scale, grad_offset = ai_infra_qat_asymmetric_per_group_backward(
    grad_output, weight, scale, offset, group_size=group_size, bit=bit, eps=1e-4, clip_val=0.99
)

print(f"Weight gradient shape: {grad_weight.shape}")
print(f"Scale gradient shape: {grad_scale.shape}")
print(f"Offset gradient shape: {grad_offset.shape}")
```

---

## Comparison with Symmetric Quantization

| Feature | Symmetric Quantization | Asymmetric Quantization |
|-----|---------|-----------|
| Quantization range | [-Q, Q] | [-Q, Q] + offset |
| Number of parameters | scale | scale + offset |
| Applicable scenario | Symmetric weight distribution | Asymmetric weight distribution |
| Quantization precision | Lower | Higher |
| Computational complexity | Lower | Higher |
