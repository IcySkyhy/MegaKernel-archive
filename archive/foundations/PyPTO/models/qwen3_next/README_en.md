# Qwen3Next Samples

This directory contains development sample code for the PyPTO Qwen3Next model. We implement the core attention mechanism of Qwen3Next, delivering the **Chunk Gated Delta Rule** operator, which is an efficient linear attention mechanism designed for long sequence modeling scenarios.

## Parameter Description/Constraints

- Shape format field meanings

| Field | Full Name/Meaning | Value Rules and Description |
|--------|---------------|----------------|
| T | Total Tokens | Range: sum of sequence lengths across all batches |
| B | Batch (input sample batch size) | Range: inferred from act_seq_len |
| L | Chunk Length | Fixed value: 128 |
| Nqk | Query/Key Head Num | Range: supports 2, 4, 16, and so on |
| Nv | Value Head Num | Range: supports 4, 8, 32, and so on, Nv // Nqk must be an integer (GQA grouping) |
| D | Head Dimension | Fixed value: 128 |
| S | Sequence Length | Range: supports dynamic sequence length |

---

# chunk_gated_delta_rule

## Description

The `chunk_gated_delta_rule` operator corresponds to the core attention computation module in the Qwen3Next network, implementing a chunked linear attention mechanism based on the **Gated Delta Rule**. This operator fuses the following key operations:

- **L2 Normalization**: Normalize Query and Key
- **Gate Cumulative Sum**: Compute the cumulative sum of gate signals
- **Decay Mask Generation**: Generate temporal decay mask matrices
- **Pre-Attention Calculation**: Compute the product of the KKT matrix and the decay mask
- **Matrix Inversion**: Efficiently invert the attention matrix
- **Recurrent State Attention**: Compute the final attention output with historical states

Through chunked processing and operator fusion, this operator significantly reduces the O(n-squared) complexity of traditional Attention, achieving linear time complexity attention computation, especially suitable for efficient inference with very long sequences.

## Mathematical Formulas

### L2 Normalization

$$
\text{L2Norm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

$$
\mathbf{q}_{norm} = \text{L2Norm}(\mathbf{q}), \quad \mathbf{k}_{norm} = \text{L2Norm}(\mathbf{k})
$$

### Gate Cumulative Sum

$$
\mathbf{g}_{cum} = \text{cumsum}(\mathbf{g}) = \mathbf{T}_{tril} \cdot \mathbf{g}
$$

Where $\mathbf{T}_{tril}$ is a lower triangular matrix of all ones.

### Decay Mask

$$
\mathbf{D}_{decay} = \exp\left((\mathbf{g}_{cum} - \mathbf{g}_{cum}^T) \odot \mathbf{T}_{tril}\right)
$$

### Pre-Attention Matrix

$$
\mathbf{k}_{\beta} = \mathbf{k} \odot \boldsymbol{\beta}
$$

$$
\mathbf{A} = (\mathbf{k}_{\beta} \cdot \mathbf{k}^T) \odot \mathbf{D}_{decay} \odot \mathbf{M}_{mask}
$$

### Matrix Inversion (Iterative Method)

$$
\mathbf{A}^{-1} = (\mathbf{I} - \mathbf{A})^{-1} \approx \mathbf{I} + \mathbf{A} + \mathbf{A}^2 + \cdots
$$

Use chunked recurrence for efficient computation:

$$
\mathbf{A}^{-1}_{i,:i} = \mathbf{A}^{-1}_{i-1,:i-1} + \mathbf{A}_{i,:i} \cdot \mathbf{A}^{-1}_{i-1,:i-1}
$$

### Value and Key Cumulative Decay

$$
\mathbf{v}_{out} = \mathbf{A}^{-1} \cdot (\mathbf{v} \odot \boldsymbol{\beta})
$$

$$
\mathbf{k}_{cumdecay} = \mathbf{A}^{-1} \cdot (\mathbf{k}_{\beta} \odot \exp(\mathbf{g}_{cum}))
$$

### Recurrent State Attention

$$
\mathbf{v}' = \mathbf{k}_{cumdecay} \cdot \mathbf{S}^T
$$

$$
\mathbf{o}_{inter} = (\mathbf{q} \odot \exp(\mathbf{g}_{cum})) \cdot \mathbf{S}^T
$$

$$
\mathbf{o}_{chunk} = \mathbf{o}_{inter} + (\mathbf{q} \cdot \mathbf{k}^T \odot \mathbf{D}_{decay} \odot \mathbf{T}_{tril}) \cdot (\mathbf{v}_{out} - \mathbf{v}')
$$

### State Update

$$
\mathbf{S}_{new} = \mathbf{S} \cdot \exp(g_{last}) + \mathbf{v}_{new}^T \cdot \mathbf{k}_{gexp}
$$

Where:
- $\mathbf{k}_{gexp} = \mathbf{k} \odot \exp(g_{last} - \mathbf{g})$
- $g_{last}$ is the gate value at the last position of the current chunk

## Function Prototype

```python
def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    states: torch.Tensor,
    mask: torch.Tensor,
    tril_mask: torch.Tensor,
    eye: torch.Tensor,
    act_seq_len: torch.Tensor,
    core_attn_out: torch.Tensor,
    last_state_data: torch.Tensor
):
```

## Parameter Description

>**Precautions:**<br>
>
>- T is the total of all batch sequence lengths, B is the input sample batch size, L is the chunk length (fixed at 128), Nqk is the number of Query/Key heads, Nv is the number of Value heads (supports GQA, Nv // Nqk must be an integer), D is the dimension of each attention head (fixed at 128).

-   **query** (`Tensor`): Query vector. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [T, Nqk, D].

-   **key** (`Tensor`): Key vector. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [T, Nqk, D].

-   **value** (`Tensor`): Value vector. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [T, Nv, D].

-   **beta** (`Tensor`): Beta scaling factor, controls the weighting of Key and Value. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [T, Nv].

-   **gate** (`Tensor`): Gate signal, controls temporal decay. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [T, Nv].

-   **states** (`Tensor`): Initial recurrent state matrix. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [B, Nv, D, D].

-   **mask** (`Tensor`): Attention mask matrix (lower triangular negative value mask). Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

-   **tril_mask** (`Tensor`): Lower triangular mask matrix. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

-   **eye** (`Tensor`): Identity matrix for matrix inversion (specially processed). Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [16, 128].

-   **act_seq_len** (`Tensor`): Cumulative sequence length index for each batch. Does not support non-contiguous. Data format supports ND. Data type supports `int32`. Shape is [B+1]. For example, [0, 4096, 8192] means batch 0 has sequence length 4096 and batch 1 has sequence length 4096.

-   **core_attn_out** (`Tensor`): Output result of attention computation. Data format supports ND. Data type supports `float32`. Shape is [T, Nv, D].

-   **last_state_data** (`Tensor`): Updated recurrent state matrix, usable for the next sequence chunk computation. Data format supports ND. Data type supports `float32`. Shape is [B, Nv, D, D].

## Submodule Description

### l2norm

L2 normalization module, normalizes Query and Key.

```python
def l2norm(
    query: pypto.Tensor,
    key: pypto.Tensor,
    eps: float = 1e-6
) -> tuple[pypto.Tensor, pypto.Tensor]:
```

**Parameters:**

- **query** (`Tensor`): Input query tensor. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **key** (`Tensor`): Input key tensor. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **eps** (`float`): Small constant to prevent division by zero. Data type supports `float`. Default value is 1e-6.

**Return Values:**

- **query_after_l2norm** (`Tensor`): Normalized query tensor. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **key_after_l2norm** (`Tensor`): Normalized key tensor. Data format supports ND. Data type supports `float32`. Shape is [L, D].

---

### pre_attn

Pre-attention computation module, computes gate cumulative sum, decay mask, pre-attention matrix, and weighted key.

```python
def pre_attn(
    gate_view: pypto.Tensor,
    key_view_2d: pypto.Tensor,
    beta_view: pypto.Tensor,
    tril: pypto.Tensor,
    mask: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]:
```

**Parameters:**

- **gate_view** (`Tensor`): Gate signal. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, 1].

- **key_view_2d** (`Tensor`): Key tensor. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **beta_view** (`Tensor`): Beta scaling factor. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, 1].

- **tril** (`Tensor`): Lower triangular matrix. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

- **mask** (`Tensor`): Mask matrix. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

**Return Values:**

- **gate_cum** (`Tensor`): Gate cumulative sum. Data format supports ND. Data type supports `float32`. Shape is [L, 1].

- **decay_mask** (`Tensor`): Decay mask. Data format supports ND. Data type supports `float32`. Shape is [L, L].

- **A** (`Tensor`): Pre-attention matrix. Data format supports ND. Data type supports `float32`. Shape is [L, L].

- **key_beta** (`Tensor`): Weighted key. Data format supports ND. Data type supports `float32`. Shape is [L, D].

---

### inverse_pto

Matrix inversion module, uses chunked recurrence algorithm to efficiently compute the inverse of large matrices.

```python
def inverse_pto(
    attn: pypto.Tensor,
    eye: pypto.Tensor,
    size: int
) -> pypto.Tensor:
```

**Parameters:**

- **attn** (`Tensor`): Attention matrix to invert. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

- **eye** (`Tensor`): Identity matrix (specially processed). Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

- **size** (`int`): Matrix size. Data type supports `int`.

**Return Values:**

- **attn_inv** (`Tensor`): Inverse matrix. Data format supports ND. Data type supports `float32`. Shape is [L, L].

---

### inverse_pto_min_length

Tail-axis concatenation optimized matrix inversion module.

```python
def inverse_pto_min_length(
    attn_dim0: pypto.Tensor,
    attn_dim1: pypto.Tensor,
    eye: pypto.Tensor,
    row_num: int,
    col_num: int,
) -> pypto.Tensor:
```

**Parameters:**

- **attn_dim0** (`Tensor`): Attention matrix block concatenated along dimension 0. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L // 8].

- **attn_dim1** (`Tensor`): Attention matrix block concatenated along dimension 1. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L // 8, L].

- **eye** (`Tensor`): Identity matrix. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

- **row_num** (`int`): Number of rows. Data type supports `int`. Value is L // 8.

- **col_num** (`int`): Number of columns. Data type supports `int`. Value is L.

**Return Values:**

- **res** (`Tensor`): Inverse result matrix. Data format supports ND. Data type supports `float32`. Shape is [L, L].

---

### inverse_matmul

Small matrix inversion module, used as a sub-computation for chunked matrix inversion.

```python
def inverse_matmul(
    attn: pypto.Tensor,
    attn_1_1_inv: pypto.Tensor,
    attn_2_2_inv: pypto.Tensor,
    x_ofs: int,
    y_ofs: int,
    len: int,
) -> pypto.Tensor:
```

**Parameters:**

- **attn** (`Tensor`): Original attention matrix. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

- **attn_1_1_inv** (`Tensor`): Inverse of the upper-left submatrix. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [len, len].

- **attn_2_2_inv** (`Tensor`): Inverse of the lower-right submatrix. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [len, len].

- **x_ofs** (`int`): Row offset. Data type supports `int`.

- **y_ofs** (`int`): Column offset. Data type supports `int`.

- **len** (`int`): Submatrix length. Data type supports `int`.

**Return Values:**

- **attn_inv** (`Tensor`): Merged inverse matrix. Data format supports ND. Data type supports `float32`. Shape is [len * 2, len * 2].

---

### cal_value_and_key_cumdecay

Compute weighted value and key cumulative decay.

```python
def cal_value_and_key_cumdecay(
    attn: pypto.Tensor,
    value_view: pypto.Tensor,
    beta_view: pypto.Tensor,
    key_beta: pypto.Tensor,
    gate_cum: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor]:
```

**Parameters:**

- **attn** (`Tensor`): Inverse attention matrix. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

- **value_view** (`Tensor`): Value tensor. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **beta_view** (`Tensor`): Beta scaling factor. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **key_beta** (`Tensor`): Weighted key. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **gate_cum** (`Tensor`): Gate cumulative sum. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, 1].

**Return Values:**

- **value_out** (`Tensor`): Weighted value output. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **key_cum_out** (`Tensor`): Key cumulative decay output. Data format supports ND. Data type supports `float32`. Shape is [L, D].

---

### recurrent_state_attn_all

Recurrent state attention computation, combines historical states to compute the final attention output and update the state.

```python
def recurrent_state_attn_all(
    query: pypto.Tensor,
    key: pypto.Tensor,
    value: pypto.Tensor,
    k_cumdecay: pypto.Tensor,
    gate: pypto.Tensor,
    state: pypto.Tensor,
    decay_mask: pypto.Tensor,
    tril: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor]:
```

**Parameters:**

- **query** (`Tensor`): Query tensor. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **key** (`Tensor`): Key tensor. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **value** (`Tensor`): Value tensor. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, Dv].

- **k_cumdecay** (`Tensor`): Key cumulative decay. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, Dk].

- **gate** (`Tensor`): Gate cumulative sum. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, 1].

- **state** (`Tensor`): Current recurrent state. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [D, D].

- **decay_mask** (`Tensor`): Decay mask. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

- **tril** (`Tensor`): Lower triangular matrix. Does not support non-contiguous. Data format supports ND. Data type supports `float32`. Shape is [L, L].

**Return Values:**

- **chunk_attn_out** (`Tensor`): Attention output of the current chunk. Data format supports ND. Data type supports `float32`. Shape is [L, D].

- **state_new** (`Tensor`): Updated state. Data format supports ND. Data type supports `float32`. Shape is [Dv, Dk].

---

## Supported Configurations

| Configuration Item | Supported Range | Description |
|--------|----------|------|
| T (total sequence length) | 1K to 1M+ | Supports very long sequences |
| B (batch size) | 1 to 128 | Adjust based on memory |
| Nqk (QK head num) | 2, 4, 16 | Query/Key multi-head count |
| Nv (V head num) | 4, 8, 32 | Value multi-head count (GQA) |
| D (head dimension) | 128 | Fixed value |
| L (chunk size) | 128 | Fixed value |

## Runtime Configuration

```python
@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "ready_on_host_tensors": ["tensor_a", "tensor_b"]
    },
    debug_options={"runtime_debug_mode": 1},
)
def test_kernel(
    tensor_a: pypto.Tensor(),
    tensor_b: pypto.Tensor(),
    tensor_c: pypto.Tensor(),
):
    ******
```

## Performance Features

1. **Linear time complexity**: Compared to traditional O(n-squared) Attention, Gated Delta Rule achieves O(n) linear complexity
2. **Chunked processing**: Uses fixed-size (128) chunked processing, effectively utilizing NPU parallel computing capability
3. **Operator fusion**: Multiple computation steps complete within a single kernel, reducing memory access overhead
4. **State reuse**: Recurrent state is reusable across sequence chunks, supporting streaming inference
5. **GQA support**: Supports Grouped Query Attention, reducing KV Cache memory usage

## Call Sample

- For details, refer to [gated_delta_rule_impl.py](./gated_delta_rule_impl.py)

## References

- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) - Theoretical foundation of Gated Delta Rule
- [Linear Attention](https://arxiv.org/abs/2006.16236) - Linear attention mechanism
- [GQA: Grouped Query Attention](https://arxiv.org/abs/2305.13245) - Grouped query attention
