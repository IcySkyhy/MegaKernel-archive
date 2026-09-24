# FFN Module (Feed-Forward Network)

This sample demonstrates how to implement a complete Feed-Forward Network (FFN) module using PyPTO. The module is designed for efficient execution on Ascend NPU hardware and supports multiple activation functions and dynamic shapes.

## Core Features

- **Multiple Activation Functions**: Supports GELU, SwiGLU, and ReLU activations.
- **Static and Dynamic Shapes**: Supports both fixed Batch Size and variable Batch Size scenarios.
- **Configurable Tiling**: Configurable tile shapes optimized for NPU performance.
- **Engineered Implementation**: Includes a configuration class (Dataclass), complete type annotations, and detailed documentation.

## Code Structure

- **`ffn_module.py`**: Core implementation of the FFN module and test script, containing multiple test cases and sample usages.

## How To Run

### Environment Preparation

```bash
# Configure CANN environment variables
# After installation, configure the environment variables. Execute the following command based on the actual path of set_env.sh.
# The above environment variable configuration only takes effect in the current window. You can write the above commands into the environment variable configuration file (such as .bashrc) as needed.

# Default path installation, using root user as a sample (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set device ID
export TILE_FWK_DEVICE_ID=0
```

### Run Tests

```bash
# Run all test cases
python3 ffn_module.py

# List all available test cases
python3 ffn_module.py --list
```

## FFN Architecture Description

The FFN module implements the standard Transformer feed-forward network logic:

### Standard FFN (GELU/ReLU)
```
Input [B, H]
  -> Gate Projection [B, H] @ [H, I] -> [B, I]
  -> Activation (GELU/ReLU)
  -> Down Projection [B, I] @ [I, H] -> [B, H]
  -> Output [B, H]
```

### SwiGLU FFN
```
Input [B, H]
  -> Gate Projection [B, H] @ [H, I] -> [B, I]
  -> Up Projection [B, H] @ [H, I] -> [B, I]
  -> SwiGLU(Gate, Up) -> [B, I]
  -> Down Projection [B, I] @ [I, H] -> [B, H]
  -> Output [B, H]
```
Where: `B` = Batch Size, `H` = Hidden Size, `I` = Intermediate Size.

## Key Configuration Parameters (`FFNConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_size` | `int` | Required | Hidden layer dimension size |
| `intermediate_size` | `int` | Required | Intermediate layer dimension size |
| `activation` | `str` | `"gelu"` | Activation function: `"gelu"`, `"swiglu"`, or `"relu"` |
| `dtype` | `pypto.DataType` | `DT_FP16` | Data type used for computation |
| `use_dynamic_shape` | `bool` | `False` | Whether to support dynamic Batch Size |
| `vec_tile_shape` | `tuple` | `(64, 128)` | Tile shape for vector operations |
| `cube_tile_shape` | `tuple` | `(64, 128, 128)` | Tile shape for matrix operations |
| `basic_batch` | `int` | `32` | Basic Batch size for dynamic processing |

## Best Practices
1. **Performance Tuning**: Adjust `vec_tile_shape` and `cube_tile_shape` based on the model scale. Larger Tiles typically provide better compute density but consume more internal cache.
2. **Data Type**: For LLM inference, use `DT_BF16` for a better balance of precision and performance.
3. **Dynamic Batch**: When the Batch Size of the inference service changes frequently, enable `use_dynamic_shape` and set a reasonable `basic_batch`.

## Precautions
- This module currently mainly supports 2D tensor input.
- GELU activation uses an approximate implementation commonly used in high-performance computing.
- Ensure that the NPU has sufficient memory to hold the weights and intermediate tensors.
