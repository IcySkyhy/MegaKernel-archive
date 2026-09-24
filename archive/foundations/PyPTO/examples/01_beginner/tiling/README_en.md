# Tiling Operations

This directory contains usage samples of Tiling configuration operations in PyPTO, covering Cube Tiling and Vector Tiling.

## Overview

Tiling is the core of Ascend NPU performance optimization. By dividing tensor blocks reasonably, you can maximize the parallel utilization of hardware units (such as Cube units and Vector units) and optimize memory access patterns.

The samples cover the following content:
- **Cube Tile Shapes**: Tiling configuration designed specifically for matrix multiplication (Cube operations).
- **Vector Tile Shapes**: Tiling configuration designed specifically for element-wise operations and vector operations.

## Code File Description

- **`tiling_config.py`**: Contains all samples related to Tiling configuration.
  - `test_set_cube_tile_shapes_basic`: Basic usage of Cube Tiling.
  - `test_set_vec_tile_shapes_basic`: Basic usage of Vector Tiling.
  - `test_different_tile_shapes_on_results`: Verifies the consistency of computation results under different Tiling shapes.
  - `test_different_tile_shapes_on_runtime`: Demonstrates the impact of different Tiling configurations on runtime performance.

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

### Run the Samples

```bash
# Run all Tiling-related samples
python3 tiling_config.py

# List all available Tiling test cases
python3 tiling_config.py --list

# Run a specific test case
python3 tiling_config.py cube_tile::test_set_cube_tile_shapes_basic
```

## Core API and Concepts

### 1. Cube Tiling (Matrix Multiplication)
Cube Tiling configures the tile sizes for three dimensions: M, K, and N.
```python
# Set Cube Tile shapes: [M_tile], [K_tile], [N_tile]
pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
```

### 2. Vector Tiling (Vector and Element-wise Operations)
The number of Vector Tiling blocks must match the number of tensor dimensions (1 to 4).
```python
# Set Vector Tile shapes for a 3-dimensional tensor
pypto.set_vec_tile_shapes(1, 2, 8)
```

### 3. Impact of Tiling on Performance
- **Consistency**: Regardless of how you partition the Tiling, the final computation results remain consistent.
- **Performance**: Reasonable Tiling shapes can significantly reduce the number of data transfers between L1, L0 caches, and Global Memory, improving the utilization of compute units.

## Best Practices
- **Match Operator Type**: Use Cube Tiling for matrix operations and Vector Tiling for vector operations.
- **Align with Hardware**: Ascend NPU hardware typically has specific alignment requirements (such as 16x16 or 32x32). Refer to the hardware architecture specifications when choosing Tiling shapes.
- **Dynamic Adjustment**: When developing complex operators, experiment with different Tiling combinations to find the optimal performance.

## Precautions
- Configure Tiling inside the JIT kernel function, before the actual computation occurs.
- When verifying tile sizes, ensure that the tile shape does not exceed the actual shape of the tensor (unless the automatic padding mechanism is enabled).
