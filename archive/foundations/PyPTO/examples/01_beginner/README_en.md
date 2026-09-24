# Beginner Examples

This directory contains PyPTO beginner samples designed to help developers quickly grasp the core concepts and basic operator operations of PyPTO.

## Sample Description

The beginner samples are divided into the following categories:

### 1. Basic Operations ([basic](./basic))
- **Content**: Demonstrates the most basic tensor creation, element-wise operations, matrix multiplication, reduction operations, Tiling configuration, and transform operations (View and Assemble). Includes tensor creation operations and the use of Symbolic Scalars.
- **Core Features**: `pypto.tensor`, `@pypto.frontend.jit`, `pypto.from_torch`, `pypto.view`, `pypto.assemble`, `pypto.scalar`.
- **Recommended Audience**: Developers new to PyPTO.

### 2. Compute Operators ([compute](./compute))
- **Content**: Shows detailed usage of various compute operators.
  - `elementwise_ops.py`: Element-wise operators (Add, Sub, Mul, Div, Exp, Log, Abs, Sqrt, Rsqrt, Clip, and so on), including broadcasting mechanism and scalar operations.
  - `matmul_ops.py`: Various configurations of matrix multiplication (Matmul).
  - `reduce_ops.py`: Reduction operators (Sum, Max, Min, and so on).
- **Core Features**: Various mathematical computation APIs.

### 3. Tiling Strategy ([tiling](./tiling))
- **Content**: Introduces how to configure hardware-related Tiling shapes to optimize execution efficiency on the Ascend NPU.
- **Core Features**: `pypto.set_vec_tile_shapes`, `pypto.set_cube_tile_shapes`.

### 4. Transform Operators ([transform](./transform))
- **Content**: Demonstrates tensor shape transformation, slicing, transposition, and other operations.
- **Core Features**: `pypto.transpose`, `pypto.reshape`, `pypto.slice`.

## How To Run

Before running any sample, ensure that the CANN environment is configured and the device ID is set:

```bash
# Configure CANN environment variables
# After installation, configure the environment variables. Execute the following command based on the actual path of set_env.sh.
# The above environment variable configuration only takes effect in the current window. You can write the above commands into the environment variable configuration file (such as .bashrc) as needed.

# Default path installation, using root user as a sample (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set device ID
export TILE_FWK_DEVICE_ID=0
```

Enter the corresponding subdirectory and run the script. For example:

```bash
cd basic
python3 basic_ops.py
```

## Learning Suggestions

1. First, read and run `basic/basic_ops.py` to understand the basic PyPTO workflow.
2. Based on your needs, further study the various operator APIs under `compute`.
3. Learn about `tiling` to understand how to optimize operator performance based on hardware characteristics.
