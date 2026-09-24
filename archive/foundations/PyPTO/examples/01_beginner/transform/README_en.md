# Transform Operations

This directory contains usage samples of tensor transform operators in PyPTO, including Assemble, Gather, Concat, and View operations.

## Overview

The transform operator samples cover the following content:
- **Assemble**: Place a small tensor at a specified offset position within a large tensor.
- **Gather**: Collect elements from a source tensor based on an index tensor.
- **Concat**: Concatenate multiple tensors along a specified dimension.
- **View**: Create a view of a tensor, supporting specified shapes and offsets without data copying.

## Code File Description

- **`transform_ops.py`**: Comprehensive sample containing all transform operators.
  - `test_assemble_basic`: Basic Assemble operation.
  - `test_gather_basic`: Basic Gather operation.
  - `test_concat_basic`: Basic Concat operation.
  - `test_view_basic`: Basic View operation.
- **`add_scalar_loop_view_assemble.py`**: Sample demonstrating scalar addition combined with View and Assemble in a loop tiling pattern.

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
# Run all transform-related samples
python3 transform_ops.py

# List all available transform test cases
python3 transform_ops.py --list

# Run a specific test case
python3 transform_ops.py assemble::test_assemble_basic
```

## Core API Description

### 1. Assemble
Commonly used to assemble computed Tile results back into a global tensor.
```python
# Place small_tensor at the [0, 0] offset in large_tensor
pypto.assemble(small_tensor, offsets=[0, 0], large_tensor)
```

### 2. Gather
Select data from an input tensor based on indices.
```python
# Gather along dimension 0
pypto.gather(input_tensor, dim=0, index_tensor)
```

### 3. Concat
Concatenate multiple tensors in order.
```python
# Concatenate two tensors along dimension 1
pypto.concat([tensor1, tensor2], dim=1)
```

### 4. View
Create a reference pointing to a partial region of the original tensor. It is key to implementing Tiling loops.
```python
# Create a 4x4 view with an offset of [0, 4]
view = pypto.view(tensor, shape=[4, 4], offsets=[0, 4])
```

## Precautions
- **Assemble and View**: These two operators are typically used together to implement manual Tiling loops.
- **Dimension Alignment**: When performing Concat operations, the shapes of all dimensions except the concatenation dimension must be consistent.
- **Data Copying**: The `view` operation does not involve data copying, while `gather` and `concat` typically produce new data copies.
