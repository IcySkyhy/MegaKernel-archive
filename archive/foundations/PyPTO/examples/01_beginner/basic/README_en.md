# Basic Operations

This sample demonstrates the basic operator operations and typical programming patterns of PyPTO. It is very suitable for beginners to quickly get started.

## Sample Code Features

- **JIT Compilation**: Use the `@pypto.jit` decorator to define kernel functions that execute on the NPU.
- **PyTorch Integration**: Directly accept PyTorch NPU tensors as input and output, seamlessly integrating with existing deep learning workflows.
- **Explicit Tiling Control**: Manually optimize hardware execution efficiency through `set_vec_tile_shapes` and `set_cube_tile_shapes`.
- **Dynamic Shape**: Supports adjusting tensor shapes dynamically at runtime, without having to determine them at compile time.

## Code Structure

- **`basic_ops.py`**: Main script containing all samples, serving as the entry point for a quick overview.
  - `test_add()`: Sample 1 - Addition.
  - `test_erfc()`: Sample 2 - Element-wise operation (ERFC).
  - `test_matmul()`: Sample 3 - Matrix multiplication.
  - `test_sum()`: Sample 4 - Reduction operation (Sum).
  - `test_dynamic_add()`: Sample 5 - Dynamic Shape.

For more detailed usage of various operators, refer to the sibling directories:

- `../compute/`: Element-wise operators, matrix multiplication, reduction operators.
- `../tiling/`: Tiling configuration strategies.
- `../transform/`: Transform operators.

## How To Run

### Environment Preparation

```bash
# Configure CANN environment variables
# After installation, configure the environment variables. Execute the following command based on the actual path of set_env.sh.
# The above environment variable configuration only takes effect in the current window. You can write the above commands into the environment variable configuration file (such as .bashrc) as needed.

# Default path installation, using root user as a sample (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Execute the Script

```bash
# Run all samples
python3 basic_ops.py

# Run a specific sample (for example, Sample 1: Addition)
python3 basic_ops.py -t add
```

## Precautions

- **Tile Size**: The choice of Tiling shapes significantly affects operator performance. Typically, you should set them according to the vector/matrix computation unit size of the NPU architecture.
- **Environment**: Ensure that `torch_npu` is correctly installed and can recognize the Ascend GPU.
