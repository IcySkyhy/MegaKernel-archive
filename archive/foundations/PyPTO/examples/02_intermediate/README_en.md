# Intermediate Examples

This directory contains PyPTO intermediate development samples. They mainly demonstrate how to build common neural network components, custom complex operators, and use the advanced features of the PyPTO Runtime.

## Sample Description

The intermediate samples are divided into the following three main categories:

### 1. Neural Network Components ([basic_nn](./basic_nn))
- **Layer Normalization ([layer_normalization](./basic_nn/layer_normalization))**:
    - Demonstrates the implementation of standard LayerNorm and RMSNorm.
    - Involves mean and variance computation.
- **FFN Module ([ffn](./basic_nn/ffn))**:
    - Implements a complete Feed-Forward Network.
    - Supports multiple activation functions (ReLU, GELU, SwiGLU).
    - Combines matrix multiplication, element-wise addition, and activation functions.

### 2. Custom Operators ([operators](./operators))
- **Custom Activation ([activation](./operators/activation))**:
    - Demonstrates how to combine basic operators to build complex activation functions, such as SiLU (Swish), GELU, SwiGLU, and GeGLU.
- **Softmax ([softmax](./operators/softmax))**:
    - Shows an in-depth step-by-step manual implementation of the Softmax operator.
    - Involves `exp` computation and cross-dimension `sum` reduction.

### 3. Runtime Features ([controlflow](./controlflow))
- **Dynamic Shapes ([dynamic.py](./controlflow/others/dynamic.py))**:
    - Demonstrates how to handle dynamic Batch Size or sequence length.
    - Uses the `dynamic_axis` parameter for annotation.
- **Condition and Loop**:
    - `condition/condition.py`: Demonstrates conditional branch logic inside operators.
    - `loop/`: Demonstrates complex loop control logic.
- **Kernel Input Order ([kernel_input.py](./controlflow/others/kernel_input.py))**:
    - Demonstrates the flexibility of JIT kernel input order.

## Core Features

In the intermediate samples, you learn:
- **Complex Operator Combination**: How to encapsulate multiple basic operators into a module with specific functionality.
- **Control Flow**: Using `pypto.loop` and conditional statements inside `@pypto.frontend.jit` kernels.
- **Memory Optimization**: Understanding how to reduce memory usage through `view` and `inplace` operations.

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

Enter the corresponding subdirectory and run the script.

## Learning Suggestions

1. First, learn `operators/activation` to understand how to combine basic operators to create new operators.
2. Learn `basic_nn/layer_normalization` to master the implementation of normalization layers involving reduction operations.
3. Dive into the `controlflow` directory to understand the powerful capabilities of PyPTO in handling real-world complex logic.
