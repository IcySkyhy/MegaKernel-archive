# Advanced Examples

This directory contains PyPTO advanced development samples, showcasing complex architecture implementations, advanced design patterns, and system-level performance tuning.

## Sample Description

The advanced samples cover the following core areas:

### 1. Complex Neural Network Architecture ([advanced_nn](./advanced_nn))
- **Attention Mechanism ([attention](./advanced_nn/attention))**:
    - Implements scaled dot-product attention.
    - Supports multi-head attention.
    - Supports dynamic batch and dynamic sequence length.
    - Demonstrates complex matrix transpose and multiplication combinations.

### 2. Design Patterns ([patterns](./patterns))
- **Multi-Function Module ([function](./patterns/function))**:
    - Shows how to compose multiple independent JIT functions.
    - Sequential composition and residual connection between functions.
    - Builds a complete Transformer block.

### 3. System-Level Tuning ([cost_model](./cost_model))
- **Cost Model ([cost_model.py](./cost_model/cost_model.py))**:
    - Demonstrates how to use the cost model to evaluate and optimize operator execution efficiency.

### 4. Graph Capturing Mode ([aclgraph](./aclgraph))
- **ACLGraph ([aclgraph.py](./aclgraph/aclgraph.py))**:
   - Demonstrates how to use graph capturing mode to optimize host-side overhead.

## Core Features

In the advanced samples, you will encounter:
- **Complex Tensor Transformations**: Frequent use of `transpose` and `reshape` in the attention mechanism.
- **Multi-Function Collaboration**: Understanding how to maintain code readability and maintainability in large-scale model development by composing multiple small functions.
- **Extreme Performance Optimization**: Squeezing hardware performance through deep adaptation of tiling, loop unrolling, and hardware units.

## Running the Samples

Before running any sample, ensure that you have configured the CANN environment and set the device ID:

```bash
# Configure the CANN environment variables
# After installation, configure the environment variables. For the actual path of set_env.sh, refer to the following command.
# The environment variable configuration above takes effect only in the current window. You can write the commands into an environment variable configuration file (for example, .bashrc) as needed.

# Default path installation, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set the device ID
export TILE_FWK_DEVICE_ID=0
```

Navigate to the corresponding subdirectory and run the script.

## Learning Suggestions

1. First, study `advanced_nn/attention` in depth, as it is the core of all modern LLMs.
2. Learn how to organize large operator projects through `patterns/function`.
3. Refer to the real model implementations in the `models` directory to apply these advanced features to production.
