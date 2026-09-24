# Graph Capturing Mode Operator Development Sample

This sample demonstrates how to enable graph capturing mode and execute a captured graph, based on a custom Softmax operator.

## Overview

As performance optimization deepens, the host-side overhead introduced by eager mode becomes a bottleneck. This directory currently focuses on:
- **ACLGraph**: Using graph capturing mode (ACLGraph) to offload related tasks to the device for execution, reducing host-side overhead. When the captured graph needs to execute multiple times, you do not need to resubmit tasks. You only need to call the `replay` interface multiple times.

## Sample Features

- **Decorator Annotation**: PyPTO custom operators that interface with `torch.compile` require the `@allow_in_graph` decorator to avoid graph segmentation errors.
- **FakeTensor Handling**: PyPTO custom operators that interface with `torch.compile` must check whether the input is a `FakeTensor`. If it is, return an empty tensor directly.
- **Model Compilation**: Call `torch.compile` to compile the model.
- **Graph Capture**: Use `with torch.npu.graph(g):` to capture the tasks of the first execution of the model.
- **Execute Captured Graph**: Call `replay` to execute the captured graph and compute the Softmax result.

## Code Structure

- **`aclgraph/`**:
  - `aclgraph.py`: Contains graph capturing mode enabling, captured graph execution, and precision comparison against native PyTorch operators.

## Running the Sample

### Environment Setup

```bash
# Configure the CANN environment variables
# After installation, configure the environment variables. For the actual path of set_env.sh, refer to the following command.
# The environment variable configuration above takes effect only in the current window. You can write the commands into an environment variable configuration file (for example, .bashrc) as needed.

# Default path installation, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set the device ID
export TILE_FWK_DEVICE_ID=0
```

### Execute the Script

```bash
python3 examples/03_advanced/aclgraph/aclgraph.py
```

## Precautions
- When the captured graph needs to execute multiple times, update the corresponding input data.
- During graph capture, calling memory synchronization functions is illegal and causes capture failure.
