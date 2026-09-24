# Hello World Sample

This sample demonstrates the simplest tensor addition operation in PyPTO. It is suitable for developers who are new to the framework to quickly understand the basic workflow.

## Overview

This sample demonstrates the complete PyPTO development process through a simple tensor addition:

- Define a kernel function using `@pypto.jit`.
- Create input data through PyTorch tensors.
- Call the JIT kernel to execute the computation and verify the result.

## Code Structure

- **`hello_world.py`**: Hello World sample script containing the most basic tensor addition operation.

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
# Run the Hello World sample
python3 hello_world.py

# Run in simulation mode
python3 hello_world.py --run_mode sim
```

## Precautions

- This sample is the first step for getting started. After completing it, continue learning with [Beginner Samples (01_beginner)](../01_beginner/README_en.md).
