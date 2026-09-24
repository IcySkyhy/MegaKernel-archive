# PyPTO Sample Code (Examples)

This directory contains a series of PyPTO development sample codes designed to guide developers on how to use this AI programming framework. The sample codes progressively demonstrate the framework's features based on the developer's learning path.

## Directory Structure

The sample codes are divided into the following levels:

- **00_hello_world (Getting Started)**: Simple tensor addition, a Hello World sample suitable for beginners to understand framework initialization.
- **01_beginner (Beginner)**: Basic operations and core concepts, suitable for developers new to PyPTO.
- **02_intermediate (Intermediate)**: Neural network components, operator combinations, and Runtime features.
- **03_advanced (Advanced)**: Complex architectures (such as Attention), advanced patterns, and system-level optimization.
- **models**: Real-world large language model (LLM) operator implementation samples.

## Quick Start

1. **First time user?** Start from [Beginner Samples (01_beginner)](01_beginner/README_en.md).
2. **Building a neural network?** Refer to [Intermediate Samples (02_intermediate)](02_intermediate/README_en.md).
3. **Exploring advanced patterns?** Refer to [Advanced Samples (03_advanced)](03_advanced/README_en.md).
4. **LLM operator implementations?** Explore [Model Samples (../models/)](../models).

### Environment Preparation
For environment preparation, refer to [Environment Setup](../docs/zh/install/prepare_environment.md) to complete the basic environment setup.

### Software Installation
For software installation, refer to [Software Installation](../docs/zh/install/build_and_install.md) to complete the PyPTO software installation.

### Pre-Run Configuration (Optional)
If you need to run in a real NPU environment, refer to the following configuration:

```bash
# Configure CANN environment variables
# After installation, configure the environment variables. Execute the following command based on the actual path of set_env.sh.
# The above environment variable configuration only takes effect in the current window. You can write the above commands into the environment variable configuration file (such as .bashrc) as needed.

# Default path installation, using root user as a sample (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set NPU device ID (required when running NPU samples)
export TILE_FWK_DEVICE_ID=0
```
Additional note: If you need to run models-related samples, run them on a real device.

## How To Run Samples

Most sample scripts support running all tests or specifying a specific test:

```bash

# Run all beginner basic operation samples (default is NPU mode)
python3 examples/01_beginner/basic/basic_ops.py

# Run a specific sample
python3 examples/01_beginner/basic/basic_ops.py matmul::test_matmul

# List all available samples in the script
python3 examples/01_beginner/basic/basic_ops.py --list

# Specify simulation (CPU) mode
python3 examples/01_beginner/basic/basic_ops.py --run_mode sim

```

## Learning Path Suggestions

1. **Phase 1: Build a Solid Foundation**
   - [Hello World](00_hello_world/hello_world.py)
   - [01_beginner/basic](01_beginner/basic/README_en.md)
   - [01_beginner/tiling](01_beginner/tiling/README_en.md)
   - [01_beginner/compute](01_beginner/compute/README_en.md)
   - [01_beginner/transform](01_beginner/transform/README_en.md)

2. **Phase 2: Advanced Components**
   - [02_intermediate/operators](02_intermediate/operators/README_en.md)
   - [02_intermediate/basic_nn](02_intermediate/basic_nn/README_en.md)
   - [02_intermediate/controlflow](02_intermediate/controlflow/README_en.md)

3. **Phase 3: In-Depth Practice**
   - [03_advanced/advanced_nn](03_advanced/advanced_nn/README_en.md)
   - [03_advanced/patterns](03_advanced/patterns/README_en.md)
   - [../models/deepseek_v32_exp](../models/deepseek_v32_exp/README_en.md)
   - [../models/glm_v4_5](../models/glm_v4_5/README_en.md)

---

**Wishing you a fruitful programming journey with PyPTO!**
