# Condition

This sample demonstrates how to use conditional branch logic inside PyPTO JIT kernels.

## Overview

In real operator development, you often need to execute different computation paths based on different conditions. This sample covers the following scenarios:
- **Conditional Statements in Nested Loops**: Use `if_else` inside multi-level loops to build conditional branches.
- **Dynamic Axis and Static Condition**: Use a compile-time boolean flag to control branches.
- **Dynamic Axis and Dynamic Condition**: Use a runtime index comparison to control branches.
- **Dynamic Axis and Loop Boundary Condition**: Use `is_loop_begin` or `is_loop_end` for boundary handling.

## Code Structure

- **`condition.py`**: Integrated script containing all conditional branch samples.
  - `test_nested_loops_with_conditions`: Conditional statements in nested loops.
  - `test_add_scalar_loop_dyn_axis_static_cond`: Dynamic axis and static condition.
  - `test_add_scalar_loop_dynamic_axis_dynamic_cond`: Dynamic axis and dynamic condition.
  - `test_add_scalar_loop_dynamic_axis_dynamic_loop_cond`: Dynamic axis and loop boundary condition.

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

### Execute the Script

```bash
# Run all conditional branch samples
python3 condition.py

# List all available test cases
python3 condition.py --list

# Run a specific test case
python3 condition.py nested_loops_with_conditions::test_nested_loops_with_conditions
```

## Precautions

- Conditional branches affect the compiler's code generation path. Complex nested conditions may increase compilation time.
- The choice between dynamic and static conditions depends on whether the condition value is known at compile time.
