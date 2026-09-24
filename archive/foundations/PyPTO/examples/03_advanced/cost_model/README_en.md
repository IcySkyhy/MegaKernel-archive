# System-Level Analysis and Optimization Samples

This directory showcases PyPTO tools for system-level analysis, evaluation, and optimization.

## Overview

After operator development, profiling is an essential step. This directory covers:
- **Cost Model**: Demonstrates how to use the PyPTO cost model to evaluate operator execution efficiency in a simulated environment and generate visual analysis results.

## Sample Features

- **Visual Analysis**: Shows how to generate swimlane diagrams to visually inspect the scheduling and execution of hardware units (such as Vector, Cube, and MTE).
- **Simulation Execution**: Demonstrates how to simulate operator execution costs on the Ascend architecture through the cost model without actual NPU hardware.
- **Bottleneck Identification**: Helps you identify memory-bound or compute-bound stages in operators through quantitative data analysis.

## Code Structure

- **`cost_model.py`**: A cost analysis sample based on Softmax, demonstrating how to configure and run a cost model simulation.

## Running the Sample

### Execute the Script

The cost model can typically run simulations in a CPU environment:

```bash
python3 cost_model.py
```

After execution, the results are usually saved as JSON files in the `./output` directory. You can open and view them using the PyPTO ToolKit.

## Precautions

- The cost model provides simulation results. Although they have high reference value, you must verify final performance on real NPU hardware.
- The generated swimlane diagram files are large. Focus on analyzing compute-intensive loop sections.
