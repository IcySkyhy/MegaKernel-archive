#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import argparse
import json
import os

import matplotlib.pyplot as plt

# Define color mapping
COLORS = {"ATTN": "blue", "FFN": "green"}


def read_json(filepath):
    """Read JSON file"""
    with open(filepath, "r") as file:
        return json.load(file)


def plot_swim_lane(input_data, out_path):
    """Draw Swim Lane diagram"""
    # Set image size and DPI
    fig, ax = plt.subplots(figsize=(20, 10), dpi=200)

    # Calculate the maximum value of the x-axis
    max_time = max(task["execEnd"] for block in input_data for task in block["tasks"])
    ax.set_xlim(0, max_time)  # Set x-axis range

    # Set spacing between blocks
    bar_height = 0.25  # Height of the bar chart
    spacing = 0.5  # Height of the spacing (white space width: bar width = 1:2)

    # Iterate through each block
    for block in input_data:
        block_idx = block["blockIdx"]
        y_position = block_idx * (bar_height + spacing)  # Calculate y-axis position

        # Iterate through each task
        for task in block["tasks"]:
            start = task["execStart"]
            end = task["execEnd"]
            task_type = task["type"]

            # Use hlines to draw horizontal bar chart
            ax.hlines(
                y=y_position,
                xmin=start,
                xmax=end,
                colors=COLORS[task_type],
                linewidth=bar_height * 100,  # Control the thickness of the bar
            )

    # Set y-axis labels
    y_ticks = [block["blockIdx"] * (bar_height + spacing) for block in input_data]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([block["coreType"] for block in input_data])

    # Remove ylabel
    ax.set_ylabel("")

    # Set chart title and x-axis label
    ax.set_title("ATTN & FFN Swim Lane Diagram, Time Cost = %d" % max_time)
    ax.set_xlabel("Time")

    # Save the chart
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a swim lane diagram from JSON data.")
    parser.add_argument("json_path", type=str, help="Path to the JSON file")
    args = parser.parse_args()

    # Get the JSON file path
    json_path = args.json_path
    if not os.path.isfile(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    # Set the output file path
    output_dir = os.path.dirname(json_path)
    output_path = os.path.join(output_dir, "attn-ffn.swim.png")

    # Read JSON data and plot the diagram
    data = read_json(json_path)
    plot_swim_lane(data, output_path)
    print(f"Swim lane diagram saved to {output_path}")


if __name__ == "__main__":
    main()
