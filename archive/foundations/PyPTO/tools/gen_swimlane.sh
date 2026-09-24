#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Default values
output_folder="./output"
input_type="python"
custom_path=""
number=1  # 默认处理1个最新输出文件夹
start_number=-1  # 起始编号，-1表示从最新开始

# Display help message
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -f, --front TYPE   Specify frontend type (python3, py, python, cpp)"
    echo "                     cpp: output_folder=\"./build/output/bin/output\""
    echo "                     others: output_folder=\"./output\""
    echo "  -p, --path PATH   Specify custom output folder path"
    echo "                     Overrides the -f option if both are provided"
    echo "  -n, --number NUM  Specify number of output folders to process (default: 1)"
    echo "                     Will process folders with consecutive numbers starting from latest or specified start"
    echo "  -s, --start NUM   Specify starting folder number (extracted from folder name after last '_')"
    echo "                     If not specified, starts from the latest folder"
    echo "  -h, --help        Display this help message and exit"
    echo ""
    echo "Default: When no options are used, output_folder=\"./output\""
    echo "Priority: --path > --front > default"
    echo ""
    echo "Folder naming convention: folder names should end with '_<number>' (e.g., output_1, output_2, etc.)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--front)
            shift
            if [[ -n "$1" ]]; then
                input_type="$1"
            else
                echo "Error: -f/--front option requires an argument"
                exit 1
            fi
            shift
            ;;
        -p|--path)
            shift
            if [[ -n "$1" ]]; then
                custom_path="$1"
            else
                echo "Error: -p/--path option requires an argument"
                exit 1
            fi
            shift
            ;;
        -n|--number)
            shift
            if [[ -n "$1" ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
                number="$1"
            else
                echo "Error: -n/--number option requires a positive integer"
                exit 1
            fi
            shift
            ;;
        -s|--start)
            shift
            if [[ -n "$1" ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
                start_number="$1"
            else
                echo "Error: -s/--start option requires a positive integer"
                exit 1
            fi
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Determine output folder based on provided options
# Priority: custom_path > input_type > default
if [[ -n "$custom_path" ]]; then
    # Use custom path if provided
    output_folder="$custom_path"
    echo "Using custom output folder: $output_folder"
else
    # Otherwise, use input type to determine output folder
    case "$input_type" in
        cpp)
            output_folder="./build/output/bin/output"
            ;;
        python3|py|python)
            output_folder="./output"
            ;;
        *)
            echo "Warning: Unknown frontend type '$input_type', using default output directory: ./output"
            output_folder="./output"
            ;;
    esac
fi

# Check if output folder exists
if [[ ! -d "$output_folder" ]]; then
    echo "Error: Output folder does not exist: $output_folder"
    exit 1
fi

# Function to extract number from folder name (after last '_')
extract_number() {
    local folder_name="$1"
    # Extract the part after the last underscore
    local number_part="${folder_name##*_}"
    # Check if it's a valid number
    if [[ "$number_part" =~ ^[0-9]+$ ]]; then
        echo "$number_part"
    else
        echo ""
    fi
}

# Get all output folders and extract their numbers
echo "Scanning output folders in: $output_folder"
output_folders_info=()

# Read all matching folders
while IFS= read -r folder; do
    folder_name=$(basename "$folder")
    folder_number=$(extract_number "$folder_name")
    if [[ -n "$folder_number" ]]; then
        output_folders_info+=("$folder_name:$folder_number")
    fi
done < <(find "$output_folder" -maxdepth 1 -type d -name "output*" 2>/dev/null | sort)

# Check if output folders were found
if [[ ${#output_folders_info[@]} -eq 0 ]]; then
    echo "Error: No valid output folders found in $output_folder"
    echo "Note: Folder names should end with '_<number>' (e.g., output_1, output_2, etc.)"
    exit 1
fi

# Sort folders by number in descending order (highest number first)
sorted_folders=($(printf "%s\n" "${output_folders_info[@]}" | sort -t: -k2 -rn))

# Get the latest folder number
latest_folder_info="${sorted_folders[0]}"
latest_folder_name="${latest_folder_info%%:*}"
latest_folder_number="${latest_folder_info##*:}"

echo "Found ${#sorted_folders[@]} output folder(s)"

# Determine starting number
if [[ $start_number -eq -1 ]]; then
    # Start from the latest folder
    start_number=$latest_folder_number
    echo "Starting from latest folder: $latest_folder_name (Process ID: $start_number)"
else
    echo "Starting from specified Process ID number: $start_number"
fi

# Create array of folders to process
folders_to_process=()
for folder_info in "${sorted_folders[@]}"; do
    folder_name="${folder_info%%:*}"
    folder_number="${folder_info##*:}"

    # Check if this folder should be included
    if [[ $folder_number -le $start_number ]] && [[ $folder_number -gt $((start_number - number)) ]]; then
        folders_to_process+=("$folder_name")
    fi
done

# Sort folders to process in ascending order (from oldest to newest in the range)
folders_to_process=($(printf "%s\n" "${folders_to_process[@]}" | sort -t: -k2 -n))

# Check if we found enough folders
if [[ ${#folders_to_process[@]} -eq 0 ]]; then
    echo "Error: No folders found in the specified range (starting from number $start_number)"
    exit 1
fi

if [[ ${#folders_to_process[@]} -lt $number ]]; then
    echo "Warning: Requested $number folders but only ${#folders_to_process[@]} found in range. Processing all available folders."
    number=${#folders_to_process[@]}
fi

echo "================================================================================"

# Process each output folder
processed_count=0
for folder_name in "${folders_to_process[@]}"; do
    folder_number=$(extract_number "$folder_name")
    echo ""
    echo "Processing folder [$((processed_count + 1))/$number]: $folder_name (number: $folder_number)"
    echo "----------------------------------------------------------------"

    # Build file paths
    prof_data_json_path="$output_folder/$folder_name/tilefwk_L1_prof_data.json"
    program_json_path="$output_folder/$folder_name/program.json"
    topo_txt_path="$output_folder/$folder_name/dyn_topo.txt"

    # Check if required files exist
    missing_files=0
    if [[ ! -f "$prof_data_json_path" ]]; then
        echo "  Warning: File does not exist: tilefwk_L1_prof_data.json"
        missing_files=$((missing_files + 1))
    fi

    if [[ ! -f "$program_json_path" ]]; then
        echo "  Warning: File does not exist: program.json"
        missing_files=$((missing_files + 1))
    fi

    if [[ ! -f "$topo_txt_path" ]]; then
        echo "  Warning: File does not exist: dyn_topo.txt"
        missing_files=$((missing_files + 1))
    fi

    if [[ $missing_files -gt 0 ]]; then
        echo "  Skipping folder $folder_name due to missing files"
        continue
    fi

    # Extract hash from program.json
    original_hash=$(grep -m 1 -Eo '"hash": [0-9]+' "$program_json_path" 2>/dev/null | awk '{print $2}')

    # Display file info
    echo "Configuration file paths:"
    echo "  prof_data_json: $prof_data_json_path"
    echo "  program_json:   $program_json_path"
    echo "  topo_txt:       $topo_txt_path"
    echo ""

    # Execute Python script
    echo "  Generating swim lane diagram..."
    python3 tools/profiling/draw_swim_lane.py "$prof_data_json_path" "$topo_txt_path" "$program_json_path" --label_type=1 --time_convert_denominator=50

    if [[ $? -eq 0 ]]; then
        echo "  ✓ Successfully generated diagram for $folder_name"
        processed_count=$((processed_count + 1))
    else
        echo "  ✗ Failed to generate diagram for $folder_name"
    fi
done

echo ""
echo "================================================================================"
echo "Summary: Processed $processed_count out of $number requested folder(s)"
if [[ $processed_count -eq 0 ]]; then
    exit 1
fi
