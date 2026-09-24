#!/usr/bin/env bash

export MLIR_ENABLE_DUMP=1
export MLIR_DUMP_PATH=/mnt/disks/data/kernel/triton/playaround/mlir/matmul.mlir
export TRITON_REPRODUCER_PATH=/mnt/disks/data/kernel/triton/playaround/mlir
export TRITON_ALWAYS_COMPILE=1

python naive-matmul.py