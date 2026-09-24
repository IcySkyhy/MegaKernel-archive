#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
harness_dir=${script_dir}/harness

python_bin=${PYTHON_BIN:-python3}
implementation=${IMPLEMENTATION:-pdl_13page_issue_r96}
warmup=${WARMUP:-5}
iterations=${ITERATIONS:-30}
position_start=${POSITION_START:-32}
position_end=${POSITION_END:-158}
edge_mask=${EDGE_MASK:-0x1f}

if [[ "${implementation}" == "direct_baseline" ]]; then
    default_cross_layer_edge=completion
else
    default_cross_layer_edge=pdl
fi
cross_layer_edge=${CROSS_LAYER_EDGE:-${default_cross_layer_edge}}
output=${OUTPUT:-${repo_root}/artifacts/local/${implementation}-p${position_start}-${position_end}-w${warmup}-i${iterations}-${cross_layer_edge}.json}

"${python_bin}" "${script_dir}/validate_bindings.py" \
    --mk-dir "${repo_root}/demos/low-latency-llama"

mkdir -p "$(dirname -- "${output}")"

"${python_bin}" "${harness_dir}/bench_direct_tk_pdl_ablation.py" \
    --repo "${repo_root}" \
    --mk-dir "${repo_root}/demos/low-latency-llama" \
    --baseline-helper-dir "${harness_dir}" \
    --output "${output}" \
    --implementation "${implementation}" \
    --group 12456 \
    --layer-count 16 \
    --cross-layer-edge "${cross_layer_edge}" \
    --edge-mask "${edge_mask}" \
    --position-start "${position_start}" \
    --position-end "${position_end}" \
    --warmup "${warmup}" \
    --iterations "${iterations}" \
    --device cuda:0
