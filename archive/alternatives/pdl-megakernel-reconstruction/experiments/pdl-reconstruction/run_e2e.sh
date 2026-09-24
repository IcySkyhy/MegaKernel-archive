#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
harness_dir=${script_dir}/harness

python_bin=${PYTHON_BIN:-python3}
model_path=${MODEL_PATH:?Set MODEL_PATH to the verified Llama-3.2-1B-Instruct directory}
native_reference_mk=${NATIVE_REFERENCE_MK:-}
output=${OUTPUT:-${repo_root}/artifacts/local/e2e-p32-d128.json}
correctness_output=${CORRECTNESS_OUTPUT:-${output%.json}.correctness.json}
preflight_output=${PREFLIGHT_OUTPUT:-${correctness_output%.json}.preflight.json}
position=${POSITION:-32}
fixed_input_token=1000
prompt_length=${PROMPT_LENGTH:-32}
output_length=${OUTPUT_LENGTH:-128}
warmup=${WARMUP:-3}
iterations=${ITERATIONS:-10}
correctness_warmup=${CORRECTNESS_WARMUP:-1}
correctness_iterations=${CORRECTNESS_ITERATIONS:-1}
graph_edge_mode=${GRAPH_EDGE_MODE:-programmatic}
edge_mask=${EDGE_MASK:-0x1f}
direct_tk_pdl_body=${DIRECT_TK_PDL_BODY:-13page_issue_r96}
expected_model_sha256=${EXPECTED_MODEL_SHA256:-1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f}
model_file=${model_path}/model.safetensors

case "${graph_edge_mode}" in
    programmatic|launch-completion) ;;
    *)
        echo "run_e2e.sh requires an editable PDL graph edge mode" >&2
        exit 64
        ;;
esac

mkdir -p "$(dirname -- "${output}")"
mkdir -p "$(dirname -- "${correctness_output}")"
mkdir -p "$(dirname -- "${preflight_output}")"

# Validate the compiled entry points and their build provenance before reading
# or hashing the multi-gigabyte checkpoint.
echo "Running binding preflight -> ${preflight_output}" >&2
if [[ -n "${native_reference_mk}" ]]; then
    "${python_bin}" "${script_dir}/validate_bindings.py" \
        --mk-dir "${repo_root}/demos/low-latency-llama" \
        --persistent-module "${native_reference_mk}" \
        > "${preflight_output}"
else
    "${python_bin}" "${script_dir}/validate_bindings.py" \
        --mk-dir "${repo_root}/demos/low-latency-llama" \
        --require-persistent \
        > "${preflight_output}"
fi

if [[ ! -f "${model_file}" ]]; then
    echo "Missing ${model_file}" >&2
    exit 66
fi

actual_model_sha256=$(sha256sum "${model_file}" | awk '{print $1}')
if [[ "${actual_model_sha256}" != "${expected_model_sha256}" ]]; then
    echo "model.safetensors SHA-256 mismatch" >&2
    echo "expected: ${expected_model_sha256}" >&2
    echo "actual:   ${actual_model_sha256}" >&2
    exit 65
fi

run_harness() {
    local harness_output=$1
    shift
    local -a harness_args=(
        --repo "${repo_root}"
        --mk-dir "${repo_root}/demos/low-latency-llama"
        --baseline-helper-dir "${harness_dir}"
        --output "${harness_output}"
        --model-path "${model_path}"
        --model-safetensors-sha256 "${actual_model_sha256}"
        --position "${position}"
        --fixed-input-token "${fixed_input_token}"
        --implementation extracted_tk
        --direct-tk-pdl-body "${direct_tk_pdl_body}"
        --graph-edge-mode "${graph_edge_mode}"
        --launch-completion-edge-mask "${edge_mask}"
        --correctness-contract "${script_dir}/correctness_contract.json"
        --binding-contract "${script_dir}/binding_contract.json"
        --binding-preflight-artifact "${preflight_output}"
        --enforce-correctness-contract
    )
    if [[ -n "${native_reference_mk}" ]]; then
        harness_args+=(--native-reference-mk "${native_reference_mk}")
    fi
    "${python_bin}" \
        "${harness_dir}/bench_full_standalone_graph_direct_tk_e2e.py" \
        "${harness_args[@]}" "$@"
}

echo "Running fixed-position tensor correctness gate -> ${correctness_output}" >&2
run_harness "${correctness_output}" \
    --warmup "${correctness_warmup}" \
    --iterations "${correctness_iterations}"

echo "Correctness gate passed; running P${prompt_length}/D${output_length} trajectory -> ${output}" >&2
run_harness "${output}" \
    --trajectory-prompt-length "${prompt_length}" \
    --trajectory-output-length "${output_length}" \
    --warmup "${warmup}" \
    --iterations "${iterations}" \
    --fixed-correctness-artifact "${correctness_output}"
