#!/usr/bin/env python3
"""Validate the compiled bindings required by the reconstruction runners."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path


PREFLIGHT_SCHEMA = "hazy-binding-preflight-v1"
TOOLCHAIN_FIELDS = ("cuda_compiler", "cuda_runtime_header_version")


@lru_cache(maxsize=None)
def file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cuda_compiler_identity(module: object) -> dict[str, int | None]:
    return {
        field: getattr(module, f"cuda_compiler_{field}", None)
        for field in ("major", "minor", "build")
    }


def module_provenance(module: object) -> dict[str, object]:
    module_file = Path(getattr(module, "__file__")).resolve()
    return {
        "path": str(module_file),
        "sha256": file_sha256(module_file),
        "cuda_compiler": cuda_compiler_identity(module),
        "cuda_compiler_version": getattr(module, "cuda_compiler_version", None),
        "cuda_runtime_header_version": getattr(
            module, "cuda_runtime_header_version", None
        ),
    }


def require_matching_toolchain(
    expected_label: str,
    expected: dict[str, object],
    observed_label: str,
    observed: dict[str, object],
) -> None:
    for field in TOOLCHAIN_FIELDS:
        if observed[field] != expected[field]:
            raise RuntimeError(
                f"{observed_label} uses different {field} from "
                f"{expected_label}: {observed[field]} != {expected[field]}"
            )


def require_origin(module: object, expected_dir: Path) -> None:
    module_file = Path(getattr(module, "__file__")).resolve()
    if module_file.parent != expected_dir:
        raise RuntimeError(
            f"{module_file.name} resolved from {module_file.parent}, "
            f"expected {expected_dir}"
        )


def load_extension(module_name: str, module_path: Path) -> object:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mk-dir", type=Path, required=True)
    persistent_source = parser.add_mutually_exclusive_group()
    persistent_source.add_argument(
        "--require-persistent", action="store_true"
    )
    persistent_source.add_argument("--persistent-module", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("binding_contract.json"),
    )
    args = parser.parse_args()

    mk_dir = args.mk_dir.expanduser().resolve()
    sys.path.insert(0, str(mk_dir))
    tk_module = importlib.import_module("mk_llama")
    simt_module = importlib.import_module("mk_mlp_simt")
    require_origin(tk_module, mk_dir)
    require_origin(simt_module, mk_dir)

    required_tk = (
        "opcode1_direct_pdl_13page_issue_r96",
        "opcode7_direct_pdl_wait",
        "direct_pdl_13page_issue_r96_metadata",
        "direct_pdl_13page_arrival_r96_metadata",
    )
    required_simt = (
        "inspect_cuda_graph_edges",
        "rewrite_cuda_graph_kernel_edges",
        "instantiate_cuda_graph_exec",
    )
    missing = [name for name in required_tk if not hasattr(tk_module, name)]
    missing += [name for name in required_simt if not hasattr(simt_module, name)]
    if args.require_persistent and not hasattr(tk_module, "mk_llama"):
        missing.append("mk_llama (persistent VM entry)")
    if missing:
        raise RuntimeError("missing compiled bindings: " + ", ".join(missing))

    for label, module in (("mk_llama", tk_module), ("mk_mlp_simt", simt_module)):
        provenance = module_provenance(module)
        if any(
            value is None for value in provenance["cuda_compiler"].values()
        ):
            raise RuntimeError(
                f"{label} does not expose unambiguous CUDA compiler provenance"
            )
        if provenance["cuda_runtime_header_version"] is None:
            raise RuntimeError(f"{label} does not expose CUDA header provenance")
    tk_provenance = module_provenance(tk_module)
    simt_provenance = module_provenance(simt_module)
    require_matching_toolchain(
        "candidate mk_llama",
        tk_provenance,
        "graph helper mk_mlp_simt",
        simt_provenance,
    )

    issue = dict(tk_module.direct_pdl_13page_issue_r96_metadata)
    arrival = dict(tk_module.direct_pdl_13page_arrival_r96_metadata)
    contract_path = args.contract.expanduser().resolve()
    contract = json.loads(contract_path.read_text())
    if contract.get("schema") != "hazy-h100-binding-contract-v1":
        raise RuntimeError(f"unrecognized binding contract: {contract_path}")
    expected_contract_keys = {
        "schema",
        "target",
        "direct_tk_common",
        "issue_trigger_wait_for_load_arrival",
        "arrival_trigger_wait_for_load_arrival",
    }
    if set(contract) != expected_contract_keys:
        raise RuntimeError(
            "binding contract has unexpected or missing keys: "
            f"{sorted(set(contract) ^ expected_contract_keys)}"
        )
    expected_common = contract["direct_tk_common"]
    expected_common_keys = {
        "alias_four_pages",
        "dynamic_shared_memory_bytes",
        "matvec_input_pipeline_stages",
        "num_pages",
        "num_threads",
    }
    if not isinstance(expected_common, dict) or set(expected_common) != (
        expected_common_keys
    ):
        raise RuntimeError("binding contract has an invalid direct_tk_common")
    for label, observed in (("issue", issue), ("arrival", arrival)):
        common = {
            key: observed.get(key)
            for key in expected_common
        }
        if common != expected_common:
            raise RuntimeError(
                f"{label} 13-page resource identity mismatch: {common}"
            )
    if issue.get("trigger_wait_for_load_arrival") is not contract[
        "issue_trigger_wait_for_load_arrival"
    ]:
        raise RuntimeError("issue entry does not report issue-trigger semantics")
    if arrival.get("trigger_wait_for_load_arrival") is not contract[
        "arrival_trigger_wait_for_load_arrival"
    ]:
        raise RuntimeError(
            "arrival entry does not report arrival-trigger semantics"
        )

    persistent_module = tk_module
    if args.persistent_module is not None:
        persistent_module = load_extension(
            "native_reference.mk_llama",
            args.persistent_module.expanduser().resolve(),
        )
    if (args.require_persistent or args.persistent_module is not None) and not hasattr(
        persistent_module, "mk_llama"
    ):
        raise RuntimeError("persistent reference module has no mk_llama entry")
    if args.persistent_module is not None:
        persistent_provenance = module_provenance(persistent_module)
        if any(
            value is None
            for value in persistent_provenance["cuda_compiler"].values()
        ):
            raise RuntimeError(
                "persistent reference does not expose unambiguous CUDA "
                "compiler provenance"
            )
        if persistent_provenance["cuda_runtime_header_version"] is None:
            raise RuntimeError(
                "persistent reference does not expose CUDA header provenance"
            )
    else:
        persistent_provenance = module_provenance(persistent_module)
    require_matching_toolchain(
        "candidate mk_llama",
        tk_provenance,
        "persistent reference mk_llama",
        persistent_provenance,
    )

    print(
        json.dumps(
            {
                "arrival_metadata": arrival,
                "binding_contract": {
                    "path": str(contract_path),
                    "sha256": file_sha256(contract_path),
                    "schema": contract["schema"],
                },
                "candidate_module": tk_provenance,
                "graph_helper_module": simt_provenance,
                "issue_metadata": issue,
                "persistent_module": persistent_provenance,
                "persistent_vm": hasattr(persistent_module, "mk_llama"),
                "schema": PREFLIGHT_SCHEMA,
                "status": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
