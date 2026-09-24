#!/usr/bin/env python3
"""Measure direct-TK completion and PDL body implementations.

P32/D128 has one attention partition, so opcode 3 is dormant and the
active body is opcode 1 -> opcode 2 -> opcode 4 -> opcode 5 -> opcode 6.
The accepted default is the correct 13-page implementation; four-page
entries remain explicitly labelled, intentionally incorrect controls.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import statistics
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mk-dir", type=Path, required=True)
    parser.add_argument("--baseline-helper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--implementation",
        choices=(
            "direct_baseline",
            "pdl_r96",
            "pdl_r48",
            "pdl_issue_r96",
            "pdl_issue_r48",
            "pdl_issue_4cw",
            "pdl_13page_arrival_r96",
            "pdl_13page_issue_r96",
            "pdl_13page_issue_r96_vtp56",
            "pdl_13page_issue_r96_compact",
            "pdl_13page_issue_r96_compact_op2",
            "pdl_13page_issue_r96_compact_op4",
            "pdl_13page_issue_r96_compact_op5",
            "pdl_s1_issue_r96",
            "pdl_s1_issue_r48",
            "pdl_s1_issue_4cw",
        ),
        required=True,
    )
    parser.add_argument(
        "--group", choices=("12", "24", "124", "12456"), required=True
    )
    parser.add_argument("--layer-count", type=int, default=16)
    parser.add_argument(
        "--cross-layer-edge",
        choices=("completion", "pdl"),
        default="completion",
        help=(
            "dependency from the final opcode of one requested layer group "
            "to the first opcode of the next; pdl is currently supported "
            "only for group 12456 with the PDL implementation"
        ),
    )
    parser.add_argument(
        "--edge-mask",
        type=lambda value: int(value, 0),
        default=0x1F,
        help=(
            "periodic PDL edge mask for opcode sequence 1,2,4,5,6: "
            "bit0=1->2, bit1=2->4, bit2=4->5, bit3=5->6, "
            "bit4=6->next-layer-1"
        ),
    )
    parser.add_argument("--position-start", type=int, default=32)
    parser.add_argument("--position-end", type=int, default=158)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--correctness-rtol", type=float, default=0.10)
    parser.add_argument("--correctness-atol", type=float, default=0.10)
    parser.add_argument(
        "--global-trace-output",
        type=Path,
        help=(
            "run one instrumented graph replay and write absolute "
            "%%globaltimer/%%smid records instead of performance timing"
        ),
    )
    parser.add_argument(
        "--trace-include-lm-head",
        action="store_true",
        help="append the final opcode6-to-opcode7 PDL edge in trace mode",
    )
    return parser.parse_args()


ARGS = parse_args()
sys.path.insert(0, str(ARGS.repo.resolve()))
sys.path.insert(0, str(ARGS.baseline_helper_dir.resolve()))

import torch  # noqa: E402

import bench_hazy as baseline  # noqa: E402
from megakernels.demos.latency.instructions import (  # noqa: E402
    LayerNormDoubleMatVecSiLU,
    O_ProjResidual,
    PartialAttention,
)
from megakernels.demos.latency.mk import interpret_with_mk  # noqa: E402
from megakernels.demos.latency.scheduler import (  # noqa: E402
    schedule_downproj,
    schedule_lm_head,
    schedule_qkv,
)
from megakernels.dispatch import make_schedule_builder  # noqa: E402
from megakernels.scheduler import assign_to_sms, tensorize_instructions  # noqa: E402


REQUESTED_GROUPS = {
    "12": (1, 2),
    "24": (2, 4),
    "124": (1, 2, 4),
    "12456": (1, 2, 4, 5, 6),
}

DIRECT_TRACE_MAGIC = 0x44545243
DIRECT_TRACE_EVENT_BASE = 8
DIRECT_TRACE_EVENT_NAMES = (
    "cta_entry",
    "roles_start",
    "consumer_start",
    "activation_wait_begin",
    "activation_wait_end",
    "activation_load_done",
    "consumer_end",
    "loader_start",
    "weight_first_issue",
    "weight_last_issue",
    "trigger_begin",
    "trigger_end",
    "loader_end",
    "launcher_start",
    "kv_first_issue",
    "kv_last_issue",
    "kv_wait_begin",
    "kv_wait_end",
    "launcher_end",
    "weight_first_ready",
    "weight_last_ready",
    "first_store",
    "last_store",
    "storer_start",
    "storer_end",
    "cta_end",
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_us": samples,
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "p10_us": percentile(samples, 0.10),
        "p90_us": percentile(samples, 0.90),
        "min_us": min(samples),
        "max_us": max(samples),
        "stdev_us": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_f = actual.float()
    expected_f = expected.float()
    absolute = (actual_f - expected_f).abs()
    denominator = actual_f.abs() + expected_f.abs() + 1e-6
    return {
        "correct": bool(
            torch.allclose(
                actual_f,
                expected_f,
                rtol=ARGS.correctness_rtol,
                atol=ARGS.correctness_atol,
            )
        ),
        "max_abs_diff": float(absolute.max().item()),
        "mean_abs_diff": float(absolute.mean().item()),
        "mean_relative_diff": float((2 * absolute / denominator).mean().item()),
    }


def make_node_globs(base_globs, instructions, *, slots: int | None = 1):
    node_globs = copy.copy(base_globs)
    queues = assign_to_sms(
        "rr", instructions=instructions, sm_count=base_globs.sm_count()
    )
    tensorize_instructions(node_globs, queues)
    actual = tuple(node_globs.instructions.shape)
    expected = (base_globs.sm_count(), slots, 32)
    if (
        actual[0] != base_globs.sm_count()
        or actual[2] != 32
        or (slots is not None and actual != expected)
    ):
        raise RuntimeError(
            f"unexpected instruction shape {node_globs.instructions.shape}; "
            f"expected {expected if slots is not None else '(sm_count, *, 32)'}"
        )
    return node_globs


def schedule_partial(globs, layer: int):
    return [
        PartialAttention(
            layer_idx=layer,
            kv_head_idx=kv_head,
            num_partials=1,
            partial_idx=0,
        )
        for kv_head in range(globs.num_kv_heads)
    ]


def schedule_oproj(globs, layer: int):
    return [
        O_ProjResidual(
            layer_idx=layer,
            start_block_idx=block,
            end_block_idx=block + 1,
            reduction_block_idx=0,
        )
        for block in range(globs.hidden_size // globs.o_proj_block_size)
    ]


def schedule_accepted_upgate(globs, layer: int):
    active_sms = 128
    output_blocks = globs.intermediate_size // globs.up_gate_proj_block_size
    return [
        LayerNormDoubleMatVecSiLU(
            layer_idx=layer,
            block_idxs=(
                list(range(sm, output_blocks, active_sms))
                if sm < active_sms
                else []
            ),
        )
        for sm in range(globs.sm_count())
    ]


def run_nodes(nodes) -> None:
    for node_globs, kernel in nodes:
        interpret_with_mk(node_globs, kernel)


def capture_prepared_raw_graph(fn, prepare, module) -> int:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            prepare()
            fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    stream_ptr = capture_stream.cuda_stream
    with torch.cuda.stream(capture_stream):
        prepare()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    with torch.cuda.stream(capture_stream):
        module.begin_cuda_graph_capture(stream_ptr)
        fn()
        raw_graph = module.end_cuda_graph_capture(stream_ptr)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return raw_graph


def set_position(globs, nodes, position: int) -> None:
    globs.pos_id = position
    for node_globs, _ in nodes:
        node_globs.pos_id = position


@torch.inference_mode()
def main() -> None:
    torch.cuda.set_device(ARGS.device)
    positions = list(range(ARGS.position_start, ARGS.position_end + 1))
    if not positions:
        raise ValueError("empty position range")

    model = baseline.make_model(ARGS.device, ARGS.position_end + 1, ARGS.seed)
    schedule = make_schedule_builder("latency").build(model)
    globs = schedule.globs
    if not globs.skip_attn_reduction:
        raise RuntimeError("P32/D128 group requires skip_attn_reduction=true")
    if globs.num_hidden_layers != 16:
        raise RuntimeError(f"expected 16 model layers, got {globs.num_hidden_layers}")
    if not (1 <= ARGS.layer_count <= globs.num_hidden_layers):
        raise ValueError(
            f"layer-count must be in [1, {globs.num_hidden_layers}], "
            f"got {ARGS.layer_count}"
        )
    if ARGS.group != "12456":
        raise ValueError("this ablation requires --group 12456")
    if (
        ARGS.cross_layer_edge == "pdl"
        and not ARGS.implementation.startswith("pdl_")
    ):
        raise ValueError(
            "--cross-layer-edge pdl requires a pdl_* implementation; "
            "direct_baseline uses ordinary completion edges"
        )
    if ARGS.trace_include_lm_head and ARGS.global_trace_output is None:
        raise ValueError(
            "--trace-include-lm-head requires --global-trace-output"
        )

    sys.path.insert(0, str(ARGS.mk_dir.resolve()))
    tk_module = importlib.import_module("mk_llama")
    simt_module = importlib.import_module("mk_mlp_simt")
    requested = REQUESTED_GROUPS[ARGS.group]
    group_width = len(requested)
    if ARGS.edge_mask < 0 or ARGS.edge_mask >= (1 << group_width):
        raise ValueError(
            f"edge-mask must be in [0, {(1 << group_width) - 1:#x}], "
            f"got {ARGS.edge_mask:#x}"
        )

    reference_kernels = {
        1: tk_module.opcode1_direct,
        2: tk_module.opcode2_direct,
        4: tk_module.opcode4_direct,
        5: tk_module.opcode5_direct,
        6: tk_module.opcode6_direct,
        7: tk_module.opcode7_direct,
    }
    vtp56 = ARGS.implementation == "pdl_13page_issue_r96_vtp56"
    suffix = (
        "13page_issue_r96"
        if vtp56
        else ARGS.implementation.removeprefix("pdl_")
    )
    if ARGS.global_trace_output is not None:
        if ARGS.implementation != "pdl_13page_issue_r96":
            raise ValueError(
                "--global-trace-output requires "
                "--implementation pdl_13page_issue_r96"
            )
        if len(positions) != 1:
            raise ValueError(
                "--global-trace-output requires one position"
            )
        suffix = "13page_issue_r96_trace"
    resource_identity = None
    if ARGS.implementation.startswith("pdl_13page_"):
        metadata_name = (
            "direct_pdl_13page_arrival_r96_metadata"
            if ARGS.implementation == "pdl_13page_arrival_r96"
            else "direct_pdl_13page_issue_r96_metadata"
        )
        if not hasattr(tk_module, metadata_name):
            raise RuntimeError(
                f"selected 13-page implementation lacks {metadata_name}"
            )
        resource_identity = dict(getattr(tk_module, metadata_name))
        expected_resource_identity = {
            "num_pages": 13,
            "matvec_input_pipeline_stages": 3,
            "alias_four_pages": False,
            "trigger_wait_for_load_arrival": (
                ARGS.implementation == "pdl_13page_arrival_r96"
            ),
        }
        observed = {
            key: resource_identity.get(key)
            for key in expected_resource_identity
        }
        if observed != expected_resource_identity:
            raise RuntimeError(
                "13-page resource identity mismatch: "
                f"expected {expected_resource_identity}, got {observed}"
            )
    compact_grid_opcodes = {
        "pdl_13page_issue_r96_compact": {2, 4, 5},
        "pdl_13page_issue_r96_compact_op2": {2},
        "pdl_13page_issue_r96_compact_op4": {4},
        "pdl_13page_issue_r96_compact_op5": {5},
    }.get(ARGS.implementation, set())
    pdl_kernels = (
        reference_kernels
        if ARGS.implementation == "direct_baseline"
        else {
            opcode: getattr(
                tk_module,
                (
                    f"opcode{opcode}_direct_pdl_"
                    "13page_issue_r96_compact"
                    if opcode in compact_grid_opcodes
                    else f"opcode{opcode}_direct_pdl_"
                    + (
                        "13page_issue_r96"
                        if compact_grid_opcodes
                        else (
                            suffix
                            if suffix.startswith(("13page_", "s1_"))
                            else "4page_" + suffix
                        )
                    )
                ),
            )
            for opcode in requested
        }
    )
    if vtp56:
        pdl_kernels[5] = getattr(
            tk_module,
            "opcode5_direct_pdl_4page_13page_issue_r96_vtp_signal",
        )
        pdl_kernels[6] = getattr(
            tk_module,
            "opcode6_direct_pdl_4page_13page_issue_r96_vtp_wait",
        )
    if ARGS.trace_include_lm_head:
        pdl_kernels[7] = tk_module.opcode7_direct_pdl_wait_trace

    split_schedules = {
        1: [schedule_qkv(globs, layer) for layer in range(ARGS.layer_count)],
        2: [schedule_partial(globs, layer) for layer in range(ARGS.layer_count)],
        4: [schedule_oproj(globs, layer) for layer in range(ARGS.layer_count)],
        5: [
            schedule_accepted_upgate(globs, layer)
            for layer in range(ARGS.layer_count)
        ],
        6: [schedule_downproj(globs, layer) for layer in range(ARGS.layer_count)],
    }
    split_globs = {
        opcode: [
            make_node_globs(globs, split_schedules[opcode][layer])
            for layer in range(ARGS.layer_count)
        ]
        for opcode in requested
    }

    reference_nodes = []
    pdl_nodes = []
    for layer in range(ARGS.layer_count):
        for opcode in requested:
            reference_nodes.append(
                (split_globs[opcode][layer], reference_kernels[opcode])
            )
            pdl_nodes.append((split_globs[opcode][layer], pdl_kernels[opcode]))
    if ARGS.trace_include_lm_head:
        lm_globs = make_node_globs(globs, schedule_lm_head(globs))
        reference_nodes.append((lm_globs, reference_kernels[7]))
        pdl_nodes.append((lm_globs, pdl_kernels[7]))
    candidate_nodes = pdl_nodes

    ready_barriers = torch.full_like(globs.barriers, 1_000_000)
    if vtp56:
        # Opcode 5 publishes 128 completed 16-element SiLU stores into each
        # of four 2048-element virtual-TP shard counters.  Only these four
        # incoming opcode-6 dependencies must start at zero.
        ready_barriers[:, 4, :4] = 0
    torch.manual_seed(ARGS.seed + 91)
    input_hidden = torch.randn_like(globs.hidden_states)
    input_attn = torch.randn_like(globs.attn_out)
    input_q = torch.randn_like(globs.post_ln_rope_q)
    input_k = torch.randn_like(globs.k_cache)
    input_v = torch.randn_like(globs.v_cache)

    def prepare(*, full: bool = False) -> None:
        globs.barriers.copy_(ready_barriers)
        globs.hidden_states.copy_(input_hidden)
        globs.attn_out.copy_(input_attn)
        globs.silu_out.zero_()
        if full:
            globs.post_ln_rope_q.copy_(input_q)
            globs.k_cache.copy_(input_k)
            globs.v_cache.copy_(input_v)

    graph_execs: dict[int, int] = {}
    edge_diagnostics = None
    for position in positions:
        set_position(globs, candidate_nodes, position)
        raw_graph = capture_prepared_raw_graph(
            lambda: run_nodes(candidate_nodes),
            lambda: prepare(full=False),
            simt_module,
        )
        if (
            ARGS.implementation.startswith("pdl_")
            and ARGS.cross_layer_edge == "pdl"
        ):
            before = dict(simt_module.inspect_cuda_graph_edges(raw_graph))
            edge_mask = ARGS.edge_mask
            rewritten = int(
                simt_module.rewrite_cuda_graph_kernel_edges(
                    raw_graph, False, edge_mask, group_width
                )
            )
            edge_count = len(candidate_nodes) - 1
            complete_periods, remainder = divmod(edge_count, group_width)
            expected_rewritten = (
                complete_periods * (edge_mask & ((1 << group_width) - 1)).bit_count()
                + (edge_mask & ((1 << remainder) - 1)).bit_count()
            )
            if rewritten != expected_rewritten:
                raise RuntimeError(
                    f"expected {expected_rewritten} rewritten edges, got {rewritten}"
                )
            if edge_diagnostics is None:
                edge_diagnostics = {
                    "before": before,
                    "after": dict(simt_module.inspect_cuda_graph_edges(raw_graph)),
                    "edge_period": group_width,
                    "edge_mask": edge_mask,
                    "rewritten": rewritten,
                    "selected": "periodic edge-mask selection",
                    "cross_layer_edge": ARGS.cross_layer_edge,
                }
        elif edge_diagnostics is None:
            edge_diagnostics = {
                "captured": dict(simt_module.inspect_cuda_graph_edges(raw_graph)),
                "rewritten": 0,
                "selected": "ordinary captured launch-completion edges",
                "cross_layer_edge": ARGS.cross_layer_edge,
            }
        graph_execs[position] = int(
            simt_module.instantiate_cuda_graph_exec(raw_graph, 0)
        )
        simt_module.destroy_cuda_graph(raw_graph)

    def replay(position: int) -> None:
        simt_module.launch_cuda_graph_exec(
            graph_execs[position], torch.cuda.current_stream().cuda_stream
        )

    def snapshot(position: int) -> dict[str, torch.Tensor]:
        values = {
            "hidden": globs.hidden_states.clone(),
            "attn": globs.attn_out.clone(),
            "q": globs.post_ln_rope_q.clone(),
            "k": globs.k_cache[:, :, position].clone(),
            "v": globs.v_cache[:, :, position].clone(),
        }
        if ARGS.trace_include_lm_head:
            values["logits"] = globs.logits.clone()
        return values

    def compare(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]):
        errors = {
            name: tensor_error(actual[name], expected[name])
            for name in actual
        }
        errors["correct"] = all(value["correct"] for value in errors.values())
        return errors

    correctness: dict[str, object] = {}
    for position in sorted({positions[0], positions[len(positions) // 2], positions[-1]}):
        set_position(globs, reference_nodes, position)
        prepare(full=True)
        run_nodes(reference_nodes)
        torch.cuda.synchronize()
        expected = snapshot(position)

        set_position(globs, candidate_nodes, position)
        prepare(full=True)
        run_nodes(candidate_nodes)
        torch.cuda.synchronize()
        eager = snapshot(position)

        set_position(globs, candidate_nodes, position)
        prepare(full=True)
        replay(position)
        torch.cuda.synchronize()
        graph = snapshot(position)
        graph_vs_eager = compare(graph, eager)
        correctness[str(position)] = {
            "graph_vs_eager": graph_vs_eager,
            "eager_vs_extracted_tk": compare(eager, expected),
            "graph_vs_extracted_tk": compare(graph, expected),
        }

    if ARGS.global_trace_output is not None:
        trace_position = positions[0]
        for node_globs, _ in candidate_nodes:
            node_globs.timings.zero_()
        set_position(globs, candidate_nodes, trace_position)
        prepare(full=True)
        replay(trace_position)
        torch.cuda.synchronize()

        records: list[dict[str, object]] = []
        node_metadata: list[dict[str, object]] = []
        for node_index, (node_globs, kernel) in enumerate(candidate_nodes):
            if node_index < ARGS.layer_count * group_width:
                layer = node_index // group_width
                opcode = requested[node_index % group_width]
                node_name = f"layer{layer:02d}.opcode{opcode}"
            else:
                layer = ARGS.layer_count
                opcode = 7
                node_name = "final.opcode7"
            host_rows = node_globs.timings[:, 0, :].cpu()
            valid_rows = 0
            for row_index, row in enumerate(host_rows):
                metadata = [int(value.item()) for value in row[:8]]
                if metadata[0] != DIRECT_TRACE_MAGIC:
                    continue
                valid_rows += 1
                for event_id, event_name in enumerate(
                    DIRECT_TRACE_EVENT_NAMES
                ):
                    slot = DIRECT_TRACE_EVENT_BASE + 2 * event_id
                    low = int(row[slot].item()) & 0xFFFFFFFF
                    high = int(row[slot + 1].item()) & 0xFFFFFFFF
                    timestamp_ns = low | (high << 32)
                    if timestamp_ns == 0:
                        continue
                    records.append(
                        {
                            "node_index": node_index,
                            "node_name": node_name,
                            "layer": layer,
                            "opcode": opcode,
                            "row_index": row_index,
                            "smid": metadata[1],
                            "block_idx": metadata[2],
                            "kernel_opcode": metadata[3],
                            "worker_id": metadata[4],
                            "position": metadata[5],
                            "grid_dim": metadata[6],
                            "block_dim": metadata[7],
                            "event_id": event_id,
                            "event": event_name,
                            "timestamp_ns": timestamp_ns,
                        }
                    )
            node_metadata.append(
                {
                    "node_index": node_index,
                    "node_name": node_name,
                    "layer": layer,
                    "opcode": opcode,
                    "kernel_entry": getattr(kernel, "__name__", repr(kernel)),
                    "timings_shape": list(node_globs.timings.shape),
                    "valid_rows": valid_rows,
                }
            )

        trace_payload = {
            "schema": "hazy-direct-tk-pdl-globaltimer-v1",
            "gpu": torch.cuda.get_device_name(ARGS.device),
            "position": trace_position,
            "layer_count": ARGS.layer_count,
            "scheduled_opcodes": (
                list(requested) + ([7] if ARGS.trace_include_lm_head else [])
            ),
            "kernel_node_count": len(candidate_nodes),
            "edge_diagnostics": edge_diagnostics,
            "timer": {
                "ptx_register": "%globaltimer",
                "stored_unit": "nanoseconds",
                "sm_register": "%smid",
                "clock64_used": False,
            },
            "instrumentation": {
                "performance_timing_valid": False,
                "reason": (
                    "global stores and timer reads intentionally perturb the "
                    "instrumented replay"
                ),
                "event_names": list(DIRECT_TRACE_EVENT_NAMES),
            },
            "correctness": correctness,
            "nodes": node_metadata,
            "records": records,
        }
        ARGS.global_trace_output.parent.mkdir(parents=True, exist_ok=True)
        ARGS.global_trace_output.write_text(
            json.dumps(trace_payload, indent=2, sort_keys=True) + "\n"
        )
        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        ARGS.output.write_text(
            json.dumps(
                {
                    "schema": "hazy-direct-tk-pdl-globaltimer-run-v1",
                    "global_trace_output": str(ARGS.global_trace_output),
                    "record_count": len(records),
                    "node_count": len(node_metadata),
                    "edge_diagnostics": edge_diagnostics,
                    "performance_timing_valid": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "global_trace_output": str(ARGS.global_trace_output),
                    "record_count": len(records),
                    "valid_rows": sum(
                        node["valid_rows"] for node in node_metadata
                    ),
                    "rewritten_edges": edge_diagnostics["rewritten"],
                },
                sort_keys=True,
            )
        )
        for executable in graph_execs.values():
            simt_module.destroy_cuda_graph_exec(executable)
        return

    per_position: dict[str, object] = {}
    all_samples: list[float] = []
    for position in positions:
        for _ in range(ARGS.warmup):
            prepare(full=False)
            replay(position)
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(ARGS.iterations):
            prepare(full=False)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            replay(position)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) * 1000.0)
        per_position[str(position)] = summarize(samples)
        all_samples.extend(samples)

    position_medians = [
        float(per_position[str(position)]["median_us"])
        for position in positions
    ]
    effective_edge_mask = (
        ARGS.edge_mask if ARGS.cross_layer_edge == "pdl" else None
    )
    edge_label = (
        f"pdl_edges_mask{effective_edge_mask:#04x}"
        if effective_edge_mask is not None
        else "completion_edges"
    )
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidate": (
            f"p32d128_opcode{ARGS.group}_group_{ARGS.implementation}_"
            f"{edge_label}"
        ),
        "implementation": ARGS.implementation,
        "requested_group": ARGS.group,
        "scheduled_opcodes": list(requested),
        "opcode3_scheduled": False,
        "opcode3_reason": (
            "positions 32-158 use one attention partition and "
            "skip_attn_reduction=true"
        ),
        "gpu": torch.cuda.get_device_name(ARGS.device),
        "positions_inclusive": [positions[0], positions[-1]],
        "position_count": len(positions),
        "layer_count": ARGS.layer_count,
        "cross_layer_edge": ARGS.cross_layer_edge,
        "edge_mask": effective_edge_mask,
        "kernel_node_count": len(candidate_nodes),
        "compact_grid_opcodes": sorted(compact_grid_opcodes),
        "launch_ctas_per_opcode": {
            str(opcode): (
                {2: 8, 4: 128, 5: 128}[opcode]
                if opcode in compact_grid_opcodes
                else 132
            )
            for opcode in requested
        },
        "instruction_queue_depths": [
            int(node_globs.instructions.shape[1])
            for node_globs, _ in candidate_nodes
        ],
        "timing_contract": {
            "cuda_graph_replay": True,
            "input_and_barrier_reset_in_timed_region": False,
            "timeline_collection_enabled": False,
            "timing_record_enabled": False,
            "warmup_per_position": ARGS.warmup,
            "iterations_per_position": ARGS.iterations,
            "correctness_expected": (
                ARGS.implementation == "direct_baseline"
                or ARGS.implementation.startswith("pdl_s1_")
                or ARGS.implementation.startswith("pdl_13page_")
            ),
            "four_page_alias_is_intentionally_wrong":
                ARGS.implementation in {
                    "pdl_r96",
                    "pdl_r48",
                    "pdl_issue_r96",
                    "pdl_issue_r48",
                    "pdl_issue_4cw",
                },
        },
        "entries": {
            "op1": getattr(pdl_kernels[1], "__name__", "opcode1_pdl") if 1 in requested else None,
            "op2": getattr(pdl_kernels[2], "__name__", "opcode2_pdl") if 2 in requested else None,
            "op4": getattr(pdl_kernels[4], "__name__", "opcode4_pdl") if 4 in requested else None,
            "op5": getattr(pdl_kernels[5], "__name__", "opcode5_pdl") if 5 in requested else None,
            "op6": getattr(pdl_kernels[6], "__name__", "opcode6_pdl") if 6 in requested else None,
        },
        "selected_kernel_resource_metadata": resource_identity,
        "correctness": correctness,
        "edge_diagnostics": edge_diagnostics,
        "all_position_samples": summarize(all_samples),
        "position_median_summary": summarize(position_medians),
        "selected_position_us": {
            str(position): per_position[str(position)]
            for position in sorted({positions[0], positions[len(positions) // 2], positions[-1]})
        },
        "per_position": per_position,
    }
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "candidate": result["candidate"],
        "deterministic_graph_vs_eager": all(
            value["graph_vs_eager"]["correct"]
            for value in correctness.values()
        ),
        "kernel_node_count": result["kernel_node_count"],
        "position_median_mean_us": result["position_median_summary"]["mean_us"],
        "selected_position_us": {
            key: value["median_us"]
            for key, value in result["selected_position_us"].items()
        },
        "rewritten_edges": edge_diagnostics["rewritten"],
    }, sort_keys=True))

    for executable in graph_execs.values():
        simt_module.destroy_cuda_graph_exec(executable)


if __name__ == "__main__":
    main()
