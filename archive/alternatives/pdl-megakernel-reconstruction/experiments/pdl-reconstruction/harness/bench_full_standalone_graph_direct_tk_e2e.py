#!/usr/bin/env python3
"""Compose all context-1 standalone Hazy opcodes into one CUDA Graph."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path


TENSOR_NAMES = ("hidden", "k", "logits", "v")
ERROR_METRICS = ("max_abs_diff", "mean_abs_diff")
RELEASE_REQUIRED_CHECKS = (
    "eager_vs_graph_tensor_envelope",
    "graph_matches_eager_token",
    "graph_edge_rewrite",
)
RELEASE_DIAGNOSTICS = (
    "elementwise_allclose_rtol_atol",
    "eager_graph_bit_exact",
    "eager_vs_persistent_tensor_envelope",
    "graph_vs_persistent_tensor_envelope",
    "graph_matches_persistent_replay_token",
    "persistent_initial_vs_replay_token",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mk-dir", type=Path, required=True)
    parser.add_argument("--baseline-helper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--position",
        type=int,
        default=0,
        help="zero-based decode input position captured by this graph",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="optional real-weight Hugging Face model directory",
    )
    parser.add_argument(
        "--model-safetensors-sha256",
        help="digest verified by the invoking runner before model loading",
    )
    parser.add_argument(
        "--native-reference-mk",
        type=Path,
        help=(
            "optional mk_llama extension containing the original persistent "
            "entry; used when the candidate extension intentionally omits it"
        ),
    )
    parser.add_argument(
        "--trajectory-prompt-length",
        type=int,
        help=(
            "run a full autoregressive trajectory instead of the position-local "
            "microbenchmark; the formal P32/D128 setting uses 32"
        ),
    )
    parser.add_argument(
        "--trajectory-output-length",
        type=int,
        default=128,
        help="total output tokens including the first token produced by prefill",
    )
    parser.add_argument(
        "--fixed-correctness-artifact",
        type=Path,
        help=(
            "successful fixed-position result required before trajectory "
            "timing; its SHA-256 is embedded in the trajectory report"
        ),
    )
    parser.add_argument("--prompt-token-base", type=int, default=1000)
    parser.add_argument(
        "--fixed-input-token",
        type=int,
        default=17,
        help=(
            "embedding token used by the fixed-position check; generic runs "
            "default to historical token 17, while run_e2e passes the "
            "versioned release-contract token"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--correctness-rtol", type=float, default=0.10)
    parser.add_argument("--correctness-atol", type=float, default=0.10)
    parser.add_argument(
        "--correctness-contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "correctness_contract.json",
        help="checked-in numerical envelope for the fixed-position release gate",
    )
    parser.add_argument(
        "--binding-contract",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "binding_contract.json",
        help="checked-in resource-identity contract recorded with results",
    )
    parser.add_argument(
        "--binding-preflight-artifact",
        type=Path,
        help=(
            "successful validate_bindings.py JSON sidecar; release runs bind "
            "its module and contract provenance into both result artifacts"
        ),
    )
    parser.add_argument(
        "--enforce-correctness-contract",
        action="store_true",
        help=(
            "enforce the checked-in release contract; generic harness runs "
            "otherwise use their position-local checks"
        ),
    )
    parser.add_argument(
        "--allow-correctness-mismatch",
        action="store_true",
        help="record an incorrect diagnostic candidate instead of aborting",
    )
    parser.add_argument(
        "--graph-edge-mode",
        choices=("capture", "programmatic", "launch-completion"),
        default="capture",
        help=(
            "retain captured graph edges or explicitly rewrite kernel edges "
            "to a CUDA programmatic/launch-completion port"
        ),
    )
    parser.add_argument(
        "--graph-exec-mode",
        choices=("torch", "custom", "device"),
        default="torch",
        help=(
            "instantiate through torch, directly through the CUDA runtime, "
            "or with cudaGraphInstantiateFlagDeviceLaunch"
        ),
    )
    parser.add_argument(
        "--graph-node-priorities",
        default="",
        help=(
            "optional comma-separated kernel-chain priority pattern; enables "
            "cudaGraphInstantiateFlagUseNodePriority"
        ),
    )
    parser.add_argument(
        "--launch-completion-edge-mask",
        type=lambda value: int(value, 0),
        default=0x1F,
        help=(
            "five-bit mask for qkv->attn, attn->oproj, oproj->upgate, "
            "upgate->down, and down->next launch-completion edges"
        ),
    )
    parser.add_argument(
        "--launch-completion-prune-triggers",
        action="store_true",
        help="remove producer trigger instructions on launch-completion edges",
    )
    parser.add_argument(
        "--launch-completion-plain-launch",
        action="store_true",
        help=(
            "capture pruned launch-completion producers without the "
            "programmatic stream-serialization launch attribute"
        ),
    )
    parser.add_argument(
        "--launch-completion-op4-prefetch-bytes",
        type=int,
        choices=(0, 128, 256, 512, 1024, 2048, 4096),
        default=128,
    )
    parser.add_argument(
        "--launch-completion-op5-prefetch-bytes",
        type=int,
        choices=(0, 128, 256, 512, 1024),
        default=128,
    )
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument(
        "--implementation",
        choices=("extracted_tk", "pure_simt"),
        default="extracted_tk",
    )
    parser.add_argument(
        "--direct-tk-pdl-body",
        choices=(
            "13page_issue_r96",
            "13page_issue_r96_compact",
            "13page_issue_r96_vtp56",
        ),
        help=(
            "replace body opcodes 1,2,4,5,6 with the arithmetic-preserving "
            "direct-TK PDL entries and use a dependency-waiting direct-TK "
            "LM head; vtp56 retains PDL admission but replaces the opcode5 "
            "to opcode6 whole-grid wait with four fine-grained shard signals"
        ),
    )
    parser.add_argument(
        "--direct-tk-opcode4-entry",
        help=(
            "optional mk_llama opcode-4 entry override for a direct-TK PDL "
            "body; all other body and tail entries remain unchanged"
        ),
    )
    parser.add_argument(
        "--direct-tk-entry-suffix",
        help=(
            "optional common mk_llama direct-PDL entry suffix for body "
            "opcodes 1,2,4,5,6; for example s1_micro16_all"
        ),
    )
    parser.add_argument(
        "--direct-tk-opcode7-entry",
        help=(
            "optional mk_llama opcode-7 entry paired with "
            "--direct-tk-entry-suffix"
        ),
    )
    parser.add_argument(
        "--simt-opcodes",
        default=None,
        help=(
            "comma-separated override for mixed full-Graph A/B runs; for "
            "example, --simt-opcodes 5 replaces only opcode 5"
        ),
    )
    parser.add_argument(
        "--simt-entry6",
        default="opcode6_simt_16w1r_v4_na",
        choices=(
            "opcode6_simt",
            "opcode6_simt_4w4r",
            "opcode6_simt_8w4r",
            "opcode6_simt_16w1r",
            "opcode6_simt_cached",
            "opcode6_simt_16w1r_v4",
            "opcode6_simt_16w1r_v4_na",
            "opcode6_simt_16w1r_v4_na_l2_256b",
            "opcode6_simt_16w1r_v4_na_prefetch4k",
            "opcode6_simt_16w1r_v4_na_prefetch8k",
            "opcode6_simt_16w1r_v4_na_prefetch16k",
            "opcode6_simt_16w1r_cp256",
            "opcode6_simt_16w1r_cp512",
            "opcode6_simt_16w1r_cp1024",
            "opcode6_simt_16w1r_cp512x3",
            "opcode6_simt_16w1r_cp1024x3",
        ),
    )
    parser.add_argument(
        "--simt-entry4",
        default="opcode4_simt_128cta_16w_v4_na",
        choices=(
            "opcode4_simt_128cta_8w2r",
            "opcode4_simt_128cta_8w2r_v4",
            "opcode4_simt_128cta_8w2r_v4_na",
            "opcode4_simt_128cta_8w2r_v4_na_l2_256b",
            "opcode4_simt_128cta_16w_v4",
            "opcode4_simt_128cta_16w_v4_na",
            "opcode4_simt_128cta_16w_v4_na_l2_256b",
        ),
    )
    parser.add_argument("--pdl-op5-op6", action="store_true")
    parser.add_argument("--pdl-op4-op5", action="store_true")
    parser.add_argument("--pdl-op1-op2", action="store_true")
    parser.add_argument("--pdl-op2-op4", action="store_true")
    parser.add_argument("--pdl-op6-next", action="store_true")
    parser.add_argument("--pdl-op6-qkv", action="store_true")
    parser.add_argument("--pdl-op6-lmhead", action="store_true")
    parser.add_argument("--pdl-full-chain", action="store_true")
    parser.add_argument("--pdl-pairwise-winners", action="store_true")
    parser.add_argument(
        "--pdl-pairwise-split-down",
        action="store_true",
        help=(
            "replace the full opcode5 completion wait with four 2048-K "
            "producer/consumer flags and a split-K opcode6"
        ),
    )
    parser.add_argument(
        "--pdl-fused-mlp",
        action="store_true",
        help=(
            "fuse each layer's accepted one-wave up+gate and down kernels "
            "with an in-kernel 132-CTA grid barrier"
        ),
    )
    parser.add_argument(
        "--pdl-pairwise-split-down-producer",
        choices=("contiguous", "phased4", "phased2-one", "phased2-pair"),
        default="phased2-one",
        help="opcode5 row ordering used by the fine-grained split-down edge",
    )
    parser.add_argument(
        "--pdl-pairwise-op6-smem-prefix-cols",
        type=int,
        choices=(0, 256, 512, 1024, 2048),
        default=0,
        help=(
            "cp.async this many down-weight columns per warp into SMEM "
            "before the opcode5->opcode6 dependency wait"
        ),
    )
    parser.add_argument(
        "--pdl-pairwise-op6-prefetch-bytes",
        type=int,
        choices=(0, 128, 256, 512, 1024, 2048),
        default=0,
        help="L2-hint prefix issued by opcode6 before its incoming PDL wait",
    )
    parser.add_argument(
        "--pdl-qkv-vready-attention",
        action="store_true",
        help="publish per-V-head readiness and bypass the full qkv grid wait",
    )
    parser.add_argument(
        "--pdl-q-ready-attention",
        action="store_true",
        help=(
            "publish per-Q-head readiness, pipeline historical KV before "
            "the dependency, and wait for grid completion only at current KV"
        ),
    )
    parser.add_argument(
        "--pdl-q-ready-attention-entry",
        default="opcode2_tk_short_8cta_1w_pdl_late_current_kv_wait_trigger",
        choices=(
            "opcode2_tk_short_8cta_1w_pdl_late_current_kv_wait_trigger",
            "opcode2_tk_short_cpasync_8cta_1w_s3_pdl_late_current_kv_wait_trigger",
            "opcode2_tk_short_cpasync_8cta_1w_s10_pdl_late_current_kv_wait_trigger",
        ),
        help="short-attention entry used by --pdl-q-ready-attention",
    )
    parser.add_argument(
        "--pdl-short-attention-entry",
        default="opcode2_tk_short_cpasync_8cta_1w_s10_hist_pdl_wait_trigger",
        choices=(
            "opcode2_tk_short_8cta_1w_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_1w_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_2w_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_4w_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_1w_regprefetch8_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_1w_regprefetch16_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_1w_regprefetch32_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_2w_regprefetch8_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_2w_regprefetch16_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_2w_regprefetch32_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_4w_regprefetch8_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_4w_regprefetch16_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_4w_regprefetch32_pdl_wait_trigger",
            "opcode2_simt_direct_gqa_8cta_4w_block16_reg32_pdl_wait_trigger",
            "opcode2_tk_short_8cta_1w_s4_pdl_wait_trigger",
            "opcode2_tk_short_8cta_1w_s5_pdl_wait_trigger",
            "opcode2_tk_short_8cta_1w_s10_pdl_wait_trigger",
            "opcode2_tk_short_8cta_1w_s3_hist_pdl_wait_trigger",
            "opcode2_tk_short_8cta_1w_s5_hist_pdl_wait_trigger",
            "opcode2_tk_short_8cta_1w_s10_hist_pdl_wait_trigger",
            "opcode2_tk_short_cpasync_8cta_1w_s3_pdl_wait_trigger",
            "opcode2_tk_short_cpasync_8cta_1w_s3_hist_pdl_wait_trigger",
            "opcode2_tk_short_cpasync_8cta_1w_s10_hist_pdl_wait_trigger",
            "opcode2_tk_short_cpasync_8cta_1w_s10_pdl_wait_trigger",
            "opcode2_tk_short_cpasync_8cta_1w_s10_hybrid6_pdl_wait_trigger",
            "opcode2_tk_short_reg1_8cta_1w_pdl_wait_trigger",
            "opcode2_tk_short_cpasync_reg1_8cta_1w_pdl_wait_trigger",
        ),
        help=(
            "short-context TK attention pipeline selected by composed PDL "
            "graphs at nonzero positions"
        ),
    )
    parser.add_argument(
        "--pdl-fuse-context1-attention",
        action="store_true",
        help=(
            "write the context-1 replicated V result in qkv's epilogue and "
            "replace opcode2 compute with a dependency-only node"
        ),
    )
    parser.add_argument(
        "--pdl-remove-context1-attention-node",
        action="store_true",
        help=(
            "after fusing context-1 attention stores into qkv, connect qkv "
            "directly to oproj and omit the dependency-only opcode2 node"
        ),
    )
    parser.add_argument(
        "--elide-static-barrier-reset",
        action="store_true",
        help=(
            "initialize Bar once when selected standalone kernels do not "
            "mutate it, rather than capturing a reset memcpy every replay"
        ),
    )
    parser.add_argument(
        "--pdl-context1-kv-only",
        action="store_true",
        help=(
            "context-1 specialization: compute persistent K/V state and "
            "fused attention output, but omit unobservable Q rows"
        ),
    )
    parser.add_argument(
        "--pdl-context1-kv-warps",
        type=int,
        choices=(4, 8, 12, 16, 24),
        default=8,
        help="CTA warp count for the 132-CTA context-1 KV-only QKV",
    )
    parser.add_argument(
        "--pdl-pairwise-op4-mode",
        choices=(
            "legacy",
            "entry128",
            "k50",
            "k75",
            "k875",
            "k9375",
            "epilogue0",
            "epilogue128",
        ),
        default="epilogue128",
    )
    parser.add_argument(
        "--pdl-pairwise-op5-prefetch-bytes",
        type=int,
        choices=(0, 128, 256, 512, 1024),
        default=128,
    )
    parser.add_argument(
        "--pdl-pairwise-op5-prefetch-matrix",
        choices=("both", "up", "gate"),
        default="both",
    )
    parser.add_argument(
        "--pdl-pairwise-op5-v4",
        action="store_true",
        help=(
            "use 64-bit BF16x4 loads in the exact 132-CTA one-wave "
            "up+gate kernel"
        ),
    )
    parser.add_argument(
        "--pdl-pairwise-op5-smem-mode",
        choices=("none", "cp128x3", "cp256x2", "cp256x3"),
        default="none",
        help=(
            "replace the opcode4->opcode5 L2-hint consumer with a "
            "consumer-warp cp.async GMEM-to-SMEM prologue"
        ),
    )
    parser.add_argument(
        "--pdl-pairwise-op5-reg-prefix-cols",
        type=int,
        choices=(0, 64, 128),
        default=0,
        help=(
            "retain this K-prefix in registers across the opcode4->opcode5 "
            "dependency wait; mutually exclusive with SMEM staging"
        ),
    )
    parser.add_argument(
        "--pdl-pairwise-op5-demand-bytes",
        type=int,
        choices=(0, 256, 512, 1024, 2048, 4096),
        default=0,
        help=(
            "issue real LDG.L2 warming for this many prefix bytes per "
            "weight row and matrix before the opcode4->opcode5 wait"
        ),
    )
    parser.add_argument(
        "--pdl-pairwise-qkv-prefetch-bytes",
        type=int,
        choices=(0, 128, 256, 512, 1024),
        default=256,
    )
    parser.add_argument(
        "--pdl-pairwise-qkv-trigger-mode",
        choices=(
            "entry",
            "mainloop",
            "k50",
            "k75",
            "k875",
            "k9375",
            "epilogue",
            "tail1",
            "tail2",
            "tail4",
        ),
        default="entry",
        help="programmatic qkv->attention trigger point for pairwise graphs",
    )
    parser.add_argument(
        "--pdl-pairwise-lm-mode",
        choices=(
            "legacy",
            "entry0",
            "entry2k",
            "epilogue0",
            "epilogue2k",
            "epilogue4k",
        ),
        default="epilogue0",
    )
    parser.add_argument(
        "--pdl-pairwise-lm-trigger-mode",
        choices=(
            "legacy",
            "entry",
            "k50",
            "k75",
            "k875",
            "k9375",
            "epilogue",
        ),
        default="legacy",
        help=(
            "override the final down->LM trigger while retaining opcode6's "
            "incoming prefetch policy"
        ),
    )
    parser.add_argument(
        "--pdl-pairwise-lm-prefetch-bytes",
        type=int,
        choices=(-1, 0, 128, 256, 512, 1024, 1536, 2048, 3072, 4096),
        default=-1,
        help="override LM-head's pre-wait L2 hint bytes per output row",
    )
    for edge in ("qkv", "op4", "op5", "op6", "lm"):
        parser.add_argument(
            f"--pdl-prefetch-policy-{edge}",
            choices=("none", "l1", "l2", "l2_evict_last"),
            default="l2",
            help=(
                f"cache policy for the {edge} pre-wait weight hint; none "
                "suppresses the hint while retaining the dependency wait"
            ),
        )
    parser.add_argument(
        "--pdl-op1-prefetch-bytes",
        type=int,
        choices=(0, 128, 256, 512, 1024),
        default=0,
    )
    parser.add_argument(
        "--pdl-op1-entry",
        default="opcode1_simt_24w_pdl_trigger",
        choices=(
            "opcode1_simt_24w_pdl_trigger",
            "opcode1_simt_24w_pdl_trigger_mainloop",
            "opcode1_simt_24w_pdl_trigger_k50",
            "opcode1_simt_24w_pdl_trigger_k75",
            "opcode1_simt_24w_pdl_trigger_k875",
            "opcode1_simt_24w_pdl_trigger_k9375",
            "opcode1_simt_24w_pdl_trigger_epilogue",
        ),
    )
    parser.add_argument(
        "--pdl-op2-entry",
        default="opcode2_simt_8cta_4w_pdl_trigger",
        choices=(
            "opcode2_simt_8cta_4w_pdl_trigger",
            "opcode2_simt_8cta_4w_pdl_trigger_epilogue",
        ),
    )
    parser.add_argument(
        "--pdl-op4-wait-entry",
        default="opcode4_simt_128cta_16w_v4_na_pdl_wait",
        choices=(
            "opcode4_simt_128cta_16w_v4_na_pdl_wait",
            "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch128b",
            "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch256b",
            "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch512b",
            "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch1k",
            "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch2k",
            "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch4k",
        ),
    )
    parser.add_argument(
        "--pdl-op4-entry",
        default="opcode4_simt_128cta_16w_v4_na_pdl_trigger",
        choices=(
            "opcode4_simt_128cta_16w_v4_na_pdl_trigger",
            "opcode4_simt_128cta_16w_v4_na_pdl_trigger_k50",
            "opcode4_simt_128cta_16w_v4_na_pdl_trigger_k75",
            "opcode4_simt_128cta_16w_v4_na_pdl_trigger_k875",
            "opcode4_simt_128cta_16w_v4_na_pdl_trigger_k9375",
            "opcode4_simt_128cta_16w_v4_na_pdl_trigger_epilogue",
        ),
    )
    parser.add_argument(
        "--pdl-op5-entry",
        default=(
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_trigger"
        ),
        choices=(
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_trigger",
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_trigger_k50",
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_trigger_k75",
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_trigger_k875",
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_trigger_k9375",
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_trigger_epilogue",
        ),
    )
    parser.add_argument(
        "--pdl-op5-chain-entry",
        default=(
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_wait_trigger"
        ),
        choices=(
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_wait_trigger",
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_wait_trigger_prefetch128b",
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_wait_trigger_prefetch256b",
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_wait_trigger_prefetch512b",
            "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_wait_trigger_prefetch1k",
        ),
    )
    parser.add_argument(
        "--pdl-op6-entry",
        default="opcode6_simt_16w1r_v4_na_pdl_wait",
        choices=(
            "opcode6_simt_16w1r_v4_na_pdl_wait",
            "opcode6_simt_16w1r_v4_na_pdl_prefetch128b",
            "opcode6_simt_16w1r_v4_na_pdl_prefetch256b",
            "opcode6_simt_16w1r_v4_na_pdl_prefetch512b",
            "opcode6_simt_16w1r_v4_na_pdl_prefetch1k",
            "opcode6_simt_16w1r_v4_na_pdl_prefetch2k",
            "opcode6_simt_16w1r_v4_na_pdl_prefetch4k",
            "opcode6_simt_16w1r_v4_na_pdl_prefetch8k",
            "opcode6_simt_16w1r_v4_na_pdl_prefetch16k",
        ),
    )
    parser.add_argument(
        "--pdl-op6-chain-entry",
        default="opcode6_simt_16w1r_v4_na_pdl_wait_trigger",
        choices=(
            "opcode6_simt_16w1r_v4_na_pdl_wait_trigger",
            "opcode6_simt_16w1r_v4_na_pdl_wait_trigger_k50",
            "opcode6_simt_16w1r_v4_na_pdl_wait_trigger_k75",
            "opcode6_simt_16w1r_v4_na_pdl_wait_trigger_k875",
            "opcode6_simt_16w1r_v4_na_pdl_wait_trigger_k9375",
            "opcode6_simt_16w1r_v4_na_pdl_wait_trigger_epilogue",
        ),
    )
    parser.add_argument(
        "--pdl-op7-entry",
        default=(
            "opcode7_simt_32w3r_na_v4_fullunroll_prefetch4096_pdl_wait"
        ),
        choices=(
            "opcode7_simt_32w3r_na_v4_fullunroll_balanced_tail_pdl_wait",
            "opcode7_simt_32w3r_na_v4_fullunroll_prefetch128_pdl_wait",
            "opcode7_simt_32w3r_na_v4_fullunroll_prefetch256_pdl_wait",
            "opcode7_simt_32w3r_na_v4_fullunroll_prefetch512_pdl_wait",
            "opcode7_simt_32w3r_na_v4_fullunroll_prefetch1024_pdl_wait",
            "opcode7_simt_32w3r_na_v4_fullunroll_prefetch1536_pdl_wait",
            "opcode7_simt_32w3r_na_v4_fullunroll_prefetch2048_pdl_wait",
            "opcode7_simt_32w3r_na_v4_fullunroll_prefetch3072_pdl_wait",
            "opcode7_simt_32w3r_na_v4_fullunroll_prefetch4096_pdl_wait",
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
PDL_OP6_QKV = (
    ARGS.pdl_op6_next or ARGS.pdl_op6_qkv or ARGS.pdl_pairwise_winners
)
PDL_OP6_LMHEAD = (
    ARGS.pdl_op6_next or ARGS.pdl_op6_lmhead or ARGS.pdl_pairwise_winners
)
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
from megakernels.llama import LlamaForCausalLM  # noqa: E402
from megakernels.model_types import BatchState, ExtraModelConfig  # noqa: E402
from megakernels.scheduler import assign_to_sms, tensorize_instructions  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def summarize(samples: list[float]) -> dict[str, object]:
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


def tensor_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 0.10,
    atol: float = 0.10,
) -> dict[str, object]:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_diff = (actual_f - expected_f).abs()
    denominator = actual_f.abs() + expected_f.abs() + 1e-6
    return {
        "correct": bool(torch.allclose(actual_f, expected_f, rtol=rtol, atol=atol)),
        "max_abs_diff": float(abs_diff.max().item()),
        "mean_abs_diff": float(abs_diff.mean().item()),
        "mean_relative_diff": float(
            (2 * abs_diff / denominator).mean().item()
        ),
    }


def make_node_globs(base_globs, instructions):
    node_globs = copy.copy(base_globs)
    queues = assign_to_sms(
        "rr",
        instructions=instructions,
        sm_count=base_globs.sm_count(),
    )
    tensorize_instructions(node_globs, queues)
    if tuple(node_globs.instructions.shape) != (
        base_globs.sm_count(),
        1,
        32,
    ):
        raise RuntimeError(
            f"unexpected node instruction shape {node_globs.instructions.shape}"
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
            start_block_idx=block_idx,
            end_block_idx=block_idx + 1,
            reduction_block_idx=0,
        )
        for block_idx in range(
            globs.hidden_size // globs.o_proj_block_size
        )
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


def reset_model_caches(model) -> None:
    model.stacked_kv_cache[0].zero_()
    model.stacked_kv_cache[1].zero_()


def make_workload_model():
    max_context = ARGS.position + 1
    if ARGS.trajectory_prompt_length is not None:
        max_context = (
            ARGS.trajectory_prompt_length + ARGS.trajectory_output_length - 1
        )
    if ARGS.model_path is None:
        return baseline.make_model(ARGS.device, max_context, ARGS.seed)

    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)
    torch.set_default_device(ARGS.device)
    torch.set_default_dtype(torch.bfloat16)
    extra = ExtraModelConfig(
        interleave_rope=True,
        max_len_override=max(8192, max_context + 2),
        max_batch_size=1,
    )
    return LlamaForCausalLM.from_pretrained(
        str(ARGS.model_path),
        extra_config=extra,
        device=ARGS.device,
        dtype=torch.bfloat16,
    )


def token_hash(tokens: torch.Tensor) -> str:
    host_tokens = tokens.detach().cpu()
    if host_tokens.dtype == torch.bfloat16:
        host_tokens = host_tokens.view(torch.int16)
    return hashlib.sha256(host_tokens.numpy().tobytes()).hexdigest()


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


def module_provenance(module: object | None) -> dict[str, object] | None:
    if module is None:
        return None
    module_path = Path(getattr(module, "__file__")).resolve()
    return {
        "path": str(module_path),
        "sha256": file_sha256(module_path),
        "cuda_compiler": cuda_compiler_identity(module),
        "cuda_compiler_version": getattr(module, "cuda_compiler_version", None),
        "cuda_runtime_header_version": getattr(
            module, "cuda_runtime_header_version", None
        ),
    }


def require_matching_module_toolchain(
    expected_label: str,
    expected: dict[str, object] | None,
    observed_label: str,
    observed: dict[str, object] | None,
) -> None:
    for label, provenance in (
        (expected_label, expected),
        (observed_label, observed),
    ):
        compiler = provenance.get("cuda_compiler") if provenance else None
        header = (
            provenance.get("cuda_runtime_header_version")
            if provenance
            else None
        )
        if (
            not isinstance(compiler, dict)
            or set(compiler) != {"major", "minor", "build"}
            or any(value is None for value in compiler.values())
            or header is None
        ):
            raise RuntimeError(
                f"{label} does not expose complete CUDA toolchain provenance"
            )
    for field in ("cuda_compiler", "cuda_runtime_header_version"):
        if observed[field] != expected[field]:
            raise RuntimeError(
                f"{observed_label} uses different {field} from "
                f"{expected_label}: {observed[field]} != {expected[field]}"
            )


@lru_cache(maxsize=1)
def load_binding_contract_artifact() -> dict[str, str]:
    contract_path = ARGS.binding_contract.expanduser().resolve()
    contract = json.loads(contract_path.read_text())
    if contract.get("schema") != "hazy-h100-binding-contract-v1":
        raise RuntimeError(f"unrecognized binding contract: {contract_path}")
    return {
        "path": str(contract_path),
        "sha256": file_sha256(contract_path),
        "schema": contract["schema"],
    }


def load_binding_preflight_artifact(
    *,
    candidate_module: object,
    reference_module: object,
    simt_module: object | None,
) -> dict[str, str]:
    artifact = ARGS.binding_preflight_artifact
    if artifact is None:
        raise RuntimeError(
            "release contract enforcement requires "
            "--binding-preflight-artifact"
        )
    artifact = artifact.expanduser().resolve()
    payload = json.loads(artifact.read_text())
    expected_keys = {
        "arrival_metadata",
        "binding_contract",
        "candidate_module",
        "graph_helper_module",
        "issue_metadata",
        "persistent_module",
        "persistent_vm",
        "schema",
        "status",
    }
    if set(payload) != expected_keys:
        raise RuntimeError(
            "binding preflight has unexpected or missing keys: "
            f"{sorted(set(payload) ^ expected_keys)}"
        )
    if payload.get("schema") != "hazy-binding-preflight-v1":
        raise RuntimeError(f"unrecognized binding preflight: {artifact}")
    if payload.get("status") != "ok" or payload.get("persistent_vm") is not True:
        raise RuntimeError(f"binding preflight did not pass: {artifact}")
    if payload.get("binding_contract") != load_binding_contract_artifact():
        raise RuntimeError("binding contract changed after preflight")
    candidate_provenance = module_provenance(candidate_module)
    expected_modules = {
        "candidate_module": candidate_provenance,
        "graph_helper_module": module_provenance(simt_module),
        "persistent_module": module_provenance(reference_module),
    }
    for label in ("graph_helper_module", "persistent_module"):
        require_matching_module_toolchain(
            "candidate_module",
            candidate_provenance,
            label,
            expected_modules[label],
        )
    for label, expected in expected_modules.items():
        if payload.get(label) != expected:
            raise RuntimeError(f"{label} changed after binding preflight")
    return {
        "path": str(artifact),
        "sha256": file_sha256(artifact),
        "schema": payload["schema"],
        "status": payload["status"],
    }


def expected_rewritten_edges(
    edge_count: int, edge_mask: int, edge_period: int
) -> int:
    complete_periods, remainder = divmod(edge_count, edge_period)
    return (
        complete_periods
        * (edge_mask & ((1 << edge_period) - 1)).bit_count()
        + (edge_mask & ((1 << remainder) - 1)).bit_count()
    )


def require_rewritten_edges(
    *,
    scope: str,
    observed: int,
    edge_count: int,
    edge_mask: int,
    edge_period: int,
) -> int:
    expected = expected_rewritten_edges(edge_count, edge_mask, edge_period)
    if observed != expected:
        raise RuntimeError(
            f"{scope}: expected {expected} rewritten edges from "
            f"{edge_count} ordinals, got {observed}"
        )
    return expected


def validate_tensor_limits(
    label: str, limits: object
) -> dict[str, dict[str, float]]:
    if not isinstance(limits, dict) or set(limits) != set(TENSOR_NAMES):
        raise RuntimeError(
            f"{label} must define exactly these tensors: {TENSOR_NAMES}"
        )
    validated: dict[str, dict[str, float]] = {}
    for tensor_name in TENSOR_NAMES:
        tensor_limits = limits[tensor_name]
        if not isinstance(tensor_limits, dict) or set(tensor_limits) != set(
            ERROR_METRICS
        ):
            raise RuntimeError(
                f"{label}.{tensor_name} must define exactly {ERROR_METRICS}"
            )
        validated[tensor_name] = {}
        for metric in ERROR_METRICS:
            value = tensor_limits[metric]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise RuntimeError(
                    f"{label}.{tensor_name}.{metric} must be finite and "
                    "nonnegative"
                )
            validated[tensor_name][metric] = float(value)
    return validated


@lru_cache(maxsize=1)
def load_correctness_contract() -> tuple[dict[str, object], dict[str, str]]:
    contract_path = ARGS.correctness_contract.expanduser().resolve()
    contract = json.loads(contract_path.read_text())
    if contract.get("schema") != "hazy-p32-fixed-correctness-contract-v2":
        raise RuntimeError(f"unrecognized correctness contract: {contract_path}")
    expected_keys = {
        "schema",
        "position",
        "fixed_input_token",
        "fixed_input_rationale",
        "model_safetensors_sha256",
        "persistent_replay_tensor_error_limits",
        "eager_graph_tensor_error_limits",
        "required_checks",
        "diagnostic_only",
        "diagnostic_rationale",
    }
    if set(contract) != expected_keys:
        raise RuntimeError(
            "correctness contract has unexpected or missing keys: "
            f"{sorted(set(contract) ^ expected_keys)}"
        )
    if contract.get("position") != ARGS.position:
        raise RuntimeError(
            f"correctness contract is for position {contract.get('position')}, "
            f"not {ARGS.position}"
        )
    if contract.get("fixed_input_token") != ARGS.fixed_input_token:
        raise RuntimeError(
            "correctness contract fixed token does not match "
            f"--fixed-input-token: {contract.get('fixed_input_token')} != "
            f"{ARGS.fixed_input_token}"
        )
    if not isinstance(contract["fixed_input_rationale"], str) or not contract[
        "fixed_input_rationale"
    ].strip():
        raise RuntimeError(
            "correctness contract fixed_input_rationale is missing"
        )
    if ARGS.model_safetensors_sha256 != contract.get(
        "model_safetensors_sha256"
    ):
        raise RuntimeError(
            "verified model digest does not match the correctness contract"
        )
    contract["persistent_replay_tensor_error_limits"] = (
        validate_tensor_limits(
            "persistent_replay_tensor_error_limits",
            contract["persistent_replay_tensor_error_limits"],
        )
    )
    contract["eager_graph_tensor_error_limits"] = validate_tensor_limits(
        "eager_graph_tensor_error_limits",
        contract["eager_graph_tensor_error_limits"],
    )
    if not isinstance(contract["required_checks"], list) or tuple(
        contract["required_checks"]
    ) != RELEASE_REQUIRED_CHECKS:
        raise RuntimeError(
            "correctness contract required_checks do not match the harness"
        )
    if not isinstance(contract["diagnostic_only"], list) or tuple(
        contract["diagnostic_only"]
    ) != RELEASE_DIAGNOSTICS:
        raise RuntimeError(
            "correctness contract diagnostic_only does not match the harness"
        )
    rationale = contract["diagnostic_rationale"]
    if (
        not isinstance(rationale, dict)
        or set(rationale) != set(RELEASE_DIAGNOSTICS)
        or not all(
            isinstance(value, str) and value.strip()
            for value in rationale.values()
        )
    ):
        raise RuntimeError(
            "correctness contract diagnostic_rationale is incomplete"
        )
    return contract, {
        "path": str(contract_path),
        "sha256": file_sha256(contract_path),
        "schema": contract["schema"],
    }


def tensor_envelope_check(
    observed: dict[str, object], limits: dict[str, object]
) -> dict[str, object]:
    tensors: dict[str, object] = {}
    for tensor_name, tensor_limits in limits.items():
        error = observed[tensor_name]
        checks = {
            metric: math.isfinite(float(error[metric]))
            and float(error[metric]) <= float(limit)
            for metric, limit in tensor_limits.items()
        }
        tensors[tensor_name] = {
            "observed": {
                metric: float(error[metric]) for metric in tensor_limits
            },
            "limits": tensor_limits,
            "passed": all(checks.values()),
        }
    return {
        "passed": all(value["passed"] for value in tensors.values()),
        "tensors": tensors,
    }


def load_fixed_correctness_gate(
    *,
    implementation_label: str,
    selected_kernel_entries: dict[str, str],
    candidate_module: object,
    reference_module: object,
    simt_module: object | None,
    binding_preflight_artifact: dict[str, str] | None,
) -> dict[str, object] | None:
    artifact = ARGS.fixed_correctness_artifact
    if artifact is None:
        if not ARGS.enforce_correctness_contract:
            return None
        raise RuntimeError(
            "trajectory timing requires --fixed-correctness-artifact"
        )
    artifact = artifact.expanduser().resolve()
    payload = json.loads(artifact.read_text())
    if payload.get("schema") != "hazy-fixed-position-correctness-v2":
        raise RuntimeError(f"unrecognized correctness artifact: {artifact}")
    if payload.get("correctness_gate_passed") is not True:
        raise RuntimeError(
            f"fixed-position correctness gate did not pass: {artifact}"
        )
    expected_model = str(ARGS.model_path.resolve()) if ARGS.model_path else None
    observed_model = payload.get("model_path")
    if observed_model is not None:
        observed_model = str(Path(observed_model).resolve())
    if observed_model != expected_model:
        raise RuntimeError(
            "fixed-position artifact uses a different model path: "
            f"{observed_model!r} != {expected_model!r}"
        )
    if payload.get("model_safetensors_sha256") != (
        ARGS.model_safetensors_sha256
    ):
        raise RuntimeError("fixed-position artifact uses a different model digest")
    if payload.get("position") != ARGS.position:
        raise RuntimeError("fixed-position artifact uses a different position")
    if payload.get("fixed_input_token") != ARGS.fixed_input_token:
        raise RuntimeError("fixed-position artifact uses a different input token")
    if payload.get("implementation") != implementation_label:
        raise RuntimeError("fixed-position artifact uses a different implementation")
    if payload.get("selected_kernel_entries") != selected_kernel_entries:
        raise RuntimeError("fixed-position artifact uses different kernel entries")
    if payload.get("direct_tk_pdl_body") != ARGS.direct_tk_pdl_body:
        raise RuntimeError("fixed-position artifact uses a different direct-TK body")
    if payload.get("graph_edge_mode") != ARGS.graph_edge_mode:
        raise RuntimeError("fixed-position artifact uses a different graph edge mode")
    if payload.get("launch_completion_edge_mask") != (
        ARGS.launch_completion_edge_mask
    ):
        raise RuntimeError("fixed-position artifact uses a different edge mask")
    expected_modules = {
        "candidate": module_provenance(candidate_module),
        "native_reference": module_provenance(reference_module),
        "graph_helper": module_provenance(simt_module),
    }
    if payload.get("module_provenance") != expected_modules:
        raise RuntimeError("compiled modules changed after the correctness gate")
    expected_binding_contract = load_binding_contract_artifact()
    if payload.get("binding_contract") != expected_binding_contract:
        raise RuntimeError(
            "binding contract changed after the fixed-position gate"
        )
    if payload.get("binding_preflight") != binding_preflight_artifact:
        raise RuntimeError(
            "binding preflight changed after the fixed-position gate"
        )
    expected_contract, expected_contract_artifact = load_correctness_contract()
    if payload.get("correctness_contract") != expected_contract_artifact:
        raise RuntimeError(
            "correctness contract changed after the fixed-position gate"
        )
    required_checks = payload.get("correctness_gate_checks")
    required_names = expected_contract["required_checks"]
    if not isinstance(required_checks, dict) or set(required_checks) != set(
        required_names
    ):
        raise RuntimeError("fixed-position artifact has an invalid check set")
    if not all(required_checks.get(name) is True for name in required_names):
        raise RuntimeError(
            "fixed-position artifact does not pass every contract check"
        )
    return {
        "path": str(artifact),
        "sha256": file_sha256(artifact),
        "correctness_gate_passed": True,
        "checks": payload.get("correctness_gate_checks"),
    }


def first_token_divergence(
    candidate: list[int], reference: list[int]
) -> int | None:
    for index, (candidate_token, reference_token) in enumerate(
        zip(candidate, reference, strict=True)
    ):
        if candidate_token != reference_token:
            return index
    return None


def capture_raw_cuda_graph(fn, simt_module) -> int:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    stream_ptr = capture_stream.cuda_stream
    with torch.cuda.stream(capture_stream):
        simt_module.begin_cuda_graph_capture(stream_ptr)
        fn()
        raw_graph = simt_module.end_cuda_graph_capture(stream_ptr)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return raw_graph


def run_trajectory(
    *,
    model,
    globs,
    nodes,
    ready_barriers,
    reference_globs,
    candidate_module,
    reference_module,
    simt_module,
    implementation_label: str,
    selected_kernel_entries: dict[str, str],
    binding_preflight_artifact: dict[str, str] | None,
) -> None:
    """Measure the exact post-prefill P32/D128 autoregressive contract.

    One CUDA Graph is captured per decode position because ``pos_id`` is a
    by-value kernel argument.  Each graph includes embedding, the barrier
    reset, all split body kernels, argmax, and output-token publication.  The
    timed host loop launches the 127 position graphs exactly as the stock
    generator launches the 127 persistent-kernel forwards.
    """

    prompt_length = ARGS.trajectory_prompt_length
    if prompt_length is None:
        raise RuntimeError("trajectory prompt length is missing")
    output_length = ARGS.trajectory_output_length
    if prompt_length < 1 or output_length < 2:
        raise ValueError("trajectory prompt length must be >=1 and output >=2")
    if ARGS.position != prompt_length:
        raise ValueError(
            "trajectory kernel selection uses --position as its first decode "
            f"position; expected {prompt_length}, got {ARGS.position}"
        )

    fixed_correctness_gate = load_fixed_correctness_gate(
        implementation_label=implementation_label,
        selected_kernel_entries=selected_kernel_entries,
        candidate_module=candidate_module,
        reference_module=reference_module,
        simt_module=simt_module,
        binding_preflight_artifact=binding_preflight_artifact,
    )

    decode_forwards = output_length - 1
    final_position = prompt_length + decode_forwards - 1
    positions = list(range(prompt_length, final_position + 1))
    trajectory_edge_period = (
        4 if ARGS.pdl_remove_context1_attention_node else 5
    )
    trajectory_edge_mask = (
        0x06
        if ARGS.pdl_remove_context1_attention_node
        else ARGS.launch_completion_edge_mask
    )
    full_edge_count = len(nodes) - 1
    candidate_output = torch.zeros(
        (1, output_length), device=ARGS.device, dtype=torch.long
    )
    native_output = torch.zeros_like(candidate_output)
    position_ids = torch.arange(prompt_length, device=ARGS.device)

    def prepare_output(output: torch.Tensor, prompt_token: int) -> None:
        reset_model_caches(model)
        output.zero_()
        input_ids = torch.full(
            (1, prompt_length),
            prompt_token,
            device=ARGS.device,
            dtype=torch.long,
        )
        prefill = model(
            BatchState(input_ids=input_ids, position_ids=position_ids)
        )
        if prefill.output_ids is None:
            raise RuntimeError("prefill did not return output IDs")
        output[:, 0].copy_(prefill.output_ids[:, -1])
        torch.cuda.synchronize()

    # Capture against one valid, sequentially-produced cache trajectory.  The
    # five warmup executions inside capture_graph rewrite only the current KV
    # slot, so historical slots from the preceding position remain intact.
    prepare_output(candidate_output, ARGS.prompt_token_base - 1)
    candidate_graphs: list[object] = []
    trajectory_rewritten_edges: list[int] = []
    for step, position in enumerate(positions):
        globs.pos_id = position
        for _, node_globs, _ in nodes:
            node_globs.pos_id = position

        def candidate_step(step: int = step) -> None:
            embedded = model.model.embed_tokens(
                BatchState(input_ids=candidate_output[:, step : step + 1])
            )
            if embedded.hidden_states is None:
                raise RuntimeError("trajectory embedding produced no hidden state")
            globs.hidden_states.copy_(embedded.hidden_states.reshape(-1))
            globs.barriers.copy_(ready_barriers)
            for _, node_globs, kernel in nodes:
                interpret_with_mk(node_globs, kernel)
            candidate_output[:, step + 1].copy_(
                torch.argmax(globs.logits, dim=-1)
            )

        if ARGS.graph_edge_mode == "capture":
            candidate_graphs.append(baseline.capture_graph(candidate_step))
        else:
            raw_graph = capture_raw_cuda_graph(candidate_step, simt_module)
            rewritten = simt_module.rewrite_cuda_graph_kernel_edges(
                raw_graph,
                ARGS.graph_edge_mode == "launch-completion",
                trajectory_edge_mask,
                trajectory_edge_period,
                full_edge_count,
            )
            rewritten = int(rewritten)
            require_rewritten_edges(
                scope=f"trajectory full position {position}",
                observed=rewritten,
                edge_count=full_edge_count,
                edge_mask=trajectory_edge_mask,
                edge_period=trajectory_edge_period,
            )
            trajectory_rewritten_edges.append(rewritten)
            executable = simt_module.instantiate_cuda_graph_exec(raw_graph, 0)
            simt_module.destroy_cuda_graph(raw_graph)
            candidate_graphs.append(int(executable))

    def candidate_generate() -> None:
        if ARGS.graph_edge_mode == "capture":
            for graph in candidate_graphs:
                graph.replay()
        else:
            stream_ptr = torch.cuda.current_stream().cuda_stream
            for executable in candidate_graphs:
                simt_module.launch_cuda_graph_exec(executable, stream_ptr)

    def native_eager_generate() -> None:
        for step, position in enumerate(positions):
            embedded = model.model.embed_tokens(
                BatchState(input_ids=native_output[:, step : step + 1])
            )
            if embedded.hidden_states is None:
                raise RuntimeError("native embedding produced no hidden state")
            reference_globs.hidden_states.copy_(
                embedded.hidden_states.reshape(-1)
            )
            reference_globs.barriers.fill_(0)
            reference_globs.pos_id = position
            interpret_with_mk(reference_globs, reference_module.mk_llama)
            native_output[:, step + 1].copy_(
                torch.argmax(reference_globs.logits, dim=-1)
            )

    # Capture the native persistent megakernel with the exact same
    # one-position-per-CUDA-Graph launch contract as the split candidate.  The
    # previous eager native control mixed graph-replay and Python-launch
    # overhead into the candidate-vs-native delta.
    prepare_output(native_output, ARGS.prompt_token_base - 2)
    native_graphs: list[object] = []
    for step, position in enumerate(positions):
        reference_globs.pos_id = position

        def native_graph_step(step: int = step) -> None:
            embedded = model.model.embed_tokens(
                BatchState(input_ids=native_output[:, step : step + 1])
            )
            if embedded.hidden_states is None:
                raise RuntimeError(
                    "native graph embedding produced no hidden state"
                )
            reference_globs.hidden_states.copy_(
                embedded.hidden_states.reshape(-1)
            )
            reference_globs.barriers.fill_(0)
            interpret_with_mk(reference_globs, reference_module.mk_llama)
            native_output[:, step + 1].copy_(
                torch.argmax(reference_globs.logits, dim=-1)
            )

        native_graphs.append(baseline.capture_graph(native_graph_step))

    def native_graph_generate() -> None:
        for graph in native_graphs:
            graph.replay()

    # Also capture a matched body+LM scope.  Both variants include the barrier
    # reset but exclude embedding, argmax, and token publication.  Subtracting
    # these matched core timings from the matched full-step timings isolates
    # whether the residual appears inside the direct-TK/PDL chain or at the
    # surrounding graph boundaries.
    core_hidden = globs.hidden_states.detach().clone()

    def prepare_candidate_core() -> None:
        reset_model_caches(model)
        globs.hidden_states.copy_(core_hidden)
        torch.cuda.synchronize()

    def prepare_native_core() -> None:
        reset_model_caches(model)
        reference_globs.hidden_states.copy_(core_hidden)
        torch.cuda.synchronize()

    prepare_candidate_core()
    candidate_core_graphs: list[object] = []
    trajectory_core_rewritten_edges: list[int] = []
    for position in positions:
        globs.pos_id = position
        for _, node_globs, _ in nodes:
            node_globs.pos_id = position

        def candidate_core_step() -> None:
            globs.barriers.copy_(ready_barriers)
            for _, node_globs, kernel in nodes:
                interpret_with_mk(node_globs, kernel)

        if ARGS.graph_edge_mode == "capture":
            candidate_core_graphs.append(
                baseline.capture_graph(candidate_core_step)
            )
        else:
            raw_graph = capture_raw_cuda_graph(
                candidate_core_step, simt_module
            )
            rewritten = simt_module.rewrite_cuda_graph_kernel_edges(
                raw_graph,
                ARGS.graph_edge_mode == "launch-completion",
                trajectory_edge_mask,
                trajectory_edge_period,
                full_edge_count,
            )
            rewritten = int(rewritten)
            require_rewritten_edges(
                scope=f"trajectory body+LM position {position}",
                observed=rewritten,
                edge_count=full_edge_count,
                edge_mask=trajectory_edge_mask,
                edge_period=trajectory_edge_period,
            )
            trajectory_core_rewritten_edges.append(rewritten)
            executable = simt_module.instantiate_cuda_graph_exec(raw_graph, 0)
            simt_module.destroy_cuda_graph(raw_graph)
            candidate_core_graphs.append(int(executable))

    def candidate_core_generate() -> None:
        if ARGS.graph_edge_mode == "capture":
            for graph in candidate_core_graphs:
                graph.replay()
        else:
            stream_ptr = torch.cuda.current_stream().cuda_stream
            for executable in candidate_core_graphs:
                simt_module.launch_cuda_graph_exec(executable, stream_ptr)

    prepare_native_core()
    native_core_graphs: list[object] = []
    for position in positions:
        reference_globs.pos_id = position

        def native_core_step() -> None:
            reference_globs.barriers.fill_(0)
            interpret_with_mk(reference_globs, reference_module.mk_llama)

        native_core_graphs.append(baseline.capture_graph(native_core_step))

    def native_core_generate() -> None:
        for graph in native_core_graphs:
            graph.replay()

    if nodes[-1][0] != "final.opcode7_lm_head":
        raise RuntimeError("trajectory body split expected opcode7 as final node")
    body_nodes = nodes[:-1]
    body_edge_count = len(body_nodes) - 1

    prepare_candidate_core()
    candidate_body_graphs: list[object] = []
    trajectory_body_rewritten_edges: list[int] = []
    for position in positions:
        globs.pos_id = position
        for _, node_globs, _ in body_nodes:
            node_globs.pos_id = position

        def candidate_body_step() -> None:
            globs.barriers.copy_(ready_barriers)
            for _, node_globs, kernel in body_nodes:
                interpret_with_mk(node_globs, kernel)

        if ARGS.graph_edge_mode == "capture":
            candidate_body_graphs.append(
                baseline.capture_graph(candidate_body_step)
            )
        else:
            raw_graph = capture_raw_cuda_graph(
                candidate_body_step, simt_module
            )
            rewritten = simt_module.rewrite_cuda_graph_kernel_edges(
                raw_graph,
                ARGS.graph_edge_mode == "launch-completion",
                trajectory_edge_mask,
                trajectory_edge_period,
                body_edge_count,
            )
            rewritten = int(rewritten)
            require_rewritten_edges(
                scope=f"trajectory body position {position}",
                observed=rewritten,
                edge_count=body_edge_count,
                edge_mask=trajectory_edge_mask,
                edge_period=trajectory_edge_period,
            )
            trajectory_body_rewritten_edges.append(rewritten)
            executable = simt_module.instantiate_cuda_graph_exec(raw_graph, 0)
            simt_module.destroy_cuda_graph(raw_graph)
            candidate_body_graphs.append(int(executable))

    def candidate_body_generate() -> None:
        if ARGS.graph_edge_mode == "capture":
            for graph in candidate_body_graphs:
                graph.replay()
        else:
            stream_ptr = torch.cuda.current_stream().cuda_stream
            for executable in candidate_body_graphs:
                simt_module.launch_cuda_graph_exec(executable, stream_ptr)

    reference_body_schedule = make_schedule_builder("latency").build(model)
    reference_body_globs = reference_body_schedule.globs
    reference_body_instructions = [
        instruction
        for instruction in reference_body_schedule.get_linear_instructions()
        if instruction.opcode() != 7
    ]
    reference_body_queues = assign_to_sms(
        "rr",
        instructions=reference_body_instructions,
        sm_count=reference_body_globs.sm_count(),
    )
    tensorize_instructions(reference_body_globs, reference_body_queues)

    def prepare_native_body() -> None:
        reset_model_caches(model)
        reference_body_globs.hidden_states.copy_(core_hidden)
        torch.cuda.synchronize()

    prepare_native_body()
    native_body_graphs: list[object] = []
    for position in positions:
        reference_body_globs.pos_id = position

        def native_body_step() -> None:
            reference_body_globs.barriers.fill_(0)
            interpret_with_mk(reference_body_globs, reference_module.mk_llama)

        native_body_graphs.append(baseline.capture_graph(native_body_step))

    def native_body_generate() -> None:
        for graph in native_body_graphs:
            graph.replay()

    def run_tokens(
        output: torch.Tensor, generate, prompt_token: int
    ) -> list[int]:
        prepare_output(output, prompt_token)
        generate()
        torch.cuda.synchronize()
        return output[0].cpu().tolist()

    correctness_prompt_token = ARGS.prompt_token_base
    native_tokens = run_tokens(
        native_output, native_eager_generate, correctness_prompt_token
    )
    native_graph_tokens = run_tokens(
        native_output, native_graph_generate, correctness_prompt_token
    )
    candidate_tokens = run_tokens(
        candidate_output, candidate_generate, correctness_prompt_token
    )
    native_repeat_tokens = run_tokens(
        native_output, native_eager_generate, correctness_prompt_token
    )
    native_graph_repeat_tokens = run_tokens(
        native_output, native_graph_generate, correctness_prompt_token
    )
    candidate_repeat_tokens = run_tokens(
        candidate_output, candidate_generate, correctness_prompt_token
    )
    first_divergence = first_token_divergence(candidate_tokens, native_tokens)
    matched_tokens = sum(
        candidate == native
        for candidate, native in zip(
            candidate_tokens, native_tokens, strict=True
        )
    )

    variant_names = (
        "candidate_full",
        "native_graph_full",
        "native_eager_full",
        "candidate_core",
        "native_graph_core",
        "candidate_body",
        "native_graph_body",
    )
    samples_ms: dict[str, list[float]] = {
        name: [] for name in variant_names
    }
    host_samples_ms: dict[str, list[float]] = {
        name: [] for name in variant_names
    }
    timed_hashes: dict[str, list[str]] = {
        name: [] for name in variant_names
    }
    variants = {
        "candidate_full": (
            candidate_output,
            candidate_generate,
            lambda prompt_token: prepare_output(
                candidate_output, prompt_token
            ),
            candidate_output,
        ),
        "native_graph_full": (
            native_output,
            native_graph_generate,
            lambda prompt_token: prepare_output(native_output, prompt_token),
            native_output,
        ),
        "native_eager_full": (
            native_output,
            native_eager_generate,
            lambda prompt_token: prepare_output(native_output, prompt_token),
            native_output,
        ),
        "candidate_core": (
            None,
            candidate_core_generate,
            lambda _prompt_token: prepare_candidate_core(),
            globs.logits,
        ),
        "native_graph_core": (
            None,
            native_core_generate,
            lambda _prompt_token: prepare_native_core(),
            reference_globs.logits,
        ),
        "candidate_body": (
            None,
            candidate_body_generate,
            lambda _prompt_token: prepare_candidate_core(),
            globs.hidden_states,
        ),
        "native_graph_body": (
            None,
            native_body_generate,
            lambda _prompt_token: prepare_native_body(),
            reference_body_globs.hidden_states,
        ),
    }
    total_iterations = ARGS.warmup + ARGS.iterations
    for iteration in range(total_iterations):
        prompt_token = ARGS.prompt_token_base + 1 + iteration
        order = (
            variant_names[iteration % len(variant_names) :]
            + variant_names[: iteration % len(variant_names)]
        )
        for variant in order:
            _output, generate, prepare, hash_tensor = variants[variant]
            prepare(prompt_token)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter()
            start.record()
            generate()
            end.record()
            torch.cuda.synchronize()
            host_elapsed_ms = (time.perf_counter() - host_start) * 1000.0
            if iteration >= ARGS.warmup:
                samples_ms[variant].append(start.elapsed_time(end))
                host_samples_ms[variant].append(host_elapsed_ms)
                timed_hashes[variant].append(token_hash(hash_tensor))

    def summarize_generation(variant: str) -> dict[str, object]:
        generation_samples = samples_ms[variant]
        host_generation_samples = host_samples_ms[variant]
        generation_mean_ms = statistics.fmean(generation_samples)
        return {
            "samples_ms": generation_samples,
            "generation_ms_mean": generation_mean_ms,
            "generation_ms_median": statistics.median(generation_samples),
            "generation_ms_stdev": (
                statistics.stdev(generation_samples)
                if len(generation_samples) > 1
                else 0.0
            ),
            "decode_us_mean": (
                generation_mean_ms * 1000.0 / decode_forwards
            ),
            "decode_forwards_per_second": (
                decode_forwards / (generation_mean_ms / 1000.0)
            ),
            "host_wall_samples_ms": host_generation_samples,
            "host_wall_generation_ms_mean": statistics.fmean(
                host_generation_samples
            ),
            "output_token_sha256s": timed_hashes[variant],
        }

    candidate_summary = summarize_generation("candidate_full")
    native_graph_summary = summarize_generation("native_graph_full")
    native_eager_summary = summarize_generation("native_eager_full")
    candidate_core_summary = summarize_generation("candidate_core")
    native_core_summary = summarize_generation("native_graph_core")
    candidate_body_summary = summarize_generation("candidate_body")
    native_body_summary = summarize_generation("native_graph_body")
    candidate_decode_us = candidate_summary["decode_us_mean"]
    native_decode_us = native_graph_summary["decode_us_mean"]
    candidate_core_decode_us = candidate_core_summary["decode_us_mean"]
    native_core_decode_us = native_core_summary["decode_us_mean"]
    candidate_body_decode_us = candidate_body_summary["decode_us_mean"]
    native_body_decode_us = native_body_summary["decode_us_mean"]
    candidate_surrounding_us = (
        candidate_decode_us - candidate_core_decode_us
    )
    native_surrounding_us = native_decode_us - native_core_decode_us
    report = {
        "schema": "hazy-p32d128-split-trajectory-v3",
        "candidate": implementation_label + "_per_position_cuda_graph",
        "workload": {
            "batch": 1,
            "prompt_tokens": prompt_length,
            "output_tokens": output_length,
            "decode_forwards": decode_forwards,
            "decode_positions_inclusive": [positions[0], positions[-1]],
            "prefill_excluded": True,
            "timed_step_scope": (
                f"embedding + barrier reset + {len(nodes)} split body kernels "
                "+ argmax + output-token publication"
            ),
        },
        "timing_record_enabled": False,
        "timeline_collection_enabled": False,
        "warmup": ARGS.warmup,
        "trials": ARGS.iterations,
        "candidate_timing": candidate_summary,
        "native_matched_timing": native_graph_summary,
        "native_eager_timing": native_eager_summary,
        "candidate_body_lm_timing": candidate_core_summary,
        "native_body_lm_timing": native_core_summary,
        "candidate_body_timing": candidate_body_summary,
        "native_body_timing": native_body_summary,
        "candidate_pct_delta_vs_matched_native": (
            (candidate_decode_us / native_decode_us - 1.0) * 100.0
        ),
        "matched_scope_decomposition_us_per_forward": {
            "candidate_full_minus_body_lm": candidate_surrounding_us,
            "native_full_minus_body_lm": native_surrounding_us,
            "body_lm_candidate_minus_native": (
                candidate_core_decode_us - native_core_decode_us
            ),
            "body_candidate_minus_native": (
                candidate_body_decode_us - native_body_decode_us
            ),
            "candidate_lm_increment": (
                candidate_core_decode_us - candidate_body_decode_us
            ),
            "native_lm_increment": (
                native_core_decode_us - native_body_decode_us
            ),
            "lm_increment_candidate_minus_native": (
                (candidate_core_decode_us - candidate_body_decode_us)
                - (native_core_decode_us - native_body_decode_us)
            ),
            "surrounding_candidate_minus_native": (
                candidate_surrounding_us - native_surrounding_us
            ),
            "full_candidate_minus_native": (
                candidate_decode_us - native_decode_us
            ),
            "native_graph_minus_eager": (
                native_decode_us - native_eager_summary["decode_us_mean"]
            ),
        },
        "five_percent_target_decode_us": native_decode_us * 1.05,
        "within_five_percent_of_matched_native": bool(
            candidate_decode_us <= native_decode_us * 1.05
        ),
        "fixed_position_correctness_gate": fixed_correctness_gate,
        "binding_contract": (
            load_binding_contract_artifact()
            if ARGS.enforce_correctness_contract
            else None
        ),
        "binding_preflight": binding_preflight_artifact,
        "enforce_correctness_contract": (
            ARGS.enforce_correctness_contract
        ),
        "correctness": {
            "prompt_token": correctness_prompt_token,
            "candidate_tokens": candidate_tokens,
            "native_tokens": native_tokens,
            "first_output_index_divergence": first_divergence,
            "matched_output_tokens": matched_tokens,
            "total_output_tokens": output_length,
            "candidate_matches_native": first_divergence is None,
            "candidate_repeat_deterministic": (
                candidate_repeat_tokens == candidate_tokens
            ),
            "native_repeat_deterministic": native_repeat_tokens == native_tokens,
            "native_graph_matches_native_eager": (
                native_graph_tokens == native_tokens
            ),
            "native_graph_repeat_deterministic": (
                native_graph_repeat_tokens == native_graph_tokens
            ),
            "native_graph_tokens": native_graph_tokens,
            "candidate_sha256": token_hash(
                torch.tensor(candidate_tokens, dtype=torch.long)
            ),
            "native_sha256": token_hash(
                torch.tensor(native_tokens, dtype=torch.long)
            ),
        },
        "kernel_node_count_per_forward": len(nodes),
        "graph_count": len(candidate_graphs),
        "native_graph_count": len(native_graphs),
        "candidate_body_lm_graph_count": len(candidate_core_graphs),
        "native_body_lm_graph_count": len(native_core_graphs),
        "candidate_body_graph_count": len(candidate_body_graphs),
        "native_body_graph_count": len(native_body_graphs),
        "graph_edge_mode": ARGS.graph_edge_mode,
        "launch_completion_edge_mask": ARGS.launch_completion_edge_mask,
        "effective_edge_mask": trajectory_edge_mask,
        "effective_edge_period": trajectory_edge_period,
        "trajectory_edge_contract": {
            "full_and_body_lm_edge_ordinals": full_edge_count,
            "body_edge_ordinals": body_edge_count,
            "expected_full_and_body_lm_rewrites": expected_rewritten_edges(
                full_edge_count,
                trajectory_edge_mask,
                trajectory_edge_period,
            ),
            "expected_body_rewrites": expected_rewritten_edges(
                body_edge_count,
                trajectory_edge_mask,
                trajectory_edge_period,
            ),
        },
        "trajectory_rewritten_edges": trajectory_rewritten_edges,
        "trajectory_core_rewritten_edges": (
            trajectory_core_rewritten_edges
        ),
        "trajectory_body_rewritten_edges": (
            trajectory_body_rewritten_edges
        ),
        "selected_kernel_entries": selected_kernel_entries,
        "direct_tk_pdl_body": ARGS.direct_tk_pdl_body,
        "module_provenance": {
            "candidate": module_provenance(candidate_module),
            "native_reference": module_provenance(reference_module),
            "graph_helper": module_provenance(simt_module),
        },
        "native_reference_mk": (
            str(ARGS.native_reference_mk.resolve())
            if ARGS.native_reference_mk else None
        ),
        "model_path": str(ARGS.model_path.resolve()) if ARGS.model_path else None,
        "model_safetensors_sha256": ARGS.model_safetensors_sha256,
        "fixed_input_token": ARGS.fixed_input_token,
        "gpu": torch.cuda.get_device_name(ARGS.device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "seed": ARGS.seed,
    }
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


@torch.inference_mode()
def main() -> None:
    if ARGS.position < 0:
        raise ValueError("--position must be nonnegative")
    if ARGS.fixed_input_token < 0:
        raise ValueError("--fixed-input-token must be nonnegative")
    if ARGS.enforce_correctness_contract:
        if ARGS.allow_correctness_mismatch:
            raise ValueError(
                "--enforce-correctness-contract is incompatible with "
                "--allow-correctness-mismatch"
            )
        if ARGS.model_path is None or ARGS.model_safetensors_sha256 is None:
            raise ValueError(
                "release contract enforcement requires --model-path and "
                "--model-safetensors-sha256"
            )
        if ARGS.binding_preflight_artifact is None:
            raise ValueError(
                "release contract enforcement requires "
                "--binding-preflight-artifact"
            )
        if ARGS.graph_edge_mode == "capture":
            raise ValueError(
                "release contract enforcement requires programmatic or "
                "launch-completion graph edges"
            )
        # Reject a stale position/token/checkpoint or malformed release
        # contract before allocating model weights or capturing any graph.
        load_binding_contract_artifact()
        load_correctness_contract()
    context1_only_options = {
        "pdl_fuse_context1_attention": ARGS.pdl_fuse_context1_attention,
        "pdl_remove_context1_attention_node": (
            ARGS.pdl_remove_context1_attention_node
        ),
        "pdl_context1_kv_only": ARGS.pdl_context1_kv_only,
        "pdl_qkv_vready_attention": ARGS.pdl_qkv_vready_attention,
    }
    if ARGS.position != 0:
        enabled = [
            name for name, value in context1_only_options.items() if value
        ]
        if enabled:
            raise ValueError(
                "nonzero position is incompatible with context-1 options: "
                + ", ".join(enabled)
            )
    torch.cuda.set_device(ARGS.device)
    model = make_workload_model()
    builder = make_schedule_builder("latency")
    token = torch.tensor(
        [[ARGS.fixed_input_token]], device=ARGS.device, dtype=torch.long
    )
    embedded_state = model.model.embed_tokens(BatchState(input_ids=token))
    if embedded_state.hidden_states is None:
        raise RuntimeError("embedding did not produce hidden states")
    input_hidden = embedded_state.hidden_states.reshape(-1).clone()

    # Both extensions are emitted beside the low-latency Llama demo rather
    # than installed as package modules.  Add the directory explicitly so the
    # harness is independent of its launch working directory/PYTHONPATH.
    extension_dir = ARGS.mk_dir.expanduser().resolve()
    if not extension_dir.is_dir():
        raise RuntimeError(f"compiled extension directory is missing: {extension_dir}")
    sys.path.insert(0, str(extension_dir))
    reference_module = importlib.import_module("mk_llama")
    if Path(reference_module.__file__).resolve().parent != extension_dir:
        raise RuntimeError(
            f"mk_llama resolved outside --mk-dir: {reference_module.__file__}"
        )
    native_reference_module = reference_module
    if ARGS.native_reference_mk is not None:
        reference_spec = importlib.util.spec_from_file_location(
            "native_reference.mk_llama",
            ARGS.native_reference_mk.resolve(),
        )
        if reference_spec is None or reference_spec.loader is None:
            raise RuntimeError(
                f"cannot load native reference extension "
                f"{ARGS.native_reference_mk}"
            )
        native_reference_module = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(native_reference_module)
    production_opcodes = {1, 2, 4, 5, 6, 7}
    if ARGS.simt_opcodes is not None:
        simt_opcodes = {
            int(value)
            for value in ARGS.simt_opcodes.split(",")
            if value.strip()
        }
    elif ARGS.implementation == "pure_simt":
        simt_opcodes = set(production_opcodes)
    else:
        simt_opcodes = set()
    unknown_opcodes = simt_opcodes - production_opcodes
    if unknown_opcodes:
        raise ValueError(f"unsupported production opcodes: {unknown_opcodes}")
    simt_module = (
        importlib.import_module("mk_mlp_simt")
        if (
            simt_opcodes
            or ARGS.direct_tk_pdl_body
            or ARGS.graph_edge_mode != "capture"
        )
        else None
    )
    if (
        simt_module is not None
        and Path(simt_module.__file__).resolve().parent != extension_dir
    ):
        raise RuntimeError(
            f"mk_mlp_simt resolved outside --mk-dir: {simt_module.__file__}"
        )
    binding_preflight_artifact = (
        load_binding_preflight_artifact(
            candidate_module=reference_module,
            reference_module=native_reference_module,
            simt_module=simt_module,
        )
        if ARGS.enforce_correctness_contract
        else None
    )
    extracted_kernels = {
        1: reference_module.opcode1_direct,
        2: reference_module.opcode2_direct,
        4: reference_module.opcode4_direct,
        # Use the arithmetic-preserving extracted operator as the non-SIMT
        # control. The historical fixed4/no-atomic entry is an optimized,
        # numerically relaxed candidate and is not a correctness baseline for
        # a full real-weight P32/D128 trajectory.
        5: reference_module.opcode5_direct,
        6: reference_module.opcode6_direct,
        7: reference_module.opcode7_direct,
    }
    simt_kernels = (
        {
            1: simt_module.opcode1_simt_24w,
            2: (
                simt_module.opcode2_simt_8cta_4w
                if ARGS.position == 0
                else simt_module.opcode2_tk_short_8cta_1w
            ),
            4: getattr(simt_module, ARGS.simt_entry4),
            5: simt_module.opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b,
            6: getattr(simt_module, ARGS.simt_entry6),
            7: simt_module.opcode7_simt_32w3r_na_v4_fullunroll_prefetch4096,
        }
        if simt_module is not None
        else {}
    )
    if ARGS.pdl_op1_op2:
        if not {1, 2}.issubset(simt_opcodes):
            raise ValueError("--pdl-op1-op2 requires SIMT opcodes 1 and 2")
        simt_kernels[1] = getattr(simt_module, ARGS.pdl_op1_entry)
        simt_kernels[2] = (
            simt_module.opcode2_simt_8cta_4w_pdl_wait
            if ARGS.position == 0
            else simt_module.opcode2_tk_short_8cta_1w_pdl_wait
        )
    if ARGS.pdl_op2_op4:
        if not {2, 4}.issubset(simt_opcodes):
            raise ValueError("--pdl-op2-op4 requires SIMT opcodes 2 and 4")
        if ARGS.position == 0:
            op2_entry = ARGS.pdl_op2_entry
            if ARGS.pdl_op1_op2:
                op2_entry = op2_entry.replace(
                    "_pdl_trigger", "_pdl_wait_trigger"
                )
            simt_kernels[2] = getattr(simt_module, op2_entry)
        else:
            simt_kernels[2] = (
                simt_module.opcode2_tk_short_8cta_1w_pdl_wait_trigger
                if ARGS.pdl_op1_op2
                else simt_module.opcode2_tk_short_8cta_1w_pdl_trigger
            )
        simt_kernels[4] = getattr(simt_module, ARGS.pdl_op4_wait_entry)
    if ARGS.pdl_op5_op6:
        if not {5, 6}.issubset(simt_opcodes):
            raise ValueError("--pdl-op5-op6 requires SIMT opcodes 5 and 6")
        simt_kernels[5] = getattr(simt_module, ARGS.pdl_op5_entry)
        simt_kernels[6] = getattr(simt_module, ARGS.pdl_op6_entry)
    if ARGS.pdl_op4_op5:
        if ARGS.pdl_op2_op4:
            raise ValueError(
                "combined opcode2->opcode4->opcode5 PDL is enabled only "
                "after the pairwise opcode2->opcode4 winner is promoted"
            )
        if not ARGS.pdl_op5_op6:
            raise ValueError("--pdl-op4-op5 currently requires --pdl-op5-op6")
        simt_kernels[4] = getattr(simt_module, ARGS.pdl_op4_entry)
        simt_kernels[5] = getattr(simt_module, ARGS.pdl_op5_chain_entry)
    pdl_first_op1 = None
    pdl_following_op1 = None
    pdl_final_op7 = None
    pdl_op6_qkv_chain = None
    pdl_op6_lm_chain = None
    pdl_fused_mlp_qkv = None
    pdl_fused_mlp_lm = None
    pdl_op5_bf16_chain = None
    if PDL_OP6_QKV or PDL_OP6_LMHEAD:
        if not (ARGS.pdl_op5_op6 or ARGS.pdl_pairwise_winners):
            raise ValueError("opcode6 downstream PDL requires --pdl-op5-op6")
        pdl_op6_qkv_chain = getattr(simt_module, ARGS.pdl_op6_chain_entry)
        pdl_op6_lm_chain = pdl_op6_qkv_chain
        if PDL_OP6_QKV:
            pdl_following_op1 = getattr(
                simt_module,
                "opcode1_simt_24w_pdl_wait"
                + (
                    f"_prefetch{ARGS.pdl_op1_prefetch_bytes}"
                    if ARGS.pdl_op1_prefetch_bytes
                    else ""
                ),
            )
        if PDL_OP6_LMHEAD:
            pdl_final_op7 = getattr(simt_module, ARGS.pdl_op7_entry)
    if ARGS.pdl_full_chain:
        if not (
            ARGS.pdl_op4_op5
            and ARGS.pdl_op5_op6
            and PDL_OP6_QKV
            and PDL_OP6_LMHEAD
        ):
            raise ValueError(
                "--pdl-full-chain requires the op4->op5, op5->op6, and "
                "op6->next PDL flags"
            )
        qkv_trigger_suffix = (
            ""
            if ARGS.pdl_pairwise_qkv_trigger_mode == "entry"
            else "_" + ARGS.pdl_pairwise_qkv_trigger_mode
        )
        pdl_first_op1 = getattr(
            simt_module,
            "opcode1_simt_24w_pdl_trigger" + qkv_trigger_suffix,
        )
        pdl_following_op1 = getattr(
            simt_module,
            "opcode1_simt_24w_pdl_wait_trigger"
            + (
                f"_prefetch{ARGS.pdl_op1_prefetch_bytes}"
                if ARGS.pdl_op1_prefetch_bytes
                else ""
            ),
        )
        simt_kernels[2] = (
            simt_module.opcode2_simt_8cta_4w_pdl_wait_trigger
            if ARGS.position == 0
            else getattr(simt_module, ARGS.pdl_short_attention_entry)
        )
        if ARGS.pdl_q_ready_attention:
            if ARGS.position == 0:
                raise ValueError("Q-ready attention requires nonzero position")
            if (
                ARGS.pdl_pairwise_qkv_prefetch_bytes != 256
                or ARGS.pdl_pairwise_qkv_trigger_mode != "entry"
                or ARGS.pdl_prefetch_policy_qkv != "l2"
            ):
                raise ValueError(
                    "Q-ready attention currently requires entry trigger and "
                    "the accepted 256-byte L2 following-QKV hint"
                )
            pdl_first_op1 = simt_module.opcode1_simt_24w_pdl_trigger_qready
            pdl_following_op1 = (
                simt_module.opcode1_simt_24w_pdl_wait_trigger_prefetch256_qready
            )
            simt_kernels[2] = getattr(
                simt_module, ARGS.pdl_q_ready_attention_entry
            )
        simt_kernels[4] = (
            simt_module.opcode4_simt_128cta_16w_v4_na_pdl_wait_trigger
        )
    if ARGS.pdl_pairwise_winners:
        if simt_opcodes != production_opcodes:
            raise ValueError(
                "--pdl-pairwise-winners requires all production SIMT opcodes"
            )
        qkv_trigger_suffix = (
            ""
            if ARGS.pdl_pairwise_qkv_trigger_mode == "entry"
            else "_" + ARGS.pdl_pairwise_qkv_trigger_mode
        )
        pdl_first_op1 = getattr(
            simt_module,
            "opcode1_simt_24w_pdl_trigger" + qkv_trigger_suffix,
        )
        qkv_prefetch_suffix = (
            f"_prefetch{ARGS.pdl_pairwise_qkv_prefetch_bytes}"
            if ARGS.pdl_pairwise_qkv_prefetch_bytes
            else ""
        )
        pdl_following_op1 = getattr(
            simt_module,
            "opcode1_simt_24w_pdl_wait_trigger"
            + qkv_prefetch_suffix
            + qkv_trigger_suffix,
        )
        simt_kernels[2] = (
            simt_module.opcode2_simt_8cta_4w_pdl_wait_trigger
            if ARGS.position == 0
            else getattr(simt_module, ARGS.pdl_short_attention_entry)
        )
        if ARGS.pdl_q_ready_attention:
            if ARGS.position == 0:
                raise ValueError("Q-ready attention requires nonzero position")
            if (
                ARGS.pdl_pairwise_qkv_prefetch_bytes != 256
                or ARGS.pdl_pairwise_qkv_trigger_mode != "entry"
                or ARGS.pdl_prefetch_policy_qkv != "l2"
            ):
                raise ValueError(
                    "Q-ready attention currently requires entry trigger and "
                    "the accepted 256-byte L2 following-QKV hint"
                )
            pdl_first_op1 = simt_module.opcode1_simt_24w_pdl_trigger_qready
            pdl_following_op1 = (
                simt_module.opcode1_simt_24w_pdl_wait_trigger_prefetch256_qready
            )
            simt_kernels[2] = getattr(
                simt_module, ARGS.pdl_q_ready_attention_entry
            )
        pairwise_op4_entries = {
            "legacy": "opcode4_simt_128cta_16w_v4_na_pdl_wait_trigger",
            "entry128": (
                "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch128b_"
                "trigger_entry"
            ),
            "k50": (
                "opcode4_simt_128cta_16w_v4_na_pdl_wait_trigger_k50"
            ),
            "k75": (
                "opcode4_simt_128cta_16w_v4_na_pdl_wait_trigger_k75"
            ),
            "k875": (
                "opcode4_simt_128cta_16w_v4_na_pdl_wait_trigger_k875"
            ),
            "k9375": (
                "opcode4_simt_128cta_16w_v4_na_pdl_wait_trigger_k9375"
            ),
            "epilogue0": (
                "opcode4_simt_128cta_16w_v4_na_pdl_wait_trigger_epilogue"
            ),
            "epilogue128": (
                "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch128b_"
                "trigger_epilogue"
            ),
        }
        simt_kernels[4] = getattr(
            simt_module, pairwise_op4_entries[ARGS.pdl_pairwise_op4_mode]
        )
        op5_early_modes = sum((
            ARGS.pdl_pairwise_op5_smem_mode != "none",
            ARGS.pdl_pairwise_op5_reg_prefix_cols != 0,
            ARGS.pdl_pairwise_op5_demand_bytes != 0,
        ))
        if op5_early_modes > 1:
            raise ValueError(
                "opcode5 SMEM staging, register-prefix prefetch, and "
                "demand warming are mutually exclusive"
            )
        if ARGS.pdl_pairwise_op5_v4 and op5_early_modes:
            raise ValueError(
                "opcode5 v4 loads are mutually exclusive with alternate "
                "early-warming modes"
            )
        if ARGS.pdl_pairwise_op5_demand_bytes != 0:
            demand_suffix = {
                256: "256b",
                512: "512b",
                1024: "1k",
                2048: "2k",
                4096: "4k",
            }[ARGS.pdl_pairwise_op5_demand_bytes]
            simt_kernels[5] = getattr(
                simt_module,
                "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                f"wait_trigger_demand{demand_suffix}",
            )
        elif ARGS.pdl_pairwise_op5_reg_prefix_cols != 0:
            simt_kernels[5] = getattr(
                simt_module,
                "opcode5_simt_32w2r_i_132cta_regprefix"
                f"{ARGS.pdl_pairwise_op5_reg_prefix_cols}_pdl_wait_trigger",
            )
        elif ARGS.pdl_pairwise_op5_smem_mode != "none":
            simt_kernels[5] = getattr(
                simt_module,
                "opcode5_simt_32w2r_"
                f"{ARGS.pdl_pairwise_op5_smem_mode}_pdl_wait_trigger",
            )
        elif ARGS.pdl_pairwise_op5_v4:
            if (
                ARGS.pdl_pairwise_op5_prefetch_bytes != 128
                or ARGS.pdl_pairwise_op5_prefetch_matrix != "both"
                or ARGS.pdl_prefetch_policy_op5 != "l2"
            ):
                raise ValueError(
                    "opcode5 v4 currently requires the 128-byte/both-matrix "
                    "L2 prefetch winner"
                )
            simt_kernels[5] = getattr(
                simt_module,
                "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_v4_pdl_"
                "wait_trigger_prefetch128b",
            )
        else:
            effective_op5_prefetch_bytes = (
                0
                if ARGS.pdl_prefetch_policy_op5 == "none"
                else ARGS.pdl_pairwise_op5_prefetch_bytes
            )
            op5_prefetch_suffix = {
                0: "",
                128: "_prefetch128b",
                256: "_prefetch256b",
                512: "_prefetch512b",
                1024: "_prefetch1k",
            }[effective_op5_prefetch_bytes]
            if ARGS.pdl_prefetch_policy_op5 not in ("none", "l2"):
                if (
                    effective_op5_prefetch_bytes != 128
                    or ARGS.pdl_pairwise_op5_prefetch_matrix != "both"
                ):
                    raise ValueError(
                        "alternate opcode5 cache policies are bound only for "
                        "the 128-byte/both-matrix candidate"
                    )
                op5_prefetch_suffix += {
                    "l1": "_l1",
                    "l2_evict_last": "_l2_evict_last",
                }[ARGS.pdl_prefetch_policy_op5]
            if ARGS.pdl_pairwise_op5_prefetch_matrix != "both":
                if effective_op5_prefetch_bytes != 128:
                    raise ValueError(
                        "single-matrix op5 prefetch is available only for "
                        "the 128-byte L2-line candidate"
                    )
                op5_prefetch_suffix += (
                    f"_{ARGS.pdl_pairwise_op5_prefetch_matrix}_only"
                )
            simt_kernels[5] = getattr(
                simt_module,
                "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                "wait_trigger" + op5_prefetch_suffix,
            )
        pdl_op5_bf16_chain = simt_kernels[5]
        if (
            ARGS.pdl_pairwise_split_down
            and (
                ARGS.pdl_pairwise_op6_smem_prefix_cols
                or ARGS.pdl_pairwise_op6_prefetch_bytes
            )
        ):
            raise ValueError(
                "fine-grained split-down and opcode6 early weight modes "
                "are mutually exclusive"
            )
        if (
            ARGS.pdl_pairwise_op6_smem_prefix_cols
            and ARGS.pdl_pairwise_op6_prefetch_bytes
        ):
            raise ValueError(
                "opcode6 SMEM staging and L2 hinting are mutually exclusive"
            )
        if ARGS.pdl_pairwise_split_down:
            if (
                ARGS.pdl_pairwise_op5_v4
                or op5_early_modes
                or ARGS.pdl_prefetch_policy_op5 != "l2"
                or (
                    ARGS.pdl_pairwise_op5_prefetch_bytes != 128
                    or ARGS.pdl_pairwise_op5_prefetch_matrix != "both"
                )
            ):
                raise ValueError(
                    "split-down currently requires the 128-byte/both-matrix "
                    "opcode5 winner without an alternate early-warming mode"
                )
            split_producer_entries = {
                "contiguous": (
                    "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                    "wait_trigger_prefetch128b_split_ready"
                ),
                "phased4": (
                    "opcode5_simt_32w2r_i_132cta_split4_pipeline_pdl_"
                    "wait_trigger_prefetch128b"
                ),
                "phased2-one": (
                    "opcode5_simt_32w1r_i_132cta_split2_pipeline_pdl_"
                    "wait_trigger_prefetch128b"
                ),
                "phased2-pair": (
                    "opcode5_simt_16w2r_i_132cta_split2_pipeline_pdl_"
                    "wait_trigger_prefetch128b"
                ),
            }
            simt_kernels[5] = getattr(
                simt_module,
                split_producer_entries[
                    ARGS.pdl_pairwise_split_down_producer
                ],
            )
            pdl_op6_qkv_chain = simt_module.opcode6_simt_split4_pdl_trigger
            pdl_op6_lm_chain = pdl_op6_qkv_chain
        elif ARGS.pdl_pairwise_op6_smem_prefix_cols:
            prefix_suffix = {
                256: "512b",
                512: "1k",
                1024: "2k",
                2048: "4k",
            }[ARGS.pdl_pairwise_op6_smem_prefix_cols]
            pdl_op6_qkv_chain = getattr(
                simt_module,
                "opcode6_simt_16w1r_smem_prefix"
                f"{prefix_suffix}_pdl_wait_trigger",
            )
            pdl_op6_lm_chain = pdl_op6_qkv_chain
        elif (
            ARGS.pdl_pairwise_op6_prefetch_bytes
            and ARGS.pdl_prefetch_policy_op6 != "none"
        ):
            prefetch_suffix = {
                128: "128b",
                256: "256b",
                512: "512b",
                1024: "1k",
                2048: "2k",
            }[ARGS.pdl_pairwise_op6_prefetch_bytes]
            op6_policy_suffix = {
                "l1": "_l1",
                "l2": "",
                "l2_evict_last": "_l2_evict_last",
            }[ARGS.pdl_prefetch_policy_op6]
            if (
                ARGS.pdl_prefetch_policy_op6 != "l2"
                and ARGS.pdl_pairwise_op6_prefetch_bytes != 1024
            ):
                raise ValueError(
                    "L1 and L2::evict_last opcode6 candidates are bound only "
                    "for the accepted 1 KiB prefix"
                )
            pdl_op6_qkv_chain = getattr(
                simt_module,
                "opcode6_simt_16w1r_v4_na_pdl_prefetch"
                f"{prefetch_suffix}{op6_policy_suffix}_trigger",
            )
            pdl_op6_lm_chain = pdl_op6_qkv_chain
        else:
            pdl_op6_qkv_chain = (
                simt_module.opcode6_simt_16w1r_v4_na_pdl_wait_trigger
            )
            if ARGS.pdl_pairwise_lm_mode.startswith("epilogue"):
                pdl_op6_lm_chain = (
                    simt_module.opcode6_simt_16w1r_v4_na_pdl_wait_trigger_epilogue
                )
            else:
                pdl_op6_lm_chain = pdl_op6_qkv_chain
        if ARGS.pdl_pairwise_lm_trigger_mode != "legacy":
            if ARGS.pdl_prefetch_policy_op6 != "l2":
                raise ValueError(
                    "explicit LM trigger modes are currently bound only for "
                    "the opcode6 L2 policy"
                )
            if ARGS.pdl_pairwise_op6_prefetch_bytes != 1024:
                raise ValueError(
                    "explicit LM trigger modes currently require the 1 KiB "
                    "opcode6 incoming hint"
                )
            lm_trigger_suffix = (
                ""
                if ARGS.pdl_pairwise_lm_trigger_mode == "entry"
                else "_" + ARGS.pdl_pairwise_lm_trigger_mode
            )
            pdl_op6_lm_chain = getattr(
                simt_module,
                "opcode6_simt_16w1r_v4_na_pdl_prefetch1k_trigger"
                + lm_trigger_suffix,
            )
        if ARGS.pdl_prefetch_policy_lm == "none":
            pdl_final_op7 = (
                simt_module.opcode7_simt_32w3r_na_v4_fullunroll_balanced_tail_pdl_wait
            )
        elif ARGS.pdl_pairwise_lm_prefetch_bytes >= 0:
            lm_prefetch_bytes = ARGS.pdl_pairwise_lm_prefetch_bytes
            if lm_prefetch_bytes == 0:
                pdl_final_op7 = (
                    simt_module.opcode7_simt_32w3r_na_v4_fullunroll_balanced_tail_pdl_wait
                )
            else:
                lm_policy_suffix = {
                    "l1": "_l1",
                    "l2": "",
                    "l2_evict_last": "_l2_evict_last",
                }[ARGS.pdl_prefetch_policy_lm]
                if (
                    ARGS.pdl_prefetch_policy_lm != "l2"
                    and lm_prefetch_bytes != 256
                ):
                    raise ValueError(
                        "L1 and L2::evict_last LM candidates are bound only "
                        "for the accepted 256-byte prefix"
                    )
                pdl_final_op7 = getattr(
                    simt_module,
                    "opcode7_simt_32w3r_na_v4_fullunroll_prefetch"
                    f"{lm_prefetch_bytes}{lm_policy_suffix}_pdl_wait",
                )
        elif ARGS.pdl_pairwise_lm_mode in ("entry0", "epilogue0"):
            pdl_final_op7 = (
                simt_module.opcode7_simt_32w3r_na_v4_fullunroll_balanced_tail_pdl_wait
            )
        else:
            lm_prefetch_bytes = (
                4096
                if ARGS.pdl_pairwise_lm_mode in ("legacy", "epilogue4k")
                else 2048
            )
            pdl_final_op7 = getattr(
                simt_module,
                "opcode7_simt_32w3r_na_v4_fullunroll_prefetch"
                f"{lm_prefetch_bytes}_pdl_wait",
            )
        if ARGS.launch_completion_prune_triggers:
            if ARGS.graph_edge_mode != "launch-completion":
                raise ValueError(
                    "trigger pruning requires --graph-edge-mode "
                    "launch-completion"
                )
            prune_mask = ARGS.launch_completion_edge_mask
            if prune_mask & 0x01:
                pdl_first_op1 = simt_module.opcode1_simt_24w
                pdl_following_op1 = getattr(
                    simt_module,
                    "opcode1_simt_24w_pdl_wait" + qkv_prefetch_suffix,
                )
            if prune_mask & 0x02:
                attention_name = "opcode2_simt_8cta_4w_pdl_wait"
                if ARGS.launch_completion_plain_launch:
                    attention_name += "_plain_launch"
                simt_kernels[2] = getattr(simt_module, attention_name)
            if prune_mask & 0x04:
                if ARGS.pdl_prefetch_policy_op4 == "none":
                    op4_prefetch_suffix = ""
                else:
                    op4_prefetch_suffix = {
                        0: "",
                        128: "_prefetch128b",
                        256: "_prefetch256b",
                        512: "_prefetch512b",
                        1024: "_prefetch1k",
                        2048: "_prefetch2k",
                        4096: "_prefetch4k",
                    }[ARGS.launch_completion_op4_prefetch_bytes]
                    if ARGS.pdl_prefetch_policy_op4 != "l2":
                        if ARGS.launch_completion_op4_prefetch_bytes != 128:
                            raise ValueError(
                                "L1 and L2::evict_last opcode4 candidates "
                                "are bound only for the accepted 128 bytes"
                            )
                        op4_prefetch_suffix += {
                            "l1": "_l1",
                            "l2_evict_last": "_l2_evict_last",
                        }[ARGS.pdl_prefetch_policy_op4]
                if (
                    ARGS.launch_completion_plain_launch
                    and ARGS.pdl_prefetch_policy_op4 not in ("none", "l2")
                ):
                    raise ValueError(
                        "alternate opcode4 policies are not bound for plain "
                        "launch"
                    )
                simt_kernels[4] = getattr(
                    simt_module,
                    "opcode4_simt_128cta_16w_v4_na_pdl_wait"
                    + op4_prefetch_suffix
                    + (
                        "_plain_launch"
                        if ARGS.launch_completion_plain_launch
                        else ""
                    ),
                )
            if prune_mask & 0x08:
                if (
                    ARGS.pdl_pairwise_op5_smem_mode != "none"
                    or ARGS.pdl_pairwise_op5_demand_bytes != 0
                    or ARGS.pdl_pairwise_split_down
                    or (
                        ARGS.pdl_pairwise_op5_prefetch_matrix != "both"
                    )
                ):
                    raise ValueError(
                        "upgate trigger pruning currently requires the "
                        "both-matrix pairwise path"
                    )
                if ARGS.pdl_pairwise_op5_reg_prefix_cols:
                    simt_kernels[5] = getattr(
                        simt_module,
                        "opcode5_simt_32w2r_i_132cta_regprefix"
                        f"{ARGS.pdl_pairwise_op5_reg_prefix_cols}_pdl_wait",
                    )
                else:
                    effective_op5_bytes = (
                        0
                        if ARGS.pdl_prefetch_policy_op5 == "none"
                        else ARGS.launch_completion_op5_prefetch_bytes
                    )
                    op5_wait_suffix = {
                        0: "0b",
                        128: "128b",
                        256: "256b",
                        512: "512b",
                        1024: "1k",
                    }[effective_op5_bytes]
                    if ARGS.pdl_prefetch_policy_op5 != "l2" and (
                        ARGS.pdl_prefetch_policy_op5 != "none"
                    ):
                        if effective_op5_bytes != 256:
                            raise ValueError(
                                "L1 and L2::evict_last opcode5 candidates "
                                "are bound only for the accepted 256 bytes"
                            )
                        op5_wait_suffix += {
                            "l1": "_l1",
                            "l2_evict_last": "_l2_evict_last",
                        }[ARGS.pdl_prefetch_policy_op5]
                    if (
                        ARGS.launch_completion_plain_launch
                        and ARGS.pdl_prefetch_policy_op5 not in ("none", "l2")
                    ):
                        raise ValueError(
                            "alternate opcode5 policies are not bound for "
                            "plain launch"
                        )
                    simt_kernels[5] = getattr(
                        simt_module,
                        "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                        f"wait_prefetch{op5_wait_suffix}"
                        + (
                            "_plain_launch"
                            if ARGS.launch_completion_plain_launch
                            else ""
                        ),
                    )
            if prune_mask & 0x10:
                if (
                    ARGS.pdl_pairwise_split_down
                    or ARGS.pdl_pairwise_op6_smem_prefix_cols
                ):
                    raise ValueError(
                        "down trigger pruning is not wired to experimental "
                        "split/staged consumers"
                    )
                if ARGS.pdl_pairwise_op6_prefetch_bytes:
                    no_trigger_suffix = {
                        128: "128b",
                        256: "256b",
                        512: "512b",
                        1024: "1k",
                        2048: "2k",
                    }[ARGS.pdl_pairwise_op6_prefetch_bytes]
                    pdl_op6_qkv_chain = getattr(
                        simt_module,
                        "opcode6_simt_16w1r_v4_na_pdl_prefetch"
                        f"{no_trigger_suffix}",
                    )
                else:
                    pdl_op6_qkv_chain = (
                        simt_module.opcode6_simt_16w1r_v4_na_pdl_wait
                    )
                pdl_op6_lm_chain = pdl_op6_qkv_chain
        if ARGS.pdl_qkv_vready_attention:
            if not (
                ARGS.graph_edge_mode == "launch-completion"
                and ARGS.launch_completion_prune_triggers
                and (ARGS.launch_completion_edge_mask & 0x02)
            ):
                raise ValueError(
                    "V-ready attention requires a pruned launch-completion "
                    "attention->oproj edge"
                )
            if ARGS.pdl_pairwise_qkv_prefetch_bytes not in (0, 128):
                raise ValueError(
                    "V-ready QKV is currently bound for 0 or 128-byte hints"
                )
            pdl_first_op1 = simt_module.opcode1_simt_24w_pdl_trigger_vready
            pdl_following_op1_name = "opcode1_simt_24w_pdl_wait_trigger"
            if ARGS.pdl_pairwise_qkv_prefetch_bytes == 128:
                pdl_following_op1_name += "_prefetch128"
            pdl_following_op1 = getattr(
                simt_module, pdl_following_op1_name + "_vready"
            )
            simt_kernels[2] = simt_module.opcode2_simt_8cta_4w_vready_wait
        if ARGS.pdl_fuse_context1_attention:
            if ARGS.pdl_qkv_vready_attention:
                raise ValueError(
                    "fused context-1 attention and V-ready attention are "
                    "mutually exclusive"
                )
            if (
                ARGS.pdl_prefetch_policy_qkv != "none"
                and ARGS.pdl_pairwise_qkv_prefetch_bytes != 128
            ):
                raise ValueError(
                    "fused context-1 attention currently requires the "
                    "128-byte QKV hint"
                )
            if ARGS.pdl_pairwise_qkv_trigger_mode != "entry":
                raise ValueError(
                    "fused context-1 attention currently requires the entry "
                    "qkv trigger"
                )
            pdl_first_op1 = (
                simt_module.opcode1_simt_24w_pdl_trigger_fused_context1_attention
            )
            if ARGS.pdl_prefetch_policy_qkv == "none":
                following_qkv_name = (
                    "opcode1_simt_24w_pdl_wait_trigger_no_prefetch_"
                    "fused_context1_attention"
                )
            else:
                qkv_policy_suffix = {
                    "l1": "_l1",
                    "l2": "",
                    "l2_evict_last": "_l2_evict_last",
                }[ARGS.pdl_prefetch_policy_qkv]
                following_qkv_name = (
                    "opcode1_simt_24w_pdl_wait_trigger_prefetch128"
                    f"{qkv_policy_suffix}_fused_context1_attention"
                )
            pdl_following_op1 = getattr(simt_module, following_qkv_name)
            simt_kernels[2] = (
                simt_module.opcode2_simt_1cta_1w_pdl_wait_bypass
            )
        if ARGS.pdl_remove_context1_attention_node:
            if not ARGS.pdl_fuse_context1_attention:
                raise ValueError(
                    "removing opcode2 requires fused context-1 attention"
                )
            if ARGS.launch_completion_edge_mask != 0x0E:
                raise ValueError(
                    "the current opcode2-removal mapping requires mask 14"
                )
        if ARGS.pdl_context1_kv_only:
            if not ARGS.pdl_fuse_context1_attention:
                raise ValueError(
                    "KV-only context-1 QKV requires fused attention output"
                )
            pdl_first_op1 = getattr(
                simt_module,
                f"opcode1_simt_{ARGS.pdl_context1_kv_warps}w_pdl_trigger_"
                "fused_context1_attention_kv_only",
            )
            if ARGS.pdl_prefetch_policy_qkv == "none":
                kv_following_name = (
                    f"opcode1_simt_{ARGS.pdl_context1_kv_warps}w_pdl_wait_"
                    "trigger_no_prefetch_fused_context1_attention_kv_only"
                )
            else:
                qkv_policy_suffix = {
                    "l1": "_l1",
                    "l2": "",
                    "l2_evict_last": "_l2_evict_last",
                }[ARGS.pdl_prefetch_policy_qkv]
                kv_following_name = (
                    f"opcode1_simt_{ARGS.pdl_context1_kv_warps}w_pdl_wait_"
                    "trigger_prefetch128"
                    f"{qkv_policy_suffix}_fused_context1_attention_kv_only"
                )
            pdl_following_op1 = getattr(simt_module, kv_following_name)

    if ARGS.pdl_fused_mlp:
        if not ARGS.pdl_pairwise_winners:
            raise ValueError("fused MLP requires --pdl-pairwise-winners")
        if (
            ARGS.pdl_pairwise_split_down
            or ARGS.pdl_pairwise_op5_v4
            or ARGS.pdl_pairwise_op5_smem_mode != "none"
            or ARGS.pdl_pairwise_op5_reg_prefix_cols
            or ARGS.pdl_pairwise_op5_demand_bytes
            or ARGS.pdl_pairwise_op5_prefetch_bytes != 128
            or ARGS.pdl_pairwise_op5_prefetch_matrix != "both"
            or ARGS.pdl_prefetch_policy_op5 != "l2"
            or ARGS.pdl_pairwise_op6_smem_prefix_cols
            or ARGS.pdl_pairwise_op6_prefetch_bytes
        ):
            raise ValueError(
                "fused MLP currently matches only the accepted opcode5 and "
                "opcode6 pairwise defaults"
            )
        if ARGS.graph_edge_mode != "capture":
            raise ValueError("fused MLP currently requires captured graph edges")
        pdl_fused_mlp_qkv = (
            simt_module.opcode56_simt_fused_132cta_pdl_wait_trigger
        )
        pdl_fused_mlp_lm = (
            simt_module.opcode56_simt_fused_132cta_pdl_wait_trigger_epilogue
        )

    def kernel_entry_name(kernel) -> str:
        return getattr(kernel, "__name__", repr(kernel))

    selected_kernel_entries = {
        "qkv_first": kernel_entry_name(
            pdl_first_op1
            if pdl_first_op1 is not None
            else simt_kernels[1]
            if simt_module is not None
            else extracted_kernels[1]
        ),
        "qkv_following": kernel_entry_name(
            pdl_following_op1
            if pdl_following_op1 is not None
            else simt_kernels[1]
            if simt_module is not None
            else extracted_kernels[1]
        ),
        "op4": kernel_entry_name(
            simt_kernels[4] if simt_module is not None else extracted_kernels[4]
        ),
        "op5": kernel_entry_name(
            simt_kernels[5] if simt_module is not None else extracted_kernels[5]
        ),
        "op5_bf16": (
            kernel_entry_name(pdl_op5_bf16_chain)
            if pdl_op5_bf16_chain is not None
            else None
        ),
        "op6_qkv": kernel_entry_name(
            pdl_op6_qkv_chain
            if pdl_op6_qkv_chain is not None
            else simt_kernels[6]
            if simt_module is not None
            else extracted_kernels[6]
        ),
        "op6_lm": kernel_entry_name(
            pdl_op6_lm_chain
            if pdl_op6_lm_chain is not None
            else simt_kernels[6]
            if simt_module is not None
            else extracted_kernels[6]
        ),
        "mlp_fused_qkv": (
            kernel_entry_name(pdl_fused_mlp_qkv)
            if pdl_fused_mlp_qkv is not None
            else None
        ),
        "mlp_fused_lm": (
            kernel_entry_name(pdl_fused_mlp_lm)
            if pdl_fused_mlp_lm is not None
            else None
        ),
        "lm": kernel_entry_name(
            pdl_final_op7
            if pdl_final_op7 is not None
            else simt_kernels[7]
            if simt_module is not None
            else extracted_kernels[7]
        ),
    }
    candidate_kernels = {
        opcode: (
            simt_kernels[opcode]
            if opcode in simt_opcodes
            else extracted_kernels[opcode]
        )
        for opcode in production_opcodes
    }
    if ARGS.direct_tk_pdl_body:
        if simt_opcodes:
            raise ValueError(
                "--direct-tk-pdl-body is incompatible with --simt-opcodes "
                "and --implementation pure_simt"
            )
        direct_suffix = (
            "13page_issue_r96"
            if ARGS.direct_tk_pdl_body == "13page_issue_r96_vtp56"
            else ARGS.direct_tk_pdl_body
        )
        for opcode in (1, 2, 4, 5, 6):
            candidate_kernels[opcode] = getattr(
                reference_module,
                f"opcode{opcode}_direct_pdl_{direct_suffix}",
            )
        if ARGS.direct_tk_entry_suffix:
            if ARGS.direct_tk_opcode4_entry:
                raise ValueError(
                    "--direct-tk-entry-suffix is incompatible with "
                    "--direct-tk-opcode4-entry"
                )
            for opcode in (1, 2, 4, 5, 6):
                candidate_kernels[opcode] = getattr(
                    reference_module,
                    (
                        f"opcode{opcode}_direct_pdl_4page_"
                        f"{ARGS.direct_tk_entry_suffix}"
                    ),
                )
        if ARGS.direct_tk_opcode4_entry:
            candidate_kernels[4] = getattr(
                reference_module, ARGS.direct_tk_opcode4_entry
            )
        if ARGS.direct_tk_pdl_body == "13page_issue_r96_vtp56":
            candidate_kernels[5] = getattr(
                reference_module,
                "opcode5_direct_pdl_4page_13page_issue_r96_vtp_signal",
            )
            candidate_kernels[6] = getattr(
                reference_module,
                "opcode6_direct_pdl_4page_13page_issue_r96_vtp_wait",
            )
        candidate_kernels[7] = (
            getattr(reference_module, ARGS.direct_tk_opcode7_entry)
            if ARGS.direct_tk_opcode7_entry
            else reference_module.opcode7_direct_pdl_wait
        )
        selected_kernel_entries.update(
            {
                "qkv_first": kernel_entry_name(candidate_kernels[1]),
                "qkv_following": kernel_entry_name(candidate_kernels[1]),
                "op4": kernel_entry_name(candidate_kernels[4]),
                "op5": kernel_entry_name(candidate_kernels[5]),
                "op5_bf16": kernel_entry_name(candidate_kernels[5]),
                "op6_qkv": kernel_entry_name(candidate_kernels[6]),
                "op6_lm": kernel_entry_name(candidate_kernels[6]),
                "lm": kernel_entry_name(candidate_kernels[7]),
            }
        )
    implementation_label = (
        "pure_simt"
        if simt_opcodes == production_opcodes
        else "extracted_tk"
        if not simt_opcodes
        else "mixed_simt_" + "_".join(str(value) for value in sorted(simt_opcodes))
    )
    if ARGS.pdl_op5_op6:
        implementation_label += "_pdl_op5_op6"
    if ARGS.pdl_op4_op5:
        implementation_label += "_pdl_op4_op5"
    if ARGS.pdl_op1_op2:
        implementation_label += "_pdl_op1_op2"
    if ARGS.pdl_op2_op4:
        implementation_label += "_pdl_op2_op4"
    if PDL_OP6_QKV:
        implementation_label += "_pdl_op6_qkv"
    if PDL_OP6_LMHEAD:
        implementation_label += "_pdl_op6_lmhead"
    if ARGS.pdl_full_chain:
        implementation_label += "_pdl_full_chain"
    if ARGS.pdl_pairwise_winners:
        implementation_label += "_pdl_pairwise_winners"
    if ARGS.pdl_pairwise_split_down:
        implementation_label += "_split_down"
    if ARGS.pdl_fused_mlp:
        implementation_label += "_fused_mlp"
    if ARGS.pdl_qkv_vready_attention:
        implementation_label += "_qkv_vready"
    if ARGS.pdl_q_ready_attention:
        implementation_label += "_q_ready_attention"
    if ARGS.pdl_fuse_context1_attention:
        implementation_label += "_fused_context1_attention"
    if ARGS.pdl_remove_context1_attention_node:
        implementation_label += "_no_opcode2_node"
    if ARGS.pdl_context1_kv_only:
        implementation_label += "_context1_kv_only"
    if ARGS.launch_completion_prune_triggers:
        implementation_label += "_pruned_triggers"
    if ARGS.direct_tk_pdl_body:
        implementation_label = (
            "direct_tk_pdl_body_" + ARGS.direct_tk_pdl_body + "_lm_pdl"
        )
        if ARGS.direct_tk_entry_suffix:
            implementation_label += "_" + ARGS.direct_tk_entry_suffix

    # Reference: the unmodified persistent schedule and arithmetic at the
    # requested decode position. Historical K/V slots remain zero in this
    # position-local harness; both paths observe the same cache contents.
    reference_schedule = builder.build(model)
    reference_queues = assign_to_sms("rr", schedule=reference_schedule)
    tensorize_instructions(reference_schedule.globs, reference_queues)
    reference_globs = reference_schedule.globs
    reference_globs.pos_id = ARGS.position
    reference_globs.barriers.zero_()
    reference_globs.hidden_states.copy_(input_hidden)
    reset_model_caches(model)
    interpret_with_mk(reference_globs, native_reference_module.mk_llama)
    torch.cuda.synchronize()
    reference = {
        "logits": reference_globs.logits.clone(),
        "hidden": reference_globs.hidden_states.clone(),
        "k": reference_globs.k_cache[:, :, ARGS.position].clone(),
        "v": reference_globs.v_cache[:, :, ARGS.position].clone(),
        "token": int(torch.argmax(reference_globs.logits).item()),
    }

    # Standalone nodes share every model/activation tensor but own immutable
    # per-node instruction pages.  This is required because CUDA Graph kernel
    # arguments capture tensor pointers rather than instruction contents.
    standalone_schedule = builder.build(model)
    globs = standalone_schedule.globs
    globs.pos_id = ARGS.position
    if not globs.skip_attn_reduction:
        raise RuntimeError("short-context graph must use the reduction-skipping path")

    nodes = []
    for layer in range(globs.num_hidden_layers):
        layer_nodes = [
                (
                    f"layer{layer:02d}.opcode1_qkv",
                    make_node_globs(globs, schedule_qkv(globs, layer)),
                    (
                        pdl_first_op1
                        if (ARGS.pdl_full_chain or ARGS.pdl_pairwise_winners)
                        and layer == 0
                        else pdl_following_op1
                        if PDL_OP6_QKV and layer > 0
                        else candidate_kernels[1]
                    ),
                ),
        ]
        if not ARGS.pdl_remove_context1_attention_node:
            layer_nodes.append(
                (
                    f"layer{layer:02d}.opcode2_partial_attention",
                    make_node_globs(globs, schedule_partial(globs, layer)),
                    candidate_kernels[2],
                )
            )
        layer_nodes.append(
            (
                f"layer{layer:02d}.opcode4_oproj",
                make_node_globs(globs, schedule_oproj(globs, layer)),
                candidate_kernels[4],
            )
        )
        if ARGS.pdl_fused_mlp:
            layer_nodes.append(
                (
                    f"layer{layer:02d}.opcode56_fused_mlp",
                    make_node_globs(
                        globs, schedule_accepted_upgate(globs, layer)
                    ),
                    pdl_fused_mlp_lm
                    if layer + 1 == globs.num_hidden_layers
                    else pdl_fused_mlp_qkv,
                )
            )
        else:
            layer_nodes.extend(
                [
                    (
                        f"layer{layer:02d}.opcode5_upgate",
                        make_node_globs(
                            globs, schedule_accepted_upgate(globs, layer)
                        ),
                        pdl_op5_bf16_chain
                        if pdl_op5_bf16_chain is not None
                        else candidate_kernels[5],
                    ),
                    (
                        f"layer{layer:02d}.opcode6_downproj",
                        make_node_globs(
                            globs, schedule_downproj(globs, layer)
                        ),
                        (
                            pdl_op6_qkv_chain
                            if (
                                PDL_OP6_QKV
                                and layer + 1 < globs.num_hidden_layers
                            )
                            else pdl_op6_lm_chain
                            if PDL_OP6_LMHEAD
                            and layer + 1 == globs.num_hidden_layers
                            else candidate_kernels[6]
                        ),
                    ),
                ]
            )
        nodes.extend(layer_nodes)
    nodes.append(
        (
            "final.opcode7_lm_head",
            make_node_globs(globs, schedule_lm_head(globs)),
            pdl_final_op7 if PDL_OP6_LMHEAD else candidate_kernels[7],
        )
    )
    expected_nodes = 81
    if ARGS.pdl_remove_context1_attention_node:
        expected_nodes -= globs.num_hidden_layers
    if ARGS.pdl_fused_mlp:
        expected_nodes -= globs.num_hidden_layers
    if len(nodes) != expected_nodes:
        raise RuntimeError(
            f"expected {expected_nodes} short-context kernel nodes, got "
            f"{len(nodes)}"
        )

    # Graph edges serialize the nodes, so all cross-op spin barriers are
    # redundant.  Pre-mark them ready at replay start; op-local CTA completion
    # remains guaranteed by each kernel node's completion edge.
    ready_barriers = torch.full_like(globs.barriers, 1_000_000)
    if ARGS.direct_tk_pdl_body == "13page_issue_r96_vtp56":
        # Opcode5 publishes one counter per 2048-element SiLU shard. Opcode6
        # reduction_block_idx 0..3 waits for the corresponding 128 stores.
        ready_barriers[:, 4, :4] = 0
    if ARGS.pdl_fused_mlp:
        ready_barriers[:, 9, :2] = 0
    if ARGS.elide_static_barrier_reset:
        if (
            ARGS.pdl_qkv_vready_attention
            or ARGS.pdl_pairwise_split_down
            or ARGS.pdl_fused_mlp
        ):
            raise ValueError(
                "static Bar reset elision is incompatible with kernels that "
                "publish fine-grained readiness"
            )
        globs.barriers.copy_(ready_barriers)
        torch.cuda.synchronize()

    if ARGS.trajectory_prompt_length is not None:
        if ARGS.elide_static_barrier_reset:
            raise ValueError(
                "trajectory mode requires a per-forward barrier reset"
            )
        run_trajectory(
            model=model,
            globs=globs,
            nodes=nodes,
            ready_barriers=ready_barriers,
            reference_globs=reference_globs,
            candidate_module=reference_module,
            reference_module=native_reference_module,
            simt_module=simt_module,
            implementation_label=implementation_label,
            selected_kernel_entries=selected_kernel_entries,
            binding_preflight_artifact=binding_preflight_artifact,
        )
        return

    def standalone_sequence() -> None:
        if not ARGS.elide_static_barrier_reset:
            globs.barriers.copy_(ready_barriers)
        globs.hidden_states.copy_(input_hidden)
        for _, node_globs, kernel in nodes:
            interpret_with_mk(node_globs, kernel)

    reset_model_caches(model)
    standalone_sequence()
    torch.cuda.synchronize()
    eager = {
        "logits": tensor_error(
            globs.logits, reference["logits"],
            rtol=ARGS.correctness_rtol, atol=ARGS.correctness_atol,
        ),
        "hidden": tensor_error(
            globs.hidden_states, reference["hidden"],
            rtol=ARGS.correctness_rtol, atol=ARGS.correctness_atol,
        ),
        "k": tensor_error(
            globs.k_cache[:, :, ARGS.position], reference["k"],
            rtol=ARGS.correctness_rtol, atol=ARGS.correctness_atol,
        ),
        "v": tensor_error(
            globs.v_cache[:, :, ARGS.position], reference["v"],
            rtol=ARGS.correctness_rtol, atol=ARGS.correctness_atol,
        ),
        "token": int(torch.argmax(globs.logits).item()),
        "k_by_layer": [
            tensor_error(
                globs.k_cache[layer, :, ARGS.position],
                reference["k"][layer],
                rtol=ARGS.correctness_rtol,
                atol=ARGS.correctness_atol,
            )
            for layer in range(globs.num_hidden_layers)
        ],
        "v_by_layer": [
            tensor_error(
                globs.v_cache[layer, :, ARGS.position],
                reference["v"][layer],
                rtol=ARGS.correctness_rtol,
                atol=ARGS.correctness_atol,
            )
            for layer in range(globs.num_hidden_layers)
        ],
    }
    eager["token_match"] = eager["token"] == reference["token"]
    eager["correct"] = bool(
        eager["token_match"]
        and eager["logits"]["correct"]
        and eager["hidden"]["correct"]
        and eager["k"]["correct"]
        and eager["v"]["correct"]
    )
    eager_outputs = {
        "logits": globs.logits.clone(),
        "hidden": globs.hidden_states.clone(),
        "k": globs.k_cache[:, :, ARGS.position].clone(),
        "v": globs.v_cache[:, :, ARGS.position].clone(),
    }

    graph_edge_diagnostics = None
    graph_priority_diagnostics = None
    graph_exec_flags = 0
    custom_graph_exec = None
    if ARGS.graph_edge_mode == "capture":
        if ARGS.graph_exec_mode != "torch" or ARGS.graph_node_priorities:
            raise ValueError(
                "custom graph instantiation requires an editable graph edge mode"
            )
        standalone_graph = baseline.capture_graph(standalone_sequence)
        standalone_replay = standalone_graph.replay
    else:
        raw_graph = capture_raw_cuda_graph(standalone_sequence, simt_module)
        graph_edge_period = (
            4 if ARGS.pdl_remove_context1_attention_node else 5
        )
        graph_edge_mask = (
            0x06
            if ARGS.pdl_remove_context1_attention_node
            else ARGS.launch_completion_edge_mask
        )
        graph_edge_diagnostics = {
            "before": simt_module.inspect_cuda_graph_edges(raw_graph),
            "edge_period": graph_edge_period,
            "effective_edge_mask": graph_edge_mask,
        }
        graph_edge_diagnostics["rewritten"] = (
            simt_module.rewrite_cuda_graph_kernel_edges(
                raw_graph,
                ARGS.graph_edge_mode == "launch-completion",
                graph_edge_mask,
                graph_edge_period,
            )
        )
        graph_edge_diagnostics["rewritten"] = int(
            graph_edge_diagnostics["rewritten"]
        )
        graph_edge_diagnostics["expected_rewritten"] = require_rewritten_edges(
            scope="fixed-position graph",
            observed=graph_edge_diagnostics["rewritten"],
            edge_count=len(nodes) - 1,
            edge_mask=graph_edge_mask,
            edge_period=graph_edge_period,
        )
        graph_edge_diagnostics["after"] = (
            simt_module.inspect_cuda_graph_edges(raw_graph)
        )
        if ARGS.graph_node_priorities:
            priority_pattern = [
                int(value)
                for value in ARGS.graph_node_priorities.split(",")
            ]
            graph_priority_diagnostics = (
                simt_module.set_cuda_graph_kernel_chain_priorities(
                    raw_graph, priority_pattern
                )
            )
            graph_exec_flags |= 0x8
        if ARGS.graph_exec_mode == "device":
            graph_exec_flags |= 0x4
        custom_graph_exec = simt_module.instantiate_cuda_graph_exec(
            raw_graph, graph_exec_flags
        )
        simt_module.destroy_cuda_graph(raw_graph)

        def standalone_replay() -> None:
            simt_module.launch_cuda_graph_exec(
                custom_graph_exec,
                torch.cuda.current_stream().cuda_stream,
            )

    standalone_replay()
    torch.cuda.synchronize()
    graph_correctness = {
        "logits": tensor_error(
            globs.logits, reference["logits"],
            rtol=ARGS.correctness_rtol, atol=ARGS.correctness_atol,
        ),
        "hidden": tensor_error(
            globs.hidden_states, reference["hidden"],
            rtol=ARGS.correctness_rtol, atol=ARGS.correctness_atol,
        ),
        "k": tensor_error(
            globs.k_cache[:, :, ARGS.position], reference["k"],
            rtol=ARGS.correctness_rtol, atol=ARGS.correctness_atol,
        ),
        "v": tensor_error(
            globs.v_cache[:, :, ARGS.position], reference["v"],
            rtol=ARGS.correctness_rtol, atol=ARGS.correctness_atol,
        ),
        "token": int(torch.argmax(globs.logits).item()),
        "k_by_layer": [
            tensor_error(
                globs.k_cache[layer, :, ARGS.position],
                reference["k"][layer],
                rtol=ARGS.correctness_rtol,
                atol=ARGS.correctness_atol,
            )
            for layer in range(globs.num_hidden_layers)
        ],
        "v_by_layer": [
            tensor_error(
                globs.v_cache[layer, :, ARGS.position],
                reference["v"][layer],
                rtol=ARGS.correctness_rtol,
                atol=ARGS.correctness_atol,
            )
            for layer in range(globs.num_hidden_layers)
        ],
    }
    graph_correctness["token_match"] = (
        graph_correctness["token"] == reference["token"]
    )
    graph_correctness["correct"] = bool(
        graph_correctness["token_match"]
        and graph_correctness["logits"]["correct"]
        and graph_correctness["hidden"]["correct"]
        and graph_correctness["k"]["correct"]
        and graph_correctness["v"]["correct"]
    )
    graph_outputs = {
        "logits": globs.logits.clone(),
        "hidden": globs.hidden_states.clone(),
        "k": globs.k_cache[:, :, ARGS.position].clone(),
        "v": globs.v_cache[:, :, ARGS.position].clone(),
    }
    replay_consistency = {
        "logits": tensor_error(
            globs.logits, eager_outputs["logits"], rtol=0.0, atol=0.0
        ),
        "hidden": tensor_error(
            globs.hidden_states, eager_outputs["hidden"], rtol=0.0, atol=0.0
        ),
        "k": tensor_error(
            globs.k_cache[:, :, ARGS.position], eager_outputs["k"],
            rtol=0.0, atol=0.0
        ),
        "v": tensor_error(
            globs.v_cache[:, :, ARGS.position], eager_outputs["v"],
            rtol=0.0, atol=0.0
        ),
    }
    replay_consistency["bit_exact"] = all(
        value["correct"] for value in replay_consistency.values()
    )
    # Matched one-node persistent control.  Its own barriers must start at
    # zero because the persistent kernel uses them for cross-op scheduling.
    reference_barrier_zeros = torch.zeros_like(reference_globs.barriers)

    def persistent_sequence() -> None:
        reference_globs.barriers.copy_(reference_barrier_zeros)
        reference_globs.hidden_states.copy_(input_hidden)
        interpret_with_mk(reference_globs, native_reference_module.mk_llama)

    persistent_graph = baseline.capture_graph(persistent_sequence)
    persistent_graph.replay()
    torch.cuda.synchronize()
    persistent_graph_token = int(torch.argmax(reference_globs.logits).item())
    persistent_graph_outputs = {
        "logits": reference_globs.logits,
        "hidden": reference_globs.hidden_states,
        "k": reference_globs.k_cache[:, :, ARGS.position],
        "v": reference_globs.v_cache[:, :, ARGS.position],
    }
    eager_vs_persistent_replay = {
        name: tensor_error(
            eager_outputs[name],
            persistent_graph_outputs[name],
            rtol=ARGS.correctness_rtol,
            atol=ARGS.correctness_atol,
        )
        for name in ("logits", "hidden", "k", "v")
    }
    graph_vs_persistent_replay = {
        name: tensor_error(
            graph_outputs[name],
            persistent_graph_outputs[name],
            rtol=ARGS.correctness_rtol,
            atol=ARGS.correctness_atol,
        )
        for name in ("logits", "hidden", "k", "v")
    }

    correctness_diagnostics = {
        "elementwise_allclose_rtol_atol": {
            "eager_vs_initial_persistent": eager["correct"],
            "graph_vs_initial_persistent": graph_correctness["correct"],
        },
        "eager_graph_bit_exact": replay_consistency["bit_exact"],
        "graph_matches_persistent_replay_token": (
            graph_correctness["token"] == persistent_graph_token
        ),
        "persistent_initial_vs_replay_token": (
            reference["token"] == persistent_graph_token
        ),
    }
    binding_contract_artifact = (
        load_binding_contract_artifact()
        if ARGS.enforce_correctness_contract
        else None
    )
    correctness_contract = None
    correctness_contract_artifact = None
    eager_vs_persistent_tensor_envelope = None
    graph_vs_persistent_tensor_envelope = None
    eager_graph_tensor_envelope = None
    if ARGS.enforce_correctness_contract:
        correctness_contract, correctness_contract_artifact = (
            load_correctness_contract()
        )
        tensor_limits = correctness_contract[
            "persistent_replay_tensor_error_limits"
        ]
        eager_vs_persistent_tensor_envelope = tensor_envelope_check(
            eager_vs_persistent_replay, tensor_limits
        )
        graph_vs_persistent_tensor_envelope = tensor_envelope_check(
            graph_vs_persistent_replay, tensor_limits
        )
        eager_graph_tensor_envelope = tensor_envelope_check(
            replay_consistency,
            correctness_contract["eager_graph_tensor_error_limits"],
        )
        correctness_diagnostics[
            "eager_vs_persistent_tensor_envelope"
        ] = eager_vs_persistent_tensor_envelope
        correctness_diagnostics[
            "graph_vs_persistent_tensor_envelope"
        ] = graph_vs_persistent_tensor_envelope
        correctness_gate_checks = {
            "eager_vs_graph_tensor_envelope": (
                eager_graph_tensor_envelope["passed"]
            ),
            "graph_matches_eager_token": (
                graph_correctness["token"] == eager["token"]
            ),
            "graph_edge_rewrite": (
                graph_edge_diagnostics is not None
                and graph_edge_diagnostics["rewritten"]
                == graph_edge_diagnostics["expected_rewritten"]
            ),
        }
        correctness_diagnostics = {
            name: correctness_diagnostics[name]
            for name in correctness_contract["diagnostic_only"]
        }
        required_checks = correctness_contract["required_checks"]
    else:
        correctness_gate_checks = {
            "eager_matches_initial_persistent": eager["correct"],
            "graph_matches_initial_persistent": graph_correctness["correct"],
            "graph_matches_eager_token": (
                graph_correctness["token"] == eager["token"]
            ),
            "persistent_replay_matches_initial_token": (
                persistent_graph_token == reference["token"]
            ),
        }
        if ARGS.graph_edge_mode != "capture":
            correctness_gate_checks["graph_edge_rewrite"] = bool(
                graph_edge_diagnostics is not None
                and graph_edge_diagnostics["rewritten"]
                == graph_edge_diagnostics["expected_rewritten"]
            )
        required_checks = tuple(correctness_gate_checks)
    correctness_gate_passed = all(
        correctness_gate_checks[name] for name in required_checks
    )
    if not correctness_gate_passed and not ARGS.allow_correctness_mismatch:
        raise RuntimeError(
            f"fixed-position correctness gate failed: {correctness_gate_checks}"
        )

    if ARGS.pdl_full_chain or ARGS.pdl_pairwise_winners:
        pdl_edge_names = [
            "opcode1->opcode2",
            "opcode2->opcode4",
            "opcode4->opcode5",
            "opcode5->opcode6",
            "opcode6->next-layer-opcode1",
            "final-opcode6->opcode7",
        ]
    else:
        pdl_edge_names = []
        if ARGS.pdl_op1_op2:
            pdl_edge_names.append("opcode1->opcode2")
        if ARGS.pdl_op2_op4:
            pdl_edge_names.append("opcode2->opcode4")
        if ARGS.pdl_op4_op5:
            pdl_edge_names.append("opcode4->opcode5")
        if ARGS.pdl_op5_op6:
            pdl_edge_names.append("opcode5->opcode6")
        if PDL_OP6_QKV:
            pdl_edge_names.append("opcode6->next-layer-opcode1")
        if PDL_OP6_LMHEAD:
            pdl_edge_names.append("final-opcode6->opcode7")
    cross_op_dependency = (
        "programmatic CUDA Graph edges for " + ", ".join(pdl_edge_names)
        + "; graph edge mode " + ARGS.graph_edge_mode
        + "; standard edges otherwise; device spin barriers pre-marked ready"
        if pdl_edge_names
        else "CUDA Graph node edges; device spin barriers pre-marked ready"
    )

    if ARGS.profile_only:
        if ARGS.trace_output is None:
            raise ValueError("--profile-only requires --trace-output")
        for _ in range(10):
            standalone_replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            standalone_replay()
            torch.cuda.synchronize()
        ARGS.trace_output.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(ARGS.trace_output))
        profile_result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "candidate": (
                f"position{ARGS.position}_16layer_"
                f"{implementation_label}_opcode_cuda_graph"
            ),
            "implementation": implementation_label,
            "simt_opcodes": sorted(simt_opcodes),
            "simt_entry6": ARGS.simt_entry6,
            "simt_entry4": ARGS.simt_entry4,
            "pdl_op5_entry": ARGS.pdl_op5_entry,
            "pdl_op4_entry": ARGS.pdl_op4_entry,
            "pdl_op5_chain_entry": ARGS.pdl_op5_chain_entry,
            "pdl_op5_op6": ARGS.pdl_op5_op6,
            "pdl_op4_op5": ARGS.pdl_op4_op5,
            "pdl_op1_op2": ARGS.pdl_op1_op2,
            "pdl_op2_op4": ARGS.pdl_op2_op4,
            "pdl_op1_entry": ARGS.pdl_op1_entry,
            "pdl_op2_entry": ARGS.pdl_op2_entry,
            "pdl_op4_wait_entry": ARGS.pdl_op4_wait_entry,
            "pdl_op6_next": ARGS.pdl_op6_next,
            "pdl_op6_qkv": PDL_OP6_QKV,
            "pdl_op6_lmhead": PDL_OP6_LMHEAD,
            "pdl_full_chain": ARGS.pdl_full_chain,
            "pdl_pairwise_winners": ARGS.pdl_pairwise_winners,
            "graph_edge_mode": ARGS.graph_edge_mode,
            "graph_exec_mode": ARGS.graph_exec_mode,
            "graph_exec_flags": graph_exec_flags,
            "graph_node_priorities": ARGS.graph_node_priorities,
            "graph_priority_diagnostics": graph_priority_diagnostics,
            "launch_completion_edge_mask": (
                ARGS.launch_completion_edge_mask
            ),
            "launch_completion_prune_triggers": (
                ARGS.launch_completion_prune_triggers
            ),
            "launch_completion_plain_launch": (
                ARGS.launch_completion_plain_launch
            ),
            "launch_completion_op4_prefetch_bytes": (
                ARGS.launch_completion_op4_prefetch_bytes
            ),
            "launch_completion_op5_prefetch_bytes": (
                ARGS.launch_completion_op5_prefetch_bytes
            ),
            "graph_edge_diagnostics": graph_edge_diagnostics,
            "pdl_pairwise_split_down": ARGS.pdl_pairwise_split_down,
            "pdl_pairwise_split_down_producer": (
                ARGS.pdl_pairwise_split_down_producer
            ),
            "pdl_pairwise_op6_smem_prefix_cols": (
                ARGS.pdl_pairwise_op6_smem_prefix_cols
            ),
            "pdl_pairwise_op6_prefetch_bytes": (
                ARGS.pdl_pairwise_op6_prefetch_bytes
            ),
            "pdl_qkv_vready_attention": ARGS.pdl_qkv_vready_attention,
            "pdl_q_ready_attention": ARGS.pdl_q_ready_attention,
            "pdl_fuse_context1_attention": (
                ARGS.pdl_fuse_context1_attention
            ),
            "pdl_remove_context1_attention_node": (
                ARGS.pdl_remove_context1_attention_node
            ),
            "elide_static_barrier_reset": ARGS.elide_static_barrier_reset,
            "pdl_context1_kv_only": ARGS.pdl_context1_kv_only,
            "pdl_context1_kv_warps": ARGS.pdl_context1_kv_warps,
            "pdl_pairwise_op4_mode": ARGS.pdl_pairwise_op4_mode,
            "pdl_pairwise_op5_prefetch_bytes": (
                ARGS.pdl_pairwise_op5_prefetch_bytes
            ),
            "pdl_pairwise_op5_prefetch_matrix": (
                ARGS.pdl_pairwise_op5_prefetch_matrix
            ),
            "pdl_pairwise_op5_v4": ARGS.pdl_pairwise_op5_v4,
            "pdl_pairwise_op5_smem_mode": (
                ARGS.pdl_pairwise_op5_smem_mode
            ),
            "pdl_pairwise_op5_reg_prefix_cols": (
                ARGS.pdl_pairwise_op5_reg_prefix_cols
            ),
            "pdl_pairwise_op5_demand_bytes": (
                ARGS.pdl_pairwise_op5_demand_bytes
            ),
            "pdl_pairwise_qkv_prefetch_bytes": (
                ARGS.pdl_pairwise_qkv_prefetch_bytes
            ),
            "pdl_pairwise_qkv_trigger_mode": (
                ARGS.pdl_pairwise_qkv_trigger_mode
            ),
            "pdl_pairwise_lm_mode": ARGS.pdl_pairwise_lm_mode,
            "pdl_pairwise_lm_trigger_mode": (
                ARGS.pdl_pairwise_lm_trigger_mode
            ),
            "pdl_pairwise_lm_prefetch_bytes": (
                ARGS.pdl_pairwise_lm_prefetch_bytes
            ),
            "pdl_prefetch_policies": {
                edge: getattr(ARGS, f"pdl_prefetch_policy_{edge}")
                for edge in ("qkv", "op4", "op5", "op6", "lm")
            },
            "selected_kernel_entries": selected_kernel_entries,
            "pdl_op1_prefetch_bytes": ARGS.pdl_op1_prefetch_bytes,
            "pdl_op6_entry": ARGS.pdl_op6_entry,
            "pdl_op6_chain_entry": ARGS.pdl_op6_chain_entry,
            "pdl_op7_entry": ARGS.pdl_op7_entry,
            "gpu": torch.cuda.get_device_name(ARGS.device),
            "kernel_node_count": len(nodes),
            "node_order": [label for label, _, _ in nodes],
            "reference_token": reference["token"],
            "graph_correctness": graph_correctness,
            "eager_graph_replay_consistency": replay_consistency,
            "timing_record_enabled": False,
            "timeline_collection_enabled": True,
            "performance_samples_collected": False,
            "trace_output": str(ARGS.trace_output),
        }
        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        ARGS.output.write_text(
            json.dumps(profile_result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(profile_result, sort_keys=True))
        return

    # No native timing, Chrome trace, CUPTI, or Nsight collection is enabled
    # in these performance loops.
    standalone_eager_samples = baseline.time_samples(
        standalone_sequence, ARGS.warmup, ARGS.iterations
    )
    standalone_graph_samples = baseline.time_samples(
        standalone_replay, ARGS.warmup, ARGS.iterations
    )
    persistent_eager_samples = baseline.time_samples(
        persistent_sequence, ARGS.warmup, ARGS.iterations
    )
    persistent_graph_samples = baseline.time_samples(
        persistent_graph.replay, ARGS.warmup, ARGS.iterations
    )

    opcode_counts = Counter(
        int(label.split("opcode", 1)[1].split("_", 1)[0])
        for label, _, _ in nodes
    )
    result = {
        "schema": "hazy-fixed-position-correctness-v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidate": (
            f"position{ARGS.position}_16layer_"
            f"{implementation_label}_opcode_cuda_graph"
        ),
        "implementation": implementation_label,
        "simt_opcodes": sorted(simt_opcodes),
        "simt_entry6": ARGS.simt_entry6,
        "simt_entry4": ARGS.simt_entry4,
        "pdl_op5_entry": ARGS.pdl_op5_entry,
        "pdl_op4_entry": ARGS.pdl_op4_entry,
        "pdl_op5_chain_entry": ARGS.pdl_op5_chain_entry,
        "pdl_op5_op6": ARGS.pdl_op5_op6,
        "pdl_op4_op5": ARGS.pdl_op4_op5,
        "pdl_op1_op2": ARGS.pdl_op1_op2,
        "pdl_op2_op4": ARGS.pdl_op2_op4,
        "pdl_op1_entry": ARGS.pdl_op1_entry,
        "pdl_op2_entry": ARGS.pdl_op2_entry,
        "pdl_op4_wait_entry": ARGS.pdl_op4_wait_entry,
        "pdl_op6_next": ARGS.pdl_op6_next,
        "pdl_op6_qkv": PDL_OP6_QKV,
        "pdl_op6_lmhead": PDL_OP6_LMHEAD,
        "pdl_full_chain": ARGS.pdl_full_chain,
        "pdl_pairwise_winners": ARGS.pdl_pairwise_winners,
        "graph_edge_mode": ARGS.graph_edge_mode,
        "graph_exec_mode": ARGS.graph_exec_mode,
        "graph_exec_flags": graph_exec_flags,
        "graph_node_priorities": ARGS.graph_node_priorities,
        "graph_priority_diagnostics": graph_priority_diagnostics,
        "launch_completion_edge_mask": ARGS.launch_completion_edge_mask,
        "launch_completion_prune_triggers": (
            ARGS.launch_completion_prune_triggers
        ),
        "launch_completion_plain_launch": (
            ARGS.launch_completion_plain_launch
        ),
        "launch_completion_op4_prefetch_bytes": (
            ARGS.launch_completion_op4_prefetch_bytes
        ),
        "launch_completion_op5_prefetch_bytes": (
            ARGS.launch_completion_op5_prefetch_bytes
        ),
        "graph_edge_diagnostics": graph_edge_diagnostics,
        "pdl_pairwise_split_down": ARGS.pdl_pairwise_split_down,
        "pdl_pairwise_split_down_producer": (
            ARGS.pdl_pairwise_split_down_producer
        ),
        "pdl_pairwise_op6_smem_prefix_cols": (
            ARGS.pdl_pairwise_op6_smem_prefix_cols
        ),
        "pdl_pairwise_op6_prefetch_bytes": (
            ARGS.pdl_pairwise_op6_prefetch_bytes
        ),
        "pdl_qkv_vready_attention": ARGS.pdl_qkv_vready_attention,
        "pdl_q_ready_attention": ARGS.pdl_q_ready_attention,
        "pdl_fuse_context1_attention": (
            ARGS.pdl_fuse_context1_attention
        ),
        "pdl_remove_context1_attention_node": (
            ARGS.pdl_remove_context1_attention_node
        ),
        "elide_static_barrier_reset": ARGS.elide_static_barrier_reset,
        "pdl_context1_kv_only": ARGS.pdl_context1_kv_only,
        "pdl_context1_kv_warps": ARGS.pdl_context1_kv_warps,
        "pdl_pairwise_op4_mode": ARGS.pdl_pairwise_op4_mode,
        "pdl_pairwise_op5_prefetch_bytes": (
            ARGS.pdl_pairwise_op5_prefetch_bytes
        ),
        "pdl_pairwise_op5_prefetch_matrix": (
            ARGS.pdl_pairwise_op5_prefetch_matrix
        ),
        "pdl_pairwise_op5_v4": ARGS.pdl_pairwise_op5_v4,
        "pdl_pairwise_op5_smem_mode": ARGS.pdl_pairwise_op5_smem_mode,
        "pdl_pairwise_op5_reg_prefix_cols": (
            ARGS.pdl_pairwise_op5_reg_prefix_cols
        ),
        "pdl_pairwise_op5_demand_bytes": (
            ARGS.pdl_pairwise_op5_demand_bytes
        ),
        "pdl_pairwise_qkv_prefetch_bytes": (
            ARGS.pdl_pairwise_qkv_prefetch_bytes
        ),
        "pdl_pairwise_qkv_trigger_mode": (
            ARGS.pdl_pairwise_qkv_trigger_mode
        ),
        "pdl_pairwise_lm_mode": ARGS.pdl_pairwise_lm_mode,
        "pdl_pairwise_lm_trigger_mode": (
            ARGS.pdl_pairwise_lm_trigger_mode
        ),
        "pdl_pairwise_lm_prefetch_bytes": (
            ARGS.pdl_pairwise_lm_prefetch_bytes
        ),
        "pdl_prefetch_policies": {
            edge: getattr(ARGS, f"pdl_prefetch_policy_{edge}")
            for edge in ("qkv", "op4", "op5", "op6", "lm")
        },
        "selected_kernel_entries": selected_kernel_entries,
        "pdl_op1_prefetch_bytes": ARGS.pdl_op1_prefetch_bytes,
        "pdl_op6_entry": ARGS.pdl_op6_entry,
        "pdl_op6_chain_entry": ARGS.pdl_op6_chain_entry,
        "pdl_op7_entry": ARGS.pdl_op7_entry,
        "direct_tk_pdl_body": ARGS.direct_tk_pdl_body,
        "gpu": torch.cuda.get_device_name(ARGS.device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "seed": ARGS.seed,
        "context": ARGS.position + 1,
        "position": ARGS.position,
        "fixed_input_token": ARGS.fixed_input_token,
        "model_path": str(ARGS.model_path.resolve()) if ARGS.model_path else None,
        "model_safetensors_sha256": ARGS.model_safetensors_sha256,
        "binding_contract": binding_contract_artifact,
        "binding_preflight": binding_preflight_artifact,
        "enforce_correctness_contract": (
            ARGS.enforce_correctness_contract
        ),
        "module_provenance": {
            "candidate": module_provenance(reference_module),
            "native_reference": module_provenance(native_reference_module),
            "graph_helper": module_provenance(simt_module),
        },
        "native_reference_mk": (
            str(ARGS.native_reference_mk.resolve())
            if ARGS.native_reference_mk else None
        ),
        "allow_correctness_mismatch": ARGS.allow_correctness_mismatch,
        "timing_record_enabled": False,
        "timeline_collection_enabled": False,
        "warmup": ARGS.warmup,
        "trials": ARGS.iterations,
        "kernel_node_count": len(nodes),
        "opcode_node_counts": {
            str(opcode): count for opcode, count in sorted(opcode_counts.items())
        },
        "opcode3_scheduled": False,
        "instruction_storage": "one immutable tensor per graph kernel node",
        "cross_op_dependency": cross_op_dependency,
        "reference_token": reference["token"],
        "eager_correctness": eager,
        "graph_correctness": graph_correctness,
        "eager_graph_replay_consistency": replay_consistency,
        "persistent_graph_token": persistent_graph_token,
        "persistent_initial_token": reference["token"],
        "eager_vs_persistent_replay": eager_vs_persistent_replay,
        "graph_vs_persistent_replay": graph_vs_persistent_replay,
        "correctness_contract": correctness_contract_artifact,
        "eager_vs_persistent_tensor_envelope": (
            eager_vs_persistent_tensor_envelope
        ),
        "graph_vs_persistent_tensor_envelope": (
            graph_vs_persistent_tensor_envelope
        ),
        "eager_graph_tensor_envelope": eager_graph_tensor_envelope,
        "correctness_diagnostics": correctness_diagnostics,
        "correctness_gate_checks": correctness_gate_checks,
        "correctness_gate_passed": correctness_gate_passed,
        "standalone_eager_sequence": summarize(standalone_eager_samples),
        "standalone_cuda_graph": summarize(standalone_graph_samples),
        "persistent_eager_sequence": summarize(persistent_eager_samples),
        "persistent_cuda_graph": summarize(persistent_graph_samples),
    }
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
