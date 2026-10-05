"""Official vLLM torch references for the QSA segment — copied VERBATIM.

Source: `vllm/tests/models/qwen4_exp/test_qsa_reference.py:93-234`
(/workspace/vllm main `8a2364605c`), the four pure-torch ground-truth helpers the
vLLM authors use to validate their Triton QSA kernels:

    _qsa_mqa_paged_reference               :93-105   indexer logits
    _qsa_relative_topk_reference           :108-125  block top-k
    _expand_qsa_indices_reference           :128-164  expand + causal tail
    _qsa_select_paged_reference             :167-195  score -> top-k
    _qsa_sparse_paged_attention_reference   :198-234  sparse attention

Why this file exists: the QSA segment is the only part of the reference with **no
runnable official implementation** (every numerical stage upstream is Triton or a
CUDA C++ op; `grep -rn forward_native vllm/models/qwen4_exp/` finds nothing, and
`test_qsa_reference.py` cannot be imported here because its own header imports
`vllm.*`). Round-1 review flagged that the QSA segment's credibility therefore
rested entirely on line-anchored traceability. These helpers are the official
authors' own torch semantics, so running them against `ref/qsa.py` gives that
segment an executable second implementation (`selfcheck.py` section K).

The bodies below are a character-for-character copy -- nothing was edited, and
`QSA_REFERENCE_0f5c5c7c115983e3b533e3d0a4b86c3d7f94bb9baa721ce5dcf8254ec982d045256` pins the extraction (`selfcheck.py` K0 re-checks it
against the source file, so upstream drift is detected rather than assumed away).

Known, DOCUMENTED differences from the kernels (both sides of each are asserted
in section K, so neither is left implicit):

  * `_qsa_mqa_paged_reference` divides the logits by `sqrt(head_dim)`
    (:103), while the Triton scoring kernel does **not**
    (`nvidia/ops/qsa_indexer.py:94-100`). A positive constant cannot change the
    top-k ORDER, so only the raw logits differ, by exactly that factor.
  * `_expand_qsa_indices_reference` compacts `-1` padding to the end with a stable
    argsort and returns `output_width` columns with **no** trailing count column,
    whereas the kernel keeps positions and adds the count
    (`nvidia/ops/qsa_indexer.py:268-276`). The SET of selected tokens is the same.
"""

from __future__ import annotations

import math

import torch

# sha256 of the extracted source lines (test_qsa_reference.py:93-234)
QSA_REFERENCE_BODY_SHA256 = "0f5c5c7c115983e3b533e3d0a4b86c3d7f94bb9baa721ce5dcf8254ec982d045"


def _qsa_mqa_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    visible_lengths: torch.Tensor,
) -> torch.Tensor:
    pages = page_table.index_select(0, token_to_req.long()).long()
    keys = k_cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(positions >= visible_lengths.unsqueeze(1), -torch.inf)


def _qsa_relative_topk_reference(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    output = torch.full(
        (logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device
    )
    for row in range(logits.shape[0]):
        start = int(row_starts[row].item())
        length = int((row_ends[row] - row_starts[row]).item())
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def _expand_qsa_indices_reference(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    rows = block_indices.shape[0]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    offsets = torch.arange(compress_ratio, device=block_indices.device)
    blocks = block_indices.long()
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=block_indices.device)
    visible_tokens = query_positions + 1
    tail_start = visible_tokens // compress_ratio * compress_ratio
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_count = (visible_tokens - tail_start).unsqueeze(1)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(output_width, device=result.device).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + output_width)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


def _qsa_select_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
) -> torch.Tensor:
    row_sequence_lengths = sequence_lengths.index_select(0, token_to_req.long())
    visible_blocks = torch.minimum(
        (query_positions + 1) // compress_ratio,
        row_sequence_lengths // compress_ratio,
    ).to(torch.int32)
    logits = _qsa_mqa_paged_reference(
        q,
        k_cache,
        page_table,
        token_to_req,
        visible_blocks,
    )
    starts = torch.zeros_like(visible_blocks)
    return _qsa_relative_topk_reference(
        logits,
        starts,
        visible_blocks,
        token_topk // compress_ratio,
    )


def _qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Dense reference for QSA sparse paged attention.

    Mirrors the kernel's dequant: fp8-e4m3 K/V caches are dequantized with the
    per-tensor k_scale/v_scale host floats; bf16 caches use unit scales.
    """
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            continue
        request = token_to_req[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = (k_cache[pages, offsets].float() * k_scale).repeat_interleave(
            repeats, dim=1
        )
        values = (v_cache[pages, offsets].float() * v_scale).repeat_interleave(
            repeats, dim=1
        )
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys)
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities, values).to(q.dtype)
    return output
