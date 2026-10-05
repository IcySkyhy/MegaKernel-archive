"""QSA ("full_attention") segment reference — Qwen4ExpQSAAttention + indexer.

Scope: what `Qwen4ExpDecoderLayer.forward` calls as
`self.self_attn(hidden_states=block_input, positions=positions)`
(vllm/models/qwen4_exp/nvidia/model.py:318-322). Input `[T, 2560]` bf16, output
`[T, 2560]` bf16.

**This segment has NO eager/`forward_native` path in vLLM** — every numerical
stage is Triton or a CUDA C++ op (grep `forward_native` under
vllm/models/qwen4_exp/ -> no hits). The reference below is therefore a
re-implementation of the *kernel math*, anchored line by line, plus the torch
references that vLLM's own test-suite uses as ground truth:

  vllm/tests/models/qwen4_exp/test_qsa_reference.py:93-105   indexer logits
      (note :103 divides by sqrt(128); the kernel does NOT -- see DEVIATIONS)
  vllm/tests/models/qwen4_exp/test_qsa_reference.py:108-125   block top-k
  vllm/tests/models/qwen4_exp/test_qsa_reference.py:128-164   expand + tail
  vllm/tests/models/qwen4_exp/test_qsa_reference.py:198-234   sparse attention
  vllm/tests/models/qwen4_exp/test_qsa_reference.py:1077      gate applied outside

Anchors (base = /workspace/vllm main `8a2364605c`):
  forward sequence     nvidia/qsa.py:500-526, :444-498, :182-249
  qk norm+rope (main)  vllm/model_executor/layers/fused_qk_norm_rope.py:80-152
                       (var = sum(x*x)/head_dim :68-69 ; w = weight + norm_beta :70 ;
                        round-trip bf16 :73 ; NeoX split-half rope :125-126 ;
                        gate copied raw :132-135)
  gate layout          model_executor/models/qwen3_next.py:427-435 + fused_qk_norm_rope.py:156
  indexer q path       nvidia/ops/qsa_pre_indexer.py:68-77, :32-53
  indexer q (unfused)  nvidia/indexer_qsa.py:323-330
  compressed key write nvidia/ops/qsa_pre_indexer.py:204-343
                       (pool + bf16 round :266-269 ; first_position :271 ;
                        norm+rope :323-333 ; store :335-343)
  pooling              nvidia/ops/qsa.py:401-433
  indexer logits       nvidia/ops/qsa_indexer.py:19-107 (relu+sum :94-100 decode,
                       :206-208 prefill ; store :101-107 / :214-218)
  block top-k          nvidia/ops/qsa_indexer.py:471-499
  expand + count col   nvidia/ops/qsa_indexer.py:221-276 (count column :268-276)
  visible_blocks       common/qsa_cache.py:253-275
  rope cache           model_executor/layers/rotary_embedding/base.py:80-102 (cast to model dtype)
  sparse attention     nvidia/ops/qsa.py:594-758, :32-233, epilogue :181-214
  o_proj               nvidia/qsa.py:525

NOTE on the `file:line` anchors below: the line number is a LOOKUP HINT against
the pinned base commit shown above; line numbers drift when upstream moves.
The stable reference is the SYMBOL. `python3 tools/check_anchors.py --verbose`
prints, for every anchor, which file it resolves to and which symbol it lands in
(it also fails if an anchor is ambiguous or out of range), and
`selfcheck.py` section I runs that check. Project rule 2026-09-26.

DEVIATIONS from the kernels (see README "Ver QSA 证明什么 / 不证明什么"):

  D1. Storage is dense per sequence instead of paged. For a single request with no
      speculative tokens this is an exact re-indexing of the two paged caches:
      `raw_key_cache` is a `CircularBufferSpec(block_size=4, ...)` ring
      (common/qsa_cache.py:838-856) and `compressed_key_cache` is
      `MLAAttentionSpec(tokens_per_state=4)` (1 row / 4 tokens). We keep the full
      raw-key history, so cross-chunk group completion needs no ring.
  D2. The compressed-row write is computed lazily from the raw keys of its group.
      The kernel writes it as soon as the group completes (qsa_pre_indexer.py:335-343),
      so for a given position the value is identical; only the *timing* differs.
  D3. `visible_blocks` is computed with the formula from the always-visible case;
      the "row" axis is a single request.
  D4. The indexer test reference divides the logits by `sqrt(128)`
      (test_qsa_reference.py:103); the Triton kernel does not
      (qsa_indexer.py:94-100). A positive constant cannot change the top-k ORDER,
      so only the raw `qsa.index_logits` dump differs. This module matches the
      kernel and reports the ratio.
  D5. RoPE is evaluated in fp32 with a bf16 cos/sin cache widened to fp32; the
      Triton kernels may keep some intermediates in bf16. Rounding points that ARE
      preserved: normed value -> bf16 before rotation, cos/sin cache stored bf16,
      q/k stored bf16. Expected agreement: <= 1 ulp class, not bit-exact.
  D6. MRoPE degenerates to plain RoPE for text: `get_mrope_input_positions`
      returns three identical rows (nvidia/model.py:846-852), so the T/H/W
      pair-to-axis assignment cannot change any number.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class _QSAConfig:
    def __init__(self, text_config: dict):
        self.hidden = text_config["hidden_size"]
        self.num_heads = text_config["num_attention_heads"]  # 24
        self.num_kv_heads = text_config["num_key_value_heads"]  # 2
        self.head_dim = text_config["head_dim"]  # 256
        self.eps = text_config["rms_norm_eps"]  # 1e-6
        prf = text_config.get("partial_rotary_factor", 0.25)
        self.rotary_dim = int(self.head_dim * prf)  # 64
        self.rope_theta = 1.0e7  # get_rope default for this family
        self.index_n_heads = text_config["indexer_n_heads"]  # 4
        self.index_kv_heads = text_config["indexer_kv_heads"]  # 1
        self.index_head_dim = text_config["indexer_head_dim"]  # 128
        self.indexer_budget = text_config["indexer_budget"]  # 2048 (tokens)
        self.compress_ratio = text_config["indexer_compress_ratio"]  # 4
        self.block_topk = self.indexer_budget // self.compress_ratio  # 512 rows
        self.output_width = self.indexer_budget + self.compress_ratio - 1  # 2051
        self.packed_width = self.output_width + 1  # 2052 (trailing count)
        self.q_size = self.num_heads * self.head_dim  # 6144
        self.kv_size = self.num_kv_heads * self.head_dim  # 512
        self.index_q_dim = self.index_n_heads * self.index_head_dim  # 512
        self.index_proj_dim = self.q_size + self.kv_size * 2  # 13312


def _rope_cache(rotary_dim: int, theta: float, max_pos: int, dtype):
    """RotaryEmbeddingBase._compute_cos_sin_cache (rotary_embedding/base.py:80-102)."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )  # [rotary_dim/2]
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_pos, rotary_dim/2]
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rotary_dim]
    return cache.to(dtype)  # base.py:59-63 casts the cache to the model dtype


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int):
    """NeoX split-half partial RoPE.

    x: [..., H, D]. For each row the first `rotary_dim` dims rotate as
    (x[:r/2], x[r/2:r]) -> (a*cos - b*sin, b*cos + a*sin); dims >= rotary_dim are
    untouched. Matches fused_qk_norm_rope.py:125-126 and
    ops/qsa_pre_indexer.py:76 (`out0 = r0 * cos - r1 * sin`) and
    ApplyRotaryEmb.forward_static (rotary_embedding/common.py:138-171).
    """
    orig = x.dtype
    half = rotary_dim // 2
    rot = x[..., :rotary_dim].float()
    a = rot[..., :half]
    b = rot[..., half:]
    # cos/sin must already be broadcastable against x[..., :half], i.e. shape
    # [..., 1, half] when x has a head axis, [..., half] when it does not.
    cosf = cos.float()
    sinf = sin.float()
    out = torch.empty_like(rot)
    out[..., :half] = a * cosf - b * sinf
    out[..., half:] = b * cosf + a * sinf
    res = torch.cat([out, x[..., rotary_dim:].float()], dim=-1)
    return res.to(orig)


def _gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """GemmaRMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + w)."""
    xf = x.float()
    rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * rms * (1.0 + weight.float())).to(x.dtype)


class QSA:
    """Reference QSA segment with dense per-sequence caches."""

    def __init__(self, weights, text_config: dict, max_seq_len: int = 8192):
        self.cfg = _QSAConfig(text_config)
        c = self.cfg
        self.weights = weights
        self.w_qkv = weights.qkv_proj.to(torch.bfloat16)  # [13312, 2560]
        self.o_proj = weights.o_proj_weight.to(torch.bfloat16)  # [2560, 6144]
        self.q_norm_w = weights.q_norm_weight.to(torch.bfloat16)  # [256]
        self.k_norm_w = weights.k_norm_weight.to(torch.bfloat16)
        self.index_proj = weights.index_qk_proj.to(torch.bfloat16)  # [640, 2560]
        self.index_q_norm_w = weights.index_q_norm_weight.to(torch.bfloat16)  # [128]
        self.index_k_norm_w = weights.index_k_norm_weight.to(torch.bfloat16)
        self.max_seq_len = max_seq_len
        cache_dtype = torch.bfloat16  # indexer_kv_dtype="auto" -> bf16
        self.cos_sin = _rope_cache(c.rotary_dim, c.rope_theta, max_seq_len, cache_dtype)
        self.seq_len = 0
        self.k_cache = torch.zeros(max_seq_len, c.num_kv_heads, c.head_dim, dtype=cache_dtype)
        self.v_cache = torch.zeros(max_seq_len, c.num_kv_heads, c.head_dim, dtype=cache_dtype)
        self.raw_key = torch.zeros(max_seq_len, c.index_head_dim, dtype=cache_dtype)
        self.compressed = torch.zeros(max_seq_len // c.compress_ratio, c.index_head_dim,
                                      dtype=cache_dtype)
        self.compressed_valid = 0

    # ---------------------------------------------------------------- main qkv
    def _project_qkv_gate(self, hidden: torch.Tensor, positions: torch.Tensor):
        """nvidia/qsa.py:505-511 + qwen3_next.py:427-435 + fused_qk_norm_rope.py."""
        c = self.cfg
        qkv = hidden.to(self.w_qkv.dtype) @ self.w_qkv.t()  # [T, 13312]
        q_gate, k_raw, v_raw = qkv.split(
            [c.q_size * 2, c.kv_size, c.kv_size], dim=-1
        )  # fused_qk_norm_rope.py:156 per head [q|gate]
        t = qkv.shape[0]
        q_gate = q_gate.view(t, c.num_heads, 2 * c.head_dim)
        q, gate = torch.chunk(q_gate, 2, dim=-1)  # qwen3_next.py:432-435
        # norm (head_dim-wide Gemma, weight+1) then partial NeoX rope
        qn = _gemma_rmsnorm(q, self.q_norm_w, c.eps)  # fused_qk_norm_rope.py:68-73
        kn = _gemma_rmsnorm(
            k_raw.view(t, c.num_kv_heads, c.head_dim), self.k_norm_w, c.eps
        )
        cos = self.cos_sin[positions][..., : c.rotary_dim // 2]
        sin = self.cos_sin[positions][..., c.rotary_dim // 2 :]
        qn = _apply_rope(qn, cos[:, None, :], sin[:, None, :], c.rotary_dim)
        kn = _apply_rope(kn, cos[:, None, :], sin[:, None, :], c.rotary_dim)
        gate = gate.reshape(t, c.num_heads, c.head_dim)  # copied verbatim, no norm
        return qn, kn, v_raw.view(t, c.num_kv_heads, c.head_dim), gate

    # ------------------------------------------------------------- indexer
    def _indexer(self, hidden: torch.Tensor, positions: torch.Tensor):
        c = self.cfg
        t = hidden.shape[0]
        proj = hidden.to(self.index_proj.dtype) @ self.index_proj.t()  # [T, 640]
        q_idx = proj[:, : c.index_q_dim].reshape(t, c.index_n_heads, c.index_head_dim)
        raw_k = proj[:, c.index_q_dim :]  # [T, 128]

        # q: Gemma-RMSNorm(128) -> partial RoPE(64) -> bf16
        #    indexer_qsa.py:323-330 (unfused) / ops/qsa_pre_indexer.py:68-77 (fused)
        qn = _gemma_rmsnorm(q_idx, self.index_q_norm_w, c.eps)
        cos = self.cos_sin[positions][..., : c.rotary_dim // 2]
        sin = self.cos_sin[positions][..., c.rotary_dim // 2 :]
        qn = _apply_rope(qn, cos[:, None, :], sin[:, None, :], c.rotary_dim)
        qn = qn.to(torch.bfloat16)  # indexer_qsa.py:330 .to(self.indexer_dtype)

        # store raw keys (compressor input); dtype bf16 like the ring cache
        self.raw_key[self.seq_len : self.seq_len + t] = raw_k.to(torch.bfloat16)

        # compressed rows for every complete group, lazily recomputed from raw_key
        first_complete = self.compressed_valid
        total_positions = self.seq_len + t
        n_complete = total_positions // c.compress_ratio
        for g in range(first_complete, n_complete):
            start = g * c.compress_ratio
            group = self.raw_key[start : start + c.compress_ratio]  # [4, 128]
            # pool in fp32 then round to bf16 (qsa_pre_indexer.py:266-269)
            pooled = (
                (group.float().sum(0) / c.compress_ratio).to(torch.bfloat16).float()
            )
            kc = _gemma_rmsnorm(
                pooled.reshape(1, c.index_head_dim), self.index_k_norm_w, c.eps
            )  # qsa_pre_indexer.py:323-333 (norm then rope)
            pos = torch.tensor([start])  # first_position = end-3 (qsa_pre_indexer.py:271)
            cosg = self.cos_sin[pos][..., : c.rotary_dim // 2]
            sing = self.cos_sin[pos][..., c.rotary_dim // 2 :]
            kc = _apply_rope(kc, cosg, sing, c.rotary_dim)  # qsa_pre_indexer.py:323-333
            self.compressed[g] = kc[0].to(torch.bfloat16)
        self.compressed_valid = n_complete

        return qn, raw_k, n_complete

    def _select(self, qn: torch.Tensor, positions: torch.Tensor, n_complete: int):
        """Indexer score -> block top-k -> expand + causal tail.

        qsa_indexer.py:94-100 (score), :471-499 (top-k), :221-276 (expand).
        """
        c = self.cfg
        t = qn.shape[0]
        seq_len = self.seq_len + t
        # The kernel keeps a logits buffer as wide as the compressed cache
        # (`logits = torch.empty((rows, columns))`, qsa_indexer.py:536, where
        # `columns = page_table_width * page_size`), i.e. it scores EVERY visible
        # compressed row -- the block_topk truncation happens in the top-k, not
        # before scoring. So the dump is [t, seq_len // CR], not [t, 512].
        n_cand = max(seq_len // c.compress_ratio, 1)
        out = torch.full((t, c.packed_width), -1, dtype=torch.int32)
        logits_out = torch.full((t, n_cand), float("-inf"), dtype=torch.float32)
        block_idx_out = torch.full((t, c.block_topk), -1, dtype=torch.int32)
        cols = torch.arange(c.output_width)
        for row in range(t):
            pos = int(positions[row])
            # common/qsa_cache.py:253-275 with a single request
            visible = min((pos + 1) // c.compress_ratio, seq_len // c.compress_ratio)
            # NOTE: `visible == 0` (the first 1..3 tokens of a sequence) must NOT
            # skip the row. The scoring kernel returns early when nothing is
            # visible (`qsa_indexer.py:56-57`; prefill `:163-164`) and `_topk`
            # (`:471-499`, called at `:563`) then fills all -1, but the EXPAND
            # stage still emits the causal tail (`:242-243` computes
            # tail_start/tail_count without consulting
            # visible_blocks). So row 0 of a fresh sequence carries exactly one
            # selected token (position 0).
            if visible > 0:
                keys = self.compressed[:visible].float()  # [V, 128]
                # qsa_indexer.py:94-100: scores = keys @ q ; relu ; sum over 4 heads.
                scores = torch.relu(keys @ qn[row].float().t())  # [V, 4]
                logit = scores.sum(-1)  # [V]  (kernel: tl.sum(scores, axis=2))
                logits_out[row, :visible] = logit
                k = min(visible, c.block_topk)
                top = torch.topk(logit, k).indices  # order is unspecified in the kernel
                block_idx_out[row, :k] = top.to(torch.int32)
            else:
                k = 0
                # dummy slot: `safe_rank` below is clamped to 0 and every
                # "expanded" column is masked out when expanded_count == 0
                top = torch.zeros(1, dtype=torch.int64)

            # expand + tail (qsa_indexer.py:221-276), vectorized over columns:
            #   columns [0, expanded_count)  -> block_rank = col // CR, offset = col % CR
            #   columns [expanded_count, ...) -> causal tail of the not-yet-complete group
            expanded_count = k * c.compress_ratio
            tail_start = ((pos + 1) // c.compress_ratio) * c.compress_ratio
            tail_count = (pos + 1) - tail_start
            is_exp = cols < expanded_count
            safe_rank = (cols // c.compress_ratio).clamp(max=max(k - 1, 0))
            expanded = top[safe_rank] * c.compress_ratio + (cols % c.compress_ratio)
            tail_offset = cols - expanded_count
            is_tail = (
                (cols >= expanded_count)
                & (tail_offset < tail_count)
                & (tail_offset < c.compress_ratio - 1)
            )
            token = torch.where(is_exp, expanded, tail_start + tail_offset)
            valid = (is_exp | is_tail) & (token >= 0) & (token < seq_len)
            out[row, : c.output_width] = torch.where(
                valid, token, torch.full_like(token, -1)
            ).to(torch.int32)
            out[row, c.output_width] = expanded_count + tail_count
        return out, logits_out, block_idx_out

    # ------------------------------------------------------- sparse attention
    def _sparse_attention(self, q, k_cache, v_cache, indices, gate):
        """nvidia/ops/qsa.py:32-233 + epilogue :181-214.

        Vectorized over the GQA group (12 query heads share one KV head) instead
        of looping head by head; the kernel does exactly this
        (`ops/qsa.py:89-100` loads a BLOCK_M=next_pow2(GROUP_SIZE) block of heads).
        Online softmax (`ops/qsa.py:162-179`, exp2-based) is replaced by one fp32 softmax
        over the whole row -- equivalent when there is no split-K, but a different
        summation order, so this is a <=1ulp-class stage, not bit-exact.
        """
        c = self.cfg
        t = q.shape[0]
        group = c.num_heads // c.num_kv_heads  # qsa.py:670-671  GROUP_SIZE = 12
        out = torch.zeros_like(q)
        scale = c.head_dim**-0.5  # ops/qsa.py:648  head_dim=256 -> 0.0625
        for row in range(t):
            tokens = indices[row, : c.output_width]
            tokens = tokens[tokens >= 0].long()
            if tokens.numel() == 0:
                continue
            for kv in range(c.num_kv_heads):
                heads = slice(kv * group, (kv + 1) * group)
                keys = k_cache[tokens, kv, :].float()  # [n, 256]
                vals = v_cache[tokens, kv, :].float()
                scores = (q[row, heads].float() @ keys.t()) * scale  # ops/qsa.py:157-161
                probs = torch.softmax(scores, dim=-1)  # ops/qsa.py:162-179
                acc = probs @ vals  # fp32 acc over bf16 values (ops/qsa.py:173-178)
                acc = acc.to(torch.bfloat16)  # ops/qsa.py:196  bf16 round before the gate
                out[row, heads, :] = (
                    acc.float() * torch.sigmoid(gate[row, heads, :].float())
                ).to(torch.bfloat16)  # ops/qsa.py:198-205
        return out

    # ------------------------------------------------------------------- fwd
    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        c = self.cfg
        taps: dict[str, torch.Tensor] = {}
        q, k, v, gate = self._project_qkv_gate(hidden_states, positions)
        taps["qsa.q"] = q
        taps["qsa.k"] = k
        taps["qsa.v"] = v
        taps["qsa.gate"] = gate

        qn, raw_k, n_complete = self._indexer(hidden_states, positions)
        taps["qsa.index_q"] = qn
        taps["qsa.index_k"] = raw_k
        taps["qsa.compressed_key"] = self.compressed[:n_complete].clone()

        indices, logits, block_idx = self._select(qn, positions, n_complete)
        taps["qsa.index_logits"] = logits
        taps["qsa.block_indices"] = block_idx
        taps["qsa.token_indices"] = indices

        # main KV write (do_kv_cache_update, nvidia/qsa.py:480-486) happens after
        # the indexer (nvidia/qsa.py:472-498 ordering).
        self.k_cache[self.seq_len : self.seq_len + hidden_states.shape[0]] = k
        self.v_cache[self.seq_len : self.seq_len + hidden_states.shape[0]] = v

        attn = self._sparse_attention(q, self.k_cache, self.v_cache, indices, gate)
        taps["qsa.attn_out"] = attn
        flat = attn.reshape(hidden_states.shape[0], -1).to(self.o_proj.dtype)
        out = flat @ self.o_proj.t()  # nvidia/qsa.py:525
        self.seq_len += hidden_states.shape[0]
        return out, taps
