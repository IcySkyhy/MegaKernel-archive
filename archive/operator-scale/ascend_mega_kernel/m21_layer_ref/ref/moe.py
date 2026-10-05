"""MoE segment reference — Qwen4ExpSparseMoeBlock (= Qwen3NextSparseMoeBlock).

Scope: what `Qwen4ExpDecoderLayer.forward` calls as `self.mlp(block_input)`
(vllm/models/qwen4_exp/nvidia/model.py:330). Input `[T, 2560]` bf16, output
`[T, 2560]` bf16.

NOTE on the `file:line` anchors below: the line number is a LOOKUP HINT against
the pinned base commit shown above; line numbers drift when upstream moves.
The stable reference is the SYMBOL. `python3 tools/check_anchors.py --verbose`
prints, for every anchor, which file it resolves to and which symbol it lands in
(it also fails if an anchor is ambiguous or out of range), and
`selfcheck.py` section I runs that check. Project rule 2026-09-26.

vLLM anchors:

  block construction      vllm/model_executor/models/qwen3_next.py:131-238
  subclass (no override)  vllm/models/qwen4_exp/nvidia/model.py:159-170
                          (only rejects sequence-parallel MoE, sets n_shared_experts)
  forward                 qwen3_next.py:240-270 -> MoERunner.forward
  router gate             vllm/model_executor/layers/fused_moe/router/gate_linear.py:18,192-249
                          (out_dtype is None here -> tier 5, plain bf16 F.linear)
  softmax / topk / renorm csrc/libtorch_stable/moe/topk_softmax_kernels.cu:113-134,497-543,581-592
                          (`renormalize` default True: config has no norm_topk_prob,
                           getattr(..., True) at qwen3_next.py:227)
  no routed scale         topk_softmax_kernels.cu:845-847 (routed_scaling_factor pinned to 1.0)
  shared expert gate      vllm/model_executor/models/qwen2_moe.py:115-123
                          `out = F.sigmoid(self.expert_gate(x)[0]) * out`
  expert MLP              qwen2_moe.py:91-118 (gate_up_proj -> SiluAndMul -> down_proj)
  SiluAndMul              vllm/model_executor/layers/activation.py:140-144
                          `silu(x[..., :d]) * x[..., d:]`
  final combine           vllm/model_executor/layers/fused_moe/runner/moe_runner.py:785-788
                          `result = shared_output + fused_output`

Final formula:  out = Σ_{k∈top10} w_k · Expert_k(x)  +  sigmoid(W_sg·x) · SharedMLP(x)

Precision: vLLM main has no `ascend` quant method (docs/14 §11 item 6), so there is
no runnable official path for THIS checkpoint's MoE. What is reproduced here is the
semantics of the block with the checkpoint's exact MXFP4 weight values (the E2M1
code set times a power-of-two scale is exactly representable in bf16, so
dequantization is lossless in bf16/fp32). `precision` selects the activation dtype:
  "fp32"  -> the L1 full-precision reference chain (default; docs/17 §1 L1)
  "bf16"  -> what a bf16 GEMM path produces
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MoE:
    def __init__(self, weights, text_config: dict, precision: str = "fp32"):
        self.w = weights
        self.num_experts = text_config["num_experts"]  # 512
        self.top_k = text_config["num_experts_per_tok"]  # 10
        self.hidden = text_config["hidden_size"]  # 2560
        self.inter = text_config["moe_intermediate_size"]  # 640
        self.renormalize = text_config.get("norm_topk_prob", None)
        if self.renormalize is None:
            self.renormalize = True  # qwen3_next.py:227 getattr(config,"norm_topk_prob",True)
        self.precision = precision
        assert precision in ("fp32", "bf16")
        self.dtype = torch.float32 if precision == "fp32" else torch.bfloat16

    # ------------------------------------------------------------ routing
    def router(self, x: torch.Tensor):
        """Returns (logits bf16, topk_ids int64 [T,k], topk_weights fp32 [T,k]).

        gate_linear.py:234-241 -> F.linear on bf16 (out_dtype is None here).
        topk_softmax_kernels.cu:118,134 -> softmax in fp32 regardless of input dtype.
        topk_softmax_kernels.cu:581-592 -> renormalize divides by the selected sum
          (denom = selected_sum if > 0 else 1) and applies routed_scaling_factor,
          which is hard-pinned to 1.0 for the softmax path (:845-847).
        """
        xin = x.to(self.w.gate_weight.dtype)
        logits = F.linear(xin, self.w.gate_weight)  # [T, 512] bf16
        probs = torch.softmax(logits.float(), dim=-1)  # fp32
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        if self.renormalize:
            denom = torch.where(
                top_w.sum(-1, keepdim=True) > 0,
                top_w.sum(-1, keepdim=True),
                torch.ones_like(top_w[:, :1]),
            )
            top_w = top_w / denom
        # routed_scaling_factor == 1.0 -> no further scaling
        return logits, top_i, top_w.contiguous()

    # ------------------------------------------------------------ experts
    def _expert(self, x: torch.Tensor, e: int) -> torch.Tensor:
        """One expert MLP on a slice of tokens (qwen2_moe.py:115-119)."""
        gu = self.w.expert_gate_up(e).to(self.dtype)  # [1280, 2560]
        dn = self.w.expert_down(e).to(self.dtype)  # [2560, 640]
        h = x @ gu.t()  # [n, 1280]
        d = self.inter
        gate = h[..., :d]  # order set by routed_experts.py:936-941 (chunk(2,dim=1)[0]=gate)
        up = h[..., d:]
        act = F.silu(gate) * up  # activation.py:140-144 (SiluAndMul)
        return act @ dn.t()  # [n, 2560]

    # ------------------------------------------------------------ shared
    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        """qwen2_moe.py:115-121 with expert_gate = shared_expert_gate."""
        xin = x.to(self.w.shared_gate_proj.dtype)
        gate_up = xin @ self.w.shared_gate_proj.t()  # [n, 640]
        up = xin @ self.w.shared_up_proj.t()  # [n, 640]
        act = F.silu(gate_up) * up
        out = act @ self.w.shared_down_proj.t()  # [n, 2560]
        sg = F.linear(x.to(self.w.shared_expert_gate_weight.dtype),
                      self.w.shared_expert_gate_weight)  # [n, 1] bf16
        return F.sigmoid(sg.float()) * out  # qwen2_moe.py:121

    # ----------------------------------------------------------------- fwd
    def _routed(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor):
        """Σ_k w_k · Expert_k(x), iterating EXPERT-major.

        The summation order matters twice: (a) it is the order the official
        grouped GEMM uses (tokens are permuted so each expert sees a contiguous
        row block, `runner/moe_runner.py`), and (b) expert-major means each
        expert's packed weights are dequantized at most once. A `k`-major loop
        would dequantize all 512 experts ten times over.
        """
        routed = torch.zeros_like(x, dtype=torch.float32)
        # (row, k) pairs grouped by expert; top-k picks distinct experts per row,
        # so a row appears at most once per expert.
        for e in torch.unique(ids).tolist():
            mask = ids == e
            rows, ks = mask.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            y = self._expert(x[rows], int(e))  # [n, 2560]
            routed[rows] += y.float() * weights[rows, ks].unsqueeze(-1)
        return routed

    def forward(self, x: torch.Tensor):
        taps: dict[str, torch.Tensor] = {}
        logits, ids, weights = self.router(x)
        taps["moe.router_logits"] = logits
        taps["moe.topk_ids"] = ids.to(torch.int32)
        taps["moe.topk_weights"] = weights

        xin = x.to(self.dtype)
        routed = self._routed(xin, ids, weights)
        taps["moe.routed_out"] = routed.to(x.dtype)

        shared = self._shared(x)  # [T, 2560]
        taps["moe.shared_out"] = shared.to(x.dtype)

        out = (routed + shared.float()).to(x.dtype)  # moe_runner.py:785-788
        taps["moe.out"] = out
        return out, taps
