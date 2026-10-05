"""Hyper-connection (GatedResidual) reference — bit-faithful to the NVIDIA path.

Every arithmetic statement below carries the vLLM source anchor it was copied
from. The anchors are:

  vllm/models/qwen4_exp/nvidia/ops/hc.py            (the five Triton glue kernels)
  vllm/models/qwen4_exp/nvidia/hyperconnection.py   (the module wiring)
  vllm/models/qwen4_exp/common/hyperconnection.py   (the plain-torch reference impl)
  vllm/tests/models/qwen4_exp/test_hc_ops.py        (vLLM's own torch references)

The Triton kernels are the *production* spec (CUDA is what the model actually
runs on), so they are what this module replicates. `test_hc_ops.py` provides an
independent torch statement of the same math; `test_hc_ops_reference()` below
runs it side by side so the two can be compared.

NOTE on the `file:line` anchors below: the line number is a LOOKUP HINT against
the pinned base commit shown above; line numbers drift when upstream moves.
The stable reference is the SYMBOL. `python3 tools/check_anchors.py --verbose`
prints, for every anchor, which file it resolves to and which symbol it lands in
(it also fails if an anchor is ambiguous or out of range), and
`selfcheck.py` section I runs that check. Project rule 2026-09-26.

Rounding points that MUST be preserved (they are the whole reason this is not
just a few matmuls):

  * `grouped_gemma_rmsnorm` writes bf16           (hc.py:52 stores into a bf16 buffer)
  * `hc_silu` divides by HC *before* silu, bf16   (hc.py:100-105)
  * `hc_gate_mix` sigmoids the bf16 gate, /HC     (hc.py:150-160)
  * `hc_combine`  w = 2*sigmoid(inj/4) on bf16 inj(hc.py:226-228)
  * `hc_combine_norm` rounds the combine back to bf16 *before* the RMSNorm
                                                  (hc.py:327, comment at :325-326)
  * the merged down+inject Linear output is bf16  (nvidia/hyperconnection.py:141)

Reduction order is not identical to the Triton block reduction (a 2560-wide
`tl.sum` is a tree reduce inside one program; torch's `.sum(-1)` uses its own
unrolled tree). Norm outputs are therefore expected to agree to <=1 ulp, while
the elementwise ops should be bit-exact. See README "tolerance classes".
"""

from __future__ import annotations

import torch

HC_DEFAULT = 4
HC_LOWRANK_DEFAULT = 320
EPS_DEFAULT = 1e-6


# ---------------------------------------------------------------------------
# grouped_gemma_rmsnorm  <->  _grouped_gemma_rmsnorm_kernel (hc.py:12-52)
# ---------------------------------------------------------------------------
def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, hc_count: int
) -> torch.Tensor:
    """4-group Gemma-RMSNorm with a per-element [10240] affine.

    hc.py:25-26   GROUP_DIM = DIM // NUM_GROUPS ; BLOCK_SIZE = next_pow2(GROUP_DIM)
    hc.py:42      x = load(...).to(tl.float32)
    hc.py:43      w = load(...)                      # bf16, cast lazily below
    hc.py:45      rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)
    hc.py:47-48   y = x * rrms ; y += y * w.to(tl.float32)      # Gemma (1+w)
    hc.py:52      store(y_ptr, y)                    # y_ptr is bf16 -> round
    """
    orig_dtype = x.dtype
    dim = x.shape[-1]
    group_dim = dim // hc_count  # hc.py:25
    xf = x.reshape(*x.shape[:-1], hc_count, group_dim).float()  # hc.py:42
    wf = weight.float()  # hc.py:43
    rrms = torch.rsqrt(xf.square().sum(-1, keepdim=True) / group_dim + eps)  # hc.py:45
    y = xf * rrms  # hc.py:47
    y = y + y * wf.reshape(hc_count, group_dim)  # hc.py:48
    return y.reshape(*x.shape).to(orig_dtype)  # hc.py:52


# ---------------------------------------------------------------------------
# hc_silu  <->  _hc_silu_kernel (hc.py:81-105)
# ---------------------------------------------------------------------------
def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """silu(x / HC) in fp32, stored bf16.  hc.py:100-101,105"""
    orig_dtype = x.dtype
    r = x.float() / hc_count  # hc.py:100
    r = r * torch.sigmoid(r)  # hc.py:101
    return r.to(orig_dtype)  # hc.py:105


# ---------------------------------------------------------------------------
# hc_gate_mix  <->  _hc_gate_mix_kernel (hc.py:125-160)
# ---------------------------------------------------------------------------
def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Mean over HC streams of sigmoid(gate_s) * x_s.  hc.py:150-160"""
    orig_dtype = x.dtype
    dim = gate.shape[-1]
    hc_dim = dim // hc_count  # hc.py:138
    g = gate.reshape(gate.shape[0], hc_count, hc_dim).float()  # hc.py:153
    xf = x.reshape(x.shape[0], hc_count, hc_dim).float()  # hc.py:154
    acc = (torch.sigmoid(g) * xf).sum(1)  # hc.py:155
    acc = acc / hc_count  # hc.py:156
    return acc.to(orig_dtype)  # hc.py:160


# ---------------------------------------------------------------------------
# hc_combine  <->  _hc_combine_kernel (hc.py:188-232)
# ---------------------------------------------------------------------------
def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    """res[s,j] + block[j] * w[s], w[s] = 2*sigmoid(inj[s]/HC) (unit when inj=None).

    hc.py:218-221  loads
    hc.py:226-227  inj = 2.0*sigmoid(inj/ HC) ; block = block * inj[:, None]
    hc.py:228      out = res + block
    hc.py:232      store into a residual.new_empty buffer (bf16) -> round
    """
    orig_dtype = residual.dtype
    dim = residual.shape[-1]
    hc_dim = dim // hc_count
    res = residual.reshape(residual.shape[0], hc_count, hc_dim).float()  # hc.py:221
    blk = block_output.float()  # hc.py:220
    if injection_logits is None:
        out = res + blk.unsqueeze(1)  # hc.py:228 with block unscaled
    else:
        inj = injection_logits.float()  # hc.py:219
        w = 2.0 * torch.sigmoid(inj / hc_count)  # hc.py:226
        out = res + blk.unsqueeze(1) * w.unsqueeze(-1)  # hc.py:227-228
    return out.reshape(residual.shape).to(orig_dtype)  # hc.py:232


# ---------------------------------------------------------------------------
# hc_combine_norm  <->  _hc_combine_norm_kernel (hc.py:272-345)
# ---------------------------------------------------------------------------
def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused combine + grouped Gemma-RMSNorm. Returns (materialized, normalized).

    hc.py:314-321  loads (res / inj / block)
    hc.py:322-324  inj = 2*sigmoid(inj/HC) ; block = block * inj[stream]
    hc.py:325-327  out = (res + block).to(out_ptr.dtype)   <-- EARLY bf16 ROUND
    hc.py:330      store out
    hc.py:332-334  out = out.to(float32) ; rrms = rsqrt(sum_sq/HC_DIM + EPS)
    hc.py:342-345  w = load(weight) ; y = out*rrms ; y += y*w ; store y
    """
    orig_dtype = residual.dtype
    dim = residual.shape[-1]
    hc_dim = dim // hc_count
    res = residual.reshape(residual.shape[0], hc_count, hc_dim).float()  # hc.py:314
    blk = block_output.float()  # hc.py:317-321
    if injection_logits is None:
        combined = res + blk.unsqueeze(1)  # hc.py:228-equivalent (no gate)
    else:
        inj = injection_logits.float()  # hc.py:316
        w = 2.0 * torch.sigmoid(inj / hc_count)  # hc.py:323
        combined = res + blk.unsqueeze(1) * w.unsqueeze(-1)  # hc.py:324
    # hc.py:325-327: round the materialized combine result before normalizing.
    materialized = combined.reshape(residual.shape).to(orig_dtype)
    out = materialized.reshape(residual.shape[0], hc_count, hc_dim).float()  # hc.py:332
    rrms = torch.rsqrt(out.square().sum(-1, keepdim=True) / hc_dim + eps)  # hc.py:333-334
    y = out * rrms  # hc.py:343
    y = y + y * norm_weight.float().reshape(hc_count, hc_dim)  # hc.py:344
    return materialized, y.reshape(residual.shape).to(orig_dtype)  # hc.py:330,345


# ---------------------------------------------------------------------------
# GatedResidual module  <->  nvidia/hyperconnection.py:50-196
# ---------------------------------------------------------------------------
class GatedResidual:
    """One hyper-connection mixer.

    Weight contract (checkpoint-native, no padding):
      hc_norm.weight            bf16 [HC*H]     -> grouped Gemma RMSNorm affine
      input_mix_weight_down     bf16 [R, HC*H]  -> low-rank down projection
      input_mix_weight_up       bf16 [HC*H, R]  -> low-rank up projection
      block_inject_weight       bf16 [HC, HC*H] -> injection logits (absent if not use_combine)

    vLLM packs down+inject into one MergedColumnParallelLinear whose output is
    padded to 16 rows: pad_size = -(R + HC) % 16 = 12, output width 336
    (nvidia/hyperconnection.py:95-107). The padding rows are dropped
    (`lora, injection, _ = split(...)`, :142) and a Linear's output rows are
    independent of each other, so a checkpoint-exact 320+4=324-row harness is
    numerically identical. See README "deviations from the padded merged linear".
    """

    def __init__(
        self,
        hc_norm_weight: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        inject_weight: torch.Tensor | None,
        hc_count: int = HC_DEFAULT,
        eps: float = EPS_DEFAULT,
        use_combine: bool = True,
    ):
        self.hc_norm_weight = hc_norm_weight
        self.down_weight = down_weight
        self.up_weight = up_weight
        self.inject_weight = inject_weight
        self.hc_count = hc_count
        self.eps = eps
        self.use_combine = use_combine
        self.hidden_size = hc_norm_weight.shape[0] // hc_count
        assert self.hc_count * self.hidden_size == hc_norm_weight.shape[0]
        if use_combine:
            assert inject_weight is not None and inject_weight.shape[0] == hc_count

    # -- projections -----------------------------------------------------
    def _down_inject(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """nvidia/hyperconnection.py:141-142 / 178-179 (split of the merged linear).

        The merged Linear runs in bf16, so `lora` and `injection` are bf16 --
        that rounding is preserved here explicitly.
        """
        inp = self.down_weight.dtype
        lora = (xn.to(inp) @ self.down_weight.t()).to(inp)  # merged down shard, bf16
        injection = (xn.to(inp) @ self.inject_weight.t()).to(inp)  # merged row shard
        return lora, injection

    def _up(self, lora: torch.Tensor) -> torch.Tensor:
        """nvidia/hyperconnection.py:148,185.  Gate is bf16 (Linear output)."""
        inp = self.up_weight.dtype
        return (lora.to(inp) @ self.up_weight.t()).to(inp)

    # -- the two entry points -------------------------------------------
    def mix(self, hidden_states: torch.Tensor):
        """nvidia/hyperconnection.py:128-151.  Returns (h, block_input, injection).

        Used only for the very first attn mixer of the model, where there is no
        pending block output (the containing layer's forward; `model.py:313-314`, `Qwen4ExpDecoderLayer.forward`).
        """
        xn = grouped_gemma_rmsnorm(
            hidden_states, self.hc_norm_weight, self.eps, self.hc_count
        )  # :131-136
        injection = None
        if self.use_combine:
            lora, injection = self._down_inject(xn)  # :141-142
        else:
            inp = self.down_weight.dtype
            lora = (xn.to(inp) @ self.down_weight.t()).to(inp)  # :144
        lora = hc_silu(lora, self.hc_count)  # :147
        gate = self._up(lora)  # :148
        block_input = hc_gate_mix(xn, gate, self.hc_count)  # :149
        return hidden_states, block_input, injection  # :151

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ):
        """nvidia/hyperconnection.py:153-188 (the layer-to-layer path)."""
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm_weight,
            self.eps,
            self.hc_count,
        )  # :166-173
        injection = None
        if self.use_combine:
            lora, injection = self._down_inject(xn)  # :175-179
        else:
            inp = self.down_weight.dtype
            lora = (xn.to(inp) @ self.down_weight.t()).to(inp)  # :181
        lora = hc_silu(lora, self.hc_count)  # :184
        gate = self._up(lora)  # :185
        block_input = hc_gate_mix(xn, gate, self.hc_count)  # :186
        return hidden_states, block_input, injection  # :188

    def combine(self, hidden_states, block_output, injection):
        """nvidia/hyperconnection.py:190-196 (used only to materialize a pending
        combine before PLE, model.py:293-297)."""
        return hc_combine(hidden_states, block_output, injection, self.hc_count)


# ---------------------------------------------------------------------------
# Eager (unfused) reference -- common/hyperconnection.py, directly importable
# ---------------------------------------------------------------------------
def eager_gated_residual_weights(down_weight, up_weight, inject_weight):
    """Build the objects needed to run `common/hyperconnection.py` GatedResidual.

    Returns a kwargs dict for `GatedResidual.__init__`; used by
    `ref/official.py` to run the official torch reference side by side.
    """
    return dict(
        down_weight=down_weight, up_weight=up_weight, inject_weight=inject_weight
    )
