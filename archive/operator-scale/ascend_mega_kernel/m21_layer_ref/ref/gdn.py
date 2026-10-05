"""GDN ("linear_attention") segment reference — QwenGatedDeltaNetAttention.

Scope: the *segment* boundary of one decoder layer, i.e. what
`Qwen4ExpDecoderLayer.forward` calls as `self.linear_attn(hidden_states=block_input)`
(vllm/models/qwen4_exp/nvidia/model.py:316-317). Input `[T, 2560]` bf16, output
`[T, 2560]` bf16, plus the two mamba states.

Replicated from these vLLM anchors:

  module construction      vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:386-539
  weight fusion            qwen_gdn_linear_attn.py:564-588 (qkvz), :590-612 (ba),
                           qwen3_5.py:219-226 (checkpoint -> fused names)
  conv weight reshape      qwen_gdn_linear_attn.py:1311-1314
                           `conv_weights = weight.view(size(0), size(2))` -> [dim, 4]
  forward split            qwen_gdn_linear_attn.py:940-946 (CUDA), :1030-1036 (CPU)
  conv1d (prefill ref)     vllm/model_executor/layers/mamba/ops/cpu/causal_conv1d.py:33-82
  conv1d (decode ref)      causal_conv1d.py:118-147
  l2norm + scale           vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:317-320
                           (decode kernel, eps 1e-6 hard-coded)
  gating                   fused_recurrent.py:322-329
  per-token recurrence     fused_recurrent.py:331-335
  output projection        qwen_gdn_linear_attn.py:845-857 + layers/layernorm.py:266-314
  state shapes             layers/mamba/mamba_utils.py:279-300

Two deliberate choices, both documented in the README:

  1. **Sequential recurrence, not the chunked factorization.** The chunked kernels
     (`third_party/flash_linear_attention/ops/chunk.py:32-77`) are an exact
     refactorization of the same recurrence; vLLM's own CPU test asserts the two
     agree (`tests/kernels/mamba/cpu/test_cpu_gdn_ops.py:308-367`). The sequential
     form is used because it is the readable spec and stays valid for any T.
  2. **l2norm eps = 1e-6** (the CUDA/Triton production value), not the 1e-5 that
     the CPU C++ port hard-codes (csrc/cpu/sgl-kernels/fla.cpp:1670). Backends
     disagree; pin one explicitly.

NOTE on the `file:line` anchors below: the line number is a LOOKUP HINT against
the pinned base commit shown above; line numbers drift when upstream moves.
The stable reference is the SYMBOL. `python3 tools/check_anchors.py --verbose`
prints, for every anchor, which file it resolves to and which symbol it lands in
(it also fails if an anchor is ambiguous or out of range), and
`selfcheck.py` section I runs that check. Project rule 2026-09-26.

Recurrence (per v-head `hv`, using k-head `i_h = hv // (num_v_heads // num_k_heads)`),
fused_recurrent.py:296-341:

    q̂ = l2norm_1e-6(q) * head_k_dim**-0.5        # fused_recurrent.py:317-320
    k̂ = l2norm_1e-6(k)                           # fused_recurrent.py:318-319
    g = -exp(A_log[hv]) * softplus(a[hv] + dt_bias[hv])   # :326-328
    β = sigmoid(b[hv])                           # :329
    h  *= exp(g)                                 # :331
    v'  = (v - h @ k̂) * β                        # :332-333
    h  += outer(v', k̂)                           # :334
    o   = h @ q̂      (uses the UPDATED h)         # :335
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

L2NORM_EPS = 1e-6  # fused_recurrent.py:318-319 and :238 (L2NORM_EPS constexpr)
SOFTPLUS_THRESHOLD = 20.0  # fused_recurrent.py (SOFTPLUS_THRESHOLD), :327


class _GDNConfig:
    def __init__(self, text_config: dict):
        self.num_k_heads = text_config["linear_num_key_heads"]  # 16
        self.num_v_heads = text_config["linear_num_value_heads"]  # 48
        self.head_k_dim = text_config["linear_key_head_dim"]  # 128
        self.head_v_dim = text_config["linear_value_head_dim"]  # 128
        self.conv_kernel_size = text_config["linear_conv_kernel_dim"]  # 4
        self.hidden_size = text_config["hidden_size"]  # 2560
        self.eps = text_config["rms_norm_eps"]  # 1e-6
        self.output_gate_type = text_config.get("output_gate_type", "silu")
        # qwen_gdn_linear_attn.py:400-402
        self.key_dim = self.head_k_dim * self.num_k_heads  # 2048
        self.value_dim = self.head_v_dim * self.num_v_heads  # 6144
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 10240


class GDN:
    """Reference GDN segment (stateful: conv_state + ssm_state)."""

    def __init__(self, weights, text_config: dict):
        self.cfg = _GDNConfig(text_config)
        c = self.cfg
        # in_proj_qkvz: MergedColumnParallelLinear(2560, [2048,2048,6144,6144])
        self.w_qkvz = weights.in_proj_qkvz.to(torch.bfloat16)  # [16384, 2560]
        # in_proj_ba: MergedColumnParallelLinear(2560, [48, 48])
        self.w_ba = weights.in_proj_ba.to(torch.bfloat16)  # [96, 2560]
        # conv1d.weight reshaped to [conv_dim, kernel] (qwen_gdn_linear_attn.py:1312-1314)
        self.conv_w = weights.conv1d_weight.to(torch.bfloat16)  # [10240, 4]
        self.norm_w = weights.norm_weight.to(torch.bfloat16)  # [128]
        self.out_proj = weights.out_proj_weight.to(torch.bfloat16)  # [2560, 6144]
        self.A_log = weights.A_log.float()  # [48] (vLLM forces fp32, :473-478)
        self.dt_bias = weights.dt_bias.to(torch.bfloat16)  # [48]

        # derived dtypes / shapes
        self.conv_dtype = torch.bfloat16  # conv_state follows model dtype
        self.state_dtype = torch.float32  # mamba_ssm_dtype = "float32"
        assert self.conv_w.shape == (c.conv_dim, c.conv_kernel_size)
        self.conv_state_len = c.conv_kernel_size - 1  # 3 (mamba_utils.py:290-293)

    # ------------------------------------------------------------------ state
    def init_states(self, dtype_state=None):
        t = self.conv_dtype
        conv_state = torch.zeros(self.cfg.conv_dim, self.conv_state_len, dtype=t)
        ssm_state = torch.zeros(
            self.cfg.num_v_heads, self.cfg.head_v_dim, self.cfg.head_k_dim,
            dtype=self.state_dtype,
        )
        return conv_state, ssm_state

    # ------------------------------------------------------------------ parts
    def _conv(self, mixed_qkv: torch.Tensor, conv_state: torch.Tensor):
        """Causal depthwise conv1d + silu, state updated in place.

        Direct transcription of `causal_conv1d_update_torch`
        (mamba/ops/cpu/causal_conv1d.py:131-147) generalized to any T, which is the
        same math as the prefill reference (:55-82). Both are the CPU reference for
        the CUDA `causal_conv1d_fn` / `causal_conv1d_update` ops.

        x: [T, D] -> [1, D, T]; weight [D, 1, K]; output [1, D, T] -> [T, D].
        """
        t, dim = mixed_qkv.shape
        state_len = self.conv_state_len
        x = mixed_qkv.transpose(0, 1).unsqueeze(0).to(conv_state.dtype)  # [1, D, T]
        x_new = torch.cat([conv_state.unsqueeze(0), x], dim=-1)  # :135
        new_state = x_new[:, :, -state_len:].contiguous()  # :136 / :81
        out = F.conv1d(
            x_new, self.conv_w.unsqueeze(1), None, padding=0, groups=dim
        )[:, :, -t:]  # :138-144
        out = F.silu(out)  # :145-146
        return out[0].transpose(0, 1).contiguous(), new_state[0]

    @staticmethod
    def _l2norm(x: torch.Tensor) -> torch.Tensor:
        """fused_recurrent.py:318-319 / fused_gdn_prefill_post_conv.py:84-91."""
        return x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + L2NORM_EPS)

    def _gating(self, a: torch.Tensor, b: torch.Tensor):
        """fused_recurrent.py:322-329.

        a, b: [T, num_v_heads] bf16 -> g fp32, beta fp32.
        """
        a_val = a.float()
        b_val = b.float()
        A_log = self.A_log  # [HV] fp32
        dt_bias = self.dt_bias.float()
        x = a_val + dt_bias  # :326
        # softplus with the threshold-20 linear tail (:327)
        sp = torch.nn.functional.softplus(x, beta=1.0, threshold=SOFTPLUS_THRESHOLD)
        g = -torch.exp(A_log) * sp  # :328
        beta = torch.sigmoid(b_val)  # :329
        return g, beta

    def _recurrence(self, q, k, v, g, beta, ssm_state):
        """Per-token delta rule, fused_recurrent.py:331-335.

        q: [T, Hk, K] fp32 (already l2normed and scaled)
        k: [T, Hk, K] fp32 (already l2normed)
        v: [T, Hv, V] fp32   g, beta: [T, Hv] fp32
        ssm_state: [Hv, V, K] fp32, updated in place (Triton stores h as [V, K]:
        `o_v[:, None] * K + o_k[None, :]`, fused_recurrent.py:306).
        Returns o: [T, Hv, V] fp32.
        """
        cfg = self.cfg
        t_len = q.shape[0]
        group = cfg.num_v_heads // cfg.num_k_heads  # 48 // 16 = 3
        o = torch.empty(t_len, cfg.num_v_heads, cfg.head_v_dim, dtype=torch.float32)
        for t in range(t_len):
            decay = torch.exp(g[t])  # :331 (exp(g_val))
            for hv in range(cfg.num_v_heads):
                ih = hv // group  # :284  i_h = i_hv // (HV // H)
                h = ssm_state[hv]  # [V, K]
                h *= decay[hv]  # :331
                kv_mem = h @ k[t, ih]  # :332  sum(b_h * b_k[None, :], 1)
                delta = (v[t, hv] - kv_mem) * beta[t, hv]  # :333
                h += delta.unsqueeze(-1) * k[t, ih].unsqueeze(0)  # :334
                o[t, hv] = h @ q[t, ih]  # :335
        return o

    def _rms_norm_gated(self, x, z):
        """RMSNormGated(norm_before_gate=True, activation='sigmoid')
        layers/layernorm.py:266-314; constructed at qwen_gdn_linear_attn.py:490-497
        with head_v_dim=128, group_size=None, eps=1e-6, activation=output_gate_type.
        """
        orig = x.dtype
        xf = x.float()
        zf = z.float()
        act = F.sigmoid if self.cfg.output_gate_type == "sigmoid" else F.silu
        variance = xf.square().mean(dim=-1, keepdim=True)  # :300
        x_normed = xf * torch.rsqrt(variance + self.cfg.eps)  # :301
        out = x_normed * self.norm_w.float()  # :302
        out = out * act(zf)  # :311-312 (norm_before_gate=True)
        return out.to(orig)

    # ------------------------------------------------------------------ fwd
    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Returns (output, new_conv_state, ssm_state, taps).

        `taps` carries the intermediate activations the harness dumps.
        `ssm_state` is modified in place (as the kernels do).
        """
        cfg = self.cfg
        taps: dict[str, torch.Tensor] = {}
        t_len = hidden_states.shape[0]

        # --- projections (qwen_gdn_linear_attn.py:1018-1019, :940-946) ---
        h = hidden_states.to(self.w_qkvz.dtype)
        mixed_qkvz = h @ self.w_qkvz.t()  # [T, 16384]
        ba = h @ self.w_ba.t()  # [T, 96]
        qkv_size = cfg.key_dim * 2 + cfg.value_dim  # :942
        z_size = cfg.value_dim  # :943
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)  # :944
        z = z.reshape(t_len, -1, cfg.head_v_dim)  # :945
        b, a = ba.chunk(2, dim=-1)  # :946 / :1036
        taps["gdn.mixed_qkvz"] = mixed_qkvz
        taps["gdn.ba"] = ba

        # --- short conv (silu) ---
        conv_out, new_conv_state = self._conv(mixed_qkv, conv_state)
        taps["gdn.conv_out"] = conv_out

        # --- split + l2norm + scale (fused_recurrent.py:310-320) ---
        q = conv_out[:, : cfg.key_dim].reshape(t_len, cfg.num_k_heads, cfg.head_k_dim)
        k = conv_out[
            :, cfg.key_dim : 2 * cfg.key_dim
        ].reshape(t_len, cfg.num_k_heads, cfg.head_k_dim)
        v = conv_out[:, 2 * cfg.key_dim :].reshape(
            t_len, cfg.num_v_heads, cfg.head_v_dim
        )
        scale = cfg.head_k_dim**-0.5  # qwen_gdn_linear_attn.py:1676 self.head_k_dim**-0.5
        qn = self._l2norm(q) * scale  # :317-320
        kn = self._l2norm(k)  # :318-319
        taps["gdn.q"] = qn.to(torch.bfloat16)
        taps["gdn.k"] = kn.to(torch.bfloat16)
        taps["gdn.v"] = v

        # --- gating ---
        g, beta = self._gating(a, b)
        taps["gdn.g"] = g
        taps["gdn.beta"] = beta

        # --- recurrence ---
        core_out = self._recurrence(qn, kn, v.float(), g, beta, ssm_state)
        taps["gdn.core_out"] = core_out.to(torch.bfloat16)

        # --- RMSNormGated + out_proj (qwen_gdn_linear_attn.py:845-857) ---
        normed = self._rms_norm_gated(core_out, z)  # :855
        taps["gdn.normed"] = normed
        flat = normed.reshape(t_len, -1).to(self.out_proj.dtype)  # :856
        out = flat @ self.out_proj.t()  # :856
        return out, new_conv_state, ssm_state, taps
