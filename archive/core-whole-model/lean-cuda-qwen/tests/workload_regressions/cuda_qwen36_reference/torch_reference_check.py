#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle
"""Development-time cross-check of the NumPy oracle fixtures against the vendored
HF torch reference (modeling_qwen3_5_excerpt.py).

Loads a fixture directory (default ./fixtures), rebuilds the tiny model with the
excerpt classes in fp32 on CPU, runs prefill + decode + autograd backward, and
compares every dump/gradient tensor. Forward gate: max abs err <= 1e-5 (per the
workstream contract, any larger disagreement means the NumPy port is wrong).
Backward (autograd vs hand-written VJP) gate: <= 1e-4.

NOT part of the runtime gates: requires torch, never imported by oracle.py or
run_test.sh's mandatory path. Usage:
    torch_reference_check.py [fixtures_dir]   (default: ./fixtures next to this file)
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modeling_qwen3_5_excerpt as ref  # noqa: E402
import oracle  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

TOL_FWD = 1e-5
TOL_BWD = 1e-4


def build_config(cfg):
    return SimpleNamespace(
        hidden_size=cfg["hidden_size"],
        linear_num_value_heads=cfg["linear_num_value_heads"],
        linear_num_key_heads=cfg["linear_num_key_heads"],
        linear_key_head_dim=cfg["linear_key_head_dim"],
        linear_value_head_dim=cfg["linear_value_head_dim"],
        linear_conv_kernel_dim=cfg["linear_conv_kernel_dim"],
        hidden_act="silu",
        rms_norm_eps=cfg["rms_norm_eps"],
        layer_types=list(cfg["layer_types"]),
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        attention_dropout=0.0,
        attention_bias=False,
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": cfg["rope_theta"],
            "partial_rotary_factor": cfg["partial_rotary_factor"],
        },
    )


def t(name, fx):
    return torch.from_numpy(np.ascontiguousarray(fx[name])).float()


def main():
    fx_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures"
    )
    _, fx = oracle.load_fixtures(fx_dir)
    cfg = oracle.CFG
    T = cfg["seq_len"]
    nk, dk = cfg["linear_num_key_heads"], cfg["linear_key_head_dim"]
    nv, dv = cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
    kd, vd = nk * dk, nv * dv
    nq, nkv, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]

    config = build_config(cfg)
    torch.manual_seed(0)

    embed = torch.nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
    layers = torch.nn.ModuleList(
        [ref.Qwen3_5DecoderLayer(config, i) for i in range(len(cfg["layer_types"]))]
    )
    final_norm = ref.Qwen3_5RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
    lm_head = torch.nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
    rotary = ref.Qwen3_5TextRotaryEmbedding(config)

    # load fixture weights (HF names; conv weight stored squeezed in fixtures)
    with torch.no_grad():
        embed.weight.copy_(t("model.embed_tokens.weight", fx))
        for i, layer in enumerate(layers):
            p = f"model.layers.{i}."
            layer.input_layernorm.weight.copy_(t(p + "input_layernorm.weight", fx))
            layer.post_attention_layernorm.weight.copy_(t(p + "post_attention_layernorm.weight", fx))
            layer.mlp.gate_proj.weight.copy_(t(p + "mlp.gate_proj.weight", fx))
            layer.mlp.up_proj.weight.copy_(t(p + "mlp.up_proj.weight", fx))
            layer.mlp.down_proj.weight.copy_(t(p + "mlp.down_proj.weight", fx))
            if cfg["layer_types"][i] == "linear_attention":
                la = layer.linear_attn
                la.in_proj_qkv.weight.copy_(t(p + "linear_attn.in_proj_qkv.weight", fx))
                la.in_proj_z.weight.copy_(t(p + "linear_attn.in_proj_z.weight", fx))
                la.in_proj_b.weight.copy_(t(p + "linear_attn.in_proj_b.weight", fx))
                la.in_proj_a.weight.copy_(t(p + "linear_attn.in_proj_a.weight", fx))
                la.conv1d.weight.copy_(t(p + "linear_attn.conv1d.weight", fx).unsqueeze(1))
                la.A_log.copy_(t(p + "linear_attn.A_log", fx))
                la.dt_bias.copy_(t(p + "linear_attn.dt_bias", fx))
                la.norm.weight.copy_(t(p + "linear_attn.norm.weight", fx))
                la.out_proj.weight.copy_(t(p + "linear_attn.out_proj.weight", fx))
            else:
                sa = layer.self_attn
                sa.q_proj.weight.copy_(t(p + "self_attn.q_proj.weight", fx))
                sa.k_proj.weight.copy_(t(p + "self_attn.k_proj.weight", fx))
                sa.v_proj.weight.copy_(t(p + "self_attn.v_proj.weight", fx))
                sa.o_proj.weight.copy_(t(p + "self_attn.o_proj.weight", fx))
                sa.q_norm.weight.copy_(t(p + "self_attn.q_norm.weight", fx))
                sa.k_norm.weight.copy_(t(p + "self_attn.k_norm.weight", fx))
        final_norm.weight.copy_(t("model.norm.weight", fx))
        lm_head.weight.copy_(t("lm_head.weight", fx))

    for mod in list(layers.modules()) + [embed, lm_head]:
        for prm in mod.parameters(recurse=False):
            prm.requires_grad_(True)

    # ---- recording wrappers around excerpt functions ----
    rec = {"conv": [], "chunk": [], "gnorm": [], "rope": [], "attn": []}
    orig_conv, orig_chunk = ref.causal_conv1d_fn, ref.torch_chunk_gated_delta_rule
    orig_recur, orig_rope, orig_attn = (
        ref.torch_recurrent_gated_delta_rule,
        ref.apply_rotary_pos_emb,
        ref.eager_attention_forward,
    )
    orig_gnorm = ref.Qwen3_5RMSNormGated.forward

    def conv_wrap(hidden_states, weight, bias=None, activation=None, **kw):
        out = orig_conv(hidden_states, weight, bias, activation, **kw)
        rec["conv"].append((hidden_states.detach(), out.detach()))
        return out

    def chunk_wrap(query, key, value, g, beta, **kw):
        out, state = orig_chunk(query, key, value, g, beta, **kw)
        rec["chunk"].append(
            (query.detach(), key.detach(), value.detach(), g.detach(), beta.detach(), out.detach())
        )
        return out, state

    def gnorm_wrap(self, hidden_states, gate=None):
        out = orig_gnorm(self, hidden_states, gate)
        rec["gnorm"].append((hidden_states.detach(), gate.detach(), out.detach()))
        return out

    def rope_wrap(q, k, cos, sin, unsqueeze_dim=1):
        qe, ke = orig_rope(q, k, cos, sin, unsqueeze_dim)
        rec["rope"].append((qe.detach(), ke.detach()))
        return qe, ke

    def attn_wrap(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, probs = orig_attn(module, query, key, value, attention_mask, scaling, dropout, **kw)
        rec["attn"].append((out.detach(), probs.detach()))
        return out, probs

    ref.causal_conv1d_fn = conv_wrap
    ref.torch_chunk_gated_delta_rule = chunk_wrap
    ref.Qwen3_5RMSNormGated.forward = gnorm_wrap
    ref.apply_rotary_pos_emb = rope_wrap
    ref.eager_attention_forward = attn_wrap

    hooks_out = {}

    def hook(name):
        def fn(module, args, output):
            hooks_out[name] = output.detach() if torch.is_tensor(output) else output[0].detach()

        return fn

    embed.register_forward_hook(hook("embed"))
    for i, layer in enumerate(layers):
        layer.input_layernorm.register_forward_hook(hook(f"layer{i}.norm1"))
        layer.mlp.register_forward_hook(hook(f"layer{i}.mlp"))
        layer.register_forward_hook(hook(f"layer{i}.out"))
        if cfg["layer_types"][i] == "linear_attention":
            layer.linear_attn.out_proj.register_forward_hook(hook(f"layer{i}.mixer"))
        else:
            layer.self_attn.o_proj.register_forward_hook(hook(f"layer{i}.mixer"))
            layer.self_attn.o_proj.register_forward_pre_hook(
                lambda module, args: hooks_out.__setitem__("layer3.postgate", args[0].detach())
            )
    final_norm.register_forward_hook(hook("final"))
    lm_head.register_forward_hook(hook("logits"))

    tokens = torch.from_numpy(fx["tokens"].astype(np.int64)).unsqueeze(0)
    targets = torch.from_numpy(fx["targets"].astype(np.int64))
    position_ids = torch.arange(T).unsqueeze(0)
    causal = torch.where(
        torch.triu(torch.ones(T, T, dtype=torch.bool), 1),
        torch.tensor(float("-inf")),
        torch.tensor(0.0),
    )[None, None]

    x = embed(tokens)
    cos, sin = rotary(x, position_ids)
    for i, layer in enumerate(layers):
        # linear_attention layers take a padding mask (None here); full_attention
        # layers take the additive causal mask.
        mask = causal if cfg["layer_types"][i] == "full_attention" else None
        x = layer(
            x,
            position_embeddings=(cos, sin),
            attention_mask=mask,
            position_ids=position_ids,
        )
    fh = final_norm(x)
    fh.retain_grad()
    logits = lm_head(fh)
    logits.retain_grad()

    loss = F.cross_entropy(logits.view(-1, cfg["vocab_size"]), targets)
    loss.backward()

    # ---- comparisons ----
    results = []

    def check(name, torch_tensor, fixture_name, tol=TOL_FWD):
        a = torch_tensor.detach().cpu().numpy().astype(np.float32)
        b = fx[fixture_name].astype(np.float32)
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs fixture {b.shape}"
        err = float(np.abs(a - b).max()) if a.size else 0.0
        results.append((name, err, tol))

    check("embed_out", hooks_out["embed"][0], "embed_out")
    check("rope.cos", cos[0], "rope.cos")
    check("rope.sin", sin[0], "rope.sin")
    li = 0
    for i, layer in enumerate(layers):
        check(f"layer{i}.norm1_out", hooks_out[f"layer{i}.norm1"][0], f"layer{i}.norm1_out")
        check(f"layer{i}.mlp_out", hooks_out[f"layer{i}.mlp"][0], f"layer{i}.mlp_out")
        check(f"layer{i}.hidden_out", hooks_out[f"layer{i}.out"][0], f"layer{i}.hidden_out")
        check(f"layer{i}.mixer_out", hooks_out[f"layer{i}.mixer"][0], f"layer{i}.mixer_out")
        if cfg["layer_types"][i] != "linear_attention":
            continue
        conv_in, conv_out = rec["conv"][li]
        q_in, k_in, v_in, g_in, b_in, dout = rec["chunk"][li]
        gn_in, gn_gate, gn_out = rec["gnorm"][li]
        check(f"layer{i}.qkv_pre_conv", conv_in[0].transpose(0, 1), f"layer{i}.qkv_pre_conv")
        check(f"layer{i}.q_post_conv", conv_out[0, :kd].transpose(0, 1), f"layer{i}.q_post_conv")
        check(f"layer{i}.k_post_conv", conv_out[0, kd : 2 * kd].transpose(0, 1), f"layer{i}.k_post_conv")
        check(f"layer{i}.v_post_conv", conv_out[0, 2 * kd :].transpose(0, 1), f"layer{i}.v_post_conv")
        check(f"layer{i}.beta", b_in[0], f"layer{i}.beta")
        check(f"layer{i}.g", g_in[0], f"layer{i}.g")
        check(f"layer{i}.q_l2", ref.l2norm(q_in, dim=-1, eps=1e-6)[0], f"layer{i}.q_l2")
        check(f"layer{i}.k_l2", ref.l2norm(k_in, dim=-1, eps=1e-6)[0], f"layer{i}.k_l2")
        check(f"layer{i}.v_heads", v_in[0], f"layer{i}.v_heads")
        check(f"layer{i}.delta_out", dout[0], f"layer{i}.delta_out")
        check(f"layer{i}.gated_out", gn_out.reshape(T, vd), f"layer{i}.gated_out")
        # S_final: torch chunk returns no state without a cache; use recurrent rule
        _, S_rec = orig_recur(
            q_in, k_in, v_in, g=g_in, beta=b_in, initial_state=None,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        check(f"layer{i}.S_final", S_rec[0], f"layer{i}.S_final")
        li += 1

    q_rope, k_rope = rec["rope"][0]
    attn_pre, attn_probs = rec["attn"][0]
    check("layer3.q_rope", q_rope[0].transpose(0, 1), "layer3.q_rope")
    check("layer3.k_rope", k_rope[0].transpose(0, 1), "layer3.k_rope")
    check("layer3.attn_probs", attn_probs[0], "layer3.attn_probs")
    check("layer3.attn_out_pre_gate", attn_pre.reshape(T, nq * dh), "layer3.attn_out_pre_gate")
    check("layer3.attn_out_post_gate", hooks_out["layer3.postgate"][0], "layer3.attn_out_post_gate")
    qg = layers[3].self_attn.q_proj(hooks_out["layer3.norm1"])
    check("layer3.gate", qg.view(T, nq, 2 * dh)[..., dh:].reshape(T, nq * dh), "layer3.gate")
    check("final_hidden", hooks_out["final"][0], "final_hidden")
    check("logits", hooks_out["logits"][0], "logits")

    # backward probes (torch autograd vs hand-written VJP fixtures)
    check("grad.logits", logits.grad[0], "grad.logits", TOL_BWD)
    check("grad.final_hidden", fh.grad[0], "grad.final_hidden", TOL_BWD)
    check(
        "grad.layer0.in_proj_qkv",
        layers[0].linear_attn.in_proj_qkv.weight.grad,
        "grad.model.layers.0.linear_attn.in_proj_qkv.weight",
        TOL_BWD,
    )
    check(
        "grad.layer3.q_proj",
        layers[3].self_attn.q_proj.weight.grad,
        "grad.model.layers.3.self_attn.q_proj.weight",
        TOL_BWD,
    )

    # ---- decode case (per linear_attention layer) ----
    for i, layer in enumerate(layers):
        if cfg["layer_types"][i] != "linear_attention":
            continue
        la = layer.linear_attn
        dp = f"decode.layer{i}."
        x_t = t(dp + "hidden_in", fx)
        with torch.no_grad():
            n1 = layer.input_layernorm(x_t.unsqueeze(0).unsqueeze(0))
            qkv = la.in_proj_qkv(n1).transpose(1, 2)  # (1, conv_dim, 1)
            conv_state = t(dp + "conv_state_in", fx).transpose(0, 1).unsqueeze(0).clone()
            conv_out = ref.causal_conv1d_update(
                qkv, conv_state, la.conv1d.weight.squeeze(1), la.conv1d.bias, la.activation
            )
            check(dp + "qkv_pre_conv", qkv[0, :, 0], dp + "qkv_pre_conv")
            check(dp + "conv_state_out", conv_state[0].transpose(0, 1), dp + "conv_state_out")
            q_c, k_c, v_c = torch.split(conv_out[0, :, 0], [kd, kd, vd])
            check(dp + "q_post_conv", q_c, dp + "q_post_conv")
            check(dp + "k_post_conv", k_c, dp + "k_post_conv")
            check(dp + "v_post_conv", v_c, dp + "v_post_conv")
            z = la.in_proj_z(n1).reshape(1, 1, nv, dv)
            b = la.in_proj_b(n1)
            a = la.in_proj_a(n1)
            beta = b.sigmoid()
            g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
            qh = q_c.reshape(1, 1, nk, dk).repeat_interleave(nv // nk, dim=2)
            kh = k_c.reshape(1, 1, nk, dk).repeat_interleave(nv // nk, dim=2)
            vh = v_c.reshape(1, 1, nv, dv)
            check(dp + "q_l2", ref.l2norm(qh, dim=-1, eps=1e-6)[0, 0], dp + "q_l2")
            check(dp + "k_l2", ref.l2norm(kh, dim=-1, eps=1e-6)[0, 0], dp + "k_l2")
            check(dp + "beta", beta[0, 0], dp + "beta")
            check(dp + "g", g[0, 0], dp + "g")
            out, S_out = orig_recur(
                qh, kh, vh, g=g, beta=beta,
                initial_state=t(dp + "S_in", fx).unsqueeze(0),
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            check(dp + "delta_out", out[0, 0], dp + "delta_out")
            check(dp + "S_out", S_out[0], dp + "S_out")
            gated = la.norm(out.reshape(-1, dv), z.reshape(-1, dv)).reshape(1, 1, vd)
            check(dp + "gated_out", gated[0, 0], dp + "gated_out")
            mixer = la.out_proj(gated)
            check(dp + "mixer_out", mixer[0, 0], dp + "mixer_out")
            h = x_t + mixer[0, 0]
            mlp_out = layer.mlp(layer.post_attention_layernorm(h.unsqueeze(0).unsqueeze(0)))
            check(dp + "mlp_out", mlp_out[0, 0], dp + "mlp_out")
            check(dp + "hidden_out", h + mlp_out[0, 0], dp + "hidden_out")

    nfail = 0
    for name, err, tol in results:
        status = "PASS" if err <= tol else "FAIL"
        if err > tol:
            nfail += 1
        print(f"  [{status}] {name}: max abs err {err:.3e} (tol {tol:.0e})")
    print(
        f"TORCH CROSS-CHECK {'PASS' if nfail == 0 else 'FAIL'} "
        f"({len(results)} tensors, torch {torch.__version__})"
    )
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
