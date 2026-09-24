"""DSv2InKernelTPEP: the host-side decode engine driving the megakernel.

Owns the symmetric-memory buffers used by the in-kernel collectives, the
compressed MLA KV cache, the TP/EP-sharded weight tensors, and the per-step
launch. `step()` advances one decode token; `generate()` wraps it in a greedy
loop. All 8 ranks run the same code under torchrun.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as sm


from dsv2mk.runtime.weight_loader import load_config, load_stacked_weights
from dsv2mk.runtime.weight_cache import (
    cache_enabled,
    cache_signature,
    default_cache_dir,
    save_cache,
    try_load_cached,
)
from dsv2mk.kernels.megakernel import attn_prologue_multilayer


def _shard_w_qb(w: torch.Tensor, tp_size: int, tp_rank: int,
                 N: int, qk_nope_dim: int, R: int) -> torch.Tensor:
    if tp_size == 1:
        return w
    per_head = qk_nope_dim + R
    per_rank_N = N // tp_size
    per_rank_rows = per_rank_N * per_head
    start = tp_rank * per_rank_rows
    return w[:, start:start + per_rank_rows, :].contiguous()


def _shard_w_o(w: torch.Tensor, tp_size: int, tp_rank: int,
               v_head_dim: int) -> torch.Tensor:
    if tp_size == 1:
        return w
    L, H, last = w.shape
    N = last // v_head_dim
    per_rank_N = N // tp_size
    per_rank_last = per_rank_N * v_head_dim
    start = tp_rank * per_rank_last
    return w[:, :, start:start + per_rank_last].contiguous()


def _shard_heads_axis1(w: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    if tp_size == 1:
        return w
    n = w.shape[1]
    per_rank = n // tp_size
    start = tp_rank * per_rank
    return w[:, start:start + per_rank].contiguous()


class DSv2InKernelTPEP:

    def __init__(self, snapshot_dir: str, *,
                 tp_size: Optional[int] = None,
                 ep_size: Optional[int] = None,
                 S_max: int = 512,
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.bfloat16,
                 use_fp8_q1: bool = False,
                 use_fp8_q2: bool = False,
                 use_fp8_routed: bool = False,
                 skip_hf_load: bool = False,
                 max_layers: Optional[int] = None,
                 include_layer0: bool = False,
                 verbose: bool = False):
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.tp_size = tp_size if tp_size is not None else self.world_size
        self.ep_size = ep_size if ep_size is not None else self.world_size
        self.tp_rank = self.rank if self.tp_size > 1 else 0
        self.ep_rank = self.rank if self.ep_size > 1 else 0
        assert self.tp_size <= self.world_size and self.ep_size <= self.world_size

        if device is None:
            device = f"cuda:{self.rank}" if self.world_size > 1 else "cuda"
        if self.world_size > 1:
            torch.cuda.set_device(self.rank)
        self.device = device
        self.dtype = dtype
        self.S_max = S_max
        self.use_fp8_q1 = use_fp8_q1
        self.use_fp8_q2 = use_fp8_q2
        self.use_fp8_routed = use_fp8_routed
        self.verbose = verbose

        from transformers.models.deepseek_v2 import DeepseekV2Config, DeepseekV2ForCausalLM
        hf_cfg = DeepseekV2Config.from_pretrained(snapshot_dir)
        if not hasattr(hf_cfg, "rope_parameters") or hf_cfg.rope_parameters is None:
            rs = hf_cfg.rope_scaling if hasattr(hf_cfg, "rope_scaling") else None
            if rs is not None:
                hf_cfg.rope_parameters = {**rs,
                                           "rope_type": rs.get("type", "yarn"),
                                           "rope_theta": hf_cfg.rope_theta}
        self.hf_cfg = hf_cfg

        if skip_hf_load:
            self.model = self._load_hf_aux_only(snapshot_dir, hf_cfg, device, dtype)
        else:
            self.model = DeepseekV2ForCausalLM.from_pretrained(
                snapshot_dir, dtype=dtype, device_map=device,
                attn_implementation="eager", config=hf_cfg,
            )
            self.model.eval()

        self.cfg = load_config(snapshot_dir)
        self.include_layer0 = include_layer0
        if max_layers is not None:
            start = 0 if include_layer0 else self.cfg.first_k_dense_replace
            self.layer_range = (start, start + max_layers)
        else:
            start = 0 if include_layer0 else self.cfg.first_k_dense_replace
            self.layer_range = (start, self.cfg.num_hidden_layers)
        self.L_moe = self.layer_range[1] - self.layer_range[0]

        fp8_stages = set()
        if use_fp8_q1: fp8_stages.add("shared_q1")
        if use_fp8_q2: fp8_stages.add("shared_q2")
        if use_fp8_routed: fp8_stages.add("routed_moe")
        fp8_stages = fp8_stages or None

        if self.rank == 0:
            print(f"[rank {self.rank}] loading megakernel weights "
                  f"(L_moe={self.L_moe}, ep={self.ep_size}/{self.ep_rank}, "
                  f"tp={self.tp_size}/{self.tp_rank}, fp8={fp8_stages}, "
                  f"include_layer0={self.include_layer0})")

        cfg = self.cfg
        H = cfg.hidden_size
        N_full = cfg.num_attention_heads
        Lkv, R = cfg.kv_lora_rank, cfg.qk_rope_head_dim
        qk_nope_dim = cfg.qk_nope_head_dim
        v_head_dim = cfg.v_head_dim

        cache_dir = default_cache_dir()
        sig = cache_signature(
            snapshot_dir, self.layer_range, dtype, fp8_stages,
            self.ep_size, self.ep_rank, self.tp_size, self.tp_rank,
            self.include_layer0,
        )
        cached = try_load_cached(
            cache_dir, sig, device, self.tp_rank, self.ep_rank,
        ) if cache_enabled() else None

        if cached is not None:
            self.weights = cached
            self.weights["config"] = cfg
            if "layer_range" not in self.weights:
                self.weights["layer_range"] = self.layer_range
            if self.rank == 0:
                print(f"[rank {self.rank}] weight cache HIT: "
                      f"{cache_dir}/rank{self.tp_rank}-{self.ep_rank}_"
                      f"{sig[:16]}.safetensors")
        else:
            if self.rank == 0:
                reason = "disabled" if not cache_enabled() else "miss"
                print(f"[rank {self.rank}] weight cache {reason}: falling back "
                      f"to slow loader (signature={sig[:16]}…)")
            self.weights = load_stacked_weights(
                snapshot_dir, layer_range=self.layer_range,
                device=device, dtype=dtype,
                fp8_stages=fp8_stages,
                ep_size=self.ep_size, ep_rank=self.ep_rank,
                include_layer0=self.include_layer0,
            )

            if self.tp_size > 1:
                assert N_full % self.tp_size == 0
                self.weights["W_qb"] = _shard_w_qb(
                    self.weights["W_qb"], self.tp_size, self.tp_rank,
                    N_full, qk_nope_dim, R,
                )
                self.weights["W_UK"] = _shard_heads_axis1(self.weights["W_UK"], self.tp_size, self.tp_rank)
                self.weights["W_UV"] = _shard_heads_axis1(self.weights["W_UV"], self.tp_size, self.tp_rank)
                self.weights["W_O"] = _shard_w_o(self.weights["W_O"], self.tp_size, self.tp_rank, v_head_dim)

            if cache_enabled():
                if self.rank == 0:
                    print(f"[rank {self.rank}] saving weight cache → "
                          f"{cache_dir}/rank{self.tp_rank}-{self.ep_rank}_"
                          f"{sig[:16]}.safetensors")
                save_cache(
                    cache_dir, sig, self.weights, self.tp_rank, self.ep_rank,
                )

        self.N_per_rank = N_full // self.tp_size

        self.kv_cache_c = torch.zeros(self.L_moe, S_max, Lkv, dtype=dtype, device=device)
        self.kv_cache_pe = torch.zeros(self.L_moe, S_max, R, dtype=dtype, device=device)
        self._res_zero_stack = torch.zeros(self.L_moe, H, dtype=dtype, device=device)
        self.moe_routed_acc_f32 = torch.zeros(self.L_moe, H, dtype=torch.float32, device=device)
        _KI = self.cfg.num_experts_per_tok * self.cfg.moe_intermediate_size
        _IS = self.cfg.I_shared_extended if self.include_layer0 else self.cfg.I_shared
        self.moe_splitk_partials = torch.zeros(
            self.L_moe, 2 * 2 * (_KI + _IS), dtype=torch.float32, device=device)

        if self.world_size > 1:
            group = dist.distributed_c10d._get_default_group()
            self.attn_proj_symm_local = sm.empty(self.L_moe, H, dtype=dtype, device=device)
            self.attn_proj_symm_handle = sm.rendezvous(
                self.attn_proj_symm_local, group=group.group_name,
            )
            self.tp_sync_local = sm.empty(self.L_moe, 132, dtype=torch.int32, device=device)
            self.tp_sync_handle = sm.rendezvous(self.tp_sync_local, group=group.group_name)
            self.moe_out_symm_local = sm.empty(self.L_moe, H, dtype=dtype, device=device)
            self.moe_out_symm_handle = sm.rendezvous(
                self.moe_out_symm_local, group=group.group_name,
            )
            self.ep_sync_local = sm.empty(self.L_moe, 132, dtype=torch.int32, device=device)
            self.ep_sync_handle = sm.rendezvous(self.ep_sync_local, group=group.group_name)
            self.attn_proj_symm_local.zero_()
            self.tp_sync_local.zero_()
            self.moe_out_symm_local.zero_()
            self.ep_sync_local.zero_()
            self.attn_proj_reduced = torch.zeros(self.L_moe, H, dtype=dtype, device=device)
            self.tp_l1_sync = torch.zeros(1, dtype=torch.int32, device=device)
            dist.barrier(device_ids=[self.rank])
            torch.cuda.synchronize()
        else:
            self.attn_proj_symm_local = None
            self.attn_proj_symm_handle = None
            self.tp_sync_local = None
            self.tp_sync_handle = None
            self.moe_out_symm_local = None
            self.moe_out_symm_handle = None
            self.ep_sync_local = None
            self.ep_sync_handle = None
            self.attn_proj_reduced = None
            self.tp_l1_sync = None

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=True)

        self._rotary = self.model.model.rotary_emb
        self._embed = self.model.model.embed_tokens
        self._layer0 = self.model.model.layers[0]
        self._final_norm = self.model.model.norm
        self._lm_head = self.model.lm_head

        self.use_inline_embed = (
            self.include_layer0
            and os.environ.get("DSV2MK_DISABLE_INLINE_EMBED", "0") != "1"
        )
        self.use_inline_rotary = (
            self.include_layer0
            and os.environ.get("DSV2MK_DISABLE_INLINE_ROTARY", "0") != "1"
        )
        if self.use_inline_embed:
            with torch.no_grad():
                self.embed_weight = (
                    self._embed.weight.detach().to(dtype=dtype, device=device).contiguous()
                )
            self.token_id_buf = torch.zeros(1, dtype=torch.int32, device=device)
            self._h_in_dummy = torch.zeros(self.cfg.hidden_size, dtype=dtype, device=device)
            self._hidden_dummy = torch.zeros(
                (1, 1, self.cfg.hidden_size), dtype=dtype, device=device
            )
        else:
            self.embed_weight = None
            self.token_id_buf = None
            self._h_in_dummy = None
            self._hidden_dummy = None

        if self.use_inline_rotary:
            R = self.cfg.qk_rope_head_dim
            with torch.no_grad():
                inv_freq = self._rotary.inv_freq.detach().to(
                    dtype=torch.float32, device=device,
                ).contiguous()
            assert inv_freq.shape == (R // 2,), (
                f"HF rotary inv_freq shape {inv_freq.shape} != (R/2={R//2},); "
                "model rotary config mismatch."
            )
            self.inv_freq = inv_freq
            self.attention_scaling = float(self._rotary.attention_scaling)
            self.position_id_buf = torch.zeros(1, dtype=torch.int32, device=device)
            self._cos_dummy = torch.zeros(R // 2, dtype=dtype, device=device)
            self._sin_dummy = torch.zeros(R // 2, dtype=dtype, device=device)
            if self.rank == 0:
                print(
                    f"[rank 0] B-2 inline rotary: inv_freq[0..3]="
                    f"{inv_freq[:4].tolist()} attention_scaling={self.attention_scaling:.6f}"
                )
        else:
            self.inv_freq = None
            self.attention_scaling = 1.0
            self.position_id_buf = None
            self._cos_dummy = None
            self._sin_dummy = None

        self.use_inline_lm_head = True
        if self.use_inline_lm_head:
            with torch.no_grad():
                self.gamma_final = (
                    self._final_norm.weight.detach().to(dtype=dtype, device=device)
                    .contiguous()
                )
                self.lm_head_weight = (
                    self._lm_head.weight.detach().to(dtype=dtype, device=device)
                    .contiguous()
                )
            assert self.gamma_final.shape == (self.cfg.hidden_size,)
            assert self.lm_head_weight.shape == (
                self.cfg.vocab_size, self.cfg.hidden_size,
            ), (
                f"lm_head_weight shape {self.lm_head_weight.shape} "
                f"!= ({self.cfg.vocab_size}, {self.cfg.hidden_size})"
            )
            if self.use_inline_embed:
                self.next_token_buf = self.token_id_buf
                loop_mode = "aliased to token_id_buf (pure on-device decode)"
            else:
                self.next_token_buf = torch.zeros(
                    1, dtype=torch.int32, device=device
                )
                loop_mode = (
                    "standalone int32 buffer (B-1 off; host .item() per step "
                    "for HF layer 0)"
                )
            if self.rank == 0:
                print(
                    f"[rank 0] B-3 inline lm_head: "
                    f"gamma_final={tuple(self.gamma_final.shape)} {self.gamma_final.dtype} "
                    f"lm_head_weight={tuple(self.lm_head_weight.shape)} {self.lm_head_weight.dtype} "
                    f"V={self.cfg.vocab_size} H={self.cfg.hidden_size}; "
                    f"next_token_buf {loop_mode}."
                )
        else:
            self.gamma_final = None
            self.lm_head_weight = None
            self.next_token_buf = None

        self.hf_past_cache = None
        self.cache_pos = 0
        self._last_h_final = None

        with torch.no_grad():
            _w = torch.zeros(128, 128, dtype=dtype, device=device)
            _ = torch.matmul(_w, _w)
            del _w
            torch.cuda.synchronize()

        self._precompile_warmup()

    def _precompile_warmup(self):
        if self.world_size <= 1:
            return
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_ptr
        import cuda.bindings.driver as cuda
        from dsv2mk.kernels.megakernel import _get_compiled, _ct

        cfg = self.cfg
        H = cfg.hidden_size
        Lkv, R = cfg.kv_lora_rank, cfg.qk_rope_head_dim
        qk_nope_dim = cfg.qk_nope_head_dim
        v_head_dim = cfg.v_head_dim
        Lq_slice = cfg.q_lora_rank if cfg.q_lora_rank is not None else 0
        OUT_QKVA = Lq_slice + Lkv + R
        OUT_QB = self.N_per_rank * (qk_nope_dim + R)
        Lq = cfg.q_lora_rank if cfg.q_lora_rank is not None else 0
        if self.rank == 0:
            print(f"[rank 0] precompile warmup: triggering compile on all {self.world_size} ranks…")
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)
        import time as _t
        t0 = _t.time()
        _get_compiled(
            _ct(self.dtype), H, OUT_QKVA, Lq, OUT_QB, Lkv, R, self.L_moe, self.S_max,
            self.N_per_rank, qk_nope_dim, v_head_dim,
            cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.n_group, cfg.topk_group,
            cfg.moe_intermediate_size,
            cfg.I_shared_extended if self.include_layer0 else cfg.I_shared,
            cfg.routed_scaling_factor,
            num_threads=256, threads_per_output=32,
            num_sms=132, stream=stream,
            tp_size=self.tp_size, ep_size=self.ep_size, ep_rank=self.ep_rank,
            V=cfg.vocab_size,
            attention_scaling=self.attention_scaling,
        )
        print(f"[rank {self.rank}] compile took {_t.time() - t0:.1f}s")
        dist.barrier(device_ids=[self.rank])
        torch.cuda.synchronize()
        if self.rank == 0:
            print(f"[rank 0] all ranks finished compile; barriered.")

    @staticmethod
    def _load_hf_aux_only(snapshot_dir, hf_cfg, device, dtype):
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from transformers.models.deepseek_v2 import DeepseekV2ForCausalLM
        from safetensors import safe_open
        import json
        from pathlib import Path

        with init_empty_weights():
            model = DeepseekV2ForCausalLM(hf_cfg)

        idx_path = Path(snapshot_dir) / "model.safetensors.index.json"
        weight_map = json.loads(idx_path.read_text())["weight_map"]

        wanted_prefixes = (
            "model.embed_tokens.",
            "model.layers.0.",
            "model.norm.",
            "lm_head.",
        )
        shards: dict[str, list[str]] = {}
        for k, fname in weight_map.items():
            if any(k.startswith(p) for p in wanted_prefixes):
                shards.setdefault(fname, []).append(k)

        def _load_named(name):
            fn = weight_map[name]
            with safe_open(str(Path(snapshot_dir) / fn), framework="pt", device="cpu") as g:
                return g.get_tensor(name)

        def _block_dequant(w_fp8, s_inv):
            N, K = w_fp8.shape
            w = w_fp8.to(torch.float32)
            s = s_inv.to(torch.float32)
            s_full = s.repeat_interleave(128, dim=0)[:N].repeat_interleave(128, dim=1)[:, :K]
            return (w * s_full).to(dtype)

        for fname, keys in shards.items():
            path = Path(snapshot_dir) / fname
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for k in keys:
                    if k.endswith("weight_scale_inv"):
                        continue
                    t = f.get_tensor(k)
                    sk = (k[:-len("weight")] + "weight_scale_inv"
                          if k.endswith("weight") else None)
                    if t.dtype == torch.float8_e4m3fn and sk in weight_map:
                        t = _block_dequant(t, _load_named(sk))
                    else:
                        t = t.to(dtype)
                    set_module_tensor_to_device(model, k, device=device, value=t, dtype=dtype)
        model.eval()
        for m in (model.model.embed_tokens, model.model.layers[0],
                  model.model.norm, model.lm_head):
            m.to(device=device, dtype=dtype)
        return model

    def reset(self):
        self.cache_pos = 0
        self.hf_past_cache = None
        self._last_h_final = None
        self.kv_cache_c.zero_()
        self.kv_cache_pe.zero_()
        if self.world_size > 1:
            self.attn_proj_symm_local.zero_()
            self.tp_sync_local.zero_()
            self.moe_out_symm_local.zero_()
            self.ep_sync_local.zero_()
            dist.barrier(device_ids=[self.rank])
            torch.cuda.synchronize()

    @torch.no_grad()
    def step(self, token_id: int | None) -> torch.Tensor | None:
        cfg = self.cfg
        H = cfg.hidden_size
        Lkv, R = cfg.kv_lora_rank, cfg.qk_rope_head_dim
        qk_nope_dim = cfg.qk_nope_head_dim
        v_head_dim = cfg.v_head_dim
        E = cfg.n_routed_experts
        K_topk = cfg.num_experts_per_tok
        n_group = cfg.n_group
        topk_group = cfg.topk_group
        I_r = cfg.moe_intermediate_size
        I_s = cfg.I_shared_extended if self.include_layer0 else cfg.I_shared

        if self.verbose:
            import time as _t
            t0 = _t.time()
        if self.use_inline_embed:
            if token_id is not None:
                self.token_id_buf.fill_(int(token_id))
            hidden = self._hidden_dummy
        else:
            assert token_id is not None, \
                "step(None) is only valid when use_inline_embed=True " \
                "(the kernel needs to read its previous prediction from token_id_buf)"
            ids = torch.tensor([[token_id]], device=self.device)
            hidden = self._embed(ids)
        if self.verbose:
            torch.cuda.synchronize()
            if self.rank == 0:
                print(f"[step {self.cache_pos}] embed {_t.time() - t0:.3f}s", flush=True)

        if self.use_inline_rotary:
            self.position_id_buf.fill_(int(self.cache_pos))
            cos_v = self._cos_dummy
            sin_v = self._sin_dummy
            position_embeddings = None
        else:
            position_ids = torch.tensor([[self.cache_pos]], device=self.device)
            position_embeddings = self._rotary(hidden, position_ids)
        if self.include_layer0:
            if self.use_inline_embed:
                h_in = self._h_in_dummy
            else:
                h_in = hidden.view(-1).contiguous().to(self.dtype)
        else:
            from transformers.cache_utils import DynamicCache
            if self.hf_past_cache is None:
                self.hf_past_cache = DynamicCache()
            if self.verbose:
                t1 = _t.time()
            position_ids_l0 = torch.tensor([[self.cache_pos]], device=self.device)
            layer0_out = self._layer0(
                hidden, attention_mask=None, position_ids=position_ids_l0,
                past_key_values=self.hf_past_cache, cache_position=position_ids_l0[0],
                position_embeddings=position_embeddings,
            )
            if self.verbose:
                torch.cuda.synchronize()
                if self.rank == 0:
                    print(f"[step {self.cache_pos}] layer0 {_t.time() - t1:.3f}s", flush=True)
            hidden = layer0_out[0] if isinstance(layer0_out, tuple) else layer0_out
            h_in = hidden.view(-1).contiguous().to(self.dtype)

        if not self.use_inline_rotary:
            cos_v = position_embeddings.real.squeeze().to(self.dtype).contiguous()
            sin_v = position_embeddings.imag.squeeze().to(self.dtype).contiguous()
        rs = cfg.rope_parameters or {}
        mscale = float(rs.get("mscale", 1.0))
        mscale_all_dim = float(rs.get("mscale_all_dim", 1.0))
        factor = float(rs.get("factor", 1.0))
        import math as _math
        if factor > 1.0:
            yarn_mscale = 0.1 * mscale_all_dim * _math.log(factor) + 1.0
        else:
            yarn_mscale = 1.0
        softmax_scale = (1.0 / _math.sqrt(qk_nope_dim + R)) * (yarn_mscale ** 2)

        mc_attn = self.attn_proj_symm_handle.multicast_ptr if self.world_size > 1 else 0
        mc_sync = self.tp_sync_handle.multicast_ptr if self.world_size > 1 else 0
        mc_moe = self.moe_out_symm_handle.multicast_ptr if self.world_size > 1 else 0
        mc_ep = self.ep_sync_handle.multicast_ptr if self.world_size > 1 else 0


        if self.verbose:
            import time as _t2
            t2 = _t2.time()
        outs = attn_prologue_multilayer(
            h_in, self._res_zero_stack,
            self.weights["gamma1"], self.weights["W_qkva"],
            self.weights["gamma2"], self.weights["W_qb"], self.weights["gamma3"],
            cos_v, sin_v,
            self.kv_cache_c, self.kv_cache_pe,
            self.weights["W_UK"], self.weights["W_UV"], self.weights["W_O"],
            self.weights["gamma4"], self.weights["W_gate_router"],
            self.weights["W_gate_routed"], self.weights["W_up_routed"],
            self.weights["W_down_routed"],
            self.weights["W_gate_shared"], self.weights["W_up_shared"],
            self.weights["W_down_shared"],
            cache_pos=self.cache_pos,
            N=self.N_per_rank, qk_nope_dim=qk_nope_dim, v_head_dim=v_head_dim,
            Lkv=Lkv, R=R,
            E=E, K_topk=K_topk, n_group=n_group, topk_group=topk_group,
            I_routed=I_r, I_shared=I_s, routed_scaling=cfg.routed_scaling_factor,
            softmax_scale=softmax_scale, eps=cfg.rms_norm_eps,
            w_gate_shared_fp8=self.weights.get("W_gate_shared_fp8"),
            w_up_shared_fp8=self.weights.get("W_up_shared_fp8"),
            sfa_gate_q1=self.weights.get("W_gate_shared_sf"),
            sfa_up_q1=self.weights.get("W_up_shared_sf"),
            w_down_shared_fp8=self.weights.get("W_down_shared_fp8"),
            sfa_down_q2=self.weights.get("W_down_shared_sf_hw"),
            w_gate_routed_fp8=self.weights.get("W_gate_routed_fp8"),
            w_up_routed_fp8=self.weights.get("W_up_routed_fp8"),
            w_down_routed_fp8=self.weights.get("W_down_routed_fp8"),
            sf_gate_routed=self.weights.get("W_gate_routed_sf"),
            sf_up_routed=self.weights.get("W_up_routed_sf"),
            sf_down_routed=self.weights.get("W_down_routed_sf"),
            sf_gate_routed_hw=self.weights.get("W_gate_routed_sf_hw"),
            sf_up_routed_hw=self.weights.get("W_up_routed_sf_hw"),
            sf_down_routed_hw=self.weights.get("W_down_routed_sf_hw"),
            attn_proj_symm_local=self.attn_proj_symm_local,
            attn_proj_symm_mc_ptr_int=mc_attn,
            tp_sync_local=self.tp_sync_local,
            tp_sync_mc_ptr_int=mc_sync,
            tp_size=self.tp_size,
            attn_proj_reduced=self.attn_proj_reduced,
            moe_routed_acc_f32=self.moe_routed_acc_f32,
            moe_splitk_partials=self.moe_splitk_partials,
            tp_l1_sync=self.tp_l1_sync,
            moe_out_symm_local=self.moe_out_symm_local,
            moe_out_symm_mc_ptr_int=mc_moe,
            ep_sync_local=self.ep_sync_local,
            ep_sync_mc_ptr_int=mc_ep,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            embed_weight=self.embed_weight,
            token_id_buf=self.token_id_buf,
            inv_freq=self.inv_freq,
            position_id_buf=self.position_id_buf,
            attention_scaling=self.attention_scaling,
            gamma_final=self.gamma_final,
            lm_head_weight=self.lm_head_weight,
            next_token_buf=self.next_token_buf,
        )
        if self.verbose:
            torch.cuda.synchronize()
            if self.rank == 0:
                print(f"[step {self.cache_pos}] megakernel {_t2.time() - t2:.3f}s", flush=True)
        h_final = outs[12]
        self._last_h_final = h_final.detach().clone()

        if self.use_inline_lm_head:
            self.cache_pos += 1
            return None
        if self.verbose:
            t3 = _t2.time()
        h_normed = self._final_norm(h_final.view(1, 1, -1))
        logits = self._lm_head(h_normed).view(-1)
        if self.verbose:
            torch.cuda.synchronize()
            if self.rank == 0:
                print(f"[step {self.cache_pos}] norm+lm_head {_t2.time() - t3:.3f}s", flush=True)
        self.cache_pos += 1
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens: int = 16, greedy: bool = True):
        self.reset()
        out_ids = list(prompt_ids)
        if self.use_inline_lm_head and self.use_inline_embed:
            for tid in prompt_ids:
                self.token_id_buf.fill_(int(tid))
                self.step(int(tid))
            first_gen = int(self.next_token_buf.item())
            out_ids.append(first_gen)
            if first_gen != self.tokenizer.eos_token_id:
                for _ in range(max_new_tokens - 1):
                    self.step(None)
                    next_tok = int(self.next_token_buf.item())
                    out_ids.append(next_tok)
                    if next_tok == self.tokenizer.eos_token_id:
                        break
            return out_ids
        if self.use_inline_lm_head:
            for tid in prompt_ids:
                self.step(int(tid))
            first_gen = int(self.next_token_buf.item())
            out_ids.append(first_gen)
            if first_gen != self.tokenizer.eos_token_id:
                for _ in range(max_new_tokens - 1):
                    self.step(first_gen)
                    next_tok = int(self.next_token_buf.item())
                    out_ids.append(next_tok)
                    if next_tok == self.tokenizer.eos_token_id:
                        break
                    first_gen = next_tok
            return out_ids
        last_logits = None
        for tid in prompt_ids:
            last_logits = self.step(int(tid))
        for _ in range(max_new_tokens):
            if greedy:
                next_tok = int(last_logits.argmax().item())
            else:
                next_tok = int(last_logits.argmax().item())
            out_ids.append(next_tok)
            if next_tok == self.tokenizer.eos_token_id:
                break
            last_logits = self.step(next_tok)
        return out_ids

