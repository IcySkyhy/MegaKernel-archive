"""Load DeepSeek-V2 safetensors and shard them for TP=8 / EP=8.

Produces the layer-stacked, rank-local tensors the megakernel indexes by
pointer arithmetic, optionally quantising the Q1/Q2/routed-expert stages to
block-scaled FP8 E4M3 with hardware-layout scale factors.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from safetensors import safe_open


@dataclass
class DSv2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: Optional[int]
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    moe_intermediate_size: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    first_k_dense_replace: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_parameters: Optional[Dict]
    scoring_func: str
    intermediate_size: Optional[int] = None

    @property
    def I_shared(self) -> int:
        return self.n_shared_experts * self.moe_intermediate_size

    @property
    def I_shared_extended(self) -> int:
        if self.intermediate_size is None:
            return self.I_shared
        return max(self.intermediate_size, self.I_shared)

    @property
    def OUT_QKVA(self) -> int:
        Lq = self.q_lora_rank if self.q_lora_rank is not None else self.hidden_size
        return Lq + self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def OUT_QB(self) -> int:
        return self.num_attention_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)

    @property
    def Lq_effective(self) -> int:
        return self.q_lora_rank if self.q_lora_rank is not None else self.hidden_size


def load_config(snapshot_dir: str | Path) -> DSv2Config:
    p = Path(snapshot_dir) / "config.json"
    cfg = json.loads(p.read_text())
    rope_params = cfg.get("rope_scaling", None) or cfg.get("rope_parameters", None)
    return DSv2Config(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        q_lora_rank=cfg.get("q_lora_rank", None),
        kv_lora_rank=cfg["kv_lora_rank"],
        qk_nope_head_dim=cfg["qk_nope_head_dim"],
        qk_rope_head_dim=cfg["qk_rope_head_dim"],
        v_head_dim=cfg["v_head_dim"],
        n_routed_experts=cfg["n_routed_experts"],
        n_shared_experts=cfg.get("n_shared_experts", 1),
        num_experts_per_tok=cfg["num_experts_per_tok"],
        n_group=cfg.get("n_group", 1) or 1,
        topk_group=cfg.get("topk_group", 1) or 1,
        moe_intermediate_size=cfg["moe_intermediate_size"],
        routed_scaling_factor=float(cfg.get("routed_scaling_factor", 1.0)),
        norm_topk_prob=bool(cfg.get("norm_topk_prob", False)),
        first_k_dense_replace=int(cfg.get("first_k_dense_replace", 0)),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        vocab_size=cfg["vocab_size"],
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
        rope_parameters=rope_params,
        scoring_func=cfg.get("scoring_func", "softmax"),
        intermediate_size=cfg.get("intermediate_size", None),
    )


class _WeightFetcher:

    def __init__(self, snapshot_dir: str | Path):
        self.snapshot_dir = Path(snapshot_dir)
        idx_path = self.snapshot_dir / "model.safetensors.index.json"
        if idx_path.exists():
            self.weight_map = json.loads(idx_path.read_text())["weight_map"]
        else:
            self.weight_map = None
        self._open_files: Dict[str, "safe_open"] = {}

    def _file_for(self, key: str) -> Path:
        if self.weight_map is None:
            files = list(self.snapshot_dir.glob("*.safetensors"))
            assert len(files) == 1, f"expected 1 safetensors file, found {len(files)}"
            return files[0]
        return self.snapshot_dir / self.weight_map[key]

    def get(self, key: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu") -> torch.Tensor:
        f = str(self._file_for(key))
        if f not in self._open_files:
            self._open_files[f] = safe_open(f, framework="pt", device=device)
        sf = self._open_files[f]
        t = sf.get_tensor(key)
        if t.dtype != dtype:
            t = t.to(dtype)
        return t

    def has(self, key: str) -> bool:
        if self.weight_map is None:
            return True
        return key in self.weight_map


def _permute_rope_rows(W: torch.Tensor, R: int) -> torch.Tensor:
    assert W.dim() == 2, f"_permute_rope_rows expects 2D weight, got {W.shape}"
    out_dim = W.shape[0]
    assert out_dim >= R
    head = W[:out_dim - R, :]
    rope = W[out_dim - R:, :]
    rope = rope.view(R // 2, 2, -1)
    rope = rope.transpose(0, 1).contiguous()
    rope = rope.view(R, -1)
    return torch.cat([head, rope], dim=0).contiguous()


def _process_kv_b_proj(
    kv_b_weight: torch.Tensor, N: int, qk_nope_dim: int, v_head_dim: int, Lkv: int
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = qk_nope_dim + v_head_dim
    assert kv_b_weight.shape == (N * head_dim, Lkv), kv_b_weight.shape
    kv_b = kv_b_weight.view(N, head_dim, Lkv)
    w_uk_hf = kv_b[:, :qk_nope_dim, :]
    w_uv_hf = kv_b[:, qk_nope_dim:qk_nope_dim + v_head_dim, :]
    w_uk = w_uk_hf.transpose(1, 2).contiguous()
    w_uv = w_uv_hf.contiguous()
    return w_uk, w_uv


def _process_qkva(
    cfg: DSv2Config, fetcher: _WeightFetcher, layer: int
) -> torch.Tensor:
    prefix = f"model.layers.{layer}.self_attn"
    kv_a = fetcher.get(f"{prefix}.kv_a_proj_with_mqa.weight")
    kv_a = _permute_rope_rows(kv_a, cfg.qk_rope_head_dim)
    if cfg.q_lora_rank is not None:
        q_a = fetcher.get(f"{prefix}.q_a_proj.weight")
        return torch.cat([q_a, kv_a], dim=0).contiguous()
    return kv_a.contiguous()


def _process_q_b(
    cfg: DSv2Config, fetcher: _WeightFetcher, layer: int
) -> torch.Tensor:
    prefix = f"model.layers.{layer}.self_attn"
    if cfg.q_lora_rank is not None:
        W = fetcher.get(f"{prefix}.q_b_proj.weight")
    else:
        W = fetcher.get(f"{prefix}.q_proj.weight")
    N = cfg.num_attention_heads
    qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    R = cfg.qk_rope_head_dim
    assert W.shape[0] == N * qk_head_dim, f"unexpected q_b/q_proj rows: {W.shape}"
    W = W.view(N, qk_head_dim, -1)
    head_nope = W[:, :qk_head_dim - R, :]
    head_rope = W[:, qk_head_dim - R:, :]
    head_rope = head_rope.view(N, R // 2, 2, -1).transpose(1, 2).contiguous().view(N, R, -1)
    W = torch.cat([head_nope, head_rope], dim=1).contiguous()
    return W.view(N * qk_head_dim, -1).contiguous()


def _process_layernorm_gammas(
    cfg: DSv2Config, fetcher: _WeightFetcher, layer: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = f"model.layers.{layer}"
    H = cfg.hidden_size
    g1 = fetcher.get(f"{prefix}.input_layernorm.weight")
    g3 = fetcher.get(f"{prefix}.self_attn.kv_a_layernorm.weight")
    g4 = fetcher.get(f"{prefix}.post_attention_layernorm.weight")
    if cfg.q_lora_rank is not None:
        g2 = fetcher.get(f"{prefix}.self_attn.q_a_layernorm.weight")
    else:
        g2 = torch.ones(H, dtype=g1.dtype, device=g1.device)
    return g1, g2, g3, g4


def _process_moe(
    cfg: DSv2Config, fetcher: _WeightFetcher, layer: int,
    ep_size: int = 1, ep_rank: int = 0,
) -> dict:
    prefix = f"model.layers.{layer}.mlp"
    H = cfg.hidden_size
    E = cfg.n_routed_experts
    I_r = cfg.moe_intermediate_size
    I_s = cfg.I_shared
    assert E % ep_size == 0, f"E={E} not divisible by ep_size={ep_size}"
    E_per_rank = E // ep_size
    e_start = ep_rank * E_per_rank
    e_end = e_start + E_per_rank

    w_router = fetcher.get(f"{prefix}.gate.weight")
    assert w_router.shape == (E, H)

    w_gr = torch.empty(E_per_rank, I_r, H, dtype=w_router.dtype, device=w_router.device)
    w_ur = torch.empty(E_per_rank, I_r, H, dtype=w_router.dtype, device=w_router.device)
    w_dr = torch.empty(E_per_rank, H, I_r, dtype=w_router.dtype, device=w_router.device)
    for local_e, global_e in enumerate(range(e_start, e_end)):
        w_gr[local_e] = fetcher.get(f"{prefix}.experts.{global_e}.gate_proj.weight")
        w_ur[local_e] = fetcher.get(f"{prefix}.experts.{global_e}.up_proj.weight")
        w_dr[local_e] = fetcher.get(f"{prefix}.experts.{global_e}.down_proj.weight")

    w_gs = fetcher.get(f"{prefix}.shared_experts.gate_proj.weight")
    w_us = fetcher.get(f"{prefix}.shared_experts.up_proj.weight")
    w_ds = fetcher.get(f"{prefix}.shared_experts.down_proj.weight")
    assert w_gs.shape == (I_s, H), f"shared gate {w_gs.shape} != ({I_s}, {H})"
    assert w_us.shape == (I_s, H)
    assert w_ds.shape == (H, I_s)

    return dict(
        W_gate_router=w_router.contiguous(),
        W_gate_routed=w_gr.contiguous(),
        W_up_routed=w_ur.contiguous(),
        W_down_routed=w_dr.contiguous(),
        W_gate_shared=w_gs.contiguous(),
        W_up_shared=w_us.contiguous(),
        W_down_shared=w_ds.contiguous(),
    )


def _process_dense_layer_mlp(
    cfg: DSv2Config, fetcher: _WeightFetcher, layer: int,
    ep_size: int = 1, ep_rank: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    prefix = f"model.layers.{layer}.mlp"
    H = cfg.hidden_size
    E = cfg.n_routed_experts
    I_r = cfg.moe_intermediate_size
    inter = cfg.intermediate_size
    assert inter is not None, "intermediate_size required for dense-layer fold"
    assert E % ep_size == 0
    E_per_rank = E // ep_size

    w_g = fetcher.get(f"{prefix}.gate_proj.weight")
    w_u = fetcher.get(f"{prefix}.up_proj.weight")
    w_d = fetcher.get(f"{prefix}.down_proj.weight")
    assert w_g.shape == (inter, H), f"dense gate {w_g.shape} != ({inter}, {H})"
    assert w_u.shape == (inter, H)
    assert w_d.shape == (H, inter)

    w_router = torch.zeros(E, H, dtype=dtype)
    w_gr = torch.zeros(E_per_rank, I_r, H, dtype=dtype)
    w_ur = torch.zeros(E_per_rank, I_r, H, dtype=dtype)
    w_dr = torch.zeros(E_per_rank, H, I_r, dtype=dtype)

    return dict(
        W_gate_router=w_router,
        W_gate_routed=w_gr,
        W_up_routed=w_ur,
        W_down_routed=w_dr,
        W_gate_shared=w_g.contiguous(),
        W_up_shared=w_u.contiguous(),
        W_down_shared=w_d.contiguous(),
    )


def load_stacked_weights(
    snapshot_dir: str | Path,
    layer_range: tuple[int, int] | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    fp8_stages: Iterable[str] | None = None,
    ep_size: int = 1,
    ep_rank: int = 0,
    include_layer0: bool = False,
) -> dict:
    snapshot_dir = Path(snapshot_dir)
    cfg = load_config(snapshot_dir)
    if layer_range is None:
        if include_layer0:
            layer_range = (0, cfg.num_hidden_layers)
        else:
            layer_range = (cfg.first_k_dense_replace, cfg.num_hidden_layers)
    start, end = layer_range
    n_moe = end - start
    assert n_moe > 0
    if include_layer0:
        assert start == 0, (
            f"include_layer0=True requires layer_range to start at 0, got {start}"
        )
    fetcher = _WeightFetcher(snapshot_dir)

    fp8_stages = set(fp8_stages or [])
    _VALID_FP8 = {"shared_q1", "shared_q2", "routed_moe"}
    unknown = fp8_stages - _VALID_FP8
    assert not unknown, f"unknown fp8_stages: {unknown} (supported: {_VALID_FP8})"

    H = cfg.hidden_size
    N = cfg.num_attention_heads
    Lkv = cfg.kv_lora_rank
    R = cfg.qk_rope_head_dim
    qk_nope_dim = cfg.qk_nope_head_dim
    v_head_dim = cfg.v_head_dim
    Lq_slice = cfg.q_lora_rank if cfg.q_lora_rank is not None else 0
    qb_K = cfg.q_lora_rank if cfg.q_lora_rank is not None else H
    OUT_QKVA = Lq_slice + Lkv + R
    OUT_QB = cfg.OUT_QB
    E = cfg.n_routed_experts
    I_r = cfg.moe_intermediate_size
    if include_layer0:
        I_s = cfg.I_shared_extended
        assert cfg.intermediate_size is not None, (
            "intermediate_size missing in cfg — required for layer-0 fold"
        )
    else:
        I_s = cfg.I_shared
    I_s_moe = cfg.I_shared

    out = {}
    out["gamma1"] = torch.empty(n_moe, H, dtype=dtype)
    out["gamma2"] = torch.empty(n_moe, max(Lq_slice, 1), dtype=dtype)
    out["gamma3"] = torch.empty(n_moe, Lkv, dtype=dtype)
    out["gamma4"] = torch.empty(n_moe, H, dtype=dtype)
    out["W_qkva"] = torch.empty(n_moe, OUT_QKVA, H, dtype=dtype)
    out["W_qb"] = torch.empty(n_moe, OUT_QB, qb_K, dtype=dtype)
    out["W_UK"] = torch.empty(n_moe, N, Lkv, qk_nope_dim, dtype=dtype)
    out["W_UV"] = torch.empty(n_moe, N, v_head_dim, Lkv, dtype=dtype)
    out["W_O"] = torch.empty(n_moe, H, N * v_head_dim, dtype=dtype)
    out["W_gate_router"] = torch.empty(n_moe, E, H, dtype=dtype)
    E_per_rank = E // ep_size
    out["W_gate_routed"] = torch.empty(n_moe, E_per_rank, I_r, H, dtype=dtype)
    out["W_up_routed"] = torch.empty(n_moe, E_per_rank, I_r, H, dtype=dtype)
    out["W_down_routed"] = torch.empty(n_moe, E_per_rank, H, I_r, dtype=dtype)
    if include_layer0:
        out["W_gate_shared"] = torch.zeros(n_moe, I_s, H, dtype=dtype)
        out["W_up_shared"] = torch.zeros(n_moe, I_s, H, dtype=dtype)
        out["W_down_shared"] = torch.zeros(n_moe, H, I_s, dtype=dtype)
    else:
        out["W_gate_shared"] = torch.empty(n_moe, I_s, H, dtype=dtype)
        out["W_up_shared"] = torch.empty(n_moe, I_s, H, dtype=dtype)
        out["W_down_shared"] = torch.empty(n_moe, H, I_s, dtype=dtype)

    for idx, layer in enumerate(range(start, end)):
        g1, g2, g3, g4 = _process_layernorm_gammas(cfg, fetcher, layer)
        out["gamma1"][idx] = g1
        if Lq_slice > 0:
            out["gamma2"][idx] = g2
        else:
            out["gamma2"][idx] = torch.ones_like(out["gamma2"][idx])
        out["gamma3"][idx] = g3
        out["gamma4"][idx] = g4
        out["W_qkva"][idx] = _process_qkva(cfg, fetcher, layer)
        out["W_qb"][idx] = _process_q_b(cfg, fetcher, layer)
        kv_b = fetcher.get(f"model.layers.{layer}.self_attn.kv_b_proj.weight")
        w_uk, w_uv = _process_kv_b_proj(kv_b, N, qk_nope_dim, v_head_dim, Lkv)
        out["W_UK"][idx] = w_uk
        out["W_UV"][idx] = w_uv
        out["W_O"][idx] = fetcher.get(f"model.layers.{layer}.self_attn.o_proj.weight")
        if include_layer0 and layer < cfg.first_k_dense_replace:
            dense = _process_dense_layer_mlp(
                cfg, fetcher, layer, ep_size=ep_size, ep_rank=ep_rank, dtype=dtype,
            )
            for k, v in dense.items():
                if k in ("W_gate_shared", "W_up_shared"):
                    assert v.shape == (I_s, H), (
                        f"dense {k} {v.shape} != extended ({I_s}, {H})"
                    )
                    out[k][idx] = v
                elif k == "W_down_shared":
                    assert v.shape == (H, I_s), (
                        f"dense {k} {v.shape} != extended ({H}, {I_s})"
                    )
                    out[k][idx] = v
                else:
                    out[k][idx] = v
        else:
            moe = _process_moe(cfg, fetcher, layer, ep_size=ep_size, ep_rank=ep_rank)
            for k, v in moe.items():
                if k in ("W_gate_shared", "W_up_shared"):
                    assert v.shape == (I_s_moe, H), v.shape
                    out[k][idx, :I_s_moe, :] = v
                elif k == "W_down_shared":
                    assert v.shape == (H, I_s_moe), v.shape
                    out[k][idx, :, :I_s_moe] = v
                else:
                    out[k][idx] = v

    if "shared_q1" in fp8_stages:
        _attach_fp8_shared_q1(out, I_s, H)
    if "shared_q2" in fp8_stages:
        _attach_fp8_shared_q2(out, I_s, H)
    if "routed_moe" in fp8_stages:
        _attach_fp8_routed_moe(out, E_per_rank, cfg.moe_intermediate_size, H)
        _attach_fp8_routed_moe_hw_sf(out, E_per_rank, cfg.moe_intermediate_size, H)
        _dummy = torch.empty(1, 1, 1, 1, dtype=dtype)
        out["W_gate_routed"] = _dummy
        out["W_up_routed"] = _dummy
        out["W_down_routed"] = _dummy
    import gc
    gc.collect()

    if device != "cpu":
        for k in list(out.keys()):
            if k == "config" or k == "layer_range":
                continue
            out[k] = out[k].to(device).contiguous()
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    out["config"] = cfg
    out["layer_range"] = (start, end)
    return out


def _attach_fp8_shared_q1(out: dict, I_s: int, H: int) -> None:
    import sys
    from dsv2mk.quant import quantize_to_fp8_with_hw_sf

    W_g = out["W_gate_shared"]
    W_u = out["W_up_shared"]
    assert W_g.shape[-2:] == (I_s, H), f"W_gate_shared {W_g.shape} != (.., {I_s}, {H})"
    assert W_u.shape[-2:] == (I_s, H), f"W_up_shared   {W_u.shape} != (.., {I_s}, {H})"
    assert I_s % 128 == 0, f"I_s={I_s} not a multiple of 128 (HW SF atom MN)"
    assert H % 128 == 0, f"H={H} not a multiple of 128 (HW SF atom K * sf_vec_size)"

    Wg_fp8, sf_g = quantize_to_fp8_with_hw_sf(W_g, logical_block_k=128, sf_vec_size=32)
    Wu_fp8, sf_u = quantize_to_fp8_with_hw_sf(W_u, logical_block_k=128, sf_vec_size=32)
    expected_sf_per_layer = I_s * H // 32
    assert sf_g.shape == (W_g.shape[0], expected_sf_per_layer), (
        f"SFA gate {sf_g.shape} != ({W_g.shape[0]}, {expected_sf_per_layer})"
    )
    assert sf_u.shape == (W_u.shape[0], expected_sf_per_layer)
    out["W_gate_shared_fp8"] = Wg_fp8.contiguous()
    out["W_up_shared_fp8"] = Wu_fp8.contiguous()
    out["W_gate_shared_sf"] = sf_g.contiguous()
    out["W_up_shared_sf"] = sf_u.contiguous()


def _attach_fp8_shared_q2(out: dict, I_s: int, H: int) -> None:
    import sys
    from dsv2mk.quant import quantize_to_fp8_with_hw_sf

    W_d = out["W_down_shared"]
    assert W_d.shape[-2:] == (H, I_s), f"W_down_shared {W_d.shape} != (.., {H}, {I_s})"
    assert I_s % 128 == 0, f"I_s={I_s} not divisible by 128 (logical SF block)"
    assert I_s % 32 == 0, f"I_s={I_s} not divisible by 32 (HW SF vec_size)"
    assert H % 128 == 0, f"H={H} not divisible by 128 (HW SF atom MN)"

    Wd_fp8_hw, sf_d_hw = quantize_to_fp8_with_hw_sf(
        W_d, logical_block_k=128, sf_vec_size=32,
    )
    expected_sf_hw_per_layer = H * I_s // 32
    assert sf_d_hw.shape == (W_d.shape[0], expected_sf_hw_per_layer), (
        f"Q2 SF_HW shape {sf_d_hw.shape} != "
        f"({W_d.shape[0]}, {expected_sf_hw_per_layer})"
    )
    out["W_down_shared_fp8"] = Wd_fp8_hw.contiguous()
    out["W_down_shared_sf_hw"] = sf_d_hw.contiguous()


def _attach_fp8_routed_moe(out: dict, E: int, I_routed: int, H: int) -> None:
    import sys
    from dsv2mk.quant import quantize_fp8_e4m3_block

    Wg = out["W_gate_routed"]
    Wu = out["W_up_routed"]
    Wd = out["W_down_routed"]

    assert Wg.shape[-3:] == (E, I_routed, H), f"W_gate_routed {Wg.shape} != (.., {E}, {I_routed}, {H})"
    assert Wu.shape == Wg.shape, f"W_up_routed {Wu.shape} != {Wg.shape}"
    assert Wd.shape[-3:] == (E, H, I_routed), f"W_down_routed {Wd.shape} != (.., {E}, {H}, {I_routed})"
    assert H % 128 == 0
    assert I_routed % 128 == 0

    Wg_fp8, sf_g = quantize_fp8_e4m3_block(Wg, block_k=128)
    Wu_fp8, sf_u = quantize_fp8_e4m3_block(Wu, block_k=128)
    Wd_fp8, sf_d = quantize_fp8_e4m3_block(Wd, block_k=128)

    n_moe = Wg.shape[0]
    assert sf_g.shape == (n_moe, E, I_routed, H // 128), f"sf_g shape mismatch: {sf_g.shape}"
    assert sf_u.shape == (n_moe, E, I_routed, H // 128), f"sf_u shape mismatch: {sf_u.shape}"
    assert sf_d.shape == (n_moe, E, H, I_routed // 128), f"sf_d shape mismatch: {sf_d.shape}"

    out["W_gate_routed_fp8"] = Wg_fp8.contiguous()
    out["W_up_routed_fp8"] = Wu_fp8.contiguous()
    out["W_down_routed_fp8"] = Wd_fp8.contiguous()
    out["W_gate_routed_sf"] = sf_g.contiguous()
    out["W_up_routed_sf"] = sf_u.contiguous()
    out["W_down_routed_sf"] = sf_d.contiguous()


def _attach_fp8_routed_moe_hw_sf(out: dict, E: int, I_routed: int, H: int) -> None:
    import sys
    from dsv2mk.quant import replicate_ue8m0_to_sf_vec_size, reorder_sf_to_hw_layout

    sf_g = out["W_gate_routed_sf"]
    sf_u = out["W_up_routed_sf"]
    sf_d = out["W_down_routed_sf"]
    assert sf_g.shape[-4:] == (sf_g.shape[0], E, I_routed, H // 128), (
        f"sf_g shape {sf_g.shape} unexpected"
    )
    assert sf_d.shape[-4:] == (sf_d.shape[0], E, H, I_routed // 128), (
        f"sf_d shape {sf_d.shape} unexpected"
    )

    def _to_hw_4d(sf_logical: torch.Tensor, MN: int, K: int) -> torch.Tensor:
        sf_per_vec = replicate_ue8m0_to_sf_vec_size(
            sf_logical, logical_block_k=128, sf_vec_size=32,
        )
        n_moe = sf_per_vec.shape[0]
        assert MN % 128 == 0, f"MN={MN} not divisible by 128 (HW SF atom MN)"
        assert (K // 32) % 4 == 0, f"K//32={K//32} not divisible by 4 (HW SF atom K)"
        rest_m = MN // 128
        rest_k = (K // 32) // 4
        sf7 = sf_per_vec.reshape(n_moe, E, rest_m, 4, 32, rest_k, 4)
        out = sf7.permute(0, 1, 2, 5, 4, 3, 6).contiguous()
        return out.reshape(n_moe, E, MN * (K // 32)).contiguous()

    out["W_gate_routed_sf_hw"] = _to_hw_4d(sf_g, I_routed, H)
    out["W_up_routed_sf_hw"] = _to_hw_4d(sf_u, I_routed, H)
    out["W_down_routed_sf_hw"] = _to_hw_4d(sf_d, H, I_routed)

