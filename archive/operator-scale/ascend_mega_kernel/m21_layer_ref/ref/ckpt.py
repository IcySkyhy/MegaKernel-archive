"""Per-layer checkpoint loader (real Qwen3.8-Flash-Next-MXFP4 weights).

Reads ONE decoder layer's tensors out of the 131-shard safetensors checkpoint via
the header index, so memory stays ~1.4 GiB instead of 170 GiB. Routed experts are
kept packed (uint8) and dequantized one expert at a time on demand, because a
single layer's gate_up_proj is 838 MB packed / 6.7 GB dequantized.

Checkpoint prefix (vLLM maps `model.language_model.` -> `model.`,
vllm/models/qwen4_exp/nvidia/model.py:657-658):

    model.language_model.layers.<L>.attn_hyper_connection.{hc_norm.weight,
        input_mix_weight_down.weight, input_mix_weight_up.weight,
        block_inject_weight.weight}
    model.language_model.layers.<L>.mlp_hyper_connection.{...same four...}
    model.language_model.layers.<L>.linear_attn.{A_log, conv1d.weight, dt_bias,
        in_proj_a.weight, in_proj_b.weight, in_proj_qkv.weight,
        in_proj_z.weight, norm.weight, out_proj.weight}
    model.language_model.layers.<L>.self_attn.{q_proj,k_proj,v_proj,o_proj,
        q_norm,k_norm}.weight + .indexer.{index_qk_proj,q_layernorm,k_layernorm}.weight
    model.language_model.layers.<L>.mlp.{gate.weight, shared_expert_gate.weight,
        experts.{gate_up_proj,down_proj}[.weight_scale],
        shared_expert.{gate_proj,up_proj,down_proj}.weight[.weight_scale]}
    model.language_model.hyper_connection_mixer.{hc_norm,input_mix_weight_down,
        input_mix_weight_up}.weight            (global final mixer, use_combine=False)

Weight-layout decisions and their sources:

  * `in_proj_qkvz` (MergedColumnParallelLinear, output_sizes [2048,2048,6144,6144],
    qwen_gdn_linear_attn.py:564-588) is the concatenation of the checkpoint's
    `in_proj_qkv.weight` [10240,2560] then `in_proj_z.weight` [6144,2560]
    (qwen3_5.py:219-226 `.in_proj_qkv -> (".in_proj_qkvz",(0,1,2))`,
    `.in_proj_z -> (".in_proj_qkvz",3)`).
  * `in_proj_ba` ([48,48]) is `in_proj_b` then `in_proj_a` (qwen3_5.py:223-224).
  * `conv1d.weight` is stored [10240,1,4] and vLLM re-adds the middle dim
    (qwen_gdn_linear_attn.py:427 `self.conv1d.weight.data.unsqueeze(1)`).
  * MoE routed experts stay packed; `gate_up_proj` rows [0,640) = gate, [640,1280) = up
    (routed_experts.py:936-941 `chunk(2, dim=1)[0] -> w1 = gate`).
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from safetensors_reader import ShardReader  # noqa: E402,F401  (verbatim upstream copy)
from torch_reader import TorchShardReader  # noqa: E402  (M39's torch-facing subclass)

DEFAULT_CKPT = "/workspace/Qwen3.8-Flash-Next-MXFP4"


def _torch():
    import torch

    return torch


@dataclass
class HyperConnWeights:
    """The four checkpoint tensors of one GatedResidual mixer."""

    hc_norm_weight: object  # bf16 [10240]
    down_weight: object  # bf16 [320, 10240]
    up_weight: object  # bf16 [10240, 320]
    inject_weight: object | None  # bf16 [4, 10240] (None for the final mixer)


@dataclass
class GDNetWeights:
    in_proj_qkvz: object  # bf16 [16384, 2560]
    in_proj_ba: object  # bf16 [96, 2560]
    conv1d_weight: object  # bf16 [10240, 4]  (middle dim dropped)
    norm_weight: object  # bf16 [128]
    out_proj_weight: object  # bf16 [2560, 6144]
    A_log: object  # fp32 [48]  (checkpoint stores bf16; vLLM forces fp32)
    dt_bias: object  # bf16 [48]


@dataclass
class QSAAttnWeights:
    qkv_proj: object  # bf16 [13312, 2560] = cat(q_proj, k_proj, v_proj)
    o_proj_weight: object  # bf16 [2560, 6144]
    q_norm_weight: object  # bf16 [256]
    k_norm_weight: object  # bf16 [256]
    index_qk_proj: object  # bf16 [640, 2560]
    index_q_norm_weight: object  # bf16 [128]
    index_k_norm_weight: object  # bf16 [128]


@dataclass
class MoEWeights:
    gate_weight: object  # bf16 [512, 2560]
    shared_expert_gate_weight: object  # bf16 [1, 2560]
    shared_gate_proj: object  # fp32 [640, 2560] (dequantized MXFP4)
    shared_up_proj: object  # fp32 [640, 2560]
    shared_down_proj: object  # fp32 [2560, 640]
    _reader: TorchShardReader | None = field(default=None, repr=False)
    _prefix: str = field(default="", repr=False)
    _cache: "OrderedDict" = field(default_factory=OrderedDict, repr=False)

    def expert_gate_up(self, expert: int):
        """fp32 [1280, 2560]; rows [0,640) = gate_proj, [640,1280) = up_proj."""
        return self._cached("gu", expert,
                            self._prefix + "mlp.experts.gate_up_proj")

    def expert_down(self, expert: int):
        """fp32 [2560, 640]."""
        return self._cached("dn", expert, self._prefix + "mlp.experts.down_proj")

    # Dequantizing one expert costs ~12 MB of fp32; keep a bounded LRU so a
    # 64-token chunk that touches many experts does not re-dequantize constantly.
    MAX_CACHED_EXPERTS = 40

    def _cached(self, kind: str, expert: int, name: str):
        key = (kind, expert)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        val = self._dequant(name, expert)
        self._cache[key] = val
        while len(self._cache) > self.MAX_CACHED_EXPERTS:
            self._cache.popitem(last=False)
        return val

    def _dequant(self, name: str, expert: int):
        from .mxfp4 import dequant_mxfp4_torch

        torch = _torch()
        packed = self._reader.load_torch(name, slice(expert, expert + 1))
        scale = self._reader.load_torch(name + ".weight_scale", slice(expert, expert + 1))
        return dequant_mxfp4_torch(packed[0], scale[0], torch)


class LayerWeights:
    """All checkpoint tensors needed to run one decoder layer on the CPU.

    ``layer_idx`` is 0-based. Use ``LayerWeights.mixer_global()`` for the final
    ``hyper_connection_mixer`` (3 tensors, ``use_combine=False``).
    """

    def __init__(self, layer_idx: int, ckpt: str = DEFAULT_CKPT):
        self.layer_idx = layer_idx
        self.ckpt = ckpt
        self.reader = _get_reader(ckpt)
        self.prefix = f"model.language_model.layers.{layer_idx}."
        self.torch = _torch()

    # ------------------------------------------------------------------ HC
    def hyper_connection(self, which: str) -> HyperConnWeights:
        assert which in ("attn_hyper_connection", "mlp_hyper_connection")
        p = self.prefix + which + "."
        r = self.reader
        return HyperConnWeights(
            hc_norm_weight=r.load_torch(p + "hc_norm.weight"),
            down_weight=r.load_torch(p + "input_mix_weight_down.weight"),
            up_weight=r.load_torch(p + "input_mix_weight_up.weight"),
            inject_weight=r.load_torch(p + "block_inject_weight.weight"),
        )

    # ---------------------------------------------------------------- GDN
    def gdn(self) -> GDNetWeights:
        p = self.prefix + "linear_attn."
        r = self.reader
        qkv = r.load_torch(p + "in_proj_qkv.weight")  # [10240, 2560]
        z = r.load_torch(p + "in_proj_z.weight")  # [6144, 2560]
        b = r.load_torch(p + "in_proj_b.weight")  # [48, 2560]
        a = r.load_torch(p + "in_proj_a.weight")  # [48, 2560]
        conv = r.load_torch(p + "conv1d.weight")  # [10240, 1, 4]
        return GDNetWeights(
            in_proj_qkvz=self.torch.cat([qkv, z], dim=0),
            in_proj_ba=self.torch.cat([b, a], dim=0),
            conv1d_weight=conv.squeeze(1).contiguous(),
            norm_weight=r.load_torch(p + "norm.weight"),
            out_proj_weight=r.load_torch(p + "out_proj.weight"),
            A_log=r.load_torch_f32(p + "A_log"),
            dt_bias=r.load_torch(p + "dt_bias"),
        )

    # ---------------------------------------------------------------- QSA
    def qsa(self) -> QSAAttnWeights:
        p = self.prefix + "self_attn."
        r = self.reader
        return QSAAttnWeights(
            qkv_proj=self.torch.cat(
                [
                    r.load_torch(p + "q_proj.weight"),
                    r.load_torch(p + "k_proj.weight"),
                    r.load_torch(p + "v_proj.weight"),
                ],
                dim=0,
            ),
            o_proj_weight=r.load_torch(p + "o_proj.weight"),
            q_norm_weight=r.load_torch(p + "q_norm.weight"),
            k_norm_weight=r.load_torch(p + "k_norm.weight"),
            index_qk_proj=r.load_torch(p + "indexer.index_qk_proj.weight"),
            index_q_norm_weight=r.load_torch(p + "indexer.q_layernorm.weight"),
            index_k_norm_weight=r.load_torch(p + "indexer.k_layernorm.weight"),
        )

    # ---------------------------------------------------------------- MoE
    def moe(self) -> MoEWeights:
        p = self.prefix
        r = self.reader
        torch = self.torch
        from .mxfp4 import dequant_mxfp4_torch

        def one(name):
            packed = r.load_torch(name + ".weight")
            scale = r.load_torch(name + ".weight_scale")
            return dequant_mxfp4_torch(packed, scale, torch)

        return MoEWeights(
            gate_weight=r.load_torch(p + "mlp.gate.weight"),
            shared_expert_gate_weight=r.load_torch(p + "mlp.shared_expert_gate.weight"),
            shared_gate_proj=one(p + "mlp.shared_expert.gate_proj"),
            shared_up_proj=one(p + "mlp.shared_expert.up_proj"),
            shared_down_proj=one(p + "mlp.shared_expert.down_proj"),
            _reader=r,
            _prefix=p,
        )

    # ------------------------------------------------------------- config
    def text_config(self) -> dict:
        import json

        return json.load(open(f"{self.ckpt}/config.json"))["text_config"]

    def layer_type(self) -> str:
        return self.text_config()["layer_types"][self.layer_idx]

    # -------------------------------------------------- global final mixer
    @classmethod
    def global_mixer(cls, ckpt: str = DEFAULT_CKPT) -> HyperConnWeights:
        r = _get_reader(ckpt)
        p = "model.language_model.hyper_connection_mixer."
        return HyperConnWeights(
            hc_norm_weight=r.load_torch(p + "hc_norm.weight"),
            down_weight=r.load_torch(p + "input_mix_weight_down.weight"),
            up_weight=r.load_torch(p + "input_mix_weight_up.weight"),
            inject_weight=None,  # vLLM: use_combine=False -> no block_inject_weight
        )

    @classmethod
    def embed_tokens(cls, ckpt: str = DEFAULT_CKPT):
        r = _get_reader(ckpt)
        return r.load_torch("model.language_model.embed_tokens.weight")


@lru_cache(maxsize=4)
def _get_reader(ckpt: str) -> TorchShardReader:
    """Reader for `ckpt`, with a loud check that the download is complete.

    `ShardReader.incomplete` maps a shard to the tensors whose byte range has not
    landed yet (upstream M8 behaviour, kept by the verbatim copy). A partial
    download must fail here, not silently mid-run.
    """
    rdr = TorchShardReader(ckpt)
    if rdr.incomplete:
        bad = {k: len(v) for k, v in rdr.incomplete.items()}
        raise RuntimeError(
            f"{ckpt}: {len(bad)} shard(s) are incomplete, "
            f"missing tensor counts {bad}. Finish the download first."
        )
    return rdr
