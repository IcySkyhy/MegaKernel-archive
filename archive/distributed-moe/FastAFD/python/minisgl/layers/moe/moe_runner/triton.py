from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from minisgl.core import get_global_ctx

from .base import MoeRunner, MoeRunnerBackend, MoeRunnerConfig

if TYPE_CHECKING:
    import torch

    from ..layer import MoELayer


class TritonRunner(MoeRunner):
    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON

    def apply(self, dispatch_output: Any, layer: "MoELayer") -> "torch.Tensor":
        hidden_states = dispatch_output.hidden_states
        if hidden_states.shape[0] == 0:
            return hidden_states
        topk_output = dispatch_output.topk_output
        if isinstance(topk_output, tuple) and len(topk_output) == 2:
            topk_weights, topk_ids = topk_output
        else:
            from minisgl.moe.fused import fused_topk

            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=topk_output,
                topk=layer.top_k,
                renormalize=layer.renormalize,
            )
        w1 = layer.gate_up_proj
        w2 = layer.down_proj
        if w1.dtype == torch.float8_e4m3fn:
            from minisgl.models.fp8_utils import dequant_fp8_block_batched

            block_size = (
                layer.quant.weight_block_size
                if getattr(layer, "quant", None) is not None
                else (128, 128)
            )
            w1 = dequant_fp8_block_batched(w1, layer.gate_up_proj_scale, block_size)
            w2 = dequant_fp8_block_batched(w2, layer.down_proj_scale, block_size)

        ctx = get_global_ctx()
        if getattr(dispatch_output, "use_expert_map_override", False):
            expert_map = dispatch_output.expert_map_override
            global_num_experts = dispatch_output.global_num_experts_override
        else:
            expert_map = layer._expert_map_dev
            global_num_experts = layer.num_experts
        return ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=expert_map,
            global_num_experts=global_num_experts,
        )


__all__ = ["TritonRunner"]
