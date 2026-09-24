from abc import ABC, abstractmethod

import torch


class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        apply_router_weight_on_input: bool,
        expert_map: torch.Tensor | None = None,
        global_num_experts: int | None = None,
    ) -> torch.Tensor: ...
