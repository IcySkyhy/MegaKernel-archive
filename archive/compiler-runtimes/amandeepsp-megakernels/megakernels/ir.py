from torch.storage import UntypedStorage
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class ValueRef:
    node: str
    output_index: int = 0
    # TODO: Nested Outputs?


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    storage_id: int
    storage_offset: int = 0
