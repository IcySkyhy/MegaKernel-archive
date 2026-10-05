"""Opaque torch mutation boundary for the independent fixed native sequence."""
import torch

from .native import _LIVE


@torch.library.custom_op('qwen3_megakernel::enqueue', mutates_args=('buffers',))
def enqueue(inputs: list[torch.Tensor], buffers: list[torch.Tensor], handle: int) -> None:
    workspace = _LIVE.get(handle)
    if workspace is None or not hasattr(workspace, '_submit_native'):
        raise RuntimeError('unknown decoder generation')
    if (len(inputs) != len(workspace.op_inputs) or len(buffers) != len(workspace.op_buffers)
            or any(tensor.data_ptr() != expected.data_ptr()
                   for tensor, expected in zip(inputs + buffers, workspace.op_inputs + workspace.op_buffers))):
        raise RuntimeError('compiler changed decoder storage bindings')
    workspace._submit_native(torch.cuda.current_stream(workspace.stream.device))


@enqueue.register_fake
def _fake_enqueue(inputs: list[torch.Tensor], buffers: list[torch.Tensor], handle: int) -> None:
    return None
