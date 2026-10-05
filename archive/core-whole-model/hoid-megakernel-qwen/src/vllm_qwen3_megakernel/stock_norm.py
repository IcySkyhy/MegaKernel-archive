"""The stock final RMSNorm, compiled natively over each bucket's summed BF16 residual.

Imports stay lazy so manifest/lifecycle validation requires neither torch nor vLLM.
"""
from __future__ import annotations


def _check_stock_norm(norm, device):
    import torch
    from vllm.model_executor.layers.layernorm import RMSNorm
    if (not isinstance(norm, RMSNorm) or norm.hidden_size != 2560
            or norm.variance_epsilon != 1e-6 or norm.variance_size_override is not None
            or not norm.has_weight or not norm.pass_weight_add):
        raise ValueError("stock final norm requires the actual weighted RMSNorm(2560, eps=1e-6)")
    if (tuple(norm.weight.shape) != (2560,) or norm.weight.device != device
            or norm.weight.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("stock final norm weight must retain its physical BF16/FP32 dtype and device")


def prepare_residual_final_norm(norm, example):
    """Compile the stock RMSNorm over a one-field [rows, hidden] final residual."""
    import torch
    from vllm import ir
    from vllm.ir.op import enable_torch_wrap

    _check_stock_norm(norm, example.device)
    if (example.ndim != 2 or example.shape[0] not in (1, 2, 4, 8) or example.shape[1] != norm.hidden_size
            or example.dtype != torch.bfloat16 or not example.is_contiguous()):
        raise ValueError("residual final norm requires contiguous BF16 [rows, hidden], rows in 1/2/4/8")
    if example.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("stock final norm must be prepared outside CUDA capture")
    compiled = torch.compile(lambda residual: norm.forward_native(residual), fullgraph=True, dynamic=False)
    rows = example.shape[0]

    def run(residual):
        if residual.shape != example.shape:
            raise ValueError("final norm prepared for another bucket")
        with torch.inference_mode(), enable_torch_wrap(False), ir.ops.rms_norm.set_priority(["native"]):
            return compiled(residual)

    normalized = run(torch.zeros_like(example))
    if normalized.dtype != torch.bfloat16 or tuple(normalized.shape) != (rows, norm.hidden_size):
        raise RuntimeError("compiled stock final norm violated its BF16 output contract")
    return run
