import logging
import pytest
import torch
import triton
import triton.language as tl

from .kernels import matmul_silu, matmul_silu_torch


logging.basicConfig(level=logging.DEBUG)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx90a"


@pytest.mark.parametrize("Z, M, N, K", [(2, 32, 256, 512)])
def test_matmul_silu(Z, M, N, K):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, M, N), device=DEVICE, dtype=torch.float16) * SCALE
    w = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    triton_output = matmul_silu(x, w)
    torch_output = matmul_silu_torch(x, w)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    # Bigger tolerance for AMD MI200 devices.
    # MI200 devices use reduced precision fp16 and bf16 and flush input and
    # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    rtol = 1e-2 if is_hip_mi200() else 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


@pytest.mark.parametrize("Z, N, K", [(2, 256, 512)])
def test_matmul_silu_vec(Z, N, K):
    test_matmul_silu(Z, 10, N, K)


if __name__ == "__main__":
    print("TESTING GEMM")
    test_matmul_silu(2, 32, 256, 512)
    print("TESTING GEMV")
    test_matmul_silu_vec(2, 256, 512)
