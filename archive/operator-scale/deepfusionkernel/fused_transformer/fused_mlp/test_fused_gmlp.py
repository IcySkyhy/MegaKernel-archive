import logging
import pytest
import torch

import triton
import triton.language as tl

from .fused_gmlp import *
from ..utils import is_cuda, is_hip_mi200


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.DEBUG)


# %%
# Unit Test
# ---------


@pytest.mark.parametrize("Z, M, N, K, T", [(2, 32, 256, 512, 256)])
def test_gatedmlp_m_tkn(Z, M, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, M, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_m_tkn(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


@pytest.mark.parametrize("Z, N, K, T", [(2, 256, 512, 256)])
def test_gatedmlp_m_tkn_vec(Z, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, 10, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_m_tkn_vec(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


@pytest.mark.parametrize("Z, M, N, K, T", [(2, 32, 256, 512, 256)])
def test_gatedmlp_mt_kn(Z, M, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, M, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_mt_kn(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


@pytest.mark.parametrize("Z, N, K, T", [(2, 256, 512, 256)])
def test_gatedmlp_mt_kn_vec(Z, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, 10, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_mt_kn_vec(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


@pytest.mark.parametrize("Z, M, N, K, T", [(2, 32, 256, 512, 256)])
def test_gatedmlp_mt_kn_keepA2(Z, M, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, M, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_mt_kn_keepA2(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


@pytest.mark.parametrize("Z, N, K, T", [(2, 256, 512, 256)])
def test_gatedmlp_mt_kn_keepA2_vec(Z, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, 10, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_mt_kn_keepA2_vec(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


@pytest.mark.parametrize("Z, M, N, K, T", [(2, 32, 256, 512, 256)])
def test_gatedmlp_t_mkn(Z, M, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, M, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_t_mkn(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


@pytest.mark.parametrize("Z, N, K, T", [(2, 256, 512, 256)])
def test_gatedmlp_t_mkn_vec(Z, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, 10, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_t_mkn_vec(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


@pytest.mark.parametrize("Z, M, N, K, T", [(2, 32, 256, 512, 256)])
def test_gatedmlp_mk_tn(Z, M, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, M, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_mk_tn(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


@pytest.mark.parametrize("M, N, K, T", [(32, 256, 512, 256)])
def test_gatedmlp_separated(M, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((M, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_separated(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


@pytest.mark.parametrize("Z, N, K, T", [(2, 256, 512, 256)])
def test_gatedmlp_separated_vec(Z, N, K, T):
    torch.manual_seed(0)
    # quantization error accumulate through matmul,
    # so we scale down X to make error magnitude consistent
    SCALE = 1 / K
    x = torch.randn((Z, 10, N), device=DEVICE, dtype=torch.float16) * SCALE
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    triton_output = gatedmlp_separated_vec(x, u, g, d, "silu")
    torch_output = gatedmlp_torch(x, u, g, d, "silu")
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


# Debug
if __name__ == "__main__":
    Z, M, N, K, T = (2, 32, 128, 128, 128)
    test_gatedmlp_m_tkn(Z, M, N, K, T)
    test_gatedmlp_mt_kn(Z, M, N, K, T)
    test_gatedmlp_mt_kn_keepA2(Z, M, N, K, T)
    test_gatedmlp_t_mkn(Z, M, N, K, T)
    test_gatedmlp_separated(M, N, K, T)
    test_gatedmlp_m_tkn_vec(Z, N, K, T)
    test_gatedmlp_mt_kn_vec(Z, N, K, T)
    test_gatedmlp_mt_kn_keepA2_vec(Z, N, K, T)
    test_gatedmlp_t_mkn_vec(Z, N, K, T)
    test_gatedmlp_separated_vec(Z, N, K, T)
    test_gatedmlp_mk_tn(Z, M, N, K, T)
