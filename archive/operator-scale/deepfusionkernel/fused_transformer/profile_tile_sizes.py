"""
Script for profiling the impact of tile sizes for gated MLP with separated kernels.
"""

import logging
from pprint import pprint
import torch

from tqdm import tqdm
import triton
import triton.language as tl

from transformers.activations import ACT2FN

from .utils import is_cuda, is_hip


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_autotune_config(separated_kernel=0, vec_input=False):
    """
    :param: separated_kernel:
      `0` for all configs.
      `1` for configs without dim `T`.
      `2` for configs without dim `N`.
    """
    if separated_kernel == 0:
        configs = [
            triton.Config(
                {
                    **({} if vec_input else {"BLOCK_M": BM}),
                    **{"BLOCK_N": BN, "BLOCK_K": BK, "BLOCK_T": BT},
                },
                num_stages=s,
                num_warps=w,
            )  # for BM in [16]\
            # for BN in [16]\
            # for BK in [16]\
            # for BT in [32]\
            # for s in ([2] if is_hip() else [4])\
            # for w in [4]\
            for BM in [32, 64]
            for BN in [32, 64, 128, 256]
            for BK in [32, 64, 128, 256]
            for BT in [32, 64, 128, 256]
            for s in ([2] if is_hip() else [3, 4, 5])
            for w in [2, 4, 8]
        ]
    elif separated_kernel == 1:
        configs = [
            triton.Config(
                {
                    **({} if vec_input else {"BLOCK_M": BM}),
                    **{"BLOCK_N": BN, "BLOCK_K": BK},
                },
                num_stages=s,
                num_warps=w,
            )  # for BM in [16]\
            # for BN in [16]\
            # for BK in [16]\
            # for s in ([2] if is_hip() else [4])\
            # for w in [4]\
            for BM in [32, 64]
            for BN in [32, 64, 128, 256]
            for BK in [32, 64, 128, 256]
            for s in ([2] if is_hip() else [3, 4, 5])
            for w in [2, 4, 8]
        ]
    elif separated_kernel == 2:
        configs = [
            triton.Config(
                {
                    **({} if vec_input else {"BLOCK_M": BM}),
                    **{"BLOCK_K": BK, "BLOCK_T": BT},
                },
                num_stages=s,
                num_warps=w,
            )  # for BM in [16]\
            # for BK in [16]\
            # for BT in [32]\
            # for s in ([2] if is_hip() else [4])\
            # for w in [4]\
            for BM in [32, 64]
            for BK in [32, 64, 128, 256]
            for BT in [32, 64, 128, 256]
            for s in ([2] if is_hip() else [3, 4, 5])
            for w in [2, 4, 8]
        ]
    else:
        raise ValueError("Invalid `separated_kernel`, should be one of [0, 1, 2]")
    return configs


@triton.jit
def silu(x):
    """
    silu(x) = x * sigmoid(x)
    """
    a = sigmoid(x)
    return x * a


@triton.jit
def sigmoid(x):
    return tl.sigmoid(x)


ACTIVATIONS = [
    "silu",
]


@triton.jit
def gatedmlp_separated_a2_kernel(
    X, U, G, A2,  # fmt: skip
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_a2z, stride_a2m, stride_a2k,  # fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate
    Z,
    M,
    N,
    K,  # fmt: skip
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
):
    """
    Kernel for first half gated MLP with ACT as Silu (computing A2).
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (M // BLOCK_M) * (K // BLOCK_K))
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    off_z = pid // (num_pid_m * num_pid_k)
    pid_sample = pid % (num_pid_m * num_pid_k)
    pid_m = pid_sample % num_pid_m
    pid_k = pid_sample // num_pid_m
    # pid_m = pid_sample // num_pid_k
    # pid_k = pid_sample % num_pid_k
    x_offset = off_z.to(tl.int64) * stride_xz
    a2_offset = off_z.to(tl.int64) * stride_a2z

    a2_ptr = tl.make_block_ptr(
        base=A2 + a2_offset,
        shape=(M, K),
        strides=(stride_a2m, stride_a2k),
        offsets=(pid_m * BLOCK_M, pid_k * BLOCK_K),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )
    x_ptr = tl.make_block_ptr(
        base=X + x_offset,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    u_ptr = tl.make_block_ptr(
        base=U,
        shape=(N, K),
        strides=(stride_un, stride_uk),
        offsets=(0, pid_k * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    g_ptr = tl.make_block_ptr(
        base=G,
        shape=(N, K),
        strides=(stride_gn, stride_gk),
        offsets=(0, pid_k * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    accu = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)  # accu = X @ U
    accg = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)  # accg = X @ G
    for n in tl.range(0, tl.cdiv(N, BLOCK_N)):
        x = tl.load(x_ptr, boundary_check=(0, 1), padding_option="zero")
        u = tl.load(u_ptr, boundary_check=(0, 1), padding_option="zero")
        g = tl.load(g_ptr, boundary_check=(0, 1), padding_option="zero")

        accu = tl.dot(x, u, accu)
        accg = tl.dot(x, g, accg)

        x_ptr = tl.advance(x_ptr, (0, BLOCK_N))
        u_ptr = tl.advance(u_ptr, (BLOCK_N, 0))
        g_ptr = tl.advance(g_ptr, (BLOCK_N, 0))

    acca = tl.sigmoid(accg)  # acca = sigmoid(X@G)
    accg = acca * accg  # Silu(accg) = accg * sigmoid(accg)
    accu = accg * accu  # accu2 = accu * Silu(accg)
    tl.store(a2_ptr, accu.to(A2.type.element_ty), boundary_check=(0, 1))


@triton.jit
def gatedmlp_separated_y_kernel(
    A2, D, Y,  # fmt: skip
    stride_a2z, stride_a2m, stride_a2k,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate
    Z,
    M,
    K,
    T,  # fmt: skip
    BLOCK_M: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
):
    """
    Kernel for last half gated MLP (computing Y).
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (M // BLOCK_M) * (T // BLOCK_T))
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_t = tl.cdiv(T, BLOCK_T)
    off_z = pid // (num_pid_m * num_pid_t)
    pid_sample = pid % (num_pid_m * num_pid_t)
    pid_m = pid_sample % num_pid_m
    pid_t = pid_sample // num_pid_m
    # pid_m = pid_sample // num_pid_t
    # pid_t = pid_sample % num_pid_t
    a2_offset = off_z.to(tl.int64) * stride_a2z
    y_offset = off_z.to(tl.int64) * stride_yz

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(M, T),
        strides=(stride_ym, stride_yt),
        offsets=(pid_m * BLOCK_M, pid_t * BLOCK_T),
        block_shape=(BLOCK_M, BLOCK_T),
        order=(0, 1),
    )
    a2_ptr = tl.make_block_ptr(
        base=A2 + a2_offset,
        shape=(M, K),
        strides=(stride_a2m, stride_a2k),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )
    d_ptr = tl.make_block_ptr(
        base=D,
        shape=(K, T),
        strides=(stride_dk, stride_dt),
        offsets=(0, pid_t * BLOCK_T),
        block_shape=(BLOCK_K, BLOCK_T),
        order=(1, 0),
    )
    accy = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)  # accy = accu2 @ D
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a2 = tl.load(a2_ptr, boundary_check=(0, 1), padding_option="zero")
        d = tl.load(d_ptr, boundary_check=(0, 1), padding_option="zero")

        accy = tl.dot(a2, d, accy)

        a2_ptr = tl.advance(a2_ptr, (0, BLOCK_K))
        d_ptr = tl.advance(d_ptr, (BLOCK_K, 0))

    tl.store(y_ptr, accy.to(Y.type.element_ty), boundary_check=(0, 1))


def profile_gatedmlp_separated(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with separated kernels for computing A2 and Y.
    """
    # Check constraints.
    assert x.shape[-1] == u.shape[0], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[1] == d.shape[0], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    Z, M, N = x.shape
    N, K = u.shape
    K, T = d.shape
    # Allocates output.
    a2 = torch.empty((Z, M, K), device=x.device, dtype=x.dtype)
    y = torch.empty((Z, M, T), device=x.device, dtype=x.dtype)

    success_configs = []
    for triton_config in tqdm(
        get_autotune_config(separated_kernel=0), desc="Profiling"
    ):
        config = triton_config.kwargs
        logger.info(
            f"Profiling config: {config}, num_stages: {triton_config.num_stages}, num_warps: {triton_config.num_warps}"
        )
        # launch kernels
        try:
            if act == "silu":
                # launch kernel for a2
                grid = lambda meta: (
                    Z
                    * triton.cdiv(M, meta["BLOCK_M"])
                    * triton.cdiv(K, meta["BLOCK_K"]),
                )
                gatedmlp_separated_a2_kernel[grid](
                    x,
                    u,
                    g,
                    a2,  # fmt: skip
                    x.stride(0),
                    x.stride(1),
                    x.stride(2),  # fmt: skip
                    u.stride(0),
                    u.stride(1),  # fmt: skip
                    g.stride(0),
                    g.stride(1),  # fmt: skip
                    a2.stride(0), a2.stride(1), a2.stride(2),  # fmt: skip
                    Z,
                    M,
                    N,
                    K,  # fmt: skip
                    BLOCK_M=config["BLOCK_M"],
                    BLOCK_N=config["BLOCK_N"],
                    BLOCK_K=config["BLOCK_K"],
                )
                # input(a2)
                # launch kernel for y
                grid = lambda meta: (
                    Z
                    * triton.cdiv(M, meta["BLOCK_M"])
                    * triton.cdiv(T, meta["BLOCK_T"]),
                )
                gatedmlp_separated_y_kernel[grid](
                    a2,
                    d,
                    y,  # fmt: skip
                    a2.stride(0),
                    a2.stride(1),
                    a2.stride(2),  # fmt: skip
                    d.stride(0),
                    d.stride(1),  # fmt: skip
                    y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
                    Z,
                    M,
                    K,
                    T,  # fmt: skip
                    BLOCK_M=config["BLOCK_M"],
                    BLOCK_K=config["BLOCK_K"],
                    BLOCK_T=config["BLOCK_T"],
                )
                # input(y)
                success_configs.append(
                    {
                        "config": config,
                        "num_stages": triton_config.num_stages,
                        "num_warps": triton_config.num_warps,
                    }
                )
            else:
                raise NotImplementedError("Activation fn other than Silu unsupported")
        except triton.runtime.errors.OutOfResources as e:
            logger.error(f"Autotuning failed with {e}")
    return success_configs


# @torch.compile(fullgraph=True)
def gatedmlp_torch(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    dtype = x.dtype
    x, u, g, d = (
        x.to(torch.float32),
        u.to(torch.float32),
        g.to(torch.float32),
        d.to(torch.float32),
    )
    a1 = torch.matmul(x, u)
    ag = torch.matmul(x, g)
    act_fn = ACT2FN[act]
    aa = act_fn(ag)
    a2 = torch.mul(a1, aa)
    # print("TORCH:", a2[0, 0:5, 5:10])
    y = torch.matmul(a2, d)
    return y.to(dtype)


if __name__ == "__main__":
    # Test correctness.
    M, N, K, T = 1, 8192, 7168, 8192
    x = torch.randn((1, M, N), device=DEVICE, dtype=torch.float16)
    u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
    d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
    success_configs = profile_gatedmlp_separated(x, u, g, d)
    logger.info("Finished profiling.")
    print("Successful configs:")
    # print(success_configs)
    print(len(success_configs))
