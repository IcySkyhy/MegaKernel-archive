import logging
import torch
import triton
import triton.language as tl

from ..utils import is_hip


def get_autotune_config(vec_input=False):
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
        for BM in [16, 32, 64, 128]
        for BN in [16, 32, 64, 128]
        for BK in [16, 32, 64, 128]
        for s in ([2] if is_hip() else [3, 4, 5])
        for w in [2, 4, 8]
    ]
    return configs


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_silu_kernel(
    X, W, Y,  # fmt: skip
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_wn, stride_wk,  # fmt: skip
    stride_yz, stride_ym, stride_yk,  # fmt: skip
    # M = seq_len, N = d_h, K = d_out
    Z,
    M,
    N,
    K,  # fmt: skip
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
):
    """
    For debugging roofline profiling...
    GEMM kernel with SiLU.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (M // BLOCK_M) * (K // BLOCK_K))
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    off_z = pid // (num_pid_m * num_pid_k)
    pid_example = pid % (num_pid_m * num_pid_k)
    pid_m = pid_example // num_pid_k
    pid_k = pid_example % num_pid_k
    x_offset = off_z.to(tl.int64) * stride_xz
    y_offset = off_z.to(tl.int64) * stride_yz

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(M, K),
        strides=(stride_ym, stride_yk),
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
    w_ptr = tl.make_block_ptr(
        base=W,
        shape=(N, K),
        strides=(stride_wn, stride_wk),
        offsets=(0, pid_k * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    accy = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in tl.range(0, tl.cdiv(N, BLOCK_N)):
        x = tl.load(x_ptr)
        w = tl.load(w_ptr)

        accy = tl.dot(x, w, accy)

        x_ptr = tl.advance(x_ptr, (0, BLOCK_N))
        w_ptr = tl.advance(w_ptr, (BLOCK_N, 0))

    # SiLU
    sigm = tl.sigmoid(accy)
    accy = sigm * accy

    tl.store(y_ptr, accy.to(Y.type.element_ty))


@triton.autotune(
    configs=get_autotune_config(vec_input=True),
    key=["N", "K"],
)
@triton.jit
def matmul_silu_vec_kernel(
    X, W, Y,  # fmt: skip
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_wn, stride_wk,  # fmt: skip
    stride_yz, stride_ym, stride_yk,  # fmt: skip
    # M = seq_len, N = d_h, K = d_out
    Z,
    M,
    N,
    K,  # fmt: skip
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
):
    """
    For debugging roofline profiling...
    GEMV kernel with SiLU.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (K // BLOCK_K), M)
    off_m = tl.program_id(axis=1)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    off_z = pid // num_pid_k
    pid_k = pid % num_pid_k
    x_offset = off_z.to(tl.int64) * stride_xz + off_m.to(tl.int64) * stride_xm
    y_offset = off_z.to(tl.int64) * stride_yz + off_m.to(tl.int64) * stride_ym

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(K,),
        strides=(stride_yk,),
        offsets=(pid_k * BLOCK_K,),
        block_shape=(BLOCK_K,),
        order=(0,),
    )
    x_ptr = tl.make_block_ptr(
        base=X + x_offset,
        shape=(N,),
        strides=(stride_xn,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    w_ptr = tl.make_block_ptr(
        base=W,
        shape=(N, K),
        strides=(stride_wn, stride_wk),
        offsets=(0, pid_k * BLOCK_K),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    accy = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for n in tl.range(0, tl.cdiv(N, BLOCK_N)):
        x = tl.load(x_ptr)
        w = tl.load(w_ptr)

        accy += tl.sum(x.to(tl.float32)[:, None] * w.to(tl.float32), axis=0)

        x_ptr = tl.advance(x_ptr, (BLOCK_N,))
        w_ptr = tl.advance(w_ptr, (BLOCK_N, 0))

    # SiLU
    sigm = tl.sigmoid(accy)
    accy = sigm * accy

    tl.store(y_ptr, accy.to(Y.type.element_ty))


def matmul_silu(x, w):
    assert x.shape[-1] == w.shape[0]
    assert x.dtype == w.dtype

    Z, M, N = x.shape
    N, K = w.shape

    y = torch.empty((Z, M, K), device=x.device, dtype=x.dtype)

    logging.debug("Invoking GEMM kernel")
    grid = lambda META: (
        Z * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_K"]),
    )
    matmul_silu_kernel[grid](
        x, w, y,  # fmt: skip
        x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
        w.stride(0), w.stride(1),  # fmt: skip
        y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
        Z, M, N, K,  # fmt: skip
    )

    return y


def matmul_silu_vec(x, w):
    assert x.shape[-1] == w.shape[0]
    assert x.dtype == w.dtype

    Z, M, N = x.shape
    N, K = w.shape

    y = torch.empty((Z, M, K), device=x.device, dtype=x.dtype)

    logging.debug("Invoking GEMV kernel")
    grid = lambda META: (Z * triton.cdiv(K, META["BLOCK_K"]), M)
    matmul_silu_vec_kernel[grid](
        x, w, y,  # fmt: skip
        x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
        w.stride(0), w.stride(1),  # fmt: skip
        y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
        Z, M, N, K,  # fmt: skip
    )

    return y


def matmul_silu_torch(x, w):
    dtype = x.dtype
    x, w = x.to(torch.float32), w.to(torch.float32)
    a = torch.matmul(x, w)
    act_fn = torch.nn.functional.silu
    y = act_fn(a)
    return y.to(dtype)
