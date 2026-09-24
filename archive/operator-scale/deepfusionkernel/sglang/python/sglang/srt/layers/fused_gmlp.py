import logging
import torch

import triton
import triton.language as tl

from transformers.activations import ACT2FN


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"



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
                {**({} if vec_input else {'BLOCK_M': BM}), **{'BLOCK_N': BN, 'BLOCK_K': BK, 'BLOCK_T': BT}}, 
                num_stages=s, 
                num_warps=w,
            ) \
            # for BM in [32]\
            # for BN in [32]\
            # for BK in [32]\
            # for BT in [32]\
            # for s in ([2] if is_hip() else [4])\
            # for w in [4]\
            for BM in [32, 64, 128, 256]\
            for BN in [32, 64, 128, 256]\
            for BK in [32, 64, 128, 256]\
            for BT in [32, 64, 128, 256]\
            for s in ([2] if is_hip() else [3, 4, 5])\
            for w in [2, 4, 8]\
        ]
    elif separated_kernel == 1:
        configs = [
            triton.Config(
                {**({} if vec_input else {'BLOCK_M': BM}), **{'BLOCK_N': BN, 'BLOCK_K': BK}}, 
                num_stages=s, 
                num_warps=w,
            ) \
            # for BM in [32]\
            # for BN in [32]\
            # for BK in [32]\
            # for s in ([2] if is_hip() else [4])\
            # for w in [4]\
            for BM in [32, 64, 128, 256]\
            for BN in [32, 64, 128, 256]\
            for BK in [32, 64, 128, 256]\
            for s in ([2] if is_hip() else [3, 4, 5])\
            for w in [2, 4, 8]\
        ]
    elif separated_kernel == 2:
        configs = [
            triton.Config(
                {**({} if vec_input else {'BLOCK_M': BM}), **{'BLOCK_K': BK, 'BLOCK_T': BT}}, 
                num_stages=s, 
                num_warps=w,
            ) \
            # for BM in [32]\
            # for BK in [32]\
            # for BT in [32]\
            # for s in ([2] if is_hip() else [4])\
            # for w in [4]\
            for BM in [32, 64, 128, 256]\
            for BK in [32, 64, 128, 256]\
            for BT in [32, 64, 128, 256]\
            for s in ([2] if is_hip() else [3, 4, 5])\
            for w in [2, 4, 8]\
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


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K", "T"],
)
@triton.jit
def gatedmlp_m_tkn_kernel(
    X, U, G, D, Y,  # fmt: skip
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate 
    Z, M, N, K, T,  # fmt: skip
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
):
    """
    Kernel for gated MLP with ACT as Silu and loop order `[m, t, k, n]`
    (i.e. tiling `Y` in row-major) with grid `[m]`.
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (M // BLOCK_M))
    num_pid_m = tl.cdiv(M, BLOCK_M)
    off_z = pid // num_pid_m
    pid_m = pid % num_pid_m
    x_offset = off_z.to(tl.int64) * stride_xz
    y_offset = off_z.to(tl.int64) * stride_yz

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(M, T),
        strides=(stride_ym, stride_yt),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_T),
        # `order` is the logical access pattern, for Triton optimisations for Hopper TMA only.
        # It has nothing to do with the underlying order the matrix is stored.
        # It concerns element access pattern within each tile, not tile access pattern.
        # The memory hierarchy may also matter e.g. concurrent access to memory banks.
        # Use (1,0) when either column-major access or column-major original data layout, but not when both. (???)
        # TODO: experiment on this. Seems not quite intrinsic...
        order=(0, 1),
    )
    for t in tl.range(0, tl.cdiv(T, BLOCK_T)):
        d_ptr = tl.make_block_ptr(
            base=D,
            shape=(K, T),
            strides=(stride_dk, stride_dt),
            offsets=(0, t * BLOCK_T),
            block_shape=(BLOCK_K, BLOCK_T),
            order=(1, 0),
        )
        accy = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)  # accy = accu2 @ D
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            d = tl.load(d_ptr, boundary_check=(0, 1), padding_option="zero")
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
                offsets=(0, k * BLOCK_K),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            g_ptr = tl.make_block_ptr(
                base=G,
                shape=(N, K),
                strides=(stride_gn, stride_gk),
                offsets=(0, k * BLOCK_K),
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
            accy = tl.dot(accu, d.to(tl.float32), accy)
            d_ptr = tl.advance(d_ptr, (BLOCK_K, 0))

        tl.store(y_ptr, accy.to(Y.type.element_ty), boundary_check=(0, 1))
        y_ptr = tl.advance(y_ptr, (0, BLOCK_T))


@triton.autotune(
    configs=get_autotune_config(vec_input=True),
    key=["N", "K", "T"],
)
@triton.jit
def gatedmlp_m_tkn_vec_kernel(
    X, U, G, D, Y,  # fmt: skip
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    # M = 1, N (= T) = d_h, K = d_intermediate 
    Z, M, N, K, T,  # fmt: skip
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
):
    """
    Kernel for gated MLP with ACT as Silu and loop order `[m=1, t, k, n]`
    (i.e. tiling `Y` in row-major) with grid `[m=1]`.
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z, M)
    off_z = pid
    off_m = tl.program_id(axis=1)
    x_offset = off_z.to(tl.int64) * stride_xz + off_m.to(tl.int64) * stride_xm
    y_offset = off_z.to(tl.int64) * stride_yz + off_m.to(tl.int64) * stride_ym

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(T, ),
        strides=(stride_yt, ),
        offsets=(0, ),
        block_shape=(BLOCK_T, ),
        order=(0, ),
    )
    for t in tl.range(0, tl.cdiv(T, BLOCK_T)):
        d_ptr = tl.make_block_ptr(
            base=D,
            shape=(K, T),
            strides=(stride_dk, stride_dt),
            offsets=(0, t * BLOCK_T),
            block_shape=(BLOCK_K, BLOCK_T),
            order=(1, 0),
        )
        accy = tl.zeros((BLOCK_T, ), dtype=tl.float32)  # accy = accu2 @ D
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            d = tl.load(d_ptr, boundary_check=(0, 1), padding_option="zero")
            x_ptr = tl.make_block_ptr(
                base=X + x_offset,
                shape=(N, ),
                strides=(stride_xn, ),
                offsets=(0, ),
                block_shape=(BLOCK_N, ),
                order=(0, ),
            )
            u_ptr = tl.make_block_ptr(
                base=U,
                shape=(N, K),
                strides=(stride_un, stride_uk),
                offsets=(0, k * BLOCK_K),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            g_ptr = tl.make_block_ptr(
                base=G,
                shape=(N, K),
                strides=(stride_gn, stride_gk),
                offsets=(0, k * BLOCK_K),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            accu = tl.zeros((BLOCK_K, ), dtype=tl.float32)  # accu = X @ U
            accg = tl.zeros((BLOCK_K, ), dtype=tl.float32)  # accg = X @ G
            for n in tl.range(0, tl.cdiv(N, BLOCK_N)):
                x = tl.load(x_ptr, boundary_check=(0, ), padding_option="zero")
                u = tl.load(u_ptr, boundary_check=(0, 1), padding_option="zero")
                g = tl.load(g_ptr, boundary_check=(0, 1), padding_option="zero")

                accu += tl.sum(x.to(tl.float32)[:, None] * u.to(tl.float32), axis=0)
                accg += tl.sum(x.to(tl.float32)[:, None] * g.to(tl.float32), axis=0)

                x_ptr = tl.advance(x_ptr, (BLOCK_N, ))
                u_ptr = tl.advance(u_ptr, (BLOCK_N, 0))
                g_ptr = tl.advance(g_ptr, (BLOCK_N, 0))

            acca = tl.sigmoid(accg)  # acca = sigmoid(X@G)
            accg = acca * accg  # Silu(accg) = accg * sigmoid(accg)
            accu = accg * accu  # accu2 = accu * Silu(accg)
            accy += tl.sum(accu.to(tl.float32)[:, None] * d.to(tl.float32), axis=0)
            d_ptr = tl.advance(d_ptr, (BLOCK_K, 0))

        tl.store(y_ptr, accy.to(Y.type.element_ty), boundary_check=(0, ))
        y_ptr = tl.advance(y_ptr, (BLOCK_T, ))


# @torch.compile(fullgraph=True)
def gatedmlp_m_tkn(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with `[m, t, k, n]` and grid `[m]`
    """
    # configs = {
    #     torch.float8_e4m3fn: {
    #         "BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_T": 32, "num_stages": 4,
    #         "num_warps": 8
    #     },
    #     torch.float16: {
    #         "BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_T": 32, "num_stages": 3,
    #         "num_warps": 8
    #     }
    # }
    # Check constraints.
    assert x.shape[2] == u.shape[0], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[1] == d.shape[0], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    Z, M, N = x.shape
    N, K = u.shape
    K, T = d.shape
    # Allocates output.
    y = torch.empty((Z, M, T), device=x.device, dtype=x.dtype)
    # launch kernel
    if act == "silu":
        grid = lambda META: (Z * triton.cdiv(M, META["BLOCK_M"]),)
        gatedmlp_m_tkn_kernel[grid](
            x, u, g, d, y,  # fmt: skip
            x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
            u.stride(0), u.stride(1),  # fmt: skip
            g.stride(0), g.stride(1),  # fmt: skip
            d.stride(0), d.stride(1),  # fmt: skip
            y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
            Z, M, N, K, T,  # fmt: skip
            # BLOCK_M=configs[dtype]["BLOCK_M"],  #
            # BLOCK_N=configs[dtype]["BLOCK_N"],  #
            # BLOCK_K=configs[dtype]["BLOCK_K"],  #
            # BLOCK_T=configs[dtype]["BLOCK_T"],  #
            # num_stages=configs[dtype]["num_stages"],  #
            # num_warps=configs[dtype]["num_warps"],  #
        )
    else:
        raise NotImplementedError("Activation fn other than Silu unsupported")
    return y


def gatedmlp_m_tkn_vec(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with `[m, t, k, n]` and grid `[m]`
    """
    # configs = {
    #     torch.float8_e4m3fn: {
    #         "BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_T": 32, "num_stages": 4,
    #         "num_warps": 8
    #     },
    #     torch.float16: {
    #         "BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16, "BLOCK_T": 32, "num_stages": 3,
    #         "num_warps": 8
    #     }
    # }
    # Check constraints.
    assert x.shape[2] == u.shape[0], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[1] == d.shape[0], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    Z, M, N = x.shape
    N, K = u.shape
    K, T = d.shape
    # Allocates output.
    y = torch.empty((Z, M, T), device=x.device, dtype=x.dtype)
    # launch kernel
    if act == "silu":
        grid = lambda META: (Z, M)
        gatedmlp_m_tkn_vec_kernel[grid](
            x, u, g, d, y,  # fmt: skip
            x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
            u.stride(0), u.stride(1),  # fmt: skip
            g.stride(0), g.stride(1),  # fmt: skip
            d.stride(0), d.stride(1),  # fmt: skip
            y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
            Z, M, N, K, T,  # fmt: skip
        )
    else:
        raise NotImplementedError("Activation fn other than Silu unsupported")
    return y



@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K", "T"],
)
@triton.jit
def gatedmlp_mt_kn_kernel(
    X, U, G, D, Y,  # fmt: skip
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate 
    Z, M, N, K, T,  # fmt: skip
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
):
    """
    Kernel for gated MLP with ACT as Silu and loop order `[m, t, k, n]`
    (i.e. tiling `Y` in row-major) with grid `[m, t]`.
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (M // BLOCK_M) * (T // BLOCK_T))
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_t = tl.cdiv(T, BLOCK_T)
    off_z = pid // (num_pid_m * num_pid_t)
    pid_sample = pid % (num_pid_m * num_pid_t)
    pid_m = pid_sample // num_pid_t
    pid_t = pid_sample % num_pid_t
    x_offset = off_z.to(tl.int64) * stride_xz
    y_offset = off_z.to(tl.int64) * stride_yz

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(M, T),
        strides=(stride_ym, stride_yt),
        offsets=(pid_m * BLOCK_M, pid_t * BLOCK_T),
        block_shape=(BLOCK_M, BLOCK_T),
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
        d = tl.load(d_ptr, boundary_check=(0, 1), padding_option="zero")
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
            offsets=(0, k * BLOCK_K),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )
        g_ptr = tl.make_block_ptr(
            base=G,
            shape=(N, K),
            strides=(stride_gn, stride_gk),
            offsets=(0, k * BLOCK_K),
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
        accy = tl.dot(accu, d.to(tl.float32), accy)
        d_ptr = tl.advance(d_ptr, (BLOCK_K, 0))

    tl.store(y_ptr, accy.to(Y.type.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=get_autotune_config(vec_input=True),
    key=["N", "K", "T"],
)
@triton.jit
def gatedmlp_mt_kn_vec_kernel(
    X, U, G, D, Y,  # fmt: skip
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    # M = 1, N (= T) = d_h, K = d_intermediate 
    Z, M, N, K, T,  # fmt: skip
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
):
    """
    Kernel for gated MLP with ACT as Silu and loop order `[m=1, t, k, n]`
    (i.e. tiling `Y` in row-major) with grid `[m=1, t]`.
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (T // BLOCK_T), M)
    num_pid_t = tl.cdiv(T, BLOCK_T)
    off_z = pid // num_pid_t
    pid_t = pid % num_pid_t
    off_m = tl.program_id(axis=1)
    x_offset = off_z.to(tl.int64) * stride_xz + off_m.to(tl.int64) * stride_xm
    y_offset = off_z.to(tl.int64) * stride_yz + off_m.to(tl.int64) * stride_ym

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(T, ),
        strides=(stride_yt, ),
        offsets=(pid_t * BLOCK_T, ),
        block_shape=(BLOCK_T, ),
        order=(0, ),
    )
    d_ptr = tl.make_block_ptr(
        base=D,
        shape=(K, T),
        strides=(stride_dk, stride_dt),
        offsets=(0, pid_t * BLOCK_T),
        block_shape=(BLOCK_K, BLOCK_T),
        order=(1, 0),
    )
    accy = tl.zeros((BLOCK_T, ), dtype=tl.float32)  # accy = accu2 @ D
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        d = tl.load(d_ptr, boundary_check=(0, 1), padding_option="zero")
        x_ptr = tl.make_block_ptr(
            base=X + x_offset,
            shape=(N, ),
            strides=(stride_xn, ),
            offsets=(0, ),
            block_shape=(BLOCK_N, ),
            order=(0, ),
        )
        u_ptr = tl.make_block_ptr(
            base=U,
            shape=(N, K),
            strides=(stride_un, stride_uk),
            offsets=(0, k * BLOCK_K),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )
        g_ptr = tl.make_block_ptr(
            base=G,
            shape=(N, K),
            strides=(stride_gn, stride_gk),
            offsets=(0, k * BLOCK_K),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )
        accu = tl.zeros((BLOCK_K, ), dtype=tl.float32)  # accu = X @ U
        accg = tl.zeros((BLOCK_K, ), dtype=tl.float32)  # accg = X @ G
        for n in tl.range(0, tl.cdiv(N, BLOCK_N)):
            x = tl.load(x_ptr, boundary_check=(0, ), padding_option="zero")
            u = tl.load(u_ptr, boundary_check=(0, 1), padding_option="zero")
            g = tl.load(g_ptr, boundary_check=(0, 1), padding_option="zero")

            accu += tl.sum(x.to(tl.float32)[:, None] * u.to(tl.float32), axis=0)
            accg += tl.sum(x.to(tl.float32)[:, None] * g.to(tl.float32), axis=0)

            x_ptr = tl.advance(x_ptr, (BLOCK_N, ))
            u_ptr = tl.advance(u_ptr, (BLOCK_N, 0))
            g_ptr = tl.advance(g_ptr, (BLOCK_N, 0))

        acca = tl.sigmoid(accg)  # acca = sigmoid(X@G)
        accg = acca * accg  # Silu(accg) = accg * sigmoid(accg)
        accu = accg * accu  # accu2 = accu * Silu(accg)
        accy += tl.sum(accu.to(tl.float32)[:, None] * d.to(tl.float32), axis=0)
        d_ptr = tl.advance(d_ptr, (BLOCK_K, 0))

    tl.store(y_ptr, accy.to(Y.type.element_ty), boundary_check=(0, ))


# @torch.compile(fullgraph=True)
def gatedmlp_mt_kn(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with `[m, t, k, n]` and grid `[m, t]`
    """
    # Check constraints.
    assert x.shape[2] == u.shape[0], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[1] == d.shape[0], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    Z, M, N = x.shape
    N, K = u.shape
    K, T = d.shape
    # Allocates output.
    y = torch.empty((Z, M, T), device=x.device, dtype=x.dtype)
    # launch kernel
    if act == "silu":
        grid = lambda META: (Z * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(T, META["BLOCK_T"]),)
        gatedmlp_mt_kn_kernel[grid](
            x, u, g, d, y,  # fmt: skip
            x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
            u.stride(0), u.stride(1),  # fmt: skip
            g.stride(0), g.stride(1),  # fmt: skip
            d.stride(0), d.stride(1),  # fmt: skip
            y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
            Z, M, N, K, T,  # fmt: skip
        )
    else:
        raise NotImplementedError("Activation fn other than Silu unsupported")
    return y


def gatedmlp_mt_kn_vec(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with `[m, t, k, n]` and grid `[m, t]`
    """
    # Check constraints.
    assert x.shape[2] == u.shape[0], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[1] == d.shape[0], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    Z, M, N = x.shape
    N, K = u.shape
    K, T = d.shape
    # Allocates output.
    y = torch.empty((Z, M, T), device=x.device, dtype=x.dtype)
    # launch kernel
    if act == "silu":
        grid = lambda META: (Z * triton.cdiv(T, META["BLOCK_T"]), M)
        gatedmlp_mt_kn_vec_kernel[grid](
            x, u, g, d, y,  # fmt: skip
            x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
            u.stride(0), u.stride(1),  # fmt: skip
            g.stride(0), g.stride(1),  # fmt: skip
            d.stride(0), d.stride(1),  # fmt: skip
            y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
            Z, M, N, K, T,  # fmt: skip
        )
    else:
        raise NotImplementedError("Activation fn other than Silu unsupported")
    return y



@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K", "T"],
)
@triton.jit
def gatedmlp_mt_kn_keepA2_kernel(
    X, U, G, D, Y,  # fmt: skip
    A2,  #fmt: skip
    Lock, 
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    stride_a2z, stride_a2m, stride_a2k,  #fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate 
    Z, M, N, K, T,  # fmt: skip
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
):
    """
    Kernel for gated MLP with ACT as Silu and loop order `[m, t, k, n]`
    (i.e. tiling `Y` in row-major) with grid `[m, t]`.
    Compute A2 = (X@W1) * ACT(X@G) only in the first program of each block_m.
    Theoretically better than gatedmlp_m_tkn_keepA2 and gatedmlp_t_mkn_keepA2.
    No L2 cache optimization.
    """
    # Use mutex for waiting for A2; need to either store computed A2 in cache, or load A2 from HBM.
    # Need to call tl.store(A2) to store A2 to L2 cache for sharing across programs;
    # Thus need to allocate space for A2 in global memory.
    # Order tiles in [m, t] is better than [t, m] for cache reuse.
    # Need to use atomic ops for locks for (actually not) mutex.
    
    pid = tl.program_id(axis=0)  # grid: (Z * (M // BLOCK_M) * (T // BLOCK_T))
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_t = tl.cdiv(T, BLOCK_T)
    off_z = pid // (num_pid_m * num_pid_t)
    pid_sample = pid % (num_pid_m * num_pid_t)
    pid_m = pid_sample // num_pid_t
    pid_t = pid_sample % num_pid_t
    x_offset = off_z.to(tl.int64) * stride_xz
    a2_offset = off_z.to(tl.int64) * stride_a2z
    y_offset = off_z.to(tl.int64) * stride_yz
    # Atomic lock offset
    lock_id = off_z * num_pid_m + pid_m
    Lock += lock_id
    
    # Only dim_t == 0 can update the lock
    if pid_t == 0:
        a2_ptr = tl.make_block_ptr(
            base=A2 + a2_offset,
            shape=(M, K),
            strides=(stride_a2m, stride_a2k),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(0, 1),
        )
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
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
                offsets=(0, k * BLOCK_K),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            g_ptr = tl.make_block_ptr(
                base=G,
                shape=(N, K),
                strides=(stride_gn, stride_gk),
                offsets=(0, k * BLOCK_K),
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
            a2_ptr = tl.advance(a2_ptr, (0, BLOCK_K))
        tl.atomic_xchg(Lock, 1, sem="release")
    else:
        while tl.atomic_cas(Lock, 1, 1, sem="acquire") == 0:
            pass

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


@triton.autotune(
    configs=get_autotune_config(vec_input=True),
    key=["N", "K", "T"],
)
@triton.jit
def gatedmlp_mt_kn_keepA2_vec_kernel(
    X, U, G, D, Y,  # fmt: skip
    A2,  #fmt: skip
    Lock, 
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    stride_a2z, stride_a2m, stride_a2k,  #fmt: skip
    # M = 1, N (= T) = d_h, K = d_intermediate 
    Z, M, N, K, T,  # fmt: skip
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
):
    """
    Kernel for gated MLP with ACT as Silu and loop order `[m=1, t, k, n]`
    (i.e. tiling `Y` in row-major) with grid `[m=1, t]`.
    Compute A2 = (X@W1) * ACT(X@G) only in the first program of each block_m.
    Theoretically better than gatedmlp_m_tkn_keepA2 and gatedmlp_t_mkn_keepA2.
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (T // BLOCK_T), M)
    num_pid_t = tl.cdiv(T, BLOCK_T)
    off_z = pid // num_pid_t
    pid_t = pid % num_pid_t
    off_m = tl.program_id(axis=1)
    x_offset = off_z.to(tl.int64) * stride_xz + off_m.to(tl.int64) * stride_xm
    a2_offset = off_z.to(tl.int64) * stride_a2z + off_m.to(tl.int64) * stride_a2m
    y_offset = off_z.to(tl.int64) * stride_yz + off_m.to(tl.int64) * stride_ym
    # Atomic lock offset
    num_pid_m = M
    lock_id = off_z * num_pid_m + off_m
    Lock += lock_id

    # Only dim_t == 0 can update the lock
    if pid_t == 0:
        a2_ptr = tl.make_block_ptr(
            base=A2 + a2_offset,
            shape=(K, ),
            strides=(stride_a2k, ),
            offsets=(0, ),
            block_shape=(BLOCK_K, ),
            order=(0, ),
        )
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            x_ptr = tl.make_block_ptr(
                base=X + x_offset,
                shape=(N, ),
                strides=(stride_xn, ),
                offsets=(0, ),
                block_shape=(BLOCK_N, ),
                order=(0, ),
            )
            u_ptr = tl.make_block_ptr(
                base=U,
                shape=(N, K),
                strides=(stride_un, stride_uk),
                offsets=(0, k * BLOCK_K),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            g_ptr = tl.make_block_ptr(
                base=G,
                shape=(N, K),
                strides=(stride_gn, stride_gk),
                offsets=(0, k * BLOCK_K),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            accu = tl.zeros((BLOCK_K, ), dtype=tl.float32)  # accu = X @ U
            accg = tl.zeros((BLOCK_K, ), dtype=tl.float32)  # accg = X @ G
            for n in tl.range(0, tl.cdiv(N, BLOCK_N)):
                x = tl.load(x_ptr, boundary_check=(0, ), padding_option="zero")
                u = tl.load(u_ptr, boundary_check=(0, 1), padding_option="zero")
                g = tl.load(g_ptr, boundary_check=(0, 1), padding_option="zero")

                accu += tl.sum(x.to(tl.float32)[:, None] * u.to(tl.float32), axis=0)
                accg += tl.sum(x.to(tl.float32)[:, None] * g.to(tl.float32), axis=0)

                x_ptr = tl.advance(x_ptr, (BLOCK_N, ))
                u_ptr = tl.advance(u_ptr, (BLOCK_N, 0))
                g_ptr = tl.advance(g_ptr, (BLOCK_N, 0))

            acca = tl.sigmoid(accg)  # acca = sigmoid(X@G)
            accg = acca * accg  # Silu(accg) = accg * sigmoid(accg)
            accu = accg * accu  # accu2 = accu * Silu(accg)
            tl.store(a2_ptr, accu.to(A2.type.element_ty), boundary_check=(0, ))
            a2_ptr = tl.advance(a2_ptr, (BLOCK_K, ))
        tl.atomic_xchg(Lock, 1, sem="release")
    else:
        while tl.atomic_cas(Lock, 1, 1, sem="acquire") == 0:
            pass

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(T, ),
        strides=(stride_yt, ),
        offsets=(pid_t * BLOCK_T, ),
        block_shape=(BLOCK_T, ),
        order=(0, ),
    )
    a2_ptr = tl.make_block_ptr(
        base=A2 + a2_offset,
        shape=(K, ),
        strides=(stride_a2k, ),
        offsets=(0, ),
        block_shape=(BLOCK_K, ),
        order=(0, ),
    )
    d_ptr = tl.make_block_ptr(
        base=D,
        shape=(K, T),
        strides=(stride_dk, stride_dt),
        offsets=(0, pid_t * BLOCK_T),
        block_shape=(BLOCK_K, BLOCK_T),
        order=(1, 0),
    )
    accy = tl.zeros((BLOCK_T, ), dtype=tl.float32)  # accy = accu2 @ D
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a2 = tl.load(a2_ptr, boundary_check=(0, ), padding_option="zero")
        d = tl.load(d_ptr, boundary_check=(0, 1), padding_option="zero")
        accy += tl.sum(a2.to(tl.float32)[:, None] * d.to(tl.float32), axis=0)
        a2_ptr = tl.advance(a2_ptr, (BLOCK_K, ))
        d_ptr = tl.advance(d_ptr, (BLOCK_K, 0))
    tl.store(y_ptr, accy.to(Y.type.element_ty), boundary_check=(0, ))


# @torch.compile(fullgraph=True)
def gatedmlp_mt_kn_keepA2(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with `[m, t, k, n]` and grid `[m, t]`, 
    reusing computed A2 = (X@W1) * ACT(X@G).
    """
    # Check constraints.
    assert x.shape[2] == u.shape[0], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[1] == d.shape[0], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    Z, M, N = x.shape
    N, K = u.shape
    K, T = d.shape
    # Allocate locks
    locks = torch.zeros((Z, M), device=x.device, dtype=torch.int32)  # Should be (Z, M//BLOCK_M)
    # Allocate output.
    a2 = torch.empty((Z, M, K), device=x.device, dtype=x.dtype)
    y = torch.empty((Z, M, T), device=x.device, dtype=x.dtype)
    # launch kernel
    if act == "silu":
        grid = lambda META: (Z * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(T, META["BLOCK_T"]),)
        gatedmlp_mt_kn_keepA2_kernel[grid](
            x, u, g, d, y,  # fmt: skip
            a2,  # fmt:skip
            locks,  # fmt: skip
            x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
            u.stride(0), u.stride(1),  # fmt: skip
            g.stride(0), g.stride(1),  # fmt: skip
            d.stride(0), d.stride(1),  # fmt: skip
            y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
            a2.stride(0), a2.stride(1), a2.stride(2),  # fmt: skip
            Z, M, N, K, T,  # fmt: skip
        )
    else:
        raise NotImplementedError("Activation fn other than Silu unsupported")
    return y


def gatedmlp_mt_kn_keepA2_vec(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with `[m, t, k, n]` and grid `[m, t]`, 
    reusing computed A2 = (X@W1) * ACT(X@G).
    """
    # Check constraints.
    assert x.shape[2] == u.shape[0], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[1] == d.shape[0], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    Z, M, N = x.shape
    N, K = u.shape
    K, T = d.shape
    # Allocate locks
    locks = torch.zeros((Z, M), device=x.device, dtype=torch.int32)  # Should be (Z, M//BLOCK_M)
    # Allocate output.
    a2 = torch.empty((Z, M, K), device=x.device, dtype=x.dtype)
    y = torch.empty((Z, M, T), device=x.device, dtype=x.dtype)
    # launch kernel
    if act == "silu":
        grid = lambda META: (Z * triton.cdiv(T, META["BLOCK_T"]), M)
        gatedmlp_mt_kn_keepA2_vec_kernel[grid](
            x, u, g, d, y,  # fmt: skip
            a2,  # fmt:skip
            locks,  # fmt: skip
            x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
            u.stride(0), u.stride(1),  # fmt: skip
            g.stride(0), g.stride(1),  # fmt: skip
            d.stride(0), d.stride(1),  # fmt: skip
            y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
            a2.stride(0), a2.stride(1), a2.stride(2),  # fmt: skip
            Z, M, N, K, T,  # fmt: skip
        )
    else:
        raise NotImplementedError("Activation fn other than Silu unsupported")
    return y



@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K", "T"],
)
@triton.jit
def gatedmlp_t_mkn_kernel(
    X, U, G, D, Y,  # fmt: skip
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate 
    Z, M, N, K, T,  # fmt: skip
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
):
    """
    Kernel for gated MLP with ACT as Silu and loop order `[t, m, k, n]`
    (i.e. tiling `Y` in column-major) with grid `[t]`.
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (T // BLOCK_T))
    num_pid_t = tl.cdiv(T, BLOCK_T)
    off_z = pid // num_pid_t
    pid_t = pid % num_pid_t
    x_offset = off_z.to(tl.int64) * stride_xz
    y_offset = off_z.to(tl.int64) * stride_yz

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(M, T),
        strides=(stride_ym, stride_yt),
        offsets=(0, pid_t * BLOCK_T),
        block_shape=(BLOCK_M, BLOCK_T),
        order=(0, 1),
    )
    for m in tl.range(0, tl.cdiv(M, BLOCK_M)):
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
            d = tl.load(d_ptr, boundary_check=(0, 1), padding_option="zero")
            x_ptr = tl.make_block_ptr(
                base=X + x_offset,
                shape=(M, N),
                strides=(stride_xm, stride_xn),
                offsets=(m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(0, 1),
            )
            u_ptr = tl.make_block_ptr(
                base=U,
                shape=(N, K),
                strides=(stride_un, stride_uk),
                offsets=(0, k * BLOCK_K),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            g_ptr = tl.make_block_ptr(
                base=G,
                shape=(N, K),
                strides=(stride_gn, stride_gk),
                offsets=(0, k * BLOCK_K),
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
            accy = tl.dot(accu, d.to(tl.float32), accy)
            d_ptr = tl.advance(d_ptr, (BLOCK_K, 0))

        tl.store(y_ptr, accy.to(Y.type.element_ty), boundary_check=(0, 1))
        y_ptr = tl.advance(y_ptr, (BLOCK_M, 0))


@triton.autotune(
    configs=get_autotune_config(vec_input=True),
    key=["N", "K", "T"],
)
@triton.jit
def gatedmlp_t_mkn_vec_kernel(
    X, U, G, D, Y,  # fmt: skip
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    # M = 1, N (= T) = d_h, K = d_intermediate 
    Z, M, N, K, T,  # fmt: skip
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
):
    """
    Should be the same as gatedmlp_mt_kn_vec_kernel.
    Kernel for gated MLP with ACT as Silu and loop order `[t, m=1, k, n]`
    (i.e. tiling `Y` in column-major) with grid `[t]`.
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (T // BLOCK_T), M)
    num_pid_t = tl.cdiv(T, BLOCK_T)
    off_z = pid // num_pid_t
    pid_t = pid % num_pid_t
    off_m = tl.program_id(axis=1)
    x_offset = off_z.to(tl.int64) * stride_xz + off_m.to(tl.int64) * stride_xm
    y_offset = off_z.to(tl.int64) * stride_yz + off_m.to(tl.int64) * stride_ym

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(T, ),
        strides=(stride_yt, ),
        offsets=(pid_t * BLOCK_T, ),
        block_shape=(BLOCK_T, ),
        order=(0, ),
    )
    d_ptr = tl.make_block_ptr(
        base=D,
        shape=(K, T),
        strides=(stride_dk, stride_dt),
        offsets=(0, pid_t * BLOCK_T),
        block_shape=(BLOCK_K, BLOCK_T),
        order=(1, 0),
    )
    accy = tl.zeros((BLOCK_T, ), dtype=tl.float32)  # accy = accu2 @ D
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        d = tl.load(d_ptr, boundary_check=(0, 1), padding_option="zero")
        x_ptr = tl.make_block_ptr(
            base=X + x_offset,
            shape=(N, ),
            strides=(stride_xn, ),
            offsets=(0, ),
            block_shape=(BLOCK_N, ),
            order=(0, ),
        )
        u_ptr = tl.make_block_ptr(
            base=U,
            shape=(N, K),
            strides=(stride_un, stride_uk),
            offsets=(0, k * BLOCK_K),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )
        g_ptr = tl.make_block_ptr(
            base=G,
            shape=(N, K),
            strides=(stride_gn, stride_gk),
            offsets=(0, k * BLOCK_K),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )
        accu = tl.zeros((BLOCK_K, ), dtype=tl.float32)  # accu = X @ U
        accg = tl.zeros((BLOCK_K, ), dtype=tl.float32)  # accg = X @ G
        for n in tl.range(0, tl.cdiv(N, BLOCK_N)):
            x = tl.load(x_ptr, boundary_check=(0, ), padding_option="zero")
            u = tl.load(u_ptr, boundary_check=(0, 1), padding_option="zero")
            g = tl.load(g_ptr, boundary_check=(0, 1), padding_option="zero")

            accu += tl.sum(x.to(tl.float32)[:, None] * u.to(tl.float32), axis=0)
            accg += tl.sum(x.to(tl.float32)[:, None] * g.to(tl.float32), axis=0)

            x_ptr = tl.advance(x_ptr, (BLOCK_N, ))
            u_ptr = tl.advance(u_ptr, (BLOCK_N, 0))
            g_ptr = tl.advance(g_ptr, (BLOCK_N, 0))

        acca = tl.sigmoid(accg)  # acca = sigmoid(X@G)
        accg = acca * accg  # Silu(accg) = accg * sigmoid(accg)
        accu = accg * accu  # accu2 = accu * Silu(accg)
        accy += tl.sum(accu.to(tl.float32)[:, None] * d.to(tl.float32), axis=0)
        d_ptr = tl.advance(d_ptr, (BLOCK_K, 0))

    tl.store(y_ptr, accy.to(Y.type.element_ty), boundary_check=(0, ))
    # y_ptr = tl.advance(y_ptr, (0, ))


# @torch.compile(fullgraph=True)
def gatedmlp_t_mkn(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with `[t, m, k, n]` and grid `[t]`
    """
    # Check constraints.
    assert x.shape[2] == u.shape[0], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[1] == d.shape[0], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    Z, M, N = x.shape
    N, K = u.shape
    K, T = d.shape
    # Allocates output.
    y = torch.empty((Z, M, T), device=x.device, dtype=x.dtype)
    # launch kernel
    if act == "silu":
        grid = lambda META: (Z * triton.cdiv(T, META["BLOCK_T"]),)
        gatedmlp_t_mkn_kernel[grid](
            x, u, g, d, y,  # fmt: skip
            x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
            u.stride(0), u.stride(1),  # fmt: skip
            g.stride(0), g.stride(1),  # fmt: skip
            d.stride(0), d.stride(1),  # fmt: skip
            y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
            Z, M, N, K, T,  # fmt: skip
        )
    else:
        raise NotImplementedError("Activation fn other than Silu unsupported")
    return y

def gatedmlp_t_mkn_vec(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with `[t, m, k, n]` and grid `[t]`
    """
    # Check constraints.
    assert x.shape[2] == u.shape[0], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[1] == d.shape[0], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    Z, M, N = x.shape
    N, K = u.shape
    K, T = d.shape
    # Allocates output.
    y = torch.empty((Z, M, T), device=x.device, dtype=x.dtype)
    # launch kernel
    if act == "silu":
        grid = lambda META: (Z * triton.cdiv(T, META["BLOCK_T"]), M)
        gatedmlp_t_mkn_vec_kernel[grid](
            x, u, g, d, y,  # fmt: skip
            x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
            u.stride(0), u.stride(1),  # fmt: skip
            g.stride(0), g.stride(1),  # fmt: skip
            d.stride(0), d.stride(1),  # fmt: skip
            y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
            Z, M, N, K, T,  # fmt: skip
        )
    else:
        raise NotImplementedError("Activation fn other than Silu unsupported")
    return y



# TODO: debug testing, should be about Y=A2@D
@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K", "T"],
)
@triton.jit
def gatedmlp_mk_tn_kernel(
    X, U, G, D, Y,  # fmt: skip
    Lock, 
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate 
    Z, M, N, K, T,  # fmt: skip
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
    **kwargs,
):
    """
    Kernel for gated MLP with ACT as Silu and loop order `[m, k, t, n]`
    (i.e. tiling `A2` in row-major) with grid `[m, k]`.
    Accumulate Y = A2 @ D across programs using mutexes.
    Theoretically better than gatedmlp_m_ktn.
    No L2 cache optimization.
    """
    # Use mutex for accumulating Y; need to either store computed Y in cache, or load Y from HBM.
    # Need to call tl.store(Y) to store Y to L2 cache for sharing across programs;
    # Order tiles in [m, k] is better than [k, m] for cache reuse.
    # Need to use atomic ops for locks for mutex.
    
    pid = tl.program_id(axis=0)  # grid: (Z * (M // BLOCK_M) * (K // BLOCK_K))
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    off_z = pid // (num_pid_m * num_pid_k)
    pid_sample = pid % (num_pid_m * num_pid_k)
    pid_m = pid_sample // num_pid_k
    pid_k = pid_sample % num_pid_k
    x_offset = off_z.to(tl.int64) * stride_xz
    y_offset = off_z.to(tl.int64) * stride_yz
    # Atomic lock offset
    num_pid_t = tl.cdiv(T, BLOCK_T)
    lock_id = (off_z * num_pid_m + pid_m) * num_pid_t
    Lock += lock_id
    
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
    a2 = accg * accu  # accu2 = accu * Silu(accg)
    # if pid == 0:
        # print("TRITON:", a2.handle.data[0:5, 5:10])

    # Accumulate Y using mutexes
    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(M, T),
        strides=(stride_ym, stride_yt),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_T),
        order=(0, 1),
    )
    d_ptr = tl.make_block_ptr(
        base=D,
        shape=(K, T),
        strides=(stride_dk, stride_dt),
        offsets=(pid_k * BLOCK_K, 0),
        block_shape=(BLOCK_K, BLOCK_T),
        order=(1, 0),
    )
    for t in tl.range(0, tl.cdiv(T, BLOCK_T)):
        d = tl.load(d_ptr, boundary_check=(0, 1), padding_option="zero")
        accy = tl.dot(a2, d.to(tl.float32))
        # Mutex: store Y tile
        while tl.atomic_cas(Lock + t, 0, 1, sem="acquire") == 1:
            pass
        y = tl.load(y_ptr, boundary_check=(0, 1), padding_option="zero")
        accy = y + accy
        tl.store(y_ptr, accy.to(Y.type.element_ty), boundary_check=(0, 1))
        tl.atomic_xchg(Lock + t, 0, sem="release")
        # Mutex ends
        d_ptr = tl.advance(d_ptr, (0, BLOCK_T))
        y_ptr = tl.advance(y_ptr, (0, BLOCK_T))


# TODO: mk_tn_vec


# @torch.compile(fullgraph=True)
def gatedmlp_mk_tn(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with `[m, k, t, n]` and grid `[m, k]`, 
    accumulating Y = A2 @ D across programs.
    """
    # Check constraints.
    assert x.shape[2] == u.shape[0], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[1] == d.shape[0], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    Z, M, N = x.shape
    N, K = u.shape
    K, T = d.shape
    # Allocate locks
    locks = torch.zeros((Z, M, T), device=x.device, dtype=torch.int32)  # Should be (Z, M//BLOCK_M, T//BLOCK_T)
    # Allocate output.
    y = torch.empty((Z, M, T), device=x.device, dtype=x.dtype)
    # launch kernel
    grid = lambda META: (Z * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_K"]),)
    if act == "silu":
        gatedmlp_mk_tn_kernel[grid](
            x, u, g, d, y,  # fmt: skip
            locks,  # fmt: skip
            x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
            u.stride(0), u.stride(1),  # fmt: skip
            g.stride(0), g.stride(1),  # fmt: skip
            d.stride(0), d.stride(1),  # fmt: skip
            y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
            Z, M, N, K, T,  # fmt: skip
        )
    else:
        raise NotImplementedError("Activation fn other than Silu unsupported")
    return y



# TODO: gatedmlp_m_ktn, gatedmlp_k_mtn??


@triton.autotune(
    configs=get_autotune_config(separated_kernel=1),
    key=["M", "N", "K"],
)
@triton.jit
def gatedmlp_separated_a2_kernel(
    X, U, G, A2,  # fmt: skip
    stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_a2m, stride_a2k,  # fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate 
    M, N, K,  # fmt: skip
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
    # num_pid_k = tl.cdiv(K, BLOCK_K)
    # off_z = pid // (num_pid_m * num_pid_k)
    pid_sample = pid #% (num_pid_m * num_pid_k)
    pid_m = pid_sample % num_pid_m
    pid_k = pid_sample // num_pid_m
    # x_offset = off_z.to(tl.int64) * stride_xz
    # a2_offset = off_z.to(tl.int64) * stride_a2z

    a2_ptr = tl.make_block_ptr(
        base=A2, # + a2_offset,
        shape=(M, K),
        strides=(stride_a2m, stride_a2k),
        offsets=(pid_m * BLOCK_M, pid_k * BLOCK_K),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )
    x_ptr = tl.make_block_ptr(
        base=X, # + x_offset,
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


@triton.autotune(
    configs=get_autotune_config(separated_kernel=2),
    key=["M", "K", "T"],
)
@triton.jit
def gatedmlp_separated_y_kernel(
    A2, D, Y,  # fmt: skip
    stride_a2m, stride_a2k,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_ym, stride_yt,  # fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate 
    M, K, T,  # fmt: skip
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
    # num_pid_t = tl.cdiv(T, BLOCK_T)
    # off_z = pid // (num_pid_m * num_pid_t)
    pid_sample = pid #% (num_pid_m * num_pid_t)
    pid_m = pid_sample % num_pid_m
    pid_t = pid_sample // num_pid_m
    # a2_offset = off_z.to(tl.int64) * stride_a2z
    # y_offset = off_z.to(tl.int64) * stride_yz

    y_ptr = tl.make_block_ptr(
        base=Y, # + y_offset,
        shape=(M, T),
        strides=(stride_ym, stride_yt),
        offsets=(pid_m * BLOCK_M, pid_t * BLOCK_T),
        block_shape=(BLOCK_M, BLOCK_T),
        order=(0, 1),
    )
    a2_ptr = tl.make_block_ptr(
        base=A2, # + a2_offset,
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


@triton.autotune(
    configs=get_autotune_config(separated_kernel=1, vec_input=True),
    key=["N", "K"],
)
@triton.jit
def gatedmlp_separated_a2_vec_kernel(
    X, U, G, A2,  # fmt: skip
    stride_xz, stride_xm, stride_xn,  # fmt: skip
    stride_un, stride_uk,  # fmt: skip
    stride_gn, stride_gk,  # fmt: skip
    stride_a2z, stride_a2m, stride_a2k,  # fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate 
    Z, M, N, K,  # fmt: skip
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
):
    """
    Kernel for first half gated MLP with ACT as Silu (computing A2).
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (K // BLOCK_K), M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    off_z = pid // num_pid_k
    pid_k = pid % num_pid_k
    off_m = tl.program_id(axis=1)
    x_offset = off_z.to(tl.int64) * stride_xz + off_m.to(tl.int64) * stride_xm
    a2_offset = off_z.to(tl.int64) * stride_a2z + off_m.to(tl.int64) * stride_a2m

    a2_ptr = tl.make_block_ptr(
        base=A2 + a2_offset,
        shape=(K, ),
        strides=(stride_a2k, ),
        offsets=(pid_k * BLOCK_K, ),
        block_shape=(BLOCK_K, ),
        order=(0, ),
    )
    x_ptr = tl.make_block_ptr(
        base=X + x_offset,
        shape=(N, ),
        strides=(stride_xn, ),
        offsets=(0, ),
        block_shape=(BLOCK_N, ),
        order=(0, ),
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
    accu = tl.zeros((BLOCK_K, ), dtype=tl.float32)  # accu = X @ U
    accg = tl.zeros((BLOCK_K, ), dtype=tl.float32)  # accg = X @ G
    for n in tl.range(0, tl.cdiv(N, BLOCK_N)):
        x = tl.load(x_ptr, boundary_check=(0, ), padding_option="zero")
        u = tl.load(u_ptr, boundary_check=(0, 1), padding_option="zero")
        g = tl.load(g_ptr, boundary_check=(0, 1), padding_option="zero")

        accu += tl.sum(x.to(tl.float32)[:, None] * u.to(tl.float32), axis=0)
        accg += tl.sum(x.to(tl.float32)[:, None] * g.to(tl.float32), axis=0)

        x_ptr = tl.advance(x_ptr, (BLOCK_N, ))
        u_ptr = tl.advance(u_ptr, (BLOCK_N, 0))
        g_ptr = tl.advance(g_ptr, (BLOCK_N, 0))

    acca = tl.sigmoid(accg)  # acca = sigmoid(X@G)
    accg = acca * accg  # Silu(accg) = accg * sigmoid(accg)
    accu = accg * accu  # accu2 = accu * Silu(accg)
    tl.store(a2_ptr, accu.to(A2.type.element_ty), boundary_check=(0, ))


@triton.autotune(
    configs=get_autotune_config(separated_kernel=2, vec_input=True),
    key=["K", "T"],
)
@triton.jit
def gatedmlp_separated_y_vec_kernel(
    A2, D, Y,  # fmt: skip
    stride_a2z, stride_a2m, stride_a2k,  # fmt: skip
    stride_dk, stride_dt,  # fmt: skip
    stride_yz, stride_ym, stride_yt,  # fmt: skip
    # M = seq_len, N (= T) = d_h, K = d_intermediate 
    Z, M, K, T,  # fmt: skip
    BLOCK_K: tl.constexpr,  #
    BLOCK_T: tl.constexpr,  #
):
    """
    Kernel for last half gated MLP (computing Y).
    No L2 cache optimization.
    """
    pid = tl.program_id(axis=0)  # grid: (Z * (T // BLOCK_T), M)
    num_pid_t = tl.cdiv(T, BLOCK_T)
    off_z = pid // num_pid_t
    pid_t = pid % num_pid_t
    off_m = tl.program_id(axis=1)
    a2_offset = off_z.to(tl.int64) * stride_a2z + off_m.to(tl.int64) * stride_a2m
    y_offset = off_z.to(tl.int64) * stride_yz + off_m.to(tl.int64) * stride_ym

    y_ptr = tl.make_block_ptr(
        base=Y + y_offset,
        shape=(T, ),
        strides=(stride_yt, ),
        offsets=(pid_t * BLOCK_T, ),
        block_shape=(BLOCK_T, ),
        order=(0, ),
    )
    a2_ptr = tl.make_block_ptr(
        base=A2 + a2_offset,
        shape=(K, ),
        strides=(stride_a2k, ),
        offsets=(0, ),
        block_shape=(BLOCK_K, ),
        order=(0, ),
    )
    d_ptr = tl.make_block_ptr(
        base=D,
        shape=(K, T),
        strides=(stride_dk, stride_dt),
        offsets=(0, pid_t * BLOCK_T),
        block_shape=(BLOCK_K, BLOCK_T),
        order=(1, 0),
    )
    accy = tl.zeros((BLOCK_T, ), dtype=tl.float32)  # accy = accu2 @ D
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a2 = tl.load(a2_ptr, boundary_check=(0, ), padding_option="zero")
        d = tl.load(d_ptr, boundary_check=(0, 1), padding_option="zero")

        accy += tl.sum(a2.to(tl.float32)[:, None] * d.to(tl.float32), axis=0)

        a2_ptr = tl.advance(a2_ptr, (BLOCK_K, ))
        d_ptr = tl.advance(d_ptr, (BLOCK_K, 0))

    tl.store(y_ptr, accy.to(Y.type.element_ty), boundary_check=(0, ))


# @torch.compile(fullgraph=True)
def gatedmlp_separated(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with separated kernels for computing A2 and Y.
    """
    # Check constraints.
    assert x.shape[-1] == u.shape[1], "Incompatible dimensions"
    assert u.shape == g.shape, "Incompatible dimensions"
    assert u.shape[0] == d.shape[1], "Incompatible dimensions"
    assert x.dtype == u.dtype == g.dtype == d.dtype, "Incompatible dtypes"
    assert act in ACTIVATIONS, "Unsupported activation function"
    # assert x.is_contiguous(), "Matrix X must be contiguous"
    M, N = x.shape
    K, N = u.shape
    T, K = d.shape
    # Allocates output.
    a2 = torch.empty((M, K), device=x.device, dtype=x.dtype)
    y = torch.empty((M, T), device=x.device, dtype=x.dtype)
    # launch kernels
    # launch kernel for a2
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_K"]),)
    gatedmlp_separated_a2_kernel[grid](
        x, u, g, a2,  # fmt: skip
        x.stride(0), x.stride(1),  # fmt: skip
        u.stride(1), u.stride(0),  # fmt: skip
        g.stride(1), g.stride(0),  # fmt: skip
        a2.stride(0), a2.stride(1),  # fmt: skip
        M, N, K,  # fmt: skip
    )
    # launch kernel for y
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(T, META["BLOCK_T"]),)
    gatedmlp_separated_y_kernel[grid](
        a2, d, y,  # fmt: skip
        a2.stride(0), a2.stride(1),  # fmt: skip
        d.stride(1), d.stride(0),  # fmt: skip
        y.stride(0), y.stride(1),  # fmt: skip
        M, K, T,  # fmt: skip
    )
    return y


def gatedmlp_separated_vec(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    """
    Gated MLP with separated kernels for computing A2 and Y.
    """
    # Check constraints.
    assert x.shape[2] == u.shape[0], "Incompatible dimensions"
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
    # launch kernels
    if act == "silu":
        # launch kernel for a2
        grid = lambda META: (Z * triton.cdiv(K, META["BLOCK_K"]), M)
        gatedmlp_separated_a2_vec_kernel[grid](
            x, u, g, a2,  # fmt: skip
            x.stride(0), x.stride(1), x.stride(2),  # fmt: skip
            u.stride(0), u.stride(1),  # fmt: skip
            g.stride(0), g.stride(1),  # fmt: skip
            a2.stride(0), a2.stride(1), a2.stride(2),  # fmt: skip
            Z, M, N, K,  # fmt: skip
        )
        # launch kernel for y
        grid = lambda META: (Z * triton.cdiv(T, META["BLOCK_T"]), M)
        gatedmlp_separated_y_vec_kernel[grid](
            a2, d, y,  # fmt: skip
            a2.stride(0), a2.stride(1), a2.stride(2),  # fmt: skip
            d.stride(0), d.stride(1),  # fmt: skip
            y.stride(0), y.stride(1), y.stride(2),  # fmt: skip
            Z, M, K, T,  # fmt: skip
        )
    else:
        raise NotImplementedError("Activation fn other than Silu unsupported")
    return y



# THE FOLLOWING MIGHT BE WRONG...
# gatedmlp_m_ktn, gatedmlp_k_mtn, and gatedmlp_m_tkn_keepA2 can't be 
# implemented without tl.load and tl.store because indexed assignment to tensor 
# is not supported by Triton.



# @torch.compile(fullgraph=True)
def gatedmlp_torch(
    x, u, g, d,  # fmt: skip
    act="silu",  # fmt: skip
):
    dtype = x.dtype
    x, u, g, d = x.to(torch.float32), u.to(torch.float32), g.to(torch.float32), d.to(torch.float32)
    a1 = torch.matmul(x, u)
    ag = torch.matmul(x, g)
    act_fn = ACT2FN[act]
    aa = act_fn(ag)
    a2 = torch.mul(a1, aa) 
    # print("TORCH:", a2[0, 0:5, 5:10])
    y = torch.matmul(a2, d)
    return y.to(dtype)
