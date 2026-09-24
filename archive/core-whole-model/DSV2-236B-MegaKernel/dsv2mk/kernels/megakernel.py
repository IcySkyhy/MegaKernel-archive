"""The DeepSeek-V2-236B decode megakernel: one persistent CuTe-DSL launch.

A single `@cute.kernel` runs all 60 decoder layers for one decode step. Weights
are stacked along a leading layer dimension and indexed with Int64 pointer
arithmetic; layers are separated by in-kernel grid syncs (atomic counter,
release/acquire at gpu scope) rather than by returning to the host.

Per layer the kernel walks these stages, all inside the one launch:

    A+B   residual add, input RMSNorm, fused q/kv_a down-projection
    C-F   q_a RMSNorm, q_b up-projection, kv_a norm, decoupled k_pe RoPE
    G+H   q_pe RoPE and the W_UK absorption BMM
    I     MLA decode over the compressed KV cache, split across CTAs
    I-c   cross-CTA partial reduction (online-softmax rescale)
    J/K   per-head output gather and the o_proj GEMM
    TP    multimem.red scatter of the attn_proj partials + TP barrier
    L     TP all-reduce consume, folded into the residual
    M+N   post-attention RMSNorm and the MoE router GEMV
    O     group-limited-greedy top-K selection (ballot + popc + ctz)
    Q1/Q2 shared-expert gate/up (FP8 block-scaled MMA), SiLU, down-projection
    EP    expert-parallel barrier
    P1/P2 routed experts, gate/up then down, FP8 block-scaled
    R     distributed multimem.ld_reduce all-reduce + intra-rank broadcast
    S     inline embedding lookup, final norm and lm_head argmax

Every op is written in CuTe-DSL. No external compute library is called from
inside the kernel body.

Target: sm_100 (B200), TP=EP=8, batch size 1, decode only.
"""

from __future__ import annotations

import math
import os
from typing import Type

import cuda.bindings.driver as cuda
import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass._mlir.dialects import llvm, arith as _arith_d, vector as _vector_d
from cutlass._mlir import ir as _mlir_ir
from cutlass.cutlass_dsl import T as _T, dsl_user_op
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from dsv2mk.kernels.softmax_primitives import (
    fmax,
    fmax_reduce,
    fadd_reduce,
)


_torch_to_cutlass_dtype = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


if hasattr(cute.arch, "clock"):
    _clock = cute.arch.clock
else:
    @dsl_user_op
    def _clock(*, loc=None, ip=None) -> Int32:
        return Int32(
            llvm.inline_asm(
                Int32.mlir_type,
                [],
                "mov.u32 $0, %clock;\n",
                "=r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )


def _ct(dtype: torch.dtype) -> Type[cutlass.Numeric]:
    return _torch_to_cutlass_dtype[dtype]


NUM_SMS = 132


class DSv2AttnPrologueMultilayerKernel:

    COPY_BITS = 128

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        H: int,
        OUT_QKVA: int,
        Lq: int,
        OUT_QB: int,
        L: int,
        S_max: int,
        N: int = 128,
        qk_nope_dim: int = 128,
        v_head_dim: int = 128,
        Lkv: int = 512,
        R: int = 64,
        V: int = 102400,
        E: int = 160,
        K_topk: int = 6,
        n_group: int = 8,
        topk_group: int = 3,
        I_routed: int = 1536,
        I_shared: int = 3072,
        routed_scaling: float = 16.0,
        moe_norm_topk_prob: bool = False,
        num_threads: int = 256,
        threads_per_output: int = 32,
        num_sms: int = NUM_SMS,
        tp_size: int = 1,
        ep_size: int = 1,
        ep_rank: int = 0,
        attention_scaling: float = 1.0,
        use_splitk: bool = False,
    ):
        self.use_q_lora = (Lq > 0)
        self.qb_K = Lq if self.use_q_lora else H
        assert OUT_QKVA == Lq + Lkv + R, (
            f"OUT_QKVA must equal Lq+Lkv+R (got {OUT_QKVA} != {Lq}+{Lkv}+{R})"
        )
        assert OUT_QB == N * (qk_nope_dim + R), (
            f"OUT_QB must equal N*(qk_nope_dim+R) (got {OUT_QB} != {N}*({qk_nope_dim}+{R}))"
        )
        self.dtype = dtype
        self.H = H
        self.OUT_QKVA = OUT_QKVA
        self.Lq = Lq
        self.OUT_QB = OUT_QB
        self.Lkv = Lkv
        self.R = R
        self.L = L
        self.S_max = S_max
        self.N = N
        self.qk_nope_dim = qk_nope_dim
        self.v_head_dim = v_head_dim
        self.HEAD_TOTAL = qk_nope_dim + R
        self.O_CONCAT = N * v_head_dim
        self.E = E
        self.K_topk = K_topk
        self.n_group = n_group
        self.topk_group = topk_group
        assert E % n_group == 0, f"E ({E}) must be divisible by n_group ({n_group})"
        self.E_per_group = E // n_group
        self.I_routed = I_routed
        self.I_shared = I_shared
        self.USE_SPLITK = use_splitk
        self.ROUTED_SPLIT = False
        self.Q1_INTER_OFFSET = K_topk * I_routed
        self.I_max_moe = K_topk * I_routed + I_shared
        self.routed_scaling = routed_scaling
        self.moe_norm_topk_prob = moe_norm_topk_prob
        self.num_threads = num_threads
        self.threads_per_output = threads_per_output
        self.num_sms = num_sms
        self.TILE_SEQ_DEC = 8
        self.NUM_STAGES_DEC = 2
        self.USE_2CTA = True
        self.CLUSTER_SHAPE_MNK = (2, 1, 1) if self.USE_2CTA else (1, 1, 1)
        self.TMEM_COLS = 512
        self.NUM_KV_SPLITS = max(1, self.num_sms // self.N)

        self.CLUSTER_SIZE = 2
        self.CLUSTER_TILE_M = 128
        self.CTA_TILE_M = self.CLUSTER_TILE_M // self.CLUSTER_SIZE
        self.NUM_HDIMV_SPLITS = 2
        self.PV_N = self.Lkv // self.NUM_HDIMV_SPLITS
        self.TILE_SEQ_DEC_NEW = 64
        self.MMA_INST_QK = (self.CLUSTER_TILE_M, self.TILE_SEQ_DEC_NEW, 16)
        self.MMA_TILER_QK = (self.CLUSTER_TILE_M, self.TILE_SEQ_DEC_NEW, 64)
        self.MMA_INST_PV = (self.CLUSTER_TILE_M, self.PV_N, 16)
        self.MMA_TILER_PV = (self.CLUSTER_TILE_M, self.PV_N, self.TILE_SEQ_DEC_NEW)
        self.NUM_QK_CHUNKS = (self.Lkv + self.R) // self.MMA_TILER_QK[2]
        assert (self.Lkv + self.R) % self.MMA_TILER_QK[2] == 0
        self.NUM_PV_N_SPLITS = self.Lkv // self.MMA_TILER_PV[1]
        assert self.Lkv % self.MMA_TILER_PV[1] == 0
        self.AB_STAGES_QK = 2
        self.V_STAGES = 1
        self.ACC_S_STAGES = 1
        self.ACC_O_STAGES = 1
        self.P_STAGES = 1
        self.NUM_EPI_WARPS = 4
        self.THREADS_IN_EPILOGUE = self.NUM_EPI_WARPS * 32
        self.MMA_WARP_ID = 4
        self.TMA_WARP_ID = 5
        self.STAGE_I_EPI_BAR_ID = 3
        self.STAGE_I_SOFTMAX_BAR_ID = 4
        self.STAGE_I_TMEM_ALLOC_BAR_ID = 5
        self.HEADS_PER_CTA = self.N // self.CLUSTER_SIZE
        assert self.N % self.CLUSTER_SIZE == 0, (
            f"N ({self.N}) must be divisible by CLUSTER_SIZE "
            f"({self.CLUSTER_SIZE}) for even cluster split."
        )
        self.M_PAD_PER_HEAD = self.CTA_TILE_M // self.HEADS_PER_CTA
        self.NUM_KV_SPLITS_MMA = min(
            self.num_sms // self.CLUSTER_SIZE,
            self.S_max // self.TILE_SEQ_DEC_NEW,
        )
        self.NUM_KV_SPLITS = self.NUM_KV_SPLITS_MMA
        assert self.num_sms % self.CLUSTER_SIZE == 0, (
            f"num_sms ({self.num_sms}) must be divisible by CLUSTER_SIZE "
            f"({self.CLUSTER_SIZE}) for 2-CTA cluster layout."
        )
        assert self.N <= self.CLUSTER_TILE_M, (
            f"N ({self.N}) > CLUSTER_TILE_M ({self.CLUSTER_TILE_M}) — "
            f"head-grouping across clusters is not yet supported."
        )
        assert self.CLUSTER_TILE_M % self.N == 0, (
            f"CLUSTER_TILE_M ({self.CLUSTER_TILE_M}) must be a multiple of N "
            f"({self.N}) so the intra-cluster M-padding is uniform."
        )
        assert self.Lkv % self.NUM_HDIMV_SPLITS == 0, (
            f"Lkv ({self.Lkv}) must be divisible by NUM_HDIMV_SPLITS "
            f"({self.NUM_HDIMV_SPLITS}) for the PV MMA's even N-split."
        )
        kdim = self.Lkv + self.R
        assert kdim % self.MMA_TILER_QK[2] == 0, (
            f"KDIM ({kdim}) must be divisible by MMA_TILER_QK[2] "
            f"({self.MMA_TILER_QK[2]}) for the QK K-loop tiling."
        )
        self.vec_size = self.COPY_BITS // dtype.width
        assert H % self.vec_size == 0
        assert Lq == 0 or Lq % self.vec_size == 0
        assert self.qb_K % self.vec_size == 0
        assert Lkv % self.vec_size == 0
        assert R % 2 == 0, "R must be even for the half-split RoPE convention"
        assert num_threads % threads_per_output == 0
        assert threads_per_output in (8, 16, 32)
        self.outs_per_cta = num_threads // threads_per_output
        self.warps_per_row = max(num_threads // 32, 1)
        self.tiles_qkva = (OUT_QKVA + self.outs_per_cta - 1) // self.outs_per_cta
        self.tiles_qb = (OUT_QB + self.outs_per_cta - 1) // self.outs_per_cta
        self.iters_qkva = (self.tiles_qkva + num_sms - 1) // num_sms
        self.iters_qb = (self.tiles_qb + num_sms - 1) // num_sms
        self.num_vec_blocks_h = max(1, (H + num_threads * self.vec_size - 1) // (num_threads * self.vec_size))
        self.num_vec_blocks_lq = max(1, (Lq + num_threads * self.vec_size - 1) // (num_threads * self.vec_size))
        self.num_vec_blocks_lkv = max(1, (Lkv + num_threads * self.vec_size - 1) // (num_threads * self.vec_size))

        self.Q1_TILE_K = 256
        self.Q1_NUM_STAGES = 2
        assert H % self.Q1_TILE_K == 0, (
            f"H ({H}) must be divisible by Q1_TILE_K ({self.Q1_TILE_K}) for "
            f"staged Q1 loader."
        )
        self.Q1_NUM_K_TILES = H // self.Q1_TILE_K
        self.q1_tma_bytes_one = self.outs_per_cta * self.Q1_TILE_K * (dtype.width // 8)
        self.Q2_TILE_K = 64
        self.Q2_NUM_STAGES = 2
        assert I_shared % self.Q2_TILE_K == 0, (
            f"I_shared ({I_shared}) must be divisible by Q2_TILE_K "
            f"({self.Q2_TILE_K}) for staged Q2 loader."
        )
        assert (I_shared // self.Q2_TILE_K) % (2 * self.Q2_NUM_STAGES) == 0, (
            f"I_shared/Q2_TILE_K ({I_shared}/{self.Q2_TILE_K}="
            f"{I_shared // self.Q2_TILE_K}) must be divisible by "
            f"2*Q2_NUM_STAGES ({2 * self.Q2_NUM_STAGES}) so the constexpr "
            f"phase formula `(kt // NUM_STAGES) % 2` is correct per-(it) "
            f"without mid-kernel mbar re-init (banned per memory)."
        )
        assert self.Q2_TILE_K % threads_per_output == 0, (
            f"Q2_TILE_K ({self.Q2_TILE_K}) must be divisible by "
            f"threads_per_output ({threads_per_output}) for clean Q2-local "
            f"vec_size = TILE_K / threads_per_output."
        )
        self.Q2_NUM_K_TILES = I_shared // self.Q2_TILE_K
        self.Q2_VEC_SIZE = self.Q2_TILE_K // threads_per_output
        self.q2_tma_bytes_one = self.outs_per_cta * self.Q2_TILE_K * (dtype.width // 8)

        self.Q1_CLUSTER_TILE_M = 128
        self.Q1_CTA_TILE_M = self.Q1_CLUSTER_TILE_M // self.CLUSTER_SIZE
        self.Q1_MMA_N = 64
        self.Q1_K_CHUNK = 32
        assert H % self.Q1_K_CHUNK == 0, (
            f"H ({H}) must be divisible by Q1_K_CHUNK ({self.Q1_K_CHUNK})"
        )
        self.Q1_NUM_K_TILES_MMA = H // self.Q1_K_CHUNK
        self.Q1_AB_STAGES = 2
        self.Q1_MMA_INST = (self.Q1_CLUSTER_TILE_M, self.Q1_MMA_N, 16)
        self.Q1_MMA_TILER = (self.Q1_CLUSTER_TILE_M, self.Q1_MMA_N, self.Q1_K_CHUNK)
        self.Q1_NUM_ACTIVE_CLUSTERS = (I_shared + self.Q1_CLUSTER_TILE_M - 1) // self.Q1_CLUSTER_TILE_M

        self.Q1_SF_VEC_SIZE = 32
        self.Q1_FP8_CLUSTER_TILE_M = 256
        self.Q1_FP8_CTA_TILE_M = self.Q1_FP8_CLUSTER_TILE_M // self.CLUSTER_SIZE
        self.Q1_FP8_MMA_N = 32
        self.Q1_FP8_K_CHUNK = 128
        assert H % self.Q1_FP8_K_CHUNK == 0, (
            f"H ({H}) must be divisible by Q1_FP8_K_CHUNK ({self.Q1_FP8_K_CHUNK})"
        )
        self.Q1_FP8_NUM_K_TILES_MMA = H // self.Q1_FP8_K_CHUNK
        self.Q1_FP8_AB_STAGES = 2
        self.Q1_FP8_MMA_INST = (self.Q1_FP8_CLUSTER_TILE_M, self.Q1_FP8_MMA_N, 32)
        self.Q1_FP8_MMA_TILER = (self.Q1_FP8_CLUSTER_TILE_M, self.Q1_FP8_MMA_N, self.Q1_FP8_K_CHUNK)
        self.Q1_FP8_NUM_ACTIVE_CLUSTERS = (
            (I_shared + self.Q1_FP8_CLUSTER_TILE_M - 1) // self.Q1_FP8_CLUSTER_TILE_M
        )
        self.TP_SIZE = tp_size
        self.EP_SIZE = ep_size
        self.EP_RANK = ep_rank
        assert E % ep_size == 0, f"n_routed_experts={E} must be divisible by ep_size={ep_size}"
        self.E_PER_RANK = E // ep_size
        self.E_LOCAL_START = ep_rank * self.E_PER_RANK
        self.E_LOCAL_END = (ep_rank + 1) * self.E_PER_RANK
        self.V = V
        self.attention_scaling = float(attention_scaling)


        self.Q2_CLUSTER_TILE_M = 128
        self.Q2_CTA_TILE_M = self.Q2_CLUSTER_TILE_M // self.CLUSTER_SIZE
        self.Q2_MMA_N = 64
        self.Q2_K_CHUNK = 32
        assert I_shared % self.Q2_K_CHUNK == 0, (
            f"I_shared ({I_shared}) must be divisible by Q2_K_CHUNK ({self.Q2_K_CHUNK})"
        )
        self.Q2_NUM_K_TILES_MMA = I_shared // self.Q2_K_CHUNK
        self.Q2_AB_STAGES = 2
        self.Q2_MMA_INST = (self.Q2_CLUSTER_TILE_M, self.Q2_MMA_N, 16)
        self.Q2_MMA_TILER = (self.Q2_CLUSTER_TILE_M, self.Q2_MMA_N, self.Q2_K_CHUNK)
        self.Q2_NUM_ACTIVE_CLUSTERS = (H + self.Q2_CLUSTER_TILE_M - 1) // self.Q2_CLUSTER_TILE_M

        self.Q2_FP8_CLUSTER_TILE_M = 256
        self.Q2_FP8_CTA_TILE_M = self.Q2_FP8_CLUSTER_TILE_M // self.CLUSTER_SIZE
        self.Q2_FP8_MMA_N = 16
        self.Q2_FP8_K_CHUNK = 128
        assert I_shared % self.Q2_FP8_K_CHUNK == 0, (
            f"I_shared ({I_shared}) must be divisible by Q2_FP8_K_CHUNK "
            f"({self.Q2_FP8_K_CHUNK})"
        )
        self.Q2_FP8_NUM_K_TILES_MMA = I_shared // self.Q2_FP8_K_CHUNK
        self.Q2_NUM_K_SPLITS = 2
        assert self.Q2_FP8_NUM_K_TILES_MMA % self.Q2_NUM_K_SPLITS == 0
        self.Q2_FP8_NUM_K_TILES_SPLIT = self.Q2_FP8_NUM_K_TILES_MMA // self.Q2_NUM_K_SPLITS
        self.Q2_FP8_AB_STAGES = 4
        self.Q2_FP8_MMA_INST = (self.Q2_FP8_CLUSTER_TILE_M, self.Q2_FP8_MMA_N, 32)
        self.Q2_FP8_MMA_TILER = (
            self.Q2_FP8_CLUSTER_TILE_M, self.Q2_FP8_MMA_N, self.Q2_FP8_K_CHUNK,
        )
        self.Q2_FP8_NUM_ACTIVE_CLUSTERS = (
            (H + self.Q2_FP8_CLUSTER_TILE_M - 1) // self.Q2_FP8_CLUSTER_TILE_M
        )
        self.Q2_SF_VEC_SIZE = 32
        self.Q2_USE_MMA = True

        self.MOE_P1_CLUSTER_TILE_M = 256
        self.MOE_P1_CTA_TILE_M = self.MOE_P1_CLUSTER_TILE_M // self.CLUSTER_SIZE
        self.MOE_P1_MMA_N = 32
        self.MOE_P1_K_CHUNK = 128
        assert H % self.MOE_P1_K_CHUNK == 0, (
            f"H ({H}) must be divisible by MOE_P1_K_CHUNK ({self.MOE_P1_K_CHUNK})"
        )
        assert I_routed % self.MOE_P1_CLUSTER_TILE_M == 0, (
            f"I_routed ({I_routed}) must be divisible by "
            f"MOE_P1_CLUSTER_TILE_M ({self.MOE_P1_CLUSTER_TILE_M})"
        )
        self.MOE_P1_NUM_K_TILES_MMA = H // self.MOE_P1_K_CHUNK
        self.MOE_P1_NUM_M_CLUSTERS = I_routed // self.MOE_P1_CLUSTER_TILE_M
        self.MOE_P1_AB_STAGES = 2
        self.MOE_P1_MMA_INST  = (self.MOE_P1_CLUSTER_TILE_M, self.MOE_P1_MMA_N, 32)
        self.MOE_P1_MMA_TILER = (
            self.MOE_P1_CLUSTER_TILE_M, self.MOE_P1_MMA_N, self.MOE_P1_K_CHUNK,
        )

        self.MOE_P2_CLUSTER_TILE_M = 256
        self.MOE_P2_CTA_TILE_M = self.MOE_P2_CLUSTER_TILE_M // self.CLUSTER_SIZE
        self.MOE_P2_MMA_N = 32
        self.MOE_P2_K_CHUNK = 128
        assert I_routed % self.MOE_P2_K_CHUNK == 0, (
            f"I_routed ({I_routed}) must be divisible by MOE_P2_K_CHUNK ({self.MOE_P2_K_CHUNK})"
        )
        assert H % self.MOE_P2_CLUSTER_TILE_M == 0, (
            f"H ({H}) must be divisible by MOE_P2_CLUSTER_TILE_M ({self.MOE_P2_CLUSTER_TILE_M})"
        )
        self.MOE_P2_NUM_K_TILES_MMA = I_routed // self.MOE_P2_K_CHUNK
        self.MOE_P2_NUM_M_CLUSTERS = H // self.MOE_P2_CLUSTER_TILE_M
        self.MOE_P2_AB_STAGES = 4
        self.MOE_P2_MMA_INST  = (self.MOE_P2_CLUSTER_TILE_M, self.MOE_P2_MMA_N, 32)
        self.MOE_P2_MMA_TILER = (
            self.MOE_P2_CLUSTER_TILE_M, self.MOE_P2_MMA_N, self.MOE_P2_K_CHUNK,
        )

        self.MOE_SF_VEC_SIZE = 32

        self.STAGE_K_CLUSTER_SIZE = self.CLUSTER_SIZE
        self.STAGE_K_CLUSTER_TILE_M = 128
        self.STAGE_K_CTA_TILE_M = (
            self.STAGE_K_CLUSTER_TILE_M // self.STAGE_K_CLUSTER_SIZE
        )
        self.STAGE_K_MMA_N = 32
        self.STAGE_K_K_CHUNK = 32
        assert self.O_CONCAT % self.STAGE_K_K_CHUNK == 0, (
            f"O_CONCAT ({self.O_CONCAT}) must be divisible by STAGE_K_K_CHUNK "
            f"({self.STAGE_K_K_CHUNK})"
        )
        self.STAGE_K_NUM_K_TILES_MMA = self.O_CONCAT // self.STAGE_K_K_CHUNK
        self.STAGE_K_AB_STAGES = 4
        self.STAGE_K_MMA_INST = (
            self.STAGE_K_CLUSTER_TILE_M, self.STAGE_K_MMA_N, 16,
        )
        self.STAGE_K_MMA_TILER = (
            self.STAGE_K_CLUSTER_TILE_M, self.STAGE_K_MMA_N, self.STAGE_K_K_CHUNK,
        )
        self.STAGE_K_NUM_ACTIVE_CLUSTERS = (
            (self.H + self.STAGE_K_CLUSTER_TILE_M - 1)
            // self.STAGE_K_CLUSTER_TILE_M
        )

        self.WS_COMM_WARP = 7

    def smem_size_in_bytes(self) -> int:
        bf16 = self.dtype.width // 8
        s_h = self.H * bf16
        s_norm = self.H * bf16
        s_qc = 0
        s_kv = 0
        reduce_b = max(
            self.warps_per_row,
            self.n_group,
            self.TILE_SEQ_DEC * self.warps_per_row,
        ) * 4
        TS_MMA = self.TILE_SEQ_DEC_NEW
        K_CHUNK = self.MMA_TILER_QK[2]
        s_sQ = self.CTA_TILE_M * K_CHUNK * bf16 * self.NUM_QK_CHUNKS
        s_sK = TS_MMA * K_CHUNK * bf16 * self.AB_STAGES_QK
        s_sP = self.CTA_TILE_M * TS_MMA * bf16
        s_sV = TS_MMA * self.PV_N * bf16 * self.NUM_PV_N_SPLITS
        s_mbar_pipes = (
            self.AB_STAGES_QK + self.V_STAGES + self.V_STAGES
            + self.ACC_S_STAGES + self.ACC_O_STAGES + self.ACC_O_STAGES
            + self.P_STAGES
        ) * 2 * 8
        s_row_max = self.CTA_TILE_M * self.CLUSTER_SIZE * 4
        Q1_K_CHUNK_C = const_expr(self.Q1_K_CHUNK)
        s_q1_wg_mma = 0
        s_q1_wu_mma = 0
        s_q1_x_mma  = 0
        sf_k_blocks = self.Q1_FP8_K_CHUNK // self.Q1_SF_VEC_SIZE
        s_q1_sfa_g  = self.Q1_FP8_CTA_TILE_M * sf_k_blocks * 1 * self.Q1_FP8_AB_STAGES
        s_q1_sfa_u  = self.Q1_FP8_CTA_TILE_M * sf_k_blocks * 1 * self.Q1_FP8_AB_STAGES
        s_q1_sfb    = self.Q1_FP8_MMA_N    * sf_k_blocks * 1 * self.Q1_FP8_AB_STAGES
        s_q1_quant_scratch = 0
        s_q1_sf_scratch    = self.H // self.Q1_SF_VEC_SIZE
        s_q1_extra  = (
            s_q1_sfa_g + s_q1_sfa_u + s_q1_sfb
            + s_q1_quant_scratch + s_q1_sf_scratch
        )
        s_q1_mbars = (2 * max(self.Q1_AB_STAGES, self.Q1_FP8_AB_STAGES, self.Q2_FP8_AB_STAGES) + 2 * 1) * 8
        s_q2_wd = self.outs_per_cta * self.Q2_TILE_K * bf16 * self.Q2_NUM_STAGES
        s_q2_bars = self.Q2_NUM_STAGES * 8
        s_q2_free_bars = self.Q2_NUM_STAGES * 8
        s_stage_k_mbars = (2 * self.STAGE_K_AB_STAGES + 2 * 1) * 8
        moe_p1_sf_k = self.MOE_P1_K_CHUNK // self.MOE_SF_VEC_SIZE
        s_moe_p1_sfa_g = (
            self.MOE_P1_CTA_TILE_M * moe_p1_sf_k * 1 * self.MOE_P1_AB_STAGES
        )
        s_moe_p1_sfa_u = (
            self.MOE_P1_CTA_TILE_M * moe_p1_sf_k * 1 * self.MOE_P1_AB_STAGES
        )
        s_moe_p1_sfb = (
            self.MOE_P1_MMA_N * moe_p1_sf_k * 1 * self.MOE_P1_AB_STAGES
        )
        s_moe_p1_mbars = (
            (2 * self.MOE_P1_AB_STAGES + 2 * 1) * 8
        )
        s_moe_p1 = (
            s_moe_p1_sfa_g + s_moe_p1_sfa_u + s_moe_p1_sfb + s_moe_p1_mbars
        )
        moe_p2_sf_k = self.MOE_P2_K_CHUNK // self.MOE_SF_VEC_SIZE
        s_moe_p2_sfa_d = (
            self.MOE_P2_CTA_TILE_M * moe_p2_sf_k * 1 * self.MOE_P2_AB_STAGES
        )
        s_moe_p2_sfb = (
            self.MOE_P2_MMA_N * moe_p2_sf_k * 1 * self.MOE_P2_AB_STAGES
        )
        s_moe_p2_mbars = (
            (2 * self.MOE_P2_AB_STAGES + 2 * 1) * 8
        )
        s_moe_p2_sf_scratch = self.I_routed // self.MOE_SF_VEC_SIZE
        s_moe_p2 = (
            s_moe_p2_sfa_d + s_moe_p2_sfb + s_moe_p2_mbars + s_moe_p2_sf_scratch
        )
        return (
            s_h + s_norm + s_qc + s_kv + reduce_b
            + s_sQ + s_sK + s_sP + s_sV
            + s_mbar_pipes + s_row_max
            + s_q1_wg_mma + s_q1_wu_mma + s_q1_x_mma + s_q1_extra + s_q1_mbars
            + s_q2_wd + s_q2_bars + s_q2_free_bars
            + s_stage_k_mbars
            + s_moe_p1
            + s_moe_p2
            + 2048
        )

    @cute.jit
    def __call__(
        self,
        h_in_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        gamma1_ptr: cute.Pointer,
        w_qkva_ptr: cute.Pointer,
        gamma2_ptr: cute.Pointer,
        w_qb_ptr: cute.Pointer,
        gamma3_ptr: cute.Pointer,
        cos_ptr: cute.Pointer,
        sin_ptr: cute.Pointer,
        qkva_out_ptr: cute.Pointer,
        q_out_ptr: cute.Pointer,
        kvc_out_ptr: cute.Pointer,
        kpe_out_ptr: cute.Pointer,
        kv_cache_c_ptr: cute.Pointer,
        kv_cache_pe_ptr: cute.Pointer,
        w_uk_ptr: cute.Pointer,
        q_nope_abs_out_ptr: cute.Pointer,
        attn_out_ptr: cute.Pointer,
        m_partial_ptr: cute.Pointer,
        l_partial_ptr: cute.Pointer,
        o_partial_ptr: cute.Pointer,
        w_uv_ptr: cute.Pointer,
        w_o_ptr: cute.Pointer,
        o_per_head_out_ptr: cute.Pointer,
        attn_proj_out_ptr: cute.Pointer,
        gamma4_ptr: cute.Pointer,
        w_gate_router_ptr: cute.Pointer,
        w_gate_routed_ptr: cute.Pointer,
        w_up_routed_ptr: cute.Pointer,
        w_down_routed_ptr: cute.Pointer,
        w_gate_shared_ptr: cute.Pointer,
        w_up_shared_ptr: cute.Pointer,
        w_down_shared_ptr: cute.Pointer,
        router_logits_ptr: cute.Pointer,
        topk_w_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        inter_tmp_ptr: cute.Pointer,
        moe_out_ptr: cute.Pointer,
        w_gate_shared_fp8_ptr: cute.Pointer,
        w_up_shared_fp8_ptr: cute.Pointer,
        sfa_gate_q1_ptr: cute.Pointer,
        sfa_up_q1_ptr: cute.Pointer,
        w_down_shared_fp8_ptr: cute.Pointer,
        sfa_down_q2_ptr: cute.Pointer,
        w_gate_routed_fp8_ptr: cute.Pointer,
        w_up_routed_fp8_ptr: cute.Pointer,
        w_down_routed_fp8_ptr: cute.Pointer,
        sf_gate_routed_ptr: cute.Pointer,
        sf_up_routed_ptr: cute.Pointer,
        sf_down_routed_ptr: cute.Pointer,
        sf_gate_routed_hw_ptr: cute.Pointer,
        sf_up_routed_hw_ptr: cute.Pointer,
        sf_down_routed_hw_ptr: cute.Pointer,
        attn_proj_symm_local_ptr: cute.Pointer,
        attn_proj_symm_mc_ptr: cute.Pointer,
        tp_sync_local_ptr: cute.Pointer,
        tp_sync_mc_ptr: cute.Pointer,
        attn_proj_reduced_ptr: cute.Pointer,
        moe_routed_acc_f32_ptr: cute.Pointer,
        tp_l1_sync_ptr: cute.Pointer,
        moe_out_symm_local_ptr: cute.Pointer,
        moe_out_symm_mc_ptr: cute.Pointer,
        ep_sync_local_ptr: cute.Pointer,
        ep_sync_mc_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        stage_ts_ptr: cute.Pointer,
        h_final_out_ptr: cute.Pointer,
        embed_weight_ptr: cute.Pointer,
        token_id_buf_ptr: cute.Pointer,
        inv_freq_ptr: cute.Pointer,
        position_id_buf_ptr: cute.Pointer,
        gamma_final_ptr: cute.Pointer,
        lm_head_weight_ptr: cute.Pointer,
        next_token_buf_ptr: cute.Pointer,
        argmax_scratch_ptr: cute.Pointer,
        moe_splitk_partials_ptr: cute.Pointer,
        cache_pos: Int32,
        softmax_scale: Float32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        H = const_expr(self.H)
        OUT_QKVA = const_expr(self.OUT_QKVA)
        Lq = const_expr(self.Lq)
        qb_K = const_expr(self.qb_K)
        use_q_lora = const_expr(self.use_q_lora)
        OUT_QB = const_expr(self.OUT_QB)
        Lkv = const_expr(self.Lkv)
        R = const_expr(self.R)
        L = const_expr(self.L)
        S_max = const_expr(self.S_max)
        num_sms = const_expr(self.num_sms)

        mH_in = cute.make_tensor(h_in_ptr, cute.make_layout((H,), stride=(1,)))
        mRes = cute.make_tensor(residual_ptr, cute.make_layout((L, H), stride=(H, 1)))
        mGamma1 = cute.make_tensor(gamma1_ptr, cute.make_layout((L, H), stride=(H, 1)))
        mWqkva = cute.make_tensor(
            w_qkva_ptr,
            cute.make_layout((L, OUT_QKVA, H), stride=(OUT_QKVA * H, H, 1)),
        )
        Lq_alloc = const_expr(max(self.Lq, 1))
        mGamma2 = cute.make_tensor(
            gamma2_ptr, cute.make_layout((L, Lq_alloc), stride=(Lq_alloc, 1)),
        )
        mWqb = cute.make_tensor(
            w_qb_ptr,
            cute.make_layout((L, OUT_QB, qb_K), stride=(OUT_QB * qb_K, qb_K, 1)),
        )
        mGamma3 = cute.make_tensor(gamma3_ptr, cute.make_layout((L, Lkv), stride=(Lkv, 1)))
        mCos = cute.make_tensor(cos_ptr, cute.make_layout((R // 2,), stride=(1,)))
        mSin = cute.make_tensor(sin_ptr, cute.make_layout((R // 2,), stride=(1,)))
        mQkvaOut = cute.make_tensor(qkva_out_ptr, cute.make_layout((L, OUT_QKVA), stride=(OUT_QKVA, 1)))
        mQout = cute.make_tensor(q_out_ptr, cute.make_layout((L, OUT_QB), stride=(OUT_QB, 1)))
        mKvcOut = cute.make_tensor(kvc_out_ptr, cute.make_layout((L, Lkv), stride=(Lkv, 1)))
        mKpeOut = cute.make_tensor(kpe_out_ptr, cute.make_layout((L, R), stride=(R, 1)))
        mKvCacheC = cute.make_tensor(
            kv_cache_c_ptr,
            cute.make_layout((L, S_max, Lkv), stride=(S_max * Lkv, Lkv, 1)),
        )
        mKvCachePe = cute.make_tensor(
            kv_cache_pe_ptr,
            cute.make_layout((L, S_max, R), stride=(S_max * R, R, 1)),
        )
        mKvCacheC_2D = cute.make_tensor(
            kv_cache_c_ptr,
            cute.make_layout((L * S_max, Lkv), stride=(Lkv, 1)),
        )
        mKvCachePe_2D = cute.make_tensor(
            kv_cache_pe_ptr,
            cute.make_layout((L * S_max, R), stride=(R, 1)),
        )

        mma_inst_qk = const_expr(self.MMA_INST_QK)
        mma_tiler_qk = const_expr(self.MMA_TILER_QK)
        mma_inst_pv = const_expr(self.MMA_INST_PV)
        mma_tiler_pv = const_expr(self.MMA_TILER_PV)
        ab_stages_qk = const_expr(self.AB_STAGES_QK)
        cluster_size = const_expr(self.CLUSTER_SIZE)
        op_qk = tcgen05.MmaF16BF16Op(
            self.dtype, Float32, mma_inst_qk,
            tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma_qk = cute.make_tiled_mma(op_qk)
        op_pv = tcgen05.MmaF16BF16Op(
            self.dtype, Float32, mma_inst_pv,
            tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
        )
        tiled_mma_pv = cute.make_tiled_mma(op_pv)
        num_qk_chunks_ce = const_expr(self.NUM_QK_CHUNKS)
        sQ_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, mma_tiler_qk, self.dtype, num_qk_chunks_ce,
        )
        sK_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, mma_tiler_qk, self.dtype, ab_stages_qk,
        )
        sP_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_pv, mma_tiler_pv, self.dtype, 1,
        )
        num_pv_n_splits_ce = const_expr(self.NUM_PV_N_SPLITS)
        sV_layout_one = sm100_utils.make_smem_layout_b(
            tiled_mma_pv, mma_tiler_pv, self.dtype, 1,
        )
        sV_layout_staged = sV_layout_one
        cta_layout_mnk_mma = cute.make_layout((cluster_size, 1, 1))
        cta_layout_vmnk_mma = cute.tiled_divide(
            cta_layout_mnk_mma, (tiled_mma_qk.thr_id,),
        )
        op_tma_mcast = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
            tcgen05.CtaGroup.TWO,
        )
        sK_layout_slice = cute.slice_(sK_layout_staged, (None, None, None, 0))
        tma_atom_kvc_mma, tma_tensor_kvc_mma = (
            cute.nvgpu.make_tiled_tma_atom_B(
                op_tma_mcast,
                mKvCacheC_2D,
                sK_layout_slice,
                mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk_mma.shape,
            )
        )
        tma_atom_kpe_mma, tma_tensor_kpe_mma = (
            cute.nvgpu.make_tiled_tma_atom_B(
                op_tma_mcast,
                mKvCachePe_2D,
                sK_layout_slice,
                mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk_mma.shape,
            )
        )
        PV_N_PER_SPLIT_HOST = const_expr(self.PV_N)
        mV0_NK = cute.make_tensor(
            kv_cache_c_ptr,
            cute.make_layout(
                (PV_N_PER_SPLIT_HOST, L * S_max),
                stride=(1, Lkv),
            ),
        )
        mV1_NK = cute.make_tensor(
            (kv_cache_c_ptr + Int64(PV_N_PER_SPLIT_HOST)).align(16),
            cute.make_layout(
                (PV_N_PER_SPLIT_HOST, L * S_max),
                stride=(1, Lkv),
            ),
        )
        sV_layout_slice = cute.slice_(sV_layout_staged, (None, None, None, 0))
        tma_atom_v0_mma, tma_tensor_v0_mma = (
            cute.nvgpu.make_tiled_tma_atom_B(
                op_tma_mcast,
                mV0_NK,
                sV_layout_slice,
                mma_tiler_pv,
                tiled_mma_pv,
                cta_layout_vmnk_mma.shape,
            )
        )
        tma_atom_v1_mma, tma_tensor_v1_mma = (
            cute.nvgpu.make_tiled_tma_atom_B(
                op_tma_mcast,
                mV1_NK,
                sV_layout_slice,
                mma_tiler_pv,
                tiled_mma_pv,
                cta_layout_vmnk_mma.shape,
            )
        )
        sQ_layout_slice = cute.slice_(sQ_layout_staged, (None, None, None, 0))

        N = const_expr(self.N)
        qk_nope_dim = const_expr(self.qk_nope_dim)
        mWuk = cute.make_tensor(
            w_uk_ptr,
            cute.make_layout(
                (L, N, Lkv, qk_nope_dim),
                stride=(N * Lkv * qk_nope_dim, Lkv * qk_nope_dim, qk_nope_dim, 1),
            ),
        )
        mQnopeAbs = cute.make_tensor(
            q_nope_abs_out_ptr,
            cute.make_layout((L, N, Lkv), stride=(N * Lkv, Lkv, 1)),
        )
        mAttnOut = cute.make_tensor(
            attn_out_ptr,
            cute.make_layout((L, N, Lkv), stride=(N * Lkv, Lkv, 1)),
        )


        NUM_KV_SPLITS_C = const_expr(self.NUM_KV_SPLITS)
        mMPartial = cute.make_tensor(
            m_partial_ptr,
            cute.make_layout((N, NUM_KV_SPLITS_C), stride=(NUM_KV_SPLITS_C, 1)),
        )
        mLPartial = cute.make_tensor(
            l_partial_ptr,
            cute.make_layout((N, NUM_KV_SPLITS_C), stride=(NUM_KV_SPLITS_C, 1)),
        )
        mOPartial = cute.make_tensor(
            o_partial_ptr,
            cute.make_layout(
                (N, NUM_KV_SPLITS_C, Lkv),
                stride=(NUM_KV_SPLITS_C * Lkv, Lkv, 1),
            ),
        )
        v_head_dim = const_expr(self.v_head_dim)
        O_CONCAT = const_expr(self.O_CONCAT)
        mWuv = cute.make_tensor(
            w_uv_ptr,
            cute.make_layout(
                (L, N, v_head_dim, Lkv),
                stride=(N * v_head_dim * Lkv, v_head_dim * Lkv, Lkv, 1),
            ),
        )
        mWo = cute.make_tensor(
            w_o_ptr,
            cute.make_layout((L, H, O_CONCAT), stride=(H * O_CONCAT, O_CONCAT, 1)),
        )
        mOperHead = cute.make_tensor(
            o_per_head_out_ptr,
            cute.make_layout((L, N, v_head_dim), stride=(N * v_head_dim, v_head_dim, 1)),
        )
        mAttnProj = cute.make_tensor(
            attn_proj_out_ptr,
            cute.make_layout((L, H), stride=(H, 1)),
        )
        mAttnProjSymmLocal = cute.make_tensor(
            attn_proj_symm_local_ptr,
            cute.make_layout((L, H), stride=(H, 1)),
        )
        mAttnProjSymmMc = cute.make_tensor(
            attn_proj_symm_mc_ptr,
            cute.make_layout((L, H), stride=(H, 1)),
        )
        mTpSyncLocal = cute.make_tensor(
            tp_sync_local_ptr,
            cute.make_layout((L, num_sms), stride=(num_sms, 1)),
        )
        mTpSyncMc = cute.make_tensor(
            tp_sync_mc_ptr,
            cute.make_layout((L, num_sms), stride=(num_sms, 1)),
        )
        mAttnProjReduced = cute.make_tensor(
            attn_proj_reduced_ptr,
            cute.make_layout((L, H), stride=(H, 1)),
        )
        mMoeRoutedAccF32 = cute.make_tensor(
            moe_routed_acc_f32_ptr,
            cute.make_layout((L, H), stride=(H, 1)),
        )
        mTpL1Sync = cute.make_tensor(
            tp_l1_sync_ptr, cute.make_layout((1,), stride=(1,)),
        )
        mMoeOutSymmLocal = cute.make_tensor(
            moe_out_symm_local_ptr,
            cute.make_layout((L, H), stride=(H, 1)),
        )
        mMoeOutSymmMc = cute.make_tensor(
            moe_out_symm_mc_ptr,
            cute.make_layout((L, H), stride=(H, 1)),
        )
        mEpSyncLocal = cute.make_tensor(
            ep_sync_local_ptr,
            cute.make_layout((L, 132), stride=(132, 1)),
        )
        mEpSyncMc = cute.make_tensor(
            ep_sync_mc_ptr,
            cute.make_layout((L, 132), stride=(132, 1)),
        )
        E = const_expr(self.E)
        K_topk = const_expr(self.K_topk)
        I_routed = const_expr(self.I_routed)
        I_shared = const_expr(self.I_shared)
        I_max_moe = const_expr(self.I_max_moe)
        Q1_INTER_OFFSET = const_expr(self.Q1_INTER_OFFSET)
        E_per_rank = const_expr(self.E_PER_RANK)
        mGamma4 = cute.make_tensor(gamma4_ptr, cute.make_layout((L, H), stride=(H, 1)))
        mWgateRouter = cute.make_tensor(
            w_gate_router_ptr,
            cute.make_layout((L, E, H), stride=(E * H, H, 1)),
        )
        mWgateRouted = cute.make_tensor(
            w_gate_routed_ptr,
            cute.make_layout(
                (L, E_per_rank, I_routed, H),
                stride=(E_per_rank * I_routed * H, I_routed * H, H, 1),
            ),
        )
        mWupRouted = cute.make_tensor(
            w_up_routed_ptr,
            cute.make_layout(
                (L, E_per_rank, I_routed, H),
                stride=(E_per_rank * I_routed * H, I_routed * H, H, 1),
            ),
        )
        mWdownRouted = cute.make_tensor(
            w_down_routed_ptr,
            cute.make_layout(
                (L, E_per_rank, H, I_routed),
                stride=(E_per_rank * H * I_routed, H * I_routed, I_routed, 1),
            ),
        )
        mWgateRouted_fp8 = cute.make_tensor(
            w_gate_routed_fp8_ptr,
            cute.make_layout(
                (L, E_per_rank, I_routed, H),
                stride=(E_per_rank * I_routed * H, I_routed * H, H, 1),
            ),
        )
        mWupRouted_fp8 = cute.make_tensor(
            w_up_routed_fp8_ptr,
            cute.make_layout(
                (L, E_per_rank, I_routed, H),
                stride=(E_per_rank * I_routed * H, I_routed * H, H, 1),
            ),
        )
        mWdownRouted_fp8 = cute.make_tensor(
            w_down_routed_fp8_ptr,
            cute.make_layout(
                (L, E_per_rank, H, I_routed),
                stride=(E_per_rank * H * I_routed, H * I_routed, I_routed, 1),
            ),
        )
        H_SF = const_expr(H // 128)
        I_R_SF = const_expr(I_routed // 128)
        mSfGateRouted = cute.make_tensor(
            sf_gate_routed_ptr,
            cute.make_layout(
                (L, E_per_rank, I_routed, H_SF),
                stride=(E_per_rank * I_routed * H_SF, I_routed * H_SF, H_SF, 1),
            ),
        )
        mSfUpRouted = cute.make_tensor(
            sf_up_routed_ptr,
            cute.make_layout(
                (L, E_per_rank, I_routed, H_SF),
                stride=(E_per_rank * I_routed * H_SF, I_routed * H_SF, H_SF, 1),
            ),
        )
        mSfDownRouted = cute.make_tensor(
            sf_down_routed_ptr,
            cute.make_layout(
                (L, E_per_rank, H, I_R_SF),
                stride=(E_per_rank * H * I_R_SF, H * I_R_SF, I_R_SF, 1),
            ),
        )
        mWgateShared = cute.make_tensor(
            w_gate_shared_ptr,
            cute.make_layout((I_shared, H, L), stride=(H, 1, I_shared * H)),
        )
        mWupShared = cute.make_tensor(
            w_up_shared_ptr,
            cute.make_layout((I_shared, H, L), stride=(H, 1, I_shared * H)),
        )
        mWdownShared = cute.make_tensor(
            w_down_shared_ptr,
            cute.make_layout((L, H, I_shared), stride=(H * I_shared, I_shared, 1)),
        )
        _A4_EL_TOTAL = E_per_rank * L
        mWgateRouted_fp8_mma = cute.make_tensor(
            w_gate_routed_fp8_ptr,
            cute.make_layout(
                (I_routed, H, _A4_EL_TOTAL),
                stride=(H, 1, I_routed * H),
            ),
        )
        mWupRouted_fp8_mma = cute.make_tensor(
            w_up_routed_fp8_ptr,
            cute.make_layout(
                (I_routed, H, _A4_EL_TOTAL),
                stride=(H, 1, I_routed * H),
            ),
        )
        mWdownRouted_fp8_mma = cute.make_tensor(
            w_down_routed_fp8_ptr,
            cute.make_layout(
                (H, I_routed, _A4_EL_TOTAL),
                stride=(I_routed, 1, H * I_routed),
            ),
        )
        mWgateShared_fp8 = cute.make_tensor(
            w_gate_shared_fp8_ptr,
            cute.make_layout((I_shared, H, L), stride=(H, 1, I_shared * H)),
        )
        mWupShared_fp8 = cute.make_tensor(
            w_up_shared_fp8_ptr,
            cute.make_layout((I_shared, H, L), stride=(H, 1, I_shared * H)),
        )
        sfa_q1_layout = blockscaled_utils.tile_atom_to_shape_SF(
            mWgateShared_fp8.shape, const_expr(self.Q1_SF_VEC_SIZE),
        )
        mSFA_g_q1 = cute.make_tensor(sfa_gate_q1_ptr, sfa_q1_layout)
        mSFA_u_q1 = cute.make_tensor(sfa_up_q1_ptr, sfa_q1_layout)

        q1_outs_per_cta_h = const_expr(self.outs_per_cta)
        q1_tile_k_h = const_expr(self.Q1_TILE_K)
        smem_layout_q1_one = cute.make_layout(
            (q1_outs_per_cta_h, q1_tile_k_h),
            stride=(q1_tile_k_h, 1),
        )
        tma_atom_wg_q1, tma_tensor_wg_q1 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mWgateShared,
            smem_layout_q1_one,
            (q1_outs_per_cta_h, q1_tile_k_h),
        )
        tma_atom_wu_q1, tma_tensor_wu_q1 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mWupShared,
            smem_layout_q1_one,
            (q1_outs_per_cta_h, q1_tile_k_h),
        )
        mWdownShared_tma = cute.make_tensor(
            w_down_shared_ptr,
            cute.make_layout((H, I_shared, L), stride=(I_shared, 1, H * I_shared)),
        )
        q2_outs_per_cta_h = const_expr(self.outs_per_cta)
        q2_tile_k_h = const_expr(self.Q2_TILE_K)
        smem_layout_q2_one = cute.make_layout(
            (q2_outs_per_cta_h, q2_tile_k_h),
            stride=(q2_tile_k_h, 1),
        )
        tma_atom_wd_q2, tma_tensor_wd_q2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mWdownShared_tma,
            smem_layout_q2_one,
            (q2_outs_per_cta_h, q2_tile_k_h),
        )

        mWdownShared_fp8_tma = cute.make_tensor(
            w_down_shared_fp8_ptr,
            cute.make_layout((H, I_shared, L), stride=(I_shared, 1, H * I_shared)),
        )
        tma_atom_wd_q2_fp8, tma_tensor_wd_q2_fp8 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mWdownShared_fp8_tma,
            smem_layout_q2_one,
            (q2_outs_per_cta_h, q2_tile_k_h),
        )
        Q2_K_SF = const_expr(I_shared // 128)
        mSF_down_q2 = cute.make_tensor(
            sfa_down_q2_ptr,
            cute.make_layout((H, Q2_K_SF, L), stride=(Q2_K_SF, 1, H * Q2_K_SF)),
        )
        smem_layout_sf_q2 = cute.make_layout(
            (q2_outs_per_cta_h, Q2_K_SF),
            stride=(Q2_K_SF, 1),
        )
        tma_atom_sf_q2, tma_tensor_sf_q2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mSF_down_q2,
            smem_layout_sf_q2,
            (q2_outs_per_cta_h, Q2_K_SF),
        )

        q1_mma_inst = const_expr(self.Q1_MMA_INST)
        q1_mma_tiler = const_expr(self.Q1_MMA_TILER)
        q1_ab_stages = const_expr(self.Q1_AB_STAGES)
        op_q1 = tcgen05.MmaF16BF16Op(
            self.dtype, Float32, q1_mma_inst,
            tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma_q1 = cute.make_tiled_mma(op_q1)
        sWg_q1_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_q1, q1_mma_tiler, self.dtype, q1_ab_stages,
        )
        sWu_q1_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_q1, q1_mma_tiler, self.dtype, q1_ab_stages,
        )
        sX_q1_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_q1, q1_mma_tiler, self.dtype, q1_ab_stages,
        )
        op_tma_q1_a = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
            tcgen05.CtaGroup.TWO,
        )
        sWg_q1_layout_slice = cute.slice_(sWg_q1_layout_staged, (None, None, None, 0))
        sWu_q1_layout_slice = cute.slice_(sWu_q1_layout_staged, (None, None, None, 0))
        tma_atom_wg_q1_mma, tma_tensor_wg_q1_mma = (
            cute.nvgpu.make_tiled_tma_atom_A(
                op_tma_q1_a,
                mWgateShared,
                sWg_q1_layout_slice,
                q1_mma_tiler,
                tiled_mma_q1,
                cta_layout_vmnk_mma.shape,
            )
        )
        tma_atom_wu_q1_mma, tma_tensor_wu_q1_mma = (
            cute.nvgpu.make_tiled_tma_atom_A(
                op_tma_q1_a,
                mWupShared,
                sWu_q1_layout_slice,
                q1_mma_tiler,
                tiled_mma_q1,
                cta_layout_vmnk_mma.shape,
            )
        )
        q1_fp8_mma_inst    = const_expr(self.Q1_FP8_MMA_INST)
        q1_fp8_mma_tiler   = const_expr(self.Q1_FP8_MMA_TILER)
        q1_fp8_ab_stages   = const_expr(self.Q1_FP8_AB_STAGES)
        q1_sf_vec_size     = const_expr(self.Q1_SF_VEC_SIZE)
        q1_fp8_mma_inst_mn     = (q1_fp8_mma_inst[0], q1_fp8_mma_inst[1])
        q1_fp8_mma_inst_mn_sfb = (
            q1_fp8_mma_inst[0] // 2,
            cute.round_up(q1_fp8_mma_inst[1], 128),
        )
        tiled_mma_q1_fp8 = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            q1_sf_vec_size,
            tcgen05.CtaGroup.TWO,
            q1_fp8_mma_inst_mn,
        )
        tiled_mma_q1_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            q1_sf_vec_size,
            tcgen05.CtaGroup.ONE,
            q1_fp8_mma_inst_mn_sfb,
        )
        q1_fp8_cluster_shape_mn = (2, 1)
        q1_fp8_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*q1_fp8_cluster_shape_mn, 1)),
            (tiled_mma_q1_fp8.thr_id.shape,),
        )
        q1_fp8_cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*q1_fp8_cluster_shape_mn, 1)),
            (tiled_mma_q1_sfb.thr_id.shape,),
        )
        sWg_q1_fp8_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_q1_fp8, q1_fp8_mma_tiler, cutlass.Float8E4M3FN, q1_fp8_ab_stages,
        )
        sWu_q1_fp8_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_q1_fp8, q1_fp8_mma_tiler, cutlass.Float8E4M3FN, q1_fp8_ab_stages,
        )
        sX_q1_fp8_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_q1_fp8, q1_fp8_mma_tiler, cutlass.Float8E4M3FN, q1_fp8_ab_stages,
        )
        sSFA_g_q1_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_q1_fp8, q1_fp8_mma_tiler, q1_sf_vec_size, q1_fp8_ab_stages,
        )
        sSFA_u_q1_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_q1_fp8, q1_fp8_mma_tiler, q1_sf_vec_size, q1_fp8_ab_stages,
        )
        sSFB_q1_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_q1_fp8, q1_fp8_mma_tiler, q1_sf_vec_size, q1_fp8_ab_stages,
        )
        sWg_q1_fp8_layout_slice = cute.slice_(sWg_q1_fp8_layout_staged, (None, None, None, 0))
        sWu_q1_fp8_layout_slice = cute.slice_(sWu_q1_fp8_layout_staged, (None, None, None, 0))
        sSFA_g_q1_layout_slice  = cute.slice_(sSFA_g_q1_layout_staged,  (None, None, None, 0))
        sSFA_u_q1_layout_slice  = cute.slice_(sSFA_u_q1_layout_staged,  (None, None, None, 0))
        sSFB_q1_layout_slice    = cute.slice_(sSFB_q1_layout_staged,    (None, None, None, 0))
        op_tma_q1_fp8_a = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
            tcgen05.CtaGroup.TWO,
        )
        tma_atom_wg_q1_fp8, tma_tensor_wg_q1_fp8 = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma_q1_fp8_a,
            mWgateShared_fp8,
            sWg_q1_fp8_layout_slice,
            q1_fp8_mma_tiler,
            tiled_mma_q1_fp8,
            q1_fp8_cluster_layout_vmnk.shape,
        )
        tma_atom_wu_q1_fp8, tma_tensor_wu_q1_fp8 = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma_q1_fp8_a,
            mWupShared_fp8,
            sWu_q1_fp8_layout_slice,
            q1_fp8_mma_tiler,
            tiled_mma_q1_fp8,
            q1_fp8_cluster_layout_vmnk.shape,
        )
        tma_atom_sfa_g_q1, tma_tensor_sfa_g_q1 = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma_q1_fp8_a,
            mSFA_g_q1,
            sSFA_g_q1_layout_slice,
            q1_fp8_mma_tiler,
            tiled_mma_q1_fp8,
            q1_fp8_cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfa_u_q1, tma_tensor_sfa_u_q1 = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma_q1_fp8_a,
            mSFA_u_q1,
            sSFA_u_q1_layout_slice,
            q1_fp8_mma_tiler,
            tiled_mma_q1_fp8,
            q1_fp8_cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        q2_mma_inst = const_expr(self.Q2_MMA_INST)
        q2_mma_tiler = const_expr(self.Q2_MMA_TILER)
        q2_ab_stages = const_expr(self.Q2_AB_STAGES)
        op_q2 = tcgen05.MmaF16BF16Op(
            self.dtype, Float32, q2_mma_inst,
            tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma_q2 = cute.make_tiled_mma(op_q2)
        sWd_q2_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_q2, q2_mma_tiler, self.dtype, q2_ab_stages,
        )
        sX_q2_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_q2, q2_mma_tiler, self.dtype, q2_ab_stages,
        )
        sWd_q2_layout_slice = cute.slice_(sWd_q2_layout_staged, (None, None, None, 0))
        tma_atom_wd_q2_mma, tma_tensor_wd_q2_mma = (
            cute.nvgpu.make_tiled_tma_atom_A(
                op_tma_q1_a,
                mWdownShared_tma,
                sWd_q2_layout_slice,
                q2_mma_tiler,
                tiled_mma_q2,
                cta_layout_vmnk_mma.shape,
            )
        )
        q2_fp8_mma_inst    = const_expr(self.Q2_FP8_MMA_INST)
        q2_fp8_mma_tiler   = const_expr(self.Q2_FP8_MMA_TILER)
        q2_fp8_ab_stages   = const_expr(self.Q2_FP8_AB_STAGES)
        q2_sf_vec_size     = const_expr(self.Q2_SF_VEC_SIZE)
        q2_fp8_mma_inst_mn     = (q2_fp8_mma_inst[0], q2_fp8_mma_inst[1])
        q2_fp8_mma_inst_mn_sfb = (
            q2_fp8_mma_inst[0] // 2,
            cute.round_up(q2_fp8_mma_inst[1], 128),
        )
        tiled_mma_q2_fp8 = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            q2_sf_vec_size,
            tcgen05.CtaGroup.TWO,
            q2_fp8_mma_inst_mn,
        )
        tiled_mma_q2_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            q2_sf_vec_size,
            tcgen05.CtaGroup.ONE,
            q2_fp8_mma_inst_mn_sfb,
        )
        q2_fp8_cluster_shape_mn = (2, 1)
        q2_fp8_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*q2_fp8_cluster_shape_mn, 1)),
            (tiled_mma_q2_fp8.thr_id.shape,),
        )
        q2_fp8_cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*q2_fp8_cluster_shape_mn, 1)),
            (tiled_mma_q2_sfb.thr_id.shape,),
        )
        sWd_q2_fp8_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_q2_fp8, q2_fp8_mma_tiler, cutlass.Float8E4M3FN, q2_fp8_ab_stages,
        )
        sX_q2_fp8_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_q2_fp8, q2_fp8_mma_tiler, cutlass.Float8E4M3FN, q2_fp8_ab_stages,
        )
        sSFA_down_q2_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_q2_fp8, q2_fp8_mma_tiler, q2_sf_vec_size, q2_fp8_ab_stages,
        )
        sSFB_q2_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_q2_fp8, q2_fp8_mma_tiler, q2_sf_vec_size, q2_fp8_ab_stages,
        )
        sWd_q2_fp8_layout_slice  = cute.slice_(sWd_q2_fp8_layout_staged,   (None, None, None, 0))
        sSFA_down_q2_layout_slice = cute.slice_(sSFA_down_q2_layout_staged, (None, None, None, 0))
        op_tma_q2_fp8_a = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
            tcgen05.CtaGroup.TWO,
        )
        tma_atom_wd_q2_fp8_mma, tma_tensor_wd_q2_fp8_mma = (
            cute.nvgpu.make_tiled_tma_atom_A(
                op_tma_q2_fp8_a,
                mWdownShared_fp8_tma,
                sWd_q2_fp8_layout_slice,
                q2_fp8_mma_tiler,
                tiled_mma_q2_fp8,
                q2_fp8_cluster_layout_vmnk.shape,
            )
        )
        sfa_q2_layout = blockscaled_utils.tile_atom_to_shape_SF(
            mWdownShared_fp8_tma.shape, const_expr(self.Q2_SF_VEC_SIZE),
        )
        mSFA_down_q2 = cute.make_tensor(sfa_down_q2_ptr, sfa_q2_layout)
        tma_atom_sfa_down_q2_mma, tma_tensor_sfa_down_q2_mma = (
            cute.nvgpu.make_tiled_tma_atom_A(
                op_tma_q2_fp8_a,
                mSFA_down_q2,
                sSFA_down_q2_layout_slice,
                q2_fp8_mma_tiler,
                tiled_mma_q2_fp8,
                q2_fp8_cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        )


        moe_p1_mma_inst    = const_expr(self.MOE_P1_MMA_INST)
        moe_p1_mma_tiler   = const_expr(self.MOE_P1_MMA_TILER)
        moe_p1_ab_stages   = const_expr(self.MOE_P1_AB_STAGES)
        moe_p2_mma_inst    = const_expr(self.MOE_P2_MMA_INST)
        moe_p2_mma_tiler   = const_expr(self.MOE_P2_MMA_TILER)
        moe_p2_ab_stages   = const_expr(self.MOE_P2_AB_STAGES)
        moe_sf_vec         = const_expr(self.MOE_SF_VEC_SIZE)
        moe_p1_mma_inst_mn = (moe_p1_mma_inst[0], moe_p1_mma_inst[1])
        moe_p2_mma_inst_mn = (moe_p2_mma_inst[0], moe_p2_mma_inst[1])
        moe_p1_mma_inst_mn_sfb = (
            moe_p1_mma_inst[0] // 2,
            cute.round_up(moe_p1_mma_inst[1], 128),
        )
        moe_p2_mma_inst_mn_sfb = (
            moe_p2_mma_inst[0] // 2,
            cute.round_up(moe_p2_mma_inst[1], 128),
        )

        tiled_mma_moe_p1_fp8 = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            moe_sf_vec,
            tcgen05.CtaGroup.TWO,
            moe_p1_mma_inst_mn,
        )
        tiled_mma_moe_p2_fp8 = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            moe_sf_vec,
            tcgen05.CtaGroup.TWO,
            moe_p2_mma_inst_mn,
        )
        tiled_mma_moe_p1_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            moe_sf_vec,
            tcgen05.CtaGroup.ONE,
            moe_p1_mma_inst_mn_sfb,
        )
        tiled_mma_moe_p2_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            moe_sf_vec,
            tcgen05.CtaGroup.ONE,
            moe_p2_mma_inst_mn_sfb,
        )

        moe_p1_cluster_shape_mn = (self.CLUSTER_SIZE, 1)
        moe_p1_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*moe_p1_cluster_shape_mn, 1)),
            (tiled_mma_moe_p1_fp8.thr_id.shape,),
        )
        moe_p1_cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*moe_p1_cluster_shape_mn, 1)),
            (tiled_mma_moe_p1_sfb.thr_id.shape,),
        )
        moe_p2_cluster_shape_mn = (self.CLUSTER_SIZE, 1)
        moe_p2_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*moe_p2_cluster_shape_mn, 1)),
            (tiled_mma_moe_p2_fp8.thr_id.shape,),
        )
        moe_p2_cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*moe_p2_cluster_shape_mn, 1)),
            (tiled_mma_moe_p2_sfb.thr_id.shape,),
        )

        sWg_moe_p1_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_moe_p1_fp8, moe_p1_mma_tiler, cutlass.Float8E4M3FN, moe_p1_ab_stages,
        )
        sWu_moe_p1_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_moe_p1_fp8, moe_p1_mma_tiler, cutlass.Float8E4M3FN, moe_p1_ab_stages,
        )
        sX_moe_p1_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_moe_p1_fp8, moe_p1_mma_tiler, cutlass.Float8E4M3FN, moe_p1_ab_stages,
        )
        sSFA_g_moe_p1_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_moe_p1_fp8, moe_p1_mma_tiler, moe_sf_vec, moe_p1_ab_stages,
        )
        sSFA_u_moe_p1_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_moe_p1_fp8, moe_p1_mma_tiler, moe_sf_vec, moe_p1_ab_stages,
        )
        sSFB_moe_p1_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_moe_p1_fp8, moe_p1_mma_tiler, moe_sf_vec, moe_p1_ab_stages,
        )
        sWd_moe_p2_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_moe_p2_fp8, moe_p2_mma_tiler, cutlass.Float8E4M3FN, moe_p2_ab_stages,
        )
        sX_moe_p2_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_moe_p2_fp8, moe_p2_mma_tiler, cutlass.Float8E4M3FN, moe_p2_ab_stages,
        )
        sSFA_d_moe_p2_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_moe_p2_fp8, moe_p2_mma_tiler, moe_sf_vec, moe_p2_ab_stages,
        )
        sSFB_moe_p2_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_moe_p2_fp8, moe_p2_mma_tiler, moe_sf_vec, moe_p2_ab_stages,
        )

        sWg_moe_p1_layout_slice = cute.slice_(sWg_moe_p1_layout_staged, (None, None, None, 0))
        sWu_moe_p1_layout_slice = cute.slice_(sWu_moe_p1_layout_staged, (None, None, None, 0))
        sSFA_g_moe_p1_layout_slice = cute.slice_(sSFA_g_moe_p1_layout_staged, (None, None, None, 0))
        sSFA_u_moe_p1_layout_slice = cute.slice_(sSFA_u_moe_p1_layout_staged, (None, None, None, 0))
        sWd_moe_p2_layout_slice = cute.slice_(sWd_moe_p2_layout_staged, (None, None, None, 0))
        sSFA_d_moe_p2_layout_slice = cute.slice_(sSFA_d_moe_p2_layout_staged, (None, None, None, 0))

        sf_layout_moe_p1 = cute.tile_to_shape(
            blockscaled_utils.BlockScaledBasicChunk(moe_sf_vec).layout,
            (I_routed, H, _A4_EL_TOTAL),
            (2, 1, 3),
        )
        sf_layout_moe_p2 = cute.tile_to_shape(
            blockscaled_utils.BlockScaledBasicChunk(moe_sf_vec).layout,
            (H, I_routed, _A4_EL_TOTAL),
            (2, 1, 3),
        )
        mSfGateRouted_hw = cute.make_tensor(sf_gate_routed_hw_ptr, sf_layout_moe_p1)
        mSfUpRouted_hw   = cute.make_tensor(sf_up_routed_hw_ptr,   sf_layout_moe_p1)
        mSfDownRouted_hw = cute.make_tensor(sf_down_routed_hw_ptr, sf_layout_moe_p2)

        op_tma_moe_p1_a = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
            tcgen05.CtaGroup.TWO,
        )
        op_tma_moe_p2_a = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
            tcgen05.CtaGroup.TWO,
        )
        tma_atom_wg_moe_p1, tma_tensor_wg_moe_p1 = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma_moe_p1_a,
            mWgateRouted_fp8_mma,
            sWg_moe_p1_layout_slice,
            moe_p1_mma_tiler,
            tiled_mma_moe_p1_fp8,
            moe_p1_cluster_layout_vmnk.shape,
        )
        tma_atom_wu_moe_p1, tma_tensor_wu_moe_p1 = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma_moe_p1_a,
            mWupRouted_fp8_mma,
            sWu_moe_p1_layout_slice,
            moe_p1_mma_tiler,
            tiled_mma_moe_p1_fp8,
            moe_p1_cluster_layout_vmnk.shape,
        )
        tma_atom_sfa_g_moe_p1, tma_tensor_sfa_g_moe_p1 = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma_moe_p1_a,
            mSfGateRouted_hw,
            sSFA_g_moe_p1_layout_slice,
            moe_p1_mma_tiler,
            tiled_mma_moe_p1_fp8,
            moe_p1_cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfa_u_moe_p1, tma_tensor_sfa_u_moe_p1 = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma_moe_p1_a,
            mSfUpRouted_hw,
            sSFA_u_moe_p1_layout_slice,
            moe_p1_mma_tiler,
            tiled_mma_moe_p1_fp8,
            moe_p1_cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_wd_moe_p2, tma_tensor_wd_moe_p2 = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma_moe_p2_a,
            mWdownRouted_fp8_mma,
            sWd_moe_p2_layout_slice,
            moe_p2_mma_tiler,
            tiled_mma_moe_p2_fp8,
            moe_p2_cluster_layout_vmnk.shape,
        )
        tma_atom_sfa_d_moe_p2, tma_tensor_sfa_d_moe_p2 = cute.nvgpu.make_tiled_tma_atom_A(
            op_tma_moe_p2_a,
            mSfDownRouted_hw,
            sSFA_d_moe_p2_layout_slice,
            moe_p2_mma_tiler,
            tiled_mma_moe_p2_fp8,
            moe_p2_cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        stage_k_mma_inst   = const_expr(self.STAGE_K_MMA_INST)
        stage_k_mma_tiler  = const_expr(self.STAGE_K_MMA_TILER)
        stage_k_ab_stages  = const_expr(self.STAGE_K_AB_STAGES)
        op_stage_k = tcgen05.MmaF16BF16Op(
            self.dtype, Float32, stage_k_mma_inst,
            tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma_stage_k = cute.make_tiled_mma(op_stage_k)
        sWo_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_stage_k, stage_k_mma_tiler, self.dtype, stage_k_ab_stages,
        )
        sOperHead_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_stage_k, stage_k_mma_tiler, self.dtype, stage_k_ab_stages,
        )
        mWo_tma = cute.make_tensor(
            w_o_ptr,
            cute.make_layout(
                (H, O_CONCAT, L),
                stride=(O_CONCAT, 1, H * O_CONCAT),
            ),
        )
        sWo_layout_slice = cute.slice_(sWo_layout_staged, (None, None, None, 0))
        tma_atom_wo_mma, tma_tensor_wo_mma = (
            cute.nvgpu.make_tiled_tma_atom_A(
                op_tma_q1_a,
                mWo_tma,
                sWo_layout_slice,
                stage_k_mma_tiler,
                tiled_mma_stage_k,
                cta_layout_vmnk_mma.shape,
            )
        )

        mRouterLogits = cute.make_tensor(
            router_logits_ptr,
            cute.make_layout((L, E), stride=(E, 1)),
        )
        mTopkW = cute.make_tensor(
            topk_w_ptr,
            cute.make_layout((L, K_topk), stride=(K_topk, 1)),
        )
        mTopkIds = cute.make_tensor(
            topk_ids_ptr,
            cute.make_layout((L, K_topk), stride=(K_topk, 1)),
        )
        mInterTmp = cute.make_tensor(
            inter_tmp_ptr,
            cute.make_layout((I_max_moe,), stride=(1,)),
        )
        SPLITK_KI = const_expr(self.K_topk * self.I_routed)
        SPLITK_IS = const_expr(self.I_shared)
        SPLITK_WIDTH = const_expr(2 * 2 * (SPLITK_KI + SPLITK_IS))
        mPartials = cute.make_tensor(
            moe_splitk_partials_ptr,
            cute.make_layout((self.L, SPLITK_WIDTH), stride=(SPLITK_WIDTH, 1)),
        )
        mMoeOut = cute.make_tensor(
            moe_out_ptr,
            cute.make_layout((L, H), stride=(H, 1)),
        )
        mCounter = cute.make_tensor(counter_ptr, cute.make_layout((1,), stride=(1,)))
        mStageTs = cute.make_tensor(
            stage_ts_ptr,
            cute.make_layout((L, 48), stride=(48, 1)),
        )
        mHFinal = cute.make_tensor(
            h_final_out_ptr, cute.make_layout((H,), stride=(1,)),
        )
        V = const_expr(self.V)
        mEmbedWeight = cute.make_tensor(
            embed_weight_ptr, cute.make_layout((V, H), stride=(H, 1)),
        )
        mToken = cute.make_tensor(
            token_id_buf_ptr, cute.make_layout((1,), stride=(1,)),
        )
        mInvFreq = cute.make_tensor(
            inv_freq_ptr, cute.make_layout((R // 2,), stride=(1,)),
        )
        mPos = cute.make_tensor(
            position_id_buf_ptr, cute.make_layout((1,), stride=(1,)),
        )
        mGammaFinal = cute.make_tensor(
            gamma_final_ptr, cute.make_layout((H,), stride=(1,)),
        )
        mLmHead = cute.make_tensor(
            lm_head_weight_ptr, cute.make_layout((V, H), stride=(H, 1)),
        )
        mNextTokenBuf = cute.make_tensor(
            next_token_buf_ptr, cute.make_layout((1,), stride=(1,)),
        )
        mArgmaxScratch = cute.make_tensor(
            argmax_scratch_ptr,
            cute.make_layout((const_expr(num_sms * 2),), stride=(1,)),
        )

        self.kernel(
            mH_in, mRes, mGamma1, mWqkva, mGamma2, mWqb,
            mGamma3, mCos, mSin,
            mQkvaOut, mQout, mKvcOut, mKpeOut,
            mKvCacheC, mKvCachePe,
            mWuk, mQnopeAbs, mAttnOut,
            mMPartial, mLPartial, mOPartial,
            mWuv, mWo, mOperHead, mAttnProj,
            mGamma4, mWgateRouter,
            mWgateRouted, mWupRouted, mWdownRouted,
            mWgateShared, mWupShared, mWdownShared,
            mRouterLogits, mTopkW, mTopkIds, mInterTmp, mPartials, mMoeOut,
            mWgateRouted_fp8, mWupRouted_fp8, mWdownRouted_fp8,
            mSfGateRouted, mSfUpRouted, mSfDownRouted,
            mAttnProjSymmLocal, mAttnProjSymmMc,
            mTpSyncLocal, mTpSyncMc,
            mAttnProjReduced, mMoeRoutedAccF32, mTpL1Sync,
            mMoeOutSymmLocal, mMoeOutSymmMc,
            mEpSyncLocal, mEpSyncMc,
            mCounter, mStageTs, mHFinal,
            mEmbedWeight, mToken,
            mInvFreq, mPos,
            mGammaFinal, mLmHead, mNextTokenBuf, mArgmaxScratch,
            cache_pos, softmax_scale, eps,
            tiled_mma_qk, tiled_mma_pv,
            tma_atom_kvc_mma, tma_tensor_kvc_mma,
            tma_atom_kpe_mma, tma_tensor_kpe_mma,
            tma_atom_v0_mma, tma_tensor_v0_mma,
            tma_atom_v1_mma, tma_tensor_v1_mma,
            sQ_layout_staged, sK_layout_staged,
            sP_layout_staged, sV_layout_staged,
            cta_layout_vmnk_mma,
            tma_atom_wg_q1, tma_tensor_wg_q1,
            tma_atom_wu_q1, tma_tensor_wu_q1,
            tma_atom_wd_q2, tma_tensor_wd_q2,
            tiled_mma_q1, tiled_mma_q2,
            tma_atom_wg_q1_mma, tma_tensor_wg_q1_mma,
            tma_atom_wu_q1_mma, tma_tensor_wu_q1_mma,
            tma_atom_wd_q2_mma, tma_tensor_wd_q2_mma,
            sWg_q1_layout_staged, sWu_q1_layout_staged, sX_q1_layout_staged,
            sWd_q2_layout_staged, sX_q2_layout_staged,
            sWg_q1_fp8_layout_staged, sWu_q1_fp8_layout_staged, sX_q1_fp8_layout_staged,
            sSFA_g_q1_layout_staged, sSFA_u_q1_layout_staged, sSFB_q1_layout_staged,
            tiled_mma_q1_fp8, tiled_mma_q1_sfb,
            tma_atom_wg_q1_fp8, tma_tensor_wg_q1_fp8,
            tma_atom_wu_q1_fp8, tma_tensor_wu_q1_fp8,
            tma_atom_sfa_g_q1, tma_tensor_sfa_g_q1,
            tma_atom_sfa_u_q1, tma_tensor_sfa_u_q1,
            tma_atom_wd_q2_fp8, tma_tensor_wd_q2_fp8,
            tma_atom_sf_q2, tma_tensor_sf_q2,
            tiled_mma_q2_fp8, tiled_mma_q2_sfb,
            sWd_q2_fp8_layout_staged, sX_q2_fp8_layout_staged,
            sSFA_down_q2_layout_staged, sSFB_q2_layout_staged,
            tma_atom_wd_q2_fp8_mma, tma_tensor_wd_q2_fp8_mma,
            tma_atom_sfa_down_q2_mma, tma_tensor_sfa_down_q2_mma,
            tiled_mma_stage_k,
            sWo_layout_staged, sOperHead_layout_staged,
            tma_atom_wo_mma, tma_tensor_wo_mma,
            tiled_mma_moe_p1_fp8, tiled_mma_moe_p1_sfb,
            tiled_mma_moe_p2_fp8, tiled_mma_moe_p2_sfb,
            sWg_moe_p1_layout_staged, sWu_moe_p1_layout_staged, sX_moe_p1_layout_staged,
            sSFA_g_moe_p1_layout_staged, sSFA_u_moe_p1_layout_staged, sSFB_moe_p1_layout_staged,
            sWd_moe_p2_layout_staged, sX_moe_p2_layout_staged,
            sSFA_d_moe_p2_layout_staged, sSFB_moe_p2_layout_staged,
            tma_atom_wg_moe_p1, tma_tensor_wg_moe_p1,
            tma_atom_wu_moe_p1, tma_tensor_wu_moe_p1,
            tma_atom_sfa_g_moe_p1, tma_tensor_sfa_g_moe_p1,
            tma_atom_sfa_u_moe_p1, tma_tensor_sfa_u_moe_p1,
            tma_atom_wd_moe_p2, tma_tensor_wd_moe_p2,
            tma_atom_sfa_d_moe_p2, tma_tensor_sfa_d_moe_p2,
        ).launch(
            grid=[num_sms, 1, 1],
            block=[self.num_threads, 1, 1],
            cluster=self.CLUSTER_SHAPE_MNK,
            smem=self.smem_size_in_bytes(),
            stream=stream,
        )

    @cute.jit
    def _tp_signal_hoist(self, tidx, bidx, warp_id, lane_id, tp_sync_mc_addr):
        cute.arch.fence_acq_rel_sys()
        if warp_id == Int32(self.WS_COMM_WARP):
            if lane_id == Int32(0):
                llvm.inline_asm(
                    None,
                    [tp_sync_mc_addr.ir_value()],
                    "multimem.red.release.sys.global.add.u32 [$0], 1;",
                    "l",
                    has_side_effects=True,
                    is_align_stack=False,
                )

    @cute.jit
    def _tp_spin_late(
        self, tidx, warp_id, lane_id, tp_sync_local_addr, cache_pos,
        mStageTs=None, layer=None, bidx=None,
    ):
        diag = const_expr(mStageTs is not None)
        if warp_id == Int32(self.WS_COMM_WARP):
            if lane_id == Int32(0):
                if const_expr(diag):
                    if bidx == Int32(0):
                        mStageTs[layer, 30] = _clock()
                tp_target = (cache_pos + Int32(1)) * Int32(self.TP_SIZE)
                done_tp = Int32(0)
                while done_tp < tp_target:
                    val_loaded = llvm.inline_asm(
                        _mlir_ir.IntegerType.get_signless(32),
                        [tp_sync_local_addr.ir_value()],
                        "ld.acquire.sys.global.L1::no_allocate.u32 $0, [$1];",
                        "=r,l",
                        has_side_effects=True,
                        is_align_stack=False,
                    )
                    done_tp = Int32(val_loaded)
                if const_expr(diag):
                    if bidx == Int32(0):
                        mStageTs[layer, 31] = _clock()
                cute.arch.fence_acq_rel_sys()
        cute.arch.barrier()

    @cute.kernel
    def kernel(
        self,
        mH_in: cute.Tensor,
        mRes: cute.Tensor,
        mGamma1: cute.Tensor,
        mWqkva: cute.Tensor,
        mGamma2: cute.Tensor,
        mWqb: cute.Tensor,
        mGamma3: cute.Tensor,
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        mQkvaOut: cute.Tensor,
        mQout: cute.Tensor,
        mKvcOut: cute.Tensor,
        mKpeOut: cute.Tensor,
        mKvCacheC: cute.Tensor,
        mKvCachePe: cute.Tensor,
        mWuk: cute.Tensor,
        mQnopeAbs: cute.Tensor,
        mAttnOut: cute.Tensor,
        mMPartial: cute.Tensor,
        mLPartial: cute.Tensor,
        mOPartial: cute.Tensor,
        mWuv: cute.Tensor,
        mWo: cute.Tensor,
        mOperHead: cute.Tensor,
        mAttnProj: cute.Tensor,
        mGamma4: cute.Tensor,
        mWgateRouter: cute.Tensor,
        mWgateRouted: cute.Tensor,
        mWupRouted: cute.Tensor,
        mWdownRouted: cute.Tensor,
        mWgateShared: cute.Tensor,
        mWupShared: cute.Tensor,
        mWdownShared: cute.Tensor,
        mRouterLogits: cute.Tensor,
        mTopkW: cute.Tensor,
        mTopkIds: cute.Tensor,
        mInterTmp: cute.Tensor,
        mPartials: cute.Tensor,
        mMoeOut: cute.Tensor,
        mWgateRouted_fp8: cute.Tensor,
        mWupRouted_fp8: cute.Tensor,
        mWdownRouted_fp8: cute.Tensor,
        mSfGateRouted: cute.Tensor,
        mSfUpRouted: cute.Tensor,
        mSfDownRouted: cute.Tensor,
        mAttnProjSymmLocal: cute.Tensor,
        mAttnProjSymmMc: cute.Tensor,
        mTpSyncLocal: cute.Tensor,
        mTpSyncMc: cute.Tensor,
        mAttnProjReduced: cute.Tensor,
        mMoeRoutedAccF32: cute.Tensor,
        mTpL1Sync: cute.Tensor,
        mMoeOutSymmLocal: cute.Tensor,
        mMoeOutSymmMc: cute.Tensor,
        mEpSyncLocal: cute.Tensor,
        mEpSyncMc: cute.Tensor,
        mCounter: cute.Tensor,
        mStageTs: cute.Tensor,
        mHFinal: cute.Tensor,
        mEmbedWeight: cute.Tensor,
        mToken: cute.Tensor,
        mInvFreq: cute.Tensor,
        mPos: cute.Tensor,
        mGammaFinal: cute.Tensor,
        mLmHead: cute.Tensor,
        mNextTokenBuf: cute.Tensor,
        mArgmaxScratch: cute.Tensor,
        cache_pos: Int32,
        softmax_scale: Float32,
        eps: Float32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_kvc_mma: cute.CopyAtom,
        tma_tensor_kvc_mma: cute.Tensor,
        tma_atom_kpe_mma: cute.CopyAtom,
        tma_tensor_kpe_mma: cute.Tensor,
        tma_atom_v0_mma: cute.CopyAtom,
        tma_tensor_v0_mma: cute.Tensor,
        tma_atom_v1_mma: cute.CopyAtom,
        tma_tensor_v1_mma: cute.Tensor,
        sQ_layout_staged: cute.ComposedLayout,
        sK_layout_staged: cute.ComposedLayout,
        sP_layout_staged: cute.ComposedLayout,
        sV_layout_staged: cute.ComposedLayout,
        cta_layout_vmnk_mma: cute.Layout,
        tma_atom_wg_q1: cute.CopyAtom,
        tma_tensor_wg_q1: cute.Tensor,
        tma_atom_wu_q1: cute.CopyAtom,
        tma_tensor_wu_q1: cute.Tensor,
        tma_atom_wd_q2: cute.CopyAtom,
        tma_tensor_wd_q2: cute.Tensor,
        tiled_mma_q1: cute.TiledMma,
        tiled_mma_q2: cute.TiledMma,
        tma_atom_wg_q1_mma: cute.CopyAtom,
        tma_tensor_wg_q1_mma: cute.Tensor,
        tma_atom_wu_q1_mma: cute.CopyAtom,
        tma_tensor_wu_q1_mma: cute.Tensor,
        tma_atom_wd_q2_mma: cute.CopyAtom,
        tma_tensor_wd_q2_mma: cute.Tensor,
        sWg_q1_layout_staged: cute.ComposedLayout,
        sWu_q1_layout_staged: cute.ComposedLayout,
        sX_q1_layout_staged: cute.ComposedLayout,
        sWd_q2_layout_staged: cute.ComposedLayout,
        sX_q2_layout_staged: cute.ComposedLayout,
        sWg_q1_fp8_layout_staged: cute.ComposedLayout,
        sWu_q1_fp8_layout_staged: cute.ComposedLayout,
        sX_q1_fp8_layout_staged: cute.ComposedLayout,
        sSFA_g_q1_layout_staged: cute.Layout,
        sSFA_u_q1_layout_staged: cute.Layout,
        sSFB_q1_layout_staged: cute.Layout,
        tiled_mma_q1_fp8: cute.TiledMma,
        tiled_mma_q1_sfb: cute.TiledMma,
        tma_atom_wg_q1_fp8: cute.CopyAtom,
        tma_tensor_wg_q1_fp8: cute.Tensor,
        tma_atom_wu_q1_fp8: cute.CopyAtom,
        tma_tensor_wu_q1_fp8: cute.Tensor,
        tma_atom_sfa_g_q1: cute.CopyAtom,
        tma_tensor_sfa_g_q1: cute.Tensor,
        tma_atom_sfa_u_q1: cute.CopyAtom,
        tma_tensor_sfa_u_q1: cute.Tensor,
        tma_atom_wd_q2_fp8: cute.CopyAtom,
        tma_tensor_wd_q2_fp8: cute.Tensor,
        tma_atom_sf_q2: cute.CopyAtom,
        tma_tensor_sf_q2: cute.Tensor,
        tiled_mma_q2_fp8: cute.TiledMma,
        tiled_mma_q2_sfb: cute.TiledMma,
        sWd_q2_fp8_layout_staged: cute.ComposedLayout,
        sX_q2_fp8_layout_staged: cute.ComposedLayout,
        sSFA_down_q2_layout_staged: cute.Layout,
        sSFB_q2_layout_staged: cute.Layout,
        tma_atom_wd_q2_fp8_mma: cute.CopyAtom,
        tma_tensor_wd_q2_fp8_mma: cute.Tensor,
        tma_atom_sfa_down_q2_mma: cute.CopyAtom,
        tma_tensor_sfa_down_q2_mma: cute.Tensor,
        tiled_mma_stage_k: cute.TiledMma,
        sWo_layout_staged: cute.ComposedLayout,
        sOperHead_layout_staged: cute.ComposedLayout,
        tma_atom_wo_mma: cute.CopyAtom,
        tma_tensor_wo_mma: cute.Tensor,
        tiled_mma_moe_p1_fp8: cute.TiledMma,
        tiled_mma_moe_p1_sfb: cute.TiledMma,
        tiled_mma_moe_p2_fp8: cute.TiledMma,
        tiled_mma_moe_p2_sfb: cute.TiledMma,
        sWg_moe_p1_layout_staged: cute.ComposedLayout,
        sWu_moe_p1_layout_staged: cute.ComposedLayout,
        sX_moe_p1_layout_staged: cute.ComposedLayout,
        sSFA_g_moe_p1_layout_staged: cute.Layout,
        sSFA_u_moe_p1_layout_staged: cute.Layout,
        sSFB_moe_p1_layout_staged: cute.Layout,
        sWd_moe_p2_layout_staged: cute.ComposedLayout,
        sX_moe_p2_layout_staged: cute.ComposedLayout,
        sSFA_d_moe_p2_layout_staged: cute.Layout,
        sSFB_moe_p2_layout_staged: cute.Layout,
        tma_atom_wg_moe_p1: cute.CopyAtom,
        tma_tensor_wg_moe_p1: cute.Tensor,
        tma_atom_wu_moe_p1: cute.CopyAtom,
        tma_tensor_wu_moe_p1: cute.Tensor,
        tma_atom_sfa_g_moe_p1: cute.CopyAtom,
        tma_tensor_sfa_g_moe_p1: cute.Tensor,
        tma_atom_sfa_u_moe_p1: cute.CopyAtom,
        tma_tensor_sfa_u_moe_p1: cute.Tensor,
        tma_atom_wd_moe_p2: cute.CopyAtom,
        tma_tensor_wd_moe_p2: cute.Tensor,
        tma_atom_sfa_d_moe_p2: cute.CopyAtom,
        tma_tensor_sfa_d_moe_p2: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        H = const_expr(self.H)
        OUT_QKVA = const_expr(self.OUT_QKVA)
        Lq = const_expr(self.Lq)
        qb_K = const_expr(self.qb_K)
        use_q_lora = const_expr(self.use_q_lora)
        OUT_QB = const_expr(self.OUT_QB)
        Lkv = const_expr(self.Lkv)
        R = const_expr(self.R)
        L = const_expr(self.L)
        SPLITK_SQ_MAX = const_expr(2)
        P1_SPLITK = const_expr(1)
        SPLITK_IS = const_expr(self.I_shared)
        SPLITK_KI = const_expr(self.K_topk * self.I_routed)
        SPLITK_P1G_OFF = const_expr(0)
        SPLITK_P1U_OFF = const_expr(SPLITK_KI * SPLITK_SQ_MAX)
        SPLITK_Q1G_OFF = const_expr(2 * SPLITK_KI * SPLITK_SQ_MAX)
        SPLITK_Q1U_OFF = const_expr(2 * SPLITK_KI * SPLITK_SQ_MAX + SPLITK_IS * SPLITK_SQ_MAX)
        S_max = const_expr(self.S_max)
        v_head_dim = const_expr(self.v_head_dim)
        O_CONCAT = const_expr(self.O_CONCAT)
        E = const_expr(self.E)
        K_topk = const_expr(self.K_topk)
        n_group = const_expr(self.n_group)
        topk_group = const_expr(self.topk_group)
        E_per_group = const_expr(self.E_per_group)
        E_per_rank = const_expr(self.E_PER_RANK)
        I_routed = const_expr(self.I_routed)
        I_shared = const_expr(self.I_shared)
        Q1_INTER_OFFSET = const_expr(self.Q1_INTER_OFFSET)
        routed_scaling = const_expr(Float32(self.routed_scaling))
        moe_norm_topk_prob = const_expr(self.moe_norm_topk_prob)
        SYNCS_PER_LAYER = const_expr(11 + 2 * self.K_topk)
        R2_ROUTED_NUM_CLUSTERS_CE = const_expr(
            self.K_topk * self.I_routed // self.MOE_P1_CLUSTER_TILE_M
        )
        R2_SHARED_NUM_CLUSTERS_CE = const_expr(self.Q2_FP8_NUM_ACTIVE_CLUSTERS)
        R2_PHASE1_END_CLUSTER_CE = const_expr(
            R2_ROUTED_NUM_CLUSTERS_CE + R2_SHARED_NUM_CLUSTERS_CE
        )
        NUM_KV_SPLITS = const_expr(self.NUM_KV_SPLITS)
        num_threads = const_expr(self.num_threads)
        num_sms = const_expr(self.num_sms)
        vec_size = const_expr(self.vec_size)
        threads_per_output = const_expr(self.threads_per_output)
        outs_per_cta = const_expr(self.outs_per_cta)
        warps_per_row = const_expr(self.warps_per_row)
        tiles_qkva = const_expr(self.tiles_qkva)
        tiles_qb = const_expr(self.tiles_qb)
        iters_qkva = const_expr(self.iters_qkva)
        iters_qb = const_expr(self.iters_qb)

        warp_id = tidx // 32
        lane_id = tidx % 32

        smem = cutlass.utils.SmemAllocator()
        sCarry = smem.allocate_tensor(
            mH_in.element_type, cute.make_layout((H,), stride=(1,)),
            byte_alignment=16,
        )
        sNorm1 = smem.allocate_tensor(
            mH_in.element_type, cute.make_layout((H,), stride=(1,)),
            byte_alignment=16,
        )
        sQc_alloc_dim = const_expr(max(self.Lq, 1))
        reduce_buf_slots = const_expr(max(
            self.warps_per_row, self.n_group,
            self.TILE_SEQ_DEC * self.warps_per_row,
        ))
        reduce_buf = smem.allocate_tensor(
            Float32, cute.make_layout((reduce_buf_slots,)), byte_alignment=4,
        )
        ab_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * self.AB_STAGES_QK,)),
            byte_alignment=8,
        )
        v0_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * self.V_STAGES,)),
            byte_alignment=8,
        )
        v1_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * self.V_STAGES,)),
            byte_alignment=8,
        )
        acc_s_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * self.ACC_S_STAGES,)),
            byte_alignment=8,
        )
        acc_o0_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * self.ACC_O_STAGES,)),
            byte_alignment=8,
        )
        acc_o1_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * self.ACC_O_STAGES,)),
            byte_alignment=8,
        )
        p_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * self.P_STAGES,)),
            byte_alignment=8,
        )
        sQ_mma = smem.allocate_tensor(
            mH_in.element_type,
            sQ_layout_staged.outer,
            byte_alignment=1024,
            swizzle=sQ_layout_staged.inner,
        )
        sK_mma = smem.allocate_tensor(
            mH_in.element_type,
            sK_layout_staged.outer,
            byte_alignment=128,
            swizzle=sK_layout_staged.inner,
        )
        sP_mma = smem.allocate_tensor(
            mH_in.element_type,
            sP_layout_staged.outer,
            byte_alignment=128,
            swizzle=sP_layout_staged.inner,
        )
        sV0_mma = smem.allocate_tensor(
            mH_in.element_type,
            sV_layout_staged.outer,
            byte_alignment=1024,
            swizzle=sV_layout_staged.inner,
        )
        sV1_mma = smem.allocate_tensor(
            mH_in.element_type,
            sV_layout_staged.outer,
            byte_alignment=1024,
            swizzle=sV_layout_staged.inner,
        )
        sRowMax_mma = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (self.CTA_TILE_M, self.CLUSTER_SIZE),
                stride=(self.CLUSTER_SIZE, 1),
            ),
            byte_alignment=4,
        )
        sQc = cute.make_tensor(
            cute.recast_ptr(sQ_mma.iterator, dtype=mH_in.element_type),
            cute.make_layout((sQc_alloc_dim,), stride=(1,)),
        )
        sKv = cute.make_tensor(
            cute.recast_ptr(sQ_mma.iterator, dtype=mH_in.element_type),
            cute.make_layout((Lkv,), stride=(1,)),
        )
        Q1_OPC = const_expr(self.outs_per_cta)
        Q1_TILE_K_C = const_expr(self.Q1_TILE_K)
        Q1_NUM_STAGES_C = const_expr(self.Q1_NUM_STAGES)
        Q1_NUM_K_TILES_C = const_expr(self.Q1_NUM_K_TILES)
        Q1_TMA_BYTES_ONE = const_expr(self.q1_tma_bytes_one)
        sWg_q1_mma = cute.make_tensor(
            cute.recast_ptr(
                sQ_mma.iterator,
                sWg_q1_fp8_layout_staged.inner,
                dtype=cutlass.Float8E4M3FN,
            ),
            sWg_q1_fp8_layout_staged.outer,
        )
        sWg_q1_bytes_per_stage = const_expr(
            self.Q1_FP8_CTA_TILE_M * self.Q1_FP8_K_CHUNK
        )
        sWg_q1_total_bytes = const_expr(
            sWg_q1_bytes_per_stage * self.Q1_FP8_AB_STAGES
        )
        sWu_q1_bf16_offset = const_expr(sWg_q1_total_bytes // 2)
        sWu_q1_mma = cute.make_tensor(
            cute.recast_ptr(
                sQ_mma.iterator + sWu_q1_bf16_offset,
                sWu_q1_fp8_layout_staged.inner,
                dtype=cutlass.Float8E4M3FN,
            ),
            sWu_q1_fp8_layout_staged.outer,
        )
        sX_q1_mma = cute.make_tensor(
            cute.recast_ptr(
                sV1_mma.iterator,
                sX_q1_fp8_layout_staged.inner,
                dtype=cutlass.Float8E4M3FN,
            ),
            sX_q1_fp8_layout_staged.outer,
        )
        sSFA_g_q1 = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            sSFA_g_q1_layout_staged,
            byte_alignment=16,
        )
        sSFA_u_q1 = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            sSFA_u_q1_layout_staged,
            byte_alignment=16,
        )
        sSFB_q1 = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            sSFB_q1_layout_staged,
            byte_alignment=16,
        )
        sX_q1_quant_scratch = cute.make_tensor(
            cute.recast_ptr(sK_mma.iterator, dtype=cutlass.Float8E4M3FN),
            cute.make_layout((H,), stride=(1,)),
        )
        sX_q1_sf_scratch = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            cute.make_layout((H // self.Q1_SF_VEC_SIZE,), stride=(1,)),
            byte_alignment=16,
        )
        sX_p1_quant_scratch = cute.make_tensor(
            cute.recast_ptr(sK_mma.iterator, dtype=cutlass.Float8E4M3FN),
            cute.make_layout((H,), stride=(1,)),
        )
        sX_p1_sf_scratch = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            cute.make_layout((H // self.MOE_SF_VEC_SIZE,), stride=(1,)),
            byte_alignment=16,
        )
        q1_ab_mma_stages = const_expr(
            max(self.Q1_AB_STAGES, self.Q1_FP8_AB_STAGES, self.Q2_FP8_AB_STAGES)
        )
        q1_ab_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * q1_ab_mma_stages,)),
            byte_alignment=8,
        )
        q1_acc_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * 1,)),
            byte_alignment=8,
        )
        stage_k_ab_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * self.STAGE_K_AB_STAGES,)),
            byte_alignment=8,
        )
        stage_k_acc_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * 1,)),
            byte_alignment=8,
        )
        sWg_moe_p1 = cute.make_tensor(
            cute.recast_ptr(
                sQ_mma.iterator,
                sWg_moe_p1_layout_staged.inner,
                dtype=cutlass.Float8E4M3FN,
            ),
            sWg_moe_p1_layout_staged.outer,
        )
        sWg_moe_p1_bytes_per_stage = const_expr(
            self.MOE_P1_CTA_TILE_M * self.MOE_P1_K_CHUNK
        )
        sWg_moe_p1_total_bytes = const_expr(
            sWg_moe_p1_bytes_per_stage * self.MOE_P1_AB_STAGES
        )
        sWu_moe_p1_bf16_offset = const_expr(
            sWg_moe_p1_total_bytes // 2
        )
        sWu_moe_p1 = cute.make_tensor(
            cute.recast_ptr(
                sQ_mma.iterator + sWu_moe_p1_bf16_offset,
                sWu_moe_p1_layout_staged.inner,
                dtype=cutlass.Float8E4M3FN,
            ),
            sWu_moe_p1_layout_staged.outer,
        )
        sX_moe_p1 = cute.make_tensor(
            cute.recast_ptr(
                sV0_mma.iterator,
                sX_moe_p1_layout_staged.inner,
                dtype=cutlass.Float8E4M3FN,
            ),
            sX_moe_p1_layout_staged.outer,
        )
        sSFA_g_moe_p1 = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            sSFA_g_moe_p1_layout_staged,
            byte_alignment=128,
        )
        sSFA_u_moe_p1 = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            sSFA_u_moe_p1_layout_staged,
            byte_alignment=128,
        )
        sSFB_moe_p1 = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            sSFB_moe_p1_layout_staged,
            byte_alignment=128,
        )
        moe_p1_ab_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * self.MOE_P1_AB_STAGES,)),
            byte_alignment=8,
        )
        moe_p1_acc_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * 1,)),
            byte_alignment=8,
        )
        sWd_moe_p2 = cute.make_tensor(
            cute.recast_ptr(
                sQ_mma.iterator,
                sWd_moe_p2_layout_staged.inner,
                dtype=cutlass.Float8E4M3FN,
            ),
            sWd_moe_p2_layout_staged.outer,
        )
        sX_moe_p2 = cute.make_tensor(
            cute.recast_ptr(
                sV0_mma.iterator,
                sX_moe_p2_layout_staged.inner,
                dtype=cutlass.Float8E4M3FN,
            ),
            sX_moe_p2_layout_staged.outer,
        )
        sX_p2_quant_scratch = cute.make_tensor(
            cute.recast_ptr(sK_mma.iterator, dtype=cutlass.Float8E4M3FN),
            cute.make_layout((I_routed,), stride=(1,)),
        )
        sX_p2_sf_scratch = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            cute.make_layout((I_routed // self.MOE_SF_VEC_SIZE,), stride=(1,)),
            byte_alignment=16,
        )
        sSFA_d_moe_p2 = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            sSFA_d_moe_p2_layout_staged,
            byte_alignment=128,
        )
        sSFB_moe_p2 = smem.allocate_tensor(
            cutlass.Float8E8M0FNU,
            sSFB_moe_p2_layout_staged,
            byte_alignment=128,
        )
        moe_p2_ab_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * self.MOE_P2_AB_STAGES,)),
            byte_alignment=8,
        )
        moe_p2_acc_mma_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((2 * 1,)),
            byte_alignment=8,
        )
        Q2_OPC = const_expr(self.outs_per_cta)
        Q2_TILE_K_C = const_expr(self.Q2_TILE_K)
        Q2_NUM_STAGES_C = const_expr(self.Q2_NUM_STAGES)
        Q2_NUM_K_TILES_C = const_expr(self.Q2_NUM_K_TILES)
        Q2_VEC_SIZE_C = const_expr(self.Q2_VEC_SIZE)
        Q2_TMA_BYTES_ONE = const_expr(self.q2_tma_bytes_one)
        sQ2_staged_layout = cute.make_layout(
            (Q2_OPC, Q2_TILE_K_C, Q2_NUM_STAGES_C),
            stride=(Q2_TILE_K_C, 1, Q2_OPC * Q2_TILE_K_C),
        )
        sWd_q2 = smem.allocate_tensor(
            cutlass.Float8E4M3FN, sQ2_staged_layout, byte_alignment=128,
        )
        Q2_K_SF_SMEM = const_expr(I_shared // 128)
        sSF_q2_layout = cute.make_layout(
            (Q2_OPC, Q2_K_SF_SMEM), stride=(Q2_K_SF_SMEM, 1),
        )
        sSF_q2 = smem.allocate_tensor(
            cutlass.Float8E8M0FNU, sSF_q2_layout, byte_alignment=16,
        )
        q2_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((Q2_NUM_STAGES_C,)),
            byte_alignment=8,
        )
        q2_free_bars = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((Q2_NUM_STAGES_C,)),
            byte_alignment=8,
        )
        q2_sf_bar = smem.allocate_tensor(
            cutlass.Int64, cute.make_layout((1,)), byte_alignment=8,
        )
        tmem_holding_buf = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((1,)), byte_alignment=4,
        )
        tmem_dealloc_mbar_buf = smem.allocate_tensor(
            cutlass.Int64, cute.make_layout((1,)), byte_alignment=8,
        )

        pipeline.pipeline_init_arrive(
            cluster_shape_mn=self.CLUSTER_SHAPE_MNK, is_relaxed=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=num_threads,
        )
        tmem_alloc = utils.TmemAllocator(
            tmem_holding_buf.iterator,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_buf.iterator,
        )
        tmem_alloc.allocate(self.TMEM_COLS)

        pipeline.pipeline_init_wait(cluster_shape_mn=self.CLUSTER_SHAPE_MNK)

        num_vec_blocks_h = const_expr(self.num_vec_blocks_h)
        num_vec_blocks_lq = const_expr(self.num_vec_blocks_lq)
        tok = mToken[0]
        for vb in cutlass.range_constexpr(num_vec_blocks_h):
            base = (vb * num_threads + tidx) * vec_size
            if base + vec_size <= H:
                for v in cutlass.range_constexpr(vec_size):
                    k = base + v
                    sCarry[k] = mEmbedWeight[tok, k]
        q2_bars_ptr_init = q2_bars.iterator
        q2_free_bars_ptr_init = q2_free_bars.iterator
        if tidx == 0:
            for s_init_q2 in cutlass.range_constexpr(Q2_NUM_STAGES_C):
                cute.arch.mbarrier_init(q2_bars_ptr_init + s_init_q2, 1)
                cute.arch.mbarrier_init(q2_free_bars_ptr_init + s_init_q2, num_threads)
            cute.arch.mbarrier_init(q2_sf_bar.iterator, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        num_vec_blocks_lkv = const_expr(self.num_vec_blocks_lkv)

        Lq_alloc_loop = const_expr(max(self.Lq, 1))
        q2_sf_phase = Int32(0)
        for layer in cutlass.range(L, unroll=1):
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 0] = _clock()
            row_offset_h = Int64(layer) * Int64(H)
            row_offset_lq = Int64(layer) * Int64(Lq_alloc_loop)
            row_offset_lkv = Int64(layer) * Int64(Lkv)
            res_ptr_l = (mRes.iterator + row_offset_h).align(16)
            g1_ptr_l = (mGamma1.iterator + row_offset_h).align(16)
            g2_ptr_l = (mGamma2.iterator + row_offset_lq).align(16)
            g3_ptr_l = (mGamma3.iterator + row_offset_lkv).align(16)
            mResL = cute.make_tensor(res_ptr_l, cute.make_layout((H,), stride=(1,)))
            mG1L = cute.make_tensor(g1_ptr_l, cute.make_layout((H,), stride=(1,)))
            mG2L = cute.make_tensor(g2_ptr_l, cute.make_layout((Lq_alloc_loop,), stride=(1,)))
            mG3L = cute.make_tensor(g3_ptr_l, cute.make_layout((Lkv,), stride=(1,)))

            partial_sq = Float32(0.0)
            for vb in cutlass.range_constexpr(num_vec_blocks_h):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= H:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        c_val = sCarry[k].to(Float32)
                        r_val = mResL[k].to(Float32)
                        s = c_val + r_val
                        sCarry[k] = s.to(mH_in.element_type)
                        partial_sq = partial_sq + s * s

            partial_sq = cute.arch.warp_reduction(
                partial_sq, op=lambda a, b: a + b, threads_in_group=32,
            )
            if lane_id == 0:
                reduce_buf[warp_id] = partial_sq
            cute.arch.barrier()
            total_sq = Float32(0.0)
            for w in cutlass.range_constexpr(warps_per_row):
                total_sq = total_sq + reduce_buf[w]
            mean_sq = total_sq / Float32(H)
            rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

            for vb in cutlass.range_constexpr(num_vec_blocks_h):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= H:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        c_val = sCarry[k].to(Float32)
                        g = mG1L[k].to(Float32)
                        ns = c_val * rstd * g
                        sNorm1[k] = ns.to(mH_in.element_type)
            cute.arch.barrier()

            w_qkva_off = Int64(layer) * Int64(OUT_QKVA) * Int64(H)
            w_qkva_l_ptr = (mWqkva.iterator + w_qkva_off).align(16)
            mWqkvaL = cute.make_tensor(
                w_qkva_l_ptr, cute.make_layout((OUT_QKVA, H), stride=(H, 1)),
            )
            copy_atom_w_qkva = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mWqkva.element_type,
                num_bits_per_copy=128,
            )
            group_idx = tidx // threads_per_output
            lane_in_group = tidx % threads_per_output
            chunk_stride_qkva = const_expr(threads_per_output * vec_size)
            num_full_chunks_qkva = const_expr(H // chunk_stride_qkva)
            for it in cutlass.range_constexpr(iters_qkva):
                tile_idx = bidx + it * num_sms
                if tile_idx < tiles_qkva:
                    j = tile_idx * outs_per_cta + group_idx
                    if j < OUT_QKVA:
                        w_row_ptr_qkva = (mWqkvaL.iterator + j * H).align(16)
                        acc = Float32(0.0)
                        for kc in cutlass.range_constexpr(num_full_chunks_qkva):
                            k_base = kc * chunk_stride_qkva + lane_in_group * vec_size
                            w_vec = cute.make_rmem_tensor(
                                cute.make_layout((vec_size,), stride=(1,)),
                                mWqkva.element_type,
                            )
                            w_slice = cute.make_tensor(
                                (w_row_ptr_qkva + k_base).align(16),
                                cute.make_layout((vec_size,), stride=(1,)),
                            )
                            cute.copy(copy_atom_w_qkva, w_slice, w_vec)
                            for v in cutlass.range_constexpr(vec_size):
                                n_val = sNorm1[k_base + v].to(Float32)
                                w_val = w_vec[v].to(Float32)
                                acc = acc + n_val * w_val
                        acc = cute.arch.warp_reduction(
                            acc, op=lambda a, b: a + b, threads_in_group=threads_per_output,
                        )
                        if lane_in_group == 0:
                            mQkvaOut[layer, j] = acc.to(mQkvaOut.element_type)

            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 35] = _clock()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected = Int32((SYNCS_PER_LAYER * layer + 1) * num_sms)
                done = Int32(0)
                while done < expected:
                    done = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 1] = _clock()

            qkva_off = Int64(layer) * Int64(OUT_QKVA)
            qkva_l_ptr = (mQkvaOut.iterator + qkva_off).align(16)
            mQkvaL = cute.make_tensor(
                qkva_l_ptr, cute.make_layout((OUT_QKVA,), stride=(1,)),
            )
            partial_sq2 = Float32(0.0)
            for vb in cutlass.range_constexpr(num_vec_blocks_lq):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= Lq:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        qc_val = mQkvaL[k].to(Float32)
                        sQc[k] = qc_val.to(mH_in.element_type)
                        partial_sq2 = partial_sq2 + qc_val * qc_val
            partial_sq2 = cute.arch.warp_reduction(
                partial_sq2, op=lambda a, b: a + b, threads_in_group=32,
            )
            if lane_id == 0:
                reduce_buf[warp_id] = partial_sq2
            cute.arch.barrier()
            total_sq2 = Float32(0.0)
            for w in cutlass.range_constexpr(warps_per_row):
                total_sq2 = total_sq2 + reduce_buf[w]
            mean_sq2 = total_sq2 / Float32(Lq)
            rstd2 = cute.math.rsqrt(mean_sq2 + eps, fastmath=True)

            for vb in cutlass.range_constexpr(num_vec_blocks_lq):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= Lq:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        xs = sQc[k].to(Float32)
                        g = mG2L[k].to(Float32)
                        ns = xs * rstd2 * g
                        sQc[k] = ns.to(mH_in.element_type)
            cute.arch.barrier()
            w_qb_off = Int64(layer) * Int64(OUT_QB) * Int64(qb_K)
            w_qb_l_ptr = (mWqb.iterator + w_qb_off).align(16)
            mWqbL = cute.make_tensor(
                w_qb_l_ptr, cute.make_layout((OUT_QB, qb_K), stride=(qb_K, 1)),
            )
            copy_atom_w_qb = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mWqb.element_type,
                num_bits_per_copy=128,
            )
            chunk_stride_qb = const_expr(threads_per_output * vec_size)
            num_full_chunks_qb = const_expr(qb_K // chunk_stride_qb)
            for it in cutlass.range_constexpr(iters_qb):
                tile_idx = bidx + it * num_sms
                if tile_idx < tiles_qb:
                    j = tile_idx * outs_per_cta + group_idx
                    if j < OUT_QB:
                        w_row_ptr_qb = (mWqbL.iterator + j * qb_K).align(16)
                        acc = Float32(0.0)
                        for kc in cutlass.range_constexpr(num_full_chunks_qb):
                            k_base = kc * chunk_stride_qb + lane_in_group * vec_size
                            w_vec = cute.make_rmem_tensor(
                                cute.make_layout((vec_size,), stride=(1,)),
                                mWqb.element_type,
                            )
                            w_slice = cute.make_tensor(
                                (w_row_ptr_qb + k_base).align(16),
                                cute.make_layout((vec_size,), stride=(1,)),
                            )
                            cute.copy(copy_atom_w_qb, w_slice, w_vec)
                            for v in cutlass.range_constexpr(vec_size):
                                a_val = sQc[k_base + v].to(Float32)
                                w_val = w_vec[v].to(Float32)
                                acc = acc + a_val * w_val
                        acc = cute.arch.warp_reduction(
                            acc, op=lambda a, b: a + b, threads_in_group=threads_per_output,
                        )
                        if lane_in_group == 0:
                            mQout[layer, j] = acc.to(mQout.element_type)

            partial_sq3 = Float32(0.0)
            for vb in cutlass.range_constexpr(num_vec_blocks_lkv):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= Lkv:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        kv_val = mQkvaL[Lq + k].to(Float32)
                        sKv[k] = kv_val.to(mH_in.element_type)
                        partial_sq3 = partial_sq3 + kv_val * kv_val
            partial_sq3 = cute.arch.warp_reduction(
                partial_sq3, op=lambda a, b: a + b, threads_in_group=32,
            )
            if lane_id == 0:
                reduce_buf[warp_id] = partial_sq3
            cute.arch.barrier()
            total_sq3 = Float32(0.0)
            for w in cutlass.range_constexpr(warps_per_row):
                total_sq3 = total_sq3 + reduce_buf[w]
            mean_sq3 = total_sq3 / Float32(Lkv)
            rstd3 = cute.math.rsqrt(mean_sq3 + eps, fastmath=True)
            if bidx == 0:
                for vb in cutlass.range_constexpr(num_vec_blocks_lkv):
                    base = (vb * num_threads + tidx) * vec_size
                    if base + vec_size <= Lkv:
                        for v in cutlass.range_constexpr(vec_size):
                            k = base + v
                            xs = sKv[k].to(Float32)
                            g = mG3L[k].to(Float32)
                            ns = xs * rstd3 * g
                            kvc_bf16 = ns.to(mH_in.element_type)
                            mKvcOut[layer, k] = kvc_bf16
                            mKvCacheC[layer, cache_pos, k] = kvc_bf16
            cute.arch.barrier()

            if bidx == 0:
                r_half = const_expr(R // 2)
                if tidx < r_half:
                    x1 = mQkvaL[Lq + Lkv + tidx].to(Float32)
                    x2 = mQkvaL[Lq + Lkv + tidx + r_half].to(Float32)
                    pos_f = mPos[0].to(Float32)
                    freq = pos_f * mInvFreq[tidx]
                    scaling = const_expr(Float32(self.attention_scaling))
                    c = Float32(cute.math.cos(freq, fastmath=True)) * scaling
                    s = Float32(cute.math.sin(freq, fastmath=True)) * scaling
                    o1 = x1 * c - x2 * s
                    o2 = x1 * s + x2 * c
                    o1_bf16 = o1.to(mH_in.element_type)
                    o2_bf16 = o2.to(mH_in.element_type)
                    mKpeOut[layer, tidx] = o1_bf16
                    mKpeOut[layer, tidx + r_half] = o2_bf16
                    mKvCachePe[layer, cache_pos, tidx] = o1_bf16
                    mKvCachePe[layer, cache_pos, tidx + r_half] = o2_bf16
            cute.arch.barrier()

            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 36] = _clock()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected2 = Int32((SYNCS_PER_LAYER * layer + 2) * num_sms)
                done2 = Int32(0)
                while done2 < expected2:
                    done2 = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 2] = _clock()

            N = const_expr(self.N)
            qk_nope_dim = const_expr(self.qk_nope_dim)
            HEAD_TOTAL = const_expr(self.HEAD_TOTAL)
            r_half_g = const_expr(R // 2)
            total_pairs = const_expr(N * r_half_g)
            q_layer_off = Int64(layer) * Int64(OUT_QB)
            q_layer_ptr = (mQout.iterator + q_layer_off).align(16)
            mQL = cute.make_tensor(q_layer_ptr, cute.make_layout((OUT_QB,), stride=(1,)))
            jobs_per_grid = const_expr(num_sms * num_threads)
            jobs_per_thread = const_expr(max(1, (total_pairs + jobs_per_grid - 1) // jobs_per_grid))
            for jt in cutlass.range_constexpr(jobs_per_thread):
                pair_idx = bidx * num_threads + tidx + jt * jobs_per_grid
                if pair_idx < total_pairs:
                    head_n = pair_idx // r_half_g
                    dim_i = pair_idx % r_half_g
                    base_off = head_n * HEAD_TOTAL + qk_nope_dim
                    x1 = mQL[base_off + dim_i].to(Float32)
                    x2 = mQL[base_off + dim_i + r_half_g].to(Float32)
                    pos_f_g = mPos[0].to(Float32)
                    freq_g = pos_f_g * mInvFreq[dim_i]
                    scaling_g = const_expr(Float32(self.attention_scaling))
                    c_g = Float32(cute.math.cos(freq_g, fastmath=True)) * scaling_g
                    s_g = Float32(cute.math.sin(freq_g, fastmath=True)) * scaling_g
                    o1g = x1 * c_g - x2 * s_g
                    o2g = x1 * s_g + x2 * c_g
                    mQL[base_off + dim_i] = o1g.to(mH_in.element_type)
                    mQL[base_off + dim_i + r_half_g] = o2g.to(mH_in.element_type)

            total_outputs_uk = const_expr(N * Lkv)
            tiles_uk = const_expr((total_outputs_uk + outs_per_cta - 1) // outs_per_cta)
            iters_uk = const_expr((tiles_uk + num_sms - 1) // num_sms)
            chunk_uk = const_expr(threads_per_output * vec_size)
            num_chunks_uk = const_expr(max(1, (qk_nope_dim + chunk_uk - 1) // chunk_uk))
            w_uk_layer_off = (
                Int64(layer) * Int64(N) * Int64(Lkv) * Int64(qk_nope_dim)
            )
            w_uk_layer_ptr = (mWuk.iterator + w_uk_layer_off).align(16)
            mWukL = cute.make_tensor(
                w_uk_layer_ptr,
                cute.make_layout((N, Lkv, qk_nope_dim), stride=(Lkv * qk_nope_dim, qk_nope_dim, 1)),
            )
            q_nope_abs_layer_off = Int64(layer) * Int64(N) * Int64(Lkv)
            q_nope_abs_layer_ptr = (mQnopeAbs.iterator + q_nope_abs_layer_off).align(16)
            mQnopeAbsL = cute.make_tensor(
                q_nope_abs_layer_ptr,
                cute.make_layout((N, Lkv), stride=(Lkv, 1)),
            )
            q_layer_off_uk = Int64(layer) * Int64(OUT_QB)
            q_layer_ptr_uk = (mQout.iterator + q_layer_off_uk).align(16)
            mQLuk = cute.make_tensor(q_layer_ptr_uk, cute.make_layout((OUT_QB,), stride=(1,)))
            for it in cutlass.range_constexpr(iters_uk):
                tile_idx = bidx + it * num_sms
                if tile_idx < tiles_uk:
                    flat_j = tile_idx * outs_per_cta + group_idx
                    if flat_j < total_outputs_uk:
                        head_n = flat_j // Lkv
                        lkv_dim = flat_j % Lkv
                        acc_uk = Float32(0.0)
                        for kc in cutlass.range_constexpr(num_chunks_uk):
                            base = kc * chunk_uk + lane_in_group * vec_size
                            if base + vec_size <= qk_nope_dim:
                                for v in cutlass.range_constexpr(vec_size):
                                    k_dim = base + v
                                    qv = mQLuk[head_n * HEAD_TOTAL + k_dim].to(Float32)
                                    wv = mWukL[head_n, lkv_dim, k_dim].to(Float32)
                                    acc_uk = acc_uk + qv * wv
                        acc_uk = cute.arch.warp_reduction(
                            acc_uk, op=lambda a, b: a + b, threads_in_group=threads_per_output,
                        )
                        if lane_in_group == 0:
                            mQnopeAbsL[head_n, lkv_dim] = acc_uk.to(mQnopeAbsL.element_type)

            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 37] = _clock()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected3 = Int32((SYNCS_PER_LAYER * layer + 3) * num_sms)
                done3 = Int32(0)
                while done3 < expected3:
                    done3 = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 3] = _clock()

            items_per_thread_lkv = const_expr(max(1, (Lkv + num_threads - 1) // num_threads))

            TS_MMA = const_expr(self.TILE_SEQ_DEC_NEW)
            KDIM_CE = const_expr(self.Lkv + self.R)
            K_CHUNK = const_expr(self.MMA_TILER_QK[2])
            HDIM_V = const_expr(self.Lkv)
            PV_N_PER_SPLIT = const_expr(self.PV_N)
            num_qk_chunks_ce = const_expr(self.NUM_QK_CHUNKS)
            num_pv_n_splits_ce = const_expr(self.NUM_PV_N_SPLITS)
            num_kvc_chunks_ce = const_expr(self.Lkv // self.MMA_TILER_QK[2])
            cluster_size_ce = const_expr(self.CLUSTER_SIZE)
            cta_tile_m_ce = const_expr(self.CTA_TILE_M)
            heads_per_cta_ce = const_expr(self.HEADS_PER_CTA)
            m_pad_per_head_ce = const_expr(self.M_PAD_PER_HEAD)
            threads_per_head_pack = const_expr(num_threads // heads_per_cta_ce)
            cols_per_thread_pack = const_expr(KDIM_CE // threads_per_head_pack)
            threads_in_epi_ce = const_expr(self.THREADS_IN_EPILOGUE)
            mma_warp_id_ce = const_expr(self.MMA_WARP_ID)
            tma_warp_id_ce = const_expr(self.TMA_WARP_ID)
            num_epi_warps_ce = const_expr(self.NUM_EPI_WARPS)
            ab_stages_ce = const_expr(self.AB_STAGES_QK)
            v_stages_ce = const_expr(self.V_STAGES)
            acc_s_stages_ce = const_expr(self.ACC_S_STAGES)
            acc_o_stages_ce = const_expr(self.ACC_O_STAGES)
            p_stages_ce = const_expr(self.P_STAGES)
            NUM_KV_SPLITS_MMA_C = const_expr(self.NUM_KV_SPLITS_MMA)
            LOG2E_F = const_expr(1.4426950408889634)
            NEG_INF_F = const_expr(float("-inf"))
            HEAD_TOTAL_CE = const_expr(self.HEAD_TOTAL)
            qk_nope_dim_ce = const_expr(self.qk_nope_dim)
            tiles_per_layer_mma = const_expr(S_max // TS_MMA)
            mma_tiler_qk = const_expr(self.MMA_TILER_QK)
            mma_tiler_pv = const_expr(self.MMA_TILER_PV)
            assert S_max % TS_MMA == 0, (
                f"S_max ({S_max}) must be divisible by TS_MMA ({TS_MMA})"
            )

            cluster_id = bidx // Int32(cluster_size_ce)
            cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
            cta_in_cluster_coord_vmnk = cta_layout_vmnk_mma.get_flat_coord(
                cta_rank_in_cluster,
            )
            mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk_mma, mode=[0])
            is_leader_cta = mma_tile_coord_v == Int32(0)
            tma_mcast_mask_kv = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cta_layout_vmnk_mma, cta_in_cluster_coord_vmnk, mcast_mode=1,
            )

            Sk_dec = Int32(cache_pos + Int32(1))
            num_n_tiles_total = (Sk_dec + Int32(TS_MMA - 1)) // Int32(TS_MMA)
            tiles_per_split = (
                num_n_tiles_total + Int32(NUM_KV_SPLITS_MMA_C - 1)
            ) // Int32(NUM_KV_SPLITS_MMA_C)

            cc_off2 = Int64(layer) * Int64(S_max) * Int64(Lkv)
            cp_off2 = Int64(layer) * Int64(S_max) * Int64(R)
            cc_ptr2 = (mKvCacheC.iterator + cc_off2).align(16)
            cp_ptr2 = (mKvCachePe.iterator + cp_off2).align(16)
            mCcL = cute.make_tensor(cc_ptr2, cute.make_layout((S_max, Lkv), stride=(Lkv, 1)))
            mCpL = cute.make_tensor(cp_ptr2, cute.make_layout((S_max, R), stride=(R, 1)))

            q_lp_i = (mQout.iterator + Int64(layer) * Int64(OUT_QB)).align(16)
            qna_lp_i = (mQnopeAbs.iterator + Int64(layer) * Int64(N) * Int64(Lkv)).align(16)
            mQLI = cute.make_tensor(q_lp_i, cute.make_layout((OUT_QB,), stride=(1,)))
            mQnaLI = cute.make_tensor(qna_lp_i, cute.make_layout((N, Lkv), stride=(Lkv, 1)))

            softmax_scale_log2 = softmax_scale * Float32(LOG2E_F)

            warp_idx_local = cute.arch.make_warp_uniform(cute.arch.warp_idx())

            if cluster_id < Int32(NUM_KV_SPLITS_MMA_C):
                n_tile_start_global = cluster_id * tiles_per_split
                n_tile_end_global = cutlass.min(
                    num_n_tiles_total, n_tile_start_global + tiles_per_split,
                )
                num_n_tiles_local = n_tile_end_global - n_tile_start_global


                tma_warp_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
                num_mcast_participants = (
                    cute.size(cta_layout_vmnk_mma, mode=[1])
                    + cute.size(cta_layout_vmnk_mma, mode=[2]) - 1
                )
                mma_consumer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, size=num_mcast_participants,
                )
                acc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
                acc_consumer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    size=threads_in_epi_ce * cluster_size_ce,
                )
                p_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    size=threads_in_epi_ce * cluster_size_ce,
                )
                p_consumer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, size=num_mcast_participants,
                )

                bytes_k_per_chunk = const_expr(
                    TS_MMA * K_CHUNK * 2
                )
                bytes_v_per_tile = const_expr(
                    TS_MMA * PV_N_PER_SPLIT * 2
                )

                ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
                    barrier_storage=ab_mma_bars.iterator,
                    num_stages=ab_stages_ce,
                    producer_group=tma_warp_group,
                    consumer_group=mma_consumer_group,
                    tx_count=bytes_k_per_chunk,
                    cta_layout_vmnk=cta_layout_vmnk_mma,
                ).make_participants()
                v0_producer, v0_consumer = pipeline.PipelineTmaUmma.create(
                    barrier_storage=v0_mma_bars.iterator,
                    num_stages=v_stages_ce,
                    producer_group=tma_warp_group,
                    consumer_group=mma_consumer_group,
                    tx_count=bytes_v_per_tile,
                    cta_layout_vmnk=cta_layout_vmnk_mma,
                ).make_participants()
                v1_producer, v1_consumer = pipeline.PipelineTmaUmma.create(
                    barrier_storage=v1_mma_bars.iterator,
                    num_stages=v_stages_ce,
                    producer_group=tma_warp_group,
                    consumer_group=mma_consumer_group,
                    tx_count=bytes_v_per_tile,
                    cta_layout_vmnk=cta_layout_vmnk_mma,
                ).make_participants()
                acc_s_producer, acc_s_consumer = pipeline.PipelineUmmaAsync.create(
                    barrier_storage=acc_s_mma_bars.iterator,
                    num_stages=acc_s_stages_ce,
                    producer_group=acc_producer_group,
                    consumer_group=acc_consumer_group,
                    cta_layout_vmnk=cta_layout_vmnk_mma,
                ).make_participants()
                acc_o0_producer, acc_o0_consumer = pipeline.PipelineUmmaAsync.create(
                    barrier_storage=acc_o0_mma_bars.iterator,
                    num_stages=acc_o_stages_ce,
                    producer_group=acc_producer_group,
                    consumer_group=acc_consumer_group,
                    cta_layout_vmnk=cta_layout_vmnk_mma,
                ).make_participants()
                acc_o1_producer, acc_o1_consumer = pipeline.PipelineUmmaAsync.create(
                    barrier_storage=acc_o1_mma_bars.iterator,
                    num_stages=acc_o_stages_ce,
                    producer_group=acc_producer_group,
                    consumer_group=acc_consumer_group,
                    cta_layout_vmnk=cta_layout_vmnk_mma,
                ).make_participants()
                p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
                    barrier_storage=p_mma_bars.iterator,
                    num_stages=p_stages_ce,
                    producer_group=p_producer_group,
                    consumer_group=p_consumer_group,
                    cta_layout_vmnk=cta_layout_vmnk_mma,
                ).make_participants()

                epi_sync_barrier = pipeline.NamedBarrier(
                    barrier_id=self.STAGE_I_EPI_BAR_ID,
                    num_threads=threads_in_epi_ce,
                )
                softmax_barrier = pipeline.NamedBarrier(
                    barrier_id=self.STAGE_I_SOFTMAX_BAR_ID,
                    num_threads=threads_in_epi_ce,
                )

                if bidx == 0 and tidx == 0:
                    mStageTs[layer, 32] = _clock()

                if num_n_tiles_local > Int32(0):

                    my_head_local = tidx // Int32(threads_per_head_pack)
                    tid_in_head_group = tidx % Int32(threads_per_head_pack)
                    head_global_qpack = (
                        my_head_local + cta_rank_in_cluster * Int32(heads_per_cta_ce)
                    )
                    for c in cutlass.range_constexpr(cols_per_thread_pack):
                        col = tid_in_head_group + Int32(c * threads_per_head_pack)
                        src = mH_in.element_type(0)
                        if col < Int32(Lkv):
                            src = mQnaLI[head_global_qpack, col].to(
                                mH_in.element_type,
                            )
                        else:
                            pe_idx = col - Int32(Lkv)
                            src = mQLI[
                                head_global_qpack * Int32(HEAD_TOTAL_CE)
                                + Int32(qk_nope_dim_ce) + pe_idx
                            ].to(mH_in.element_type)
                        chunk_idx_q = col // Int32(K_CHUNK)
                        k_in_chunk = col % Int32(K_CHUNK)
                        k_lo = k_in_chunk % Int32(16)
                        k_hi = k_in_chunk // Int32(16)
                        for r in cutlass.range_constexpr(m_pad_per_head_ce):
                            row = my_head_local * Int32(m_pad_per_head_ce) + Int32(r)
                            sQ_mma[((row, k_lo), 0, k_hi, chunk_idx_q)] = src
                    cute.arch.barrier()
                    cute.arch.fence_view_async_shared()

                    if bidx == 0 and tidx == 0:
                        mStageTs[layer, 33] = _clock()

                    thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
                    thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

                    gK_mma = cute.local_tile(
                        tma_tensor_kvc_mma,
                        cute.slice_(mma_tiler_qk, (0, None, None)),
                        (None, None),
                    )
                    gKpe_mma = cute.local_tile(
                        tma_tensor_kpe_mma,
                        cute.slice_(mma_tiler_qk, (0, None, None)),
                        (None, None),
                    )
                    tCgK = thr_mma_qk.partition_B(gK_mma)
                    tCgKpe = thr_mma_qk.partition_B(gKpe_mma)
                    gV0_mma = cute.local_tile(
                        tma_tensor_v0_mma,
                        cute.slice_(mma_tiler_pv, (0, None, None)),
                        (None, None),
                    )
                    gV1_mma = cute.local_tile(
                        tma_tensor_v1_mma,
                        cute.slice_(mma_tiler_pv, (0, None, None)),
                        (None, None),
                    )
                    tCgV0 = thr_mma_pv.partition_B(gV0_mma)
                    tCgV1 = thr_mma_pv.partition_B(gV1_mma)

                    tCrQ = tiled_mma_qk.make_fragment_A(sQ_mma)
                    tCrK = tiled_mma_qk.make_fragment_B(sK_mma)
                    tCrP = tiled_mma_pv.make_fragment_A(sP_mma)
                    tCrV0 = tiled_mma_pv.make_fragment_B(sV0_mma)
                    tCrV1 = tiled_mma_pv.make_fragment_B(sV1_mma)

                    acc_shape_s = tiled_mma_qk.partition_shape_C(mma_tiler_qk[:2])
                    tCtAcc_s_fake = tiled_mma_qk.make_fragment_C(
                        cute.append(acc_shape_s, acc_s_stages_ce),
                    )
                    acc_shape_o = tiled_mma_pv.partition_shape_C(mma_tiler_pv[:2])
                    tCtAcc_o_fake = tiled_mma_pv.make_fragment_C(acc_shape_o)

                    tmem_cols_o_per_split = const_expr(
                        PV_N_PER_SPLIT // cluster_size_ce
                    )
                    tmem_offset_s = const_expr(0)
                    tmem_offset_o0 = const_expr(TS_MMA // cluster_size_ce)
                    tmem_offset_o1 = const_expr(
                        tmem_offset_o0 + tmem_cols_o_per_split
                    )

                    tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_kvc_mma,
                        cta_in_cluster_coord_vmnk[1],
                        cute.make_layout(
                            cute.size(cta_layout_vmnk_mma, mode=[1]),
                        ),
                        cute.group_modes(sK_mma, 0, 3),
                        cute.group_modes(tCgK, 0, 3),
                    )
                    tKpesKpe, tKpegKpe = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_kpe_mma,
                        cta_in_cluster_coord_vmnk[1],
                        cute.make_layout(
                            cute.size(cta_layout_vmnk_mma, mode=[1]),
                        ),
                        cute.group_modes(sK_mma, 0, 3),
                        cute.group_modes(tCgKpe, 0, 3),
                    )
                    tV0sV0, tV0gV0 = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v0_mma,
                        cta_in_cluster_coord_vmnk[1],
                        cute.make_layout(
                            cute.size(cta_layout_vmnk_mma, mode=[1]),
                        ),
                        cute.group_modes(sV0_mma, 0, 3),
                        cute.group_modes(tCgV0, 0, 3),
                    )
                    tV1sV1, tV1gV1 = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v1_mma,
                        cta_in_cluster_coord_vmnk[1],
                        cute.make_layout(
                            cute.size(cta_layout_vmnk_mma, mode=[1]),
                        ),
                        cute.group_modes(sV1_mma, 0, 3),
                        cute.group_modes(tCgV1, 0, 3),
                    )

                    layer_tile_off_mma = Int32(layer) * Int32(tiles_per_layer_mma)

                    if warp_idx_local == Int32(tma_warp_id_ce):
                        cpasync.prefetch_descriptor(tma_atom_kvc_mma)
                        cpasync.prefetch_descriptor(tma_atom_kpe_mma)
                        cpasync.prefetch_descriptor(tma_atom_v0_mma)
                        cpasync.prefetch_descriptor(tma_atom_v1_mma)
                        for n_tile_local in cutlass.range(num_n_tiles_local, unroll=1):
                            n_tile_global = n_tile_start_global + n_tile_local
                            n_tile_cache = layer_tile_off_mma + n_tile_global
                            v0_empty = v0_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v0_mma,
                                tV0gV0[(None, 0, n_tile_cache)],
                                tV0sV0[(None, v0_empty.index)],
                                tma_bar_ptr=v0_empty.barrier,
                                mcast_mask=tma_mcast_mask_kv,
                            )
                            v1_empty = v1_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v1_mma,
                                tV1gV1[(None, 0, n_tile_cache)],
                                tV1sV1[(None, v1_empty.index)],
                                tma_bar_ptr=v1_empty.barrier,
                                mcast_mask=tma_mcast_mask_kv,
                            )
                            for k_chunk_idx in cutlass.range_constexpr(num_qk_chunks_ce):
                                ab_empty = ab_producer.acquire_and_advance()
                                if k_chunk_idx < num_kvc_chunks_ce:
                                    cute.copy(
                                        tma_atom_kvc_mma,
                                        tKgK[(None, n_tile_cache, k_chunk_idx)],
                                        tKsK[(None, ab_empty.index)],
                                        tma_bar_ptr=ab_empty.barrier,
                                        mcast_mask=tma_mcast_mask_kv,
                                    )
                                else:
                                    cute.copy(
                                        tma_atom_kpe_mma,
                                        tKpegKpe[(None, n_tile_cache, 0)],
                                        tKpesKpe[(None, ab_empty.index)],
                                        tma_bar_ptr=ab_empty.barrier,
                                        mcast_mask=tma_mcast_mask_kv,
                                    )
                        ab_producer.tail()
                        v0_producer.tail()
                        v1_producer.tail()

                    elif warp_idx_local == Int32(mma_warp_id_ce):
                        tmem_ptr = tmem_alloc.retrieve_ptr(Float32)
                        tCtAcc_s_base = cute.make_tensor(
                            tmem_ptr, tCtAcc_s_fake.layout,
                        )
                        tCtAcc_o0_base = cute.make_tensor(
                            tmem_ptr + tmem_offset_o0, tCtAcc_o_fake.layout,
                        )
                        tCtAcc_o1_base = cute.make_tensor(
                            tmem_ptr + tmem_offset_o1, tCtAcc_o_fake.layout,
                        )

                        if is_leader_cta:
                            num_k_blocks_qk = cute.size(tCrQ, mode=[2])
                            num_k_blocks_pv = cute.size(tCrP, mode=[2])
                            for n_tile_local in cutlass.range(
                                num_n_tiles_local, unroll=1,
                            ):
                                acc_s_empty = acc_s_producer.acquire_and_advance()
                                tCtAcc_s = tCtAcc_s_base[
                                    (None, None, None, acc_s_empty.index)
                                ]
                                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                                for k_chunk_idx in cutlass.range_constexpr(
                                    num_qk_chunks_ce,
                                ):
                                    ab_full = ab_consumer.wait_and_advance()
                                    for k_block_idx in cutlass.range_constexpr(
                                        num_k_blocks_qk,
                                    ):
                                        cute.gemm(
                                            tiled_mma_qk,
                                            tCtAcc_s,
                                            tCrQ[(
                                                None, None,
                                                k_block_idx, k_chunk_idx,
                                            )],
                                            tCrK[(
                                                None, None,
                                                k_block_idx, ab_full.index,
                                            )],
                                            tCtAcc_s,
                                        )
                                        tiled_mma_qk.set(
                                            tcgen05.Field.ACCUMULATE, True,
                                        )
                                    ab_full.release()
                                acc_s_empty.commit()

                                p_full = p_consumer.wait_and_advance()

                                acc_o0_empty = acc_o0_producer.acquire_and_advance()
                                v0_full = v0_consumer.wait_and_advance()
                                tiled_mma_pv.set(
                                    tcgen05.Field.ACCUMULATE,
                                    cutlass.Boolean(n_tile_local > Int32(0)),
                                )
                                for k_block_idx in cutlass.range_constexpr(
                                    num_k_blocks_pv,
                                ):
                                    cute.gemm(
                                        tiled_mma_pv,
                                        tCtAcc_o0_base,
                                        tCrP[(None, None, k_block_idx, p_full.index)],
                                        tCrV0[(None, None, k_block_idx, 0)],
                                        tCtAcc_o0_base,
                                    )
                                    tiled_mma_pv.set(
                                        tcgen05.Field.ACCUMULATE, True,
                                    )
                                acc_o0_empty.commit()
                                v0_full.release()

                                acc_o1_empty = acc_o1_producer.acquire_and_advance()
                                v1_full = v1_consumer.wait_and_advance()
                                tiled_mma_pv.set(
                                    tcgen05.Field.ACCUMULATE,
                                    cutlass.Boolean(n_tile_local > Int32(0)),
                                )
                                for k_block_idx in cutlass.range_constexpr(
                                    num_k_blocks_pv,
                                ):
                                    cute.gemm(
                                        tiled_mma_pv,
                                        tCtAcc_o1_base,
                                        tCrP[(None, None, k_block_idx, p_full.index)],
                                        tCrV1[(None, None, k_block_idx, 0)],
                                        tCtAcc_o1_base,
                                    )
                                    tiled_mma_pv.set(
                                        tcgen05.Field.ACCUMULATE, True,
                                    )
                                acc_o1_empty.commit()
                                v1_full.release()

                                p_full.release()
                        acc_s_producer.tail()
                        acc_o0_producer.tail()
                        acc_o1_producer.tail()

                    elif warp_idx_local < Int32(num_epi_warps_ce):
                        tmem_ptr = tmem_alloc.retrieve_ptr(Float32)
                        tCtAcc_s_base = cute.make_tensor(
                            tmem_ptr, tCtAcc_s_fake.layout,
                        )
                        tCtAcc_o0_base = cute.make_tensor(
                            tmem_ptr + tmem_offset_o0, tCtAcc_o_fake.layout,
                        )
                        tCtAcc_o1_base = cute.make_tensor(
                            tmem_ptr + tmem_offset_o1, tCtAcc_o_fake.layout,
                        )

                        tStS_acc = tCtAcc_s_base[(None, None), 0, 0, 0]
                        copy_atom_s_t2r = cute.make_copy_atom(
                            tcgen05.Ld32x32bOp(
                                tcgen05.Repetition.x32, tcgen05.Pack.NONE,
                            ),
                            Float32,
                        )
                        tiled_copy_s_t2r = tcgen05.make_tmem_copy(
                            copy_atom_s_t2r, tStS_acc,
                        )
                        thr_copy_s_t2r = tiled_copy_s_t2r.get_slice(tidx)
                        tStS_t2r = thr_copy_s_t2r.partition_S(tStS_acc)
                        cS = cute.make_identity_tensor(mma_tiler_qk[:2])
                        tScS = tiled_mma_qk.get_slice(mma_tile_coord_v).partition_C(cS)[
                            (None, None), 0, 0
                        ]
                        tScS_t2r = thr_copy_s_t2r.partition_D(tScS)
                        tSrS_t2r = cute.make_rmem_tensor(
                            tScS_t2r.shape, Float32,
                        )

                        universal_copy_bits = 128
                        smem_store_atom_p = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            mH_in.element_type,
                            num_bits_per_copy=universal_copy_bits,
                        )
                        smem_store_tiled_p = cute.make_tiled_copy_D(
                            smem_store_atom_p, tiled_copy_s_t2r,
                        )
                        smem_store_thr_p = smem_store_tiled_p.get_slice(tidx)
                        sP_slice = sP_mma[None, None, None, 0]
                        sP_mn = cute.make_tensor(
                            sP_slice.iterator,
                            cute.make_layout(
                                (
                                    (sP_slice.shape[0][0], sP_slice.shape[1]),
                                    (sP_slice.shape[0][1], sP_slice.shape[2]),
                                ),
                                stride=(
                                    (sP_slice.stride[0][0], sP_slice.stride[1]),
                                    (sP_slice.stride[0][1], sP_slice.stride[2]),
                                ),
                            ),
                        )
                        sP_smem_view = smem_store_thr_p.partition_D(sP_mn)

                        tOtO_template = tCtAcc_o0_base[(None, None), 0, 0]
                        corr_tile_size = const_expr(
                            math.gcd(32, PV_N_PER_SPLIT // cluster_size_ce),
                        )
                        copy_atom_o_t2r = cute.make_copy_atom(
                            tcgen05.Ld32x32bOp(
                                tcgen05.Repetition(corr_tile_size),
                                tcgen05.Pack.NONE,
                            ),
                            Float32,
                        )
                        copy_atom_o_r2t = cute.make_copy_atom(
                            tcgen05.St32x32bOp(
                                tcgen05.Repetition(corr_tile_size),
                                tcgen05.Unpack.NONE,
                            ),
                            Float32,
                        )
                        tiled_copy_o_t2r = tcgen05.make_tmem_copy(
                            copy_atom_o_t2r, tOtO_template,
                        )
                        tiled_copy_o_r2t = tcgen05.make_tmem_copy(
                            copy_atom_o_r2t, tOtO_template,
                        )
                        thr_copy_o_t2r = tiled_copy_o_t2r.get_slice(tidx)
                        thr_copy_o_r2t = tiled_copy_o_r2t.get_slice(tidx)
                        tOtO0_t2r = thr_copy_o_t2r.partition_S(
                            tCtAcc_o0_base[(None, None), 0, 0],
                        )
                        tOtO0_r2t = thr_copy_o_r2t.partition_D(
                            tCtAcc_o0_base[(None, None), 0, 0],
                        )
                        tOtO1_t2r = thr_copy_o_t2r.partition_S(
                            tCtAcc_o1_base[(None, None), 0, 0],
                        )
                        tOtO1_r2t = thr_copy_o_r2t.partition_D(
                            tCtAcc_o1_base[(None, None), 0, 0],
                        )

                        cO_split = cute.make_identity_tensor(
                            (cta_tile_m_ce, PV_N_PER_SPLIT),
                        )
                        tOicO_t2r = thr_copy_o_t2r.partition_D(
                            cO_split,
                        )
                        tOrO_t2r = cute.make_rmem_tensor(
                            tOicO_t2r.shape, Float32,
                        )

                        m_i = cute.make_rmem_tensor(1, Float32)
                        l_i = cute.make_rmem_tensor(1, Float32)
                        m_i[0] = Float32(NEG_INF_F)
                        l_i[0] = Float32(0.0)
                        my_row = tidx % Int32(cta_tile_m_ce)
                        my_col_half = tidx // Int32(cta_tile_m_ce)

                        for n_tile_local in cutlass.range(num_n_tiles_local, unroll=1):
                            n_tile_global = n_tile_start_global + n_tile_local

                            acc_s_full = acc_s_consumer.wait_and_advance()
                            cute.copy(tiled_copy_s_t2r, tStS_t2r, tSrS_t2r)
                            cute.arch.fence_view_async_tmem_load()
                            acc_s_full.release()

                            n_tile_seq_base = n_tile_global * Int32(TS_MMA)
                            for i in cutlass.range_constexpr(cute.size(tSrS_t2r)):
                                n_coord = Int32(tScS_t2r[i][1])
                                seq_pos = n_tile_seq_base + n_coord
                                if seq_pos >= Sk_dec:
                                    tSrS_t2r[i] = Float32(NEG_INF_F)

                            is_first = n_tile_local == Int32(0)
                            s_ssa = tSrS_t2r.load()
                            row_max_local = fmax_reduce(s_ssa, arch=100)
                            softmax_barrier.arrive_and_wait()
                            sRowMax_mma[my_row, my_col_half] = row_max_local
                            softmax_barrier.arrive_and_wait()
                            row_max_0 = sRowMax_mma[my_row, Int32(0)]
                            row_max_1 = sRowMax_mma[my_row, Int32(1)]
                            row_max_full = fmax(row_max_0, row_max_1)
                            row_max_safe = (
                                row_max_full
                                if row_max_full != Float32(NEG_INF_F)
                                else Float32(0.0)
                            )
                            acc_scale = Float32(1.0)
                            if not is_first:
                                row_max_old = m_i[0]
                                acc_scale_log = (
                                    row_max_old - row_max_safe
                                ) * softmax_scale_log2
                                acc_scale = cute.math.exp2(
                                    acc_scale_log, fastmath=True,
                                )
                            m_i[0] = row_max_full
                            row_max_scaled = row_max_safe * softmax_scale_log2
                            for i in cutlass.range_constexpr(cute.size(tSrS_t2r)):
                                tSrS_t2r[i] = cute.math.exp2(
                                    tSrS_t2r[i] * softmax_scale_log2
                                    - row_max_scaled,
                                    fastmath=True,
                                )
                            p_ssa = tSrS_t2r.load()
                            row_sum_local = fadd_reduce(p_ssa, arch=100)
                            softmax_barrier.arrive_and_wait()
                            sRowMax_mma[my_row, my_col_half] = row_sum_local
                            softmax_barrier.arrive_and_wait()
                            row_sum_0 = sRowMax_mma[my_row, Int32(0)]
                            row_sum_1 = sRowMax_mma[my_row, Int32(1)]
                            row_sum_full = row_sum_0 + row_sum_1
                            if is_first:
                                l_i[0] = row_sum_full
                            else:
                                l_i[0] = l_i[0] * acc_scale + row_sum_full

                            if not is_first:
                                acc_o0_full = acc_o0_consumer.wait_and_advance()
                                cute.copy(tiled_copy_o_t2r, tOtO0_t2r, tOrO_t2r)
                                cute.arch.fence_view_async_tmem_load()
                                o0_ssa = tOrO_t2r.load()
                                tOrO_t2r.store(o0_ssa * acc_scale)
                                cute.copy(tiled_copy_o_r2t, tOrO_t2r, tOtO0_r2t)
                                cute.arch.fence_view_async_tmem_store()
                                acc_o0_full.release()
                                acc_o1_full = acc_o1_consumer.wait_and_advance()
                                cute.copy(tiled_copy_o_t2r, tOtO1_t2r, tOrO_t2r)
                                cute.arch.fence_view_async_tmem_load()
                                o1_ssa = tOrO_t2r.load()
                                tOrO_t2r.store(o1_ssa * acc_scale)
                                cute.copy(tiled_copy_o_r2t, tOrO_t2r, tOtO1_r2t)
                                cute.arch.fence_view_async_tmem_store()
                                acc_o1_full.release()

                            p_empty = p_producer.acquire_and_advance()
                            tPrP = cute.make_rmem_tensor(
                                tSrS_t2r.shape, mH_in.element_type,
                            )
                            tPrP.store(tSrS_t2r.load().to(mH_in.element_type))
                            rP_smem_view = smem_store_thr_p.retile(tPrP)
                            cute.copy(smem_store_tiled_p, rP_smem_view, sP_smem_view)
                            cute.arch.fence_view_async_shared()
                            p_empty.commit()


                        acc_o0_full = acc_o0_consumer.wait_and_advance()
                        cute.copy(tiled_copy_o_t2r, tOtO0_t2r, tOrO_t2r)
                        cute.arch.fence_view_async_tmem_load()
                        acc_o0_full.release()
                        for i in cutlass.range_constexpr(cute.size(tOrO_t2r)):
                            row_i = Int32(tOicO_t2r[i][0])
                            col_in_split = Int32(tOicO_t2r[i][1])
                            head_local_i = row_i // Int32(m_pad_per_head_ce)
                            replica_i = row_i % Int32(m_pad_per_head_ce)
                            head_global_i = (
                                head_local_i
                                + cta_rank_in_cluster * Int32(heads_per_cta_ce)
                            )
                            col_global_i = (
                                Int32(0) * Int32(PV_N_PER_SPLIT) + col_in_split
                            )
                            if replica_i == Int32(0):
                                mOPartial[
                                    head_global_i, cluster_id, col_global_i
                                ] = tOrO_t2r[i]

                        acc_o1_full = acc_o1_consumer.wait_and_advance()
                        cute.copy(tiled_copy_o_t2r, tOtO1_t2r, tOrO_t2r)
                        cute.arch.fence_view_async_tmem_load()
                        acc_o1_full.release()
                        for i in cutlass.range_constexpr(cute.size(tOrO_t2r)):
                            row_i = Int32(tOicO_t2r[i][0])
                            col_in_split = Int32(tOicO_t2r[i][1])
                            head_local_i = row_i // Int32(m_pad_per_head_ce)
                            replica_i = row_i % Int32(m_pad_per_head_ce)
                            head_global_i = (
                                head_local_i
                                + cta_rank_in_cluster * Int32(heads_per_cta_ce)
                            )
                            col_global_i = (
                                Int32(1) * Int32(PV_N_PER_SPLIT) + col_in_split
                            )
                            if replica_i == Int32(0):
                                mOPartial[
                                    head_global_i, cluster_id, col_global_i
                                ] = tOrO_t2r[i]

                        head_local_ml = my_row // Int32(m_pad_per_head_ce)
                        replica_ml = my_row % Int32(m_pad_per_head_ce)
                        head_global_ml = (
                            head_local_ml
                            + cta_rank_in_cluster * Int32(heads_per_cta_ce)
                        )
                        if my_col_half == Int32(0) and replica_ml == Int32(0):
                            mMPartial[head_global_ml, cluster_id] = m_i[0]
                            mLPartial[head_global_ml, cluster_id] = l_i[0]

                        epi_sync_barrier.arrive_and_wait()

                else:
                    if warp_idx_local < Int32(num_epi_warps_ce):
                        if tidx < Int32(heads_per_cta_ce):
                            head_local_si = tidx
                            head_global_si = (
                                head_local_si
                                + cta_rank_in_cluster * Int32(heads_per_cta_ce)
                            )
                            mMPartial[head_global_si, cluster_id] = Float32(NEG_INF_F)
                            mLPartial[head_global_si, cluster_id] = Float32(0.0)

            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 34] = _clock()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected4 = Int32((SYNCS_PER_LAYER * layer + 4) * num_sms)
                done4 = Int32(0)
                while done4 < expected4:
                    done4 = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 4] = _clock()

            if bidx < Int32(N):
                head_n_i_c = bidx
                out_lp_c = (mAttnOut.iterator + Int64(layer) * Int64(N) * Int64(Lkv)).align(16)
                mOutLI_c = cute.make_tensor(out_lp_c, cute.make_layout((N, Lkv), stride=(Lkv, 1)))

                m_final = Float32(-1.0e30)
                for s in cutlass.range_constexpr(NUM_KV_SPLITS):
                    m_s = mMPartial[head_n_i_c, s]
                    m_final = (
                        cute.arch.fmax(m_final, m_s)
                        if hasattr(cute.arch, "fmax")
                        else (m_final if m_final > m_s else m_s)
                    )

                l_final = Float32(0.0)
                o_accum = cute.make_fragment(items_per_thread_lkv, Float32)
                for i in cutlass.range_constexpr(items_per_thread_lkv):
                    o_accum[i] = Float32(0.0)
                for s in cutlass.range_constexpr(NUM_KV_SPLITS):
                    m_s = mMPartial[head_n_i_c, s]
                    l_s = mLPartial[head_n_i_c, s]
                    scale_s = cute.math.exp(m_s - m_final, fastmath=True)
                    l_final = l_final + scale_s * l_s
                    for i in cutlass.range_constexpr(items_per_thread_lkv):
                        idx_v = tidx * items_per_thread_lkv + i
                        if idx_v < Lkv:
                            o_accum[i] = o_accum[i] + scale_s * mOPartial[head_n_i_c, s, idx_v]

                inv_l_final = Float32(1.0) / l_final
                for i in cutlass.range_constexpr(items_per_thread_lkv):
                    idx_v = tidx * items_per_thread_lkv + i
                    if idx_v < Lkv:
                        mOutLI_c[head_n_i_c, idx_v] = (o_accum[i] * inv_l_final).to(mOutLI_c.element_type)

            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 38] = _clock()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected5 = Int32((SYNCS_PER_LAYER * layer + 5) * num_sms)
                done5 = Int32(0)
                while done5 < expected5:
                    done5 = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 5] = _clock()

            total_outputs_uv = const_expr(N * v_head_dim)
            tiles_uv = const_expr((total_outputs_uv + outs_per_cta - 1) // outs_per_cta)
            iters_uv = const_expr((tiles_uv + num_sms - 1) // num_sms)
            chunk_uv = const_expr(threads_per_output * vec_size)
            num_chunks_uv = const_expr(max(1, (Lkv + chunk_uv - 1) // chunk_uv))
            w_uv_layer_off = (
                Int64(layer) * Int64(N) * Int64(v_head_dim) * Int64(Lkv)
            )
            w_uv_layer_ptr = (mWuv.iterator + w_uv_layer_off).align(16)
            mWuvL = cute.make_tensor(
                w_uv_layer_ptr,
                cute.make_layout(
                    (N, v_head_dim, Lkv),
                    stride=(v_head_dim * Lkv, Lkv, 1),
                ),
            )
            attn_layer_ptr = (mAttnOut.iterator + Int64(layer) * Int64(N) * Int64(Lkv)).align(16)
            mAttnL = cute.make_tensor(attn_layer_ptr, cute.make_layout((N, Lkv), stride=(Lkv, 1)))
            o_per_head_layer_ptr = (
                mOperHead.iterator + Int64(layer) * Int64(N) * Int64(v_head_dim)
            ).align(16)
            mOperHeadL = cute.make_tensor(
                o_per_head_layer_ptr,
                cute.make_layout((N, v_head_dim), stride=(v_head_dim, 1)),
            )
            for it in cutlass.range_constexpr(iters_uv):
                tile_idx = bidx + it * num_sms
                if tile_idx < tiles_uv:
                    flat_j = tile_idx * outs_per_cta + group_idx
                    if flat_j < total_outputs_uv:
                        head_n_uv = flat_j // v_head_dim
                        v_dim_uv = flat_j % v_head_dim
                        acc_uv = Float32(0.0)
                        for kc in cutlass.range_constexpr(num_chunks_uv):
                            base = kc * chunk_uv + lane_in_group * vec_size
                            if base + vec_size <= Lkv:
                                for v in cutlass.range_constexpr(vec_size):
                                    l_dim = base + v
                                    attn_v = mAttnL[head_n_uv, l_dim].to(Float32)
                                    wv_v = mWuvL[head_n_uv, v_dim_uv, l_dim].to(Float32)
                                    acc_uv = acc_uv + attn_v * wv_v
                        acc_uv = cute.arch.warp_reduction(
                            acc_uv, op=lambda a, b: a + b, threads_in_group=threads_per_output,
                        )
                        if lane_in_group == 0:
                            mOperHeadL[head_n_uv, v_dim_uv] = acc_uv.to(mOperHeadL.element_type)

            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 39] = _clock()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected6 = Int32((SYNCS_PER_LAYER * layer + 6) * num_sms)
                done6 = Int32(0)
                while done6 < expected6:
                    done6 = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 6] = _clock()

            tiles_o = const_expr((H + outs_per_cta - 1) // outs_per_cta)
            iters_o = const_expr((tiles_o + num_sms - 1) // num_sms)
            chunk_stride_o = const_expr(threads_per_output * vec_size)
            num_full_chunks_o = const_expr(O_CONCAT // chunk_stride_o)
            w_o_layer_off = Int64(layer) * Int64(H) * Int64(O_CONCAT)
            w_o_layer_ptr = (mWo.iterator + w_o_layer_off).align(16)
            mWoL = cute.make_tensor(
                w_o_layer_ptr, cute.make_layout((H, O_CONCAT), stride=(O_CONCAT, 1)),
            )
            copy_atom_w_o = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mWo.element_type,
                num_bits_per_copy=128,
            )
            mOperHeadL_flat = cute.make_tensor(
                o_per_head_layer_ptr, cute.make_layout((O_CONCAT,), stride=(1,)),
            )
            attn_proj_layer_ptr = (mAttnProj.iterator + Int64(layer) * Int64(H)).align(16)
            mAttnProjL = cute.make_tensor(
                attn_proj_layer_ptr, cute.make_layout((H,), stride=(1,)),
            )
            attn_proj_symm_local_base = (
                mAttnProjSymmLocal.iterator.toint() + Int64(layer) * Int64(H) * Int64(2)
            )
            attn_proj_symm_mc_base = (
                mAttnProjSymmMc.iterator.toint() + Int64(layer) * Int64(H) * Int64(2)
            )
            tp_sync_local_addr = (
                mTpSyncLocal.iterator.toint()
                + Int64(layer) * Int64(num_sms) * Int64(4)
                + Int64(bidx) * Int64(4)
            )
            tp_sync_mc_addr = (
                mTpSyncMc.iterator.toint()
                + Int64(layer) * Int64(num_sms) * Int64(4)
                + Int64(bidx) * Int64(4)
            )
            attn_proj_red_base_i64 = (
                mAttnProjReduced.iterator.toint()
                + Int64(layer) * Int64(H) * Int64(2)
            )
            warp_idx_k = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if cluster_id < Int32(self.STAGE_K_NUM_ACTIVE_CLUSTERS):
              SK_CTA_TILE_M     = const_expr(self.STAGE_K_CTA_TILE_M)
              SK_CLUSTER_TILE_M = const_expr(self.STAGE_K_CLUSTER_TILE_M)
              SK_K_CHUNK        = const_expr(self.STAGE_K_K_CHUNK)
              SK_NUM_K_TILES    = const_expr(self.STAGE_K_NUM_K_TILES_MMA)
              SK_AB_STAGES      = const_expr(self.STAGE_K_AB_STAGES)
              SK_MMA_N          = const_expr(self.STAGE_K_MMA_N)
              sk_mma_tiler_c    = const_expr(self.STAGE_K_MMA_TILER)
              bytes_sk_per_cta = const_expr(SK_CTA_TILE_M * SK_K_CHUNK * 2)
              bytes_sk_ab_per_stage = const_expr(
                  cluster_size_ce * bytes_sk_per_cta
              )
              SK_TMEM_COLS_PER_ACC = const_expr(SK_MMA_N // cluster_size_ce)

              sWo = cute.make_tensor(
                  cute.recast_ptr(
                      sQ_mma.iterator,
                      sWo_layout_staged.inner,
                      dtype=self.dtype,
                  ),
                  sWo_layout_staged.outer,
              )
              sOperHead = cute.make_tensor(
                  cute.recast_ptr(
                      sV0_mma.iterator,
                      sOperHead_layout_staged.inner,
                      dtype=self.dtype,
                  ),
                  sOperHead_layout_staged.outer,
              )

              tma_warp_group_k = pipeline.CooperativeGroup(
                  pipeline.Agent.Thread,
              )
              mma_consumer_group_k = pipeline.CooperativeGroup(
                  pipeline.Agent.Thread, size=1,
              )
              acc_producer_group_k = pipeline.CooperativeGroup(
                  pipeline.Agent.Thread,
              )
              acc_consumer_group_k = pipeline.CooperativeGroup(
                  pipeline.Agent.Thread,
                  size=threads_in_epi_ce * cluster_size_ce,
              )
              sk_ab_producer, sk_ab_consumer = (
                  pipeline.PipelineTmaUmma.create(
                      barrier_storage=stage_k_ab_mma_bars.iterator,
                      num_stages=SK_AB_STAGES,
                      producer_group=tma_warp_group_k,
                      consumer_group=mma_consumer_group_k,
                      tx_count=bytes_sk_ab_per_stage,
                      cta_layout_vmnk=cta_layout_vmnk_mma,
                  ).make_participants()
              )
              sk_acc_producer, sk_acc_consumer = (
                  pipeline.PipelineUmmaAsync.create(
                      barrier_storage=stage_k_acc_mma_bars.iterator,
                      num_stages=1,
                      producer_group=acc_producer_group_k,
                      consumer_group=acc_consumer_group_k,
                      cta_layout_vmnk=cta_layout_vmnk_mma,
                  ).make_participants()
              )

              gWo_mma = cute.local_tile(
                  tma_tensor_wo_mma,
                  cute.slice_(sk_mma_tiler_c, (None, 0, None)),
                  (None, None, Int32(layer)),
              )
              thr_mma_sk_slice = tiled_mma_stage_k.get_slice(mma_tile_coord_v)
              tCgWo = thr_mma_sk_slice.partition_A(gWo_mma)
              tCrWo = tiled_mma_stage_k.make_fragment_A(sWo)
              tCrOperHead = tiled_mma_stage_k.make_fragment_B(sOperHead)
              tWoS, tWoG = cute.nvgpu.cpasync.tma_partition(
                  tma_atom_wo_mma,
                  cta_in_cluster_coord_vmnk[2],
                  cute.make_layout(
                      cute.size(cta_layout_vmnk_mma, mode=[2]),
                  ),
                  cute.group_modes(sWo, 0, 3),
                  cute.group_modes(tCgWo, 0, 3),
              )

              acc_shape_sk = tiled_mma_stage_k.partition_shape_C(
                  sk_mma_tiler_c[:2],
              )
              tCtAcc_sk_fake = tiled_mma_stage_k.make_fragment_C(acc_shape_sk)
              SK_TMEM_OFF_ACC = const_expr(0)

              if warp_idx_k == Int32(tma_warp_id_ce):
                  cpasync.prefetch_descriptor(tma_atom_wo_mma)
                  lane_k = tidx % Int32(32)
                  sOperHead_u32 = cute.make_tensor(
                      cute.recast_ptr(
                          sV0_mma.iterator,
                          sOperHead_layout_staged.inner,
                          dtype=cutlass.Uint32,
                      ),
                      cute.recast_layout(
                          32, 16, sOperHead_layout_staged.outer,
                      ),
                  )
                  mOperHeadL_u32 = cute.recast_tensor(
                      mOperHeadL_flat, cutlass.Uint32,
                  )
                  lane_K_atom_u32 = lane_k & Int32(7)
                  lane_N_off = (lane_k >> Int32(3)) & Int32(3)
                  SK_NN_QUAD_ITERS = const_expr(SK_MMA_N // 8)
                  for kt in cutlass.range(SK_NUM_K_TILES, unroll=1):
                      ab_empty = sk_ab_producer.acquire_and_advance()
                      k_base_k = kt * Int32(SK_K_CHUNK)
                      val_k0 = mOperHeadL_u32[
                          (k_base_k >> Int32(1)) + lane_K_atom_u32
                      ]
                      val_k1 = mOperHeadL_u32[
                          (k_base_k >> Int32(1)) + Int32(8) + lane_K_atom_u32
                      ]
                      for nn_quad in cutlass.range_constexpr(SK_NN_QUAD_ITERS):
                          nn = Int32(nn_quad) * Int32(4) + lane_N_off
                          sOperHead_u32[
                              ((nn, lane_K_atom_u32), 0, 0, ab_empty.index)
                          ] = val_k0
                          sOperHead_u32[
                              ((nn, lane_K_atom_u32), 0, 1, ab_empty.index)
                          ] = val_k1
                      cute.arch.fence_view_async_shared()
                      cute.copy(
                          tma_atom_wo_mma,
                          tWoG[(None, cluster_id, kt)],
                          tWoS[(None, ab_empty.index)],
                          tma_bar_ptr=ab_empty.barrier,
                          mcast_mask=tma_mcast_mask_kv,
                      )
                  sk_ab_producer.tail()

              elif warp_idx_k == Int32(mma_warp_id_ce):
                  if is_leader_cta:
                      tmem_ptr_sk = tmem_alloc.retrieve_ptr(Float32)
                      tCtAcc_sk = cute.make_tensor(
                          tmem_ptr_sk + SK_TMEM_OFF_ACC,
                          tCtAcc_sk_fake.layout,
                      )
                      num_k_blocks_sk = cute.size(tCrWo, mode=[2])
                      acc_sk_empty = sk_acc_producer.acquire_and_advance()

                      ab_full = sk_ab_consumer.wait_and_advance()
                      tiled_mma_stage_k.set(
                          tcgen05.Field.ACCUMULATE, False,
                      )
                      for kb in cutlass.range_constexpr(num_k_blocks_sk):
                          cute.gemm(
                              tiled_mma_stage_k,
                              tCtAcc_sk,
                              tCrWo[(None, None, kb, ab_full.index)],
                              tCrOperHead[(None, None, kb, ab_full.index)],
                              tCtAcc_sk,
                          )
                          tiled_mma_stage_k.set(
                              tcgen05.Field.ACCUMULATE, True,
                          )
                      ab_full.release()

                      for kt in cutlass.range(
                          SK_NUM_K_TILES - 1, unroll=1,
                      ):
                          ab_full2 = sk_ab_consumer.wait_and_advance()
                          for kb in cutlass.range_constexpr(num_k_blocks_sk):
                              cute.gemm(
                                  tiled_mma_stage_k,
                                  tCtAcc_sk,
                                  tCrWo[(None, None, kb, ab_full2.index)],
                                  tCrOperHead[(None, None, kb, ab_full2.index)],
                                  tCtAcc_sk,
                              )
                          ab_full2.release()
                      acc_sk_empty.commit()
                  sk_acc_producer.tail()

              elif warp_idx_k < Int32(num_epi_warps_ce):
                  tmem_ptr_sk = tmem_alloc.retrieve_ptr(Float32)
                  tCtAcc_sk = cute.make_tensor(
                      tmem_ptr_sk + SK_TMEM_OFF_ACC,
                      tCtAcc_sk_fake.layout,
                  )
                  tStAcc_sk = tCtAcc_sk[(None, None), 0, 0]
                  copy_atom_sk_t2r = cute.make_copy_atom(
                      tcgen05.Ld32x32bOp(
                          tcgen05.Repetition.x16, tcgen05.Pack.NONE,
                      ),
                      Float32,
                  )
                  tiled_copy_sk_t2r = tcgen05.make_tmem_copy(
                      copy_atom_sk_t2r, tStAcc_sk,
                  )
                  thr_copy_sk_t2r = tiled_copy_sk_t2r.get_slice(tidx)
                  tStAcc_t2r = thr_copy_sk_t2r.partition_S(tStAcc_sk)
                  cSk = cute.make_identity_tensor(
                      (SK_CTA_TILE_M, SK_MMA_N),
                  )
                  tSkcSk_t2r = thr_copy_sk_t2r.partition_D(cSk)
                  tSkrAcc_t2r = cute.make_rmem_tensor(
                      tSkcSk_t2r.shape, Float32,
                  )

                  acc_sk_full = sk_acc_consumer.wait_and_advance()
                  cute.copy(tiled_copy_sk_t2r, tStAcc_t2r, tSkrAcc_t2r)
                  cute.arch.fence_view_async_tmem_load()
                  acc_sk_full.release()

                  sStageK = cute.make_tensor(
                      cute.recast_ptr(
                          sV0_mma.iterator, dtype=mAttnProjL.element_type,
                      ),
                      cute.make_layout((SK_CTA_TILE_M,), stride=(1,)),
                  )
                  for i in cutlass.range_constexpr(cute.size(tSkrAcc_t2r)):
                      coord_i = tSkcSk_t2r[i]
                      if coord_i[1] == Int32(0):
                          sStageK[coord_i[0]] = tSkrAcc_t2r[i].to(
                              mAttnProjL.element_type,
                          )
                  sk_epi_bar = pipeline.NamedBarrier(
                      barrier_id=self.STAGE_I_EPI_BAR_ID,
                      num_threads=threads_in_epi_ce,
                  )
                  sk_epi_bar.arrive_and_wait()
                  sStageK_u32 = cute.make_tensor(
                      cute.recast_ptr(sV0_mma.iterator, dtype=cutlass.Int32),
                      cute.make_layout((SK_CTA_TILE_M // 2,), stride=(1,)),
                  )
                  if tidx < Int32(SK_CTA_TILE_M // 8):
                      h_base = (
                          cluster_id * Int32(SK_CLUSTER_TILE_M)
                          + cta_rank_in_cluster * Int32(SK_CTA_TILE_M)
                          + tidx * Int32(8)
                      )
                      if h_base + Int32(8) <= Int32(H):
                          u_off = tidx * Int32(4)
                          mc_addr_i64 = (
                              attn_proj_symm_mc_base + Int64(h_base) * Int64(2)
                          )
                          llvm.inline_asm(
                              None,
                              [
                                  mc_addr_i64.ir_value(),
                                  sStageK_u32[u_off + Int32(0)].ir_value(),
                                  sStageK_u32[u_off + Int32(1)].ir_value(),
                                  sStageK_u32[u_off + Int32(2)].ir_value(),
                                  sStageK_u32[u_off + Int32(3)].ir_value(),
                              ],
                              "multimem.red.relaxed.sys.global.add.v4.bf16x2 [$0], {$1, $2, $3, $4};",
                              "l,r,r,r,r",
                              has_side_effects=True,
                              is_align_stack=False,
                          )

            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 40] = _clock()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 7] = _clock()

            self._tp_signal_hoist(tidx, bidx, warp_id, lane_id, tp_sync_mc_addr)

            self._tp_spin_late(
                tidx, warp_id, lane_id, tp_sync_local_addr, cache_pos,
                mStageTs=mStageTs, layer=layer, bidx=bidx,
            )

            if bidx == 0 and tidx == 0:
                mStageTs[layer, 26] = _clock()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 28] = _clock()
            attn_proj_symm_layer_ptr = (
                mAttnProjSymmLocal.iterator + Int64(layer) * Int64(H)
            ).align(16)
            mAttnProjSymmL = cute.make_tensor(
                attn_proj_symm_layer_ptr,
                cute.make_layout((H,), stride=(1,)),
            )
            for vb in cutlass.range_constexpr(num_vec_blocks_h):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= H:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        old = sCarry[k].to(Float32)
                        add = mAttnProjSymmL[k].to(Float32)
                        sCarry[k] = (old + add).to(mH_in.element_type)
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 29] = _clock()
            cute.arch.barrier()

            if bidx == 0 and tidx == 0:
                mStageTs[layer, 27] = _clock()


            gamma4_layer_off = Int64(layer) * Int64(H)
            g4_ptr_l = (mGamma4.iterator + gamma4_layer_off).align(16)
            mG4L = cute.make_tensor(g4_ptr_l, cute.make_layout((H,), stride=(1,)))

            partial_sq_m = Float32(0.0)
            for vb in cutlass.range_constexpr(num_vec_blocks_h):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= H:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        x = sCarry[k].to(Float32)
                        partial_sq_m = partial_sq_m + x * x
            partial_sq_m = cute.arch.warp_reduction(
                partial_sq_m, op=lambda a, b: a + b, threads_in_group=32,
            )
            if lane_id == 0:
                reduce_buf[warp_id] = partial_sq_m
            cute.arch.barrier()
            total_sq_m = Float32(0.0)
            for w in cutlass.range_constexpr(warps_per_row):
                total_sq_m = total_sq_m + reduce_buf[w]
            mean_sq_m = total_sq_m / Float32(H)
            rstd_m = cute.math.rsqrt(mean_sq_m + eps, fastmath=True)
            for vb in cutlass.range_constexpr(num_vec_blocks_h):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= H:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        x = sCarry[k].to(Float32)
                        g = mG4L[k].to(Float32)
                        sNorm1[k] = (x * rstd_m * g).to(mH_in.element_type)
            cute.arch.barrier()

            w_router_layer_off = Int64(layer) * Int64(E) * Int64(H)
            w_router_ptr_l = (mWgateRouter.iterator + w_router_layer_off).align(16)
            mWrouterL = cute.make_tensor(
                w_router_ptr_l, cute.make_layout((E, H), stride=(H, 1)),
            )
            logits_layer_off = Int64(layer) * Int64(E)
            logits_ptr_l = (mRouterLogits.iterator + logits_layer_off).align(16)
            mLogitsL = cute.make_tensor(logits_ptr_l, cute.make_layout((E,), stride=(1,)))

            tiles_router = const_expr((E + outs_per_cta - 1) // outs_per_cta)
            iters_router = const_expr((tiles_router + num_sms - 1) // num_sms)
            chunk_r = const_expr(threads_per_output * vec_size)
            num_chunks_router = const_expr(max(1, (H + chunk_r - 1) // chunk_r))
            for it in cutlass.range_constexpr(iters_router):
                tile_idx = bidx + it * num_sms
                if tile_idx < tiles_router:
                    e_out = tile_idx * outs_per_cta + group_idx
                    if e_out < E:
                        acc_r = Float32(0.0)
                        for kc in cutlass.range_constexpr(num_chunks_router):
                            base = kc * chunk_r + lane_in_group * vec_size
                            if base + vec_size <= H:
                                for v in cutlass.range_constexpr(vec_size):
                                    k = base + v
                                    acc_r = acc_r + sNorm1[k].to(Float32) * mWrouterL[e_out, k].to(Float32)
                        acc_r = cute.arch.warp_reduction(
                            acc_r, op=lambda a, b: a + b, threads_in_group=threads_per_output,
                        )
                        if lane_in_group == 0:
                            mLogitsL[e_out] = acc_r.to(mLogitsL.element_type)

            cute.arch.barrier()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected8 = Int32((SYNCS_PER_LAYER * layer + 8) * num_sms)
                done8 = Int32(0)
                while done8 < expected8:
                    done8 = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 8] = _clock()
            attn_proj_symm_zero_ptr = (
                mAttnProjSymmLocal.iterator + Int64(layer) * Int64(H)
            ).align(16)
            mAttnProjSymmZero = cute.make_tensor(
                attn_proj_symm_zero_ptr,
                cute.make_layout((H,), stride=(1,)),
            )
            for vb in cutlass.range_constexpr(num_vec_blocks_h):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= H:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        if bidx == 1:
                            mAttnProjSymmZero[k] = Float32(0.0).to(mH_in.element_type)

            topk_w_layer_off = Int64(layer) * Int64(K_topk)
            topk_ids_layer_off = Int64(layer) * Int64(K_topk)
            topk_w_ptr_l = (mTopkW.iterator + topk_w_layer_off).align(4)
            topk_ids_ptr_l = (mTopkIds.iterator + topk_ids_layer_off).align(4)
            mTopkWL = cute.make_tensor(topk_w_ptr_l, cute.make_layout((K_topk,), stride=(1,)))
            mTopkIdsL = cute.make_tensor(topk_ids_ptr_l, cute.make_layout((K_topk,), stride=(1,)))

            if bidx == 0:
                NEG_INF = Float32(-3.4e38)
                local_max = NEG_INF
                for e_off in cutlass.range_constexpr((E + num_threads - 1) // num_threads):
                    e = e_off * num_threads + tidx
                    if e < E:
                        v = mLogitsL[e].to(Float32)
                        if v > local_max:
                            local_max = v
                local_max = cute.arch.warp_reduction(
                    local_max, op=lambda a, b: a if a > b else b, threads_in_group=32,
                )
                if lane_id == 0:
                    reduce_buf[warp_id] = local_max
                cute.arch.barrier()
                if warp_id == 0 and lane_id < warps_per_row:
                    local_max = reduce_buf[lane_id]
                else:
                    local_max = NEG_INF
                local_max = cute.arch.warp_reduction(
                    local_max, op=lambda a, b: a if a > b else b, threads_in_group=warps_per_row,
                )
                if warp_id == 0 and lane_id == 0:
                    reduce_buf[0] = local_max
                cute.arch.barrier()
                local_max = reduce_buf[0]

                local_sum = Float32(0.0)
                for e_off in cutlass.range_constexpr((E + num_threads - 1) // num_threads):
                    e = e_off * num_threads + tidx
                    if e < E:
                        v = mLogitsL[e].to(Float32)
                        ev_raw = cute.math.exp(v - local_max, fastmath=True)
                        ev = Float32(0.0) + ev_raw
                        mLogitsL[e] = ev.to(mLogitsL.element_type)
                        local_sum = local_sum + ev
                local_sum = cute.arch.warp_reduction(
                    local_sum, op=lambda a, b: a + b, threads_in_group=32,
                )
                if lane_id == 0:
                    reduce_buf[warp_id] = local_sum
                cute.arch.barrier()
                if warp_id == 0 and lane_id < warps_per_row:
                    local_sum = reduce_buf[lane_id]
                else:
                    local_sum = Float32(0.0)
                local_sum = cute.arch.warp_reduction(
                    local_sum, op=lambda a, b: a + b, threads_in_group=warps_per_row,
                )
                if warp_id == 0 and lane_id == 0:
                    reduce_buf[0] = local_sum
                cute.arch.barrier()
                local_sum = reduce_buf[0]

                inv_sum = Float32(1.0) / local_sum
                for e_off in cutlass.range_constexpr((E + num_threads - 1) // num_threads):
                    e = e_off * num_threads + tidx
                    if e < E:
                        v = mLogitsL[e].to(Float32)
                        mLogitsL[e] = (v * inv_sum).to(mLogitsL.element_type)
                cute.arch.barrier()

                if warp_id == Int32(0):
                    NUM_PER_LANE = const_expr((E + 31) // 32)
                    local_vals = cute.make_fragment(NUM_PER_LANE, Float32)
                    for j in cutlass.range_constexpr(NUM_PER_LANE):
                        local_vals[j] = NEG_INF
                    for j in cutlass.range_constexpr(NUM_PER_LANE):
                        e_idx = lane_id * Int32(NUM_PER_LANE) + Int32(j)
                        if e_idx < Int32(E):
                            local_vals[j] = mLogitsL[e_idx].to(Float32)

                    my_lane_max = NEG_INF
                    for j in cutlass.range_constexpr(NUM_PER_LANE):
                        if local_vals[j] > my_lane_max:
                            my_lane_max = local_vals[j]
                    g_max = cute.arch.warp_reduction(
                        my_lane_max,
                        op=lambda a, b: a if a > b else b,
                        threads_in_group=4,
                    )

                    if (lane_id % Int32(4)) == Int32(0):
                        reduce_buf[lane_id // Int32(4)] = g_max
                    cute.arch.sync_warp()
                    if lane_id == Int32(0):
                        for _kg in cutlass.range_constexpr(topk_group):
                            best_max = NEG_INF
                            best_g = Int32(0)
                            for g in cutlass.range_constexpr(n_group):
                                gm = reduce_buf[g]
                                if gm > best_max:
                                    best_max = gm
                                    best_g = Int32(g)
                            reduce_buf[best_g] = NEG_INF
                    cute.arch.sync_warp()

                    my_group = lane_id // Int32(4)
                    group_marker = reduce_buf[my_group]
                    is_in_top = group_marker < Float32(-1e30)
                    for j in cutlass.range_constexpr(NUM_PER_LANE):
                        local_vals[j] = local_vals[j] if is_in_top else NEG_INF

                    for k_idx in cutlass.range_constexpr(K_topk):
                        local_best_val = NEG_INF
                        local_best_off = Int32(0)
                        for j in cutlass.range_constexpr(NUM_PER_LANE):
                            if local_vals[j] > local_best_val:
                                local_best_val = local_vals[j]
                                local_best_off = Int32(j)
                        global_val = cute.arch.warp_reduction(
                            local_best_val,
                            op=lambda a, b: a if a > b else b,
                            threads_in_group=32,
                        )
                        is_owner = local_best_val == global_val
                        ballot = cute.arch.vote_ballot_sync(is_owner)
                        lsb_only = ballot & (-ballot)
                        first_owner_lane = cute.arch.popc(lsb_only - Int32(1))
                        owner_off = cute.arch.shuffle_sync(
                            local_best_off,
                            offset=first_owner_lane,
                            mask=-1,
                            mask_and_clamp=31,
                        )
                        global_idx = first_owner_lane * Int32(NUM_PER_LANE) + owner_off
                        if lane_id == Int32(0):
                            mTopkWL[k_idx] = global_val * routed_scaling
                            mTopkIdsL[k_idx] = global_idx
                        if lane_id == first_owner_lane:
                            for j in cutlass.range_constexpr(NUM_PER_LANE):
                                if Int32(j) == owner_off:
                                    local_vals[j] = NEG_INF

            moe_out_layer_off = Int64(layer) * Int64(H)
            moe_out_ptr_l = (mMoeOut.iterator + moe_out_layer_off).align(16)
            mMoeOutL = cute.make_tensor(moe_out_ptr_l, cute.make_layout((H,), stride=(1,)))
            moe_routed_symm_local_base = (
                mMoeOutSymmLocal.iterator.toint() + Int64(layer) * Int64(H) * Int64(2)
            )
            moe_routed_symm_mc_base = (
                mMoeOutSymmMc.iterator.toint() + Int64(layer) * Int64(H) * Int64(2)
            )
            ep_sync_local_addr = (
                mEpSyncLocal.iterator.toint()
                + (Int64(layer) * Int64(num_sms) + Int64(bidx)) * Int64(4)
            )
            ep_sync_mc_addr = (
                mEpSyncMc.iterator.toint()
                + (Int64(layer) * Int64(num_sms) + Int64(bidx)) * Int64(4)
            )
            moe_routed_symm_local_ptr_l = (
                mMoeOutSymmLocal.iterator + Int64(layer) * Int64(H)
            ).align(16)
            mMoeRoutedSymmL = cute.make_tensor(
                moe_routed_symm_local_ptr_l,
                cute.make_layout((H,), stride=(1,)),
            )
            zero_base = (bidx * num_threads + tidx) * vec_size
            if zero_base + vec_size <= H:
                for v in cutlass.range_constexpr(vec_size):
                    kz = zero_base + v
                    mMoeOutL[kz] = Float32(0.0).to(mH_in.element_type)
                    mMoeRoutedSymmL[kz] = Float32(0.0).to(mH_in.element_type)
                    mMoeRoutedAccF32[layer, kz] = Float32(0.0)

            cute.arch.barrier()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected9 = Int32((SYNCS_PER_LAYER * layer + 9) * num_sms)
                done9 = Int32(0)
                while done9 < expected9:
                    done9 = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 9] = _clock()


            warp_idx_q1 = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            Q1_NUM_ACTIVE_CLUSTERS_C = const_expr(
                self.Q1_FP8_NUM_ACTIVE_CLUSTERS
            )
            q1_n_active = Int32(cutlass.select_(
                layer == Int32(0),
                Int32(Q1_NUM_ACTIVE_CLUSTERS_C),
                Int32(Q1_NUM_ACTIVE_CLUSTERS_C // 4),
            ))
            Q1_CLUSTER_TILE_M_C = const_expr(self.Q1_CLUSTER_TILE_M)
            Q1_CTA_TILE_M_C = const_expr(self.Q1_CTA_TILE_M)
            Q1_MMA_N_C = const_expr(self.Q1_MMA_N)
            Q1_K_CHUNK_C = const_expr(self.Q1_K_CHUNK)
            Q1_NUM_K_TILES_MMA_C = const_expr(self.Q1_NUM_K_TILES_MMA)
            Q1_AB_STAGES_C = const_expr(self.Q1_AB_STAGES)
            q1_mma_tiler_c = const_expr(self.Q1_MMA_TILER)
            bytes_q1_per_weight = const_expr(Q1_CTA_TILE_M_C * Q1_K_CHUNK_C * 2)
            bytes_q1_ab_per_stage = const_expr(
                2 * cluster_size_ce * bytes_q1_per_weight
            )
            Q1_TMEM_COLS_PER_ACC = const_expr(Q1_MMA_N_C // cluster_size_ce)
            shared_cluster_id_q1 = cluster_id - Int32(R2_ROUTED_NUM_CLUSTERS_CE)
            Q1_SPLITK = const_expr(2 if self.USE_SPLITK else 1)
            q1_m_tile = shared_cluster_id_q1 % q1_n_active
            q1_k_split = shared_cluster_id_q1 // q1_n_active
            if (cluster_id >= Int32(R2_ROUTED_NUM_CLUSTERS_CE)) and (shared_cluster_id_q1 < q1_n_active * Int32(Q1_SPLITK)):
              Q1F_CTA_TILE_M     = const_expr(self.Q1_FP8_CTA_TILE_M)
              Q1F_CLUSTER_TILE_M = const_expr(self.Q1_FP8_CLUSTER_TILE_M)
              Q1F_K_CHUNK        = const_expr(self.Q1_FP8_K_CHUNK)
              Q1F_NUM_K_TILES    = const_expr(self.Q1_FP8_NUM_K_TILES_MMA)
              Q1F_K_TILES_PER_SPLIT = const_expr(Q1F_NUM_K_TILES // Q1_SPLITK)
              q1_k_start = q1_k_split * Int32(Q1F_K_TILES_PER_SPLIT)
              Q1F_AB_STAGES      = const_expr(self.Q1_FP8_AB_STAGES)
              Q1F_MMA_N          = const_expr(self.Q1_FP8_MMA_N)
              Q1F_X_BROADCAST_N  = const_expr(self.Q1_FP8_MMA_N // self.CLUSTER_SIZE)
              SF_VEC             = const_expr(self.Q1_SF_VEC_SIZE)
              Q1F_NUM_SF_BLOCKS_PER_KT = const_expr(Q1F_K_CHUNK // SF_VEC)
              Q1F_TOTAL_SF_BLOCKS = const_expr(H // SF_VEC)
              q1f_mma_tiler_c    = const_expr(self.Q1_FP8_MMA_TILER)
              Q1F_TMEM_COLS_ACC  = const_expr(Q1F_MMA_N)

              warp_id_q1f = tidx // Int32(32)
              lane_q1f = tidx % Int32(32)
              num_warps_quant = const_expr(8)
              LOGICAL_BLOCK_K = const_expr(128)
              SUB_PER_LOG = const_expr(LOGICAL_BLOCK_K // SF_VEC)
              num_logical_blocks_q = const_expr(H // LOGICAL_BLOCK_K)
              logblks_per_warp_q = const_expr(num_logical_blocks_q // num_warps_quant)
              if warp_id_q1f < Int32(num_warps_quant):
                  for lb in cutlass.range_constexpr(logblks_per_warp_q):
                      log_block_idx = (
                          warp_id_q1f * Int32(logblks_per_warp_q) + Int32(lb)
                      )
                      local_amax = Float32(0.0)
                      for s in cutlass.range_constexpr(SUB_PER_LOG):
                          k_pos = log_block_idx * Int32(LOGICAL_BLOCK_K) + Int32(s) * Int32(SF_VEC) + lane_q1f
                          x_bf16 = sNorm1[k_pos]
                          absx = cute.math.absf(x_bf16.to(Float32))
                          local_amax = cutlass.max(local_amax, absx)
                      amax = cute.arch.warp_reduction_max(
                          local_amax, threads_in_group=32,
                      )
                      ratio = amax * Float32(1.0 / 448.0)
                      bits_i32 = ratio.bitcast(cutlass.Int32)
                      exp_field = (bits_i32 >> Int32(23)) & Int32(0xFF)
                      mantissa_field = bits_i32 & Int32(0x7FFFFF)
                      mantissa_nonzero = Int32(0)
                      if mantissa_field != Int32(0):
                          mantissa_nonzero = Int32(1)
                      exp_ceil = exp_field + mantissa_nonzero
                      exp_clamped = cutlass.min(
                          cutlass.max(exp_ceil, Int32(0)), Int32(254),
                      )
                      scale_bits = exp_clamped << Int32(23)
                      scale_f32 = scale_bits.bitcast(Float32)
                      inv_scale = Float32(1.0) / scale_f32
                      inv_scale = Float32(cutlass.select_(
                          amax == Float32(0.0), Float32(0.0), inv_scale,
                      ))
                      xf_q1 = cute.make_fragment(SUB_PER_LOG, Float32)
                      for s in cutlass.range_constexpr(SUB_PER_LOG):
                          k_pos = log_block_idx * Int32(LOGICAL_BLOCK_K) + Int32(s) * Int32(SF_VEC) + lane_q1f
                          x_bf16 = sNorm1[k_pos]
                          x_scaled = x_bf16.to(Float32) * inv_scale
                          xf_q1[s] = cutlass.min(
                              cutlass.max(x_scaled, Float32(-448.0)),
                              Float32(448.0),
                          )
                      xf8_q1 = cute.make_fragment(SUB_PER_LOG, cutlass.Float8E4M3FN)
                      xf8_q1.store(xf_q1.load().to(cutlass.Float8E4M3FN))
                      for s in cutlass.range_constexpr(SUB_PER_LOG):
                          k_pos = log_block_idx * Int32(LOGICAL_BLOCK_K) + Int32(s) * Int32(SF_VEC) + lane_q1f
                          sX_q1_quant_scratch[k_pos] = xf8_q1[s]
                      if lane_q1f == Int32(0):
                          sf_u8 = exp_clamped.to(cutlass.Uint8)
                          sf_byte = sf_u8.bitcast(cutlass.Float8E8M0FNU)
                          for s in cutlass.range_constexpr(SUB_PER_LOG):
                              sX_q1_sf_scratch[log_block_idx * Int32(SUB_PER_LOG) + Int32(s)] = sf_byte
              cute.arch.barrier()

              tma_warp_group_q1f = pipeline.CooperativeGroup(
                  pipeline.Agent.Thread,
              )
              mma_consumer_group_q1f = pipeline.CooperativeGroup(
                  pipeline.Agent.Thread, size=1,
              )
              acc_producer_group_q1f = pipeline.CooperativeGroup(
                  pipeline.Agent.Thread,
              )
              acc_consumer_group_q1f = pipeline.CooperativeGroup(
                  pipeline.Agent.Thread,
                  size=threads_in_epi_ce * cluster_size_ce,
              )
              bytes_q1f_per_cta = const_expr(
                  2 * Q1F_CTA_TILE_M * Q1F_K_CHUNK
                  + 2 * Q1F_CTA_TILE_M * Q1F_K_CHUNK // SF_VEC
              )
              bytes_q1f_ab_per_stage = const_expr(
                  cluster_size_ce * bytes_q1f_per_cta
              )
              q1f_ab_producer, q1f_ab_consumer = pipeline.PipelineTmaUmma.create(
                  barrier_storage=q1_ab_mma_bars.iterator,
                  num_stages=Q1F_AB_STAGES,
                  producer_group=tma_warp_group_q1f,
                  consumer_group=mma_consumer_group_q1f,
                  tx_count=bytes_q1f_ab_per_stage,
                  cta_layout_vmnk=cta_layout_vmnk_mma,
              ).make_participants()
              q1f_acc_producer, q1f_acc_consumer = pipeline.PipelineUmmaAsync.create(
                  barrier_storage=q1_acc_mma_bars.iterator,
                  num_stages=1,
                  producer_group=acc_producer_group_q1f,
                  consumer_group=acc_consumer_group_q1f,
                  cta_layout_vmnk=cta_layout_vmnk_mma,
              ).make_participants()

              gWg_q1f = cute.local_tile(
                  tma_tensor_wg_q1_fp8,
                  cute.slice_(q1f_mma_tiler_c, (None, 0, None)),
                  (None, None, Int32(layer)),
              )
              gWu_q1f = cute.local_tile(
                  tma_tensor_wu_q1_fp8,
                  cute.slice_(q1f_mma_tiler_c, (None, 0, None)),
                  (None, None, Int32(layer)),
              )
              gSFA_g_q1f = cute.local_tile(
                  tma_tensor_sfa_g_q1,
                  cute.slice_(q1f_mma_tiler_c, (None, 0, None)),
                  (None, None, Int32(layer)),
              )
              gSFA_u_q1f = cute.local_tile(
                  tma_tensor_sfa_u_q1,
                  cute.slice_(q1f_mma_tiler_c, (None, 0, None)),
                  (None, None, Int32(layer)),
              )

              thr_mma_q1f = tiled_mma_q1_fp8.get_slice(mma_tile_coord_v)
              tCgWg_f = thr_mma_q1f.partition_A(gWg_q1f)
              tCgWu_f = thr_mma_q1f.partition_A(gWu_q1f)
              tCgSFA_g_f = thr_mma_q1f.partition_A(gSFA_g_q1f)
              tCgSFA_u_f = thr_mma_q1f.partition_A(gSFA_u_q1f)

              tCrWg_f = tiled_mma_q1_fp8.make_fragment_A(sWg_q1_mma)
              tCrWu_f = tiled_mma_q1_fp8.make_fragment_A(sWu_q1_mma)
              tCrX_f = tiled_mma_q1_fp8.make_fragment_B(sX_q1_mma)

              a_cta_layout_f = cute.make_layout(
                  cute.size(cta_layout_vmnk_mma, mode=[2]),
              )
              tWgSWg_f, tWgGWg_f = cute.nvgpu.cpasync.tma_partition(
                  tma_atom_wg_q1_fp8,
                  cta_in_cluster_coord_vmnk[2],
                  a_cta_layout_f,
                  cute.group_modes(sWg_q1_mma, 0, 3),
                  cute.group_modes(tCgWg_f, 0, 3),
              )
              tWuSWu_f, tWuGWu_f = cute.nvgpu.cpasync.tma_partition(
                  tma_atom_wu_q1_fp8,
                  cta_in_cluster_coord_vmnk[2],
                  a_cta_layout_f,
                  cute.group_modes(sWu_q1_mma, 0, 3),
                  cute.group_modes(tCgWu_f, 0, 3),
              )
              tSFAgS_f, tSFAgG_f = cute.nvgpu.cpasync.tma_partition(
                  tma_atom_sfa_g_q1,
                  cta_in_cluster_coord_vmnk[2],
                  a_cta_layout_f,
                  cute.group_modes(sSFA_g_q1, 0, 3),
                  cute.group_modes(tCgSFA_g_f, 0, 3),
              )
              tSFAgS_f = cute.filter_zeros(tSFAgS_f)
              tSFAgG_f = cute.filter_zeros(tSFAgG_f)
              tSFAuS_f, tSFAuG_f = cute.nvgpu.cpasync.tma_partition(
                  tma_atom_sfa_u_q1,
                  cta_in_cluster_coord_vmnk[2],
                  a_cta_layout_f,
                  cute.group_modes(sSFA_u_q1, 0, 3),
                  cute.group_modes(tCgSFA_u_f, 0, 3),
              )
              tSFAuS_f = cute.filter_zeros(tSFAuS_f)
              tSFAuG_f = cute.filter_zeros(tSFAuG_f)

              acc_shape_q1f = tiled_mma_q1_fp8.partition_shape_C(
                  (q1f_mma_tiler_c[0], q1f_mma_tiler_c[1]),
              )
              tCtAcc_q1f_fake = tiled_mma_q1_fp8.make_fragment_C(acc_shape_q1f)

              Q1F_TMEM_COLS_SF   = const_expr(16)
              Q1F_TMEM_OFF_G     = const_expr(0)
              Q1F_TMEM_OFF_U     = const_expr(Q1F_TMEM_COLS_ACC)
              Q1F_TMEM_OFF_SFA_G = const_expr(2 * Q1F_TMEM_COLS_ACC)
              Q1F_TMEM_OFF_SFA_U = const_expr(
                  2 * Q1F_TMEM_COLS_ACC + Q1F_TMEM_COLS_SF
              )
              Q1F_TMEM_OFF_SFB   = const_expr(
                  2 * Q1F_TMEM_COLS_ACC + 2 * Q1F_TMEM_COLS_SF
              )

              sfa_slice_f = cute.slice_(
                  sSFA_g_q1_layout_staged, (None, None, None, 0),
              )
              sfb_slice_f = cute.slice_(
                  sSFB_q1_layout_staged, (None, None, None, 0),
              )
              tCtSFA_layout_f = blockscaled_utils.make_tmem_layout_sfa(
                  tiled_mma_q1_fp8, q1f_mma_tiler_c, SF_VEC, sfa_slice_f,
              )
              tCtSFB_layout_f = blockscaled_utils.make_tmem_layout_sfb(
                  tiled_mma_q1_fp8, q1f_mma_tiler_c, SF_VEC, sfb_slice_f,
              )

              if warp_idx_q1 == Int32(tma_warp_id_ce):
                  cpasync.prefetch_descriptor(tma_atom_wg_q1_fp8)
                  cpasync.prefetch_descriptor(tma_atom_wu_q1_fp8)
                  cpasync.prefetch_descriptor(tma_atom_sfa_g_q1)
                  cpasync.prefetch_descriptor(tma_atom_sfa_u_q1)
                  for kt_local in cutlass.range(Q1F_K_TILES_PER_SPLIT, unroll=1):
                      kt = q1_k_start + Int32(kt_local)
                      ab_empty_f = q1f_ab_producer.acquire_and_advance()
                      k_base_f = kt * Int32(Q1F_K_CHUNK)
                      sX_q1_mma_u32 = cute.recast_tensor(sX_q1_mma, cutlass.Uint32)
                      sX_quant_u32 = cute.recast_tensor(
                          sX_q1_quant_scratch, cutlass.Uint32,
                      )
                      atom_k_u32_lane_q1 = lane_q1f % Int32(8)
                      kk_lane_q1 = lane_q1f // Int32(8)
                      xfp8_u32_q1 = sX_quant_u32[(k_base_f >> 2) + lane_q1f]
                      for nn in cutlass.range_constexpr(Q1F_X_BROADCAST_N):
                          sX_q1_mma_u32[
                              ((nn, atom_k_u32_lane_q1),
                               0, kk_lane_q1, ab_empty_f.index)
                          ] = xfp8_u32_q1
                      sSFB_q1_u8 = cute.recast_tensor(sSFB_q1, cutlass.Uint8)
                      sX_sf_u8 = cute.recast_tensor(sX_q1_sf_scratch, cutlass.Uint8)
                      for sbb in cutlass.range_constexpr(
                          Q1F_NUM_SF_BLOCKS_PER_KT
                      ):
                          sf_idx_global = Int32(kt) * Int32(
                              Q1F_NUM_SF_BLOCKS_PER_KT
                          ) + Int32(sbb)
                          sf_byte = sX_sf_u8[sf_idx_global]
                          i_n = lane_q1f
                          for j_n in cutlass.range_constexpr(4):
                              sSFB_q1_u8[
                                  (((i_n, j_n), 0),
                                   0, sbb, ab_empty_f.index)
                              ] = sf_byte
                      cute.arch.fence_view_async_shared()
                      cute.copy(
                          tma_atom_wg_q1_fp8,
                          tWgGWg_f[(None, q1_m_tile, kt)],
                          tWgSWg_f[(None, ab_empty_f.index)],
                          tma_bar_ptr=ab_empty_f.barrier,
                          mcast_mask=tma_mcast_mask_kv,
                      )
                      cute.copy(
                          tma_atom_wu_q1_fp8,
                          tWuGWu_f[(None, q1_m_tile, kt)],
                          tWuSWu_f[(None, ab_empty_f.index)],
                          tma_bar_ptr=ab_empty_f.barrier,
                          mcast_mask=tma_mcast_mask_kv,
                      )
                      cute.copy(
                          tma_atom_sfa_g_q1,
                          tSFAgG_f[(None, q1_m_tile, kt)],
                          tSFAgS_f[(None, ab_empty_f.index)],
                          tma_bar_ptr=ab_empty_f.barrier,
                          mcast_mask=tma_mcast_mask_kv,
                      )
                      cute.copy(
                          tma_atom_sfa_u_q1,
                          tSFAuG_f[(None, q1_m_tile, kt)],
                          tSFAuS_f[(None, ab_empty_f.index)],
                          tma_bar_ptr=ab_empty_f.barrier,
                          mcast_mask=tma_mcast_mask_kv,
                      )
                  q1f_ab_producer.tail()

              elif warp_idx_q1 == Int32(mma_warp_id_ce):
                  if is_leader_cta:
                      tmem_ptr_q1f = tmem_alloc.retrieve_ptr(Float32)
                      tCtAcc_q1f_g = cute.make_tensor(
                          tmem_ptr_q1f + Q1F_TMEM_OFF_G,
                          tCtAcc_q1f_fake.layout,
                      )
                      tCtAcc_q1f_u = cute.make_tensor(
                          tmem_ptr_q1f + Q1F_TMEM_OFF_U,
                          tCtAcc_q1f_fake.layout,
                      )
                      sfa_g_tmem_ptr = cute.recast_ptr(
                          tmem_ptr_q1f + Q1F_TMEM_OFF_SFA_G,
                          dtype=cutlass.Float8E8M0FNU,
                      )
                      sfa_u_tmem_ptr = cute.recast_ptr(
                          tmem_ptr_q1f + Q1F_TMEM_OFF_SFA_U,
                          dtype=cutlass.Float8E8M0FNU,
                      )
                      sfb_tmem_ptr = cute.recast_ptr(
                          tmem_ptr_q1f + Q1F_TMEM_OFF_SFB,
                          dtype=cutlass.Float8E8M0FNU,
                      )
                      tCtSFA_g_f = cute.make_tensor(
                          sfa_g_tmem_ptr, tCtSFA_layout_f,
                      )
                      tCtSFA_u_f = cute.make_tensor(
                          sfa_u_tmem_ptr, tCtSFA_layout_f,
                      )
                      tCtSFB_f = cute.make_tensor(
                          sfb_tmem_ptr, tCtSFB_layout_f,
                      )

                      copy_atom_s2t_sf = cute.make_copy_atom(
                          tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO),
                          cutlass.Float8E8M0FNU,
                      )
                      sSFA_g_compact = cute.filter_zeros(sSFA_g_q1)
                      sSFA_u_compact = cute.filter_zeros(sSFA_u_q1)
                      sSFB_compact = cute.filter_zeros(sSFB_q1)
                      tCtSFA_g_compact = cute.filter_zeros(tCtSFA_g_f)
                      tCtSFA_u_compact = cute.filter_zeros(tCtSFA_u_f)
                      tCtSFB_compact = cute.filter_zeros(tCtSFB_f)
                      tiled_copy_s2t_sfa_g = tcgen05.make_s2t_copy(
                          copy_atom_s2t_sf, tCtSFA_g_compact,
                      )
                      tiled_copy_s2t_sfa_u = tcgen05.make_s2t_copy(
                          copy_atom_s2t_sf, tCtSFA_u_compact,
                      )
                      tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(
                          copy_atom_s2t_sf, tCtSFB_compact,
                      )
                      thr_s2t_sfa_g = tiled_copy_s2t_sfa_g.get_slice(0)
                      thr_s2t_sfa_u = tiled_copy_s2t_sfa_u.get_slice(0)
                      thr_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
                      tCsSFA_g_s2t_ = thr_s2t_sfa_g.partition_S(sSFA_g_compact)
                      tCsSFA_u_s2t_ = thr_s2t_sfa_u.partition_S(sSFA_u_compact)
                      tCsSFB_s2t_ = thr_s2t_sfb.partition_S(sSFB_compact)
                      tCsSFA_g_s2t = tcgen05.get_s2t_smem_desc_tensor(
                          tiled_copy_s2t_sfa_g, tCsSFA_g_s2t_,
                      )
                      tCsSFA_u_s2t = tcgen05.get_s2t_smem_desc_tensor(
                          tiled_copy_s2t_sfa_u, tCsSFA_u_s2t_,
                      )
                      tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
                          tiled_copy_s2t_sfb, tCsSFB_s2t_,
                      )
                      tCtSFA_g_s2t = thr_s2t_sfa_g.partition_D(
                          tCtSFA_g_compact,
                      )
                      tCtSFA_u_s2t = thr_s2t_sfa_u.partition_D(
                          tCtSFA_u_compact,
                      )
                      tCtSFB_s2t = thr_s2t_sfb.partition_D(tCtSFB_compact)

                      num_kblocks_f = cute.size(tCrWg_f, mode=[2])
                      acc_q1f_empty = q1f_acc_producer.acquire_and_advance()
                      tiled_mma_q1_fp8.set(tcgen05.Field.ACCUMULATE, False)
                      for kt in cutlass.range(Q1F_K_TILES_PER_SPLIT, unroll=1):
                          ab_full_f = q1f_ab_consumer.wait_and_advance()
                          stage_coord = (
                              None, None, None, None, ab_full_f.index,
                          )
                          cute.copy(
                              tiled_copy_s2t_sfa_g,
                              tCsSFA_g_s2t[stage_coord],
                              tCtSFA_g_s2t,
                          )
                          cute.copy(
                              tiled_copy_s2t_sfa_u,
                              tCsSFA_u_s2t[stage_coord],
                              tCtSFA_u_s2t,
                          )
                          cute.copy(
                              tiled_copy_s2t_sfb,
                              tCsSFB_s2t[stage_coord],
                              tCtSFB_s2t,
                          )
                          for kb in cutlass.range_constexpr(num_kblocks_f):
                              sf_kb_coord = (None, None, kb)
                              tiled_mma_q1_fp8.set(
                                  tcgen05.Field.SFA,
                                  tCtSFA_g_f[sf_kb_coord].iterator,
                              )
                              tiled_mma_q1_fp8.set(
                                  tcgen05.Field.SFB,
                                  tCtSFB_f[sf_kb_coord].iterator,
                              )
                              cute.gemm(
                                  tiled_mma_q1_fp8,
                                  tCtAcc_q1f_g,
                                  tCrWg_f[
                                      (None, None, kb, ab_full_f.index)
                                  ],
                                  tCrX_f[
                                      (None, None, kb, ab_full_f.index)
                                  ],
                                  tCtAcc_q1f_g,
                              )
                              tiled_mma_q1_fp8.set(
                                  tcgen05.Field.SFA,
                                  tCtSFA_u_f[sf_kb_coord].iterator,
                              )
                              cute.gemm(
                                  tiled_mma_q1_fp8,
                                  tCtAcc_q1f_u,
                                  tCrWu_f[
                                      (None, None, kb, ab_full_f.index)
                                  ],
                                  tCrX_f[
                                      (None, None, kb, ab_full_f.index)
                                  ],
                                  tCtAcc_q1f_u,
                              )
                              tiled_mma_q1_fp8.set(
                                  tcgen05.Field.ACCUMULATE, True,
                              )
                          ab_full_f.release()
                      acc_q1f_empty.commit()
                  q1f_acc_producer.tail()

              elif warp_idx_q1 < Int32(num_epi_warps_ce):
                  tmem_ptr_q1f = tmem_alloc.retrieve_ptr(Float32)
                  tCtAcc_q1f_g = cute.make_tensor(
                      tmem_ptr_q1f + Q1F_TMEM_OFF_G,
                      tCtAcc_q1f_fake.layout,
                  )
                  tCtAcc_q1f_u = cute.make_tensor(
                      tmem_ptr_q1f + Q1F_TMEM_OFF_U,
                      tCtAcc_q1f_fake.layout,
                  )
                  tStG_acc_f = tCtAcc_q1f_g[(None, None), 0, 0]
                  tStU_acc_f = tCtAcc_q1f_u[(None, None), 0, 0]
                  copy_atom_q1f_t2r = cute.make_copy_atom(
                      tcgen05.Ld32x32bOp(
                          tcgen05.Repetition.x32, tcgen05.Pack.NONE,
                      ),
                      Float32,
                  )
                  tiled_copy_q1f_t2r = tcgen05.make_tmem_copy(
                      copy_atom_q1f_t2r, tStG_acc_f,
                  )
                  thr_copy_q1f_t2r = tiled_copy_q1f_t2r.get_slice(tidx)
                  tStG_t2r_f = thr_copy_q1f_t2r.partition_S(tStG_acc_f)
                  tStU_t2r_f = thr_copy_q1f_t2r.partition_S(tStU_acc_f)

                  cQ1f = cute.make_identity_tensor(
                      (Q1F_CTA_TILE_M, Q1F_MMA_N),
                  )
                  tQ1fcQ1f_t2r = thr_copy_q1f_t2r.partition_D(cQ1f)
                  tQ1frG_t2r = cute.make_rmem_tensor(
                      tQ1fcQ1f_t2r.shape, Float32,
                  )
                  tQ1frU_t2r = cute.make_rmem_tensor(
                      tQ1fcQ1f_t2r.shape, Float32,
                  )

                  acc_q1f_full = q1f_acc_consumer.wait_and_advance()
                  cute.copy(tiled_copy_q1f_t2r, tStG_t2r_f, tQ1frG_t2r)
                  cute.copy(tiled_copy_q1f_t2r, tStU_t2r_f, tQ1frU_t2r)
                  cute.arch.fence_view_async_tmem_load()
                  acc_q1f_full.release()

                  for i in cutlass.range_constexpr(cute.size(tQ1frG_t2r)):
                      coord_i = tQ1fcQ1f_t2r[i]
                      row_in_cta_f = coord_i[0]
                      col_f = coord_i[1]
                      if col_f == Int32(0):
                          i_out_f = (
                              q1_m_tile * Int32(Q1F_CLUSTER_TILE_M)
                              + cta_rank_in_cluster
                              * Int32(Q1F_CTA_TILE_M)
                              + row_in_cta_f
                          )
                          if i_out_f < Int32(I_shared):
                              g_val_f = tQ1frG_t2r[i]
                              u_val_f = tQ1frU_t2r[i]
                              if const_expr(self.USE_SPLITK):
                                  q1_slot = q1_k_split * Int32(SPLITK_IS)
                                  mPartials[layer, Int32(SPLITK_Q1G_OFF) + q1_slot + i_out_f] = g_val_f
                                  mPartials[layer, Int32(SPLITK_Q1U_OFF) + q1_slot + i_out_f] = u_val_f
                              else:
                                  sig_f = Float32(1.0) / (
                                      Float32(1.0)
                                      + cute.math.exp(-g_val_f, fastmath=True)
                                  )
                                  inter_val_f = (g_val_f * sig_f) * u_val_f
                                  mInterTmp[Int32(Q1_INTER_OFFSET) + i_out_f] = inter_val_f.to(
                                      mInterTmp.element_type,
                                  )

            tiles_i_routed = const_expr((I_routed + outs_per_cta - 1) // outs_per_cta)
            iters_i_routed = const_expr((tiles_i_routed + num_sms - 1) // num_sms)
            num_chunks_i_routed = const_expr(max(1, (H + chunk_r - 1) // chunk_r))
            chunk_d_routed = const_expr(threads_per_output * vec_size)
            num_chunks_d_routed = const_expr(max(1, (I_routed + chunk_d_routed - 1) // chunk_d_routed))
            tiles_h_routed = const_expr((H + outs_per_cta - 1) // outs_per_cta)
            iters_h_routed = const_expr((tiles_h_routed + num_sms - 1) // num_sms)

            FP8_VEC = const_expr(16)
            chunk_r_fp8 = const_expr(threads_per_output * FP8_VEC)
            num_chunks_i_routed_fp8 = const_expr(
                max(1, (H + chunk_r_fp8 - 1) // chunk_r_fp8)
            )
            chunk_d_routed_fp8 = const_expr(threads_per_output * FP8_VEC)
            num_chunks_d_routed_fp8 = const_expr(
                max(1, (I_routed + chunk_d_routed_fp8 - 1) // chunk_d_routed_fp8)
            )
            copy_atom_w_routed_fp8 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float8E4M3FN,
                num_bits_per_copy=128,
            )
            warp_id_p = tidx // Int32(32)
            lane_p = tidx % Int32(32)
            P_NUM_WARPS_QUANT = const_expr(8)
            P_LOGICAL_BLOCK_K = const_expr(128)
            P_SF_VEC = const_expr(self.MOE_SF_VEC_SIZE)
            P_SUB_PER_LOG = const_expr(P_LOGICAL_BLOCK_K // P_SF_VEC)
            P_NUM_LOG_BLOCKS = const_expr(H // P_LOGICAL_BLOCK_K)
            P_LOGBLKS_PER_WARP = const_expr(
                P_NUM_LOG_BLOCKS // P_NUM_WARPS_QUANT
            )
            if warp_id_p < Int32(P_NUM_WARPS_QUANT):
                for lb in cutlass.range_constexpr(P_LOGBLKS_PER_WARP):
                    log_block_idx = (
                        warp_id_p * Int32(P_LOGBLKS_PER_WARP) + Int32(lb)
                    )
                    local_amax = Float32(0.0)
                    for s in cutlass.range_constexpr(P_SUB_PER_LOG):
                        k_pos = (
                            log_block_idx * Int32(P_LOGICAL_BLOCK_K)
                            + Int32(s) * Int32(P_SF_VEC) + lane_p
                        )
                        x_bf16 = sNorm1[k_pos]
                        absx = cute.math.absf(x_bf16.to(Float32))
                        local_amax = cutlass.max(local_amax, absx)
                    amax = cute.arch.warp_reduction_max(
                        local_amax, threads_in_group=32,
                    )
                    ratio = amax * Float32(1.0 / 448.0)
                    bits_i32 = ratio.bitcast(cutlass.Int32)
                    exp_field = (bits_i32 >> Int32(23)) & Int32(0xFF)
                    mantissa_field = bits_i32 & Int32(0x7FFFFF)
                    mantissa_nonzero = Int32(0)
                    if mantissa_field != Int32(0):
                        mantissa_nonzero = Int32(1)
                    exp_ceil = exp_field + mantissa_nonzero
                    exp_clamped = cutlass.min(
                        cutlass.max(exp_ceil, Int32(0)), Int32(254),
                    )
                    scale_bits = exp_clamped << Int32(23)
                    scale_f32 = scale_bits.bitcast(Float32)
                    inv_scale = Float32(1.0) / scale_f32
                    inv_scale = Float32(cutlass.select_(
                        amax == Float32(0.0), Float32(0.0), inv_scale,
                    ))
                    xf_p1 = cute.make_fragment(P_SUB_PER_LOG, Float32)
                    for s in cutlass.range_constexpr(P_SUB_PER_LOG):
                        k_pos = (
                            log_block_idx * Int32(P_LOGICAL_BLOCK_K)
                            + Int32(s) * Int32(P_SF_VEC) + lane_p
                        )
                        x_bf16 = sNorm1[k_pos]
                        x_scaled = x_bf16.to(Float32) * inv_scale
                        xf_p1[s] = cutlass.min(
                            cutlass.max(x_scaled, Float32(-448.0)),
                            Float32(448.0),
                        )
                    xf8_p1 = cute.make_fragment(P_SUB_PER_LOG, cutlass.Float8E4M3FN)
                    xf8_p1.store(xf_p1.load().to(cutlass.Float8E4M3FN))
                    for s in cutlass.range_constexpr(P_SUB_PER_LOG):
                        k_pos = (
                            log_block_idx * Int32(P_LOGICAL_BLOCK_K)
                            + Int32(s) * Int32(P_SF_VEC) + lane_p
                        )
                        sX_p1_quant_scratch[k_pos] = xf8_p1[s]
                    if lane_p == Int32(0):
                        sf_u8 = exp_clamped.to(cutlass.Uint8)
                        sf_byte = sf_u8.bitcast(cutlass.Float8E8M0FNU)
                        for s in cutlass.range_constexpr(P_SUB_PER_LOG):
                            sX_p1_sf_scratch[
                                log_block_idx * Int32(P_SUB_PER_LOG)
                                + Int32(s)
                            ] = sf_byte
            cute.arch.barrier()
            warp_idx_p1 = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            P1_NUM_ACTIVE_CLUSTERS = const_expr(
                K_topk * I_routed // self.MOE_P1_CLUSTER_TILE_M
            )
            P1_M_CLUSTERS_PER_EXPERT = const_expr(
                I_routed // self.MOE_P1_CLUSTER_TILE_M
            )
            if cluster_id < Int32(P1_NUM_ACTIVE_CLUSTERS):
                expert_slot = cluster_id // Int32(
                    P1_M_CLUSTERS_PER_EXPERT
                )
                m_cluster_idx = cluster_id % Int32(
                    P1_M_CLUSTERS_PER_EXPERT
                )
                p1_k_split = Int32(0)
                expert_id_p1 = mTopkIdsL[expert_slot]
                is_local_p1 = (
                    (expert_id_p1 >= Int32(self.E_LOCAL_START))
                    and (expert_id_p1 < Int32(self.E_LOCAL_END))
                )
                local_idx_p1 = Int32(cutlass.select_(
                    is_local_p1,
                    expert_id_p1 - Int32(self.E_LOCAL_START),
                    Int32(0),
                ))
                if is_local_p1:
                    P1_CTA_TILE_M     = const_expr(self.MOE_P1_CTA_TILE_M)
                    P1_CLUSTER_TILE_M = const_expr(self.MOE_P1_CLUSTER_TILE_M)
                    P1_K_CHUNK        = const_expr(self.MOE_P1_K_CHUNK)
                    P1_NUM_K_TILES    = const_expr(self.MOE_P1_NUM_K_TILES_MMA)
                    P1_AB_STAGES      = const_expr(self.MOE_P1_AB_STAGES)
                    P1_MMA_N          = const_expr(self.MOE_P1_MMA_N)
                    P1_X_BROADCAST_N  = const_expr(
                        self.MOE_P1_MMA_N // self.CLUSTER_SIZE
                    )
                    P1_SF_VEC         = const_expr(self.MOE_SF_VEC_SIZE)
                    P1_NUM_SF_BLOCKS_PER_KT = const_expr(
                        P1_K_CHUNK // P1_SF_VEC
                    )
                    p1_mma_tiler_c    = const_expr(self.MOE_P1_MMA_TILER)
                    P1_TMEM_COLS_ACC  = const_expr(P1_MMA_N)

                    tma_warp_group_p1 = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                    )
                    mma_consumer_group_p1 = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, size=1,
                    )
                    acc_producer_group_p1 = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                    )
                    acc_consumer_group_p1 = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        size=threads_in_epi_ce * cluster_size_ce,
                    )
                    bytes_p1_per_cta = const_expr(
                        2 * P1_CTA_TILE_M * P1_K_CHUNK
                        + 2 * P1_CTA_TILE_M * P1_K_CHUNK // P1_SF_VEC
                    )
                    bytes_p1_ab_per_stage = const_expr(
                        cluster_size_ce * bytes_p1_per_cta
                    )
                    p1_ab_producer, p1_ab_consumer = pipeline.PipelineTmaUmma.create(
                        barrier_storage=moe_p1_ab_mma_bars.iterator,
                        num_stages=P1_AB_STAGES,
                        producer_group=tma_warp_group_p1,
                        consumer_group=mma_consumer_group_p1,
                        tx_count=bytes_p1_ab_per_stage,
                        cta_layout_vmnk=cta_layout_vmnk_mma,
                    ).make_participants()
                    p1_acc_producer, p1_acc_consumer = pipeline.PipelineUmmaAsync.create(
                        barrier_storage=moe_p1_acc_mma_bars.iterator,
                        num_stages=1,
                        producer_group=acc_producer_group_p1,
                        consumer_group=acc_consumer_group_p1,
                        cta_layout_vmnk=cta_layout_vmnk_mma,
                    ).make_participants()

                    combined_el_p1 = (
                        Int32(local_idx_p1)
                        + Int32(layer) * Int32(E_per_rank)
                    )
                    gWg_p1 = cute.local_tile(
                        tma_tensor_wg_moe_p1,
                        cute.slice_(p1_mma_tiler_c, (None, 0, None)),
                        (None, None, combined_el_p1),
                    )
                    gWu_p1 = cute.local_tile(
                        tma_tensor_wu_moe_p1,
                        cute.slice_(p1_mma_tiler_c, (None, 0, None)),
                        (None, None, combined_el_p1),
                    )
                    gSFA_g_p1 = cute.local_tile(
                        tma_tensor_sfa_g_moe_p1,
                        cute.slice_(p1_mma_tiler_c, (None, 0, None)),
                        (None, None, combined_el_p1),
                    )
                    gSFA_u_p1 = cute.local_tile(
                        tma_tensor_sfa_u_moe_p1,
                        cute.slice_(p1_mma_tiler_c, (None, 0, None)),
                        (None, None, combined_el_p1),
                    )

                    thr_mma_p1 = tiled_mma_moe_p1_fp8.get_slice(
                        mma_tile_coord_v,
                    )
                    tCgWg_p1 = thr_mma_p1.partition_A(gWg_p1)
                    tCgWu_p1 = thr_mma_p1.partition_A(gWu_p1)
                    tCgSFA_g_p1 = thr_mma_p1.partition_A(gSFA_g_p1)
                    tCgSFA_u_p1 = thr_mma_p1.partition_A(gSFA_u_p1)

                    tCrWg_p1 = tiled_mma_moe_p1_fp8.make_fragment_A(sWg_moe_p1)
                    tCrWu_p1 = tiled_mma_moe_p1_fp8.make_fragment_A(sWu_moe_p1)
                    tCrX_p1  = tiled_mma_moe_p1_fp8.make_fragment_B(sX_moe_p1)

                    a_cta_layout_p1 = cute.make_layout(
                        cute.size(cta_layout_vmnk_mma, mode=[2]),
                    )
                    tWgSWg_p1, tWgGWg_p1 = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_wg_moe_p1,
                        cta_in_cluster_coord_vmnk[2],
                        a_cta_layout_p1,
                        cute.group_modes(sWg_moe_p1, 0, 3),
                        cute.group_modes(tCgWg_p1, 0, 3),
                    )
                    tWuSWu_p1, tWuGWu_p1 = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_wu_moe_p1,
                        cta_in_cluster_coord_vmnk[2],
                        a_cta_layout_p1,
                        cute.group_modes(sWu_moe_p1, 0, 3),
                        cute.group_modes(tCgWu_p1, 0, 3),
                    )
                    tSFAgS_p1, tSFAgG_p1 = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_sfa_g_moe_p1,
                        cta_in_cluster_coord_vmnk[2],
                        a_cta_layout_p1,
                        cute.group_modes(sSFA_g_moe_p1, 0, 3),
                        cute.group_modes(tCgSFA_g_p1, 0, 3),
                    )
                    tSFAgS_p1 = cute.filter_zeros(tSFAgS_p1)
                    tSFAgG_p1 = cute.filter_zeros(tSFAgG_p1)
                    tSFAuS_p1, tSFAuG_p1 = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_sfa_u_moe_p1,
                        cta_in_cluster_coord_vmnk[2],
                        a_cta_layout_p1,
                        cute.group_modes(sSFA_u_moe_p1, 0, 3),
                        cute.group_modes(tCgSFA_u_p1, 0, 3),
                    )
                    tSFAuS_p1 = cute.filter_zeros(tSFAuS_p1)
                    tSFAuG_p1 = cute.filter_zeros(tSFAuG_p1)

                    acc_shape_p1 = tiled_mma_moe_p1_fp8.partition_shape_C(
                        (p1_mma_tiler_c[0], p1_mma_tiler_c[1]),
                    )
                    tCtAcc_p1_fake = tiled_mma_moe_p1_fp8.make_fragment_C(
                        acc_shape_p1,
                    )
                    P1_TMEM_COLS_SF   = const_expr(16)
                    P1_TMEM_OFF_G     = const_expr(0)
                    P1_TMEM_OFF_U     = const_expr(P1_TMEM_COLS_ACC)
                    P1_TMEM_OFF_SFA_G = const_expr(2 * P1_TMEM_COLS_ACC)
                    P1_TMEM_OFF_SFA_U = const_expr(
                        2 * P1_TMEM_COLS_ACC + P1_TMEM_COLS_SF
                    )
                    P1_TMEM_OFF_SFB   = const_expr(
                        2 * P1_TMEM_COLS_ACC + 2 * P1_TMEM_COLS_SF
                    )
                    sfa_slice_p1 = cute.slice_(
                        sSFA_g_moe_p1_layout_staged,
                        (None, None, None, 0),
                    )
                    sfb_slice_p1 = cute.slice_(
                        sSFB_moe_p1_layout_staged,
                        (None, None, None, 0),
                    )
                    tCtSFA_layout_p1 = blockscaled_utils.make_tmem_layout_sfa(
                        tiled_mma_moe_p1_fp8, p1_mma_tiler_c, P1_SF_VEC,
                        sfa_slice_p1,
                    )
                    tCtSFB_layout_p1 = blockscaled_utils.make_tmem_layout_sfb(
                        tiled_mma_moe_p1_fp8, p1_mma_tiler_c, P1_SF_VEC,
                        sfb_slice_p1,
                    )

                    if warp_idx_p1 == Int32(tma_warp_id_ce):
                        cpasync.prefetch_descriptor(tma_atom_wg_moe_p1)
                        cpasync.prefetch_descriptor(tma_atom_wu_moe_p1)
                        cpasync.prefetch_descriptor(tma_atom_sfa_g_moe_p1)
                        cpasync.prefetch_descriptor(tma_atom_sfa_u_moe_p1)
                        for kt in cutlass.range(P1_NUM_K_TILES, unroll=1):
                            ab_empty_p1 = p1_ab_producer.acquire_and_advance()
                            k_base_p1 = kt * Int32(P1_K_CHUNK)
                            sX_moe_p1_u32 = cute.recast_tensor(
                                sX_moe_p1, cutlass.Uint32,
                            )
                            sX_p1_quant_u32 = cute.recast_tensor(
                                sX_p1_quant_scratch, cutlass.Uint32,
                            )
                            lane_p1 = tidx % Int32(32)
                            atom_k_u32_lane_p1 = lane_p1 % Int32(8)
                            kk_lane_p1 = lane_p1 // Int32(8)
                            xfp8_u32_p1 = sX_p1_quant_u32[
                                (k_base_p1 >> 2) + lane_p1
                            ]
                            for nn in cutlass.range_constexpr(P1_X_BROADCAST_N):
                                sX_moe_p1_u32[
                                    ((nn, atom_k_u32_lane_p1),
                                     0, kk_lane_p1, ab_empty_p1.index)
                                ] = xfp8_u32_p1
                            sSFB_moe_p1_u8 = cute.recast_tensor(
                                sSFB_moe_p1, cutlass.Uint8,
                            )
                            sX_p1_sf_u8 = cute.recast_tensor(
                                sX_p1_sf_scratch, cutlass.Uint8,
                            )
                            for sbb in cutlass.range_constexpr(
                                P1_NUM_SF_BLOCKS_PER_KT
                            ):
                                sf_idx_global_p1 = Int32(kt) * Int32(
                                    P1_NUM_SF_BLOCKS_PER_KT
                                ) + Int32(sbb)
                                sf_byte_p1 = sX_p1_sf_u8[sf_idx_global_p1]
                                i_n_p1 = lane_p1
                                for j_n in cutlass.range_constexpr(4):
                                    sSFB_moe_p1_u8[
                                        (((i_n_p1, j_n), 0),
                                         0, sbb, ab_empty_p1.index)
                                    ] = sf_byte_p1
                            cute.arch.fence_view_async_shared()
                            cute.copy(
                                tma_atom_wg_moe_p1,
                                tWgGWg_p1[(None, m_cluster_idx, kt)],
                                tWgSWg_p1[(None, ab_empty_p1.index)],
                                tma_bar_ptr=ab_empty_p1.barrier,
                                mcast_mask=tma_mcast_mask_kv,
                            )
                            cute.copy(
                                tma_atom_wu_moe_p1,
                                tWuGWu_p1[(None, m_cluster_idx, kt)],
                                tWuSWu_p1[(None, ab_empty_p1.index)],
                                tma_bar_ptr=ab_empty_p1.barrier,
                                mcast_mask=tma_mcast_mask_kv,
                            )
                            cute.copy(
                                tma_atom_sfa_g_moe_p1,
                                tSFAgG_p1[(None, m_cluster_idx, kt)],
                                tSFAgS_p1[(None, ab_empty_p1.index)],
                                tma_bar_ptr=ab_empty_p1.barrier,
                                mcast_mask=tma_mcast_mask_kv,
                            )
                            cute.copy(
                                tma_atom_sfa_u_moe_p1,
                                tSFAuG_p1[(None, m_cluster_idx, kt)],
                                tSFAuS_p1[(None, ab_empty_p1.index)],
                                tma_bar_ptr=ab_empty_p1.barrier,
                                mcast_mask=tma_mcast_mask_kv,
                            )
                        p1_ab_producer.tail()

                    elif warp_idx_p1 == Int32(mma_warp_id_ce):
                        if is_leader_cta:
                            tmem_ptr_p1 = tmem_alloc.retrieve_ptr(Float32)
                            tCtAcc_p1_g = cute.make_tensor(
                                tmem_ptr_p1 + P1_TMEM_OFF_G,
                                tCtAcc_p1_fake.layout,
                            )
                            tCtAcc_p1_u = cute.make_tensor(
                                tmem_ptr_p1 + P1_TMEM_OFF_U,
                                tCtAcc_p1_fake.layout,
                            )
                            sfa_g_tmem_ptr_p1 = cute.recast_ptr(
                                tmem_ptr_p1 + P1_TMEM_OFF_SFA_G,
                                dtype=cutlass.Float8E8M0FNU,
                            )
                            sfa_u_tmem_ptr_p1 = cute.recast_ptr(
                                tmem_ptr_p1 + P1_TMEM_OFF_SFA_U,
                                dtype=cutlass.Float8E8M0FNU,
                            )
                            sfb_tmem_ptr_p1 = cute.recast_ptr(
                                tmem_ptr_p1 + P1_TMEM_OFF_SFB,
                                dtype=cutlass.Float8E8M0FNU,
                            )
                            tCtSFA_g_p1 = cute.make_tensor(
                                sfa_g_tmem_ptr_p1, tCtSFA_layout_p1,
                            )
                            tCtSFA_u_p1 = cute.make_tensor(
                                sfa_u_tmem_ptr_p1, tCtSFA_layout_p1,
                            )
                            tCtSFB_p1 = cute.make_tensor(
                                sfb_tmem_ptr_p1, tCtSFB_layout_p1,
                            )
                            copy_atom_s2t_sf_p1 = cute.make_copy_atom(
                                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO),
                                cutlass.Float8E8M0FNU,
                            )
                            sSFA_g_compact_p1 = cute.filter_zeros(
                                sSFA_g_moe_p1,
                            )
                            sSFA_u_compact_p1 = cute.filter_zeros(
                                sSFA_u_moe_p1,
                            )
                            sSFB_compact_p1 = cute.filter_zeros(sSFB_moe_p1)
                            tCtSFA_g_compact_p1 = cute.filter_zeros(
                                tCtSFA_g_p1,
                            )
                            tCtSFA_u_compact_p1 = cute.filter_zeros(
                                tCtSFA_u_p1,
                            )
                            tCtSFB_compact_p1 = cute.filter_zeros(tCtSFB_p1)
                            tiled_copy_s2t_sfa_g_p1 = tcgen05.make_s2t_copy(
                                copy_atom_s2t_sf_p1, tCtSFA_g_compact_p1,
                            )
                            tiled_copy_s2t_sfa_u_p1 = tcgen05.make_s2t_copy(
                                copy_atom_s2t_sf_p1, tCtSFA_u_compact_p1,
                            )
                            tiled_copy_s2t_sfb_p1 = tcgen05.make_s2t_copy(
                                copy_atom_s2t_sf_p1, tCtSFB_compact_p1,
                            )
                            thr_s2t_sfa_g_p1 = tiled_copy_s2t_sfa_g_p1.get_slice(0)
                            thr_s2t_sfa_u_p1 = tiled_copy_s2t_sfa_u_p1.get_slice(0)
                            thr_s2t_sfb_p1 = tiled_copy_s2t_sfb_p1.get_slice(0)
                            tCsSFA_g_s2t_p1_ = thr_s2t_sfa_g_p1.partition_S(
                                sSFA_g_compact_p1,
                            )
                            tCsSFA_u_s2t_p1_ = thr_s2t_sfa_u_p1.partition_S(
                                sSFA_u_compact_p1,
                            )
                            tCsSFB_s2t_p1_ = thr_s2t_sfb_p1.partition_S(
                                sSFB_compact_p1,
                            )
                            tCsSFA_g_s2t_p1 = tcgen05.get_s2t_smem_desc_tensor(
                                tiled_copy_s2t_sfa_g_p1, tCsSFA_g_s2t_p1_,
                            )
                            tCsSFA_u_s2t_p1 = tcgen05.get_s2t_smem_desc_tensor(
                                tiled_copy_s2t_sfa_u_p1, tCsSFA_u_s2t_p1_,
                            )
                            tCsSFB_s2t_p1 = tcgen05.get_s2t_smem_desc_tensor(
                                tiled_copy_s2t_sfb_p1, tCsSFB_s2t_p1_,
                            )
                            tCtSFA_g_s2t_p1 = thr_s2t_sfa_g_p1.partition_D(
                                tCtSFA_g_compact_p1,
                            )
                            tCtSFA_u_s2t_p1 = thr_s2t_sfa_u_p1.partition_D(
                                tCtSFA_u_compact_p1,
                            )
                            tCtSFB_s2t_p1 = thr_s2t_sfb_p1.partition_D(
                                tCtSFB_compact_p1,
                            )

                            num_kblocks_p1 = cute.size(tCrWg_p1, mode=[2])
                            acc_p1_empty = p1_acc_producer.acquire_and_advance()
                            tiled_mma_moe_p1_fp8.set(
                                tcgen05.Field.ACCUMULATE, False,
                            )
                            for kt in cutlass.range(P1_NUM_K_TILES, unroll=1):
                                ab_full_p1 = p1_ab_consumer.wait_and_advance()
                                stage_coord_p1 = (
                                    None, None, None, None,
                                    ab_full_p1.index,
                                )
                                cute.copy(
                                   tiled_copy_s2t_sfa_g_p1,
                                   tCsSFA_g_s2t_p1[stage_coord_p1],
                                   tCtSFA_g_s2t_p1,
                                )
                                cute.copy(
                                   tiled_copy_s2t_sfa_u_p1,
                                   tCsSFA_u_s2t_p1[stage_coord_p1],
                                   tCtSFA_u_s2t_p1,
                                )
                                cute.copy(
                                   tiled_copy_s2t_sfb_p1,
                                   tCsSFB_s2t_p1[stage_coord_p1],
                                   tCtSFB_s2t_p1,
                                )
                                for kb in cutlass.range_constexpr(
                                   num_kblocks_p1
                                ):
                                   sf_kb_coord_p1 = (None, None, kb)
                                   tiled_mma_moe_p1_fp8.set(
                                       tcgen05.Field.SFA,
                                       tCtSFA_g_p1[sf_kb_coord_p1].iterator,
                                   )
                                   tiled_mma_moe_p1_fp8.set(
                                       tcgen05.Field.SFB,
                                       tCtSFB_p1[sf_kb_coord_p1].iterator,
                                   )
                                   cute.gemm(
                                       tiled_mma_moe_p1_fp8,
                                       tCtAcc_p1_g,
                                       tCrWg_p1[
                                           (None, None, kb, ab_full_p1.index)
                                       ],
                                       tCrX_p1[
                                           (None, None, kb, ab_full_p1.index)
                                       ],
                                       tCtAcc_p1_g,
                                   )
                                   tiled_mma_moe_p1_fp8.set(
                                       tcgen05.Field.SFA,
                                       tCtSFA_u_p1[sf_kb_coord_p1].iterator,
                                   )
                                   cute.gemm(
                                       tiled_mma_moe_p1_fp8,
                                       tCtAcc_p1_u,
                                       tCrWu_p1[
                                           (None, None, kb, ab_full_p1.index)
                                       ],
                                       tCrX_p1[
                                           (None, None, kb, ab_full_p1.index)
                                       ],
                                       tCtAcc_p1_u,
                                   )
                                   tiled_mma_moe_p1_fp8.set(
                                       tcgen05.Field.ACCUMULATE, True,
                                   )
                                ab_full_p1.release()
                            acc_p1_empty.commit()
                        p1_acc_producer.tail()

                    elif warp_idx_p1 < Int32(num_epi_warps_ce):
                        tmem_ptr_p1_epi = tmem_alloc.retrieve_ptr(Float32)
                        tCtAcc_p1_g_epi = cute.make_tensor(
                            tmem_ptr_p1_epi + P1_TMEM_OFF_G,
                            tCtAcc_p1_fake.layout,
                        )
                        tCtAcc_p1_u_epi = cute.make_tensor(
                            tmem_ptr_p1_epi + P1_TMEM_OFF_U,
                            tCtAcc_p1_fake.layout,
                        )
                        tStG_acc_p1 = tCtAcc_p1_g_epi[(None, None), 0, 0]
                        tStU_acc_p1 = tCtAcc_p1_u_epi[(None, None), 0, 0]
                        copy_atom_p1_t2r = cute.make_copy_atom(
                            tcgen05.Ld32x32bOp(
                                tcgen05.Repetition.x32, tcgen05.Pack.NONE,
                            ),
                            Float32,
                        )
                        tiled_copy_p1_t2r = tcgen05.make_tmem_copy(
                            copy_atom_p1_t2r, tStG_acc_p1,
                        )
                        thr_copy_p1_t2r = tiled_copy_p1_t2r.get_slice(tidx)
                        tStG_t2r_p1 = thr_copy_p1_t2r.partition_S(tStG_acc_p1)
                        tStU_t2r_p1 = thr_copy_p1_t2r.partition_S(tStU_acc_p1)
                        cP1 = cute.make_identity_tensor(
                            (P1_CTA_TILE_M, P1_MMA_N),
                        )
                        tP1cP1_t2r = thr_copy_p1_t2r.partition_D(cP1)
                        tP1rG_t2r = cute.make_rmem_tensor(
                            tP1cP1_t2r.shape, Float32,
                        )
                        tP1rU_t2r = cute.make_rmem_tensor(
                            tP1cP1_t2r.shape, Float32,
                        )
                        acc_p1_full = p1_acc_consumer.wait_and_advance()
                        cute.copy(tiled_copy_p1_t2r, tStG_t2r_p1, tP1rG_t2r)
                        cute.copy(tiled_copy_p1_t2r, tStU_t2r_p1, tP1rU_t2r)
                        cute.arch.fence_view_async_tmem_load()
                        acc_p1_full.release()
                        for i in cutlass.range_constexpr(
                            cute.size(tP1rG_t2r)
                        ):
                            coord_i_p1 = tP1cP1_t2r[i]
                            row_in_cta_p1 = coord_i_p1[0]
                            col_p1 = coord_i_p1[1]
                            if col_p1 == Int32(0):
                                i_out_p1 = (
                                    m_cluster_idx * Int32(P1_CLUSTER_TILE_M)
                                    + cta_rank_in_cluster
                                    * Int32(P1_CTA_TILE_M)
                                    + row_in_cta_p1
                                )
                                if i_out_p1 < Int32(I_routed):
                                    g_val_p1 = tP1rG_t2r[i]
                                    u_val_p1 = tP1rU_t2r[i]
                                    if const_expr(self.USE_SPLITK and self.ROUTED_SPLIT):
                                        p1_row = Int32(expert_slot * I_routed) + i_out_p1
                                        mPartials[layer, Int32(SPLITK_P1G_OFF) + p1_k_split * Int32(SPLITK_KI) + p1_row] = g_val_p1
                                        mPartials[layer, Int32(SPLITK_P1U_OFF) + p1_k_split * Int32(SPLITK_KI) + p1_row] = u_val_p1
                                    else:
                                        sig_p1 = Float32(1.0) / (
                                            Float32(1.0)
                                            + cute.math.exp(-g_val_p1, fastmath=True)
                                        )
                                        inter_val_p1 = (
                                            g_val_p1 * sig_p1
                                        ) * u_val_p1
                                        mInterTmp[
                                            Int32(expert_slot * I_routed)
                                            + i_out_p1
                                        ] = inter_val_p1.to(
                                            mInterTmp.element_type,
                                        )
                                        if layer == Int32(0) or layer == Int32(1):
                                            if cluster_id == Int32(0):
                                                if cta_rank_in_cluster == Int32(0):
                                                    if (tidx % Int32(32)) == Int32(0):
                                                        if i_out_p1 == Int32(0):
                                                            g_bits = g_val_p1.bitcast(cutlass.Int32)
                                                            u_bits = u_val_p1.bitcast(cutlass.Int32)
                                                            inter_bits = inter_val_p1.bitcast(cutlass.Int32)
                                                            llvm.inline_asm(
                                                                None,
                                                                [g_bits.ir_value(), u_bits.ir_value(), inter_bits.ir_value()],
                                                                "",
                                                                "r,r,r",
                                                                has_side_effects=True,
                                                                asm_dialect=0,
                                                            )

            for k_exp in cutlass.range_constexpr(K_topk):
                expert_id = mTopkIdsL[k_exp]
                is_local_b = (expert_id >= Int32(self.E_LOCAL_START)) and (expert_id < Int32(self.E_LOCAL_END))
                local_idx = Int32(cutlass.select_(
                    is_local_b,
                    expert_id - Int32(self.E_LOCAL_START),
                    Int32(0),
                ))

                gate_base_ptr = (
                    mWgateRouted.iterator
                    + Int64(layer) * Int64(E_per_rank) * Int64(I_routed) * Int64(H)
                    + Int64(local_idx) * Int64(I_routed) * Int64(H)
                ).align(16)
                up_base_ptr = (
                    mWupRouted.iterator
                    + Int64(layer) * Int64(E_per_rank) * Int64(I_routed) * Int64(H)
                    + Int64(local_idx) * Int64(I_routed) * Int64(H)
                ).align(16)
                mWgL = cute.make_tensor(gate_base_ptr, cute.make_layout((I_routed, H), stride=(H, 1)))
                mWuL = cute.make_tensor(up_base_ptr, cute.make_layout((I_routed, H), stride=(H, 1)))
                H_SF_K = const_expr(H // 128)
                g_fp8_base = (
                    mWgateRouted_fp8.iterator
                    + Int64(layer) * Int64(E_per_rank) * Int64(I_routed) * Int64(H)
                    + Int64(local_idx) * Int64(I_routed) * Int64(H)
                ).align(16)
                u_fp8_base = (
                    mWupRouted_fp8.iterator
                    + Int64(layer) * Int64(E_per_rank) * Int64(I_routed) * Int64(H)
                    + Int64(local_idx) * Int64(I_routed) * Int64(H)
                ).align(16)
                g_sf_base = (
                    mSfGateRouted.iterator
                    + Int64(layer) * Int64(E_per_rank) * Int64(I_routed) * Int64(H_SF_K)
                    + Int64(local_idx) * Int64(I_routed) * Int64(H_SF_K)
                ).align(16)
                u_sf_base = (
                    mSfUpRouted.iterator
                    + Int64(layer) * Int64(E_per_rank) * Int64(I_routed) * Int64(H_SF_K)
                    + Int64(local_idx) * Int64(I_routed) * Int64(H_SF_K)
                ).align(16)
                mWgL_fp8 = cute.make_tensor(g_fp8_base, cute.make_layout((I_routed, H), stride=(H, 1)))
                mWuL_fp8 = cute.make_tensor(u_fp8_base, cute.make_layout((I_routed, H), stride=(H, 1)))
                mSfG_L = cute.make_tensor(g_sf_base, cute.make_layout((I_routed, H_SF_K), stride=(H_SF_K, 1)))
                mSfU_L = cute.make_tensor(u_sf_base, cute.make_layout((I_routed, H_SF_K), stride=(H_SF_K, 1)))
                pass

                cute.arch.barrier()
                if tidx == 0:
                    cute.arch.atomic_add(
                        mCounter.iterator, Int32(1), sem="release", scope="gpu",
                    )
                cute.arch.barrier()
                if bidx == 0 and tidx == 0:
                    mStageTs[layer, Int32(11 + k_exp)] = _clock()

            if tidx == 0:
                expected_p1_pass = Int32((SYNCS_PER_LAYER * layer + 9 + K_topk) * num_sms)
                done_p1_pass = Int32(0)
                while done_p1_pass < expected_p1_pass:
                    done_p1_pass = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, Int32(43)] = _clock()

            if const_expr(self.USE_SPLITK):
                splitk_q1_rows = Int32(cutlass.select_(
                    layer == Int32(0), Int32(I_shared), Int32(I_shared // 4)))
                splitk_gtid = Int32(bidx) * Int32(num_threads) + Int32(tidx)
                if splitk_gtid < splitk_q1_rows:
                    g_sk = Float32(0.0)
                    u_sk = Float32(0.0)
                    for ks_sk in cutlass.range_constexpr(SPLITK_SQ_MAX):
                        g_sk = g_sk + mPartials[
                            layer, Int32(SPLITK_Q1G_OFF) + Int32(ks_sk * SPLITK_IS) + splitk_gtid]
                        u_sk = u_sk + mPartials[
                            layer, Int32(SPLITK_Q1U_OFF) + Int32(ks_sk * SPLITK_IS) + splitk_gtid]
                    sig_sk = Float32(1.0) / (
                        Float32(1.0) + cute.math.exp(-g_sk, fastmath=True)
                    )
                    mInterTmp[Int32(Q1_INTER_OFFSET) + splitk_gtid] = (
                        (g_sk * sig_sk) * u_sk
                    ).to(mInterTmp.element_type)
                cute.arch.barrier()

            P2_NUM_ACTIVE_CLUSTERS = const_expr(H // self.MOE_P2_CLUSTER_TILE_M)
            P2_SLOTS_PER_PASS = const_expr(
                (self.num_sms // self.CLUSTER_SIZE) // P2_NUM_ACTIVE_CLUSTERS
            )
            P2_NUM_PASSES = const_expr(
                (K_topk + P2_SLOTS_PER_PASS - 1) // P2_SLOTS_PER_PASS
            )
            P2_PARALLEL_CLUSTERS = const_expr(
                P2_SLOTS_PER_PASS * P2_NUM_ACTIVE_CLUSTERS
            )
            for sub_pass in cutlass.range_constexpr(P2_NUM_PASSES):
                p2_slot_in_pass = cluster_id // Int32(P2_NUM_ACTIVE_CLUSTERS)
                k_exp_raw = Int32(sub_pass * P2_SLOTS_PER_PASS) + p2_slot_in_pass
                k_exp = Int32(cutlass.select_(
                    k_exp_raw < Int32(K_topk), k_exp_raw, Int32(K_topk - 1),
                ))
                expert_id = mTopkIdsL[k_exp]
                weight_k = mTopkWL[k_exp]
                is_local_b = (expert_id >= Int32(self.E_LOCAL_START)) and (expert_id < Int32(self.E_LOCAL_END))
                local_idx = Int32(cutlass.select_(
                    is_local_b,
                    expert_id - Int32(self.E_LOCAL_START),
                    Int32(0),
                ))
                weight_k = Float32(cutlass.select_(
                    is_local_b,
                    weight_k,
                    Float32(0.0),
                ))

                down_base_ptr = (
                    mWdownRouted.iterator
                    + Int64(layer) * Int64(E_per_rank) * Int64(H) * Int64(I_routed)
                    + Int64(local_idx) * Int64(H) * Int64(I_routed)
                ).align(16)
                mWdL = cute.make_tensor(down_base_ptr, cute.make_layout((H, I_routed), stride=(I_routed, 1)))
                I_R_SF_K = const_expr(I_routed // 128)
                d_fp8_base = (
                    mWdownRouted_fp8.iterator
                    + Int64(layer) * Int64(E_per_rank) * Int64(H) * Int64(I_routed)
                    + Int64(local_idx) * Int64(H) * Int64(I_routed)
                ).align(16)
                d_sf_base = (
                    mSfDownRouted.iterator
                    + Int64(layer) * Int64(E_per_rank) * Int64(H) * Int64(I_R_SF_K)
                    + Int64(local_idx) * Int64(H) * Int64(I_R_SF_K)
                ).align(16)
                mWdL_fp8 = cute.make_tensor(d_fp8_base, cute.make_layout((H, I_routed), stride=(I_routed, 1)))
                mSfD_L = cute.make_tensor(d_sf_base, cute.make_layout((H, I_R_SF_K), stride=(I_R_SF_K, 1)))
                warp_idx_p2 = cute.arch.make_warp_uniform(cute.arch.warp_idx())
                if cluster_id < Int32(P2_PARALLEL_CLUSTERS):
                    m_cluster_idx_p2 = cluster_id % Int32(P2_NUM_ACTIVE_CLUSTERS)
                    if is_local_b:
                        P2_CTA_TILE_M     = const_expr(self.MOE_P2_CTA_TILE_M)
                        P2_CLUSTER_TILE_M = const_expr(self.MOE_P2_CLUSTER_TILE_M)
                        P2_K_CHUNK        = const_expr(self.MOE_P2_K_CHUNK)
                        P2_NUM_K_TILES    = const_expr(self.MOE_P2_NUM_K_TILES_MMA)
                        P2_AB_STAGES      = const_expr(self.MOE_P2_AB_STAGES)
                        P2_MMA_N          = const_expr(self.MOE_P2_MMA_N)
                        P2_X_BROADCAST_N  = const_expr(
                            self.MOE_P2_MMA_N // self.CLUSTER_SIZE
                        )
                        P2_SF_VEC         = const_expr(self.MOE_SF_VEC_SIZE)
                        P2_NUM_SF_BLOCKS_PER_KT = const_expr(
                            P2_K_CHUNK // P2_SF_VEC
                        )
                        p2_mma_tiler_c    = const_expr(self.MOE_P2_MMA_TILER)
                        P2_TMEM_COLS_ACC  = const_expr(P2_MMA_N)
                        P2_TMEM_COLS_SF   = const_expr(16)
                        P2_TMEM_OFF_D     = const_expr(0)
                        P2_TMEM_OFF_SFA_D = const_expr(P2_TMEM_COLS_ACC)
                        P2_TMEM_OFF_SFB   = const_expr(
                            P2_TMEM_COLS_ACC + P2_TMEM_COLS_SF
                        )

                        P2_NUM_WARPS_QUANT_Q = const_expr(6)
                        P2_NUM_LOG_BLOCKS_Q  = const_expr(I_routed // P2_K_CHUNK)
                        P2_LOGBLKS_PER_WARP_Q = const_expr(
                            P2_NUM_LOG_BLOCKS_Q // P2_NUM_WARPS_QUANT_Q
                        )
                        P2_SUB_PER_LOG_Q     = const_expr(P2_K_CHUNK // P2_SF_VEC)
                        warp_id_p2_q = tidx // Int32(32)
                        lane_p2_q    = tidx % Int32(32)
                        inter_val_base_q = Int32(k_exp * I_routed)
                        if warp_id_p2_q < Int32(P2_NUM_WARPS_QUANT_Q):
                            for lb in cutlass.range_constexpr(P2_LOGBLKS_PER_WARP_Q):
                                log_block_idx_q = (
                                    warp_id_p2_q * Int32(P2_LOGBLKS_PER_WARP_Q)
                                    + Int32(lb)
                                )
                                local_amax_q = Float32(0.0)
                                for s in cutlass.range_constexpr(P2_SUB_PER_LOG_Q):
                                    k_pos_q = (
                                        log_block_idx_q * Int32(P2_K_CHUNK)
                                        + Int32(s) * Int32(P2_SF_VEC) + lane_p2_q
                                    )
                                    x_bf16_q = mInterTmp[
                                        inter_val_base_q + k_pos_q
                                    ]
                                    absx_q = cute.math.absf(x_bf16_q.to(Float32))
                                    local_amax_q = cutlass.max(local_amax_q, absx_q)
                                amax_q = cute.arch.warp_reduction_max(
                                    local_amax_q, threads_in_group=32,
                                )
                                if const_expr(lb == 0):
                                    if layer == Int32(0) or layer == Int32(1):
                                        if cluster_id == Int32(0):
                                            if cta_rank_in_cluster == Int32(0):
                                                if warp_id_p2_q == Int32(0):
                                                    if lane_p2_q == Int32(0):
                                                        _direct0 = mInterTmp[
                                                            inter_val_base_q
                                                        ].to(Float32)
                                                        _direct32 = mInterTmp[
                                                            inter_val_base_q + Int32(32)
                                                        ].to(Float32)
                                                        _direct64 = mInterTmp[
                                                            inter_val_base_q + Int32(64)
                                                        ].to(Float32)
                                                        amax_bits = amax_q.bitcast(cutlass.Int32)
                                                        d0_bits = _direct0.bitcast(cutlass.Int32)
                                                        d32_bits = _direct32.bitcast(cutlass.Int32)
                                                        d64_bits = _direct64.bitcast(cutlass.Int32)
                                                        llvm.inline_asm(
                                                            None,
                                                            [amax_bits.ir_value(), d0_bits.ir_value(), d32_bits.ir_value(), d64_bits.ir_value()],
                                                            "",
                                                            "r,r,r,r",
                                                            has_side_effects=True,
                                                            asm_dialect=0,
                                                        )
                                ratio_q = amax_q * Float32(1.0 / 448.0)
                                bits_i32_q = ratio_q.bitcast(cutlass.Int32)
                                exp_field_q = (bits_i32_q >> Int32(23)) & Int32(0xFF)
                                mantissa_field_q = bits_i32_q & Int32(0x7FFFFF)
                                mantissa_nonzero_q = Int32(0)
                                if mantissa_field_q != Int32(0):
                                    mantissa_nonzero_q = Int32(1)
                                exp_ceil_q = exp_field_q + mantissa_nonzero_q
                                exp_clamped_q = cutlass.min(
                                    cutlass.max(exp_ceil_q, Int32(0)), Int32(254),
                                )
                                scale_bits_q = exp_clamped_q << Int32(23)
                                scale_f32_q = scale_bits_q.bitcast(Float32)
                                inv_scale_q = Float32(1.0) / scale_f32_q
                                inv_scale_q = Float32(cutlass.select_(
                                    amax_q == Float32(0.0),
                                    Float32(0.0),
                                    inv_scale_q,
                                ))
                                xf_p2 = cute.make_fragment(P2_SUB_PER_LOG_Q, Float32)
                                for s in cutlass.range_constexpr(P2_SUB_PER_LOG_Q):
                                    k_pos_q = (
                                        log_block_idx_q * Int32(P2_K_CHUNK)
                                        + Int32(s) * Int32(P2_SF_VEC) + lane_p2_q
                                    )
                                    x_bf16_q = mInterTmp[
                                        inter_val_base_q + k_pos_q
                                    ]
                                    x_scaled_q = x_bf16_q.to(Float32) * inv_scale_q
                                    xf_p2[s] = cutlass.min(
                                        cutlass.max(x_scaled_q, Float32(-448.0)),
                                        Float32(448.0),
                                    )
                                xf8_p2 = cute.make_fragment(P2_SUB_PER_LOG_Q, cutlass.Float8E4M3FN)
                                xf8_p2.store(xf_p2.load().to(cutlass.Float8E4M3FN))
                                for s in cutlass.range_constexpr(P2_SUB_PER_LOG_Q):
                                    k_pos_q = (
                                        log_block_idx_q * Int32(P2_K_CHUNK)
                                        + Int32(s) * Int32(P2_SF_VEC) + lane_p2_q
                                    )
                                    sX_p2_quant_scratch[k_pos_q] = xf8_p2[s]
                                if lane_p2_q == Int32(0):
                                    sf_u8_q = exp_clamped_q.to(cutlass.Uint8)
                                    sf_byte_q = sf_u8_q.bitcast(
                                        cutlass.Float8E8M0FNU,
                                    )
                                    for s in cutlass.range_constexpr(
                                        P2_SUB_PER_LOG_Q
                                    ):
                                        sX_p2_sf_scratch[
                                            log_block_idx_q
                                            * Int32(P2_SUB_PER_LOG_Q)
                                            + Int32(s)
                                        ] = sf_byte_q
                        cute.arch.barrier()

                        tma_warp_group_p2 = pipeline.CooperativeGroup(
                            pipeline.Agent.Thread,
                        )
                        mma_consumer_group_p2 = pipeline.CooperativeGroup(
                            pipeline.Agent.Thread, size=1,
                        )
                        acc_producer_group_p2 = pipeline.CooperativeGroup(
                            pipeline.Agent.Thread,
                        )
                        acc_consumer_group_p2 = pipeline.CooperativeGroup(
                            pipeline.Agent.Thread,
                            size=threads_in_epi_ce * cluster_size_ce,
                        )
                        bytes_p2_per_cta = const_expr(
                            P2_CTA_TILE_M * P2_K_CHUNK
                            + P2_CTA_TILE_M * P2_K_CHUNK // P2_SF_VEC
                        )
                        bytes_p2_ab_per_stage = const_expr(
                            cluster_size_ce * bytes_p2_per_cta
                        )
                        p2_ab_producer, p2_ab_consumer = pipeline.PipelineTmaUmma.create(
                            barrier_storage=moe_p2_ab_mma_bars.iterator,
                            num_stages=P2_AB_STAGES,
                            producer_group=tma_warp_group_p2,
                            consumer_group=mma_consumer_group_p2,
                            tx_count=bytes_p2_ab_per_stage,
                            cta_layout_vmnk=cta_layout_vmnk_mma,
                        ).make_participants()
                        p2_acc_producer, p2_acc_consumer = pipeline.PipelineUmmaAsync.create(
                            barrier_storage=moe_p2_acc_mma_bars.iterator,
                            num_stages=1,
                            producer_group=acc_producer_group_p2,
                            consumer_group=acc_consumer_group_p2,
                            cta_layout_vmnk=cta_layout_vmnk_mma,
                        ).make_participants()

                        combined_el_p2 = (
                            Int32(local_idx) + Int32(layer) * Int32(E_per_rank)
                        )
                        gWd_p2 = cute.local_tile(
                            tma_tensor_wd_moe_p2,
                            cute.slice_(p2_mma_tiler_c, (None, 0, None)),
                            (None, None, combined_el_p2),
                        )
                        gSFA_d_p2 = cute.local_tile(
                            tma_tensor_sfa_d_moe_p2,
                            cute.slice_(p2_mma_tiler_c, (None, 0, None)),
                            (None, None, combined_el_p2),
                        )
                        thr_mma_p2 = tiled_mma_moe_p2_fp8.get_slice(
                            mma_tile_coord_v,
                        )
                        tCgWd_p2    = thr_mma_p2.partition_A(gWd_p2)
                        tCgSFA_d_p2 = thr_mma_p2.partition_A(gSFA_d_p2)
                        tCrWd_p2 = tiled_mma_moe_p2_fp8.make_fragment_A(sWd_moe_p2)
                        tCrX_p2  = tiled_mma_moe_p2_fp8.make_fragment_B(sX_moe_p2)

                        a_cta_layout_p2 = cute.make_layout(
                            cute.size(cta_layout_vmnk_mma, mode=[2]),
                        )
                        tWdSWd_p2, tWdGWd_p2 = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_wd_moe_p2,
                            cta_in_cluster_coord_vmnk[2],
                            a_cta_layout_p2,
                            cute.group_modes(sWd_moe_p2, 0, 3),
                            cute.group_modes(tCgWd_p2, 0, 3),
                        )
                        tSFAdS_p2, tSFAdG_p2 = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_sfa_d_moe_p2,
                            cta_in_cluster_coord_vmnk[2],
                            a_cta_layout_p2,
                            cute.group_modes(sSFA_d_moe_p2, 0, 3),
                            cute.group_modes(tCgSFA_d_p2, 0, 3),
                        )
                        tSFAdS_p2 = cute.filter_zeros(tSFAdS_p2)
                        tSFAdG_p2 = cute.filter_zeros(tSFAdG_p2)

                        acc_shape_p2 = tiled_mma_moe_p2_fp8.partition_shape_C(
                            (p2_mma_tiler_c[0], p2_mma_tiler_c[1]),
                        )
                        tCtAcc_p2_fake = tiled_mma_moe_p2_fp8.make_fragment_C(
                            acc_shape_p2,
                        )
                        sfa_slice_p2 = cute.slice_(
                            sSFA_d_moe_p2_layout_staged,
                            (None, None, None, 0),
                        )
                        sfb_slice_p2 = cute.slice_(
                            sSFB_moe_p2_layout_staged,
                            (None, None, None, 0),
                        )
                        tCtSFA_layout_p2 = blockscaled_utils.make_tmem_layout_sfa(
                            tiled_mma_moe_p2_fp8, p2_mma_tiler_c, P2_SF_VEC,
                            sfa_slice_p2,
                        )
                        tCtSFB_layout_p2 = blockscaled_utils.make_tmem_layout_sfb(
                            tiled_mma_moe_p2_fp8, p2_mma_tiler_c, P2_SF_VEC,
                            sfb_slice_p2,
                        )

                        if warp_idx_p2 == Int32(tma_warp_id_ce):
                            cpasync.prefetch_descriptor(tma_atom_wd_moe_p2)
                            cpasync.prefetch_descriptor(tma_atom_sfa_d_moe_p2)
                            for kt in cutlass.range(P2_NUM_K_TILES, unroll=1):
                                ab_empty_p2 = p2_ab_producer.acquire_and_advance()
                                k_base_p2 = kt * Int32(P2_K_CHUNK)
                                sX_moe_p2_u32 = cute.recast_tensor(
                                    sX_moe_p2, cutlass.Uint32,
                                )
                                sX_p2_quant_u32 = cute.recast_tensor(
                                    sX_p2_quant_scratch, cutlass.Uint32,
                                )
                                lane_p2_b = tidx % Int32(32)
                                atom_k_u32_lane_p2 = lane_p2_b % Int32(8)
                                kk_lane_p2 = lane_p2_b // Int32(8)
                                xfp8_u32_p2 = sX_p2_quant_u32[
                                    (k_base_p2 >> 2) + lane_p2_b
                                ]
                                for nn in cutlass.range_constexpr(P2_X_BROADCAST_N):
                                    sX_moe_p2_u32[
                                        ((nn, atom_k_u32_lane_p2),
                                         0, kk_lane_p2, ab_empty_p2.index)
                                    ] = xfp8_u32_p2
                                sSFB_moe_p2_u8 = cute.recast_tensor(
                                    sSFB_moe_p2, cutlass.Uint8,
                                )
                                sX_p2_sf_u8 = cute.recast_tensor(
                                    sX_p2_sf_scratch, cutlass.Uint8,
                                )
                                for sbb in cutlass.range_constexpr(
                                    P2_NUM_SF_BLOCKS_PER_KT
                                ):
                                    sf_idx_global_p2 = Int32(kt) * Int32(
                                        P2_NUM_SF_BLOCKS_PER_KT
                                    ) + Int32(sbb)
                                    sf_byte_p2 = sX_p2_sf_u8[sf_idx_global_p2]
                                    i_n_p2 = lane_p2_b
                                    for j_n in cutlass.range_constexpr(4):
                                        sSFB_moe_p2_u8[
                                            (((i_n_p2, j_n), 0),
                                             0, sbb, ab_empty_p2.index)
                                        ] = sf_byte_p2
                                cute.arch.fence_view_async_shared()
                                cute.copy(
                                    tma_atom_wd_moe_p2,
                                    tWdGWd_p2[(None, m_cluster_idx_p2, kt)],
                                    tWdSWd_p2[(None, ab_empty_p2.index)],
                                    tma_bar_ptr=ab_empty_p2.barrier,
                                    mcast_mask=tma_mcast_mask_kv,
                                )
                                cute.copy(
                                    tma_atom_sfa_d_moe_p2,
                                    tSFAdG_p2[(None, m_cluster_idx_p2, kt)],
                                    tSFAdS_p2[(None, ab_empty_p2.index)],
                                    tma_bar_ptr=ab_empty_p2.barrier,
                                    mcast_mask=tma_mcast_mask_kv,
                                )
                            p2_ab_producer.tail()

                        elif warp_idx_p2 == Int32(mma_warp_id_ce):
                            if is_leader_cta:
                                tmem_ptr_p2 = tmem_alloc.retrieve_ptr(Float32)
                                tCtAcc_p2_d = cute.make_tensor(
                                    tmem_ptr_p2 + P2_TMEM_OFF_D,
                                    tCtAcc_p2_fake.layout,
                                )
                                sfa_d_tmem_ptr_p2 = cute.recast_ptr(
                                    tmem_ptr_p2 + P2_TMEM_OFF_SFA_D,
                                    dtype=cutlass.Float8E8M0FNU,
                                )
                                sfb_tmem_ptr_p2 = cute.recast_ptr(
                                    tmem_ptr_p2 + P2_TMEM_OFF_SFB,
                                    dtype=cutlass.Float8E8M0FNU,
                                )
                                tCtSFA_d_p2 = cute.make_tensor(
                                    sfa_d_tmem_ptr_p2, tCtSFA_layout_p2,
                                )
                                tCtSFB_p2 = cute.make_tensor(
                                    sfb_tmem_ptr_p2, tCtSFB_layout_p2,
                                )
                                copy_atom_s2t_sf_p2 = cute.make_copy_atom(
                                    tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO),
                                    cutlass.Float8E8M0FNU,
                                )
                                sSFA_d_compact_p2 = cute.filter_zeros(
                                    sSFA_d_moe_p2,
                                )
                                sSFB_compact_p2 = cute.filter_zeros(sSFB_moe_p2)
                                tCtSFA_d_compact_p2 = cute.filter_zeros(
                                    tCtSFA_d_p2,
                                )
                                tCtSFB_compact_p2 = cute.filter_zeros(tCtSFB_p2)
                                tiled_copy_s2t_sfa_d_p2 = tcgen05.make_s2t_copy(
                                    copy_atom_s2t_sf_p2, tCtSFA_d_compact_p2,
                                )
                                tiled_copy_s2t_sfb_p2 = tcgen05.make_s2t_copy(
                                    copy_atom_s2t_sf_p2, tCtSFB_compact_p2,
                                )
                                thr_s2t_sfa_d_p2 = tiled_copy_s2t_sfa_d_p2.get_slice(0)
                                thr_s2t_sfb_p2 = tiled_copy_s2t_sfb_p2.get_slice(0)
                                tCsSFA_d_s2t_p2_ = thr_s2t_sfa_d_p2.partition_S(
                                    sSFA_d_compact_p2,
                                )
                                tCsSFB_s2t_p2_ = thr_s2t_sfb_p2.partition_S(
                                    sSFB_compact_p2,
                                )
                                tCsSFA_d_s2t_p2 = tcgen05.get_s2t_smem_desc_tensor(
                                    tiled_copy_s2t_sfa_d_p2, tCsSFA_d_s2t_p2_,
                                )
                                tCsSFB_s2t_p2 = tcgen05.get_s2t_smem_desc_tensor(
                                    tiled_copy_s2t_sfb_p2, tCsSFB_s2t_p2_,
                                )
                                tCtSFA_d_s2t_p2 = thr_s2t_sfa_d_p2.partition_D(
                                    tCtSFA_d_compact_p2,
                                )
                                tCtSFB_s2t_p2 = thr_s2t_sfb_p2.partition_D(
                                    tCtSFB_compact_p2,
                                )
                                num_kblocks_p2 = cute.size(tCrWd_p2, mode=[2])
                                acc_p2_empty = p2_acc_producer.acquire_and_advance()
                                tiled_mma_moe_p2_fp8.set(
                                    tcgen05.Field.ACCUMULATE, False,
                                )
                                for kt in cutlass.range(
                                    P2_NUM_K_TILES, unroll=1,
                                ):
                                    ab_full_p2 = p2_ab_consumer.wait_and_advance()
                                    stage_coord_p2 = (
                                        None, None, None, None,
                                        ab_full_p2.index,
                                    )
                                    cute.copy(
                                        tiled_copy_s2t_sfa_d_p2,
                                        tCsSFA_d_s2t_p2[stage_coord_p2],
                                        tCtSFA_d_s2t_p2,
                                    )
                                    cute.copy(
                                        tiled_copy_s2t_sfb_p2,
                                        tCsSFB_s2t_p2[stage_coord_p2],
                                        tCtSFB_s2t_p2,
                                    )
                                    for kb in cutlass.range_constexpr(
                                        num_kblocks_p2
                                    ):
                                        sf_kb_coord_p2 = (None, None, kb)
                                        tiled_mma_moe_p2_fp8.set(
                                            tcgen05.Field.SFA,
                                            tCtSFA_d_p2[sf_kb_coord_p2].iterator,
                                        )
                                        tiled_mma_moe_p2_fp8.set(
                                            tcgen05.Field.SFB,
                                            tCtSFB_p2[sf_kb_coord_p2].iterator,
                                        )
                                        cute.gemm(
                                            tiled_mma_moe_p2_fp8,
                                            tCtAcc_p2_d,
                                            tCrWd_p2[
                                                (None, None, kb, ab_full_p2.index)
                                            ],
                                            tCrX_p2[
                                                (None, None, kb, ab_full_p2.index)
                                            ],
                                            tCtAcc_p2_d,
                                        )
                                        tiled_mma_moe_p2_fp8.set(
                                            tcgen05.Field.ACCUMULATE, True,
                                        )
                                    ab_full_p2.release()
                                acc_p2_empty.commit()
                            p2_acc_producer.tail()

                        elif warp_idx_p2 < Int32(num_epi_warps_ce):
                            tmem_ptr_p2_epi = tmem_alloc.retrieve_ptr(Float32)
                            tCtAcc_p2_d_epi = cute.make_tensor(
                                tmem_ptr_p2_epi + P2_TMEM_OFF_D,
                                tCtAcc_p2_fake.layout,
                            )
                            tStD_acc_p2 = tCtAcc_p2_d_epi[(None, None), 0, 0]
                            copy_atom_p2_t2r = cute.make_copy_atom(
                                tcgen05.Ld32x32bOp(
                                    tcgen05.Repetition.x32,
                                    tcgen05.Pack.NONE,
                                ),
                                Float32,
                            )
                            tiled_copy_p2_t2r = tcgen05.make_tmem_copy(
                                copy_atom_p2_t2r, tStD_acc_p2,
                            )
                            thr_copy_p2_t2r = tiled_copy_p2_t2r.get_slice(tidx)
                            tStD_t2r_p2 = thr_copy_p2_t2r.partition_S(tStD_acc_p2)
                            cP2 = cute.make_identity_tensor(
                                (P2_CTA_TILE_M, P2_MMA_N),
                            )
                            tP2cP2_t2r = thr_copy_p2_t2r.partition_D(cP2)
                            tP2rD_t2r = cute.make_rmem_tensor(
                                tP2cP2_t2r.shape, Float32,
                            )
                            acc_p2_full = p2_acc_consumer.wait_and_advance()
                            cute.copy(tiled_copy_p2_t2r, tStD_t2r_p2, tP2rD_t2r)
                            cute.arch.fence_view_async_tmem_load()
                            acc_p2_full.release()
                            for i in cutlass.range_constexpr(
                                cute.size(tP2rD_t2r)
                            ):
                                coord_i_p2 = tP2cP2_t2r[i]
                                if coord_i_p2[1] == Int32(0):
                                    h_out_p2 = (
                                        m_cluster_idx_p2 * Int32(P2_CLUSTER_TILE_M)
                                        + cta_rank_in_cluster * Int32(P2_CTA_TILE_M)
                                        + coord_i_p2[0]
                                    )
                                    if h_out_p2 < Int32(H):
                                        d_val_p2 = tP2rD_t2r[i]
                                        if layer == Int32(0) or layer == Int32(1):
                                            if cluster_id == Int32(0):
                                                if cta_rank_in_cluster == Int32(0):
                                                    if (tidx % Int32(32)) == Int32(0):
                                                        llvm.inline_asm(
                                                            None,
                                                            [d_val_p2.ir_value()],
                                                            "",
                                                            "f",
                                                            has_side_effects=True,
                                                            asm_dialect=0,
                                                        )
                                        p2_add_f32 = weight_k * d_val_p2
                                        p2_acc_addr = (
                                            mMoeRoutedAccF32.iterator.toint()
                                            + (Int64(layer) * Int64(H)
                                               + Int64(h_out_p2)) * Int64(4)
                                        )
                                        llvm.inline_asm(
                                            None,
                                            [
                                                p2_acc_addr.ir_value(),
                                                p2_add_f32.ir_value(),
                                            ],
                                            "red.global.add.f32 [$0], $1;",
                                            "l,f",
                                            has_side_effects=True,
                                            is_align_stack=False,
                                        )
                pass

                cute.arch.barrier()
                for p2_bk in cutlass.range_constexpr(P2_SLOTS_PER_PASS):
                    if tidx == 0:
                        cute.arch.atomic_add(
                            mCounter.iterator, Int32(1), sem="release", scope="gpu",
                        )
                    if bidx == 0 and tidx == 0:
                        mStageTs[
                            layer,
                            Int32(17 + sub_pass * P2_SLOTS_PER_PASS + p2_bk),
                        ] = _clock()
                cute.arch.barrier()

            if tidx == 0:
                expected_p2_pass = Int32((SYNCS_PER_LAYER * layer + 9 + 2 * K_topk) * num_sms)
                done_p2_pass = Int32(0)
                while done_p2_pass < expected_p2_pass:
                    done_p2_pass = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            cute.arch.fence_acq_rel_sys()
            if tidx == Int32(0):
                llvm.inline_asm(
                    None,
                    [ep_sync_mc_addr.ir_value()],
                    "multimem.red.release.sys.global.add.u32 [$0], 1;",
                    "l",
                    has_side_effects=True,
                    is_align_stack=False,
                )


            cute.arch.barrier()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected_q1 = Int32((SYNCS_PER_LAYER * layer + 9 + 2 * K_topk + 1) * num_sms)
                done_q1 = Int32(0)
                while done_q1 < expected_q1:
                    done_q1 = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 23] = _clock()

            warp_idx_q2 = cute.arch.make_warp_uniform(cute.arch.warp_idx())

            q2_k_split = cluster_id % Int32(self.Q2_NUM_K_SPLITS)
            shared_cluster_id_q2 = cluster_id // Int32(self.Q2_NUM_K_SPLITS)
            if cluster_id < Int32(self.Q2_FP8_NUM_ACTIVE_CLUSTERS * self.Q2_NUM_K_SPLITS):
              Q2F_CTA_TILE_M     = const_expr(self.Q2_FP8_CTA_TILE_M)
              Q2F_CLUSTER_TILE_M = const_expr(self.Q2_FP8_CLUSTER_TILE_M)
              Q2F_K_CHUNK        = const_expr(self.Q2_FP8_K_CHUNK)
              Q2F_NUM_K_TILES    = const_expr(self.Q2_FP8_NUM_K_TILES_MMA)
              Q2F_NUM_K_TILES_SPLIT = const_expr(self.Q2_FP8_NUM_K_TILES_SPLIT)
              q2_n_split = Int32(cutlass.select_(
                  layer == Int32(0),
                  Int32(Q2F_NUM_K_TILES_SPLIT),
                  Int32(Q2F_NUM_K_TILES_SPLIT // 2),
              ))
              Q2F_NUM_K_SPLITS   = const_expr(self.Q2_NUM_K_SPLITS)
              mQ2Acc_sk = cute.make_tensor(
                  mOPartial.iterator,
                  cute.make_layout((Q2F_NUM_K_SPLITS, H), stride=(H, 1)),
              )
              Q2F_AB_STAGES      = const_expr(self.Q2_FP8_AB_STAGES)
              Q2F_MMA_N          = const_expr(self.Q2_FP8_MMA_N)
              Q2F_X_BROADCAST_N  = const_expr(self.Q2_FP8_MMA_N // self.CLUSTER_SIZE)
              Q2F_SF_VEC         = const_expr(self.Q2_SF_VEC_SIZE)
              Q2F_NUM_SF_BLOCKS_PER_KT = const_expr(Q2F_K_CHUNK // Q2F_SF_VEC)
              Q2F_TOTAL_SF_BLOCKS_K    = const_expr(I_shared // Q2F_SF_VEC)
              q2f_mma_tiler_c    = const_expr(self.Q2_FP8_MMA_TILER)
              Q2F_TMEM_COLS_ACC  = const_expr(Q2F_MMA_N)
              Q2F_TMEM_COLS_SF   = const_expr(16)

              sWd_q2_fp8 = cute.make_tensor(
                  cute.recast_ptr(
                      sQ_mma.iterator,
                      sWd_q2_fp8_layout_staged.inner,
                      dtype=cutlass.Float8E4M3FN,
                  ),
                  sWd_q2_fp8_layout_staged.outer,
              )
              sX_q2_fp8 = cute.make_tensor(
                  cute.recast_ptr(
                      sV0_mma.iterator,
                      sX_q2_fp8_layout_staged.inner,
                      dtype=cutlass.Float8E4M3FN,
                  ),
                  sX_q2_fp8_layout_staged.outer,
              )
              sInterTmp_quant_scratch = cute.make_tensor(
                  cute.recast_ptr(sV1_mma.iterator, dtype=cutlass.Float8E4M3FN),
                  cute.make_layout((I_shared,), stride=(1,)),
              )
              sSFA_down_q2 = cute.make_tensor(
                  cute.recast_ptr(sK_mma.iterator, dtype=cutlass.Float8E8M0FNU),
                  sSFA_down_q2_layout_staged,
              )
              sSFB_q2 = cute.make_tensor(
                  cute.recast_ptr(sP_mma.iterator, dtype=cutlass.Float8E8M0FNU),
                  sSFB_q2_layout_staged,
              )
              sInterTmp_sf_scratch = cute.make_tensor(
                  cute.recast_ptr(sRowMax_mma.iterator, dtype=cutlass.Float8E8M0FNU),
                  cute.make_layout((I_shared // Q2F_SF_VEC,), stride=(1,)),
              )

              warp_id_q2f = tidx // Int32(32)
              lane_q2f = tidx % Int32(32)
              num_warps_quant_q2 = const_expr(8)
              Q2F_LOGICAL_BLOCK_K = const_expr(128)
              Q2F_SUB_PER_LOG = const_expr(Q2F_LOGICAL_BLOCK_K // Q2F_SF_VEC)
              num_logical_blocks_q2 = const_expr(I_shared // Q2F_LOGICAL_BLOCK_K)
              logblks_per_warp_q2 = const_expr(num_logical_blocks_q2 // num_warps_quant_q2)
              logblks_per_warp_q2_dyn = Int32(cutlass.select_(
                  layer == Int32(0),
                  Int32(logblks_per_warp_q2),
                  Int32(logblks_per_warp_q2 // 2),
              ))
              if warp_id_q2f < Int32(num_warps_quant_q2):
                  for lb in cutlass.range(logblks_per_warp_q2_dyn, unroll=1):
                      log_block_idx_q2 = (
                          warp_id_q2f * logblks_per_warp_q2_dyn + lb
                      )
                      local_amax_q2 = Float32(0.0)
                      for s in cutlass.range_constexpr(Q2F_SUB_PER_LOG):
                          k_pos_q2 = (
                              log_block_idx_q2 * Int32(Q2F_LOGICAL_BLOCK_K)
                              + Int32(s) * Int32(Q2F_SF_VEC) + lane_q2f
                          )
                          x_bf16_q2 = mInterTmp[Int32(Q1_INTER_OFFSET) + k_pos_q2]
                          absx_q2 = cute.math.absf(x_bf16_q2.to(Float32))
                          local_amax_q2 = cutlass.max(local_amax_q2, absx_q2)
                      amax_q2 = cute.arch.warp_reduction_max(
                          local_amax_q2, threads_in_group=32,
                      )
                      ratio_q2 = amax_q2 * Float32(1.0 / 448.0)
                      bits_i32_q2 = ratio_q2.bitcast(cutlass.Int32)
                      exp_field_q2 = (bits_i32_q2 >> Int32(23)) & Int32(0xFF)
                      mantissa_field_q2 = bits_i32_q2 & Int32(0x7FFFFF)
                      mantissa_nonzero_q2 = Int32(0)
                      if mantissa_field_q2 != Int32(0):
                          mantissa_nonzero_q2 = Int32(1)
                      exp_ceil_q2 = exp_field_q2 + mantissa_nonzero_q2
                      exp_clamped_q2 = cutlass.min(
                          cutlass.max(exp_ceil_q2, Int32(0)), Int32(254),
                      )
                      scale_bits_q2 = exp_clamped_q2 << Int32(23)
                      scale_f32_q2 = scale_bits_q2.bitcast(Float32)
                      inv_exp_biased_q2 = Int32(254) - exp_clamped_q2
                      inv_scale_bits_q2 = inv_exp_biased_q2 << Int32(23)
                      inv_scale_q2 = inv_scale_bits_q2.bitcast(Float32)
                      xf_q2 = cute.make_fragment(Q2F_SUB_PER_LOG, Float32)
                      for s in cutlass.range_constexpr(Q2F_SUB_PER_LOG):
                          k_pos_q2 = (
                              log_block_idx_q2 * Int32(Q2F_LOGICAL_BLOCK_K)
                              + Int32(s) * Int32(Q2F_SF_VEC) + lane_q2f
                          )
                          x_bf16_q2 = mInterTmp[Int32(Q1_INTER_OFFSET) + k_pos_q2]
                          x_scaled_q2 = x_bf16_q2.to(Float32) * inv_scale_q2
                          xf_q2[s] = cutlass.min(
                              cutlass.max(x_scaled_q2, Float32(-448.0)),
                              Float32(448.0),
                          )
                      xf8_q2 = cute.make_fragment(Q2F_SUB_PER_LOG, cutlass.Float8E4M3FN)
                      xf8_q2.store(xf_q2.load().to(cutlass.Float8E4M3FN))
                      for s in cutlass.range_constexpr(Q2F_SUB_PER_LOG):
                          k_pos_q2 = (
                              log_block_idx_q2 * Int32(Q2F_LOGICAL_BLOCK_K)
                              + Int32(s) * Int32(Q2F_SF_VEC) + lane_q2f
                          )
                          sInterTmp_quant_scratch[k_pos_q2] = xf8_q2[s]
                      if lane_q2f == Int32(0):
                          sf_u8_q2 = exp_clamped_q2.to(cutlass.Uint8)
                          sf_byte_q2 = sf_u8_q2.bitcast(cutlass.Float8E8M0FNU)
                          for s in cutlass.range_constexpr(Q2F_SUB_PER_LOG):
                              sInterTmp_sf_scratch[
                                  log_block_idx_q2 * Int32(Q2F_SUB_PER_LOG) + Int32(s)
                              ] = sf_byte_q2
              cute.arch.barrier()
              if bidx == 0 and tidx == 0:
                  mStageTs[layer, 41] = _clock()

              tma_warp_group_q2f = pipeline.CooperativeGroup(pipeline.Agent.Thread)
              mma_consumer_group_q2f = pipeline.CooperativeGroup(
                  pipeline.Agent.Thread, size=1,
              )
              acc_producer_group_q2f = pipeline.CooperativeGroup(pipeline.Agent.Thread)
              acc_consumer_group_q2f = pipeline.CooperativeGroup(
                  pipeline.Agent.Thread,
                  size=threads_in_epi_ce * cluster_size_ce,
              )
              bytes_q2f_per_cta = const_expr(
                  Q2F_CTA_TILE_M * Q2F_K_CHUNK
                  + Q2F_CTA_TILE_M * Q2F_K_CHUNK // Q2F_SF_VEC
              )
              bytes_q2f_ab_per_stage = const_expr(
                  cluster_size_ce * bytes_q2f_per_cta
              )
              q2f_ab_producer, q2f_ab_consumer = pipeline.PipelineTmaUmma.create(
                  barrier_storage=q1_ab_mma_bars.iterator,
                  num_stages=Q2F_AB_STAGES,
                  producer_group=tma_warp_group_q2f,
                  consumer_group=mma_consumer_group_q2f,
                  tx_count=bytes_q2f_ab_per_stage,
                  cta_layout_vmnk=cta_layout_vmnk_mma,
              ).make_participants()
              q2f_acc_producer, q2f_acc_consumer = pipeline.PipelineUmmaAsync.create(
                  barrier_storage=q1_acc_mma_bars.iterator,
                  num_stages=1,
                  producer_group=acc_producer_group_q2f,
                  consumer_group=acc_consumer_group_q2f,
                  cta_layout_vmnk=cta_layout_vmnk_mma,
              ).make_participants()

              gWd_q2f = cute.local_tile(
                  tma_tensor_wd_q2_fp8_mma,
                  cute.slice_(q2f_mma_tiler_c, (None, 0, None)),
                  (None, None, Int32(layer)),
              )
              gSFA_q2f = cute.local_tile(
                  tma_tensor_sfa_down_q2_mma,
                  cute.slice_(q2f_mma_tiler_c, (None, 0, None)),
                  (None, None, Int32(layer)),
              )

              thr_mma_q2f = tiled_mma_q2_fp8.get_slice(mma_tile_coord_v)
              tCgWd_f = thr_mma_q2f.partition_A(gWd_q2f)
              tCgSFA_f = thr_mma_q2f.partition_A(gSFA_q2f)

              tCrWd_f = tiled_mma_q2_fp8.make_fragment_A(sWd_q2_fp8)
              tCrX_q2f = tiled_mma_q2_fp8.make_fragment_B(sX_q2_fp8)

              a_cta_layout_q2f = cute.make_layout(
                  cute.size(cta_layout_vmnk_mma, mode=[2]),
              )
              tWdS_f, tWdG_f = cute.nvgpu.cpasync.tma_partition(
                  tma_atom_wd_q2_fp8_mma,
                  cta_in_cluster_coord_vmnk[2],
                  a_cta_layout_q2f,
                  cute.group_modes(sWd_q2_fp8, 0, 3),
                  cute.group_modes(tCgWd_f, 0, 3),
              )
              tSFAS_f, tSFAG_f = cute.nvgpu.cpasync.tma_partition(
                  tma_atom_sfa_down_q2_mma,
                  cta_in_cluster_coord_vmnk[2],
                  a_cta_layout_q2f,
                  cute.group_modes(sSFA_down_q2, 0, 3),
                  cute.group_modes(tCgSFA_f, 0, 3),
              )
              tSFAS_f = cute.filter_zeros(tSFAS_f)
              tSFAG_f = cute.filter_zeros(tSFAG_f)

              acc_shape_q2f = tiled_mma_q2_fp8.partition_shape_C(
                  (q2f_mma_tiler_c[0], q2f_mma_tiler_c[1]),
              )
              tCtAcc_q2f_fake = tiled_mma_q2_fp8.make_fragment_C(acc_shape_q2f)

              Q2F_TMEM_OFF_ACC = const_expr(0)
              Q2F_TMEM_OFF_SFA = const_expr(Q2F_TMEM_COLS_ACC)
              Q2F_TMEM_OFF_SFB = const_expr(
                  Q2F_TMEM_COLS_ACC + Q2F_TMEM_COLS_SF,
              )

              sfa_slice_q2f = cute.slice_(
                  sSFA_down_q2_layout_staged, (None, None, None, 0),
              )
              sfb_slice_q2f = cute.slice_(
                  sSFB_q2_layout_staged, (None, None, None, 0),
              )
              tCtSFA_layout_q2f = blockscaled_utils.make_tmem_layout_sfa(
                  tiled_mma_q2_fp8, q2f_mma_tiler_c, Q2F_SF_VEC, sfa_slice_q2f,
              )
              tCtSFB_layout_q2f = blockscaled_utils.make_tmem_layout_sfb(
                  tiled_mma_q2_fp8, q2f_mma_tiler_c, Q2F_SF_VEC, sfb_slice_q2f,
              )

              if warp_idx_q2 == Int32(tma_warp_id_ce):
                  cpasync.prefetch_descriptor(tma_atom_wd_q2_fp8_mma)
                  cpasync.prefetch_descriptor(tma_atom_sfa_down_q2_mma)
                  for kt_local in cutlass.range(q2_n_split, unroll=1):
                      kt = q2_k_split * q2_n_split + kt_local
                      ab_empty_f = q2f_ab_producer.acquire_and_advance()
                      k_base_f = kt * Int32(Q2F_K_CHUNK)
                      sX_q2_fp8_u32 = cute.recast_tensor(sX_q2_fp8, cutlass.Uint32)
                      sX_quant_u32_q2 = cute.recast_tensor(
                          sInterTmp_quant_scratch, cutlass.Uint32,
                      )
                      atom_k_u32_lane = lane_q2f % Int32(8)
                      kk_lane = lane_q2f // Int32(8)
                      xfp8_u32 = sX_quant_u32_q2[(k_base_f >> 2) + lane_q2f]
                      for nn in cutlass.range_constexpr(Q2F_X_BROADCAST_N):
                          sX_q2_fp8_u32[
                              ((nn, atom_k_u32_lane), 0, kk_lane, ab_empty_f.index)
                          ] = xfp8_u32
                      sSFB_q2_u8 = cute.recast_tensor(sSFB_q2, cutlass.Uint8)
                      sX_sf_u8_q2 = cute.recast_tensor(
                          sInterTmp_sf_scratch, cutlass.Uint8,
                      )
                      for sbb in cutlass.range_constexpr(Q2F_NUM_SF_BLOCKS_PER_KT):
                          sf_idx_global_q2 = (
                              Int32(kt) * Int32(Q2F_NUM_SF_BLOCKS_PER_KT)
                              + Int32(sbb)
                          )
                          sf_byte_b_q2 = sX_sf_u8_q2[sf_idx_global_q2]
                          i_n_q2 = lane_q2f
                          for j_n in cutlass.range_constexpr(4):
                              sSFB_q2_u8[
                                  (((i_n_q2, j_n), 0), 0, sbb, ab_empty_f.index)
                              ] = sf_byte_b_q2
                      cute.arch.fence_view_async_shared()
                      cute.copy(
                          tma_atom_wd_q2_fp8_mma,
                          tWdG_f[(None, shared_cluster_id_q2, kt)],
                          tWdS_f[(None, ab_empty_f.index)],
                          tma_bar_ptr=ab_empty_f.barrier,
                          mcast_mask=tma_mcast_mask_kv,
                      )
                      cute.copy(
                          tma_atom_sfa_down_q2_mma,
                          tSFAG_f[(None, shared_cluster_id_q2, kt)],
                          tSFAS_f[(None, ab_empty_f.index)],
                          tma_bar_ptr=ab_empty_f.barrier,
                          mcast_mask=tma_mcast_mask_kv,
                      )
                  q2f_ab_producer.tail()

              elif warp_idx_q2 == Int32(mma_warp_id_ce):
                  if is_leader_cta:
                      tmem_ptr_q2f = tmem_alloc.retrieve_ptr(Float32)
                      tCtAcc_q2f = cute.make_tensor(
                          tmem_ptr_q2f + Q2F_TMEM_OFF_ACC,
                          tCtAcc_q2f_fake.layout,
                      )
                      sfa_tmem_ptr_q2 = cute.recast_ptr(
                          tmem_ptr_q2f + Q2F_TMEM_OFF_SFA,
                          dtype=cutlass.Float8E8M0FNU,
                      )
                      sfb_tmem_ptr_q2 = cute.recast_ptr(
                          tmem_ptr_q2f + Q2F_TMEM_OFF_SFB,
                          dtype=cutlass.Float8E8M0FNU,
                      )
                      tCtSFA_f_q2 = cute.make_tensor(
                          sfa_tmem_ptr_q2, tCtSFA_layout_q2f,
                      )
                      tCtSFB_f_q2 = cute.make_tensor(
                          sfb_tmem_ptr_q2, tCtSFB_layout_q2f,
                      )

                      copy_atom_s2t_sf_q2 = cute.make_copy_atom(
                          tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO),
                          cutlass.Float8E8M0FNU,
                      )
                      sSFA_down_compact_q2 = cute.filter_zeros(sSFA_down_q2)
                      sSFB_compact_q2 = cute.filter_zeros(sSFB_q2)
                      tCtSFA_compact_q2 = cute.filter_zeros(tCtSFA_f_q2)
                      tCtSFB_compact_q2 = cute.filter_zeros(tCtSFB_f_q2)
                      tiled_copy_s2t_sfa_q2 = tcgen05.make_s2t_copy(
                          copy_atom_s2t_sf_q2, tCtSFA_compact_q2,
                      )
                      tiled_copy_s2t_sfb_q2 = tcgen05.make_s2t_copy(
                          copy_atom_s2t_sf_q2, tCtSFB_compact_q2,
                      )
                      thr_s2t_sfa_q2 = tiled_copy_s2t_sfa_q2.get_slice(0)
                      thr_s2t_sfb_q2 = tiled_copy_s2t_sfb_q2.get_slice(0)
                      tCsSFA_s2t_q2_ = thr_s2t_sfa_q2.partition_S(sSFA_down_compact_q2)
                      tCsSFB_s2t_q2_ = thr_s2t_sfb_q2.partition_S(sSFB_compact_q2)
                      tCsSFA_s2t_q2 = tcgen05.get_s2t_smem_desc_tensor(
                          tiled_copy_s2t_sfa_q2, tCsSFA_s2t_q2_,
                      )
                      tCsSFB_s2t_q2 = tcgen05.get_s2t_smem_desc_tensor(
                          tiled_copy_s2t_sfb_q2, tCsSFB_s2t_q2_,
                      )
                      tCtSFA_s2t_q2 = thr_s2t_sfa_q2.partition_D(tCtSFA_compact_q2)
                      tCtSFB_s2t_q2 = thr_s2t_sfb_q2.partition_D(tCtSFB_compact_q2)

                      num_kblocks_q2f = cute.size(tCrWd_f, mode=[2])
                      acc_q2f_empty = q2f_acc_producer.acquire_and_advance()
                      tiled_mma_q2_fp8.set(tcgen05.Field.ACCUMULATE, False)
                      for kt_local in cutlass.range(q2_n_split, unroll=1):
                          kt = q2_k_split * q2_n_split + kt_local
                          ab_full_f = q2f_ab_consumer.wait_and_advance()
                          stage_coord_q2 = (None, None, None, None, ab_full_f.index)
                          cute.copy(
                              tiled_copy_s2t_sfa_q2,
                              tCsSFA_s2t_q2[stage_coord_q2],
                              tCtSFA_s2t_q2,
                          )
                          cute.copy(
                              tiled_copy_s2t_sfb_q2,
                              tCsSFB_s2t_q2[stage_coord_q2],
                              tCtSFB_s2t_q2,
                          )
                          for kb in cutlass.range_constexpr(num_kblocks_q2f):
                              sf_kb_coord_q2 = (None, None, kb)
                              tiled_mma_q2_fp8.set(
                                  tcgen05.Field.SFA,
                                  tCtSFA_f_q2[sf_kb_coord_q2].iterator,
                              )
                              tiled_mma_q2_fp8.set(
                                  tcgen05.Field.SFB,
                                  tCtSFB_f_q2[sf_kb_coord_q2].iterator,
                              )
                              cute.gemm(
                                  tiled_mma_q2_fp8,
                                  tCtAcc_q2f,
                                  tCrWd_f[(None, None, kb, ab_full_f.index)],
                                  tCrX_q2f[(None, None, kb, ab_full_f.index)],
                                  tCtAcc_q2f,
                              )
                              tiled_mma_q2_fp8.set(tcgen05.Field.ACCUMULATE, True)
                          ab_full_f.release()
                      acc_q2f_empty.commit()
                  q2f_acc_producer.tail()

              elif warp_idx_q2 < Int32(num_epi_warps_ce):
                  tmem_ptr_q2f_epi = tmem_alloc.retrieve_ptr(Float32)
                  tCtAcc_q2f_epi = cute.make_tensor(
                      tmem_ptr_q2f_epi + Q2F_TMEM_OFF_ACC,
                      tCtAcc_q2f_fake.layout,
                  )
                  tStAcc_q2f = tCtAcc_q2f_epi[(None, None), 0, 0]
                  copy_atom_q2f_t2r = cute.make_copy_atom(
                      tcgen05.Ld32x32bOp(
                          tcgen05.Repetition.x16, tcgen05.Pack.NONE,
                      ),
                      Float32,
                  )
                  tiled_copy_q2f_t2r = tcgen05.make_tmem_copy(
                      copy_atom_q2f_t2r, tStAcc_q2f,
                  )
                  thr_copy_q2f_t2r = tiled_copy_q2f_t2r.get_slice(tidx)
                  tStAcc_t2r_q2f = thr_copy_q2f_t2r.partition_S(tStAcc_q2f)

                  cQ2f = cute.make_identity_tensor(
                      (Q2F_CTA_TILE_M, Q2F_MMA_N),
                  )
                  tQ2fcQ2f_t2r = thr_copy_q2f_t2r.partition_D(cQ2f)
                  tQ2frAcc_t2r = cute.make_rmem_tensor(
                      tQ2fcQ2f_t2r.shape, Float32,
                  )

                  acc_q2f_full = q2f_acc_consumer.wait_and_advance()
                  cute.copy(tiled_copy_q2f_t2r, tStAcc_t2r_q2f, tQ2frAcc_t2r)
                  cute.arch.fence_view_async_tmem_load()
                  acc_q2f_full.release()

                  for i in cutlass.range_constexpr(cute.size(tQ2frAcc_t2r)):
                      coord_i_q2 = tQ2fcQ2f_t2r[i]
                      row_in_cta_q2 = coord_i_q2[0]
                      col_q2 = coord_i_q2[1]
                      if col_q2 == Int32(0):
                          h_out_f = (
                              shared_cluster_id_q2 * Int32(Q2F_CLUSTER_TILE_M)
                              + cta_rank_in_cluster * Int32(Q2F_CTA_TILE_M)
                              + row_in_cta_q2
                          )
                          if h_out_f < Int32(H):
                              mQ2Acc_sk[q2_k_split, h_out_f] = tQ2frAcc_t2r[i]

            Q2_ACTIVE_CTAS = const_expr(
                self.Q2_FP8_NUM_ACTIVE_CLUSTERS * self.Q2_NUM_K_SPLITS
                * self.CLUSTER_SIZE
            )
            if bidx >= Int32(Q2_ACTIVE_CTAS):
                hide_base = (
                    (bidx - Int32(Q2_ACTIVE_CTAS)) * num_threads + tidx
                ) * vec_size
                if hide_base + vec_size <= H:
                    for v in cutlass.range_constexpr(vec_size):
                        kkh = hide_base + v
                        mMoeRoutedSymmL[kkh] = mMoeRoutedAccF32[layer, kkh].to(
                            mH_in.element_type
                        )

            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 42] = _clock()
            if tidx == 0:
                cute.arch.atomic_add(
                    mCounter.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected_q2 = Int32((SYNCS_PER_LAYER * layer + 9 + 2 * K_topk + 2) * num_sms)
                done_q2 = Int32(0)
                while done_q2 < expected_q2:
                    done_q2 = cute.arch.atomic_add(
                        mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                    )
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:

                mStageTs[layer, 10] = _clock()
            if tidx == Int32(0):
                ep_tp_target = (cache_pos + Int32(1)) * Int32(self.EP_SIZE)
                done_ep = Int32(0)
                while done_ep < ep_tp_target:
                    val_loaded = llvm.inline_asm(
                        _mlir_ir.IntegerType.get_signless(32),
                        [ep_sync_local_addr.ir_value()],
                        "ld.acquire.sys.global.L1::no_allocate.u32 $0, [$1];",
                        "=r,l",
                        has_side_effects=True,
                        is_align_stack=False,
                    )
                    done_ep = Int32(val_loaded)
                cute.arch.fence_acq_rel_sys()
            cute.arch.barrier()

            if bidx == 0 and tidx == 0:
                mStageTs[layer, 24] = _clock()

            mQ2Acc_sk_r = cute.make_tensor(
                mOPartial.iterator,
                cute.make_layout((self.Q2_NUM_K_SPLITS, H), stride=(H, 1)),
            )
            moe_resid_bcast_ptr = (
                mAttnProjReduced.iterator + Int64(layer) * Int64(H)
            ).align(16)
            mMoeResidBcast = cute.make_tensor(
                moe_resid_bcast_ptr, cute.make_layout((H,), stride=(1,)),
            )
            NUM_VB_TOTAL_R = const_expr((H + vec_size - 1) // vec_size)
            global_vb_r = bidx * Int32(num_threads) + tidx
            if global_vb_r < Int32(NUM_VB_TOTAL_R):
                base = global_vb_r * Int32(vec_size)
                mc_addr_i64 = moe_routed_symm_mc_base + Int64(base) * Int64(2)
                struct_ty = _mlir_ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>")
                result_struct = llvm.inline_asm(
                    struct_ty,
                    [mc_addr_i64.ir_value()],
                    "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {$0, $1, $2, $3}, [$4];",
                    "=r,=r,=r,=r,l",
                    has_side_effects=True,
                    is_align_stack=False,
                )
                i32_t = _mlir_ir.IntegerType.get_signless(32)
                packed_regs = (
                    llvm.extractvalue(i32_t, result_struct, [0]),
                    llvm.extractvalue(i32_t, result_struct, [1]),
                    llvm.extractvalue(i32_t, result_struct, [2]),
                    llvm.extractvalue(i32_t, result_struct, [3]),
                )
                for v_pair in cutlass.range_constexpr(vec_size // 2):
                    v = v_pair * 2
                    k = base + v
                    old0 = sCarry[k].to(Float32)
                    old1 = sCarry[k + 1].to(Float32)
                    sh0_sk = Float32(0.0)
                    sh1_sk = Float32(0.0)
                    for sk in cutlass.range_constexpr(self.Q2_NUM_K_SPLITS):
                        sh0_sk = sh0_sk + mQ2Acc_sk_r[sk, k]
                        sh1_sk = sh1_sk + mQ2Acc_sk_r[sk, k + 1]
                    shared0 = mMoeOutL[k].to(Float32) + sh0_sk
                    shared1 = mMoeOutL[k + 1].to(Float32) + sh1_sk
                    packed_i32 = cutlass.Int32(packed_regs[v_pair])
                    lo_bits = (packed_i32 & Int32(0xFFFF)).to(cutlass.Int16)
                    hi_bits = ((packed_i32 >> Int32(16)) & Int32(0xFFFF)).to(cutlass.Int16)
                    routed0 = lo_bits.bitcast(cutlass.BFloat16).to(Float32)
                    routed1 = hi_bits.bitcast(cutlass.BFloat16).to(Float32)
                    mMoeResidBcast[k] = (old0 + shared0 + routed0).to(mH_in.element_type)
                    mMoeResidBcast[k + 1] = (old1 + shared1 + routed1).to(mH_in.element_type)
            cute.arch.barrier()
            if tidx == 0:
                cute.arch.atomic_add(
                    mTpL1Sync.iterator, Int32(1), sem="release", scope="gpu",
                )
                expected_r = Int32((layer + Int32(1)) * Int32(num_sms))
                done_r = Int32(0)
                r_addr = mTpL1Sync.iterator.toint()
                while done_r < expected_r:
                    val_r = llvm.inline_asm(
                        _mlir_ir.IntegerType.get_signless(32),
                        [r_addr.ir_value()],
                        "ld.acquire.gpu.global.L1::no_allocate.u32 $0, [$1];",
                        "=r,l",
                        has_side_effects=True,
                        is_align_stack=False,
                    )
                    done_r = Int32(val_r)
            cute.arch.barrier()
            for vb in cutlass.range_constexpr(num_vec_blocks_h):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= H:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        sCarry[k] = mMoeResidBcast[k]
            cute.arch.barrier()
            if bidx == 0 and tidx == 0:
                mStageTs[layer, 25] = _clock()

        if bidx == Int32(0):
            for vb in cutlass.range_constexpr(num_vec_blocks_h):
                base = (vb * num_threads + tidx) * vec_size
                if base + vec_size <= H:
                    for v in cutlass.range_constexpr(vec_size):
                        k = base + v
                        mHFinal[k] = sCarry[k]
        V_LM = const_expr(self.V)
        outs_lm = const_expr((V_LM + num_sms - 1) // num_sms)
        assert H % num_threads == 0, (
            f"Stage T requires H ({H}) divisible by num_threads ({num_threads})"
        )

        cute.arch.barrier()

        partial_sq_T = Float32(0.0)
        for vb in cutlass.range_constexpr(num_vec_blocks_h):
            base = (vb * num_threads + tidx) * vec_size
            if base + vec_size <= H:
                for v in cutlass.range_constexpr(vec_size):
                    k = base + v
                    c_val = sCarry[k].to(Float32)
                    partial_sq_T = partial_sq_T + c_val * c_val
        partial_sq_T = cute.arch.warp_reduction(
            partial_sq_T, op=lambda a, b: a + b, threads_in_group=32,
        )
        if lane_id == 0:
            reduce_buf[warp_id] = partial_sq_T
        cute.arch.barrier()
        total_sq_T = Float32(0.0)
        for w in cutlass.range_constexpr(warps_per_row):
            total_sq_T = total_sq_T + reduce_buf[w]
        mean_sq_T = total_sq_T / Float32(H)
        rstd_T = cute.math.rsqrt(mean_sq_T + eps, fastmath=True)
        cute.arch.barrier()
        for vb in cutlass.range_constexpr(num_vec_blocks_h):
            base = (vb * num_threads + tidx) * vec_size
            if base + vec_size <= H:
                for v in cutlass.range_constexpr(vec_size):
                    k = base + v
                    c_val = sCarry[k].to(Float32)
                    g_val = mGammaFinal[k].to(Float32)
                    sCarry[k] = (c_val * rstd_T * g_val).to(mH_in.element_type)
        cute.arch.barrier()

        warp_outs = const_expr((outs_lm + warps_per_row - 1) // warps_per_row)
        per_warp_vb = const_expr((H + 32 * vec_size - 1) // (32 * vec_size))
        assert H % (32 * vec_size) == 0, (
            f"Stage T.1 v3 requires H ({H}) divisible by 32*vec_size "
            f"({32 * vec_size}) for the warp-coalesced LDG.E.128 pattern."
        )
        assert reduce_buf_slots >= 2 * warps_per_row, (
            f"Stage T.1 v3 needs reduce_buf_slots >= {2 * warps_per_row}, "
            f"got {reduce_buf_slots}."
        )

        copy_atom_w_lm = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mLmHead.element_type,
            num_bits_per_copy=128,
        )

        v_warp_base_T = bidx * Int32(outs_lm) + Int32(warp_id) * Int32(warp_outs)

        if lane_id == 0:
            reduce_buf[Int32(warp_id) * Int32(2)] = Float32(-3.4e38)
            reduce_buf[Int32(warp_id) * Int32(2) + Int32(1)] = (
                Int32(-1).bitcast(Float32)
            )
        cute.arch.barrier()

        for v_local in cutlass.range(warp_outs, unroll=1):
            v = v_warp_base_T + Int32(v_local)
            in_range = v < Int32(V_LM)
            acc_T = Float32(0.0)
            if in_range:
                w_row_off = Int64(v) * Int64(H)
                w_row_ptr = (mLmHead.iterator + w_row_off).align(16)
                for vb_w in cutlass.range_constexpr(per_warp_vb):
                    k_base = (vb_w * 32 + lane_id) * vec_size
                    w_vec = cute.make_rmem_tensor(
                        cute.make_layout((vec_size,), stride=(1,)),
                        mLmHead.element_type,
                    )
                    w_slice = cute.make_tensor(
                        (w_row_ptr + k_base).align(16),
                        cute.make_layout((vec_size,), stride=(1,)),
                    )
                    cute.copy(copy_atom_w_lm, w_slice, w_vec)
                    for v_inner in cutlass.range_constexpr(vec_size):
                        w_val = w_vec[v_inner].to(Float32)
                        h_val = sCarry[k_base + v_inner].to(Float32)
                        acc_T = acc_T + w_val * h_val
            acc_T = cute.arch.warp_reduction(
                acc_T, op=lambda a, b: a + b, threads_in_group=32,
            )
            if lane_id == 0:
                if in_range:
                    cur_max = reduce_buf[Int32(warp_id) * Int32(2)]
                    if acc_T > cur_max:
                        reduce_buf[Int32(warp_id) * Int32(2)] = acc_T
                        reduce_buf[Int32(warp_id) * Int32(2) + Int32(1)] = (
                            v.bitcast(Float32)
                        )

        cute.arch.barrier()
        local_max_val = Float32(-3.4e38)
        local_max_idx = Int32(-1)
        if tidx == 0:
            for w in cutlass.range_constexpr(warps_per_row):
                candidate_val = reduce_buf[Int32(w * 2)]
                candidate_idx = reduce_buf[Int32(w * 2 + 1)].bitcast(Int32)
                if candidate_val > local_max_val:
                    local_max_val = candidate_val
                    local_max_idx = candidate_idx

        if tidx == 0:
            mArgmaxScratch[bidx * Int32(2)] = local_max_val.bitcast(cutlass.Int32)
            mArgmaxScratch[bidx * Int32(2) + Int32(1)] = local_max_idx
        cute.arch.barrier()
        if tidx == 0:
            cute.arch.atomic_add(
                mCounter.iterator, Int32(1), sem="release", scope="gpu",
            )
            expected_T = Int32((SYNCS_PER_LAYER * L + 1) * num_sms)
            done_T = Int32(0)
            while done_T < expected_T:
                done_T = cute.arch.atomic_add(
                    mCounter.iterator, Int32(0), sem="acquire", scope="gpu",
                )
        cute.arch.barrier()
        if bidx == Int32(0) and tidx == Int32(0):
            g_max_val = Float32(-3.4e38)
            g_max_idx = Int32(-1)
            for s in cutlass.range_constexpr(num_sms):
                v_bits = mArgmaxScratch[Int32(s * 2)]
                v_idx = mArgmaxScratch[Int32(s * 2 + 1)]
                v_val = v_bits.bitcast(Float32)
                if v_val > g_max_val:
                    g_max_val = v_val
                    g_max_idx = v_idx
            mNextTokenBuf[0] = g_max_idx

        tmem_alloc.relinquish_alloc_permit()
        tmem_ptr_for_free = tmem_alloc.retrieve_ptr(Float32)
        pipeline.sync(barrier_id=2)
        tmem_alloc.free(tmem_ptr_for_free)


_kernel_cache: dict = {}


def _get_counter(dev):
    return torch.zeros(1, dtype=torch.int32, device=dev)


def _get_compiled(dtype, H, OUT_QKVA, Lq, OUT_QB, Lkv, R, L, S_max, N, qk_nope_dim, v_head_dim,
                  E, K_topk, n_group, topk_group, I_routed, I_shared, routed_scaling,
                  num_threads, threads_per_output, num_sms, stream,
                  tp_size=1, ep_size=1, ep_rank=0,
                  V=102400, attention_scaling=1.0):
    _use_splitk = int(os.environ.get("MOE_SPLITK", "0")) != 0
    key = (dtype, H, OUT_QKVA, Lq, OUT_QB, Lkv, R, L, S_max, N, qk_nope_dim, v_head_dim,
           E, K_topk, n_group, topk_group, I_routed, I_shared, routed_scaling,
           num_threads, threads_per_output, num_sms,
           tp_size, ep_size, ep_rank,
           V, float(attention_scaling), _use_splitk)
    if key in _kernel_cache:
        return _kernel_cache[key]
    obj = DSv2AttnPrologueMultilayerKernel(
        dtype, H=H, OUT_QKVA=OUT_QKVA, Lq=Lq, OUT_QB=OUT_QB, L=L, S_max=S_max,
        N=N, qk_nope_dim=qk_nope_dim, v_head_dim=v_head_dim, Lkv=Lkv, R=R,
        E=E, K_topk=K_topk, n_group=n_group, topk_group=topk_group,
        I_routed=I_routed, I_shared=I_shared, routed_scaling=routed_scaling,
        num_threads=num_threads, threads_per_output=threads_per_output, num_sms=num_sms,
        tp_size=tp_size, ep_size=ep_size, ep_rank=ep_rank,
        V=V, attention_scaling=float(attention_scaling), use_splitk=_use_splitk,
    )
    p = lambda: make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    pi32 = lambda: make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=4)
    pf32 = lambda: make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16)
    pi64 = lambda: make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=4)
    pfp8 = lambda: make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16)
    psf  = lambda: make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, assumed_align=16)
    compiled = cute.compile(
        obj,
        p(), p(), p(), p(), p(), p(),
        p(), p(), p(),
        p(), p(), p(), p(),
        p(), p(),
        p(), p(), p(),
        pf32(), pf32(), pf32(),
        p(), p(), p(), p(),
        p(), p(),
        p(), p(), p(),
        p(), p(), p(),
        pf32(), pf32(), pi32(), p(), p(),
        pfp8(), pfp8(), psf(), psf(),
        pfp8(), psf(),
        pfp8(), pfp8(), pfp8(), psf(), psf(), psf(),
        psf(), psf(), psf(),
        p(), p(), pi32(), pi32(),
        p(), pf32(), pi32(),
        p(), p(), pi32(), pi32(),
        pi32(), pi64(), p(), p(), pi32(), pf32(), pi32(),
        p(), p(), pi32(), pi32(),
        pf32(),
        Int32(0), Float32(1.0), Float32(1e-6), stream,
    )
    _kernel_cache[key] = (obj, compiled)
    return obj, compiled


def attn_prologue_multilayer(
    h_in: Tensor,
    residual_stacked: Tensor,
    gamma1_stacked: Tensor,
    w_qkva_stacked: Tensor,
    gamma2_stacked: Tensor,
    w_qb_stacked: Tensor,
    gamma3_stacked: Tensor,
    cos: Tensor,
    sin: Tensor,
    kv_cache_c: Tensor,
    kv_cache_pe: Tensor,
    w_uk_stacked: Tensor,
    w_uv_stacked: Tensor,
    w_o_stacked: Tensor,
    gamma4_stacked: Tensor,
    w_gate_router_stacked: Tensor,
    w_gate_routed_stacked: Tensor,
    w_up_routed_stacked: Tensor,
    w_down_routed_stacked: Tensor,
    w_gate_shared_stacked: Tensor,
    w_up_shared_stacked: Tensor,
    w_down_shared_stacked: Tensor,
    cache_pos: int,
    N: int = 128,
    qk_nope_dim: int = 128,
    v_head_dim: int = 128,
    Lkv: int = 512,
    R: int = 64,
    E: int = 160,
    K_topk: int = 6,
    n_group: int = 8,
    topk_group: int = 3,
    I_routed: int = 1536,
    I_shared: int = 3072,
    routed_scaling: float = 16.0,
    softmax_scale: float | None = None,
    eps: float = 1e-6,
    w_gate_shared_fp8: Tensor = None,
    w_up_shared_fp8: Tensor = None,
    sfa_gate_q1: Tensor = None,
    sfa_up_q1: Tensor = None,
    w_down_shared_fp8: Tensor = None,
    sfa_down_q2: Tensor = None,
    w_gate_routed_fp8: Tensor = None,
    w_up_routed_fp8: Tensor = None,
    w_down_routed_fp8: Tensor = None,
    sf_gate_routed: Tensor = None,
    sf_up_routed: Tensor = None,
    sf_down_routed: Tensor = None,
    sf_gate_routed_hw: Tensor = None,
    sf_up_routed_hw: Tensor = None,
    sf_down_routed_hw: Tensor = None,
    attn_proj_symm_local: Tensor = None,
    attn_proj_symm_mc_ptr_int: int = 0,
    tp_sync_local: Tensor = None,
    tp_sync_mc_ptr_int: int = 0,
    tp_size: int = 1,
    attn_proj_reduced: Tensor = None,
    moe_routed_acc_f32: Tensor = None,
    tp_l1_sync: Tensor = None,
    moe_out_symm_local: Tensor = None,
    moe_out_symm_mc_ptr_int: int = 0,
    ep_sync_local: Tensor = None,
    ep_sync_mc_ptr_int: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    embed_weight: Tensor = None,
    token_id_buf: Tensor = None,
    inv_freq: Tensor = None,
    position_id_buf: Tensor = None,
    attention_scaling: float = 1.0,
    gamma_final: Tensor = None,
    lm_head_weight: Tensor = None,
    next_token_buf: Tensor = None,
    moe_splitk_partials: Tensor = None,
) -> tuple[Tensor, ...]:
    assert h_in.is_cuda and h_in.is_contiguous()
    assert residual_stacked.is_cuda and residual_stacked.is_contiguous()
    assert gamma1_stacked.is_cuda and gamma1_stacked.is_contiguous()
    assert w_qkva_stacked.is_cuda and w_qkva_stacked.is_contiguous()
    assert gamma2_stacked.is_cuda and gamma2_stacked.is_contiguous()
    assert w_qb_stacked.is_cuda and w_qb_stacked.is_contiguous()
    assert gamma3_stacked.is_cuda and gamma3_stacked.is_contiguous()
    assert cos.is_cuda and cos.is_contiguous()
    assert sin.is_cuda and sin.is_contiguous()
    assert kv_cache_c.is_cuda and kv_cache_c.is_contiguous()
    assert kv_cache_pe.is_cuda and kv_cache_pe.is_contiguous()
    assert w_uk_stacked.is_cuda and w_uk_stacked.is_contiguous()
    assert w_uv_stacked.is_cuda and w_uv_stacked.is_contiguous()
    assert w_o_stacked.is_cuda and w_o_stacked.is_contiguous()
    for t, name in [
        (gamma4_stacked, "gamma4"),
        (w_gate_router_stacked, "w_gate_router"),
        (w_gate_routed_stacked, "w_gate_routed"),
        (w_up_routed_stacked, "w_up_routed"),
        (w_down_routed_stacked, "w_down_routed"),
        (w_gate_shared_stacked, "w_gate_shared"),
        (w_up_shared_stacked, "w_up_shared"),
        (w_down_shared_stacked, "w_down_shared"),
    ]:
        assert t.is_cuda and t.is_contiguous(), f"{name} not cuda/contiguous"

    H = h_in.shape[0]
    L, OUT_QKVA, H2 = w_qkva_stacked.shape
    assert H == H2
    L2, OUT_QB, qb_K = w_qb_stacked.shape
    assert L == L2
    if OUT_QKVA == Lkv + R:
        Lq = 0
        assert qb_K == H, f"Lite path expects W_qb input dim = H, got {qb_K}"
    else:
        Lq = OUT_QKVA - Lkv - R
        assert qb_K == Lq, f"236B path expects W_qb input dim = Lq={Lq}, got {qb_K}"
    assert residual_stacked.shape == (L, H)
    assert gamma1_stacked.shape == (L, H)
    assert gamma2_stacked.shape == (L, max(Lq, 1))
    assert gamma3_stacked.shape == (L, Lkv)
    assert cos.shape == (R // 2,)
    assert sin.shape == (R // 2,)
    L3, S_max, Lkv2 = kv_cache_c.shape
    assert L3 == L and Lkv2 == Lkv
    L4, S_max2, R2 = kv_cache_pe.shape
    assert L4 == L and S_max2 == S_max and R2 == R
    assert 0 <= cache_pos < S_max
    L5, N_w, Lkv_w, qk_w = w_uk_stacked.shape
    assert L5 == L and N_w == N and Lkv_w == Lkv and qk_w == qk_nope_dim, (
        f"w_uk_stacked shape mismatch: {w_uk_stacked.shape} vs "
        f"({L}, {N}, {Lkv}, {qk_nope_dim})"
    )
    L6, N_v, vhd, Lkv_v = w_uv_stacked.shape
    assert L6 == L and N_v == N and vhd == v_head_dim and Lkv_v == Lkv, (
        f"w_uv_stacked shape mismatch: {w_uv_stacked.shape} vs "
        f"({L}, {N}, {v_head_dim}, {Lkv})"
    )
    L7, H_o, Oconcat = w_o_stacked.shape
    O_CONCAT = N * v_head_dim
    assert L7 == L and H_o == H and Oconcat == O_CONCAT, (
        f"w_o_stacked shape mismatch: {w_o_stacked.shape} vs ({L}, {H}, {O_CONCAT})"
    )
    assert gamma4_stacked.shape == (L, H)
    assert w_gate_router_stacked.shape == (L, E, H), \
        f"w_gate_router shape {w_gate_router_stacked.shape} != ({L}, {E}, {H})"
    E_per_rank = E // ep_size
    assert w_gate_shared_stacked.shape == (L, I_shared, H)
    assert w_up_shared_stacked.shape == (L, I_shared, H)
    assert w_down_shared_stacked.shape == (L, H, I_shared)
    assert E % n_group == 0, f"E ({E}) must be divisible by n_group ({n_group})"

    dev, dt = h_in.device, h_in.dtype
    qkva_out = torch.empty(L, OUT_QKVA, dtype=dt, device=dev)
    q_out = torch.empty(L, OUT_QB, dtype=dt, device=dev)
    kvc_out = torch.empty(L, Lkv, dtype=dt, device=dev)
    kpe_out = torch.empty(L, R, dtype=dt, device=dev)
    q_nope_abs_out = torch.empty(L, N, Lkv, dtype=dt, device=dev)
    attn_out = torch.empty(L, N, Lkv, dtype=dt, device=dev)
    o_per_head_out = torch.empty(L, N, v_head_dim, dtype=dt, device=dev)
    attn_proj_out = torch.empty(L, H, dtype=dt, device=dev)
    CLUSTER_SIZE_FOR_SPLITS = 2
    TILE_SEQ_DEC_NEW_FOR_SPLITS = 64
    num_kv_splits = min(NUM_SMS // CLUSTER_SIZE_FOR_SPLITS,
                        S_max // TILE_SEQ_DEC_NEW_FOR_SPLITS)
    num_kv_splits = max(1, num_kv_splits)
    m_partial_buf = torch.zeros(N, num_kv_splits, dtype=torch.float32, device=dev)
    l_partial_buf = torch.zeros(N, num_kv_splits, dtype=torch.float32, device=dev)
    o_partial_buf = torch.zeros(N, num_kv_splits, Lkv, dtype=torch.float32, device=dev)
    I_max_moe = K_topk * I_routed + I_shared
    router_logits_out = torch.zeros(L, E, dtype=torch.float32, device=dev)
    topk_w_out = torch.zeros(L, K_topk, dtype=torch.float32, device=dev)
    topk_ids_out = torch.zeros(L, K_topk, dtype=torch.int32, device=dev)
    inter_tmp = torch.zeros(I_max_moe, dtype=dt, device=dev)
    moe_out = torch.zeros(L, H, dtype=dt, device=dev)
    h_final_out = torch.zeros(H, dtype=dt, device=dev)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(qk_nope_dim + R)
    counter = _get_counter(dev)
    counter.zero_()

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    dtype = _ct(dt)
    assert OUT_QB == N * (qk_nope_dim + R), (
        f"OUT_QB must equal N*(qk_nope_dim+R) (got {OUT_QB} != {N}*({qk_nope_dim}+{R}))"
    )
    assert embed_weight is not None, "embed_weight (V, H) bf16 required"
    assert token_id_buf is not None, "token_id_buf (1,) int32 required"
    assert embed_weight.is_cuda and embed_weight.is_contiguous()
    assert embed_weight.dtype == dt, (
        f"embed_weight dtype {embed_weight.dtype} != h_in dtype {dt}"
    )
    V_w, H_w = embed_weight.shape
    assert H_w == H, f"embed_weight H={H_w} != model H={H}"
    assert token_id_buf.is_cuda and token_id_buf.is_contiguous()
    assert token_id_buf.dtype == torch.int32 and token_id_buf.numel() == 1, (
        f"token_id_buf must be (1,) int32, got {token_id_buf.shape} {token_id_buf.dtype}"
    )
    V_kernel = V_w
    assert inv_freq is not None, "inv_freq (R/2,) fp32 required"
    assert position_id_buf is not None, "position_id_buf (1,) int32 required"
    assert inv_freq.is_cuda and inv_freq.is_contiguous()
    assert inv_freq.dtype == torch.float32, f"inv_freq must be fp32, got {inv_freq.dtype}"
    assert inv_freq.shape == (R // 2,), (
        f"inv_freq shape {inv_freq.shape} != (R/2={R//2},)"
    )
    assert position_id_buf.is_cuda and position_id_buf.is_contiguous()
    assert position_id_buf.dtype == torch.int32 and position_id_buf.numel() == 1, (
        f"position_id_buf must be (1,) int32, got {position_id_buf.shape} {position_id_buf.dtype}"
    )
    assert gamma_final is not None, "gamma_final (H,) bf16 required"
    assert lm_head_weight is not None, "lm_head_weight (V, H) bf16 required"
    assert next_token_buf is not None, "next_token_buf (1,) int32 required"
    assert gamma_final.is_cuda and gamma_final.is_contiguous()
    assert gamma_final.dtype == dt, (
        f"gamma_final dtype {gamma_final.dtype} != h_in dtype {dt}"
    )
    assert gamma_final.shape == (H,), f"gamma_final shape {gamma_final.shape} != (H={H},)"
    assert lm_head_weight.is_cuda and lm_head_weight.is_contiguous()
    assert lm_head_weight.dtype == dt, (
        f"lm_head_weight dtype {lm_head_weight.dtype} != h_in dtype {dt}"
    )
    V_lm, H_lm = lm_head_weight.shape
    assert H_lm == H, f"lm_head_weight H={H_lm} != model H={H}"
    assert V_lm == V_kernel, (
        f"lm_head_weight V={V_lm} != embed_weight V={V_kernel}"
    )
    assert next_token_buf.is_cuda and next_token_buf.is_contiguous()
    assert next_token_buf.dtype == torch.int32 and next_token_buf.numel() == 1, (
        f"next_token_buf must be (1,) int32, got {next_token_buf.shape} {next_token_buf.dtype}"
    )
    obj, compiled = _get_compiled(
        dtype, H, OUT_QKVA, Lq, OUT_QB, Lkv, R, L, S_max, N, qk_nope_dim, v_head_dim,
        E, K_topk, n_group, topk_group, I_routed, I_shared, routed_scaling,
        num_threads=256, threads_per_output=32, num_sms=NUM_SMS, stream=stream,
        tp_size=tp_size, ep_size=ep_size, ep_rank=ep_rank,
        V=V_kernel, attention_scaling=float(attention_scaling),
    )
    p = lambda t: make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    pi32 = lambda t: make_ptr(cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    pf32 = lambda t: make_ptr(cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    pi64 = lambda t: make_ptr(cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    pfp8 = lambda t: make_ptr(cutlass.Float8E4M3FN, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    psf  = lambda t: make_ptr(cutlass.Float8E8M0FNU, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    _u8_dummy = lambda: torch.empty(16, dtype=torch.uint8, device=dev)
    w_gate_shared_fp8_d = w_gate_shared_fp8 if w_gate_shared_fp8 is not None else _u8_dummy()
    w_up_shared_fp8_d   = w_up_shared_fp8   if w_up_shared_fp8   is not None else _u8_dummy()
    sfa_gate_q1_d       = sfa_gate_q1       if sfa_gate_q1       is not None else _u8_dummy()
    sfa_up_q1_d         = sfa_up_q1         if sfa_up_q1         is not None else _u8_dummy()
    w_down_shared_fp8_d = w_down_shared_fp8 if w_down_shared_fp8 is not None else _u8_dummy()
    sfa_down_q2_d       = sfa_down_q2       if sfa_down_q2       is not None else _u8_dummy()
    w_gate_routed_fp8_d = w_gate_routed_fp8 if w_gate_routed_fp8 is not None else _u8_dummy()
    w_up_routed_fp8_d   = w_up_routed_fp8   if w_up_routed_fp8   is not None else _u8_dummy()
    w_down_routed_fp8_d = w_down_routed_fp8 if w_down_routed_fp8 is not None else _u8_dummy()
    sf_gate_routed_d    = sf_gate_routed    if sf_gate_routed    is not None else _u8_dummy()
    sf_up_routed_d      = sf_up_routed      if sf_up_routed      is not None else _u8_dummy()
    sf_down_routed_d    = sf_down_routed    if sf_down_routed    is not None else _u8_dummy()
    sf_gate_routed_hw_d = sf_gate_routed_hw if sf_gate_routed_hw is not None else _u8_dummy()
    sf_up_routed_hw_d   = sf_up_routed_hw   if sf_up_routed_hw   is not None else _u8_dummy()
    sf_down_routed_hw_d = sf_down_routed_hw if sf_down_routed_hw is not None else _u8_dummy()
    attn_proj_symm_local_d = attn_proj_symm_local if attn_proj_symm_local is not None else _u8_dummy()
    tp_sync_local_d = tp_sync_local if tp_sync_local is not None else _u8_dummy()
    attn_proj_symm_mc_addr = attn_proj_symm_mc_ptr_int if attn_proj_symm_mc_ptr_int != 0 else attn_proj_symm_local_d.data_ptr()
    tp_sync_mc_addr = tp_sync_mc_ptr_int if tp_sync_mc_ptr_int != 0 else tp_sync_local_d.data_ptr()
    attn_proj_reduced_d = attn_proj_reduced if attn_proj_reduced is not None else _u8_dummy()
    moe_routed_acc_f32_d = moe_routed_acc_f32 if moe_routed_acc_f32 is not None else _u8_dummy()
    tp_l1_sync_d = tp_l1_sync if tp_l1_sync is not None else _u8_dummy()
    if tp_l1_sync is not None:
        tp_l1_sync_d.zero_()
    moe_out_symm_local_d = moe_out_symm_local if moe_out_symm_local is not None else _u8_dummy()
    ep_sync_local_d = ep_sync_local if ep_sync_local is not None else _u8_dummy()
    moe_out_symm_mc_addr = moe_out_symm_mc_ptr_int if moe_out_symm_mc_ptr_int != 0 else moe_out_symm_local_d.data_ptr()
    ep_sync_mc_addr = ep_sync_mc_ptr_int if ep_sync_mc_ptr_int != 0 else ep_sync_local_d.data_ptr()
    argmax_scratch_d = torch.zeros(NUM_SMS * 2, dtype=torch.int32, device=dev)
    moe_splitk_partials_d = moe_splitk_partials if moe_splitk_partials is not None else _u8_dummy()
    stage_ts = _get_stage_ts(L, dev)
    stage_ts.zero_()
    compiled(
        p(h_in), p(residual_stacked), p(gamma1_stacked), p(w_qkva_stacked),
        p(gamma2_stacked), p(w_qb_stacked),
        p(gamma3_stacked), p(cos), p(sin),
        p(qkva_out), p(q_out), p(kvc_out), p(kpe_out),
        p(kv_cache_c), p(kv_cache_pe),
        p(w_uk_stacked), p(q_nope_abs_out), p(attn_out),
        pf32(m_partial_buf), pf32(l_partial_buf), pf32(o_partial_buf),
        p(w_uv_stacked), p(w_o_stacked), p(o_per_head_out), p(attn_proj_out),
        p(gamma4_stacked), p(w_gate_router_stacked),
        p(w_gate_routed_stacked), p(w_up_routed_stacked), p(w_down_routed_stacked),
        p(w_gate_shared_stacked), p(w_up_shared_stacked), p(w_down_shared_stacked),
        pf32(router_logits_out), pf32(topk_w_out), pi32(topk_ids_out),
        p(inter_tmp), p(moe_out),
        pfp8(w_gate_shared_fp8_d), pfp8(w_up_shared_fp8_d),
        psf(sfa_gate_q1_d), psf(sfa_up_q1_d),
        pfp8(w_down_shared_fp8_d), psf(sfa_down_q2_d),
        pfp8(w_gate_routed_fp8_d), pfp8(w_up_routed_fp8_d), pfp8(w_down_routed_fp8_d),
        psf(sf_gate_routed_d), psf(sf_up_routed_d), psf(sf_down_routed_d),
        psf(sf_gate_routed_hw_d), psf(sf_up_routed_hw_d), psf(sf_down_routed_hw_d),
        p(attn_proj_symm_local_d),
        make_ptr(dtype, attn_proj_symm_mc_addr, cute.AddressSpace.gmem, assumed_align=16),
        pi32(tp_sync_local_d),
        make_ptr(cutlass.Int32, tp_sync_mc_addr, cute.AddressSpace.gmem, assumed_align=4),
        p(attn_proj_reduced_d), pf32(moe_routed_acc_f32_d), pi32(tp_l1_sync_d),
        p(moe_out_symm_local_d),
        make_ptr(dtype, moe_out_symm_mc_addr, cute.AddressSpace.gmem, assumed_align=16),
        pi32(ep_sync_local_d),
        make_ptr(cutlass.Int32, ep_sync_mc_addr, cute.AddressSpace.gmem, assumed_align=4),
        pi32(counter), pi64(stage_ts), p(h_final_out),
        p(embed_weight), pi32(token_id_buf),
        pf32(inv_freq), pi32(position_id_buf),
        p(gamma_final), p(lm_head_weight),
        pi32(next_token_buf), pi32(argmax_scratch_d),
        pf32(moe_splitk_partials_d),
        Int32(cache_pos), Float32(softmax_scale), Float32(eps), stream,
    )
    global _last_stage_ts
    _last_stage_ts = stage_ts
    return (qkva_out, q_out, kvc_out, kpe_out, q_nope_abs_out, attn_out,
            o_per_head_out, attn_proj_out,
            router_logits_out, topk_w_out, topk_ids_out, moe_out,
            h_final_out)


_last_stage_ts: Tensor | None = None
_stage_ts_cache: dict = {}


def _get_stage_ts(L: int, dev) -> Tensor:
    key = (L, str(dev))
    buf = _stage_ts_cache.get(key)
    if buf is None:
        buf = torch.zeros(L, 48, dtype=torch.int32, device=dev)
        _stage_ts_cache[key] = buf
    return buf


def get_last_stage_ts() -> Tensor | None:
    return _last_stage_ts

