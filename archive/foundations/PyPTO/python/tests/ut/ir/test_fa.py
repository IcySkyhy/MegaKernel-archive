# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from dataclasses import dataclass

import pypto
from pypto.pil.compile_pipeline import compile_new_ir


@dataclass
class TileShapeConfig:
    g_tile: int
    s2_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def fa_kernel(q, k, v, block_table, kv_act_seqs, atten_out, softmax_scale, tile_config):
    shape_q = q.shape
    shape_k = k.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = kv_act_seqs.shape[0]

    dtype = q.dtype
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_config.g_tile
    s2_tile = tile_config.s2_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = [block_num_scalar * block_size, n2_sym * dn]
    q_2d_shape = [b_scalar * s1_scalar * nq, dn]

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    for b_idx in pypto.loop(b_scalar, name="LOOP_b"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2_sym, name="LOOP_n2"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g"):
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", unroll_list=[4, 1]):
                        block_num = s2_tile // block_size
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])

                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                k_2d, [block_size, dn], [block_idx_valid * block_size, n2_idx * dn]
                            )
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])

                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        if s2_idx == 0:
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                            tsub = pypto.sub(sij_scale, tilda_mij)
                            tilda_pij = pypto.exp(tsub)
                            tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                            sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            max_update[:] = tilda_mij

                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                    v_2d, [block_size, dn], [block_idx_valid * block_size, n2_idx * dn]
                                )
                            vj_assemble = pypto.view(
                                vj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn]
                            )
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_tmp
                        else:
                            pypto.set_pass_options(sg_set_scope=1)
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                            max_new = pypto.maximum(max_update, tilda_mij)
                            tsub = pypto.sub(sij_scale, max_new)
                            tilda_pij = pypto.exp(tsub)
                            tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                            sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            pypto.set_pass_options(sg_set_scope=-1)

                            pypto.set_pass_options(sg_set_scope=2)
                            tsub2 = pypto.sub(max_update, max_new)
                            max_update[:] = max_new
                            update_mul = pypto.exp(tsub2)
                            sum_update[:] = sum_update * update_mul + sum_local
                            pypto.set_pass_options(sg_set_scope=-1)

                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = pypto.view(
                                    v_2d, [block_size, dn], [block_idx_valid * block_size, n2_idx * dn]
                                )
                            vj_assemble = pypto.view(
                                vj_assemble, [s2_tile, dn], [0, 0], valid_shape=[actual_s2_tile, dn]
                            )
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_update * update_mul + oi_tmp
                        if s2_idx == s2_loop - 1:
                            oi_final = pypto.div(oi_update, sum_update, precision_type=pypto.PrecisionType.INTRINSIC)
                            pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                            oi_final_3d = pypto.cast(pypto.reshape(oi_final, [1, g_tile, dn]), dtype)
                            pypto.assemble(oi_final_3d, oi_ofs, atten_out)


def test_fa_compile():
    m_tile = 128
    cube_tile = 128
    tile_config = TileShapeConfig(
        g_tile=12,
        s2_tile=512,
        c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        v1_tile_shape=[m_tile, 512],
        c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        v2_tile_shape=[m_tile, cube_tile],
    )

    s2 = 16384
    qd = 128
    nq = 12
    nkv = 1
    block_size = 128
    scale = qd**-0.5
    num_blocks = (s2 + block_size - 1) // block_size

    q = pypto.Tensor([-1, nq, qd], pypto.DT_BF16, "q")
    k = pypto.Tensor([-1, block_size, nkv, qd], pypto.DT_BF16, "k")
    v = pypto.Tensor([-1, block_size, nkv, qd], pypto.DT_BF16, "v")
    block_table = pypto.Tensor([-1, num_blocks], pypto.DT_INT32, "block_table")
    actual_seq = pypto.Tensor([-1], pypto.DT_INT32, "actual_seq")
    atten_out = pypto.Tensor([-1, nq, qd], pypto.DT_BF16, "atten_out")

    compile_new_ir(fa_kernel, q, k, v, block_table, actual_seq, atten_out, scale, tile_config)


if __name__ == "__main__":
    test_fa_compile()
