/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file calc_error.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {

// Calculator 层错误码从 0xBF000U 开始，只在 calculator/ 目录内部使用，

enum class CalculatorErrorScene : uint32_t {
    // Range / 生成相关
    RANGE_NUMEL_MISMATCH = 0xBF000U, // torch::arange 生成的 numel 与 out.shape 展开后的元素个数不一致

    // 比较运算相关
    COMPARE_UNSUPPORTED_TYPE = 0xBF001U, // CompareImpl 收到未支持的 CmpOperationType
    BITMODE_LAST_DIM_INVALID = 0xBF002U, // CmpModeType::BIT 模式下，最后一维尺寸不是 NUM_VALUE_8 的倍数

    // 形状约束 / 格式转换
    FORMAT_ND2NZ_RANK_LT_2 = 0xBF003U, // FormatND2NZ 要求 rank >= 2，实际小于 2
    FORMAT_NZ2ND_RANK_LT_2 = 0xBF004U, // FormatNZ2ND 要求 rank >= 2，实际小于 2

    // 量化预处理相关
    QUANTPRECOMPUTE_NULL_DATAPTR = 0xBF005U,   // QuantPreCompute 中 out/self 的 dataPtr 为空
    QUANTPRECOMPUTE_DTYPE_MISMATCH = 0xBF006U, // QuantPreCompute 要求 out=FP16/self=INT32 的 dtype 组合不满足

    // GatherMask / pattern 模式
    GATHERMASK_PATTERNMODE_INVALID = 0xBF007U, // GatherMask 中 patternMode 不在 [1,7] 合法范围内

    // TopK/MrgSort 轴约束
    MRGSORT_AXIS_OUT_OF_RANGE = 0xBF008U, // MrgSort/TiledMrgSort 等中 axis 超出张量维度范围

    // Scatter/ScatterUpdate 相关
    SCATTER_BLOCKSIZE_ZERO = 0xBF009U,          // ScatterUpdate 中 blockSize 为 0
    SCATTER_INDICES_DIM_INVALID = 0xBF00AU,     // indices 维度不是期望的 2 维
    SCATTER_SRC_RET_DIM_UNSUPPORTED = 0xBF00BU, // src/ret 的维度不是 2 或 4，当前实现不支持
    SCATTER_SRC_RET_DIM_MISMATCH = 0xBF00CU,    // src 与 ret 的维度数量不一致

    // MatMul 形状约束
    MATMUL_INPUT_SHAPE_MISMATCH = 0xBF00DU, // MatMul/MX MatMul 输入shape不符合预期

    // QuantMX 相关
    QUANTMX_RANK_INVALID = 0xBF010U, // QuantMX 仅支持 1D~4D 输入

    // Gather / GatherINUB 相关
    GATHER_AXIS_OUT_OF_RANGE = 0xBF00EU,             // Gather 中 axis 超出 params 维度范围
    GATHER_INUB_DEVICE_INVALID = 0xBF00FU,           // GatherINUB 要求 params/indices/pageTable/out 全部在 CPU 上
    GATHER_INUB_AXIS_INVALID = 0xBF010U,             // GatherINUB 仅支持 axis == 0
    GATHER_INUB_BLOCKSIZE_INVALID = 0xBF011U,        // GatherINUB 中 blockSize 非法（<= 0）
    GATHER_INUB_SHAPE_INVALID = 0xBF012U,            // GatherINUB 输入/输出 shape 不满足约束
    GATHER_INUB_DTYPE_INVALID = 0xBF013U,            // GatherINUB dtype 约束不满足
    GATHER_INUB_LOGICAL_INDEX_INVALID = 0xBF014U,    // GatherINUB 逻辑索引越界或非法
    GATHER_INUB_PAGETABLE_NUMEL_MISMATCH = 0xBF015U, // GatherINUB pageTable 展开后元素个数异常
    GATHER_INUB_LOGICAL_BLOCK_INVALID = 0xBF016U,    // GatherINUB logical_block 越界或非法
    GATHER_INUB_PHYSICAL_INDEX_INVALID = 0xBF017U,   // GatherINUB physical 索引非法

    // FP4 打包转换相关
    FP4_PACKED_LAST_DIM_INVALID = 0xBF018U, // FP4 packed 转换要求最后一维必须是偶数

    // Pack/UnPack 相关
    PACKORUNPACK_MEMCPY_FAILED = 0xBF019U, // PackOrUnPack 中 memcpy_s 失败
};

} // namespace npu::tile_fwk
