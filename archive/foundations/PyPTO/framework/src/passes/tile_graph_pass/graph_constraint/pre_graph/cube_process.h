/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cube_process.h
 * \brief
 */

#ifndef PASS_CUBE_PROCESS_H
#define PASS_CUBE_PROCESS_H
#include "pre_graph_common.h"

namespace npu::tile_fwk {
const std::string MATMUL_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";
const std::string A_MUL_B_SCALE_ATTR = OP_ATTR_PREFIX + "scale_value";
const std::string A_MUL_B_RELU_ATTR = OP_ATTR_PREFIX + "relu_type";

/* L1 Copy In 的内外轴大小 */
const std::string L1_COPY_IN_INNER = OP_ATTR_PREFIX + "inner_value";
const std::string L1_COPY_IN_OUTER = OP_ATTR_PREFIX + "outer_value";

/* L0C Copy Out 的内外轴大小 */
const std::string L0C_COPY_OUT_OUTER = OP_ATTR_PREFIX + "curH";
const std::string L0C_COPY_OUT_INNER = OP_ATTR_PREFIX + "curW";

/* 是否做搬运随路转Nz */
const std::string COPY_IS_NZ = OP_ATTR_PREFIX + "is_nz";

/* Matmul支持的数据类型
key: L0A 和 L0B 的数据类型
vaule: L0C 对应的数据类型
*/
const std::map<std::pair<DataType, DataType>, DataType> supportDtypeMap = {
    {{DataType::DT_FP16, DataType::DT_FP16}, DataType::DT_FP32},
    {{DataType::DT_BF16, DataType::DT_BF16}, DataType::DT_FP32},
    {{DataType::DT_FP32, DataType::DT_FP32}, DataType::DT_FP32},
    {{DataType::DT_FP32, DataType::DT_FP16}, DataType::DT_FP32},
    {{DataType::DT_FP32, DataType::DT_BF16}, DataType::DT_FP32},
    {{DataType::DT_FP16, DataType::DT_FP32}, DataType::DT_FP32},
    {{DataType::DT_BF16, DataType::DT_FP32}, DataType::DT_FP32},
    {{DataType::DT_HF8, DataType::DT_HF8}, DataType::DT_FP32},
    {{DataType::DT_INT8, DataType::DT_INT8}, DataType::DT_INT32},
    {{DataType::DT_INT4, DataType::DT_INT4}, DataType::DT_INT32},
    {{DataType::DT_INT16, DataType::DT_INT16}, DataType::DT_INT32},
    {{DataType::DT_FP8E5M2, DataType::DT_FP8E5M2}, DataType::DT_FP32},
    {{DataType::DT_FP8E4M3, DataType::DT_FP8E4M3}, DataType::DT_FP32},
    {{DataType::DT_FP8E5M2, DataType::DT_FP8E4M3}, DataType::DT_FP32},
    {{DataType::DT_FP8E4M3, DataType::DT_FP8E5M2}, DataType::DT_FP32},
    {{DataType::DT_FP4_E2M1X2, DataType::DT_FP4_E2M1X2}, DataType::DT_FP32},
    {{DataType::DT_FP4_E1M2X2, DataType::DT_FP4_E1M2X2}, DataType::DT_FP32},
    {{DataType::DT_FP4_E2M1, DataType::DT_FP4_E2M1}, DataType::DT_FP32},
    {{DataType::DT_FP4_E1M2, DataType::DT_FP4_E1M2}, DataType::DT_FP32},
};

class CubeProcess {
public:
    CubeProcess() {}
    ~CubeProcess() = default;

    Status CheckValidCube(const Operation& op);
    Status UpdateCubeOp(Function& function);
    Status UpdateCopyAttr(Operation& op) const;
    Status AddL1CopyInAttr(const std::shared_ptr<LogicalTensor> input, int nzValue, int mValue, int kValue,
                           int nValue) const;
    Status AddL0cCopyOutAttr(const std::shared_ptr<LogicalTensor> output, int nzValue, int mValue, int nValue) const;
    Status UpdateL0cDtype(Operation& op);
    Status AlignGMTensor(Function& function, std::vector<Operation*>& l0CCopyOuts, Operation& mulOp);
    void DFSSearch(Operation* op, std::vector<Operation*>& l0CCopyOuts, std::unordered_set<Operation*>& visitedOp);
    Status GetL0CCopyOuts(Operation& op, std::vector<Operation*>& l0CCopyOuts);
    Status ReconnectGraph(Operation& mulOp, std::vector<Operation*> copyOutOps);
    Status TransferAttr(Operation& mulOp, std::vector<Operation*> copyOutOps);
};
} // namespace npu::tile_fwk
#endif // PASS_CUBE_PROCESS_H
