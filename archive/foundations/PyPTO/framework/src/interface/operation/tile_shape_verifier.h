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
 * \file tile_shape_verifier.h
 * \brief
 */

#pragma once

#include "operation.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {
using VerifyFunc = std::function<bool(const Operation& op, std::ostream& oss, const LogicalTensorPtr& tensor)>;

constexpr uint32_t VERIFY_SHAPE_SIZE = 0x0001;
constexpr uint32_t VERIFY_TAIL_ALIGN = 0x0002;
constexpr uint32_t VERIFY_FIX_AXIS = 0x0004;
constexpr uint32_t VERIFY_SHAPE_SIZE_LAST_INPUT = 0x0008;

// customize the check items of opcode
// for example : {OP_XX, VERIFY_TAIL_ALIGN | VERIFY_FIX_AXIS}
const std::unordered_map<Opcode, uint32_t> verify_cfg = {{Opcode::OP_INDEX_PUT, VERIFY_SHAPE_SIZE_LAST_INPUT},
                                                         {Opcode::OP_INDEX_ADD, VERIFY_SHAPE_SIZE_LAST_INPUT}};

const std::unordered_map<Opcode, std::string> axis_name_map = {{Opcode::OP_EXPAND, "expand_dims"},
                                                               {Opcode::OP_GATHER, "axis"}};

class TileShapeVerifier {
public:
    static bool Verify([[maybe_unused]] const Function& func, const Operation& op, std::ostream& oss)
    {
        auto config = GetVerifyConfig(op.GetOpcode());
        if ((config & VERIFY_SHAPE_SIZE) && !RunVerifyFunc(op, oss, VerifyTileShapeSize)) {
            return false;
        }
        if ((config & VERIFY_TAIL_ALIGN) && !RunVerifyFunc(op, oss, VerifyTileShapeTailAxisAlign)) {
            return false;
        }
        if ((config & VERIFY_FIX_AXIS)) {
            auto tensor = op.GetIOperands().front();
            if (!VerifyTileShapeFixAxis(op, oss, tensor)) {
                return false;
            }
        }
        if ((config & VERIFY_SHAPE_SIZE_LAST_INPUT) && !RunVerifyFuncLastInput(op, oss, VerifyTileShapeSize)) {
            return false;
        }
        return true;
    }

private:
    static bool RunVerifyFunc(const Operation& op, std::ostream& oss, const VerifyFunc& func)
    {
        return func(op, oss, op.GetOOperands().front());
    }
    static bool RunVerifyFuncLastInput(const Operation& op, std::ostream& oss, const VerifyFunc& func)
    {
        return func(op, oss, op.GetIOperands().back());
    }
    static bool VerifyTileShapeSize(const Operation& op, std::ostream& oss, const LogicalTensorPtr& tensor)
    {
        auto tile_size = op.GetTileShape().GetVecTile().size();
        auto shape_size = tensor->GetShape().size();
        if (tile_size < shape_size) {
            auto opName = OpcodeManager::Inst().GetOpcodeStr(op.GetOpcode());
            oss << "Operation: " << opName << ". Tile shape size " << tile_size
                << " is not matched the output shape size " << shape_size << ".";
            return false;
        }
        return true;
    }

    static bool VerifyTileShapeTailAxisAlign(const Operation& op, std::ostream& oss, const LogicalTensorPtr& tensor)
    {
        if (GetTensorMemoryType(op, tensor) != MemoryType::MEM_UB) {
            return true;
        }
        auto tail_axis = op.GetTileShape().GetVecTile().tile.back();
        auto data_type = tensor->Datatype();
        if (tail_axis * BytesOf(data_type) % BLOCK_SIZE != 0) {
            auto opName = OpcodeManager::Inst().GetOpcodeStr(op.GetOpcode());
            oss << "Operation: " << opName << ". The last axis of Tile shape " << tail_axis << " is not align 32B.";
            return false;
        }
        return true;
    }

    static bool VerifyTileShapeFixAxis(const Operation& op, std::ostream& oss, const LogicalTensorPtr& tensor)
    {
        auto shape = tensor->GetShape();
        int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + axis_name_map.at(op.GetOpcode()));
        axis = (axis == -1) ? (shape.size() - 1) : axis;
        if (op.GetTileShape().GetVecTile()[axis] != shape[axis]) {
            auto opName = OpcodeManager::Inst().GetOpcodeStr(op.GetOpcode());
            oss << "Operation: " << opName << ". Tile shape's " << std::to_string(axis)
                << " dim is not equal to output's " << std::to_string(axis) << " dim.";
            return false;
        }
        return true;
    }

    static uint32_t GetVerifyConfig(const Opcode& opcode)
    {
        if (verify_cfg.find(opcode) == verify_cfg.end()) {
            return VERIFY_SHAPE_SIZE;
        }
        return verify_cfg.at(opcode);
    }

    static MemoryType GetTensorMemoryType(const Operation& op, const LogicalTensorPtr& tensor)
    {
        auto index = op.GetOOperandIndex(tensor);
        if (index > 0) {
            return OpcodeManager::Inst().GetOutputsMemType(op.GetOpcode())[index];
        }
        index = op.GetIOperandIndex(tensor);
        return OpcodeManager::Inst().GetInputsMemType(op.GetOpcode())[index];
    }
};
} // namespace npu::tile_fwk
