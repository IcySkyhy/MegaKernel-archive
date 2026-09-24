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
 * \file generate_move_op.h
 * \brief
 */

#ifndef PASS_GENERATE_MOVE_OP_H_
#define PASS_GENERATE_MOVE_OP_H_

#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "interface/operation/attribute.h"

namespace npu::tile_fwk {
/*
    GenerateMoveOp: 将view和sassemble翻译成copyin和copyout,并将连续的copyin和copyout合并为一个，删除冗余copyout
*/

/*
key：vector of pair,每个pair记录convert op的from和to的内存类型
value：Opcode类型
*/
const std::map<std::pair<MemoryType, MemoryType>, Opcode> platformPathMap = {
    {{MEM_DEVICE_DDR, MEM_L1}, Opcode::OP_COPY_IN},
    {{MEM_DEVICE_DDR, MEM_UB}, Opcode::OP_COPY_IN},
    {{MEM_L1, MEM_L0A}, Opcode::OP_L1_TO_L0A},
    {{MEM_L1, MEM_L0B}, Opcode::OP_L1_TO_L0B},
    {{MEM_L0C, MEM_DEVICE_DDR}, Opcode::OP_COPY_OUT},
    {{MEM_UB, MEM_DEVICE_DDR}, Opcode::OP_COPY_OUT},
    {{MEM_L0C, MEM_L1}, Opcode::OP_L0C_TO_L1},
    {{MEM_L1, MEM_BT}, Opcode::OP_L1_TO_BT},
    {{MEM_L1, MEM_FIX_QUANT_PRE}, Opcode::OP_L1_TO_FIX_QUANT_PRE},
    {{MEM_L0C, MEM_UB}, Opcode::OP_L0C_COPY_UB},
    {{MEM_UB, MEM_L1}, Opcode::OP_UB_COPY_L1},
    {{MEM_L1, MEM_L0AMX}, Opcode::OP_L1_TO_L0A_SCALE},
    {{MEM_L1, MEM_L0BMX}, Opcode::OP_L1_TO_L0B_SCALE},
};

class GenerateMoveOp : public Pass {
public:
    GenerateMoveOp() : Pass("GenerateMoveOp") {}
    ~GenerateMoveOp() override = default;

private:
    static bool IsTransFormatOp(Opcode opcode);
    static bool HasConvConsumer(const Operation& op);
    static bool HasConvProducer(const Operation& op);
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    Status CreateMoveOp(Function& function) const;
    void SetCopyAttr(Operation& op, ViewOpAttribute* viewOpAttribute) const;
    void SetL0C2L1CopyAttr(Operation& op, const Shape& realShape, const std::vector<OpImmediate>& fromOffset,
                           const std::vector<OpImmediate>& toOffset,
                           Matrix::CopyMode mode = Matrix::CopyMode::UNKNOWN) const;
    void SetL0C2UBCopyAttr(Operation& op, const Shape& realShape, const std::vector<OpImmediate>& fromOffset,
                           const std::vector<OpImmediate>& toOffset,
                           Matrix::CopyMode mode = Matrix::CopyMode::UNKNOWN) const;
    void SetUB2L1CopyAttr(Operation& op, const Shape& copyShape, const std::vector<OpImmediate>& fromOffset,
                          const std::vector<OpImmediate>& toOffset,
                          Matrix::CopyMode mode = Matrix::CopyMode::UNKNOWN) const;
    Status SetOpcodeByMemPath(Operation& op, MemoryType from, MemoryType to) const;
    bool HasSpecificConsumer(const Operation& op) const;
    void ConvertViewToCopyInWhenInputGm(Operation& op, ViewOpAttribute* viewOpAttribute) const;
    Status A23CreateMoveOpForView(Function& function, Operation& op) const;
    Status A5CreateMoveOpForView(Function& function, Operation& op) const;
    Status ProcessGmInput(bool& isGmOutput, Operation& op, ViewOpAttribute* viewOpAttribute) const;
    Status CollapseTransFormatL1Consumer(Operation& op, ViewOpAttribute* viewOpAttribute) const;
    Status ProcessL0A(Operation& op, ViewOpAttribute* viewOpAttribute) const;
    Status ProcessL0B(Operation& op, ViewOpAttribute* viewOpAttribute) const;
    Status ProcessL0AMX(Operation& op, ViewOpAttribute* viewOpAttribute) const;
    Status ProcessL0BMX(Operation& op, ViewOpAttribute* viewOpAttribute) const;
    Status ProcessDefault(Function& function, Operation& op, ViewOpAttribute* viewOpAttribute) const;
    Status ProcessConvAssemble(Operation& op, AssembleOpAttribute* assembleOpAttribute) const;
    bool TryInsertModeDispatch(Function& function, Operation& op, AssembleOpAttribute* assembleOpAttribute,
                               MemoryType inputMemtype, MemoryType outputMemtype) const;
    Status CreateMoveOpForAssemble(Function& function, Operation& op) const;
    Status CreateMoveOpForConvert(Function& function, Operation& op) const;
    void ProcessUB2L1(Function& function, Operation& op) const;
    static int64_t PadUB(int64_t dim, int64_t padValue);
    Status ProcessL1CopyInConv(Operation& op) const;
    Status ProcessL0CCopyOutConv(Operation& op) const;
    Status ProcessDuplicateOp(Operation& op) const;
    Status ProcessViewOp(Function& function, Operation& op) const;
};
} // namespace npu::tile_fwk
#endif // PASS_GENERATE_MOVE_OP_H_
