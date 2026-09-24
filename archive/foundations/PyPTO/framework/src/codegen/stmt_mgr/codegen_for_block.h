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
 * \file codegen_for_block.h
 * \brief
 */

#pragma once

#include <unordered_set>
#include <vector>

#include "interface/operation/opcode.h"
#include "interface/tensor/symbolic_scalar.h"
#include "codegen/symbol_mgr/codegen_symbol.h"

namespace npu::tile_fwk {
constexpr const int MAX_LOOP_DEPTH = 3;

struct ForNode {
    std::string loopVar;
    SymbolicScalar start;
    SymbolicScalar extent;
    SymbolicScalar step;

    std::string Print() const;
    void PrintInit(std::ostringstream& os) const;
    void PrintCond(std::ostringstream& os) const;
    void PrintUpdate(std::ostringstream& os) const;
};

using OffsetInLoop = std::vector<std::string>;
class ForBlockManager {
public:
    explicit ForBlockManager(const std::shared_ptr<SymbolManager>& symbolManager) : sm_(symbolManager)
    {
        forNodes_.reserve(MAX_LOOP_DEPTH);
    };
    ~ForBlockManager() = default;

    void UpdateAxesList(const std::vector<SymbolicScalar>& axesList);

    void LoopStart() { isInLoop_ = true; }

    void OutLoop()
    {
        sm_->OutForLoop();
        axesList_.clear();
        forNodes_.clear();
        tensorOffset_.clear();
        allOffsetsInLoop_.clear();
        opList_.clear();

        isInLoop_ = false;
        offsetCnt_ = 0;
    }

    bool IsInLoop() { return isInLoop_; }

    void AddTensorInLoopBody(const std::string& tensorFullDim, const TileTensor& tileTensor, const Operation& oper,
                             Opcode opCode);

    void AddOpInLoopBody(std::string& op)
    {
        CODEGEN_LOGI("AddOpInLoopBody add op : %s", op.c_str());
        opList_.emplace_back(op);
    }

    std::string Print() const;

private:
    void PrintForHeader(std::ostringstream& os) const;
    void PrintForBody(std::ostringstream& os) const;
    void PrintForEnd(std::ostringstream& os) const;
    void PrintOffsetDef(std::ostringstream& os) const;
    void PrintSetAddrs(std::ostringstream& os) const;
    void PrintSetAddrSingle(std::ostringstream& os, const std::string& tensor, const std::string& offset) const;
    void PrintTileOps(std::ostringstream& os) const;
    void UpdateTensorOffsetInLoop(Opcode opCode, int tensorMagic, const Operation& oper,
                                  const std::string& tensorNameInLoop);
    OffsetInLoop BuildOffsetInLoop(int tensorMagic, const Operation& oper);

    std::shared_ptr<SymbolManager> sm_;
    std::vector<SymbolicScalar> axesList_;
    std::vector<ForNode> forNodes_;
    std::vector<std::string> opList_;
    // tensorNameInLoop -> offset var, e.g. ubTensor_2 -> tileOffset_1
    std::unordered_map<std::string, std::string> tensorOffset_;
    // OffsetInLoop -> offset var, e.g. (0, idx1, 0) -> tileOffset_1
    std::map<OffsetInLoop, std::string> allOffsetsInLoop_;
    // default offset, e.g. (idx0, idx1, idx2)
    OffsetInLoop defaultOffset_;

    bool isInLoop_{false};
    int offsetCnt_{0};
};
} // namespace npu::tile_fwk
