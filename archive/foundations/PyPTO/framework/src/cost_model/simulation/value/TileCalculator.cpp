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
 * \file TileCalculator.cpp
 * \brief
 */

#include "cost_model/simulation/value/TileCalculator.h"
#include <vector>
#include "interface/inner/hash_buffer.h"
#include "tilefwk/pypto_fwk_log.h"

namespace CostModel {
TileCalculator TileCalculator::instance;

static inline bool IsCopyOutOp(const Opcode& op)
{
    return (op == Opcode::OP_COPY_OUT || op == Opcode::OP_L0C_COPY_OUT || op == Opcode::OP_TRANSPOSE_MOVEOUT ||
            op == Opcode::OP_INDEX_OUTCAST);
}

static inline bool IsCopyInOp(const Opcode& op)
{
    return (op == Opcode::OP_COPY_IN || op == Opcode::OP_L1_COPY_IN || op == Opcode::OP_TRANSPOSE_MOVEIN);
}

static uint64_t CalculateInputHash(const TilePtr& tile)
{
    npu::tile_fwk::HashBuffer buffer(tile->shape, tile->offset, tile->dataType, tile->bufType, tile->symbol);
    return static_cast<uint64_t>(buffer.Digest());
}

static uint64_t LoadTile(TileState::TileStateKeyTy& k, std::shared_ptr<TileState> local,
                         std::shared_ptr<TileState> global)
{
    if (k.bufType == BUF_DDR) {
        return global->Load(k);
    } else {
        return local->Load(k);
    }
}

static void StoreTile(TileState::TileStateKeyTy& k, uint64_t& value, std::shared_ptr<TileState> local,
                      std::shared_ptr<TileState> global)
{
    if (k.bufType == BUF_DDR) {
        return global->Store(k, value);
    } else {
        return local->Store(k, value);
    }
}

static uint64_t CalculateOutputHash(TileOpPtr& op, size_t idx, FunctionInvokeInfo& invoke,
                                    std::shared_ptr<TileState> local, std::shared_ptr<TileState> global)
{
    std::vector<uint64_t> hash;
    for (auto& incast : op->iOperand) {
        auto bind = invoke.Bind(incast->rawMagic);
        if (!bind) {
            bind = incast;
        }

        auto k = TileState::TileKey(bind->rawMagic, bind->bufType, bind->shape, bind->offset);
        auto value = LoadTile(k, local, global);
        hash.push_back(value);
    }

    auto tile = op->oOperand[idx];
    auto bind = invoke.Bind(tile->rawMagic);
    if (!bind) {
        bind = tile;
    }
    auto k = TileState::TileKey(bind->rawMagic, bind->bufType, bind->shape, bind->offset);
    auto value = LoadTile(k, local, global);
    npu::tile_fwk::HashBuffer buffer(op->opcode, hash, idx, value);
    return static_cast<uint64_t>(buffer.Digest());
}

void TileCalculator::Reset() { seq = 0; }

void TileCalculator::CalculateInput(TilePtr tile, std::shared_ptr<TileState> global)
{
    auto value = CalculateInputHash(tile);
    auto key = TileState::TileKey(tile->rawMagic, tile->bufType, tile->shape, tile->offset);
    global->Store(key, value);
}

inline bool IsCopyIn(const std::string& op) { return op.find("COPY_IN") != std::string::npos; }

inline bool IsCopyOut(const std::string& op)
{
    return (op == "COPY_OUT" || op == "L0C_COPY_OUT" || op == "TRANSPOSE_MOVEOUT" || op == "INDEX_OUTCAST");
}

void TileCalculator::Calculate(TileOpPtr op, FunctionInvokeInfo& invoke, std::shared_ptr<TileState> local,
                               std::shared_ptr<TileState> global)
{
    seq++;

    if (op->opcode == "RESHAPE") {
        auto bind = invoke.Bind(op->oOperand[0]->rawMagic);
        if (!bind) {
            bind = op->oOperand[0];
        }
        auto dk = TileState::TileKey(bind->rawMagic, bind->bufType, bind->shape, bind->offset);

        bind = invoke.Bind(op->iOperand[0]->rawMagic);
        if (!bind) {
            bind = op->iOperand[0];
        }
        auto sk = TileState::TileKey(bind->rawMagic, bind->bufType, bind->shape, bind->offset);
        global->Ref(dk, sk);
        global->Load(sk);
        global->Load(dk);
    } else {
        for (auto& incast : op->iOperand) {
            auto bind = invoke.Bind(incast->rawMagic);
            if (!bind) {
                bind = incast;
            }
            auto k = TileState::TileKey(bind->rawMagic, bind->bufType, bind->shape, bind->offset);
            LoadTile(k, local, global);
        }

        for (size_t i = 0; i < op->oOperand.size(); i++) {
            auto outcast = op->oOperand[i];
            auto bind = invoke.Bind(outcast->rawMagic);
            if (!bind) {
                bind = outcast;
            }
            auto k = TileState::TileKey(bind->rawMagic, bind->bufType, bind->shape, bind->offset);
            auto value = CalculateOutputHash(op, i, invoke, local, global);
            StoreTile(k, value, local, global);
        }
    }
}
} // namespace CostModel
