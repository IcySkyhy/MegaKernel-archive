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
 * \file symbol_id_gen.h
 * \brief Simple ID generator for codegen symbol manager (non-thread-safe)
 */

#pragma once

#include <array>
#include <cstdint>
#include "tilefwk/error.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

enum class SymbolIdType {
    CG_USING_NAME, // gen using name for codegen
    CG_VAR_NAME,   // gen variable for codegen
    CG_ID_TYPE_END,
};

class SymbolIdGen {
public:
    SymbolIdGen() = default;

    int NewId() { return id_++; }

    int CurId() const { return id_; }

    void Reset() { id_ = 0; }

    void SetId(int id) { id_ = id; }

private:
    int id_{0};
};

class SymbolIdGenMgr {
public:
    SymbolIdGenMgr() = default;

    template <SymbolIdType T>
    int NewId()
    {
        ASSERT(GenCodeErr::SYMBOL_ID_INVALID, T < SymbolIdType::CG_ID_TYPE_END)
            << "Invalid SymbolIdType: " << ToUnderlying(T);
        return gens_[static_cast<size_t>(T)].NewId();
    }

    template <SymbolIdType T>
    int CurId() const
    {
        ASSERT(GenCodeErr::SYMBOL_ID_INVALID, T < SymbolIdType::CG_ID_TYPE_END)
            << "Invalid SymbolIdType: " << ToUnderlying(T);
        return gens_[static_cast<size_t>(T)].CurId();
    }

    template <SymbolIdType T>
    void Reset()
    {
        ASSERT(GenCodeErr::SYMBOL_ID_INVALID, T < SymbolIdType::CG_ID_TYPE_END)
            << "Invalid SymbolIdType: " << ToUnderlying(T);
        gens_[static_cast<size_t>(T)].Reset();
    }

    template <SymbolIdType T>
    void SetId(int id)
    {
        ASSERT(GenCodeErr::SYMBOL_ID_INVALID, T < SymbolIdType::CG_ID_TYPE_END)
            << "Invalid SymbolIdType: " << ToUnderlying(T);
        gens_[static_cast<size_t>(T)].SetId(id);
    }

private:
    std::array<SymbolIdGen, static_cast<size_t>(SymbolIdType::CG_ID_TYPE_END)> gens_;
};

} // namespace npu::tile_fwk
