/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file checker_utils.h
 * \brief
 */

#pragma once

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"

namespace npu {
namespace tile_fwk {
class OpChecker {
public:
    class BaseChecker {
    public:
        virtual bool check(Operation* op) const = 0;
        virtual ~BaseChecker() = default;
    };

    class CalcTypeChecker : public BaseChecker {
        std::vector<OpCalcType> conditions;

    public:
        CalcTypeChecker(std::vector<OpCalcType> calcTypes) : conditions(std::move(calcTypes)) {}
        CalcTypeChecker(OpCalcType calcType) : conditions({calcType}) {}
        bool check(Operation* op) const override;
    };

    class CoreTypeChecker : public BaseChecker {
        std::vector<OpCoreType> conditions;

    public:
        CoreTypeChecker(std::vector<OpCoreType> coreTypes) : conditions(std::move(coreTypes)) {}
        CoreTypeChecker(OpCoreType coreType) : conditions({coreType}) {}
        bool check(Operation* op) const override;
    };

    class InputMemTypeChecker : public BaseChecker {
        std::vector<MemoryType> conditions;

    public:
        InputMemTypeChecker(std::vector<MemoryType> inputMemTypes) : conditions(std::move(inputMemTypes)) {}
        InputMemTypeChecker(MemoryType inputMemType) : conditions({inputMemType}) {}
        bool check(Operation* op) const override;
    };

    class OutputMemTypeChecker : public BaseChecker {
        std::vector<MemoryType> conditions;

    public:
        OutputMemTypeChecker(std::vector<MemoryType> outputMemTypes) : conditions(std::move(outputMemTypes)) {}
        OutputMemTypeChecker(MemoryType outputMemType) : conditions({outputMemType}) {}
        bool check(Operation* op) const override;
    };

    template <typename... Checkers>
    static bool check(Operation* op, Checkers&&... checkers)
    {
        return (checkers.check(op) && ...);
    }

    template <typename... Checkers>
    static bool check(Operation& op, Checkers&&... checkers)
    {
        return check(&op, std::forward<Checkers>(checkers)...);
    }
};
} // namespace tile_fwk
} // namespace npu
