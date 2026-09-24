/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root directory of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ir/program.h"
#include "tilefwk/tilefwk.h"

using namespace pypto;

namespace npu::tile_fwk {

class ProgramBuilder {
    using Body = std::function<void()>;
    using LoopBody = std::function<void(SymbolicScalar, const std::vector<ir::VarPtr>&)>;

public:
    ProgramBuilder();
    ~ProgramBuilder();

    ProgramBuilder(const ProgramBuilder&) = delete;
    ProgramBuilder& operator=(const ProgramBuilder&) = delete;

    /**
     * \brief Begin a new function.
     *
     * \param name The name of the function.
     * \param inputs The input tensors of the function.
     */
    void BeginFunction(const std::string& name, std::vector<std::reference_wrapper<const Tensor>> inputs);

    /**
     * \brief End the current function.
     *
     * \return The program pointer.
     */
    ir::ProgramPtr EndFunction();

    /**
     * \brief Allocate a new tensor.
     *
     * \param dtype The data type of the tensor.
     * \param shape The shape of the tensor.
     * \param name The name of the tensor.
     * \return The tensor tensor.
     */
    Tensor Alloc(DataType dtype, std::vector<int64_t> shape, std::string name);

    /**
     * \brief Create a new if statement.
     *
     * \param cond The condition of the if statement.
     * \param thenFn The then branch function.
     * \param elseFn The else branch function.
     * \return The if statement tensor.
     */
    std::vector<ir::ExprPtr> If(SymbolicScalar cond, Body thenFn, Body elseFn, const char* file = __builtin_FILE(),
                                int line = __builtin_LINE());

    /**
     * \brief Yield a value.
     *
     * \param value The value to yield.
     */
    template <typename... Args>
    void Yield(Args... args)
    {
        Yield({Unwrap(args)...});
    }

    /**
     * \brief Yield a value.
     *
     * \param value The value to yield.
     */
    void Yield(std::vector<ir::ExprPtr> value);

    /**
     * \brief Create a new for loop.
     *
     * \param start The start value of the loop.
     * \param stop The stop value of the loop.
     * \param step The step of the loop.
     * \param carries The carries of the loop, in the form of `(name, init)`.
     * \param body The body of the loop.
     * \return The loop tensor.
     */
    std::vector<ir::VarPtr> For(SymbolicScalar start, SymbolicScalar stop, SymbolicScalar step,
                                std::vector<std::pair<std::string, std::reference_wrapper<Tensor>>> carries,
                                LoopBody body, const char* file = __builtin_FILE(), int line = __builtin_LINE());

    /**
     * \brief Continue the loop.
     *
     * \param value The value to continue with.
     */
    template <typename... Args>
    void Continue(Args... args)
    {
        Continue({Unwrap(args)...});
    }

    /**
     * \brief Continue the loop.
     *
     * \param value The value to continue with.
     */
    void Continue(std::vector<ir::ExprPtr> value);

    /**
     * \brief Convert an expression to a tensor.
     *
     * \param e The expression to convert.
     * \return The tensor tensor.
     */
    Tensor AsTensor(const ir::ExprPtr& e);

    /**
     * \brief Convert an expression to a symbolic scalar.
     *
     * \param e The expression to convert.
     * \return The symbolic scalar tensor.
     */
    SymbolicScalar AsSymbol(const ir::ExprPtr& e);

private:
    ir::ExprPtr Unwrap(const Tensor& t);
    ir::ExprPtr Unwrap(SymbolicScalar s);
    ir::ExprPtr Unwrap(const ir::ExprPtr& e);
    ir::ExprPtr Unwrap(int x);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace npu::tile_fwk
