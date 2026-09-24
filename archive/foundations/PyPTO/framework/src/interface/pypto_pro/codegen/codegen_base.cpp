/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/codegen_base.h"

#include <optional>
#include <string>

#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir; // NOLINT(build/namespaces)

std::string CodegenBase::TryGetVarName(const ir::ExprPtr& expr) const
{
    if (auto var = As<Var>(expr)) {
        return var->name_;
    }
    return "";
}

namespace {

std::string BuildBinaryExprString(const CodegenBase& codegen, const ir::ExprPtr& left, const std::string& op,
                                  const ir::ExprPtr& right)
{
    return "(" + codegen.GenerateExprString(left) + " " + op + " " + codegen.GenerateExprString(right) + ")";
}

std::string GetScalarCTypeString(const ir::ExprPtr& expr, const std::string& op_name)
{
    auto scalar_type = As<ScalarType>(expr->GetType());
    INTERNAL_CHECK(scalar_type) << "Internal error: " << op_name << " expression should have ScalarType, got "
                                << expr->GetType()->TypeName();
    return scalar_type->dtype_.ToCTypeString();
}

std::optional<std::string> TryGenerateSimpleExprString(const CodegenBase& codegen, const ir::ExprPtr& expr)
{
    std::string var_name = codegen.TryGetVarName(expr);
    if (!var_name.empty()) {
        return var_name;
    }
    if (auto const_int = As<ConstInt>(expr)) {
        return std::to_string(const_int->value_);
    }
    if (auto const_float = As<ConstFloat>(expr)) {
        return std::to_string(const_float->value_);
    }
    if (auto const_bool = As<ConstBool>(expr)) {
        return const_bool->value_ ? "1" : "0";
    }
    return std::nullopt;
}

std::optional<std::string> TryGenerateBinaryExprString(const CodegenBase& codegen, const ir::ExprPtr& expr)
{
    if (auto add = As<Add>(expr)) {
        return BuildBinaryExprString(codegen, add->left_, "+", add->right_);
    }
    if (auto sub = As<Sub>(expr)) {
        return BuildBinaryExprString(codegen, sub->left_, "-", sub->right_);
    }
    if (auto mul = As<Mul>(expr)) {
        return BuildBinaryExprString(codegen, mul->left_, "*", mul->right_);
    }
    if (auto floor_div = As<FloorDiv>(expr)) {
        return BuildBinaryExprString(codegen, floor_div->left_, "/", floor_div->right_);
    }
    if (auto floor_mod = As<FloorMod>(expr)) {
        return BuildBinaryExprString(codegen, floor_mod->left_, "%", floor_mod->right_);
    }
    if (auto eq = As<Eq>(expr)) {
        return BuildBinaryExprString(codegen, eq->left_, "==", eq->right_);
    }
    if (auto ne = As<Ne>(expr)) {
        return BuildBinaryExprString(codegen, ne->left_, "!=", ne->right_);
    }
    if (auto lt = As<Lt>(expr)) {
        return BuildBinaryExprString(codegen, lt->left_, "<", lt->right_);
    }
    if (auto le = As<Le>(expr)) {
        return BuildBinaryExprString(codegen, le->left_, "<=", le->right_);
    }
    if (auto gt = As<Gt>(expr)) {
        return BuildBinaryExprString(codegen, gt->left_, ">", gt->right_);
    }
    if (auto ge = As<Ge>(expr)) {
        return BuildBinaryExprString(codegen, ge->left_, ">=", ge->right_);
    }
    return std::nullopt;
}

std::optional<std::string> TryGenerateMinMaxExprString(const CodegenBase& codegen, const ir::ExprPtr& expr)
{
    if (auto min_expr = As<Min>(expr)) {
        std::string cpp_type = GetScalarCTypeString(expr, "Min");
        return "std::min<" + cpp_type + ">(" + codegen.GenerateExprString(min_expr->left_) + ", " +
               codegen.GenerateExprString(min_expr->right_) + ")";
    }
    if (auto max_expr = As<Max>(expr)) {
        std::string cpp_type = GetScalarCTypeString(expr, "Max");
        return "std::max<" + cpp_type + ">(" + codegen.GenerateExprString(max_expr->left_) + ", " +
               codegen.GenerateExprString(max_expr->right_) + ")";
    }
    return std::nullopt;
}

std::optional<std::string> TryGenerateUnaryExprString(const CodegenBase& codegen, const ir::ExprPtr& expr)
{
    if (auto neg = As<Neg>(expr)) {
        return "(-" + codegen.GenerateExprString(neg->operand_) + ")";
    }
    if (auto cast_expr = As<Cast>(expr)) {
        std::string cpp_type = GetScalarCTypeString(expr, "Cast");
        return "static_cast<" + cpp_type + ">(" + codegen.GenerateExprString(cast_expr->operand_) + ")";
    }
    return std::nullopt;
}

std::optional<std::string> TryGenerateTupleGetItemExprString(const CodegenBase& codegen, const ir::ExprPtr& expr)
{
    auto get_item = As<GetItemExpr>(expr);
    if (!get_item || !As<TupleType>(get_item->value_->GetType())) {
        return std::nullopt;
    }

    auto const_idx = As<ConstInt>(get_item->slice_);
    INTERNAL_CHECK(const_idx) << "GetItemExpr on a tuple requires a ConstInt slice, got "
                              << get_item->slice_->TypeName();
    return codegen.GenerateExprString(get_item->value_) + "_" + std::to_string(const_idx->value_);
}

} // namespace

std::string CodegenBase::GenerateExprString(const ir::ExprPtr& expr) const
{
    if (auto simple = TryGenerateSimpleExprString(*this, expr)) {
        return *simple;
    }
    if (auto binary = TryGenerateBinaryExprString(*this, expr)) {
        return *binary;
    }
    if (auto min_max = TryGenerateMinMaxExprString(*this, expr)) {
        return *min_max;
    }
    if (auto unary = TryGenerateUnaryExprString(*this, expr)) {
        return *unary;
    }
    if (auto get_item = TryGenerateTupleGetItemExprString(*this, expr)) {
        return *get_item;
    }
    throw NotImplementedError("GenerateExprString not implemented for expression type: " + expr->TypeName());
}

std::string CodegenBase::GetRuntimeDataTypeString(const ir::DataType& dtype) const
{
    if (dtype == DataType::FP16)
        return "DataType::FLOAT16";
    if (dtype == DataType::FP32)
        return "DataType::FLOAT32";
    if (dtype == DataType::INT32)
        return "DataType::INT32";
    if (dtype == DataType::INT16)
        return "DataType::INT16";
    if (dtype == DataType::INT8)
        return "DataType::INT8";
    if (dtype == DataType::UINT8)
        return "DataType::UINT8";
    if (dtype == DataType::BF16)
        return "DataType::BFLOAT16";
    // INDEX is a semantic type in the IR; the runtime represents it as INT64
    if (dtype == DataType::INDEX)
        return "DataType::INT64";
    if (dtype == DataType::INT64)
        return "DataType::INT64";
    if (dtype == DataType::UINT64)
        return "DataType::UINT64";
    return "DataType::UNKNOWN";
}

} // namespace codegen
} // namespace pypto
