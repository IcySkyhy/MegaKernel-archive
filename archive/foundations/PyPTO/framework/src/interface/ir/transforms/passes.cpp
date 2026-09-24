/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/transforms/passes.h"

#include "ir/kind_traits.h"
#include "ir/memref.h"

#include "ir/transforms/utils/dead_code_elimination.h"
#include "ir/transforms/utils/canonicalize.h"
#include "ir/transforms/infer_token_pass.h"
#include "ir/transforms/merge_stmts_pass.h"
#include "ir/transforms/remove_redundant_token_pass.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/error.h"
#include "core/logging.h"
#include "interface/tensor/irbuilder.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/transforms/ir_property.h"
#include "ir/transforms/pass_context.h"
#include "ir/verifier/verifier.h"

namespace pypto {
namespace ir {

// Pass class implementation using pimpl pattern

Pass::Pass() : impl_(nullptr) {}

Pass::Pass(std::shared_ptr<PassImpl> impl) : impl_(std::move(impl)) {}

Pass::~Pass() = default;

Pass::Pass(const Pass& other) = default;
Pass& Pass::operator=(const Pass& other) = default;
Pass::Pass(Pass&& other) noexcept = default;
Pass& Pass::operator=(Pass&& other) noexcept = default;

ProgramPtr Pass::operator()(const ProgramPtr& program) const
{
    INTERNAL_CHECK(impl_) << "Pass has null implementation";
    INTERNAL_CHECK(program) << "Pass cannot run on null program";

    auto* ctx = PassContext::Current();
    if (ctx) {
        ctx->RunBeforePass(*this, program);
    }

    ProgramPtr result = (*impl_)(program);
    INTERNAL_CHECK(result) << "Pass '" << GetName() << "' returned null program";

    if (ctx) {
        ctx->RunAfterPass(*this, result);
    }

    return result;
}

ProgramPtr Pass::run(const ProgramPtr& program) const { return (*this)(program); }

std::string Pass::GetName() const
{
    if (!impl_) {
        return "NullPass";
    }
    return impl_->GetName();
}

IRPropertySet Pass::GetRequiredProperties() const
{
    if (!impl_) {
        return {};
    }
    return impl_->GetRequiredProperties();
}

IRPropertySet Pass::GetProducedProperties() const
{
    if (!impl_) {
        return {};
    }
    return impl_->GetProducedProperties();
}

IRPropertySet Pass::GetInvalidatedProperties() const
{
    if (!impl_) {
        return {};
    }
    return impl_->GetInvalidatedProperties();
}

// Utility pass implementations

namespace {

/**
 * \brief Pass implementation that wraps a program transform function
 */
class ProgramPassImpl : public PassImpl {
public:
    ProgramPassImpl(std::function<ProgramPtr(const ProgramPtr&)> transform, std::string name, PassProperties properties)
        : transform_(std::move(transform)), name_(std::move(name)), properties_(properties)
    {}

    ProgramPtr operator()(const ProgramPtr& program) override
    {
        INTERNAL_CHECK(program) << "ProgramPass cannot run on null program";
        return transform_(program);
    }

    [[nodiscard]] std::string GetName() const override { return name_.empty() ? "ProgramPass" : name_; }
    [[nodiscard]] IRPropertySet GetRequiredProperties() const override { return properties_.required; }
    [[nodiscard]] IRPropertySet GetProducedProperties() const override { return properties_.produced; }
    [[nodiscard]] IRPropertySet GetInvalidatedProperties() const override { return properties_.invalidated; }

private:
    std::function<ProgramPtr(const ProgramPtr&)> transform_;
    std::string name_;
    PassProperties properties_;
};

/**
 * \brief Pass implementation that applies a function transform to each function in program
 */
class FunctionPassImpl : public PassImpl {
public:
    FunctionPassImpl(std::function<FunctionPtr(const FunctionPtr&)> transform, std::string name,
                     PassProperties properties)
        : transform_(std::move(transform)), name_(std::move(name)), properties_(properties)
    {}

    ProgramPtr operator()(const ProgramPtr& program) override
    {
        INTERNAL_CHECK(program) << "FunctionPass cannot run on null program";

        // Apply the function transform to each function in the program
        std::vector<FunctionPtr> transformed_functions;
        transformed_functions.reserve(program->functions_.size());

        for (const auto& entry : program->functions_) {
            const auto& func = entry.second;
            FunctionPtr transformed_func = transform_(func);
            transformed_functions.push_back(transformed_func);
        }

        // Create a new program with the transformed functions
        return std::make_shared<const Program>(transformed_functions, program->name_, program->span_);
    }

    [[nodiscard]] std::string GetName() const override { return name_.empty() ? "FunctionPass" : name_; }
    [[nodiscard]] IRPropertySet GetRequiredProperties() const override { return properties_.required; }
    [[nodiscard]] IRPropertySet GetProducedProperties() const override { return properties_.produced; }
    [[nodiscard]] IRPropertySet GetInvalidatedProperties() const override { return properties_.invalidated; }

private:
    std::function<FunctionPtr(const FunctionPtr&)> transform_;
    std::string name_;
    PassProperties properties_;
};

} // namespace

// Factory functions for utility passes
namespace pass {

Pass CreateProgramPass(std::function<ProgramPtr(const ProgramPtr&)> transform, const std::string& name,
                       const PassProperties& properties)
{
    return Pass(std::make_shared<ProgramPassImpl>(std::move(transform), name, properties));
}

Pass CreateFunctionPass(std::function<FunctionPtr(const FunctionPtr&)> transform, const std::string& name,
                        const PassProperties& properties)
{
    return Pass(std::make_shared<FunctionPassImpl>(std::move(transform), name, properties));
}

Pass RunVerifier(const std::vector<std::string>& disabled_rules)
{
    auto disabled_rules_snapshot = std::make_shared<const std::vector<std::string>>(disabled_rules);
    return CreateProgramPass(
        [disabled_rules_snapshot](const ProgramPtr& program) -> ProgramPtr {
            // Create default verifier with all rules
            IRVerifier verifier = IRVerifier::CreateDefault();

            // Disable requested rules
            for (const auto& rule_name : *disabled_rules_snapshot) {
                verifier.DisableRule(rule_name);
            }

            // Run verification and collect diagnostics
            auto diagnostics = verifier.Verify(program);
            // Log diagnostics
            if (!diagnostics.empty()) {
                std::string report = IRVerifier::GenerateReport(diagnostics);
                IR_LOGI() << "IR Verification Report:\n" << report;
            }

            // Return the same program (verification doesn't modify IR)
            return program;
        },
        "IRVerifier");
}

Pass Canonicalize()
{
    return CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            if (!func || !func->body_)
                return func;
            auto body = CanonicalizeSeqStmts(func->body_);
            if (body == func->body_)
                return func;
            return std::make_shared<const Function>(func->name_, func->params_, func->returnTypes_, body, func->span_,
                                                    func->funcType_, func->entry_, func->attrs_);
        },
        "Canonicalize");
}

Pass InferTokenPass()
{
    return CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            if (!func || !func->body_)
                return func;
            if (!npu::tile_fwk::IRContext::Get().AssembleNewLogicalTensor())
                return func;
            auto seq = SeqStmts::AsMut(func->body_);
            auto inferredBody = ::pypto::ir::RunInferTokenPass(seq);
            if (inferredBody == func->body_)
                return func;
            return std::make_shared<const Function>(func->name_, func->params_, func->returnTypes_, inferredBody,
                                                    func->span_, func->funcType_, func->entry_, func->attrs_);
        },
        "InferTokenPass");
}

Pass RemoveRedundantTokenPass()
{
    return CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            if (!func || !func->body_)
                return func;
            if (!npu::tile_fwk::IRContext::Get().AssembleNewLogicalTensor())
                return func;
            auto body = ::pypto::ir::RunRemoveRedundantTokenPass(SeqStmts::AsMut(func->body_));
            if (body == func->body_)
                return func;
            return std::make_shared<const Function>(func->name_, func->params_, func->returnTypes_, body, func->span_,
                                                    func->funcType_, func->entry_);
        },
        "RemoveRedundantTokenPass");
}

Pass MergeStmtsIntoIf()
{
    return CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            if (!func || !func->body_)
                return func;
            std::vector<std::string> extVarNames;
            for (const auto& param : func->params_) {
                if (param) {
                    extVarNames.push_back(param->name_);
                }
            }
            auto seq = SeqStmts::AsMut(func->body_);
            auto mergedBody = MergeStmtsIntoIfStmt(seq, extVarNames);
            if (mergedBody == func->body_)
                return func;
            return std::make_shared<const Function>(func->name_, func->params_, func->returnTypes_, mergedBody,
                                                    func->span_, func->funcType_, func->entry_, func->attrs_);
        },
        "MergeStmtsIntoIf");
}

} // namespace pass

// ---------------------------------------------------------------------------
// Stub (no-op) implementations for passes whose .cpp files were removed.
// Each returns the program unchanged.  The factory functions are still declared
// in passes.h and referenced from Python bindings.
// ---------------------------------------------------------------------------
namespace pass {

static Pass MakeIdentityPass(const std::string& name)
{
    return CreateProgramPass([](const ProgramPtr& p) { return p; }, name);
}

Pass InitMemRef() { return MakeIdentityPass("InitMemRef"); }
Pass BasicMemoryReuse() { return MakeIdentityPass("BasicMemoryReuse"); }
Pass AllocateMemoryAddr() { return MakeIdentityPass("AllocateMemoryAddr"); }
Pass OutlineIncoreScopes() { return MakeIdentityPass("OutlineIncoreScopes"); }
Pass ConvertTensorToBlockOps() { return MakeIdentityPass("ConvertTensorToBlockOps"); }
Pass FlattenCallExpr() { return MakeIdentityPass("FlattenCallExpr"); }
Pass NormalizeStmtStructure() { return MakeIdentityPass("NormalizeStmtStructure"); }
Pass FlattenSingleStmt() { return MakeIdentityPass("FlattenSingleStmt"); }

} // namespace pass

// PassPipeline implementation

PassPipeline::PassPipeline() = default;

void PassPipeline::AddPass(Pass pass) { passes_.push_back(std::move(pass)); }

ProgramPtr PassPipeline::Run(const ProgramPtr& program) const
{
    ProgramPtr current = program;

    // Automatic verification: verify lightweight properties exactly once each.
    // Level comes from active PassContext, or env-var default if no context.
    for (const auto& p : passes_) {
        current = p(current);
    }
    return current;
}

std::vector<std::string> PassPipeline::GetPassNames() const
{
    std::vector<std::string> names;
    names.reserve(passes_.size());
    for (const auto& p : passes_) {
        names.push_back(p.GetName());
    }
    return names;
}

} // namespace ir
} // namespace pypto
