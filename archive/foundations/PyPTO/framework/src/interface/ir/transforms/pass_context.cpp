/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/transforms/pass_context.h"

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "core/error.h"
#include "core/logging.h"
#include "ir/program.h"
#include "ir/transforms/ir_property.h"
#include "ir/transforms/passes.h"
#include "ir/verifier/verifier.h"
#include "ir/transforms/pass_context.h"
namespace pypto {
namespace ir {
// Thread-local current context (top of stack)
thread_local PassContext* PassContext::current_ = nullptr;
// CallbackInstrument
CallbackInstrument::CallbackInstrument(Callback before_pass, Callback after_pass, std::string name)
    : before_pass_(std::move(before_pass)), after_pass_(std::move(after_pass)), name_(std::move(name))
{}

void CallbackInstrument::RunBeforePass(const Pass& pass, const ProgramPtr& program)
{
    if (before_pass_)
        before_pass_(pass, program);
}

void CallbackInstrument::RunAfterPass(const Pass& pass, const ProgramPtr& program)
{
    if (after_pass_)
        after_pass_(pass, program);
}

std::string CallbackInstrument::GetName() const { return name_; }

// PassContext
PassContext::PassContext(std::vector<PassInstrumentPtr> instruments, VerificationLevel verification_level)
    : instruments_(std::move(instruments)), verification_level_(verification_level), previous_(nullptr)
{}

VerificationLevel PassContext::GetVerificationLevel() const { return verification_level_; }

const std::vector<PassInstrumentPtr>& PassContext::GetInstruments() const { return instruments_; }

void PassContext::EnterContext()
{
    previous_ = current_;
    current_ = this;
}

void PassContext::ExitContext()
{
    INTERNAL_CHECK(current_ == this)
        << "PassContext::ExitContext called out of order or without a matching EnterContext";
    current_ = previous_;
    previous_ = nullptr;
}

void PassContext::RunBeforePass(const Pass& pass, const ProgramPtr& program)
{
    for (const auto& instrument : instruments_) {
        INTERNAL_CHECK(instrument != nullptr) << "PassContext contains a null PassInstrument";
        instrument->RunBeforePass(pass, program);
    }
}

void PassContext::RunAfterPass(const Pass& pass, const ProgramPtr& program)
{
    for (const auto& instrument : instruments_) {
        INTERNAL_CHECK(instrument != nullptr) << "PassContext contains a null PassInstrument";
        instrument->RunAfterPass(pass, program);
    }
}

PassContext* PassContext::Current() { return current_; }

} // namespace ir
} // namespace pypto
