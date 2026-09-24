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
 * \file operator_tracer.h
 * \brief
 */

#pragma once

#include <sys/cdefs.h>
#include <cstdint>

namespace npu::tile_fwk {
class OperatorChecker {
public:
    void PreCheck();
    void PostCheck();

private:
    int preOpCount;
    int preMagic;
    int preOp;
    int preRawMagic;
    int enable;
};

struct OperatorTracer {
    OperatorTracer(const void*)
    {
        if (IsCheckerEnabled()) {
            checker.PreCheck();
        }
    }
    ~OperatorTracer()
    {
        if (IsCheckerEnabled())
            checker.PostCheck();
    }
    OperatorChecker checker;

    bool IsCheckerEnabled() const;
};
} // namespace npu::tile_fwk

#define DECLARE_TRACER() auto __tracer = npu::tile_fwk::OperatorTracer(__builtin_return_address(0))
#define DECLARE_TRACERX(lr) auto __tracer = npu::tile_fwk::OperatorTracer(lr)
