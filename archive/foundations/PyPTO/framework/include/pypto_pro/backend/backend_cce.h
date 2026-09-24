/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_BACKEND_BACKEND_CCE_H_
#define PYPTO_BACKEND_BACKEND_CCE_H_

#include <string>

#include "backend/common/backend.h"

namespace pypto {
namespace backend {

class BackendCCE : public Backend {
public:
    static BackendCCE& Instance();

    [[nodiscard]] std::string GetTypeName() const override { return "CCE"; }

private:
    BackendCCE();
};

} // namespace backend
} // namespace pypto

#endif // PYPTO_BACKEND_BACKEND_CCE_H_
