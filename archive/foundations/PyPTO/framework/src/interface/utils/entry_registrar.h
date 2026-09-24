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
 * \file entry_registrar.h
 * \brief
 */

#pragma once

namespace npu::tile_fwk {

struct EntryRegistrarNode {
    typedef void (*Entry)(void* data);

    Entry entry{nullptr};
    const char* name{nullptr};

    EntryRegistrarNode* next{nullptr};

    template <typename T>
    EntryRegistrarNode(T& group, Entry entryArg, const char* nameArg = nullptr) : entry(entryArg), name(nameArg)
    {
        group.Append(this);
    }
};

struct EntryRegistrarGroup {
    EntryRegistrarNode* head{nullptr};

    void Append(EntryRegistrarNode* node);

    // Register one fresh attribute instance per type into the given manager.
    void Init(void* data);
};

} // namespace npu::tile_fwk
