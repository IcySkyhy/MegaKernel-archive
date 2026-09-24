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
 * \file adapter_manager.h
 * \brief
 */

#pragma once

#include "adapter/manager/types/acl_adapter_types.h"
#include "adapter/manager/types/adump_adapter_types.h"
#include "adapter/manager/types/hal_adapter_types.h"
#include "adapter/manager/types/hccl_adapter_types.h"
#include "adapter/manager/types/msprof_adapter_types.h"
#include "adapter/manager/types/runtime_adapter_types.h"
#include "adapter/manager/cann_adapter.h"

namespace npu::tile_fwk {
class AdapterManager {
public:
    static AdapterManager& Instance();
    const CannAdapter<AclFunc>& GetAclAdapter() const { return aclAdapter_; }
    const CannAdapter<AdumpFunc>& GetAdumpAdapter() const { return adumpAdapter_; }
    const CannAdapter<HalFunc>& GetHalAdapter() const { return halAdapter_; }
    const CannAdapter<HcclFunc>& GetHcclAdapter() const { return hcclAdapter_; }
    const CannAdapter<MsprofFunc>& GetMsprofAdapter() const { return msprofAdapter_; }
    const CannAdapter<RuntimeFunc>& GetRuntimeAdapter() const { return runtimeAdapter_; }

private:
    AdapterManager();
    ~AdapterManager();
    CannAdapter<AclFunc> aclAdapter_;
    CannAdapter<AdumpFunc> adumpAdapter_;
    CannAdapter<HalFunc> halAdapter_;
    CannAdapter<HcclFunc> hcclAdapter_;
    CannAdapter<MsprofFunc> msprofAdapter_;
    CannAdapter<RuntimeFunc> runtimeAdapter_;
};
} // namespace npu::tile_fwk
