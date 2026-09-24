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
 * \file device_ctrl.cpp
 * \brief DO NOT MODIFY THIS FILE!!!
 *      This file is an entry file which is only used to export symbol.
 *      Please modify the implementation of *DeviceKernelArgs* when needed.
 */

#include "device_ctrl.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
DeviceCtrlMachine g_ctrl_machine;
}

extern "C" __attribute__((visibility("default"))) int PyptoKernelCtrlServerRegisterTaskInspector(
    DeviceTaskInspectorEntry inspectorEntry, void* inspector)
{
    g_ctrl_machine.RegisterTaskInspector(inspectorEntry, inspector);
    return 0;
}

extern "C" __attribute__((visibility("default"))) int PyptoKernelCtrlServerInit(void* targ)
{
    DeviceKernelArgs* kargs = (DeviceKernelArgs*)targ;
    return g_ctrl_machine.EntryInit(kargs);
}

extern "C" __attribute__((visibility("default"))) int PyptoKernelCtrlServer(void* targ)
{
    DeviceKernelArgs* kargs = (DeviceKernelArgs*)targ;
    return g_ctrl_machine.EntryMain(kargs);
}
