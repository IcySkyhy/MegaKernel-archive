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
 * \file pypto_bundle_api.h
 * \brief Two-operation C ABI for a PyPTO kernel bundle (.pyptokb): Workspace + Launch, in a path-based and an
 *        in-memory flavor.
 *
 * Each entry takes a single unified tensor list (all operands including outputs, indexed by flat position). Init /
 * Load / Unload / device-memory are internalized: an idempotent load (cached by hashKey) seeds the AiCpu .so from
 * the bundle, arch-checks, and registers the kernel, all lazily.
 */

#ifndef PYPTO_BUNDLE_API_H
#define PYPTO_BUNDLE_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {      // one tensor operand passed across the C ABI
    void* addr;       // device memory address
    int32_t dataType; // matches the internal DataType enum (tilefwk/data_type.h)
    int32_t rank;
    int64_t shape[8]; // up to 8 dims
} PyptoTensorDesc;

/* Workspace query for the given tensor shapes. Returns the workspace bytes the caller must allocate before Launch
 * (0 on failure / no workspace). Pure host computation: does NOT initialize the device. */
uint64_t PyptoWorkspace(const char* path, const PyptoTensorDesc* tensors, uint32_t nTensors);

/* Launch the bundle on the given tensors. workspaceAddr may be NULL when PyptoWorkspace returned 0. The first
 * Launch of the process lazily initializes the runtime. sync!=0 waits for completion. Returns 0 on success. */
int PyptoLaunch(const char* path, const PyptoTensorDesc* tensors, uint32_t nTensors, void* workspaceAddr, void* stream,
                int sync);

/* In-memory variants: same semantics, but `bundleData`/`bundleSize` is a caller-loaded .pyptokb byte image. */
uint64_t PyptoWorkspaceFromMemory(const void* bundleData, uint64_t bundleSize, const PyptoTensorDesc* tensors,
                                  uint32_t nTensors);

int PyptoLaunchFromMemory(const void* bundleData, uint64_t bundleSize, const PyptoTensorDesc* tensors,
                          uint32_t nTensors, void* workspaceAddr, void* stream, int sync);

#ifdef __cplusplus
}
#endif

#endif // PYPTO_BUNDLE_API_H
