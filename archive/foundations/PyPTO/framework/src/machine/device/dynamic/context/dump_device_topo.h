/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dump_device_topo.h
 * \brief
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "machine/device/dynamic/context/device_slot_context.h"
#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/utils/dynamic/dev_encode.h"

namespace npu::tile_fwk::dynamic::topo_dump {

void DumpProducerCellAccess(uint32_t devTaskId, int slotIdx, uint32_t devNextIdx, DevAscendFunction& devRootSrc,
                            DevAscendFunctionOutcast& outcast, const DeviceExecuteSlot& slot,
                            const uint64_t* expressionList);

/// Dump a single consumer cell access event (read) for one (slot, op).
void DumpConsumerCellAccess(uint32_t devTaskId, int slotIdx, uint32_t devNextIdx, DevAscendFunction& nextSrc,
                            const DevAscendFunctionCallOperandUse& consumer,
                            const DevCellMatchTableDesc& cellMatchTableDesc, const uint64_t* expressionList);

/// Dump a single stitch edge to dyn_stitch_edges.csv.
void DumpStitchEdge(const DevAscendFunctionDupped& producerDup, const DevAscendFunctionDupped& consumerDup,
                    size_t producerOperationIdx, size_t consumerIdx, size_t consumerOperationIdx,
                    DeviceStitchContext::StitchKind stitchKind, int slotIdx);

} // namespace npu::tile_fwk::dynamic::topo_dump
