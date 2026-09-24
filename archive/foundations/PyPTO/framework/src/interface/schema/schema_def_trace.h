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
 * \file schema_def_trace.h
 * \brief
 */

#pragma once

SCHEMA_DEF_TYPE_UNION(DeviceTaskIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(DupIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(RootIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(OperationIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(LeafIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(SlotIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(IOperandIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(OOperandIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(IncastIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(OutcastIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(InputIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(OutputIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(AicoreIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(ControlThreadIndexType, none, Int64Type);
SCHEMA_DEF_TYPE_UNION(ScheduleThreadIndexType, none, Int64Type);

SCHEMA_DEF_TYPE_UNION(DupAddress, none, AddressType);
SCHEMA_DEF_TYPE_UNION(RawTensorAddress, none, AddressType);

SCHEMA_DEF_TYPE_UNION(DimShape, none, CoordType);
SCHEMA_DEF_TYPE_UNION(DimOffset, none, CoordType);

SCHEMA_DEF_ATTR(LUid, DeviceTaskIndexType, DupIndexType, RootIndexType, OperationIndexType, LeafIndexType);
SCHEMA_DEF_ATTR(RUid, DeviceTaskIndexType, DupIndexType, RootIndexType);
SCHEMA_DEF_ATTR(DUid, DeviceTaskIndexType);

SCHEMA_DEF_TYPE_UNION(LUidType, LUid);
SCHEMA_DEF_TYPE_UNION(RUidType, RUid);
SCHEMA_DEF_TYPE_UNION(DUidType, DUid);

SCHEMA_DEF_ATTR(LActStart, AicoreIndexType);
SCHEMA_DEF_ATTR(LActFinish, AicoreIndexType);
SCHEMA_DEF_ATTR(LActIncastCount, Int64Type);
SCHEMA_DEF_ATTR(LActIncast, shape, offset, range);
SCHEMA_DEF_ATTR(LActOutcastCount, Int64Type);
SCHEMA_DEF_ATTR(LActOutcast, shape, offset, range);
SCHEMA_DEF_TYPE_UNION(LActType, LActStart, LActFinish, coa, succ, LActIncastCount, LActIncast, LActOutcastCount,
                      LActOutcast);

SCHEMA_DEF_ATTR(RActDup, name);
SCHEMA_DEF_ATTR(RActStitch);
SCHEMA_DEF_ATTR(RActIncastCount, Int64Type);
SCHEMA_DEF_ATTR(RActIncast, incast, range);
SCHEMA_DEF_ATTR(RActOutcastCount, Int64Type);
SCHEMA_DEF_ATTR(RActOutcast, outcast, range);
SCHEMA_DEF_ATTR(RActRawTensorCount, Int64Type);
SCHEMA_DEF_ATTR(RActRawTensor, rawTensor, rawDesc);
SCHEMA_DEF_ATTR(RActExpressionCount, Int64Type);
SCHEMA_DEF_ATTR(RActWorkspace, range);
SCHEMA_DEF_TYPE_UNION(RActType, RActDup, RActStitch, RActIncastCount, RActIncast, RActOutcastCount, RActOutcast, expr,
                      RActRawTensorCount, RActRawTensor, RActWorkspace, RActExpressionCount);

SCHEMA_DEF_ATTR(Producer, LUidType, OOperandIndexType, OutcastIndexType, SlotIndexType, DimOffset, DimShape);
SCHEMA_DEF_TYPE_UNION(ProducerType, Producer);

SCHEMA_DEF_ATTR(Consumer, LUidType, IOperandIndexType, IncastIndexType, SlotIndexType, DimOffset, DimShape);
SCHEMA_DEF_TYPE_UNION(ConsumerType, Consumer);

SCHEMA_DEF_ATTR(StitchReasonSomeMatch);
SCHEMA_DEF_ATTR(StitchReasonUniqueMatch);
SCHEMA_DEF_ATTR(StitchReasonOneToManyMatch);
SCHEMA_DEF_ATTR(StitchReasonManyToOneMatch);
SCHEMA_DEF_ATTR(StitchReasonWorkspaceReuse);
SCHEMA_DEF_TYPE_UNION(StitchReasonType, StitchReasonSomeMatch, StitchReasonUniqueMatch, StitchReasonOneToManyMatch,
                      StitchReasonManyToOneMatch, StitchReasonWorkspaceReuse);

SCHEMA_DEF_ATTR(DActSubmit, Int64Type);
SCHEMA_DEF_ATTR(DActStitchStart, RUidType);
SCHEMA_DEF_ATTR(DActStitchFinish, RUidType);
SCHEMA_DEF_ATTR(DActStitchEdge, ProducerType, ConsumerType, StitchReasonType);
SCHEMA_DEF_TYPE_UNION(DActType, DActSubmit, DActStitchStart, DActStitchFinish, DActStitchEdge);

SCHEMA_DEF_ATTR(LEvent, LUidType, LActType);
SCHEMA_DEF_ATTR(REvent, RUidType, RActType);
SCHEMA_DEF_ATTR(DEvent, DUidType, DActType);

SCHEMA_DEF_ATTR(ThreadStart);
SCHEMA_DEF_ATTR(ThreadFinish);
SCHEMA_DEF_ATTR(ControlFlowCacheFullRunCache);
SCHEMA_DEF_ATTR(ControlFlowCacheFullRunControl);
SCHEMA_DEF_ATTR(ControlFlowCachePartRunCache, Int64Type, Int64Type);
SCHEMA_DEF_ATTR(ControlFlowCachePartRunControl);
SCHEMA_DEF_ATTR(ControlFlowCachePartRunControlContinue);
SCHEMA_DEF_ATTR(RawWorkspace, AddressType, AddressType);
SCHEMA_DEF_ATTR(Workspace, range);
SCHEMA_DEF_ATTR(WorkspaceMetadataGeneral, range);
SCHEMA_DEF_ATTR(WorkspaceMetadataStitch, range);
SCHEMA_DEF_ATTR(WorkspaceInnerTensor, range);
SCHEMA_DEF_ATTR(WorkspacePartialOutcast, range);
SCHEMA_DEF_ATTR(WorkspaceInDeviceTaskOutcast, range);
SCHEMA_DEF_ATTR(WorkspaceCrossDeviceTaskOutcast, range);
SCHEMA_DEF_ATTR(WorkspaceSpill, mem, Int64Type, range);
SCHEMA_DEF_ATTR(InputTensorCount, Int64Type);
SCHEMA_DEF_ATTR(InputTensorRange, Int64Type, range);
SCHEMA_DEF_ATTR(InputTensorElement, Int64Type, AddressType, Int64Type);
SCHEMA_DEF_ATTR(OutputTensorCount, Int64Type);
SCHEMA_DEF_ATTR(OutputTensorRange, Int64Type, range);
SCHEMA_DEF_ATTR(OutputTensorElement, Int64Type, AddressType, Int64Type);

SCHEMA_DEF_TYPE_UNION(CtrlActType, ThreadStart, ThreadFinish, ControlFlowCacheFullRunCache,
                      ControlFlowCacheFullRunControl, ControlFlowCachePartRunCache, ControlFlowCachePartRunControl,
                      ControlFlowCachePartRunControlContinue, RawWorkspace, Workspace, WorkspaceMetadataGeneral,
                      WorkspaceMetadataStitch, WorkspaceInnerTensor, WorkspacePartialOutcast,
                      WorkspaceInDeviceTaskOutcast, WorkspaceCrossDeviceTaskOutcast, WorkspaceSpill, InputTensorCount,
                      InputTensorRange, InputTensorElement, OutputTensorCount, OutputTensorRange, OutputTensorElement);
SCHEMA_DEF_TYPE_UNION(ScheActType, ThreadStart, ThreadFinish);

SCHEMA_DEF_ATTR(CtrlEvent, ControlThreadIndexType, CtrlActType);
SCHEMA_DEF_ATTR(ScheEvent, ScheduleThreadIndexType, ScheActType);
