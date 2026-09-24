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
 * \file device_error_tracking.cpp
 * \brief
 */

#include "machine/runtime/runner/device_error_tracking.h"
#include <cstdint>
#include <iostream>
#include "adapter/api/acl_api.h"
#include "tilefwk/device_error_code.h"

namespace npu::tile_fwk {

struct ErrorCodeEntry {
    int32_t retcode;
    const char* msg;
};

// Sorted by retcode in ascending order, for binary search.
static const ErrorCodeEntry kErrorCodeTable[] = {
    // 107xxx
    {PYPTO_DEVICE_ERROR_PARAM_INVALID, "param invalid"},
    {PYPTO_DEVICE_ERROR_INVALID_DEVICEID, "invalid device id"},
    {PYPTO_DEVICE_ERROR_CONTEXT_NULL, "current context null"},
    {PYPTO_DEVICE_ERROR_STREAM_CONTEXT, "stream not in current context"},
    {PYPTO_DEVICE_ERROR_MODEL_CONTEXT, "model not in current context"},
    {PYPTO_DEVICE_ERROR_STREAM_MODEL, "stream not in model"},
    {PYPTO_DEVICE_ERROR_EVENT_TIMESTAMP_INVALID, "event timestamp invalid"},
    {PYPTO_DEVICE_ERROR_EVENT_TIMESTAMP_REVERSAL, "event timestamp reversal"},
    {PYPTO_DEVICE_ERROR_ADDR_UNALIGNED, "memory address unaligned"},
    {PYPTO_DEVICE_ERROR_FILE_OPEN, "open file failed"},
    {PYPTO_DEVICE_ERROR_FILE_WRITE, "write file failed"},
    {PYPTO_DEVICE_ERROR_STREAM_SUBSCRIBE, "error subscribe stream"},
    {PYPTO_DEVICE_ERROR_THREAD_SUBSCRIBE, "error subscribe thread"},
    {PYPTO_DEVICE_ERROR_GROUP_NOT_SET, "group not set"},
    {PYPTO_DEVICE_ERROR_GROUP_NOT_CREATE, "group not create"},
    {PYPTO_DEVICE_ERROR_STREAM_NO_CB_REG, "callback not register to stream"},
    {PYPTO_DEVICE_ERROR_INVALID_MEMORY_TYPE, "invalid memory type"},
    {PYPTO_DEVICE_ERROR_INVALID_HANDLE, "invalid handle"},
    {PYPTO_DEVICE_ERROR_INVALID_MALLOC_TYPE, "invalid malloc type"},
    {PYPTO_DEVICE_ERROR_WAIT_TIMEOUT, "wait timeout"},
    {PYPTO_DEVICE_ERROR_TASK_TIMEOUT, "task timeout"},
    {PYPTO_DEVICE_ERROR_SYSPARAMOPT_NOT_SET, "not set sysparamopt"},
    {PYPTO_DEVICE_ERROR_DEVICE_TASK_ABORT, "device task aborting"},
    {PYPTO_DEVICE_ERROR_STREAM_ABORT, "stream aborting"},
    {PYPTO_DEVICE_ERROR_CAPTURE_DEPENDENCY, "capture dependency failure"},
    {PYPTO_DEVICE_ERROR_STREAM_UNJOINED, "invalid capture model"},
    {PYPTO_DEVICE_ERROR_MODEL_CAPTURED, "model is captured"},
    {PYPTO_DEVICE_ERROR_STREAM_CAPTURED, "stream is captured"},
    {PYPTO_DEVICE_ERROR_EVENT_CAPTURED, "event is captured"},
    {PYPTO_DEVICE_ERROR_STREAM_NOT_CAPTURED, "stream is not in capture status"},
    {PYPTO_DEVICE_ERROR_CAPTURE_MODE_NOT_SUPPORT, "stream is captured, not support current oper"},
    {PYPTO_DEVICE_ERROR_STREAM_CAPTURE_IMPLICIT, "a disallowed implicit dependency from default stream"},
    {PYPTO_DEVICE_ERROR_STREAM_CAPTURE_CONFLICT, "interdependent stream cannot begin capture together"},
    {PYPTO_DEVICE_ERROR_STREAM_TASK_GROUP_STATUS, "task group status error"},
    {PYPTO_DEVICE_ERROR_STREAM_TASK_GROUP_INTR, "task group interrupted"},
    {PYPTO_DEVICE_ERROR_TASK_ABORT_STOP, "device task aborting stop before post process"},
    {PYPTO_DEVICE_ERROR_STREAM_CAPTURE_UNMATCHED, "the capture was not initiated in this stream"},
    {PYPTO_DEVICE_ERROR_MODEL_RUNNING, "the model is still running"},
    {PYPTO_DEVICE_ERROR_STREAM_CAPTURE_WRONG_THREAD, "the thread of end capture and begin capture is not same"},
    {PYPTO_DEVICE_ERROR_INSUFFICIENT_INPUT_ARRAY, "input array capacity insufficient"},
    {PYPTO_DEVICE_ERROR_MODEL_UPDATE_FAILED, "the model update failed"},
    {PYPTO_DEVICE_ERROR_CAPTURE_MODE_BLOCK_ASYNC, "async oper convert to sync oper, stream is captured"},
    {PYPTO_DEVICE_ERROR_SYMBOL_NOT_FOUND, "symbol not found"},
    // 207xxx
    {PYPTO_DEVICE_ERROR_FEATURE_NOT_SUPPORT, "feature not support"},
    {PYPTO_DEVICE_ERROR_MEMORY_ALLOCATION, "memory allocation error"},
    {PYPTO_DEVICE_ERROR_MEMORY_FREE, "memory free error"},
    {PYPTO_DEVICE_ERROR_AICORE_OVER_FLOW, "aicore over flow"},
    {PYPTO_DEVICE_ERROR_NO_DEVICE, "no device"},
    {PYPTO_DEVICE_ERROR_RESOURCE_ALLOC_FAIL, "resource alloc fail"},
    {PYPTO_DEVICE_ERROR_NO_PERMISSION, "no permission"},
    {PYPTO_DEVICE_ERROR_NO_EVENT_RESOURCE, "no event resource"},
    {PYPTO_DEVICE_ERROR_NO_STREAM_RESOURCE, "no stream resource"},
    {PYPTO_DEVICE_ERROR_NO_NOTIFY_RESOURCE, "no notify resource"},
    {PYPTO_DEVICE_ERROR_NO_MODEL_RESOURCE, "no model resource"},
    {PYPTO_DEVICE_ERROR_NO_CDQ_RESOURCE, "no cdq resource"},
    {PYPTO_DEVICE_ERROR_OVER_LIMIT, "over limit"},
    {PYPTO_DEVICE_ERROR_QUEUE_EMPTY, "queue is empty"},
    {PYPTO_DEVICE_ERROR_QUEUE_FULL, "queue is full"},
    {PYPTO_DEVICE_ERROR_REPEATED_INIT, "repeated init"},
    {PYPTO_DEVICE_ERROR_AIVEC_OVER_FLOW, "aivec over flow"},
    {PYPTO_DEVICE_ERROR_OVER_FLOW, "common over flow"},
    {PYPTO_DEVICE_ERROR_DEVICE_OOM, "device oom"},
    {PYPTO_DEVICE_ERROR_FEATURE_NOT_SUPPORT_UPDATE_OP, "not support to update this op"},
    {PYPTO_DEVICE_ERROR_TIMEOUT, "driver timeout"},
    // 507xxx
    {PYPTO_DEVICE_ERROR_INTERNAL_ERROR, "runtime internal error"},
    {PYPTO_DEVICE_ERROR_TS_ERROR, "ts internal error"},
    {PYPTO_DEVICE_ERROR_STREAM_TASK_FULL, "task full in stream"},
    {PYPTO_DEVICE_ERROR_STREAM_TASK_EMPTY, "task empty in stream"},
    {PYPTO_DEVICE_ERROR_STREAM_NOT_COMPLETE, "stream not complete"},
    {PYPTO_DEVICE_ERROR_END_OF_SEQUENCE, "end of sequence"},
    {PYPTO_DEVICE_ERROR_EVENT_NOT_COMPLETE, "event not complete"},
    {PYPTO_DEVICE_ERROR_CONTEXT_RELEASE_ERROR, "context release error"},
    {PYPTO_DEVICE_ERROR_SOC_VERSION, "soc version error"},
    {PYPTO_DEVICE_ERROR_TASK_TYPE_NOT_SUPPORT, "task type not support"},
    {PYPTO_DEVICE_ERROR_LOST_HEARTBEAT, "ts lost heartbeat"},
    {PYPTO_DEVICE_ERROR_MODEL_EXECUTE, "model execute failed"},
    {PYPTO_DEVICE_ERROR_REPORT_TIMEOUT, "report timeout"},
    {PYPTO_DEVICE_ERROR_SYS_DMA, "sys dma error"},
    {PYPTO_DEVICE_ERROR_AICORE_TIMEOUT, "aicore timeout"},
    {PYPTO_DEVICE_ERROR_AICORE_EXCEPTION, "aicore exception"},
    {PYPTO_DEVICE_ERROR_AICORE_TRAP_EXCEPTION, "aicore trap exception"},
    {PYPTO_DEVICE_ERROR_AICPU_TIMEOUT, "aicpu timeout"},
    {PYPTO_DEVICE_ERROR_AICPU_EXCEPTION, "aicpu exception"},
    {PYPTO_DEVICE_ERROR_AICPU_DATADUMP_RSP_ERR, "aicpu datadump response error"},
    {PYPTO_DEVICE_ERROR_AICPU_MODEL_RSP_ERR, "aicpu model operate response error"},
    {PYPTO_DEVICE_ERROR_PROFILING_ERROR, "profiling error"},
    {PYPTO_DEVICE_ERROR_IPC_ERROR, "ipc error"},
    {PYPTO_DEVICE_ERROR_MODEL_ABORT_NORMAL, "model abort normal"},
    {PYPTO_DEVICE_ERROR_KERNEL_UNREGISTERING, "kernel unregistering"},
    {PYPTO_DEVICE_ERROR_RINGBUFFER_NOT_INIT, "ringbuffer not init"},
    {PYPTO_DEVICE_ERROR_RINGBUFFER_NO_DATA, "ringbuffer no data"},
    {PYPTO_DEVICE_ERROR_KERNEL_LOOKUP, "kernel lookup error"},
    {PYPTO_DEVICE_ERROR_KERNEL_DUPLICATE, "kernel register duplicate"},
    {PYPTO_DEVICE_ERROR_DEBUG_REGISTER_FAIL, "debug register failed"},
    {PYPTO_DEVICE_ERROR_DEBUG_UNREGISTER_FAIL, "debug unregister failed"},
    {PYPTO_DEVICE_ERROR_LABEL_CONTEXT, "label not in current context"},
    {PYPTO_DEVICE_ERROR_PROGRAM_USE_OUT, "program register num use out"},
    {PYPTO_DEVICE_ERROR_DEV_SETUP_ERROR, "device setup error"},
    {PYPTO_DEVICE_ERROR_VECTOR_CORE_TIMEOUT, "vector core timeout"},
    {PYPTO_DEVICE_ERROR_VECTOR_CORE_EXCEPTION, "vector core exception"},
    {PYPTO_DEVICE_ERROR_VECTOR_CORE_TRAP_EXCEPTION, "vector core trap exception"},
    {PYPTO_DEVICE_ERROR_CDQ_BATCH_ABNORMAL, "cdq alloc batch abnormal"},
    {PYPTO_DEVICE_ERROR_DIE_MODE_CHANGE_ERROR, "can not change die mode"},
    {PYPTO_DEVICE_ERROR_DIE_SET_ERROR, "single die mode can not set die"},
    {PYPTO_DEVICE_ERROR_INVALID_DIEID, "invalid die id"},
    {PYPTO_DEVICE_ERROR_DIE_MODE_NOT_SET, "die mode not set"},
    {PYPTO_DEVICE_ERROR_AICORE_TRAP_READ_OVERFLOW, "aic trap read overflow"},
    {PYPTO_DEVICE_ERROR_AICORE_TRAP_WRITE_OVERFLOW, "aic trap write overflow"},
    {PYPTO_DEVICE_ERROR_VECTOR_CORE_TRAP_READ_OVERFLOW, "aiv trap read overflow"},
    {PYPTO_DEVICE_ERROR_VECTOR_CORE_TRAP_WRITE_OVERFLOW, "aiv trap write overflow"},
    {PYPTO_DEVICE_ERROR_STREAM_SYNC_TIMEOUT, "stream sync time out"},
    {PYPTO_DEVICE_ERROR_EVENT_SYNC_TIMEOUT, "event sync time out"},
    {PYPTO_DEVICE_ERROR_FFTS_PLUS_TIMEOUT, "ffts+ timeout"},
    {PYPTO_DEVICE_ERROR_FFTS_PLUS_EXCEPTION, "ffts+ exception"},
    {PYPTO_DEVICE_ERROR_FFTS_PLUS_TRAP_EXCEPTION, "ffts+ trap exception"},
    {PYPTO_DEVICE_ERROR_SEND_MSG, "hdc send msg fail"},
    {PYPTO_DEVICE_ERROR_COPY_DATA, "copy data fail"},
    {PYPTO_DEVICE_ERROR_DEVICE_MEM_ERROR, "device MEM ERROR"},
    {PYPTO_DEVICE_ERROR_HBM_MULTI_BIT_ECC_ERROR, "hbm Multi-bit ECC error"},
    {PYPTO_DEVICE_ERROR_SUSPECT_DEVICE_MEM_ERROR, "suspect device MEM ERROR"},
    {PYPTO_DEVICE_ERROR_LINK_ERROR, "link ERROR"},
    {PYPTO_DEVICE_ERROR_SUSPECT_REMOTE_ERROR, "suspect remote ERROR"},
    {PYPTO_DEVICE_ERROR_DRV_INTERNAL_ERROR, "drv internal error"},
    {PYPTO_DEVICE_ERROR_AICPU_INTERNAL_ERROR, "aicpu internal error"},
    {PYPTO_DEVICE_ERROR_SOCKET_CLOSE, "hdc disconnect"},
    {PYPTO_DEVICE_ERROR_AICPU_INFO_LOAD_RSP_ERR, "aicpu info load response error"},
    {PYPTO_DEVICE_ERROR_STREAM_CAPTURE_INVALIDATED, "capture status is invalidated"},
    {PYPTO_DEVICE_ERROR_COMM_OP_RETRY_FAIL, "hccl operation retry failed"},
};

const char* GetRetcodeMessage(int32_t retcode)
{
    for (const auto& entry : kErrorCodeTable) {
        if (entry.retcode == retcode) {
            return entry.msg;
        }
    }
    return "unknown error";
}

void PyPTOExceptionInfoCallBack(AclRtExceptionInfo* exceptionInfo)
{
    const char* errMsg = GetRetcodeMessage(static_cast<int32_t>(exceptionInfo->retcode));
    const char* kernelName = "(Null)";
    if (exceptionInfo->expandInfo.type == RtExceptionExpandType::AICORE) {
        kernelName = exceptionInfo->expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName;
    }
    printf("[Error]: %s, device_id: %u, stream_id: %u, task_id: %u, retcode: %u, kernelName: %s\n", errMsg,
           exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid, exceptionInfo->retcode, kernelName);
    printf("        Rectify the fault based on the error information in the ascend log.\n");
    printf("PyPTO error: PyPTO Internal Error. Please rectify the fault based on the error information "
           "in the ascend log. (function PyPTOExceptionInfoCallBack)\n");
}

void InitializeErrorCallback()
{
    AclError ret = AclRtSetExceptionInfoCallback(&PyPTOExceptionInfoCallBack);
    if (ret != ACLRT_SUCCESS) {
        printf("Failed to set exception callback: %d\n", ret);
    }
}
} // namespace npu::tile_fwk
