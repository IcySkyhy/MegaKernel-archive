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
 * \file device_error_code.h
 * \brief
 */

#pragma once

namespace npu::tile_fwk {

// =============================================================================
// Runtime Device Error (ACL API return codes)
// =============================================================================
#define PYPTO_DEVICE_SUCCESS 0                                // success
#define PYPTO_DEVICE_ERROR_PARAM_INVALID 107000               // param invalid
#define PYPTO_DEVICE_ERROR_INVALID_DEVICEID 107001            // invalid device id
#define PYPTO_DEVICE_ERROR_CONTEXT_NULL 107002                // current context null
#define PYPTO_DEVICE_ERROR_STREAM_CONTEXT 107003              // stream not in current context
#define PYPTO_DEVICE_ERROR_MODEL_CONTEXT 107004               // model not in current context
#define PYPTO_DEVICE_ERROR_STREAM_MODEL 107005                // stream not in model
#define PYPTO_DEVICE_ERROR_EVENT_TIMESTAMP_INVALID 107006     // event timestamp invalid
#define PYPTO_DEVICE_ERROR_EVENT_TIMESTAMP_REVERSAL 107007    // event timestamp reversal
#define PYPTO_DEVICE_ERROR_ADDR_UNALIGNED 107008              // memory address unaligned
#define PYPTO_DEVICE_ERROR_FILE_OPEN 107009                   // open file failed
#define PYPTO_DEVICE_ERROR_FILE_WRITE 107010                  // write file failed
#define PYPTO_DEVICE_ERROR_STREAM_SUBSCRIBE 107011            // error subscribe stream
#define PYPTO_DEVICE_ERROR_THREAD_SUBSCRIBE 107012            // error subscribe thread
#define PYPTO_DEVICE_ERROR_GROUP_NOT_SET 107013               // group not set
#define PYPTO_DEVICE_ERROR_GROUP_NOT_CREATE 107014            // group not create
#define PYPTO_DEVICE_ERROR_STREAM_NO_CB_REG 107015            // callback not register to stream
#define PYPTO_DEVICE_ERROR_INVALID_MEMORY_TYPE 107016         // invalid memory type
#define PYPTO_DEVICE_ERROR_INVALID_HANDLE 107017              // invalid handle
#define PYPTO_DEVICE_ERROR_INVALID_MALLOC_TYPE 107018         // invalid malloc type
#define PYPTO_DEVICE_ERROR_WAIT_TIMEOUT 107019                // wait timeout
#define PYPTO_DEVICE_ERROR_TASK_TIMEOUT 107020                // task timeout
#define PYPTO_DEVICE_ERROR_SYSPARAMOPT_NOT_SET 107021         // not set sysparamopt
#define PYPTO_DEVICE_ERROR_DEVICE_TASK_ABORT 107022           // device task aborting
#define PYPTO_DEVICE_ERROR_STREAM_ABORT 107023                // stream aborting
#define PYPTO_DEVICE_ERROR_CAPTURE_DEPENDENCY 107024          // capture dependency failure
#define PYPTO_DEVICE_ERROR_STREAM_UNJOINED 107025             // invalid capture model
#define PYPTO_DEVICE_ERROR_MODEL_CAPTURED 107026              // model is captured
#define PYPTO_DEVICE_ERROR_STREAM_CAPTURED 107027             // stream is captured
#define PYPTO_DEVICE_ERROR_EVENT_CAPTURED 107028              // event is captured
#define PYPTO_DEVICE_ERROR_STREAM_NOT_CAPTURED 107029         // stream is not in capture status
#define PYPTO_DEVICE_ERROR_CAPTURE_MODE_NOT_SUPPORT 107030    // stream is captured, not support current oper
#define PYPTO_DEVICE_ERROR_STREAM_CAPTURE_IMPLICIT 107031     // a disallowed implicit dependency from default stream
#define PYPTO_DEVICE_ERROR_STREAM_CAPTURE_CONFLICT 107032     // interdependent stream cannot begin capture together
#define PYPTO_DEVICE_ERROR_STREAM_TASK_GROUP_STATUS 107033    // task group status error
#define PYPTO_DEVICE_ERROR_STREAM_TASK_GROUP_INTR 107034      // task group interrupted
#define PYPTO_DEVICE_ERROR_TASK_ABORT_STOP 107035             // device task aborting stop before post process
#define PYPTO_DEVICE_ERROR_STREAM_CAPTURE_UNMATCHED 107036    // the capture was not initiated in this stream
#define PYPTO_DEVICE_ERROR_MODEL_RUNNING 107037               // the model is still running
#define PYPTO_DEVICE_ERROR_STREAM_CAPTURE_WRONG_THREAD 107038 // the thread of end capture and begin capture is not same
#define PYPTO_DEVICE_ERROR_INSUFFICIENT_INPUT_ARRAY 107039    // input array capacity insufficient
#define PYPTO_DEVICE_ERROR_MODEL_UPDATE_FAILED 107040         // the model update failed
#define PYPTO_DEVICE_ERROR_CAPTURE_MODE_BLOCK_ASYNC \
    107041 // async oper convert to sync oper, stream is captured, not support current oper
#define PYPTO_DEVICE_ERROR_SYMBOL_NOT_FOUND 107042 // symbol not found

#define PYPTO_DEVICE_ERROR_FEATURE_NOT_SUPPORT 207000           // feature not support
#define PYPTO_DEVICE_ERROR_MEMORY_ALLOCATION 207001             // memory allocation error, only used by out of memory
#define PYPTO_DEVICE_ERROR_MEMORY_FREE 207002                   // memory free error
#define PYPTO_DEVICE_ERROR_AICORE_OVER_FLOW 207003              // aicore over flow
#define PYPTO_DEVICE_ERROR_NO_DEVICE 207004                     // no device
#define PYPTO_DEVICE_ERROR_RESOURCE_ALLOC_FAIL 207005           // resource alloc fail
#define PYPTO_DEVICE_ERROR_NO_PERMISSION 207006                 // no permission
#define PYPTO_DEVICE_ERROR_NO_EVENT_RESOURCE 207007             // no event resource
#define PYPTO_DEVICE_ERROR_NO_STREAM_RESOURCE 207008            // no stream resource
#define PYPTO_DEVICE_ERROR_NO_NOTIFY_RESOURCE 207009            // no notify resource
#define PYPTO_DEVICE_ERROR_NO_MODEL_RESOURCE 207010             // no model resource
#define PYPTO_DEVICE_ERROR_NO_CDQ_RESOURCE 207011               // no cdq resource
#define PYPTO_DEVICE_ERROR_OVER_LIMIT 207012                    // over limit
#define PYPTO_DEVICE_ERROR_QUEUE_EMPTY 207013                   // queue is empty
#define PYPTO_DEVICE_ERROR_QUEUE_FULL 207014                    // queue is full
#define PYPTO_DEVICE_ERROR_REPEATED_INIT 207015                 // repeated init
#define PYPTO_DEVICE_ERROR_AIVEC_OVER_FLOW 207016               // aivec over flow
#define PYPTO_DEVICE_ERROR_OVER_FLOW 207017                     // common over flow
#define PYPTO_DEVICE_ERROR_DEVICE_OOM 207018                    // device oom
#define PYPTO_DEVICE_ERROR_FEATURE_NOT_SUPPORT_UPDATE_OP 207019 // not support to update this op
#define PYPTO_DEVICE_ERROR_TIMEOUT 207020                       // driver timeout

#define PYPTO_DEVICE_ERROR_INTERNAL_ERROR 507000                  // runtime internal error
#define PYPTO_DEVICE_ERROR_TS_ERROR 507001                        // ts internel error
#define PYPTO_DEVICE_ERROR_STREAM_TASK_FULL 507002                // task full in stream
#define PYPTO_DEVICE_ERROR_STREAM_TASK_EMPTY 507003               // task empty in stream
#define PYPTO_DEVICE_ERROR_STREAM_NOT_COMPLETE 507004             // stream not complete
#define PYPTO_DEVICE_ERROR_END_OF_SEQUENCE 507005                 // end of sequence
#define PYPTO_DEVICE_ERROR_EVENT_NOT_COMPLETE 507006              // event not complete
#define PYPTO_DEVICE_ERROR_CONTEXT_RELEASE_ERROR 507007           // context release error
#define PYPTO_DEVICE_ERROR_SOC_VERSION 507008                     // soc version error
#define PYPTO_DEVICE_ERROR_TASK_TYPE_NOT_SUPPORT 507009           // task type not support
#define PYPTO_DEVICE_ERROR_LOST_HEARTBEAT 507010                  // ts lost heartbeat
#define PYPTO_DEVICE_ERROR_MODEL_EXECUTE 507011                   // model execute failed
#define PYPTO_DEVICE_ERROR_REPORT_TIMEOUT 507012                  // report timeout
#define PYPTO_DEVICE_ERROR_SYS_DMA 507013                         // sys dma error
#define PYPTO_DEVICE_ERROR_AICORE_TIMEOUT 507014                  // aicore timeout
#define PYPTO_DEVICE_ERROR_AICORE_EXCEPTION 507015                // aicore exception
#define PYPTO_DEVICE_ERROR_AICORE_TRAP_EXCEPTION 507016           // aicore trap exception
#define PYPTO_DEVICE_ERROR_AICPU_TIMEOUT 507017                   // aicpu timeout
#define PYPTO_DEVICE_ERROR_AICPU_EXCEPTION 507018                 // aicpu exception
#define PYPTO_DEVICE_ERROR_AICPU_DATADUMP_RSP_ERR 507019          // aicpu datadump response error
#define PYPTO_DEVICE_ERROR_AICPU_MODEL_RSP_ERR 507020             // aicpu model operate response error
#define PYPTO_DEVICE_ERROR_PROFILING_ERROR 507021                 // profiling error
#define PYPTO_DEVICE_ERROR_IPC_ERROR 507022                       // ipc error
#define PYPTO_DEVICE_ERROR_MODEL_ABORT_NORMAL 507023              // model abort normal
#define PYPTO_DEVICE_ERROR_KERNEL_UNREGISTERING 507024            // kernel unregistering
#define PYPTO_DEVICE_ERROR_RINGBUFFER_NOT_INIT 507025             // ringbuffer not init
#define PYPTO_DEVICE_ERROR_RINGBUFFER_NO_DATA 507026              // ringbuffer no data
#define PYPTO_DEVICE_ERROR_KERNEL_LOOKUP 507027                   // kernel lookup error
#define PYPTO_DEVICE_ERROR_KERNEL_DUPLICATE 507028                // kernel register duplicate
#define PYPTO_DEVICE_ERROR_DEBUG_REGISTER_FAIL 507029             // debug register failed
#define PYPTO_DEVICE_ERROR_DEBUG_UNREGISTER_FAIL 507030           // debug unregister failed
#define PYPTO_DEVICE_ERROR_LABEL_CONTEXT 507031                   // label not in current context
#define PYPTO_DEVICE_ERROR_PROGRAM_USE_OUT 507032                 // program register num use out
#define PYPTO_DEVICE_ERROR_DEV_SETUP_ERROR 507033                 // device setup error
#define PYPTO_DEVICE_ERROR_VECTOR_CORE_TIMEOUT 507034             // vector core timeout
#define PYPTO_DEVICE_ERROR_VECTOR_CORE_EXCEPTION 507035           // vector core exception
#define PYPTO_DEVICE_ERROR_VECTOR_CORE_TRAP_EXCEPTION 507036      // vector core trap exception
#define PYPTO_DEVICE_ERROR_CDQ_BATCH_ABNORMAL 507037              // cdq alloc batch abnormal
#define PYPTO_DEVICE_ERROR_DIE_MODE_CHANGE_ERROR 507038           // can not change die mode
#define PYPTO_DEVICE_ERROR_DIE_SET_ERROR 507039                   // single die mode can not set die
#define PYPTO_DEVICE_ERROR_INVALID_DIEID 507040                   // invalid die id
#define PYPTO_DEVICE_ERROR_DIE_MODE_NOT_SET 507041                // die mode not set
#define PYPTO_DEVICE_ERROR_AICORE_TRAP_READ_OVERFLOW 507042       // aic trap read overflow
#define PYPTO_DEVICE_ERROR_AICORE_TRAP_WRITE_OVERFLOW 507043      // aic trap write overflow
#define PYPTO_DEVICE_ERROR_VECTOR_CORE_TRAP_READ_OVERFLOW 507044  // aiv trap read overflow
#define PYPTO_DEVICE_ERROR_VECTOR_CORE_TRAP_WRITE_OVERFLOW 507045 // aiv trap write overflow
#define PYPTO_DEVICE_ERROR_STREAM_SYNC_TIMEOUT 507046             // stream sync time out
#define PYPTO_DEVICE_ERROR_EVENT_SYNC_TIMEOUT 507047              // event sync time out
#define PYPTO_DEVICE_ERROR_FFTS_PLUS_TIMEOUT 507048               // ffts+ timeout
#define PYPTO_DEVICE_ERROR_FFTS_PLUS_EXCEPTION 507049             // ffts+ exception
#define PYPTO_DEVICE_ERROR_FFTS_PLUS_TRAP_EXCEPTION 507050        // ffts+ trap exception
#define PYPTO_DEVICE_ERROR_SEND_MSG 507051                        // hdc send msg fail
#define PYPTO_DEVICE_ERROR_COPY_DATA 507052                       // copy data fail
#define PYPTO_DEVICE_ERROR_DEVICE_MEM_ERROR 507053                // device MEM ERROR
#define PYPTO_DEVICE_ERROR_HBM_MULTI_BIT_ECC_ERROR 507054         // hbm Multi-bit ECC error
#define PYPTO_DEVICE_ERROR_SUSPECT_DEVICE_MEM_ERROR 507055        // suspect device MEM ERROR
#define PYPTO_DEVICE_ERROR_LINK_ERROR 507056                      // link ERROR
#define PYPTO_DEVICE_ERROR_SUSPECT_REMOTE_ERROR 507057            // suspect remote ERROR
#define PYPTO_DEVICE_ERROR_DRV_INTERNAL_ERROR 507899              // drv internal error
#define PYPTO_DEVICE_ERROR_AICPU_INTERNAL_ERROR 507900            // aicpu internal error
#define PYPTO_DEVICE_ERROR_SOCKET_CLOSE 507901                    // hdc disconnect
#define PYPTO_DEVICE_ERROR_AICPU_INFO_LOAD_RSP_ERR 507902         // aicpu info load response error
#define PYPTO_DEVICE_ERROR_STREAM_CAPTURE_INVALIDATED 507903      // capture status is invalidated
#define PYPTO_DEVICE_ERROR_COMM_OP_RETRY_FAIL 507904              // hccl operation retry failed

} // namespace npu::tile_fwk
