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
 * \file error_code.h
 * \brief Centralized error code definitions for all components.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace npu::tile_fwk {
template <typename T, typename = std::enable_if_t<std::is_enum_v<T>>>
inline constexpr std::underlying_type_t<T> ToUnderlying(T value)
{
    return static_cast<std::underlying_type_t<T>>(value);
}

// =============================================================================
// F0XXXX: external limitation
// =============================================================================
enum class ExternalError : uint32_t {
    COMMON_EXTERNAL_ERROR = 0x0FFFFU,
    INVALID_TYPE = 0x00001U,
    INVALID_VAL = 0x00002U,
    RUNTIME_ERROR = 0x00003U,
    NAME_ERROR = 0x00004U,
    NOT_IMPLEMENTED_ERROR = 0x00005U,
    KEY_ERROR = 0x00006U,
    INVALID_OPERATION = 0x00007U,
    OUT_OF_RANGE = 0x00008U,
    BAD_FD = 0x00009U,
    DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED = 0x0000AU,
    INVALID_SHAPE = 0x0000BU,
    INVALID_TILE = 0x0000CU,
    INVALID_FORMAT = 0x0000DU,
    UNKNOWN = 0x0FFFFU
};

// =============================================================================
// F1XXXX-: internal limitation
// =============================================================================
enum class InternalError : uint32_t {
    COMMON_INNER_ERROR = 0x1FFFFU,
    FE_INNER_ERROR = 0x2FFFFU,
    PASS_INNER_ERROR = 0x4FFFFU,
    CODEGEN_INNER_ERROR = 0x6FFFFU,
    MACHINE_INNER_ERROR = 0x7FFFFU,
    SIM_INNER_ERROR = 0x9FFFFU,
};

// =============================================================================
// F2-F3XXXX: FUNCTION
// =============================================================================
enum class FeError : uint32_t {
    INVALID_OPERATION = 0x21001U,
    INVALID_TYPE = 0x21002U,
    INVALID_VAL = 0x21003U,
    INVALID_PTR = 0x21004U,
    OUT_OF_RANGE = 0x21005U,
    IS_EXIST = 0x21006U,
    NOT_EXIST = 0x21007U,
    DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED = 0x21008U,
    OP_DEPENDENCY_CYCLE = 0x21009U,
    BAD_FD = 0x29001U,
    INVALID_FILE = 0x29002U,
    UNKNOWN = 0x3FFFFU
};

// =============================================================================
// F4-F5XXXX: PASS
// =============================================================================
enum class TensorErr : uint32_t {
    TENSOR_NULL_POINTER = 0x40000U,
    TENSOR_INVALID_MEMORY_TYPE = 0x40001U,
    TENSOR_SUBGRAPH_BOUNDARY = 0x40002U,
    TENSOR_SHAPE_MISMATCH = 0x40003U,
    TENSOR_UNSUPPORTED_DATATYPE = 0x40004U,
    TENSOR_MEMORY_ALLOCATION = 0x40005U,
    TENSOR_DYNAMIC_ATTR = 0x40006U
};

enum class OperationErr : uint32_t {
    OP_INVALID_OPERAND_COUNT = 0x41000U,
    OP_NULL_POINTER = 0x41001U,
    OP_INVALID_OPCODE = 0x41002U,
    OP_PRODUCER_CONSUMER = 0x41003U,
    OP_SPECIAL_CONSTRAINT = 0x41004U,
    OP_NESTING_DEPTH = 0x41005U,
    OP_SEQUENCE_ERROR = 0x41006U,
    OP_SCOPE_ERROR = 0x41007U
};

enum class FunctionErr : uint32_t {
    FUNCTION_GRAPH_STRUCTURE = 0x42000U,
    FUNCTION_BOUNDARY_COMPLETENESS = 0x42001U,
    FUNCTION_GRAPH_CONNECTION = 0x42002U,
    FUNCTION_EXPAND_FEATURE = 0x42003U,
    FUNCTION_MEMORY_REACHABILITY = 0x42004U,
    FUNCTION_UNIQUENESS = 0x42005U,
    FUNCTION_SPECIAL_STRUCTURE = 0x42006U
};

enum class GraphErr : uint32_t {
    GRAPH_LOOP_DETECTION = 0x43000U,
    GRAPH_TOPOLOGY_STRUCTURE = 0x43001U,
    GRAPH_SUBGRAPH_EMPTY = 0x43002U,
    GRAPH_SUBGRAPH_ID_INVALID = 0x43003U,
    GRAPH_EDGE_CONSISTENCY = 0x43004U,
    GRAPH_COLOR_CONSISTENCY = 0x43005U,
    GRAPH_READY_STATE = 0x43006U,
    GRAPH_AIV_AIC_MIX = 0x43007U
};

enum class ConfigErr : uint32_t {
    CONFIG_MEMORY_TYPE_REACHABLE = 0x44000U,
    CONFIG_SUBGRAPH_BOUNDARY = 0x44001U,
    CONFIG_TENSOR_MEMORY_TYPE = 0x44002U,
    CONFIG_FILE_FAILED = 0x44003U
};

enum class ManagerErr : uint32_t {};

// =============================================================================
// F6XXXX: CODEGEN
// =============================================================================
enum class CodeGenErrorCategory {
    FRAMEWORK = 0x60000U,
    OPERATION_ADAPTER = 0x61000U,
    GEN_OP_CODE = 0x62000U,
    COMPILE_CODE = 0x63000U,
};

enum class FwkErr : uint32_t {
    PLATFORM_NOT_SUPPORTED = static_cast<uint32_t>(CodeGenErrorCategory::FRAMEWORK) + 1U,
    INVALID_FUNCTION,
    TASK_RUN_FAILED
};

enum class OperErr : uint32_t {
    ATTRIBUTE_INVALID = static_cast<uint32_t>(CodeGenErrorCategory::OPERATION_ADAPTER) + 1U,
    TENSOR_DIM_EXCEEDED,
    OPERAND_COUNT_EXCEEDED,
    OPERAND_COUNT_NOT_MATCHED,
    OPERATION_INIT_FAILED,
    OPERAND_TYPE_UNSUPPORTED,
};

enum class GenCodeErr : uint32_t {
    GEN_OP_CODE_FAILED = static_cast<uint32_t>(CodeGenErrorCategory::GEN_OP_CODE) + 1U,
    OP_CODE_UNSUPPORTED,
    PRINT_FAILED,
    PRINT_MODE_ERROR,
    DATA_TYPE_MISMATCHED,
    DATA_TYPE_UNSUPPORTED,
    TENSOR_SHAPE_INVALID,
    TENSOR_SHAPE_MISMATCHED,
    TENSOR_DIM_UNSUPPORTED,
    TENSOR_OFFSET_INVALID,
    TENSOR_MAGIC_CONFLICT,
    PARAM_IDX_INVALID,
    TENSOR_NOT_FOUND,
    SYMBOL_NOT_FOUND,
    PIPE_ID_NOT_FOUND,
    SYMBOL_ID_INVALID,
};

enum class CmpCodeErr : uint32_t {
    COMPILE_CODE_FAILED = static_cast<uint32_t>(CodeGenErrorCategory::COMPILE_CODE) + 1U,
    INCLUDE_FILE_NOT_FOUND,
    PTO_ISA_NOT_FOUND,
    CMD_CHECK_FAILED,
    FILE_IO_FAILED,
};

// =============================================================================
// F7-F8XXXX: MACHINE
// =============================================================================
enum class MachineError : uint32_t {
    HOST_BACKEND = 0x70000U,
    HOST_LAUNCHER = 0x71000U,
    SCHEDULE = 0x72000U,
    CONTROL_FLOW = 0x73000U,
    WORKSPACE = 0x74000U,
    DUMP_DFX = 0x75000U,
    PROGRAM_ENCODE = 0x76000U,
    TENSOR_META = 0x77000U,
    SERVER_KERNEL = 0x78000U,
    THREAD_MACHINE = 0x79000U,
    DEV_DATA = 0x7A000U,
    DEV_COMMON = 0x7A000U,
    RUNTIME_ERROR = 0x7B000U,
    UNKNOWN = 0x7F000U,
};

enum class DevDataErr : uint32_t {
    DEV_RELOC_VECTOR_INDEX_OOB = static_cast<uint32_t>(MachineError::DEV_DATA) + 0x01U,
    SMALL_ARRAY_RESIZE_OOB,
    VECTOR_UNINITIALIZED,
    VECTOR_INDEX_OUT_OF_RANGE,
    VECTOR_EMPTY_ACCESS,
    ITEM_POOL_UNINITIALIZED,
    ITEM_POOL_FREE_LIST_INVALID,
    ITEM_POOL_INDEX_OUT_OF_RANGE,
    SHEET_COLUMN_MISMATCH,
    SHEET_COLUMN_INDEX_OUT_OF_RANGE,
};

enum class DevCommonErr : uint32_t {
    MEMCPY_FAILED = static_cast<uint32_t>(MachineError::DEV_COMMON) + 0x01U,
    ALLOC_FAILED,
    MALLOC_FAILED,
    MEMRESET_FAILED,
    NULLPTR,
    PARAM_INVALID,
    PARAM_CHECK_FAILED,
    FILE_ERROR,
    SYSTEM_CALL_FAILED,
    GET_ENV_FAILED,
    GET_HANDLE_FAILED,
    FREE_FAILED,
    LOAD_LIBRARY_FAILED,
    INIT_FAILED,
    EXEC_THRID_API_FAILD,
    CANN_API_NOT_FOUND,
};

enum class HostBackEndErr : uint32_t {
    COMPILE_AICORE_FAILED = static_cast<uint32_t>(MachineError::HOST_BACKEND) + 0x01U,
    COMPILE_CCEC_FAILED,
    LINK_FAILED,
    GEN_AICORE_FILE_FAILED,
    GEN_DYNAMIC_OP_FAILED,
    PRECOMPILE_FAILED,
    FUNCTION_CACHE_HASH_MISS,
    DUPLICATE_LEAF_FUNC_HASH,
    RUN_PASS_FAILED,
};

enum class HostLauncherErr : uint32_t {
    LAUNCH_AICPU_FAILED = static_cast<uint32_t>(MachineError::HOST_LAUNCHER) + 0x01U,
    LAUNCH_PREPARE_FAILED,
    LAUNCH_CUSTOM_AICPU_FAILED,
    LAUNCH_AICORE_FAILED,
    LAUNCH_BUILTIN_OP_NULL_FAILED,
    REGISTER_KERNEL_FAILED,
    PREPARE_ARGS_FAILED,
    MAP_REG_ADDR_FAILED,
    MEM_POOL_CHECK_ALL_SENTINELS_FAILED,
    TRIPLE_STREAM_ERROR,
    SYNC_FAILED,
};

enum class SchedErr : uint32_t {
    TASK_WAIT_TIMEOUT = static_cast<uint32_t>(MachineError::SCHEDULE) + 0x01U,
    HANDSHAKE_TIMEOUT,
    READY_QUEUE_OVERFLOW,
    CORE_TASK_EXEC_FAILED,
    CORE_TASK_PROCESS_FAILED,
    RINGBUFFER_WAIT_TIMEOUT,
    ABNOMAL_LAST_WORD,
    SCH_DEVTASK_CTX_FULL,
    FSM_STATUS_ERROR,
    SCH_PARALLEL_DEVTASK_TIMEOUT,
    CORE_INFO_INVALID,
    WAIT_CTRL_TIMEOUT
};

enum class CtrlErr : uint32_t {
    CTRL_FLOW_EXEC_FAILED = static_cast<uint32_t>(MachineError::CONTROL_FLOW) + 0x01U,
    ROOT_ALLOC_CTX_NULL,
    ROOT_STITCH_CTX_NULL,
    DEVICE_TASK_BUILD_FAILED,
    TASK_STATS_ABNORMAL,
    CTRL_INIT_FAILED,
    CTRL_SIM_FAILED,
    CTRL_ALLOC_TIMEOUT,
    CELL_MATCH_FILL_OP_NOT_ENOUGH,
    CELL_MATCH_OP_TYPE_NOT_SUPPORTED,
    CELL_MATCH_OP_ID_INVALID,
};

enum class WsErr : uint32_t {
    SLAB_ADD_CACHE_FAILED = static_cast<uint32_t>(MachineError::WORKSPACE) + 0x01U,
    SLAB_STAGE_LIST_INCONSISTENT,
    SLAB_TYPE_INVALID,
    SLAB_MAGIC_CORRUPTED,
    WORKSPACE_INIT_RESOURCE_ERROR,
    WORKSPACE_INIT_PARAM_INVALID,
    WS_TENSOR_ADDRESS_OUT_OF_RANGE,
    WORKSPACE_ITER_INVALID,
    WORKSPACE_REFCOUNT_INVALID,
    WORKSPACE_ALLOCATOR_REGIST_FAILED,
    WORKSPACE_CATEGORY_INVALID,
    WORKSPACE_CAPACITY_INSUFFICIENT,
    WORKSPACE_BASE_ADDR_OUT_OF_RANGE,
};

enum class ProgEncodeErr : uint32_t {
    DYNFUNC_DATA_ALIGNMENT_ERROR = static_cast<uint32_t>(MachineError::PROGRAM_ENCODE) + 0x01U,
    FUNC_OP_SIZE_MISMATCH,
    STITCH_PRED_SUCC_MISMATCH,
    STITCH_LIST_TOO_LARGE,
    STITCH_HANDLE_INDEX_OUT_OF_RANGE,
    CELL_MATCH_PARAM_INVALID,
    RANGE_VERIFY_FAILED,
    CACHE_RELOC_KIND_INVALID,
    ADDR_OFFSET_RAW_MAGIC_MISMATCH,
    CALL_OP_COUNT_EXCEEDS_UINT16_MAX,
    CELL_MATCH_DIM_ZERO,
    ASSEMBLE_STITCH_MEMORY_EXCESS,
    LEAF_CALLEE_ATTR_NULL,
    CELL_MATCH_LAUNCH_PREPARE_FAILED,
};

enum class TensorMetaErr : uint32_t {
    TENSOR_DIM_COUNT_EXCEEDED = static_cast<uint32_t>(MachineError::TENSOR_META) + 0x01U,
    TENSOR_ENCODE_PTR_MISMATCH,
    RAW_TENSOR_INDEX_OUT_OF_RANGE,
    SHAPE_VALUE_MISMATCH,
    INCAST_ADDRESS_NULL,
    OUTCAST_ADDRESS_NULL,
    RUNTIME_WORKSPACE_NULL,
};

enum class ServerKernelErr : uint32_t {
    KERNEL_EXEC_FAILED = static_cast<uint32_t>(MachineError::SERVER_KERNEL) + 0x01U,
};

enum class ThreadErr : uint32_t {
    SIGNAL_HANDLER_ABNORMAL = static_cast<uint32_t>(MachineError::THREAD_MACHINE) + 0x01U,
    THREAD_CPU_ALLOC_FAILED = static_cast<uint32_t>(MachineError::THREAD_MACHINE),
    THREAD_CPU_WAIT_FINISH_TIMEOUT = static_cast<uint32_t>(MachineError::THREAD_MACHINE),
};

enum class RtErr : uint32_t {
    RT_INIT_FAILED = static_cast<uint32_t>(MachineError::RUNTIME_ERROR) + 0x01U,
    RT_MEMCPY_FAILED,
    RT_MEMSET_FAILED,
    RT_MALLOC_FAILED,
    RT_LAUNCH_FAILED,
    RT_EVENT_FAILED,
    RT_CAPTURE_FAILED,
    RT_REGISTER_FAILED,
    RT_LOAD_FAILED,
    RT_GET_FUNC_FAILED,
    RT_DEVICE_FAILED,
};

// =============================================================================
// FAXXXX: DISTRIBUTED
// =============================================================================
enum class DistributedErrorCode : uint32_t {
    INVALID_GROUP_NAME = 0xA0000,
    INVALID_WORLD_SIZE = 0xA0001,
    INVALID_TENSOR_DIM = 0xA0002,
    INVALID_TENSOR_SHAPE = 0xA0003,
    INVALID_TENSOR_DTYPE = 0xA0004,
    INVALID_TENSOR_FORMAT = 0xA0005,
    INVALID_OPERAND_NUM = 0xA0006,
    INVALID_SHMEM_TENSOR = 0xA0007,
    INVALID_SHMEM_VIEW_PARAM = 0xA0008,
    INVALID_OP_TYPE = 0xA0009,
    INVALID_MOE_EXPERT_NUM = 0xA000A,
    INVALID_MOE_TOP_K = 0xA000B,
    INVALID_EXPERT_NUM_PER_RANK = 0xA000C,
    INVALID_TILE_DIM = 0xA1000,
    INVALID_TILE_SHAPE = 0xA1001,
    INVALID_ALIGNMENT = 0xA1002,
    WIN_SIZE_EXCEED_LIMIT = 0xA2000,
    TILE_NUM_EXCEED_LIMIT = 0xA2001,
    DIVISION_BY_ZERO = 0xA2002,
    AICPU_TASK_TIMEOUT = 0xA3000,
    AICPU_TASK_NUM_EXCEED_LIMIT = 0xA3001,
    AICPU_TASK_QUEUE_EMPTY = 0xA3002,
    AICPU_TASKID_NOT_IN_MAP = 0xA3003,
    INVALID_GROUP_INDEX = 0xA3004,
    NULLPTR = 0xA3005,
    INVALID_GROUP_COUNT = 0xA3006,
    HCCL_ALLOC_RESOURCE_FAILED = 0xA4000,
    INVALID_HCCL_TOPO = 0xA4001,
    CONTEXT_CONFIGURE_FAILED = 0xA4002,
    UNKNOW_ERROR = 0xAFFFF
};

// =============================================================================
// FBXXXX: VERIFY
// =============================================================================
enum class VerifyErrorCategory : uint32_t {
    VERIFY_ENABLE = 0xB0000U,
    CONTROL_FLOW = 0xB1000U,
    EXECUTE_OPERATION = 0xB2000U,
    OP_DUMP = 0xB3000U,
    VERIFY_RESULT = 0xB4000U,
};

enum class VerifyEnableScene : uint32_t {
    VERIFY_NOT_ENABLE = 0xB0001U,
    VERIFY_LOAD_CALC_OPS_FAILED = 0xB0002U,
    TOLERANCE_MISMATCH = 0xB0003,
};

enum class ControlFlowScene : uint32_t {
    INVALID_FUNC_IO_SPEC = 0xB1001U,
    INVALID_INPLACE_CHAIN = 0xB1002U,
    INVALID_CALLEE_MAPPING = 0xB1003U,
    FUNC_IO_DATAVIEW_NULL = 0xB1004U,
    FUNC_INCAST_COUNT_MISMATCH = 0xB1005U,
    FUNC_OUTCAST_COUNT_MISMATCH = 0xB1006U,
    FUNC_TENSOR_DATAVIEW_MISMATCH = 0xB1007U,
    FUNC_TENSOR_DATAVIEW_LIST_SIZE_MISMATCH = 0xB1008U,
    FUNC_INPLACE_ALLOC_CONFLICT = 0xB1009U,
    FUNC_TENSOR_DATAVIEW_DUP = 0xB100AU,
    FUNC_SPILL_RAW_TENSOR_DUP = 0xB100BU,
    FUNC_INPLACE_GROUP_NO_FUNC_IO = 0xB100CU,
    FUNC_SLOT_IO_COUNT_MISMATCH = 0xB100DU,
    FUNC_SLOT_MISSING = 0xB100EU,
    FUNC_UNKNOWN_IO_TYPE = 0xB100FU,
    MIX_GLOBAL_TENSOR_WAIT_TIMEOUT = 0xB1010U,
    MIX_SPLIT_PARALLEL_LIMIT_EXCEEDED = 0xB1011U,
    INTERPRETER_SYNC_SIM_WAIT_TIMEOUT = 0xB1012U,
    FUNC_RAW_TENSOR_NULL = 0xB1013U,
    FUNC_RAW_TENSOR_SIZE_MISMATCH = 0xB1014U,
    INVALID_TENSOR_SHAPE = 0xB1015U,
};

enum class ExecuteOperationScene : uint32_t {
    INVALID_TENSOR_SHAPE = 0xB2001U,
    INVALID_TENSOR_DTYPE = 0xB2002U,
    INVALID_TENSOR_SIZE = 0xB2003U,
    CTX_NULL = 0xB2004U,
    CTX_OP_NULL = 0xB2005U,
    CTX_INPUT_COUNT_MISMATCH = 0xB2006U,
    CTX_OUTPUT_COUNT_MISMATCH = 0xB2007U,
    CTX_INPUT_VIEW_NULL = 0xB2008U,
    CTX_OUTPUT_VIEW_NULL = 0xB2009U,
    UNSUPPORTED_OPCODE = 0xB200AU,
    EMPTY_VALIDSHAPE = 0xB200BU,
    VIEWTYPE_BYTES_MISMATCH = 0xB200CU,
    AMULACC_ACC_DTYPE_UNSUPPORTED = 0xB200DU,
    L0C_TO_L1_SHAPE_NOT_2D = 0xB200EU,
    RUNTIME_EXCEPTION = 0xB200FU,
    VIEW_VALIDSHAPE_EXCEEDS_INPUT = 0xB2010U, // view toDynValidShape 大于输入 validShape，场景不合法
};

enum class OpDumpScene : uint32_t {
    DUMP_OPEN_FILE_FAILED = 0xB3001U,
    DUMP_WRITE_FILE_FAILED = 0xB3002U,
};

enum class VerifyResultScene : uint32_t {
    VERIFY_RESULT_MISMATCH = 0xB4001U,
    VERIFY_RESULT_SHAPE_DIFF = 0xB4002U,
    VERIFY_RESULT_DTYPE_DIFF = 0xB4003U,
    VERIFY_RESULT_INDEX_OUTOFBOUNDS = 0xB4004U
};

enum class ElementScene : uint32_t {
    INVALID_ELEMENT_DTYPE = 0xB5001U,
};

// =============================================================================
// FCXXXX: OPERATION
// FC0-FC2XXX: VECTOR
// FC3-FC5XXX: MATMUL
// FC6-FC8XXX: CONV
// FC9XXX: VIEW OP
// =============================================================================
enum class VectorErrorCode : uint32_t {
    ERR_PARAM_INVALID = 0xC0000U,
    ERR_PARAM_DTYPE_UNSUPPORTED = 0xC0001U,
    ERR_PARAM_SHAPE_DIM_UNSUPPORTED = 0xC0002U,
    ERR_PARAM_COUNT_INVALID = 0xC0003U,
    ERR_CONFIG_TILE = 0xC1000U,
    ERR_CONFIG_ALIGNMENT = 0xC1001U,
    ERR_RUNTIME_NULLPTR = 0xC2000U,
    ERR_RUNTIME_LOGIC = 0xC2001U,
};

enum class MatmulErrorCode : uint32_t {
    ERR_PARAM_INVALID = 0xC3000U,
    ERR_PARAM_MISMATCH = 0xC3001U,
    ERR_PARAM_UNSUPPORTED = 0xC3002U,
    ERR_CONFIG_TILE = 0xC4000U,
    ERR_CONFIG_ALIGNMENT = 0xC4001U,
    ERR_CONFIG_UNSUPPORTED = 0xC4002U,
    ERR_RUNTIME_NULLPTR = 0xC5000U,
    ERR_RUNTIME_STATE = 0xC5001U,
    ERR_RUNTIME_LOGIC = 0xC5002U,
};

enum class ConvOperationError : uint32_t { OVER_BUFFER_LIMIT = 0xC6101U, UNKNOWN = 0xC6199U };

enum class ConvExpandFuncError : uint32_t {
    EXPANDFUNC_TENSOR_OP_NULLPTR = 0xC6201U,
    EXPANDFUNC_TENSOR_ATTR_GET_FAILED = 0xC6202U,
    EXPANDFUNC_TILE_OP_NULLPTR = 0xC6203U,
    EXPANDFUNC_PARAMS_INVALID = 0xC6204U,
    EXPANDFUNC_INNER_STATUS_FAILED = 0xC6205U,
    UNKNOWN = 0xC6299U
};

enum class ConvCodenGenError : uint32_t {
    CODEGEN_GET_ATTR_FAILED = 0xC6301U,
    CODEGEN_CHECK_ATTR_INVALID = 0xC6302U,
    CODEGEN_CHECK_DIM_INVALID = 0xC6303U,
    UNKNOWN = 0xC6399U
};

enum class ConvTileOpError : uint32_t {
    TILEOP_TENSOR_FORMAT_FAILED = 0xC6401U,
    TILEOP_SHAPE_SIZE_FAILED = 0xC6402U,
    TILEOP_STC_SHAPE_INVALID = 0xC6403U,
    TILEOP_INDEX_INVALID = 0xC6404U,
    UNKNOWN = 0xC6499U
};

} // namespace npu::tile_fwk

// =============================================================================
// F9XXXX: SIMULATION
// =============================================================================
namespace CostModel {
enum class SimulationErrorCategory {
    INTERNEL_ERROR = 90000U,
    EXTERNAL_ERROR = 91000U,
    FORWARD_SIM = 92000U,
    POST_SIM = 93000U,
    PRECISION_SIM = 94000U,
    UNKNOWN = 99000U,
};

enum class ExternalErrorScene : uint32_t {
    INVALID_CONFIG = 91001U,
    INVALID_CONFIG_NAME = 91002U,
    FILE_FORMAT_ERROR = 91003U,
    FILE_CONTENT_ERROR = 91004U,
    INVALID_PATH = 91005U,
    FILE_OPEN_FAILED = 91006U,
    PYTHON_CMD_ERROR = 91007U
};

enum class ForwardSimErrorScene : uint32_t { INVALID_PIPE_TYPE = 92006U, SHAPE_INVALID = 92007U, DEAD_LOCK = 92010U };

enum class PrecisionSimErrorScene : uint32_t {
    NO_SO_EXISTS = 94001U,
    CANN_LOAD_FAILED = 94002U,
    CMD_ERROR = 94003U,
    CANNSIM_FAILED = 94005U
};
} // namespace CostModel
