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
 * \file codegen_common.h
 * \brief
 */

#ifndef CODEGEN_COMMON_H
#define CODEGEN_COMMON_H

#include <map>

#include "interface/utils/common.h"
#include "tilefwk/data_type.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
const std::string GM_TENSOR_PARAM_STR = "param";
const std::string PREFIX_STR_RAW_SHAPE = "RAWSHAPE";
const std::string PREFIX_STR_STRIDE = "STRIDE";
const std::string PREFIX_STR_OFFSET = "OFFSET";
constexpr const int MAX_DIM = 5;
constexpr const int UPDATE_SHAPE_MAX_DIM = 6;

const std::string GET_PARAM_VALID_SHAPE_BY_IDX = "GET_PARAM_VALID_SHAPE_BY_IDX";
const std::string GET_PARAM_OFFSET_BY_IDX = "GET_PARAM_OFFSET_BY_IDX";

const std::string GM_PARAM_TYPE_FOR_DYN = "CoreFuncParam";
const std::string GM_STACK_BASE = "GMStackBase";

const std::string CG_ERROR = "CG_ERROR";

const std::pair<std::string, std::string> DELIMITER_PARENTHESES("(", ")");
const std::pair<std::string, std::string> DELIMITER_ANGLE_BRACKETS("<", ">");
const std::string CONN_COMMA = ", ";
const std::string SEMICOLON = ";";
const std::string SEMICOLON_BLANK = "; ";
const std::string STMT_END = ";\n";
const std::string END_LINE = "\n";

constexpr const int K_BYTES_OF16_BIT = 2;
constexpr const int K_BYTES_OF32_BIT = 4;

constexpr const unsigned ID0 = 0;
constexpr const unsigned ID1 = 1;
constexpr const unsigned ID2 = 2;
constexpr const unsigned ID3 = 3;
constexpr const unsigned ID4 = 4;
constexpr const unsigned ID5 = 5;
constexpr const unsigned ID6 = 6;
constexpr const unsigned ID7 = 7;

constexpr const int BUFFER_SIZE_256 = 256;
constexpr const int BUFFER_SIZE_512 = 512;
constexpr const int BUFFER_SIZE_1024 = 1024;

const std::string FP16_INF_POS = "0x7C00";
const std::string FP16_INF_NEG = "0xFC00";
const std::string FP16_NAN = "0x7C01";
const std::string FP32_INF_POS = "0x7F800000";
const std::string FP32_INF_NEG = "0xFF800000";
const std::string FP32_NAN = "0x7FC00000";
const std::string BF16_INF_POS = "0x7F80";
const std::string BF16_INF_NEG = "0xFF80";
const std::string BF16_NAN = "0x7FC0";

const int INVALID_TENSOR_MAGIC = -1;

// multi input single output
enum class MISOIdx : unsigned {
    DST_IDX = 0,
    SRC0_IDX = 1,
    SRC1_IDX = 2,
    SRC2_IDX = 3,
};

// multi input multi output
enum class MIMOIdx : unsigned {
    DST_IDX = 0,
    TMP_IDX = 1,
    SRC0_IDX = 2,
    SRC1_IDX = 3,
};

// multi input latched output
enum class MILOIdx : unsigned {
    DST_IDX = 0,
    TMP_IDX = 1,
    TMP2_IDX = 2,
    SRC0_IDX = 3,
};

const std::unordered_map<OperandType, std::string> OPERAND_TYPE_TO_ADDR_TYPE{
    {BUF_DDR, "__gm__"}, {BUF_UB, "__ubuf__"},  {BUF_L1, "__cbuf__"}, {BUF_L0A, "__ca__"}, {BUF_L0B, "__cb__"},
    {BUF_L0C, "__cc__"}, {BUF_FIX, "__fbuf__"}, {BUF_BT, "__cc__"},   {BUF_L0AMX, ""},     {BUF_L0BMX, ""},
};

const std::map<PipeType, std::string> PIPE_ID{
    {PIPE_MTE1, "PIPE_MTE1"}, {PIPE_MTE2, "PIPE_MTE2"}, {PIPE_MTE3, "PIPE_MTE3"}, {PIPE_V, "PIPE_V"},
    {PIPE_M, "PIPE_M"},       {PIPE_FIX, "PIPE_FIX"},   {PIPE_S, "PIPE_S"},       {PIPE_ALL, "PIPE_ALL"},
};

const std::map<OperandType, std::string> BUFFER_TYPE_TO_PREFIX = {
    {OperandType::BUF_UB, "UB"},        {OperandType::BUF_L1, "L1"},   {OperandType::BUF_L0A, "L0A"},
    {OperandType::BUF_L0B, "L0B"},      {OperandType::BUF_L0C, "L0C"}, {OperandType::BUF_FIX, "FIXBUF"},
    {OperandType::BUF_BT, "BIAS"},      {OperandType::BUF_DDR, "GM"},  {OperandType::BUF_L0AMX, "L0A_MX"},
    {OperandType::BUF_L0BMX, "L0B_MX"},
};

// lowercase version
const std::map<OperandType, std::string> BUFFER_TYPE_TO_PREFIX_LC = {
    {OperandType::BUF_UB, "ub"},        {OperandType::BUF_L1, "l1"},   {OperandType::BUF_L0A, "l0a"},
    {OperandType::BUF_L0B, "l0b"},      {OperandType::BUF_L0C, "l0c"}, {OperandType::BUF_FIX, "fbuf"},
    {OperandType::BUF_BT, "bt"},        {OperandType::BUF_DDR, "gm"},  {OperandType::BUF_L0AMX, "l0a_mx"},
    {OperandType::BUF_L0BMX, "l0b_mx"},
};

enum class VecScalMode { VEC_MODE, SCALAR_MODE };

enum class TransMode : int { CAST_NONE = 0, CAST_RINT = 1, CAST_ROUND = 2 };

enum class CopyOutMode : int {
    COPY_MOD_INVALID = -1,
    COPY_MOD_NZ2ND = 0,
    COPY_MOD_NZ2NZ,
    COPY_MOD_ND2ND,
    COPY_MOD_NZ2DN
};

struct CodeGenCtx {
    std::string includePath;
    std::string cceDir;
    bool isMainBlock{false};
    bool isDynamicAligned{false};
    int devRootIndex{0};
    CodeGenCtx() = default;
    CodeGenCtx(std::string inPath, std::string cmpPath, bool isMainBlk = false, bool isDynAligned = false,
               int devRootIdx = 0)
        : includePath(std::move(inPath)),
          cceDir(std::move(cmpPath)),
          isMainBlock(isMainBlk),
          isDynamicAligned(isDynAligned),
          devRootIndex(devRootIdx)
    {}
    bool IsCCEPathEmpty() const { return cceDir.empty(); }
    bool IsIncludePathEmpty() const { return includePath.empty(); }
};

const std::map<MemoryType, OperandType> OPERAND_TYPE_TO_MEMORY_TYPE{
    {MemoryType::MEM_UB, BUF_UB},
    {MemoryType::MEM_L1, BUF_L1},
    {MemoryType::MEM_L0A, BUF_L0A},
    {MemoryType::MEM_L0B, BUF_L0B},
    {MemoryType::MEM_L0C, BUF_L0C},
    {MemoryType::MEM_DEVICE_DDR, BUF_DDR},
    {MemoryType::MEM_BT, BUF_BT},
    {MemoryType::MEM_FIX, BUF_FIX},
    {MemoryType::MEM_FIX_QUANT_PRE, BUF_FIX},
    {MemoryType::MEM_FIX_RELU_PRE, BUF_FIX},
    {MemoryType::MEM_FIX_RELU_POST, BUF_FIX},
    {MemoryType::MEM_FIX_QUANT_POST, BUF_FIX},
    {MemoryType::MEM_FIX_ELT_ANTIQ, BUF_FIX},
    {MemoryType::MEM_FIX_MTE2_ANTIQ, BUF_FIX},
    {MemoryType::MEM_L0AMX, BUF_L0AMX},
    {MemoryType::MEM_L0BMX, BUF_L0BMX},
};

enum class BroadcastOperand : int64_t {
    NONE = 0,
    LEFT_OPERAND = 1,
    RIGHT_OPERAND = 2,
};

} // namespace npu::tile_fwk

#endif // CODEGEN_COMMON_H
