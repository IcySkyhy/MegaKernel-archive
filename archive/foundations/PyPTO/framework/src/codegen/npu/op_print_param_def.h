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
 * \file op_print_param_def.h
 * \brief
 */

#ifndef OP_PRINT_PARAM_DEF_H
#define OP_PRINT_PARAM_DEF_H

#include <string>
#include <vector>

namespace npu::tile_fwk {
struct PrintScatterElemParam {
    const int axis;
    const int scatterMode;
    const std::string& dVar;
    const std::string& s0Var;
    const std::string& s1Var;
    std::vector<int64_t>& dstRawShape;
    std::vector<int64_t>& src1RawShape;
    const std::vector<std::string>& dataTypeExpr;
};

struct PrintScatterParam {
    const int axis;
    const int scatterMode;
    const std::string& dVar;
    const std::string& s1Var;
    const std::string& s2Var;
    std::vector<int64_t>& dstRawShape;
    std::vector<int64_t>& src1RawShape;
    std::vector<int64_t>& src2RawShape;
    const std::vector<std::string>& dataTypeExpr;
};

struct PrintIndexAddParam {
    const int axis;
    const std::string& dstVar;
    const std::string& srcVar;
    const std::string& indicesVar;
    std::vector<int64_t>& dstRawShape;
    std::vector<int64_t>& srcRawShape;
    const std::vector<std::string>& dataTypeExpr;
};

struct PrintIndexPutParam {
    const std::string& dVar;
    const std::string& s1Var;
    const std::vector<std::string>& s2Var;
    const std::vector<int64_t>& gmShape;
    const std::vector<int64_t>& src1RawShape;
    const std::vector<std::string>& dataTypeExpr;
    const bool accumulate;
};

enum class WhereOpIdx : unsigned { resIdx = 0, tempIdx, condIdx, src0Idx, src1Idx };
struct WhereParam {
    std::vector<std::string> templateList;
    std::vector<std::string> paramList;
    std::vector<std::string> dynParamList;
    std::vector<std::string> varExpr;
    std::vector<std::string> dataTypeExpr;
};

struct PrintDupOpParam {
    const std::string& dVar;
    const std::string& dstDtypeStr;
    const std::string& dupV;
};

struct PrintUnaryParam {
    const std::string& s0Var;
    const std::string& dVar;
    const std::string& srcDtypeStr;
    const std::string& dstDtypeStr;
};

struct PrintUnaryTmpParam {
    const std::string& s0Var;
    const std::string& dVar;
    const std::string& tmpVar;
    const std::string& srcDtypeStr;
    const std::string& dstDtypeStr;
    const std::string& tmpDtypeStr;
};

struct PrintUnaryTmpBuffParam {
    const std::string& s0Var;
    const std::string& tmpVar;
    const std::string& dVar;
    const std::string& srcDtypeStr;
    const std::string& tmpDtypeStr;
    const std::string& dstDtypeStr;
};

struct PrintMemCopyWithL0CParam {
    const unsigned uf;
    const unsigned gmIdx;
    const unsigned localIdx;
    const std::vector<std::string>& addrTypeHead;
    const std::vector<std::string>& addrExpr;
    const std::vector<int64_t>& gmShape;
    const std::vector<int64_t>& localRawShape;
    const std::vector<std::string>& dataTypeExpr;
};

struct PrintMemCopyWithL1Param {
    const bool isCopyLocalToGM;
    const bool isSpillingToGM;
    const unsigned uf;
    const unsigned gmIdx;
    const unsigned localIdx;
    const std::vector<std::string>& addrTypeHead;
    const std::vector<std::string>& addrExpr;
    const std::vector<int64_t>& gmShape;
    const std::vector<int64_t>& localRawShape;
    const std::vector<std::string>& dataTypeExpr;
};

struct PrintMemCopyWithUBParam {
    const unsigned gmIdx;
    const unsigned localIdx;
    const bool isSpillingToGM;
    const std::vector<std::string>& addrTypeHead;
    std::vector<std::string>& addrExpr;
    std::vector<std::string>& dataTypeExpr;
};

struct PrintGatherParam {
    const std::string& s0Var;
    const std::string& s1Var;
    const std::string& dVar;
    const std::string& src0DtypeStr;
    const std::string& src1DtypeStr;
    const std::string& dstDtypeStr;
    const int64_t axis;
};

struct PrintBinaryScalarParam {
    const std::string& s0Var;
    const std::string& dVar;
    const std::string& src0DtypeStr;
    const std::string& dstDtypeStr;
    const size_t dim;
};

struct PrintBinaryParam {
    const std::string& s0Var;
    const std::string& s1Var;
    const std::string& dVar;
    const std::string& src0DtypeStr;
    const std::string& src1DtypeStr;
    const std::string& dstDtypeStr;
};

struct PrintBinaryBrcParam {
    const std::string& s0Var;
    const std::string& s1Var;
    const std::string& dVar;
    const std::string& tmpVar;
    const std::string& src0DtypeStr;
    const std::string& src1DtypeStr;
    const std::string& dstDtypeStr;
    const std::string& tmpDtypeStr;
};

struct PrintTransposeDataMoveParam {
    const unsigned gmIdx;
    const unsigned localIdx;
    const std::string& localVar;
    const std::vector<int64_t>& gmShape;
    const std::string& localDtypeStr;
    const std::string& gmDtypeStr;
};

struct PrintGatherEleParam {
    const int axis;
    const std::string& dVar;
    const std::string& s0Var;
    const std::string& s1Var;
    std::vector<int64_t>& dstOriginShape;
    std::vector<int64_t>& dstRawShape;
    std::vector<int64_t>& src0RawShape;
    std::vector<int64_t>& src1RawShape;
    const std::vector<std::string>& dataTypeExpr;
};

struct PrintIndexOutCastParam {
    const std::string& s0Var;
    const std::string& s1Var;
    const std::vector<std::string>& addrExpr;
    const std::vector<int64_t>& gmShape;
    std::vector<int64_t>& src0OriginShape;
    std::vector<int64_t>& src0RawShape;
    std::vector<int64_t>& src1OriginShape;
    std::vector<int64_t>& src1RawShape;
    const std::vector<std::string>& dataTypeExpr;
    const std::string& cacheMode;
    const std::string& blockSize;
};

struct PrintCumSumParam {
    const int axis;
    const bool flag;
    const std::string& dVar;
    const std::string& inputVar;
    std::vector<int64_t>& inputRawShape;
    const std::string* dataTypeExpr;
};

struct DynamicParamPackMTE {
    std::vector<std::string> gmShapeExpr;
    std::vector<std::string> gmOffsetExpr;
    std::vector<std::string> paramList;
};

struct SortParam {
    std::vector<int64_t> dstShape{4, 1};
    std::vector<int64_t> tmpShape{4, 1};
    std::vector<int64_t> srcShape{4, 1};
    const std::string s0Var;
    const std::string dVar;
    const std::string tVar;
    const std::string srcDtypeStr;
    const std::string dstDtypeStr;
    const std::string tmpDtypeStr;
};

struct TiledSortParam {
    std::vector<int64_t> dstShape{4, 1};
    std::vector<int64_t> srcShape{5, 1};
    const std::string s0Var;
    const std::string s1Var;
    const std::string s2Var;
    const std::string s3Var;
    const std::string tmpVar;
    const std::string dVar;
    const std::string srcDtypeStr;
    const std::string dstDtypeStr;
};

struct PrintQuantizeParam {
    const std::string& dstVar;
    const std::string& srcVar;
    const std::string& scaleVar;
    const std::string& offsetVar;
    const bool isAsymmetric;
};

} // namespace npu::tile_fwk

#endif // OP_PRINT_PARAM_DEF_H
