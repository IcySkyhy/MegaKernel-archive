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
 * \file codegen_op_litenpu.h
 * \brief
 */

#ifndef CODEGEN_OP_LITENPU_H
#define CODEGEN_OP_LITENPU_H

#include "codegen/npu/codegen_op_npu.h"

namespace npu::tile_fwk {

class CodeGenOpLiteNPU : public CodeGenOpNPU {
public:
    explicit CodeGenOpLiteNPU(const CodeGenOpNPUCtx& ctx);
    ~CodeGenOpLiteNPU() override = default;

private:
    TileTensor QueryTileTensorByIdx(int paramIdx) const override;
    std::vector<SymbolicScalar> GetOffsetFromAttr(int idx) const override;

    std::string GenGmParamVar(unsigned gmParamIdx) const override;

    void UpdateGemvTileOpName() override;

    TileTensor BuildTileTensor(int paramIdx, const std::string& usingType,
                               const TileTensorShape& tileTensorShape = {}) override;

    void UpdateTileTensorShapeAndStride(int paramIdx, TileTensor& tileTensor, bool isSpillToGm,
                                        const TileTensorShape& tileTensorShape = {}) override;

    std::vector<std::string> GetGmOffsetForTileTensor(unsigned gmIdx) const override;

    void UpdateGmParamIdx(const Operation& oper);
    bool ContainsRuntimeCoaGetParmOffset(const std::vector<SymbolicScalar>& offsets) const;
    void ExtractCoaGetParamOffset(const SymbolicScalar& expr, int& base, int& idx) const;
    std::vector<SymbolicScalar> GetStaticOffsetFromLinearArgList(const std::vector<SymbolicScalar>& dynOffset) const;
    Function& subFunc;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_OP_LITENPU_H
