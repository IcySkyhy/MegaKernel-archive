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
 * \file codegen_cloudnpu.cpp
 * \brief
 */
#include "codegen_cloudnpu.h"

#include "codegen_op_cloudnpu.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/utils/parallel_execute.h"
#include "interface/configs/config_manager_ng.h"

namespace npu::tile_fwk {
const int PMU_ID_FROM_FUNC_HASH_LEN = 3;

bool CodeGenCloudNPU::IsEnablePMUTrace() const
{
    if (platform_ != NPUArch::DAV_3510) {
        return false;
    }
    return config::GetCodeGenOption<bool>(ENABLE_PMU_TRACE);
}

std::string CodeGenCloudNPU::GenPMUId(const Function& subFunc) const
{
    std::string id;
    if (ctx.isMainBlock) {
        id += "1";
    }
    std::string funcHash = subFunc.GetFunctionHash();
    ASSERT(FwkErr::INVALID_FUNCTION, funcHash.size() >= PMU_ID_FROM_FUNC_HASH_LEN)
        << "funcHash size is less than PMU_ID_FROM_FUNC_HASH_LEN";
    funcHash = funcHash.substr(funcHash.size() - PMU_ID_FROM_FUNC_HASH_LEN);
    id += funcHash; // the max value supported by bisheng is 4096

    // Emit canonical decimal to avoid C++ octal literal parsing (e.g. 058, 095).
    constexpr unsigned kMaxPmuId = 4096;
    unsigned idVal = std::stoul(id);
    idVal %= kMaxPmuId;
    return std::to_string(idVal);
}

void CodeGenCloudNPU::PrintPMUTraceAhead(const Function& subFunc, std::ostringstream& oss)
{
    if (!IsEnablePMUTrace()) {
        return;
    }

    pmuId_ = GenPMUId(subFunc);

    static const std::string pmuTraceAhead = R"!!!(
// ------------------- For PMU Trace Start -------------------
__asm__ volatile("bar.all");
bisheng::cce::mark_stamp<PIPE_MTE2, ${id_value}$>();
__asm__ volatile(".rept 100\n\tNOP \n\t.endr");
bisheng::cce::mark_stamp<PIPE_MTE2, ${id_value}$>();
// ------------------- For PMU Trace End -------------------
)!!!";
    oss << StringSubstitute(pmuTraceAhead, {{"id_value", pmuId_}}) << "\n";
}

void CodeGenCloudNPU::PrintPMUTraceAfter(std::ostringstream& oss) const
{
    if (!IsEnablePMUTrace()) {
        return;
    }

    static const std::string pmuTraceAhead = R"!!!(
// ------------------- For PMU Trace Start -------------------
__asm__ volatile("bar.all");
bisheng::cce::mark_stamp<PIPE_MTE3, ${id_value}$>();
__asm__ volatile(".rept 1000\n\tNOP \n\t.endr");
bisheng::cce::mark_stamp<PIPE_MTE3, ${id_value}$>();
// ------------------- For PMU Trace End -------------------
)!!!";
    oss << StringSubstitute(pmuTraceAhead, {{"id_value", pmuId_}});
}

void CodeGenCloudNPU::GenFuncBody(Function& subFunc, Function& topFunc, std::ostringstream& oss)
{
    PrintPMUTraceAhead(subFunc, oss);

    OperationsViewer operationList = subFunc.Operations(false);
    if (operationList.IsEmpty()) {
        CODEGEN_LOGW("operationList from PASS is empty, func magic name: %s, func hash: %s",
                     subFunc.GetMagicName().c_str(), subFunc.GetFunctionHash().c_str());
    }

    CODEGEN_LOGI("TopFunc Type is %s\nFunction to codegen:\n %s\n", topFunc.GetFunctionTypeStr().c_str(),
                 subFunc.Dump().c_str());

    std::shared_ptr<SymbolManager> symbolMgr = std::make_shared<SymbolManager>();
    std::shared_ptr<ForBlockManager> forBlkMgr = std::make_shared<ForBlockManager>(symbolMgr);
    FloatSpecValMgr floatSpecValMgr;
    std::string tileOpSourceRegion;
    tileOpSourceRegion.reserve(CODE_RESERVED_SIZE);
    for (const auto& op : operationList) {
        CODEGEN_LOGI("======================== Op CodeGenNPU Start ========================\nGen OP IS: %s",
                     op.Dump().c_str());
        if (SKIP_OPCODE_FOR_CODEGEN.find(op.GetOpcode()) != SKIP_OPCODE_FOR_CODEGEN.end()) {
            CODEGEN_LOGI("ignore this op\n------------------------ Op CodeGenNPU Finish -----------------------");
            continue;
        }

        GenAllocForLocalBuffer(op, symbolMgr);
        floatSpecValMgr.UpdateByOp(op);

        CodeGenOpCloudNPU cop({symbolMgr, topFunc, subFunc, op, ctx.isMainBlock, ctx.isDynamicAligned, forBlkMgr});
        std::string tileOpSourceCode = cop.GenOpCode();
        ASSERT(GenCodeErr::GEN_OP_CODE_FAILED, tileOpSourceCode.find(CG_ERROR) == tileOpSourceCode.npos)
            << "Generate code of op failed, op is " << op.Dump();

        tileOpSourceRegion.append(symbolMgr->GenNewTileTensorDefs());
        tileOpSourceRegion.append(tileOpSourceCode);
        CODEGEN_LOGI("------------------------ Op CodeGenNPU Finish -----------------------");
    }

    floatSpecValMgr.PrintFloatSpecVal(oss);
    symbolMgr->PrintAddrAllocs(oss);
    GenDynParamForExpr(oss, subFunc);
    symbolMgr->GenUsingList(oss);
    oss << tileOpSourceRegion;

    PrintPMUTraceAfter(oss);
}

} // namespace npu::tile_fwk
