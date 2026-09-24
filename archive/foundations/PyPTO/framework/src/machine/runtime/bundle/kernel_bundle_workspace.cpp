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
 * \file kernel_bundle_workspace.cpp
 * \brief Load-side dynamic-workspace evaluation (see header). The pack-side counterpart is pack/kernel_bundle_pack.cpp,
 *        which stays in tile_fwk_runtime while this file ships in libtile_fwk_bundle.so.
 */

#include "machine/runtime/bundle/kernel_bundle_workspace.h"

#include <exception>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "machine/runtime/bundle/kernel_bundle_loader.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "interface/function/function.h"
#include "interface/tensor/symbolic_scalar.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::bundle {

uint64_t EvalWorkspaceForShapes(LoadedBundle& bundle, const std::vector<dynamic::DeviceTensorData>& inputs,
                                const std::vector<dynamic::DeviceTensorData>& outputs)
{
    auto* devProg = reinterpret_cast<dynamic::DevAscendProgram*>(bundle.devProgram.data());
    if (devProg == nullptr) {
        return 0;
    }
    if (bundle.symbolMeta.empty()) {
        return devProg->memBudget.Total(); // old/static bundle: no dynamic trees, keep baked constant
    }

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(bundle.symbolMeta.begin(), bundle.symbolMeta.end());
    } catch (const std::exception& e) {
        MACHINE_LOGW("[kernel-bundle] SymbolMeta parse failed (%s); using static workspace", e.what());
        return devProg->memBudget.Total();
    }

    // Rebuild the recording-time symbol dict; the Evaluator resolves runtime shape dims from the tensors, so the
    // dynamism comes from `inputs`/`outputs`, not this dict.
    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    if (root.contains("symbols")) {
        for (auto it = root["symbols"].begin(); it != root["symbols"].end(); ++it) {
            symbolDict[it.key()] = static_cast<ScalarImmediateType>(it.value().get<int64_t>());
        }
    }

    // Mirror ExportedOperator::GetWorkSpaceSize / RefillDynamicMemBudgets: evaluate the trees against the real
    // shapes and write the concrete values into memBudget in place.
    dynamic::Evaluator eval{symbolDict, &inputs, &outputs};

    // Rebuild + apply the dynamic cell-match stride patches for these shapes. The patched host desc rides to the
    // device with the devProgram; the same patches also go into the launch args (re-applied on device via
    // ApplyDynamicCellMatchDescPatchesFromLaunchArgs). Without this the device reads a stale baked stride.
    bundle.cellMatchStridePatches.clear();
    if (root.contains("cellMatchLaunch")) {
        std::vector<DyndevFunctionAttribute::DynamicCellMatchLaunchMeta> metaList;
        for (const auto& m : root["cellMatchLaunch"]) {
            DyndevFunctionAttribute::DynamicCellMatchLaunchMeta lm;
            lm.slotIndex = m.at("slot").get<int>();
            lm.descOffset = m.at("descOffset").get<uint64_t>();
            lm.cellShape = m.at("cellShape").get<std::vector<int>>();
            for (const auto& row : m.at("cand")) {
                std::vector<SymbolicScalar> dims;
                for (const auto& e : row) {
                    dims.push_back(LoadSymbolicScalar(e));
                }
                lm.candidateRawDims.push_back(std::move(dims));
            }
            metaList.push_back(std::move(lm));
        }
        bundle.cellMatchStridePatches = dynamic::PrepareDynamicCellMatchDescPatches(metaList, eval);
        dynamic::PatchHostDynamicCellMatchTableDesc(devProg, bundle.cellMatchStridePatches);
    }

    if (root.contains("assembleMem")) {
        const SymbolicScalar assembleMem = LoadSymbolicScalar(root["assembleMem"]);
        if (assembleMem.IsValid()) {
            devProg->memBudget.tensor.maxDynamicAssembleOutcastMem = static_cast<uint64_t>(eval.Evaluate(assembleMem));
        }
    }
    if (root.contains("cellMatchMem")) {
        const SymbolicScalar cellMatchMem = LoadSymbolicScalar(root["cellMatchMem"]);
        if (cellMatchMem.IsValid()) {
            devProg->memBudget.metadata.maxDynamicCellMatchTableMem = static_cast<uint64_t>(
                eval.Evaluate(cellMatchMem));
            const uint64_t slotNum = devProg->memBudget.metadata.dynamicCellMatchSlotNum;
            devProg->memBudget.metadata.dynamicCellMatch = slotNum *
                                                           devProg->memBudget.metadata.maxDynamicCellMatchTableMem;
        }
    }
    return devProg->memBudget.Total();
}

} // namespace npu::tile_fwk::bundle
