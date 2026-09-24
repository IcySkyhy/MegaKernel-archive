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
 * \file device_launcher_binding.h
 * \brief
 */

#ifndef SRC_RUNTIME_DEVICE_LAUNCHER_BINDING_H
#define SRC_RUNTIME_DEVICE_LAUNCHER_BINDING_H

#include <vector>
#include "adapter/api/acl_define.h"
#include "adapter/api/runtime_define.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/runtime/launcher/device_launcher_types.h"

namespace npu::tile_fwk::dynamic {
using DeviceStream = unsigned long long;

struct Evaluator : npu::tile_fwk::SymbolicClosure {
    const std::vector<DeviceTensorData>* inputs{nullptr};
    const std::vector<DeviceTensorData>* outputs{nullptr};

    Evaluator(std::unordered_map<std::string, ScalarImmediateType>& symbolDictArg,
              const std::vector<DeviceTensorData>* inputsArg, const std::vector<DeviceTensorData>* outputsArg)
    {
        ASSERT(DevCommonErr::PARAM_CHECK_FAILED, inputsArg != nullptr && outputsArg != nullptr);
        inputs = inputsArg;
        outputs = outputsArg;
        symbolValueDict = std::shared_ptr<SymbolValueDictType>(&symbolDictArg, [](SymbolValueDictType*) {});
    }

private:
    ScalarImmediateType GetinputShapeDim(ScalarImmediateType argIdx, ScalarImmediateType dim) const
    {
        if (argIdx < static_cast<ScalarImmediateType>(inputs->size())) {
            return (*inputs)[static_cast<size_t>(argIdx)].GetShape()[static_cast<size_t>(dim)];
        }
        size_t outIdx = static_cast<size_t>(argIdx - static_cast<ScalarImmediateType>(inputs->size()));
        return (*outputs)[outIdx].GetShape()[static_cast<size_t>(dim)];
    }

    ScalarImmediateType GetViewValidShapeDim(ScalarImmediateType validshape, ScalarImmediateType viewoffset,
                                             ScalarImmediateType viewshape) const
    {
        validshape -= viewoffset;
        if (validshape > viewshape)
            validshape = viewshape;
        else if (validshape < 0)
            validshape = 0;
        return validshape;
    }

    ScalarImmediateType EvaluateSymbolicCall(const std::string& name, std::vector<ScalarImmediateType>& vals) const
    {
        if (name == "RUNTIME_GetInputShapeDim") {
            return GetinputShapeDim(vals[0], vals[1]);
        } else if (name == "RUNTIME_GetViewValidShapeDim") {
            return GetViewValidShapeDim(vals[0], vals[1], vals[0x2]);
        } else {
            ASSERT(DevCommonErr::PARAM_INVALID, false) << "unsupported call " << name;
            return 0;
        }
    }

    virtual ScalarImmediateType EvaluateExpressionCall(const RawSymbolicScalarPtr ptr) const
    {
        auto expr = std::static_pointer_cast<RawSymbolicExpression>(ptr);
        auto iops = expr->OperandList();
        std::vector<ScalarImmediateType> vals;
        for (size_t i = 1; i < iops.size(); i++) {
            vals.emplace_back(Evaluate(iops[i]));
        }
        auto name = std::static_pointer_cast<RawSymbolicSymbol>(iops[0])->Name();
        return EvaluateSymbolicCall(name, vals);
    }
};

class ExportedOperator : public CachedOperator {
public:
    void ResetFunction(Function* func) { func_ = Program::GetInstance().GetFunctionSharedPtr(func); }

    Function* GetFunction() const { return func_.get(); }

    uint64_t GetWorkSpaceSize(const std::vector<DeviceTensorData>& inputs,
                              const std::vector<DeviceTensorData>& outputs) const
    {
        auto dynAttr = func_->GetDyndevAttribute();
        std::vector<uint8_t>& devProgData = dynAttr->devProgBinary;
        auto* devProg = reinterpret_cast<DevAscendProgram*>(devProgData.data());
        Evaluator eval{dynAttr->inputSymbolDict, &inputs, &outputs};
        if (devProg == nullptr) {
            return 0;
        }
        devProg->memBudget.tensor.maxDynamicAssembleOutcastMem = eval.Evaluate(dynAttr->maxDynamicAssembleOutcastMem);
        uint64_t workspaceSize = devProg->memBudget.Total();
        return workspaceSize;
    }

private:
    std::shared_ptr<Function> func_;
};

int ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(ExportedOperator* op,
                                                         const std::vector<DeviceTensorData>& inputList,
                                                         const std::vector<DeviceTensorData>& outputList,
                                                         DeviceStream aicoreStream, bool streamSynchronize,
                                                         uint8_t* devCtrlCache = nullptr,
                                                         const DeviceLauncherConfig& config = DeviceLauncherConfig());

int DeviceSynchronize(DeviceStream aicpuStream);

int DeviceRunOnce(Function* function, uint8_t* hostCtrlCache = nullptr,
                  const DeviceLauncherConfig& config = DeviceLauncherConfig());

int HasInplaceArgs(Function* function);

void DeviceLauncherInit();

void DeviceLauncherFini();

ExportedOperator* ExportedOperatorBegin();

void ExportedOperatorEnd(ExportedOperator* op);

void CopyDevToHost(const DeviceTensorData& devTensor, DeviceTensorData& hostTensor);

void CopyHostToDev(const DeviceTensorData& devTensor, DeviceTensorData& hostTensor);

uint8_t* CopyHostToDev(uint8_t* data, uint64_t size);

void ChangeCaptureModeRelax();

void ChangeCaptureModeGlobal();

void GetCaptureInfo(RtStream aicoreStream, AclMdlRI& rtModel, bool& isCapture);

void* RegisterKernelBinary(const std::vector<uint8_t>& kernelBinary);

void UnregisterKernelBinary(void* hdl);
} // namespace npu::tile_fwk::dynamic
#endif // SRC_MACHINE_DEVICE_LAUNCHER_H
