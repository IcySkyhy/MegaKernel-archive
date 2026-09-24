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
 * \file flow_verifier.cpp
 * \brief
 */

#include <cmath>
#include <limits>
#include <numeric>

#include "flow_verifier.h"
#include "tilefwk/tilefwk.h"
#include "interface/interpreter/interpreter_log.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/error_code.h"
#include "tilefwk/comm_group_recorder.h"
#include "communication.h"

namespace npu::tile_fwk {

namespace {

// Scalar decode: FP8 aligned with calculator fp8 paths; HF8 aligned with calculator/fp_convert.cpp (Hf8ToFloat32).
float DecodeFp8E4M3(uint8_t x)
{
    const int xi = static_cast<int>(x);
    const float sign = (xi & 0x80) != 0 ? -1.0f : 1.0f;
    const int expBits = (xi >> 3) & 0xF;
    const int mantBits = xi & 0x7;
    if (expBits == 0) {
        return sign * (static_cast<float>(mantBits) / 8.0f) * (1.0f / 64.0f);
    }
    if (expBits >= 0x1 && expBits <= 0xe) {
        const float expVal = static_cast<float>(expBits) - 7.0f;
        const float mantVal = 1.0f + static_cast<float>(mantBits) / 8.0f;
        return sign * std::pow(2.0f, expVal) * mantVal;
    }
    // expBits == 15: OCP float8_e4m3fn — mant==111 is NaN; else finite (1+m/8)*2^8, max 448 at mant=110.
    if (mantBits == 0x7) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float mantVal = 1.0f + static_cast<float>(mantBits) / 8.0f;
    return sign * std::pow(2.0f, 8.0f) * mantVal;
}

float DecodeFp8E5M2(uint8_t x)
{
    const int xi = static_cast<int>(x);
    const float sign = (xi & 0x80) != 0 ? -1.0f : 1.0f;
    const int expBits = (xi >> 2) & 0x1F;
    const int mantBits = xi & 0x3;
    if (expBits == 0) {
        return sign * (static_cast<float>(mantBits) / 4.0f) * (1.0f / 16384.0f);
    }
    if (expBits >= 0x1 && expBits <= 0x1e) {
        const float expVal = static_cast<float>(expBits) - 15.0f;
        const float mantVal = 1.0f + static_cast<float>(mantBits) / 4.0f;
        return sign * std::pow(2.0f, expVal) * mantVal;
    }
    if (mantBits == 0) {
        return sign * std::numeric_limits<float>::infinity();
    }
    return std::numeric_limits<float>::quiet_NaN();
}

float DecodeFp8E8M0(uint8_t x)
{
    const int xi = static_cast<int>(x);
    const float sign = (xi & 0x80) != 0 ? -1.0f : 1.0f;
    const int expBits = xi & 0x7F;
    const float expVal = static_cast<float>(expBits) - 63.0f;
    return sign * std::pow(2.0f, expVal);
}

float DecodeHf8(uint8_t x)
{
    const int signBit = (x >> 7) & 0x1;
    const float sign = signBit != 0 ? -1.0f : 1.0f;
    const int lower7 = x & 0x7F;
    const int top4 = lower7 >> 3;
    if (top4 == 0) { // Subnormal: D=0000, M in [0,7]
        const int mv = lower7 & 0x7;
        return sign * std::pow(2.0f, static_cast<float>(mv - 0x17));
    }
    if (top4 == 1) { // Normal: D=0001, E_v=0
        const int mv = lower7 & 0x7;
        return sign * (1.0f + static_cast<float>(mv) / 8.0f);
    }
    const int top3 = lower7 >> 4;
    if (top3 == 1) { // D=001, E bit count = 1, |E_v|=1
        const int eb = (lower7 >> 3) & 0x1;
        const int ev = (eb == 0) ? 1 : -1;
        const int mv = lower7 & 0x7;
        return sign * std::pow(2.0f, static_cast<float>(ev)) * (1.0f + static_cast<float>(mv) / 8.0f);
    }
    const int top2 = lower7 >> 5;
    if (top2 == 1) { // D=01, E bit count = 2, |E_v| in [2,3]
        const int eb = (lower7 >> 3) & 0x3;
        const int evSign = (eb >> 1) & 0x1;
        const int evAbs = 2 + (eb & 0x1);
        const int ev = evSign ? -evAbs : evAbs;
        const int mv = lower7 & 0x7;
        return sign * std::pow(2.0f, static_cast<float>(ev)) * (1.0f + static_cast<float>(mv) / 8.0f);
    }
    if (top2 == 0x2) { // D=10, E bit count = 3, |E_v| in [4,7]
        const int eb = (lower7 >> 2) & 0x7;
        const int evSign = (eb >> 2) & 0x1;
        const int evAbs = 4 + (eb & 0x3);
        const int ev = evSign ? -evAbs : evAbs;
        const int mv = lower7 & 0x3;
        return sign * std::pow(2.0f, static_cast<float>(ev)) * (1.0f + static_cast<float>(mv) / 4.0f);
    }
    // D=11, E bit count = 4, |E_v| in [8,15]
    const int eb = (lower7 >> 1) & 0xF;
    const int evSign = (eb >> 3) & 0x1;
    const int evAbs = 8 + (eb & 0x7);
    const int ev = evSign ? -evAbs : evAbs;
    const int mv = lower7 & 0x1;
    return sign * std::pow(2.0f, static_cast<float>(ev)) * (1.0f + static_cast<float>(mv) / 2.0f);
}

double Fp8StorageToDouble(uint8_t bits, DataType fmt)
{
    float v = 0.0f;
    switch (fmt) {
        case DT_FP8:
        case DT_FP8E4M3:
            v = DecodeFp8E4M3(bits);
            break;
        case DT_HF8:
            v = DecodeHf8(bits);
            break;
        case DT_FP8E5M2:
            v = DecodeFp8E5M2(bits);
            break;
        case DT_FP8E8M0:
            v = DecodeFp8E8M0(bits);
            break;
        default:
            ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, false);
            break;
    }
    return static_cast<double>(v);
}

} // namespace

// uint8-backed low-precision float formats: DT_FP8*, DT_HF8. Decodes each byte with Fp8StorageToDouble(fmt).
FlowVerifier::CompareResult FlowVerifier::CompareFp8TensorData(const std::shared_ptr<LogicalTensorData>& goldenDataView,
                                                               const std::shared_ptr<LogicalTensorData>& outputDataView,
                                                               DataType fp8Format, float rtol, float atol,
                                                               int errorCountThreshold, int failNum)
{
    auto& validShape = goldenDataView->GetValidShape();
    const auto size = std::accumulate(validShape.begin(), validShape.end(), 1, std::multiplies<>());
    CompareResult compareResult(size, rtol, atol, errorCountThreshold, failNum, validShape);
    CompareDataRecursiveWithLeaf(
        compareResult, 0, goldenDataView->GetStorageOffset(), outputDataView->GetStorageOffset(), goldenDataView,
        outputDataView, 0,
        [&](CompareResult& cr, size_t lastCount, int64_t outOff, int64_t gOff, int64_t index,
            const std::shared_ptr<LogicalTensorData>& gv, const std::shared_ptr<LogicalTensorData>& ov) {
            const uint8_t* gp = &gv->Get<uint8_t>(gOff);
            const uint8_t* op = &ov->Get<uint8_t>(outOff);
            for (size_t i = 0; i < lastCount; i++) {
                CompareScalarPair(cr, index + static_cast<int64_t>(i), Fp8StorageToDouble(gp[i], fp8Format),
                                  Fp8StorageToDouble(op[i], fp8Format));
            }
        });
    compareResult.UpdateErrorCountThreshold();
    return compareResult;
}

FlowVerifier::CompareResult FlowVerifier::VerifyResult(const std::shared_ptr<LogicalTensorData>& goldenDataView,
                                                       const std::shared_ptr<LogicalTensorData>& outputDataView,
                                                       float rtol, float atol)
{
    // tensor maybe padded during PadLocalBuffer Pass, tensor shape maybe changed, just check the valid data
    if (goldenDataView->GetValidShape() != outputDataView->GetValidShape()) {
        auto dtype = outputDataView->GetData()->GetDataType();
        const auto& rawShape = outputDataView->GetData()->GetShape();
        size_t requiredSize = RawTensorData::CalcRequiredSize(dtype, rawShape);
        ASSERT(VerifyResultScene::VERIFY_RESULT_MISMATCH, goldenDataView->GetData()->size() == requiredSize)
            << "RawTensor size mismatch when updating: existing size=" << goldenDataView->GetData()->size()
            << ", required size=" << requiredSize;
        goldenDataView->UpdateValidShape(outputDataView->GetValidShape(), outputDataView->GetOffset());
        goldenDataView->GetData()->UpdateShapeAndDtype(dtype, rawShape);
    }
    ASSERT(VerifyResultScene::VERIFY_RESULT_SHAPE_DIFF,
           goldenDataView->GetValidShape() == outputDataView->GetValidShape());
    ASSERT(VerifyResultScene::VERIFY_RESULT_DTYPE_DIFF, goldenDataView->GetDataType() == outputDataView->GetDataType());
    switch (goldenDataView->GetDataType()) {
        case DT_INT8:
            return CompareData<int8_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_INT16:
            return CompareData<int16_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_INT32:
            return CompareData<int32_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_INT64:
            return CompareData<int64_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_FP16:
            return CompareData<npu::tile_fwk::float16, float>(goldenDataView, outputDataView, rtol, atol);
        case DT_FP32:
            return CompareData<float, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_BF16:
            return CompareData<npu::tile_fwk::bfloat16, float>(goldenDataView, outputDataView, rtol, atol);
        case DT_UINT8:
            return CompareData<uint8_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_UINT16:
            return CompareData<uint16_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_UINT32:
            return CompareData<uint32_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_UINT64:
            return CompareData<uint64_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_DOUBLE:
            return CompareData<double, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_BOOL:
            return CompareData<uint8_t, double>(goldenDataView, outputDataView, rtol, atol);
        case DT_FP8:
        case DT_HF8:
        case DT_FP8E4M3:
        case DT_FP8E5M2:
        case DT_FP8E8M0:
            return CompareFp8TensorData(goldenDataView, outputDataView, goldenDataView->GetDataType(), rtol, atol);
        default:
            ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, false);
            break;
    }
    return CompareResult();
}

bool FlowVerifier::VerifyResult(const std::vector<std::shared_ptr<LogicalTensor>>& tensorDatalist,
                                const std::vector<std::shared_ptr<LogicalTensor>>& goldenDatalist,
                                const std::string& key, const std::string tensorName,
                                const std::vector<std::shared_ptr<LogicalTensorData>>& goldenDataViewList,
                                const std::vector<std::shared_ptr<LogicalTensorData>>& tensorDataViewList, float rtol,
                                float atol)
{
    bool result = true;
    if (goldenDataViewList.size() != tensorDataViewList.size()) {
        INTERPRETER_EVENT("%s Verify NO_COMPARE", key.c_str());
        return result;
    }
    for (size_t k = 0; k < tensorDataViewList.size(); k++) {
        if (!goldenDataViewList[k]) {
            INTERPRETER_EVENT("%s Verify for %zu data view list index %zu result NO_COMPARE", key.c_str(),
                              goldenDataViewList.size(), k);
            continue;
        }
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        auto ts = tv.tv_sec * 1000000 + tv.tv_usec; // 1000000 is us per sec
        std::string goldenFileName;
        std::string fileName = tensorName + "~" + std::to_string(k) + "~" + std::to_string(ts) + ".data";
        functionInterpreter_->DumpTensorBinary(tensorDataViewList[k], fileName);

        std::vector<std::string> ProgrameInfo(toIndex(ProgrameInfoCsvHeader::COL_COUNT));
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::passName)] = functionInterpreter_->execDumpPassName;
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::pathFuncMagicName)] = functionInterpreter_->execDumpFunPath;
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::pathFuncMagic)] = std::to_string(
            functionInterpreter_->pathFuncMagic);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::pathFuncHash)] = std::to_string(
                                                                         functionInterpreter_->pathFuncHash) +
                                                                     "'";
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::outputRawShape)] = functionInterpreter_->ShapeToString(
            tensorDataViewList[k]->GetData()->GetShape());
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::outputShape)] = functionInterpreter_->ShapeToString(
            tensorDataViewList[k]->GetShape());
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::outputValidShape)] = functionInterpreter_->ShapeToString(
            tensorDataViewList[k]->GetValidShape());
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::outputDtype)] = DataType2String(
            tensorDataViewList[k]->GetDataType(), true);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::outputFormat)] = std::to_string(
            tensorDatalist[k]->GetRawTensor()->format);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::outputRawMagic)] = std::to_string(
            tensorDatalist[k]->GetRawTensor()->GetRawMagic());
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::outputSymbol)] = tensorDatalist[k]->GetRawTensor()->GetSymbol();
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::outputTensor)] = fileName;
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::verifyResult)] = "PASS";
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::aTimeStamp)] = std::to_string(ts) + "'";
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::bTimeStamp)] = std::to_string(ts) + "'";
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::loopInfo)] = functionInterpreter_->GetLoopSymbolString();
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::rtolAndAtol)] = functionInterpreter_->ToStrWithPrecision(rtol) +
                                                                    "/" +
                                                                    functionInterpreter_->ToStrWithPrecision(atol);
        if (functionInterpreter_->execDumpPassName == "tensor_graph") {
            ProgrameInfo[toIndex(ProgrameInfoCsvHeader::goldenPassName)] = "user_golden";
            ProgrameInfo[toIndex(ProgrameInfoCsvHeader::ioflag)] = "a" + std::to_string(k);
            goldenFileName = "usergolden~" + std::to_string(k) + ".data";
        } else {
            ProgrameInfo[toIndex(ProgrameInfoCsvHeader::goldenPassName)] = "tensor_graph";
            ProgrameInfo[toIndex(ProgrameInfoCsvHeader::ioflag)] = "o" + std::to_string(k);
            goldenFileName = tensorName + "~" + std::to_string(k) + "~" + "golden" + "~" + std::to_string(ts) + ".data";
            ProgrameInfo[toIndex(ProgrameInfoCsvHeader::goldenRawMagic)] = std::to_string(
                goldenDatalist[k]->GetRawTensor()->GetRawMagic());
        }
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::goldenTensor)] = goldenFileName;
        functionInterpreter_->DumpTensorBinary(goldenDataViewList[k], goldenFileName);

        auto tensorGraphResult = VerifyResult(goldenDataViewList[k], tensorDataViewList[k], rtol, atol);
        if (!tensorGraphResult.Check()) {
            INTERPRETER_LOGE_FULL(VerifyResultScene::VERIFY_RESULT_MISMATCH,
                                  "%s Verify for %zu data view list index %zu result FAILED", key.c_str(),
                                  goldenDataViewList.size(), k);
            fprintf(functionInterpreter_->execDumpErrorFile, "[VERIFY:FAIL] %s, %s, %s, %s, %s\n",
                    functionInterpreter_->execDumpPassName.c_str(), functionInterpreter_->execDumpFunPath.c_str(),
                    functionInterpreter_->GetLoopSymbolString().c_str(),
                    std::to_string(tensorDatalist[k]->GetRawTensor()->GetRawMagic()).c_str(),
                    tensorDatalist[k]->GetRawTensor()->GetSymbol().c_str());
            std::ostringstream oss;
            tensorGraphResult.DumpDataDetail(oss);
            fprintf(functionInterpreter_->execDumpErrorFile, "%s", oss.str().c_str());
            ProgrameInfo[toIndex(ProgrameInfoCsvHeader::verifyResult)] = "FAIL";
            result = false;
        } else {
            INTERPRETER_EVENT("%s Verify for %zu data view list index %zu result PASS", key.c_str(),
                              goldenDataViewList.size(), k);
        }
        CompareResultDetail res = tensorGraphResult.Dump();
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::failCnt)] = std::to_string(res.failNum) + "/" +
                                                                std::to_string(res.warnNum) + "/" +
                                                                std::to_string(res.toleranceCnt);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::totalCnt)] = std::to_string(res.totalCnt) + "/" +
                                                                 std::to_string(res.zeroCnt) + "/" +
                                                                 std::to_string(res.infnanCnt);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::mre)] = functionInterpreter_->ToStrWithPrecision(res.mre);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::mreTop8)] = functionInterpreter_->ToStrWithPrecision(res.mreTop8);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::mreTop1Permil)] = functionInterpreter_->ToStrWithPrecision(
            res.mreTop1Permil);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::mae)] = functionInterpreter_->ToStrWithPrecision(res.mae);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::maeTop8)] = functionInterpreter_->ToStrWithPrecision(res.maeTop8);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::maeTop1Permil)] = functionInterpreter_->ToStrWithPrecision(
            res.maeTop1Permil);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::aMax)] = functionInterpreter_->ToStrWithPrecision(res.aMax);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::aMin)] = functionInterpreter_->ToStrWithPrecision(res.aMin);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::aAvg)] = functionInterpreter_->ToStrWithPrecision(res.aAvg);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::aAavg)] = functionInterpreter_->ToStrWithPrecision(res.aAavg);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::aZero)] = std::to_string(res.aZero);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::aInfnan)] = std::to_string(res.aInfnan);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::bMax)] = functionInterpreter_->ToStrWithPrecision(res.bMax);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::bMin)] = functionInterpreter_->ToStrWithPrecision(res.bMin);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::bAvg)] = functionInterpreter_->ToStrWithPrecision(res.bAvg);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::bAavg)] = functionInterpreter_->ToStrWithPrecision(res.bAavg);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::bZero)] = std::to_string(res.bZero);
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::bInfnan)] = std::to_string(res.bInfnan);
        functionInterpreter_->WriteCsvRow(ProgrameInfo, functionInterpreter_->ProgrameRowNum,
                                          functionInterpreter_->execProgrameResultFile);
    }
    return result;
}

void FlowVerifier::UpdateInterpreterCache()
{
    auto& cache = Program::GetInstance().GetFunctionCache();
    std::unordered_map<FunctionHash, Function*> hashDict;
    cache.BuildHashDict(functionInterpreter_->GetEntry(), hashDict);
    functionInterpreter_->UpdateHashDict(hashDict);
}

std::string FlowVerifier::ParseErrorMsg(std::string errorMsg)
{
    std::string msg = functionInterpreter_->execDumpPassName + ", " + functionInterpreter_->execDumpFunPath + ", " +
                      functionInterpreter_->GetLoopSymbolString();
    auto pos = errorMsg.find("OpError");
    if (pos != std::string::npos) {
        return "[VERIFY:EXCEPTION:OP] " + msg + ", " + errorMsg.substr(0, pos) + errorMsg.substr(pos + 0x7);
    } else {
        return "[VERIFY:EXCEPTION:PATH] " + msg + "\n" + errorMsg;
    }
}

void FlowVerifier::WriteUserGolden(const std::vector<std::shared_ptr<LogicalTensorData>>& goldenDataViewList)
{
    std::vector<std::string> OpInfo(toIndex(OpInfoCsvHeader::COL_COUNT));
    for (size_t k = 0; k < goldenDataViewList.size(); k++) {
        std::string goldenFileName = "usergolden~" + std::to_string(k) + ".data";
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        auto ts = tv.tv_sec * 1000000 + tv.tv_usec;
        OpInfo[toIndex(OpInfoCsvHeader::outputTensor)] = goldenFileName;
        OpInfo[toIndex(OpInfoCsvHeader::inputTensors)] = goldenFileName;
        OpInfo[toIndex(OpInfoCsvHeader::passName)] = "user_golden";
        OpInfo[toIndex(OpInfoCsvHeader::timeStamp)] = std::to_string(ts) + "'";
        OpInfo[toIndex(OpInfoCsvHeader::ioflag)] = "a" + std::to_string(k);
    }
    functionInterpreter_->WriteCsvRow(OpInfo, functionInterpreter_->opInfoRowNum,
                                      functionInterpreter_->execOpResultFile);
}

void FlowVerifier::WriteException()
{
    std::vector<std::string> ProgrameInfo(toIndex(ProgrameInfoCsvHeader::COL_COUNT));
    ProgrameInfo[toIndex(ProgrameInfoCsvHeader::passName)] = functionInterpreter_->execDumpPassName;
    ProgrameInfo[toIndex(ProgrameInfoCsvHeader::pathFuncMagicName)] = functionInterpreter_->execDumpFunPath;
    ProgrameInfo[toIndex(ProgrameInfoCsvHeader::pathFuncMagic)] = std::to_string(functionInterpreter_->pathFuncMagic);
    ProgrameInfo[toIndex(ProgrameInfoCsvHeader::pathFuncHash)] = std::to_string(functionInterpreter_->pathFuncHash) +
                                                                 "'";
    ProgrameInfo[toIndex(ProgrameInfoCsvHeader::loopInfo)] = functionInterpreter_->GetLoopSymbolString();
    ProgrameInfo[toIndex(ProgrameInfoCsvHeader::verifyResult)] = "EXCEPTION";
    if (functionInterpreter_->execDumpPassName == "tensor_graph") {
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::goldenPassName)] = "user_golden";
    } else {
        ProgrameInfo[toIndex(ProgrameInfoCsvHeader::goldenPassName)] = "tensor_graph";
    }
    functionInterpreter_->WriteCsvRow(ProgrameInfo, functionInterpreter_->ProgrameRowNum,
                                      functionInterpreter_->execProgrameResultFile);
}

void FlowVerifier::VerifyTensorGraph(Function* entry,
                                     const std::vector<std::shared_ptr<LogicalTensorData>>& inputDataViewList,
                                     const std::vector<std::shared_ptr<LogicalTensorData>>& outputDataViewList,
                                     const std::vector<std::shared_ptr<LogicalTensorData>>& goldenDataViewList,
                                     const std::shared_ptr<TensorSlotManager>& slotManager)
{
    const std::vector<std::string> groupNames = Distributed::CommGroupRecorder::GetInstance().Output();
    if (!groupNames.empty()) {
        for (const std::string& groupName : groupNames) {
            SimulationCommManager::Instance().CreateSimulationCommContext(groupName, 0);
        }
    }

    entry_ = entry;
    inputDataViewList_ = inputDataViewList;
    outputDataViewList_ = outputDataViewList;
    goldenDataViewList_ = goldenDataViewList;

    ASSERT(VerifyEnableScene::VERIFY_NOT_ENABLE, calc::IsVerifyEnabled()) << "Verify not supported";
    auto attr = entry->GetDyndevAttribute();
    std::vector<int> inputSlotList = slotManager->LookupSlotIndexConst(attr->startArgsInputTensorList);
    std::vector<int> outputSlotList = slotManager->LookupSlotIndexConst(attr->startArgsOutputTensorList);

    std::unordered_map<int, TileOpFormat> slotTileOpFormatDict;
    std::unordered_map<int, std::shared_ptr<StorageData>> slotDataViewDict;
    std::unordered_map<int, std::shared_ptr<LogicalTensorData>> slotLogicalDataViewDict;
    std::unordered_set<int> outputSlotSet;

    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC, inputSlotList.size() == attr->startArgsInputTensorList.size());
    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC, inputDataViewList.size() == inputSlotList.size());
    for (size_t i = 0; i < inputDataViewList.size(); i++) {
        auto inputTensor = attr->startArgsInputTensorList[i].get().GetStorage();
        if (inputTensor == nullptr) {
            continue;
        }
        auto tileop = inputTensor->Format();

        auto input = inputDataViewList[i];
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE, inputTensor->Datatype() == input->GetDataType());
        if (tileop == TileOpFormat::TILEOP_NZ) {
            slotTileOpFormatDict[inputSlotList[i]] = TileOpFormat::TILEOP_NZ;
        }
        slotDataViewDict[inputSlotList[i]] = input->GetData()->GetRawData();
        slotLogicalDataViewDict[inputSlotList[i]] = input;
    }
    ASSERT(ControlFlowScene::INVALID_FUNC_IO_SPEC, outputDataViewList.size() == outputSlotList.size());
    for (size_t i = 0; i < outputDataViewList.size(); i++) {
        slotDataViewDict[outputSlotList[i]] = outputDataViewList[i]->GetData()->GetRawData();
        slotLogicalDataViewDict[outputSlotList[i]] = outputDataViewList[i];
        auto outputTensor = attr->startArgsOutputTensorList[i].get().GetStorage();
        auto tileop = outputTensor->Format();
        if (tileop == TileOpFormat::TILEOP_NZ) {
            slotTileOpFormatDict[outputSlotList[i]] = TileOpFormat::TILEOP_NZ;
        }
    }
    if (outputDataViewList.size() == 0) {
        outputSlotSet.insert(inputSlotList.begin(), inputSlotList.end());
    } else {
        outputSlotSet.insert(outputSlotList.begin(), outputSlotList.end());
    }

    std::unordered_map<std::string, ScalarImmediateType> controlFlowSymbolDict;
    const std::vector<std::string>& inputNameList = slotManager->GetInputNameList();
    const std::vector<std::string>& outputNameList = slotManager->GetOutputNameList();
    size_t idx = 0;
    for (size_t i = 0; i < inputNameList.size(); i++) {
        controlFlowSymbolDict[AddArgPrefix(inputNameList[i])] = idx++;
    }
    for (size_t i = 0; i < outputNameList.size(); i++) {
        controlFlowSymbolDict[AddArgPrefix(outputNameList[i])] = idx++;
    }

    std::vector<std::shared_ptr<LogicalTensorData>> inoutDataViewList = inputDataViewList_;
    inoutDataViewList.insert(inoutDataViewList.end(), outputDataViewList.begin(), outputDataViewList.end());
    functionInterpreter_ = std::make_shared<FunctionInterpreter>();
    functionInterpreter_->Initialize(entry, inoutDataViewList);
    functionInterpreter_->verifyType = VerifyType::TENSOR_GRAPH;
    functionInterpreter_->execDumpPassName = "tensor_graph";
    if (goldenDataViewList.size() != 0) {
        WriteUserGolden(goldenDataViewList);
    }
    UpdateInterpreterCache();

    if (config::GetVerifyOption<bool>(KEY_PASS_VERIFY_SAVE_TENSOR)) {
        functionInterpreter_->DumpSetLevelTensor();
    }

    auto tensorDir = config::LogTopFolder() + "/tensor";
    CreateDir(tensorDir, true);

    try {
        controlFlowExecution_ = functionInterpreter_->RunForControlFlow("tensor_graph", slotTileOpFormatDict,
                                                                        slotDataViewDict, slotLogicalDataViewDict,
                                                                        outputSlotSet, controlFlowSymbolDict);
        functionInterpreter_->DumpReset();
        bool res = true;

        std::vector<double> tolerance = config::GetVerifyOption<std::vector<double>>(KEY_PASS_VERIFY_ERROR_TOL);
        float rtol = static_cast<float>(tolerance[0]);
        float atol = static_cast<float>(tolerance[1]);
        if (outputDataViewList.size() == 0) {
            res = VerifyResult(attr->startArgsInputLogicalTensorList, {}, "tensor_graph", "tensor_graph",
                               goldenDataViewList_, inputDataViewList_, rtol, atol);
        } else {
            res = VerifyResult(attr->startArgsOutputLogicalTensorList, {}, "tensor_graph", "tensor_graph",
                               goldenDataViewList_, outputDataViewList_, rtol, atol);
        }
        if (!res) {
            checkResult = false;
        }
    } catch (std::exception& e) {
        std::string msg = e.what();
        fprintf(functionInterpreter_->execDumpErrorFile, "%s\n", ParseErrorMsg(msg).c_str());
        WriteException();
        throw std::runtime_error(e.what());
    }
}

template <typename T>
static std::string ToString(const T& val, size_t totalSize)
{
    std::string data = std::to_string(val);
    if (totalSize < data.size()) {
        return data;
    } else {
        return std::string(totalSize - data.size(), '0') + data;
    }
}

void FlowVerifier::VerifyPass(Function* func, int passIndex, const std::string& passIdentifier)
{
    functionInterpreter_->verifyType = VerifyType::PASS;
    functionInterpreter_->passIndex = passIndex;
    functionInterpreter_->execDumpPassName = "Pass_" + ToString(passIndex, 0x2) + "_" + passIdentifier;
    functionInterpreter_->execDumpFunPath = func->GetMagicName();
    functionInterpreter_->pathFuncMagic = func->GetFuncMagic();
    functionInterpreter_->pathFuncHash = func->GetFunctionHash().GetHash();
    UpdateInterpreterCache();
    if (controlFlowExecution_->executionListDict.count(func) == 0) {
        return;
    }

    std::vector<std::string> passFilter = config::GetVerifyOption<std::vector<std::string>>(KEY_PASS_VERIFY_FILTER);
    if (passFilter.empty()) {
        return;
    }

    if (std::find(passFilter.begin(), passFilter.end(), "all") == passFilter.end()) {
        auto it = std::find(passFilter.begin(), passFilter.end(), passIdentifier);
        if (it == passFilter.end()) {
            return;
        }
    }

    const std::vector<std::string> groupNames = Distributed::CommGroupRecorder::GetInstance().Output();
    if (!groupNames.empty()) {
        for (const std::string& groupName : groupNames) {
            SimulationCommManager::Instance().CreateSimulationCommContext(groupName, passIndex + 1);
        }
    }

    std::vector<double> tolerance = config::GetVerifyOption<std::vector<double>>(KEY_PASS_VERIFY_ERROR_TOL);
    ASSERT(VerifyEnableScene::TOLERANCE_MISMATCH, tolerance.size() == 0x2)
        << "Expected tolerance size: " << 0x2 << ", actual tolerance size: " << tolerance.size();
    float rtol = static_cast<float>(tolerance[0]);
    float atol = static_cast<float>(tolerance[1]);

    auto& captureList = controlFlowExecution_->executionListDict.find(func)->second;

    if (config::GetVerifyOption<bool>(KEY_PASS_VERIFY_SAVE_TENSOR)) {
        functionInterpreter_->DumpSetLevelTensor();
    }
    for (size_t captureIndex = 0; captureIndex < captureList.size(); captureIndex++) {
        const std::string key = functionInterpreter_->execDumpFunPath + "_" + functionInterpreter_->execDumpPassName;
        functionInterpreter_->captureIndex = captureIndex;

        std::shared_ptr<FunctionCaptureExecution> capture = nullptr;
        capture = captureList[captureIndex];

        std::shared_ptr<FunctionCaptureExecution> captureExecution = nullptr;
        try {
            captureExecution = functionInterpreter_->RunForPass(functionInterpreter_->execDumpPassName, func, capture);

            auto goldenDataViewList = capture->golden->outcastDataViewList;
            auto executeDataViewList = captureExecution->golden->outcastDataViewList;
            std::string tensorName = "tensor~" + func->GetMagicName() + "~" + passIdentifier + "~" +
                                     functionInterpreter_->GetLoopSymbolString(false);

            auto res = VerifyResult(func->GetOutcast(), capture->func->GetOutcast(), key, tensorName,
                                    goldenDataViewList, executeDataViewList, rtol, atol);
            if (!res) {
                checkResult = false;
            }
        } catch (std::exception& e) {
            INTERPRETER_LOGE_FULL(VerifyResultScene::VERIFY_RESULT_MISMATCH,
                                  "VerifyPass failed for function %s, pass %s (passIndex: %d, captureIndex: %zu): %s",
                                  func->GetMagicName().c_str(), passIdentifier.c_str(), passIndex, captureIndex,
                                  e.what());
            std::string msg = e.what();
            fprintf(functionInterpreter_->execDumpErrorFile, "%s\n", ParseErrorMsg(msg).c_str());
            WriteException();
            checkResult = false;
            continue;
        }
    }
    functionInterpreter_->DumpReset();
}

FlowVerifier& FlowVerifier::GetInstance()
{
    static FlowVerifier flowVerifier;
    return flowVerifier;
}
} // namespace npu::tile_fwk
