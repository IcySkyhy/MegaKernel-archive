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
 * \file function.cpp
 * \brief
 */

#include "interface/interpreter/function.h"
#include "interface/interpreter/flow_verifier.h"
#include "interface/interpreter/interpreter_log.h"
#include "interface/configs/config_manager.h"
#include "utils/file_utils.h"

#include <cstdio>
#include <vector>

namespace npu::tile_fwk {
namespace {

std::vector<std::string> MakeOpInfoCsvHeader()
{
    return {
        "NO.",
        "PHASE_NAME",
        "PATH_FUNC:func_magicname",
        "PATH_FUNC:funcmagic",
        "PATH_FUNC:hash",
        "LOOP_INFO",
        "ROOT_FUNC:functype",
        "ROOT_FUNC:graphtype",
        "ROOT_FUNC:funcmagic",
        "ROOT_FUNC:hash",
        "FUNC:functype",
        "FUNC:graphtype",
        "FUNC:funcmagic",
        "FUNC:hash",
        ":rawmagic",
        ":rawshape",
        ":datatype",
        ":format",
        ":symbol",
        ":magic",
        ":offset",
        ":shape",
        ":validshape",
        "EVAL:dynvalidshape",
        "ROOT_CALL:opmagic",
        "ROOT_CALL:rawmagic",
        ":opmagic",
        ":opcode",
        "OP_ATTR_SYM_OFFSET",
        "OP_ATTR_ATOMIC",
        "OP_IO_FLAG",
        "TIMESTAMP",
        "FILENAME",
        "INPUT:FILENAMES",
        "INPUT:validshape",
        "INPUT:rawmagic",
    };
}

std::vector<std::string> MakeProgrameInfoCsvHeader()
{
    return {
        "NO.",
        "A>PHASE_NAME",
        "B>PHASE_NAME",
        "PATH_FUNC:func_magicname",
        "PATH_FUNC:funcmagic",
        "PATH_FUNC:hash",
        "LOOP_INFO",
        "IO_FLAG",
        "A>:rawmagic",
        "B>:rawmagic",
        ":rawshape",
        ":datatype",
        ":format",
        ":symbol",
        ":shape",
        ":validshape",
        "A>TIMESTAMP",
        "A>FILENAME",
        "B>TIMESTAMP",
        "B>FILENAME",
        "AB>RESULT",
        "AB>rtol/atol",
        "AB>fail_cnt/warn_cnt/tol_cnt",
        "AB>total_cnt/zero_cnt/infnan_cnt",
        "AB>mre",
        "AB>mre_top8",
        "AB>mre_top1permil",
        "AB>mae",
        "AB>mae_top8",
        "AB>mae_top1permil",
        "A>max",
        "A>min",
        "A>avg",
        "A>aavg",
        "A>zero",
        "A>infnan",
        "B>max",
        "B>min",
        "B>avg",
        "B>aavg",
        "B>zero",
        "B>infnan",
    };
}
} // anonymous namespace

FunctionInterpreter::FunctionInterpreter()
    : interpreterSyncSimulation_(std::make_shared<InterpreterSyncSimulationState>()), interpreterThreadPool_(0x3)
{
    dumpPath = config::GetVerifyOption<std::string>(KEY_PASS_VERIFY_SAVE_TENSOR_DIR);
    if (dumpPath.empty()) {
        dumpPath = config::LogTopFolder();
    }
    dumpPath = dumpPath + "/verify/";
    CreateDir(dumpPath, true);
    interpreter::SetLogFilePath(dumpPath + "interpreter.log");
    const std::string opResultFilePath = dumpPath + "verify_graph_data_metainfo.csv";
    execOpResultFile = fopen(opResultFilePath.c_str(), "w");
    ASSERT(OpDumpScene::DUMP_OPEN_FILE_FAILED, execOpResultFile != nullptr)
        << "open file failed : " << opResultFilePath.c_str();
    const std::string programeResultFilePath = dumpPath + "verify_graph_result_brief.csv";
    execProgrameResultFile = fopen(programeResultFilePath.c_str(), "w");
    ASSERT(OpDumpScene::DUMP_OPEN_FILE_FAILED, execProgrameResultFile != nullptr)
        << "open file failed : " << programeResultFilePath.c_str();
    const std::string dumpErrorFilePath = dumpPath + "verify_graph_result_brief.log";
    execDumpErrorFile = fopen(dumpErrorFilePath.c_str(), "w");
    ASSERT(OpDumpScene::DUMP_OPEN_FILE_FAILED, execDumpErrorFile != nullptr)
        << "open file failed : " << dumpErrorFilePath.c_str();
    auto opCsvHeader = MakeOpInfoCsvHeader();
    auto programeCsvHeader = MakeProgrameInfoCsvHeader();
    WriteCsvRow(opCsvHeader, opInfoRowNum, execOpResultFile);
    WriteCsvRow(programeCsvHeader, ProgrameRowNum, execProgrameResultFile);
}

constexpr int MAX_IDENT_LEVEL = 20;
const std::unordered_set<std::string> copyOpCode = {"COPY_IN",           "COPY_OUT",           "L1_TO_L0A",
                                                    "L1_TO_L0B",         "L1_TO_L0At",         "FIX_COPY_IN_QUANT_PRE",
                                                    "L1_TO_L0Bt",        "L0C_COPY_L1",        "L1_TO_BT",
                                                    "TRANSPOSE_MOVEIN",  "TRANSPOSE_MOVEOUT",  "INDEX_OUTCAST",
                                                    "RESHAPE_COPY_IN",   "RESHAPE_COPY_OUT",   "INDEX_ADD",
                                                    "SHMEM_PUT",         "SHMEM_LOAD",         "SHMEM_SET",
                                                    "SHMEM_SIGNL",       "L1_COPY_IN_A_SCALE", "L1_COPY_IN_B_SCALE",
                                                    "L1_TO_L0A_SCALE",   "L1_TO_L0B_SCALE",    "L0C_RESHAPE_COPY_OUT",
                                                    "L1_RESHAPE_COPY_IN"};
const std::unordered_set<std::string> convertOpCode = {"L0C_COPY_UB", "CONVERT", "UB_COPY_ND2NZ", "UB_COPY_L1_ND",
                                                       "UB_COPY_L1"};

static std::string HtmlEscape(const std::string& src, bool escapeLineBreak = true)
{
    std::string ret;
    for (auto& c : src) {
        switch (c) {
            case '<':
                ret += "&lt;";
                break;
            case '>':
                ret += "&gt;";
                break;
            case '&':
                ret += "&amp;";
                break;
            case '\n':
                if (escapeLineBreak) {
                    ret += "<br/>";
                }
                ret.push_back(c);
                break;
            default:
                ret.push_back(c);
        }
    }
    return ret;
}

void FunctionInterpreter::DumpFunctionHead(Function* func)
{
    if (execDumpLevel < EXEC_DUMP_LEVEL_OPERATION) {
        return;
    }
    int indent = GetFrameSize();
    auto head = func->DumpSSATitle();
    auto raw = func->DumpSSARawTensor(indent);
    auto incast = func->DumpSSAIncast(indent);
    auto outcast = func->DumpSSAOutcast(indent);
    auto attr = func->DumpSSAAttribute(indent);
    auto symbol = DumpSymbolDict();
    if (!execDumpFile) {
        return;
    }
    fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, execDumpFuncKey.c_str());
    fprintf(execDumpFile, "<div class=\"function indent_%d\">frameIndex=%s</div>", indent, GetFrameCurrIndex().c_str());
    fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(head).c_str());
    fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(raw).c_str());
    fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(incast).c_str());
    fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(outcast).c_str());
    fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(attr).c_str());
    fprintf(execDumpFile, "<div class=\"function indent_%d\">%s</div>", indent, HtmlEscape(symbol).c_str());
}

void FunctionInterpreter::DumpOperation(Operation* op)
{
    if (execDumpLevel < EXEC_DUMP_LEVEL_OPERATION)
        return;
    int indent = GetFrameSize();
    auto dump = op->Dump();
    if (execDumpFile) {
        std::string tensorId = GetDumpTensorId(GetFrameCurr(), op);
        std::string operationId = GetDumpOperationId(GetFrameCurr(), op);
        fprintf(execDumpFile,
                "<div class=\"indent_%d\" id=\"%s\">"
                "  <div class=\"operation\" id=\"%s\">%s</div>"
                "</div>\n",
                indent, tensorId.c_str(), operationId.c_str(), HtmlEscape(dump).c_str());
    }
}

std::string FunctionInterpreter::GetDumpFilePath(const std::string& lv0, const std::string& lv1,
                                                 const std::string& filename)
{
    std::string baseDirName = lv0 + "/" + lv1;
    if (!IsPathExist(baseDirName)) {
        CreateDir(baseDirName);
    }
    return baseDirName + "/" + filename;
}

void FunctionInterpreter::DumpBinary(std::vector<int64_t>& shape, std::vector<int64_t>& stride,
                                     std::vector<int64_t>& offset, FILE* fdata, uint8_t* data, size_t dtypeSize)
{
    if (shape.size() > 1) {
        for (int64_t k = 0; k < shape[0]; k++) {
            auto newOffset = std::vector<int64_t>(offset.begin() + 1, offset.end());
            auto newStride = std::vector<int64_t>(stride.begin() + 1, stride.end());
            auto newShape = std::vector<int64_t>(shape.begin() + 1, shape.end());
            auto newData = data + offset[0] * dtypeSize * stride[0] + k * stride[0] * dtypeSize;
            DumpBinary(newShape, newStride, newOffset, fdata, newData, dtypeSize);
        }
    } else {
        size_t res = fwrite(data + offset[0] * dtypeSize, dtypeSize, shape[0], fdata);
        if (res != static_cast<size_t>(shape[0])) {
            INTERPRETER_LOGW("Write size is not equal actual size.");
        }
    }
}

void FunctionInterpreter::DumpTensorBinary(const std::shared_ptr<LogicalTensor>& tensor,
                                           const std::shared_ptr<LogicalTensorData>& dataView)
{
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR || !execDumpFile)
        return;
    std::string dumpTensorDirName = GetDumpFrameDirName();
    std::string dumpTensorFileName = GetDumpTensorFileName(tensor);
    std::string dumpTensorFilePath = GetDumpFilePath(execDumpDir, dumpTensorDirName, dumpTensorFileName);
    dataView->Save(dumpTensorFilePath);
}

void FunctionInterpreter::DumpTensorBinary(const std::shared_ptr<LogicalTensorData>& dataView,
                                           std::string dumpTensorFileName, bool isRaw)
{
    std::string dumpTensorFilePath = execDumpDir + "/" + dumpTensorFileName;
    auto rawShape = dataView->GetData()->GetShape();
    if (std::any_of(rawShape.begin(), rawShape.end(), [](const int64_t& val) { return val <= 0; })) {
        INTERPRETER_LOGW("The tensor size is not greater than 0.");
        return;
    }
    auto validShape = dataView->GetValidShape();
    auto offset = dataView->GetOffset();
    if (isRaw) {
        validShape = rawShape;
        std::fill(offset.begin(), offset.end(), 0);
    }
    if (std::any_of(validShape.begin(), validShape.end(), [](const int64_t& val) { return val <= 0; })) {
        return;
    }

    auto stride = dataView->GetData()->GetStride();
    if (offset.size() != validShape.size() || stride.size() != validShape.size()) {
        return;
    }
    FILE* fdata = fopen(dumpTensorFilePath.c_str(), "wb");
    if (fdata == nullptr) {
        INTERPRETER_LOGE(OpDumpScene::DUMP_OPEN_FILE_FAILED, "Failed to open file: %s", dumpTensorFilePath.c_str());
        return;
    }
    DumpBinary(validShape, stride, offset, fdata, dataView->GetData()->data(), BytesOf(dataView->GetDataType()));
    fclose(fdata);
}

std::shared_ptr<LogicalTensorData> FunctionInterpreter::LoadTensorBinary(const std::shared_ptr<LogicalTensor>& tensor,
                                                                         const std::string filepath)
{
    if (!IsPathExist(filepath)) {
        return nullptr;
    }
    std::vector<int64_t> shape = tensor->GetShape();
    if (std::any_of(shape.begin(), shape.end(), [](int64_t num) { return num <= 0; })) {
        return nullptr;
    }
    FILE* fdata = fopen(filepath.c_str(), "rb");
    if (fdata == nullptr) {
        INTERPRETER_LOGE(OpDumpScene::DUMP_OPEN_FILE_FAILED, "Failed to open file: %s", filepath.c_str());
        return nullptr;
    }
    auto data = std::make_shared<RawTensorData>(static_cast<DataType>(tensor->Datatype()), shape);
    if (fread(data->data(), 1, data->size(), fdata) != data->size()) {
        fclose(fdata);
        return nullptr;
    }
    auto dataView = std::make_shared<LogicalTensorData>(data, shape, shape, std::vector<int64_t>(shape.size(), 0));
    fclose(fdata);
    return dataView;
}

void FunctionInterpreter::FillOperationBasicInfo(Operation* op, FunctionFrame* frame, std::vector<std::string>& opInfo)
{
    opInfo[toIndex(OpInfoCsvHeader::rootFuncID)] = std::to_string(frame->rootFuncIndex);
    opInfo[toIndex(OpInfoCsvHeader::rootFuncHash)] = std::to_string(frame->rootFuncHash) + "'";
    opInfo[toIndex(OpInfoCsvHeader::rootFuncType)] = frame->rootFuncType;
    opInfo[toIndex(OpInfoCsvHeader::rootFuncGraphType)] = frame->rootFuncGraphType;
    opInfo[toIndex(OpInfoCsvHeader::funcID)] = std::to_string(frame->funcIndex);
    opInfo[toIndex(OpInfoCsvHeader::funcHash)] = std::to_string(frame->funcHash) + "'";
    opInfo[toIndex(OpInfoCsvHeader::funcType)] = frame->funcType;
    opInfo[toIndex(OpInfoCsvHeader::funcGraphType)] = frame->funcGraphType;
    opInfo[toIndex(OpInfoCsvHeader::passName)] = execDumpPassName;
    opInfo[toIndex(OpInfoCsvHeader::pathFuncMagicName)] = execDumpFunPath;
    opInfo[toIndex(OpInfoCsvHeader::pathFuncMagic)] = std::to_string(pathFuncMagic);
    opInfo[toIndex(OpInfoCsvHeader::pathFuncHash)] = std::to_string(pathFuncHash) + "'";
    opInfo[toIndex(OpInfoCsvHeader::loopInfo)] = GetLoopSymbolString();
    opInfo[toIndex(OpInfoCsvHeader::opCode)] = op->GetOpcodeStr();
    opInfo[toIndex(OpInfoCsvHeader::opMagic)] = std::to_string(op->GetOpMagic());
    if (frame->callop != nullptr) {
        opInfo[toIndex(OpInfoCsvHeader::callopMagic)] = std::to_string(frame->callop->GetOpMagic());
    }
}

void FunctionInterpreter::FillOperationOffsetInfo(Operation* op, FunctionFrame* frame,
                                                  const std::vector<SymbolicScalar>& linearArgList,
                                                  std::vector<std::string>& opInfo)
{
    if (convertOpCode.count(op->GetOpcodeStr())) { // convert op has no offset
        return;
    }
    auto opAttr = std::static_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
    if (opAttr) {
        if (copyOpCode.count(op->GetOpcodeStr())) {
            auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
            auto offset = copyAttr->IsCopyOut() ? copyAttr->GetToOffset() : copyAttr->GetFromOffset();
            auto offsetView = GetOperationInterpreterForThisThread().EvaluateOpImmediate(frame, offset);
            opInfo[toIndex(OpInfoCsvHeader::attrOffset)] = ShapeToString(offsetView);
        } else {
            Offset offsetView = EvaluateOffset(opAttr->GetFromOffset(), opAttr->GetFromDynOffset(), linearArgList);
            opInfo[toIndex(OpInfoCsvHeader::attrOffset)] = ShapeToString(offsetView);
        }
    }
    if (op->HasAttribute(OP_ATTR_PREFIX + "atomic_add")) {
        opInfo[toIndex(OpInfoCsvHeader::attrAtomic)] = "True";
    } else {
        opInfo[toIndex(OpInfoCsvHeader::attrAtomic)] = "False";
    }
}

void FunctionInterpreter::FillOperationInputInfo(
    Operation* op, FunctionFrame* frame, const std::vector<std::shared_ptr<LogicalTensorData>>* ioperandDataViewList,
    std::vector<std::string>& opInfo)
{
    auto iopSize = op->GetIOperands().size();
    for (size_t k = 0; k < iopSize; k++) {
        if (k >= ioperandDataViewList->size()) {
            return;
        }
        auto dataView = ioperandDataViewList->at(k);
        if (k > 0) {
            opInfo[toIndex(OpInfoCsvHeader::inputTensors)] += ";";
            opInfo[toIndex(OpInfoCsvHeader::inputValidShape)] += ";";
            opInfo[toIndex(OpInfoCsvHeader::inputRawMagic)] += ";";
        }
        opInfo[toIndex(OpInfoCsvHeader::inputValidShape)] += ShapeToString(dataView->GetValidShape());
        opInfo[toIndex(OpInfoCsvHeader::inputRawMagic)] += std::to_string(
            op->GetIOperands()[k]->GetRawTensor()->GetRawMagic());
        auto it = frame->tensorDataBinDict.find(op->GetIOperands()[k]);
        if (it != frame->tensorDataBinDict.end()) {
            opInfo[toIndex(OpInfoCsvHeader::inputTensors)] += it->second;
        }
        if (op->GetOpcode() == Opcode::OP_COPY_IN) {
            auto itTmp = frame->callopDataViewTensorDict.find(dataView);
            if (itTmp != frame->callopDataViewTensorDict.end()) {
                opInfo[toIndex(OpInfoCsvHeader::callopRawMagic)] = std::to_string(
                    itTmp->second->GetRawTensor()->GetRawMagic());
            }
        }
    }
}

void FunctionInterpreter::FillOperationOutputInfo(
    Operation* op, FunctionFrame* frame, const std::vector<std::shared_ptr<LogicalTensorData>>* ooperandDataViewList,
    const std::vector<SymbolicScalar>& linearArgList, int indent, std::vector<std::string>& opInfo)
{
    auto oopSize = op->GetOOperands().size();
    for (size_t k = 0; k < oopSize; k++) {
        if (k < ooperandDataViewList->size()) {
            auto dataView = ooperandDataViewList->at(k);
            std::string dumpTensorFileName = GetDumpTensorFileName(op->GetOOperands()[k], op, frame);
            bool isRaw = (op->GetOpcode() == Opcode::OP_COPY_OUT || IsAssembleLike(op->GetOpcode()));
            DumpTensorBinary(dataView, dumpTensorFileName, isRaw);
            if (execDumpFile) {
                fprintf(execDumpFile, "<div class=\"detail indent_%d\">%s</a></div>\n", indent,
                        dumpTensorFileName.c_str());
            }
            struct timeval tv;
            gettimeofday(&tv, nullptr);
            auto ts = tv.tv_sec * 1000000 + tv.tv_usec;

            frame->tensorDataBinDict[op->GetOOperands()[k]] = dumpTensorFileName;
            opInfo[toIndex(OpInfoCsvHeader::tensorMagic)] = std::to_string(op->GetOOperands()[k]->GetMagic());
            opInfo[toIndex(OpInfoCsvHeader::rawTensorMagic)] = std::to_string(
                op->GetOOperands()[k]->GetRawTensor()->GetRawMagic());
            opInfo[toIndex(OpInfoCsvHeader::outputShape)] = ShapeToString(dataView->GetShape());
            opInfo[toIndex(OpInfoCsvHeader::outputRawShape)] = ShapeToString(dataView->GetData()->GetShape());
            opInfo[toIndex(OpInfoCsvHeader::outputValidShape)] = ShapeToString(dataView->GetValidShape());
            opInfo[toIndex(OpInfoCsvHeader::outputDynValidShape)] = ShapeToString(
                EvaluateValidShape((op->GetOOperands()[k]->GetDynValidShape()), linearArgList));
            opInfo[toIndex(OpInfoCsvHeader::outputDtype)] = DataType2String(dataView->GetDataType(), true);
            opInfo[toIndex(OpInfoCsvHeader::tensorOffset)] = ShapeToString(dataView->GetOffset());
            opInfo[toIndex(OpInfoCsvHeader::outputTensor)] = dumpTensorFileName;
            opInfo[toIndex(OpInfoCsvHeader::timeStamp)] = std::to_string(ts) + "'";
            opInfo[toIndex(OpInfoCsvHeader::outputSymbol)] = op->GetOOperands()[k]->GetRawTensor()->GetSymbol();
            opInfo[toIndex(OpInfoCsvHeader::outputFormat)] = std::to_string(
                op->GetOOperands()[k]->GetRawTensor()->format);
            opInfo[toIndex(OpInfoCsvHeader::ioflag)] = "o" + std::to_string(k);

            if (op->GetOpcode() == Opcode::OP_COPY_OUT) {
                auto itTmp = frame->callopDataViewTensorDict.find(dataView);
                if (itTmp != frame->callopDataViewTensorDict.end()) {
                    opInfo[toIndex(OpInfoCsvHeader::callopRawMagic)] = std::to_string(
                        itTmp->second->GetRawTensor()->GetRawMagic());
                }
            }
        }
        WriteCsvRow(opInfo, opInfoRowNum, execOpResultFile);
    }
}

void FunctionInterpreter::DumpOperationTensor(
    Operation* op, FunctionFrame* frame, const std::vector<std::shared_ptr<LogicalTensorData>>* ooperandDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>>* ioperandDataViewList)
{
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR || !execDumpFile)
        return;

    int indent = GetFrameSize();
    std::vector<SymbolicScalar> linearArgList;
    if (frame->callopAttr != nullptr) {
        linearArgList = frame->callopAttr->GetLinearArgList();
    }

    std::vector<std::string> opInfo(toIndex(OpInfoCsvHeader::COL_COUNT));
    FillOperationBasicInfo(op, frame, opInfo);
    FillOperationOffsetInfo(op, frame, linearArgList, opInfo);
    FillOperationInputInfo(op, frame, ioperandDataViewList, opInfo);
    FillOperationOutputInfo(op, frame, ooperandDataViewList, linearArgList, indent, opInfo);
}

void FunctionInterpreter::DumpPassTensorDiff(const std::shared_ptr<FunctionCaptureExecution>& captureExecution,
                                             const std::shared_ptr<FunctionCaptureExecution>& captureGolden)
{
    if (execDumpLevel < EXEC_DUMP_LEVEL_TENSOR)
        return;
    if (captureExecution->GetFrameList().size() != captureGolden->GetFrameList().size())
        return;

    std::vector<double> tolerance = config::GetVerifyOption<std::vector<double>>(KEY_PASS_VERIFY_ERROR_TOL);
    float rtol = static_cast<float>(tolerance[0]);
    float atol = static_cast<float>(tolerance[1]);

    std::string dumpStyleFilePath = execDumpDir + "/entry_" + std::to_string(captureIndex) + ".css";
    execDumpStyleFile = fopen(dumpStyleFilePath.c_str(), "w");
    if (execDumpStyleFile == nullptr) {
        INTERPRETER_LOGE(OpDumpScene::DUMP_OPEN_FILE_FAILED, "Failed to open file: %s", dumpStyleFilePath.c_str());
        return;
    }
    for (size_t idx = 0; idx < captureGolden->GetFrameList().size(); idx++) {
        auto frameExecution = captureExecution->GetFrameList()[idx];
        auto frameGolden = captureGolden->GetFrameList()[idx];
        const auto& tensorDictExecution = frameExecution->tensorDataBinDict;
        const auto& tensorDictGolden = frameGolden->tensorDataBinDict;
        std::vector<std::shared_ptr<LogicalTensor>> tensorList;
        for (auto& [tensor, dataViewExecution] : tensorDictExecution) {
            (void)dataViewExecution;
            if (tensorDictGolden.count(tensor)) {
                tensorList.push_back(tensor);
            }
        }
        for (size_t k = 0; k < tensorList.size(); k++) {
            auto tensor = tensorList[k];
            auto executionFileName = tensorDictExecution.find(tensor)->second;
            auto dataViewExecution = LoadTensorBinary(tensor, execDumpDir + "/" + executionFileName);
            auto filename = tensorDictGolden.find(tensor)->second;
            auto dataViewGolden = LoadTensorBinary(tensor, dumpPath + "/tensor_graph/" + filename);
            if (!dataViewExecution || !dataViewGolden) {
                continue;
            }
            auto compare = FlowVerifier::VerifyResult(dataViewGolden, dataViewExecution, rtol, atol);
            std::string tensorId = GetDumpTensorId(frameExecution, tensor);
            if (compare.Check()) {
                fprintf(execDumpStyleFile, "#%s { background-color: LightGreen; }\n", tensorId.c_str());
            } else {
                fprintf(execDumpStyleFile, "#%s { background-color: LightCoral; }\n", tensorId.c_str());
            }
        }
    }
    fclose(execDumpStyleFile);
}

void FunctionInterpreter::DumpBegin()
{
    frameCount = 0;
    execDumpDir = dumpPath + execDumpFuncKey;
    CreateDir(execDumpDir, true);

    if (execDumpLevel < EXEC_DUMP_LEVEL_OPERATION)
        return;

    std::string styleFilePath = dumpPath + "/verifier.css";
    if (GetFileSize(styleFilePath) == 0) {
        FILE* fcss = fopen(styleFilePath.c_str(), "w");
        if (fcss == nullptr) {
            INTERPRETER_LOGE(OpDumpScene::DUMP_OPEN_FILE_FAILED, "Failed to open file: %s", styleFilePath.c_str());
        } else {
            for (int i = 0; i < MAX_IDENT_LEVEL; i++) {
                fprintf(fcss, ".indent_%d { margin-left: %dpx; }\n", i, i * 50); // 50 is left margin
            }
            fprintf(fcss, ".table_head { width: 10%%; }\n");
            fprintf(fcss, ".tensor_data { vertical-align: top; }\n");
            fclose(fcss);
        }
    }

    std::string dumpFilePath = execDumpDir + "/entry_" + std::to_string(captureIndex) + ".html";
    execDumpFile = fopen(dumpFilePath.c_str(), "w");
    if (execDumpFile == nullptr) {
        INTERPRETER_LOGE(OpDumpScene::DUMP_OPEN_FILE_FAILED, "Failed to open file: %s", dumpFilePath.c_str());
        return;
    }
    fprintf(execDumpFile, R"HTML(
<html>
    <head>
    <link rel="stylesheet" type="text/css" href="../verifier.css">
    <link rel="stylesheet" type="text/css" href="entry_%s.css">
    </head>
    <body>
)HTML",
            std::to_string(captureIndex).c_str());
}

void FunctionInterpreter::DumpEnd()
{
    if (execDumpLevel < EXEC_DUMP_LEVEL_OPERATION || execDumpFile == nullptr)
        return;
    fprintf(execDumpFile, R"HTML(
    </body>
</html>
)HTML");
    fclose(execDumpFile);
}

} // namespace npu::tile_fwk
