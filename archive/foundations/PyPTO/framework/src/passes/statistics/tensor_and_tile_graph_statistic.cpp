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
 * \file tensor_and_tile_graph_statistic.cpp
 * \brief
 */

#include "tensor_and_tile_graph_statistic.h"

#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <queue>
#include <climits>
#include <nlohmann/json.hpp>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "utils/file_utils.h"
#include "interface/tensor/hypercube_overlap_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/graph_utils.h"

#define MODULE_NAME "TensorAndTileGraphStatistic"

namespace npu {
namespace tile_fwk {
using json = nlohmann::json;

template <typename tType>
uint64_t CalcTensorSize(const std::vector<tType>& curShape)
{
    uint64_t res = 1;
    for (auto& dim : curShape) {
        res *= dim;
    }
    return res;
}

void CalcOperatorInfo(Function& function, json& report)
{
    MetricData memoryMetric;

    auto operationViewer = function.Operations();
    report["totalOpCount"] = operationViewer.size();

    // 非静态场景下 无法计算size 直接返回
    if (function.GetFunctionType() != FunctionType::STATIC) {
        APASS_LOG_INFO_F(Elements::Function,
                         "PeakMemory can't be calculated; because functiontype is not static, name %s, hash %lu.",
                         function.GetMagicName().c_str(), function.GetFunctionHash().GetHash());
        return;
    }
    uint64_t totalCopySize = 0;
    for (size_t i = 0; i < operationViewer.size(); i++) {
        auto opCode = operationViewer[i].GetOpcode();
        int opMagic = operationViewer[i].GetOpMagic();
        uint64_t iShapeSize = 0;
        for (auto& inOperand : operationViewer[i].GetIOperands()) {
            iShapeSize += CalcTensorSize(inOperand->GetShape());
        }
        if (IsCopyOut(opCode)) {
            totalCopySize += iShapeSize;
        }
        uint64_t oShapeSize = 0;
        for (auto& outOperand : operationViewer[i].GetOOperands()) {
            oShapeSize += CalcTensorSize(outOperand->GetShape());
        }
        if (IsCopyIn(opCode)) {
            totalCopySize += oShapeSize;
        }
        memoryMetric.UpdateMetricData(iShapeSize + oShapeSize, opMagic);
    }

    // 写出memoryMetric和copyMetric
    report["peakMemory"] = {{"peakMemoryUsage", memoryMetric.GetMaxSize()},
                            {"peakMemoryUsageOps", *memoryMetric.GetMaxNodes()}};
    report["copyDataCount"] = totalCopySize;
}

void CalcTensorInfo(Function& function, json& report)
{
    MetricData consumerMetric;
    MetricData producerMetric;
    TensorSet allTensors = GraphUtils::GetAllTensors(function);
    for (auto tensor : allTensors) {
        int tensorMagic = tensor->GetMagic();
        uint64_t consumerSize = static_cast<uint64_t>(tensor->GetConsumers().size());
        consumerMetric.UpdateMetricData(consumerSize, tensorMagic);
        uint64_t producerSize = static_cast<uint64_t>(tensor->GetProducers().size());
        producerMetric.UpdateMetricData(producerSize, tensorMagic);
    }

    // 写出consumerMetric和producerMetric
    report["totalTensorCount"] = allTensors.size();
    report["maxConsumerCount"] = consumerMetric.GetMaxSize();
    report["maxConsumerTensors"] = *consumerMetric.GetMaxNodes();
    report["maxproducerCount"] = producerMetric.GetMaxSize();
    report["maxproducerTensors"] = *producerMetric.GetMaxNodes();
}

void GetOpConnectionMap(Function& function, std::vector<std::vector<int>>& inMap, std::vector<std::vector<int>>& outMap,
                        std::vector<bool>& actualMagic)
{
    // 找到最大的magic编号
    MetricData magicNum;
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        magicNum.UpdateMetricData(static_cast<uint64_t>(operationViewer[i].GetOpMagic()), static_cast<int>(i));
    }
    size_t magicUpperRange = static_cast<size_t>(magicNum.GetMaxSize()) + static_cast<size_t>(1);

    // 生成inMap和outMap
    inMap.resize(magicUpperRange);
    outMap.resize(magicUpperRange);
    actualMagic.resize(magicUpperRange);
    std::fill(actualMagic.begin(), actualMagic.end(), false);
    for (size_t i = 0; i < operationViewer.size(); i++) {
        int childMagic = operationViewer[i].GetOpMagic();
        actualMagic[childMagic] = true;
        for (auto& input : operationViewer[i].GetIOperands()) {
            for (auto& parentOpPtr : input->GetProducers()) {
                int parentMagic = parentOpPtr->GetOpMagic();
                auto it = std::find(inMap[childMagic].begin(), inMap[childMagic].end(), parentMagic);
                if (it == inMap[childMagic].end()) {
                    inMap[childMagic].push_back(parentMagic);
                    outMap[parentMagic].push_back(childMagic);
                }
            }
        }
    }
}

void TraversePathUp(const int parent, const std::vector<std::vector<int>>& outMap, std::vector<int>& layerMap)
{
    int parentLayer = layerMap[parent];
    for (auto child : outMap[parent]) {
        if (layerMap[child] <= parentLayer) {
            layerMap[child] = parentLayer + 1;
            TraversePathUp(child, outMap, layerMap);
        }
    }
}

void CalcGraphMetrics(const std::vector<std::vector<int>>& inMap, const std::vector<std::vector<int>>& outMap,
                      const std::vector<bool>& actualVertex, json& report)
{
    MetricData inDegreeMetric;
    MetricData outDegreeMetric;
    std::vector<int> layerMap = std::vector<int>(inMap.size(), 0);

    // 计算每层节点层数，inDegree和outDegree
    for (size_t i = 0; i < inMap.size(); i++) {
        if (actualVertex[i]) {
            inDegreeMetric.UpdateMetricData(static_cast<uint64_t>(inMap[i].size()), static_cast<int>(i));
            outDegreeMetric.UpdateMetricData(static_cast<uint64_t>(outMap[i].size()), static_cast<int>(i));
            if (inMap[i].size() == 0) {
                TraversePathUp(static_cast<int>(i), outMap, layerMap);
            }
        }
    }

    // 找到最大层数，最大层数等于最长路径的长度
    int maxLayerNum = *std::max_element(layerMap.begin(), layerMap.end()) + 1;

    // 找到最宽层的节点数
    std::vector<int> layerCount = std::vector<int>(maxLayerNum, 0);
    for (size_t i = 0; i < inMap.size(); i++) {
        if (actualVertex[i]) {
            layerCount[layerMap[i]]++;
        }
    }
    int maxLayerWidth = *std::max_element(layerCount.begin(), layerCount.end());

    // 写出inDegreeMetric, outDegreeMetric, maxLayerNum, maxLayerWidth
    report["maxFanin"] = inDegreeMetric.GetMaxSize();
    report["maxFaninOps"] = *inDegreeMetric.GetMaxNodes();
    report["maxFanout"] = outDegreeMetric.GetMaxSize();
    report["maxFanoutOps"] = *outDegreeMetric.GetMaxNodes();
    report["maxDepth"] = maxLayerNum;
    report["maxWidth"] = maxLayerWidth;
}

void CalShapeInt(Operation* copyin, std::vector<OpImmediate>& shape, std::vector<int>& shapeInt, bool& continueFlag)
{
    for (auto s : shape) {
        SymbolicScalar& value = s.GetSpecifiedValue();
        if (value.Raw()->Kind() != SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE) {
            APASS_LOG_WARN_F(Elements::Operation, "%d COPYIN Shape is dynamic, CalShapeInt not support!",
                             copyin->GetOpMagic());
            continueFlag = true;
            return;
        }
        RawSymbolicImmediate* immediate = dynamic_cast<RawSymbolicImmediate*>(value.Raw().get());
        shapeInt.emplace_back(static_cast<int>(immediate->Immediate()));
    }
}

void CalOffsetInt(Operation* copyin, std::vector<OpImmediate>& offset, std::vector<int>& offsetInt, bool& continueFlag)
{
    for (auto o : offset) {
        SymbolicScalar& value = o.GetSpecifiedValue();
        if (value.Raw()->Kind() != SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE) {
            APASS_LOG_WARN_F(Elements::Operation, "%d COPYIN Offset is dynamic, CalOffsetInt not support!",
                             copyin->GetOpMagic());
            continueFlag = true;
            return;
        }
        RawSymbolicImmediate* immediate = dynamic_cast<RawSymbolicImmediate*>(value.Raw().get());
        offsetInt.emplace_back(static_cast<int>(immediate->Immediate()));
    }
}

std::unordered_map<std::string, int> redundantCopyMemoryMap = {
    {"MEM_UB", 0}, {"MEM_L1", 0}, {"MEM_L0A", 0}, {"MEM_L0B", 0}, {"MEM_L0C", 0}};

Status CalRedundantCopy(Function& function, json& report)
{
    for (auto tensor : GraphUtils::GetAllTensors(function)) {
        auto dataSize = BytesOf(tensor->Datatype());
        std::unordered_map<MemoryType, HypercubeOverlapChecker<Operation*>> overlapChecker;
        for (auto& consumer : tensor->GetConsumers()) {
            if (consumer->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
                if (consumer->GetOpAttribute() == nullptr) {
                    APASS_LOG_ERROR_F(Elements::Operation, "%d COPYIN op attr is nullptr, CalRedundantCopy failed!",
                                      consumer->GetOpMagic());
                    return FAILED;
                }
                std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(
                    consumer->GetOpAttribute());
                auto shape = attr->GetShape();
                auto offset = attr->GetCopyInAttr().first;
                auto dstMemType = attr->GetCopyInAttr().second;
                std::vector<int> shapeInt;
                bool continueFlag = false;
                CalShapeInt(consumer, shape, shapeInt, continueFlag);
                if (continueFlag) {
                    continue;
                }
                std::vector<int> offsetInt;
                CalOffsetInt(consumer, offset, offsetInt, continueFlag);
                if (continueFlag) {
                    continue;
                }
                std::vector<int> hypercube;
                for (size_t i = 0; i < shapeInt.size(); i++) {
                    auto min = offsetInt[i];
                    auto max = offsetInt[i] + shapeInt[i];
                    hypercube.emplace_back(min);
                    hypercube.emplace_back(max);
                }
                int64_t* overlapPtr = new int64_t(0);
                overlapChecker[dstMemType].Find(hypercube, overlapPtr);
                redundantCopyMemoryMap[MemoryTypeToString(dstMemType)] += *overlapPtr * dataSize;
                delete overlapPtr;
                overlapChecker[dstMemType].Insert(hypercube, consumer);
            }
        }
    }
    report["redundantCopyCount(Bytes)"] = redundantCopyMemoryMap;
    return SUCCESS;
}

void WriteHealthReport(const json& report, const std::string& reportPath, const std::string& filename)
{
    if (!CreateDir(reportPath, true)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to create directory for health report");
    }
    std::ofstream out(reportPath + "/" + filename);
    if (!out.is_open()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to open health report file for writing");
        return;
    }
    out << report.dump(DUMP_WIDTH);
    out.close();
}

Status CheckTileShapeAIV(Operation* op, std::vector<int>& res)
{
    auto tileSize = op->GetTileShape().GetVecTile().tile;
    for (auto input : op->GetIOperands()) {
        auto tensorShape = input->GetShape();
        if (tileSize.size() != tensorShape.size()) {
            APASS_LOG_WARN_F(
                Elements::Tensor,
                "%d Tensorshape size %zu is not equal to %d %s tileshape size %zu, CheckTileShapeAIV failed!",
                input->GetMagic(), tensorShape.size(), op->GetOpMagic(), op->GetOpcodeStr().c_str(), tileSize.size());
            return FAILED;
        }
        bool devisible = true;
        for (size_t i = 0; i < tensorShape.size(); i++) {
            if (tensorShape[i] % tileSize[i] != 0) {
                devisible = false;
                break;
            }
        }
        if (!devisible) {
            res.emplace_back(op->GetOpMagic());
        }
    }
    return SUCCESS;
}

Status CheckTileShapeAIC(Operation* op, std::vector<int>& res)
{
    auto tileSize = op->GetTileShape().GetCubeTile();
    auto mL1 = tileSize.m[1];
    auto kL1A = tileSize.k[1];
    auto kL1B = tileSize.k[2];
    auto nL1 = tileSize.n[1];
    auto mL0 = tileSize.m[0];
    auto kL0 = tileSize.k[0];
    auto nL0 = tileSize.n[0];
    if (op->GetIOperands().size() != CUDE_IOPERAND_NUM2 && op->GetIOperands().size() != CUDE_IOPERAND_NUM3) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Cube operation %d %s ioperands size is %zu, should be 2 or 3, CheckTileShapeAIC failed!",
                         op->GetOpMagic(), op->GetOpcodeStr().c_str(), op->GetIOperands().size());
        return FAILED;
    }
    auto TensorA = op->GetIOperands()[0];
    auto TensorB = op->GetIOperands()[1];
    if (TensorA->GetShape().size() != NUM2 || TensorB->GetShape().size() != NUM2) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Cube operation %d %s input tensor shape size is not 2, CheckTileShapeAIC failed!",
                         op->GetOpMagic(), op->GetOpcodeStr().c_str());
        return FAILED;
    }
    bool L1A = false;
    if (TensorA->GetShape()[0] % mL1 == 0 && TensorA->GetShape()[1] % kL1A == 0) {
        L1A = true;
    }
    bool L1B = false;
    if (TensorB->GetShape()[0] % kL1B == 0 && TensorB->GetShape()[1] % nL1 == 0) {
        L1B = true;
    }
    bool L0A = false;
    if (L1A && (mL1 % mL0 == 0) && (kL1A % kL0 == 0)) {
        L0A = true;
    }
    bool L0B = false;
    if (L1B && (kL1B % kL0 == 0) && (nL1 % nL0 == 0)) {
        L0B = true;
    }
    if (L1A && L1B && L0A && L0B) {
        res.emplace_back(op->GetOpMagic());
    }
    return SUCCESS;
}

void FindShapeNotdevisibleOp(Function& function, json& report)
{
    std::vector<int> res;
    for (auto op : function.Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_VIEW || op->GetOpcode() == Opcode::OP_ASSEMBLE ||
            op->GetOpcode() == Opcode::OP_RESHAPE) {
            continue;
        }
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op->GetOpcode());
        if (opcfg.coreType_ == CoreType::AIV) {
            if (CheckTileShapeAIV(op, res) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "FindShapeNotdevisibleOp faild at function CheckTileShapeAIV!");
                return;
            }
        } else if (opcfg.coreType_ == CoreType::AIC) {
            if (CheckTileShapeAIC(op, res) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "FindShapeNotdevisibleOp faild at function CheckTileShapeAIC!");
                return;
            }
        }
    }
    report["tileShapeNotDevisibleOp"] = res;
}

void HealthCheckTensorGraph(Function& function, const std::string& reportPath, const std::string& fileName)
{
    json tensorGraphReport;

    // 1. 计算operation节点信息
    tensorGraphReport["totalOpCount"] = function.Operations().size();

    // 2. 计算tensor节点信息
    CalcTensorInfo(function, tensorGraphReport);

    // 3. 构建operation节点图
    std::vector<std::vector<int>> inMap;
    std::vector<std::vector<int>> outMap;
    std::vector<bool> actualMagic;
    GetOpConnectionMap(function, inMap, outMap, actualMagic);
    if (inMap.size() == 0) {
        return;
    }

    // 4. 计算图信息
    CalcGraphMetrics(inMap, outMap, actualMagic, tensorGraphReport);

    // 5. 不整除shape统计
    FindShapeNotdevisibleOp(function, tensorGraphReport);

    // 6. 写出健康报告
    std::string graphName = fileName + "_TensorGraphHealthReport.json";
    WriteHealthReport(tensorGraphReport, reportPath, graphName);
}

void HealthCheckTileGraph(Function& function, const std::string& reportPath, const std::string& fileName)
{
    json tileGraphReport;

    // 1. 计算operation节点信息
    CalcOperatorInfo(function, tileGraphReport);

    // 2. 计算tensor节点信息
    CalcTensorInfo(function, tileGraphReport);

    // 3. 构建operation节点图
    std::vector<std::vector<int>> inMap; // magic到magic的映射，in - parent, out - child
    std::vector<std::vector<int>> outMap;
    std::vector<bool> actualMagic;
    GetOpConnectionMap(function, inMap, outMap, actualMagic);
    if (inMap.size() == 0) {
        return;
    }

    // 4. 计算图信息
    CalcGraphMetrics(inMap, outMap, actualMagic, tileGraphReport);

    // 5. 冗余搬运统计
    if (CalRedundantCopy(function, tileGraphReport) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "HealthCheckTileGraph failed at function CalRedundantCopy!");
    }

    // 6. 写出健康报告
    std::string graphName = fileName + "_TileGraphHealthReport.json";
    WriteHealthReport(tileGraphReport, reportPath, graphName);
}

} // namespace tile_fwk
} // namespace npu
