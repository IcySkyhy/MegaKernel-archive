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
 * \file ParseInput.cpp
 * \brief
 */

#include "cost_model/simulation/tools/ParseInput.h"

#include <vector>

#include "nlohmann/json.hpp"
#include "cost_model/simulation/tools/visualizer.h"
#include "cost_model/simulation/base/ModelTop.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/pypto_fwk_log.h"

using namespace std;

namespace CostModel {

using Json = nlohmann::json;

void ParseInput::ParseJson(std::shared_ptr<CostModel::SimSys> sim, const std::string& jsonPath)
{
    std::ifstream input(jsonPath);
    if (!input.is_open()) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_OPEN_FAILED, "Error: fail to open file: %s",
                        jsonPath.c_str());
        return;
    }
    Json j;
    input >> j;

    sim->enableExpectValue = false;

    // Get Function From Json Input
    const auto& functions = j.at("functions");
    bool foundStartFunc = false;
    for (const auto& function : functions) {
        std::unordered_map<int, int> tensorMagicIdMap;
        tensorMagicIdMap.clear();
        FunctionPtr func = std::make_shared<Function>();
        func->functionHash = std::stoull(function.at("hash").get<std::string>());
        func->magic = function.at("magic");
        func->funcName = function.at("magicname");
        if (!foundStartFunc) {
            size_t pos = func->funcName.find(sim->config.startFunctionLabel);
            if (pos != std::string::npos) {
                foundStartFunc = true;
                sim->startFuncName = func->funcName;
                sim->startFuncHash = func->functionHash;
            }
        }
        bool isCube = false;
        const auto& operations = function.at("operations");
        for (const auto& op : operations) {
            if (op.at("opcode") == "NOP") {
                continue;
            }
            TileOpPtr tileOp = std::make_shared<TileOp>();
            tileOp->funcPtr = func;
            const auto& iOperand = op.at("ioperands");
            for (const auto& in : iOperand) {
                int magic = in.at("magic");
                auto it = tensorMagicIdMap.find(magic);
                if (it != tensorMagicIdMap.end()) {
                    func->tiles[it->second]->consumers.emplace_back(tileOp);
                    tileOp->iOperand.emplace_back(func->tiles[it->second]);
                    continue;
                }
                tensorMagicIdMap[magic] = int(func->tiles.size());
                TilePtr tensor = std::make_shared<Tile>(in.dump());
                tensor->funcPtr = func;
                tensor->consumers.emplace_back(tileOp);
                func->tiles.emplace_back(tensor);
                tileOp->iOperand.emplace_back(tensor);
                if (tensor->nodeType == NodeType::INCAST) {
                    func->incastMagic.emplace_back(magic);
                }
                func->tileMap[tensor->magic] = tensor;
            }
            const auto& output = op.at("ooperands");
            for (const auto& out : output) {
                int magic = out["magic"];
                auto it = tensorMagicIdMap.find(magic);
                if (it != tensorMagicIdMap.end()) {
                    func->tiles[it->second]->producer = tileOp;
                    func->tiles[it->second]->producers.emplace_back(tileOp);
                    tileOp->oOperand.emplace_back(func->tiles[it->second]);
                    continue;
                }
                tensorMagicIdMap[magic] = int(func->tiles.size());
                TilePtr tensor = std::make_shared<Tile>(out.dump());
                tensor->funcPtr = func;
                tensor->producer = tileOp;
                tensor->producers.emplace_back(tileOp);
                func->tiles.emplace_back(tensor);
                tileOp->oOperand.emplace_back(tensor);
                if (tensor->nodeType == NodeType::OUTCAST) {
                    func->outcastMagic.emplace_back(magic);
                }
                func->tileMap[tensor->magic] = tensor;
            }
            tileOp->opcode = op.at("opcode");
            tileOp->magic = op.at("opmagic");
            tileOp->bufType = OperandType::BUF_UB;
            if (tileOp->IsCall()) {
                tileOp->calleeHash = std::stoull(op.at("calleehash").get<std::string>());
            }
            tileOp->GetPipeType();
            if (tileOp->pipeType == CorePipeType::PIPE_CUBE) {
                isCube = true;
            }
            func->tileOps.emplace_back(tileOp);
        }

        if (isCube) {
            func->machineType = MachineType::AIC;
            sim->stats->totalFunctionCube++;
        } else {
            func->machineType = MachineType::AIV;
            sim->stats->totalFunctionVec++;
        }
        if (sim->drawGraph) {
            ModelVisualizer visualizer;
            visualizer.DrawFunction(func, sim->graphsOutdir);
        }
        sim->stats->totalFunctionNum++;
        sim->stats->totalFunctionTileOps += func->tileOps.size();
        sim->functionCache.Insert(func);
    }
}

bool ParseInput::FilterOpcode(std::string& opcode)
{
    if (opcode == "NOP") {
        return true;
    }

    auto allocQuery = opcode.find("ALLOC");
    if (allocQuery != std::string::npos) {
        return true;
    }

    auto phaseQuery = opcode.find("PHASE");
    if (phaseQuery != std::string::npos) {
        return true;
    }

    auto syncQuery = opcode.find("SYNC");
    if (syncQuery != std::string::npos) {
        return true;
    }

    auto barQuery = opcode.find("BAR");
    if (barQuery != std::string::npos) {
        return true;
    }

    return false;
}

void ParseInput::BuildTile(npu::tile_fwk::Function& func, std::shared_ptr<npu::tile_fwk::LogicalTensor> logicalTensor,
                           TilePtr tile)
{
    tile->magic = logicalTensor->magic;
    for (auto& s : logicalTensor->shape) {
        tile->shape.emplace_back(s);
    }
    for (auto& o : logicalTensor->offset) {
        tile->offset.emplace_back(o);
    }
    tile->bufferType = npu::tile_fwk::MemoryTypeToString(logicalTensor->GetMemoryTypeOriginal());
    tile->bufType = CostModel::BufferNameToType(tile->bufferType);

    tile->GetPipeType();

    tile->symbol = logicalTensor->tensor->symbol;

    tile->dataTypeStr = npu::tile_fwk::DataType2String(logicalTensor->tensor->datatype);
    tile->dataType = CostModel::ToDataType(tile->dataTypeStr);
    std::string type = npu::tile_fwk::NodeType2String(FunctionUtils::GetNodeType(*logicalTensor, func));
    tile->nodeType = CostModel::ToNodeType(type);

    tile->rawMagic = logicalTensor->tensor->rawmagic;
    for (auto& value : logicalTensor->tensor->rawshape) {
        tile->rawShape.emplace_back(value);
    }
}

void ParseInput::BuildFunctionInvoke(FunctionPtr root, std::shared_ptr<CostModel::SimSys> sim)
{
    auto cache = sim->functionCache.cache;
    int esgId = 0;
    for (auto& op : root->tileOps) {
        if (op->IsCall()) {
            auto& callee = cache[op->calleeHash];

            // Incast
            const auto& incast1 = op->iOperand;
            const auto& incast2 = callee->incastMagic;
            for (size_t i = 0; i < incast1.size(); i++) {
                auto& t1 = incast1[i];
                auto& t2 = incast2[i];
                callee->invoke[esgId].binds[t2] = t1;
            }

            // Outcast
            const auto& outcast1 = op->oOperand;
            const auto& outcast2 = callee->outcastMagic;
            for (size_t i = 0; i < outcast1.size(); i++) {
                auto& t1 = outcast1[i];
                auto& t2 = outcast2[i];
                callee->invoke[esgId].binds[t2] = t1;
            }
            esgId++;
        }
    }
}

void ParseInput::GetTileAllocSeq(const std::vector<Operation*>& operationList, FunctionPtr func)
{
    if (operationList.empty()) {
        return;
    }
    func->tileAllocSequence.clear();
    func->tileAllocSequence.resize(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));
    for (const auto& op : operationList) {
        std::string opcode = op->GetOpcodeStr();
        if (FilterOpcode(opcode) || op->IsCall()) {
            continue;
        }
        auto& tileOp = func->tileOpMap[op->opmagic];
        bool srcTileHasProducesor = false;
        bool allDstTileMemKnown = true;
        for (auto& in : tileOp->iOperand) {
            if (!in->exeInfo.visited) {
                in->exeInfo.visited = true;
                func->tileAllocSequence[static_cast<int>(in->pipeType)].emplace_back(in->magic);
            }
            if (!in->producers.empty()) {
                srcTileHasProducesor = true;
            }
        }

        for (auto& out : tileOp->oOperand) {
            if (!out->exeInfo.visited) {
                out->exeInfo.visited = true;
                func->tileAllocSequence[static_cast<int>(out->pipeType)].emplace_back(out->magic);
            }
            if (out->bufType == BUF_UNKNOWN || out->bufType == BUF_DDR) {
                allDstTileMemKnown = false;
            }
        }

        // For tileOp without tile wakeup execute
        if ((tileOp->iOperand.size() == 0 || !srcTileHasProducesor) &&
            (tileOp->oOperand.size() == 0 || !allDstTileMemKnown)) {
            func->tileAllocSequence[static_cast<int>(tileOp->pipeType)].emplace_back(tileOp->magic);
        }
    }
    bool fullCover = true;
    for (auto& tile : func->tiles) {
        if (!tile->exeInfo.visited) {
            fullCover = false;
            break;
        }
    }
    func->hasSchedule = fullCover;
}

void ParseInput::BuildFunction(std::shared_ptr<CostModel::SimSys> sim, npu::tile_fwk::Function* parentFunc,
                               FunctionPtr func)
{
    std::unordered_map<int, int> tileMagicIdMap;
    tileMagicIdMap.clear();
    bool isCube = false;
    func->parentFunction = parentFunc;
    func->functionHash = parentFunc->GetFunctionHash().GetHash();
    func->magic = parentFunc->GetFuncMagic();
    func->funcName = parentFunc->GetMagicName();
    func->InitPipeExecTime();

    for (const auto& incast : parentFunc->inCasts_) {
        func->incastMagic.emplace_back(incast->magic);
    }

    for (const auto& outcast : parentFunc->outCasts_) {
        func->outcastMagic.emplace_back(outcast->magic);
    }
    bool hasCall = false;
    const auto& opAfterOOOPass = parentFunc->OperationsAfterOOO();
    uint64_t seq = 0;
    for (auto& op : opAfterOOOPass) {
        std::string opcode = op.GetOpcodeStr();
        if (FilterOpcode(opcode)) {
            continue;
        }
        func->opSequenceAfterOOO_[op.GetOpMagic()] = seq++;
        func->opMagicSequence.emplace_back(op.GetOpMagic());
    }
    const auto& operations = parentFunc->Operations();
    for (auto& op : operations) {
        std::string opcode = op.GetOpcodeStr();
        if (FilterOpcode(opcode)) {
            continue;
        }
        TileOpPtr tileOp = std::make_shared<TileOp>();
        tileOp->funcPtr = func;
        tileOp->operation = &op;

        if (op.HasAttr(OpAttributeKey::scalar)) {
            tileOp->scalarVld = true;
            tileOp->scalarVal = op.GetElementAttribute(OpAttributeKey::scalar);
        }

        for (auto& input : op.GetIOperands()) {
            int magic = input->magic;
            auto it = tileMagicIdMap.find(magic);
            if (it != tileMagicIdMap.end()) {
                func->tiles[it->second]->consumers.emplace_back(tileOp);
                tileOp->iOperand.emplace_back(func->tiles[it->second]);
                continue;
            }
            tileMagicIdMap[magic] = func->tiles.size();

            TilePtr tile = std::make_shared<Tile>();
            BuildTile(*parentFunc, input, tile);
            tile->funcPtr = func;
            tile->consumers.emplace_back(tileOp);
            func->tiles.emplace_back(tile);
            tileOp->iOperand.emplace_back(tile);
            func->tileMap[tile->magic] = tile;
        }
        for (const auto& out : op.GetOOperands()) {
            int magic = out->magic;
            auto it = tileMagicIdMap.find(magic);
            if (it != tileMagicIdMap.end()) {
                func->tiles[it->second]->producer = tileOp;
                func->tiles[it->second]->producers.emplace_back(tileOp);
                tileOp->oOperand.emplace_back(func->tiles[it->second]);
                continue;
            }
            tileMagicIdMap[magic] = int(func->tiles.size());
            TilePtr tile = std::make_shared<Tile>();
            BuildTile(*parentFunc, out, tile);
            tile->funcPtr = func;
            tile->producer = tileOp;
            tile->producers.emplace_back(tileOp);
            func->tiles.emplace_back(tile);
            tileOp->oOperand.emplace_back(tile);
            func->tileMap[tile->magic] = tile;
        }
        tileOp->opcode = op.GetOpcodeStr();
        tileOp->magic = op.GetOpMagic();
        tileOp->subgraphId = op.GetSubgraphID();
        tileOp->semanticLabel = op.GetSemanticLabelStr();
        tileOp->bufType = OperandType::BUF_UB;
        if (tileOp->IsCall()) {
            hasCall = true;
            tileOp->calleeHash = op.GetCalleeHash().GetHash();
        }
        tileOp->GetPipeType();
        tileOp->IsSpecial();
        if (tileOp->pipeType == CorePipeType::PIPE_CUBE) {
            isCube = true;
        }
        func->tileOps.emplace_back(tileOp);
        func->tileOpMap[tileOp->magic] = tileOp;
    }
    ASSERT(
        npu::tile_fwk::InternalError::SIM_INNER_ERROR,
        hasCall || func->opSequenceAfterOOO_.size() == 0 || (func->tileOps.size() == func->opSequenceAfterOOO_.size()))
        << "[SIMULATION]: "
        << "hasCall=" << hasCall << " func->opSequenceAfterOOO_.size=" << func->opSequenceAfterOOO_.size()
        << " func->tileOps.size=" << func->tileOps.size();
    if (sim->config.useOOOPassSeq) {
        GetTileAllocSeq(parentFunc->Operations().DuplicatedOpList(), func);
    }

    if (isCube) {
        func->machineType = MachineType::AIC;
    } else {
        func->machineType = MachineType::AIV;
    }
    CheckFunction(parentFunc, func);
}

void ParseInput::CheckTileOp(FunctionPtr func)
{
    SIMULATION_LOGW("\n[Simulation Check Function]: %s", func->funcName.c_str());
    for (const auto& op : func->tileOps) {
        if (op->IsCall()) {
            continue;
        }
        if (op->iOperand.size() == 0) {
            SIMULATION_LOGW("TileOp has no input: %s", func->funcName.c_str());
            if (op->operation != nullptr) {
                SIMULATION_LOGW("Frontend Operation: %s", op->operation->Dump().c_str());
            }
            SIMULATION_LOGW("Simulation Op: %s", op->Dump(true).c_str());
        }
        if (op->oOperand.size() == 0) {
            SIMULATION_LOGW("Function: %s Op: %s has no input", func->funcName.c_str(), op->Dump(true).c_str());
        }
    }
}

void ParseInput::CheckTile(FunctionPtr func)
{
    // Check Tile
    for (auto& tile : func->tiles) {
        if (tile->producers.size() == 0) {
            if (std::find(func->incastMagic.begin(), func->incastMagic.end(), tile->magic) == func->incastMagic.end()) {
                SIMULATION_LOGW("Tile has no producer, but not incast: %s", tile->Dump().c_str());
                func->incastMagic.emplace_back(tile->magic);
            }
        }
        if (tile->consumers.size() == 0) {
            if (std::find(func->outcastMagic.begin(), func->outcastMagic.end(), tile->magic) ==
                func->outcastMagic.end()) {
                SIMULATION_LOGW("Tile has no consumer, but not outcast: %s", tile->Dump().c_str());
                func->outcastMagic.emplace_back(tile->magic);
            }
        }
    }
}

void ParseInput::CheckInOutCast(FunctionPtr func)
{
    // Check outcast/incast
    auto inIdx = func->incastMagic.begin();
    while (inIdx != func->incastMagic.end()) {
        if (func->tileMap.find((*inIdx)) == func->tileMap.end()) {
            SIMULATION_LOGW("Incast not found in tileMap: %d", (*inIdx));
            inIdx = func->incastMagic.erase(inIdx);
            continue;
        }
        auto& incast = func->tileMap[(*inIdx)];
        incast->nodeType = NodeType::INCAST;
        if (incast->producers.size() != 0) {
            SIMULATION_LOGW("Incast has producer %s", incast->Dump().c_str());
        }
        if (incast->consumers.size() == 0) {
            SIMULATION_LOGW("Incast has no consumer %s", incast->Dump().c_str());
        }
        inIdx++;
    }
    auto outIdx = func->outcastMagic.begin();
    while (outIdx != func->outcastMagic.end()) {
        if (func->tileMap.find((*outIdx)) == func->tileMap.end()) {
            SIMULATION_LOGW("Outcast not found in tileMap %d", (*outIdx));
            outIdx = func->outcastMagic.erase(outIdx);
            continue;
        }
        auto& outcast = func->tileMap[(*outIdx)];
        outcast->nodeType = NodeType::OUTCAST;
        if (outcast->producers.size() == 0) {
            SIMULATION_LOGW("Outcast has no producer %s", outcast->Dump().c_str());
        }
        if (outcast->consumers.size() != 0) {
            SIMULATION_LOGW("Outcast has no consumer %s", outcast->Dump().c_str());
        }
        outIdx++;
    }
}

void ParseInput::CheckFunction(npu::tile_fwk::Function* parentFunc, FunctionPtr func)
{
    (void)parentFunc;
    CheckTileOp(func);
    CheckTile(func);
    CheckInOutCast(func);
}

void ParseInput::ParseFunction(std::shared_ptr<CostModel::SimSys> sim,
                               std::vector<npu::tile_fwk::Function*>& inputFuncs, bool topoFromRootFunc)
{
    if (topoFromRootFunc) {
        sim->enableExpectValue = true;
        ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, inputFuncs.size() == 1)
            << "[SIMULATION]: inputFuncs.size is not equals to 1."
            << "inputFuncs.size=" << inputFuncs.size();
        for (const auto& rootFunction : inputFuncs) {
            FunctionPtr func = std::make_shared<Function>();
            func->functionHash = rootFunction->GetFunctionHash().GetHash();
            func->magic = rootFunction->GetFuncMagic();
            func->funcName = rootFunction->GetMagicName();

            sim->startFuncName = func->funcName;
            sim->startFuncHash = func->functionHash;
            const auto& operations = rootFunction->Operations();
            for (auto& topo : rootFunction->topoInfo_.GetTopology()) {
                // Copy input topoinfo.
                TopoInfoEntry entry;
                entry.eSgId = topo.esgId;
                entry.readyState = topo.readyState;
                entry.outGraph = topo.outGraph;
                entry.calleeHash = operations[topo.esgId].GetCalleeHash().GetHash();
                func->inputTopo.push_back(entry);
            }
            BuildFunction(sim, rootFunction, func);
            if (sim->drawGraph) {
                ModelVisualizer visualizer;
                visualizer.DrawFunction(func, sim->graphsOutdir);
            }
            func->topoFromRootFunc = true;
            sim->functionCache.Insert(func);

            // Build Leaf Functions
            for (auto& leafFunc : rootFunction->programs_) {
                FunctionPtr lFunc = std::make_shared<Function>();
                BuildFunction(sim, leafFunc.second, lFunc);
                lFunc->pSgId = leafFunc.first;
                lFunc->parentFunction = leafFunc.second;
                if (lFunc->machineType == MachineType::AIC) {
                    sim->stats->totalFunctionCube++;
                } else {
                    sim->stats->totalFunctionVec++;
                }
                sim->stats->totalFunctionNum++;
                sim->stats->totalFunctionTileOps += lFunc->tileOps.size();
                sim->functionCache.Insert(lFunc);
                func->tileMap.insert(lFunc->tileMap.begin(), lFunc->tileMap.end());
                if (sim->drawGraph) {
                    ModelVisualizer visualizer;
                    visualizer.DrawFunction(lFunc, sim->graphsOutdir);
                }
            }
            BuildFunctionInvoke(func, sim);
        }
        return;
    }

    sim->enableExpectValue = false;
    // Get Function From parentFunctions Input
    bool foundStartFunc = false;
    for (const auto& function : inputFuncs) {
        FunctionPtr func = std::make_shared<Function>();
        BuildFunction(sim, function, func);
        func->parentFunction = function;
        if (!foundStartFunc) {
            size_t pos = func->funcName.find(sim->config.startFunctionLabel);
            if (pos != std::string::npos) {
                foundStartFunc = true;
                sim->startFuncName = func->funcName;
                sim->startFuncHash = func->functionHash;
            }
        }
        if (sim->drawGraph) {
            ModelVisualizer visualizer;
            visualizer.DrawFunction(func, sim->graphsOutdir);
        }
        sim->stats->totalFunctionNum++;
        sim->stats->totalFunctionTileOps += func->tileOps.size();
        sim->functionCache.Insert(func);
        BuildFunctionInvoke(func, sim);
    }
}

void ParseInput::ParseSingleFunction(std::shared_ptr<CostModel::SimSys> sim, npu::tile_fwk::Function* func)
{
    FunctionPtr lFunc = std::make_shared<Function>();
    BuildFunction(sim, func, lFunc);
    sim->functionCache.Insert(lFunc);
    sim->singleFuncHash = lFunc->functionHash;
    if (sim->drawGraph) {
        ModelVisualizer visualizer;
        visualizer.DrawFunction(lFunc, sim->graphsOutdir);
    }
}

void ParseInput::ParseJsonConfig(const std::string& path, std::vector<std::string>& cfg) const
{
    std::ifstream file(path);
    if (!file.is_open()) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_OPEN_FAILED, "Error: fail to open file: %s", path.c_str());
        return;
    }
    Json j;
    file >> j;
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::string c = it.key() + "=" + it.value().dump();
        cfg.emplace_back(c);
    }
    file.close();
}

void ParseInput::ParseConfig(const std::string& path, std::vector<std::string>& cfg) const
{
    std::ifstream file(path);
    if (!file.is_open()) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_OPEN_FAILED, "Error: fail to open file: %s", path.c_str());
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            cfg.emplace_back(line);
        } else {
            SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_CONTENT_ERROR, "Parse Config File Error: %s",
                            line.c_str());
        }
    }
    file.close();
}

void ParseInput::ParseCalendarJson(std::shared_ptr<CostModel::SimSys> sim, const std::string& jsonPath) const
{
    std::ifstream jsonInput(jsonPath);
    if (!jsonInput.is_open()) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_OPEN_FAILED, "Error: fail to open file: %s",
                        jsonPath.c_str());
        return;
    }
    Json calendarJson;
    jsonInput >> calendarJson;

    int counterNum = calendarJson["numSupportedCounters"].get<int>();

    sim->calendarCounter.resize(counterNum, 0);

    std::vector<std::pair<int, int>> waitVector;
    int taskId;
    for (const auto& core : calendarJson["cores"]) {
        for (const auto& task : core["tasks"]) {
            // change to functionHash
            if (task.contains("functionHash")) {
                sim->taskWaitMap[task["taskId"].get<int>()] = waitVector;
                sim->corePacketMap[core["coreId"].get<int>()].push_back(
                    {task["taskId"].get<int>(), std::stoull(task["functionHash"].get<std::string>())});
                taskId = task["taskId"].get<int>();
                if (sim->config.calendarMode == static_cast<uint64_t>(CalendarMode::GLOBAL_COUNTER)) {
                    ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, waitVector.size() == 1)
                        << "[SIMULATION]: task has two wait in calendar global counter."
                        << "waitVector.size=" << waitVector.size();
                    sim->taskFirstSetMap[taskId] = waitVector[0].second + 1;
                }
                waitVector.clear();
            } else {
                if (task["operation"].get<std::string>() == "wait") {
                    waitVector.push_back({task["counterId"].get<int>(), task["expectedValue"].get<int>()});
                } else if (task["operation"].get<std::string>() == "set") {
                    sim->taskSetMap[taskId] = task["counterId"].get<int>();
                    sim->taskSetExpectMap[taskId] = task["expectedValue"];
                } else if (task["operation"].get<std::string>() == "setWait") {
                    sim->taskWaitBeforeSetMap[taskId].push_back(
                        {task["counterId"].get<int>(), task["expectedValue"].get<int>()});
                }
            }
        }
    }
}

void ParseInput::ParseFixedLatencyTask(std::shared_ptr<CostModel::SimSys> sim, std::string const& path)
{
    std::ifstream jsonInput(path);
    if (!jsonInput.is_open()) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_OPEN_FAILED, "Error: fail to open file: %s", path.c_str());
        return;
    }
    Json fixedLatencyTask;
    jsonInput >> fixedLatencyTask;

    uint64_t virtualRootHash = 0;
    uint64_t virtualLeafHash = 1;
    std::unordered_map<std::string, uint64_t> leafVirturalHashMap;
    std::unordered_map<std::string, MachineType> leafMachineTypeMap;

    // Build virtual root function
    FunctionPtr func = std::make_shared<Function>();
    func->functionHash = virtualRootHash;
    func->magic = virtualRootHash;
    func->funcName = "virtual_root_function";
    sim->startFuncName = func->funcName;
    sim->startFuncHash = func->functionHash;
    uint64_t cycleConvert = sim->config.fixedLatencyTimeConvert;

    for (const auto& item : fixedLatencyTask) {
        TopoInfoEntry entry;
        entry.eSgId = item["taskId"].get<uint64_t>();
        entry.readyState = item["remainingPredecessors"];
        entry.readyState *= -1;
        entry.outGraph = item["successors"].get<setType>();
        std::string funcName = item["funcName"].get<std::string>();
        if (leafVirturalHashMap.find(funcName) == leafVirturalHashMap.end()) {
            leafVirturalHashMap[funcName] = virtualLeafHash++;
        }
        entry.calleeHash = leafVirturalHashMap[funcName];
        double exeTime = item["execTime"].get<double>();
        entry.fixedLatency = true;
        entry.fixedLatencyVal = static_cast<uint64_t>(std::trunc(exeTime * cycleConvert));
        std::string machineType = item["coreType"];
        entry.mType = ToMachineType(machineType);
        leafMachineTypeMap[funcName] = entry.mType;
        func->inputTopo.push_back(entry);
    }
    func->topoFromRootFunc = true;
    sim->functionCache.Insert(func);

    // Build virtual leaf function
    for (auto& [funcName, funcHash] : leafVirturalHashMap) {
        FunctionPtr leafFunc = std::make_shared<Function>();
        leafFunc->functionHash = funcHash;
        leafFunc->machineType = leafMachineTypeMap[funcName];
        leafFunc->funcName = funcName;
        sim->functionCache.Insert(leafFunc);
    }
}

void ParseInput::ParseTopoJson(std::string path, std::deque<TaskMap>& taskMapQueue)
{
    std::ifstream jsonInput(path);
    if (!jsonInput.is_open()) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_OPEN_FAILED, "Error: fail to open file: %s", path.c_str());
        return;
    }
    Json topoJson;
    jsonInput >> topoJson;
    std::map<uint64_t, TaskMap> groupTaskMap;
    for (const auto& item : topoJson) {
        auto subtask = std::make_shared<Task>();
        subtask->seqNo = item.value("seqNo", 0);
        subtask->taskId = item.value("taskId", 0);
        subtask->leafIndex = item.value("leafIndex", 0);
        subtask->opmagic = item.value("opmagic", 0);
        subtask->psgId = item.value("psgId", -1);
        subtask->rootIndex = item.value("rootIndex", 0);
        subtask->uniqueKey = item.value("uniqueKey", subtask->taskId);
        subtask->functionHash = item["funcHash"].get<uint64_t>();
        subtask->machineType = ToMachineType(item["coreType"]);
        subtask->successors = item["successors"].get<std::vector<uint64_t>>();
        groupTaskMap[subtask->seqNo][subtask->taskId] = subtask;
    }
    for (auto& taskMap : groupTaskMap) {
        for (auto& task : taskMap.second) {
            for (auto& successor : task.second->successors) {
                taskMap.second.at(successor)->predecessors.push_back(task.first);
            }
        }
        for (auto& task : taskMap.second) {
            task.second->remainingPredecessors = task.second->predecessors.size();
        }
    }
    for (auto& entry : groupTaskMap) {
        taskMapQueue.push_back(entry.second);
    }
}

void ParseInput::ParseReplayInfoJson(const std::string& path,
                                     std::unordered_map<uint64_t, std::deque<ReplayTaskEntry>>& replayTasksInfoMap)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        SIMULATION_LOGE(CostModel::ExternalErrorScene::FILE_OPEN_FAILED, "Error: fail to open file: %s", path.c_str());
        return;
    }
    Json j;
    file >> j;
    for (const auto& item : j) {
        uint64_t blockIdx = item["blockIdx"];
        std::string coreTypeStr = item["coreType"];
        MachineType coreType = ToMachineType(coreTypeStr);
        if (!IsCoreMachine(coreType)) {
            continue;
        }
        uint64_t machineId = GetProcessID(coreType, blockIdx);
        const auto& tasks = item["tasks"];
        replayTasksInfoMap[machineId] = std::deque<ReplayTaskEntry>();
        auto& machineTaskQ = replayTasksInfoMap[machineId];
        for (const auto& task : tasks) {
            uint64_t seqNo = task["seqNo"];
            uint64_t taskId = task["taskId"];
            uint64_t beginCycle = task["execStart"];
            uint64_t endCycle = task["execEnd"];
            machineTaskQ.push_back(ReplayTaskEntry(seqNo, taskId, beginCycle, endCycle));
        }
    }
}
} // namespace CostModel
