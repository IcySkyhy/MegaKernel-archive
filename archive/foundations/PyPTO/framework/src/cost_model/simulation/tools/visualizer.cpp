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
 * \file visualizer.cpp
 * \brief
 */

#include "cost_model/simulation/tools/visualizer.h"

#include <iostream>
#include <fstream>
#include <set>

#include "utils/file_utils.h"
#include "tilefwk/pypto_fwk_log.h"

namespace CostModel {

void ModelVisualizer::DrawTile(std::ofstream& os, TilePtr tensor, bool debug) const
{
    std::string label = "\"Tile\\n" + tensor->Dump();
    std::string fillColor = GetColor(tensor->magic);
    std::string fontColor = GetReverseColor(tensor->magic);
    if (debug) {
        bool unissue = !tensor->exeInfo.isWritten || !tensor->exeInfo.isAllocated;
        label += ("\\nUNISSUE:" + std::to_string(unissue));
    }
    label += "\"";
    os << "\tT" << tensor->magic << " [label=" << label;
    os << ", shape=box,style=filled,fillcolor=\"" << fillColor << "\",fontcolor=\"" << fontColor << "\"";
    os << "];" << std::endl;
}

void ModelVisualizer::DrawTileOp(std::ofstream& os, TileOpPtr tileop, FunctionPtr func, bool debug) const
{
    std::string label = "\"TileOp\\nopmagic:" + std::to_string(tileop->magic) + "\\nopcode:" + tileop->opcode;
    std::string fillColor = GetColor(tileop->magic);
    std::string fontColor = GetReverseColor(tileop->magic);
    if (tileop->IsCall()) {
        label += "\\ncalleeHash:" + std::to_string(tileop->calleeHash);
    }
    label += "\\nSeq:" + std::to_string(func->opSequenceAfterOOO_[tileop->magic]);
    if (debug) {
        bool unissue = !tileop->exeInfo.issued || !tileop->exeInfo.retired;
        label += "\\nUNISSUE:" + std::to_string(unissue);
    }
    label += "\"";
    os << "\tN" << tileop->magic << " [label=" << label;
    os << ",shape=box,style=filled,fillcolor=\"" << fillColor << "\",fontcolor=\"" << fontColor << "\"";
    os << "];" << std::endl;
}

void ModelVisualizer::DrawTask(std::ofstream& os, std::shared_ptr<Task> task, bool detail)
{
    std::string label = "\"TaskID:" + std::to_string(task->taskId);
    if (detail) {
        label += "_detail";
    }
    std::string fillColor = GetTaskColor(task->machineType, task->taskId);
    std::string fontColor = GetTaskFontColor(task->machineType, task->taskId);
    label += "|FunctionName:" + task->functionName;
    label += "|FunctionHash:" + std::to_string(task->functionHash);
    label += "|" + MachineName(task->machineType);
    label += "\"";
    os << "\tN" << task->taskId << " [label=" << label;
    os << ",shape=record,style=filled,fillcolor=\"" << fillColor << "\",fontcolor=\"" << fontColor << "\"";
    os << "];" << std::endl;
}

void ModelVisualizer::DrawFunction(FunctionPtr func, const std::string& outdir, bool debug) const
{
    std::string path = outdir + "/" + func->funcName + "_graph.dot";
    std::ofstream os(path);

    os << "digraph {" << std::endl;
    os << "\tlabel=\"" << func->funcName << "\\nhash:" << std::to_string(func->functionHash) << "\";" << std::endl;
    os << "\trankdir=LR;" << std::endl;
    for (auto tensor : func->tiles) {
        DrawTile(os, tensor, debug);
    }

    for (auto tileop : func->tileOps) {
        DrawTileOp(os, tileop, func, debug);
        for (auto in : tileop->iOperand) {
            os << "\tT" << in->magic << " -> N" << tileop->magic << ";" << std::endl;
        }
        for (auto out : tileop->oOperand) {
            os << "\tN" << tileop->magic << " -> T" << out->magic << ";" << std::endl;
        }
    }
    os << "}" << std::endl;
    os.close();
    SIMULATION_LOGW("Path: %s", path.c_str());
}

void ModelVisualizer::DebugFunction(FunctionPtr func, std::unordered_map<int, TilePtr>& tiles,
                                    std::unordered_map<int, TileOpPtr>& tileOps, const std::string& outdir) const
{
    std::string path = outdir + "/" + func->funcName + ".deadlock_debug_graph.dot";
    std::ofstream os(path);

    os << "digraph {" << std::endl;
    os << "\tlabel=\"deadlock_debug_graph\";" << std::endl;
    os << "\trankdir=LR;" << std::endl;
    for (auto tensor : tiles) {
        DrawTile(os, tensor.second, true);
    }

    for (auto tileop : tileOps) {
        DrawTileOp(os, tileop.second, func, true);
        for (auto in : tileop.second->iOperand) {
            os << "\tT" << in->magic << " -> N" << tileop.second->magic << ";" << std::endl;
        }
        for (auto out : tileop.second->oOperand) {
            os << "\tN" << tileop.second->magic << " -> T" << out->magic << ";" << std::endl;
        }
    }
    os << "}" << std::endl;
    os.close();
    SIMULATION_LOGW("Path: %s", path.c_str());
}

void ModelVisualizer::DrawTasks(const TaskMap& taskMap, bool drawDetail, std::string outPath)
{
    std::string globalLabel = drawDetail ? "Tasks Graph" : "Tasks Thumbnail Graph";
    std::ofstream os(outPath);

    os << "digraph {" << std::endl;
    os << "\tlabel=\"" << globalLabel << "\";" << std::endl;
    os << "\trankdir=LR;" << std::endl;

    for (auto& task : taskMap) {
        DrawTask(os, task.second, true);
    }

    for (auto& task : taskMap) {
        for (auto& src : task.second->predecessors) {
            os << "\tN" << src << " -> N" << task.second->taskId << ";" << std::endl;
        }
    }

    os << "}" << std::endl;
    os.close();
}

std::string ModelVisualizer::GetColor(uint64_t color) const
{
    uint64_t randColor = color % uint64_t(Modulor::MODULOR_NUM);
    switch (static_cast<Modulor>(randColor)) {
        case Modulor::MODULOR_0:
            return "azure3";
        case Modulor::MODULOR_1:
            return "plum3";
        case Modulor::MODULOR_2:
            return "#a9def9";
        case Modulor::MODULOR_3:
            return "#005f73";
        case Modulor::MODULOR_4:
            return "#F1766D";
        case Modulor::MODULOR_5:
            return "#839DD1";
        case Modulor::MODULOR_6:
            return "orange1";
        case Modulor::MODULOR_7:
            return "#9932CC";
        case Modulor::MODULOR_8:
            return "royalblue1";
        case Modulor::MODULOR_9:
            return "cyan";
        case Modulor::MODULOR_10:
            return "aquamarine";
        case Modulor::MODULOR_11:
            return "#d0f4de";
        case Modulor::MODULOR_12:
            return "#9FD4AE";
        case Modulor::MODULOR_13:
            return "#FDD379";
        case Modulor::MODULOR_14:
            return "#7A70B5";
        default:
            return "#4a5759";
    }
}

std::string ModelVisualizer::GetReverseColor(uint64_t color) const
{
    uint64_t randColor = color % uint64_t(Modulor::MODULOR_NUM);
    switch (static_cast<Modulor>(randColor)) {
        case Modulor::MODULOR_0:
            return "black";
        case Modulor::MODULOR_1:
            return "black";
        case Modulor::MODULOR_2:
            return "black";
        case Modulor::MODULOR_3:
            return "white";
        case Modulor::MODULOR_4:
            return "black";
        case Modulor::MODULOR_5:
            return "white";
        case Modulor::MODULOR_6:
            return "black";
        case Modulor::MODULOR_7:
            return "white";
        case Modulor::MODULOR_8:
            return "white";
        case Modulor::MODULOR_9:
            return "black";
        case Modulor::MODULOR_10:
            return "black";
        case Modulor::MODULOR_11:
            return "black";
        case Modulor::MODULOR_12:
            return "white";
        case Modulor::MODULOR_13:
            return "black";
        case Modulor::MODULOR_14:
            return "white";
        default:
            return "white";
    }
}

std::string ModelVisualizer::GetTaskColor(MachineType type, uint64_t taskId)
{
    auto query = taskColorMap.find(type);
    if (query != taskColorMap.end()) {
        uint64_t idx = taskId % uint64_t(query->second.size());
        return query->second.at(idx);
    }
    return "gray";
}

std::string ModelVisualizer::GetTaskFontColor(CostModel::MachineType type, uint64_t taskId)
{
    auto query = taskFontColorMap.find(type);
    if (query != taskFontColorMap.end()) {
        uint64_t idx = taskId % uint64_t(query->second.size());
        return query->second.at(idx);
    }
    return "white";
}
} // namespace CostModel
