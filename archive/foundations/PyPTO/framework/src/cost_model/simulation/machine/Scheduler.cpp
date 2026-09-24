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
 * \file Scheduler.cpp
 * \brief
 */

#include "cost_model/simulation/machine/Scheduler.h"

#include <algorithm>

#include "cost_model/simulation/base/ModelTop.h"
#include "tilefwk/pypto_fwk_log.h"

namespace CostModel {

std::shared_ptr<SimSys> Scheduler::GetSim() { return sim; }

void Scheduler::MergeCopyOutGroup(int srcCopyOutIdx, int curCopyOutIdx, std::map<int, int>& copyOutSeq)
{
    if (srcCopyOutIdx != curCopyOutIdx) {
        while (copyOutSeq[srcCopyOutIdx] != srcCopyOutIdx) {
            srcCopyOutIdx = copyOutSeq[srcCopyOutIdx];
        }
        copyOutSeq[srcCopyOutIdx] = copyOutSeq[curCopyOutIdx];
    }
}

void Scheduler::TileInsertQueue(TilePtr tile, std::vector<std::vector<int>>& tileAllocSequence)
{
    int queueIndex = static_cast<int>(tile->pipeType);
    auto it = pipeIssueOrders[queueIndex].find(tile->magic);
    if (it != pipeIssueOrders[queueIndex].end()) {
        return;
    }
    SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] pop tile magic: %d, sequence: %d", GetSim()->GetCycles(),
                    tile->magic, issueSequencePtr[queueIndex]);
    pipeIssueOrders[queueIndex][tile->magic] = issueSequencePtr[queueIndex];
    issueSequencePtr[queueIndex]++;
    tileAllocSequence[queueIndex].emplace_back(tile->magic);
}

void Scheduler::TileOpInsertQueue(TileOpPtr tileOp)
{
    int queueIndex = static_cast<int>(tileOp->pipeType);
    auto it = pipeIssueOrders[queueIndex].find(tileOp->magic);
    if (it != pipeIssueOrders[queueIndex].end()) {
        return;
    }
    pipeIssueOrders[queueIndex][tileOp->magic] = issueSequencePtr[queueIndex];
    tileOp->exeInfo.sequenceToIssue = issueSequencePtr[queueIndex];
    issueSequencePtr[queueIndex]++;
    SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] pop tileop opmagic: %d to %s , seq: %d", GetSim()->GetCycles(),
                    tileOp->magic, CorePipeName(tileOp->pipeType).c_str(), tileOp->exeInfo.sequenceToIssue);
}

void Scheduler::SortTile(std::unordered_map<int, TilePtr>& tiles, std::unordered_map<int, TileOpPtr>& tileOps,
                         std::vector<std::vector<int>>& tileAllocSequence)
{
    if (tiles.empty() || tileOps.empty()) {
        return;
    }
    std::deque<std::pair<int, bool>> queue;
    std::map<int, bool> tilesVisited;
    std::map<int, bool> tileOpsVisited;

    // Get all outcast nodes or non_consumers nodes
    for (auto& tile : tiles) {
        tilesVisited[tile.first] = false;
        if (tile.second->exeInfo.isOutcast) {
            SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] outcast index: %d", GetSim()->GetCycles(), tile.first);
            queue.emplace_back(tile.first, true);
        } else if (tile.second->consumers.empty()) {
            SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] no consumers tile index: %d, magic: %d",
                            GetSim()->GetCycles(), tile.first, tile.second->magic);
            queue.emplace_back(tile.first, true);
        }
    }
    for (auto& tileOp : tileOps) {
        tileOpsVisited[tileOp.first] = false;
        if (tileOp.second->oOperand.empty()) {
            queue.emplace_back(tileOp.first, false);
        }
    }
    SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] output nodes queue size: %zu", GetSim()->GetCycles(),
                    queue.size());

    ASSERT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, !queue.empty())
        << "[SIMULATION]: Sort Tile Alloc not find output nodes";

    // Merge And Sort outcast
    std::deque<std::pair<int, bool>> tmpQueue;
    std::map<int, bool> tmpTileOpsVisited;
    std::map<int, bool> tmpTileVisited;
    for (auto& tile : tiles) {
        tmpTileVisited[tile.first] = false;
    }
    for (auto& tileOp : tileOps) {
        tmpTileOpsVisited[tileOp.first] = false;
    }
    std::map<int, int> copyOutSeq;
    for (auto& it : queue) {
        if (it.second) {
            tiles[it.first]->exeInfo.copyOutIdx = it.first;
        } else {
            tileOps[it.first]->exeInfo.copyOutIdx = it.first;
        }
        copyOutSeq[it.first] = it.first;
        tmpQueue.push_back(it);
    }
    while (!tmpQueue.empty()) {
        auto back = tmpQueue.back();
        tmpQueue.pop_back();
        if (back.second) { // Is tile
            tmpTileVisited[back.first] = true;
            auto tile = tiles[back.first];
            SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] tmpQueue::tile::: %d", GetSim()->GetCycles(),
                            back.first);
            if (tile->producers.empty()) {
                continue;
            }
            for (const auto& producer : tile->producers) {
                // Push tile's producer_tileop to tmp_queue_back
                auto producerTileop = producer;
                if (!tmpTileOpsVisited[producerTileop->magic]) {
                    tmpQueue.emplace_back(producerTileop->magic, false); // 第一次被找到时的copy_out_idx
                    producerTileop->exeInfo.copyOutIdx = tile->exeInfo.copyOutIdx;
                } else {
                    MergeCopyOutGroup(producerTileop->exeInfo.copyOutIdx, tile->exeInfo.copyOutIdx, copyOutSeq);
                }
            }
        } else { // Is operation
            tmpTileOpsVisited[back.first] = true;
            auto tileop = tileOps[back.first];
            SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] tmpQueue::tileOps:::%d, opmagic: %d",
                            GetSim()->GetCycles(), back.first, tileop->magic);
            for (const auto& srcTile : tileop->iOperand) {
                if (!tmpTileVisited[srcTile->magic]) {
                    tmpQueue.emplace_back(srcTile->magic, true); // 第一次被找到时的copy_out_idx
                    srcTile->exeInfo.copyOutIdx = tileop->exeInfo.copyOutIdx;
                } else {
                    MergeCopyOutGroup(srcTile->exeInfo.copyOutIdx, tileop->exeInfo.copyOutIdx, copyOutSeq);
                }
            }
        }
    }

    for (auto& it : copyOutSeq) {
        int father = it.first;
        while (father != copyOutSeq[father]) {
            father = copyOutSeq[father];
        }
        copyOutSeq[it.first] = father;
        SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] %d->%d", GetSim()->GetCycles(), it.first, father);
    }

    sort(queue.begin(), queue.end(),
         [&copyOutSeq](auto& a, auto& b) { return copyOutSeq[a.first] < copyOutSeq[b.first]; });

    // Sort Tile Alloc And TileOp(Operation)
    issueSequencePtr.clear();
    pipeIssueOrders.clear();
    issueSequencePtr.resize(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE), 0);
    pipeIssueOrders.resize(static_cast<int>(CorePipeType::TOTAL_CORE_PIPE_TYPE));
    while (!queue.empty()) {
        auto back = queue.back();
        if (back.second) { // Is tile
            if (tilesVisited[back.first]) {
                queue.pop_back();
                auto tile = tiles[back.first];
                TileInsertQueue(tile, tileAllocSequence);
                continue;
            }

            tilesVisited[back.first] = true;
            auto tile = tiles[back.first];
            SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] read tile magic: %d", GetSim()->GetCycles(),
                            tile->magic);
            if (!tile->producers.empty()) {
                for (const auto& producer : tile->producers) {
                    // Push tile's producer_tileop to queue_back
                    SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] push tileop opmagic: %d, domCount: %d",
                                    GetSim()->GetCycles(), producer->magic, producer->exeInfo.domCount);
                    queue.emplace_back(producer->magic, false);
                }
            }
        } else {
            if (tileOpsVisited[back.first]) {
                queue.pop_back();
                auto tileop = tileOps[back.first];
                TileOpInsertQueue(tileop);
                continue;
            }

            tileOpsVisited[back.first] = true;
            auto tileop = tileOps[back.first];
            SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] read tileop opmagic: %d", GetSim()->GetCycles(),
                            tileop->magic);

            // Sort the src of this node according to the rules.
            std::vector<std::pair<int, int>> srcStat;
            if (sortTileAllocPolicy == "DOM_COUNT") {
                for (const auto& srcTile : tileop->iOperand) {
                    srcStat.emplace_back(srcTile->magic, srcTile->exeInfo.domCount);
                }
            }

            sort(srcStat.begin(), srcStat.end(),
                 [](std::pair<int, int>& a, std::pair<int, int>& b) { return a.second < b.second; });
            for (auto it : srcStat) {
                if (tilesVisited[it.first]) {
                    continue;
                }
                SIMULATION_LOGI("[Cycle: %lu][Scheduler][SortTile] push tile magic: %d, domCount: %d",
                                GetSim()->GetCycles(), tiles[it.first]->magic, tiles[it.first]->exeInfo.domCount);
                queue.emplace_back(it.first, true);
            }
            if (srcStat.empty() || (tileop->exeInfo.noDstWakeup && tileop->exeInfo.noSrcWakeup)) {
                tileAllocSequence[static_cast<int>(tileop->pipeType)].emplace_back(back.first);
            }
        }
    }
}
} // namespace CostModel
