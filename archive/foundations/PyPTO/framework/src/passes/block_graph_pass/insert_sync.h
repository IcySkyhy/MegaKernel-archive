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
 * \file insert_sync.h
 * \brief
 */

#ifndef PASS_INSERT_SYNC_H
#define PASS_INSERT_SYNC_H

#include <queue>
#include "interface/utils/common.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/platform.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/opcode.h"
#include "passes/pass_interface/pass.h"

namespace npu {
namespace tile_fwk {
constexpr uint64_t EVENTID_DEADLOCK_ENTER_TIME = 5;
constexpr uint64_t DEADLOCK_TIME_THRESHOLD = 2;
constexpr uint64_t LEFT_OFFSET1 = 32;
constexpr uint64_t LEFT_OFFSET2 = 16;
constexpr uint64_t LEFT_OFFSET3 = 8;
constexpr uint64_t LEFT_OFFSET4 = 24;
constexpr uint64_t MAX_POP = 8;
// 两个op之间插入的set_flag/wait_flag数量的最大值为192
constexpr uint64_t SEQUENCE_IDX = 300;
constexpr uint64_t HALF_SEQUENCE_IDX = 150;
constexpr uint64_t FORCE_SYNC_OP_NUM = 15;
constexpr uint64_t NUM15 = 15;
constexpr uint64_t NUM31 = 31;
struct Interval {
    int start;
    int end;
    int idx;
    Interval(int s, int e, int id) : start(s), end(e), idx(id) {}
};

struct IntervalTreeNode {
    Interval interval;
    int max;
    IntervalTreeNode* left;
    IntervalTreeNode* right;
    explicit IntervalTreeNode(Interval i) : interval(i), max(i.end), left(nullptr), right(nullptr) {}
};

class RangeSearchTree {
public:
    ~RangeSearchTree() { FreeTree(); }
    void Insert(int left, int right, int idx);
    std::set<int> GetCovered(int left, int right);

private:
    void FreeTree();
    Status ProcessTreeNode(const Interval& interval, IntervalTreeNode* currPtr,
                           std::vector<IntervalTreeNode*>& intervalStack);
    Status InsertInterval(const Interval& interval);
    void OverlapSearch(const Interval& interval, std::set<int>& result);
    IntervalTreeNode* treeRoot = nullptr;
};

class DataDependencySearcher {
public:
    std::set<int> Find(Operation* opWait);
    void Insert(const Operation* opSet, int idx);
    std::unordered_map<int, TileRange> ubTensorRangeMap;

private:
    void CheckRAWSearchTree(Operation* opWait, std::set<int>& res);
    void CheckWAWSearchTree(Operation* opWait, std::set<int>& res);
    void CheckWARSearchTree(Operation* opWait, std::set<int>& res);
    void InsertWAWSearchTree(const Operation* opSet, int idx);
    void InsertRAWSearchTree(const Operation* opSet, int idx);
    void InsertWARSearchTree(const Operation* opSet, int idx);

    std::unordered_map<MemoryType, RangeSearchTree> wawSearchTree_;
    std::unordered_map<MemoryType, RangeSearchTree> warSearchTree_;
    std::unordered_map<MemoryType, RangeSearchTree> rawSearchTree_;
    std::unordered_map<int, std::set<int>> readDdrMemMap;
    std::unordered_map<int, std::set<int>> writeDdrMemMap;
};

using IndexOp = std::pair<uint64_t, std::reference_wrapper<Operation>>;
enum class PipeSeq {
    AIC_MTE2 = 0,
    AIC_MTE1,
    AIC_M,
    AIC_FIX,
    AIV0_MTE2,
    AIV1_MTE2,
    AIV0_V,
    AIV1_V,
    AIV0_MTE3,
    AIV1_MTE3,
    AIC_MTE3,
    AIV0_S,
    AIV1_S,
    AIC_S,
    PIPE_END
};

class PipeSync {
public:
    PipeSync() { InitIssueQueue(); }
    Status InsertSync(Function& function, std::vector<Operation*>& syncedOpLog);
    void PhaseKernelProcess(Function& function, const std::vector<Operation*>& srcLog, std::vector<Operation*>& dstLog);
    const std::vector<Operation*>& GetOriOpList() const { return oriOpList_; }
    std::unordered_map<Operation*, Operation*> setOpMap;
    std::unordered_map<Operation*, Operation*> waitOpMap;

private:
    friend class TuneTileOpSeqForVF;
    friend class TuneSyncForVF;
    friend class InsertSync;

    struct PipeCoreReal {
        PipeCoreReal(PipeType p, CoreType c) : pipe(p), core(c) {}
        PipeType pipe;
        CoreType core;

        bool operator==(const PipeCoreReal& t) const { return (this->pipe == t.pipe && this->core == t.core); }

        bool operator!=(const PipeCoreReal& t) const { return !(*this == t); }
    };

    // 包含AIVCore类型的PipeCoreReal
    struct PipeCoreRealEx {
        PipeCoreRealEx(PipeType p, CoreType c, AIVCore a = AIVCore::UNSPECIFIED) : pipe(p), core(c), aivCore(a) {}
        PipeCoreRealEx(PipeCoreReal p, AIVCore a = AIVCore::UNSPECIFIED) : pipe(p.pipe), core(p.core), aivCore(a) {}
        PipeType pipe;
        CoreType core;
        AIVCore aivCore{AIVCore::UNSPECIFIED};

        bool operator==(const PipeCoreRealEx& t) const
        {
            return (this->pipe == t.pipe && this->core == t.core && this->aivCore == t.aivCore);
        }

        bool operator!=(const PipeCoreRealEx& t) const { return !(*this == t); }
    };

    struct PipeCoreRealExCompare {
        bool operator()(const PipeCoreRealEx& lhs, const PipeCoreRealEx& rhs) const
        {
            if (lhs.core != rhs.core) {
                return static_cast<uint64_t>(lhs.core) < static_cast<uint64_t>(rhs.core);
            }
            if (lhs.pipe != rhs.pipe) {
                return static_cast<uint64_t>(lhs.pipe) < static_cast<uint64_t>(rhs.pipe);
            }
            return static_cast<int>(lhs.aivCore) < static_cast<int>(rhs.aivCore);
        }
    };

    struct PipeCoreRealCompare {
        bool operator()(const PipeCoreReal& lhs, const PipeCoreReal& rhs) const
        {
            return ((static_cast<uint64_t>(lhs.core) << LEFT_OFFSET2) |
                    (static_cast<uint64_t>(lhs.pipe) << LEFT_OFFSET3)) <
                   ((static_cast<uint64_t>(rhs.core) << LEFT_OFFSET2) |
                    (static_cast<uint64_t>(rhs.pipe) << LEFT_OFFSET3));
        }
    };

    struct PipeCore {
        PipeCore(PipeType ps, PipeType pe, CoreType c, AIVCore a) : pipeStart(ps), pipeEnd(pe), core(c), aivCore(a) {}
        PipeType pipeStart;
        PipeType pipeEnd;
        CoreType core;
        AIVCore aivCore;

        bool operator==(const PipeCore& t) const
        {
            return (this->pipeStart == t.pipeStart && this->pipeEnd == t.pipeEnd && this->core == t.core &&
                    this->aivCore == t.aivCore);
        }

        bool operator!=(const PipeCore& t) const { return !(*this == t); }
    };

    struct PipeCoreCompare {
        bool operator()(const PipeCore& lhs, const PipeCore& rhs) const
        {
            return ((static_cast<uint64_t>(lhs.core) << LEFT_OFFSET1) |
                    (static_cast<uint64_t>(lhs.pipeStart) << LEFT_OFFSET4) |
                    (static_cast<uint64_t>(lhs.pipeEnd) << LEFT_OFFSET2) |
                    (static_cast<uint64_t>(lhs.aivCore) << LEFT_OFFSET3)) <
                   ((static_cast<uint64_t>(rhs.core) << LEFT_OFFSET1) |
                    (static_cast<uint64_t>(rhs.pipeStart) << LEFT_OFFSET4) |
                    (static_cast<uint64_t>(rhs.pipeEnd) << LEFT_OFFSET2) |
                    (static_cast<uint64_t>(rhs.aivCore) << LEFT_OFFSET3));
        }
    };

    using PipePair = std::pair<PipeCoreReal, PipeCoreReal>; // setPipe, waitPipe
    using PipePairEx = std::pair<PipeCoreRealEx, PipeCoreRealEx>;
    using CoreTypeDetail = std::pair<CoreType, AIVCore>;
    using CorePair = std::pair<CoreTypeDetail, CoreTypeDetail>;

    struct PipePairHash {
        std::size_t operator()(const PipePair& pp) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, pp.first.pipe);
            HashCombine(res, pp.first.core);
            HashCombine(res, pp.second.pipe);
            HashCombine(res, pp.second.core);
            return res;
        };
    };

    struct PipePairExHash {
        std::size_t operator()(const PipePairEx& pp) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, pp.first.pipe);
            HashCombine(res, pp.first.core);
            HashCombine(res, pp.first.aivCore);
            HashCombine(res, pp.second.pipe);
            HashCombine(res, pp.second.core);
            HashCombine(res, pp.second.aivCore);
            return res;
        };
    };

    struct CorePairHash {
        std::size_t operator()(const CorePair& pp) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, pp.first.first);
            HashCombine(res, pp.first.second);
            HashCombine(res, pp.second.first);
            HashCombine(res, pp.second.second);
            return res;
        };
    };

    struct IndexVecHash {
        std::size_t operator()(const std::pair<size_t, size_t>& pp) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, pp.first);
            HashCombine(res, pp.second);
            return res;
        };
    };

    struct DepOp {
        DepOp(size_t i, PipeCore pipeCore) : idx(i), selfPipeCore(pipeCore) {}
        size_t idx = SIZE_MAX;       // idx in oplog
        size_t idxInPipe = SIZE_MAX; // idx in the pipe belonging to
        PipeCore selfPipeCore;
        bool issued{false};

        std::vector<size_t> setPipe;  // this op will set_flag for op in setPipe; 后
        std::vector<size_t> waitPipe; // this op will wait_flag for op in waitPipe; 前
        std::string DumpDepOp(const std::vector<Operation*>& opLog);
    };

    struct IssueQueue {
        explicit IssueQueue(PipeCoreRealEx pipe) : selfPipeCore(pipe) {}
        PipeCoreRealEx selfPipeCore;
        size_t currOp{0};
        std::vector<size_t> ops;
        std::string DumpIssueQueue(const std::vector<Operation*>& opLogPtr);
    };

    struct PipeDepInfo {
        size_t waitIdx;
        std::map<PipeCoreRealEx, size_t, PipeCoreRealExCompare> setPipes;
        std::string DumpPipeDepInfo();
    };

    struct DataDepInfo {
        PipeType setp;
        CoreType setc;
        PipeType waitp;
        CoreType waitc;
        AIVCore setaivc;
        AIVCore waitaivc;
        std::vector<int> setOpIdList{};      // 对应sync_src/cv_sync_src在syncedOpLog中的idx
        std::vector<int> setOpEventIdList{}; // eventid
        // sync_src/cv_sync_src对应的setop和waitop的idx pair {setop idx, waitop idx}
        std::vector<std::pair<int, int>> opDepList{};
        std::string DumpDataDepInfo(const std::vector<IndexOp>& syncedOpLog, const std::vector<Operation*>& oriOpList);
    };

    struct IssueNum {
        // max op can be issued this round
        std::unordered_map<PipePairEx, size_t, PipePairExHash> maxIssueNum;
        // already issued op this round
        std::unordered_map<PipePairEx, size_t, PipePairExHash> currIssueNum;
        // max cv op can be issued this round
        std::unordered_map<CorePair, size_t, CorePairHash> maxCvIssueNum;
        // already issued cv op this round
        std::unordered_map<CorePair, size_t, CorePairHash> currCvIssueNum;
    };

    // HandleEventID 辅助结构：封装处理上下文
    struct EventIdProcessContext {
        DepOp& op;
        size_t eleIdx;
        IssueQueue& issueQ;
        IssueNum& issuenum;
        std::vector<IndexOp>& syncedOpLog;
        bool deadlock;
        bool eventIdOk{true};
        bool failedFlag{false};
    };

    struct EventResource {
        int eventId;
        CoreTypeDetail srcCore;
        CoreTypeDetail dstCore;
        PipeType srcPipe;
        PipeType dstPipe;

        bool operator==(const EventResource& other) const
        {
            return eventId == other.eventId && srcCore == other.srcCore && dstCore == other.dstCore &&
                   srcPipe == other.srcPipe && dstPipe == other.dstPipe;
        }
    };

    struct EventResourceHash {
        std::size_t operator()(const EventResource& er) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, er.eventId);
            HashCombine(res, er.srcCore.first);
            HashCombine(res, er.srcCore.second);
            HashCombine(res, er.dstCore.first);
            HashCombine(res, er.dstCore.second);
            HashCombine(res, er.srcPipe);
            HashCombine(res, er.dstPipe);
            return res;
        }
    };

    struct PipeCoreRealExHash {
        std::size_t operator()(const PipeCoreRealEx& p) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, p.pipe);
            HashCombine(res, p.core);
            HashCombine(res, p.aivCore);
            return res;
        }
    };

    std::string PipeSeqName(PipeSeq seq) const;
    PipeSeq GetPipeSeq(PipeCoreRealEx pipe);
    PipeCoreRealEx GetPipeFromSeq(PipeSeq seq);
    Status PipeDispatch(const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog);
    Status AdjustCopyInCfg(TileOpCfg& opcfg, const Operation& op);
    Status AdjustCopyOutCfg(TileOpCfg& opcfg, const Operation& op);
    Status AdjustOpCfg(TileOpCfg& opcfg, const Operation& op);
    void InitIssueQueue();
    void EnqueueOp(DepOp& op, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog);
    void RemoveOpDep(DepOp& setOp, DepOp& waitOp) const;
    void AddPhaseOp1(Function& function, const std::vector<Operation*>& srcLog, std::vector<Operation*>& dstLog,
                     size_t& i, size_t& prerun);
    void AddPhaseOp2(Function& function, std::vector<Operation*>& dstLog, size_t& prerun);
    Status AddOpDep(DepOp& setOp, DepOp& waitOp, bool isMergeCvSyncBase = false);
    Status AdjustOpDep(DepOp& op, size_t waitOpIdx, IssueQueue& issueQ, bool& failedFlag);
    Status HandleEventID(DepOp& op, IssueQueue& issueQ, IssueNum& issuenum, bool& deadlock, bool& res,
                         std::vector<IndexOp>& syncedOpLog);
    Status ProcessEventIdElement(EventIdProcessContext& ctx);
    Status ProcessSameCoreCase(const PipePairEx& pp, EventIdProcessContext& ctx);
    Status PopFromQueue(IssueQueue& issueQ, std::vector<size_t>& poped, bool& deadlock,
                        std::vector<IndexOp>& syncedOpLog);
    Status InjectWaitFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog);
    Status InjectSetFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog);
    Status InjectSync(Function& function, const std::vector<Operation*>& opLogPtr, size_t idx,
                      std::vector<IndexOp>& syncedOpLog);
    Status IssueOpPipeSeq(Function& function, const std::vector<Operation*>& opLogPtr,
                          std::vector<IndexOp>& syncedOpLog, bool& eventIdDeadlock, size_t& issued);
    Status IssueSyncOp(Function& function, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog,
                       size_t& totalIssued, size_t& allIssued);
    Status IssueOp(Function& function, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog);
    Status ProcessDeadLock(uint64_t& eventIdDeadlockEnterTimes, bool& eventIdDeadlock,
                           std::vector<IndexOp>& syncedOpLog);
    Status SynDependency(int maxOverlapDepIdx, const DataDepInfo& depInfo, const PipePairEx& pipePairEx,
                         std::vector<IndexOp>& syncedOpLog);
    Status GetDepInfo(std::vector<IndexOp>& syncedOpLog, const PipePairEx& pipePairEx, DataDepInfo& depInfo);
    Status RelaxFakeDataDep(std::vector<IndexOp>& syncedOpLog);
    bool CheckIssuedOp(const DepOp& op);
    bool ConstructDepInfo(DataDepInfo& depInfo, std::vector<IndexOp>& syncedOpLog, int i);
    bool FindDataDep(DataDepInfo& depInfo, std::vector<IndexOp>& syncedOpLog, int i);
    bool FindMaxOverlap(DataDepInfo& depInfo, int& maxOverlapDepIdx);
    bool GenSyncOp(PipeCoreRealEx set, PipeCoreRealEx wait, int eventId, bool isSet, Operation& op);
    Status GetEventId(const PipePairEx& pp, int& eventId, size_t setIdx, size_t& waitIdx,
                      std::vector<IndexOp>& syncedOpLog, Function& function);
    bool HasFreeEventId(const PipePairEx& pp);
    bool BufOverlap(const TileRange& range1, const TileRange& range2) const;
    bool CheckWawDependency(const Operation& opSet, const Operation& opWait);
    bool CheckRawDependency(const Operation& opSet, const Operation& opWait);
    bool CheckWarDependency(const Operation& opSet, const Operation& opWait);
    bool HasDataDependency(const Operation& opSet, const Operation& opWait);
    void UpdateDep(DepOp& currOp, DepOp& prevOp);
    bool IgnorableIntraPipeDep(size_t prev, size_t curr, const std::vector<Operation*>& opLogPtr);
    void FindDep(DepOp& op, const std::vector<Operation*>& opLogPtr, size_t idx,
                 DataDependencySearcher& dataDependencySearcher);
    void InitCVEventIdQ();
    Status RecycleCrossCoreEventIds(const PipeCoreRealEx& currPipeRealEx);
    static void PushEventIdIfAbsent(std::deque<int>& queue, int eventId);
    Status UpdateSyncArriveStatus(int eventId, const PipeCore& setPipe, const PipeCore& currPipe,
                                  const PipeCoreRealEx& setPipeRealEx, const PipeCoreRealEx& currPipeRealEx);
    void RemoveEventResourceFromSyncArriveStatus(const PipePairEx& pp, int eventId);
    std::deque<int>* GetCrossCoreEventIdQPtr(const PipePairEx& pp);
    void RemoveEventIdFromCrossCoreQueues(int eventId);
    void RemoveSetIntraBlockAndOpDep(std::vector<IndexOp>& syncedOpLog);
    void AddCrossCoreForceSyncOps(size_t setIdx, std::vector<IndexOp>& syncedOpLog, Function& function);
    void CreateForceSyncOp(Opcode opcode, PipeType pipe, CoreType core, AIVCore aivCore, uint64_t& insertIdx,
                           Function& function, std::vector<IndexOp>& syncedOpLog);
    void CreateBarAllOp(AIVCore aivCore, uint64_t& insertIdx, Function& function, std::vector<IndexOp>& syncedOpLog);
    std::deque<int>& GetFreeEventIdQueue(const PipePairEx& pp);
    int GetSyncSrcLogIdx(const std::vector<IndexOp>& syncedOpLog, int i);
    int GetMaxEventId(const PipePairEx& pp);
    std::string DumpLatestPipeDepMap();
    void BuildTensorRangeMap(Operation* op);
    bool IsMixCvCore(CoreType coreOne, CoreType coreTwo);

    std::vector<DepOp> depOps_;
    // Cube: MTE2, MTE1, M, FIX, Vector: MTE2, V, MTE3
    std::vector<IssueQueue> issueState_;
    std::unordered_map<PipePairEx, std::deque<int>, PipePairExHash> freeEventId_;

    // set pipe可用的eventid
    std::unordered_map<PipeCoreRealEx, std::array<std::deque<int>, NUM2>, PipeCoreRealExHash> crossCoreFreeEventId_;
    std::unordered_map<PipeCoreRealEx, std::unordered_set<EventResource, EventResourceHash>, PipeCoreRealExHash>
        syncArriveStatus;
    std::vector<std::pair<std::pair<size_t, size_t>, Operation*>> NoWaitCVPairs_;

    std::unordered_map<std::pair<size_t, size_t>, int, IndexVecHash> setWaitPairMap_;
    std::map<PipeCoreRealEx, PipeDepInfo, PipeCoreRealExCompare> latestPipeDep_;
    static std::map<PipeCoreRealEx, PipeSeq, PipeCoreRealExCompare> pipe2Seq;
    static std::map<PipeSeq, PipeCoreRealEx> seq2pipe;
    static std::vector<PipePair> dataDepPair;
    static std::vector<PipePair> mixCVDataDepPair;
    static std::vector<PipeCoreRealEx> cvPipeCoreEx;

    static constexpr int EVENT_NUM = 8;
    static constexpr int CROSS_CORE_EVENT_NUM = 16;
    static constexpr int EVENT_ID7 = 7;
    static constexpr int INTRA_CORE_START_EVENT_ID = 1;                            // 核内 eventid 使用 1 ~ 7
    std::unordered_map<PipePairEx, std::vector<int>, PipePairExHash> doublePipeOp; // pipepair, opmagic
    std::queue<size_t> orderedOpList_;
    std::vector<Operation*> oriOpList_;
    std::unordered_map<int, TileRange> ubTensorRangeMap;
    IRBuilder irBuilder_;
};

class InsertSync : public Pass {
public:
    InsertSync() : Pass("InsertSync") {}
    ~InsertSync() override {}
    void SetEnableDebug(bool enableDebug) { enableDebug_ = enableDebug; }

private:
    Status RunOnFunction(Function& function) override;
    void InsertCvPipeAll(Function* subGraphFunc);
    void InsertCvSyncOps(Function* subGraphFunc, std::vector<Operation*>& newOpList);
    Status GenNewOpList(Function* subGraphFunc, std::vector<Operation*>& opListNew);
    Status CheckNewOpListSeq(const std::vector<Operation*>& oriOpList, const std::vector<Operation*>& opListNew);
    Status InsertSyncMainLoop(Function* subGraphFunc);
    bool enableDebug_{false};
    IRBuilder irBuilder_;
};
} // namespace tile_fwk
} // namespace npu

#endif // PASS_INSERT_SYNC_H
