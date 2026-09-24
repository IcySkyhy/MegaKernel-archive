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
 * \file operation.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_set>
#include <variant>
#include <nlohmann/json.hpp>
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"
#include "interface/utils/common.h"
#include "opcode.h"
#include "attribute.h"
#include "attr_holder.h"
#include "interface/tensor/logical_tensor.h"
#include "operation_common.h"
#include "ir/stmt.h"

using Json = nlohmann::json;
using namespace pypto;

namespace npu::tile_fwk {
constexpr size_t NON_GROUP = -1;
constexpr int32_t TILE_STR_PREFIX_LEN = 5;

#define AICPU_CALL_NUM_COPYOUT_RESOLVE 1
#define AICPU_CALL_NUM_BIT 16
#define AICPU_CALL_ARG_BIT 16
#define AICPU_CALL_TASK_BIT 32

class OpAttributeKey {
public:
    static const std::string aicpuCall;
    static const std::string scalar;
    static const std::string vectorScalar;
    static const std::string dynScalar;
    static const std::string isGlobalInput;
    static const std::string seqNo;
    static const std::string color;
    static const std::string isCube;
    static const std::string blockPadding;
    static const std::string broadcastLastAxis;
    static const std::string tilePadding;
    static const std::string reshapePadding;
    static const std::string shapePadded;
    static const std::string needAlloc;
    static const std::string dontTouch;
    static const std::string tag;
    static const std::string distTilingInfo;
    static const std::string sameInOut;
    static const std::string expandDims;
    static const std::string inputCombineAxis;
    static const std::string outputCombineAxis;
    static const std::string inputCombineAxisDone;
    static const std::string outputCombineAxisDone;
    static const std::string inplaceIdx;
    static const std::string inplaceInfo;
    static const std::string cacheMode;
    static const std::string panzBlockSize;
    static const std::string requiresBoundaryCopy;
    static const std::string excludeBufferReuse;
    static const std::string bindTensor;
    static const std::string startOffset;
    static const std::string distOpAttr;
    static const std::string isDistCopyOut;
    static const std::string subBlockIdx;
    static const std::string accumulate;
    static const std::string indicesSize;
    static const std::string brcbIdx;
    static const std::string brcOperand;
    static const std::string topkAlgo;
    static const std::string quantFlag;
    static const std::string loopGroup;
    static const std::string loopAxes;
    static const std::string loopGroupStart;
    static const std::string loopGroupEnd;
    static const std::string dynloopGroup;
    static const std::string dynloopAxes;
    static const std::string dynloopGroupStart;
    static const std::string dynloopGroupEnd;
    static const std::string lastUse;
    static const std::string isUpper;
    static const std::string blockSize;
    static const std::string transMode;
    static const std::string workspaceBaseOffset;
    static const std::string copyInMode;
    static const std::string copyOutMode;
    static const std::string localCopyLocalMode;
    static const std::string copyIsNZ;
    static const std::string isGemv;
    static const std::string scaleValue;
    static const std::string rowPad;
    static const std::string ownerRank;
    static const std::string maxTileNum;
    static const std::string perRankDataShape;
    static const std::string perRankTileNum;
    static const std::string totalTileNum;
    static const std::string precisionType;
    static const std::string perm;
    static const std::string mxQuantMode;
    static const std::string mxQuantAxis;
    static const std::string mxQuantPerformanceMode;
    static const std::string gmTensorParamIdxInCall;
    static const std::string staticValidShape;
    // HashOrder attribute keys
    static const std::string l1ReuseHashOrder;
    static const std::string l1ReuseSubgraphCount;
    static const std::string cubeMergeHashOrder;
    static const std::string cubeMergeSubgraphCount;
    static const std::string vecMergeHashOrder;
    static const std::string vecMergeSubgraphCount;
    static const std::string atomicAdd;
    static const std::string splitMN; // for L0C->UB Dual Dst
    static const std::string reduceCopyPreSubgraphId;
    // HashOrder attribute keys for conv
    static const std::string postK;
    static const std::string postM;
    static const std::string postN;
    static const std::string filterH;
    static const std::string filterW;
    static const std::string strideH;
    static const std::string strideW;
    static const std::string dilationH;
    static const std::string dilationW;
    static const std::string paddingLeft;
    static const std::string paddingRight;
    static const std::string paddingTop;
    static const std::string paddingBottom;
    static const std::string padValue;
    static const std::string repeatStride;
    static const std::string repeatTime;
    static const std::string wStride;
    static const std::string srcGmConvValidShape;
    static const std::string l0cValidMN;
    static const std::string rmwMode;
    static const std::string transDataOffset;
    static const std::string isConv;
    static const std::string writeConflict;
    static const std::string parallel;
};

const std::string CONV_GROUPS_ATTR = "op_attr_groups";
const std::string FAKE_TRANS_IN_FORMAT_ATTR = "op_attr_fake_trans_in_format";
const std::string FAKE_TRANS_OUT_FORMAT_ATTR = "op_attr_fake_trans_out_format";

class ConvOpAttributeKey {
public:
    static const std::string cin;
    static const std::string cout;
    static const std::string paddingLeft;
    static const std::string paddingTop;
    static const std::string paddingRight;
    static const std::string paddingBottom;
    static const std::string strideh;
    static const std::string stridew;
    static const std::string hposX;
    static const std::string hsteP;
    static const std::string wposX;
    static const std::string wstep;
    static const std::string hoffsetY;
    static const std::string woffsetY;
    static const std::string reluType;
    static const std::string reluAlpha;
    static const std::string clearFlag;
    static const std::string hasAccFlag;
    static const std::string hasEltFlag;
    static const std::string hasBiasFlag;
    static const std::string eltBrcbFlag;
    static const std::string fmapSrcNum;
    static const std::string eltMode;
    static const std::string fmapC0;
};

class FixpOpAttributeKey {
public:
    static const std::string hStart;
    static const std::string hEnd;
    static const std::string quantPreScalar;
    static const std::string quantPostScalar;
    static const std::string antiqScalar;
    static const std::string hasQuantPreVector;
    static const std::string hasQuantPostVector;
    static const std::string hasAntiqVector;
    static const std::string fbAddrSpace;
};

class PoolOpAttributeKey {
public:
    static const std::string poolh;
    static const std::string poolw;
};

class TensorAttributeKey {
public:
    static const std::string tensorAddr;
};

enum class FbBufferSpace { QUANT_PRE = 0, RELU_PRE, RELU_POST, QUANT_POST, ANTIQ_ELT, ANTIQ_MTE2 };

enum class AIVCore {
    UNSPECIFIED = -1, // 未指定或非Vector组件
    AIV0 = 0,         // 在AIVE0核上执行
    AIV1 = 1          // 在AIVE1核上执行
};

inline const BiMap<AIVCore>& GetAIVCoreDict()
{
    static BiMap<AIVCore> dict{{
        {AIVCore::UNSPECIFIED, "AIC"},
        {AIVCore::AIV0, "AIV0"},
        {AIVCore::AIV1, "AIV1"},
    }};
    return dict;
}

struct OperandAttribute {
    OperandAttribute() : offset(-1), attr(0) {}
    OperandAttribute(int off_in, int attr_in = 0) : offset(off_in), attr(attr_in) {}
    bool isAtomic() const { return attr & 0x1; }

    int offset;
    int attr;
};

class Function;
// Class to represent an operation (opcode) and its operands
class Operation : public ir::TensorOpStmt, public std::enable_shared_from_this<Operation>, public AttrHolder {
public:
    // MixSubgraphSplit相关字段的结构体
    struct MixSubgraphFields {
        int internalSubgraphID{NOT_IN_SUBGRAPH};
        AIVCore aivCore{AIVCore::UNSPECIFIED};
    };

    // sg_set_scope 相关字段的结构体
    struct ScopeInfo {
        int scopeId{-1};
        bool allowParallelMerge{false};
        bool allowCrossScopeMerge{false};
        int cvFuseId{-1}; // 仅由graph_partition标记

        ScopeInfo() = default;
        explicit ScopeInfo(int id) : scopeId(id) {}
        static ScopeInfo FromConfig(const std::vector<int64_t>& config)
        {
            ScopeInfo info;
            info.scopeId = static_cast<int>(config[0]);
            info.allowParallelMerge = static_cast<bool>(config[1]);
            info.allowCrossScopeMerge = static_cast<bool>(config[2]);
            return info;
        }
        void SetCvFuseId(int id) { cvFuseId = id; }
    };
    friend class Function;
    LogicalTensors iOperand;      // Input operands (now actual objects, not shared_ptr)
    LogicalTensors oOperand;      // Output operands (now actual objects, not shared_ptr)
    LogicalTensors dependOperand; // Depend Operands
    int opmagic;                  // The magic number for the operation, default value -1
    int programFuncMagic_;        // function magic of leafFunction
    int outcastRefcount{0};

    int cycles{0};
    int cycleStart{0};
    int cycleEnd{0};
    int cubeDepId{-1};

    std::vector<int> inParamLocation_;
    std::vector<int> outParamLocation_;
    OpSyncQueue syncQueue_;
    QueueType queueType;

    // Constructor to initialize the opcode, input operands, output operands, and opmagic
    Operation(Function& cur, Opcode opcode, LogicalTensors iOperands, LogicalTensors oOperands, int opMagic = -1);

    Operation(Function& cur, Opcode opcode) : Operation(cur, opcode, {}, {}) {}

    Operation(Function& cur, const std::string& op, const LogicalTensors& input, const LogicalTensors& output)
        : Operation(cur, FindOpcode(op), input, output)
    {
        if (op.substr(0, TILE_STR_PREFIX_LEN) == "TILE_") {
            isTileOp_ = true;
        }
    };

    Operation(const Operation& other) = delete;
    Operation(Operation&& other) = delete;
    Operation& operator=(const Operation& other) = delete;
    Operation& operator=(Operation&& other) = delete;

    Function* BelongTo() const { return function_; }

    const QueueType& GetQueueType() const { return queueType; }

    const OpSyncQueue& GetSyncQueue() const { return syncQueue_; }

    const TileShape& GetTileShape() const { return tileShape_; }
    void UpdateTileShape(const TileShape newTileShape) { tileShape_ = newTileShape; }

    TileShape& GetTileShapeForSetting() { return tileShape_; }

    [[nodiscard]] std::string GetStringAttribute(const std::string& key) const;
    void SetAttribute(const std::string& key, const std::string& value);

    [[nodiscard]] bool GetBoolAttribute(const std::string& key) const;
    void SetAttribute(const std::string& key, bool value);

    [[nodiscard]] int64_t GetIntAttribute(const std::string& key) const;
    void SetAttribute(const std::string& key, int64_t value);
    void SetAttribute(const std::string& key, int value) { SetAttribute(key, static_cast<int64_t>(value)); }

    [[nodiscard]] int GetInplaceIndex(size_t outputIndex) const;

    [[nodiscard]] Element GetElementAttribute(const std::string& key) const;
    std::vector<Element> GetVectorElementAttribute(const std::string& key) const;
    void SetAttribute(const std::string& key, Element value);

    template <typename T = int64_t>
    std::vector<T> GetVectorIntAttribute(const std::string& key) const
    {
        static_assert(std::is_integral_v<T>);
        std::vector<int64_t> val;
        GetAttr(key, val);
        if constexpr (std::is_same_v<T, int64_t>) {
            return val;
        }
        std::vector<T> ret;
        for (auto& x : val) {
            ret.emplace_back(static_cast<T>(x));
        }
        return ret;
    }

    template <typename T = int64_t>
    void SetAttribute(const std::string& key, const std::vector<T>& value)
    {
        static_assert(std::is_integral_v<T>);
        if constexpr (std::is_same_v<T, int64_t>) {
            SetAttr(key, value);
        } else {
            std::vector<int64_t> nvalue;
            for (auto& x : value) {
                nvalue.emplace_back(static_cast<int64_t>(x));
            }
            SetAttr(key, nvalue);
        }
    }

    [[nodiscard]] CastMode GetCastModeAttribute(const std::string& key) const;
    void SetAttribute(const std::string& key, CastMode value);

    [[nodiscard]] SymbolicScalar GetSymbolicScalarAttribute(const std::string& key) const;
    void SetAttribute(const std::string& key, const SymbolicScalar& value);

    [[nodiscard]] std::vector<SymbolicScalar> GetVectorSymbolicScalarAttribute(const std::string& key) const;
    void SetAttribute(const std::string& key, const std::vector<SymbolicScalar>& value);
    void SetAttribute(const std::string& key, const std::vector<Element>& value);

    [[nodiscard]] bool HasAttribute(const std::string& key) const { return HasAttr(key); }

    [[nodiscard]] std::map<std::string, std::any> GetAllAttribute() const;

    Json DumpJson(bool dumpTensor = true) const;
    static std::shared_ptr<Operation> LoadJson(
        Function& cur, const std::unordered_map<int, std::shared_ptr<LogicalTensor>>& tensorDict, const Json& opDump);

private:
    void UnlinkFromLogicalTensors();
    void DumpOperandsJson(Json& opDump, bool dumpTensor) const;
    void DumpCalleeHashJson(Json& opDump) const;
    void DumpLocationJson(Json& opDump) const;
    void DumpParamLocationJson(Json& opDump) const;
    void DumpCallOpInfoJson(Json& opDump) const;
    void DumpTileInfoJson(Json& opDump) const;
    void DumpAttributesJson(Json& opDump) const;

    static void LoadOperandsFromJson(const Json& opDump,
                                     const std::unordered_map<int, std::shared_ptr<LogicalTensor>>& tensorDict,
                                     std::vector<std::shared_ptr<LogicalTensor>>& ioperands,
                                     std::vector<std::shared_ptr<LogicalTensor>>& ooperands);
    void LoadLocationFromJson(const Json& opDump);
    void LoadBasicInfoFromJson(const Json& opDump);
    void LoadTileInfoFromJson(const Json& opDump);
    void LoadOpAttributeFromJson(const Json& opDump, Opcode opcode);
    void LoadExtraInfoFromJson(const Json& opDump);

public:
    [[nodiscard]] std::string DumpSSA(const std::string& prefix = "") const;

    [[nodiscard]] std::string Dump() const;

    [[nodiscard]] int GetOpMagic() const { return opmagic; }

    [[nodiscard]] const LogicalTensors& GetIOperands() const { return iOperand; }
    LogicalTensors& GetIOperands() { return iOperand; }

    [[nodiscard]] const LogicalTensors& GetOOperands() const { return oOperand; }
    LogicalTensors& GetOOperands() { return oOperand; }

    [[nodiscard]] const LogicalTensors& GetDependOperands() const { return dependOperand; }
    LogicalTensors& GetDependOperands() { return dependOperand; }

    void AddDependOperand(LogicalTensorPtr dependoperand);

    size_t GetInputOperandSize() const { return iOperand.size(); }

    size_t GetOutputOperandSize() const { return oOperand.size(); }

    size_t GetDependOperandSize() const { return dependOperand.size(); }

    LogicalTensorPtr GetInputOperand(const size_t index) const;

    LogicalTensorPtr GetOutputOperand(const size_t index) const;

    int GetIOperandIndex(const LogicalTensorPtr& ioperand) const;
    int GetOOperandIndex(const LogicalTensorPtr& ooperand) const;

    void ReplaceInputOperand(const LogicalTensorPtr& originInput, const LogicalTensorPtr& newInput);

    void ReplaceOutputOperand(const LogicalTensorPtr& originOutput, const LogicalTensorPtr& newOutput);

    void UpdateInputOperand(const size_t index, const std::shared_ptr<LogicalTensor>& newInput);

    void UpdateOutputOperand(const size_t index, const std::shared_ptr<LogicalTensor>& newOutput);

    std::unordered_set<Operation*> ConsumerOps() const;
    std::unordered_set<Operation*> ProducerOps() const;

    class OperationComparator {
    public:
        bool operator()(const Operation* lhs, const Operation* rhs) const
        {
            return lhs->GetOpMagic() < rhs->GetOpMagic();
        }
    };
    std::set<Operation*, OperationComparator> ConsumerOpsOrdered() const;
    std::set<Operation*, OperationComparator> ProducerOpsOrdered() const;

    std::unordered_set<Operation*> ConsumerOpsByToken() const;
    std::unordered_set<Operation*> ProducerOpsByToken() const;

    [[nodiscard]] const std::unordered_set<Operation*>& GetInCtrlOperations() const { return inputCtrlOps; }

    [[nodiscard]] const std::unordered_set<Operation*>& GetOutCtrlOperations() const { return outputCtrlOps; }

    void ClearInCtrlOperations() { inputCtrlOps.clear(); }

    void ClearOutCtrlOperations() { outputCtrlOps.clear(); }

    ScopeInfo scopeInfo_;
    void SetScopeId(int scopeId) { scopeInfo_.scopeId = scopeId; };
    void SetScopeInfo(const ScopeInfo& info) { scopeInfo_ = info; };
    const ScopeInfo& GetScopeInfo() const { return scopeInfo_; };
    int GetScopeId() const { return scopeInfo_.scopeId; };
    bool GetAllowParallelMerge() const { return scopeInfo_.allowParallelMerge; };
    bool GetAllowCrossScopeMerge() const { return scopeInfo_.allowCrossScopeMerge; };
    int GetCvFuseId() const { return scopeInfo_.cvFuseId; };

    int GetOooScopeId() const { return oooScopeId_; }
    void SetOooScopeId(int id) { oooScopeId_ = id; }

    void AddInCtrlOperation(Operation& operation);

    void RemoveInCtrlOperation(Operation& operation);

    void AddOutCtrlOperation(Operation& operation);

    void RemoveOutCtrlOperation(Operation& operation);

    Operation& CloneOperation(Function& func, const LogicalTensors& iOperandList,
                              const LogicalTensors& oOperandList) const;

    std::shared_ptr<Operation> CloneTensorOpStmt(const LogicalTensors& iOperandList, const LogicalTensors& oOperandList,
                                                 const std::vector<ir::VarPtr>& resultTokens,
                                                 const std::vector<ir::VarPtr>& tokens, ir::Span span,
                                                 Function* targetFunc = nullptr) const;

    [[nodiscard]] std::string GetOpcodeStr(bool appendTile = false) const;
    [[nodiscard]] CoreType GetCoreType() const { return coreType_; }
    void SetCoreType(CoreType ct) { coreType_ = ct; }
    [[nodiscard]] std::string GetCoreTypeStr() const;

    unsigned long ComputeHash();
    unsigned long ComputeHashOrderless() const;
    [[nodiscard]] bool IsCall() const;
    [[nodiscard]] bool IsNOP() const;

    bool IsIsolatedOp() const;

    bool OnlyHasCtrlEdgeToOp(Operation& op) const;

    const std::shared_ptr<OpAttribute>& GetOpAttribute() const { return opAttribute_; }
    std::shared_ptr<OpAttribute>& GetOpAttribute() { return opAttribute_; }

    void SetOpAttribute(const std::shared_ptr<OpAttribute>& attr)
    {
        opAttribute_ = attr;
        static std::unordered_set<Opcode> copyOpAttrOpTypes{Opcode::OP_L1_COPY_IN,
                                                            Opcode::OP_L1_COPY_OUT,
                                                            Opcode::OP_COPY_IN,
                                                            Opcode::OP_L0C_TO_L1,
                                                            Opcode::OP_L0C_COPY_UB,
                                                            Opcode::OP_L0C_COPY_UB_DUAL_DST,
                                                            Opcode::OP_L1_TO_BT,
                                                            Opcode::OP_L1_TO_FIX_QUANT_PRE,
                                                            Opcode::OP_L1_TO_L0A,
                                                            Opcode::OP_L1_TO_L0B,
                                                            Opcode::OP_L1_TO_L0_AT,
                                                            Opcode::OP_L1_TO_L0_BT,
                                                            Opcode::OP_UB_COPY_L1,
                                                            Opcode::OP_COPY_OUT,
                                                            Opcode::OP_RESHAPE_COPY_IN,
                                                            Opcode::OP_RESHAPE_COPY_OUT,
                                                            Opcode::OP_INDEX_OUTCAST,
                                                            Opcode::OP_INDEX_PUT,
                                                            Opcode::OP_INDEX_ADD,
                                                            Opcode::OP_TRANSPOSE_MOVEIN,
                                                            Opcode::OP_TRANSPOSE_MOVEOUT,
                                                            Opcode::OP_NCHW2NC1HWC0,
                                                            Opcode::OP_NCHW2Fractal_Z,
                                                            Opcode::OP_NC1HWC02NCHW,
                                                            Opcode::OP_NCDHW2NDC1HWC0,
                                                            Opcode::OP_NCDHW2FRACTAL_Z_3D,
                                                            Opcode::OP_NDC1HWC02NCDHW,
                                                            Opcode::OP_SHMEM_PUT,
                                                            Opcode::OP_SHMEM_STORE,
                                                            Opcode::OP_SHMEM_SIGNAL,
                                                            Opcode::OP_SHMEM_GET,
                                                            Opcode::OP_SHMEM_LOAD,
                                                            Opcode::OP_SHMEM_SET,
                                                            Opcode::OP_GATHER_IN_UB,
                                                            Opcode::OP_L1_COPY_IN_A_SCALE,
                                                            Opcode::OP_L1_COPY_IN_B_SCALE,
                                                            Opcode::OP_L1_TO_L0A_SCALE,
                                                            Opcode::OP_L1_TO_L0B_SCALE,
                                                            Opcode::OP_L1_COPY_IN_CONV,
                                                            Opcode::OP_L0C_COPY_OUT,
                                                            Opcode::OP_L0C_COPY_OUT_CONV,
                                                            Opcode::OP_L1_RESHAPE_COPY_IN,
                                                            Opcode::OP_L0C_RESHAPE_COPY_OUT};
        if (copyOpAttrOpTypes.count(opcode_) > 0) {
            FE_ASSERT(std::dynamic_pointer_cast<CopyOpAttribute>(opAttribute_) != nullptr);
            return;
        }

        switch (opcode_) {
            case Opcode::OP_VIEW:
            case Opcode::OP_SLICE: {
                FE_ASSERT(std::dynamic_pointer_cast<ViewOpAttribute>(opAttribute_) != nullptr);
                break;
            }
            case Opcode::OP_ASSEMBLE:
            case Opcode::OP_CONTRACT: {
                FE_ASSERT(std::dynamic_pointer_cast<AssembleOpAttribute>(opAttribute_) != nullptr);
                break;
            }
            case Opcode::OP_ASSEMBLE_SSA:
            case Opcode::OP_ATOMIC_RMW:
                FE_ASSERT(std::dynamic_pointer_cast<AssembleOpAttribute>(opAttribute_) != nullptr ||
                          std::dynamic_pointer_cast<CopyOpAttribute>(opAttribute_) != nullptr);
                break;
            case Opcode::OP_BLOCK_CALL:
            case Opcode::OP_CALL: {
                FE_ASSERT(std::dynamic_pointer_cast<CallOpAttribute>(opAttribute_) != nullptr);
                break;
            }
            case Opcode::OP_CONVERT: {
                FE_ASSERT(std::dynamic_pointer_cast<ConvertOpAttribute>(opAttribute_) != nullptr);
                break;
            }
            default:
                FE_ASSERT(opAttribute_ == nullptr);
        }
    }

    void SetAssembleOpAttribute(const std::vector<int64_t>& toOffset,
                                const std::vector<SymbolicScalar>& toDynOffset = {})
    {
        FE_ASSERT(opcode_ == Opcode::OP_ASSEMBLE || opcode_ == Opcode::OP_CONTRACT ||
                  opcode_ == Opcode::OP_ASSEMBLE_SSA || opcode_ == Opcode::OP_ATOMIC_RMW);
        SetOpAttribute(std::make_shared<AssembleOpAttribute>(toOffset, toDynOffset));
    }

    void ReplaceIOperand(size_t index, std::shared_ptr<LogicalTensor> newTensor);
    void ReplaceOOperand(size_t index, std::shared_ptr<LogicalTensor> newTensor);

    std::string GetCalleeMagicName() const
    {
        FE_ASSERT(IsCall());
        return std::static_pointer_cast<CallOpAttribute>(opAttribute_)->GetCalleeMagicName();
    }

    const std::string& GetCalleeBracketName() const
    {
        return std::static_pointer_cast<CallOpAttribute>(opAttribute_)->GetCalleeBracketName();
    }

    const FunctionHash& GetCalleeHash() const
    {
        FE_ASSERT(IsCall() || opcode_ == Opcode::OP_BLOCK_CALL);
        auto callop = std::dynamic_pointer_cast<CallOpAttribute>(opAttribute_);
        return callop->GetCalleeHash();
    }

    void EraseInput(const std::shared_ptr<LogicalTensor>& input);
    void EraseDependTensor(const std::shared_ptr<LogicalTensor>& dependTensor);
    void ReplaceInput(const std::shared_ptr<LogicalTensor>& newInput, const std::shared_ptr<LogicalTensor>& oldInput);
    void ReplaceOutput(const std::shared_ptr<LogicalTensor>& newOutput,
                       const std::shared_ptr<LogicalTensor>& oldOutput);

    Opcode GetOpcode() const { return opcode_; }

    void SetOpCode(Opcode opcode)
    {
        opcode_ = opcode;
        TensorOpStmt::opcode_ = OpcodeManager::Inst().GetOpcodeStr(opcode);
    }
    int GetLatency() const { return latency_; }
    void UpdateLatency(int latency) { latency_ = latency; }

    int GetRemainingTime() const { return remainingTime_; }
    void UpdateRemainingTime(int remainingTime) { remainingTime_ = remainingTime; }

    int GetSubgraphID() const { return subgraphID_; }
    int GetInternalSubgraphID() const
    {
        return mixSubgraphFields_ ? mixSubgraphFields_->internalSubgraphID : NOT_IN_SUBGRAPH;
    }
    void UpdateSubgraphID(int subgraphID) { subgraphID_ = subgraphID; }
    // HashOrder attribute setters/getters using attribute mechanism
    void SetHashOrderInfo(const std::string& hashOrderKey, const std::string& countKey, const std::string& hashOrder,
                          size_t subgraphCount)
    {
        SetAttr(hashOrderKey, hashOrder);
        SetAttr(countKey, static_cast<int64_t>(subgraphCount));
    }
    std::pair<std::string, size_t> GetHashOrderInfo(const std::string& hashOrderKey, const std::string& countKey) const
    {
        std::string hashOrder;
        int64_t count = 0;
        GetAttr(hashOrderKey, hashOrder);
        GetAttr(countKey, count);
        return {hashOrder, static_cast<size_t>(count)};
    }
    void UpdateInternalSubgraphID(int internalSubgraphID)
    {
        ensureMixSubgraphFields();
        mixSubgraphFields_->internalSubgraphID = internalSubgraphID;
    }

    auto GroupID() const { return groupID_; }
    void SetGroupID(size_t groupID) const { groupID_ = groupID; }

    void SetSemanticLabel(std::shared_ptr<SemanticLabel> label) { semanticLabel_ = label; }
    const std::string& GetSemanticLabelStr() const
    {
        static std::string empty = "";
        return semanticLabel_ ? semanticLabel_->label : empty;
    }
    std::shared_ptr<SemanticLabel> GetSemanticLabel() const { return semanticLabel_; }

    void SetAsDeleted() { isDeleted_ = true; }
    void SetAsNotDeleted() { isDeleted_ = false; }
    [[nodiscard]] bool IsDeleted() const { return isDeleted_; }

    [[nodiscard]] AIVCore GetAIVCore() const
    {
        return mixSubgraphFields_ ? mixSubgraphFields_->aivCore : AIVCore::UNSPECIFIED;
    }
    [[nodiscard]] std::string GetAIVCoreStr() const
    {
        if (!mixSubgraphFields_) {
            return "UNKNOWN";
        }
        return GetAIVCoreDict().Find(mixSubgraphFields_->aivCore);
    }
    void SetAIVCore(AIVCore aivCore)
    {
        ensureMixSubgraphFields();
        mixSubgraphFields_->aivCore = aivCore;
    }

    void SetSubFuncInvokeInfo(const SubfuncInvokeInfoTy& invokeInfo);

    SubfuncInvokeInfoTy& GetSubFuncInvokeInfo()
    {
        auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(opAttribute_);
        FE_ASSERT(callAttr != nullptr);
        return *(callAttr->invokeInfo_);
    }

    int GetProgramId();

    bool IsNeedStackGM() const;

    int GetIOpAttrOffset(int pos) const { return iOpAttr_.empty() ? -1 : iOpAttr_[pos].offset; }
    int GetOOpAttrOffset(int pos) const { return oOpAttr_.empty() ? -1 : oOpAttr_[pos].offset; }
    OperandAttribute GetIOpAttr(int pos) const { return iOpAttr_.empty() ? OperandAttribute() : iOpAttr_[pos]; }
    OperandAttribute GetOOpAttr(int pos) const { return oOpAttr_.empty() ? OperandAttribute() : oOpAttr_[pos]; }
    void SetIOpAtt(int pos, int offset)
    {
        if (iOpAttr_.empty()) {
            iOpAttr_.resize(iOperand.size());
        }
        iOpAttr_[pos].offset = offset;
    }
    void SetOOpAtt(int pos, int offset, int attr = 0)
    {
        if (oOpAttr_.empty()) {
            oOpAttr_.resize(oOperand.size());
        }
        if (pos < (int)oOpAttr_.size()) {
            oOpAttr_[pos] = {offset, attr};
        }
    }
    const std::vector<OperandAttribute>& GetIOpAttr() const { return iOpAttr_; }
    const std::vector<OperandAttribute>& GetOOpAttr() const { return oOpAttr_; }
    void SetOperandAttr(const std::vector<OperandAttribute>& iAttr, const std::vector<OperandAttribute>& oAttr)
    {
        iOpAttr_ = iAttr;
        oOpAttr_ = oAttr;
    }

    std::vector<std::reference_wrapper<SymbolicScalar>> GetDynamicAttributeList();
    const ir::Span& GetSpan() const { return span_; }
    void SetSpan(ir::Span span) { span_ = span; }
    const std::vector<std::string>& GetCommentList() const { return commentList_; }
    std::vector<std::string>& GetCommentList() { return commentList_; }

private:
    int oooScopeId_{-1};

    void InitCoreTypeAndTileShape(Opcode opcode);
    void InitTensorGraphMetadata();
    void InitLatency(Opcode opcode);

    Opcode opcode_{Opcode::OP_UNKNOWN};
    int subgraphID_{NOT_IN_SUBGRAPH};
    bool isTileOp_{false};
    TileShape tileShape_;
    std::shared_ptr<OpAttribute> opAttribute_;
    unsigned long operationHash_{0};
    int latency_{1};

    std::vector<OperandAttribute> iOpAttr_;
    std::vector<OperandAttribute> oOpAttr_;
    int remainingTime_{INVALID_TIME};
    CoreType coreType_{CoreType::MIX};
    std::unordered_set<Operation*> inputCtrlOps;
    std::unordered_set<Operation*> outputCtrlOps;
    mutable size_t groupID_{NON_GROUP};
    bool isDeleted_{false};

    std::shared_ptr<SemanticLabel> semanticLabel_;
    Function* function_;

    std::vector<std::string> commentList_;
    std::unique_ptr<MixSubgraphFields> mixSubgraphFields_;
    void ensureMixSubgraphFields()
    {
        if (!mixSubgraphFields_) {
            mixSubgraphFields_ = std::make_unique<MixSubgraphFields>();
        }
    }
};
using OperationPtr = std::shared_ptr<Operation>;

// Custom comparator for Operation in magic order
struct OperationCmp {
    bool operator()(const Operation* lhs, const Operation* rhs) const;
};

/*  ！！！！！！！！！对外开放OP接口请在 tilefwk_op.h 中添加 ！！！！！！！！！！！！！！！！！！！！*/

} // namespace npu::tile_fwk
