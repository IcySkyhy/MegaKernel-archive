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
 * \file operation.cpp
 * \brief
 */

#include "operation.h"

#include <functional>

#include "tilefwk/data_type.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/opcode.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/inner/tile_shape.h"
#include "interface/utils/error.h"
#include "interface/program/program.h"
#include "interface/operation/cycles.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/utils/serialization.h"
#include "passes/pass_utils/subfunc_utils.h"
#include "interface/configs/config_manager_ng.h"

namespace npu::tile_fwk {
const std::string OpAttributeKey::aicpuCall = "AICPU_CALL";
const std::string OpAttributeKey::color = "COLOR";
const std::string OpAttributeKey::scalar = "SCALAR";
const std::string OpAttributeKey::dynScalar = "DYN_SCALAR";
const std::string OpAttributeKey::vectorScalar = "VECTORSCALAR";
const std::string OpAttributeKey::isGlobalInput = "IS_GLOBAL_INPUT";
const std::string OpAttributeKey::seqNo = "SEQ_NO";
const std::string OpAttributeKey::isCube = "IS_CUBE";
const std::string OpAttributeKey::blockPadding = "BLOCK_PADDING";
const std::string OpAttributeKey::tilePadding = "TILE_PADDING";
const std::string OpAttributeKey::reshapePadding = "RESHAPE_PADDING";
const std::string OpAttributeKey::shapePadded = "SHAPE_PADDED";
const std::string OpAttributeKey::needAlloc = "NEED_ALLOC";
const std::string OpAttributeKey::broadcastLastAxis = "BROADCAST_LAST_AXIS";
const std::string OpAttributeKey::dontTouch = "DONT_TOUCH";
const std::string OpAttributeKey::tag = "TAG";
const std::string OpAttributeKey::distTilingInfo = "DIST_TILING_INFO";
const std::string OpAttributeKey::sameInOut = "SAME_IN_OUT";
const std::string OpAttributeKey::expandDims = "op_attr_expand_dims";
const std::string OpAttributeKey::inputCombineAxis = "op_attr_input_combine_axis";
const std::string OpAttributeKey::outputCombineAxis = "op_attr_output_combine_axis";
const std::string OpAttributeKey::inplaceIdx = "INPLACE_IDX";
const std::string OpAttributeKey::inplaceInfo = "INPLACE_INFO";
const std::string OpAttributeKey::cacheMode = "CACHE_MODE";
const std::string OpAttributeKey::panzBlockSize = "PA_NZ_BLOCK_SIZE";
const std::string OpAttributeKey::inputCombineAxisDone = "input_combine_axis_done";   // only for flow verify tool
const std::string OpAttributeKey::outputCombineAxisDone = "output_combine_axis_done"; // flow verify tool only
const std::string OpAttributeKey::requiresBoundaryCopy = "requires_boundary_copy";
const std::string OpAttributeKey::excludeBufferReuse = "exclude_buffer_reuse";
const std::string OpAttributeKey::bindTensor = "BIND_TENSOR";
const std::string OpAttributeKey::startOffset = "start_offset";
const std::string OpAttributeKey::distOpAttr = "DIST_OP_ATTR";
const std::string OpAttributeKey::isDistCopyOut = "IS_DIST_COPY_OUT";
const std::string OpAttributeKey::subBlockIdx = "SUB_BLOCK_IDX";
const std::string OpAttributeKey::accumulate = "accumulate";
const std::string OpAttributeKey::indicesSize = "indicesSize";
const std::string OpAttributeKey::brcbIdx = "brcb_idx";
const std::string OpAttributeKey::brcOperand = "op_attr_brc_operand";
const std::string OpAttributeKey::topkAlgo = "topk_algo";
const std::string OpAttributeKey::quantFlag = "op_attr_vector_quant_flag";
const std::string OpAttributeKey::loopGroup = "LOOP_GROUP";
const std::string OpAttributeKey::loopAxes = "LOOP_AXES";
const std::string OpAttributeKey::loopGroupStart = "LOOP_GROUP_START";
const std::string OpAttributeKey::loopGroupEnd = "LOOP_GROUP_END";
const std::string OpAttributeKey::dynloopGroup = "DYN_LOOP_GROUP";
const std::string OpAttributeKey::dynloopAxes = "DYN_LOOP_AXES";
const std::string OpAttributeKey::dynloopGroupStart = "DYN_LOOP_GROUP_START";
const std::string OpAttributeKey::dynloopGroupEnd = "DYN_LOOP_GROUP_END";
const std::string OpAttributeKey::lastUse = "last_use";
const std::string OpAttributeKey::isUpper = "is_upper";
const std::string OpAttributeKey::blockSize = "block_size";
const std::string OpAttributeKey::transMode = "op_attr_trans_mode";
const std::string OpAttributeKey::workspaceBaseOffset = "workspace_base_offset";
const std::string OpAttributeKey::copyInMode = "op_attr_copy_in_mode";
const std::string OpAttributeKey::copyOutMode = "op_attr_copy_out_mode";
const std::string OpAttributeKey::localCopyLocalMode = "op_attr_local_copy_local_mode";
const std::string OpAttributeKey::copyIsNZ = "op_attr_is_nz";
const std::string OpAttributeKey::isGemv = "op_attr_is_gemv";
const std::string OpAttributeKey::scaleValue = "op_attr_scale_value";
const std::string OpAttributeKey::rowPad = "op_attr_row_pad";
const std::string OpAttributeKey::ownerRank = "owner_rank";
const std::string OpAttributeKey::maxTileNum = "max_tile_num";
const std::string OpAttributeKey::perRankDataShape = "per_rank_data_shape";
const std::string OpAttributeKey::perRankTileNum = "per_rank_tile_num";
const std::string OpAttributeKey::totalTileNum = "total_tile_num";
const std::string OpAttributeKey::precisionType = "precision_type";
const std::string OpAttributeKey::perm = "perm";
const std::string OpAttributeKey::mxQuantMode = "op_attr_mx_quant_mode";
const std::string OpAttributeKey::mxQuantAxis = "op_attr_mx_quant_axis";
const std::string OpAttributeKey::mxQuantPerformanceMode = "op_attr_mx_quant_performance_mode";
const std::string OpAttributeKey::gmTensorParamIdxInCall = "gm_tensor_param_idx_in_call";
const std::string OpAttributeKey::staticValidShape = "op_attr_static_valid_shape";
const std::string OpAttributeKey::l1ReuseHashOrder = "l1_reuse_hashOrder";
const std::string OpAttributeKey::l1ReuseSubgraphCount = "l1_reuse_subgraphCount";
const std::string OpAttributeKey::cubeMergeHashOrder = "cube_merge_hashOrder";
const std::string OpAttributeKey::cubeMergeSubgraphCount = "cube_merge_subgraphCount";
const std::string OpAttributeKey::vecMergeHashOrder = "vec_merge_hashOrder";
const std::string OpAttributeKey::vecMergeSubgraphCount = "vec_merge_subgraphCount";
const std::string OpAttributeKey::atomicAdd = "op_attr_atomic_add";
const std::string OpAttributeKey::splitMN = "op_attr_splitMN";
const std::string OpAttributeKey::reduceCopyPreSubgraphId = "reduce_copy_pre_subgraph_id";
const std::string OpAttributeKey::postK = "POST_K";
const std::string OpAttributeKey::postM = "POST_M";
const std::string OpAttributeKey::postN = "POST_N";
const std::string OpAttributeKey::filterH = "FILTER_H";
const std::string OpAttributeKey::filterW = "FILTER_W";
const std::string OpAttributeKey::strideH = "STRIDE_H";
const std::string OpAttributeKey::strideW = "STRIDE_W";
const std::string OpAttributeKey::dilationH = "DILATION_H";
const std::string OpAttributeKey::dilationW = "DILATION_W";
const std::string OpAttributeKey::paddingLeft = "PAD_LEFT";
const std::string OpAttributeKey::paddingRight = "PAD_RIGHT";
const std::string OpAttributeKey::paddingTop = "PAD_TOP";
const std::string OpAttributeKey::paddingBottom = "PAD_BOTTOM";
const std::string OpAttributeKey::padValue = "PAD_VALUE";
const std::string OpAttributeKey::repeatStride = "REPEAT_STRIDE";
const std::string OpAttributeKey::repeatTime = "REPEAT_TIME";
const std::string OpAttributeKey::wStride = "W_STRIDE";
const std::string OpAttributeKey::srcGmConvValidShape = "SRC_CONV_GM_VALID_SHAPE";
const std::string OpAttributeKey::l0cValidMN = "L0C_VALID_MN";
const std::string OpAttributeKey::rmwMode = "op_attr_rmw_mode";
const std::string OpAttributeKey::transDataOffset = "TRANSDATA_OFFSET";
const std::string OpAttributeKey::isConv = "isConv";
const std::string OpAttributeKey::writeConflict = "write_conflict";
const std::string OpAttributeKey::parallel = "parallel";

const std::string ConvOpAttributeKey::cin = "CIN";
const std::string ConvOpAttributeKey::cout = "COUT";
const std::string ConvOpAttributeKey::paddingLeft = "PADDING_LEFT";
const std::string ConvOpAttributeKey::paddingTop = "PADDING_TOP";
const std::string ConvOpAttributeKey::paddingRight = "PADDING_RIGHT";
const std::string ConvOpAttributeKey::paddingBottom = "PADDING_BOTTOM";
const std::string ConvOpAttributeKey::strideh = "STRIDEH";
const std::string ConvOpAttributeKey::stridew = "STRIDEW";
const std::string ConvOpAttributeKey::hposX = "HPOS_X";
const std::string ConvOpAttributeKey::hsteP = "HSTEP";
const std::string ConvOpAttributeKey::wposX = "WPOS_X";
const std::string ConvOpAttributeKey::wstep = "WSTEP";
const std::string ConvOpAttributeKey::hoffsetY = "HOFFSET_Y";
const std::string ConvOpAttributeKey::woffsetY = "WOFFSET_Y";
const std::string ConvOpAttributeKey::reluType = "RELU_TYPE";
const std::string ConvOpAttributeKey::reluAlpha = "RELU_ALPHA";
const std::string ConvOpAttributeKey::clearFlag = "CLEAR_FLAG";
const std::string ConvOpAttributeKey::hasAccFlag = "HAS_ACC_FLAG";
const std::string ConvOpAttributeKey::hasEltFlag = "HAS_ELT_FLAG";
const std::string ConvOpAttributeKey::hasBiasFlag = "HAS_BIAS_FLAG";
const std::string ConvOpAttributeKey::eltBrcbFlag = "ELT_BRCB_FLAG";
const std::string ConvOpAttributeKey::fmapSrcNum = "FMAP_SRC_NUM";

const std::string ConvOpAttributeKey::eltMode = "ELT_MODE";

const std::string ConvOpAttributeKey::fmapC0 = "FMAP_C0";
const std::string FixpOpAttributeKey::quantPreScalar = "QUANT_PRE_SCALAR";
const std::string FixpOpAttributeKey::quantPostScalar = "QUANT_POST_SCALAR";
const std::string FixpOpAttributeKey::antiqScalar = "ANTIQ_SCALAR";
const std::string FixpOpAttributeKey::hasQuantPreVector = "HAS_QUANT_PRE_VECTOR";
const std::string FixpOpAttributeKey::hasQuantPostVector = "HAS_QUANT_POST_VECTOR";
const std::string FixpOpAttributeKey::hasAntiqVector = "HAS_ANTIQ_VECTOR";
const std::string FixpOpAttributeKey::fbAddrSpace = "FIX_BUFFER_ADDR_SPACE";

const std::string PoolOpAttributeKey::poolh = "POOL_WIN_H";
const std::string PoolOpAttributeKey::poolw = "POOL_WIN_W";

const std::string TensorAttributeKey::tensorAddr = "tensorAddr";

bool OperationCmp::operator()(const Operation* lhs, const Operation* rhs) const
{
    return lhs->GetOpMagic() < rhs->GetOpMagic();
}

int64_t GetLatencyByOperands(const LogicalTensors& operands, const std::string& opcode)
{
    std::vector<std::vector<int64_t>> shape;
    shape.reserve(operands.size());
    for (const auto& operand : operands) {
        shape.emplace_back(operand->shape);
    }
    return GetCycles(opcode, shape, operands[0]->tensor->datatype);
}

Operation::Operation(Function& cur, Opcode opcode, LogicalTensors iOperands, LogicalTensors oOperands, int opMagic)
    : ir::TensorOpStmt(ir::Span()),
      iOperand(std::move(iOperands)),
      oOperand(std::move(oOperands)),
      opmagic(opMagic),
      opcode_(opcode),
      isTileOp_(!cur.IsGraphType(GraphType::TENSOR_GRAPH)),
      coreType_(CoreType::MIX),
      function_(&cur)
{
    if (opcode != Opcode::OP_CALL) {
        FE_ASSERT(FeError::INVALID_TYPE, cur.GetFunctionType() != FunctionType::EAGER);
    }

    iOpAttr_.resize(iOperands.size());
    oOpAttr_.resize(oOperands.size());
    tileShape_.Reset();
    if (opmagic == -1) {
        opmagic = cur.opSeed_++;
    }
    TensorOpStmt::opcode_ = OpcodeManager::Inst().GetOpcodeStr(opcode_);
    for (auto& t : iOperand) {
        TensorOpStmt::args_.push_back(t);
    }
    for (auto& t : oOperand) {
        TensorOpStmt::result_.push_back(t);
    }
    InitCoreTypeAndTileShape(opcode);
    InitTensorGraphMetadata();

    for (auto& input : GetIOperands()) {
        input->AddConsumer(this);
    }

    for (auto& output : GetOOperands()) {
        output->AddProducer(this);
    }

    InitLatency(opcode);
}

void Operation::InitCoreTypeAndTileShape(Opcode opcode)
{
    auto opCoreType = OpcodeManager::Inst().GetCoreType(opcode);
    switch (opCoreType) {
        case OpCoreType::AIC:
            coreType_ = CoreType::AIC;
            break;
        case OpCoreType::AIV:
            coreType_ = CoreType::AIV;
            break;
        case OpCoreType::AICPU:
            coreType_ = CoreType::AICPU;
            break;
        default:
            break;
    }

    if (function_->IsGraphType(GraphType::TENSOR_GRAPH)) {
        tileShape_ = TileShape::Current();
    }
}

void Operation::InitTensorGraphMetadata()
{
    if (!function_->IsGraphType({GraphType::TENSOR_GRAPH, GraphType::TILE_GRAPH})) {
        return;
    }
    SetSemanticLabel(config::GetSemanticLabel());
    SetSpan(ir::Span::Current());
}

void Operation::InitLatency(Opcode opcode)
{
    const auto& latencyOperands = (opcode == Opcode::OP_COPY_IN || opcode == Opcode::OP_VIEW ||
                                   opcode == Opcode::OP_SLICE) ?
                                      GetOOperands() :
                                      GetIOperands();
    if (!latencyOperands.empty()) {
        // Get operation latency
        latency_ = GetLatencyByOperands(latencyOperands, GetOpcodeStr());
    }
}

std::string Operation::GetStringAttribute(const std::string& key) const
{
    FE_ASSERT(FeError::NOT_EXIST, HasAttr(key)) << "Operation doesn't have attribute " << key;
    std::string attrVal;
    GetAttr(key, attrVal);
    return attrVal;
}

void Operation::SetAttribute(const std::string& key, const std::string& value) { SetAttr(key, value); }

bool Operation::GetBoolAttribute(const std::string& key) const
{
    if (!HasAttr(key)) {
        return false;
    }
    bool attrVal = false;
    GetAttr(key, attrVal);
    return attrVal;
}

void Operation::SetAttribute(const std::string& key, bool value) { SetAttr(key, value); }

int64_t Operation::GetIntAttribute(const std::string& key) const
{
    FE_ASSERT(FeError::NOT_EXIST, HasAttr(key)) << "Operation doesn't have attribute " << key;
    int64_t attrVal = 0;
    GetAttr(key, attrVal);
    return attrVal;
}

void Operation::SetAttribute(const std::string& key, int64_t value) { SetAttr(key, value); }

int Operation::GetInplaceIndex(size_t outputIndex) const
{
    struct {
        Opcode opcode;
        size_t outputIndex;
        int inputIndex;
    } inplaceInfo[] = {{Opcode::OP_INDEX_OUTCAST, 0, 2}};
    for (const auto& info : inplaceInfo) {
        if (info.opcode == GetOpcode() && outputIndex == info.outputIndex) {
            return info.inputIndex;
        }
    }
    if (HasAttribute(OpAttributeKey::inplaceIdx)) {
        ASSERT(ControlFlowScene::INVALID_INPLACE_CHAIN, outputIndex == 0);
        return static_cast<int>(GetIntAttribute(OpAttributeKey::inplaceIdx));
    }
    if (GetBoolAttribute(OP_ATTR_PREFIX + "isInplace")) {
        return 0;
    }

    std::map<int, int> attrInplaceInfo;
    if (GetAttr(OpAttributeKey::inplaceInfo, attrInplaceInfo)) {
        for (const auto& [inputIndex, attrOutputIndex] : attrInplaceInfo) {
            if (attrOutputIndex >= 0 && static_cast<size_t>(attrOutputIndex) == outputIndex) {
                return inputIndex;
            }
        }
    }
    return -1;
}

CastMode Operation::GetCastModeAttribute(const std::string& key) const
{
    FE_ASSERT(FeError::NOT_EXIST, HasAttr(key)) << "Operation doesn't have attribute " << key;
    int attrVal = GetIntAttribute(key);
    FE_ASSERT(FeError::INVALID_VAL, attrVal >= CAST_NONE && attrVal <= CAST_ODD);
    return static_cast<CastMode>(attrVal);
}

void Operation::SetAttribute(const std::string& key, CastMode value) { SetAttr(key, static_cast<int64_t>(value)); }

SymbolicScalar Operation::GetSymbolicScalarAttribute(const std::string& key) const
{
    FE_ASSERT(FeError::NOT_EXIST, HasAttr(key)) << "Operation doesn't have attribute " << key;
    SymbolicScalar attrVal = 0;
    GetAttr(key, attrVal);
    FE_ASSERT(FeError::INVALID_VAL, attrVal.IsValid());
    return attrVal;
}

void Operation::SetAttribute(const std::string& key, const SymbolicScalar& value)
{
    FE_ASSERT(FeError::INVALID_VAL, value.IsValid());
    SetAttr(key, value);
}

std::vector<SymbolicScalar> Operation::GetVectorSymbolicScalarAttribute(const std::string& key) const
{
    FE_ASSERT(FeError::NOT_EXIST, HasAttr(key)) << "Operation doesn't have attribute " << key;
    std::vector<SymbolicScalar> attrVal;
    GetAttr(key, attrVal);
    for (auto& attr : attrVal) {
        FE_ASSERT(FeError::INVALID_VAL, attr.IsValid());
    }
    return attrVal;
}

void Operation::SetAttribute(const std::string& key, const std::vector<SymbolicScalar>& value)
{
    for (auto& attr : value) {
        FE_ASSERT(FeError::INVALID_VAL, attr.IsValid());
    }
    SetAttr(key, value);
}

[[nodiscard]] Element Operation::GetElementAttribute(const std::string& key) const
{
    FE_ASSERT(FeError::NOT_EXIST, HasAttr(key)) << "Operation doesn't have attribute " << key;
    Element attrVal;
    GetAttr(key, attrVal);
    return attrVal;
}

std::vector<Element> Operation::GetVectorElementAttribute(const std::string& key) const
{
    FE_ASSERT(FeError::NOT_EXIST, HasAttr(key)) << "Operation doesn't have attribute " << key;
    std::vector<Element> attrVal;
    GetAttr(key, attrVal);
    return attrVal;
}

void Operation::SetAttribute(const std::string& key, const std::vector<Element>& value) { SetAttr(key, value); }

void Operation::SetAttribute(const std::string& key, Element value) { SetAttr(key, value); }

std::map<std::string, std::any> Operation::GetAllAttribute() const { return GetAllAttr(); }

void DebugJson(const Json& j)
{
    constexpr int32_t dumpLen = 2;
    std::string s = j.dump(dumpLen);
    printf("%s\n", s.c_str());
}

void Operation::DumpOperandsJson(Json& opDump, bool dumpTensor) const
{
    Json ioperandsDump = Json::array();
    Json ooperandsDump = Json::array();
    for (auto& i : iOperand) {
        if (dumpTensor) {
            ioperandsDump.push_back(i->DumpJson(*this->function_));
        } else {
            ioperandsDump.push_back(i->GetMagic());
        }
    }
    for (auto& o : oOperand) {
        if (dumpTensor) {
            ooperandsDump.push_back(o->DumpJson(*this->function_));
        } else {
            ooperandsDump.push_back(o->GetMagic());
        }
    }
    opDump["ioperands"] = ioperandsDump;
    opDump["ooperands"] = ooperandsDump;

    Json itokensDump = Json::array();
    for (const auto& token : tokens_) {
        itokensDump.push_back(token->name_);
    }
    Json otokensDump = Json::array();
    for (const auto& token : result_token_) {
        otokensDump.push_back(token->name_);
    }
    opDump["itokens"] = itokensDump;
    opDump["otokens"] = otokensDump;
}

void Operation::DumpCalleeHashJson(Json& opDump) const
{
    if (!IsCall()) {
        return;
    }
    auto calleeHash = std::static_pointer_cast<CallOpAttribute>(GetOpAttribute())->GetCalleeHash();
    std::shared_ptr<Function> callee = nullptr;
    for (auto& ele : Program::GetInstance().GetFunctionMap()) {
        if (ele.second->GetFunctionHash() == calleeHash) {
            callee = ele.second;
        }
    }
    if (callee == nullptr) {
        FE_LOGE(FeError::NOT_EXIST, "Cannot find function by calleeHash %s", calleeHash.c_str());
        return;
    }
    if (callee->rootFunc_ == nullptr) {
        opDump["calleehash"] = calleeHash.Data();
    } else {
        opDump["calleehash"] = callee->rootFunc_->GetFunctionHash().Data();
    }
}

void Operation::DumpLocationJson(Json& opDump) const
{
    opDump["opmagic"] = GetOpMagic();
    if (semanticLabel_) {
        Json jlabel;
        jlabel["label"] = semanticLabel_->label;
        jlabel["filename"] = semanticLabel_->filename;
        jlabel["lineno"] = semanticLabel_->lineno;
        opDump["semantic_label"] = jlabel;
    }
    if (!span_.IsUnknown() && config::GetPlatformConfig(KEY_DUMP_SOURCE_LOCATION, 0)) {
        opDump["file"] = span_.Filename();
        opDump["line"] = span_.BeginLine();
    }
}

void Operation::DumpParamLocationJson(Json& opDump) const
{
    Json inLocation = Json::array();
    Json outLocation = Json::array();
    for (auto& inLoc : inParamLocation_) {
        inLocation.emplace_back(inLoc);
    }
    for (auto& outLoc : outParamLocation_) {
        outLocation.emplace_back(outLoc);
    }
    if (!inParamLocation_.empty()) {
        opDump["in_param_loc"] = inLocation;
        opDump["static"]["in_param_loc"] = opDump["in_param_loc"];
    }
    if (!outParamLocation_.empty()) {
        opDump["out_param_loc"] = outLocation;
        opDump["static"]["out_param_loc"] = opDump["out_param_loc"];
    }
}

void Operation::DumpCallOpInfoJson(Json& opDump) const
{
    if (opcode_ != Opcode::OP_CALL ||
        !BelongTo()->IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::EXECUTE_GRAPH)) {
        return;
    }
    auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(GetOpAttribute());
    FE_ASSERT(FeError::INVALID_PTR, callAttr != nullptr);
    auto programId = callAttr->invokeInfo_->GetProgramId();
    auto programIter = function_->programs_.find(programId);
    if (programIter != function_->programs_.end()) {
        opDump["program_funcmagic"] = programIter->second->GetFuncMagic();
    } else {
        opDump["program_funcmagic"] = programFuncMagic_;
    }
    auto attr = std::dynamic_pointer_cast<CallOpAttribute>(GetOpAttribute());
    FE_ASSERT(FeError::INVALID_PTR, attr != nullptr);
    opDump["invoke_info"] = attr->DumpInvokeInfoJson();
    opDump["static"]["invoke_info"] = opDump["invoke_info"];
}

void Operation::DumpTileInfoJson(Json& opDump) const
{
    if (!isTileOp_) {
        return;
    }
    HashBuffer vecBuffer, cubeBuffer, distBuffer;
    opDump["tile"]["vec"] = std::basic_string(SerializeTo(tileShape_.GetVecTile(), vecBuffer));
    opDump["tile"]["cube"] = std::basic_string(SerializeTo(tileShape_.GetCubeTile(), cubeBuffer));
    opDump["tile"]["comm"] = std::basic_string(SerializeTo(tileShape_.GetDistTile(), distBuffer));
}

void Operation::DumpAttributesJson(Json& opDump) const
{
    if (GetOpAttribute() != nullptr) {
        opDump["attr"] = GetOpAttribute()->DumpDynJson();
    }
    for (const auto& pair : GetAllAttr()) {
        opDump["op_attr"][pair.first] = DumpAttrJson(pair.first);
    }
    opDump["sync_queue"] = syncQueue_.ToJson();
    opDump["static"]["sync_queue"] = opDump["sync_queue"];
}

Json Operation::DumpJson(bool dumpTensor) const
{
    Json opDump;
    opDump[T_FIELD_KIND] = static_cast<int>(Kind::T_KIND_OPERATION);

    DumpOperandsJson(opDump, dumpTensor);
    opDump["opcode"] = GetOpcodeStr();
    opDump["latency"] = GetLatency();

    DumpCalleeHashJson(opDump);
    DumpLocationJson(opDump);

    opDump["subgraphid"] = subgraphID_;
    opDump["mixSubgraphCore"] = GetAIVCoreStr();

    DumpParamLocationJson(opDump);
    DumpCallOpInfoJson(opDump);
    DumpTileInfoJson(opDump);
    DumpAttributesJson(opDump);

    return opDump;
}

void Operation::LoadOperandsFromJson(const Json& opDump,
                                     const std::unordered_map<int, std::shared_ptr<LogicalTensor>>& tensorDict,
                                     std::vector<std::shared_ptr<LogicalTensor>>& ioperands,
                                     std::vector<std::shared_ptr<LogicalTensor>>& ooperands)
{
    for (auto& i : opDump["ioperands"]) {
        std::shared_ptr<LogicalTensor> tensor;
        if (i.is_number()) {
            int magic = i.get<int>();
            FE_ASSERT(FeError::NOT_EXIST, tensorDict.count(magic));
            tensor = tensorDict.find(magic)->second;
        } else {
            tensor = tensorDict.find(i["magic"].get<int>())->second;
        }
        ioperands.push_back(tensor);
    }
    for (auto& o : opDump["ooperands"]) {
        std::shared_ptr<LogicalTensor> tensor;
        if (o.is_number()) {
            int magic = o.get<int>();
            FE_ASSERT(FeError::NOT_EXIST, tensorDict.count(magic));
            tensor = tensorDict.find(magic)->second;
        } else {
            tensor = tensorDict.find(o["magic"].get<int>())->second;
        }
        ooperands.push_back(tensor);
    }
}

void Operation::LoadLocationFromJson(const Json& opDump)
{
    if (opDump.count("semantic_label")) {
        auto jlabel = opDump["semantic_label"];
        semanticLabel_ = std::make_shared<SemanticLabel>(
            jlabel["label"].get<std::string>(), jlabel["filename"].get<std::string>(), jlabel["lineno"].get<int>());
    }
    if (opDump.count("file")) {
        span_ = ir::Span(opDump["file"].get<std::string>(), opDump["line"].get<int>(), 0);
    }
}

void Operation::LoadBasicInfoFromJson(const Json& opDump)
{
    subgraphID_ = opDump["subgraphid"].get<int>();

    UpdateLatency(opDump["latency"].get<int>());
    if (opDump.count("in_param_loc")) {
        for (auto& inLoc : opDump["in_param_loc"]) {
            inParamLocation_.emplace_back(inLoc);
        }
    }
    if (opDump.count("out_param_loc")) {
        for (auto& outLoc : opDump["out_param_loc"]) {
            outParamLocation_.emplace_back(outLoc);
        }
    }
}

void Operation::LoadTileInfoFromJson(const Json& opDump)
{
    if (opDump.count("tile")) {
        HashBuffer vecBuffer = opDump["tile"]["vec"].get<HashBuffer>();
        HashBuffer cubeBuffer = opDump["tile"]["cube"].get<HashBuffer>();
        HashBuffer distBuffer = opDump["tile"]["comm"].get<HashBuffer>();
        isTileOp_ = true;
        DeserializeFrom(vecBuffer, tileShape_.GetVecTile());
        DeserializeFrom(cubeBuffer, tileShape_.GetCubeTile());
        DeserializeFrom(distBuffer, tileShape_.GetDistTile());
    } else {
        isTileOp_ = false;
    }
}

void Operation::LoadOpAttributeFromJson(const Json& opDump, Opcode opcode)
{
    if (opDump.count("attr")) {
        auto& attrJson = opDump["attr"];
        std::shared_ptr<OpAttribute> opAttribute;
        switch (opcode) {
            case Opcode::OP_VIEW:
            case Opcode::OP_SLICE:
                opAttribute = DeserializeFrom<ViewOpAttribute>(attrJson);
                break;
            case Opcode::OP_ASSEMBLE:
            case Opcode::OP_CONTRACT:
                opAttribute = DeserializeFrom<AssembleOpAttribute>(attrJson);
                break;
            case Opcode::OP_CALL:
                opAttribute = DeserializeFrom<CallOpAttribute>(opDump, function_);
                break;
            case Opcode::OP_CONVERT:
                opAttribute = DeserializeFrom<ConvertOpAttribute>(attrJson);
                break;
            case Opcode::OP_COPY_IN:
            case Opcode::OP_L1_TO_L0A:
            case Opcode::OP_L1_TO_L0B:
            case Opcode::OP_L1_TO_L0_AT:
            case Opcode::OP_L1_TO_L0_BT:
            case Opcode::OP_COPY_OUT:
            case Opcode::OP_TRANSPOSE_MOVEIN:
            case Opcode::OP_TRANSPOSE_MOVEOUT:
            case Opcode::OP_INDEX_PUT:
            case Opcode::OP_INDEX_ADD:
            case Opcode::OP_INDEX_OUTCAST:
                opAttribute = DeserializeFrom<CopyOpAttribute>(attrJson);
                break;
            default:
                break;
        }
        SetOpAttribute(opAttribute);
    }
    if (opDump.count("op_attr")) {
        auto& opAttrJson = opDump["op_attr"];
        for (auto it = opAttrJson.begin(); it != opAttrJson.end(); ++it) {
            LoadAttrJson(it.key(), it.value());
        }
    }
}

void Operation::LoadExtraInfoFromJson(const Json& opDump)
{
    if (opDump.count("program_funcmagic")) {
        programFuncMagic_ = opDump["program_funcmagic"].get<int>();
    }
    if (opDump.count("sync_queue")) {
        syncQueue_.FromJson(opDump["sync_queue"]);
    }
    if (opDump.count("mixSubgraphCore")) {
        std::string aivCoreStr = opDump["mixSubgraphCore"].get<std::string>();
        if (aivCoreStr != "UNKNOWN") {
            AIVCore core = GetAIVCoreDict().Find(aivCoreStr, AIVCore::UNSPECIFIED);
            SetAIVCore(core);
        }
    }
}

std::shared_ptr<Operation> Operation::LoadJson(
    Function& cur, const std::unordered_map<int, std::shared_ptr<LogicalTensor>>& tensorDict, const Json& opDump)
{
    FE_ASSERT(FeError::INVALID_TYPE, opDump[T_FIELD_KIND].get<int>() == static_cast<int>(Kind::T_KIND_OPERATION));

    std::vector<std::shared_ptr<LogicalTensor>> ioperands;
    std::vector<std::shared_ptr<LogicalTensor>> ooperands;
    LoadOperandsFromJson(opDump, tensorDict, ioperands, ooperands);

    Opcode opcode = FindOpcode(opDump["opcode"].get<std::string>());
    int opMagic = opDump["opmagic"].get<int>();
    std::shared_ptr<Operation> op = std::make_shared<Operation>(cur, opcode, ioperands, ooperands, opMagic);

    op->LoadLocationFromJson(opDump);
    op->LoadBasicInfoFromJson(opDump);
    op->LoadTileInfoFromJson(opDump);
    op->LoadOpAttributeFromJson(opDump, opcode);
    op->LoadExtraInfoFromJson(opDump);

    return op;
}

std::string Operation::DumpSSA(const std::string& prefix) const
{
    std::ostringstream oss;

    if (!span_.IsUnknown()) {
        oss << prefix << "/* " << span_.ToString() << " */\n";
    }

    if (GetCommentList().size() != 0) {
        for (auto& c : GetCommentList()) {
            oss << prefix << "/* " + c + " */\n";
        }
    }

    bool hasResult = false;
    oss << prefix;
    for (size_t i = 0; i < oOperand.size(); i++) {
        oss << ((i != 0) ? ", " : "");
        oss << oOperand[i]->DumpSSA(false, true, true);
        hasResult = true;
    }
    for (const auto& token : result_token_) {
        oss << ((hasResult) ? ", " : "") << "Token(" << token->name_ << ")";
        hasResult = true;
    }
    oss << ((hasResult) ? " = " : "");
    oss << "!" << GetOpMagic() << " " << GetOpcodeStr(true);
    oss << "(g:" << GetSubgraphID() << ", s:" << GetScopeId() << ")";
    for (size_t i = 0; i < iOperand.size(); i++) {
        oss << ((i == 0) ? " " : ", ");
        oss << iOperand[i]->DumpSSA(false, true, false);
    }
    if (!tokens_.empty()) {
        oss << " #tokens{";
        for (size_t i = 0; i < tokens_.size(); i++) {
            oss << ((i == 0) ? "" : ", ") << tokens_[i]->name_;
        }
        oss << "}";
    }
    if (opAttribute_ != nullptr) {
        oss << " " << opAttribute_->Dump();
    }
    if (GetAllAttr().size()) {
        oss << " " << DumpAttr();
    }
    if (OpcodeManager::Inst().IsSync(GetOpcode())) {
        oss << " #sync{" << GetSyncQueue().Dump() << "}";
    }
    oss << "\n";
    return oss.str();
}

std::string Operation::Dump() const { return DumpSSA(); }

void Operation::ReplaceInputOperand(const std::shared_ptr<LogicalTensor>& originInput,
                                    const std::shared_ptr<LogicalTensor>& newInput)
{
    if (originInput == nullptr || newInput == nullptr) {
        return;
    }
    for (size_t i = 0; i < iOperand.size(); ++i) {
        FE_ASSERT(FeError::INVALID_PTR, iOperand[i] != nullptr);
        if (iOperand[i] == originInput) {
            iOperand[i] = newInput;
            continue;
        }
    }
}

void Operation::ReplaceOutputOperand(const std::shared_ptr<LogicalTensor>& originOutput,
                                     const std::shared_ptr<LogicalTensor>& newOutput)
{
    if (originOutput == nullptr || newOutput == nullptr) {
        return;
    }
    for (size_t i = 0; i < oOperand.size(); ++i) {
        FE_ASSERT(FeError::INVALID_PTR, oOperand[i] != nullptr);
        if (oOperand[i] == originOutput) {
            oOperand[i] = newOutput;
            continue;
        }
    }
}

void Operation::ReplaceIOperand(size_t index, std::shared_ptr<LogicalTensor> newTensor)
{
    FE_ASSERT(FeError::OUT_OF_RANGE, index < GetIOperands().size());
    GetIOperands()[index]->RemoveConsumer(*this);
    GetIOperands()[index] = std::move(newTensor);
    GetIOperands()[index]->AddConsumer(*this);
    TensorOpStmt::args_[index] = GetIOperands()[index];

    operationHash_ = 0;
}

void Operation::ReplaceOOperand(size_t index, std::shared_ptr<LogicalTensor> newTensor)
{
    FE_ASSERT(FeError::OUT_OF_RANGE, index < GetOOperands().size());
    GetOOperands()[index]->RemoveProducer(*this);
    GetOOperands()[index] = std::move(newTensor);
    GetOOperands()[index]->AddProducer(*this);
    TensorOpStmt::result_[index] = GetOOperands()[index];

    operationHash_ = 0;
}

LogicalTensorPtr Operation::GetInputOperand(const size_t index) const
{
    if (index >= iOperand.size()) {
        return nullptr;
    }
    return iOperand[index];
}

LogicalTensorPtr Operation::GetOutputOperand(const size_t index) const
{
    if (index >= oOperand.size()) {
        return nullptr;
    }
    return oOperand[index];
}

int Operation::GetIOperandIndex(const LogicalTensorPtr& ioperand) const
{
    for (size_t i = 0; i < iOperand.size(); ++i) {
        FE_ASSERT(FeError::INVALID_PTR, iOperand[i] != nullptr);
        if (iOperand[i] == ioperand) {
            return (int)i;
        }
    }
    return -1;
}
int Operation::GetOOperandIndex(const LogicalTensorPtr& ooperand) const
{
    for (size_t i = 0; i < oOperand.size(); ++i) {
        FE_ASSERT(FeError::INVALID_PTR, oOperand[i] != nullptr);
        if (oOperand[i] == ooperand) {
            return (int)i;
        }
    }
    return -1;
}

void Operation::AddDependOperand(LogicalTensorPtr dependoperand)
{
    for (const auto& operand : dependOperand) {
        if (operand == dependoperand) {
            return;
        }
    }
    dependOperand.emplace_back(dependoperand);
}

std::unordered_set<Operation*> Operation::ConsumerOps() const
{
    std::unordered_set<Operation*> consumers;
    for (const auto& output : GetOOperands()) {
        for (const auto& consumer : output->GetConsumers()) {
            if (consumer->BelongTo() == function_) {
                consumers.emplace(consumer);
            }
        }
    }
    return consumers;
}

std::unordered_set<Operation*> Operation::ProducerOps() const
{
    std::unordered_set<Operation*> producers;
    for (const auto& input : GetIOperands()) {
        for (const auto& producer : input->GetProducers()) {
            if (producer->BelongTo() == function_) {
                producers.emplace(producer);
            }
        }
    }
    return producers;
}

std::set<Operation*, Operation::OperationComparator> Operation::ConsumerOpsOrdered() const
{
    auto ops = ConsumerOps();
    std::set<Operation*, OperationComparator> consumers(ops.begin(), ops.end());
    return consumers;
}
std::set<Operation*, Operation::OperationComparator> Operation::ProducerOpsOrdered() const
{
    auto ops = ProducerOps();
    std::set<Operation*, OperationComparator> producers(ops.begin(), ops.end());
    return producers;
}

std::unordered_set<Operation*> Operation::ConsumerOpsByToken() const
{
    std::unordered_set<Operation*> consumers;
    if (function_) {
        for (const auto& token : result_token_) {
            for (const auto& stmt : function_->GetVarDependency().GetConsumers(token)) {
                consumers.emplace(static_cast<Operation*>(const_cast<ir::Stmt*>(stmt.get())));
            }
        }
    }
    return consumers;
}

std::unordered_set<Operation*> Operation::ProducerOpsByToken() const
{
    std::unordered_set<Operation*> producers;
    if (function_) {
        for (const auto& token : tokens_) {
            for (const auto& stmt : function_->GetVarDependency().GetProducers(token)) {
                producers.emplace(static_cast<Operation*>(const_cast<ir::Stmt*>(stmt.get())));
            }
        }
    }
    return producers;
}

void Operation::UpdateInputOperand(const size_t index, const std::shared_ptr<LogicalTensor>& newInput)
{
    if (newInput == nullptr || index >= iOperand.size()) {
        return;
    }
    iOperand[index]->RemoveConsumer(this);
    iOperand[index] = newInput;
    iOperand[index]->AddConsumer(this);
}

void Operation::UpdateOutputOperand(const size_t index, const std::shared_ptr<LogicalTensor>& newOutput)
{
    if (newOutput == nullptr || index >= oOperand.size()) {
        return;
    }
    oOperand[index]->RemoveProducer(this);
    oOperand[index] = newOutput;
    oOperand[index]->AddProducer(this);
}

void Operation::AddInCtrlOperation(Operation& operation)
{
    if (operation.BelongTo() != BelongTo()) {
        return;
    }
    if (this->GetOpMagic() == operation.GetOpMagic()) {
        return;
    }
    inputCtrlOps.emplace(&operation);
}

void Operation::RemoveInCtrlOperation(Operation& operation)
{
    auto iter = inputCtrlOps.find(&operation);
    if (iter != inputCtrlOps.end()) {
        inputCtrlOps.erase(iter);
    }
}

void Operation::AddOutCtrlOperation(Operation& operation)
{
    if (operation.BelongTo() != BelongTo()) {
        return;
    }
    if (this->GetOpMagic() == operation.GetOpMagic()) {
        return;
    }
    outputCtrlOps.emplace(&operation);
}

void Operation::RemoveOutCtrlOperation(Operation& operation)
{
    auto iter = outputCtrlOps.find(&operation);
    if (iter != outputCtrlOps.end()) {
        outputCtrlOps.erase(iter);
    }
}

Operation& Operation::CloneOperation(Function& func, const LogicalTensors& iOperandList,
                                     const LogicalTensors& oOperandList) const
{
    Operation& op = func.AddRawOperation(opcode_, iOperandList, oOperandList);
    op.SetScopeInfo(scopeInfo_);
    op.SetOooScopeId(oooScopeId_);
    if (opAttribute_) {
        op.opAttribute_ = opAttribute_->Clone();
    }
    op.attributes = attributes;
    op.UpdateTileShape(tileShape_);
    return op;
}

void Operation::UnlinkFromLogicalTensors()
{
    for (auto& input : iOperand) {
        input->RemoveConsumer(this);
    }
    for (auto& output : oOperand) {
        output->RemoveProducer(this);
    }
}

std::shared_ptr<Operation> Operation::CloneTensorOpStmt(const LogicalTensors& iOperandList,
                                                        const LogicalTensors& oOperandList,
                                                        const std::vector<ir::VarPtr>& resultTokens,
                                                        const std::vector<ir::VarPtr>& tokens, ir::Span span,
                                                        Function* targetFunc) const
{
    const_cast<Operation*>(this)->UnlinkFromLogicalTensors();
    auto applyCloneMetadata = [&](Operation& op) {
        op.SetScopeInfo(scopeInfo_);
        op.SetOooScopeId(oooScopeId_);
        if (opAttribute_) {
            op.opAttribute_ = opAttribute_->Clone();
        }
        op.attributes = attributes;
        op.UpdateTileShape(tileShape_);
        op.coreType_ = coreType_;
        op.result_token_ = resultTokens;
        op.tokens_ = tokens;
        op.attrs_ = attrs_;
        if (semanticLabel_) {
            op.semanticLabel_ = semanticLabel_;
        }
        if (!span.IsUnknown()) {
            op.SetSpan(span);
        }
    };

    if (targetFunc != nullptr) {
        for (const auto& operand : iOperandList) {
            operand->UpdateBelongFunction(*targetFunc);
        }
        for (const auto& operand : oOperandList) {
            operand->UpdateBelongFunction(*targetFunc);
        }
        Operation& op = targetFunc->AddRawOperation(opcode_, iOperandList, oOperandList, span);
        applyCloneMetadata(op);
        return std::static_pointer_cast<Operation>(op.shared_from_this());
    }

    FE_ASSERT(BelongTo() != nullptr);
    auto clone = std::make_shared<Operation>(*BelongTo(), opcode_, iOperandList, oOperandList);
    clone->UnlinkFromLogicalTensors();
    applyCloneMetadata(*clone);
    return clone;
}

std::string Operation::GetOpcodeStr(bool appendTile) const
{
    if (appendTile && isTileOp_) {
        return "TILE_" + TensorOpStmt::opcode_;
    }
    return TensorOpStmt::opcode_;
}

[[nodiscard]] std::string Operation::GetCoreTypeStr() const
{
    switch (coreType_) {
        case CoreType::AIC:
            return "AIC";
        case CoreType::AIV:
            return "AIV";
        case CoreType::AICPU:
            return "AICPU";
        default:
            return "MIX";
    }
}

unsigned long Operation::ComputeHash()
{
    // compute has every time to avoid member changed
    operationHash_ = ComputeHashOrderless();
    return operationHash_;
}

unsigned long Operation::ComputeHashOrderless() const
{
    std::stringstream ss;
    ss << GetOpcodeStr(true);
    for (const auto& incast : iOperand) {
        ss << "[i";
        ss << "$" << incast->tensor->DumpSSA(false, false);
        ss << incast->DumpType();
        ss << "(";
        for (size_t i = 0; i < incast->offset.size(); ++i) {
            ss << incast->offset[i];
            if (i != incast->offset.size() - 1) {
                ss << ", ";
            }
        }
        ss << ")";
        ss << "]";
    }

    if (opAttribute_ != nullptr) {
        ss << " " << opAttribute_->Dump();
    }
    for (const auto& attr : OpcodeManager::Inst().GetAttrs(opcode_)) {
        ss << " attr: [" << attr << " : " << DumpAttr(attr) << "]";
    }

    ss << "id" << subgraphID_;

    std::string s = ss.str();

    std::hash<std::string> hasher;
    auto result = hasher(s);
    return result;
}

bool Operation::IsCall() const { return opcode_ == Opcode::OP_CALL; }

bool Operation::IsNOP() const { return opcode_ == Opcode::OP_NOP; }

bool Operation::IsIsolatedOp() const
{
    return iOperand.empty() && oOperand.empty() && inputCtrlOps.empty() && outputCtrlOps.empty();
}

bool Operation::OnlyHasCtrlEdgeToOp(Operation& op) const
{
    if (!iOperand.empty() || !oOperand.empty() || !inputCtrlOps.empty()) {
        return false;
    }
    return outputCtrlOps.size() == 1 && outputCtrlOps.count(&op) != 0;
}

void Operation::EraseInput(const std::shared_ptr<LogicalTensor>& input)
{
    for (auto iter = iOperand.begin(); iter != iOperand.end();) {
        if (iter->get()->magic == input->magic) {
            iter = iOperand.erase(iter);
        } else {
            ++iter;
        }
    }
}

void Operation::EraseDependTensor(const std::shared_ptr<LogicalTensor>& dependTensor)
{
    for (auto iter = dependOperand.begin(); iter != dependOperand.end();) {
        if (iter->get()->magic == dependTensor->magic) {
            iter = dependOperand.erase(iter);
        } else {
            ++iter;
        }
    }
}

void Operation::ReplaceInput(const std::shared_ptr<LogicalTensor>& newInput,
                             const std::shared_ptr<LogicalTensor>& oldInput)
{
    for (size_t i = 0; i < GetIOperands().size(); i++) {
        if (iOperand[i]->magic == oldInput->magic) {
            ReplaceIOperand(i, newInput);
        }
    }
}

void Operation::ReplaceOutput(const std::shared_ptr<LogicalTensor>& newOutput,
                              const std::shared_ptr<LogicalTensor>& oldOutput)
{
    for (int i = 0; static_cast<size_t>(i) < oOperand.size(); ++i) {
        if (oOperand[i]->magic == oldOutput->magic) {
            ReplaceOOperand(i, newOutput);
            break;
        }
    }
}

void Operation::SetSubFuncInvokeInfo(const SubfuncInvokeInfoTy& invokeInfo)
{
    auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(opAttribute_);
    FE_ASSERT(FeError::INVALID_PTR, callAttr != nullptr);
    callAttr->invokeInfo_ = std::make_shared<SubfuncInvokeInfoTy>(invokeInfo);
}

int Operation::GetProgramId()
{
    auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(opAttribute_);
    FE_ASSERT(FeError::INVALID_PTR, callAttr != nullptr);
    return callAttr->invokeInfo_->GetProgramId();
}

bool Operation::IsNeedStackGM() const
{
    auto isStack = [](const std::shared_ptr<LogicalTensor>& tensor) {
        return tensor->GetRawMagic() == SYMBOL_STACK_BASE;
    };

    return (OpcodeManager::Inst().IsCopyIn(opcode_) && std::any_of(iOperand.cbegin(), iOperand.cend(), isStack)) ||
           (OpcodeManager::Inst().IsCopyOut(opcode_) && std::any_of(oOperand.cbegin(), oOperand.cend(), isStack));
}

std::vector<std::reference_wrapper<SymbolicScalar>> Operation::GetDynamicAttributeList()
{
    std::vector<std::reference_wrapper<SymbolicScalar>> dynamicAttributeList;
    switch (GetOpcode()) {
        case Opcode::OP_VIEW:
        case Opcode::OP_SLICE: {
            auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(GetOpAttribute());
            if (viewAttr != nullptr) {
                std::vector<SymbolicScalar>& viewFromDynOffset = viewAttr->GetFromDynOffset();
                for (auto& offset : viewFromDynOffset) {
                    dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(offset));
                }
                std::vector<SymbolicScalar>& viewToDynValidShape = viewAttr->GetToDynValidShape();
                for (auto& shape : viewToDynValidShape) {
                    dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(shape));
                }
            }
            for (auto& shape : oOperand[0]->GetDynValidShape()) {
                dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(shape));
            }
        } break;
        case Opcode::OP_ASSEMBLE:
        case Opcode::OP_CONTRACT:
            [[fallthrough]];
        case Opcode::OP_ATOMIC_RMW: {
            auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(GetOpAttribute());
            if (assembleAttr != nullptr) {
                std::vector<SymbolicScalar>& assembleToDynOffset = assembleAttr->GetToDynOffset();
                for (auto& offset : assembleToDynOffset) {
                    dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(offset));
                }
                std::vector<SymbolicScalar>& assembleFromDynValidShape = assembleAttr->GetFromDynValidShape();
                for (auto& shape : assembleFromDynValidShape) {
                    dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(shape));
                }
            }
            for (auto& shape : iOperand[0]->GetDynValidShape()) {
                dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(shape));
            }
        } break;
        case Opcode::OP_COPY_IN:
            [[fallthrough]];
        case Opcode::OP_UB_COPY_IN: {
            auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(GetOpAttribute());
            if (copyAttr != nullptr) {
                for (auto& offset : copyAttr->GetFromOffset()) {
                    if (offset.IsSpecified()) {
                        dynamicAttributeList.push_back(
                            std::reference_wrapper<SymbolicScalar>(offset.GetSpecifiedValue()));
                    }
                }
                for (auto& shape : copyAttr->GetToDynValidShape()) {
                    if (shape.IsSpecified()) {
                        dynamicAttributeList.push_back(
                            std::reference_wrapper<SymbolicScalar>(shape.GetSpecifiedValue()));
                    }
                }
            }
        } break;
        case Opcode::OP_COPY_OUT:
            [[fallthrough]];
        case Opcode::OP_UB_COPY_OUT: {
            auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(GetOpAttribute());
            if (copyAttr) {
                for (auto& offset : copyAttr->GetToOffset()) {
                    if (offset.IsSpecified()) {
                        dynamicAttributeList.push_back(
                            std::reference_wrapper<SymbolicScalar>(offset.GetSpecifiedValue()));
                    }
                }
                for (auto& shape : copyAttr->GetFromDynValidShape()) {
                    if (shape.IsSpecified()) {
                        dynamicAttributeList.push_back(
                            std::reference_wrapper<SymbolicScalar>(shape.GetSpecifiedValue()));
                    }
                }
            }
        } break;
        case Opcode::OP_VEC_DUP: {
            auto scalar = GetAttr<SymbolicScalar>(OpAttributeKey::dynScalar);
            if (scalar) {
                dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(*scalar));
            }
        } break;
        case Opcode::OP_PRINT: {
            auto cond = GetAttr<SymbolicScalar>(OP_ATTR_PREFIX + "cond");
            if (cond) {
                dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(*cond));
            }
            auto scalars = GetAttr<std::vector<SymbolicScalar>>(OP_ATTR_PREFIX + "scalars");
            if (scalars) {
                for (auto& scalar : *scalars) {
                    dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(scalar));
                }
            }
        } break;
        case Opcode::OP_BIND_TENSOR: {
            auto& attrDict = GetAllAttr();
            auto it = attrDict.find(OpAttributeKey::bindTensor);
            if (it != attrDict.end()) {
                auto& value = pypto::AnyCastRef<SymbolicScalar>(it->second);
                dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(value));
            }
        } break;
        case Opcode::OP_CALL: {
            auto callAttr = std::static_pointer_cast<CallOpAttribute>(GetOpAttribute());
            if (callAttr != nullptr) {
                for (auto& arg : callAttr->GetLinearArgList()) {
                    dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(arg));
                }
            }
        } break;
        case Opcode::OP_SHMEM_PUT:
        case Opcode::OP_SHMEM_STORE:
        case Opcode::OP_SHMEM_GET:
        case Opcode::OP_SHMEM_LOAD: {
            auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(GetOpAttribute());
            if (copyAttr == nullptr) {
                break;
            }
            for (auto& shape : copyAttr->GetToDynValidShape()) {
                if (!shape.IsSpecified()) {
                    continue;
                }
                dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(shape.GetSpecifiedValue()));
            }
            for (auto& shape : copyAttr->GetFromDynValidShape()) {
                if (!shape.IsSpecified()) {
                    continue;
                }
                dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(shape.GetSpecifiedValue()));
            }
            for (auto& offset : copyAttr->GetToOffset()) {
                if (!offset.IsSpecified()) {
                    continue;
                }
                dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(offset.GetSpecifiedValue()));
            }
            for (auto& offset : copyAttr->GetFromOffset()) {
                if (!offset.IsSpecified()) {
                    continue;
                }
                dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(offset.GetSpecifiedValue()));
            }
        } break;
        case Opcode::OP_L1_RESHAPE_COPY_IN:
            [[fallthrough]];
        case Opcode::OP_L0C_RESHAPE_COPY_OUT:
            [[fallthrough]];
        case Opcode::OP_RESHAPE_COPY_OUT:
            [[fallthrough]];
        case Opcode::OP_RESHAPE_COPY_IN: {
            auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(GetOpAttribute());
            if (copyAttr) {
                for (auto& shape : copyAttr->GetFromDynValidShape()) {
                    if (shape.IsSpecified()) {
                        dynamicAttributeList.push_back(
                            std::reference_wrapper<SymbolicScalar>(shape.GetSpecifiedValue()));
                    }
                }
                for (auto& shape : copyAttr->GetToDynValidShape()) {
                    if (shape.IsSpecified()) {
                        dynamicAttributeList.push_back(
                            std::reference_wrapper<SymbolicScalar>(shape.GetSpecifiedValue()));
                    }
                }
            }
        } break;
        default:
            break;
    }
    return dynamicAttributeList;
}

} // namespace npu::tile_fwk
