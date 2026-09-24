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
 * \file convert_op_inserter.h
 * \brief
 */

#ifndef PASS_CONVERT_OP_INSERTER_LEGACY_H_
#define PASS_CONVERT_OP_INSERTER_LEGACY_H_

#include <unordered_map>
#include <unordered_set>
#include <set>

#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"

namespace npu {
namespace tile_fwk {
namespace legacy {

struct OpMagicComparator {
    bool operator()(const Operation* a, const Operation* b) const { return a->GetOpMagic() < b->GetOpMagic(); }
};

struct ConvertOpInfo {
    MemoryType from;
    MemoryType to;
    std::shared_ptr<LogicalTensor> input;
    std::shared_ptr<LogicalTensor> output;
};

class ConvertInserter {
public:
    ConvertInserter() = default;
    ~ConvertInserter() = default;

    std::vector<ConvertOpInfo> converts;
    std::unordered_map<int, std::shared_ptr<RawTensor>> oldRawToNewRaw;

    /*
        key: Tensor 指针
        value: Opmagic到consumer op的指针和该op所需内存类型对的映射map
    */
    std::map<LogicalTensorPtr, std::map<int, std::pair<Operation*, MemoryType>>> tensorTobeMap;
    std::unordered_map<int, std::map<MemoryType, std::set<Operation*, OpMagicComparator>>> conflictMap;

    // 设置指定tensor的指定consumer op所需的mem tobe 类型
    void UpdateTensorTobeMap(const LogicalTensorPtr& tensor, Operation& operation, MemoryType t,
                             const char* reason = nullptr);

    std::map<MemoryType, std::set<Operation*, OpMagicComparator>> GetRequiredTobe(LogicalTensorPtr& tensor) const;

    // 查询指定tensor是否存在指定consumer op所需的mem类型
    bool HasRequirement(const LogicalTensorPtr& tensor, const Operation& operation) const;

    // 安全提取指定tensor的指定consumer op所需的mem类型，不存在时返回MEM_UNKNOWN
    MemoryType GetRequirementOrUnknown(const LogicalTensorPtr& tensor, const Operation& operation) const;

    // 提取指定tensor的所有consumer op和所需的mem类型
    std::map<Operation*, MemoryType, OpMagicComparator> GetConsumerRequirements(const LogicalTensorPtr& tensor) const;

    // 提取指定tensor的所有已知consumer需求mem类型
    std::set<MemoryType> GetKnownRequiredTypes(const LogicalTensorPtr& tensor) const;

    // 如果指定tensor的已知consumer需求mem类型唯一则返回该类型，否则返回MEM_UNKNOWN
    MemoryType TryGetUniqueKnownRequiredType(const LogicalTensorPtr& tensor) const;

    // 清空convert插入规划态，不清空tensorTobeMap
    void ClearPlanningState();

    // 过滤得到所有有conflict的tensor信息
    void FilterConflictTensor();

    // tobe Map转换类型，以memory type为key
    std::map<MemoryType, std::set<Operation*, OpMagicComparator>> ReformMap(
        const std::map<int, std::pair<Operation*, MemoryType>>& oriMap) const;

    // 遍历所有tensor，如果有Mem conflict，记录到converts中
    Status RecordConflict(Function& function);

    // 根据已记录的converts插入OP_CONVERT
    void InsertConvertOps(Function& function);

    // 将插入的OP_CONVERT转化为View和Assemble，后续GenerateMoveOp时会转化为copy类Op
    bool CreateMoveOpForConvert(Operation& op);

    // 判断是否跨Memory层级
    bool CrossCore(const MemoryType from, const MemoryType to) const;

    // 更新消费者并重连graph
    void UpdateConsumerAndReconnect(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor,
                                    Operation* op) const;

    // 对外总接口
    Status DoInsertion(Function& function);

    // 构建转换路径
    Status ConstructPath(MemoryType from, MemoryType to, std::vector<MemoryType>& paths,
                         const std::shared_ptr<LogicalTensor>& oOperand, const Operation& op) const;

    // 检查tensor是否需要跳过
    bool SkipOperand(const std::shared_ptr<LogicalTensor>& oOperand,
                     const std::unordered_set<int>& visitedTensor) const;

    // 检查tensor生产者是否都是assemble
    bool isAllProducerAssemble(const std::shared_ptr<LogicalTensor>& oOperand) const;

    // 检查tensor所有的消费者是否都有效
    bool isAllConsumersValid(Function& function, const std::set<Operation*, OpMagicComparator>& consumers) const;

    // 为每个存在内存冲突的消费者插入convert op
    void InsertConvertOpForEachConsumer(Function& function, const Operation& op,
                                        const std::shared_ptr<LogicalTensor>& oOperand,
                                        std::set<Operation*, OpMagicComparator>& consumers,
                                        std::vector<MemoryType>& paths);

    // 记录需要插入的convert op
    std::shared_ptr<LogicalTensor> RecordInsertConvertOp(const std::shared_ptr<LogicalTensor>& oOperand,
                                                         const std::vector<MemoryType>& paths, Function& function,
                                                         const Operation& op);

    // 创建convert链路上的中间tensor，并继承输入tensor的动态shape状态
    std::shared_ptr<LogicalTensor> CreateTensorLikeForConvert(const std::shared_ptr<LogicalTensor>& input,
                                                              MemoryType outputMemoryType) const;

    // 构造convert转view时使用的ViewOpAttribute
    std::shared_ptr<ViewOpAttribute> BuildViewAttrForConvert(const Operation& op, MemoryType to) const;

    // 构造convert转assemble时使用的AssembleOpAttribute
    std::shared_ptr<AssembleOpAttribute> BuildAssembleAttrForConvert(const Operation& op, MemoryType from) const;

    // graph重连
    void GraphReconnect(const std::shared_ptr<LogicalTensor>& oOperand, std::shared_ptr<LogicalTensor> output,
                        const std::set<Operation*, OpMagicComparator>& consumers, Function& function) const;

    // cube级联场景
    bool IsNotValidDataType(const std::shared_ptr<LogicalTensor>& firstCVOutput) const;

    // l0c2l1场景，限制数据类型和数据对齐
    bool FitL0C2L1(const LogicalTensorPtr& tensor);

    // l0c2l1场景规避条件检查，检查op输入是否满足无非立即数的validShape
    bool FitL0C2L1(const Operation& op);

    // 检查是否适合 UB2L1 小搬大
    bool FitUB2L1(const LogicalTensorPtr& tensor) const;

    // 判断同一源 tensor 是否有并行 consumer 需要不同的 memory 去向
    bool HasParallelDifferentConsumerRequirement(const LogicalTensorPtr& tensor, MemoryType targetType) const;

    // 特殊场景处理：生成者均为Assemble或者消费者均为View/Assemble，且mem路径中经过DDR
    void ProcessSpecialProducersOrConsumers(Function& function, const Operation& op,
                                            const std::shared_ptr<LogicalTensor>& oOperand,
                                            std::set<Operation*, OpMagicComparator>& consumers,
                                            MemoryType& requiredMemoryType);

    // 构造转换路径
    Status ProcessConvertPath(const Operation& op, const std::shared_ptr<LogicalTensor>& oOperand,
                              MemoryType requiredMemoryType, std::vector<MemoryType>& paths);
};
static constexpr int MATMUL_DIM_NUM = 2;
static constexpr int L0C2L1_DIM1_SHAPE_RESTICT = 16; // l0c2l1要求输入的外轴（第一轴）元素数量必须是16的倍数
static constexpr int L0C2L1_DIM2_BYTE_RESTICT = 32; // l0c2l1要求输入的内轴（第二轴）必须是32Byte对齐
} // namespace legacy
} // namespace tile_fwk
} // namespace npu
#endif // PASS_CONVERT_OP_INSERTER_LEGACY_H_
