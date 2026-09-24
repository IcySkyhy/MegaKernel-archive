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
 * \file checker.h
 * \brief
 */

#ifndef CHECKER_H
#define CHECKER_H

#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class Checker {
public:
    /**
     * \brief Destroy the Checker object
     */
    virtual ~Checker() = default;
    /**
     * \brief Do the PreCheck for current pass.
     *        If not overriden, check nothing and return SUCCESS.
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the function passes the precheck.
     */
    virtual Status DoPreCheck(Function& function);
    /**
     * \brief Do the PostCheck for current pass.
     *        If not overriden, check nothing and return SUCCESS.
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the function passes the postcheck.
     */
    virtual Status DoPostCheck(Function& function);
    /**
     * \brief Do the DefaultEnabledPreCheck for current pass, the check items must be executed.
     *        If not overriden, check nothing and return SUCCESS.
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the function passes the precheck.
     */
    virtual Status DoDefaultEnabledPreCheck(Function& function);
    /**
     * \brief Do the DefaultEnabledPostCheck for current pass, the check items must be executed.
     *        If not overriden, check nothing and return SUCCESS.
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the function passes the postcheck.
     */
    virtual Status DoDefaultEnabledPostCheck(Function& function);

protected:
    /**
     * \brief Check whether consumers and producers of the tensor are valid (not null).
     * \param tensor : This parameter indicates the source tensor.
     * \return Status, indicating whether the tensor has null consumer or null producer.
     */
    Status CheckConsumerProducer(const LogicalTensorPtr& tensor);
    /**
     * \brief Check whether the function has invalid op (null op).
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the function has null op.
     */
    Status CheckValidOp(Function& function);
    /**
     * \brief Check whether ops are valid (has null input/output).
     *        Besides, check whether the input and the output has null consumer/producer.
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the function has an op with null input/output,
     *                 or there exists an op has an input or output with null consumer/producer.
     */
    Status CheckOpIOValid(Function& function);
    /**
     * \brief Check whether the incasts and outcasts of the function are valid (not empty).
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the function has valid incast/outcast.
     */
    Status CheckCompleteness(Function& function);
    /**
     * \brief Check whether the graph has loop.
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the graph has a loop.
     */
    Status CheckGraphLoop(Function& function);
    /**
     * \brief Common verification.
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the public verification is passed.
     */
    Status PublicCheck(Function& function);
    /**
     * \brief Check whether the fromDynOffset_ and toDynValidShape_ of the OP_VIEW are valid.
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the fromDynOffset_ or toDynValidShape_ of OP_VIEW is empty.
     */
    Status CheckDynAttrForView(Function& function);
    /**
     * \brief Check whether the toDynOffset_ of OP_ASSEMBLE is valid.
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the toDynOffset_ of OP_ASSEMBLE is empty.
     */
    Status CheckToDynOffsetForAssemble(Function& function);
    /**
     * \brief Check whether locally defined tensors are valid.
     * \param function : This parameter indicates the function to be checked.
     * \return Status, indicating whether the local tensors have valid producers.
     */
    Status CheckLocalTensor(Function& function);
};
} // namespace tile_fwk
} // namespace npu
#endif // CHECKER_H
