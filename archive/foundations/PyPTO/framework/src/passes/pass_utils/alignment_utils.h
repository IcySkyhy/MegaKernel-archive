/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file alignment_utils.h
 * \brief
 */

#pragma once

#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

class AlignmentUtils {
public:
    /**
     * @brief Get last-dimension alignment base in elements.
     *
     * @param tensor input logical tensor.
     * @return alignment base in elements; return 0 when tensor/dtype is invalid.
     */
    static int64_t GetLastDimAlignBase(const LogicalTensorPtr& tensor);

    /**
     * @brief Check whether the last dimension is 32-byte aligned.
     *
     * @param tensor input logical tensor.
     * @return true if last dim bytes is aligned to BLOCK_SIZE; otherwise false.
     */
    static bool IsLastDim32BAligned(const LogicalTensorPtr& tensor);

    /**
     * @brief Align dimension up to the nearest multiple of padValue.
     *
     * @param dim input dimension value.
     * @param padValue alignment value; return dim directly when padValue is 0.
     * @return aligned dimension value.
     */
    static int64_t Pad(int64_t dim, int64_t padValue);

    /**
     * @brief Round dim up using a row-pad base (dim + padValue - 1).
     *
     * @param dim input dimension value.
     * @param padValue padding base.
     * @return padded dimension.
     */
    static int64_t PadRowDim(int64_t dim, int64_t padValue);

    /**
     * @brief Get the byte size of the last raw-shape dimension.
     *
     * @param tensor input logical tensor.
     * @return last-dim element count multiplied by element size; 0 when rawshape is empty.
     */
    static size_t GetLastDimBytes(const LogicalTensorPtr& tensor);

    /**
     * @brief Pad last dimension to 32-byte aligned on UB memory.
     *
     * If tensor is on UB memory and last dimension is not 32-byte aligned,
     * this function will pad the last dimension to meet 32-byte alignment requirement.
     * Original shape is saved before modification.
     *
     * @param tensor input logical tensor to be processed.
     */
    static void ProcessLastDim32BAlignedOnUB(LogicalTensorPtr tensor);

private:
    /**
     * @brief Check tensor basic validity for last-dimension alignment calculation.
     *
     * @param tensor input logical tensor.
     * @return true if tensor and shape are valid for last-dim check.
     */
    static bool IsValidForLastDimCheck(const LogicalTensorPtr& tensor);

    /**
     * @brief Check whether current operand index is marked as combined axis.
     *
     * @param combineAxis combine-axis attribute vector.
     * @param index operand index.
     * @return true if current index is marked combined and should be skipped.
     */
    static bool IsCombinedAxis(const std::vector<bool>& combineAxis, size_t index);
};

} // namespace npu::tile_fwk
