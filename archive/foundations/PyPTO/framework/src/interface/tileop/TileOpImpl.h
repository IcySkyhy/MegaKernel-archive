/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file TileOpImpl.h
 * \brief
 */

#ifndef __LOGICALTENSOR_TILE_OP_IMPL__
#define __LOGICALTENSOR_TILE_OP_IMPL__

#include "aicore_runtime.h"
#ifndef __LITE_NPU
#include "mte.h"
#include "dynamic/mte_dyn.h"
#include "vector.h"
#include "dynamic/vector_dyn.h"
#include "cube.h"
#include "dynamic/cube_dyn.h"
#include "fixpipe.h"
#include "distributed/common.h"
#include "dynamic/aicpu_call.h"
#endif

#ifdef SUPPORT_TILE_TENSOR
#ifndef __LITE_NPU
#include "distributed/tileop_shmem.h"
#endif
#include "utils/layout.h"
#include "vector/unary.h"
#include "vector/trans.h"
#include "vector/trans_data.h"
#include "vector/binary.h"
#include "vector/binary_scalar.h"
#include "vector/cast.h"
#include "vector/sign.h"
#include "vector/signbit.h"
#include "vector/isnan.h"
#include "vector/tanh.h"
#include "vector/tan.h"
#include "vector/atanh.h"
#include "vector/reduce.h"
#include "vector/softmax.h"
#include "vector/mte.h"
#include "vector/atan.h"
#include "vector/logicalnot.h"
#include "vector/compare.h"
#include "vector/hypot.h"
#include "vector/prelu.h"
#include "vector/gather.h"
#include "vector/indexadd.h"
#include "vector/scatter.h"
#include "vector/expand.h"
#include "vector/expand_exp_dif.h"
#include "vector/cum_operation.h"
#include "vector/extract.h"
#include "vector/pair_binary.h"
#include "vector/where.h"
#include "vector/pad.h"
#include "vector/fillpad.h"
#include "vector/interleave.h"
#ifndef __LITE_NPU
#include "vector/sort.h"
#include "vector/radix_select/radix_select.h"
#include "vector/quantize.h"
#include "vector/dequantize.h"
#endif
#include "vector/logicaland.h"
#include "vector/log1p.h"
#include "vector/vector_dup.h"
#include "vector/range.h"
#include "vector/uniform.h"
#include "vector/triul.h"
#include "vector/onehot.h"
#include "vector/index_outcast.h"
#include "vector/bitwise_shift.h"
#include "vector/copysign.h"
#include "vector/permute.h"
#include "cube/cube_pto.h"
#include "conv/conv_pto.h"
#endif

#endif
