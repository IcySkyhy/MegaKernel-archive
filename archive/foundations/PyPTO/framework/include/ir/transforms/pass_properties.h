/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_
#define PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_

#include "ir/transforms/ir_property.h"

namespace pypto {
namespace ir {
namespace pass {

inline const PassProperties kUnrollLoopsProperties{{IRProperty::TypeChecked}, {IRProperty::TypeChecked}, {}};

inline const PassProperties kSplitChunkedLoopsProperties{
    {IRProperty::TypeChecked, IRProperty::SSAForm}, {IRProperty::TypeChecked, IRProperty::SSAForm}, {}};

inline const PassProperties kInterchangeChunkLoopsProperties{
    {IRProperty::TypeChecked, IRProperty::SSAForm}, {IRProperty::TypeChecked, IRProperty::SSAForm}, {}};

inline const PassProperties kConvertToSSAProperties{
    {IRProperty::TypeChecked},
    {IRProperty::TypeChecked, IRProperty::SSAForm},
    {IRProperty::NormalizedStmtStructure, IRProperty::FlattenedSingleStmt}};

inline const PassProperties kFlattenCallExprProperties{
    {IRProperty::TypeChecked},
    {IRProperty::TypeChecked, IRProperty::NoNestedCalls},
    {IRProperty::NormalizedStmtStructure, IRProperty::FlattenedSingleStmt}};

inline const PassProperties kNormalizeStmtStructureProperties{
    {IRProperty::TypeChecked},
    {IRProperty::TypeChecked, IRProperty::NormalizedStmtStructure},
    {IRProperty::FlattenedSingleStmt}};

inline const PassProperties kFlattenSingleStmtProperties{{IRProperty::TypeChecked},
                                                         {IRProperty::TypeChecked, IRProperty::FlattenedSingleStmt},
                                                         {IRProperty::NormalizedStmtStructure}};

inline const PassProperties kOutlineIncoreScopesProperties{
    {IRProperty::TypeChecked, IRProperty::SSAForm}, {IRProperty::SplitIncoreOrch}, {}};

inline const PassProperties kConvertTensorToBlockOpsProperties{
    {IRProperty::SplitIncoreOrch}, {IRProperty::IncoreBlockOps}, {}};

inline const PassProperties kInitMemRefProperties{
    {IRProperty::TypeChecked, IRProperty::SSAForm, IRProperty::SplitIncoreOrch, IRProperty::IncoreBlockOps},
    {IRProperty::HasMemRefs, IRProperty::NormalizedStmtStructure},
    {IRProperty::SSAForm}};

inline const PassProperties kBasicMemoryReuseProperties{
    {IRProperty::TypeChecked, IRProperty::SplitIncoreOrch, IRProperty::IncoreBlockOps, IRProperty::HasMemRefs}, {}, {}};

inline const PassProperties kAllocateMemoryAddrProperties{
    {IRProperty::TypeChecked, IRProperty::SplitIncoreOrch, IRProperty::IncoreBlockOps, IRProperty::HasMemRefs},
    {IRProperty::AllocatedMemoryAddr},
    {}};

} // namespace pass
} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TRANSFORMS_PASS_PROPERTIES_H_
