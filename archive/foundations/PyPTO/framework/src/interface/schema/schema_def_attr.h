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
 * \file schema_def_attr.h
 * \brief
 */

#pragma once

SCHEMA_DEF_TYPE_INT64(rawTensor, '@');
SCHEMA_DEF_TYPE_INT64(tensor, '%');
SCHEMA_DEF_TYPE_INT64(operation, '!');

SCHEMA_DEF_ATTR(incast, Int64Type);
SCHEMA_DEF_ATTR(outcast, Int64Type);
SCHEMA_DEF_ATTR(name, StringType);

SCHEMA_DEF_TYPE_UINT64(memType);
SCHEMA_DEF_ATTR(mem, memType);
SCHEMA_DEF_ATTR(off, memType);
SCHEMA_DEF_ATTR(memOut, memType);

SCHEMA_DEF_TYPE_ARRAY(coaType, TextType);
SCHEMA_DEF_ATTR(coa, coaType);

SCHEMA_DEF_TYPE_ARRAY(shapeList, Int64Type);
SCHEMA_DEF_ATTR(shape, shapeList);

SCHEMA_DEF_TYPE_ARRAY(offsetList, Int64Type);
SCHEMA_DEF_ATTR(offset, offsetList);

SCHEMA_DEF_ATTR(pred, Int64Type);
SCHEMA_DEF_TYPE_ARRAY(OperationList, operation);
SCHEMA_DEF_ATTR(succ, OperationList);

SCHEMA_DEF_TYPE_ARRAY(outSuccIndexList, Int64Type);
SCHEMA_DEF_ATTR(outSuccIndex, outSuccIndexList);

SCHEMA_DEF_TYPE_ARRAY(ExpressionTable, Int64Type);
SCHEMA_DEF_ATTR(expr, ExpressionTable);
SCHEMA_DEF_ATTR(range, AddressType, AddressType, AddressType);
SCHEMA_DEF_TYPE_UNION(AttrType, expr, range, incast, outcast, name, coa);

SCHEMA_DEF_ATTR(rawDesc, Int32Type, Int64Type, Int64Type);
