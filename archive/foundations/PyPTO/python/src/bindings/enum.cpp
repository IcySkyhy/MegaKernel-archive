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
 * \file enum.cpp
 * \brief
 */

#include "pybind_common.h"
#include "tilefwk/error_code.h"

using namespace npu::tile_fwk;

namespace pypto {
void BindEnum(py::module_& m)
{
    // clang-format off
    py::enum_<DataType>(m, "DataType")
#define DTYPE_DESC(name, byte, bit, is_float, type, cann_value) \
        .value(#name, DataType::name)
        DATA_TYPE_ALL
#undef DTYPE_DESC
        .value("DT_BOTTOM", DataType::DT_BOTTOM)
        .export_values();
    // clang-format on

    py::enum_<NodeType>(m, "NodeType")
        .value("LOCAL", NodeType::LOCAL)
        .value("INCAST", NodeType::INCAST)
        .value("OUTCAST", NodeType::OUTCAST)
        .export_values();

    py::enum_<ExternalError>(m, "ExternalError")
        .value("COMMON_EXTERNAL_ERROR", ExternalError::COMMON_EXTERNAL_ERROR)
        .value("RUNTIME_ERROR", ExternalError::RUNTIME_ERROR)
        .value("NAME_ERROR", ExternalError::NAME_ERROR)
        .value("NOT_IMPLEMENTED_ERROR", ExternalError::NOT_IMPLEMENTED_ERROR)
        .value("KEY_ERROR", ExternalError::KEY_ERROR)
        .value("INVALID_OPERATION", ExternalError::INVALID_OPERATION)
        .value("INVALID_TYPE", ExternalError::INVALID_TYPE)
        .value("INVALID_VAL", ExternalError::INVALID_VAL)
        .value("OUT_OF_RANGE", ExternalError::OUT_OF_RANGE)
        .value("BAD_FD", ExternalError::BAD_FD)
        .value("DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED", ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED)
        .value("UNKNOWN", ExternalError::UNKNOWN)
        .export_values();

    py::enum_<TileOpFormat>(m, "TileOpFormat")
        .value("TILEOP_ND", TileOpFormat::TILEOP_ND)
        .value("TILEOP_NZ", TileOpFormat::TILEOP_NZ)
        .export_values();

    py::enum_<CachePolicy>(m, "CachePolicy").value("NONE_CACHEABLE", CachePolicy::NONE_CACHEABLE).export_values();

    py::enum_<ReduceMode>(m, "ReduceMode").value("ATOMIC_ADD", ReduceMode::ATOMIC_ADD).export_values();

    py::enum_<ScatterMode>(m, "ScatterMode")
        .value("NONE", ScatterMode::NONE)
        .value("ADD", ScatterMode::ADD)
        .value("MULTIPLY", ScatterMode::MULTIPLY)
        .export_values();

    py::enum_<FunctionType>(m, "FunctionType")
        .value("STATIC", FunctionType::STATIC)
        .value("DYNAMIC", FunctionType::DYNAMIC)
        .value("DYNAMIC_LOOP", FunctionType::DYNAMIC_LOOP)
        .export_values();

    py::enum_<GraphType>(m, "GraphType").value("TENSOR_GRAPH", GraphType::TENSOR_GRAPH).export_values();

    py::enum_<CastMode>(m, "CastMode")
        .value("CAST_NONE", CastMode::CAST_NONE)
        .value("CAST_RINT", CastMode::CAST_RINT)
        .value("CAST_ROUND", CastMode::CAST_ROUND)
        .value("CAST_FLOOR", CastMode::CAST_FLOOR)
        .value("CAST_CEIL", CastMode::CAST_CEIL)
        .value("CAST_TRUNC", CastMode::CAST_TRUNC)
        .value("CAST_ODD", CastMode::CAST_ODD)
        .export_values();

    py::enum_<SaturationMode>(m, "SaturationMode")
        .value("OFF", SaturationMode::OFF)
        .value("ON", SaturationMode::ON)
        .export_values();

    py::enum_<AtomicRMWMode>(m, "AtomicRMWMode")
        .value("ADD", AtomicRMWMode::ADD)
        .value("MIN", AtomicRMWMode::MIN)
        .value("MAX", AtomicRMWMode::MAX)
        .export_values();

    py::enum_<PrecisionType>(m, "PrecisionType")
        .value("INTRINSIC", PrecisionType::INTRINSIC)
        .value("HIGH_PRECISION", PrecisionType::HIGH_PRECISION)
        .export_values();

    py::enum_<DequantScaleRoundingMode>(m, "DequantScaleRoundingMode")
        .value("ROUND_UP", DequantScaleRoundingMode::ROUND_UP)
        .value("ROUND_DOWN", DequantScaleRoundingMode::ROUND_DOWN)
        .export_values();

    py::enum_<TileType>(m, "TileType")
        .value("VEC", TileType::VEC)
        .value("CUBE", TileType::CUBE)
        .value("DIST", TileType::DIST)
        .value("MAX", TileType::MAX)
        .export_values();

    py::enum_<OpType>(m, "OpType")
        .value("EQ", OpType::EQ)
        .value("NE", OpType::NE)
        .value("LT", OpType::LT)
        .value("LE", OpType::LE)
        .value("GT", OpType::GT)
        .value("GE", OpType::GE)
        .export_values();

    py::enum_<OutType>(m, "OutType").value("BOOL", OutType::BOOL).value("BIT", OutType::BIT).export_values();

    py::enum_<TopKAlgo>(m, "TopKAlgo")
        .value("MERGE_SORT", TopKAlgo::MERGE_SORT)
        .value("RADIX_SELECT", TopKAlgo::RADIX_SELECT)
        .export_values();

    py::enum_<Matrix::ReLuType>(m, "ReLuType")
        .value("NO_RELU", Matrix::ReLuType::NoReLu)
        .value("RELU", Matrix::ReLuType::ReLu)
        .export_values();

    py::enum_<Conv::ReLuType>(m, "ConvReLuType")
        .value("NO_RELU", Conv::ReLuType::NoReLu)
        .value("RELU", Conv::ReLuType::ReLu)
        .export_values();

    py::enum_<Matrix::TransMode>(m, "TransMode")
        .value("CAST_NONE", Matrix::TransMode::CAST_NONE)
        .value("CAST_RINT", Matrix::TransMode::CAST_RINT)
        .value("CAST_ROUND", Matrix::TransMode::CAST_ROUND)
        .export_values();

    py::enum_<LogBaseType>(m, "LogBaseType")
        .value("LOG_E", LogBaseType::LOG_E)
        .value("LOG_2", LogBaseType::LOG_2)
        .value("LOG_10", LogBaseType::LOG_10)
        .export_values();

    py::enum_<Distributed::AtomicType>(m, "AtomicType")
        .value("SET", Distributed::AtomicType::SET)
        .value("ADD", Distributed::AtomicType::ADD)
        .export_values();
}
} // namespace pypto
