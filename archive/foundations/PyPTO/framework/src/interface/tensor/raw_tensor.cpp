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
 * \file raw_tensor.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/serialization.h"
#include "raw_tensor.h"
#include <string>
#include <cstdint>

using namespace npu::tile_fwk;

RawTensor::RawTensor(DataType t, std::vector<int64_t> tshape, TileOpFormat tformat, std::string tname, int trawmagic)
    : rawmagic((trawmagic == -1) ? IdGen<IdType::RAW_TENSOR>::Inst().NewId() : trawmagic),
      rawshape(std::move(tshape)),
      datatype(t),
      format(tformat),
      symbol(std::move(tname))
{
    dynRawShape = SymbolicScalar::FromConcrete(rawshape);
    memoryId = rawmagic;
}

Json RawTensor::DumpJson() const
{
    Json result;
    result[T_FIELD_KIND] = static_cast<int>(Kind::T_KIND_RAW_TENSOR);
    result["datatype"] = datatype;
    result["format"] = format;
    result["rawshape"] = rawshape;
    result["rawmagic"] = rawmagic;
    if (actualRawmagic != -1) {
        result["actual_rawmagic"] = actualRawmagic;
    }

    if (!symbol.empty()) {
        result["symbol"] = symbol;
    }
    return result;
}

std::shared_ptr<RawTensor> RawTensor::LoadJson(const Json& rawTensorDump)
{
    FE_ASSERT(rawTensorDump[T_FIELD_KIND].get<int>() == static_cast<int>(Kind::T_KIND_RAW_TENSOR))
        << rawTensorDump[T_FIELD_KIND].get<int>() << " != " << static_cast<int>(Kind::T_KIND_RAW_TENSOR);
    DataType dtype = static_cast<DataType>(rawTensorDump["datatype"].get<int>());
    TileOpFormat format = static_cast<TileOpFormat>(rawTensorDump["format"].get<int>());
    std::vector<int64_t> rawshapeJson = rawTensorDump["rawshape"].get<std::vector<int64_t>>();
    int dumpRawmagic = rawTensorDump["rawmagic"].get<int>();
    std::string dumpSymbol;
    if (rawTensorDump.contains("symbol")) {
        dumpSymbol = rawTensorDump["symbol"].get<std::string>();
    }
    auto ret = std::make_shared<RawTensor>(dtype, rawshapeJson, format, dumpSymbol, dumpRawmagic);
    if (rawTensorDump.count("actual_rawmagic") != 0) {
        ret->actualRawmagic = rawTensorDump["actual_rawmagic"].get<int>();
    }
    return ret;
}

std::string RawTensor::DumpType() const
{
    std::string result = "<";
    for (auto& value : rawshape) {
        result += std::to_string((value)) + " x ";
    }
    result += DataType2String(datatype);
    if (format == TileOpFormat::TILEOP_NZ) {
        result += "_NZ";
    }
    result += ">";
    return result;
}

std::string RawTensor::DumpSSA(bool showType, bool showSymbol) const
{
    std::ostringstream oss;
    if (showType) {
        oss << DumpType() << " ";
    }
    oss << "@" << GetRawMagic();
    if (showSymbol) {
        if (GetSymbol().size() != 0) {
            oss << "\"" << GetSymbol() << "\"";
        }
    }
    return oss.str();
}

std::string RawTensor::Dump() const { return DumpSSA(); }

bool RawTensor::IsDummy() const { return isDummy_; }

void RawTensor::SetIsDummy(bool dummy) { isDummy_ = dummy; }

void RawTensor::AddRefCount(int value)
{
    FE_ASSERT(value == 1 || value == -1) << "value: " << value;
    refCount_ += value;
    if (refCount_ < 0) {
        FE_LOGI("rawmagic = %d, refCount_ is negative: %d", rawmagic, refCount_);
    }
}

int64_t RawTensor::GetRawDataSize() const
{
    if (HasNegativeNum<int64_t>(rawshape)) {
        FE_LOGD("Raw tensor shape has negative. It has dynamic axis.");
        return INT64_MAX;
    }
    return DataSizeOf(GetRawShapeSize(), datatype);
}

int64_t RawTensor::GetRawShapeSize() const
{
    return std::accumulate(rawshape.begin(), rawshape.end(), INT64_C(1), std::multiplies<int64_t>());
}
