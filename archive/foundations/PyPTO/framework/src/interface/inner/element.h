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
 * \file element.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <securec.h>
#include "tilefwk/element.h"
#include "interface/tensor/float.h"

using Json = nlohmann::json;

constexpr int SET_NUM_TWO = 2;

namespace std {
inline std::string to_string(npu::tile_fwk::bfloat16 data) { return std::to_string(static_cast<float>(data)); }
inline std::string to_string(npu::tile_fwk::float16 data) { return std::to_string(static_cast<float>(data)); }
} // namespace std

#define DISPATCH_DATA_TYPE(f, ...)                              \
    f(DT_INT8, int8_t, int64_t, ##__VA_ARGS__);                 \
    f(DT_INT16, int16_t, int64_t, ##__VA_ARGS__);               \
    f(DT_INT32, int32_t, int64_t, ##__VA_ARGS__);               \
    f(DT_INT64, int64_t, int64_t, ##__VA_ARGS__);               \
    f(DT_FP16, npu::tile_fwk::float16, double, ##__VA_ARGS__);  \
    f(DT_FP32, float, double, ##__VA_ARGS__);                   \
    f(DT_BF16, npu::tile_fwk::bfloat16, double, ##__VA_ARGS__); \
    f(DT_UINT8, uint8_t, uint64_t, ##__VA_ARGS__);              \
    f(DT_UINT16, uint16_t, uint64_t, ##__VA_ARGS__);            \
    f(DT_UINT32, uint32_t, uint64_t, ##__VA_ARGS__);            \
    f(DT_UINT64, uint64_t, uint64_t, ##__VA_ARGS__);            \
    f(DT_DOUBLE, double, double, ##__VA_ARGS__)

namespace npu::tile_fwk {

Json ToJson(const Element& elem);
Element parseElement(const Json& j);
std::vector<float> ConvElementVecToFloatVec(const std::vector<Element>& input);
std::vector<int> ConvElementVecToIntVec(const std::vector<Element>& input);

struct ElementDump {
    struct SmallString {
        uint8_t size;
        char data[sizeof(uint64_t) * SET_NUM_TWO - 1];
    };
    union {
        SmallString data;
        uint64_t padding[2];
    };

    void Clear()
    {
        padding[0] = 0;
        padding[1] = 0;
    }

    ElementDump() { Clear(); }

    const char* c_str() const { return data.data; }
    size_t size() const { return data.size; }

    bool operator!=(const ElementDump rhs) const
    {
        return padding[0] != rhs.padding[0] || padding[1] != rhs.padding[1];
    }

    void DumpElement(int64_t v)
    {
        Clear();
        data.size = snprintf_s(data.data, sizeof(data.data), sizeof(data.data) - 1, "%ld", static_cast<long>(v));
    }
    void DumpElement(uint64_t v)
    {
        Clear();
        data.size = snprintf_s(data.data, sizeof(data.data), sizeof(data.data) - 1, "%lu",
                               static_cast<unsigned long>(v));
    }
    void DumpElement(double v)
    {
        Clear();
        double vabs = std::abs(v);
        if (vabs < 1e-4 || vabs > 1e6) {
            data.size = snprintf_s(data.data, sizeof(data.data), sizeof(data.data) - 1, "%0.5le", v);
        } else if (vabs < 1) {
            data.size = snprintf_s(data.data, sizeof(data.data), sizeof(data.data) - 1, "%0.9lf", v);
        } else {
            data.size = snprintf_s(data.data, sizeof(data.data), sizeof(data.data) - 1, "%0.5lf", v);
        }
    }
};

} // namespace npu::tile_fwk
