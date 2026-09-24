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
 * \file attr_holder.h
 * \brief
 */

#pragma once
#include <iostream>
#include <map>
#include <string>
#include <any>

#include "core/any_cast.h"

#include "tilefwk/symbolic_scalar.h"
#include "interface/utils/common.h"
#include "interface/utils/string_utils.h"
#include "tilefwk/error_code.h"
#include "interface/inner/element.h"

namespace npu::tile_fwk {
const std::string OP_ATTR_PREFIX = "op_attr_";
const std::string OP_EMUOP_PREFIX = "op_emuop_";

class AttrHolder {
protected:
    std::map<std::string, std::any> attributes;

private:
    static std::string DumpElementAttr(const Element& tensorElement)
    {
        if (tensorElement.IsSigned()) {
            return std::to_string(tensorElement.GetSignedData());
        }
        if (tensorElement.IsUnsigned()) {
            return std::to_string(tensorElement.GetUnsignedData());
        }
        if (tensorElement.IsFloat()) {
            return std::to_string(tensorElement.GetFloatData());
        }
        return "";
    }

    static std::string DumpSymbolicScalarListAttr(const std::vector<SymbolicScalar>& scalarList)
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t k = 0; k < scalarList.size(); k++) {
            oss << ((k != 0) ? "," : "") << scalarList[k].Dump();
        }
        oss << "]";
        return oss.str();
    }

    static std::string DumpMapIntIntAttr(const std::map<int, int>& dict)
    {
        std::ostringstream oss;
        size_t index = 0;
        for (const auto& [src, dst] : dict) {
            oss << ((index++ == 0) ? "" : ",") << src << ":" << dst;
        }
        return oss.str();
    }

public:
    const std::map<std::string, std::any>& GetAllAttr() const { return attributes; }
    std::map<std::string, std::any>& GetAllAttr() { return attributes; }

    bool HasAttr(const std::string& key) const
    {
        if (key.empty()) {
            return false;
        }
        return attributes.find(key) != attributes.end();
    }

    // 设置属性值
    template <typename T>
    void SetAttr(const std::string& key, const T& value)
    {
        static_assert(!std::is_same_v<T, int>);
        static_assert(!std::is_same_v<T, std::vector<int>>);
        attributes[key] = value;
    }

    std::any GetRawAttr(const std::string& key) const
    {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            return it->second;
        }
        return std::any();
    }

    template <typename T>
    bool GetAttr(const std::string& key, T& value) const
    {
        static_assert(!std::is_same_v<T, int>);
        static_assert(!std::is_same_v<T, std::vector<int>>);
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            if (it->second.type() == typeid(T)) {
                value = pypto::AnyCast<T>(it->second);
            } else {
                std::cout << "Type mismatch: " << it->second.type().name() << " != " << typeid(T).name() << std::endl;
                return false;
            }
        } else {
            return false;
        }
        return true;
    }

    template <typename T>
    T* GetAttr(const std::string& key)
    {
        auto it = attributes.find(key);
        if (it != attributes.end() && it->second.type() == typeid(T)) {
            return &pypto::AnyCastRef<T>(it->second);
        }
        return nullptr;
    }

    // 移除属性
    void RemoveAttr(const std::string& key)
    {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            attributes.erase(it);
        } else {
            throw std::out_of_range("Attribute not found: " + key);
        }
    }

    void CopyAttrFrom(const AttrHolder& holder, const std::string& prefix)
    {
        for (const auto& pair : holder.attributes) {
            if (StringUtils::StartsWith(pair.first, prefix)) {
                attributes[pair.first] = pair.second;
            }
        }
    }

    std::string DumpAttr() const
    {
        std::ostringstream oss;
        int index = 0;
        for (auto& it : attributes) {
            oss << ((index++ == 0) ? "" : " ");
            oss << "#" << it.first << "{" << DumpAttr(it.first) << "}";
        }
        return oss.str();
    }

    // 打印所有属性
    std::string DumpAttr(const std::string& key) const
    {
        auto it = attributes.find(key);
        if (it == attributes.end()) {
            return "Invalid attribute key " + key;
        }

        std::string result;
        if (it->second.type() == typeid(int64_t)) {
            result = std::to_string(pypto::AnyCastRef<int64_t>(it->second));
        } else if (it->second.type() == typeid(float)) {
            result = std::to_string(pypto::AnyCastRef<float>(it->second));
        } else if (it->second.type() == typeid(double)) {
            result = std::to_string(pypto::AnyCastRef<double>(it->second));
        } else if (it->second.type() == typeid(std::string)) {
            result = pypto::AnyCastRef<std::string>(it->second);
        } else if (it->second.type() == typeid(bool)) {
            result = std::to_string(pypto::AnyCastRef<bool>(it->second));
        } else if (it->second.type() == typeid(std::vector<int64_t>)) {
            result = IntVecToStr(pypto::AnyCastRef<std::vector<int64_t>>(it->second));
        } else if (it->second.type() == typeid(Element)) {
            result = DumpElementAttr(pypto::AnyCastRef<Element>(it->second));
        } else if (it->second.type() == typeid(SymbolicScalar)) {
            auto scalar = pypto::AnyCastRef<SymbolicScalar>(it->second);
            result = scalar.Dump();
        } else if (it->second.type() == typeid(std::vector<SymbolicScalar>)) {
            result = DumpSymbolicScalarListAttr(pypto::AnyCastRef<std::vector<SymbolicScalar>>(it->second));
        } else if (it->second.type() == typeid(std::vector<bool>)) {
            auto scalarList = pypto::AnyCastRef<std::vector<bool>>(it->second);
            result = IntVecToStr<bool>(scalarList);
        } else if (it->second.type() == typeid(std::map<int, int>)) {
            result = DumpMapIntIntAttr(pypto::AnyCastRef<std::map<int, int>>(it->second));
        } else {
            result += "unsupported type ";
            result += it->second.type().name();
        }
        return result;
    }

    nlohmann::json DumpAttrJson() const
    {
        nlohmann::json attrJson;
        for (const auto& pair : attributes) {
            attrJson[pair.first] = DumpAttr(pair.first);
        }
        return attrJson;
    }

    nlohmann::json DumpAttrJson(const std::string& key) const
    {
        auto iter = attributes.find(key);
        if (iter != attributes.end()) {
            auto& second = iter->second;
            try {
                if (second.type() == typeid(int64_t)) {
                    return nlohmann::json(pypto::AnyCastRef<int64_t>(second));
                } else if (second.type() == typeid(std::vector<int64_t>)) {
                    return nlohmann::json(pypto::AnyCastRef<std::vector<int64_t>>(second));
                } else if (second.type() == typeid(std::vector<float>)) {
                    return nlohmann::json(pypto::AnyCastRef<std::vector<float>>(second));
                } else if (second.type() == typeid(std::vector<bool>)) {
                    return nlohmann::json(pypto::AnyCastRef<std::vector<bool>>(second));
                } else if (second.type() == typeid(double)) {
                    return nlohmann::json(pypto::AnyCastRef<double>(second));
                } else if (second.type() == typeid(float)) {
                    return nlohmann::json(pypto::AnyCastRef<float>(second));
                } else if (second.type() == typeid(std::string)) {
                    return nlohmann::json(pypto::AnyCastRef<std::string>(second));
                } else if (second.type() == typeid(bool)) {
                    return nlohmann::json(pypto::AnyCastRef<bool>(second));
                } else if (second.type() == typeid(Element)) {
                    return ToJson(pypto::AnyCastRef<Element>(second));
                } else if (second.type() == typeid(std::map<int, int>)) {
                    nlohmann::json dict = nlohmann::json::object();
                    auto inplaceInfo = pypto::AnyCast<std::map<int, int>>(second);
                    for (const auto& [src, dst] : inplaceInfo) {
                        dict[std::to_string(src)] = dst;
                    }
                    return dict;
                } else {
                    return nlohmann::json("Unsupported type");
                }
            } catch (const std::bad_any_cast&) {
                std::cout << "Bad any cast" << second.type().name();
            }
        }
        return nlohmann::json();
    }

    void LoadVecAttr(const std::string& key, const std::vector<nlohmann::json>& vec)
    {
        if (vec[0].is_string()) {
            std::vector<std::string> strVec;
            for (const auto& j : vec) {
                strVec.emplace_back(j.get<std::string>());
            }
            SetAttr(key, strVec);
        } else if (vec[0].is_number()) {
            if (vec[0].is_number_integer()) {
                std::vector<int64_t> intVec;
                for (const auto& j : vec) {
                    intVec.emplace_back(j.get<int64_t>());
                }
                SetAttr(key, intVec);
            } else {
                std::vector<float> floatVec;
                for (const auto& j : vec) {
                    floatVec.emplace_back(j.get<float>());
                }
                SetAttr(key, floatVec);
            }
        } else if (vec[0].is_boolean()) {
            std::vector<bool> boolVec;
            for (const auto& j : vec) {
                boolVec.emplace_back(j.get<bool>());
            }
            SetAttr(key, boolVec);
        } else {
            return;
        }
    }

    void LoadAttrJson(const std::string& key, const nlohmann::json& attrJson)
    {
        try {
            if (attrJson.is_array()) {
                // 处理数组
                std::vector<nlohmann::json> vec;
                for (const auto& elem : attrJson) {
                    vec.push_back(elem);
                }
                if (!vec.empty()) {
                    LoadVecAttr(key, vec);
                }
            } else if (attrJson.is_object()) {
                SetAttr(key, parseElement(attrJson));
            } else if (attrJson.is_string()) {
                SetAttr(key, attrJson.get<std::string>());
            } else if (attrJson.is_number()) {
                if (attrJson.is_number_integer()) {
                    SetAttr(key, attrJson.get<int64_t>());
                } else {
                    SetAttr(key, attrJson.get<float>());
                }
            } else if (attrJson.is_boolean()) {
                SetAttr(key, attrJson.get<bool>());
            } else if (attrJson.is_null()) {
                return;
            }
        } catch (...) {
            FE_LOGE(FeError::INVALID_FILE, "json parse error");
        }
    }
};
} // namespace npu::tile_fwk
