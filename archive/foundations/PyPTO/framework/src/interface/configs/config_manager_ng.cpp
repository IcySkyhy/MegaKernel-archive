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
 * \file config_manager_ng.cpp
 * \brief
 */
#include <string>
#include <map>
#include <typeinfo>
#include <fstream>
#include <sstream>
#include <list>
#include <stack>
#include <mutex>
#include <typeindex>
#include <climits>
#include <utility>

#include <nlohmann/json.hpp>

#include "interface/utils/common.h"
#include "utils/file_utils.h"
#include "interface/utils/string_utils.h"

#include "config_manager_ng.h"
#include "tilefwk/tile_shape.h"
#include "tilefwk/pypto_fwk_log.h"
#include "interface/utils/error.h"

namespace npu::tile_fwk {

namespace {
std::mutex mtx;

std::string GetReadableTypeName(const std::type_info& type)
{
    static const std::unordered_map<std::type_index, std::string> kTypeNames = {
        {typeid(void), "unknown"},
        {typeid(bool), "bool"},
        {typeid(int64_t), "int64"},
        {typeid(double), "double"},
        {typeid(std::string), "string"},
        {typeid(std::vector<int64_t>), "list[int64]"},
        {typeid(std::vector<double>), "list[double]"},
        {typeid(std::vector<std::string>), "list[string]"},
        {typeid(std::map<int64_t, int64_t>), "dict[int64, int64]"},
        {typeid(CubeTile), "CubeTile"},
        {typeid(ConvTile), "ConvTile"},
        {typeid(DistTile), "DistTile"},
    };

    auto it = kTypeNames.find(std::type_index(type));
    return it != kTypeNames.end() ? it->second : type.name();
}
} // namespace

struct TypeInfo {
    TypeInfo() = default;

    void LoadConf(const std::string& path)
    {
        std::ifstream infile(path);
        FE_ASSERT(FeError::BAD_FD, infile.is_open()) << "Open file " << path << " failed";
        nlohmann::json jData;
        infile >> jData;

        build_type_infos(jData, "");
    }

    void build_type_infos(const nlohmann::json& jData, const std::string& prefix)
    {
        if (jData.contains("properties")) {
            // 递归解析properties字段
            const auto& properties = jData["properties"];
            for (const auto& [key, value] : properties.items()) {
                const std::string new_prefix = prefix.empty() ? key : prefix + "." + key;
                build_type_infos(value, new_prefix);
            }
        } else if (jData.contains("type")) {
            // 处理type字段
            const std::string& type = jData["type"];
            if (type == "string") {
                typeInfos.insert({prefix, typeid(std::string)});
            } else if (type == "integer") {
                typeInfos.insert({prefix, typeid(int64_t)});
                parse_range_info(jData, prefix, "minimum", "maximum");
            } else if (type == "boolean") {
                typeInfos.insert({prefix, typeid(bool)});
            } else if (type == "array") {
                parse_array_type(jData, prefix);
            } else if (type == "object") {
                parse_object_type(jData, prefix);
            } else {
                FE_LOGE(FeError::INVALID_TYPE, "invalid type: %s at %s", type.c_str(), prefix.c_str());
            }
        } else {
            FE_LOGE(FeError::NOT_EXIST,
                    "Label<%s> field['type', 'properties'] not found in tile_fwk_config_schema.json", prefix.c_str());
        }
    }

    void parse_range_info(const nlohmann::json& jData, const std::string& prefix, const std::string& min_key,
                          const std::string& max_key)
    {
        int64_t minBound = jData.contains(min_key) ? jData[min_key].get<int64_t>() : INT_MIN;
        int64_t maxBound = jData.contains(max_key) ? jData[max_key].get<int64_t>() : INT_MAX;
        rangeInfos.insert({prefix, {minBound, maxBound}});
    }

    void parse_array_type(const nlohmann::json& jData, const std::string& prefix)
    {
        const std::string& jitem_type = jData["items"]["type"];
        if (jitem_type == "string") {
            typeInfos.insert({prefix, typeid(std::vector<std::string>)});
        } else if (jitem_type == "integer") {
            typeInfos.insert({prefix, typeid(std::vector<int64_t>)});
        } else if (jitem_type == "double") {
            typeInfos.insert({prefix, typeid(std::vector<double>)});
        }
    }

    void parse_object_type(const nlohmann::json& jData, const std::string& prefix)
    {
        const std::string& typeHints = jData["typeHints"];
        if (typeHints == "intmap") {
            typeInfos.insert({prefix, typeid(std::map<int64_t, int64_t>)});
            parse_range_info(jData, prefix + "_key", "key_minimum", "key_maximum");
            parse_range_info(jData, prefix + "_val", "value_minimum", "value_maximum");
        } else if (typeHints == "stringmap") {
            typeInfos.insert({prefix, typeid(std::map<std::string, int64_t>)});
        }
    }

    const std::type_info& Type(const std::string& name) const
    {
        if (typeInfos.find(name) == typeInfos.end()) {
            return typeid(void);
        }
        return typeInfos.at(name);
    }

    std::map<std::string, const std::type_info&> typeInfos;
    std::map<std::string, std::pair<int64_t, int64_t>> rangeInfos;
};

const std::any& ConfigScope::GetAnyConfig(const std::string& key) const
{
    if (values_.find(key) == values_.end()) {
        if (parent_) {
            return parent_->GetAnyConfig(key);
        }
        throw std::runtime_error("Config " + key + " not found");
    }
    return values_.at(key);
}

bool ConfigScope::HasConfig(const std::string& key) const
{
    if (key == "matrix_size" || key == "vec_tile_shapes" || key == "conv_tile_shapes" || key == "cube_tile_shapes") {
        return true;
    }

    return values_.find(key) != values_.end() || (parent_ && parent_->HasConfig(key));
}

void ConfigScope::Clear()
{
    values_.clear();
    FE_LOGD("Clear config scope successfully.");
}

const std::type_info& ConfigScope::Type(const std::string& key) const
{
    return ConfigManagerNg::GetInstance().Type(key);
}

ConfigScope::ConfigScope(ConfigScopePtr parent) : parent_(parent)
{
    if (parent_) {
        parent_->children_.push_back(this);
    }
}

TileShape ConfigScope::GenerateTileShape() const
{
    std::vector<int64_t> vecTile = GetConfig<std::vector<int64_t>>("vec_tile_shapes");
    CubeTile cubeTile = GetConfig<CubeTile>("cube_tile_shapes");
    ConvTile convTile = GetConfig<ConvTile>("conv_tile_shapes");
    DistTile distTile = GetConfig<DistTile>("dist_tile_shapes");
    std::vector<int64_t> matrixSize = GetConfig<std::vector<int64_t>>("matrix_size");
    TileShape tileShape(vecTile, cubeTile, convTile, distTile, matrixSize);
    return tileShape;
}

ConfigScope::~ConfigScope()
{
    if (parent_) {
        parent_->children_.remove(this);
    }
}

void DumpMap(std::stringstream& os, const std::map<int64_t, int64_t>& map)
{
    os << '{';
    bool is_first = true;
    for (const auto& [k, v] : map) {
        if (!is_first) {
            os << ", ";
        }
        os << "{" << k << ", " << v << "}";
        is_first = false;
    }
    os << '}';
}

void DumpValue(std::stringstream& os, const std::string& key, const std::any& val, const std::string& prefix)
{
    os << prefix << key << ": ";
    const auto& type = val.type();

    if (type == typeid(int64_t)) {
        os << (AnyCast<int64_t>(val));
    } else if (type == typeid(bool)) {
        os << AnyCast<bool>(val);
    } else if (type == typeid(std::string)) {
        os << (AnyCast<std::string>(val));
    } else if (type == typeid(std::vector<int64_t>)) {
        os << (AnyCast<std::vector<int64_t>>(val));
    } else if (type == typeid(std::vector<std::string>)) {
        os << (AnyCast<std::vector<std::string>>(val));
    } else if (type == typeid(std::map<int64_t, int64_t>)) {
        DumpMap(os, AnyCast<std::map<int64_t, int64_t>>(val));
    } else if (type == typeid(CubeTile)) {
        os << (AnyCast<CubeTile>(val).ToString());
    } else if (type == typeid(DistTile)) {
        os << (AnyCast<DistTile>(val).ToString());
    } else {
        os << "unknow type: " << type.name();
    }
}

void DumpValues(std::stringstream& os, const std::map<std::string, std::any>& values, const std::string& prefix)
{
    for (const auto& [key, val] : values) {
        DumpValue(os, key, val, prefix);
        os << "\n";
    }
}

void DumpRange(std::stringstream& os, const std::type_info& type, const std::string& key,
               const std::map<std::string, std::pair<int64_t, int64_t>>& rangeInfos)
{
    os << "Range: ";
    if (type == typeid(std::map<int64_t, int64_t>)) {
        os << "{[" << rangeInfos.at(key + "_key").first << ", " << rangeInfos.at(key + "_key").second << "], ["
           << rangeInfos.at(key + "_val").first << ", " << rangeInfos.at(key + "_val").second << "]}";
    } else {
        os << "[" << rangeInfos.at(key).first << ", " << rangeInfos.at(key).second << "]";
    }
}

bool HasRangeConstraint(const std::string& key, const std::type_info& type,
                        const std::map<std::string, std::pair<int64_t, int64_t>>& rangeInfos)
{
    if (rangeInfos.count(key) != 0) {
        return true;
    }

    if (type == typeid(std::map<int64_t, int64_t>)) {
        return rangeInfos.count(key + "_key") != 0 && rangeInfos.count(key + "_val") != 0;
    }

    return false;
}

void ValidateConfigValueType(const std::string& key, const std::any& value)
{
    const auto& expectedType = ConfigManagerNg::GetInstance().Type(key);
    if (expectedType == typeid(void) || value.type() == expectedType) {
        return;
    }

    std::stringstream os;
    os << "Option '" << key << "' has invalid type. Expected " << GetReadableTypeName(expectedType) << ", but got "
       << GetReadableTypeName(value.type());
    CHECK(ExternalError::INVALID_TYPE, false) << os.str();
}

std::string ConfigScope::ToString() const
{
    auto values = GetAllConfig();
    std::stringstream os;
    DumpValues(os, values, "");
    os << "\n";
    return os.str();
}

const std::map<std::string, std::any> ConfigScope::GetAllConfig() const
{
    std::map<std::string, std::any> values;
    auto scope = this;
    while (scope) {
        for (auto& [key, val] : scope->values_) {
            if (!values.count(key)) {
                values[key] = val;
            }
        }
        scope = scope->parent_.get();
    }
    return values;
}

void ConfigScope::AddValue(const std::string& key, std::any value)
{
    std::lock_guard<std::mutex> lock(mtx);
    values_[key] = value;
}

void ConfigScope::UpdateValueWithAny(const std::string& key, std::any value)
{
    ValidateConfigValueType(key, value);
    const auto& rangeInfos = ConfigManagerNg::GetInstance().Range();
    if (HasRangeConstraint(key, value.type(), rangeInfos) &&
        !ConfigManagerNg::GetInstance().IsWithinRange(key, value)) {
        std::stringstream os("Option:");
        std::map<std::string, std::any> node;
        node[key] = value;
        DumpValues(os, node, "");
        os << "its value doesn't within the value range. ";
        DumpRange(os, value.type(), key, rangeInfos);
        CHECK(ExternalError::INVALID_VAL, false) << os.str();
    }
    std::stringstream oss;
    DumpValue(oss, key, value, "");
    FE_LOGD("Set option successfully: %s ", oss.str().c_str());
    std::lock_guard<std::mutex> lock(mtx);
    values_[key] = value;
}

struct ConfigManagerImpl {
    TypeInfo typeInfo;
    std::stack<ConfigScopePtr> scopes;
    ConfigScopePtr root;

    ConfigManagerImpl()
    {
        typeInfo.LoadConf(GetConfDir() + "tile_fwk_config_schema.json");
        root = std::make_shared<ConfigScope>(nullptr);
        root->name_ = "default";
        LoadConf();
        InitTileShape();
        scopes.push(root);
        auto global = std::make_shared<ConfigScope>(root);
        global->name_ = "global";
        scopes.push(global);
    }

    void PushScope(ConfigScopePtr scope)
    {
        // Ensure the provided scope is not null
        FE_ASSERT(scope != nullptr) << "Cannot push a null scope.";
        scopes.push(scope);
    }

    inline bool IntervalJudge(const int64_t& stand, const int64_t& lf, const int64_t& rf) const
    {
        return stand >= lf && stand <= rf;
    }

    bool IsWithinRange(const std::string& properties, const int64_t& value) const
    {
        return IntervalJudge(value, typeInfo.rangeInfos.at(properties).first,
                             typeInfo.rangeInfos.at(properties).second);
    }

    bool IsWithinRange(const std::string& properties, const std::map<int64_t, int64_t>& value) const
    {
        const auto& ins = typeInfo.rangeInfos;
        for (auto& [lf, rf] : value) {
            if (!IntervalJudge(lf, ins.at(properties + "_key").first, ins.at(properties + "_key").second) ||
                !IntervalJudge(rf, ins.at(properties + "_val").first, ins.at(properties + "_val").second)) {
                return false;
            }
        }
        return true;
    }

    void BeginScope(const std::string& name, std::map<std::string, std::any>&& values, const char* file, int lino)
    {
        auto scope = std::make_shared<ConfigScope>(scopes.top());
        scope->values_ = std::move(values);
        scope->begin_file_ = file;
        scope->begin_lino_ = lino;
        scope->name_ = name;
        scopes.push(scope);
    }

    void EndScope(const char* file, int lino)
    {
        /* at least default and global two levels */
        FE_ASSERT(scopes.size() >= 0x2) << "No scope to pop.";
        auto& scope = scopes.top();
        scope->end_file_ = file;
        scope->end_lino_ = lino;
        scopes.pop();
    }

    void SetScope(std::map<std::string, std::any>&& values, const char* file, int lino)
    {
        if (scopes.top().use_count() > 1) { // clone if shared
            auto oldvalues = scopes.top()->values_;
            auto name = scopes.top()->name_;
            EndScope(file, lino);
            BeginScope(name, std::move(oldvalues), file, lino);
        }
        auto& scope = scopes.top();
        for (auto& it : values) {
            CHECK(ExternalError::INVALID_VAL, scope->HasConfig(it.first))
                << "key: " << it.first.c_str() << " does not exist.";
            scope->UpdateValueWithAny(it.first, it.second);
        }
    }

    void SetGlobalConfig(std::map<std::string, std::any>&& values, const char* file, int lino)
    {
        if (values.empty()) {
            FE_LOGW("No values provided to set in global config. Locations: %s:%d", file, lino);
            return;
        }
        for (auto& it : values) {
            try {
                root->AddValue(it.first, it.second);
                FE_LOGD("Set option successfully, Key: %s", it.first.c_str());
            } catch (const std::exception& e) {
                FE_LOGE(FeError::INVALID_VAL, "Failed to set option. Key: %s, Error: %s", it.first.c_str(), e.what());
            }
        }
    }

    void Dump(std::stringstream& os, ConfigScope* node, const std::string& prefix)
    {
        if (!node->begin_file_.empty()) {
            os << prefix << "scope_start: " << node->begin_file_ << ":" << node->begin_lino_ << "\n";
        }
        if (!node->end_file_.empty()) {
            os << prefix << "scope_end: " << node->end_file_ << ":" << node->end_lino_ << "\n";
        }
        if (!node->name_.empty()) {
            os << prefix << "scope: " << node->name_ << "\n";
        }
        DumpValues(os, node->values_, prefix);
        os << "\n";
        for (auto child : node->children_) {
            os << prefix << "--------\n";
            Dump(os, child, prefix + ' ');
        }
    }

    std::string GetOptionsTree()
    {
        std::stringstream os;
        Dump(os, root.get(), "");
        return os.str();
    }

private:
    std::string GetConfDir() { return GetPyptoLibPath() + "/configs/"; }

    void LoadConf(const nlohmann::json& jData, const std::string& prefix)
    {
        if (jData.is_string()) {
            root->AddValue(prefix, jData.get<std::string>());
        } else if (jData.is_number()) {
            root->AddValue(prefix, jData.get<int64_t>());
        } else if (jData.is_boolean()) {
            root->AddValue(prefix, jData.get<bool>());
        } else if (typeInfo.Type(prefix) == typeid(std::map<int64_t, int64_t>)) {
            std::map<int64_t, int64_t> mapJson;
            for (const auto& pair : jData) {
                auto arr = pair.get<std::vector<int64_t>>();
                if (arr.size() >= 0x2) {
                    mapJson[arr[0]] = arr[1];
                }
            }
            root->AddValue(prefix, mapJson);
        } else if (typeInfo.Type(prefix) == typeid(std::map<std::string, int64_t>)) {
            std::map<std::string, int64_t> mapJson;
            if (jData.is_object()) {
                for (auto& it : jData.items()) {
                    mapJson[it.key()] = it.value().get<int64_t>();
                }
            }
            root->AddValue(prefix, mapJson);
        } else if (jData.is_array()) {
            if (typeInfo.Type(prefix) == typeid(std::vector<int64_t>)) {
                root->AddValue(prefix, jData.get<std::vector<int64_t>>());
            } else if (typeInfo.Type(prefix) == typeid(std::vector<double>)) {
                root->AddValue(prefix, jData.get<std::vector<double>>());
            } else {
                root->AddValue(prefix, jData.get<std::vector<std::string>>());
            }
        } else if (jData.is_object()) {
            for (auto& it : jData.items()) {
                const std::string& key = it.key();
                if (prefix.empty()) {
                    LoadConf(it.value(), key);
                } else {
                    LoadConf(it.value(), prefix + "." + key);
                }
            }
        }
    }

    void LoadConf()
    {
        std::string confPath = GetEnvVar("TILEFWK_CONFIG_PATH");
        if (confPath.empty()) {
            confPath = GetConfDir() + "tile_fwk_config.json";
        }
        std::ifstream ifs(confPath);
        CHECK(ExternalError::BAD_FD, ifs.is_open()) << "Open file: " << confPath << " failed";
        nlohmann::json jData;
        ifs >> jData;
        LoadConf(jData, "");
    }

    void InitTileShape()
    {
        TileShape tileShape;
        tileShape.Reset();
        root->AddValue("cube_tile_shapes", tileShape.GetCubeTile());
        root->AddValue("vec_tile_shapes", tileShape.GetVecTile().tile);
        root->AddValue("conv_tile_shapes", tileShape.GetConvTile());
        root->AddValue("matrix_size", tileShape.GetMatrixSize());
        root->AddValue("dist_tile_shapes", tileShape.GetDistTile());
    }
};

void ConfigManagerNg::BeginScope(const std::string& name, std::map<std::string, std::any>&& values, const char* file,
                                 int lino)
{
    impl_->BeginScope(name, std::move(values), file, lino);
}

void ConfigManagerNg::EndScope(const char* file, int lino) { impl_->EndScope(file, lino); }

ConfigManagerNg::ScopedRestore::ScopedRestore(std::shared_ptr<ConfigScope> scope)
{
    ConfigManagerNg::GetInstance().PushScope(std::move(scope));
}

ConfigManagerNg::ScopedRestore::~ScopedRestore() { ConfigManagerNg::GetInstance().EndScope(); }

ConfigManagerNg::JitScopeGuard::JitScopeGuard(const std::string& name, std::map<std::string, std::any>&& values,
                                              const char* file, int lino)
{
    ConfigManagerNg::GetInstance().BeginScope(name, std::move(values), file, lino);
}

ConfigManagerNg::JitScopeGuard::~JitScopeGuard() { ConfigManagerNg::GetInstance().EndScope(); }

void ConfigManagerNg::SetScope(std::map<std::string, std::any>&& values, const char* file, int lino)
{
    return impl_->SetScope(std::move(values), file, lino);
}

void ConfigManagerNg::SetGlobalConfig(std::map<std::string, std::any>&& values, const char* file, int lino)
{
    return impl_->SetGlobalConfig(std::move(values), file, lino);
}

void ConfigManagerNg::PushScope(ConfigScopePtr scope) { impl_->PushScope(scope); }

std::shared_ptr<ConfigScope> ConfigManagerNg::CurrentScope() { return GetInstance().impl_->scopes.top(); }

std::shared_ptr<ConfigScope> ConfigManagerNg::GlobalScope() { return GetInstance().impl_->root; }

bool ConfigManagerNg::IsWithinRange(const std::string& properties, std::any& value) const
{
    try {
        if (value.type() == typeid(std::map<int64_t, int64_t>)) {
            return impl_->IsWithinRange(properties, AnyCast<std::map<int64_t, int64_t>>(value));
        } else if (value.type() == typeid(int64_t)) {
            return impl_->IsWithinRange(properties, AnyCast<int64_t>(value));
        }
    } catch (const std::out_of_range& e) {
        FE_LOGE(FeError::INVALID_VAL, "key[%s] has been not loaded form tile_fwk_config_schema.json.",
                properties.c_str());
        return false;
    }
    return true;
}

const std::type_info& ConfigManagerNg::Type(const std::string& key) const { return impl_->typeInfo.Type(key); }

const std::map<std::string, std::pair<int64_t, int64_t>>& ConfigManagerNg::Range() const
{
    return impl_->typeInfo.rangeInfos;
}

std::string ConfigManagerNg::GetOptionsTree() { return impl_->GetOptionsTree(); }

ConfigManagerNg::ConfigManagerNg() : impl_(std::make_unique<ConfigManagerImpl>()) { globalScope = impl_->root; }

ConfigManagerNg& ConfigManagerNg::GetInstance()
{
    static ConfigManagerNg instance;
    return instance;
}

ConfigManagerNg::~ConfigManagerNg() = default;

namespace config {

template <typename T>
void SetOptionsNg(const std::string& key, const T& value)
{
    ConfigManagerNg::CurrentScope()->UpdateValue(key, value);
}

template void SetOptionsNg<bool>(const std::string& key, const bool& value);
template void SetOptionsNg<int>(const std::string& key, const int& value);
template void SetOptionsNg<double>(const std::string& key, const double& value);
template void SetOptionsNg<std::string>(const std::string& key, const std::string& value);
template void SetOptionsNg<long>(const std::string& key, const long& value);
template void SetOptionsNg<uint8_t>(const std::string& key, const uint8_t& value);
template void SetOptionsNg<std::map<int, int>>(const std::string& key, const std::map<int, int>& value);
template void SetOptionsNg<std::map<long, long>>(const std::string& key, const std::map<long, long>& value);
template void SetOptionsNg<std::vector<int>>(const std::string& key, const std::vector<int>& value);
template void SetOptionsNg<std::vector<std::string>>(const std::string& key, const std::vector<std::string>& value);
template void SetOptionsNg<std::vector<double>>(const std::string& key, const std::vector<double>& value);
template void SetOptionsNg<std::vector<int64_t>>(const std::string& key, const std::vector<int64_t>& value);

std::shared_ptr<ConfigScope> Duplicate() { return ConfigManagerNg::CurrentScope(); }

} // namespace config
} // namespace npu::tile_fwk
