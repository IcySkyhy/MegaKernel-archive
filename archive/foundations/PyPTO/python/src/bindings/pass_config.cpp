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
 * \file pass_config.cpp
 * \brief
 */

#include "pybind_common.h"

using namespace npu::tile_fwk;

namespace pypto {
void BindPassConst(py::module_& m)
{
    m.attr("KEY_DUMP_GRAPH") = KEY_DUMP_GRAPH;
    m.attr("KEY_PRINT_GRAPH") = KEY_PRINT_GRAPH;
    m.attr("KEY_PRE_CHECK") = KEY_PRE_CHECK;
    m.attr("KEY_POST_CHECK") = KEY_POST_CHECK;
    m.attr("KEY_HEALTH_CHECK") = KEY_HEALTH_CHECK;
}

// config pass_global_configs.x parameters
void BindPassGlobalConfig(py::module_& m)
{
    m.def(
        "GetPassGlobalConfig",
        [](const std::string& key, const bool& default_value) -> py::object {
            return py::cast(config::GetPassGlobalConfig<bool>(key, default_value));
        },
        py::arg("key"), py::arg("default_value"));
    m.def(
        "GetPassGlobalConfig",
        [](const std::string& key, const int64_t& default_value) -> py::object {
            return py::cast(config::GetPassGlobalConfig<int64_t>(key, default_value));
        },
        py::arg("key"), py::arg("default_value"));
    m.def(
        "GetPassGlobalConfig",
        [](const std::string& key, const std::string& default_value) -> py::object {
            return py::cast(config::GetPassGlobalConfig<std::string>(key, default_value));
        },
        py::arg("key"), py::arg("default_value"));

    m.def(
        "SetPassGlobalConfig",
        [](const std::string& key, const bool& value) { config::SetPassGlobalConfig<bool>(key, value); },
        py::arg("key"), py::arg("value"));
    m.def(
        "SetPassGlobalConfig",
        [](const std::string& key, const int64_t& value) { config::SetPassGlobalConfig<int64_t>(key, value); },
        py::arg("key"), py::arg("value"));
    m.def(
        "SetPassGlobalConfig",
        [](const std::string& key, const std::string& value) { config::SetPassGlobalConfig<std::string>(key, value); },
        py::arg("key"), py::arg("value"));
}

// config pass_global_configs.default_pass_configs.x parameters
void BindPassDefaultConfig(py::module_& m)
{
    m.def(
        "GetPassDefaultConfig",
        [](const std::string& key, const bool& default_value) -> py::object {
            return py::cast(config::GetPassDefaultConfig<bool>(key, default_value));
        },
        py::arg("key"), py::arg("default_value"));
    m.def(
        "GetPassDefaultConfig",
        [](const std::string& key, const int64_t& default_value) -> py::object {
            return py::cast(config::GetPassDefaultConfig<int64_t>(key, default_value));
        },
        py::arg("key"), py::arg("default_value"));
    m.def(
        "GetPassDefaultConfig",
        [](const std::string& key, const std::string& default_value) -> py::object {
            return py::cast(config::GetPassDefaultConfig<std::string>(key, default_value));
        },
        py::arg("key"), py::arg("default_value"));

    m.def(
        "SetPassDefaultConfig",
        [](const std::string& key, const bool& value) { config::SetPassDefaultConfig<bool>(key, value); },
        py::arg("key"), py::arg("value"));
    m.def(
        "SetPassDefaultConfig",
        [](const std::string& key, const int64_t& value) { config::SetPassDefaultConfig<int64_t>(key, value); },
        py::arg("key"), py::arg("value"));
    m.def(
        "SetPassDefaultConfig",
        [](const std::string& key, const std::string& value) { config::SetPassDefaultConfig<std::string>(key, value); },
        py::arg("key"), py::arg("value"));
}

// config strategies.x parameters
void BindPassConfig(py::module_& m)
{
    m.def(
        "GetPassConfig",
        [](const std::string& strategy, const std::string& identifier, const std::string& key,
           const bool& default_value) -> py::object {
            return py::cast(config::GetPassConfig<bool>(strategy, identifier, key, default_value));
        },
        py::arg("strategy"), py::arg("identifier"), py::arg("key"), py::arg("default_value"));
    m.def(
        "GetPassConfig",
        [](const std::string& strategy, const std::string& identifier, const std::string& key,
           const int64_t& default_value) -> py::object {
            return py::cast(config::GetPassConfig<int64_t>(strategy, identifier, key, default_value));
        },
        py::arg("strategy"), py::arg("identifier"), py::arg("key"), py::arg("default_value"));
    m.def(
        "GetPassConfig",
        [](const std::string& strategy, const std::string& identifier, const std::string& key,
           const std::string& default_value) -> py::object {
            return py::cast(config::GetPassConfig<std::string>(strategy, identifier, key, default_value));
        },
        py::arg("strategy"), py::arg("identifier"), py::arg("key"), py::arg("default_value"));

    m.def(
        "SetPassConfig",
        [](const std::string& strategy, const std::string& identifier, const std::string& key, const bool& value) {
            config::SetPassConfig<bool>(strategy, identifier, key, value);
        },
        py::arg("strategy"), py::arg("identifier"), py::arg("key"), py::arg("value"));
    m.def(
        "SetPassConfig",
        [](const std::string& strategy, const std::string& identifier, const std::string& key, const int64_t& value) {
            config::SetPassConfig<int64_t>(strategy, identifier, key, value);
        },
        py::arg("strategy"), py::arg("identifier"), py::arg("key"), py::arg("value"));
    m.def(
        "SetPassConfig",
        [](const std::string& strategy, const std::string& identifier, const std::string& key,
           const std::string& value) { config::SetPassConfig<std::string>(strategy, identifier, key, value); },
        py::arg("strategy"), py::arg("identifier"), py::arg("key"), py::arg("value"));
}

void BindPassConfigs(py::module_& m)
{
    py::class_<PassConfigs>(m, "PassConfigs")
        .def_readonly("printGraph", &PassConfigs::printGraph)
        .def_readonly("dumpGraph", &PassConfigs::dumpGraph)
        .def_readonly("dumpPassTimeCost", &PassConfigs::dumpPassTimeCost)
        .def_readonly("preCheck", &PassConfigs::preCheck)
        .def_readonly("postCheck", &PassConfigs::postCheck)
        .def_readonly("disablePass", &PassConfigs::disablePass)
        .def_readonly("healthCheck", &PassConfigs::healthCheck);
    m.def(
        "GetPassConfigs",
        [](const std::string& strategy, const std::string& identifier) -> PassConfigs {
            return ConfigManager::Instance().GetPassConfigs(strategy, identifier);
        },
        py::arg("strategy"), py::arg("identifier"));
}

void BindPass(py::module_& m)
{
    BindPassConst(m);
    BindPassGlobalConfig(m);
    BindPassDefaultConfig(m);
    BindPassConfig(m);
    BindPassConfigs(m);
}
} // namespace pypto
