/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "backend/common/backend.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "bindings/pypto_pro/bindings.h"
#include "backend/backend_cce.h"
#include "backend/common/soc.h"
#include "ir/memref.h"
#include "ir/pipe.h"

namespace py = pybind11;

namespace pypto {
namespace ir {

using pypto::backend::Backend;
using pypto::backend::BackendCCE;
using pypto::backend::Cluster;
using pypto::backend::Core;
using pypto::backend::Die;
using pypto::backend::Mem;
using pypto::backend::SoC;
using pypto::ir::CoreType;
using pypto::ir::MemorySpace;

void BindBackend(py::module_& m)
{
    py::module_ backend_mod = m.def_submodule("backend", "PyPTO Backend module");

    // ========== Mem class ==========
    py::class_<Mem>(backend_mod, "Mem", "Memory component")
        .def(py::init<MemorySpace, uint64_t, uint64_t>(), py::arg("mem_type"), py::arg("mem_size"),
             py::arg("alignment"), "Create a memory component")
        .def_property_readonly("mem_type", &Mem::GetMemType, "Memory space type")
        .def_property_readonly("mem_size", &Mem::GetMemSize, "Memory size in bytes")
        .def_property_readonly("alignment", &Mem::GetAlignment, "Memory alignment in bytes")
        .def("__repr__", [](const Mem& mem) {
            return "Mem(type=" + ir::MemorySpaceToString(mem.GetMemType()) +
                   ", size=" + std::to_string(mem.GetMemSize()) + ", alignment=" + std::to_string(mem.GetAlignment()) +
                   ")";
        });

    // ========== Core class ==========
    py::class_<Core>(backend_mod, "Core", "Processing core")
        .def(py::init<CoreType, std::vector<Mem>>(), py::arg("core_type"), py::arg("mems"), "Create a processing core")
        .def_property_readonly("core_type", &Core::GetCoreType, "Core type (CUBE or VECTOR)")
        .def_property_readonly("mems", &Core::GetMems, "List of memory components")
        .def("__repr__", [](const Core& core) {
            return "Core(type=" + std::to_string(static_cast<int>(core.GetCoreType())) +
                   ", mems=" + std::to_string(core.GetMems().size()) + ")";
        });

    // ========== Cluster class ==========
    py::class_<Cluster>(backend_mod, "Cluster", "Cluster of processing cores")
        .def(py::init<std::map<Core, int>>(), py::arg("core_counts"), "Create cluster from core counts map")
        .def(py::init<const Core&, int>(), py::arg("core"), py::arg("count"), "Create cluster with single core type")
        .def_property_readonly("core_counts", &Cluster::GetCoreCounts, "Map of core configurations to counts")
        .def("total_core_count", &Cluster::TotalCoreCount, "Get total number of cores in cluster")
        .def("__repr__", [](const Cluster& cluster) {
            return "Cluster(total_cores=" + std::to_string(cluster.TotalCoreCount()) + ")";
        });

    // ========== Die class ==========
    py::class_<Die>(backend_mod, "Die", "Die containing clusters")
        .def(py::init<std::map<Cluster, int>>(), py::arg("cluster_counts"), "Create die from cluster counts map")
        .def(py::init<const Cluster&, int>(), py::arg("cluster"), py::arg("count"),
             "Create die with single cluster type")
        .def_property_readonly("cluster_counts", &Die::GetClusterCounts, "Map of cluster configurations to counts")
        .def("total_cluster_count", &Die::TotalClusterCount, "Get total number of clusters in die")
        .def("total_core_count", &Die::TotalCoreCount, "Get total number of cores in die")
        .def("__repr__", [](const Die& die) {
            return "Die(clusters=" + std::to_string(die.TotalClusterCount()) +
                   ", cores=" + std::to_string(die.TotalCoreCount()) + ")";
        });

    // ========== SoC class ==========
    py::class_<SoC>(backend_mod, "SoC", "System on Chip")
        .def(py::init<std::map<Die, int>>(), py::arg("die_counts"), "Create SoC from die counts map")
        .def(py::init<const Die&, int>(), py::arg("die"), py::arg("count"), "Create SoC with single die type")
        .def_property_readonly("die_counts", &SoC::GetDieCounts, "Map of die configurations to counts")
        .def("total_die_count", &SoC::TotalDieCount, "Get total number of dies in SoC")
        .def("total_cluster_count", &SoC::TotalClusterCount, "Get total number of clusters in SoC")
        .def("total_core_count", &SoC::TotalCoreCount, "Get total number of cores in SoC")
        .def("__repr__", [](const SoC& soc) {
            return "SoC(dies=" + std::to_string(soc.TotalDieCount()) +
                   ", clusters=" + std::to_string(soc.TotalClusterCount()) +
                   ", cores=" + std::to_string(soc.TotalCoreCount()) + ")";
        });

    // ========== Backend abstract base class ==========
    py::class_<Backend>(backend_mod, "Backend", "Abstract backend base class")
        .def("get_type_name", &Backend::GetTypeName, "Get backend type name")
        .def(
            "get_op_pipe",
            [](const Backend& backend, const std::string& op_name) -> std::optional<ir::PipeType> {
                const auto* op_info = backend.GetOpInfo(op_name);
                if (op_info == nullptr) {
                    return std::nullopt;
                }
                return op_info->pipe;
            },
            py::arg("op_name"), "Get the registered pipeline type for an operator, or None if unregistered")
        .def("find_mem_path", &Backend::FindMemPath, py::arg("from_mem"), py::arg("to_mem"),
             "Find memory path from source to destination")
        .def("get_mem_size", &Backend::GetMemSize, py::arg("mem_type"), "Get total memory size for given memory type")
        .def_property_readonly(
            "soc", [](const Backend& backend) -> const SoC& { return backend.GetSoC(); }, "Get SoC object");

    // ========== BackendCCE concrete implementation ==========
    py::class_<BackendCCE, Backend>(backend_mod, "BackendCCE", "CCE backend implementation")
        .def_static("instance", &BackendCCE::Instance, py::return_value_policy::reference,
                    "Get singleton instance of CCE backend");
}

} // namespace ir
} // namespace pypto
