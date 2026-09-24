# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.3)

project(PTO_Framework_PyPTO_Impl LANGUAGES C CXX)

if (NOT DEFINED PTO_FWK_SRC_ROOT)  # 在线编译
    # 路径配置
    get_filename_component(_WhlLibDir    "${CMAKE_CURRENT_SOURCE_DIR}/../../"    REALPATH)
    get_filename_component(_WhlLibFwkDir "${_WhlLibDir}/framework"               REALPATH)

    # 参数检查
    if ("${Python3_EXECUTABLE}x" STREQUAL "x")
        message(FATAL_ERROR "Python3 interpreter path not defined, can't build pypto_impl online")
    endif ()
    get_filename_component(_Python3__EXECUTABLE "${Python3_EXECUTABLE}"     REALPATH)
    get_filename_component(_Python3_ROOT_DIR    "${_Python3__EXECUTABLE}"   DIRECTORY)
    get_filename_component(_Python3_ROOT_DIR    "${_Python3_ROOT_DIR}/../"  REALPATH)
    set(Python3_ROOT_DIR "${_Python3_ROOT_DIR}" CACHE INTERNAL "Python3 root path" FORCE)
    message(STATUS "Python3_ROOT_DIR=${Python3_ROOT_DIR}")

    if ("${Python3_EXECUTABLE_VERSION}x" STREQUAL "x")
        message(FATAL_ERROR "Python3 interpreter version not defined, can't build pypto_impl online")
    endif ()
    find_package(Python3 ${Python3_EXECUTABLE_VERSION} EXACT COMPONENTS Development)
    if (NOT Python3_Development_FOUND)
        message(FATAL_ERROR "Can't get python3-dev, can't build pypto_impl online.")
    endif ()

    if ("${Python3_MOD_PYBIND11_CMAKE_DIR}x" STREQUAL "x")
        message(FATAL_ERROR "Can't get pybind11 cmake dir, can't build pypto_impl online")
    endif ()
    find_package(pybind11 CONFIG REQUIRED PATHS ${Python3_MOD_PYBIND11_CMAKE_DIR} NO_DEFAULT_PATH)

    # 构建阶段(Build)
    #   语言标准
    set(CMAKE_C_STANDARD 11)  # 指定 C 语言使用 ISO C11 标准
    set(CMAKE_C_STANDARD_REQUIRED ON)  # 要求严格支持 C11 标准
    set(CMAKE_C_EXTENSIONS OFF)  # 禁用 C 编译器扩展, 使用纯 ISO C 标准
    set(CMAKE_CXX_STANDARD 17)  # 指定 C++ 语言使用 ISO C++17 标准
    set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 要求严格支持 C++17 标准
    set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用 C 编译器扩展, 使用纯 ISO C++17 标准
    include("${_WhlLibFwkDir}/cmake/intf_pub.cmake")
    include("${_WhlLibFwkDir}/cmake/intf.cmake")

    option(BUILD_WITH_CANN "Build with CANN." ON)  # 在线编译场景仅会出现在内置 CANN 包场景
endif ()

pybind11_add_module(pypto_impl
        MODULE
            pybind11.cpp
            bindings/torch_tensor_converter.cpp
            bindings/enum.cpp
            bindings/element.cpp
            bindings/tensor.cpp
            bindings/symbolic_scalar.cpp
            bindings/controller.cpp
            bindings/operation.cpp
            bindings/runtime.cpp
            bindings/cost_model.cpp
            bindings/pass_config.cpp
            bindings/function.cpp
            bindings/distributed.cpp
            bindings/platform.cpp
            bindings/utils.cpp
            bindings/ir/ir.cpp
            bindings/ir/builder.cpp
            bindings/ir/logging.cpp
            bindings/ir/pass.cpp
            bindings/pypto_pro/codegen.cpp
            bindings/pypto_pro/backend.cpp
)
target_include_directories(pypto_impl
        PRIVATE
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:${_WhlLibFwkDir}/3rd/include>
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:${_WhlLibFwkDir}/include>
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:${_WhlLibFwkDir}/include/pypto_pro>
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:${_WhlLibFwkDir}/src>
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:${_WhlLibFwkDir}/src/interface>
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:${_WhlLibFwkDir}/src/interface/machine/device>
            ${PTO_FWK_SRC_ROOT}/framework/include/pypto_pro
            ${CMAKE_CURRENT_LIST_DIR}
)
target_compile_definitions(pypto_impl
        PRIVATE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>>
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:nlohmann=ascend_nlohmann>
)

# 主构建: 所依赖的本项目 so 均在 pypto 包 __init__.py 中加载, 此处不再 link
# 在线编译: 必须显式链接 tile_fwk_interface, 否则 pypto_impl.so 缺少 DT_NEEDED 条目,
#           运行时无法解析 ConfigManager::GetJsonNode 等符号
target_link_libraries(pypto_impl
        PRIVATE
            $<$<BOOL:${PTO_FWK_SRC_ROOT}>:json>
            c_sec
            tile_fwk_intf_pub
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:tile_fwk_interface>
)
target_link_directories(pypto_impl
        PRIVATE
           $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:${_WhlLibDir}>
)
if (DEFINED PTO_FWK_SRC_ROOT)
    if (TARGET c_sec)
        add_dependencies(pypto_impl c_sec)
    endif ()

    PTO_Fwk_GenerateGCovConfigJson(TARGET pypto_impl)
else ()
    install(TARGETS pypto_impl LIBRARY DESTINATION ".")
endif ()
