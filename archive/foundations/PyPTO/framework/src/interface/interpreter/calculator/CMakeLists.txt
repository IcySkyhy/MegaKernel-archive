# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.3)

project(PTO_Framework_Calculator LANGUAGES C CXX)

# 预定义变量检查
if (NOT DEFINED PY3_MOD_TORCH_VERSION)
    message(FATAL_ERROR "PY3_MOD_TORCH_VERSION is not defined")
endif()
if (NOT DEFINED PY3_MOD_TORCH_ROOT_PATH)
    message(FATAL_ERROR "PY3_MOD_TORCH_ROOT_PATH is not defined")
endif()
if (NOT DEFINED PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI)
    message(FATAL_ERROR "PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI is not defined")
endif()

if (NOT DEFINED PTO_FWK_SRC_ROOT)  # 在线编译
    # 路径配置
    get_filename_component(_WhlLibDir    "${CMAKE_CURRENT_SOURCE_DIR}/../../../"    REALPATH)
    get_filename_component(_WhlLibFwkDir "${_WhlLibDir}/framework"                  REALPATH)

    # 参数检查
    if ("${PY3_MOD_TORCH_VERSION}" STRLESS "2.1.0")
        message(FATAL_ERROR "Torch(${PY3_MOD_TORCH_VERSION}) Version less 2.1.0, interpreter not supported.")
    endif ()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.4.0")
        message(FATAL_ERROR "Cxx Compiler(${CMAKE_CXX_COMPILER_ID}) Version(${CMAKE_CXX_COMPILER_VERSION}) less than 9.4.0, interpreter not supported.")
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "15.0.0")
        message(FATAL_ERROR "Cxx Compiler(${CMAKE_CXX_COMPILER_ID}) Version(${CMAKE_CXX_COMPILER_VERSION}) less than 15.0.0, interpreter not supported.")
    endif()

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
endif ()
file(GLOB_RECURSE _Calculator_Srcs *.cpp)
message(STATUS "_Calculator_Srcs=${_Calculator_Srcs}")

# 编译流程
add_library(tile_fwk_calculator SHARED)
target_sources(tile_fwk_calculator PRIVATE ${_Calculator_Srcs})
target_compile_options(tile_fwk_calculator
        PRIVATE
            -D__DEVICE__
)
target_include_directories(tile_fwk_calculator
        SYSTEM PRIVATE
            # torch
            ${PY3_MOD_TORCH_ROOT_PATH}/include
            ${PY3_MOD_TORCH_ROOT_PATH}/include/torch/csrc/api/include
            # 3rd
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:${_WhlLibFwkDir}/3rd/include>
            # src
            $<$<BOOL:${PTO_FWK_SRC_ROOT}>:${PTO_FWK_SRC_ROOT}/framework/include>
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:${_WhlLibDir}/framework/include>
)
target_link_directories(tile_fwk_calculator
        PUBLIC
            ${PY3_MOD_TORCH_ROOT_PATH}/lib
            ${PY3_MOD_TORCH_ROOT_PATH}/../torch.libs
            $<$<NOT:$<BOOL:${PTO_FWK_SRC_ROOT}>>:${_WhlLibDir}/>
)
target_link_libraries(tile_fwk_calculator
        PRIVATE
            c_sec
            torch
            tile_fwk_intf_pub
)
target_link_options(tile_fwk_calculator
        PRIVATE
            -Wl,-no-as-needed,${PY3_MOD_TORCH_ROOT_PATH}/lib/libtorch_cpu.so -Wl,-as-needed
            -Wl,-no-as-needed,${PY3_MOD_TORCH_ROOT_PATH}/lib/libc10.so -Wl,-as-needed
)

set_target_properties(tile_fwk_calculator PROPERTIES CANN_GLIBCXX_USE_CXX11_ABI ${PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI})

if (NOT DEFINED PTO_FWK_SRC_ROOT)
    install(TARGETS tile_fwk_calculator LIBRARY DESTINATION "lib")
endif ()
