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

project(tile_fwk_server LANGUAGES C CXX)

########################################################################################################################
# 基础配置
########################################################################################################################
add_subdirectory(stub)

file(GLOB _Device_Utils_Srcs
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_function.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_function_dupped_data.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_program.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_cell_match_dump.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_program_ctrlflow_cache.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_workspace.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/simulation/aicore_hardware.cpp
)

add_subdirectory(dynamic)

########################################################################################################################
# 编译流程
########################################################################################################################

file(GLOB_RECURSE _Device_Srcs *.cpp)
file(GLOB_RECURSE _Stub_Srcs stub/*.cpp)
list(REMOVE_ITEM _Device_Srcs ${_Stub_Srcs})

add_library(tile_fwk_server SHARED)
target_sources(tile_fwk_server PRIVATE ${_Device_Srcs} ${_Device_Utils_Srcs})
target_compile_options(tile_fwk_server
        PRIVATE
            -Wall
            -Wextra
            -std=gnu++17
            -rdynamic
            -O3
            -march=armv8-a
            -shared
            -fPIC
            -D__DEVICE__
            -Werror
            -fno-common
            -fstack-protector-strong
)
target_include_directories(tile_fwk_server
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/framework/include
            ${PTO_FWK_SRC_ROOT}/framework/src
            ${PTO_FWK_SRC_ROOT}/framework/src/include
            ${PTO_FWK_SRC_ROOT}/framework/src/interface
            ${PTO_FWK_SRC_ROOT}/framework/src/interface/machine/device
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
)
target_link_libraries(tile_fwk_server
        PRIVATE
            $<BUILD_INTERFACE:intf_pub>
            $<BUILD_INTERFACE:ascend_hal_headers>
            $<BUILD_INTERFACE:atrace_headers>
            $<BUILD_INTERFACE:slog_headers>
            -Wl,--no-as-needed
            c_sec
            slog_stub
            -ldl
)
target_link_options(tile_fwk_server
        PRIVATE
            -Wl,-z,relro
            -Wl,-z,now
            -s
)
set_target_properties(tile_fwk_server PROPERTIES OUTPUT_NAME tilefwk_backend_server)

install(TARGETS tile_fwk_server LIBRARY DESTINATION lib COMPONENT pypto)
install(FILES ${PTO_FWK_SRC_ROOT}/framework/src/machine/runtime/runner/pypto_op_info.json DESTINATION lib COMPONENT pypto)
