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

project(pypto_ctrl_server LANGUAGES C CXX)

########################################################################################################################
# 基础配置
########################################################################################################################

message(STATUS "CMAKE_GENERATOR=${CMAKE_GENERATOR}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

########################################################################################################################
# 编译流程
########################################################################################################################

file(GLOB CTRL_FILES RELATIVE ${CMAKE_CURRENT_LIST_DIR}
    device_ctrl.cpp
    context/*.cpp
)

add_library(pypto_ctrl_server STATIC)
target_sources(pypto_ctrl_server PRIVATE ${CTRL_FILES} ${_Device_Utils_Srcs})
target_compile_options(pypto_ctrl_server
        PRIVATE
            -Wall
            -Wextra
            -std=gnu++17
            -rdynamic
            -O3
            -shared
            -fPIC
            -D__DEVICE__
            -Werror
            -D__USE_CUSTOM_CTRLFLOW__
            -fstack-protector-strong
)
target_include_directories(pypto_ctrl_server
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/framework/include
            ${PTO_FWK_SRC_ROOT}/framework/src
            ${PTO_FWK_SRC_ROOT}/framework/src/include
            ${PTO_FWK_SRC_ROOT}/framework/src/interface
            ${PTO_FWK_SRC_ROOT}/framework/src/interface/machine/device
)
target_link_libraries(pypto_ctrl_server
        PRIVATE
            $<BUILD_INTERFACE:intf_pub>
            $<BUILD_INTERFACE:atrace_headers>
            $<BUILD_INTERFACE:slog_headers>
            -Wl,--no-as-needed
            c_sec
            slog_stub
            -ldl
)
set_target_properties(pypto_ctrl_server PROPERTIES OUTPUT_NAME pypto_ctrl_server)

install(TARGETS pypto_ctrl_server ARCHIVE DESTINATION lib COMPONENT pypto)
