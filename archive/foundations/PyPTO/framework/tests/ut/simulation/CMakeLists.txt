# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_UTestNamePrefix}_simulation)

file(GLOB_RECURSE _Sources src/*.cpp)

PTO_Fwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PRIVATE_INCLUDE_DIRECTORIES ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
        LINK_LIBRARIES tile_fwk_operator tile_fwk_simulation dl
)
add_dependencies(${_Target} tile_fwk_simulation_pv)

# 可执行程序运行依赖的配置, 脚本配置
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config/* ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/
        COMMENT "Copy ${_Target} dependence config file to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/"
)
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/print_swim_lane.py         ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/draw_comm_swim_lane_png.py ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/draw_pipe_swim_lane.py     ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/tools/profiling/draw_swim_lane.py                                      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/tools/profiling/function_json_convert.py                               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/tools/profiling/parse_pipe_time_trace.py                               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/
        COMMENT "Copy ${_Target} dependence scripts file to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scripts/"
)
