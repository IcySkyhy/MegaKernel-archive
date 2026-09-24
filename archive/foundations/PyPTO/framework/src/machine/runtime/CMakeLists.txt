# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE _Runtime_Srcs *.cpp)
# The kernel-bundle load/query/launch half ships as its own ABI library (libtile_fwk_bundle.so, see bundle/
# CMakeLists.txt) and must NOT be linked into tile_fwk_runtime as well. Only the pack half stays here, because
# its hooks are called from emulation_launcher.cpp / kernel_binary.cpp -- that is what would otherwise make the
# two targets circularly dependent.
#
# The split is by DIRECTORY, not by a file whitelist: everything directly under bundle/ belongs to the ABI
# library, the pack half lives one level down in bundle/pack/ and is picked up by the recursive glob above.
# bundle/CMakeLists.txt globs its own sources with the same non-recursive pattern, so new files on either side
# land in the right target automatically with no list to keep in sync.
file(GLOB _Bundle_Load_Srcs ${CMAKE_CURRENT_SOURCE_DIR}/bundle/*.cpp)
list(REMOVE_ITEM _Runtime_Srcs ${_Bundle_Load_Srcs})
file(GLOB_RECURSE _Device_Srcs ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/*.cpp)
file(GLOB _Device_Utils_Srcs
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_function.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_function_dupped_data.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_program.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_cell_match_dump.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_encode_program_ctrlflow_cache.cpp
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/utils/dynamic/dev_workspace.cpp
)
add_library(tile_fwk_runtime SHARED)
file(GLOB_RECURSE _Simulation_Srcs ${PTO_FWK_SRC_ROOT}/framework/src/machine/simulation/*.cpp)
target_sources(tile_fwk_runtime PRIVATE ${_Runtime_Srcs} ${_Device_Srcs} ${_Device_Utils_Srcs} ${_Simulation_Srcs})
target_include_directories(tile_fwk_runtime
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
)
target_compile_definitions(tile_fwk_runtime
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_STEST_INTERPRETER_CONFIG}>:ENABLE_STEST_INTERPRETER_CONFIG>>
)
target_link_libraries(tile_fwk_runtime
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_utils>>
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_adapter>>
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_interface>>
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_compiler>>
            $<BUILD_INTERFACE:json>
        PRIVATE
            tile_fwk_intf_pub
            dl
            c_sec
)

PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_runtime)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_runtime)

# Standalone kernel-bundle ABI library; must come after tile_fwk_runtime, which it links against.
add_subdirectory(bundle)
