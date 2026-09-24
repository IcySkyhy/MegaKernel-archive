# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE _Simulation_Srcs *.cpp)
#PTO_Fwk_Debug_Source_List(NAME "_Simulation_Srcs" LIST ${_Simulation_Srcs} DETAIL OFF)

add_library(tile_fwk_simulation SHARED)
target_sources(tile_fwk_simulation PRIVATE ${_Simulation_Srcs})
target_include_directories(tile_fwk_simulation
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
)
target_link_libraries(tile_fwk_simulation
        PUBLIC
            dl
            $<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_utils>
            $<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_interface>
        PRIVATE
            tile_fwk_intf_pub
            c_sec
            json
)
PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_simulation)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_simulation)
