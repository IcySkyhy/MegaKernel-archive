# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


file(GLOB_RECURSE _CodeGen_Srcs *.cpp)
#PTO_Fwk_Debug_Source_List(NAME "_CodeGen_Srcs" LIST ${_CodeGen_Srcs} DETAIL OFF)

add_library(tile_fwk_codegen SHARED)
target_sources(tile_fwk_codegen PRIVATE ${_CodeGen_Srcs})
target_link_libraries(tile_fwk_codegen
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_utils>>
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_COMPILE_DEPENDENCY_CHECK}>:tile_fwk_interface>>
            $<BUILD_INTERFACE:json>
        PRIVATE
            tile_fwk_intf_pub
            c_sec
)

target_compile_definitions(tile_fwk_codegen
        PRIVATE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>>
)

PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_codegen)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_codegen)
