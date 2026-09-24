# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB _Cann_Host_Runtime_Srcs *.cpp)
add_library(tile_fwk_cann_host_runtime SHARED)
target_sources(tile_fwk_cann_host_runtime PRIVATE ${_Cann_Host_Runtime_Srcs})
target_compile_definitions(tile_fwk_cann_host_runtime
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_STEST_INTERPRETER_CONFIG}>:ENABLE_STEST_INTERPRETER_CONFIG>>
)
target_link_libraries(tile_fwk_cann_host_runtime
        PRIVATE
            tile_fwk_intf_pub
            dl
            c_sec
            tile_fwk_utils
            json
)
target_compile_definitions(tile_fwk_cann_host_runtime
        PRIVATE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:PROCESSOR_SUBPATH="${CMAKE_SYSTEM_PROCESSOR}-linux">>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>>
)

PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_cann_host_runtime)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_cann_host_runtime)
