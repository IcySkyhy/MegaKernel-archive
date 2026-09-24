# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_STestNamePrefix}_utils)
add_library(${_Target} INTERFACE)
target_include_directories(${_Target}
        INTERFACE
            include
            ${PTO_FWK_SRC_ROOT}/framework/src/machine/runtime
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../utils>
)
target_compile_definitions(${_Target}
        INTERFACE
            $<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>
            $<$<BOOL:${ENABLE_STEST_INTERPRETER_CONFIG}>:ENABLE_STEST_INTERPRETER_CONFIG>
            $<$<BOOL:${ENABLE_STEST_BINARY_CACHE}>:ENABLE_STEST_BINARY_CACHE>
            $<$<BOOL:${ENABLE_STEST_DUMP_JSON}>:ENABLE_STEST_DUMP_JSON>
)
target_compile_options(${_Target}
        INTERFACE
            -fno-access-control
)
target_link_libraries(${_Target}
        INTERFACE
            $<BUILD_INTERFACE:c_sec_headers>
            tile_fwk_intf_pub
            json
)
