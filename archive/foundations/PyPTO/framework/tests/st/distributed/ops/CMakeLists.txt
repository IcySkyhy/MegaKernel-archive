# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_STestNamePrefix}_distributed_ops)

file(GLOB_RECURSE _Sources src/*.cpp)
file(GLOB_RECURSE _Device_Srcs ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/*.cpp)
list(APPEND _Sources ${_Device_Srcs})

get_target_property(GTest_Gtest_Inc GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
add_library(Dist_GTest_Inc INTERFACE)
target_compile_options(Dist_GTest_Inc
        INTERFACE
            -Wno-float-equal
)
target_include_directories(Dist_GTest_Inc
        INTERFACE
            ${GTest_Gtest_Inc}
)

add_library(${_Target} SHARED)
target_sources(${_Target}
        PRIVATE
            ${_Sources}
            ${_Device_Srcs}
)
target_include_directories(${_Target}
        PUBLIC
            include
)
target_link_libraries(${_Target}
        PRIVATE
            ${PTO_Fwk_STestNamePrefix}_utils
            Dist_GTest_Inc
            tile_fwk_utils
            tile_fwk_interface
            tile_fwk_compiler
            tile_fwk_runtime
            tile_fwk_simulation
)
