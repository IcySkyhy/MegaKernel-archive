# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_UTestNamePrefix}_interface)

file(GLOB_RECURSE _Sources
        src/*.cpp
)
if (NOT ENABLE_TORCH_VERIFIER)
    list(FILTER _Sources EXCLUDE REGEX "interpreter/")
endif ()

set(LinkLibraries)
if (TARGET tile_fwk_calculator)
    list(APPEND LinkLibraries tile_fwk_calculator)
endif ()

PTO_Fwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PRIVATE_INCLUDE_DIRECTORIES ${PTO_FWK_SRC_ROOT}/framework/include/pypto_pro
        LINK_LIBRARIES ${LinkLibraries}
)

target_compile_definitions(${_Target} PRIVATE
        IO_TEXT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/ir/test_io_text_dump_data"
)
