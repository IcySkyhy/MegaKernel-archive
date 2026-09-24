# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
include(../fetch_cann_cmake.cmake)
project (device_pypto)

init_cann_project(PREPEND_MODULE_PATH PRODUCT_SIDE "device")
add_cann_target_options()
include(../intf.cmake)

get_filename_component(PYPTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(PTO_FWK_SRC_ROOT ${PYPTO_DIR})
set(BUILD_WITH_CANN ON)

find_cann_package(ascend_hal REQUIRED)
find_cann_package(atrace REQUIRED)
find_cann_package(securec REQUIRED)
find_cann_package(slog REQUIRED)
add_subdirectory(${PYPTO_DIR}/framework/src/machine/device ${CMAKE_BINARY_DIR}/framework/src/machine/device)

set_cann_subprj_package()
