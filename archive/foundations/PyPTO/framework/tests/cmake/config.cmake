# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


########################################################################################################################
# 预处理
########################################################################################################################

# 预定义变量
set(PTO_Fwk_UTestNamePrefix tile_fwk_utest)
set(PTO_Fwk_STestNamePrefix tile_fwk_stest)

set(PTO_Fwk_StestExecuteDeviceIdList)
if (NOT ENABLE_STEST_EXECUTE_DEVICE_ID)
    set(PTO_Fwk_StestExecuteDeviceIdList 0)
else ()
    string(REPLACE ":" ";" PTO_Fwk_StestExecuteDeviceIdList "${ENABLE_STEST_EXECUTE_DEVICE_ID}")
endif ()
list(GET PTO_Fwk_StestExecuteDeviceIdList 0 TileFwkStestExecuteDeviceIdPref)

if (ENABLE_STEST_DISTRIBUTED)
    find_program(MPIRUN_EXECUTABLE NAMES mpirun mpiexec)
    if (NOT MPIRUN_EXECUTABLE)
        message(FATAL_ERROR "mpirun not found! Distributed tests require MPI runtime")
    else ()
        message(STATUS "Found mpirun: ${MPIRUN_EXECUTABLE}")
    endif ()
endif ()

# 预定义路径
if (ENABLE_STEST_GOLDEN_PATH)
    get_filename_component(ENABLE_STEST_GOLDEN_PATH "${ENABLE_STEST_GOLDEN_PATH}" REALPATH)
else ()
    get_filename_component(ENABLE_STEST_GOLDEN_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/golden" REALPATH)
endif ()
