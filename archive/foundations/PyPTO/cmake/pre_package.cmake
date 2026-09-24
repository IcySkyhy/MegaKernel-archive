# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

get_filename_component(_PYPTO_DIR "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)

if(CPACK_GENERATOR STREQUAL "External")
    set(PYPTO_STAGING_DIR "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging")
else()
    set(PYPTO_STAGING_DIR "${DEB_DELIVERY}")
endif()

execute_process(
    COMMAND env PYPTO_UNIFIED_BUILD=1 "PYPTO_STAGING_DIR=${PYPTO_STAGING_DIR}" python3 -m build --no-isolation --wheel --config-setting=--build-option=--plat-name=manylinux2014_${CMAKE_SYSTEM_PROCESSOR} -C=--build-option=--py-limited-api=cp37 --outdir ${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    WORKING_DIRECTORY ${_PYPTO_DIR}
    ERROR_VARIABLE error
    RESULT_VARIABLE ret
)
if(NOT ret EQUAL 0)
    message(FATAL_ERROR "build pypto wheel failed! error messages is ${error}")
endif()

# 文件删除放在这里，支持重复调用whl打包
file(REMOVE_RECURSE "${PYPTO_STAGING_DIR}/lib")
file(REMOVE_RECURSE "${PYPTO_STAGING_DIR}/pypto")
