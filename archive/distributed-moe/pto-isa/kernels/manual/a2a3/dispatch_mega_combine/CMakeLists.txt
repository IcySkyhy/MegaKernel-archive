# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

get_filename_component(ASCEND_BISHENG_COMPILER "${ASCEND_HOME_PATH}/bin/bisheng" REALPATH)
if(NOT EXISTS "${ASCEND_BISHENG_COMPILER}")
    message(FATAL_ERROR "Cannot find ${ASCEND_HOME_PATH}/bin/bisheng, please run the CANN set_env.sh.")
endif()
set(CMAKE_C_COMPILER "${ASCEND_BISHENG_COMPILER}" CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "${ASCEND_BISHENG_COMPILER}" CACHE FILEPATH "CXX compiler" FORCE)

project(pto_example LANGUAGES C CXX)

# bisheng/cce-ld does not support the linker depfile option that CMake 3.27+ may enable.
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

# The CANN environment exposes include/bin links for its active ARM or x86 package.
# Resolve that link instead of duplicating CANN's platform selection logic here.
get_filename_component(ASCEND_KERNEL_PLATFORM_INCLUDE "${ASCEND_HOME_PATH}/include" REALPATH)
get_filename_component(ASCEND_KERNEL_PLATFORM_ROOT "${ASCEND_KERNEL_PLATFORM_INCLUDE}" DIRECTORY)
if(NOT IS_DIRECTORY "${ASCEND_KERNEL_PLATFORM_INCLUDE}")
    message(FATAL_ERROR "Cannot resolve the active CANN platform directory from ${ASCEND_HOME_PATH}/include.")
endif()

# CANN 9.x uses asc/, while older compatibility packages use include/ascendc/.
set(KERNEL_INCLUDE_CANDIDATES
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/interface
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/utils
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/impl/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/basic_api/interface
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/basic_api/impl
)

set(KERNEL_INCLUDE_DIRS)
set(KERNEL_OPERATOR_HEADER_FOUND FALSE)
set(KERNEL_TPIPE_HEADER_FOUND FALSE)
set(KERNEL_TPIPE_IMPL_HEADER_FOUND FALSE)
foreach(KERNEL_INCLUDE_CANDIDATE IN LISTS KERNEL_INCLUDE_CANDIDATES)
    if(IS_DIRECTORY "${KERNEL_INCLUDE_CANDIDATE}")
        list(APPEND KERNEL_INCLUDE_DIRS "${KERNEL_INCLUDE_CANDIDATE}")
        if(EXISTS "${KERNEL_INCLUDE_CANDIDATE}/kernel_operator.h")
            set(KERNEL_OPERATOR_HEADER_FOUND TRUE)
        endif()
        if(EXISTS "${KERNEL_INCLUDE_CANDIDATE}/kernel_tpipe.h")
            set(KERNEL_TPIPE_HEADER_FOUND TRUE)
        endif()
        if(EXISTS "${KERNEL_INCLUDE_CANDIDATE}/kernel_tpipe_impl.h")
            set(KERNEL_TPIPE_IMPL_HEADER_FOUND TRUE)
        endif()
    endif()
endforeach()

if(NOT KERNEL_OPERATOR_HEADER_FOUND
   OR NOT KERNEL_TPIPE_HEADER_FOUND
   OR NOT KERNEL_TPIPE_IMPL_HEADER_FOUND)
    message(FATAL_ERROR
        "Incomplete AscendC headers under ASCEND_HOME_PATH=${ASCEND_HOME_PATH}: expected kernel_operator.h, "
        "kernel_tpipe.h, and kernel_tpipe_impl.h. Check the active CANN environment.")
endif()

if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/highlevel_api/lib/std/tuple.h)
    list(APPEND KERNEL_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/highlevel_api)
endif()
if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/pkg_inc/runtime/runtime)
    list(APPEND KERNEL_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/pkg_inc/runtime/runtime)
endif()
list(REMOVE_DUPLICATES KERNEL_INCLUDE_DIRS)
add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
    )

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CCE_AICORE_ARCH "dav-c220")

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

get_filename_component(PTO_REPO_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../../../../include" ABSOLUTE)

include_directories(
    ${PTO_REPO_INCLUDE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof    
)

function(pto_example_mixed NAME)
    add_library(${NAME}_kernel SHARED kernel_launch.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=${CCE_AICORE_ARCH} -std=c++17)
    target_include_directories(${NAME}_kernel BEFORE PRIVATE
        ${PTO_REPO_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/op_kernel
        ${PROJECT_SOURCE_DIR}/op_kernel/utils
        ${PROJECT_BINARY_DIR}
        ${KERNEL_INCLUDE_DIRS}
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME}
        main.cpp
        runtime_context.cpp
        tiling_builder.cpp
        data_utils.cpp
    )
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} BEFORE PRIVATE
        ${PTO_REPO_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/op_kernel
        ${PROJECT_SOURCE_DIR}/op_kernel/utils
        ${PROJECT_BINARY_DIR}
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        stdc++ ascendcl hcomm runtime m tiling_api platform c_sec dl nnopbase pthread
    )
endfunction()

pto_example_mixed(dispatch_mega_combine)
