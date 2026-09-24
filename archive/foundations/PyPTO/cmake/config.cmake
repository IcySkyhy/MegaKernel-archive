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
# 环境检查
########################################################################################################################

# Python3 分析
# 1. Python3_EXECUTABLE 用以标识 Python3 可执行文件路径, -D 指定的优先级高于环境变量;
# 2. 刷新 CMake 内部变量 Python3_ROOT_DIR 值, 以便后续 find python 相关 package 使用(主要是 Development 组件);
if (DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${Python3_EXECUTABLE}")  # 不可直接获取绝对路径, 避免不同 executable 权限差异导致的 pip 包查找问题
elseif (DEFINED ENV{Python3_EXECUTABLE})
    set(Python3_EXECUTABLE "$$ENV{Python3_EXECUTABLE}")  # 不可直接获取绝对路径, 避免不同 executable 权限差异导致的 pip 包查找问题
else ()
    # 当外部未指定 Python3 时, 一般是从 CMake 为入口触发的编译, 此时直接 find.
    find_package(Python3 COMPONENTS Interpreter Development)
    if ("${Python3_EXECUTABLE}x" STREQUAL "x")
        message(FATAL_ERROR "Can't find python3 Interpreter.")
    endif ()
endif ()
message(STATUS "Python3_EXECUTABLE=${Python3_EXECUTABLE}")
if (NOT (DEFINED Python3_ROOT_DIR OR DEFINED ENV{Python3_ROOT_DIR}))
    get_filename_component(_Python3__EXECUTABLE "${Python3_EXECUTABLE}"     REALPATH)
    get_filename_component(_Python3_ROOT_DIR    "${_Python3__EXECUTABLE}"   DIRECTORY)
    get_filename_component(_Python3_ROOT_DIR    "${_Python3_ROOT_DIR}/../"  REALPATH)
    set(Python3_ROOT_DIR "${_Python3_ROOT_DIR}" CACHE INTERNAL "Python3 root path" FORCE)
    message(STATUS "Python3_ROOT_DIR=${Python3_ROOT_DIR}")
endif ()

get_filename_component(_Py3CMakeFile "${PTO_FWK_BIN_ROOT}/_pypto_py3_env.cmake" REALPATH)
PTO_Fwk_AnalysisPython3Environ(OUTPUT_FILE ${_Py3CMakeFile})
include(${_Py3CMakeFile})

if (ENABLE_FEATURE_PYTHON_FRONT_END)
    find_package(Python3 ${PYTHON3_VERSION_ID} EXACT COMPONENTS Development)
    if (NOT Python3_Development_FOUND)
        message(FATAL_ERROR "Can't get python3-dev, Python Frontend can't build.")
    endif ()
    if ("${PY3_MOD_PYBIND11_CMAKE_DIR}x" STREQUAL "x")
        message(FATAL_ERROR "Can't get pybind11 cmake dir, Python Frontend can't build.")
    endif ()
    find_package(pybind11 CONFIG REQUIRED PATHS ${PY3_MOD_PYBIND11_CMAKE_DIR} NO_DEFAULT_PATH)
endif ()


# 获取 CANN 路径
set(ASCEND_CANN_PACKAGE_PATH)
if (BUILD_WITH_CANN)
    if (CUSTOM_ASCEND_CANN_PACKAGE_PATH)
        get_filename_component(ASCEND_CANN_PACKAGE_PATH "${CUSTOM_ASCEND_CANN_PACKAGE_PATH}" REALPATH)
    elseif (DEFINED ENV{ASCEND_HOME_PATH})
        get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" REALPATH)
    else ()
        set(ASCEND_CANN_PACKAGE_PATH)
    endif ()
    if ("${ASCEND_CANN_PACKAGE_PATH}x" STREQUAL "x" OR NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
        set(BUILD_WITH_CANN OFF)
        message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} is empty or not exist, auto turn off BUILD_WITH_CANN")
    endif ()
endif ()
message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "BUILD_WITH_CANN=${BUILD_WITH_CANN}")


########################################################################################################################
# CMake 选项, 缺省参数设置
#   按 CMake 构建过程对 CMake 选项, CMake 缺省参数进行配置
#   CMake 构建过程: 1) 配置阶段(Configure); 2) 构建阶段(Build); 3) 安装阶段(Install);
########################################################################################################################

# 构建阶段(Build)
#   构建类型
#       CMake中的Generator(生成器)是用于生成本地/本机构建系统的工具. 一般分为两种:
#       1. 单配置生成器(Single-configuration generator):
#          在配置(Configuration)阶段, 仅允许指定一种构建类型, 通过变量 CMAKE_BUILD_TYPE 指定;
#          在构建阶段(Build)无法更改构建类型, 仅允许使用配置(Configuration)阶段通过变量 CMAKE_BUILD_TYPE 指定的构建类型;
#          常见的此类型生成器有: Ninja, Unix Makefiles
#       2. 多配置生成器(Multi-configuration generator) :
#          在配置(Configuration)阶段, 仅指定构建阶段(Build)可用的构建类型列表, 通过变量 CMAKE_CONFIGURATION_TYPES 指定;
#          在构建阶段(Build)通过 "--config" 参数, 指定构建阶段具体的构建类型;
#          常见的此类型生成器有: Xcode, Visual Studio
#       所以:
#           1. 单配置生成器(Single-configuration generator)场景下, 如果构建类型(CMAKE_BUILD_TYPE)未指定, 则默认为 Debug ;
#           2. 多配置生成器(Multi-configuration generator)场景下, 如果构建阶段可选的构建类型(CMAKE_CONFIGURATION_TYPES)未指定,
#              则默认将其指定为CMake允许的构建类型全集 [Debug;Release;MinSizeRel;RelWithDebInfo]
message(STATUS "CMAKE_GENERATOR=${CMAKE_GENERATOR}")
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (GENERATOR_IS_MULTI_CONFIG)
    if (NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;MinSizeRel;RelWithDebInfo" CACHE STRING "Configuration Build type" FORCE)
    endif ()
else ()
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type(default Release)" FORCE)
    endif ()
    message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif ()


# 构建阶段(Build)
#   可执行文件运行时库文件搜索路径 RPATH
#       在 UTest 及 STest 场景不略去 RPATH
string(REPLACE "," ":" ENABLE_UTEST "${ENABLE_UTEST}")
string(REPLACE "," ":" ENABLE_STEST "${ENABLE_STEST}")
string(REPLACE "," ":" ENABLE_STEST_DISTRIBUTED "${ENABLE_STEST_DISTRIBUTED}")
string(REPLACE "," ":" ENABLE_STEST_GROUP "${ENABLE_STEST_GROUP}")
if (NOT BUILD_WITH_CANN)
    if (ENABLE_STEST)
        set(ENABLE_STEST OFF)
        message(STATUS "Build without CANN, auto turn off ENABLE_STEST")
    endif ()
    if (ENABLE_STEST_DISTRIBUTED)
        set(ENABLE_STEST_DISTRIBUTED OFF)
        message(STATUS "Build without CANN, auto turn off ENABLE_STEST_DISTRIBUTED")
    endif ()
endif ()
if (ENABLE_UTEST OR ENABLE_STEST OR ENABLE_STEST_DISTRIBUTED)
    set(ENABLE_TESTS ON)
else ()
    set(ENABLE_TESTS OFF)
endif ()

if (ENABLE_TESTS)
    set(CMAKE_SKIP_RPATH FALSE)
else ()
    set(CMAKE_SKIP_RPATH TRUE)
endif ()

# 构建阶段(Build)
#   语言标准
set(CMAKE_C_STANDARD 11)  # 指定 C 语言使用 ISO C11 标准
set(CMAKE_C_STANDARD_REQUIRED ON)  # 要求严格支持 C11 标准
set(CMAKE_C_EXTENSIONS OFF)  # 禁用 C 编译器扩展, 使用纯 ISO C 标准
set(CMAKE_CXX_STANDARD 17)  # 指定 C++ 语言使用 ISO C++17 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 要求严格支持 C++17 标准
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用 C 编译器扩展, 使用纯 ISO C++17 标准


# 构建阶段(Build)
#   CCACHE 配置
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM} CACHE PATH "C cache Compiler")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM} CACHE PATH "CXX cache Compiler")
    message(STATUS "Use ccache, CCACHE_BASEDIR=$ENV{CCACHE_BASEDIR}")
else ()
    message(STATUS "ccache not found.")
endif ()

# 构建阶段(Build)
#   输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PTO_FWK_BIN_OUTPUT_ROOT}/bin)     # 设置可执行文件输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PTO_FWK_BIN_OUTPUT_ROOT}/lib)     # 设置动态库输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PTO_FWK_BIN_OUTPUT_ROOT}/lib)     # 设置静态库输出目录


# 安装阶段(Install)
#   安装路径
#       未显示设置 CMAKE_INSTALL_PREFIX (即 CMAKE_INSTALL_PREFIX 取缺省值)时, 设置与构建树根目录 CMAKE_CURRENT_BINARY_DIR 平级
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    get_filename_component(_Install_Path_Prefix "${CMAKE_CURRENT_BINARY_DIR}/../output" REALPATH)
    set(CMAKE_INSTALL_PREFIX    "${_Install_Path_Prefix}"  CACHE STRING "Install path" FORCE)
endif ()
message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")


########################################################################################################################
# 预处理
########################################################################################################################
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    if (ENABLE_GCOV)
        set(ENABLE_GCOV OFF)
        message(WARNING "GCov only supported in GNU Compiler, Current Compiler is ${CMAKE_C_COMPILER_ID}, Auto turn off it.")
    endif ()
    if (ENABLE_FEATURE_PYTHON_FRONT_END)
        message(FATAL_ERROR "Python frontend only supported GNU Compiler yet.")
    endif ()
endif ()

# ASAN / UBSAN 场景随编译执行用例场景下, 将相关检查在编译前执行, 避免出现编译完成后又无法执行的情况, 影响使用体验.
if ((ENABLE_ASAN OR ENABLE_UBSAN) AND (ENABLE_TESTS_EXECUTE OR ENABLE_FEATURE_PYTHON_FRONT_END))
    # 用于向外传递 XSAN 相关配置, 以保证 whl 包 ASan 场景的兼容性
    set(XSAN_CONFIG_FILE "${PTO_FWK_BIN_ROOT}/_pypto_xsan_config.txt")
    file(WRITE "${XSAN_CONFIG_FILE}" "")

    # LD_PRELOAD, 仅 GNU 编译器需要设置
    set(XSAN_LD_PRELOAD)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if (ENABLE_ASAN)
            # libasan.so
            execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libasan.so
                    RESULT_VARIABLE _RST
                    OUTPUT_VARIABLE ASAN_SHARED_PATH)
            if (_RST)
                message(FATAL_ERROR "Can't get libasan.so path with ${CMAKE_C_COMPILER}")
            endif ()
            get_filename_component(ASAN_SHARED_PATH "${ASAN_SHARED_PATH}" DIRECTORY)
            get_filename_component(ASAN_SHARED_PATH "${ASAN_SHARED_PATH}/libasan.so" REALPATH)
            if (NOT EXISTS ${ASAN_SHARED_PATH})
                message(FATAL_ERROR
                        "ASAN_SHARED_PATH=${ASAN_SHARED_PATH} not exist. Please check the completeness of the compiler installation.")
            endif ()
            list(APPEND XSAN_LD_PRELOAD ${ASAN_SHARED_PATH})
        endif ()
        if (ENABLE_UBSAN)
            # libubsan.so
            execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libubsan.so
                    RESULT_VARIABLE _RST
                    OUTPUT_VARIABLE UBSAN_SHARED_PATH)
            if (_RST)
                message(FATAL_ERROR "Can't get libubsan.so path with ${CMAKE_C_COMPILER}")
            endif ()
            get_filename_component(UBSAN_SHARED_PATH "${UBSAN_SHARED_PATH}" DIRECTORY)
            get_filename_component(UBSAN_SHARED_PATH "${UBSAN_SHARED_PATH}/libubsan.so" REALPATH)
            if (NOT EXISTS ${UBSAN_SHARED_PATH})
                message(FATAL_ERROR
                        "UBSAN_SHARED_PATH=${UBSAN_SHARED_PATH} not exist. Please check the completeness of the compiler installation.")
            endif ()
            list(APPEND XSAN_LD_PRELOAD ${UBSAN_SHARED_PATH})
        endif ()
        # libstdc++.so
        execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libstdc++.so
                RESULT_VARIABLE _RST
                OUTPUT_VARIABLE STDC_SHARED_PATH)
        if (_RST)
            message(FATAL_ERROR "Can't get libstdc++.so path with ${CMAKE_C_COMPILER}")
        endif ()
        get_filename_component(STDC_SHARED_PATH "${STDC_SHARED_PATH}" DIRECTORY)
        get_filename_component(STDC_SHARED_PATH "${STDC_SHARED_PATH}/libstdc++.so" REALPATH)
        if (NOT EXISTS ${STDC_SHARED_PATH})
            message(FATAL_ERROR
                    "STDC_SHARED_PATH=${STDC_SHARED_PATH} not exist. Please check the completeness of the compiler installation.")
        endif ()
        list(APPEND XSAN_LD_PRELOAD ${STDC_SHARED_PATH})
        # 结果修正
        string(REPLACE ";" ":" XSAN_LD_PRELOAD "${XSAN_LD_PRELOAD}")
        set(XSAN_LD_PRELOAD "LD_PRELOAD=${XSAN_LD_PRELOAD}")
    endif ()
    if (XSAN_LD_PRELOAD)
        file(APPEND "${XSAN_CONFIG_FILE}" "${XSAN_LD_PRELOAD}\n")
    endif ()

    set(ASAN_OPTIONS)
    if (ENABLE_ASAN)
        # 谨慎修改 ASAN_OPTIONS 取值, 当前出现告警会使 GTest 失败.
        # halt_on_error=1, 出现告警时停止运行进而触发构建失败, 避免主进程或 fork 出的子进程出现错误无法发现的情况
        # detect_stack_use_after_return=1, 栈空间返回后使用检测
        # check_initialization_order, 尝试捕获初始化顺序问题
        # strict_init_order, 动态初始化器永远不能访问来自其他模块的全局变量, 及时或者已经初始化
        # strict_string_checks, 检查字符串参数是否正确以 null 终止
        # detect_leaks=1, 内存泄漏检测
        set(ASAN_OPTIONS "ASAN_OPTIONS=halt_on_error=1,detect_stack_use_after_return=1,check_initialization_order=1,strict_init_order=1,strict_string_checks=1,symbolize=1,detect_leaks=1")
        if (ENABLE_FEATURE_PYTHON_FRONT_END)
            set(LSAN_OPTIONS "LSAN_OPTIONS=suppressions=${PTO_FWK_SRC_ROOT}/cmake/asan_suppressions.txt")
            file(APPEND "${XSAN_CONFIG_FILE}" "${LSAN_OPTIONS}\n")
        endif ()
    endif ()
    if (ASAN_OPTIONS)
        file(APPEND "${XSAN_CONFIG_FILE}" "${ASAN_OPTIONS}\n")
    endif ()

    set(UBSAN_OPTIONS)
    if (ENABLE_UBSAN)
        # 谨慎修改 UBSAN_OPTIONS 取值, 当前出现告警会使 UT 失败.
        # halt_on_error=1, 出现告警时停止运行进而触发构建失败, 避免主进程或 fork 出的子进程出现错误无法发现的情况
        # print_stacktrace=1, 出错时打印调用栈
        set(UBSAN_OPTIONS "UBSAN_OPTIONS=halt_on_error=0,print_stacktrace=1")
    endif ()
    if (UBSAN_OPTIONS)
        file(APPEND "${XSAN_CONFIG_FILE}" "${UBSAN_OPTIONS}\n")
    endif ()
endif ()


########################################################################################################################
# 三方库
########################################################################################################################

# torch optional
set(ENABLE_TORCH_VERIFIER OFF)
if (ENABLE_TESTS AND NOT ENABLE_FEATURE_PYTHON_FRONT_END)
    # calculator 二进制在 Python 场景仅在运行时编译, 仅在 C++ 场景, Tests 场景下才会编译
    if ("${PY3_MOD_TORCH_VERSION}" STRGREATER_EQUAL "2.1.0")
        if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "9.4.0") OR
            (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "15.0.0"))
            set(ENABLE_TORCH_VERIFIER ON)
        endif()
    endif()
endif()
message(STATUS "ENABLE_TORCH_VERIFIER=${ENABLE_TORCH_VERIFIER}")
