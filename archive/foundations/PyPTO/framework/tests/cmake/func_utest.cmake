# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(PTO_Fwk_UTestCaseLibraries         "" CACHE INTERNAL "" FORCE)     # UTest 各模块 用例实现二进制
set(PTO_Fwk_UTestCaseLkLibraries       "" CACHE INTERNAL "" FORCE)     # UTest 各模块 额外 Link 二进制
set(PTO_Fwk_UTestCaseLdLibrariesExt    "" CACHE INTERNAL "" FORCE)     # UTest 各模块 额外 Load 二进制

# UTest 添加测试用例二进制库
#[[
Parameters:
  one_value_keywords:
      TARGET                      : [Required] 具体测试用例二进制库名称
  multi_value_keywords:
      SOURCES                     : [Required] 编译源码
      PRIVATE_INCLUDE_DIRECTORIES : [Optional] 头文件搜索路径(PRIVATE)
      LINK_LIBRARIES              : [Optional] 链接库(PUBLIC), 仅会在最终编译出可执行文件时链接;
      LD_LIBRARIES_EXT            : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
Attention:
    1. 一般 LD_LIBRARIES_EXT 内配置的二进制, 在正常 source CANN 包环境变量后, LD_LIBRARY_PATH 内也应包含其所在路径;
]]
function(PTO_Fwk_UTest_AddCaseLib)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "SOURCES;PRIVATE_INCLUDE_DIRECTORIES;LINK_LIBRARIES;LD_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    add_library(${ARG_TARGET} STATIC)
    target_sources(${ARG_TARGET} PRIVATE ${ARG_SOURCES})
    target_include_directories(${ARG_TARGET} PRIVATE ${ARG_PRIVATE_INCLUDE_DIRECTORIES})
    target_link_libraries(${ARG_TARGET}
            PRIVATE
                ${PTO_Fwk_UTestNamePrefix}_utils
                GTest::gtest
                json
                c_sec
    )
    set(PTO_Fwk_UTestCaseLibraries       ${PTO_Fwk_UTestCaseLibraries}       ${ARG_TARGET}            CACHE INTERNAL "" FORCE)
    set(PTO_Fwk_UTestCaseLkLibrariesExt  ${PTO_Fwk_UTestCaseLkLibrariesExt}  ${ARG_LINK_LIBRARIES}    CACHE INTERNAL "" FORCE)
    set(PTO_Fwk_UTestCaseLdLibrariesExt  ${PTO_Fwk_UTestCaseLdLibrariesExt}  ${ARG_LD_LIBRARIES_EXT}  CACHE INTERNAL "" FORCE)
endfunction()

# UTest 执行可执行程序
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 用于指定具体 GTest 可执行目标, 用例会在该目标编译完成后(POST_BUILD)启动执行
  multi_value_keywords:
      LD_LIBRARIES_EXT   : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
      ENV_LINES_EXT      : [Optional] 需要额外配置的环境变量, 按照 "K=V" 格式组织
      GTEST_FILTER_LIST  : [Optional] GTestFilter 配置, Filter 间以 ';' 分割
Attention:
    1. 可以多次调用本函数以添加多个'执行任务'; 单次调用本函数时, 可以通过在 GTEST_FILTER_LIST 中配置多个过滤条件('gtest_filter') 以实现执行多用例;
]]
function(PTO_Fwk_UTest_RunExe)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "LD_LIBRARIES_EXT;ENV_LINES_EXT;GTEST_FILTER_LIST"
            ""
            ${ARGN}
    )
    if (ENABLE_TESTS_EXECUTE)
        # GCOV 并发写隔离: 传递 GCOV 数据目录, 使 utest_accelerate.py 为每个 Cntr 设置独立 GCOV_PREFIX
        if (ENABLE_GCOV)
            list(APPEND ARG_ENV_LINES_EXT "PYPTO_GCOV_ENABLED=1")
            list(APPEND ARG_ENV_LINES_EXT "PYPTO_GCOV_DATA_DIR=${PTO_FWK_BIN_ROOT}/gcov_prefix_data")
        endif ()
        # 命令行参数处理
        PTO_Fwk_GTest_RunExe_GetPreExecSetup(PyCmdSetup PyEnvLines BashCmdSetup
                TARGET              ${ARG_TARGET}
                ENV_LINES_EXT       ${ARG_ENV_LINES_EXT}
                LD_LIBRARIES_EXT    ${ARG_LD_LIBRARIES_EXT}
        )
        # 执行流程
        list(LENGTH ARG_GTEST_FILTER_LIST GtestFilterListLen)
        string(REPLACE ";" ":" GtestFilterStr "${ARG_GTEST_FILTER_LIST}")
        message(STATUS "Run GTest(${ARG_TARGET}), XSAN(ASAN:${ENABLE_ASAN} UBSAN:${ENABLE_UBSAN}), GTestFilter(${GtestFilterListLen})=${GtestFilterStr}")
        set(Comment "Run GTest(${ARG_TARGET}), XSAN(ASAN:${ENABLE_ASAN} UBSAN:${ENABLE_UBSAN})")

        if (ARG_GTEST_FILTER_LIST)
            # 使能并行执行
            set(_File $<TARGET_FILE:${ARG_TARGET}>)
            set(_Args "-t=${_File}" "--cases=${GtestFilterStr}" "--halt_on_error")
            if (PyEnvLines)
                list(APPEND _Args "--env" "${PyEnvLines}")
            endif ()
            get_filename_component(ParallelPy    "${PTO_FWK_SRC_ROOT}/cmake/scripts/utest_accelerate.py" REALPATH)
            get_filename_component(ParallelPyCwd "${PTO_FWK_SRC_ROOT}/cmake/scripts" REALPATH)
            add_custom_command(
                    TARGET ${ARG_TARGET} POST_BUILD
                    COMMAND ${PyCmdSetup} ${Python3_EXECUTABLE} ${ParallelPy} ARGS ${_Args}
                    COMMENT "${Comment} With Parallel Execute Accelerate"
                    WORKING_DIRECTORY ${ParallelPyCwd}
            )
        else ()
            add_custom_command(
                    TARGET ${ARG_TARGET} POST_BUILD
                    COMMAND ${BashCmdSetup} ./${ARG_TARGET}
                    COMMENT "${Comment}"
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            )
        endif ()
    endif ()
endfunction()

# UTest 添加并执行可执行程序
#[[
Parameters:
  one_value_keywords:
      TARGET                        : [Required] 用于指定具体 GTest 可执行目标, 用例会在该目标编译完成后(POST_BUILD)启动执行
Attention:
    1. 执行本函数后, 会产生名称为 TARGET 内容指定的构建目标, 外部可根据该目标名设置其他 custom_command 或处理依赖关系;
    2. 串行执行场景下 TARGET 内容指定最终 executable 名称;
    3. 并行执行场景下 TARGET 内容指定中间 custom_target 名称, 并行执行的各 executable 会依赖该 custom_target;
]]
function(PTO_Fwk_UTest_AddExe_RunExe)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            ""
            ""
            ${ARGN}
    )
    set(_Sources ${CMAKE_CURRENT_BINARY_DIR}/${PTO_Fwk_UTestNamePrefix}_main_stub.cpp)
    execute_process(COMMAND touch ${_Sources})

    # 默认全部执行
    set(GTestFilterList "*")
    # 支持由 ENABLE_UTEST 传入指定的 Filter
    if (NOT "${ENABLE_UTEST}" STREQUAL "ON")
        set(GTestFilterList ${ENABLE_UTEST})
        string(REPLACE ":" ";" GTestFilterList "${GTestFilterList}")
    endif ()

    list(REMOVE_DUPLICATES PTO_Fwk_UTestCaseLibraries)
    list(REMOVE_DUPLICATES PTO_Fwk_UTestCaseLkLibrariesExt)
    list(REMOVE_DUPLICATES PTO_Fwk_UTestCaseLdLibrariesExt)
    list(REMOVE_DUPLICATES GTestFilterList)

    if (NOT "$ENV{GTEST_START}" STREQUAL "")
        list(FIND GTestFilterList $ENV{GTEST_START} idx)
        if (NOT ${idx} EQUAL -1)
            list(SUBLIST GTestFilterList ${idx} -1 GTestFilterList)
        endif ()
    endif ()

    set(_PrivateLinkLibraries
            ${PTO_Fwk_UTestNamePrefix}_utils
            # 基本依赖
            # Interface 内 HostMachine 存在 dlopen 逻辑, 此处增加对应库连接, 触发相关 so 被添加到可执行程序依赖中
            tile_fwk_utils
            tile_fwk_adapter
            tile_fwk_cann_host_runtime
            tile_fwk_platform
            tile_fwk_interface
            tile_fwk_codegen
            tile_fwk_compiler
            # 用例特殊依赖
            ${PTO_Fwk_UTestCaseLkLibrariesExt}
            ${PTO_Fwk_UTestCaseLibraries}
    )
    list(REMOVE_DUPLICATES _PrivateLinkLibraries)
    PTO_Fwk_GTest_AddExe(
            TARGET                      ${ARG_TARGET}
            SOURCES                     ${_Sources}
            PRIVATE_INCLUDE_DIRECTORIES ${ARG_PRIVATE_INCLUDE_DIRECTORIES}
            PRIVATE_LINK_LIBRARIES      ${_PrivateLinkLibraries}
    )
    PTO_Fwk_UTest_RunExe(
            TARGET              ${ARG_TARGET}
            LD_LIBRARIES_EXT    ${PTO_Fwk_UTestCaseLdLibrariesExt}
            GTEST_FILTER_LIST   ${GTestFilterList}
    )

    # 生成覆盖率
    PTO_Fwk_GTest_GenerateCoverage(TARGET ${ARG_TARGET})
endfunction()

function(PTO_Fwk_UTest_AddModuleDir)
    cmake_parse_arguments(
            ARG
            ""
            "DIR"
            ""
            ${ARGN}
    )
    PTO_Fwk_GTest_AddModuleDir(
            MARK "UTest"
            DIR ${ARG_DIR}
            MODULE_LIST ${ENABLE_UTEST_MODULE}
    )
endfunction()
