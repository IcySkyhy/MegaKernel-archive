# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# GTest 生成覆盖率
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 指定所依赖的目标(POST_BUILD)
  multi_value_keywords:
      FILTER_DIRECTORIES : [Optional] 覆盖率结果过滤目录
]]
function(PTO_Fwk_GTest_GenerateCoverage)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "FILTER_DIRECTORIES"
            ""
            ${ARGN}
    )
    if (ENABLE_TESTS_EXECUTE AND ENABLE_GCOV)
        # 获取 gcc 默认头文件搜索路径
        execute_process(
                COMMAND ${CMAKE_C_COMPILER} --print-sysroot
                RESULT_VARIABLE _RST
                OUTPUT_VARIABLE _SUFFIX
                ERROR_QUIET
        )
        if (_RST)
            get_filename_component(SYS_ROOT "/usr/include" REALPATH)
        else ()
            get_filename_component(SYS_ROOT "${_SUFFIX}/usr/include" REALPATH)
        endif ()

        # 参数组织
        find_program(LCOV lcov REQUIRED)
        get_filename_component(GenCoveragePy ${PTO_FWK_SRC_ROOT}/cmake/scripts/gen_coverage.py REALPATH)
        get_filename_component(GenCoverageDataDir "${PTO_FWK_BIN_ROOT}" REALPATH)
        set(_Args "-s=${PTO_FWK_SRC_ROOT}" "-d=${GenCoverageDataDir}")

        set(Filter_Dirs
                ${PTO_FWK_SRC_ROOT}/framework/tests
                $<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:json,INTERFACE_INCLUDE_DIRECTORIES>
                ${SYS_ROOT}
                ${Python3_INCLUDE_DIRS}
                ${pybind11_INCLUDE_DIR}
                ${ARG_FILTER_DIRECTORIES}
        )
        if (ENABLE_TORCH_VERIFIER)
            list(APPEND Filter_Dirs ${PY3_MOD_TORCH_ROOT_PATH}/include)
        endif ()
        if (BUILD_WITH_CANN)
            list(APPEND Filter_Dirs ${ASCEND_CANN_PACKAGE_PATH}/include)
            list(APPEND Filter_Dirs ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc)
        endif ()
        foreach (_dir ${Filter_Dirs})
            list(APPEND _Args "-f=${_dir}")
        endforeach ()
        list(APPEND _Args "--gcov")
        list(REMOVE_DUPLICATES _Args)

        add_custom_command(
                TARGET ${ARG_TARGET} POST_BUILD
                COMMAND ${Python3_EXECUTABLE} ${GenCoveragePy} ARGS ${_Args}
                VERBATIM
                COMMENT "Generate coverage for ${ARG_TARGET}"
        )
    endif ()
endfunction()

# GTest 获取可执行程序在执行前需要的命令行配置和环境变量配置
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 用于指定具体 GTest 可执行目标
  multi_value_keywords:
      PY_CMD_SETUP       : [Required] 输出 Python 场景命令行配置
      PY_ENV_LINES       : [Required] 输出 Python 场景环境变量(按照 "K=V" 格式组织)
      BASH_CMD_SETUP     : [Required] 输出 bash   场景命令行配置(内部包含命令行配置+环境变量配置)

      LD_LIBRARIES_EXT   : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
      CMD_SETUP_EXT      : [Optional] 附加命令行配置, 要求调用者设置 export 及多命令行配置间的 && 连接
      ENV_LINES_EXT      : [Optional] 附加环境变量配置(按照 "K=V" 格式组织)
]]
function(PTO_Fwk_GTest_RunExe_GetPreExecSetup PY_CMD_SETUP PY_ENV_LINES BASH_CMD_SETUP)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "LD_LIBRARIES_EXT;CMD_SETUP_EXT;ENV_LINES_EXT"
            ""
            ${ARGN}
    )

    # 命令行
    set(CmdSetup)
    # 处理变量 CMD_SETUP_EXT
    if (NOT "${ARG_CMD_SETUP_EXT}x" STREQUAL "x")
        list(APPEND CmdSetup ${ARG_CMD_SETUP_EXT})
    endif ()
    # 处理变量内部处理(XSan 相关处理)
    if (ENABLE_ASAN OR ENABLE_UBSAN)
        if (NOT "${CmdSetup}x" STREQUAL "x")
            list(APPEND CmdSetup &&)
        endif ()
        list(APPEND CmdSetup ulimit -s 32768)
    endif ()

    # 环境变量
    set(EnvLines)
    # 处理变量 LD_LIBRARIES_EXT 及环境变量 LD_LIBRARY_PATH 及 CMAKE_LIBRARY_OUTPUT_DIRECTORY
    # 1. 当前 UTest/STest 已把动态库生成路径设置到 CMAKE_LIBRARY_OUTPUT_DIRECTORY 路径下, 此处需增加该路径配置;
    # 2. 保留 LD_LIBRARY_PATH_EXT 处理逻辑, 已供后续其他场景使用;
    set(LD_LIBRARY_PATH_EXT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    foreach (LIBRARY ${ARG_LD_LIBRARIES_EXT})
        add_dependencies(${ARG_TARGET} ${LIBRARY})
        list(APPEND LD_LIBRARY_PATH_EXT "$<TARGET_FILE_DIR:${LIBRARY}>")
    endforeach ()
    string(REPLACE ";" ":" LD_LIBRARY_PATH_EXT "${LD_LIBRARY_PATH_EXT}")
    set(LD_LIBRARY_PATH "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}")
    if (NOT "${LD_LIBRARY_PATH_EXT}x" STREQUAL "x")
        set(LD_LIBRARY_PATH "LD_LIBRARY_PATH=${LD_LIBRARY_PATH_EXT}:$ENV{LD_LIBRARY_PATH}")
    endif ()
    list(APPEND EnvLines ${LD_LIBRARY_PATH})
    # 处理环境变量 PATH
    # 处理变量 ENV_SETUP_EXT
    list(REMOVE_ITEM ARG_ENV_LINES_EXT export)
    list(REMOVE_ITEM ARG_ENV_LINES_EXT &)
    list(REMOVE_ITEM ARG_ENV_LINES_EXT &&)
    if (NOT "${ARG_ENV_LINES_EXT}x" STREQUAL "x")
        list(APPEND EnvLines ${ARG_ENV_LINES_EXT})
    endif ()
    # 处理 ASAN / UBSAN 场景
    if (ENABLE_ASAN OR ENABLE_UBSAN)
        if (NOT "${XSAN_LD_PRELOAD}x" STREQUAL "x")
            list(APPEND EnvLines ${XSAN_LD_PRELOAD})
        endif ()
    endif ()

    # 输出处理
    set(PyCmdSetup ${CmdSetup})
    if (NOT "${PyCmdSetup}x" STREQUAL "x")
        list(APPEND PyCmdSetup &&)
    endif ()
    set(${PY_CMD_SETUP} ${PyCmdSetup} PARENT_SCOPE)

    # 输出处理
    set(XSan_Options)
    if (ENABLE_ASAN OR ENABLE_UBSAN)
        set(XSan_Options ${ASAN_OPTIONS} ${UBSAN_OPTIONS})
    endif ()
    set(PyEnvLines ${EnvLines} ${XSan_Options})
    set(${PY_ENV_LINES} ${PyEnvLines} PARENT_SCOPE)

    # 输出处理
    set(BashSetup)
    if (NOT "${EnvLines}x" STREQUAL "x")
        foreach (_line ${EnvLines})
            if (NOT "${BashSetup}x" STREQUAL "x")
                list(APPEND BashSetup &&)
            endif ()
            list(APPEND BashSetup export ${_line})
        endforeach ()
    endif ()
    if (NOT "${CmdSetup}x" STREQUAL "x")
        if (NOT "${BashSetup}x" STREQUAL "x")
            list(APPEND BashSetup &&)
        endif ()
        list(APPEND BashSetup ${CmdSetup})
    endif ()
    # XSan 特殊处理(XSan_Options)
    # 1. 当其存在时, 其需要在 BashSetup 尾部, 其前需要 && 与前述命令行连接, 其后不需要补充 && 连接符
    # 2. 当不存在时, BashSetup 尾部需要补充 && 连接符;
    if (NOT "${BashSetup}x" STREQUAL "x")
        list(APPEND BashSetup &&)
    endif ()
    if (NOT "${XSan_Options}x" STREQUAL "x")
        list(APPEND BashSetup ${XSan_Options})
    endif ()
    set(${BASH_CMD_SETUP} ${BashSetup} PARENT_SCOPE)
endfunction()

# GTest 添加可执行程序
#[[
Parameters:
  one_value_keywords:
      TARGET                        : [Required] 用于指定具体 GTest 可执行目标, 用例会在该目标编译完成后(POST_BUILD)启动执行
  multi_value_keywords:
      SOURCES                       : [Optional] 额外的编译源码
      PRIVATE_INCLUDE_DIRECTORIES   : [Optional] Private 头文件查找路径
      PRIVATE_LINK_LIBRARIES        : [Optional] Private 链接库
]]
function(PTO_Fwk_GTest_AddExe)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "SOURCES;PRIVATE_INCLUDE_DIRECTORIES;PRIVATE_LINK_LIBRARIES"
            ""
            ${ARGN}
    )
    add_executable(${ARG_TARGET})
    target_sources(${ARG_TARGET}
            PRIVATE
                ${ARG_SOURCES}
                ${PTO_FWK_SRC_ROOT}/framework/tests/main.cpp
    )
    target_include_directories(${ARG_TARGET}
            PRIVATE
                ${ARG_PRIVATE_INCLUDE_DIRECTORIES}
    )
    target_compile_definitions(${ARG_TARGET}
            PRIVATE
                $<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>
                $<$<BOOL:${ENABLE_UTEST}>:ENABLE_UTEST>
                $<$<BOOL:${ENABLE_STEST}>:ENABLE_STEST>
    )
    target_link_libraries(${ARG_TARGET}
            PRIVATE
                GTest::gtest
                -Wl,--no-as-needed
                -Wl,--whole-archive
                ${ARG_PRIVATE_LINK_LIBRARIES}
                -Wl,--as-needed
                -Wl,--no-whole-archive
                -rdynamic
                -pthread
    )
    # 模拟配置文件 Install 流程, 为便于调试, 使用创建软连接方式模拟安装
    get_filename_component(InstallConfigsDir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/configs" REALPATH)
    add_custom_command(
            TARGET ${ARG_TARGET} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${InstallConfigsDir}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${InstallConfigsDir}
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/framework/src/interface/configs/*.json"                         "${InstallConfigsDir}/"
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/framework/src/passes/pass_config/tile_fwk_platform_info.json"   "${InstallConfigsDir}/"
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/framework/src/platform/parser/platforminfo.ini"   "${InstallConfigsDir}/"
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/framework/src/platform/parser/simulation_platform/platform_config/*.ini"   "${InstallConfigsDir}/"
            COMMENT "Soft link of configs(*.json) has been created at ${InstallConfigsDir}"
    )
    # 模拟脚本文件 Install 流程, 为便于调试, 使用创建软连接方式模拟安装
    get_filename_component(InstallScriptsDir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/scripts" REALPATH)
    add_custom_command(
            TARGET ${ARG_TARGET} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${InstallScriptsDir}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${InstallScriptsDir}
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/tools/profiling/draw_swim_lane.py" "${InstallScriptsDir}/"
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/draw_pipe_swim_lane.py" "${InstallScriptsDir}/"
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/draw_comm_swim_lane_png.py" "${InstallScriptsDir}/"
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/print_swim_lane.py" "${InstallScriptsDir}/"
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/tools/profiling/function_json_convert.py" "${InstallScriptsDir}/"
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/tools/profiling/parse_pipe_time_trace.py" "${InstallScriptsDir}/"
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/tools/scripts/machine_perf_trace.py" "${InstallScriptsDir}/"
            COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/tools/scripts/extract_pass_log.py" "${InstallScriptsDir}/"
            COMMENT "Soft link of scripts has been created at ${InstallScriptsDir}"
    )
    # 模拟头文件 Install 流程, 为便于调试, 使用创建软连接方式模拟安装
    get_filename_component(InstallIncludeDir "${PTO_FWK_BIN_OUTPUT_ROOT}/include" REALPATH)
    add_custom_command(
            TARGET ${ARG_TARGET} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${InstallIncludeDir}/
            COMMAND ${CMAKE_COMMAND} -E make_directory ${InstallIncludeDir}/
            COMMAND ln -sf ${PTO_FWK_SRC_ROOT}/framework/include/tilefwk ${InstallIncludeDir}
            COMMENT "Soft link of include directory has been created at ${InstallIncludeDir}"
    )
    get_filename_component(InstallLibIncludeDir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/include" REALPATH)
    add_custom_command(
            TARGET ${ARG_TARGET} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${InstallLibIncludeDir}/
            COMMAND ${CMAKE_COMMAND} -E make_directory ${InstallLibIncludeDir}/
            COMMAND ln -sf ${PTO_FWK_SRC_ROOT}/framework/src/interface/tileop ${InstallLibIncludeDir}
            COMMAND ln -sf ${PTO_FWK_SRC_ROOT}/framework/src/interface/machine/device/tilefwk ${InstallLibIncludeDir}
            COMMENT "Soft link of library include directory has been created at ${InstallLibIncludeDir}"
    )
endfunction()

# 用于获取 GTestFilter(str)
#[[
Parameters:
  one_value_keywords:
      CLASSIFY                      : [Required] Classify 配置文件
      TESTS_TYPE                    : [Required] 测试类型, 支持 [utest, stest]
      TESTS_GROUP                   : [Optional] 测试分组
      CHANGED_FILE                  : [Optional] 修改文件
]]
function(PTO_Fwk_GTest_GetGTestFilterStr GTEST_FILTER_STR)
    cmake_parse_arguments(
            ARG
            ""
            "CLASSIFY;TESTS_TYPE;TESTS_GROUP;CHANGED_FILE"
            ""
            ""
            ${ARGN}
    )
    get_filename_component(_Py "${PTO_FWK_SRC_ROOT}/cmake/scripts/analysis_changed_files.py" REALPATH)
    set(_Args "-r=${ARG_CLASSIFY}" "-t=${ARG_TESTS_TYPE}")
    if (ARG_TESTS_GROUP AND NOT "${ARG_TESTS_GROUP}" STREQUAL "ON")
        string(REPLACE ":" "," TestsGroupStr "${ARG_TESTS_GROUP}")
        list(APPEND _Args "-g=${TestsGroupStr}")
    endif ()
    if (ARG_CHANGED_FILE AND NOT "${ARG_CHANGED_FILE}" STREQUAL "ON")
        list(APPEND _Args "-c=${ARG_CHANGED_FILE}")
    endif ()
    execute_process(
            COMMAND ${Python3_EXECUTABLE} ${_Py} ${_Args}
            OUTPUT_VARIABLE OutputVariable
    )
    string(REPLACE "," ":" OutputVariable "${OutputVariable}")
    set(${GTEST_FILTER_STR} ${OutputVariable} PARENT_SCOPE)
endfunction()

# GTest 按模块添加路径
#[[
Parameters:
  one_value_keywords:
      MARK                          : [Required] 标识测试类型
      DIR                           : [Required] 待添加的模块名(与子路径同名)
  multi_value_keywords:
      MODULE_LIST                   : [Optional] 配置的模块名列表
]]
function(PTO_Fwk_GTest_AddModuleDir)
    cmake_parse_arguments(
            ARG
            ""
            "MARK;DIR"
            "MODULE_LIST"
            ""
            ${ARGN}
    )
    if ((NOT ARG_MODULE_LIST) OR "${ARG_MODULE_LIST}x" STREQUAL "ONx")
        set(_TestModuleList "ALL")
    else ()
        string(REPLACE "," ";" _TestModuleList "${ARG_MODULE_LIST}")
        string(REPLACE ":" ";" _TestModuleList "${_TestModuleList}")
    endif ()
    if (("${_TestModuleList}" STREQUAL "ALL") OR ("${ARG_DIR}" IN_LIST _TestModuleList))
        add_subdirectory(${ARG_DIR})
        message(STATUS "${ARG_MARK} Add module(${ARG_DIR})")
    endif()
endfunction()
